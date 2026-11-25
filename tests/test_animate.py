#!/usr/bin/env python3
"""The Animate stage (scripts/animate.py, scripts/skeleton.py, POST /2d/animate/<stem>) with LTX-2.3 stood in for.

The stand-in is a fake ltx-2-mlx clone whose command line writes the skeleton video back, negated (coloured
lines on white), as "the clip": the real launcher (scripts/ltx_run.py) still runs it under python -I, so what
reaches LTX (local paths only, offline, isolated) is checked here. The real model's run is
tests/test_animate_ltx.py.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import animate as A  # noqa: E402
import factory_http  # noqa: E402
import flow  # noqa: E402
import skeleton as SK  # noqa: E402
from test_factory_http import http_json  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
CYCLE_MAKER = A.cycle_maker(ROOT)
# the walker's rig (out/walker/stack/character.json, pipeline#6) without its parts: what skeleton-cycle.ts reads
WALKER = {"format": "pipeline-sprite-character", "version": 1, "stem": "walker", "height": 64, "parts": [],
          "rig": {"thigh": 9, "shin": 9, "ankle": 4, "hipHalf": 4.5, "hipFwd": -2.0, "kneeHint": 1, "standHip": 21.28,
                  "stride": 28.6, "duty": 0.6, "lift": 3.06, "bob": 0.55, "twist": 0.14, "spine": 26, "neck": 28,
                  "arms": True, "shoulderHalf": 9.5, "shoulderFwd": -3.0, "upperArm": 12, "foreArm": 13,
                  "armSwing": 0.45}}

FAKE_CLI = '''
import json, os, subprocess, sys

def main():
    a = sys.argv[1:]
    out = a[a.index("--output") + 1]
    control = a[a.index("--video-conditioning") + 1]
    record = os.environ.get("FAKE_LTX_RECORD")
    if record:
        json.dump({"argv": a, "isolated": sys.flags.isolated, "offline": os.environ.get("HF_HUB_OFFLINE"),
                   "path0": sys.path[0]}, open(record, "w"))
    if os.environ.get("FAKE_LTX_FAIL"):
        print("Error: fake LTX failed", file=sys.stderr)
        return 3
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", control, "-vf", "negate", "-pix_fmt", "yuv420p", out],
                   check=True)
    return 0
'''

# What Apple Vision reads from the operator picture (video-trials clips/operator-704.png): a realistic body, built
# unlike the walker skeletons (hips at 0.479 against their 0.637, shoulders 0.15 wide against 0.24).
OPERATOR = json.loads('''{"head_joint": [0.489, 0.13, 0.869], "left_ear_joint": [0.535, 0.129, 0.912], "left_eye_joint": [0.504, 0.114, 0.942], "left_foot_joint": [0.593, 0.851, 0.676], "left_forearm_joint": [0.645, 0.349, 0.8], "left_hand_joint": [0.677, 0.456, 0.794], "left_leg_joint": [0.563, 0.681, 0.644], "left_shoulder_1_joint": [0.582, 0.226, 0.817], "left_upLeg_joint": [0.548, 0.477, 0.546], "neck_1_joint": [0.506, 0.232, 0.803], "right_ear_joint": [0.459, 0.128, 0.529], "right_eye_joint": [0.474, 0.114, 0.927], "right_foot_joint": [0.459, 0.848, 0.773], "right_forearm_joint": [0.395, 0.363, 0.75], "right_hand_joint": [0.335, 0.457, 0.856], "right_leg_joint": [0.449, 0.662, 0.741], "right_shoulder_1_joint": [0.43, 0.239, 0.79], "right_upLeg_joint": [0.452, 0.481, 0.591], "root": [0.5, 0.479, 0.569]}''')

FAKE_BODYPOSE = '''#!{python}
# Stands in for bin/bodypose: the control skeleton back, every joint moved 1% of the width to the right; the
# guide picture alone (animate.py's start/ folder) reads as the operator, or as nothing with FAKE_BODYPOSE_NONE.
import json, os, sys
frames = sorted(f for f in os.listdir(sys.argv[1]) if f.endswith(".png"))
if os.path.basename(sys.argv[1].rstrip("/")) == "start":
    j = {{}} if os.environ.get("FAKE_BODYPOSE_NONE") else {operator}
    json.dump({{"frames": [{{"file": frames[0], "joints2d": j}}]}}, open(sys.argv[2], "w"))
    sys.exit(0)
control = json.load(open(os.path.join(os.path.dirname(sys.argv[1].rstrip("/")), "control.json")))["frames"]
out = []
for i, f in enumerate(frames):
    j = {{k: [v[0] + 0.01, v[1], v[2]] for k, v in control[min(i, len(control) - 1)]["joints2d"].items()}}
    out.append({{"file": f, "joints2d": j}})
json.dump({{"frames": out}}, open(sys.argv[2], "w"))
'''


def fake_ltx(base: Path) -> dict[str, str]:
    """A fake clone, weights folder, LoRA and text encoder; the environment animate.py reads them from."""
    clone = base / "ltx-2-mlx"
    pkg = clone / "packages" / "ltx-pipelines-mlx" / "src" / "ltx_pipelines_mlx"
    pkg.mkdir(parents=True)
    (clone / "packages" / "ltx-core-mlx" / "src").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "cli.py").write_text(textwrap.dedent(FAKE_CLI))
    model = base / "weights" / "ltx-2.3-mlx-q8"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    lora = base / "weights" / "lora.safetensors"
    lora.write_bytes(b"x")
    gemma = base / "weights" / "gemma"
    gemma.mkdir()
    bodypose = base / "bodypose"
    bodypose.write_text(FAKE_BODYPOSE.format(python=sys.executable, operator=json.dumps(OPERATOR)))
    bodypose.chmod(0o755)
    return {"PIPELINE_LTX_DIR": str(clone), "PIPELINE_LTX_PYTHON": sys.executable, "PIPELINE_LTX_MODEL": str(model),
            "PIPELINE_LTX_LORA": str(lora), "PIPELINE_LTX_GEMMA": str(gemma), "PIPELINE_BODYPOSE": str(bodypose),
            "FAKE_LTX_RECORD": str(base / "ltx-call.json")}


def make_root(base: Path, stem: str = "op") -> Path:
    root = base / "root"
    (root / "out" / stem).mkdir(parents=True)
    (root / "scripts").symlink_to(ROOT / "scripts")
    # A character picture on a light grey ground, like in/operator.jpg: a dark figure, standing.
    im = Image.new("RGB", (512, 512), (222, 224, 221))
    for y in range(120, 470):
        for x in range(225, 287):
            im.putpixel((x, y), (60, 80, 50))
    im.save(root / "out" / stem / "input.png")
    return root


def add_front_view(root: Path, stem: str = "op") -> Path:
    """A drawn front view, a cut-out (transparent ground): a figure 40 wide and 100 tall at (30..70, 10..110)."""
    im = Image.new("RGBA", (100, 120), (0, 0, 0, 0))
    for y in range(10, 110):
        for x in range(30, 70):
            im.putpixel((x, y), (180, 40, 40, 255))
    p = root / "out" / stem / "flat" / "views" / "front.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p)
    return p


def man_files(root: Path, take_id: str) -> list[str]:
    man = json.loads((root / "out" / "op" / "flow.json").read_text())
    return next(t for t in man["stages"]["flat"]["takes"] if t["id"] == take_id)["files"]


class SkeletonTests(unittest.TestCase):
    def test_walk_skeleton_ships(self):
        self.assertIn("walk_side", SK.shipped())
        frames = SK.load(SK.named("walk_side"))
        self.assertEqual(len(frames), 73)
        for name in ("head_joint", "left_foot_joint", "right_hand_joint"):
            self.assertIn(name, frames[0])

    def test_code_cycles_ship_with_their_figure(self):
        for cycle, loop in (("walk", 28), ("run", 18), ("attack", 24)):
            for facing in SK.FACINGS:
                name = SK.shipped_cycle(cycle, facing)
                self.assertIn(name, SK.shipped())
                meta = SK.info(SK.named(name))
                self.assertEqual((meta["frames"], meta["turn_frames"], meta["cycle_frames"]), (73, 16, loop))
                self.assertEqual(meta["figure"], {"top": 0.0909, "ground": 0.9091, "x": 0.5})
        self.assertIsNone(SK.info(SK.named("walk_side"))["figure"])
        with self.assertRaises(SK.SkeletonError):
            SK.shipped_cycle("dance", "right")
        with self.assertRaises(SK.SkeletonError):
            SK.shipped_cycle("walk", "up")
        self.assertIn("turns to face left", SK.cycle_prompt("run", "left"))

    def test_scale_about_keeps_the_feet(self):
        got = SK.scale_about([{"a": [0.5, 0.9, 1], "b": [0.7, 0.1, 0.8]}], 0.5, 0.9, 0.5)
        self.assertEqual(got[0]["a"], [0.5, 0.9, 1])
        self.assertEqual([round(v, 4) for v in got[0]["b"]], [0.6, 0.5, 0.8])

    def test_unknown_skeleton_is_refused(self):
        with self.assertRaises(SK.SkeletonError):
            SK.named("../etc/passwd")
        with self.assertRaises(SK.SkeletonError):
            SK.named("moonwalk")

    def test_retime_and_fit(self):
        frames = [{"head_joint": [i / 10, 0.5, 1.0]} for i in range(5)]
        half = SK.retime(frames, 24, 48, 9)
        self.assertAlmostEqual(half[1]["head_joint"][0], 0.05)
        self.assertAlmostEqual(half[8]["head_joint"][0], 0.4)
        self.assertEqual(len(SK.fit_frames(frames, 7)), 7)
        self.assertEqual(SK.fit_frames(frames, 7)[-1], frames[-1])
        self.assertEqual(len(SK.fit_frames(frames, 3)), 3)

    def test_pose_error_is_percent_of_width(self):
        a = [{"head_joint": [0.5, 0.5, 1.0], "left_foot_joint": [0.4, 0.9, 1.0]}] * 4
        b = [{"head_joint": [0.53, 0.5, 1.0], "left_foot_joint": [0.43, 0.9, 1.0]}] * 4
        res = SK.pose_error(a, b, skip=1)
        self.assertEqual(res["frames_compared"], 3)
        self.assertAlmostEqual(res["mean_joint_error_pct"], 3.0, places=2)
        self.assertAlmostEqual(res["mean_limb_error_pct"], 3.0, places=2)
        self.assertEqual(SK.pose_error(a, a)["mean_joint_error_pct"], 0)

    def test_low_confidence_joints_are_not_compared(self):
        a = [{"head_joint": [0.5, 0.5, 1.0], "root": [0.5, 0.6, 0.1]}]
        b = [{"head_joint": [0.5, 0.5, 1.0], "root": [0.9, 0.6, 1.0]}]
        self.assertEqual(SK.pose_error(a, b)["mean_joint_error_pct"], 0)

    def test_drawn_frame_has_openpose_colours(self):
        im = SK.draw_frame(SK.load(SK.named("walk_side"))[0], 256, 256)
        colours = {c for _, c in im.getcolors(256 * 256)}
        self.assertIn((0, 0, 0), colours)
        self.assertIn((255, 0, 0), colours)  # neck to right shoulder, OpenPose's first limb


class RefitTests(unittest.TestCase):
    """skeleton.refit: a shipped skeleton refitted to a picture's own body (the operator's, as Vision reads it)."""

    @staticmethod
    def length(j, a, b):
        return ((j[a][0] - j[b][0]) ** 2 + (j[a][1] - j[b][1]) ** 2) ** 0.5

    def test_first_frame_takes_the_pictures_proportions(self):
        walker = SK.load(SK.named("walker_walk_right"))
        fitted, info = SK.refit(walker, OPERATOR)
        self.assertEqual(len(fitted), len(walker))
        self.assertEqual(set(info["fitted_parts"]), set(SK.PARTS))
        f0 = fitted[0]
        # The hips where the operator's are (not the walker's 0.637), the lower foot on the operator's ground.
        self.assertAlmostEqual(f0["root"][0], OPERATOR["root"][0], places=3)
        self.assertLess(abs(f0["root"][1] - OPERATOR["root"][1]), 0.01)
        self.assertAlmostEqual(max(f0[f][1] for f in SK.FEET), max(OPERATOR[f][1] for f in SK.FEET), places=3)
        # Shoulder and hip widths, and each limb, at the operator's length (left and right averaged).
        for a, b in (("left_shoulder_1_joint", "right_shoulder_1_joint"), ("left_upLeg_joint", "right_upLeg_joint")):
            self.assertAlmostEqual(self.length(f0, a, b), self.length(OPERATOR, a, b), delta=0.005)
        for side in ("left", "right"):
            for a, b in ((f"{side}_upLeg_joint", f"{side}_leg_joint"), (f"{side}_leg_joint", f"{side}_foot_joint"),
                         (f"{side}_shoulder_1_joint", f"{side}_forearm_joint")):
                mean = (self.length(OPERATOR, a.replace(side, "left"), b.replace(side, "left")) +
                        self.length(OPERATOR, a.replace(side, "right"), b.replace(side, "right"))) / 2
                self.assertAlmostEqual(self.length(f0, a, b), mean, delta=0.01, msg=(a, b))
        # Much closer to the standing picture than moving and scaling alone (pipeline#18's x1.117 fit).
        moved = [{n: [1.117 * (v[0] - 0.5) + 0.5, 1.117 * (v[1] - 0.858) + 0.850, v[2]] for n, v in j.items()}
                 for j in walker]
        before = SK.pose_error(moved[:9], [OPERATOR] * 9, skip=1)["mean_joint_error_pct"]
        after = SK.pose_error(fitted[:9], [OPERATOR] * 9, skip=1)["mean_joint_error_pct"]
        self.assertGreater(before, 6)
        self.assertLess(after, before / 1.8)

    def test_motion_is_kept(self):
        walker = SK.load(SK.named("walker_walk_right"))
        fitted, info = SK.refit(walker, OPERATOR)
        r = info["ratios"]["thigh"]
        for i in (8, 24, 40, 60):  # each thigh bone is the walker's, scaled: same direction, the operator's length
            a, b = walker[i], fitted[i]
            for side in ("left", "right"):
                up, knee = f"{side}_upLeg_joint", f"{side}_leg_joint"
                for c in (0, 1):
                    self.assertAlmostEqual(b[knee][c] - b[up][c], (a[knee][c] - a[up][c]) * r, places=3)

    def test_every_shipped_skeleton_stays_grounded_and_in_frame(self):
        ground = max(OPERATOR[f][1] for f in SK.FEET)
        for name in SK.shipped():
            src = SK.load(SK.named(name))
            fitted, info = SK.refit(src, OPERATOR)
            g0 = max(src[0][f][1] for f in SK.FEET)
            for i, (a, b) in enumerate(zip(src, fitted)):
                lift = (max(a[f][1] for f in SK.FEET) - g0) * info["scale"]
                self.assertAlmostEqual(max(b[f][1] for f in SK.FEET), ground + lift, places=3, msg=(name, i))
                for joint, v in b.items():
                    self.assertTrue(0 <= v[0] <= 1 and 0 <= v[1] <= 1, (name, i, joint, v))

    def test_a_part_the_picture_lacks_takes_the_bodys_scale(self):
        pic = {k: v for k, v in OPERATOR.items() if "hand" not in k}
        _, info = SK.refit(SK.load(SK.named("walker_walk_right")), pic)
        self.assertNotIn("forearm", info["fitted_parts"])
        self.assertEqual(info["ratios"]["forearm"], info["scale"])
        squashed = dict(OPERATOR, left_shoulder_1_joint=[0.505, 0.232, 0.9], right_shoulder_1_joint=[0.507, 0.232, 0.9])
        _, info = SK.refit(SK.load(SK.named("walker_walk_right")), squashed)  # a side-on picture: no width to read
        self.assertAlmostEqual(info["ratios"]["shoulders"], info["scale"] / SK.RATIO_BAND, places=3)

    def test_no_body_to_fit_to_is_refused(self):
        with self.assertRaises(SK.SkeletonError):
            SK.refit(SK.load(SK.named("walk_side")), {})
        with self.assertRaises(SK.SkeletonError):
            SK.refit(SK.load(SK.named("walk_side")), {"head_joint": [0.5, 0.1, 0.9]})


class LtxRunTests(unittest.TestCase):
    def test_appended_keyframe_tokens_are_dropped_before_unpatchify(self):
        import numpy as np

        class VideoLatentPatchifier:
            def unpatchify(self, tokens, spatial_dims):
                f, h, w = spatial_dims
                return tokens.reshape(tokens.shape[0], f, h, w, tokens.shape[2])

        mod = type(sys)("ltx_core_mlx.components.patchifiers")
        mod.VideoLatentPatchifier = VideoLatentPatchifier
        names = ("ltx_core_mlx", "ltx_core_mlx.components", "ltx_core_mlx.components.patchifiers")
        with mock.patch.dict(sys.modules, {n: type(sys)(n) for n in names[:2]} | {names[2]: mod}):
            import ltx_run

            ltx_run.strip_appended()
        tokens = np.arange(1 * (2 * 3 * 3 + 9) * 4).reshape(1, 2 * 3 * 3 + 9, 4)  # 9 keyframe tokens appended
        out = VideoLatentPatchifier().unpatchify(tokens, (2, 3, 3))
        self.assertEqual(out.shape, (1, 2, 3, 3, 4))
        self.assertEqual(out.reshape(1, -1, 4)[0, -1, 0], tokens[0, 17, 0])


class LostCutoutTests(unittest.TestCase):
    """A cut-out that lost the character's legs (Apple Vision drops motion-blurred ones) is found by its size
    against the frames around it, redone against the ground behind it, or else left out of the strip."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        (self.base / "frames").mkdir()
        (self.base / "cutouts").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make(self, n=15, cut_legs=(), no_legs=()):
        """A dark figure on a light ground that shades left to right (as a studio floor does, so one ground colour
        cannot key it): a body and a leg that swings. Cut-outs are the figure, minus the legs in cut_legs; a frame
        in no_legs has no legs in the picture either."""
        from PIL import ImageDraw

        frames, cuts = [], []
        for i in range(n):
            im = Image.new("RGB", (96, 96))
            px = im.load()
            for x in range(96):
                for y in range(96):
                    px[x, y] = (230 - x, 230 - x, 228 - x)
            cut = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
            leg_x = 30 + (i * 7) % 30
            for d in (ImageDraw.Draw(im), ImageDraw.Draw(cut)):
                d.rectangle((38, 10, 56, 50), fill=(40, 60, 40, 255))
            if i not in no_legs:
                ImageDraw.Draw(im).rectangle((leg_x, 51, leg_x + 8, 88), fill=(30, 40, 30))
                if i not in cut_legs:
                    ImageDraw.Draw(cut).rectangle((leg_x, 51, leg_x + 8, 88), fill=(30, 40, 30, 255))
            f, c = self.base / "frames" / f"{i:04d}.png", self.base / "cutouts" / f"{i:04d}.png"
            im.save(f)
            cut.save(c)
            frames.append(f)
            cuts.append(c)
        return frames, cuts

    def test_whole_cutouts_are_left_alone(self):
        frames, cuts = self.make()
        before = [c.read_bytes() for c in cuts]
        self.assertEqual(A.check_cutouts(frames, cuts), [])
        self.assertEqual([c.read_bytes() for c in cuts], before)

    def test_lost_legs_are_found_by_size_and_feet(self):
        shapes = [(1000, 10, 90)] * 9
        shapes[2] = (600, 10, 90)  # much of the character gone
        shapes[4] = (950, 10, 60)  # the feet gone: shorter, its lowest point well above the others'
        lost = A.lost_frames(shapes, 100)
        self.assertEqual(sorted(lost), [2, 4])
        self.assertIn("area", lost[2])

    def test_lost_legs_are_put_back_from_the_ground_behind(self):
        frames, cuts = self.make(cut_legs=(7,))
        rows = A.check_cutouts(frames, cuts)
        self.assertEqual([(r["frame"], r["fix"]) for r in rows], [(7, "repaired")])
        with Image.open(cuts[7]) as im:
            alpha = im.getchannel("A")
            leg_x = 30 + (7 * 7) % 30
            self.assertGreater(alpha.getpixel((leg_x + 4, 80)), 200)  # the leg is back
            self.assertEqual(alpha.getpixel((5, 80)), 0)  # the shaded ground is not
            self.assertGreater(alpha.getpixel((45, 30)), 200)  # the body stays
        self.assertFalse(list((self.base / "cutouts").glob("*.redo.png")))

    def test_a_cutout_that_cannot_be_put_back_is_left_out_of_the_strip(self):
        frames, cuts = self.make(n=22, no_legs=(7,))
        rows = A.check_cutouts(frames, cuts)
        self.assertEqual([(r["frame"], r["fix"]) for r in rows], [(7, "dropped")])
        cells, picked, _ = A.strip_cells(cuts, 3, 64, dropped={7})
        self.assertNotIn(7, picked)
        self.assertEqual(picked, [1, 4, 10, 13, 16, 19])
        self.assertEqual(len(cells), 6)


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class AnimateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = make_root(self.base)
        self.env = mock.patch.dict(os.environ, fake_ltx(self.base))
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self._tmp.cleanup()

    def run_it(self, **kw):
        opts = {"prompt": "The soldier faces right and walks to the right.", "frames": 25, "size": 256}
        opts.update(kw)
        return A.animate(self.root, "op", **opts)

    def test_frames_cutouts_strip_and_version(self):
        doc = self.run_it()
        work = self.root / "out" / "op" / "flat" / "anim" / "walk"
        self.assertEqual(len(list((work / "frames").glob("*.png"))), 25)
        cuts = sorted((work / "cutouts").glob("*.png"))
        self.assertEqual(len(cuts), 25)
        with Image.open(cuts[5]) as im:
            self.assertEqual(im.mode, "RGBA")
            alpha = im.getchannel("A")
            self.assertEqual(alpha.getpixel((0, 0)), 0)  # the white ground is gone
            self.assertGreater(alpha.getextrema()[1], 200)  # the skeleton's lines stay
        for name in ("guide.png", "skeleton.mp4", "clip.mp4", "control.json", "anim.json"):
            self.assertTrue((work / name).is_file(), name)

        # The strip: frames 1, 4, ... 22 (8 of them), one cell each, in the 2D stage's motion-strip slot.
        sheets = self.root / "out" / "op" / "flat" / "sheets"
        manifest = json.loads((sheets / "manifest.json").read_text())
        row = manifest["actions"]["walk"]
        self.assertEqual(row["frames"], 8)
        self.assertEqual(row["from_frames"], [1, 4, 7, 10, 13, 16, 19, 22])
        self.assertEqual(row["made_by"], "animate")
        self.assertEqual(row["fps"], 8)
        with Image.open(sheets / "walk.png") as strip:
            self.assertEqual(strip.mode, "RGBA")
            self.assertEqual(strip.size, (row["cell"]["w"] * 8, row["cell"]["h"]))
        view = json.loads((self.root / "out" / "op" / "flat" / "view.json").read_text())
        self.assertEqual(view["center"], "sheets/walk.png")

        # What made it, and the licence's disclosure.
        self.assertTrue(doc["machine_generated"])
        self.assertIn("machine-generated", doc["disclosure"])
        self.assertIn("LTX-2 Community License", doc["licence"])
        self.assertEqual(doc["skeleton"], "walk_side")
        self.assertEqual(doc["frames"], 25)
        self.assertIsInstance(doc["lost_cutouts"], list)

        # A new 2D Version holds all of it.
        man = json.loads((self.root / "out" / "op" / "flow.json").read_text())
        flat = man["stages"]["flat"]
        take = next(t for t in flat["takes"] if t["id"] == flat["pick"])
        self.assertIn("flat/sheets/walk.png", take["files"])
        self.assertIn("flat/anim/walk/cutouts/0010.png", take["files"])
        self.assertIn("flat/anim/walk/frames/0010.png", take["files"])
        self.assertIn("Animate", take["note"])
        self.assertTrue((self.root / "out" / "op" / "takes" / "flat" / take["id"] / "flat/sheets/walk.png").is_file())

    def test_ltx_runs_isolated_offline_from_local_paths(self):
        self.run_it()
        call = json.loads((self.base / "ltx-call.json").read_text())
        self.assertEqual(call["isolated"], 1)
        self.assertEqual(call["offline"], "1")
        argv = call["argv"]
        self.assertEqual(argv[0], "ic-lora")
        for flag in ("--model", "--gemma", "--lora"):
            self.assertTrue(Path(argv[argv.index(flag) + 1]).exists(), flag)
        for flag in ("--video-conditioning", "--image"):  # built in the scratch folder, moved in after
            self.assertIn("/out/op/.animate/walk/", argv[argv.index(flag) + 1], flag)
        self.assertEqual(argv[argv.index("-f") + 1], "25")
        self.assertEqual(argv[argv.index("--frame-rate") + 1], "24")
        self.assertIn("faces right", argv[argv.index("--prompt") + 1])

    def test_side_view_is_a_second_guide(self):
        # A side view on white (a different ground from the first picture's light grey), as a cut-out would be.
        side = Image.new("RGBA", (300, 500), (0, 0, 0, 0))
        for y in range(80, 460):
            for x in range(130, 170):
                side.putpixel((x, y), (60, 80, 50, 255))
        (self.root / "out" / "op" / "flat").mkdir(parents=True, exist_ok=True)
        side.save(self.root / "out" / "op" / "flat" / "side.png")
        doc = self.run_it(side="flat/side.png", side_frame=16, side_strength=0.7)
        argv = json.loads((self.base / "ltx-call.json").read_text())["argv"]
        at = [i for i, a in enumerate(argv) if a == "--image"]
        self.assertEqual(len(at), 2)
        self.assertTrue(argv[at[0] + 1].endswith("/guide.png"))
        self.assertEqual(argv[at[0] + 2:at[0] + 4], ["0", "1.0"])
        self.assertTrue(argv[at[1] + 1].endswith("/.animate/walk/side.png"))
        self.assertEqual(argv[at[1] + 2:at[1] + 4], ["16", "0.7"])
        self.assertEqual(doc["side"], {"picture": "flat/side.png", "frame": 16, "strength": 0.7,
                                       "fitted_to_figure": True})
        work = self.root / "out" / "op" / "flat" / "anim" / "walk"

        def figure(path):
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                self.assertEqual(rgb.size, (256, 256))
                self.assertEqual(rgb.getpixel((3, 3)), (222, 224, 221))  # on the first picture's ground
                return rgb.point(lambda v: 255 if v < 150 else 0).convert("L").getbbox()

        front, side_box = figure(work / "guide.png"), figure(work / "side.png")
        # The side view's character stands where the first picture's does, as tall, on the same foot line.
        self.assertLessEqual(abs((side_box[3] - side_box[1]) - (front[3] - front[1])), 2)
        self.assertLessEqual(abs(side_box[3] - front[3]), 2)
        self.assertLessEqual(abs((side_box[0] + side_box[2]) - (front[0] + front[2])), 4)
        self.assertEqual(json.loads((work / "anim.json").read_text())["side"]["frame"], 16)

    def test_without_a_side_view_there_is_one_guide(self):
        doc = self.run_it()
        argv = json.loads((self.base / "ltx-call.json").read_text())["argv"]
        self.assertEqual(argv.count("--image"), 1)
        self.assertIsNone(doc["side"])

    def test_side_view_refusals(self):
        with self.assertRaises(A.AnimateError):
            self.run_it(side="../../etc/hosts")
        with self.assertRaisesRegex(A.AnimateError, "side-frame"):
            self.run_it(side="input.png", side_frame=25)
        with self.assertRaisesRegex(A.AnimateError, "side-frame"):
            self.run_it(side="input.png", side_frame=0)
        with self.assertRaisesRegex(A.AnimateError, "side-strength"):
            self.run_it(side="input.png", side_strength=0)

    def draw_side(self, rel: str, nose: str = "left") -> None:
        """A side view on white: a dark body with a nose sticking out on one side, so a mirror shows."""
        im = Image.new("RGB", (300, 500), (255, 255, 255))
        for y in range(80, 460):
            for x in range(130, 170):
                im.putpixel((x, y), (60, 80, 50))
        nx = range(100, 130) if nose == "left" else range(170, 200)
        for y in range(100, 130):
            for x in nx:
                im.putpixel((x, y), (60, 80, 50))
        path = self.root / "out" / "op" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path)

    def nose_side(self) -> str:
        """Which side of the fitted side guide's body the nose is on."""
        with Image.open(self.root / "out" / "op" / "flat" / "anim" / "walk" / "side.png") as im:
            rgb = im.convert("RGB")
        dark = rgb.point(lambda v: 255 if v < 150 else 0).convert("L")
        body = dark.getbbox()
        w, h = rgb.size
        # A row through the nose (drawn 5% to 13% down the figure).
        row = body[1] + (body[3] - body[1]) // 10
        xs = [x for x in range(w) if dark.getpixel((x, row))]
        mid = [x for x in range(w) if dark.getpixel((x, body[3] - 5))]
        centre = (min(mid) + max(mid)) / 2
        return "left" if (min(xs) + max(xs)) / 2 < centre else "right"

    def test_side_with_no_picture_finds_the_assets_own(self):
        self.draw_side("flat/views/side.png", nose="left")  # the pixel sheet's side view: front to the LEFT
        doc = self.run_it(side=A.FIND_SIDE, prompt="The soldier faces left and walks to the left.")
        argv = json.loads((self.base / "ltx-call.json").read_text())["argv"]
        self.assertEqual(argv.count("--image"), 2)
        self.assertEqual(doc["side"]["picture"], "flat/views/side.png")
        self.assertEqual((doc["side"]["found"], doc["side"]["facing"], doc["side"]["moves"]), (True, "left", "left"))
        self.assertFalse(doc["side"]["mirrored"])
        self.assertEqual(self.nose_side(), "left")

    def test_side_view_facing_the_wrong_way_is_mirrored(self):
        self.draw_side("character/views/side.png", nose="left")  # the sprite stack's copy, front to the LEFT
        doc = self.run_it(side=A.FIND_SIDE)  # "faces right and walks to the right"
        self.assertEqual(doc["side"]["picture"], "character/views/side.png")
        self.assertEqual((doc["side"]["facing"], doc["side"]["moves"], doc["side"]["mirrored"]),
                         ("left", "right", True))
        self.assertEqual(self.nose_side(), "right")
        # The drawn view itself is left as it was.
        with Image.open(self.root / "out" / "op" / "character" / "views" / "side.png") as im:
            self.assertEqual(im.getpixel((110, 110)), (60, 80, 50))

    def test_the_turnaround_profile_facing_the_move_is_picked(self):
        self.draw_side("flat/views/right.png", nose="right")
        self.draw_side("flat/views/left.png", nose="left")
        doc = self.run_it(side=A.FIND_SIDE, skeleton="walker_walk_left", prompt="The soldier walks.")
        self.assertEqual(doc["side"]["picture"], "flat/views/left.png")
        self.assertEqual((doc["side"]["moves"], doc["side"]["mirrored"]), ("left", False))
        doc = self.run_it(side=A.FIND_SIDE)
        self.assertEqual(doc["side"]["picture"], "flat/views/right.png")
        self.assertFalse(doc["side"]["mirrored"])
        # A left panel drawn head-on (the turnaround's 'side' flaw) is passed over: the right one, mirrored.
        (self.root / "out" / "op" / "flat" / "views.json").write_text(json.dumps({"flaws": ["side"]}))
        doc = self.run_it(side=A.FIND_SIDE, skeleton="walker_walk_left", prompt="The soldier walks.")
        self.assertEqual(doc["side"]["picture"], "flat/views/right.png")
        self.assertTrue(doc["side"]["mirrored"])
        self.assertEqual(self.nose_side(), "left")
        self.assertFalse((self.root / "out" / "op" / "flat" / "anim" / "walk" / "side-mirrored.png").exists())

    def test_side_with_no_picture_refusals(self):
        with self.assertRaisesRegex(A.AnimateError, "no side view of its own"):
            self.run_it(side=A.FIND_SIDE)
        self.draw_side("flat/views/side.png")
        with self.assertRaisesRegex(A.AnimateError, "moves left or right"):
            self.run_it(side=A.FIND_SIDE, prompt="The soldier walks on the spot.")

    def test_which_way_the_character_moves(self):
        self.assertEqual(A.moving_way("walker_run_left", "faces right"), "left")  # the skeleton's name wins
        self.assertEqual(A.moving_way("walk_side", "The soldier faces right and walks to the right."), "right")
        self.assertEqual(A.moving_way("walk_side", "She walks left, her left arm swinging"), "left")
        self.assertEqual(A.moving_way("walk_side", "a robot heading leftward"), "left")
        self.assertIsNone(A.moving_way("walk_side", "his left foot then his right foot"))
        self.assertIsNone(A.moving_way("walk_side", "faces left, then turns to the right"))

    def test_side_with_no_picture_on_the_command_line(self):
        with mock.patch.object(A, "animate", return_value={}) as run:
            self.assertEqual(A.main(["--root", str(self.root), "--stem", "op", "--prompt", "walks right",
                                     "--side", "--side-frame", "12"]), 0)
        self.assertEqual(run.call_args.kwargs["side"], A.FIND_SIDE)
        self.assertEqual(run.call_args.kwargs["side_frame"], 12)
        with mock.patch.object(A, "animate", return_value={}) as run:
            A.main(["--root", str(self.root), "--stem", "op", "--prompt", "walks right", "--side", "flat/x.png"])
        self.assertEqual(run.call_args.kwargs["side"], "flat/x.png")

    def test_skeleton_check(self):
        doc = self.run_it()
        check = doc["pose_check"]
        self.assertEqual(check["frames_compared"], 24)  # frame 0 is the picture, not a pose
        self.assertAlmostEqual(check["mean_joint_error_pct"], 1.0, places=1)

    def test_skeleton_is_fitted_to_the_picture(self):
        doc = self.run_it(skeleton="walker_walk_right", action="walk-right")
        work = self.root / "out" / "op" / "flat" / "anim" / "walk-right"
        control = json.loads((work / "control.json").read_text())
        self.assertTrue(control["fit"]["fitted"])
        self.assertEqual(doc["skeleton_fit"], control["fit"])
        self.assertGreater(control["fit"]["ratios"]["thigh"], 1.5)  # the operator's long legs
        f0 = control["frames"][0]["joints2d"]
        self.assertLess(abs(f0["root"][1] - OPERATOR["root"][1]), 0.01)
        self.assertNotAlmostEqual(f0["root"][1], SK.load(SK.named("walker_walk_right"))[0]["root"][1], places=2)
        self.assertFalse((work / "start").exists())
        self.assertFalse((work / "start.json").exists())

    def test_skeleton_is_used_as_shipped_when_vision_reads_no_body(self):
        with mock.patch.dict(os.environ, {"FAKE_BODYPOSE_NONE": "1"}):
            doc = self.run_it()
        self.assertFalse(doc["skeleton_fit"]["fitted"])
        control = json.loads((self.root / "out" / "op" / "flat" / "anim" / "walk" / "control.json").read_text())
        self.assertEqual(control["frames"][0]["joints2d"]["root"], SK.load(SK.named("walk_side"))[0]["root"])

    def test_skeleton_is_used_as_shipped_without_bodypose(self):
        with mock.patch.dict(os.environ, {"PIPELINE_BODYPOSE": str(self.base / "missing")}):
            doc = self.run_it(check=False)
        self.assertEqual(doc["skeleton_fit"], {"fitted": False, "why": "no bin/bodypose"})

    def test_guide_keeps_a_light_ground_and_whitens_magenta(self):
        g = self.root / "g.png"
        self.assertEqual(A.make_guide(self.root / "out" / "op" / "input.png", 256, g), (222, 224, 221))
        with Image.open(g) as im:
            self.assertEqual(im.size, (256, 256))
        mag = Image.new("RGB", (300, 400), (255, 0, 255))
        for y in range(100, 300):
            for x in range(120, 180):
                mag.putpixel((x, y), (30, 30, 200))
        mag.save(self.root / "m.png")
        self.assertEqual(A.make_guide(self.root / "m.png", 256, g), (255, 255, 255))
        with Image.open(g) as im:
            self.assertEqual(im.getpixel((5, 128)), (255, 255, 255))

    def test_guide_fitted_to_the_figure(self):
        """A code cycle's figure box: the character's own box scaled to it, feet on its ground, centred on it."""
        fig = {"top": 0.0909, "ground": 0.9091, "x": 0.5}
        g = self.root / "g.png"
        ground, shrink = A.fit_guide(add_front_view(self.root), 704, g, fig)
        self.assertEqual((ground, shrink), ((255, 255, 255), 1.0))
        with Image.open(g) as im:
            box = ImageChops.invert(im.convert("L")).point(lambda v: 255 if v > 40 else 0).getbbox()
        self.assertEqual(box[1], 64)  # top: 0.0909 of 704
        self.assertEqual(box[3], 640)  # ground: 0.9091 of 704 (exclusive)
        self.assertAlmostEqual((box[0] + box[2]) / 2, 352, delta=1)
        self.assertAlmostEqual(box[2] - box[0], 40 * 576 / 100, delta=2)  # the same shape, scaled
        # A character too wide for the frame at that height is made smaller, and says how much.
        wide = Image.new("RGBA", (400, 100), (10, 10, 10, 255))
        wide.save(self.root / "wide.png")
        _, shrink = A.fit_guide(self.root / "wide.png", 704, g, fig)
        self.assertAlmostEqual(shrink, 0.96 * 704 / (400 * 5.76), places=3)
        # The light-grey operator picture (no transparency): its ground is kept, the figure still fitted.
        self.assertEqual(A.fit_guide(self.root / "out" / "op" / "input.png", 704, g, fig)[0], (222, 224, 221))
        with Image.open(g) as im:
            self.assertEqual(im.getpixel((352, 300)), (60, 80, 50))
            self.assertEqual(im.getpixel((352, 50)), (222, 224, 221))

    def test_a_code_cycle(self):
        add_front_view(self.root)
        # no sprite-stack character here anyway; the refit to the picture's body (fit_control) is left out so the
        # cycle that went in can be compared with the shipped one
        with mock.patch.object(A, "cycle_maker", return_value=None), \
                mock.patch.object(A, "fit_control", side_effect=lambda root, c, work: (c, {"fitted": False})):
            doc = self.run_it(prompt="", cycle="run", facing="left")
        self.assertEqual(doc["skeleton"], "walker_run_left")
        self.assertEqual(doc["cycle"], {"name": "run", "facing": "left", "made_for": "the walker", "turn_frames": 16,
                                        "cycle_frames": 18})
        self.assertEqual(doc["picture"], "flat/views/front.png")  # it starts facing the camera
        self.assertIn("turns to face left", doc["prompt"])
        self.assertEqual(doc["fitted_to_figure"]["shrink"], 1.0)
        # The strip is the loop after the turn (frames 16 on), not the turn: 25 frames hold 16, 19 and 22 of it.
        self.assertEqual(doc["strip"]["loop"], [16, 18])
        self.assertEqual(doc["strip"]["from_frames"], [16, 19, 22])
        anim = self.root / "out" / "op" / "flat" / "anim" / "run"
        self.assertTrue((anim / "anim.json").is_file())
        control = json.loads((anim / "control.json").read_text())["frames"]
        shipped = SK.load(SK.named("walker_run_left"))
        self.assertEqual(len(control), 25)
        self.assertEqual(control[20]["joints2d"]["left_foot_joint"], shipped[20]["left_foot_joint"])
        with self.assertRaises(A.AnimateError):
            self.run_it(cycle="run", skeleton="walk_side")
        with self.assertRaises(SK.SkeletonError):
            self.run_it(cycle="dance")

    @unittest.skipUnless(CYCLE_MAKER, "needs bun and desktop/frontend/scripts/skeleton-cycle.ts")
    def test_a_sprite_stack_character_gets_its_own_cycle(self):
        add_front_view(self.root)
        (self.root / "out" / "op" / "stack").mkdir()
        (self.root / "out" / "op" / "stack" / "character.json").write_text(json.dumps(WALKER))
        with mock.patch.object(A, "cycle_maker", return_value=CYCLE_MAKER):
            self.assertEqual(A.offered(self.root, "op")["cycles"][0]["made_for"], "this character")
            doc = self.run_it(prompt="", cycle="attack", size=704)  # the size the shipped ones are made at
        self.assertEqual(doc["cycle"]["made_for"], "this character")
        self.assertFalse(doc["skeleton_fit"]["fitted"])  # its own body already: not refitted to the picture
        anim = self.root / "out" / "op" / "flat" / "anim" / "attack"
        self.assertTrue((anim / "cycle" / "attack_right.pose.json").is_file())
        # Made for the walker's own body, it is the walker's shipped attack, frame for frame.
        control = json.loads((anim / "control.json").read_text())["frames"]
        shipped = SK.load(SK.named("walker_attack_right"))
        self.assertEqual([f["joints2d"] for f in control], shipped[:25])

    def test_offered(self):
        with mock.patch.object(A, "cycle_maker", return_value=None):
            got = A.offered(self.root, "op")
        self.assertEqual([(c["cycle"], c["facing"]) for c in got["cycles"]],
                         [(c, f) for c in ("walk", "run", "attack") for f in ("right", "left")])
        self.assertEqual({c["made_for"] for c in got["cycles"]}, {"the walker"})
        self.assertIn("walk_side", got["skeletons"])
        self.assertNotIn("walker_walk_right", got["skeletons"])
        self.assertIsNone(got["front_picture"])
        add_front_view(self.root)
        self.assertEqual(A.offered(self.root, "op")["front_picture"], "flat/views/front.png")

    def test_refusals(self):
        with self.assertRaises(A.AnimateError):
            self.run_it(prompt="  ")
        with self.assertRaises(A.AnimateError):
            self.run_it(frames=24)
        with self.assertRaises(A.AnimateError):
            self.run_it(picture="../../etc/hosts")
        with mock.patch.dict(os.environ, {"PIPELINE_LTX_MODEL": str(self.base / "nowhere")}):
            with self.assertRaisesRegex(A.AnimateError, "bootstrap.sh animate"):
                self.run_it()

    def test_failed_ltx_keeps_the_last_version(self):
        self.run_it()
        man = json.loads((self.root / "out" / "op" / "flow.json").read_text())
        before = man["stages"]["flat"]["pick"]
        with mock.patch.dict(os.environ, {"FAKE_LTX_FAIL": "1"}):
            with self.assertRaisesRegex(A.AnimateError, "exit 3"):
                self.run_it(action="run")
        man = json.loads((self.root / "out" / "op" / "flow.json").read_text())
        self.assertEqual(man["stages"]["flat"]["pick"], before)
        # Nothing half-made is left among the 2D files, and the scratch folder is gone.
        self.assertFalse((self.root / "out" / "op" / "flat" / "anim" / "run").exists())
        self.assertFalse((self.root / "out" / "op" / ".animate" / "run").exists())
        self.assertEqual(flow.stage_files(self.root / "out" / "op", "flat"), man_files(self.root, before))


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = make_root(self.base)
        self.env = mock.patch.dict(os.environ, fake_ltx(self.base))
        self.env.start()
        self.factory = factory_http.Factory(self.root)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), factory_http.make_handler(self.factory))
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        job = self.factory._snapshot()
        if job and job.get("state") == "running":
            self.factory.cancel(job["job_id"])
            self.wait()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.env.stop()
        self._tmp.cleanup()

    def wait(self, timeout: float = 120) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            _, st = http_json(self.base_url + "/status")
            if st.get("state") != "running":
                return st
            time.sleep(0.2)
        self.fail("animate job did not finish")

    def post(self, body: dict, stem: str = "op"):
        return http_json(self.base_url + f"/2d/animate/{stem}", "POST", body)

    def test_bad_requests_leave_the_slot_idle(self):
        self.assertEqual(self.post({})[0], 400)
        self.assertEqual(self.post({"prompt": "walks right", "skeleton": "moonwalk"})[0], 400)
        self.assertEqual(self.post({"prompt": "walks right", "frames": 24})[0], 400)
        self.assertEqual(self.post({"prompt": "walks right", "picture": "/etc/hosts"})[0], 400)
        self.assertEqual(self.post({"prompt": "walks right", "side": "../x.png"})[0], 400)
        self.assertEqual(self.post({"prompt": "walks right", "side_frame": 0})[0], 400)
        self.assertEqual(self.post({"prompt": "walks right"}, stem="nobody")[0], 404)
        with mock.patch.dict(os.environ, {"PIPELINE_LTX_LORA": str(self.base / "nowhere")}):
            code, doc = self.post({"prompt": "walks right"})
        self.assertEqual(code, 400)
        self.assertIn("LoRA", doc["error"])
        self.assertEqual(http_json(self.base_url + "/status")[1]["state"], "idle")

    def test_offers_the_code_cycles(self):
        code, doc = http_json(self.base_url + "/2d/animate/op")
        self.assertEqual(code, 200, doc)
        self.assertEqual(len(doc["cycles"]), 6)
        self.assertEqual(doc["cycles"][0]["skeleton"], "walker_walk_right")
        self.assertEqual(http_json(self.base_url + "/2d/animate/nobody")[0], 404)

    def test_cycle_requests(self):
        argv = factory_http.animate_argv("py", self.root, "op", {"cycle": "attack", "facing": "left"})
        self.assertEqual(argv[argv.index("--cycle") + 1], "attack")
        self.assertEqual(argv[argv.index("--facing") + 1], "left")
        self.assertEqual(argv[argv.index("--action") + 1], "attack")
        self.assertIn("turns to face left", argv[argv.index("--prompt") + 1])
        self.assertNotIn("--skeleton", argv)
        self.assertEqual(self.post({"cycle": "dance"})[0], 400)
        self.assertEqual(self.post({"cycle": "walk", "facing": "up"})[0], 400)
        self.assertEqual(self.post({"cycle": "walk", "skeleton": "walk_side"})[0], 400)
        self.assertEqual(http_json(self.base_url + "/status")[1]["state"], "idle")

    def test_skeleton_list(self):
        code, doc = http_json(self.base_url + "/2d/skeletons")
        self.assertEqual(code, 200)
        self.assertEqual([s["name"] for s in doc["skeletons"]], SK.shipped())
        self.assertEqual(doc["default"], "walk_side")
        by = {s["name"]: s for s in doc["skeletons"]}
        self.assertEqual(by["walk_side"], {"name": "walk_side", "label": "Walk, side on", "action": "walk"})
        self.assertEqual(by["walker_run_left"]["label"], "Run, facing left")
        actions = [s["action"] for s in doc["skeletons"]]
        self.assertEqual(len(set(actions)), len(actions))  # one strip per skeleton: none overwrites another
        for a in actions:  # every default strip name is one POST /2d/animate takes
            self.assertEqual(factory_http.animate_argv("py", self.root, "op", {"prompt": "x", "action": a})[-1], a)

    def test_animate_job(self):
        code, doc = self.post({"prompt": "The soldier faces right and walks to the right.", "frames": 17,
                               "size": 256, "action": "walk"})
        self.assertEqual(code, 202, doc)
        self.assertEqual(doc["kind"], "animate")
        self.assertEqual(self.post({"prompt": "again"})[0], 409)
        st = self.wait()
        self.assertEqual(st["state"], "ok", st.get("log_tail"))
        keys = [s["key"] for s in st["progress"]["stages"]]
        self.assertEqual(keys, list(A.STAGES))
        self.assertEqual(st["progress"]["stages"][2]["label"], "Animate with LTX-2.3")
        log = (self.root / "out" / "op" / "animate.log").read_text()
        self.assertIn("note:", log.splitlines()[0])  # the draw model was freed first
        state = flow.state(self.root, "op")
        flat = next(s for s in state["stages"] if s["stage"] == "flat")
        self.assertTrue(flat["live"])

    def test_side_view_reaches_the_command_line(self):
        argv = factory_http.animate_argv("py", self.root, "op", {"prompt": "walks right", "side": "flat/side.png",
                                                                 "side_frame": 30})
        self.assertEqual(argv[argv.index("--side") + 1], "flat/side.png")
        self.assertEqual(argv[argv.index("--side-frame") + 1], "30")

    def test_own_side_view_reaches_the_command_line(self):
        argv = factory_http.animate_argv("py", self.root, "op", {"prompt": "walks right", "side": True,
                                                                 "side_frame": 30})
        self.assertEqual(argv[argv.index("--side") + 1], "--side-frame")  # --side with no picture
        self.assertNotIn("--side", factory_http.animate_argv("py", self.root, "op",
                                                             {"prompt": "walks right", "side": False}))

    def test_failed_job_says_why(self):
        with mock.patch.dict(os.environ, {"FAKE_LTX_FAIL": "1"}):
            code, _ = self.post({"prompt": "walks right", "frames": 17, "size": 256})
            self.assertEqual(code, 202)
            st = self.wait()
        self.assertEqual(st["state"], "error")
        self.assertIn("LTX-2.3 stopped", st["error"])


if __name__ == "__main__":
    unittest.main()
