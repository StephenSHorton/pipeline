#!/usr/bin/env python3
"""generate.sh with stub binaries: the no-picture guard, turnaround views, the shape fallbacks and loud degradation,
a supplied mesh (--mesh), no paint (--paint off) and the views' photo baking over the drawn views.

Each test builds a throwaway root (scripts copied from the repo, stub hy3d / lift-subject / shape_mv.sh) so
generate.sh runs end to end in seconds without a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NO_PICTURE = "3D needs a picture. Draw or add one first."
REAL_SCRIPTS = ["generate.sh", "_common.sh", "mv_orient.py", "reduce_mesh.py", "reduce.sh", "inspect_glb.py",
                "psx_lowpoly.py", "key_spill.py"]

HY3D_STUB = r'''
import os, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image
args = sys.argv[1:]
root = Path(__file__).resolve().parents[1]
with open(root / "hy3d.calls", "a") as fh:
    fh.write(" ".join(args) + "\n")
out = Path(args[args.index("-o") + 1])
weights = args[args.index("--weights") + 1] if "--weights" in args else ""
bad = os.environ.get("HY3D_FAIL_WEIGHTS")
if bad and bad in weights:
    print("hy3d: cannot load " + weights, file=sys.stderr)
    sys.exit(1)
box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
if args[0] == "paint":
    uv = np.random.default_rng(0).random((len(box.vertices), 2))
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (64, 64), (200, 120, 40)))
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
box.export(out)
'''

LIFT_STUB = r'''
import sys
from PIL import Image
Image.open(sys.argv[1]).convert("RGBA").save(sys.argv[2])
'''

RETEXTURE_STUB = r'''
import json, shutil, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(__file__).resolve().parents[1]
(root / "retexture.args.json").write_text(json.dumps(args))
shutil.copy(args[0], args[args.index("-o") + 1])
'''

# scripts/workbench/unwrap_cli.py <in.glb> -o <out.glb> [--texture N]: UVs and a plain atlas.
UNWRAP_STUB = r'''
import json, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image
args = sys.argv[1:]
root = Path(__file__).resolve().parents[2]
(root / "unwrap.args.json").write_text(json.dumps(args))
mesh = trimesh.load(args[0], force="mesh")
size = int(args[args.index("--texture") + 1]) if "--texture" in args else 1024
uv = np.random.default_rng(0).random((len(mesh.vertices), 2))
mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (size, size), (128, 128, 128)))
mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
mesh.export(args[args.index("-o") + 1])
'''

SHAPE_MV_OK = r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "$*" >> "$ROOT/shape_mv.calls"
ls "$ROOT/out/$1/views/lifted" >> "$ROOT/shape_mv.calls"
"$ROOT/.venv/bin/python" -c "import sys, trimesh; trimesh.creation.box(extents=(1,1,1)).export(sys.argv[1])" "$2"
echo "SHAPE_MV_RESULT $2"
'''

SHAPE_MV_FAIL = r'''#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "$*" >> "$ROOT/shape_mv.calls"
echo "shape_mv: stub"
echo "SHAPE_MV_FAIL no_backend: stub backend"
exit 2
'''


def square_png(path: Path, size: int = 96, alpha: bool = True) -> Path:
    """A centred square subject on a transparent (or magenta) ground: a box seen from any side."""
    a = np.zeros((size, size, 4), np.uint8)
    a[..., 0], a[..., 2] = 255, 255
    a[16:80, 16:80] = (200, 120, 40, 255)
    if not alpha:
        a[..., 3] = 255
    Image.fromarray(a, "RGBA").save(path)
    return path


class GenerateViewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="gen-views-"))
        self.root = self.tmp / "root"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "bin").mkdir()
        (self.root / ".venv" / "bin").mkdir(parents=True)
        for name in REAL_SCRIPTS:
            shutil.copy(ROOT / "scripts" / name, self.root / "scripts" / name)
        py = self.root / ".venv" / "bin" / "python"
        py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        py.chmod(0o755)
        self.stub("bin/hy3d_stub.py", HY3D_STUB, exe=False)
        self.stub("bin/hy3d", f'#!/bin/sh\nexec "{py}" "$(dirname "$0")/hy3d_stub.py" "$@"\n')
        self.stub("bin/lift_stub.py", LIFT_STUB, exe=False)
        self.stub("bin/lift-subject", f'#!/bin/sh\nexec "{py}" "$(dirname "$0")/lift_stub.py" "$@"\n')
        (self.root / "bin" / "mlx.metallib").write_bytes(b"")
        self.stub("scripts/retexture.py", RETEXTURE_STUB, exe=False)
        self.stub("scripts/shape_mv.sh", SHAPE_MV_FAIL)
        v20 = self.root / "weights" / "tencent" / "Hunyuan3D-2" / "hunyuan3d-dit-v2-0"
        v20.mkdir(parents=True)
        (v20 / "model.fp16.safetensors").write_bytes(b"stub")
        (self.root / "weights" / "shape-large").mkdir(parents=True)
        unet = self.root / "weights" / "paint" / "hunyuan3d-paint-v2-0" / "unet"
        unet.mkdir(parents=True)
        (unet / "diffusion_pytorch_model.safetensors").write_bytes(b"stub")
        self.src = self.tmp / "in"
        self.src.mkdir()
        self.image = square_png(self.src / "pic.png", alpha=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stub(self, rel: str, text: str, exe: bool = True) -> Path:
        p = self.root / rel
        p.write_text(text)
        if exe:
            p.chmod(0o755)
        return p

    def views_json(self, ids=("front", "right", "back", "left"), consistent=True, bad: str | None = None) -> Path:
        flat = self.tmp / "asset" / "flat"
        (flat / "views").mkdir(parents=True, exist_ok=True)
        views = []
        for vid in ids:
            p = flat / "views" / f"{vid}.png"
            if vid == bad:
                p.write_text("not a picture")
            else:
                square_png(p)
            views.append({"id": vid, "file": f"flat/views/{vid}.png"})
        doc = {"version": 1, "naming": "hunyuan", "canonical": "right", "facing": "right",
               "views": views, "consistent": consistent, "flaws": [] if consistent else ["height spread 0.2"]}
        (flat / "views.json").write_text(json.dumps(doc))
        return flat / "views.json"

    def run_gen(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        e = {k: v for k, v in os.environ.items() if k not in {"FACTORY_ROOT", "DEGRADED_JSON"}}
        e["FACTORY_ROOT"] = str(self.root)  # no tee into the log
        e.update(env or {})
        return subprocess.run(["bash", str(self.root / "scripts" / "generate.sh"), "--stem", "t", "--no-stack",
                               *args], capture_output=True, text=True, env=e, timeout=300)

    def out(self, *parts: str) -> Path:
        return self.root.joinpath("out", "t", *parts)

    def degraded_lines(self, proc) -> list[str]:
        return [ln for ln in proc.stdout.splitlines() if ln.startswith("DEGRADED ")]

    def hy3d_calls(self) -> list[str]:
        p = self.root / "hy3d.calls"
        return p.read_text().splitlines() if p.exists() else []

    # --- the guard ----------------------------------------------------------------------------------------

    def assert_refused(self, proc) -> None:
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn(NO_PICTURE, proc.stderr)
        self.assertFalse(self.out().exists(), "a refused run must not start an asset")
        self.assertEqual(self.hy3d_calls(), [])

    def test_no_image_is_refused(self) -> None:
        self.assert_refused(self.run_gen())

    def test_missing_image_is_refused(self) -> None:
        self.assert_refused(self.run_gen("--image", str(self.src / "nope.png")))

    def test_text_file_named_png_is_refused(self) -> None:
        fake = self.src / "fake.png"
        fake.write_text("hello")
        self.assert_refused(self.run_gen("--image", str(fake)))

    def test_mesh_is_refused(self) -> None:
        mesh = self.src / "thing.glb"
        trimesh.creation.box().export(mesh)
        self.assert_refused(self.run_gen("--image", str(mesh)))

    def test_bad_ref_is_refused(self) -> None:
        self.assert_refused(self.run_gen("--image", str(self.image), "--ref", str(self.src / "missing.jpg")))
        txt = self.src / "notes.txt"
        txt.write_text("x")
        self.assert_refused(self.run_gen("--image", str(self.image), "--ref", str(txt)))

    def test_bad_view_file_is_refused(self) -> None:
        vj = self.views_json(bad="back")
        self.assert_refused(self.run_gen("--image", str(self.image), "--views-json", str(vj)))

    # --- turnaround views ---------------------------------------------------------------------------------

    def test_views_json_isolates_hunyuan_named_views_and_uses_2mv(self) -> None:
        self.stub("scripts/shape_mv.sh", SHAPE_MV_OK)
        vj = self.views_json()
        proc = self.run_gen("--image", str(self.image), "--views-json", str(vj), "--ref", str(self.image))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lifted = sorted(p.name for p in self.out("views", "lifted").iterdir())
        self.assertEqual(lifted, ["back.png", "front.png", "left.png", "ref-1.png", "right.png"])
        # shape_mv sees the four named views (and the ref, which it ignores by name).
        calls = (self.root / "shape_mv.calls").read_text()
        self.assertIn("right.png", calls)
        # The ref is not a shape view: no single-image hy3d call ran.
        self.assertEqual([c for c in self.hy3d_calls() if c.startswith("shape")], [])
        manifest = json.loads(self.out("views.json").read_text())
        roles = {v["id"]: v["role"] for v in manifest["views"]}
        self.assertEqual(roles, {"input": "shape", "front": "shape", "right": "shape", "back": "shape",
                                 "left": "shape", "ref-1": "ref"})
        shape = json.loads(self.out("shape.json").read_text())
        self.assertEqual(shape["mode"], "multiview")
        self.assertEqual(shape["model"], "hunyuan3d-2mv-turbo")
        self.assertEqual(shape["steps"], 5)
        self.assertEqual(shape["orient"]["canonical"], "right")
        self.assertGreaterEqual(shape["orient"]["iou"], 0.6)
        self.assertEqual(self.degraded_lines(proc), [])
        self.assertEqual(json.loads(self.out("degraded.json").read_text())["degraded"], [])
        self.assertEqual(json.loads(self.out("inspect.json").read_text())["degraded"], [])

    def test_views_despill_the_key_on_the_rim(self) -> None:
        self.stub("scripts/shape_mv.sh", SHAPE_MV_OK)
        vj = self.views_json()
        # A pink rim pixel next to the ground.
        p = vj.parent / "views" / "right.png"
        a = np.asarray(Image.open(p)).copy()
        a[16, 40] = (230, 60, 220, 255)
        Image.fromarray(a, "RGBA").save(p)
        proc = self.run_gen("--image", str(self.image), "--views-json", str(vj))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = np.asarray(Image.open(self.out("views", "lifted", "right.png")))
        r, g, b, _ = (int(x) for x in out[16, 40])
        self.assertLessEqual(min(r, b) - g, 0, "the rim keeps the key's tint")
        self.assertEqual(tuple(int(x) for x in out[48, 48][:3]), (200, 120, 40), "the inside is untouched")

    def test_see_through_parts_lose_the_key(self) -> None:
        # Glass in the middle of the subject shows the magenta key; brass beside it keeps its colour. The same goes
        # for the main picture when a draw's key.json sits beside it.
        self.stub("scripts/shape_mv.sh", SHAPE_MV_OK)
        vj = self.views_json()
        (vj.parent / "key.json").write_text(json.dumps({"rgb": [166, 25, 106], "tol": 96}))
        for p in (vj.parent / "views" / "right.png", vj.parent / "source.png"):
            if not p.exists():
                square_png(p)
            a = np.asarray(Image.open(p)).copy()
            a[40:56, 40:56] = (145, 61, 99, 255)
            Image.fromarray(a, "RGBA").save(p)
        proc = self.run_gen("--image", str(vj.parent / "source.png"), "--views-json", str(vj))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("key spill 256 px", proc.stdout)
        for out in (self.out("views", "lifted", "right.png"), self.out("lifted.png")):
            px = np.asarray(Image.open(out))
            r, g, b, _ = (int(x) for x in px[48, 48])
            self.assertLess(max(r, g, b) - min(r, g, b), 15, (out.name, r, g, b))
            self.assertEqual(tuple(int(x) for x in px[20, 20][:3]), (200, 120, 40), "brass is untouched")

    def test_shape_mv_failure_is_degraded_and_merged_into_inspect(self) -> None:
        vj = self.views_json()
        proc = self.run_gen("--image", str(self.image), "--views-json", str(vj))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lines = self.degraded_lines(proc)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("DEGRADED shape mv_failed: "), lines[0])
        self.assertIn("no_backend: stub backend", lines[0])
        deg = json.loads(self.out("degraded.json").read_text())["degraded"]
        self.assertEqual([(d["stage"], d["code"]) for d in deg], [("shape", "mv_failed")])
        self.assertEqual(json.loads(self.out("inspect.json").read_text())["degraded"], deg)
        shape_calls = [c for c in self.hy3d_calls() if c.startswith("shape")]
        self.assertEqual(len(shape_calls), 1)
        self.assertIn("hunyuan3d-dit-v2-0", shape_calls[0])
        self.assertIn("--steps 30", shape_calls[0])
        self.assertEqual(json.loads(self.out("shape.json").read_text())["mode"], "single")

    def test_no_views_is_one_loud_single_view_on_v2_0(self) -> None:
        proc = self.run_gen("--image", str(self.image))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lines = self.degraded_lines(proc)
        self.assertEqual(lines, ["DEGRADED shape single_view: one picture"])
        self.assertFalse((self.root / "shape_mv.calls").exists())
        shape_calls = [c for c in self.hy3d_calls() if c.startswith("shape")]
        self.assertEqual(len(shape_calls), 1)
        self.assertIn("hunyuan3d-dit-v2-0", shape_calls[0])
        self.assertIn("--steps 30", shape_calls[0])
        self.assertIn("--octree 256", shape_calls[0])
        shape = json.loads(self.out("shape.json").read_text())
        self.assertEqual((shape["model"], shape["steps"], shape["octree"]), ("hunyuan3d-dit-v2-0", 30, 256))
        self.assertEqual(len(json.loads(self.out("inspect.json").read_text())["degraded"]), 1)

    def test_the_factorys_fallbacks_reach_degraded_and_inspect_json(self) -> None:
        stale = "views stale: flat/views.json was drawn for another picture; using one picture"
        proc = self.run_gen("--image", str(self.image), "--degraded", stale)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # The factory printed its own line already; generate.sh only records it, and adds no second
        # "one picture" row: the stale entry already says why there is one picture.
        self.assertEqual(self.degraded_lines(proc), [])
        deg = json.loads(self.out("degraded.json").read_text())["degraded"]
        self.assertEqual([(d["stage"], d["code"]) for d in deg], [("views", "stale")])
        shape_calls = [c for c in self.hy3d_calls() if c.startswith("shape")]
        self.assertIn("hunyuan3d-dit-v2-0", shape_calls[-1])
        self.assertEqual(deg[0]["message"], "flat/views.json was drawn for another picture; using one picture")
        self.assertEqual(json.loads(self.out("inspect.json").read_text())["degraded"], deg)
        bad = self.run_gen("--image", str(self.image), "--degraded", "no colon here")
        self.assertEqual(bad.returncode, 2, bad.stdout + bad.stderr)

    def test_v2_0_unloadable_falls_back_to_shape_large_at_8_steps(self) -> None:
        proc = self.run_gen("--image", str(self.image), env={"HY3D_FAIL_WEIGHTS": "hunyuan3d-dit-v2-0"})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        codes = [ln.split(":")[0] for ln in self.degraded_lines(proc)]
        self.assertEqual(codes, ["DEGRADED shape single_view", "DEGRADED shape v2_0_unavailable"])
        last = [c for c in self.hy3d_calls() if c.startswith("shape")][-1]
        self.assertIn("shape-large", last)
        self.assertIn("--steps 8", last)

    def test_inconsistent_views_take_the_single_view_path(self) -> None:
        self.stub("scripts/shape_mv.sh", SHAPE_MV_OK)
        vj = self.views_json(consistent=False)
        proc = self.run_gen("--image", str(self.image), "--views-json", str(vj))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lines = self.degraded_lines(proc)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("DEGRADED shape single_view: turnaround views failed"), lines[0])
        self.assertFalse((self.root / "shape_mv.calls").exists())

    def test_octree_defaults_per_poly(self) -> None:
        for poly, octree in (("high", "384"), ("game", "128")):  # medium 256: test_no_views_...
            (self.root / "hy3d.calls").unlink(missing_ok=True)
            proc = self.run_gen("--image", str(self.image), "--poly", poly)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn(f"--octree {octree}", self.hy3d_calls()[0], poly)

    # --- texture ------------------------------------------------------------------------------------------

    def retexture_args(self) -> list[str]:
        return json.loads((self.root / "retexture.args.json").read_text())

    def test_retexture_gets_every_named_view_canonical_first_without_mirror(self) -> None:
        self.stub("scripts/shape_mv.sh", SHAPE_MV_OK)
        vj = self.views_json()
        proc = self.run_gen("--image", str(self.image), "--views-json", str(vj), "--quality", "textured")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        args = self.retexture_args()
        views = [Path(args[i + 1]).name for i, a in enumerate(args) if a == "--view"]
        self.assertEqual(views, ["right.png", "front.png", "back.png", "left.png"])
        self.assertEqual(args[args.index("--first-facing") + 1], "right")
        self.assertNotIn("--mirror", args)
        self.assertIn("remesh: off (opt-in)", proc.stdout)
        self.assertEqual(self.degraded_lines(proc), [])
        lod = json.loads(self.out("lod_textured_game.json").read_text())
        self.assertTrue(lod["has_uvs"])

    def test_retexture_mirrors_a_lone_side_view(self) -> None:
        proc = self.run_gen("--image", str(self.image), "--quality", "textured", "--facing", "right")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        args = self.retexture_args()
        views = [Path(args[i + 1]).name for i, a in enumerate(args) if a == "--view"]
        self.assertEqual(views, ["lifted.png"])
        self.assertEqual(args[args.index("--first-facing") + 1], "right")
        self.assertIn("--mirror", args)

    def test_the_views_photo_bakes_first_and_wins(self) -> None:
        # A photo draw's views.json names the photo it drew from: it is cut out and goes first with --first-wins,
        # keeping the canonical's --first-facing (the views were drawn in its pose, and a free fit of a front
        # photo can land on the back); the drawn views follow as before.
        self.stub("scripts/shape_mv.sh", SHAPE_MV_OK)
        vj = self.views_json()
        (self.root / "in").mkdir()
        photo = square_png(self.root / "in" / "cow.png")
        doc = json.loads(vj.read_text())
        doc["sheet"] = {"reference": {"file": "in/cow.png", "sha256": hashlib.sha256(photo.read_bytes()).hexdigest()}}
        vj.write_text(json.dumps(doc))
        proc = self.run_gen("--image", str(self.image), "--views-json", str(vj), "--quality", "textured")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        args = self.retexture_args()
        views = [Path(args[i + 1]).name for i, a in enumerate(args) if a == "--view"]
        self.assertEqual(views, ["photo.png", "right.png", "front.png", "back.png", "left.png"])
        self.assertEqual(args[args.index("--view") + 2], "--first-wins")
        self.assertEqual(args[args.index("--first-facing") + 1], "right")
        self.assertNotIn("--mirror", args)
        self.assertEqual(self.degraded_lines(proc), [])
        self.assertTrue(self.out("views", "lifted", "photo.png").is_file())
        roles = {v["id"]: v["role"] for v in json.loads(self.out("views.json").read_text())["views"]}
        self.assertEqual(roles["photo"], "reference")
        # A photo changed since the draw is not the subject any more: the drawn views bake alone, as before.
        square_png(photo, size=64)
        proc = self.run_gen("--image", str(self.image), "--views-json", str(vj), "--quality", "textured")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("changed since the draw", proc.stdout)
        self.assertEqual([ln.split(":")[0] for ln in self.degraded_lines(proc)], ["DEGRADED retexture photo_skipped"])
        deg = json.loads(self.out("degraded.json").read_text())["degraded"]
        self.assertIn("photo_skipped", [r["code"] for r in deg])
        args = self.retexture_args()
        self.assertEqual([Path(args[i + 1]).name for i, a in enumerate(args) if a == "--view"][0], "right.png")
        self.assertEqual(args[args.index("--first-facing") + 1], "right")
        self.assertNotIn("--first-wins", args)

    def test_a_supplied_mesh_is_the_shape(self) -> None:
        # The supplied mesh may be a product of the last run, which the clean-up removes: it must survive.
        prev = self.out("textured_photo.glb")
        prev.parent.mkdir(parents=True)
        trimesh.creation.cylinder(radius=0.5, height=1.0).export(prev)
        faces = len(trimesh.load(prev, force="mesh").faces)
        proc = self.run_gen("--image", str(self.image), "--mesh", str(prev))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual([c for c in self.hy3d_calls() if c.startswith("shape")], [])
        self.assertEqual(self.degraded_lines(proc), [], "a supplied mesh is no single-view fallback")
        shape = json.loads(self.out("shape.json").read_text())
        self.assertEqual((shape["model"], shape["mode"], shape["source"]), ("supplied", "supplied", str(prev)))
        self.assertIn("iou", shape["orient"])
        self.assertEqual(len(trimesh.load(self.out("mesh_hero.glb"), force="mesh").faces), faces)
        self.assertTrue(self.out("lifted.png").is_file(), "isolate still runs: the bake reads lifted.png")
        # The factory takes the extension in any case, so this script must too.
        upper = self.tmp / "Clay.GLB"
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(upper, file_type="glb")
        proc = self.run_gen("--image", str(self.image), "--mesh", str(upper))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_a_supplied_mesh_with_views_is_not_turned(self) -> None:
        self.stub("scripts/shape_mv.sh", SHAPE_MV_OK)
        clay = self.tmp / "clay.glb"
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(clay)
        vj = self.views_json()
        proc = self.run_gen("--image", str(self.image), "--views-json", str(vj), "--mesh", str(clay),
                            "--quality", "textured")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertFalse((self.root / "shape_mv.calls").exists())
        shape = json.loads(self.out("shape.json").read_text())
        self.assertEqual(shape["mode"], "supplied")
        self.assertEqual((shape["orient"]["yaw"], shape["orient"]["turned"]), (0, False))
        self.assertEqual(shape["orient"]["canonical"], "right")
        # The drawn views still bake, canonical first.
        args = self.retexture_args()
        self.assertEqual([Path(args[i + 1]).name for i, a in enumerate(args) if a == "--view"][0], "right.png")

    def test_a_bad_mesh_is_refused(self) -> None:
        txt = self.src / "thing.glb"
        txt.write_text("not a glb")
        for bad in (str(txt), str(self.src / "missing.glb"), str(self.image)):
            with self.subTest(mesh=bad):
                proc = self.run_gen("--image", str(self.image), "--mesh", bad)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn("--mesh wants a .glb", proc.stderr)
                self.assertFalse(self.out().exists())

    def test_paint_off_unwraps_then_bakes(self) -> None:
        (self.root / "scripts" / "workbench").mkdir()
        self.stub("scripts/workbench/unwrap_cli.py", UNWRAP_STUB, exe=False)
        shutil.rmtree(self.root / "weights" / "paint")  # no paint weights needed
        clay = self.tmp / "clay.glb"
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(clay)
        # No hy3d at all: a supplied mesh that is not painted loads no model.
        (self.root / "bin" / "hy3d").unlink()
        proc = self.run_gen("--image", str(self.image), "--mesh", str(clay), "--paint", "off", "--quality", "psx")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.hy3d_calls(), [])
        stages = next(ln for ln in proc.stdout.splitlines() if ln.startswith("STAGES ")).split()[1:]
        self.assertIn("unwrap", stages)
        self.assertNotIn("paint", stages)
        unwrap = json.loads((self.root / "unwrap.args.json").read_text())
        self.assertEqual(Path(unwrap[0]).name, "mesh_lookdev.glb")
        self.assertEqual(unwrap[1:], ["-o", str(self.out("textured.glb")), "--texture", "1024"])
        self.assertEqual(self.retexture_args()[0], str(self.out("textured.glb")))
        self.assertTrue(self.out("psx.glb").is_file())

    def test_paint_off_needs_the_unwrap_cli_and_no_pbr(self) -> None:
        proc = self.run_gen("--image", str(self.image), "--paint", "off", "--quality", "textured")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("scripts/workbench/unwrap_cli.py, which is missing", proc.stderr)
        self.assertFalse(self.out().exists())
        for args, word in ((("--paint", "off", "--quality", "pbr"), "pbr"), (("--paint", "maybe"), "--paint")):
            proc = self.run_gen("--image", str(self.image), *args)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn(word, proc.stderr)

    def test_missing_paint_weights_fail_without_downloading(self) -> None:
        shutil.rmtree(self.root / "weights" / "paint")
        proc = self.run_gen("--image", str(self.image), "--quality", "textured")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("run scripts/bootstrap.sh textured", proc.stderr)


if __name__ == "__main__":
    unittest.main()
