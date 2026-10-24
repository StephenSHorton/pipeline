#!/usr/bin/env python3
"""The four-view sheet (plan 1.1-1.3): split, cut-out, the hard checks, the facing mirror, the redraw cap,
views.json, and the backends' licences. Runs on the stub backend: no model, no vision model."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import facing  # noqa: E402
from flat2d import optional, pose, views  # noqa: E402
from flat2d.parts import ground_key  # noqa: E402

STUB_ENV = {"FLAT2D_STILL": "stub", "FLAT2D_STUB_STEP": "0", "FLAT2D_STUB_LOAD": "0", "FLAT2D_STUB_FLAWS": "0",
            "FLAT2D_PREVIEW": "off", "FLAT2D_SHEET": "1152x1152"}
NO_VISION = lambda crop: None  # noqa: E731 - Apple Vision off: the magenta key cuts the views out


def det(head, profile=True, confidence=0.9, how="vlm"):
    return {"head": head, "profile": profile, "confidence": confidence, "how": how}


def right(_png):
    return det("right")


def check(flaw="", kind="avian", layout="grid", detect=right, size=(1152, 1152)):
    im = optional._stub_sheet(7, *size, flaw)
    return im, views.check_panels(im, ground_key(im), kind, layout, vision=NO_VISION, detect=detect)


class SplitTests(unittest.TestCase):
    def test_strip_splits_at_the_magenta_gaps(self) -> None:
        im = optional._stub_sheet(3, 2048, 640)
        got = views.split_sheet(im, ground_key(im), "strip")
        self.assertEqual(got["how"], "gaps")
        self.assertEqual(list(got["boxes"]), list(pose.PANELS))
        edges = [b[0] for b in got["boxes"].values()]
        self.assertEqual(edges, sorted(edges))
        for pid, (x0, y0, x1, y1) in got["boxes"].items():
            self.assertEqual((y0, y1), (0, 640))
            self.assertLess(abs((x1 - x0) - 512), 160, pid)  # cut mid-gap, not on the quarter

    def test_grid_splits_row_by_row(self) -> None:
        im = optional._stub_sheet(3, 1152, 1152)
        got = views.split_sheet(im, ground_key(im), "grid")
        self.assertEqual(got["how"], "gaps")
        b = got["boxes"]
        self.assertEqual((b["front"][1], b["right"][1]), (0, 0))  # top row: front | right
        self.assertGreater(b["back"][1], 400)  # bottom row: back | left
        self.assertLess(b["front"][0], b["right"][0])
        self.assertLess(b["back"][0], b["left"][0])

    def test_no_gaps_falls_back_to_equal_quarters(self) -> None:
        im = Image.new("RGB", (2048, 640), (255, 0, 255))
        ImageDraw.Draw(im).rectangle((40, 200, 2000, 440), fill=(30, 200, 60))  # one long thing, no gaps
        got = views.split_sheet(im, ground_key(im), "strip")
        self.assertEqual(got["how"], "quarters")
        self.assertEqual([b[0] for b in got["boxes"].values()], [0, 512, 1024, 1536])

    def test_a_thin_full_width_line_is_dropped(self) -> None:
        im = optional._stub_sheet(3, 2048, 640)
        ImageDraw.Draw(im).rectangle((0, 600, 2047, 603), fill=(20, 20, 20))  # a drawn baseline
        key = ground_key(im)
        fg = ~views.ground_of(np.asarray(im), key)
        self.assertTrue(fg[601].all())
        self.assertFalse(views.drop_lines(fg)[601].any())
        self.assertEqual(views.split_sheet(im, key, "strip")["how"], "gaps")
        _, r = check(size=(2048, 640), layout="strip")
        self.assertEqual(r["flaws"], [])

    def test_a_tall_column_is_never_dropped(self) -> None:
        fg = np.zeros((100, 50), bool)
        fg[:, 20:22] = True  # two stacked subjects in a grid can fill a column: not a line
        self.assertTrue(views.drop_lines(fg)[:, 20:22].all())


class CheckTests(unittest.TestCase):
    def test_a_clean_sheet_passes_every_check(self) -> None:
        _, r = check()
        self.assertEqual(r["flaws"], [])
        c = r["checks"]
        self.assertLessEqual(c["height_spread"], views.HEIGHT_SPREAD)
        self.assertLessEqual(c["colour_de"], views.COLOUR_DE)
        self.assertGreaterEqual(c["side_iou"], views.SIDE_IOU)
        self.assertEqual(c["facing"]["how"], "vlm")
        self.assertTrue(all(a >= views.AREA_MIN for a in c["areas"].values()))
        self.assertIn("mask_key", r["notes"])  # vision off in the test: said, not silent

    def test_each_check_fails_on_its_own_flaw(self) -> None:
        for flaw, want in (("colour", "colour"), ("side", "side"), ("empty", "empty:front")):
            with self.subTest(flaw=flaw):
                _, r = check(flaw)
                self.assertIn(want, r["flaws"])

    def test_a_view_drawn_bigger_is_a_note_not_a_flaw(self) -> None:
        # Hunyuan3D-2mv's recenter and retexture's silhouette fit both frame each view on its own: size on the sheet
        # never reaches the mesh or the paint, so it never throws four good views away.
        _, r = check("height")
        self.assertGreater(r["checks"]["height_spread"], views.HEIGHT_SPREAD)
        self.assertNotIn("height", r["flaws"])
        self.assertIn("height", r["notes"])
        self.assertIn("height", views.NOTE_TEXT)

    def test_a_props_thin_handle_is_not_a_height_flaw(self) -> None:
        # A lantern's bail stands up in the side views only: a 6 px wire, 0.12 of the cell tall.
        def handled():
            im = optional._stub_sheet(7, 1152, 1152)
            d = ImageDraw.Draw(im)
            for ox, oy in ((576, 0), (576, 576)):  # right and left cells of the grid
                d.rectangle((ox + 285, oy + 30, ox + 291, oy + 150), fill=optional._stub_color(7))
            return im
        im = handled()
        prop = views.check_panels(im, ground_key(im), None, "grid", vision=NO_VISION)
        self.assertGreater(prop["checks"]["height_spread"], views.HEIGHT_SPREAD)
        self.assertLessEqual(prop["checks"]["body_height_spread"], views.HEIGHT_SPREAD)
        self.assertNotIn("height", prop["notes"])
        creature = views.check_panels(im, ground_key(im), "avian", "grid", vision=NO_VISION, detect=right)
        self.assertIn("height", creature["notes"])  # a creature's thin parts (legs) are part of its height
        _, real = check("height", kind=None)
        self.assertIn("height", real["notes"])  # a prop drawn smaller in one view is still said

    def test_two_subjects_in_one_panel(self) -> None:
        im = optional._stub_sheet(7, 1152, 1152)
        ImageDraw.Draw(im).ellipse((420, 60, 540, 180), fill=(20, 200, 40))  # a second thing beside the front view
        r = views.check_panels(im, ground_key(im), None, "grid", vision=NO_VISION)
        self.assertIn("pieces:front", r["flaws"])

    def test_one_subject_cut_by_a_thin_gap_is_not_two(self) -> None:
        # the tractor: its drawbar taken for a line cut the back and side views in two across a 9 px gap
        im = optional._stub_sheet(7, 1152, 1152)
        r0 = views.check_panels(im, ground_key(im), None, "grid", vision=NO_VISION)
        x0, y0, x1, y1 = r0["views"][0]["bbox"]
        mid = (y0 + y1) // 2
        ImageDraw.Draw(im).rectangle((0, mid, 575, mid + 8), fill=im.getpixel((3, 3)))
        r = views.check_panels(im, ground_key(im), None, "grid", vision=NO_VISION)
        self.assertNotIn("pieces:front", r["flaws"])

    def test_thresholds_are_the_plan(self) -> None:
        self.assertEqual((views.AREA_MIN, views.HEIGHT_SPREAD, views.COLOUR_DE, views.SIDE_IOU),
                         (0.02, 0.06, 18.0, 0.6))

    def test_lab_is_cie(self) -> None:
        lab = views.to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]]))
        np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.05)
        np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.05)
        np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)

    def test_only_one_vision_call_on_the_right_panel(self) -> None:
        seen = []

        def detect(png):
            seen.append(Image.open(png).size)
            return det("right")

        im, r = check(detect=detect)
        self.assertEqual(len(seen), 1)
        x0, y0, x1, y1 = r["split"]["boxes"]["right"]
        self.assertEqual(seen[0], (x1 - x0, y1 - y0))

    def test_a_prop_has_no_vision_call(self) -> None:
        with mock.patch.object(facing, "detect_image") as detect:
            _, r = check(kind=None, detect=None)
        detect.assert_not_called()
        self.assertEqual(r["checks"]["facing"]["how"], "not_needed")

    def test_head_left_mirrors_every_panel_and_keeps_the_labels(self) -> None:
        im, plain = check()
        _, r = check(detect=lambda p: det("left"))
        self.assertEqual(r["flaws"], [])
        self.assertTrue(r["mirrored"])
        fc = r["checks"]["facing"]
        self.assertEqual((fc["how"], fc["head_before"], fc["head"]), ("mirrored", "left", "right"))
        for pid in pose.PANELS:
            self.assertEqual(ImageOps.mirror(plain["panels"][pid]).tobytes(), r["panels"][pid].tobytes(), pid)
            np.testing.assert_array_equal(np.fliplr(plain["masks"][pid]), r["masks"][pid])
        self.assertEqual([v["id"] for v in r["views"]], list(pose.PANELS))
        w = r["views"][1]["cell"][2] - r["views"][1]["cell"][0]
        a, b = plain["views"][1]["bbox"], r["views"][1]["bbox"]
        self.assertEqual(b, [w - a[2], a[1], w - a[0], a[3]])

    def test_a_head_on_or_three_quarter_right_panel_is_a_flaw(self) -> None:
        _, r = check(detect=lambda p: det("camera", profile=False))
        self.assertIn("facing", r["flaws"])
        self.assertIn("profile", r["flaws"])
        self.assertFalse(r["mirrored"])

    def test_vision_model_unavailable_is_a_hard_flaw_never_a_pass(self) -> None:
        _, r = check(detect=lambda p: det(None, None, 0.0, how="unavailable"))
        self.assertEqual(r["checks"]["facing"]["how"], "unavailable")
        self.assertIn("facing_unavailable", r["flaws"])
        self.assertEqual(views.redrawable(r["flaws"]), [])  # a new seed cannot bring it back
        verdict = facing.judge({"how": "unavailable"}, "right")
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["flaws"], ["facing_unavailable"])

    def test_cloudy_ground_is_a_note(self) -> None:
        im = optional._stub_sheet(7, 1152, 1152)
        rgb = np.asarray(im, dtype=np.int16)
        ground = (rgb == (255, 0, 255)).all(-1)
        ramp = np.linspace(0, 70, 1152, dtype=np.int16)[:, None]
        rgb[..., 0] = np.where(ground, rgb[..., 0] - ramp, rgb[..., 0])
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        r = views.check_panels(im, ground_key(im), "avian", "grid", vision=NO_VISION, detect=right)
        self.assertIn("cloudy", r["notes"])
        self.assertEqual(r["flaws"], [])


class GenerateViewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "out").mkdir()
        self.env = mock.patch.dict("os.environ", {**STUB_ENV, "FLAT2D_LIFT": str(self.root / "no-lift")})
        self.env.start()
        for k in ("FLAT2D_STUB_SHEET_FLAW", "FLAT2D_STUB_SHEET_FLAWS"):
            os.environ.pop(k, None)
        optional._STUB_LOADED = False
        self.drawn: list[tuple] = []
        real = optional.still_image

        def still(prompt, seed=1, width=1024, height=1024):
            self.drawn.append((prompt, seed, width, height))
            return real(prompt, seed=seed, width=width, height=height)

        self.still = mock.patch.object(optional, "still_image", side_effect=still)
        self.still.start()

    def tearDown(self) -> None:
        self.still.stop()
        self.env.stop()
        self._tmp.cleanup()

    def flat(self, stem="chick") -> Path:
        return self.root / "out" / stem / "flat"

    def draw(self, answers=(det("right"),), intent=None, prompt="A PS1 style chicken", stem="chick"):
        seq = list(answers)
        with mock.patch.object(facing, "detect_image", side_effect=lambda p: seq.pop(0) if len(seq) > 1 else seq[0]):
            return optional.draw(self.root, stem, prompt, intent=intent or {"output": "ps1"})

    def test_views_json_schema(self) -> None:
        saved = self.draw()
        doc = json.loads((self.flat() / "views.json").read_text())
        for k in views.VIEWS_KEYS:
            self.assertIn(k, doc)
        self.assertEqual((doc["version"], doc["naming"], doc["canonical"], doc["facing"]),
                         (1, "hunyuan", "right", "right"))
        src = (self.flat() / "source.png").read_bytes()
        self.assertEqual(doc["source_sha256"], hashlib.sha256(src).hexdigest())
        self.assertEqual([v["id"] for v in doc["views"]], ["front", "right", "back", "left"])
        for v in doc["views"]:
            self.assertEqual(v["file"], f"flat/views/{v['id']}.png")
            self.assertTrue((self.root / "out" / "chick" / v["file"]).is_file())
            self.assertEqual(len(v["bbox"]), 4)
            self.assertEqual(len(v["key"]["rgb"]), 3)  # each view's own ground key, for its despill
            self.assertGreater(v["key"]["tol"], 0)
            self.assertGreater(v["height_frac"], 0)
        self.assertTrue(doc["consistent"])
        self.assertEqual(doc["flaws"], [])
        for k in ("height_spread", "colour_de", "side_iou", "facing"):
            self.assertIn(k, doc["checks"])
        sheet = doc["sheet"]
        self.assertEqual((sheet["w"], sheet["h"], sheet["redraws"]), (1152, 1152, 0))
        self.assertEqual((sheet["model"], sheet["licence"]), ("stub", "none"))
        self.assertIn("seed", sheet)
        self.assertTrue((self.flat() / "sheet.png").is_file())
        # The canonical (right) panel is source.png, and key.json points at the views.
        right = Image.open(self.flat() / "views" / "right.png").convert("RGB")
        self.assertEqual(Image.open(self.flat() / "source.png").convert("RGB").tobytes(), right.tobytes())
        key = json.loads((self.flat() / "key.json").read_text())
        self.assertEqual(key["views"], "flat/views.json")
        self.assertEqual((key["pose"], key["facing"], key["canonical"]), ("profile_right", "right", True))
        self.assertEqual(saved["views"], "flat/views.json")
        self.assertEqual(len(self.drawn), 1)
        self.assertIn("View 2 (top right)", self.drawn[0][0])
        self.assertEqual(self.drawn[0][2:], (1152, 1152))

    def test_a_prop_sheet_is_front_canonical(self) -> None:
        with mock.patch.object(facing, "detect_image") as detect:
            optional.draw(self.root, "lamp", "a brass lantern", intent={"output": "ps1"})
        detect.assert_not_called()
        doc = json.loads((self.flat("lamp") / "views.json").read_text())
        self.assertEqual((doc["canonical"], doc["facing"]), ("front", "camera"))
        front = Image.open(self.flat("lamp") / "views" / "front.png").convert("RGB")
        self.assertEqual(Image.open(self.flat("lamp") / "source.png").convert("RGB").tobytes(), front.tobytes())
        self.assertNotIn("canonical", json.loads((self.flat("lamp") / "key.json").read_text()))

    def test_one_redraw_for_a_hard_flaw(self) -> None:
        os.environ.update({"FLAT2D_STUB_SHEET_FLAW": "colour", "FLAT2D_STUB_SHEET_FLAWS": "1"})
        saved = self.draw()
        self.assertEqual(len(self.drawn), 2)
        self.assertEqual(saved["redrawn"], ["colour"])
        doc = json.loads((self.flat() / "views.json").read_text())
        self.assertTrue(doc["consistent"])
        self.assertEqual(doc["sheet"]["redraws"], 1)

    def test_never_more_than_one_redraw(self) -> None:
        os.environ.update({"FLAT2D_STUB_SHEET_FLAW": "side", "FLAT2D_STUB_SHEET_FLAWS": "9"})
        self.draw()
        self.assertEqual(len(self.drawn), 2)  # the first sheet and ONE redraw
        doc = json.loads((self.flat() / "views.json").read_text())
        self.assertFalse(doc["consistent"])
        self.assertIn("side", doc["flaws"])
        key = json.loads((self.flat() / "key.json").read_text())
        self.assertFalse(key["canonical"])

    def test_no_redraw_for_a_cloudy_ground(self) -> None:
        os.environ["FLAT2D_STUB_FLAWS"] = "5"
        saved = self.draw()
        self.assertEqual(len(self.drawn), 1)
        doc = json.loads((self.flat() / "views.json").read_text())
        self.assertIn("cloudy", doc["notes"])
        self.assertTrue(doc["consistent"])
        self.assertEqual(saved["redrawn"], [])

    def test_no_redraw_when_the_vision_model_is_down(self) -> None:
        saved = self.draw([det(None, None, 0.0, how="unavailable")])
        self.assertEqual(len(self.drawn), 1)
        doc = json.loads((self.flat() / "views.json").read_text())
        self.assertFalse(doc["consistent"])
        self.assertEqual(doc["checks"]["facing"]["how"], "unavailable")
        self.assertIn(("draw", "facing_unchecked"), [(d["stage"], d["code"]) for d in doc["degraded"]])
        self.assertEqual(saved["degraded"], doc["degraded"])

    def test_head_left_sheet_is_mirrored_in_the_files(self) -> None:
        self.draw([det("left")])
        doc = json.loads((self.flat() / "views.json").read_text())
        self.assertTrue(doc["mirrored"])
        self.assertEqual(doc["checks"]["facing"]["how"], "mirrored")
        sheet = Image.open(self.flat() / "sheet.png").convert("RGB")  # as drawn
        x0, y0, x1, y1 = doc["views"][1]["cell"]
        right = Image.open(self.flat() / "views" / "right.png").convert("RGB")
        self.assertEqual(ImageOps.mirror(sheet.crop((x0, y0, x1, y1))).tobytes(), right.tobytes())

    def test_views_1_keeps_the_single_draw(self) -> None:
        self.draw()  # a sheet first: its files must not outlive a single still
        with mock.patch.object(facing, "detect_image", return_value=det("right")):
            saved = optional.draw(self.root, "chick", "a chicken", intent={"output": "ps1"}, views=1)
        self.assertNotIn("views", saved)
        self.assertEqual(self.drawn[-1][2:], (1024, 1024))
        for name in ("views.json", "sheet.png", "views"):
            self.assertFalse((self.flat() / name).exists(), name)
        self.assertNotIn("views", json.loads((self.flat() / "key.json").read_text()))

    def test_a_2d_only_draw_is_one_still(self) -> None:
        self.assertFalse(optional.wants_views(None))
        self.assertFalse(optional.wants_views({"output": "sprite"}))
        self.assertTrue(optional.wants_views({"output": "ps1"}))
        self.assertFalse(optional.wants_views({"output": "ps1"}, views=1))
        self.assertFalse(optional.wants_views({"output": "model", "views": 1}))

    def test_a_non_default_backend_is_degraded(self) -> None:
        with mock.patch("flat2d.optional.still_backend", return_value="z-image"), mock.patch(
            "flat2d.optional.zimage_still", side_effect=lambda p, seed, width, height: optional._stub_sheet(seed, width, height)
        ):
            self.still.stop()
            try:
                self.draw()
            finally:
                self.still.start()
        doc = json.loads((self.flat() / "views.json").read_text())
        self.assertIn(("draw", "backend"), [(d["stage"], d["code"]) for d in doc["degraded"]])
        self.assertEqual((doc["sheet"]["model"], doc["sheet"]["licence"]), ("z-image-turbo", "apache-2.0"))


class PhotoDrawTests(unittest.TestCase):
    """A photo the user added is a reference the four views are drawn FROM (the edit model; the stub here), never a
    view: flat/source.png is the canonical drawn panel, and views.json records the photo and the likeness ask."""

    setUp_views = GenerateViewsTests.setUp
    tearDown_views = GenerateViewsTests.tearDown
    flat = GenerateViewsTests.flat

    def setUp(self) -> None:
        self.setUp_views()
        (self.root / "in").mkdir()
        self.photo = self.root / "in" / "cow.jpg"
        Image.new("RGB", (640, 480), (120, 100, 80)).save(self.photo)
        self.edits: list[tuple] = []
        real = optional.edit_image

        def edit(prompt, reference, seed=1, width=1024, height=1024):
            self.edits.append((prompt, str(reference), width, height))
            return real(prompt, reference, seed=seed, width=width, height=height)

        self.edit = mock.patch.object(optional, "edit_image", side_effect=edit)
        self.edit.start()

    def tearDown(self) -> None:
        self.edit.stop()
        self.tearDown_views()

    @staticmethod
    def codes(saved):
        """The degraded codes but the test machine's own (no Apple Vision here: views cut out on the key)."""
        return [(d["stage"], d["code"]) for d in saved["degraded"] if d["code"] != "mask_key"]

    def photo_draw(self, prompt="", asks=("YES",), intent=None):
        seq = list(asks)

        def ask(_png, question, keep=False):
            if not seq:
                raise AssertionError(f"unexpected vision ask: {question[:40]}")
            got = seq.pop(0)
            if isinstance(got, Exception):
                raise got
            return got

        with mock.patch.object(facing, "ask", side_effect=ask), \
                mock.patch.object(facing, "detect_image", return_value=det("right")):
            saved = optional.draw(self.root, "cow", prompt, intent=intent or {"output": "ps1", "kind": "quadruped"},
                                  reference=self.photo)
        self.assertEqual(seq, [], "every scripted vision answer was asked for")
        return saved

    def test_the_views_are_drawn_from_the_photo_never_the_photo_itself(self) -> None:
        saved = self.photo_draw("a dairy cow")
        self.assertEqual(self.drawn, [], "a photo draw never runs the text-to-image model")
        self.assertEqual(len(self.edits), 1)
        prompt, ref, w, h = self.edits[0]
        self.assertEqual((ref, w, h), (str(self.photo), 1152, 1152))
        self.assertTrue(prompt.startswith(optional.PHOTO_LEAD))
        self.assertIn("a dairy cow.", prompt)
        src = (self.flat("cow") / "source.png").read_bytes()
        self.assertNotEqual(hashlib.sha256(src).hexdigest(), hashlib.sha256(self.photo.read_bytes()).hexdigest())
        doc = json.loads((self.flat("cow") / "views.json").read_text())
        self.assertEqual(doc["source_sha256"], hashlib.sha256(src).hexdigest())
        self.assertTrue(doc["consistent"])
        ref = doc["sheet"]["reference"]
        self.assertEqual(ref["file"], "in/cow.jpg")
        self.assertEqual(ref["sha256"], hashlib.sha256(self.photo.read_bytes()).hexdigest())
        self.assertIs(ref["likeness"]["same"], True)
        self.assertEqual((doc["sheet"]["backend"], doc["sheet"]["licence"]), ("stub", "none"))
        self.assertEqual(saved["reference"], "in/cow.jpg")
        self.assertEqual(self.codes(saved), [])

    def test_no_words_the_vision_model_names_the_photo(self) -> None:
        saved = self.photo_draw("", asks=("A black and white dairy cow.", "YES"))
        self.assertEqual(saved["prompt"], "A black and white dairy cow")
        self.assertIn("A black and white dairy cow.", self.edits[0][0])
        self.assertEqual((self.flat("cow") / "prompt.txt").read_text().strip(), "A black and white dairy cow")

    def test_a_drift_is_redrawn_then_kept_degraded_not_inconsistent(self) -> None:
        saved = self.photo_draw("a dairy cow", asks=("NO", "No."))
        self.assertEqual(len(self.edits), 2)
        self.assertIn("drift", saved["redrawn"])
        self.assertIn(("views", "drift"), [(d["stage"], d["code"]) for d in saved["degraded"]])
        doc = json.loads((self.flat("cow") / "views.json").read_text())
        self.assertTrue(doc["consistent"], "the views agree with each other; drift is not a views flaw")

    def test_a_drift_fixed_by_the_redraw_keeps_the_redraw(self) -> None:
        saved = self.photo_draw("a dairy cow", asks=("NO", "YES"))
        self.assertEqual(len(self.edits), 2)
        self.assertEqual(self.codes(saved), [])
        doc = json.loads((self.flat("cow") / "views.json").read_text())
        self.assertIs(doc["sheet"]["reference"]["likeness"]["same"], True)

    def test_no_vision_model_is_loud_never_a_pass(self) -> None:
        gone = facing.VLMUnavailable("vision model unavailable (off)")
        saved = self.photo_draw("", asks=(gone, gone))
        codes = self.codes(saved)
        self.assertIn(("draw", "caption_unavailable"), codes)
        self.assertIn(("views", "likeness_unchecked"), codes)
        self.assertEqual(saved["prompt"], "the subject of the picture")

    def test_no_photo_draw_model_uses_the_photo_as_one_picture_loudly(self) -> None:
        self.edit.stop()
        try:
            with mock.patch("flat2d.sheet.edit_backend", return_value=""):
                saved = optional.draw(self.root, "cow", "a cow", intent={"output": "ps1"}, reference=self.photo)
        finally:
            self.edit.start()
        self.assertEqual([(d["stage"], d["code"]) for d in saved["degraded"]], [("draw", "no_photo_draw")])
        self.assertFalse((self.flat("cow") / "views.json").exists())
        self.assertTrue((self.flat("cow") / "source.png").is_file())

    def test_caption_cleanup(self) -> None:
        from flat2d.photo import clean_caption

        self.assertEqual(clean_caption('"The subject is a brass lantern."'), "a brass lantern")
        self.assertEqual(clean_caption("a red barn\nwith a roof"), "a red barn")
        self.assertIsNone(clean_caption(""))
        self.assertIsNone(clean_caption("x" * 80))
        self.assertIsNone(clean_caption("1234"))

    def test_reference_must_be_a_picture_under_in_or_out(self) -> None:
        from flat2d.photo import reference_path

        self.assertEqual(reference_path(self.root, "in/cow.jpg"), self.photo.resolve())
        with self.assertRaises(ValueError):
            reference_path(self.root, "../cow.jpg")
        with self.assertRaises(ValueError):
            reference_path(self.root, "")
        with self.assertRaises(FileNotFoundError):
            reference_path(self.root, "in/missing.jpg")
        (self.root / "in" / "notes.png").write_text("not a picture")
        with self.assertRaises(ValueError):
            reference_path(self.root, "in/notes.png")


class PromptTests(unittest.TestCase):
    def test_sheet_prompt(self) -> None:
        p = pose.sheet_prompt("A PS1 style chicken", "avian")
        self.assertTrue(p.startswith("A PS1 style chicken. Character turnaround sheet in a 2 by 2 grid"))
        for text in ("View 1 (top left): front", "View 2 (top right): the creature turned to face RIGHT",
                     "View 3 (bottom left): back", "View 4 (bottom right): the creature turned to face LEFT",
                     "#FF00FF", "Exactly four views", "wings folded"):
            self.assertIn(text, p)
        self.assertNotIn("baseline", p.lower())
        self.assertNotIn("same ground", p.lower())
        strip = pose.sheet_prompt("a lantern", None, "strip")
        self.assertIn("in one row", strip)
        self.assertIn("floats in the air", strip)
        self.assertIn("View 2: right side view", strip)
        self.assertNotIn("baseline", strip.lower())

    def test_canonical_view(self) -> None:
        self.assertEqual(pose.canonical_view("avian"), "right")
        self.assertEqual(pose.canonical_view("quadruped"), "right")
        self.assertEqual(pose.canonical_view("humanoid"), "front")
        self.assertEqual(pose.canonical_view(None), "front")


class UnloadTests(unittest.TestCase):
    def test_unload_frees_the_model_and_clears_the_mlx_cache(self) -> None:
        import types

        calls = []
        mx = types.SimpleNamespace(clear_cache=lambda: calls.append("clear"))
        mlx_pkg = types.ModuleType("mlx")
        mlx_pkg.core = mx
        with mock.patch.dict(sys.modules, {"mlx": mlx_pkg, "mlx.core": mx}), mock.patch.object(
            optional, "_Q2512", object()
        ), mock.patch.object(optional, "_Q2512_RELAY", object()), mock.patch.object(optional, "_STUB_LOADED", False):
            got = optional.unload()
            self.assertIsNone(optional._Q2512)
            self.assertIsNone(optional._Q2512_RELAY)
            self.assertFalse(optional.model_loaded())
            self.assertEqual(optional.unload()["freed"], [])  # nothing loaded: still fine
        self.assertEqual(got["freed"], ["qwen-2512"])
        self.assertEqual(calls, ["clear", "clear"])


class LightningScheduleTests(unittest.TestCase):
    def test_lightning_sigmas(self) -> None:
        from flat2d.turbo import lightning_sigmas

        s = lightning_sigmas(8)
        self.assertEqual(len(s), 9)
        self.assertAlmostEqual(s[0], 1.0)
        self.assertEqual(s[-1], 0.0)
        self.assertTrue(all(a > b for a, b in zip(s, s[1:])))
        self.assertAlmostEqual(s[-2], 3 * 0.125 / (1 + 2 * 0.125))  # shift 3 on the last linspace node


if __name__ == "__main__":
    unittest.main()
