#!/usr/bin/env python3
"""A sprite stack straight from the concept picture, no mesh."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import factory_http  # noqa: E402
import flow  # noqa: E402
import stack_image  # noqa: E402


def disc_still(path: Path, color=(200, 60, 40)) -> None:
    im = Image.new("RGB", (120, 120), (255, 0, 255))
    ImageDraw.Draw(im).ellipse((20, 20, 99, 99), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    (path.parent / "key.json").write_text(json.dumps({"rgb": [255, 0, 255], "tol": 36}))


class StackPictureTest(unittest.TestCase):
    def test_disc_becomes_a_ball(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        yy, xx = np.mgrid[:80, :80]
        mask[(yy - 40) ** 2 + (xx - 40) ** 2 <= 30**2] = True
        im = Image.new("RGBA", (80, 80), (10, 120, 200, 255))
        result = stack_image.stack_picture(im, mask, res=64)
        slices = result["slices"]
        self.assertEqual(result["layers"], len(slices))
        # Widest in the middle layer, thin at the top and bottom, round in plan view.
        widths = [int((s[:, :, 3] > 0).any(axis=0).sum()) for s in slices]
        depths = [int((s[:, :, 3] > 0).any(axis=1).sum()) for s in slices]
        mid = len(slices) // 2
        self.assertGreater(widths[mid], widths[0] * 3)
        self.assertLess(abs(depths[mid] - widths[mid]), widths[mid] * 0.25)
        self.assertEqual(tuple(slices[mid][32, 32, :3]), (10, 120, 200))

    def test_nearness_thickens_the_near_side(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:35, 5:35] = True
        im = Image.new("RGBA", (40, 40), (90, 90, 90, 255))
        near = np.zeros((40, 40), dtype=np.float32)
        near[:, 20:] = 1.0
        half = stack_image.half_thickness(mask, near)
        self.assertGreater(half[20, 26], half[20, 13])

    def test_thin_bar_stays_thin(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        mask[10:70, 38:43] = True  # a blade five pixels wide
        half = stack_image.half_thickness(mask, None)
        self.assertLessEqual(half.max(), 3.0)
        self.assertGreater(half[40, 40], 1.5)

    def test_edge_has_no_key_color(self) -> None:
        # A soft-edged disc on magenta, cut out with its blended rim, as Vision does.
        big = Image.new("RGB", (480, 480), (255, 0, 255))
        ImageDraw.Draw(big).ellipse((80, 80, 399, 399), fill=(40, 160, 70))
        im = big.resize((120, 120), Image.LANCZOS).convert("RGBA")
        px = np.asarray(im, dtype=np.int32)[..., :3]
        mask = np.abs(px - [255, 0, 255]).sum(-1) > 12
        def pink(image) -> int:
            result = stack_image.stack_picture(image, stack_image.main_piece(mask), res=64)
            n = 0
            for sl in result["slices"]:
                c = sl[sl[..., 3] > 0][:, :3].astype(int)
                n += int(((c[:, 0] > 150) & (c[:, 2] > 150) & (c[:, 1] < 120)).sum())
            return n
        self.assertGreater(pink(im), 0)
        clean = stack_image.clean_colors(im, mask, ((255, 0, 255), 36))
        self.assertEqual(pink(clean), 0)
        self.assertEqual(tuple(np.asarray(clean)[60, 60, :3]), (40, 160, 70))

    def test_specks_are_dropped(self) -> None:
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True
        mask[2, 2] = True
        self.assertFalse(stack_image.main_piece(mask)[2, 2])
        with self.assertRaises(stack_image.StackError):
            stack_image.main_piece(np.zeros((10, 10), dtype=bool))


class StackRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.symlink(ROOT / "scripts", self.root / "scripts")
        self.factory = factory_http.Factory(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def post(self, stem: str) -> tuple[int, dict]:
        env = {"FLAT2D_LIFT": str(self.root / "no-lift"), "STACK_DEPTH": "off"}
        with mock.patch.dict(os.environ, env):
            status, raw, _ = factory_http.dispatch(self.factory, "POST", f"/stack/{stem}", {}, b"{}")
        return status, json.loads(raw)

    def test_builds_stack_and_take_from_concept(self) -> None:
        disc_still(self.root / "out" / "lamp" / "flat" / "source.png")
        flow.snapshot(self.root, "lamp", "concept")
        status, data = self.post("lamp")
        self.assertEqual(status, 200, data)
        meta = json.loads((self.root / "out" / "lamp" / "stack" / "stack.json").read_text())
        self.assertEqual(meta["maker"], "picture")
        self.assertEqual(meta["source"], "flat/source.png")
        self.assertGreater(meta["layers"], 50)
        with Image.open(self.root / "out" / "lamp" / "stack" / "sheet.png") as sheet:
            self.assertEqual(sheet.size, (160 * meta["layers"], 160))
        stack = next(s for s in flow.state(self.root, "lamp")["stages"] if s["stage"] == "stack")
        self.assertEqual(stack["takes"][0]["from"], {"concept": "t1"})
        self.assertFalse(stack["unsaved"])

    def test_no_picture(self) -> None:
        (self.root / "out" / "empty").mkdir(parents=True)
        status, data = self.post("empty")
        self.assertEqual(status, 400)
        self.assertIn("no concept picture", data["error"])
        status, data = self.post("missing")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
