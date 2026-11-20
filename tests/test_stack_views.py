#!/usr/bin/env python3
"""A sprite stack carved from four hand-made pixel art views (scripts/stack_views.py). No image model."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stack_views  # noqa: E402

CLEAR = (0, 0, 0, 0)
TOP = (200, 40, 40, 255)
SIDE = (40, 160, 40, 255)
FRONT = (40, 40, 200, 255)
BACK = (220, 200, 40, 255)
INK = (20, 16, 24, 255)  # the dark outline
GLASS = (120, 200, 230, 255)
TYRE = (50, 50, 60, 255)


def view(rows: list[str], colors: dict[str, tuple[int, int, int, int]], margin: int = 1) -> np.ndarray:
    """A small RGBA picture from text: '.' is transparent, each other letter a colour. A transparent margin
    around it, as a drawn view has."""
    h, w = len(rows), len(rows[0])
    out = np.zeros((h + 2 * margin, w + 2 * margin, 4), np.uint8)
    for y, row in enumerate(rows):
        assert len(row) == w, row
        for x, ch in enumerate(row):
            out[y + margin, x + margin] = CLEAR if ch == "." else colors[ch]
    return out


def solid(w: int, h: int, ch: str) -> list[str]:
    return [ch * w] * h


def outlined(w: int, h: int, ch: str) -> list[str]:
    """A w x h rectangle of ch with a 1 px ink border."""
    rows = ["k" * w]
    rows += ["k" + ch * (w - 2) + "k" for _ in range(h - 2)]
    rows += ["k" * w]
    return rows


C = {"t": TOP, "s": SIDE, "f": FRONT, "b": BACK, "k": INK, "g": GLASS, "y": TYRE}


def px(a: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in a)


class BoxTest(unittest.TestCase):
    """A 4 wide, 5 long, 3 high box with one colour per view."""

    def setUp(self) -> None:
        self.views = {
            "top": view(solid(4, 5, "t"), C),
            "side": view(solid(5, 3, "s"), C),
            "front": view(solid(4, 3, "f"), C),
            "back": view(solid(4, 3, "b"), C),
        }
        self.solid, self.color = stack_views.carve(stack_views.prepare(self.views))

    def test_every_cell_is_solid(self) -> None:
        self.assertEqual(self.solid.shape, (3, 5, 4))  # layers, length (back -> front), width
        self.assertTrue(self.solid.all())

    def test_colours_pixel_by_pixel(self) -> None:
        # Top layer: the top view's colour everywhere.
        for z in range(5):
            for x in range(4):
                self.assertEqual(px(self.color[2, z, x]), TOP, (2, z, x))
        # Lower layers: row 0 (the back) the back view, the last row (the front) the front view, the rest the side.
        for h in (0, 1):
            for x in range(4):
                self.assertEqual(px(self.color[h, 0, x]), BACK, (h, 0, x))
                self.assertEqual(px(self.color[h, 4, x]), FRONT, (h, 4, x))
            for z in (1, 2, 3):
                for x in range(4):
                    self.assertEqual(px(self.color[h, z, x]), SIDE, (h, z, x))

    def test_stack_is_square_and_centred(self) -> None:
        result = stack_views.stack_views(stack_views.prepare(self.views))
        self.assertEqual(result["resolution"], 5)
        self.assertEqual(result["layers"], 3)
        sl = result["slices"][2]
        self.assertEqual(sl.shape, (5, 5, 4))
        # 4 wide in a 5 wide square: column 4 (the pad) is empty, columns 0-3 hold the box.
        self.assertTrue((sl[:, :4, 3] == 255).all())
        self.assertTrue((sl[:, 4, 3] == 0).all())


class OutlineTest(unittest.TestCase):
    def test_outline_is_recoloured_not_carved(self) -> None:
        views = {
            "top": view(outlined(5, 6, "t"), C),
            "side": view(outlined(6, 4, "s"), C),
            "front": view(outlined(5, 4, "f"), C),
            "back": view(outlined(5, 4, "b"), C),
        }
        sol, color = stack_views.carve(stack_views.prepare(views))
        self.assertEqual(sol.shape, (4, 6, 5))
        self.assertTrue(sol.all(), "the outline is part of the silhouette")
        used = {px(c) for c in color[sol]}
        self.assertNotIn(INK, used)
        self.assertEqual(used, {TOP, SIDE, FRONT, BACK})
        self.assertEqual(px(color[3, 0, 0]), TOP)  # a corner of the top layer, ink in the top view
        self.assertEqual(px(color[0, 2, 0]), SIDE)  # the side's bottom edge, ink in the side view

    def test_off_keeps_the_ink(self) -> None:
        views = {
            "top": view(outlined(5, 6, "t"), C),
            "side": view(outlined(6, 4, "s"), C),
            "front": view(outlined(5, 4, "f"), C),
            "back": view(outlined(5, 4, "b"), C),
        }
        _, color = stack_views.carve(stack_views.prepare(views, outline="off"))
        self.assertEqual(px(color[3, 0, 0]), INK)


class LShapeTest(unittest.TestCase):
    """Seen from the side an L: a tall block at the back, a low step at the front (nose at the left)."""

    def setUp(self) -> None:
        side = [
            "..sss",  # nose at the left: the front step is the left two columns
            "..sss",
            "sssss",
        ]
        self.views = {
            "top": view(solid(3, 5, "t"), C),
            "side": view(side, C),
            "front": view(solid(3, 3, "f"), C),
            "back": view(solid(3, 3, "b"), C),
        }
        self.solid, self.color = stack_views.carve(stack_views.prepare(self.views))

    def test_shape(self) -> None:
        # Bottom layer: all five rows. Layers 1 and 2: only the back three rows (z 0-2).
        self.assertTrue(self.solid[0].all())
        for h in (1, 2):
            self.assertTrue(self.solid[h, :3].all(), h)
            self.assertFalse(self.solid[h, 3:].any(), h)

    def test_colours(self) -> None:
        # The step's top (layer 0, rows 3-4) shows from above: top colour.
        for z in (3, 4):
            for x in range(3):
                self.assertEqual(px(self.color[0, z, x]), TOP, (0, z, x))
        # The tall block's front face (row 2) in layers 0-1: covered in layer 0 by the step, open in layer 1.
        for x in range(3):
            self.assertEqual(px(self.color[1, 2, x]), FRONT, (1, 2, x))
            self.assertEqual(px(self.color[0, 2, x]), SIDE, (0, 2, x))
            self.assertEqual(px(self.color[2, 2, x]), TOP, (2, 2, x))
            self.assertEqual(px(self.color[0, 0, x]), BACK, (0, 0, x))
            self.assertEqual(px(self.color[1, 0, x]), BACK, (1, 0, x))
        # The middle row of the tall block below its top: side colour.
        for h in (0, 1):
            self.assertEqual(px(self.color[h, 1, 1]), SIDE, (h, 1, 1))


class CarTest(unittest.TestCase):
    """A car-like shape: a long low body on two tyres, a narrower glass cabin on top, outlined like pixel art."""

    def setUp(self) -> None:
        # 7 wide, 10 long. The cabin (rows 3-6 of the top view) is glass and 5 wide.
        top = [
            "ttttttt",
            "ttttttt",
            "ttttttt",
            "tgggggt",
            "tgggggt",
            "tgggggt",
            "tgggggt",
            "ttttttt",
            "ttttttt",
            "ttttttt",
        ]
        # 10 long, 5 high, nose at the left. Cabin over columns 3-6, tyres at 1-2 and 7-8 under the body.
        side = [
            "...gggg...",
            "...gggg...",
            "ssssssssss",
            "ssssssssss",
            ".yy....yy.",
        ]
        # 7 wide, 5 high: the cabin is 5 wide, the tyres are the outer columns.
        front = [
            ".ggggg.",
            ".ggggg.",
            "fffffff",
            "fffffff",
            "yy...yy",
        ]
        back = [
            ".ggggg.",
            ".ggggg.",
            "bbbbbbb",
            "bbbbbbb",
            "yy...yy",
        ]
        self.views = {"top": view(top, C), "side": view(side, C), "front": view(front, C), "back": view(back, C)}
        self.solid, self.color = stack_views.carve(stack_views.prepare(self.views, outline="off"))

    def test_layer_shapes_pixel_by_pixel(self) -> None:
        self.assertEqual(self.solid.shape, (5, 10, 7))
        # Layer 0 (the tyres): the side's columns 1-2 and 7-8 are the nose end, so rows 7-8 and 1-2 (back -> front),
        # and only the outer two columns on each side.
        want0 = np.zeros((10, 7), bool)
        for z in (1, 2, 7, 8):
            want0[z, [0, 1, 5, 6]] = True
        np.testing.assert_array_equal(self.solid[0], want0)
        # Layers 1-2: the whole body.
        self.assertTrue(self.solid[1].all() and self.solid[2].all())
        # Layers 3-4: the cabin, rows 3-6 and columns 1-5.
        want3 = np.zeros((10, 7), bool)
        want3[3:7, 1:6] = True
        np.testing.assert_array_equal(self.solid[3], want3)
        np.testing.assert_array_equal(self.solid[4], want3)

    def test_colours(self) -> None:
        # The roof shows from above: the top view's glass.
        for z in range(3, 7):
            for x in range(1, 6):
                self.assertEqual(px(self.color[4, z, x]), GLASS)
        # The hood and the boot show from above in layer 2: the top view's red.
        for z in (0, 1, 2, 7, 8, 9):
            for x in range(7):
                self.assertEqual(px(self.color[2, z, x]), TOP, (z, x))
        # The bumpers below the top layer: front (last row) blue, back (row 0) yellow.
        for x in range(7):
            self.assertEqual(px(self.color[1, 9, x]), FRONT)
            self.assertEqual(px(self.color[1, 0, x]), BACK)
        # The body's flanks: the side view's green.
        for z in range(1, 9):
            self.assertEqual(px(self.color[1, z, 0]), SIDE)
            self.assertEqual(px(self.color[1, z, 6]), SIDE)
        # The tyres take the tyre colour.
        for z in (1, 2, 7, 8):
            for x in (0, 1, 5, 6):
                self.assertEqual(px(self.color[0, z, x]), TYRE, (z, x))

    def test_no_blended_colours(self) -> None:
        used = {px(c) for c in self.color[self.solid]}
        self.assertTrue(used <= {TOP, SIDE, FRONT, BACK, GLASS, TYRE}, used)


class FixtureCarTest(unittest.TestCase):
    """The outlined car in tests/fixtures/views_car.py, the stack opened in the app's Stack stage."""

    def test_carves_clean(self) -> None:
        sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
        import views_car

        views = {n: views_car.draw(rows) for n, rows in views_car.VIEWS.items()}
        result = stack_views.stack_views(stack_views.prepare(views))
        self.assertEqual((result["resolution"], result["layers"]), (22, 9))
        self.assertEqual(result["size"], {"width": 12, "length": 22, "height": 9})
        palette = {(*c, 255) for c in views_car.PALETTE.values()}
        used = {px(c) for sl in result["slices"] for c in sl.reshape(-1, 4) if c[3]}
        self.assertTrue(used <= palette, used - palette)  # no outline ink, nothing blended
        self.assertEqual(result["colors"], len(palette))
        # Padded 12 -> 22 wide: columns 5-16. Layer 4 (the body's top): headlights at the front, tail lights at the back.
        top = result["slices"][4]
        self.assertEqual(px(top[20, 6]), (*views_car.PALETTE["h"], 255))
        self.assertEqual(px(top[1, 6]), (*views_car.PALETTE["t"], 255))
        self.assertTrue((top[:, :5, 3] == 0).all() and (top[:, 17:, 3] == 0).all())
        # Layer 0: four tyres, 4 long and 3 wide each.
        self.assertEqual(int((result["slices"][0][:, :, 3] > 0).sum()), 4 * 4 * 3)


class OrientationTest(unittest.TestCase):
    def test_back_is_mirrored_and_side_nose_faces_the_camera(self) -> None:
        # The back view's left half is yellow, its right half ink: seen from behind, its left is the car's left,
        # which is the right of the top and front views (x = 2-3).
        back = ["bbkk", "bbkk", "bbkk"]
        # The side is taller at the nose (left column).
        side = ["s..", "sss", "sss"]
        views = {
            "top": view(solid(4, 3, "t"), C),
            "side": view(side, C),
            "front": view(solid(4, 3, "f"), C),
            "back": view(back, C),
        }
        sol, color = stack_views.carve(stack_views.prepare(views, outline="off"))
        # The tall end is the last row (it faces the camera).
        self.assertTrue(sol[2, 2].all())
        self.assertFalse(sol[2, :2].any())
        self.assertEqual(px(color[0, 0, 0]), INK)
        self.assertEqual(px(color[0, 0, 3]), BACK)


class ErrorsTest(unittest.TestCase):
    def test_mismatched_sizes_name_them(self) -> None:
        views = {
            "top": view(solid(4, 5, "t"), C),
            "side": view(solid(6, 3, "s"), C),
            "front": view(solid(4, 3, "f"), C),
            "back": view(solid(4, 3, "b"), C),
        }
        with self.assertRaises(stack_views.ViewsError) as e:
            stack_views.prepare(views)
        self.assertIn("side view is 6x3 but must be 5x3", str(e.exception))
        sol, _ = stack_views.carve(stack_views.prepare(views, fit=True))
        self.assertEqual(sol.shape, (3, 5, 4))

    def test_key_colour_ground(self) -> None:
        im = np.zeros((5, 6, 4), np.uint8)
        im[:] = (255, 0, 255, 255)
        im[1:4, 1:5] = TOP
        mask = stack_views.subject_mask(im, alpha=False)
        self.assertEqual(int(mask.sum()), 12)
        self.assertTrue(mask[1:4, 1:5].all())
        # An RGB picture (no alpha channel) is read the same way.
        rgb, alpha = stack_views._rgba(Image.fromarray(im[:, :, :3], "RGB"))
        self.assertFalse(alpha)
        np.testing.assert_array_equal(stack_views.subject_mask(rgb, alpha=alpha), mask)

    def test_view_filling_its_whole_picture(self) -> None:
        # No transparent margin: the outline on the picture's border is the object's, not the ground.
        views = {
            "top": view(outlined(5, 6, "t"), C, margin=0),
            "side": view(outlined(6, 4, "s"), C, margin=0),
            "front": view(outlined(5, 4, "f"), C, margin=0),
            "back": view(outlined(5, 4, "b"), C, margin=0),
        }
        sol, _ = stack_views.carve(stack_views.prepare(views))
        self.assertEqual(sol.shape, (4, 6, 5))
        self.assertTrue(sol.all())


class CommandTest(unittest.TestCase):
    def test_writes_the_stack_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "out" / "boxcar" / "flat" / "pixel"
            folder.mkdir(parents=True)
            views = {
                "top": view(outlined(6, 9, "t"), C),
                "side": view(outlined(9, 4, "s"), C),
                "front": view(outlined(6, 4, "f"), C),
                "back": view(outlined(6, 4, "b"), C),
            }
            for name, a in views.items():
                Image.fromarray(a, "RGBA").save(folder / f"{name}.png")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "stack_views.py"), "--stem", "boxcar", "--root", str(root)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            stack = root / "out" / "boxcar" / "stack"
            meta = json.loads((stack / "stack.json").read_text())
            self.assertEqual(meta["maker"], "views")
            self.assertEqual(meta["resolution"], 9)
            self.assertEqual(meta["layers"], 4)
            self.assertEqual(meta["source"], "flat/pixel")
            self.assertEqual(meta["size"], {"width": 6, "length": 9, "height": 4})
            d = meta["defaults"]
            self.assertEqual((d["texels"], d["contrast"], d["saturation"], d["brightness"]), (9, 1.0, 1.0, 1.0))
            sheet = np.array(Image.open(stack / "sheet.png").convert("RGBA"))
            self.assertEqual(sheet.shape, (9, 9 * 4, 4))
            # Layer 3 (the top), padded 6 -> 9 wide with the box in columns 1-6: all the top view's red.
            top = sheet[:, 27:36]
            self.assertTrue((top[:, 1:7] == np.array(TOP, np.uint8)).all())
            self.assertTrue((top[:, [0, 7, 8], 3] == 0).all())
            self.assertTrue((stack / "preview.png").is_file())

    def test_missing_view_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "stack_views.py"), "--stem", "nothing", "--root", tmp],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("no top view", proc.stderr)


if __name__ == "__main__":
    unittest.main()
