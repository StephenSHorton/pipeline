#!/usr/bin/env python3
"""The pixel art sheet (flat2d/pixelart.py, pose.pixel_sheet_prompt, optional.generate_pixel_views): its prompt, the
split and size checks, and the cleaner that turns a soft drawing of fake pixels back into true pixel art. A synthetic
sheet stands in for the model: a known 32-cell car, each cell drawn 13.7 px wide, blurred and noised."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import flow  # noqa: E402
from flat2d import optional, pixelart, pose  # noqa: E402
from flat2d.parts import ground_key  # noqa: E402

OUT, BODY, GLASS, RED, TYRE = (20, 20, 30), (40, 90, 200), (150, 200, 240), (200, 40, 40), (60, 60, 60)
L, W, H = 32, 14, 10
MAGENTA = (255, 0, 255)


def car_views() -> dict[str, np.ndarray]:
    def view(w, h):
        a = np.zeros((h, w, 3), np.uint8)
        a[:] = BODY
        a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = OUT
        return a

    top, side, front, back = view(W, L), view(L, H), view(W, H), view(W, H)
    top[10:18, 3:11] = GLASS
    side[2:5, 10:20] = GLASS
    side[7:9, 4:9] = TYRE
    side[7:9, 23:28] = TYRE
    front[2:5, 3:11] = GLASS
    back[3:5, 2:5] = RED  # a small distinct colour: 6 cells of 1,200
    return {"top": top, "side": side, "front": front, "back": back}


def soft_sheet(cells: dict[str, np.ndarray], cell: float = 13.7, blur: float = 1.6, seed: int = 1,
               stretch: dict[str, tuple[int, int]] | None = None) -> Image.Image:
    """The views drawn the way the model draws pixel art: fake pixels `cell` px wide, soft rims, a little noise."""
    rng = np.random.default_rng(seed)
    sheet = Image.new("RGB", (1152, 1152), MAGENTA)
    for pid, (cx, cy) in {"top": (288, 288), "side": (864, 288), "front": (288, 864), "back": (864, 864)}.items():
        a = cells[pid]
        h, w = a.shape[:2]
        if stretch and pid in stretch:
            w, h = stretch[pid]
        im = Image.fromarray(a).resize((round(w * cell), round(h * cell)), Image.Resampling.NEAREST)
        im = im.filter(ImageFilter.GaussianBlur(blur))
        arr = np.asarray(im).astype(int) + rng.integers(-8, 9, (im.height, im.width, 3))
        sheet.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), (cx - im.width // 2, cy - im.height // 2))
    return sheet


class PromptTest(unittest.TestCase):
    def test_the_layout_and_style_are_asked_for(self) -> None:
        p = pose.pixel_sheet_prompt("a small blue 90s sports race car")
        self.assertTrue(p.startswith("a small blue 90s sports race car."))
        for words in ("2 by 2 grid", "View 1 (top left): top view", "pointing DOWN toward the bottom edge",
                      "View 2 (top right): side view", "pointing LEFT toward the left edge",
                      "View 3 (bottom left): front view", "View 4 (bottom right): back view", "orthographic",
                      "same scale", "1 pixel dark outline", "solid flat colour fills", "no soft edges",
                      "No shadows", "no ground", "no text", "no grid lines", "magenta #FF00FF"):
            self.assertIn(words, p)

    def test_a_chosen_size_is_asked_for(self) -> None:
        self.assertIn("about 32 pixels long", pose.pixel_sheet_prompt("a car"))
        self.assertIn("about 128 pixels long", pose.pixel_sheet_prompt("a car", 128))
        self.assertNotIn("32 pixels", pose.pixel_sheet_prompt("a car", 96))
        self.assertEqual(pose.pixel_size(None), pose.PIXEL_SIZE)
        self.assertEqual(pose.pixel_size("96"), 96)
        for bad in ("big", 8, 1000, "9.5"):
            with self.assertRaises(ValueError):
                pose.pixel_size(bad)

    def test_the_clean_style_has_no_pixel_wording(self) -> None:
        pixel = pose.pixel_sheet_prompt("a small blue 90s sports race car", 96)
        clean = pose.pixel_sheet_prompt("a small blue 90s sports race car", 96, "clean")
        self.assertNotIn("pixel", clean.lower())
        self.assertNotIn("96", clean)
        for words in ("2 by 2 grid", "View 1 (top left): top view", "View 2 (top right): side view",
                      "View 3 (bottom left): front view", "View 4 (bottom right): back view", "orthographic",
                      "same scale", "level, facing the viewer", "hard edges", "solid flat colour fills",
                      "no lighting", "no ground line", "No shadows", "magenta #FF00FF", "digital game art"):
            self.assertIn(words, clean)
        self.assertEqual(pose.pixel_sheet_prompt("a car", 96, None), pose.pixel_sheet_prompt("a car", 96))
        self.assertIn("about 96 pixels long", pixel)
        self.assertEqual(pose.pixel_style(" Clean "), "clean")
        for bad in ("photo", 3):
            with self.assertRaises(ValueError):
                pose.pixel_style(bad)

    def test_front_and_back_are_asked_level_and_to_scale(self) -> None:
        p = pose.pixel_sheet_prompt("a car")
        self.assertIn("level, facing the viewer, its top not visible", p)
        self.assertEqual(p.count("exactly as wide as the top view and exactly as tall as the side view"), 2)

    def test_the_turnaround_is_unchanged(self) -> None:
        self.assertEqual(pose.PANELS, ("front", "right", "back", "left"))
        self.assertNotIn("pixel", pose.sheet_prompt("a lantern", None, "grid").lower())


class CleanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = car_views()

    def clean(self, sheet: Image.Image, allow: tuple[str, ...] = (), target: int = pixelart.TARGET) -> dict:
        key = ground_key(sheet)
        checked = pixelart.check_sheet(sheet, key)
        self.assertEqual([f for f in checked["flaws"] if f not in allow], [])
        return pixelart.clean(checked["panels"], key, target=target)

    def test_a_soft_drawing_comes_back_as_its_true_pixels(self) -> None:
        got = self.clean(soft_sheet(self.cells))
        self.assertEqual(got["grid"]["how"], "found")
        self.assertAlmostEqual(got["grid"]["cell_px"], 13.7, delta=0.3)
        for pid, want in self.cells.items():
            im = np.asarray(got["views"][pid])
            self.assertEqual(im.shape[:2], want.shape[:2], pid)
            self.assertTrue((im[..., 3] == 255).all(), pid)
            err = np.abs(im[..., :3].astype(int) - want.astype(int)).sum(-1)
            self.assertLessEqual(int(err.max()), 24, pid)  # every cell its own colour, the red lights kept
        self.assertEqual(got["flaws"], [])
        self.assertEqual(len(got["palette"]), 5)

    def test_cells_are_never_blended(self) -> None:
        got = self.clean(soft_sheet(self.cells))
        pal = {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in got["palette"]}
        for im in got["views"].values():
            colours = {tuple(c) for c in np.asarray(im)[..., :3].reshape(-1, 3)}
            self.assertLessEqual(colours, pal)

    def test_the_palette_is_capped(self) -> None:
        rng = np.random.default_rng(3)
        views = {"a": (rng.integers(0, 256, (60, 60, 3)).astype(np.uint8), np.ones((60, 60), bool))}
        self.assertLessEqual(len(pixelart.build_palette(views, 24)), 24)

    def test_a_view_drawn_a_cell_long_is_fitted_with_nearest_pixels(self) -> None:
        got = self.clean(soft_sheet(self.cells, stretch={"top": (W, L + 1)}))
        self.assertEqual(got["sizes"]["drawn_cells"]["top"], [W, L + 1])
        self.assertEqual(got["sizes"]["final"]["top"], [W, L])
        self.assertIn("top", got["sizes"]["fitted"])

    def test_views_that_disagree_are_a_flaw(self) -> None:
        got = self.clean(soft_sheet(self.cells, stretch={"front": (W + 6, H), "back": (W + 6, H)}),
                         allow=("stretch:front", "stretch:back"))
        self.assertIn("disagree:width", got["flaws"])
        self.assertFalse(got["ok"])

    def test_a_front_drawn_too_tall_is_a_redrawable_sheet_flaw(self) -> None:
        sheet = soft_sheet(self.cells, stretch={"front": (W, H + 5)})
        got = pixelart.check_sheet(sheet, ground_key(sheet))
        self.assertIn("stretch:front", got["flaws"])
        self.assertNotIn("stretch:back", got["flaws"])
        self.assertGreater(got["stretch"]["front"], 0.4)
        self.assertLess(abs(got["stretch"]["back"]), pixelart.AGREE)
        self.assertEqual(pixelart.redrawable(got["flaws"]), ["stretch:front"])

    def test_a_finer_drawing_is_found_at_a_finer_target(self) -> None:
        # A 96-cell car with detail in every cell (a 32-cell car blown up would read as its 32 cells).
        rng = np.random.default_rng(5)
        big = {}
        for pid, a in self.cells.items():
            a = np.repeat(np.repeat(a, 3, 0), 3, 1)
            inner = np.zeros(a.shape[:2], bool)
            inner[1:-1, 1:-1] = True
            dots = inner & (rng.random(a.shape[:2]) < 0.3)
            a[dots] = GLASS
            big[pid] = a
        got = self.clean(soft_sheet(big, cell=5.0, blur=0.8), target=96)
        self.assertEqual(got["grid"]["how"], "found")
        self.assertEqual(got["sizes"]["final"]["side"], [3 * L, 3 * H])
        self.assertEqual(got["flaws"], [])

    def test_a_short_top_view_is_a_sheet_flaw(self) -> None:
        sheet = soft_sheet(self.cells, stretch={"top": (W, 6)})
        self.assertIn("top_short", pixelart.check_sheet(sheet, ground_key(sheet))["flaws"])

    def test_no_grid_falls_back_to_the_target_length(self) -> None:
        sheet = soft_sheet(self.cells)
        key = ground_key(sheet)
        with mock.patch.object(pixelart, "GRID_MIN_SCORE", 99.0):
            got = pixelart.clean(pixelart.check_sheet(sheet, key)["panels"], key)
        self.assertEqual(got["grid"]["how"], "fallback")
        self.assertEqual(got["sizes"]["final"]["side"][0], pixelart.TARGET)
        self.assertTrue(got["notes"])

    def test_a_sideways_top_view_is_a_sheet_flaw(self) -> None:
        cells = dict(self.cells, top=np.ascontiguousarray(np.rot90(self.cells["top"])))
        sheet = soft_sheet(cells)
        self.assertIn("top_sideways", pixelart.check_sheet(sheet, ground_key(sheet))["flaws"])

    def test_fit_repeats_or_drops_rows_only(self) -> None:
        a = np.arange(12).reshape(3, 4)
        b = pixelart.fit(a, 4, 4)
        self.assertEqual(b.shape, (4, 4))
        self.assertTrue(set(np.unique(b)) <= set(np.unique(a)))


class DrawTest(unittest.TestCase):
    def test_views_pixel_draws_cleans_and_saves_takes(self) -> None:
        sheet = soft_sheet(car_views())
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"FLAT2D_STILL": "stub"}), \
                mock.patch.object(optional, "still_image", return_value=sheet) as drawn:
            root = Path(tmp)
            (root / "out" / "car").mkdir(parents=True)
            saved = optional.draw(root, "car", "a small blue 90s sports race car", views="pixel")
            self.assertEqual(drawn.call_count, 1)
            ask = drawn.call_args.args[0]
            self.assertIn("View 1 (top left): top view", ask)
            flat = root / "out" / "car" / "flat"
            for pid in ("top", "side", "front", "back"):
                self.assertTrue((flat / "views" / f"{pid}.png").is_file())
                self.assertEqual(Image.open(flat / "pixel" / f"{pid}.png").mode, "RGBA")
            doc = json.loads((flat / "views.json").read_text())
            self.assertEqual(doc["layout"], "pixel")
            self.assertEqual(saved["pixel"], "flat/pixel/pixel.json")
            pix = json.loads((flat / "pixel" / "pixel.json").read_text())
            self.assertEqual(pix["sizes"]["final"]["side"], [L, H])
            self.assertTrue((flat / "pixel" / "preview.png").is_file())
            self.assertEqual(doc["sheet"]["size"], 32)
            self.assertIn("flat/pixel/top.png", flow.stage_files(root / "out" / "car", "flat"))
            self.assertIn("flat/sheet.png", flow.stage_files(root / "out" / "car", "concept"))

    def test_a_chosen_size_reaches_the_prompt_and_the_cleaner(self) -> None:
        sheet = soft_sheet(car_views())
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"FLAT2D_STILL": "stub"}), \
                mock.patch.object(optional, "still_image", return_value=sheet) as drawn, \
                mock.patch.object(pixelart, "clean_folder", wraps=pixelart.clean_folder) as cleaned:
            root = Path(tmp)
            (root / "out" / "car").mkdir(parents=True)
            optional.draw(root, "car", "a car", views="pixel", size=96)
            self.assertIn("about 96 pixels long", drawn.call_args.args[0])
            self.assertEqual(cleaned.call_args.kwargs["target"], 96)
            doc = json.loads((root / "out" / "car" / "flat" / "views.json").read_text())
            self.assertEqual(doc["sheet"]["size"], 96)
            self.assertEqual(doc["sheet"]["style"], "pixel")

    def test_the_clean_style_reaches_the_prompt_and_is_cleaned_the_same(self) -> None:
        sheet = soft_sheet(car_views())
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"FLAT2D_STILL": "stub"}), \
                mock.patch.object(optional, "still_image", return_value=sheet) as drawn, \
                mock.patch.object(pixelart, "clean_folder", wraps=pixelart.clean_folder) as cleaned:
            root = Path(tmp)
            (root / "out" / "car").mkdir(parents=True)
            optional.draw(root, "car", "a car", views="pixel", size=96, style="clean")
            self.assertNotIn("pixel", drawn.call_args.args[0].lower())
            self.assertEqual(cleaned.call_args.kwargs["target"], 96)
            doc = json.loads((root / "out" / "car" / "flat" / "views.json").read_text())
            self.assertEqual((doc["sheet"]["size"], doc["sheet"]["style"]), (96, "clean"))


if __name__ == "__main__":
    unittest.main()
