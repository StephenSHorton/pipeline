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

    def test_the_roof_prompt_leads_with_a_plan_and_rules_out_the_walls(self) -> None:
        text = "a red brick shop, a shop front with a green awning, a flat roof with a low parapet, a water tank"
        p = pose.pixel_roof_prompt(text, 160, "clean", ratio=1.2)
        self.assertTrue(p.startswith("Top-down plan view"))
        for words in ("straight down", "No walls or sides are visible", "1.2 times as long",
                      "a flat roof with a low parapet, a water tank.", "magenta #FF00FF"):
            self.assertIn(words, p)
        for words in ("awning", "shop front", "red brick"):  # the walls' words pulled the model into the facade
            self.assertNotIn(words, p)
        self.assertIn("flat roof of the object", pose.pixel_roof_prompt("a tall red box"))
        self.assertNotIn("pixel", p.lower())
        self.assertIn("about 160 pixels long", pose.pixel_roof_prompt("a shop", 160, "pixel"))
        self.assertIn("about as wide as it is long", pose.pixel_roof_prompt("a shop", ratio=1.0))

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
                      "no lighting", "no ground line", "no pavement", "No shadows", "magenta #FF00FF",
                      "digital game art", "only its upper surfaces visible"):
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

    def test_a_tall_building_is_not_flagged_by_the_long_object_checks(self) -> None:
        # pipeline#15: a building's side and front are taller than its top view is long, and its front is wider than
        # its side. A good sheet of one is not a flaw.
        a = car_views()
        cells = {"top": a["top"][:12, :], "side": np.full((20, 12, 3), BODY, np.uint8),
                 "front": np.full((20, 14, 3), BODY, np.uint8), "back": np.full((20, 14, 3), BODY, np.uint8)}
        for c in cells.values():
            c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = OUT
        sheet = soft_sheet(cells, cell=10.0)
        got = pixelart.check_sheet(sheet, ground_key(sheet))
        self.assertEqual(got["flaws"], [])

    def test_a_top_view_drawn_as_a_facade_is_not_judged_for_a_tall_object(self) -> None:
        # The top view drawn as the front seen from above (pipeline#15): not a flaw any more, the roof picture
        # replaces it (pipeline#16); the front and back are judged against the roof's size instead.
        side = np.full((20, 12, 3), BODY, np.uint8)
        front = np.full((20, 14, 3), BODY, np.uint8)
        cells = {"top": front.copy(), "side": side, "front": front, "back": front.copy()}
        sheet = soft_sheet(cells, cell=10.0)
        got = pixelart.check_sheet(sheet, ground_key(sheet))
        self.assertTrue(got["tall"] and got["roof"])
        self.assertEqual(got["flaws"], [])
        self.assertIn("top_length", got["notes"])
        self.assertEqual(pixelart.roof_size(got["sizes"]), (got["sizes"]["front"][0], got["sizes"]["side"][0]))

    def test_a_tall_object_whose_top_view_fits_keeps_it(self) -> None:
        a = car_views()
        cells = {"top": a["top"][:12, :], "side": np.full((20, 12, 3), BODY, np.uint8),
                 "front": np.full((20, 14, 3), BODY, np.uint8), "back": np.full((20, 14, 3), BODY, np.uint8)}
        sheet = soft_sheet(cells, cell=10.0)
        got = pixelart.check_sheet(sheet, ground_key(sheet))
        self.assertTrue(got["tall"])
        self.assertFalse(got["roof"])

    def test_the_front_wall_under_a_roof_plan_is_cut(self) -> None:
        roof = Image.new("RGB", (1024, 1024), MAGENTA)
        roof.paste(Image.new("RGB", (600, 400), GLASS), (200, 300))
        roof.paste(Image.new("RGB", (560, 30), RED), (220, 700))  # the facade peeking out, set in from the sides
        fg = pixelart.subject_mask(np.asarray(roof), ground_key(roof))
        self.assertEqual(pixelart.roof_edge_rows(fg), 30)
        panel = pixelart.roof_panel(roof, (100, 100))
        px = np.asarray(panel).reshape(-1, 3).astype(int)
        self.assertFalse((np.abs(px - RED).sum(1) < 60).any())
        square = Image.new("RGB", (1024, 1024), MAGENTA)
        square.paste(Image.new("RGB", (600, 400), GLASS), (200, 300))
        self.assertEqual(pixelart.roof_edge_rows(pixelart.subject_mask(np.asarray(square), ground_key(square))), 0)

    def test_a_strip_under_the_views_is_painted_out(self) -> None:
        # pipeline#15: a grey pavement strip wider than the walls under the side, front and back views.
        def building(w, h):
            a = np.full((h + 1, w + 6, 3), MAGENTA, np.uint8)
            a[:h, 3:3 + w] = BODY
            a[h:, :] = TYRE  # the strip: one cell, three cells wider on each side
            return a

        cells = {"top": np.full((12, 14, 3), BODY, np.uint8), "side": building(12, 20),
                 "front": building(14, 20), "back": building(14, 20)}
        sheet = soft_sheet(cells, cell=10.0, blur=0.6)
        got = pixelart.check_sheet(sheet, ground_key(sheet))
        self.assertEqual(set(got["base"]), {"side", "front", "back"})
        self.assertTrue(all(8 <= n <= 14 for n in got["base"].values()), got["base"])
        self.assertAlmostEqual(got["sizes"]["side"][0], 120, delta=6)  # the walls, not the strip
        self.assertAlmostEqual(got["sizes"]["front"][1], 200, delta=6)
        side = np.asarray(got["panels"]["side"])
        self.assertFalse(pixelart.subject_mask(side, ground_key(got["panels"]["side"]))[-60:].any(1).sum() > 25)

    def test_wheels_under_a_car_are_not_a_strip(self) -> None:
        sheet = soft_sheet(self.cells)
        self.assertEqual(pixelart.check_sheet(sheet, ground_key(sheet))["base"], {})

    def test_the_roof_picture_is_fitted_as_the_top_view(self) -> None:
        roof = Image.new("RGB", (1024, 1024), MAGENTA)
        roof.paste(Image.new("RGB", (600, 500), BODY), (200, 260))
        check = pixelart.check_roof(roof)
        self.assertEqual(check["flaws"], [])
        panel = pixelart.roof_panel(roof, (140, 120), check["key"])
        self.assertEqual(panel.size, (140 + 2 * pixelart.ROOF_MARGIN, 120 + 2 * pixelart.ROOF_MARGIN))
        fg = pixelart.subject_mask(np.asarray(panel), ground_key(panel))
        box = pixelart.bbox(fg)
        self.assertEqual((box[2] - box[0], box[3] - box[1]), (140, 120))
        self.assertIn("empty:roof", pixelart.check_roof(Image.new("RGB", (256, 256), MAGENTA))["flaws"])

    def test_no_grid_search_cuts_to_the_target_even_when_a_grid_is_there(self) -> None:
        # The clean style has no pixel grid by design; brickwork that lines up must not set the size (pipeline#15).
        sheet = soft_sheet(self.cells)
        key = ground_key(sheet)
        got = pixelart.clean(pixelart.check_sheet(sheet, key)["panels"], key, target=40, grid=False)
        self.assertEqual(got["grid"]["how"], "fallback")
        self.assertEqual(got["sizes"]["final"]["side"][0], 40)
        self.assertIn("no pixel grid", got["notes"][0])

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

    def test_a_tall_object_gets_its_top_view_from_a_roof_picture(self) -> None:
        side = np.full((20, 12, 3), BODY, np.uint8)
        front = np.full((20, 14, 3), BODY, np.uint8)
        for c in (side, front):
            c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = OUT
        sheet = soft_sheet({"top": front.copy(), "side": side, "front": front, "back": front.copy()}, cell=10.0)
        roof = Image.new("RGB", (1024, 1024), MAGENTA)
        roof.paste(Image.new("RGB", (500, 500), GLASS), (262, 262))
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"FLAT2D_STILL": "stub"}), \
                mock.patch.object(optional, "still_image", side_effect=[sheet, roof]) as drawn:
            root = Path(tmp)
            (root / "out" / "shop").mkdir(parents=True)
            optional.draw(root, "shop", "a shop", views="pixel", size=40, style="clean")
            self.assertEqual(drawn.call_count, 2)
            self.assertTrue(drawn.call_args_list[1].args[0].startswith("Top-down plan view"))
            self.assertEqual(drawn.call_args_list[1].kwargs["width"], pose.ROOF_SIZE[0])
            flat = root / "out" / "shop" / "flat"
            doc = json.loads((flat / "views.json").read_text())
            self.assertEqual(doc["roof"]["file"], "views/roof.png")
            self.assertTrue((flat / "views" / "roof.png").is_file())
            top = np.asarray(Image.open(flat / "pixel" / "top.png"))
            colours = {tuple(int(v) for v in px[:3]) for px in top.reshape(-1, 4) if px[3]}
            self.assertTrue(any(sum(abs(a - b) for a, b in zip(c, GLASS)) < 40 for c in colours))  # the roof's
            self.assertFalse(any(sum(abs(a - b) for a, b in zip(c, BODY)) < 40 for c in colours))  # not the walls'
            self.assertEqual(Image.open(flat / "pixel" / "top.png").size[1], 40)
            self.assertIn("flat/views/roof.png", flow.stage_files(root / "out" / "shop", "concept"))

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
            self.assertTrue(cleaned.call_args.kwargs["grid"])
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
            self.assertFalse(cleaned.call_args.kwargs["grid"])
            doc = json.loads((root / "out" / "car" / "flat" / "views.json").read_text())
            self.assertEqual((doc["sheet"]["size"], doc["sheet"]["style"]), (96, "clean"))


if __name__ == "__main__":
    unittest.main()
