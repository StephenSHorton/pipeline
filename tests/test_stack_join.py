#!/usr/bin/env python3
"""stack_join: buildings already made, joined into one stack (a row, a corner with a wing). No model."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import stack_building as SB  # noqa: E402
import stack_join as SJ  # noqa: E402
from test_stack_building import SMALL, synthetic_walls  # noqa: E402


class JoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        (cls.root / "out").mkdir()
        walls = synthetic_walls()
        q = lambda t: None  # noqa: E731
        cls.a = SB.build(cls.root, "a", walls, size=SMALL, line=q)
        cls.b = SB.build(cls.root, "b", walls, size=SMALL, wide=2, roof="gable", line=q)
        cls.wing = SB.build(cls.root, "wing", walls, size=SMALL, wide=4, deep=2, line=q)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_a_row_puts_the_walls_side_by_side_with_fronts_in_line(self) -> None:
        out = SJ.write_joined(self.root, "row", [("a", 0), ("b", 0)], "row", line=lambda t: None)
        pa, pb = SJ.read_part(self.root, "a"), SJ.read_part(self.root, "b")
        wa, wb = pa["wall"], pb["wall"]
        self.assertEqual(out["layers"], max(self.a["layers"], self.b["layers"]))
        # The two wall boxes touch: the joined width is both walls plus whatever hangs past the outer ends.
        self.assertGreaterEqual(out["size"]["width"], (wa[3] - wa[2] + 1) + (wb[3] - wb[2] + 1))
        (ya, xa), (yb, xb) = (p["at"] for p in out["parts"])
        self.assertEqual(ya + wa[1], yb + wb[1])  # fronts in line
        self.assertEqual(xa + wa[3] + 1, xb + wb[2])  # walls touching
        meta = json.loads((self.root / "out" / "row" / "stack" / "stack.json").read_text())
        self.assertEqual((meta["maker"], meta["shape"], meta["joined"]["layout"]), ("walls", "joined", "row"))
        self.assertTrue((self.root / "out" / "row" / "flow.json").is_file())  # saved as a Stack version
        # Every building's cell is one size, so the row is as wide in the world as its parts side by side.
        pa_meta = json.loads((self.root / "out" / "a" / "stack" / "stack.json").read_text())
        pb_meta = json.loads((self.root / "out" / "b" / "stack" / "stack.json").read_text())
        self.assertAlmostEqual(pa_meta["pitch"], pb_meta["pitch"])
        self.assertAlmostEqual(meta["pitch"], pa_meta["pitch"])
        self.assertAlmostEqual(meta["extents"][0], out["size"]["width"] * pa_meta["pitch"])
        self.assertGreater(meta["extents"][0], pa_meta["extents"][0] + pb_meta["extents"][0] - 4 * pa_meta["pitch"])

    def test_a_corner_is_an_l_with_the_wing_facing_right(self) -> None:
        SJ.write_joined(self.root, "corner", [("a", 0), ("wing", 0)], "corner", line=lambda t: None)
        slices = np.stack(SB.read_sheet_slices(self.root / "out" / "corner" / "stack"))
        solid = slices[..., 3] > 127
        ys, xs = np.nonzero(solid.any(axis=0))
        solid = solid[:, ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        mid = solid[solid.shape[0] // 2]
        L, W = mid.shape
        self.assertFalse(mid[: L // 4, : W // 4].any())  # the back left is open: an L
        self.assertTrue(mid[: L // 4, -W // 4 :].any())  # the wing at the back right
        self.assertTrue(mid[-L // 4 :, : W // 4].any())  # the main building across the front
        # Turned once, the wing's awning sticks out to the right of the main building's right wall.
        aw = self.wing["awning"]["layers"]
        main_right = SJ.read_part(self.root, "a")["wall"][3]
        self.assertTrue(solid[aw[0], : L // 3, main_right + 2 :].any())

    def test_a_corner_has_one_roof_with_no_parapet_where_its_parts_meet(self) -> None:
        out = SJ.write_joined(self.root, "corner1", [("a", 0), ("wing", 0)], "corner", line=lambda t: None)
        roof = out["roof"]
        self.assertTrue(roof["one"])
        meta = json.loads((self.root / "out" / "corner1" / "stack" / "stack.json").read_text())
        self.assertTrue(meta["joined"]["roof"]["one"])
        slices = np.stack(SB.read_sheet_slices(self.root / "out" / "corner1" / "stack"))
        solid = slices[..., 3] > 127
        D, P = roof["deck"], roof["parapet"]
        main = out["parts"][0]
        (my, mx), (_, ml, mw) = main["at"], main["size"]
        ma = SJ.read_part(self.root, "a")
        my0, _, mx0, mx1 = ma["wall"]
        # Where the parts meet: the main building's back wall, under the wing. Find it in the joined slices.
        ys, xs = np.nonzero(solid.any(axis=0))
        oy, ox = int(ys.min()), int(xs.min())
        seam_y = oy + my + my0
        wing_x = slice(ox + mx + mx1 - 2 * roof["rim"] - 2, ox + mx + mx1 - 2 * roof["rim"])  # inside the wing's span
        # No parapet over the seam: open sky over the deck there, and deck colour at the deck layer.
        self.assertFalse(solid[D + 1 : D + P + 1, seam_y - 1 : seam_y + 2, wing_x].any())
        self.assertTrue(solid[D, seam_y - 1 : seam_y + 2, wing_x].all())
        deck_rgb = slices[D, seam_y, wing_x.start, :3]
        centre = slices[D, oy + my + (my0 + ma["wall"][1]) // 2, ox + mx + (mx0 + mx1) // 2, :3]
        np.testing.assert_array_equal(deck_rgb, centre)
        # The parapet still runs round the outside: the main building's front edge.
        front_y = oy + my + ma["wall"][1]
        self.assertTrue(solid[D + P, front_y - roof["rim"] + 1 : front_y + 1, ox + mx + (mx0 + mx1) // 2].any())
        # The deck is one region: every deck cell open to the sky is joined to every other.
        from scipy import ndimage
        open_deck = solid[D] & ~solid[D + 1 :].any(axis=0)
        self.assertEqual(ndimage.label(open_deck)[1], 1)

    def test_a_row_keeps_each_buildings_parapet(self) -> None:
        out = SJ.write_joined(self.root, "row1", [("a", 0), ("a", 0)], "row", line=lambda t: None)
        self.assertIsNone(out["roof"])
        slices = np.stack(SB.read_sheet_slices(self.root / "out" / "row1" / "stack"))
        solid = slices[..., 3] > 127
        ra = SJ.roof_of(SJ.read_part(self.root, "a"))
        open_deck = solid[ra["deck"]] & ~solid[ra["deck"] + 1 :].any(axis=0)
        from scipy import ndimage
        self.assertEqual(ndimage.label(open_deck)[1], 2)  # two decks, a parapet between them

    def test_a_gable_part_keeps_the_parts_roofs(self) -> None:
        out = SJ.write_joined(self.root, "corner2", [("b", 0), ("wing", 0)], "corner", line=lambda t: None)
        self.assertFalse(out["roof"]["one"])
        self.assertIsNone(SJ.roof_of(SJ.read_part(self.root, "b")))
        self.assertIsNotNone(SJ.roof_of(SJ.read_part(self.root, "a")))

    def test_turns_and_bad_joins(self) -> None:
        p0, p1 = SJ.read_part(self.root, "a"), SJ.read_part(self.root, "a", 1)
        self.assertEqual(p0["solid"].shape[1:], p1["solid"].shape[2:0:-1])
        with self.assertRaises(SJ.JoinError):
            SJ.write_joined(self.root, "x", [("a", 0)], "row")
        with self.assertRaises(SJ.JoinError):
            SJ.write_joined(self.root, "x", [("a", 0), ("missing", 0)], "row")
        with self.assertRaises(SJ.JoinError):
            SJ.write_joined(self.root, "x", [("a", 0), ("b", 0), ("wing", 0)], "corner")
        with self.assertRaises(SJ.JoinError):
            SJ.write_joined(self.root, "a", [("a", 0), ("b", 0)], "row")


if __name__ == "__main__":
    unittest.main()
