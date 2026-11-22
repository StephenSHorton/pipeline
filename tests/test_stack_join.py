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
