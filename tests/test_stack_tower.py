#!/usr/bin/env python3
"""stack_tower: a tower with setbacks from a building's drawn walls (floors found and repeated, narrower blocks
stacked on wider ones). No model."""
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
import stack_tower as ST  # noqa: E402
from test_stack_building import BRICK, SMALL, _have, synthetic_walls  # noqa: E402

Q = lambda t: None  # noqa: E731


class FloorTests(unittest.TestCase):
    def test_floor_order_keeps_the_top_and_ground_floors(self) -> None:
        self.assertEqual(SB.floor_order(3, 3), [0, 1, 2])
        self.assertEqual(SB.floor_order(3, 5), [0, 1, 1, 1, 2])  # the typical floor repeated
        self.assertEqual(SB.floor_order(3, 2), [0, 2])
        self.assertEqual(SB.floor_order(3, 1), [2])
        self.assertEqual(SB.floor_order(2, 4), [0, 0, 0, 1])
        self.assertEqual(SB.floor_order(3, 4, ground=False), [0, 1, 1, 1])  # upper floors: no ground floor
        with self.assertRaises(SB.WallsError):
            SB.floor_order(3, 0)
        with self.assertRaises(SB.WallsError):
            SB.floor_order(1, 3)

    def test_floors_on_a_drawn_wall_repeat_its_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out").mkdir()
            SB.build(root, "s", synthetic_walls(), size=SMALL, line=Q)
            view = SB.read_pixel(root, "s")
            bands = SB.find_floors(view["side"])
            self.assertEqual(len(bands), 2)  # two rows of windows
            self.assertEqual(bands[-1][1], int(np.flatnonzero((view["side"][..., 3] > 127).any(1))[-1]) + 1)
            out = SB.set_floors(view, 4)
            (a, b), top = bands[0], bands[0][0]
            self.assertEqual(out["order"], [0, 0, 0, 1])
            self.assertEqual(out["height"], view["side"].shape[0] + 2 * (b - a))
            side = out["view"]["side"]
            np.testing.assert_array_equal(side[:top], view["side"][:top])  # the cornice stays on top
            np.testing.assert_array_equal(side[top + (b - a) : top + 2 * (b - a)], view["side"][a:b])
            np.testing.assert_array_equal(side[-(bands[1][1] - bands[1][0]) :], view["side"][bands[1][0] : bands[1][1]])
            for f in ("front", "back"):
                self.assertEqual(out["view"][f].shape, side.shape)

    @unittest.skipUnless(_have(BRICK), "the proven brick walls are not on this machine")
    def test_the_brick_wall_has_three_floors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out").mkdir()
            SB.build(root, "brick", {n: BRICK / f"{n}.png" for n in SB.WALLS}, line=Q)
            self.assertEqual(len(SB.find_floors(SB.read_pixel(root, "brick")["side"])), 3)


class TowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        (cls.root / "out").mkdir()
        SB.build(cls.root, "s", synthetic_walls(), size=SMALL, line=Q)
        cls.out = ST.write_tower(cls.root, "tower", "s", [(3, 4, 3), (4, 2, 2)], line=Q)
        slices = np.stack(SB.read_sheet_slices(cls.root / "out" / "tower" / "stack"))
        cls.solid = slices[..., 3] > 127

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_upper_block_is_narrower_and_stands_on_the_base_deck(self) -> None:
        base, up = self.out["tiers"]
        (z0, y0, x0), (h0, d0, w0) = base["at"], base["size"]
        (z1, y1, x1), (h1, d1, w1) = up["at"], up["size"]
        self.assertLess(w1, w0)
        self.assertLess(d1, d0)
        self.assertEqual(z1, base["deck"] + 1)  # on the base's roof deck
        self.assertGreater(z1 + h1, h0)  # taller than the base alone
        self.assertEqual((x1 - x0) * 2, w0 - w1)  # in the middle
        footprint = lambda h: self.solid[h].any(axis=0) | self.solid[h].any(axis=1)[:, None]  # noqa: E731
        wide = lambda h: int(np.ptp(np.flatnonzero(self.solid[h].any(axis=0)))) + 1  # noqa: E731
        mid_base, mid_up = h0 // 2, z1 + h1 // 2
        self.assertGreaterEqual(wide(mid_base), w0)
        self.assertEqual(wide(mid_up), w1)  # the upper floors step in
        # The base's terrace: deck cells between the upper block's walls and the base's parapet, open to the sky.
        deck = base["deck"]
        sky = ~self.solid[deck + 1 :].any(axis=0)
        self.assertTrue((self.solid[deck] & sky).any())
        self.assertTrue(footprint(deck).any())

    def test_the_upper_floors_are_the_drawn_wall(self) -> None:
        drawn = SB.read_pixel(self.root, "s")
        bands = SB.find_floors(drawn["side"])
        up = self.out["tiers"][1]
        self.assertEqual(up["floor_order"], [0, 0, 0, 0])
        self.assertFalse(up["awning"]["found"])  # no shop front up there
        self.assertTrue(self.out["tiers"][0]["awning"]["found"])
        # The upper block's left wall is the plain wall's top floor, over and over: one floor's layers are the next
        # floor's, and every colour on it is one the plain wall has.
        slices = np.stack(SB.read_sheet_slices(self.root / "out" / "tower" / "stack"))
        ys, xs = np.nonzero(self.solid.any(axis=0))
        oy, ox = int(ys.min()), int(xs.min())
        (z1, y1, x1), (h1, d1, w1) = up["at"], up["size"]
        a, b = bands[0]
        f = b - a
        face = slices[z1 : z1 + h1, oy + y1 + 1 : oy + y1 + d1 - 1, ox + x1]
        np.testing.assert_array_equal(face[:f], face[f : 2 * f])
        np.testing.assert_array_equal(face[f : 2 * f], face[2 * f : 3 * f])
        side = drawn["side"]
        drawn_cols = {tuple(c) for c in side[side[..., 3] > 127][:, :3]}
        face_cols = {tuple(c) for c in face[face[..., 3] > 127][:, :3]}
        self.assertTrue(face_cols <= drawn_cols)

    def test_meta_version_and_size(self) -> None:
        asset = self.root / "out" / "tower"
        meta = json.loads((asset / "stack" / "stack.json").read_text())
        self.assertEqual((meta["maker"], meta["shape"], meta["tower"]["from"]), ("walls", "tower", "s"))
        self.assertTrue((asset / "flow.json").is_file())  # saved as a Stack version
        src = json.loads((self.root / "out" / "s" / "stack" / "stack.json").read_text())
        self.assertAlmostEqual(meta["pitch"], src["pitch"])  # one cell is one size in every building

    def test_bad_towers(self) -> None:
        with self.assertRaises(SB.WallsError):
            ST.write_tower(self.root, "x", "s", [(3, 4, 3), (4, 4, 3)], line=Q)  # not narrower
        with self.assertRaises(SB.WallsError):
            ST.write_tower(self.root, "x", "s", [(3, 2, 2), (4, 3, 2)], line=Q)  # wider on top
        with self.assertRaises(SB.WallsError):
            ST.write_tower(self.root, "s", "s", [(3, 4, 3)], line=Q)
        with self.assertRaises(SB.WallsError):
            ST.parse_tier("3:4")
        self.assertEqual(ST.parse_tier("3:4:2"), (3, 4, 2))
        self.assertFalse((self.root / "out" / "x").exists())

    def test_a_gable_on_the_top_block_only(self) -> None:
        out = ST.write_tower(self.root, "gabled", "s", [(2, 4, 3), (3, 2, 2)], roof="gable", line=Q)
        self.assertEqual(out["tiers"][0]["roof"]["kind"], "flat")
        self.assertEqual(out["tiers"][1]["roof"]["kind"], "gable")


class RequestTests(unittest.TestCase):
    """A tower asked for by the app: two numbers, tiers stepped in from the building's own bays."""

    def test_stepped_tiers_step_one_bay_in_on_every_side(self) -> None:
        self.assertEqual(ST.stepped_tiers(7, 5, 3, 4), [(4, 7, 5), (4, 5, 3), (4, 3, 1)])
        self.assertEqual(ST.stepped_tiers(4, 4, 2, 6), [(6, 4, 4), (6, 2, 2)])
        self.assertEqual(ST.stepped_tiers(1, 1, 1, 3), [(3, 1, 1)])

    def test_when_the_bays_run_out_it_says_so(self) -> None:
        with self.assertRaises(SB.WallsError) as e:
            ST.stepped_tiers(7, 4, 3, 4)  # 4 along: 4, 2, then nothing
        self.assertIn("room for 2 tiers, not 3", str(e.exception))
        self.assertIn("7 window bays across and 4 along", str(e.exception))
        with self.assertRaises(SB.WallsError):
            ST.stepped_tiers(12, 12, ST.MAX_TIERS + 1, 3)
        with self.assertRaises(SB.WallsError):
            ST.stepped_tiers(5, 5, 2, 0)

    def test_parse_request(self) -> None:
        ok = ST.parse_request({"stem": "t", "from": "s", "tiers": 3, "floors": 4, "project": None})
        self.assertEqual(ok, {"stem": "t", "from": "s", "tiers": 3, "floors": 4})
        full = ST.parse_request({"stem": "t", "from": "s", "tiers": 2, "floors": 1, "roof": "Gable", "ridge": "along",
                                 "roof_colour": "#8a4a34"})
        self.assertEqual((full["roof"], full["ridge"], full["roof_colour"]), ("gable", "along", "#8a4a34"))
        for bad in ([], {"from": "s", "tiers": 2, "floors": 3},  # no stem
                    {"stem": "t", "tiers": 2, "floors": 3},  # no building
                    {"stem": "s", "from": "s", "tiers": 2, "floors": 3},  # over itself
                    {"stem": "t", "from": "s", "tiers": "2", "floors": 3},
                    {"stem": "t", "from": "s", "tiers": True, "floors": 3},
                    {"stem": "t", "from": "s", "tiers": 2.5, "floors": 3},
                    {"stem": "t", "from": "s", "tiers": 0, "floors": 3},
                    {"stem": "t", "from": "s", "tiers": ST.MAX_TIERS + 1, "floors": 3},
                    {"stem": "t", "from": "s", "tiers": 2, "floors": SB.FLOOR_MAX + 1},
                    {"stem": "t", "from": "s", "tiers": 2},
                    {"stem": "t", "from": "s", "tiers": 2, "floors": 3, "roof": "dome"},
                    {"stem": "t", "from": "s", "tiers": 2, "floors": 3, "ridge": "sideways"},
                    {"stem": "t", "from": "s", "tiers": 2, "floors": 3, "roof_colour": "#zzzzzz"},
                    {"stem": "../t", "from": "s", "tiers": 2, "floors": 3}):
            with self.assertRaises(ValueError, msg=bad):
                ST.parse_request(bad)

    def test_a_requested_tower_steps_in_from_the_building_as_made(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out").mkdir()
            SB.build(root, "s", synthetic_walls(), size=SMALL, line=Q)  # drawn: four bays a face
            self.assertEqual(ST.building_bays(root, "s"), (4, 4))
            out = ST.write_requested(root, ST.parse_request({"stem": "t", "from": "s", "tiers": 2, "floors": 3}),
                                     line=Q)
            self.assertEqual([(t["floors"], t["wide"], t["deep"]) for t in out["tiers"]], [(3, 4, 4), (3, 2, 2)])
            (b, u) = out["tiers"]
            self.assertEqual(b["size"][2] - u["size"][2], b["size"][1] - u["size"][1])  # in by the same on each side
            self.assertEqual(json.loads((root / "out" / "t" / "stack" / "stack.json").read_text())["shape"], "tower")
            flow = json.loads((root / "out" / "t" / "flow.json").read_text())
            self.assertTrue(flow)  # saved as a Stack version
            # A third tier has no bays left: said plainly, nothing written.
            with self.assertRaises(SB.WallsError) as e:
                ST.write_requested(root, ST.parse_request({"stem": "t3", "from": "s", "tiers": 3, "floors": 3}), line=Q)
            self.assertIn("room for 2 tiers, not 3", str(e.exception))
            self.assertFalse((root / "out" / "t3").exists())
            # A tower is not made from a tower: only a building with drawn walls.
            with self.assertRaises(SB.WallsError):
                ST.building_bays(root, "t")

    def test_a_building_made_narrower_starts_from_its_own_bays_and_roof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "out").mkdir()
            SB.build(root, "s", synthetic_walls(), size=SMALL, line=Q, wide=6, deep=5, roof="gable", ridge="along",
                     roof_colour="tile")
            self.assertEqual(ST.building_bays(root, "s"), (6, 5))
            self.assertEqual(ST.building_roof(root, "s")["roof"], "gable")
            out = ST.write_requested(root, ST.parse_request({"stem": "t", "from": "s", "tiers": 2, "floors": 2}),
                                     line=Q)
            self.assertEqual([(t["wide"], t["deep"]) for t in out["tiers"]], [(6, 5), (4, 3)])
            self.assertEqual([t["roof"]["kind"] for t in out["tiers"]], ["flat", "gable"])  # the building's roof on top
            self.assertEqual(out["tiers"][1]["roof"]["ridge"], "along")
            flat = ST.write_requested(root, ST.parse_request({"stem": "f", "from": "s", "tiers": 2, "floors": 2,
                                                              "roof": "flat"}), line=Q)
            self.assertEqual(flat["tiers"][1]["roof"]["kind"], "flat")
            # Three tiers would leave one bay along the top, which this wall (two middle bays, no door) cannot cut:
            # said plainly, with the tier, nothing written.
            with self.assertRaises(SB.WallsError) as e:
                ST.write_requested(root, ST.parse_request({"stem": "t3", "from": "s", "tiers": 3, "floors": 2}), line=Q)
            self.assertIn("tier 3 (2 by 1 window bays) cannot be made", str(e.exception))
            self.assertFalse((root / "out" / "t3").exists())


if __name__ == "__main__":
    unittest.main()
