#!/usr/bin/env python3
"""A walking character (pipeline#2): a person carved from front, side and back views, cut into twelve part stacks
that put back together into the body, with a rig measured from the joints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stack_character as C  # noqa: E402

H = 60


def person() -> dict:
    """A plain blocky person, 60 cells tall: legs apart, arms hanging clear of the body, a head."""
    W, L = 30, 12
    front = np.zeros((H, W), bool)  # row 0 at the top, as a picture
    side = np.zeros((H, L), bool)
    rows = lambda a, b: slice(H - b, H - a)  # layers a..b (from the bottom) as picture rows
    front[rows(0, 22), 8:13] = True  # right leg (the viewer's left)
    front[rows(0, 22), 17:22] = True  # left leg
    front[rows(22, 46), 8:22] = True  # torso
    front[rows(20, 45), 2:6] = True  # right arm, a gap from the body
    front[rows(20, 45), 24:28] = True  # left arm
    front[rows(43, 46), 2:28] = True  # shoulders joining the arms
    front[rows(46, 48), 13:17] = True  # neck
    front[rows(48, 60), 10:20] = True  # head
    side[rows(0, 3), 2:11] = True  # feet stick out forward
    side[rows(3, 60), 3:9] = True
    rgba = lambda m, c: np.where(m[..., None], np.array([*c, 255], np.uint8), np.uint8(0))
    return {
        "front": (rgba(front, (200, 40, 40)), front),
        "side": (rgba(side, (40, 200, 40)), side),
        "back": (rgba(front, (40, 40, 200)), front),
    }


class CharacterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.solid, self.rgba, self.m = C.carve_character(person())
        self.parts = C.cut_parts(self.solid, self.rgba, self.m)

    def test_twelve_parts_put_back_together_into_the_body(self) -> None:
        self.assertEqual(sorted(self.parts), sorted(C.PART_NAMES))
        C.check_cut(self.solid, self.rgba, self.parts)  # raises when a cell is lost, doubled or recoloured

    def test_a_broken_cut_is_caught(self) -> None:
        p = self.parts["thigh_R"]
        p["cells"] = p["cells"][1:]
        with self.assertRaises(C.PartsError):
            C.check_cut(self.solid, self.rgba, self.parts)

    def test_lower_limbs_hang_on_the_upper_limbs_axis(self) -> None:
        for side in "RL":
            self.assertEqual(self.parts["shin_" + side]["axis"], self.parts["thigh_" + side]["axis"])
            self.assertEqual(self.parts["foot_" + side]["axis"], self.parts["thigh_" + side]["axis"])
            self.assertEqual(self.parts["forearm_" + side]["axis"], self.parts["upperarm_" + side]["axis"])

    def test_the_rig_is_measured_from_the_joints(self) -> None:
        m = self.m
        rig = C.rig_numbers(m, self.parts)
        self.assertEqual(rig["thigh"] + rig["shin"] + rig["ankle"], m["crotch"])  # straight legs reach the crotch
        self.assertEqual(rig["spine"], m["shoulder"] - m["crotch"])
        self.assertGreater(rig["hipHalf"], 0)  # the left hip is on +y
        self.assertGreater(rig["shoulderHalf"], rig["hipHalf"])
        self.assertLess(rig["standHip"], m["crotch"])  # knees a little bent when standing


class KneeTest(unittest.TestCase):
    """pipeline#36: the knee is the hollow between the calf and the thigh, not halfway up to the crotch."""

    def test_a_straight_leg_bends_halfway(self) -> None:
        m = C.carve_character(person())[2]
        self.assertEqual(m["knee"], m["ankle"] + round((m["crotch"] - m["ankle"]) / 2))

    def test_the_knee_is_the_hollow_above_the_calf(self) -> None:
        views = person()
        side = views["side"][1].copy()
        rows = lambda a, b: slice(H - b, H - a)
        side[rows(5, 10), 2:10] = True  # a calf (or a boot's cuff), deeper than the leg
        side[rows(15, 17), 3] = False  # the hollow under the knee, shallower
        views["side"] = (views["side"][0], side)
        m = C.carve_character(views)[2]
        half = m["ankle"] + round((m["crotch"] - m["ankle"]) / 2)
        self.assertIn(m["knee"], (15, 16))
        self.assertGreater(m["knee"], half)


class SmallBodyTest(unittest.TestCase):
    """pipeline#6: the body shrunk to a flat sprite stack's size (32 or 64 layers) and cut into parts again."""

    def setUp(self) -> None:
        self.solid, self.rgba, _ = C.carve_character(person())

    def test_a_shrunk_body_is_cut_into_parts_that_put_back_together(self) -> None:
        s, r = C.shrink(self.solid, self.rgba, 2)
        self.assertEqual(s.shape[0], H // 2)
        m = C.measure(s.any(axis=1), s.any(axis=2))
        self.assertLess(m["crotch"], m["shoulder"])
        C.check_cut(s, r, C.cut_parts(s, r, m))

    def test_shrinking_picks_colours_and_never_blends_them(self) -> None:
        s, r = C.shrink(self.solid, self.rgba, 3)
        before = {tuple(c) for c in self.rgba[self.solid][:, :3]}
        after = {tuple(c) for c in r[s][:, :3]}
        self.assertTrue(after <= before)

    def test_shades_of_one_colour_vote_together(self) -> None:
        cols = np.array([[200, 30, 30], [180, 20, 25], [215, 45, 40], [10, 10, 10]], np.uint8)
        votes = np.array([[1, 1, 1, 2]], float)  # three reds once each, a black line twice
        self.assertIn(int(C._pick_colour(votes, cols)[0]), (0, 1, 2))

    def test_inside_cells_take_the_outside_colour(self) -> None:
        solid = np.ones((5, 5, 5), bool)
        rgba = np.zeros((5, 5, 5, 4), np.uint8)
        rgba[...] = (200, 30, 30, 255)
        rgba[1:4, 1:4, 1:4] = (240, 200, 160, 255)  # a skin-coloured surface inside (a head under its hair)
        out = C._outside_colours(solid, rgba)
        self.assertEqual({tuple(c) for c in out[solid][:, :3]}, {(200, 30, 30)})

    def test_a_model_is_turned_to_face_the_front_the_drawing_shows(self) -> None:
        solid = np.ones((4, 6, 6), bool)
        rgba = np.zeros((4, 6, 6, 4), np.uint8)
        rgba[...] = (40, 40, 200, 255)
        rgba[:, :, 0] = (200, 40, 40, 255)  # the red face is on the character's right side (column 0)
        drawn = np.zeros((4, 6, 4), np.uint8)
        drawn[...] = (200, 40, 40, 255)  # the drawing's front is red
        s, r = C._face_front(solid, rgba, drawn)
        self.assertTrue((r[:, -1, :, 0] == 200).all())  # the front row (L-1) is the red face now


class MeshBodyTest(unittest.TestCase):
    """pipeline#6: a body sliced from a 3D model is a solid volume the height asked, every layer filled."""

    def test_a_box_model_slices_into_full_layers(self) -> None:
        import tempfile

        import trimesh
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp)
            (asset / "character" / "views").mkdir(parents=True)
            Image.new("RGBA", (10, 40), (128, 128, 128, 255)).save(asset / "character" / "views" / "front.png")
            box = trimesh.creation.box(extents=(0.5, 2.0, 0.25))  # its sides fall on cell edges
            box.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.5, 0.5, 0.5, 1.0]))
            glb = asset / "box.glb"
            box.export(glb)
            solid, rgba = C.mesh_volume(asset, 40, glb=glb)
        self.assertEqual(solid.shape[0], 40)
        self.assertIn(solid.shape[2], (10, 11))  # 0.5 wide at 2.0 / 40 a cell, give or take a cell edge
        self.assertTrue(all(solid[h].sum() >= 0.8 * solid[h].size for h in range(40)))  # every layer filled in
        self.assertTrue((rgba[solid][:, 3] == 255).all())


if __name__ == "__main__":
    unittest.main()
