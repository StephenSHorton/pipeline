#!/usr/bin/env python3
"""scripts/workbench/solid.py: a hole through a ring is filled, only inside the sphere when one is given, a carved
bump is gone without leaving a hole, and gaps finds the hole first."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from workbench import solid  # noqa: E402


def two_rings() -> trimesh.Trimesh:
    a = trimesh.creation.torus(1.0, 0.35)
    b = trimesh.creation.torus(1.0, 0.35)
    b.apply_translation([3.0, 0, 0])
    return trimesh.util.concatenate([a, b])


class Solid(unittest.TestCase):
    def test_a_ring_closes_to_one_solid(self) -> None:
        t = trimesh.creation.torus(1.0, 0.35)
        out, info = solid.solidify(t, grid=96)
        self.assertTrue(info["watertight"])
        self.assertEqual(out.euler_number, 2)  # genus 0: the hole is gone
        self.assertGreater(out.volume, t.volume * 1.25)

    def test_only_inside_the_sphere(self) -> None:
        out, info = solid.solidify(two_rings(), center=[0, 0, 0], radius=1.0, grid=128)
        self.assertGreater(info["filled"], 0)
        self.assertEqual(len(out.split(only_watertight=False)), 2)  # both rings kept
        inside = out.contains([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        self.assertEqual(inside.tolist(), [True, False])  # only the first ring's hole is filled

    def test_the_filled_hole_is_the_sphere_one(self) -> None:
        m = two_rings()
        inside, origin, h = solid.voxels(m, 128)
        add = solid._through(inside)
        idx = np.argwhere(add)
        p = origin + h * (idx + 0.5)
        self.assertTrue((np.abs(p[:, 0]) < 1).any() and (np.abs(p[:, 0] - 3) < 1).any())

    def test_carve_takes_a_bump_off(self) -> None:
        body = trimesh.creation.box(extents=(1, 1, 1))
        bump = trimesh.creation.icosphere(radius=0.2)
        bump.apply_translation([0.6, 0, 0])
        m = trimesh.util.concatenate([body, bump])
        out, info = solid.solidify(m, close=False, carve=[{"center": [0.7, 0, 0], "radius": 0.25}], grid=96)
        self.assertGreater(info["carved"], 0)
        self.assertLess(out.bounds[1, 0], 0.56)
        self.assertTrue(info["watertight"])

    def test_gaps_finds_the_hole(self) -> None:
        g = solid.gaps(trimesh.creation.torus(1.0, 0.35), grid=96)
        self.assertTrue(g)
        self.assertLess(np.linalg.norm(g[0]["center"]), 0.1)


if __name__ == "__main__":
    unittest.main()
