#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_mesh  # noqa: E402


class CleanMeshTests(unittest.TestCase):
    def test_floaters_drops_tiny_island(self) -> None:
        main = trimesh.creation.box(extents=(1, 1, 1))
        speck = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
        speck.apply_translation([4, 0, 0])
        junk = trimesh.util.concatenate([main, speck])
        cleaned, info = clean_mesh.remove_floaters(junk, keep=1)
        self.assertEqual(info["components"], 2)
        self.assertEqual(info["kept"], 1)
        self.assertLess(info["faces_out"], info["faces_in"])
        self.assertGreater(len(cleaned.faces), 4)

    def test_sphere_deletes_local_chunk(self) -> None:
        mesh = trimesh.creation.box(extents=(1, 1, 1))
        before = len(mesh.faces)
        out, info = clean_mesh.delete_sphere(mesh, [0.5, 0.5, 0.5], 0.35)
        self.assertLess(info["faces_out"], before)
        self.assertGreater(info["faces_deleted"], 0)
        self.assertGreater(len(out.faces), 0)


if __name__ == "__main__":
    unittest.main()
