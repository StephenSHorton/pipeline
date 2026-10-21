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
import reduce_mesh  # noqa: E402
from PIL import Image  # noqa: E402


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


class ReducePiecesTests(unittest.TestCase):
    """reduce_mesh drops pieces under 2% of the area, and a textured mesh keeps a texture when decimated."""

    @staticmethod
    def with_crumb(scale: float) -> trimesh.Trimesh:
        main = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        crumb = trimesh.creation.icosphere(subdivisions=1, radius=scale)
        crumb.apply_translation([3, 0, 0])
        return trimesh.util.concatenate([main, crumb])

    def test_drops_a_piece_under_two_percent(self) -> None:
        mesh = self.with_crumb(0.1)  # area share ~1%
        out, info = reduce_mesh.drop_small_pieces(mesh)
        self.assertEqual((info["pieces"], info["kept"]), (2, 1))
        self.assertLess(out.bounds[1][0], 1.01)

    def test_keeps_a_piece_over_two_percent(self) -> None:
        mesh = self.with_crumb(0.3)  # area share ~8%
        out, info = reduce_mesh.drop_small_pieces(mesh)
        self.assertEqual(info["kept"], 2)
        self.assertEqual(len(out.faces), len(mesh.faces))

    def test_keeps_a_long_thin_piece(self) -> None:
        # A frame rod the shape model left unjoined: ~0.5% of the area but as long as the subject is tall.
        main = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        rod = trimesh.creation.cylinder(radius=0.02, height=1.2)
        rod.apply_translation([1.2, 0, 0])
        out, info = reduce_mesh.drop_small_pieces(trimesh.util.concatenate([main, rod]))
        self.assertEqual((info["pieces"], info["kept"], info["kept_long"]), (2, 2, 1))
        self.assertGreater(out.bounds[1][0], 1.2)

    def test_uv_seams_do_not_split_a_piece(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=3)
        # Unshared corners: every face is its own island by index, one piece by position.
        v = mesh.vertices[mesh.faces.reshape(-1)]
        f = np.arange(len(v)).reshape(-1, 3)
        uv = np.random.default_rng(0).random((len(v), 2))
        tex = trimesh.visual.TextureVisuals(uv=uv, image=Image.new("RGB", (8, 8)))
        seamy = trimesh.Trimesh(vertices=v, faces=f, visual=tex, process=False)
        out, info = reduce_mesh.drop_small_pieces(seamy)
        self.assertEqual(info["pieces"], 1)
        self.assertIs(out, seamy)

    def test_decimated_textured_mesh_is_rebaked_with_uvs(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=4)
        uv = np.stack([mesh.vertices[:, 0] * 0.5 + 0.5, mesh.vertices[:, 1] * 0.5 + 0.5], axis=1)
        img = Image.new("RGB", (32, 32), (10, 200, 30))
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        low = reduce_mesh.reduce_mesh(mesh, 400)
        self.assertIsNone(getattr(low.visual, "uv", None))
        baked = reduce_mesh.rebake(mesh, img, low, 128)
        self.assertIsNotNone(baked.visual.uv)
        self.assertLessEqual(len(baked.faces), 400)
        px = np.asarray(baked.visual.material.baseColorTexture)
        self.assertTrue(np.allclose(px.reshape(-1, 3).mean(0), (10, 200, 30), atol=2))


if __name__ == "__main__":
    unittest.main()
