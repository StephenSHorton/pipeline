#!/usr/bin/env python3
"""scripts/workbench/review.py on synthetic assets: a textured box whose bake matches its paint, or carries a
painted patch the paint has not (a feature in the wrong place); the grey render's shading; the subject fallback;
and a whole review writing review.json and the sheet."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from workbench import review  # noqa: E402


def textured_box(path: Path, texture: np.ndarray) -> Path:
    """A long box (along +Z, on the ground) with one UV chart per face laid out in a 3 x 2 grid of the texture."""
    box = trimesh.creation.box(extents=(0.6, 0.6, 1.4)).subdivide().subdivide()
    box.apply_translation([0, 0.3, 0])
    box.unmerge_vertices()
    V, F = box.vertices, box.faces
    n = box.face_normals
    uv = np.zeros((len(V), 2))
    for fi, f in enumerate(F):
        axis = int(np.argmax(np.abs(n[fi])))
        slot = axis * 2 + (n[fi, axis] > 0)
        cu, cv = slot % 3, slot // 3
        a, b = [i for i in range(3) if i != axis]
        lo, hi = V[:, [a, b]].min(0), V[:, [a, b]].max(0)
        t = (V[f][:, [a, b]] - lo) / (hi - lo)
        uv[f] = np.c_[(cu + 0.05 + 0.9 * t[:, 0]) / 3, (cv + 0.05 + 0.9 * t[:, 1]) / 2]
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture.astype(np.uint8)))
    mesh = trimesh.Trimesh(V, F, visual=trimesh.visual.TextureVisuals(uv=uv, material=mat), process=False)
    mesh.export(path)
    return path


def skin(size: int = 256) -> np.ndarray:
    tex = np.zeros((size, size, 3))
    tex[:] = [200, 150, 140]
    return tex


class Review(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.paint = textured_box(cls.dir / "paint.glb", skin())
        cls.same = textured_box(cls.dir / "same.glb", skin() * 1.1)  # a gain the check matches out
        ghost = skin()
        ghost[60:120, 100:180] = [90, 40, 40]  # a dark patch across most of one chart: an ear where none is
        cls.ghost = textured_box(cls.dir / "ghost.glb", ghost)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_matching_bake_agrees_with_its_paint(self) -> None:
        res, tiles = review.paint_check(self.same, self.paint, size=256)
        self.assertLess(res["worst"], 0.01)
        self.assertTrue(tiles)

    def test_a_painted_patch_the_paint_has_not_is_flagged(self) -> None:
        res, _ = review.paint_check(self.ghost, self.paint, size=256)
        self.assertGreater(res["worst"], review.PAINT_MAX)

    def test_grey_is_the_shading_alone(self) -> None:
        def grey(glb):
            sc = review.Scene(glb, 192)
            return review.grey(sc.render("q", "texture_px"), sc.render("q"))
        # the same mesh painted differently greys the same: a painted patch leaves no trace, a real shape would
        self.assertLess(float(np.abs(grey(self.ghost) - grey(self.paint)).max()), 3.0)
        g = grey(self.paint)
        self.assertGreater(float(np.ptp(g[review.subject_mask(g)].mean(1))), 10.0)  # the faces are lit differently

    def test_subject_falls_back_to_the_facing(self) -> None:
        d = self.dir / "old"
        d.mkdir(exist_ok=True)
        (d / "intent.json").write_text(json.dumps({"psx": {"facing": "right"}}))
        self.assertEqual(review.subject(d), "animal")
        (d / "intent.json").write_text(json.dumps({"psx": {"height": "50"}}))
        self.assertEqual(review.subject(d), "prop")
        (d / "intent.json").write_text(json.dumps({"subject": "person", "psx": {"facing": "right"}}))
        self.assertEqual(review.subject(d), "person")

    def test_review_writes_the_json_and_the_sheet(self) -> None:
        d = self.dir / "asset"
        d.mkdir(exist_ok=True)
        (d / "intent.json").write_text(json.dumps({"subject": "animal", "psx": {"facing": "right"}}))
        textured_box(d / "textured.glb", skin())
        ghost = skin()
        ghost[60:120, 100:180] = [90, 40, 40]
        textured_box(d / "textured_photo.glb", ghost)
        (d / "textured_photo.select.json").write_text("{}")
        (d / "degraded.json").write_text(json.dumps({"degraded": [{"code": "single_view", "message": "one picture"}]}))
        res = review.review(d, glb=d / "textured_photo.glb")
        names = {c["name"]: c for c in res["checks"]}
        self.assertFalse(names["paint"]["ok"])
        self.assertFalse(names["degraded"]["ok"])
        self.assertIn("paint", res["failed"])
        self.assertTrue(res["verdict"].startswith("fix"))
        self.assertTrue(Path(res["sheet"]).is_file())
        self.assertEqual(json.loads((d / "workbench" / "review" / "review.json").read_text())["failed"], res["failed"])

    def test_paint_does_not_apply_to_an_older_bake(self) -> None:
        d = self.dir / "older"
        d.mkdir(exist_ok=True)
        textured_box(d / "textured.glb", skin())
        textured_box(d / "textured_photo.glb", skin() * 0.3)
        res = review.review(d, glb=d / "textured_photo.glb")
        paint = next(c for c in res["checks"] if c["name"] == "paint")
        self.assertIsNone(paint["ok"])
        self.assertEqual(res["verdict"], "look")


if __name__ == "__main__":
    unittest.main()
