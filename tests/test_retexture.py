#!/usr/bin/env python3
"""retexture.py --first-wins: a real photo bakes over the drawn views instead of being averaged with them, and is
fitted near its --first-facing, anchored on a drawn side view."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def cutout(path: Path, rgb: tuple[int, int, int], size: int = 128) -> Path:
    """A square subject on a transparent ground: what a box looks like from the front."""
    a = np.zeros((size, size, 4), np.uint8)
    a[24:104, 24:104] = (*rgb, 255)
    Image.fromarray(a, "RGBA").save(path)
    return path


def textured_box(path: Path) -> None:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).subdivide().subdivide()
    # Unshared corners, so each triangle owns its own patch of the atlas.
    mesh = trimesh.Trimesh(vertices=box.vertices[box.faces].reshape(-1, 3),
                           faces=np.arange(len(box.faces) * 3).reshape(-1, 3), process=False)
    n = len(mesh.faces)
    cols = int(np.ceil(np.sqrt(n)))
    uv = np.zeros((n * 3, 2))
    for i in range(n):
        x, y = (i % cols) / cols, (i // cols) / cols
        d = 0.9 / cols
        uv[i * 3:i * 3 + 3] = [(x, y), (x + d, y), (x, y + d)]
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (256, 256), (128, 128, 128)))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    mesh.export(path)


class FirstWinsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.glb = self.tmp / "box.glb"
        textured_box(self.glb)
        self.photo = cutout(self.tmp / "photo.png", (220, 30, 30))
        self.drawn = cutout(self.tmp / "front.png", (30, 30, 220))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def bake(self, *extra: str) -> tuple[np.ndarray, list[dict]]:
        out = self.tmp / "out.glb"
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "retexture.py"), str(self.glb), "-o", str(out),
                               "--view", str(self.photo), *extra, "--view", str(self.drawn), "--samples", "40000"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        albedo = np.asarray(Image.open(out.with_suffix(".albedo.png")).convert("RGB")).astype(int)
        return albedo, json.loads(out.with_suffix(".fit.json").read_text())

    def test_the_first_view_wins_where_it_lands(self) -> None:
        # Both views see the same face. Averaged, the baked texels are purple; with the photo on top, its red.
        def shares(albedo: np.ndarray) -> tuple[float, float]:
            r, g, b = albedo[..., 0], albedo[..., 1], albedo[..., 2]
            baked = g < 80  # the grey base atlas has g 128
            purple = baked & (r > 80) & (b > 80)
            red = baked & (r > 180) & (b < 70)
            return purple.sum() / max(baked.sum(), 1), red.sum() / max(baked.sum(), 1)

        mixed, fit = self.bake()
        self.assertGreater(shares(mixed)[0], 0.9)
        self.assertNotIn("on_top", fit[0])
        top, fit = self.bake("--first-wins")
        purple, red = shares(top)
        self.assertGreater(red, 0.9)
        self.assertLess(purple, 0.1)  # the soft seam where the photo's edge fades out
        self.assertEqual([f.get("on_top") for f in fit if Path(f["view"]).name == "photo.png"], [True])

    def test_a_drawn_side_view_anchors_the_photo(self) -> None:
        # The photo shares the canonical's facing: a drawn side view is fitted first, and the photo is then searched
        # only within 50 degrees of that facing instead of all the way round, where a mirrored match could win.
        side = cutout(self.tmp / "right.png", (30, 30, 220))
        out = self.tmp / "out.glb"
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "retexture.py"), str(self.glb), "-o", str(out),
                               "--view", str(self.photo), "--first-wins", "--view", str(side),
                               "--first-facing", "right", "--samples", "20000"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        fit = json.loads(out.with_suffix(".fit.json").read_text())
        self.assertEqual([Path(f["view"]).name for f in fit], ["right.png", "photo.png"])
        gap = (fit[1]["az"] - fit[0]["az"] + 180) % 360 - 180
        self.assertLessEqual(abs(gap), 50)


if __name__ == "__main__":
    unittest.main()
