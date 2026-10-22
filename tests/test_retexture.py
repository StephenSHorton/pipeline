#!/usr/bin/env python3
"""retexture.py --first-wins: a real photo bakes over the drawn views instead of being averaged with them, and is
fitted near the drawn view it looks like."""
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

    def test_the_photo_is_searched_near_the_view_it_looks_like(self) -> None:
        # A front photo of a symmetric subject fits the back as well; the drawn view it resembles picks the window.
        sys.path.insert(0, str(ROOT / "scripts"))
        import retexture

        back = cutout(self.tmp / "back.png", (30, 200, 30))
        wide = np.zeros((128, 128, 4), np.uint8)
        wide[44:84, 8:120] = (220, 30, 30, 255)  # the photo's colour, a profile's outline
        Image.fromarray(wide, "RGBA").save(self.tmp / "right.png")
        self.drawn = cutout(self.tmp / "front.png", (220, 30, 30))
        named = [(str(self.tmp / "right.png"), -90.0), (str(self.drawn), 0.0), (str(back), 180.0)]
        self.assertEqual(retexture.photo_offset(str(self.photo), named), (0.0, str(self.drawn)))
        self.assertEqual(retexture.photo_offset(str(back), named)[0], 180.0)
        self.assertEqual(retexture.photo_offset(str(self.tmp / "right.png"), named)[0], -90.0)

        out = self.tmp / "out.glb"
        side = cutout(self.tmp / "right.png", (30, 30, 220))  # the bake: the box looks the same from every side
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "retexture.py"), str(self.glb), "-o", str(out),
                               "--view", str(self.photo), "--first-wins", "--view", str(side), "--view",
                               str(self.drawn), "--view", str(back), "--samples", "20000"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("photo.png looks most like front.png", proc.stdout)
        fit = {Path(f["view"]).name: f for f in json.loads(out.with_suffix(".fit.json").read_text())}
        front_az = fit["right.png"]["az"] + 90.0  # right is -90 from the front camera
        gap = (fit["photo.png"]["az"] - front_az + 180) % 360 - 180
        self.assertLessEqual(abs(gap), 50)


def patched(mesh: trimesh.Trimesh, size: int, colour=(128, 128, 128)) -> trimesh.Trimesh:
    """The mesh with unshared corners, each triangle its own patch of a size atlas filled with colour."""
    out = trimesh.Trimesh(vertices=mesh.vertices[mesh.faces].reshape(-1, 3),
                          faces=np.arange(len(mesh.faces) * 3).reshape(-1, 3), process=False)
    n = len(out.faces)
    cols = int(np.ceil(np.sqrt(n)))
    uv = np.zeros((n * 3, 2))
    d = 0.9 / cols
    for i in range(n):
        x, y = (i % cols) / cols, (i // cols) / cols
        uv[i * 3:i * 3 + 3] = [(x, y), (x + d, y), (x, y + d)]
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (size, size), colour))
    out.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return out


class CoverAndEndMirrorTests(unittest.TestCase):
    """The coverage image beside the albedo, and --mirror for a front-on photo: the far end gets the photo too."""

    SIZE = 512

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # A short drum lying along z, like the hay bale: its two round ends face +z and -z.
        drum = trimesh.creation.cylinder(radius=0.5, height=0.6, sections=24).subdivide()
        self.mesh = patched(drum, self.SIZE)
        self.glb = self.tmp / "drum.glb"
        self.mesh.export(self.glb)
        a = np.zeros((128, 128, 4), np.uint8)
        yy, xx = np.mgrid[:128, :128]
        a[(yy - 64) ** 2 + (xx - 64) ** 2 <= 54 ** 2] = (220, 30, 30, 255)  # the round end, seen head-on
        Image.fromarray(a, "RGBA").save(self.tmp / "photo.png")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def bake(self, *extra: str) -> tuple[np.ndarray, np.ndarray]:
        out = self.tmp / "out.glb"
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "retexture.py"), str(self.glb), "-o", str(out),
                               "--view", str(self.tmp / "photo.png"), "--first-facing", "camera", "--samples",
                               "120000", *extra], capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        albedo = np.asarray(Image.open(out.with_suffix(".albedo.png")).convert("RGB")).astype(int)
        cover = Image.open(out.with_suffix(".cover.png"))
        self.assertEqual(cover.mode, "L")
        return albedo, np.asarray(cover)

    def cap_red(self, albedo: np.ndarray) -> tuple[float, float]:
        """Share of each end's triangles whose patch centre came out red: (+z end, -z end)."""
        uv = np.asarray(self.mesh.visual.uv).reshape(-1, 3, 2).mean(1)
        px = np.clip((uv[:, 0] * (self.SIZE - 1)).astype(int), 0, self.SIZE - 1)
        py = np.clip(((1 - uv[:, 1]) * (self.SIZE - 1)).astype(int), 0, self.SIZE - 1)
        c = albedo[py, px]
        red = (c[:, 0] > 160) & (c[:, 1] < 90)
        nz = self.mesh.face_normals[:, 2]
        return float(red[nz > 0.9].mean()), float(red[nz < -0.9].mean())

    def test_cover_image_is_written_beside_the_albedo(self) -> None:
        albedo, cover = self.bake()
        self.assertEqual(cover.shape, albedo.shape[:2])
        self.assertEqual(cover.dtype, np.uint8)
        self.assertGreater(int(cover.max()), 200)  # clipped confidence 1 reads 255 where the photo landed
        self.assertEqual(int(cover.min()), 0)       # and 0 where nothing did
        # Every texel the bake turned red is one the cover marks as reached (the fade ring below keep included).
        red = (albedo[..., 0] > 160) & (albedo[..., 1] < 90)
        self.assertGreater(red.sum(), 100)
        self.assertGreater(float((cover[red] > 0).mean()), 0.99)
        self.assertGreater(float((cover[red] >= 0.18 * 255).mean()), 0.8)

    def test_mirror_puts_a_front_on_photo_on_the_far_end_too(self) -> None:
        # A free fit of a disc on a drum may land on either end, so ask that one end only is red, then both.
        plain = self.cap_red(self.bake()[0])
        self.assertGreater(max(plain), 0.6, plain)
        self.assertLess(min(plain), 0.1, plain)
        albedo, cover = self.bake("--mirror")
        both = self.cap_red(albedo)
        self.assertGreater(min(both), 0.6, both)
        # The barrel between the ends, side-on to the camera, stays mostly unreached: the cover says so.
        self.assertLess(float((cover >= 0.18 * 255).mean()), 0.9)

    def test_mirror_az_reflects_across_the_mid_plane_across_the_line_of_sight(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import retexture

        self.assertAlmostEqual(retexture.mirror_az(25.0, 0.0), 155.0)    # looking along z: through z
        self.assertAlmostEqual(retexture.mirror_az(80.0, 0.0), 280.0)    # looking along x: through x
        self.assertAlmostEqual(retexture.mirror_az(80.0, -90.0), 100.0)  # side-on: always the z mid-plane, as before


if __name__ == "__main__":
    unittest.main()
