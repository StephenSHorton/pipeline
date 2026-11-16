#!/usr/bin/env python3
"""retexture.py --first-wins: a real photo bakes over the drawn views instead of being averaged with them, and is
fitted near the drawn view it looks like."""
from __future__ import annotations

import json
import os
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
        albedo, cover = self.bake("--mirror", "--mirror-ends")
        both = self.cap_red(albedo)
        self.assertGreater(min(both), 0.6, both)
        # The barrel between the ends, side-on to the camera, stays mostly unreached: the cover says so.
        self.assertLess(float((cover >= 0.18 * 255).mean()), 0.9)

    def test_mirror_az_reflects_across_the_mid_plane_across_the_line_of_sight(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import retexture

        # --mirror-ends: the vertical mid-plane most across the line of sight.
        self.assertAlmostEqual(retexture.mirror_az(25.0, 0.0, True), 155.0)    # looking along z: through z
        self.assertAlmostEqual(retexture.mirror_az(80.0, 0.0, True), 280.0)    # looking along x: through x
        self.assertAlmostEqual(retexture.mirror_az(80.0, -90.0, True), 100.0)  # side-on: always the z mid-plane
        # Plain --mirror is what it always was, for every view: through z (a ref of the front end fitted at 105).
        for az, off in ((80.0, 0.0), (105.0, None), (80.0, 180.0), (25.0, 0.0), (80.0, -90.0)):
            self.assertAlmostEqual(retexture.mirror_az(az, off), (180.0 - az) % 360)


class SelectOcclusionTests(unittest.TestCase):
    """--select's visibility: a face hidden behind another from a view's camera does not take that view."""

    def test_a_hidden_face_does_not_take_the_view_in_front_of_it(self) -> None:
        # A small box hidden behind a big one from the front: its front face, which no camera sees, keeps the paint.
        # observe() once got NDC depth against a world-units tol, which hid nothing: that face came out red.
        big = trimesh.creation.box(extents=(1.0, 1.0, 0.5))
        big.apply_translation((0.0, 0.0, 0.5))
        small = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
        small.apply_translation((0.0, 0.0, -0.5))
        both = trimesh.util.concatenate([big.subdivide().subdivide(), small.subdivide().subdivide()])
        mesh = patched(both, 512)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            glb, out = tmp / "boxes.glb", tmp / "out.glb"
            mesh.export(glb)
            front = cutout(tmp / "front.png", (220, 30, 30))
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "retexture.py"), str(glb), "-o", str(out),
                                   "--view", str(front), "--first-facing", "camera", "--front-az", "0", "--select",
                                   "--samples", "60000"], capture_output=True, text=True, timeout=300,
                                  env={**os.environ, "RETEXTURE_LABELS": "1"})
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            albedo = np.asarray(Image.open(out.with_suffix(".albedo.png")).convert("RGB")).astype(int)
        h, w = albedo.shape[:2]
        centre = mesh.triangles_center
        uv = mesh.visual.uv[mesh.faces].mean(axis=1)
        px = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
        py = np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
        col = albedo[py, px]
        red = (col[:, 0] > 150) & (col[:, 1] < 120)  # RETEXTURE_LABELS: the view's faces red, the paint's green
        nz = mesh.face_normals[:, 2]
        seen = (nz > 0.9) & (centre[:, 2] > 0.7)      # the big box's front face
        hidden = (nz > 0.9) & (centre[:, 2] < -0.2)   # the small box's front face, behind the big box
        self.assertGreater(float(red[seen].mean()), 0.9)
        self.assertLess(float(red[hidden].mean()), 0.1)


class FlatPaintTests(unittest.TestCase):
    """--flat-paint helpers: small dark features drawn bolder, and a side view's head fitted on its own."""

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import retexture
        self.rt = retexture

    def test_small_dark_features_grow_and_big_dark_patches_stay(self) -> None:
        rgb = np.full((200, 200, 3), 220.0, np.float32)
        rgb[100:104, 60:64] = 20.0      # an eye: 4 px
        rgb[20:80, 120:180] = 20.0      # a dark patch: 3,600 px, shading or a dark material
        alpha = np.ones((200, 200), bool)
        out, n = self.rt.bold_features(rgb, alpha, grow=3)
        self.assertEqual(n, 1)
        grown = int((out[85:120, 45:80, 0] < 60).sum())
        self.assertGreater(grown, 3 * 16)  # the 4 px eye, grown 3 px all round (a diamond, not quite 10 x 10)
        self.assertLessEqual(grown, 10 * 10)
        np.testing.assert_array_equal(out[10:90, 110:190], rgb[10:90, 110:190])

    def test_the_head_is_fitted_on_the_ear(self) -> None:
        # Mesh head samples from the neck (x 100) to the nose (x 200), its ear tip at x 180; the drawing's ear tip
        # is at x 140. The ear moves onto the drawing's, the neck and the nose stay.
        xs = np.arange(100, 201)
        px = np.concatenate([xs, xs, [180]]).astype(np.int32)
        py = np.concatenate([np.full(len(xs), 80), np.full(len(xs), 60), [20]]).astype(np.int32)
        head = np.ones(len(px), bool)
        inb = np.ones(len(px), bool)
        px = np.concatenate([px, [10, 20]]).astype(np.int32)  # the body behind, so the nose points right
        py = np.concatenate([py, [70, 70]]).astype(np.int32)
        head = np.concatenate([head, [False, False]])
        inb = np.concatenate([inb, [True, True]])
        alpha = np.zeros((120, 240), bool)
        alpha[60:81, 100:201] = True
        alpha[20:60, 138:143] = True  # the drawn ear
        nx, ny = self.rt.fit_head(px, py, inb, head, alpha)
        self.assertTrue(self.rt.fit_head.report["moved"])
        self.assertEqual(int(nx[2 * len(xs)]), 140)                         # the ear tip
        self.assertEqual((int(nx[0]), int(nx[len(xs) - 1])), (100, 200))  # neck and nose
        np.testing.assert_array_equal(ny, py)
        np.testing.assert_array_equal(nx[-2:], px[-2:])                    # the body does not move


if __name__ == "__main__":
    unittest.main()

    def test_fit_snout_puts_the_mesh_snout_on_the_drawn_nostrils(self) -> None:
        h = w = 200
        rgb = np.full((h, w, 3), 200.0)
        alpha = np.zeros((h, w), bool)
        alpha[20:190, 30:170] = True
        for cx, cy, r in ((75, 60, 7), (125, 60, 7), (88, 110, 8), (112, 110, 8)):  # eyes, then nostrils
            yy, xx = np.ogrid[:h, :w]
            rgb[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 20.0
        x, y = self.rt.nostrils(rgb, alpha)
        self.assertAlmostEqual(x, 100.0, delta=1.5)
        self.assertAlmostEqual(y, 110.0, delta=1.5)
        rng = np.random.default_rng(0)
        pts = rng.uniform(-1, 1, (4000, 3))
        tip = pts[:, 2] > np.percentile(pts[:, 2], 99.7)
        px = np.full(len(pts), 100, np.int32)
        py = np.where(tip, 70, rng.integers(25, 185, len(pts))).astype(np.int32)
        inb = np.ones(len(pts), bool)
        _, out = self.rt.fit_snout(px, py, inb, pts, np.array([0.0, 0.0, 5.0]), np.zeros(3), rgb, alpha)
        self.assertTrue(self.rt.fit_snout.report["moved"])
        self.assertAlmostEqual(float(np.median(out[tip])), 110.0, delta=1.0)
        self.assertEqual((int(out.min()) >= 20, int(out.max()) <= 189), (True, True))
        # no nostrils drawn: left as it was
        _, same = self.rt.fit_snout(px, py, inb, pts, np.array([0.0, 0.0, 5.0]), np.zeros(3), np.full((h, w, 3), 200.0), alpha)
        np.testing.assert_array_equal(same, py)
