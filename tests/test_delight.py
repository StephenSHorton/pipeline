#!/usr/bin/env python3
"""delight.py takes a photo's lighting out before the bake: shade lifted toward the lit level, texture, hue, the
background and alpha kept. retexture.py --delight bakes the de-lit colours from the same cameras, and fill.py takes
its swatch from the same de-lit pixels."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import delight  # noqa: E402
import retexture  # noqa: E402

N = 256


def shaded_disc(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rgba, mask, patch): seeded noise stripes (the material) times a smooth shadow ramp (the light: the right side
    at 20% brightness) in a disc on a magenta ground. patch marks a uniformly coloured square on the shaded side."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    mask = (yy - N / 2) ** 2 + (xx - N / 2) ** 2 <= (N * 0.42) ** 2
    stripes = 1.0 + 0.25 * np.sign(np.sin(xx * 0.9 + yy * 0.4)) * rng.uniform(0.5, 1.0, (N, N))
    albedo = np.stack([0.55 * stripes, 0.42 * stripes, 0.18 * stripes], -1)  # straw, linear
    patch = (yy >= 110) & (yy < 140) & (xx >= 170) & (xx < 200)
    albedo[patch] = (0.30, 0.12, 0.40)
    t = np.clip((xx - N * 0.1) / (N * 0.8), 0.0, 1.0)
    light = 1.0 - 0.8 * (t * t * (3 - 2 * t))  # 1.0 on the left, 0.2 on the right, smooth
    lin = np.clip(albedo * light[..., None], 0.0, 1.0)
    rgb = np.rint(delight._to_srgb(lin) * 255).astype(np.uint8)
    a = np.zeros((N, N, 4), np.uint8)
    a[..., 0], a[..., 2] = 255, 255
    a[mask, :3] = rgb[mask]
    a[..., 3] = np.where(mask, 255, 0)
    return a, mask, patch


def log_lum(rgba: np.ndarray) -> np.ndarray:
    lin = delight._to_linear(rgba[..., :3].astype(np.float64) / 255.0)
    return np.log(np.maximum(delight._luminance(lin), 1e-4))


def low_high(rgba: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask-normalised low-pass of log-luminance and the residual, inside the mask."""
    L = log_lum(rgba)
    w = mask.astype(np.float64)
    low = ndimage.gaussian_filter(L * w, 12) / np.maximum(ndimage.gaussian_filter(w, 12), 1e-6)
    return low, L - low


class DelightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgba, self.mask, self.patch = shaded_disc()
        self.inner = ndimage.binary_erosion(self.mask, iterations=10)
        self.out = delight.delight(self.rgba)

    def test_the_shadow_ramp_is_lifted(self) -> None:
        before, _ = low_high(self.rgba, self.mask)
        after, _ = low_high(self.out, self.mask)
        sb, sa = before[self.inner].std(), after[self.inner].std()
        self.assertLess(sa, 0.3 * sb, (sb, sa))
        # Lifted toward the lit level, not everything darkened: the right side came up, the left barely moved.
        right = self.inner & (np.arange(N)[None, :] > N * 0.7)
        left = self.inner & (np.arange(N)[None, :] < N * 0.3)
        self.assertGreater(after[right].mean() - before[right].mean(), 1.0)
        self.assertLess(abs(after[left].mean() - before[left].mean()), 0.15)

    def test_the_texture_keeps_its_contrast(self) -> None:
        _, hb = low_high(self.rgba, self.mask)
        _, ha = low_high(self.out, self.mask)
        sel = self.inner & ~ndimage.binary_dilation(self.patch, iterations=14)
        self.assertGreater(ha[sel].std(), 0.75 * hb[sel].std(), (hb[sel].std(), ha[sel].std()))

    def test_a_coloured_patch_keeps_its_hue_and_chroma(self) -> None:
        core = ndimage.binary_erosion(self.patch, iterations=3)

        def chroma(rgba):
            lin = delight._to_linear(rgba[core, :3].astype(np.float64) / 255.0).mean(0)
            return lin / lin.sum()

        cb, ca = chroma(self.rgba), chroma(self.out)
        np.testing.assert_allclose(ca, cb, atol=0.01)
        hb = np.asarray(Image.fromarray(self.rgba[..., :3]).convert("HSV"))[core][:, 0].astype(float).mean()
        ha = np.asarray(Image.fromarray(self.out[..., :3]).convert("HSV"))[core][:, 0].astype(float).mean()
        self.assertLess(abs(ha - hb), 3.0)
        self.assertGreater(self.out[core, :3].astype(float).mean(), self.rgba[core, :3].astype(float).mean() + 20)

    def test_outside_the_mask_and_alpha_are_untouched(self) -> None:
        np.testing.assert_array_equal(self.out[~self.mask], self.rgba[~self.mask])
        np.testing.assert_array_equal(self.out[..., 3], self.rgba[..., 3])

    def test_strength_zero_is_identity(self) -> None:
        np.testing.assert_array_equal(delight.delight(self.rgba, strength=0.0), self.rgba)

    def test_nothing_clips_and_pil_in_pil_out(self) -> None:
        # The gain only rolls bright pixels off toward white: no lifted channel reaches 255 unless it already did.
        newly = (self.out[..., :3] == 255) & (self.rgba[..., :3] < 255)
        self.assertEqual(int(newly[self.mask].sum()), 0)
        im = delight.delight(Image.fromarray(self.rgba, "RGBA"))
        self.assertIsInstance(im, Image.Image)
        np.testing.assert_array_equal(np.asarray(im), self.out)

    def test_trust_falls_only_where_the_lift_is_deep(self) -> None:
        out, report, trust = delight.delight_full(self.rgba)
        np.testing.assert_array_equal(out, self.out)
        left = self.inner & (np.arange(N)[None, :] < N * 0.3)
        right = self.inner & (np.arange(N)[None, :] > N * 0.85)
        self.assertEqual(float(trust[left].min()), 1.0)
        self.assertLess(float(trust[right].mean()), 0.2)
        self.assertTrue(((trust >= 0) & (trust <= 1)).all())
        self.assertTrue((trust[~self.mask] == 1).all())
        self.assertGreater(report["low_trust"], 0.0)
        _, _, flat = delight.delight_full(self.rgba, strength=0.0)
        self.assertTrue((flat == 1).all())

    def test_a_bake_skips_the_after_summary(self) -> None:
        # report False (what retexture.load_photo asks for) skips the second edge-aware blur, nothing else.
        out, report, trust = delight.delight_full(self.rgba)
        out2, report2, trust2 = delight.delight_full(self.rgba, report=False)
        np.testing.assert_array_equal(out2, out)
        np.testing.assert_array_equal(trust2, trust)
        self.assertNotIn("lowfreq_var_after", report2)
        self.assertEqual(report2, {k: v for k, v in report.items() if k != "lowfreq_var_after"})

    def test_a_hard_shadow_border_leaves_no_band(self) -> None:
        # Straw noise under a shadow 4x darker than the lit side, its border hard (the bale's right side). Beside the
        # border the lit straw must not glow and the shade must come all the way up: a guide blurred too wide gave
        # both sides a mid level there (lit 1.33x, shade at 0.74 of lit 15 px out), which baked in as grey streaks.
        n = 600
        rng = np.random.default_rng(1)
        yy, xx = np.mgrid[:n, :n].astype(np.float64)
        mask = (yy - n / 2) ** 2 + (xx - n / 2) ** 2 <= (n * 0.42) ** 2
        tex = np.clip(0.55 + 0.25 * rng.standard_normal((n, n)), 0.05, 1.0)
        albedo = np.stack([tex * 0.9, tex * 0.75, tex * 0.4], -1)
        border = int(n / 2 + 0.1 * n)
        light = np.where(xx >= border, 0.25, 1.0)
        rgba = np.zeros((n, n, 4), np.uint8)
        rgba[..., :3] = np.rint(delight._to_srgb(albedo * light[..., None]) * 255)
        rgba[..., 3] = np.where(mask, 255, 0)
        out = delight.delight(rgba)
        ratio = delight._luminance(delight._to_linear(out[..., :3] / 255.0)) / delight._luminance(albedo)
        rows = slice(n // 2 - 25, n // 2 + 25)
        for d in (3, 8):
            lit = float(np.median(ratio[rows, border - 1 - d]))
            shade = float(np.median(ratio[rows, border + d]))
            self.assertLess(lit, 1.12, d)
            self.assertGreater(shade, 0.88, d)

    def test_cli_prints_a_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.png", Path(td) / "out.png"
            Image.fromarray(self.rgba, "RGBA").save(src)
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "delight.py"), str(src), "-o", str(dst)],
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertGreater(report["shadow_lift"], 2.0)
            self.assertLess(report["lowfreq_var_after"], 0.3 * report["lowfreq_var_before"])
            np.testing.assert_array_equal(np.asarray(Image.open(dst)), self.out)


def ramp_box(path: Path) -> None:
    """A unit box, each triangle its own patch of a grey atlas (as test_retexture's)."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).subdivide().subdivide()
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


class RetextureDelightTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.glb = self.tmp / "box.glb"
        ramp_box(self.glb)
        # The box from the front: a square of straw stripes, its right half in shadow.
        rgba, _, _ = shaded_disc()
        a = np.zeros((N, N, 4), np.uint8)
        a[48:208, 48:208] = rgba[48:208, 48:208]
        a[48:208, 48:208, 3] = 255
        a[48:208, 48:208, :3] = np.where(rgba[48:208, 48:208, 3:] > 0, rgba[48:208, 48:208, :3], 120)
        self.photo = self.tmp / "front.png"
        Image.fromarray(a, "RGBA").save(self.photo)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def bake(self, *extra: str) -> tuple[np.ndarray, list[dict], Path]:
        out = self.tmp / f"out{len(extra)}.glb"
        real = trimesh.sample.sample_surface
        argv = ["retexture.py", str(self.glb), "-o", str(out), "--view", str(self.photo), "--samples", "40000",
                *extra]
        # The same surface samples both times (sample_surface is unseeded), so the fits can be compared exactly.
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(trimesh.sample, "sample_surface", lambda m, n, **k: real(m, n, seed=7, **k)):
            self.assertEqual(retexture.main(), 0)
        albedo = np.asarray(Image.open(out.with_suffix(".albedo.png")).convert("RGB")).astype(float)
        return albedo, json.loads(out.with_suffix(".fit.json").read_text()), out

    def test_the_fit_is_the_same_and_the_shade_bakes_brighter(self) -> None:
        plain, fit0, out0 = self.bake()
        lit, fit1, out1 = self.bake("--delight", "1")
        self.assertEqual(fit0, fit1)
        # The masks come from the untouched alpha. Only shade too deep to trust splats lighter: no texel gains
        # coverage, and the lit side keeps exactly what it had.
        rgb0, af0, ab0 = retexture.load_photo(self.photo)
        rgb1, af1, ab1, trust = retexture.load_photo(self.photo, 1.0, with_trust=True)
        np.testing.assert_array_equal(af0, af1)
        np.testing.assert_array_equal(ab0, ab1)
        cover0 = np.asarray(Image.open(out0.with_suffix(".cover.png"))).astype(int)
        cover1 = np.asarray(Image.open(out1.with_suffix(".cover.png"))).astype(int)
        self.assertTrue((cover1 <= cover0).all())
        self.assertLess(trust[ab0].min(), 0.5)
        self.assertEqual(float(trust[ab0 & (np.arange(N)[None, :] < 100)].min()), 1.0)
        # Shaded photo pixels (right half of the square) come out brighter, so the texels they bake to do too.
        shade = np.zeros(af0.shape, bool)
        shade[60:196, 150:196] = True
        self.assertGreater(rgb1[shade].mean(), rgb0[shade].mean() + 25)
        hit = cover0 > 0.18 * 255
        dark = hit & (plain.mean(-1) < np.percentile(plain[hit].mean(-1), 25))
        self.assertGreater(dark.sum(), 50)
        self.assertGreater(lit[dark].mean(), plain[dark].mean() + 20)

    def test_fill_takes_its_swatch_from_the_de_lit_photo(self) -> None:
        from workbench import fill

        _, _, out = self.bake("--delight", "1")
        plain = fill.fill_unseen(out, out.with_suffix(".cover.png"), self.photo, self.tmp / "f0.glb")
        lit = fill.fill_unseen(out, out.with_suffix(".cover.png"), self.photo, self.tmp / "f1.glb", delight=1.0)
        self.assertNotIn("delight", plain)
        self.assertEqual(lit["delight"], 1.0)
        # Same swatch window (the shade is judged on the photo as lit either way), de-lit pixels in it.
        self.assertEqual(lit["swatch"], plain["swatch"])
        # The pixels the swatch is cut from are the de-lit photo's, the shade judged on the photo as it was lit.
        seen = []
        real = fill.pick_swatch
        with mock.patch.object(fill, "pick_swatch",
                               lambda rgb, *a, **k: (seen.append((rgb, k.get("shade_rgb"))), real(rgb, *a, **k))[1]):
            fill.fill_unseen(out, out.with_suffix(".cover.png"), self.photo, self.tmp / "f2.glb", delight=1.0)
        (rgb, shade_rgb), = seen
        np.testing.assert_array_equal(rgb, retexture.load_photo(self.photo, 1.0)[0])
        np.testing.assert_array_equal(shade_rgb, retexture.load_photo(self.photo)[0])
        self.assertGreater(np.abs(rgb.astype(float) - shade_rgb).max(), 25)

    def test_the_swatch_shade_is_judged_on_the_lit_photo(self) -> None:
        from workbench import fill

        # De-lit, the left half is plain straw and the right half (lifted shade) the busiest texture, which the
        # swatch score would pick. As lit, that right half was at a fifth of the brightness: given the lit photo,
        # the window stays on the left, and its pixels are still the de-lit ones.
        rng = np.random.default_rng(3)
        rgb = np.clip(rng.normal(140, 8, (120, 200, 3)), 0, 255).astype(np.float32)
        rgb[:, 100:] = np.clip(rng.normal(140, 40, (120, 100, 3)), 0, 255)
        lit = rgb.copy()
        lit[:, 100:] *= 0.2
        mask = np.ones((120, 200), bool)
        _, (x, _, side) = fill.pick_swatch(rgb, mask, side=40)
        self.assertGreaterEqual(x, 100)
        sw, (x, y, side) = fill.pick_swatch(rgb, mask, side=40, shade_rgb=lit)
        self.assertLessEqual(x + side, 100)
        self.assertGreater(sw.mean(), 100)


if __name__ == "__main__":
    unittest.main()
