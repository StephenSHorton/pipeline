#!/usr/bin/env python3
"""workbench/fill.py: every texel the photo bake did not reach gets a triplanar swatch of the photo's subject.

A synthetic drum (xatlas UVs, so real seams) stands in for the clay bale: its +z end and the front of its barrel
are "baked" (coverage 255, a known straw-like colour), the rest is the flat base colour retexture leaves. After the
fill no texel may still be that base colour, the baked texels must be untouched, and the colour must run on across
UV seams about as smoothly as it does inside a chart.
"""
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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from workbench import fill, mesher  # noqa: E402

BASE = (200, 160, 80)  # the clay's flat hay-tan
SIZE = 256


def straw_photo(path: Path, size: int = 160, shadow: bool = False) -> Path:
    """A disc of streaky straw (a known texture) on a transparent ground. shadow: its left half is in deep shade
    and flat, so the swatch must come from the right."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    streak = np.sin(xx * 0.9 + 3 * np.sin(yy * 0.15)) * 30 + rng.normal(0, 12, (size, size))
    rgb = np.stack([170 + streak, 140 + streak * 0.9, 60 + streak * 0.5], -1)
    if shadow:
        rgb[:, : size // 2] = (40, 32, 15)
    a = np.zeros((size, size, 4), np.uint8)
    a[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    a[..., 3] = np.where((yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size * 0.45) ** 2, 255, 0)
    Image.fromarray(a, "RGBA").save(path)
    return path


class FillTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        drum = trimesh.creation.cylinder(radius=0.5, height=0.6, sections=32)
        vmap, faces, uvs = mesher.unwrap(drum.vertices, drum.faces, SIZE)
        verts = np.asarray(drum.vertices)[vmap]
        f, bary, tx, ty = mesher.texels(uvs, faces, SIZE)
        pos = np.einsum("ni,nij->nj", bary, verts[faces[f]])
        # Baked: the +z end and the front half of the barrel (the side the photo saw), in a known colour per point.
        seen = (pos[:, 2] > 0.29) | (pos[:, 1] > 0.1)
        albedo = np.zeros((SIZE, SIZE, 3), np.uint8)
        albedo[:] = BASE
        cover = np.zeros((SIZE, SIZE), np.uint8)
        known = (np.stack([150 + 40 * np.sin(pos[:, 0] * 40), 120 + 30 * np.sin(pos[:, 1] * 40),
                           60 + 20 * np.sin(pos[:, 2] * 40)], -1)).astype(np.uint8)
        albedo[ty[seen], tx[seen]] = known[seen]
        cover[ty[seen], tx[seen]] = 255
        mesh = trimesh.Trimesh(verts, faces, process=False)
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(albedo, "RGB"))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)
        self.glb = self.tmp / "baked.glb"
        mesh.export(self.glb)
        self.cover = self.tmp / "baked.cover.png"
        Image.fromarray(cover, "L").save(self.cover)
        self.photo = straw_photo(self.tmp / "photo.png")
        self.albedo, self.cover_arr = albedo, cover
        self.texels = (f, tx, ty, pos)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def filled(self) -> np.ndarray:
        out = self.tmp / "filled.glb"
        report = fill.fill_unseen(self.glb, self.cover, self.photo, out)
        self.assertEqual(report["covered_after"], 1.0)
        self.assertGreater(report["filled"], 1000)
        mesh = trimesh.load(out, force="mesh")
        tex = mesh.visual.material.baseColorTexture
        return np.asarray(tex.convert("RGB")).astype(int)

    def test_no_texel_keeps_the_flat_base_colour(self) -> None:
        self.assertGreater(int((self.albedo == BASE).all(-1).sum()), 10000)  # plenty of base before
        out = self.filled()
        self.assertEqual(int((out == BASE).all(-1).sum()), 0)

    def test_baked_texels_are_unchanged(self) -> None:
        out = self.filled()
        hit = self.cover_arr >= 0.18 * 255
        self.assertGreater(int(hit.sum()), 1000)
        np.testing.assert_array_equal(out[hit], self.albedo[hit].astype(int))

    def test_the_fill_runs_on_across_uv_seams(self) -> None:
        # Pairs of unseen texels about one texel apart on the surface: those on different charts (far apart in the
        # atlas, i.e. across a seam) must differ no more than neighbours inside one chart, give or take.
        out = self.filled()
        f, tx, ty, pos = self.texels
        unseen = self.cover_arr[ty, tx] == 0
        p, x, y = pos[unseen], tx[unseen], ty[unseen]
        step = 1.6 * np.pi * 0.5 / SIZE  # about a texel on the surface
        pairs = np.array(sorted(cKDTree(p).query_pairs(step, output_type="set")))
        self.assertGreater(len(pairs), 1000)
        uv_gap = np.hypot(x[pairs[:, 0]] - x[pairs[:, 1]], y[pairs[:, 0]] - y[pairs[:, 1]])
        diff = np.abs(out[y[pairs[:, 0]], x[pairs[:, 0]]] - out[y[pairs[:, 1]], x[pairs[:, 1]]]).mean(-1)
        across, inside = diff[uv_gap > 8], diff[uv_gap <= 2]
        self.assertGreater(len(across), 20, "the drum should have seams in the unseen part")
        self.assertLess(float(across.mean()), 1.5 * float(inside.mean()) + 4.0, (across.mean(), inside.mean()))
        self.assertLess(float(across.mean()), 30.0)

    def test_the_fill_matches_the_baked_colour(self) -> None:
        out = self.filled()
        hit = self.cover_arr >= 0.18 * 255
        f, tx, ty, pos = self.texels
        unseen = np.zeros_like(hit)
        unseen[ty, tx] = True
        unseen &= ~hit
        gap = np.abs(out[unseen].mean(0) - self.albedo[hit].mean(0))
        self.assertLess(float(gap.max()), 25.0, gap)

    def test_charts_join_up_inside_and_split_at_seams(self) -> None:
        f, tx, ty, pos = self.texels
        mesh = trimesh.load(self.glb, force="mesh", process=False)
        verts, faces, uvs = np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.visual.uv)
        charts = fill.uv_charts(verts, uvs, faces)
        self.assertGreater(len(np.unique(charts)), 1)   # a drum unwraps into several charts
        # A triangle soup (no shared vertices) groups into the same charts.
        soup = fill.uv_charts(verts[faces].reshape(-1, 3), uvs[faces].reshape(-1, 2), np.arange(faces.size).reshape(-1, 3))
        pairs = set(zip(charts.tolist(), soup.tolist()))
        self.assertEqual(len(pairs), len(np.unique(charts)))

    def test_the_photo_does_not_leak_to_a_chart_beside_it_in_the_atlas(self) -> None:
        # Only the largest chart is baked, in a strong blue; the charts packed next to it are unseen. Their texels
        # within a feather of it in the atlas (another part of the model) must look like the rest of their chart.
        mesh = trimesh.load(self.glb, force="mesh", process=False)
        f, tx, ty, pos = self.texels
        charts = fill.uv_charts(np.asarray(mesh.vertices), np.asarray(mesh.visual.uv), np.asarray(mesh.faces))
        chart = np.full((SIZE, SIZE), -1)
        chart[ty, tx] = charts[f]
        big = chart == int(np.argmax(np.bincount(charts[f])))
        albedo = np.zeros((SIZE, SIZE, 3), np.uint8)
        albedo[:] = BASE
        albedo[big] = (40, 60, 220)
        cover = np.where(big, 255, 0).astype(np.uint8)
        mesh.visual.material.baseColorTexture = Image.fromarray(albedo, "RGB")
        mesh.export(self.glb)
        Image.fromarray(cover, "L").save(self.cover)
        out = self.filled().astype(float)
        from scipy import ndimage
        d = ndimage.distance_transform_edt(~big)
        other = (chart >= 0) & ~big
        near, far = other & (d <= 6), other & (d > 20)
        self.assertGreater(int(near.sum()), 50, "the drum's charts should pack within a feather of the big one")
        blue = lambda m: float((out[m][:, 2] - out[m][:, 0]).mean())  # noqa: E731
        # Leaning on the atlas neighbour made this band about 60 bluer than the rest; same chart only, about 0.
        self.assertLess(abs(blue(near) - blue(far)), 10.0, (blue(near), blue(far)))
        np.testing.assert_array_equal(out[big], albedo[big].astype(float))

    def test_swatch_avoids_the_shade_and_tiles(self) -> None:
        rgb = np.asarray(Image.open(straw_photo(self.tmp / "shade.png", shadow=True)).convert("RGBA"))
        mask = rgb[..., 3] > 220
        swatch, (x0, y0, side) = fill.pick_swatch(rgb[..., :3].astype(np.float32), mask)
        self.assertGreaterEqual(x0, 160 // 2 - 2, "the swatch should come from the lit, textured half")
        self.assertTrue(mask[y0:y0 + side, x0:x0 + side].all())
        tile = fill.tileable(swatch)
        # Wrapping costs no more than a step inside the tile: the last row meets the first like any two rows.
        wrap = np.abs(tile[0] - tile[-1]).mean()
        step = np.abs(np.diff(tile, axis=0)).mean()
        self.assertLess(wrap, 1.5 * step + 1.0)

    def test_cli_writes_the_glb_albedo_and_report(self) -> None:
        out = self.tmp / "cli.glb"
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "workbench" / "fill.py"), str(self.glb), "-o",
                               str(out), "--photo", str(self.photo)], capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)  # --cover defaults to <glb>.cover.png
        self.assertTrue(out.is_file())
        self.assertTrue(out.with_suffix(".albedo.png").is_file())
        report = json.loads(out.with_suffix(".fill.json").read_text())
        self.assertLess(report["covered_before"], 0.9)
        missing = subprocess.run([sys.executable, str(ROOT / "scripts" / "workbench" / "fill.py"), str(self.glb), "-o",
                                  str(out), "--photo", str(self.photo), "--cover", str(self.tmp / "none.png")],
                                 capture_output=True, text=True, timeout=300)
        self.assertEqual(missing.returncode, 1)
        self.assertIn("no coverage image", missing.stderr)


if __name__ == "__main__":
    unittest.main()
