#!/usr/bin/env python3
"""proportion_check.py on a synthetic box creature: a side-on photo restores a squashed length, a front-on or
three-quarter photo leaves it alone, a small difference is skipped and the clamp holds.

The creature is boxes (body, four legs with gaps, a head out front), long along x, so its side view (az 0 after
mv_orient, canonical right) and its end views differ in more than their aspect. The "photo" is an orthographic
silhouette of the true creature; the mesh under test is that creature squashed along x.
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "proportion_check.py"
sys.path.insert(0, str(ROOT / "scripts"))
from mv_orient import silhouette  # noqa: E402


def creature() -> trimesh.Trimesh:
    """Length 2.325 (x, the head out past the body), height 1.25 (y), width 0.7 (z); the head toward +x."""
    def box(ext, at):
        b = trimesh.creation.box(extents=ext)
        b.apply_translation(at)
        return b
    parts = [box((2.0, 0.7, 0.7), (0.0, 0.85, 0.0)),            # body
             box((0.45, 0.4, 0.4), (1.1, 1.05, 0.0))]           # head, out past the body
    for x in (-0.8, 0.8):
        for z in (-0.25, 0.25):
            parts.append(box((0.15, 0.55, 0.15), (x, 0.275, z)))  # legs, floor at y 0
    return trimesh.util.concatenate(parts)


def squashed(f: float) -> trimesh.Trimesh:
    m = creature()
    m.apply_scale((f, 1.0, 1.0))
    return m


def photo_png(path: Path, az: float) -> Path:
    """An RGBA cut-out of the true creature from az, padded like a photo's cut-out."""
    pts, _ = trimesh.sample.sample_surface(creature(), 120_000, seed=1)
    mask = silhouette(np.asarray(pts), az, res=480)
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    a = np.zeros((crop.shape[0] + 80, crop.shape[1] + 120, 4), np.uint8)
    a[40:-40, 60:-60, :3] = 200
    a[40:-40, 60:-60, 3] = crop * 255
    Image.fromarray(a, "RGBA").save(path)
    return path


def aspect_xy(mesh: trimesh.Trimesh) -> float:
    ext = mesh.extents
    return float(ext[1] / ext[0])


class ProportionCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.true_aspect = aspect_xy(creature())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_check(self, mesh: trimesh.Trimesh, az: float, *extra: str):
        glb = self.dir / "mesh_hero.glb"
        mesh.export(glb)
        photo = photo_png(self.dir / "photo.png", az)
        sj = self.dir / "shape.json"
        sj.write_text(json.dumps({"model": "stub", "orient": {"yaw": 90}}))
        proc = subprocess.run([sys.executable, str(SCRIPT), str(glb), "--photo", str(photo), "--canonical", "right",
                               "--json", str(sj), *extra], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        data = json.loads(sj.read_text())
        self.assertEqual(data["model"], "stub")            # merged, not replaced
        self.assertIn("orient", data)
        return data["proportion"], trimesh.load(glb, force="mesh"), proc.stdout

    def test_a_side_on_photo_restores_a_squashed_length(self) -> None:
        before = squashed(0.7)
        prop, after, out = self.run_check(before, 0.0)
        self.assertTrue(prop["applied"], prop)
        self.assertTrue(prop["side_on"])
        self.assertAlmostEqual(prop["factor"], 1 / 0.7, delta=0.03 / 0.7)
        self.assertLess(abs(aspect_xy(after) / self.true_aspect - 1), 0.03)
        self.assertLess(abs(after.extents[0] / creature().extents[0] - 1), 0.03)          # the true length, within 3%
        # Height and width stay; the centre stays.
        np.testing.assert_allclose(after.extents[1:], before.extents[1:], rtol=1e-6)
        np.testing.assert_allclose(after.bounds.mean(0), before.bounds.mean(0), atol=1e-6)
        self.assertAlmostEqual(prop["mesh_aspect_before"], aspect_xy(before), places=3)
        self.assertAlmostEqual(prop["mesh_aspect_after"], aspect_xy(after), places=3)
        self.assertIn("as long", out)
        self.assertIn("PROPORTION ", out)

    def test_the_mirrored_side_also_restores(self) -> None:
        prop, after, _ = self.run_check(squashed(0.7), 180.0)
        self.assertTrue(prop["applied"], prop)
        self.assertLess(abs(after.extents[0] / creature().extents[0] - 1), 0.03)

    def test_a_front_on_photo_leaves_it_alone(self) -> None:
        before = squashed(0.7)
        prop, after, out = self.run_check(before, 90.0)
        self.assertFalse(prop["applied"], prop)
        self.assertFalse(prop["side_on"])
        self.assertEqual(prop["factor"], 1.0)
        np.testing.assert_allclose(after.vertices, before.vertices, atol=1e-6)
        self.assertIn("left alone", out)

    def test_a_three_quarter_photo_leaves_it_alone(self) -> None:
        before = squashed(0.7)
        prop, after, _ = self.run_check(before, 45.0)
        self.assertFalse(prop["applied"], prop)
        self.assertIn("three-quarter", prop["reason"])
        np.testing.assert_allclose(after.vertices, before.vertices, atol=1e-6)

    def test_a_small_difference_is_skipped(self) -> None:
        before = squashed(0.97)
        prop, after, _ = self.run_check(before, 0.0)
        self.assertTrue(prop["side_on"], prop)
        self.assertFalse(prop["applied"])
        self.assertIn("within 5%", prop["reason"])
        np.testing.assert_allclose(after.vertices, before.vertices, atol=1e-6)

    def test_the_clamp_holds(self) -> None:
        before = squashed(0.5)                               # wants 2x; the clamp stops at 1.5
        prop, after, _ = self.run_check(before, 0.0)
        self.assertTrue(prop["applied"], prop)
        self.assertTrue(prop["clamped"])
        self.assertGreater(prop["factor_raw"], 1.8)
        self.assertEqual(prop["factor"], 1.5)
        self.assertAlmostEqual(after.extents[0] / before.extents[0], 1.5, places=4)
        # A tighter clamp from the command line holds as well.
        prop, after2, _ = self.run_check(squashed(0.5), 0.0, "--max-factor", "1.2")
        self.assertAlmostEqual(after2.extents[0] / before.extents[0], 1.2, places=4)

    def test_an_unreadable_photo_fails_loudly(self) -> None:
        glb = self.dir / "mesh_hero.glb"
        squashed(0.7).export(glb)
        bad = self.dir / "photo.png"
        bad.write_text("not a picture")
        sj = self.dir / "shape.json"
        proc = subprocess.run([sys.executable, str(SCRIPT), str(glb), "--photo", str(bad), "--canonical", "right",
                               "--json", str(sj)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("failed", proc.stderr)
        prop = json.loads(sj.read_text())["proportion"]
        self.assertFalse(prop["applied"])
        self.assertIn("error", prop)


if __name__ == "__main__":
    unittest.main()
