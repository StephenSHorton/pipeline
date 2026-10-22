#!/usr/bin/env python3
"""mesh_edit's proportion ops (scripts/workbench/edit.py) on synthetic quadrupeds: thicken_legs, shorten_legs,
widen_head. No models load; the meshes are built here."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from workbench import edit  # noqa: E402

LEGS = [(x, z) for x in (-0.22, 0.22) for z in (-0.6, 0.6)]
LEG_R = 0.08


def tube(x: float, z: float, r: float = LEG_R, h: float = 0.6, rings: int = 13, sections: int = 16) -> trimesh.Trimesh:
    """A capped cylinder from the floor up with rings along its height (trimesh's has only its two ends)."""
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ys = np.linspace(0, h, rings)
    V = [[x + r * np.cos(t), y, z + r * np.sin(t)] for y in ys for t in a]
    F = []
    for i in range(rings - 1):
        for j in range(sections):
            p, q = i * sections + j, i * sections + (j + 1) % sections
            F += [[p, q + sections, q], [p, p + sections, q + sections]]
    V += [[x, 0, z], [x, h, z]]
    b, t = len(V) - 2, len(V) - 1
    for j in range(sections):
        F += [[b, j, (j + 1) % sections]]
        top0 = (rings - 1) * sections
        F += [[t, top0 + (j + 1) % sections, top0 + j]]
    return trimesh.Trimesh(np.array(V, float), np.array(F), process=False)


def quadruped() -> tuple[trimesh.Trimesh, dict]:
    """Body along z, head at +z, four ringed legs. Returns the mesh and each part's vertex index range."""
    body = trimesh.creation.icosphere(subdivisions=4)
    body.apply_scale([0.4, 0.35, 1.0])
    body.apply_translation([0, 0.8, 0])
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.25).apply_translation([0, 1.0, 1.15])
    parts = [("body", body), ("head", head)] + [(f"leg{i}", tube(x, z)) for i, (x, z) in enumerate(LEGS)]
    ranges, off = {}, 0
    for name, m in parts:
        ranges[name] = np.arange(off, off + len(m.vertices))
        off += len(m.vertices)
    return trimesh.util.concatenate([m for _, m in parts]), ranges


class ThickenTests(unittest.TestCase):
    def test_doubles_leg_width_keeps_body_and_feet(self) -> None:
        mesh, idx = quadruped()
        V = np.asarray(mesh.vertices)
        out, info = edit.thicken_legs(mesh, 2.0)
        self.assertEqual(info["legs_found"], 4)
        top = info["top_y"]
        self.assertGreater(top, 0.35)
        self.assertLess(top, 0.6)
        for i, (x, z) in enumerate(LEGS):
            leg = idx[f"leg{i}"]
            low = leg[V[leg, 1] < 0.5 * top]
            before = np.ptp(V[low][:, [0, 2]], axis=0)
            after = np.ptp(out[low][:, [0, 2]], axis=0)
            np.testing.assert_allclose(after / before, 2.0, rtol=0.1)
            np.testing.assert_allclose(out[leg, 1], V[leg, 1])  # height untouched: the feet stay on the floor
        self.assertAlmostEqual(float(out[:, 1].min()), float(V[:, 1].min()), places=9)
        body = np.concatenate([idx["body"], idx["head"]])
        upper = body[V[body, 1] >= top]
        np.testing.assert_allclose(out[upper], V[upper], atol=1e-9)
        # The legs' centres were found where they are.
        found = sorted(tuple(np.round(leg["centre_xz"], 1)) for leg in info["legs"])
        self.assertEqual(found, sorted(tuple(np.round(p, 1)) for p in LEGS))

    def test_touching_feet(self) -> None:
        """Two legs whose hooves touch still count as two: they part higher up."""
        mesh, _ = quadruped()
        V = np.asarray(mesh.vertices).copy()
        # Splay the two front feet together at the floor: hooves meet at x = 0.
        front = (V[:, 2] > 0.4) & (V[:, 1] < 0.6) & (np.abs(V[:, 0]) > 0.1)
        V[front, 0] -= np.sign(V[front, 0]) * 0.15 * np.clip(1 - V[front, 1] / 0.3, 0, 1)
        out, info = edit.thicken_legs(trimesh.Trimesh(V, mesh.faces, process=False), 1.5)
        self.assertEqual(info["legs_found"], 4)

    def test_no_legs(self) -> None:
        with self.assertRaises(edit.EditError):
            edit.thicken_legs(trimesh.creation.box([1, 1, 2]), 1.5)


class ShortenTests(unittest.TestCase):
    def test_squashes_legs_and_lowers_body(self) -> None:
        mesh, idx = quadruped()
        V = np.asarray(mesh.vertices)
        k = 0.6
        out, info = edit.shorten_legs(mesh, k)
        floor, top = float(V[:, 1].min()), info["top_y"]
        band = top - floor
        # The visible leg (floor to the belly) is factor as long.
        body = idx["body"]
        self.assertAlmostEqual((out[body, 1].min() - floor) / (V[body, 1].min() - floor), k, delta=0.03 * k)
        # Total height drops by (1 - factor) of the leg band.
        drop = (1 - k) * band
        self.assertAlmostEqual(np.ptp(V[:, 1]) - np.ptp(out[:, 1]), drop, delta=0.03 * drop)
        self.assertAlmostEqual(info["lowered_by"], drop, places=3)
        self.assertAlmostEqual(float(out[:, 1].min()), floor, places=9)
        # Above the blend the body is the same shape, moved down.
        above = np.nonzero(V[:, 1] > top + 0.1 * band + 1e-6)[0]
        d = out[above] - V[above]
        np.testing.assert_allclose(d[:, [0, 2]], 0, atol=1e-12)
        np.testing.assert_allclose(d[:, 1], d[0, 1], atol=1e-12)  # one rigid move
        self.assertAlmostEqual(float(d[0, 1]), -drop, places=3)
        # No crease: the height map is monotonic and its slope runs between factor and 1.
        ys = np.linspace(floor, V[:, 1].max(), 2000)
        probe = trimesh.Trimesh(np.vstack([V, np.c_[np.full_like(ys, 5.0), ys, np.zeros_like(ys)]]), mesh.faces,
                                process=False)  # a column of loose vertices beside the sheep: the height map itself
        y2 = edit.shorten_legs(probe, k, top=info["top"])[0][len(V):, 1]
        slope = np.diff(y2) / np.diff(ys)
        self.assertGreaterEqual(slope.min(), k - 1e-6)
        self.assertLessEqual(slope.max(), 1 + 1e-6)
        self.assertLess(np.abs(np.diff(slope)).max(), 0.02)  # continuous slope


class WidenTests(unittest.TestCase):
    def test_widens_head_keeps_rump(self) -> None:
        mesh, idx = quadruped()
        V = np.asarray(mesh.vertices)
        f = 1.5
        out, info = edit.widen_head(mesh, f)
        self.assertEqual(info["head"], "+z")
        self.assertEqual(info["lateral_axis"], "x")
        lo, hi = V[:, 2].min(), V[:, 2].max()
        full = lo + (info["start"] + info["ramp"]) * (hi - lo)
        self.assertLess(full, 0.95)  # the head sphere starts at z = 0.9
        head = idx["head"]
        tip = head[V[head, 2] > full]
        self.assertAlmostEqual(np.ptp(out[tip, 0]) / np.ptp(V[tip, 0]), f, delta=0.05 * f)
        np.testing.assert_allclose(out[:, [1, 2]], V[:, [1, 2]])  # depth and height untouched
        rump = V[:, 2] < 0
        np.testing.assert_allclose(out[rump], V[rump])
        # No crease: no edge stretches more than the full scale would, even in the ramp.
        E = mesh.edges_unique
        before = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        after = np.linalg.norm(out[E[:, 0]] - out[E[:, 1]], axis=1)
        self.assertLess((after / before).max(), f * 1.05)

    def test_head_given_and_minus_end(self) -> None:
        mesh, _ = quadruped()
        flipped = mesh.copy()
        flipped.vertices = np.asarray(mesh.vertices) * [1, 1, -1]
        _, info = edit.widen_head(flipped, 1.2)
        self.assertEqual(info["head"], "-z")
        _, info = edit.widen_head(flipped, 1.2, head="-z", start=0.7)
        self.assertEqual(info["start_from"], "given")
        with self.assertRaises(edit.EditError):
            edit.widen_head(flipped, 1.2, head="up")


class TextureTests(unittest.TestCase):
    def test_glb_keeps_uvs_and_material(self) -> None:
        mesh, _ = quadruped()
        uv = np.random.default_rng(0).random((len(mesh.vertices), 2))
        img = Image.new("RGB", (8, 8), (200, 30, 30))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=img))
        tmp = Path(tempfile.mkdtemp())
        mesh.export(tmp / "a.glb")
        src = trimesh.load(tmp / "a.glb", force="mesh", process=False)
        out, _ = edit.apply(src, "widen_head", factor=1.3)
        out, _ = edit.apply(out, "thicken_legs", factor=1.4)
        out.export(tmp / "b.glb")
        back = trimesh.load(tmp / "b.glb", force="mesh", process=False)
        np.testing.assert_allclose(back.visual.uv, src.visual.uv, atol=1e-6)
        self.assertIsNotNone(back.visual.material.baseColorTexture)
        self.assertEqual(len(back.faces), len(src.faces))


if __name__ == "__main__":
    unittest.main()
