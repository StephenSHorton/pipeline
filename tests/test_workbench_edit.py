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


def biped(gap: float) -> tuple[trimesh.Trimesh, list[np.ndarray]]:
    """A round body on two legs along z, gap apart (surface to surface), head at +x. Returns the legs' indices."""
    body = trimesh.creation.icosphere(subdivisions=4)
    body.apply_scale([0.5, 0.4, 0.35])
    body.apply_translation([0, 0.9, 0])
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.15).apply_translation([0.5, 1.35, 0])
    zc = LEG_R + gap / 2
    legs = [tube(0, -zc), tube(0, zc)]
    off = len(body.vertices) + len(head.vertices)
    idx = [np.arange(off + i * len(legs[0].vertices), off + (i + 1) * len(legs[0].vertices)) for i in range(2)]
    return trimesh.util.concatenate([body, head] + legs), idx


def with_tail(length: float, y: float = 0.85) -> trimesh.Trimesh:
    """The quadruped with a level tail of this length at the rump (-z), at the height of the body's middle."""
    mesh, _ = quadruped()
    tail = trimesh.creation.cylinder(radius=0.05, height=length)  # along z already
    tail.apply_translation([0, y, -0.98 - length / 2])
    return trimesh.util.concatenate([mesh, tail])


def headless() -> dict[str, trimesh.Trimesh]:
    """Shapes with no head: an upright cylinder, a ball, a box on four wheels (wheel axes along z)."""
    cyl = trimesh.creation.cylinder(radius=0.6, height=1.0, sections=48).apply_translation([0, 0, 0.5])
    cyl.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # axis up (y)
    ball = trimesh.creation.icosphere(subdivisions=4, radius=0.5).apply_translation([0, 0.5, 0])
    parts = [trimesh.creation.box([2.0, 0.6, 1.0]).apply_translation([0, 0.7, 0])]
    for x in (-0.7, 0.7):
        for z in (-0.6, 0.6):
            w = trimesh.creation.cylinder(radius=0.35, height=0.2, sections=32).apply_translation([x, 0.35, z])
            parts.append(w)
    return {"upright cylinder": cyl, "ball": ball, "box on wheels": trimesh.util.concatenate(parts)}


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

    def test_overlapping_pair_is_refused_not_splayed(self) -> None:
        # Front pair overlapping (one round-ish blob), rear pair apart: the blob is two legs, not one fat one.
        r = 0.08
        for ov in (0.02, 0.05, 0.08):
            body = trimesh.creation.icosphere(subdivisions=4)
            body.apply_scale([1.0, 0.35, 0.4])
            body.apply_translation([0, 0.8, 0])
            head = trimesh.creation.icosphere(subdivisions=3, radius=0.22).apply_translation([1.15, 1.05, 0])
            legs = [tube(0.6, -(r - ov / 2), r=r), tube(0.6, r - ov / 2, r=r), tube(-0.6, -0.22, r=r), tube(-0.6, 0.22, r=r)]
            with self.subTest(overlap=ov), self.assertRaisesRegex(edit.EditError, "touch along their length"):
                edit.thicken_legs(trimesh.util.concatenate([body, head, *legs]), 1.5)

    def test_close_legs_each_about_its_own_centre(self) -> None:
        """Legs nearly touching (or touching) make one footprint blob; it is split at the waist, so each leg
        thickens about its own centre instead of the pair splaying apart about their middle."""
        for gap in (0.0, 0.005, 0.02, 0.05):
            mesh, legs = biped(gap)
            V = np.asarray(mesh.vertices)
            out, info = edit.thicken_legs(mesh, 2.0)
            self.assertEqual(info["legs_found"], 2, gap)
            zc = LEG_R + gap / 2
            for leg in legs:
                low = leg[V[leg, 1] < 0.15]
                before, after = np.median(V[low, 2]), np.median(out[low, 2])
                self.assertAlmostEqual(abs(before), zc, delta=1e-6)
                self.assertLess(abs(after - before) / zc, 0.06, gap)  # the centre stays put
                self.assertAlmostEqual(np.ptp(out[low, 2]) / np.ptp(V[low, 2]), 2.0, delta=0.1)

    def test_one_part_on_the_floor_refused(self) -> None:
        """A ball on the floor has one 'leg' (its bottom): refused, nothing moves."""
        ball = trimesh.creation.icosphere(subdivisions=4, radius=0.5).apply_translation([0, 0.5, 0])
        V = np.asarray(ball.vertices).copy()
        with self.assertRaisesRegex(edit.EditError, "found 1 leg-like part"):
            edit.thicken_legs(ball, 1.5)
        with self.assertRaisesRegex(edit.EditError, "found 1 leg-like part; give top"):
            edit.shorten_legs(ball, 0.7)
        np.testing.assert_array_equal(ball.vertices, V)
        out, info = edit.shorten_legs(ball, 0.7, top=0.2)  # a given top is the caller's call
        self.assertEqual(info["top_from"], "given")


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

    def test_level_tail_is_not_the_head(self) -> None:
        """A tail sticks out past the hind legs further than the head past the front ones, but it is thin and sits
        at the body's mid-height; the head rises above it."""
        for length in (0.3, 0.5, 0.7):
            for y in (0.75, 0.85):
                _, info = edit.widen_head(with_tail(length, y), 1.3)
                self.assertEqual(info["head"], "+z", (length, y))
                self.assertGreater(info["start"], 0.6)
            flipped = with_tail(length)
            flipped.vertices = np.asarray(flipped.vertices) * [1, 1, -1]
            self.assertEqual(edit.widen_head(flipped, 1.3)[1]["head"], "-z")

    def test_headless_shapes_refused(self) -> None:
        for name, mesh in headless().items():
            V = np.asarray(mesh.vertices).copy()
            with self.assertRaises(edit.EditError, msg=name):
                edit.widen_head(mesh, 1.3)
            np.testing.assert_array_equal(mesh.vertices, V)
            # Given both, the caller decides: it runs.
            head = "+x" if np.ptp(V[:, 0]) >= np.ptp(V[:, 2]) else "+z"
            out, info = edit.widen_head(mesh, 1.3, head=head, start=0.7)
            self.assertEqual((info["head_from"], info["start_from"]), ("given", "given"))
            self.assertGreater(info["head_width_after"], info["head_width_before"])
        # A barrel has no neck: a given head end alone still needs a start.
        with self.assertRaisesRegex(edit.EditError, "no neck"):
            edit.widen_head(headless()["box on wheels"], 1.3, head="+x")


def cone_down(x: float, z: float, tip: float, top: float = 0.55, r: float = 0.04, sections: int = 12) -> trimesh.Trimesh:
    """A spike of wool: a cone hanging point down from inside the body to tip."""
    c = trimesh.creation.cone(radius=r, height=top - tip, sections=sections)  # apex up the +z axis
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # apex down -y
    c.apply_translation([x, top, z])
    return c


def keel(z0: float, z1: float, bottom: float, top: float = 0.55, half: float = 0.03, n: int = 12) -> trimesh.Trimesh:
    """A keel of wool: a thin wedge in the y-z plane, half wide at the top tapering to an edge at the bottom, with
    rows along its height and length so the lift has vertices to move."""
    zs = np.linspace(z0, z1, n)
    ys = np.linspace(bottom, top, n)
    V, idx = [], {}
    for s in (-1, 1):
        for i, y in enumerate(ys):
            for j, z in enumerate(zs):
                w = half * (y - bottom) / (top - bottom)
                idx[s, i, j] = len(V)
                V.append([s * w, y, z])
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            for s in (-1, 1):
                a, b, c, d = idx[s, i, j], idx[s, i, j + 1], idx[s, i + 1, j + 1], idx[s, i + 1, j]
                F += [[a, b, c], [a, c, d]] if s > 0 else [[a, c, b], [a, d, c]]
    for i in range(n - 1):  # the two sheets meet along the bottom edge; close the ends
        for j, sgn in ((0, 1), (n - 1, -1)):
            a, b, c, d = idx[-1, i, j], idx[1, i, j], idx[1, i + 1, j], idx[-1, i + 1, j]
            F += [[a, b, c], [a, c, d]] if sgn > 0 else [[a, c, b], [a, d, c]]
    return trimesh.Trimesh(np.array(V, float), np.array(F), process=False)


def hairy_quadruped() -> tuple[trimesh.Trimesh, dict]:
    """The quadruped (body bottom at y 0.45) with a keel hanging to 0.3 between the front legs and spikes hanging to
    0.35 along the underside between the legs."""
    mesh, idx = quadruped()
    extra = [("keel", keel(0.25, 0.6, 0.30))]
    extra += [(f"spike{i}", cone_down(x, z, 0.35)) for i, (x, z) in enumerate([(0.0, -0.3), (0.12, 0.0), (-0.1, -0.15),
                                                                                (0.05, 0.15)])]
    off = len(mesh.vertices)
    for name, m in extra:
        idx[name] = np.arange(off, off + len(m.vertices))
        off += len(m.vertices)
    return trimesh.util.concatenate([mesh] + [m for _, m in extra]), idx


class TrimBellyTests(unittest.TestCase):
    def test_lifts_keel_and_spikes_keeps_legs(self) -> None:
        mesh, idx = hairy_quadruped()
        V = np.asarray(mesh.vertices)
        out, info = edit.trim_belly(mesh)
        line = info["line_y"] - 1e-4  # reported to 4 places
        self.assertEqual(info["legs_found"], 4)
        self.assertEqual(info["line_from"] in ("belly underside", "leg band top"), True)
        self.assertGreater(line, 0.42)  # about the body's bottom (0.45), well above the keel and spike tips
        self.assertLess(line, 0.6)
        # Nothing of the body lies below the line outside the legs (clear of every leg by LEG_FREE radii).
        R = max(leg["radius"] for leg in info["legs"])
        d = np.min([np.hypot(out[:, 0] - x, out[:, 2] - z) for x, z in LEGS], axis=0)
        legs = np.concatenate([idx[f"leg{i}"] for i in range(4)])
        rest = np.setdiff1d(np.arange(len(V)), legs)
        clear = rest[d[rest] > edit.LEG_FREE * R]
        self.assertGreaterEqual(float(out[clear, 1].min()), line - 1e-9)
        for part in ["keel"] + [f"spike{i}" for i in range(4)]:
            self.assertGreater(float(out[idx[part], 1].min()), line - 1e-9, part)
            self.assertLess(float(V[idx[part], 1].min()), line - 0.05, part)  # they did hang below it
        self.assertAlmostEqual(info["max_lift"], line - 0.30, delta=0.01)  # the keel's tip came up to the line
        # Legs, head and everything above the knee are as they were.
        np.testing.assert_array_equal(out[legs], V[legs])
        np.testing.assert_array_equal(out[idx["head"]], V[idx["head"]])
        high = V[:, 1] > line + info["blend"] * np.ptp(V[:, 1]) + 1e-9
        np.testing.assert_array_equal(out[high], V[high])
        np.testing.assert_array_equal(out[:, [0, 2]], V[:, [0, 2]])  # only heights move
        self.assertTrue((out[:, 1] >= V[:, 1] - 1e-12).all())  # a trim only lifts
        # No face turns over: every face that keeps an area faces the same side as before.
        T0, T1 = V[mesh.faces], out[mesh.faces]
        n0 = np.cross(T0[:, 1] - T0[:, 0], T0[:, 2] - T0[:, 0])
        n1 = np.cross(T1[:, 1] - T1[:, 0], T1[:, 2] - T1[:, 0])
        keep = np.linalg.norm(n1, axis=1) > 1e-12
        self.assertTrue(((n0 * n1).sum(1)[keep] > -1e-12).all())

    def test_box_on_wheels_refused_unless_line_given(self) -> None:
        car = headless()["box on wheels"]
        with self.assertRaisesRegex(edit.EditError, "give line"):
            edit.trim_belly(car)
        edit.trim_belly(car, line=0.4)  # the caller's own line still works

    def test_given_line_and_no_legs(self) -> None:
        box = trimesh.creation.box([1, 1, 2]).apply_translation([0, 0.5, 0])
        with self.assertRaisesRegex(edit.EditError, "give line"):
            edit.trim_belly(box)
        sub = box.subdivide().subdivide()
        out, info = edit.trim_belly(sub, line=0.3)  # a given line is the caller's call: lifted everywhere
        self.assertEqual((info["line_from"], info["legs_found"]), ("given", 0))
        self.assertGreaterEqual(float(out[:, 1].min()), 0.3 - 1e-9)
        mesh, idx = hairy_quadruped()
        out, info = edit.trim_belly(mesh, line=0.45, blend=0.02)
        self.assertEqual(info["line"], 0.45)
        self.assertGreater(float(out[idx["keel"], 1].min()), info["line_y"] - 1e-9)

    def test_feet_stay_on_the_floor(self) -> None:
        """A toe splaying past its leg's radius at the floor is foot, not belly: it stays."""
        mesh, idx = quadruped()
        toe = trimesh.creation.box([0.06, 0.04, 0.2]).apply_translation([0.22, 0.02, 0.72])
        both = trimesh.util.concatenate([mesh, toe])
        V = np.asarray(both.vertices)
        out, _ = edit.trim_belly(both)
        t = np.arange(len(mesh.vertices), len(V))
        np.testing.assert_array_equal(out[t], V[t])


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
