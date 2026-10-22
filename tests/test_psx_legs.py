"""psx_lowpoly --legs clean: lumpy, slanted legs come out as straight tapered prisms on the floor, inside the triangle
budget; a shape with no legs keeps its own with a note; the default (keep) is the cut it always was."""
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
sys.path.insert(0, str(ROOT / "scripts"))

import psx_lowpoly as P  # noqa: E402

LEG_R = 0.08
FEET = [(x, z) for x in (-0.22, 0.22) for z in (-0.6, 0.6)]
SLANT = 0.12  # the top of each leg sits this far further forward (+z) than its foot


def lumpy_leg(x: float, z: float, seed: int, h: float = 0.62, rings: int = 25, sections: int = 16) -> trimesh.Trimesh:
    """A capped tube from the floor up, slanted forward, its radius bulging and its axis wobbling with height."""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    V = []
    for y in np.linspace(0, h, rings):
        t = y / h
        r = LEG_R * (1 + 0.3 * np.sin(9 * t + seed) + 0.1 * rng.uniform(-1, 1))
        cx = x + 0.02 * np.sin(7 * t + seed)
        cz = z + SLANT * t + 0.02 * np.cos(5 * t + seed)
        V += [[cx + r * (1 + 0.15 * rng.uniform(-1, 1)) * np.cos(q), y, cz + r * np.sin(q)] for q in a]
    F = []
    for i in range(rings - 1):
        for j in range(sections):
            p, q = i * sections + j, i * sections + (j + 1) % sections
            F += [[p, q + sections, q], [p, p + sections, q + sections]]
    V += [[x, 0, z], [x, h, z + SLANT]]
    b, top = len(V) - 2, len(V) - 1
    t0 = (rings - 1) * sections
    for j in range(sections):
        F += [[b, j, (j + 1) % sections], [top, t0 + (j + 1) % sections, t0 + j]]
    return trimesh.Trimesh(np.array(V, float), np.array(F), process=False)


def quadruped() -> trimesh.Trimesh:
    body = trimesh.creation.icosphere(subdivisions=4)
    body.apply_scale([0.4, 0.35, 1.0])
    body.apply_translation([0, 0.8, 0])
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.25).apply_translation([0, 1.0, 1.15])
    legs = [lumpy_leg(x, z, i) for i, (x, z) in enumerate(FEET)]
    return trimesh.util.concatenate([body, head, *legs])


def bale() -> trimesh.Trimesh:
    """A hay bale: a fat cylinder lying on its side."""
    m = trimesh.creation.cylinder(radius=0.5, height=1.2, sections=48)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m = m.subdivide().subdivide()
    m.apply_translation([0, 0.5, 0])
    return m


def textured(mesh: trimesh.Trimesh, path: Path) -> Path:
    """The mesh with planar UVs and a texture whose colour runs with height (dark legs, light body), as a GLB."""
    V = np.asarray(mesh.vertices)
    lo, hi = V.min(0), V.max(0)
    uv = np.stack([(V[:, 2] - lo[2]) / (hi[2] - lo[2]), (V[:, 1] - lo[1]) / (hi[1] - lo[1])], axis=1)
    ramp = np.linspace(40, 220, 64).astype(np.uint8)
    img = np.repeat(ramp[::-1, None], 64, axis=1)
    image = Image.fromarray(np.stack([img] * 3, axis=2), "RGB")
    out = trimesh.Trimesh(V, mesh.faces, process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uv, image=image))
    out.export(path, file_type="glb")
    return path


def run(src: Path, out: Path, *extra: str) -> dict:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "psx_lowpoly.py"), str(src), "--out", str(out),
                           "--name", "m", "--height", "30", "--faces", "600", "--atlas", "64", "--facing-check", "off",
                           "--glb", str(out / "m.glb"), *extra], capture_output=True, text=True, timeout=300,
                          env={"PSX_FACING_CHECK": "off", "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    return {"report": json.loads((out / "psx.json").read_text()), "stderr": proc.stderr, "stdout": proc.stdout}


class CleanLegsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.high = P.place(quadruped(), 30.0, 0.0)
        cls.low, cls.info, cls.regions = P.clean_legs(cls.high, 600, 1.0, sides=6, scale=1.5, taper=0.8)

    def columns(self) -> list[np.ndarray]:
        F = np.asarray(self.low.faces)
        V = np.asarray(self.low.vertices)
        return [V[np.unique(F[r])] for r, _ in self.regions]

    def test_four_straight_tapered_columns_on_the_floor(self) -> None:
        self.assertEqual(self.info["legs_found"], 4)
        cols = self.columns()
        self.assertEqual(len(cols), 4)
        orig = quadruped()
        scale = 30.0 / np.ptp(orig.vertices[:, 1])
        centre = orig.bounds.mean(0)[[0, 2]]  # place() centres x and z
        for C in cols:
            self.assertEqual(len(C), 12)  # two rings of six, nothing else
            ys = np.unique(np.round(C[:, 1], 9))
            self.assertEqual(len(ys), 2)  # horizontal rings
            bottom, top = C[C[:, 1] < ys.mean()], C[C[:, 1] > ys.mean()]
            self.assertAlmostEqual(float(bottom[:, 1].max()), 0.0, places=9)  # on the floor
            rb = np.linalg.norm(bottom[:, [0, 2]] - bottom[:, [0, 2]].mean(0), axis=1)
            rt = np.linalg.norm(top[:, [0, 2]] - top[:, [0, 2]].mean(0), axis=1)
            # Every vertex on its ring's circle: the column is exactly a straight tapered prism.
            self.assertLess(float(np.ptp(rb)), 1e-9)
            self.assertLess(float(np.ptp(rt)), 1e-9)
            self.assertAlmostEqual(float(rb.mean() / rt.mean()), 0.8, places=9)
            # About 1.5 measured radii (the lumpy legs average LEG_R), standing where the leg stands.
            self.assertGreater(rt.mean(), 1.1 * LEG_R * scale)
            self.assertLess(rt.mean(), 2.2 * LEG_R * scale)
            foot = bottom[:, [0, 2]].mean(0)
            near = min(np.linalg.norm(foot - (np.array(f) - centre) * scale) for f in FEET)
            self.assertLess(near, 1.5 * LEG_R * scale)
            # Up into the body past the band top.
            self.assertGreater(float(top[:, 1].min()), self.info["top_y"])

    def test_lumpy_leg_faces_are_gone(self) -> None:
        F = np.asarray(self.low.faces)
        V = np.asarray(self.low.vertices)
        cols = np.concatenate([r for r, _ in self.regions])
        rest = np.setdiff1d(np.arange(len(F)), cols)
        low_faces = rest[(V[F[rest], 1] < 0.8 * self.info["top_y"]).all(1)]
        self.assertEqual(len(low_faces), 0)

    def test_within_budget(self) -> None:
        self.assertLessEqual(len(self.low.faces), 600)
        self.assertEqual(self.info["column_triangles"], 4 * P.column_faces(6))
        self.assertGreater(len(self.low.faces), 500)  # the freed leg triangles went back to the body

    def test_columns_bake_from_their_legs(self) -> None:
        for lows, highs in self.regions:
            hc = np.asarray(self.high.triangles_center)[highs]
            self.assertLess(float(hc[:, 1].max()), self.info["top_y"])

    def test_bale_has_no_legs(self) -> None:
        from workbench import edit
        with self.assertRaises(edit.EditError):
            P.clean_legs(P.place(bale(), 30.0, 0.0), 600, 1.0)


class MainTests(unittest.TestCase):
    def test_bale_falls_back_to_keep_with_a_note(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = textured(bale(), d / "bale.glb")
            got = run(src, d / "clean", "--legs", "clean")
            legs = got["report"]["legs"]
            self.assertEqual(legs["mode"], "keep")
            self.assertEqual(legs["asked"], "clean")
            self.assertTrue(legs["why"])
            self.assertEqual(got["report"]["degraded"][0]["code"], "legs_clean")
            self.assertIn("kept the decimated legs", got["stderr"])
            plain = run(src, d / "keep")
            self.assertEqual((d / "keep" / "m.glb").read_bytes(), (d / "clean" / "m.glb").read_bytes())
            self.assertNotIn("legs", plain["report"])

    def test_default_is_keep_and_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = textured(quadruped(), d / "q.glb")
            run(src, d / "default")
            run(src, d / "keep", "--legs", "keep")
            clean = run(src, d / "clean", "--legs", "clean", "--leg-scale", "1.5")
            for f in ("m.glb", "m.obj", "m.png"):
                self.assertEqual((d / "default" / f).read_bytes(), (d / "keep" / f).read_bytes(), f)
            self.assertEqual((d / "default" / "psx.json").read_text(), (d / "keep" / "psx.json").read_text())
            self.assertNotEqual((d / "default" / "m.glb").read_bytes(), (d / "clean" / "m.glb").read_bytes())
            self.assertEqual(clean["report"]["legs"]["legs_found"], 4)
            self.assertLessEqual(clean["report"]["triangles"], 600)
            # The columns took the dark leg texels, not the light body: the texture darkens toward the floor.
            img = np.asarray(Image.open(d / "clean" / "m.png").convert("L"), float)
            self.assertLess(img.min(), 90)


if __name__ == "__main__":
    unittest.main()
