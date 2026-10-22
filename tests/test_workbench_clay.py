#!/usr/bin/env python3
"""Clay blobs mesh to the right shape and to a GLB that retexture and psx accept."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import retexture  # noqa: E402
from workbench import clay, mesher, unwrap_cli  # noqa: E402


def blob(**kw) -> clay.Blob:
    kw.setdefault("name", "b")
    return clay.Blob.from_dict(kw)


def end_circularity(mesh: trimesh.Trimesh, axis: int) -> float:
    """4*pi*A/P^2 of the silhouette seen down `axis` (1 for a circle, pi/4 for a square).

    The silhouette is the convex hull of the projected vertices, exact for the convex shapes tested here.
    eyes.py owns the official metric; this is only enough to tell a round end from a square one.
    """
    keep = [a for a in range(3) if a != axis]
    hull = ConvexHull(np.asarray(mesh.vertices)[:, keep])
    return float(4 * np.pi * hull.volume / hull.area ** 2)


class ClayShapeTests(unittest.TestCase):
    def test_sphere_volume_within_three_percent(self) -> None:
        mesh = mesher.mesh_clay([blob(form={"type": "Sphere", "radius": 0.5})], grid=64)
        exact = 4 / 3 * np.pi * 0.5 ** 3
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(mesh.volume, 0)
        self.assertLess(abs(mesh.volume - exact) / exact, 0.03)

    def test_mirrored_blob_is_symmetric(self) -> None:
        blobs = [
            blob(name="body", form={"type": "Ellipsoid", "radii": [0.3, 0.25, 0.5]}),
            blob(name="ear", form={"type": "Capsule", "radius": 0.06, "height": 0.3}, translation=[0.25, 0.3, 0.2],
                 rotation=[0, 0, -35], mirror=True, blend=0.04),
        ]
        rng = np.random.default_rng(0)
        p = rng.uniform(-0.6, 0.6, (2000, 3))
        np.testing.assert_allclose(clay.sdf(blobs, p), clay.sdf(blobs, p * [-1, 1, 1]), atol=1e-9)
        mesh = mesher.mesh_clay(blobs, grid=64)
        flipped = np.asarray(mesh.vertices) * [-1, 1, 1]
        gap, _ = cKDTree(mesh.vertices).query(flipped)
        # The lattice is symmetric about X = 0 when a blob mirrors, so every vertex has an exact twin.
        self.assertLess(float(gap.max()), 1e-6)
        # Both ears are there: material on either side of X = 0 at ear height.
        ear = np.asarray(mesh.vertices)[:, 1] > 0.35
        self.assertTrue((mesh.vertices[ear, 0] > 0.1).any() and (mesh.vertices[ear, 0] < -0.1).any())

    def test_cylinder_end_is_round_and_box_is_not(self) -> None:
        cyl = mesher.mesh_clay([blob(form={"type": "Cylinder", "radius": 0.5, "height": 0.8}, rotation=[90, 0, 0])],
                               grid=64)
        box = mesher.mesh_clay([blob(form={"type": "Box", "size": [1.0, 1.0, 0.8]})], grid=64)
        c, b = end_circularity(cyl, axis=2), end_circularity(box, axis=2)
        self.assertGreater(c, 0.95)
        self.assertLess(b, 0.85)

    def test_carve_reduces_volume(self) -> None:
        solid = [blob(name="block", form={"type": "Box", "size": [1, 1, 1]})]
        carved = solid + [blob(name="bite", form={"type": "Sphere", "radius": 0.4}, op="carve",
                               translation=[0.5, 0.5, 0.5])]
        v0 = mesher.mesh_clay(solid, grid=48).volume
        v1 = mesher.mesh_clay(carved, grid=48).volume
        bite = (4 / 3 * np.pi * 0.4 ** 3) / 8  # one octant of the sphere sits inside the block
        self.assertLess(v1, v0)
        self.assertAlmostEqual(v0 - v1, bite, delta=bite * 0.15)

    def test_blend_fills_the_joint(self) -> None:
        pair = [dict(form={"type": "Sphere", "radius": 0.3}, translation=[-0.25, 0, 0]),
                dict(form={"type": "Sphere", "radius": 0.3}, translation=[0.25, 0, 0])]
        hard = [blob(name="a", **pair[0]), blob(name="b", **pair[1])]
        soft = [blob(name="a", **pair[0]), blob(name="b", blend=0.15, **pair[1])]
        neck = np.array([[0.0, 0.27, 0.0]])  # just above the gap between them
        self.assertGreater(clay.sdf(hard, neck)[0], clay.sdf(soft, neck)[0])


class ClayCharacterTests(unittest.TestCase):
    """Rounding, bulge, noise and a carved ground contact: what keeps a clay bale from reading as a CAD cylinder."""

    def test_rounding_takes_off_the_rims(self) -> None:
        r, h, rr = 0.5, 1.0, 0.2
        sharp = mesher.mesh_clay([blob(form={"type": "Cylinder", "radius": r, "height": h})], grid=64)
        soft = mesher.mesh_clay([blob(form={"type": "Cylinder", "radius": r, "height": h, "rounding": rr})], grid=64)
        # Steiner: the rounded cylinder is the inner cylinder (radius a, length l) grown by rr.
        a, ln = r - rr, h - 2 * rr
        exact = (np.pi * a * a * ln + (2 * np.pi * a * a + 2 * np.pi * a * ln) * rr
                 + (np.pi * ln + np.pi ** 2 * a) * rr ** 2 + 4 / 3 * np.pi * rr ** 3)
        lost = np.pi * r * r * h - exact                     # about 0.049 m3, 6% of the sharp one
        self.assertTrue(soft.is_watertight)
        # Same grid for both, so the meshing error mostly cancels in the difference.
        self.assertAlmostEqual(sharp.volume - soft.volume, lost, delta=lost * 0.25)
        # The SDF itself is exact: zero on the rim's torus, rr in from the cylinder's corner.
        corner = np.array([[r - rr + rr / np.sqrt(2), h / 2 - rr + rr / np.sqrt(2), 0.0]])
        self.assertAlmostEqual(float(clay.sdf([blob(form={"type": "Cylinder", "radius": r, "height": h,
                                                          "rounding": rr})], corner)[0]), 0.0, places=9)

    def test_bulge_widens_the_middle_and_leaves_the_caps(self) -> None:
        r, h, bulge = 0.5, 1.0, 0.2
        mesh = mesher.mesh_clay([blob(form={"type": "Cylinder", "radius": r, "height": h, "bulge": bulge})], grid=80)
        v = np.asarray(mesh.vertices)
        radial = np.hypot(v[:, 0], v[:, 2])
        middle = radial[np.abs(v[:, 1]) < 0.03].max()
        near_cap = radial[np.abs(np.abs(v[:, 1]) - (h / 2 - 0.02)) < 0.01].max()
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(middle, r * (1 + bulge), delta=0.01)
        # 0.02 below the cap the parabola has only grown by about bulge * r * 0.08.
        self.assertAlmostEqual(near_cap, r * (1 + bulge * 0.0784), delta=0.01)
        self.assertAlmostEqual(float(v[:, 1].max()), h / 2, delta=0.01)   # the caps stay where they were

    def test_bulge_keeps_the_rounding_it_was_given(self) -> None:
        # The flat cap face is radius - rounding whatever the bulge: rounding is not squashed by the barrel slope.
        r, h = 0.6, 0.9
        rho = np.linspace(0.0, r, 6001)
        for bulge, rr in ((0.6, 0.1), (1.0, 0.1), (0.5, 0.3)):
            b = blob(form={"type": "Cylinder", "radius": r, "height": h, "rounding": rr, "bulge": bulge})
            on_cap = np.stack([rho, np.full_like(rho, h / 2), np.zeros_like(rho)], axis=1)
            face = rho[clay.sdf([b], on_cap) <= 1e-9].max()
            self.assertAlmostEqual(face, r - rr, delta=1e-3, msg=f"bulge {bulge}, rounding {rr}")
            # The middle still swells to r * (1 + bulge), where the slope (and the scale) is 1.
            mid = clay.sdf([b], np.array([[r * (1 + bulge), 0.0, 0.0]]))[0]
            self.assertAlmostEqual(float(mid), 0.0, places=9)
        mesh = mesher.mesh_clay([blob(form={"type": "Cylinder", "radius": r, "height": h, "rounding": 0.3,
                                            "bulge": 0.5})], grid=64)
        self.assertTrue(mesh.is_watertight)
        v = np.asarray(mesh.vertices)
        top = v[v[:, 1] > h / 2 - 0.005]
        self.assertGreater(np.hypot(top[:, 0], top[:, 2]).max(), 0.25)   # a flat lid, not a nub

    NOISY = {"type": "Cylinder", "radius": 0.5, "height": 0.8, "rounding": 0.08}

    def noisy(self, amp: float, seed: int = 0) -> list:
        return [blob(name="bale", form=self.NOISY, noise={"amp": amp, "scale": 0.15, "seed": seed, "octaves": 2})]

    def test_noise_is_seeded(self) -> None:
        a = mesher.mesh_clay(self.noisy(0.015, seed=1), grid=48)
        b = mesher.mesh_clay(self.noisy(0.015, seed=1), grid=48)
        c = mesher.mesh_clay(self.noisy(0.015, seed=2), grid=48)
        self.assertTrue(np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces))
        p = np.random.default_rng(0).uniform(-0.6, 0.6, (4000, 3))
        self.assertFalse(np.allclose(clay.sdf(self.noisy(0.015, 1), p), clay.sdf(self.noisy(0.015, 2), p)))
        self.assertTrue(a.is_watertight and c.is_watertight)

    def test_roughness_grows_with_amp(self) -> None:
        smooth = [blob(name="bale", form=self.NOISY)]
        rough = []
        for amp in (0.0, 0.01, 0.02):
            mesh = mesher.mesh_clay(self.noisy(amp) if amp else smooth, grid=64)
            self.assertTrue(mesh.is_watertight)
            # How far the noisy surface wanders from the smooth one.
            rough.append(float(np.std(clay.sdf(smooth, mesh.vertices))))
        self.assertLess(rough[0], 0.002)
        self.assertLess(rough[0], rough[1])
        self.assertLess(rough[1], rough[2])
        self.assertLess(rough[2], 0.02)

    def test_noisy_flat_caps_stay_watertight(self) -> None:
        # A noisy cap lying along a grid plane makes cell faces whose corners alternate in and out; surface nets
        # gave those edges four triangles until mesher joined them. Several seeds and grids, all watertight.
        for seed in range(4):
            for grid in (48, 64):
                mesh = mesher.mesh_clay(self.noisy(0.012, seed=seed), grid=grid)
                self.assertTrue(mesh.is_watertight, f"seed {seed} grid {grid}")

    def test_checkerboard_face_is_joined(self) -> None:
        d = np.ones((3, 3, 3))
        d[1, 1, 1] = d[1, 2, 2] = -1.0       # diagonal corners of the x = 1 face: a checkerboard
        d[1, 2, 1] = 0.2                     # the outside corner nearer the surface
        out = mesher.join_checkerboards(d, 0.1)
        self.assertLess(out[1, 2, 1], 0)
        self.assertEqual(out[1, 1, 2], 1.0)
        self.assertEqual(int((out < 0).sum()), 3)

    def test_noise_amp_is_clamped_to_a_usable_distance(self) -> None:
        b = blob(form={"type": "Sphere", "radius": 0.5}, noise={"amp": 1.0, "scale": 0.1, "octaves": 4})
        self.assertAlmostEqual(b.noise.amp, clay.Noise.max_amp(0.1, 4), places=6)
        self.assertLess(b.noise.amp, 0.1 / 10)
        # The field's slope stays near 1, so the distance is still a distance.
        p = np.random.default_rng(1).uniform(-0.7, 0.7, (3000, 3))
        g = mesher.gradient(lambda q: b.sdf(q), p, 1e-4)
        slope = np.linalg.norm(g, axis=1)
        self.assertGreater(slope.min(), 1 - clay.NOISE_MAX_SLOPE - 0.05)
        self.assertLess(slope.max(), 1 + clay.NOISE_MAX_SLOPE + 0.05)
        self.assertIsNone(blob(form={"type": "Sphere", "radius": 0.5}, noise={"amp": 0, "scale": 0.1}).noise)
        with self.assertRaisesRegex(ValueError, "octaves"):
            blob(form={"type": "Sphere", "radius": 0.5}, noise={"amp": 0.01, "scale": 0.1, "octaves": 5})
        with self.assertRaisesRegex(ValueError, "scale"):
            blob(form={"type": "Sphere", "radius": 0.5}, noise={"amp": 0.01})

    def test_noise_travels_with_the_blob(self) -> None:
        here = blob(form={"type": "Sphere", "radius": 0.3}, noise={"amp": 0.01, "scale": 0.1, "seed": 3})
        there = blob(form={"type": "Sphere", "radius": 0.3}, noise={"amp": 0.01, "scale": 0.1, "seed": 3},
                     translation=[1.0, 0.5, 0.0], rotation=[0, 40, 0])
        p = np.random.default_rng(2).uniform(-0.4, 0.4, (500, 3))
        moved = there._rot().apply(p) + there.translation
        np.testing.assert_allclose(here.sdf(p), there.sdf(moved), atol=1e-9)

    def test_carved_floor_gives_a_flat_contact_patch(self) -> None:
        bale = blob(name="bale", form={"type": "Cylinder", "radius": 0.5, "height": 0.8}, rotation=[90, 0, 0],
                    noise={"amp": 0.01, "scale": 0.15})
        floor = blob(name="floor", form={"type": "Box", "size": [3, 1, 3]}, op="carve", translation=[0, -0.95, 0],
                     blend=0.03)
        mesh = mesher.mesh_clay([bale, floor], grid=64)
        v = np.asarray(mesh.vertices)
        self.assertTrue(mesh.is_watertight)
        # The cut sits at y = -0.45; the noisy bale would otherwise reach below -0.5.
        self.assertAlmostEqual(float(v[:, 1].min()), -0.45, delta=0.01)
        flat = v[np.abs(v[:, 1] + 0.45) < 0.003]
        # A chord 0.05 up from the bottom of a 0.5 m circle is 2 * sqrt(0.5^2 - 0.45^2) = 0.44 wide.
        self.assertGreater(float(np.ptp(flat[:, 0])), 0.3)


class ClayJsonTests(unittest.TestCase):
    # Verbatim out/mlg-hay_bale_v2/workbench/clay.json as clay_set wrote it before rounding, bulge and noise.
    OLD_BALE = """{
  "units": "m",
  "up": "Y",
  "blobs": [
    {
      "name": "bale",
      "form": {
        "type": "Cylinder",
        "radius": 0.75,
        "height": 1.2,
        "note": "Clide3D has no Cylinder; it calls this a Lathe (a revolved rectangle profile)."
      },
      "op": "add",
      "blend": 0.0,
      "color": [
        0.7843,
        0.6353,
        0.3098
      ],
      "mirror": false,
      "translation": [
        0.0,
        0.0,
        0.0
      ],
      "rotation": [
        90.0,
        0.0,
        0.0
      ]
    }
  ]
}
"""

    def test_old_clay_json_saves_back_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "old.json", Path(tmp) / "again.json"
            src.write_text(self.OLD_BALE)
            box = clay.to_json(clay.from_json([{"name": "b", "form": {"type": "Box", "size": [1, 1, 1]}}]))
            clay.save(clay.load(src), out)
            self.assertEqual(out.read_text(), self.OLD_BALE)
        # Box always wrote its rounding, and still does.
        self.assertEqual(box["blobs"][0]["form"]["rounding"], 0.0)

    def test_new_fields_round_trip(self) -> None:
        b = blob(name="bale", form={"type": "Cylinder", "radius": 0.75, "height": 1.2, "rounding": 0.12,
                                    "bulge": 0.05}, noise={"amp": 0.015, "scale": 0.15, "seed": 7})
        d = b.to_dict()
        self.assertEqual((d["form"]["rounding"], d["form"]["bulge"]), (0.12, 0.05))
        self.assertEqual(d["noise"], {"amp": 0.015, "scale": 0.15, "seed": 7, "octaves": 2})
        self.assertEqual(clay.Blob.from_dict(d).to_dict(), d)
        # amp 0 turns noise off, with or without a scale (clay_edit on a blob that never had noise).
        self.assertIsNone(blob(form={"type": "Sphere", "radius": 1}, noise={"amp": 0}).noise)
        with self.assertRaises(ValueError):
            blob(form={"type": "Sphere", "radius": 1}, noise={"amp": 0.01})   # real noise still needs a scale
        plain = blob(form={"type": "Cylinder", "radius": 0.5, "height": 1.0, "bulge": 0.0}).to_dict()
        self.assertNotIn("bulge", plain["form"])
        self.assertNotIn("noise", plain)

    def test_round_trip_and_cylinder_note(self) -> None:
        blobs = [blob(name="bale", form={"type": "Cylinder", "radius": 0.6, "height": 0.9}, color="#c8a85a",
                      rotation=[0, 0, 90], translation=[0, 0.6, 0]),
                 blob(name="hole", form={"type": "Torus", "major": 0.3, "minor": 0.05}, op="carve")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clay.json"
            clay.save(blobs, path)
            data = clay.to_json(blobs)
            self.assertEqual(data["blobs"][0]["form"]["type"], "Cylinder")
            self.assertIn("Lathe", data["blobs"][0]["form"]["note"])
            again = clay.load(path)
        self.assertEqual([b.to_dict() for b in again], [b.to_dict() for b in blobs])
        self.assertAlmostEqual(again[0].color[0], 200 / 255, places=3)

    def test_lathe_loads_as_cylinder(self) -> None:
        b = blob(form={"type": "Lathe", "radius": 0.2, "height": 0.5})
        self.assertEqual(b.form.type, "Cylinder")

    def test_bad_input_says_what_is_wrong(self) -> None:
        with self.assertRaisesRegex(ValueError, "radius"):
            blob(form={"type": "Sphere"})
        with self.assertRaisesRegex(ValueError, "not one of"):
            blob(form={"type": "Cone", "radius": 1})
        with self.assertRaisesRegex(ValueError, "carve"):
            blob(form={"type": "Sphere", "radius": 1}, op="subtract")
        with self.assertRaisesRegex(ValueError, "repeat"):
            clay.from_json([{"name": "a", "form": {"type": "Sphere", "radius": 1}}] * 2)


class ClayGlbTests(unittest.TestCase):
    BLOBS = [
        {"name": "body", "form": {"type": "Capsule", "radius": 0.3, "height": 1.0}, "rotation": [90, 0, 0],
         "translation": [0, 0.7, 0], "color": [0.9, 0.9, 0.85]},
        {"name": "leg", "form": {"type": "Cylinder", "radius": 0.07, "height": 0.5}, "translation": [0.18, 0.25, 0.3],
         "mirror": True, "color": [0.2, 0.15, 0.1], "blend": 0.03},
    ]

    def test_glb_has_uvs_texture_and_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "clay.glb"
            info = mesher.clay_glb(clay.from_json(self.BLOBS), out, grid=64, height=1.2, texture=512)
            mesh = trimesh.load(out, force="mesh")
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertEqual(mesh.visual.uv.shape, (len(mesh.vertices), 2))
        self.assertAlmostEqual(float(mesh.bounds[0][1]), 0.0, places=4)
        self.assertAlmostEqual(float(mesh.bounds[1][1]), 1.2, places=3)
        self.assertEqual(info["faces"], len(mesh.faces))
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).reshape(-1, 3)
        # Both blob colours are baked: pale wool and dark legs.
        self.assertTrue((tex.mean(1) > 200).any())
        self.assertTrue((tex.mean(1) < 60).any())

    def test_retexture_reads_it(self) -> None:
        # retexture.main() loads inline: trimesh.load(force="mesh"), visual.uv not None, then a base texture
        # from baseColorTexture or image. Same checks here, then its own surface sampler and UV lookup.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "clay.glb"
            mesher.clay_glb(clay.from_json(self.BLOBS), out, grid=48, height=1.0, texture=256)
            mesh = trimesh.load(str(out), force="mesh")
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertIsNotNone(getattr(mesh.visual, "uv", None))
        mat = mesh.visual.material
        image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        self.assertIsNotNone(image)
        self.assertEqual(np.asarray(image.convert("RGB")).shape, (256, 256, 3))
        pts, faces, bary, nrm = retexture.sample_surface(mesh, 5000)
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
        uv_pts = np.einsum("ni,nij->nj", bary, uvs[faces])
        self.assertTrue(((uv_pts >= 0) & (uv_pts <= 1)).all())


class UnwrapCliTests(unittest.TestCase):
    def test_vertex_colours_become_a_texture(self) -> None:
        box = trimesh.creation.box(extents=(1, 1, 1))
        box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=np.tile([200, 40, 40, 255], (len(box.vertices), 1)))
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in.glb", Path(tmp) / "out.glb"
            box.export(src)
            run = subprocess.run([sys.executable, str(ROOT / "scripts" / "workbench" / "unwrap_cli.py"), str(src),
                                  "-o", str(out), "--texture", "128"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("vertex colours", run.stdout)
            mesh = trimesh.load(out, force="mesh")
        self.assertIsNotNone(mesh.visual.uv)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).reshape(-1, 3)
        np.testing.assert_allclose(np.median(tex, axis=0), [200, 40, 40], atol=2)

    def test_plain_mesh_gets_grey(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in.glb", Path(tmp) / "out.glb"
            trimesh.creation.icosphere(2).export(src)
            info = unwrap_cli.unwrap_glb(src, out, texture=64)
            mesh = trimesh.load(out, force="mesh")
        self.assertIn(info["source"], ("flat",))
        self.assertIsNotNone(mesh.visual.uv)

    def test_textured_mesh_kept_unless_forced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clay_glb = Path(tmp) / "clay.glb"
            mesher.clay_glb(clay.from_json(self.BLOBS_TWO_TONE), clay_glb, grid=40, texture=128)
            kept = unwrap_cli.unwrap_glb(clay_glb, Path(tmp) / "kept.glb")
            self.assertEqual(kept["source"], "kept")
            self.assertEqual((Path(tmp) / "kept.glb").read_bytes(), clay_glb.read_bytes())
            redone = unwrap_cli.unwrap_glb(clay_glb, Path(tmp) / "redone.glb", texture=128, force=True)
            self.assertEqual(redone["source"], "texture")
            mesh = trimesh.load(Path(tmp) / "redone.glb", force="mesh")
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).reshape(-1, 3)
        self.assertTrue((tex[:, 0] > 180).any() and (tex[:, 2] > 180).any())

    BLOBS_TWO_TONE = [
        {"name": "red", "form": {"type": "Box", "size": [0.5, 0.5, 0.5]}, "translation": [-0.3, 0, 0], "color": [1, 0, 0]},
        {"name": "blue", "form": {"type": "Box", "size": [0.5, 0.5, 0.5]}, "translation": [0.3, 0, 0], "color": [0, 0, 1]},
    ]


class Decimate(unittest.TestCase):
    """Clay has exactly flat areas (a cylinder's caps); quadric decimation used to stall at about half the faces
    there, so a clay hay bale came out of the PS1 pass with ~52k triangles instead of 1,200."""

    def test_clay_cylinder_reaches_a_ps1_budget(self) -> None:
        from psx_lowpoly import decimate
        from reduce_mesh import reduce_mesh
        bale = clay.from_json([{"name": "bale", "form": {"type": "Cylinder", "radius": 0.5, "height": 1.0},
                                "rotation": [0, 0, 90]}])
        mesh = mesher.mesh_clay(bale, grid=64)
        self.assertLessEqual(len(decimate(mesh, 1200).faces), 1800)
        self.assertLessEqual(len(reduce_mesh(mesh, 6000).faces), 9000)


if __name__ == "__main__":
    unittest.main()
