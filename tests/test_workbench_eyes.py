#!/usr/bin/env python3
"""scripts/workbench/eyes.py on synthetic meshes: a primitive quadruped (head centred or shifted sideways), a
cylinder, a box, an upright post; look's contact sheet, and compare_reference against a silhouette rendered from
the quadruped itself."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import retexture  # noqa: E402
from workbench import eyes  # noqa: E402

WIDTH = 0.8  # the body ellipsoid's width
N = 60_000


def quadruped(head_dx: float = 0.0, head: bool = True, head_y: float = 1.0) -> trimesh.Trimesh:
    """Body along +Z on the ground: ellipsoid barrel, a head sphere in front of it (above it unless grazing), four
    legs."""
    body = trimesh.creation.icosphere(subdivisions=3)
    body.apply_scale([WIDTH / 2, 0.35, 1.0])
    body.apply_translation([0, 0.8, 0])
    parts = [body]
    if head:
        parts.append(trimesh.creation.icosphere(subdivisions=3, radius=0.25).apply_translation([head_dx, head_y, 1.15]))
    for x in (-0.22, 0.22):
        for z in (-0.6, 0.6):
            leg = upright(trimesh.creation.cylinder(radius=0.08, height=0.6, sections=16))
            parts.append(leg.apply_translation([x, 0.3, z]))
    return trimesh.util.concatenate(parts)


def upright(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """trimesh's cylinders run along Z; stand one up along Y."""
    return m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))


def lying_cylinder() -> trimesh.Trimesh:
    """A bale-like cylinder on its side, axis along Z, longer than tall so the body frame applies."""
    return trimesh.creation.cylinder(radius=0.5, height=1.4, sections=64).apply_translation([0, 0.5, 0])


class Eyes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def glb(self, name: str, mesh: trimesh.Trimesh) -> Path:
        p = self.dir / f"{name}.glb"
        mesh.export(p)
        return p

    def test_head_offset_shifted_and_centred(self):
        shifted = eyes.measure(self.glb("shifted", quadruped(0.2 * WIDTH)), n=N)
        self.assertAlmostEqual(shifted["head_offset_pct"], 20, delta=3)
        self.assertEqual(shifted["head_offset_side"], "left")  # +X is the creature's left when it faces +Z
        other = eyes.measure(self.glb("shifted_r", quadruped(-0.2 * WIDTH)), n=N)
        self.assertAlmostEqual(other["head_offset_pct"], 20, delta=3)
        self.assertEqual(other["head_offset_side"], "right")
        centred = eyes.measure(self.glb("centred", quadruped()), n=N)
        self.assertLess(centred["head_offset_pct"], 1.5)
        self.assertLess(abs(centred["head_yaw_deg"]), 3)

    def test_grazing_head_found_by_facing(self):
        # Head below the rump: rest_check's highest-end rule takes the rump for the head; the +Z facing doesn't.
        glb = self.glb("grazing", quadruped(0.2 * WIDTH, head_y=0.55))
        self.assertAlmostEqual(eyes.measure(glb, n=N)["head_offset_pct"], 20, delta=3)
        self.assertLess(eyes.measure(glb, n=N, toward=None)["head_offset_pct"], 10)

    def test_mirror_iou(self):
        centred = eyes.measure(self.glb("centred_m", quadruped()), n=N)
        shifted = eyes.measure(self.glb("shifted_m", quadruped(0.2 * WIDTH)), n=N)
        self.assertGreater(centred["mirror_iou"], 0.95)
        self.assertLess(shifted["mirror_iou"], centred["mirror_iou"] - 0.02)

    def test_mirror_is_left_right_when_wider_than_deep(self):
        # Shoulders wider than the body is deep make X the long ground axis; the mirror must still be about X = 0.
        def figure(*arm_x):
            parts = [trimesh.creation.box([0.6, 1.7, 0.3]).apply_translation([0, 0.85, 0])]
            parts += [trimesh.creation.box([0.6, 0.12, 0.12]).apply_translation([x, 1.4, 0]) for x in arm_x]
            return trimesh.util.concatenate(parts)

        both = eyes.measure(self.glb("arms", figure(-0.6, 0.6)), n=N)
        one = eyes.measure(self.glb("arm", figure(0.6)), n=N)
        self.assertGreater(both["mirror_iou"], 0.97)
        self.assertLess(one["mirror_iou"], 0.65)
        # A wheel or bale face-on to the camera (+Z) is round looking along forward: the end view.
        wheel = trimesh.creation.cylinder(radius=0.5, height=0.4, sections=64).apply_translation([0, 0.5, 0])
        r = eyes.measure(self.glb("wheel", wheel), n=N)
        self.assertGreater(r["circularity"], 0.95)
        self.assertEqual(r["circularity_view"], "end")

    def test_circularity(self):
        cyl = eyes.measure(self.glb("cyl", lying_cylinder()), n=N)
        box = eyes.measure(self.glb("box", trimesh.creation.box([1.4, 1.0, 1.0]).apply_translation([0, 0.5, 0])), n=N)
        self.assertGreater(cyl["circularity"], 0.95)
        self.assertEqual(cyl["circularity_view"], "end")  # looking along the cylinder's axis
        self.assertLess(box["circularity"], 0.75)
        rounded = eyes.ellipse_fill(ndimage.binary_dilation(np.pad(np.ones((80, 80), bool), 40), iterations=12))
        self.assertTrue(0.64 < rounded < 0.95, rounded)  # a rounded box sits between the two

    def test_bounds_tris_pieces(self):
        m = quadruped()
        r = eyes.measure(self.glb("parts", m), n=N)
        self.assertEqual(r["tris"], len(m.faces))
        self.assertEqual(r["pieces"], 6)
        np.testing.assert_allclose(r["bounds"]["max"], m.bounds[1], atol=1e-4)
        soup = trimesh.creation.box([1, 1, 1])
        soup.unmerge_vertices()  # PSX-style: every triangle has its own corners
        self.assertEqual(eyes.pieces(soup), 1)

    def test_upright_has_no_head_offset(self):
        post = upright(trimesh.creation.cylinder(radius=0.2, height=2.0, sections=32)).apply_translation([0, 1.0, 0])
        r = eyes.measure(self.glb("post", post), n=N)
        self.assertIsNone(r["head_offset_pct"])
        self.assertIsNone(r["head_yaw_deg"])
        self.assertIn("upright", r["head_reason"])
        self.assertGreater(r["circularity"], 0.95)
        self.assertEqual(r["circularity_view"], "top")

    def test_look_sheet(self):
        out = eyes.look(self.glb("look", quadruped()), ["side", "front", "top", "q"], 128, self.dir / "look.png")
        with Image.open(out) as im:
            self.assertEqual(im.size, (3 * 128, 2 * (128 + 18)))

    def reference(self, mesh: trimesh.Trimesh, az: float, el: float = 5.0) -> Path:
        """RGBA silhouette of the mesh from retexture's camera at (az, el)."""
        pts, _ = trimesh.sample.sample_surface(mesh, 120_000, seed=1)
        pts = np.asarray(pts, np.float32)
        V = np.asarray(mesh.vertices, np.float32)
        centre = V.mean(0)
        eye = retexture.eye_on_sphere(centre, float(np.linalg.norm(np.ptp(V, axis=0))) * 1.45, az, el)
        vp = retexture.perspective(np.deg2rad(40.0), 1.0, 0.05, 40.0) @ retexture.look_at(eye, centre)
        px, py, _, inb = retexture.project(pts, vp, 320, 320)
        m = np.zeros((320, 320), bool)
        m[py[inb], px[inb]] = True
        m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
        rgba = np.zeros((320, 320, 4), np.uint8)
        rgba[m] = (200, 180, 150, 255)
        p = self.dir / f"ref_{az:g}.png"
        Image.fromarray(rgba, "RGBA").save(p)
        return p

    def test_compare_reference(self):
        full = quadruped()
        ref = self.reference(full, az=-90.0)  # camera at -X: the head points to image right
        same = eyes.compare_reference(self.glb("cmp_full", full), ref, facing="right", out=self.dir / "same.png", n=N)
        self.assertGreater(same["iou"], 0.88)
        self.assertFalse([g for g in same["regions"] if g["area_pct"] > 4], same["regions"])
        self.assertTrue(Path(same["png"]).is_file())

        headless = eyes.compare_reference(self.glb("cmp_headless", quadruped(head=False)), ref, facing="right",
                                          out=self.dir / "headless.png", n=N)
        self.assertLess(headless["iou"], same["iou"])
        missing = [g for g in headless["regions"] if g["kind"] == "missing"]
        self.assertTrue(missing, headless["regions"])
        top = max(missing, key=lambda g: g["area_pct"])
        self.assertGreater(top["area_pct"], 5)
        self.assertGreater(top["world_hint"][2], 0.6)  # the head's end of the body (+Z)
        self.assertGreater(top["bbox"][0], 160)  # and the right half of the photo


if __name__ == "__main__":
    unittest.main()
