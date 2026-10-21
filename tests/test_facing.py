#!/usr/bin/env python3
"""facing.py: the picture detector (stubbed VLM), the mesh check and the psx_lowpoly gate, clean_psx."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import facing  # noqa: E402
import factory_env  # noqa: E402
import psx_lowpoly  # noqa: E402


def bird(head_z: float = 1.0):
    """A synthetic standing bird, head toward +Z * head_z: body ellipsoid, a smaller head sphere up front, a tail
    wedge behind, two legs (stacked so the shin band has vertices) and toe boxes pointing forward.
    Returns (verts, faces, uvs) with H = 1."""
    import trimesh

    s = float(np.sign(head_z))
    parts = []
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([0.22, 0.2, 0.32])
    body.apply_translation([0, 0.55, 0])
    parts.append(body)
    head = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    head.apply_translation([0, 0.88, 0.3 * s])
    parts.append(head)
    tail = trimesh.creation.box(extents=[0.3, 0.25, 0.2])
    tail.apply_translation([0, 0.62, -0.38 * s])
    parts.append(tail)
    for x in (-0.08, 0.08):
        for y0 in np.arange(0.035, 0.40, 0.04):
            leg = trimesh.creation.cylinder(radius=0.02, height=0.04, sections=8)
            leg.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            leg.apply_translation([x, y0 + 0.02, 0])
            parts.append(leg)
        toe = trimesh.creation.box(extents=[0.05, 0.02, 0.16])
        toe.apply_translation([x, 0.01, 0.1 * s])
        parts.append(toe)
    m = trimesh.util.concatenate(parts)
    v = np.asarray(m.vertices, float)
    uv = np.c_[(v[:, 0] - v[:, 0].min()) / np.ptp(v[:, 0]), (v[:, 1] - v[:, 1].min()) / np.ptp(v[:, 1])]
    return v, np.asarray(m.faces), uv


def write_bird_glb(path: Path, head_z: float = 1.0) -> None:
    from PIL import Image

    v, f, uv = bird(head_z)
    psx_lowpoly.write_glb(path, v / psx_lowpoly.INCH, f, uv, Image.new("RGB", (16, 16), (200, 180, 60)), "bird")


def vlm_says_head(z_sign: int):
    """A view-aware stub for facing.ask: the answers a VLM gives for a head at z_sign * Z, by render name."""
    def ask(png, question, keep=False):
        stem = Path(png).stem
        for view in ("xside", "side", "front", "back"):
            if stem.endswith("_" + view):
                break
        else:
            return "NONE"
        if view == "side":
            return "RIGHT" if z_sign < 0 else "LEFT"
        if view == "xside":
            return "LEFT" if z_sign < 0 else "RIGHT"
        if view == "front":
            return "AWAY" if z_sign < 0 else "FACING"
        return "FACING" if z_sign < 0 else "AWAY"
    return ask


class DetectImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        from PIL import Image

        self.png = Path(self._tmp.name) / "source.png"
        Image.new("RGB", (32, 24), (255, 0, 255)).save(self.png)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def stub(self, head: str, mirrored: str, profile: str = "YES"):
        def ask(png, question, keep=False):
            if question == facing.PROFILE_Q:
                return profile
            return mirrored if Path(png).stem.endswith("_mirrored") else head
        return mock.patch.object(facing, "ask", side_effect=ask)

    def test_right_profile(self) -> None:
        with self.stub("RIGHT", "Left.", "Yes.") as ask:
            got = facing.detect_image(self.png)
        self.assertEqual((got["head"], got["profile"], got["how"]), ("right", True, "vlm"))
        self.assertGreaterEqual(got["confidence"], facing.ACCEPT)
        self.assertEqual(ask.call_count, 3)
        self.assertTrue(facing.judge(got, "right")["ok"])

    def test_words_parse(self) -> None:
        for word, mirror, want in (("LEFT", "RIGHT", "left"), ("CAMERA", "CAMERA", "camera"),
                                   ("FACING", "facing", "camera"), ("away.", "AWAY", "away"),
                                   ("**Right**", "left", "right"), ("The head is on the LEFT.", "RIGHT", "left")):
            with self.subTest(word=word), self.stub(word, mirror, "NO"):
                got = facing.detect_image(self.png)
                self.assertEqual(got["head"], want)
                self.assertIs(got["profile"], False)

    def test_inconsistent_mirror_is_low_confidence(self) -> None:
        with self.stub("RIGHT", "RIGHT"):
            got = facing.detect_image(self.png)
        self.assertLess(got["confidence"], facing.ACCEPT)
        self.assertFalse(facing.judge(got, "right")["ok"])

    def test_garbage_and_unavailable(self) -> None:
        with self.stub("I cannot tell", "maybe"):
            got = facing.detect_image(self.png)
        self.assertIsNone(got["head"])
        self.assertEqual(got["confidence"], 0.0)
        with mock.patch.object(facing, "ask", side_effect=facing.VLMUnavailable("down")):
            got = facing.detect_image(self.png)
        self.assertEqual(got["how"], "unavailable")
        verdict = facing.judge(got, "right")  # nothing saw the drawing: never a pass
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["flaws"], ["facing_unavailable"])
        self.assertFalse(verdict["checked"])
        with mock.patch.dict(os.environ, {"FACING_VLM": "off"}), self.assertRaises(facing.VLMUnavailable):
            facing.ask(self.png, facing.HEAD_Q)

    def test_judge(self) -> None:
        left = {"head": "left", "profile": True, "confidence": 0.9, "how": "vlm"}
        j = facing.judge(left, "right")
        self.assertEqual((j["ok"], j["mirror"], j["flaws"], j["rank"]), (False, True, ["facing"], 1))
        tq = {"head": "right", "profile": False, "confidence": 0.9, "how": "vlm"}
        j = facing.judge(tq, "right")
        self.assertEqual((j["ok"], j["mirror"], j["flaws"], j["rank"]), (False, False, ["profile"], 2))
        cam = {"head": "camera", "profile": False, "confidence": 0.9, "how": "vlm"}
        self.assertTrue(facing.judge(cam, "camera")["ok"])


class CheckMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_backwards_bird_needs_a_flip(self) -> None:
        glb = self.dir / "back.glb"
        write_bird_glb(glb, head_z=-1)
        with mock.patch.object(facing, "ask", side_effect=vlm_says_head(-1)):
            got = facing.check_mesh(glb, "avian", "right")
        self.assertEqual(got["votes"]["toes"], -1)
        self.assertEqual(got["votes"]["vlm"], -1)
        self.assertEqual(got["head_axis"], "-Z")
        self.assertEqual(got["vlm"]["how"], "textured_views")  # the four CPU renders, not the Blender fallback
        self.assertEqual(set(got["vlm"]["answers"]), set(facing.MESH_VIEWS))
        self.assertTrue(got["flipped_needed"])
        self.assertFalse(got["agrees"])

    def test_right_way_bird_agrees(self) -> None:
        glb = self.dir / "fwd.glb"
        write_bird_glb(glb, head_z=1)
        with mock.patch.object(facing, "ask", side_effect=vlm_says_head(1)):
            got = facing.check_mesh(glb, "avian", "right")
        self.assertEqual((got["votes"]["toes"], got["votes"]["vlm"]), (1, 1))
        self.assertTrue(got["agrees"])
        self.assertFalse(got["flipped_needed"])

    def test_toes_alone_do_not_flip(self) -> None:
        glb = self.dir / "back.glb"
        write_bird_glb(glb, head_z=-1)
        with mock.patch.object(facing, "ask", side_effect=facing.VLMUnavailable("down")):
            got = facing.check_mesh(glb, "avian", "right")
        self.assertEqual(got["votes"]["toes"], -1)
        self.assertEqual(got["votes"]["vlm"], 0)
        self.assertFalse(got["flipped_needed"])


class PsxGateTests(unittest.TestCase):
    """psx_lowpoly end to end on the synthetic bird: a backwards model is turned and psx.json says so."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_psx(self, src: Path, z_sign: int, *extra: str) -> tuple[dict, np.ndarray]:
        import rig

        out = self.dir / f"psx{z_sign}"
        argv = ["psx_lowpoly.py", str(src), "--out", str(out), "--name", "bird", "--height", "48",
                "--facing", "camera", "--faces", "20000", "--atlas", "64", "--glb", str(out / "psx.glb"), *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(facing, "ask", side_effect=vlm_says_head(z_sign)), \
                mock.patch("sys.stdout"):
            self.assertEqual(psx_lowpoly.main(), 0)
        rep = json.loads((out / "psx.json").read_text())
        doc, b = rig.read_glb(out / "psx.glb")
        P = rig.accessor(doc, b, doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"])
        return rep, P

    @staticmethod
    def toe_z(P: np.ndarray) -> float:
        H = np.ptp(P[:, 1])
        feet = P[P[:, 1] < P[:, 1].min() + 0.02 * H]
        return float(feet[:, 2].mean())

    def test_backwards_bird_is_flipped(self) -> None:
        src = self.dir / "src_back.glb"
        write_bird_glb(src, head_z=-1)  # placed with facing camera (yaw 0), so the head stays at -Z
        rep, P = self.run_psx(src, -1)
        fc = rep["facing_check"]
        self.assertTrue(fc["flipped"], fc)
        self.assertIn("Facing corrected", fc["why"])
        self.assertEqual(rep["facing_source"], "cues")
        self.assertGreater(self.toe_z(P), 0)  # the toes, and so the head, now at +Z

    def test_right_way_bird_is_kept(self) -> None:
        src = self.dir / "src_fwd.glb"
        write_bird_glb(src, head_z=1)
        rep, P = self.run_psx(src, 1, "--canonical", "--facing-source", "drawn")
        fc = rep["facing_check"]
        self.assertFalse(fc["flipped"])
        self.assertTrue(fc["agrees"])
        self.assertEqual(fc["decided_by"], "declared")
        self.assertEqual(rep["facing_source"], "drawn")
        self.assertGreater(self.toe_z(P), 0)

    def test_canonical_from_key_json(self) -> None:
        """generate.sh passes only --facing: canonical and facing_source come from the asset's flat/key.json."""
        (self.dir / "flat").mkdir()
        (self.dir / "flat" / "key.json").write_text(json.dumps({"canonical": True, "facing": "camera"}))
        src = self.dir / "src_fwd.glb"
        write_bird_glb(src, head_z=1)
        rep, _ = self.run_psx(src, 1)
        self.assertTrue(rep["canonical"])
        self.assertEqual(rep["facing_source"], "drawn")
        self.assertEqual(rep["facing_check"]["decided_by"], "declared")

    def test_check_off(self) -> None:
        src = self.dir / "src_back.glb"
        write_bird_glb(src, head_z=-1)
        rep, P = self.run_psx(src, -1, "--facing-check", "off")
        self.assertEqual(rep["facing_check"], {"skipped": True})
        self.assertLess(self.toe_z(P), 0)

    def test_gate_skips_a_prop(self) -> None:
        (self.dir / "intent.json").write_text(json.dumps({"output": "ps1", "name": "crate", "subject": "prop"}))
        try:
            src = self.dir / "src_back.glb"
            write_bird_glb(src, head_z=-1)
            rep, P = self.run_psx(src, -1)
            self.assertTrue(rep["facing_check"]["skipped"])
            self.assertIn("not a creature", rep["facing_check"]["why"])
            self.assertLess(self.toe_z(P), 0)  # never turned
            self.assertEqual(psx_lowpoly.subject_kind(src), (False, None))
            (self.dir / "intent.json").write_text(json.dumps({"output": "ps1", "name": "chicken"}))
            self.assertTrue(psx_lowpoly.subject_kind(src)[0])
        finally:
            (self.dir / "intent.json").unlink(missing_ok=True)

    def test_turn_180_is_a_rotation(self) -> None:
        v = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])
        self.assertTrue(np.allclose(psx_lowpoly.turn_180(v), [[-1, 2, -3], [1, 0, -0.5]]))
        placed = psx_lowpoly.place(__import__("trimesh").Trimesh(vertices=v, faces=[[0, 1, 0]], process=False), 1, 180)
        base = psx_lowpoly.place(__import__("trimesh").Trimesh(vertices=v, faces=[[0, 1, 0]], process=False), 1, 0)
        self.assertTrue(np.allclose(placed.vertices, psx_lowpoly.turn_180(base.vertices)))


class CleanPsxTests(unittest.TestCase):
    def test_bad_facing_raises(self) -> None:
        with self.assertRaises(ValueError):
            factory_env.clean_psx({"facing": "sideways"})
        with self.assertRaises(ValueError):
            factory_env.clean_psx({"facing": "Right"})

    def test_missing_and_good_facing(self) -> None:
        self.assertEqual(factory_env.clean_psx({"height": 48}), {"height": "48"})
        self.assertEqual(factory_env.clean_psx({"facing": None}), {})
        for f in ("camera", "left", "right", "away"):
            self.assertEqual(factory_env.clean_psx({"facing": f})["facing"], f)


if __name__ == "__main__":
    unittest.main()
