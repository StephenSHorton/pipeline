#!/usr/bin/env python3
"""scripts/rig_weights.py and POST /rig/<stem>/weights: hand-edited skin weights spliced into a copy of the shipped
sheep rig (out/mlg-sheep/rigged.glb), read back, only the weight bytes changed; bad payloads refused; a busy stem 409;
the rig kept as a version before and the edit saved as a new one. The FBX exporter is a stub (no Blender here).

    .venv/bin/python tests/test_rig_weights.py
"""
from __future__ import annotations

import base64
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import factory_http  # noqa: E402
import flow  # noqa: E402
import rig_weights as rw  # noqa: E402

SHEEP = next((p for p in (ROOT / "out" / "mlg-sheep", Path.home() / "projects" / "ai-asset-pipeline" / "out" / "mlg-sheep") if (p / "rigged.glb").is_file()), ROOT / "out" / "mlg-sheep")
STEM = "mlg-sheep"


def copy_sheep(root: Path) -> Path:
    folder = root / "out" / STEM
    (folder / "rig").mkdir(parents=True)
    shutil.copy2(SHEEP / "rigged.glb", folder / "rigged.glb")
    shutil.copy2(SHEEP / "rig" / "rigged_allclips.glb", folder / "rig" / "rigged_allclips.glb")
    info = json.loads((SHEEP / "rig" / "rig.json").read_text())
    (folder / "rig" / "rig.json").write_text(json.dumps(info))
    return folder


def bone_index(glb: rw.Glb, name: str) -> int:
    joints = glb.doc["skins"][0]["joints"]
    return [glb.doc["nodes"][j]["name"] for j in joints].index(name)


def edited(glb: rw.Glb) -> dict:
    """Half of every point that the left hind thigh moves goes to the hips instead."""
    (key, (j, w)), = glb.weights().items()
    j, w = j.copy(), w.copy()
    thigh, hips = bone_index(glb, "back_leg_upper_l"), bone_index(glb, "hips")
    for v in range(len(j)):
        row = dict((int(a), float(b)) for a, b in zip(j[v], w[v]) if b > 0)
        if thigh not in row:
            continue
        give = row[thigh] / 2
        row[thigh] -= give
        row[hips] = row.get(hips, 0.0) + give
        top = sorted(row.items(), key=lambda x: -x[1])[:4]
        s = sum(b for _, b in top)
        top += [(0, 0.0)] * (4 - len(top))
        j[v] = [a for a, _ in top]
        w[v] = [b / s for _, b in top]
    return {key: (j, w)}


def fake_fbx(calls: list):
    def export(glb: Path, fbx: Path, clips):
        calls.append((glb, fbx, clips))
        fbx.parent.mkdir(parents=True, exist_ok=True)
        fbx.write_bytes(b"fbx")
        return "FBX checked: stub", {"ok": True, "problems": [], "warnings": [], "scene_fps": 30, "actions": []}
    return export


@unittest.skipUnless((SHEEP / "rigged.glb").is_file(), "needs out/mlg-sheep/rigged.glb")
class RoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.folder = copy_sheep(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_only_the_weights_change(self) -> None:
        before_bytes = (self.folder / "rigged.glb").read_bytes()
        glb = rw.Glb(before_bytes)
        edit = edited(glb)
        calls: list = []
        out = rw.save_weights(self.folder, rw.payload_from(glb, edit, note="less hind thigh"), fake_fbx(calls))
        after_bytes = (self.folder / "rigged.glb").read_bytes()
        self.assertEqual(len(after_bytes), len(before_bytes))
        after = rw.Glb(after_bytes)
        self.assertEqual(after.doc, glb.doc)  # the JSON chunk as it was
        # Every byte outside JOINTS_0 / WEIGHTS_0 is the same.
        (p,) = glb.skinned()
        spans = []
        for acc in (p["joints_acc"], p["weights_acc"]):
            _, _, off, size, stride = glb._layout(acc)
            spans.append((off, off + stride * (p["count"] - 1) + size))
        mask = np.ones(len(before_bytes), bool)
        for a, b in spans:
            mask[a:b] = False
        diff = np.frombuffer(before_bytes, np.uint8) != np.frombuffer(after_bytes, np.uint8)
        self.assertFalse(diff[mask].any())
        self.assertTrue(diff[~mask].any())
        # Read back: what was sent.
        (key, (j, w)), = after.weights().items()
        ej, ew = edit[key]
        np.testing.assert_array_equal(j, ej)
        np.testing.assert_allclose(w, ew, atol=1e-6)
        np.testing.assert_allclose(w.sum(axis=1), 1, atol=1e-5)
        self.assertGreater(out["changed"], 0)
        # The same-mesh sibling took the edit too; rig.json says so; the FBX was asked for with the shipped clips.
        sib = rw.Glb.read(self.folder / "rig" / "rigged_allclips.glb")
        np.testing.assert_array_equal(list(sib.weights().values())[0][0], ej)
        self.assertIn("rig/rigged_allclips.glb", out["files"])
        info = json.loads((self.folder / "rig" / "rig.json").read_text())
        self.assertEqual(info["edited_weights"]["saves"], 1)
        self.assertEqual(info["edited_weights"]["note"], "less hind thigh")
        self.assertEqual(info["edited_weights"]["changed_points"], out["changed"])
        self.assertEqual(info["fbx_note"], "FBX checked: stub")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], self.folder / info["fbx"])
        self.assertEqual(out["fbx"], info["fbx"])
        # A second save of the same weights changes nothing more and counts the save.
        out2 = rw.save_weights(self.folder, rw.payload_from(after), None)
        self.assertEqual(out2["changed"], 0)
        self.assertEqual((self.folder / "rigged.glb").read_bytes(), after_bytes)
        self.assertEqual(json.loads((self.folder / "rig" / "rig.json").read_text())["edited_weights"]["saves"], 2)

    def test_lists_work_too(self) -> None:
        glb = rw.Glb.read(self.folder / "rigged.glb")
        (key, (j, w)), = edited(glb).items()
        body = {"primitives": [{"mesh": key[0], "primitive": key[1], "joints": j.ravel().tolist(), "weights": w.ravel().tolist()}]}
        rw.save_weights(self.folder, body, None)
        np.testing.assert_array_equal(list(rw.Glb.read(self.folder / "rigged.glb").weights().values())[0][0], j)

    def test_no_blender_keeps_the_glb(self) -> None:
        glb = rw.Glb.read(self.folder / "rigged.glb")
        out = rw.save_weights(self.folder, rw.payload_from(glb, edited(glb)), lambda g, f, c: ("Blender not found; skipped the FBX", None))
        self.assertFalse(out["blender"])
        self.assertIn("Blender not found", out["fbx_note"])
        self.assertIn("rigged.glb", out["files"])

    def test_bad_payloads(self) -> None:
        before = (self.folder / "rigged.glb").read_bytes()
        glb = rw.Glb(before)
        good = rw.payload_from(glb)
        n = good["primitives"][0]["count"]
        j, w = list(glb.weights().values())[0]

        def with_(**kw):
            body = json.loads(json.dumps(good))
            body["primitives"][0].update(kw)
            return body

        def b64(a, dt):
            return base64.b64encode(np.asarray(a, dt).tobytes()).decode()

        unnorm = w.copy()
        unnorm[5] *= 0.5
        nan = w.copy()
        nan[7, 0] = np.nan
        neg = w.copy()
        neg[3] = [1.5, -0.5, 0, 0]
        far = j.copy()
        far[2, 0] = 56
        bad = [
            [],
            {},
            {"primitives": []},
            {"primitives": [good["primitives"][0], good["primitives"][0]]},
            with_(mesh=3),
            with_(count=n + 1),
            with_(joints=good["primitives"][0]["joints"][:-8]),
            with_(weights="not base64!"),
            with_(weights=b64(unnorm, "<f4")),
            with_(weights=b64(nan, "<f4")),
            with_(weights=b64(neg, "<f4")),
            with_(joints=b64(far, "<u2")),
            with_(joints=[0.5] * (n * 4)),
            with_(weights=5),
        ]
        for body in bad:
            with self.subTest(body=str(body)[:80]), self.assertRaises(rw.WeightsError):
                rw.save_weights(self.folder, body, None)
        self.assertEqual((self.folder / "rigged.glb").read_bytes(), before)
        self.assertNotIn("edited_weights", json.loads((self.folder / "rig" / "rig.json").read_text()))

    def test_normalised_u8_weights_add_to_one(self) -> None:
        w = np.array([[0.333, 0.333, 0.334, 0.0], [0.1, 0.2, 0.3, 0.4]])
        q = rw._encode_weights(w, np.dtype("u1"), True)
        self.assertTrue(np.all(q.astype(int).sum(axis=1) == 255))


@unittest.skipUnless((SHEEP / "rigged.glb").is_file(), "needs out/mlg-sheep/rigged.glb")
class Route(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.folder = copy_sheep(self.root)
        self.factory = factory_http.Factory(self.root)
        self.calls: list = []
        real = self.factory.save_rig_weights
        self.factory.save_rig_weights = lambda stem, raw: real(stem, raw, exporter=fake_fbx(self.calls))  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def post(self, body) -> tuple[int, dict]:
        status, data, _ = factory_http.dispatch(self.factory, "POST", f"/rig/{STEM}/weights", {}, json.dumps(body).encode())
        return status, json.loads(data)

    def test_save_keeps_a_version_then_makes_one(self) -> None:
        glb = rw.Glb.read(self.folder / "rigged.glb")
        status, out = self.post(rw.payload_from(glb, edited(glb)))
        self.assertEqual(status, 200, out)
        self.assertTrue(out["ok"])
        takes = flow.read_manifest(self.folder)["stages"]["rig"]["takes"]
        self.assertEqual([t["id"] for t in takes], ["t1", "t2"])
        self.assertEqual(out["take"], "t2")
        self.assertEqual(takes[1]["note"], "Edited weights")
        self.assertEqual(flow.read_manifest(self.folder)["stages"]["rig"]["pick"], "t2")
        # The version kept first is the rig as it was.
        kept = self.folder / flow.TAKES / "rig" / "t1" / "rigged.glb"
        self.assertEqual(kept.read_bytes(), (SHEEP / "rigged.glb").read_bytes())
        self.assertIn("rigged.glb", out["files"])
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.factory.writing(STEM))

    def test_busy_stem_is_409(self) -> None:
        self.factory._job = {"state": "running", "stem": STEM, "job_id": "j1"}
        before = (self.folder / "rigged.glb").read_bytes()
        status, out = self.post(rw.payload_from(rw.Glb(before)))
        self.assertEqual(status, 409)
        self.assertTrue(out["busy"])
        self.assertEqual((self.folder / "rigged.glb").read_bytes(), before)
        self.assertEqual(self.calls, [])

    def test_bad_body_is_400_and_unknown_asset_404(self) -> None:
        status, out = self.post({"primitives": "x"})
        self.assertEqual(status, 400)
        self.assertIn("primitives", out["error"])
        status, _, _ = factory_http.dispatch(self.factory, "POST", "/rig/nobody/weights", {}, b"{}")
        self.assertEqual(status, 404)
        status, _, _ = factory_http.dispatch(self.factory, "POST", f"/rig/{STEM}/weights", {}, b"{nope")
        self.assertEqual(status, 400)


class Synthetic(unittest.TestCase):
    """A strided, u8-joint GLB (no sheep needed): the splice keeps the interleaved bytes it doesn't own."""

    def test_strided_u8(self) -> None:
        n = 3
        stride = 12  # 4 x u8 joints + 2 x 4 pad... a view holding joints then padding
        jbin = bytearray(stride * n)
        for v in range(n):
            jbin[v * stride:v * stride + 4] = bytes([0, 1, 0, 0])
            jbin[v * stride + 4:v * stride + 12] = b"PADPADPA"
        wbin = np.tile(np.array([0.5, 0.5, 0, 0], "<f4"), n).tobytes()
        binary = bytes(jbin) + wbin
        doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
               "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(jbin), "byteStride": stride},
                               {"buffer": 0, "byteOffset": len(jbin), "byteLength": len(wbin)}],
               "accessors": [{"bufferView": 0, "componentType": 5121, "count": n, "type": "VEC4"},
                             {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC4"}],
               "meshes": [{"primitives": [{"attributes": {"JOINTS_0": 0, "WEIGHTS_0": 1}}]}],
               "nodes": [{"mesh": 0, "skin": 0}, {"name": "a"}, {"name": "b"}], "skins": [{"joints": [1, 2]}]}
        js = json.dumps(doc).encode()
        js += b" " * (-len(js) % 4)
        data = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(js) + 8 + len(binary)) + struct.pack("<II", len(js), rw.JSON_CHUNK) + js \
            + struct.pack("<II", len(binary), rw.BIN_CHUNK) + binary
        glb = rw.Glb(data)
        j = np.array([[1, 0, 0, 0]] * n)
        w = np.array([[1.0, 0, 0, 0]] * n)
        out = rw.Glb(rw.splice(glb, {(0, 0): (j, w)}))
        oj, ow = out.weights()[(0, 0)]
        np.testing.assert_array_equal(oj, j)
        np.testing.assert_allclose(ow, w)
        self.assertEqual(out.data.count(b"PADPADPA"), n)
        with self.assertRaises(rw.WeightsError):
            rw.parse_payload({"primitives": [{"mesh": 0, "primitive": 0, "joints": [2, 0, 0, 0] * n, "weights": [1, 0, 0, 0] * n}]}, glb)


if __name__ == "__main__":
    unittest.main()
