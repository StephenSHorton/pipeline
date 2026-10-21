#!/usr/bin/env python3
"""Takes: save a stage, step back to an earlier one, see what went stale."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import flow  # noqa: E402


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class FlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.asset = self.root / "out" / "radio"
        write(self.asset / "flat" / "source.png", "still one")
        write(self.asset / "mesh.glb", "mesh from still one")
        write(self.asset / "stack" / "stack.json", "{}")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def stage(self, name: str) -> dict:
        return next(s for s in flow.state(self.root, "radio")["stages"] if s["stage"] == name)

    def test_old_asset_reads_as_unsaved(self) -> None:
        self.assertTrue(self.stage("concept")["unsaved"])
        self.assertTrue(self.stage("mesh")["live"])
        self.assertFalse(self.stage("ideas")["live"])
        self.assertFalse((self.asset / flow.MANIFEST).exists())

    def test_snapshot_twice_keeps_one_take(self) -> None:
        first = flow.snapshot(self.root, "radio", "concept")
        again = flow.snapshot(self.root, "radio", "concept")
        self.assertTrue(first["created"])
        self.assertFalse(again["created"])
        self.assertEqual(len(self.stage("concept")["takes"]), 1)
        self.assertFalse(self.stage("concept")["unsaved"])

    def test_step_back_restores_files_and_marks_mesh_stale(self) -> None:
        flow.snapshot(self.root, "radio", "concept")
        flow.snapshot(self.root, "radio", "mesh")
        self.assertEqual(self.stage("mesh")["takes"][0]["from"], {"concept": "t1"})

        write(self.asset / "flat" / "source.png", "still two")
        flow.snapshot(self.root, "radio", "concept", note="brighter")
        self.assertEqual(self.stage("mesh")["stale"], ["concept"])

        flow.pick(self.root, "radio", "concept", "t1")
        self.assertEqual((self.asset / "flat" / "source.png").read_text(), "still one")
        self.assertEqual(self.stage("mesh")["stale"], [])
        self.assertFalse(self.stage("concept")["unsaved"])

    def test_pick_keeps_unsaved_work(self) -> None:
        flow.snapshot(self.root, "radio", "concept")
        write(self.asset / "flat" / "source.png", "edited, never saved")
        write(self.asset / "input.png", "an extra file")
        state = flow.pick(self.root, "radio", "concept", "t1")
        concept = next(s for s in state["stages"] if s["stage"] == "concept")
        self.assertEqual([t["id"] for t in concept["takes"]], ["t1", "t2"])
        self.assertFalse((self.asset / "input.png").exists())
        flow.pick(self.root, "radio", "concept", "t2")
        self.assertEqual((self.asset / "input.png").read_text(), "an extra file")

    def test_routes(self) -> None:
        status, body, _ = flow.handle(self.root, "POST", "/flow/radio/snapshot", b'{"stage": "stack"}')
        self.assertEqual(status, 200, body)
        status, body, _ = flow.handle(self.root, "GET", "/flow/radio", b"")
        stack = next(e for e in json.loads(body)["stages"] if e["stage"] == "stack")
        self.assertEqual(stack["pick"], "t1")
        status, _, _ = flow.handle(self.root, "POST", "/flow/radio/pick", b'{"stage": "nope", "take": "t1"}')
        self.assertEqual(status, 400)
        status, _, _ = flow.handle(self.root, "GET", "/flow/missing", b"")
        self.assertEqual(status, 404)
        status, _, _ = flow.handle(self.root, "POST", "/flow/radio/pick", b"{}", busy=lambda stem: True)
        self.assertEqual(status, 409)
        self.assertIsNone(flow.handle(self.root, "GET", "/2d/status", b""))

    def test_autosave_skips_empty_stages(self) -> None:
        self.assertEqual(flow.autosave(self.root, "radio", ("concept", "ideas", "mesh", "stack")), ["concept", "mesh", "stack"])
        self.assertEqual(self.stage("stack")["takes"][0]["from"], {"mesh": "t1", "concept": "t1"})

    def test_3d_run_files_do_not_make_a_new_concept(self) -> None:
        # A 3D build from a 2D still: save before, the run copies input.png and cuts lifted.png, save after.
        flow.autosave(self.root, "radio", ("concept", "mesh", "stack"))
        write(self.asset / "input.png", "copy of still one")
        write(self.asset / "lifted.png", "cutout")
        write(self.asset / "mesh.glb", "new mesh")
        flow.autosave(self.root, "radio", ("concept", "mesh", "stack"))
        concept = self.stage("concept")
        self.assertEqual([t["id"] for t in concept["takes"]], ["t1"])
        self.assertFalse(concept["unsaved"])
        self.assertIn("lifted.png", concept["takes"][0]["files"])
        self.assertEqual(self.stage("mesh")["stale"], [])

        # The take keeps the cutout, so stepping away and back restores it.
        write(self.asset / "flat" / "source.png", "still two")
        flow.snapshot(self.root, "radio", "concept")
        self.assertEqual(self.stage("mesh")["stale"], ["concept"])
        flow.pick(self.root, "radio", "concept", "t1")
        self.assertEqual((self.asset / "lifted.png").read_text(), "cutout")
        self.assertEqual(self.stage("mesh")["stale"], [])

    def test_photo_input_still_defines_the_concept(self) -> None:
        (self.asset / "flat" / "source.png").unlink()
        write(self.asset / "input.jpg", "photo one")
        flow.snapshot(self.root, "radio", "concept")
        write(self.asset / "input.jpg", "photo two, longer")
        self.assertTrue(flow.snapshot(self.root, "radio", "concept")["created"])

    def test_takes_saved_before_still_match(self) -> None:
        write(self.asset / "lifted.png", "cutout")
        flow.snapshot(self.root, "radio", "concept")
        data = flow.read_manifest(self.asset)
        data["stages"]["concept"]["takes"][0]["print"] = flow.fingerprint(self.asset, flow.stage_files(self.asset, "concept"))
        flow.write_manifest(self.asset, data)
        self.assertFalse(self.stage("concept")["unsaved"])
        self.assertFalse(flow.snapshot(self.root, "radio", "concept")["created"])

    def rig_files(self) -> None:
        write(self.asset / "rigged.glb", "rig one")
        write(self.asset / "rig" / "rig.json", "{}")
        write(self.asset / "rig" / "rig.log", "rig: log")

    def test_checking_a_rig_again_keeps_its_version(self) -> None:
        self.rig_files()
        flow.snapshot(self.root, "radio", "rig")
        write(self.asset / "rig" / "inspect.json", '{"version": 2, "longer": true}')
        write(self.asset / "rig" / "summary.json", '{"line": "No tears"}')
        rig = self.stage("rig")
        self.assertFalse(rig["unsaved"])
        self.assertEqual(rig["summary"], {"line": "No tears"})
        self.assertIsNone(rig["takes"][0]["summary"])  # the saved copy has no summary until it is refreshed
        flow.resave(self.root, "radio", "rig", "t1")
        rig = self.stage("rig")
        self.assertEqual([t["id"] for t in rig["takes"]], ["t1"])
        self.assertEqual(rig["takes"][0]["summary"], {"line": "No tears"})
        self.assertIn("rig/summary.json", rig["takes"][0]["files"])
        self.assertFalse(rig["unsaved"])
        self.assertNotIn("summary", flow.read_manifest(self.asset)["stages"]["rig"]["takes"][0])
        write(self.asset / "rigged.glb", "rig two, longer")
        self.assertTrue(self.stage("rig")["unsaved"])

    def test_resave_drops_files_that_are_gone(self) -> None:
        self.rig_files()
        write(self.asset / "rig" / "old.fbx", "fbx")
        flow.snapshot(self.root, "radio", "rig")
        (self.asset / "rig" / "old.fbx").unlink()
        take = flow.resave(self.root, "radio", "rig", "t1")
        self.assertNotIn("rig/old.fbx", take["files"])
        self.assertFalse((self.asset / "takes" / "rig" / "t1" / "rig" / "old.fbx").exists())
        with self.assertRaises(flow.FlowError):
            flow.resave(self.root, "radio", "rig", "t9")

    def test_rig_takes_saved_before_still_match(self) -> None:
        self.rig_files()
        write(self.asset / "rig" / "inspect.json", "{}")
        flow.snapshot(self.root, "radio", "rig")
        data = flow.read_manifest(self.asset)
        data["stages"]["rig"]["takes"][0]["print"] = flow.fingerprint(self.asset, flow.stage_files(self.asset, "rig"))
        flow.write_manifest(self.asset, data)
        self.assertFalse(self.stage("rig")["unsaved"])
        self.assertFalse(flow.snapshot(self.root, "radio", "rig")["created"])
        self.assertIsNone(self.stage("mesh").get("summary"))
        self.assertNotIn("summary", self.stage("mesh"))


class FactoryAutosaveTest(unittest.TestCase):
    def test_2d_writes_save_takes(self) -> None:
        import base64
        import io

        import factory_http
        from PIL import Image

        im = Image.new("RGB", (48, 32), (255, 0, 255))
        im.paste((180, 40, 40), (4, 4, 14, 14))
        im.paste((40, 40, 40), (28, 16, 42, 28))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factory = factory_http.Factory(root)
            body = json.dumps({"stem": "unit", "data": data}).encode("utf-8")
            status, _, _ = factory_http.dispatch(factory, "POST", "/2d/drop", {}, body)
            self.assertEqual(status, 201)
            status, _, _ = factory_http.dispatch(factory, "POST", "/2d/split", {}, b'{"stem": "unit"}')
            self.assertEqual(status, 200)
            stages = {s["stage"]: s for s in flow.state(root, "unit")["stages"]}
            self.assertEqual(stages["concept"]["pick"], "t1")
            self.assertEqual(stages["flat"]["takes"][0]["from"], {"concept": "t1"})
            self.assertFalse(stages["flat"]["unsaved"])


if __name__ == "__main__":
    unittest.main()
