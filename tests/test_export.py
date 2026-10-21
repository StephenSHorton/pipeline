#!/usr/bin/env python3
"""Export: one folder per asset a game can load."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_asset  # noqa: E402
import factory_http  # noqa: E402
import flow  # noqa: E402


def png(path: Path, size=(8, 8), color=(200, 60, 40, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data if isinstance(data, str) else json.dumps(data))


class ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.asset = self.root / "out" / "knight"
        self.factory = factory_http.Factory(self.root)
        # The key cuts the test still, not Apple Vision.
        env = mock.patch.dict(os.environ, {"FLAT2D_LIFT": str(self.root / "no-lift")})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def flat(self) -> None:
        a = self.asset
        still = Image.new("RGB", (32, 32), (255, 0, 255))
        still.paste((200, 60, 40), (8, 8, 24, 24))
        (a / "flat").mkdir(parents=True, exist_ok=True)
        still.save(a / "flat" / "source.png")
        write(a / "flat" / "key.json", {"rgb": [255, 0, 255], "tol": 36})
        write(a / "flat" / "prompt.txt", "a knight\n")
        png(a / "flat" / "parts" / "part-01.png")
        png(a / "flat" / "parts" / "part-02.png")
        write(a / "flat" / "sounds" / "part-01.wav", b"RIFF")
        rows = [
            {"id": "part-01", "file": "parts/part-01.png", "name": "Sword", "role": "weapon", "motion": "swing", "sound": "sounds/part-01.wav"},
            {"id": "part-02", "file": "parts/part-02.png", "name": "sword", "role": "other", "voice": "../../secret.wav"},
        ]
        write(a / "flat" / "mechanics.json", {"ok": True, "parts": rows, "note": "test"})
        write(a / "flat" / "parts.json", {"parts": rows})
        png(a / "flat" / "sheets" / "idle.png", (24, 8))
        write(
            a / "flat" / "sheets" / "manifest.json",
            {"actions": {"idle": {"file": "sheets/idle.png", "frames": 3, "fps": 8, "cell": {"w": 8, "h": 8}, "frame_layout": []}}},
        )

    def stack(self) -> None:
        png(self.asset / "stack" / "sheet.png", (16, 4))
        png(self.asset / "stack" / "preview.png")
        write(self.asset / "stack" / "stack.json", {"resolution": 4, "layers": 4, "pitch": 0.25, "maker": "picture"})

    def mesh(self) -> None:
        for name in ("mesh.glb", "textured.glb", "mesh_game.glb", "mesh_proxy.glb"):
            write(self.asset / name, b"glTF" + b"\0" * 80)
        write(self.asset / "lods.json", {"lods": [{"id": "game", "file": "mesh_game.glb", "faces": 5000}]})

    def test_everything_lands_with_a_manifest(self) -> None:
        self.flat()
        self.stack()
        self.mesh()
        result = export_asset.build(self.root, "knight")
        out = self.asset / "export"
        doc = json.loads((out / "asset.json").read_text())
        self.assertEqual(doc["format"], "pipeline-asset")
        self.assertEqual(doc["prompt"], "a knight")
        self.assertEqual(doc["concept"], "concept.png")
        with Image.open(out / "concept.png") as concept:
            self.assertEqual(concept.mode, "RGBA")
            self.assertEqual(concept.getpixel((0, 0)), (200, 60, 40, 0))
            self.assertEqual(concept.getpixel((16, 16)), (200, 60, 40, 255))
        parts = doc["2d"]["parts"]
        self.assertEqual([p["image"] for p in parts], ["2d/parts/sword.png", "2d/parts/sword-2.png"])
        self.assertEqual(parts[0]["sound"], "2d/sounds/sword.wav")
        self.assertEqual(parts[0]["motion"], "swing")
        # A row pointing outside flat/voices is not followed.
        self.assertNotIn("voice", parts[1])
        self.assertEqual(doc["2d"]["motions"]["idle"]["frames"], 3)
        self.assertEqual(doc["stack"]["layers"], 4)
        self.assertEqual(doc["stack"]["maker"], "picture")
        self.assertEqual(doc["3d"]["model"]["from"], "textured.glb")
        self.assertEqual(doc["3d"]["game"]["faces"], 5000)
        self.assertEqual(set(doc["3d"]), {"model", "game", "proxy"})
        for rel in [doc["concept"], *(p["image"] for p in parts), "2d/sheets/idle.png", "stack/sheet.png", "3d/game.glb"]:
            self.assertTrue((out / rel).is_file(), rel)
        self.assertIn("export/asset.json", result["files"])

    def test_photo_bake_is_the_model_and_psx_is_its_own_role(self) -> None:
        self.mesh()
        write(self.asset / "textured_photo.glb", b"glTF" + b"\0" * 80)
        write(self.asset / "psx.glb", b"glTF" + b"\0" * 80)
        write(self.asset / "psx" / "cow.obj", "o cow\n")
        write(self.asset / "psx" / "cow.mtl", "newmtl cow\n")
        png(self.asset / "psx" / "cow.png")
        write(self.asset / "psx" / "psx.json", {
            "name": "cow", "triangles": 1200, "atlas": 256, "height": 58, "facing": "right",
            "obj_axes": "z-up, +x front", "glb_axes": "y-up, +z front (glTF), metres", "size": [87, 31, 58]})
        doc = export_asset.build(self.root, "knight")["asset"]
        out = self.asset / "export"
        self.assertEqual(doc["3d"]["model"]["from"], "textured_photo.glb")
        psx = doc["3d"]["psx"]
        self.assertEqual((psx["file"], psx["obj"], psx["texture"]), ("3d/psx.glb", "3d/psx/cow.obj", "3d/psx/cow.png"))
        self.assertEqual(psx["axes"]["obj"], "z-up, +x front")
        self.assertEqual(psx["triangles"], 1200)
        for rel in (psx["file"], psx["obj"], psx["mtl"], psx["texture"]):
            self.assertTrue((out / rel).is_file(), rel)

    def test_export_again_replaces_the_folder(self) -> None:
        self.stack()
        self.mesh()
        export_asset.build(self.root, "knight")
        self.assertTrue((self.asset / "export" / "3d" / "proxy.glb").is_file())
        (self.asset / "mesh_proxy.glb").unlink()
        export_asset.build(self.root, "knight")
        self.assertFalse((self.asset / "export" / "3d" / "proxy.glb").exists())
        self.assertFalse((self.asset / "export.tmp").exists())

    def test_nothing_to_export(self) -> None:
        png(self.asset / "flat" / "source.png")
        with self.assertRaises(export_asset.ExportError):
            export_asset.build(self.root, "knight")
        self.assertFalse((self.asset / "export").exists())

    def test_route_saves_a_take_that_goes_stale(self) -> None:
        self.flat()
        flow.snapshot(self.root, "knight", "concept")
        flow.snapshot(self.root, "knight", "flat")
        status, raw, _ = factory_http.dispatch(self.factory, "POST", "/export/knight", {}, b"{}")
        self.assertEqual(status, 200, raw)
        status, raw, _ = factory_http.dispatch(self.factory, "GET", "/export/knight", {}, b"")
        self.assertEqual(json.loads(raw)["asset"]["stem"], "knight")
        entry = next(s for s in flow.state(self.root, "knight")["stages"] if s["stage"] == "export")
        self.assertEqual(entry["takes"][0]["from"], {"concept": "t1", "flat": "t1"})
        self.assertFalse(entry["unsaved"])
        self.assertEqual(entry["stale"], [])
        # A new 2D take makes the export out of date.
        png(self.asset / "flat" / "parts" / "part-03.png")
        flow.snapshot(self.root, "knight", "flat")
        entry = next(s for s in flow.state(self.root, "knight")["stages"] if s["stage"] == "export")
        self.assertEqual(entry["stale"], ["flat"])

    def test_route_errors(self) -> None:
        status, _, _ = factory_http.dispatch(self.factory, "POST", "/export/nobody", {}, b"{}")
        self.assertEqual(status, 404)
        status, _, _ = factory_http.dispatch(self.factory, "POST", "/export/..", {}, b"{}")
        self.assertIn(status, (400, 404))
        self.asset.mkdir(parents=True)
        status, raw, _ = factory_http.dispatch(self.factory, "POST", "/export/knight", {}, b"{}")
        self.assertEqual(status, 400)
        self.assertIn("nothing to export", json.loads(raw)["error"])
        status, _, _ = factory_http.dispatch(self.factory, "GET", "/export/knight", {}, b"")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
