#!/usr/bin/env python3
"""scripts/lods.py: lods.json built from an asset folder (reduce.sh's list) and one entry upserted (the workbench's
psx), in scratch folders."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lods  # noqa: E402


class Lods(unittest.TestCase):
    def setUp(self) -> None:
        self.out = Path(tempfile.mkdtemp(prefix="lods_")) / "bale"
        self.out.mkdir()
        self.addCleanup(shutil.rmtree, self.out.parent, True)

    def touch(self, *names: str) -> None:
        for n in names:
            (self.out / n).parent.mkdir(parents=True, exist_ok=True)
            (self.out / n).write_bytes(b"x" * 100)

    def meta(self, name: str, doc: dict) -> None:
        (self.out / name).parent.mkdir(parents=True, exist_ok=True)
        (self.out / name).write_text(json.dumps(doc))

    def test_build_order_and_metadata(self) -> None:
        self.touch("mesh.glb", "textured_photo.glb", "psx.glb", "textured_game.glb", "mesh_game.glb", "mesh_proxy.glb")
        self.meta("psx/psx.json", {"triangles": 1200, "atlas": 256})
        self.meta("lod_textured_game.json", {"faces": 12000, "vertices": 6845, "has_uvs": False})
        self.meta("lod_game.json", {"faces": 12000, "vertices": 6800, "has_uvs": False})
        self.meta("stack/stack.json", {})
        doc = lods.build(self.out, "bale")
        self.assertEqual([x["id"] for x in doc["lods"]], ["photo", "psx", "hero", "game", "proxy", "stack"])
        self.assertEqual(doc["lods"][1], {"id": "psx", "file": "psx.glb", "label": "PSX", "faces": 1200,
                                          "vertices": None, "has_uvs": True})
        self.assertEqual(doc["lods"][3]["file"], "mesh_game.glb")  # the textured game LOD lost its UVs

    def test_put_after_photo_replacing_in_place(self) -> None:
        self.meta("lods.json", {"stem": "bale", "lods": [
            {"id": "paint", "file": "textured_paint.glb", "label": "Painted"},
            {"id": "photo", "file": "textured_photo.glb", "label": "Photo bake"},
            {"id": "hero", "file": "mesh.glb", "label": "Hero"}]})
        lods.put(self.out, "bale", {"id": "psx", "file": "psx.glb", "label": "PSX", "faces": 1})
        lods.put(self.out, "bale", {"id": "psx", "file": "psx.glb", "label": "PSX", "faces": 2})
        doc = json.loads((self.out / "lods.json").read_text())
        self.assertEqual([x["id"] for x in doc["lods"]], ["paint", "photo", "psx", "hero"])
        self.assertEqual(doc["lods"][2]["faces"], 2)

    def test_put_without_photo_or_file(self) -> None:
        lods.put(self.out, "bale", {"id": "psx", "file": "psx.glb", "label": "PSX"})
        self.assertEqual(json.loads((self.out / "lods.json").read_text()),
                         {"stem": "bale", "lods": [{"id": "psx", "file": "psx.glb", "label": "PSX"}]})
        self.meta("lods.json", {"stem": "bale", "lods": [{"id": "hero", "file": "mesh.glb", "label": "Hero"}]})
        lods.put(self.out, "bale", {"id": "psx", "file": "psx.glb", "label": "PSX"})
        doc = json.loads((self.out / "lods.json").read_text())
        self.assertEqual([x["id"] for x in doc["lods"]], ["psx", "hero"])  # nothing ordered before psx: first

    def test_put_without_lods_json_keeps_the_folders_models(self) -> None:
        # A make that stopped before its lods stage: the photo bake and hero are there, lods.json is not.
        self.touch("textured_photo.glb", "mesh.glb", "psx.glb")
        lods.put(self.out, "bale", {"id": "psx", "file": "psx.glb", "label": "PSX", "faces": 9})
        doc = json.loads((self.out / "lods.json").read_text())
        self.assertEqual([x["id"] for x in doc["lods"]], ["photo", "psx", "hero"])
        self.assertEqual(doc["lods"][1]["faces"], 9)  # the given entry, not build()'s

    def test_cli_is_what_reduce_runs(self) -> None:
        self.assertIn('"$ROOT/scripts/lods.py" "$OUT"', (ROOT / "scripts" / "reduce.sh").read_text())
        self.touch("mesh.glb")
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / "lods.py"), str(self.out)],
                             capture_output=True, text=True, check=True)
        self.assertIn("reduce: wrote", res.stdout)
        self.assertEqual(json.loads((self.out / "lods.json").read_text()),
                         {"stem": "bale", "lods": [{"id": "hero", "file": "mesh.glb", "label": "Hero"}]})


if __name__ == "__main__":
    unittest.main()
