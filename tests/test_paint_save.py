#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paint_save  # noqa: E402


class PaintSaveTests(unittest.TestCase):
    def test_stack_sheet_writes_preview(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            asset = Path(td)
            res, layers = 8, 4
            sheet = Image.new("RGBA", (res * layers, res), (0, 0, 0, 0))
            for i in range(layers):
                for y in range(res):
                    for x in range(res):
                        sheet.putpixel((i * res + x, y), (10 * i, 80, 40, 255))
            buf = BytesIO()
            sheet.save(buf, format="PNG")
            (asset / "stack").mkdir()
            (asset / "stack" / "stack.json").write_text(
                json.dumps({"layers": layers, "resolution": res}), encoding="utf-8"
            )
            info = paint_save.save_stack_sheet(asset, buf.getvalue())
            self.assertTrue(info["ok"])
            self.assertTrue((asset / "stack" / "sheet.png").is_file())
            self.assertTrue((asset / "stack" / "preview.png").is_file())
            self.assertEqual(info["layers"], layers)

    def test_patch_lods_inserts_paint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            asset = Path(td)
            (asset / "lods.json").write_text(json.dumps({"lods": [{"id": "hero", "file": "mesh.glb"}]}), encoding="utf-8")
            paint_save._patch_lods(asset, "textured_paint.glb")
            data = json.loads((asset / "lods.json").read_text(encoding="utf-8"))
            self.assertEqual(data["lods"][0]["id"], "paint")
            self.assertEqual(data["lods"][1]["id"], "hero")


if __name__ == "__main__":
    unittest.main()
