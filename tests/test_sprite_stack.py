#!/usr/bin/env python3
"""Sprite stack is a factory product: slice a mesh, write the out/<stem>/stack contract."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import sprite_stack  # noqa: E402


def write_box_glb(path: Path) -> None:
    mesh = trimesh.creation.box(extents=(1.2, 2.0, 0.7))
    mesh.visual.face_colors = [40, 160, 70, 255]
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


class SpriteStackTests(unittest.TestCase):
    def test_pick_glb_prefers_textured(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            asset = Path(td)
            write_box_glb(asset / "mesh.glb")
            write_box_glb(asset / "textured.glb")
            picked = sprite_stack.pick_glb(asset)
            self.assertEqual(picked.name, "textured.glb")

    def test_stack_mesh_writes_sheet_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "stack"
            glb = Path(td) / "box.glb"
            write_box_glb(glb)
            mesh = sprite_stack._as_mesh(trimesh.load(str(glb), force="mesh"))
            result = sprite_stack.stack_mesh(mesh, res=24, fill=True)
            result["stem"] = "box"
            result["source"] = "box.glb"
            self.assertGreater(result["layers"], 8)
            self.assertEqual(result["resolution"], 24)
            sprite_stack.write_outputs(result, dest, "box.glb")
            sheet = Image.open(dest / "sheet.png")
            self.assertEqual(sheet.size[1], 24)
            self.assertEqual(sheet.size[0], 24 * result["layers"])
            meta = json.loads((dest / "stack.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["kind"], "stack")
            self.assertEqual(meta["layers"], result["layers"])
            self.assertEqual(meta["defaults"]["texels"], 120)
            self.assertEqual(meta["defaults"]["contrast"], 1.2)
            arr = np.array(sheet)
            self.assertGreater(int((arr[..., 3] > 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
