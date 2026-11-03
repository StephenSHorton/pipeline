#!/usr/bin/env python3
"""The code make's headless export (#360), for real: a page in headless Chrome, its Export pressed, the GLB saved.
Skipped without Playwright, a browser, or the frontend's local Three.js (the test never reaches the network)."""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import code_model  # noqa: E402
import inspect_glb  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script type="importmap">{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script></head>
<body><button id="export">Export GLB</button><script type="module">
import * as THREE from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
const scene = new THREE.Scene();
const mat = new THREE.MeshStandardMaterial({ color: 0x9a6b3f, flatShading: true });
const group = new THREE.Group();           // nested, moved: what a Three.js prop is made of
group.position.set(0, 0.5, 0);
const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat);
box.position.set(0, 0.5, 0);
group.add(box);
scene.add(group);
document.getElementById('export').onclick = () => new GLTFExporter().parse(scene, (buf) => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([buf], { type: 'model/gltf-binary' }));
  a.download = 'model.glb';
  a.click();
}, (e) => console.error(e), { binary: true });
</script></body></html>
"""


def browser_ready() -> str | None:
    if importlib.util.find_spec("playwright") is None:
        return "no playwright (uv pip install playwright)"
    if code_model.local_three() is None:
        return "no desktop/frontend/node_modules/three (the test stays offline)"
    if not (shutil.which("google-chrome") or Path("/Applications/Google Chrome.app").exists()
            or any(Path.home().glob("Library/Caches/ms-playwright/chromium-*"))):
        return "no Chrome or Playwright Chromium"
    return None


@unittest.skipIf(browser_ready(), browser_ready() or "")
class HeadlessExport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_export_saves_the_glb_offline(self) -> None:
        page = self.tmp / "model.html"
        page.write_text(PAGE)
        res = code_model.export(page, self.tmp / "out.glb", self.tmp / "shot.png", timeout_s=30)
        self.assertEqual(res["errors"], [])
        # Three.js came from the local copy, whatever version the page named.
        self.assertIn("build/three.module.js", res["three"]["served"])
        self.assertTrue((self.tmp / "shot.png").is_file())
        self.assertEqual(inspect_glb.inspect(res["glb"])["faces"], 12)
        bounds = trimesh.load(res["glb"], force="mesh").bounds
        self.assertAlmostEqual(float(bounds[0][1]), 0.5, places=4)  # the group's lift and the box's, both applied
        self.assertAlmostEqual(float(bounds[1][1]), 1.5, places=4)

    def test_no_button_is_an_error(self) -> None:
        page = self.tmp / "bare.html"
        page.write_text("<!doctype html><p>nothing to press</p>")
        with self.assertRaises(code_model.CodeModelError) as cm:
            code_model.export(page, self.tmp / "out.glb", timeout_s=3)
        self.assertIn("no Export button", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
