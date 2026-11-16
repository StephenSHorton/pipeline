#!/usr/bin/env python3
"""Mesh pane game look: low-res pass, iso snap, outlines, shadows, in the standalone viewer's module.

The desktop app's copy went with its retired classic page; the app's port (src/screens/asset/stack/gameLook.ts)
checks its shaders against viewer/pixel-look.js in look.test.ts.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "viewer" / "pixel-look.js").read_text(encoding="utf-8")
PAGES = {
    "viewer": (ROOT / "viewer" / "index.html").read_text(encoding="utf-8"),
}


class PixelLookTests(unittest.TestCase):
    def test_pages_wire_the_look(self) -> None:
        for name, html in PAGES.items():
            with self.subTest(name):
                for el in ("game-look", "game-iso", "game-outline", "game-shadow", "game-dither", "game-px", "game-levels"):
                    self.assertIn(f'id="{el}"', html)
                self.assertIn("createPixelLook({ THREE, renderer, scene, camera, controls, subject: root, key, ground, blob })", html)
                self.assertIn("bindPixelLookUi([pixelLook, stackLab.pixelLook]);", html)
                self.assertIn("pixelLook.render();", html)
                self.assertIn("carveRay.setFromCamera(carveNdc, pixelLook.viewCamera());", html)

    def test_step_bar_pixel_switch_is_bound(self) -> None:
        self.assertIn('"game-look-bar": "on"', JS)

    def test_stack_pane_uses_the_same_look(self) -> None:
        for path in (ROOT / "viewer" / "stack-lab.js",):
            with self.subTest(str(path)):
                js = path.read_text(encoding="utf-8")
                self.assertIn('import { createPixelLook } from "./pixel-look.js";', js)
                self.assertIn("subject: stackGroup", js)
                self.assertIn("gameLook.render();", js)
                self.assertIn("paintRay.setFromCamera(paintNdc, gameLook.viewCamera());", js)
                self.assertIn("pixelLook: gameLook,", js)

    def test_unlit_ground_gets_a_shadow_catcher(self) -> None:
        self.assertIn("ShadowMaterial", JS)
        self.assertIn("isMeshBasicMaterial", JS)

    def test_pass_shape(self) -> None:
        self.assertIn("OrthographicCamera", JS)
        self.assertIn("Math.round(r)", JS)  # whole-texel snap
        self.assertIn("vUv * lowRes + 1.0 + frac", JS)  # remainder shifted back on upscale
        self.assertIn("MeshNormalMaterial", JS)
        self.assertIn("DepthTexture", JS)
        self.assertIn("PCFSoftShadowMap", JS)
        self.assertIn("bayer4", JS)
        self.assertIn("#include <colorspace_fragment>", JS)


if __name__ == "__main__":
    unittest.main()
