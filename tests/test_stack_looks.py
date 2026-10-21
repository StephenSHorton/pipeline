#!/usr/bin/env python3
"""Viewer ships voxel / plane / contour looks; contour is marching squares, not rounded cubes."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "viewer" / "stack-lab.js").read_text(encoding="utf-8")


class StackLookTests(unittest.TestCase):
    def test_look_buttons_exist(self) -> None:
        for look in ("look-voxel", "look-plane", "look-contour"):
            self.assertIn(f'id="{look}"', HTML)

    def test_default_is_voxel_with_screen_pixel(self) -> None:
        self.assertIn('look: LOOKS.includes(urlLook) ? urlLook : "voxel"', JS)
        self.assertIn("stack-pixel", HTML)
        self.assertIn("fwidth(vMapUv)", JS)
        self.assertIn("smoothstep(vec2(1.0) - filterSize", JS)
        self.assertIn("Parked", HTML)

    def test_round_slider_merges_neighbors(self) -> None:
        self.assertIn('id="round"', HTML)
        self.assertIn("buildMergedVoxelSurface", JS)
        self.assertIn("laplacianSmooth", JS)
        self.assertIn("const ease = amount * amount", JS)
        self.assertIn("squared curve", HTML)
        self.assertNotIn("BoxGeometry(1, 1, 1, 5, 5, 5)", JS)

    def test_pixel_pass_keeps_linear_energy(self) -> None:
        self.assertIn("MeshBasicMaterial", JS)
        self.assertIn("toneMapped: false", JS)
        self.assertIn("diffuseColor.rgb * 0.72", JS)

    def test_main_is_boolean_keep_largest(self) -> None:
        self.assertIn('id="stack-main"', HTML)
        self.assertNotIn('id="islands"', HTML)
        self.assertIn("rebuildIslandMask", JS)
        self.assertIn("state.main", JS)
        self.assertIn("Keep only the largest connected voxel piece", HTML)

    def test_round_defaults_to_045(self) -> None:
        self.assertIn("return 0.45", JS)
        self.assertIn('value="0.45"', HTML)

    def test_quantize_keeps_distinct_hues(self) -> None:
        self.assertIn("srgbToOkLab", JS)
        self.assertIn("peakC", JS)
        self.assertIn("chroma-weighted", HTML)
        self.assertNotIn("Math.random() * N", JS)

    def test_contour_stays_parked_not_product(self) -> None:
        self.assertIn("marchingSquares", JS)
        self.assertIn("bevelEnabled: false", JS)
        self.assertIn("Not the product path", HTML)

    def test_full_orbit(self) -> None:
        self.assertIn("controls.maxPolarAngle = Math.PI;", JS)
        self.assertIn("controls.maxPolarAngle = Math.PI;", HTML)


if __name__ == "__main__":
    unittest.main()
