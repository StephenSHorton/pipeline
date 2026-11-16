#!/usr/bin/env python3
"""scripts/workbench/wrap.py: a striped picture wrapped round a lying ellipsoid puts the stripes along its length
on every side, the far side matching the near side, with the picture's shading taken out."""
from __future__ import annotations

import sys
import unittest

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))

from workbench import wrap  # noqa: E402


def picture(h: int = 200, w: int = 400):
    """An ellipse lying lengthwise with stripes that follow its outline toward the ends, as a melon's do, darker
    toward the bottom."""
    yy, xx = np.mgrid[0:h, 0:w]
    alpha = ((xx - w / 2) / (w * 0.45)) ** 2 + ((yy - h / 2) / (h * 0.45)) ** 2 < 1
    half = h * 0.45 * np.sqrt(np.clip(1 - ((xx - w / 2) / (w * 0.45)) ** 2, 1e-3, 1))
    stripe = np.where(np.floor((yy - h / 2) / half * 5) % 2 == 0, 200.0, 60.0)
    shade = 1.0 - 0.5 * yy / h
    rgb = np.dstack([stripe * shade] * 3)
    return rgb, alpha


class Wrap(unittest.TestCase):
    def ring(self, x: float, n: int = 64) -> np.ndarray:
        """Points round the long axis (z) at one place along it, angles evenly spaced."""
        t = np.linspace(-np.pi, np.pi, n, endpoint=False)
        return np.c_[0.3 * np.cos(t), 0.3 * np.sin(t), np.full(n, x)], t

    def test_colour_does_not_change_along_the_length(self) -> None:
        rgb, alpha = picture()
        ring, _ = self.ring(0.0)
        ends = np.array([[0, 0, -1.0], [0, 0, 1.0]])
        a = wrap.wrap_colours(np.r_[ring, ends], rgb, alpha)[:-2]
        b = wrap.wrap_colours(np.r_[self.ring(0.4)[0], ends], rgb, alpha)[:-2]
        self.assertLess(float(np.abs(a - b).mean()), 8.0)  # stripes run end to end

    def test_far_side_mirrors_the_near_side(self) -> None:
        rgb, alpha = picture()
        ring, t = self.ring(0.0)
        col = wrap.wrap_colours(np.r_[ring, [[0, 0, -1.0], [0, 0, 1.0]]], rgb, alpha)[:-2, 0]
        # the point at angle t and its mirror pi - t take the same colour: no seam on the far side
        mirror = (np.pi - t + np.pi) % (2 * np.pi) - np.pi
        j = np.array([int(np.argmin(np.abs(np.angle(np.exp(1j * (t - m)))))) for m in mirror])
        self.assertLess(float(np.abs(col - col[j]).mean()), 6.0)
        self.assertGreater(float(np.ptp(col)), 60.0)  # the stripes are there, round the ring

    def test_the_shading_is_taken_out(self) -> None:
        rgb, alpha = picture()
        flat = wrap.flat_rows(rgb, alpha)
        rows = np.array([flat[y][alpha[y]].mean() for y in range(40, 160, 40)])
        raw = np.array([rgb[y][alpha[y]].mean() for y in range(40, 160, 40)])
        self.assertLess(float(np.ptp(rows)), float(np.ptp(raw)) * 0.6)


if __name__ == "__main__":
    unittest.main()
