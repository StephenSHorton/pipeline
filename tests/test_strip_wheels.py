"""scripts/strip_wheels.py: finding two tyres under a car's silhouette, and leaving no wheel below the underside."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from strip_wheels import find_wheels, refine  # noqa: E402


def car_mask(h=300, w=600, tyres=((150, 60), (450, 60)), r=40):
    """A box body (y 100..200) on two round tyres whose bottoms hang to y=240."""
    m = np.zeros((h, w), bool)
    m[100:200, 50:550] = True
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, _ in tyres:
        m |= np.hypot(xx - cx, yy - (240 - r)) <= r
    return m


def test_finds_both_tyres_and_the_underside():
    wheels, sill = find_wheels(car_mask())
    assert len(wheels) == 2
    assert abs(sill - 199) <= 1
    for w, cx in zip(wheels, (150, 450)):
        assert abs(w["cx"] - cx) <= 3
        assert abs(w["r"] - 40) <= 6
        assert abs(w["bottom"] - 239) <= 1


def test_refine_finds_the_hub_and_sets_the_radius_from_the_lowest_point():
    m = car_mask()
    rgb = np.full(m.shape + (3,), 90, np.uint8)  # mid-grey body
    yy, xx = np.mgrid[0:300, 0:600]
    rgb[m & (yy >= 200)] = 25  # dark tyres
    rgb[np.hypot(xx - 150, yy - 200) <= 22] = 210  # bright rim
    rgb[np.hypot(xx - 450, yy - 200) <= 22] = 210
    wheels = [refine(rgb, w) for w in find_wheels(m)[0]]
    for w, cx in zip(wheels, (150, 450)):
        assert w["hub"]
        assert abs(w["cx"] - cx) <= 1.5 and abs(w["cy"] - 200) <= 3
        assert abs(w["r"] - 40) <= 3


def test_a_car_with_no_tyres_finds_none():
    m = np.zeros((300, 600), bool)
    m[100:200, 50:550] = True
    wheels, _ = find_wheels(m)
    assert wheels == []
