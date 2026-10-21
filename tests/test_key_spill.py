#!/usr/bin/env python3
"""key_spill: the magenta key comes out of see-through parts, never out of a subject's own colours."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import key_spill  # noqa: E402

KEY = (166, 25, 106)


def subject(*colours: tuple[int, int, int]) -> np.ndarray:
    """A 10-row strip per colour, all solid, on a transparent ground row."""
    rows = [np.tile(np.array([*c, 255], np.uint8), (10, 20, 1)) for c in colours]
    rows.append(np.zeros((2, 20, 4), np.uint8))
    return np.concatenate(rows)


class KeySpillTests(unittest.TestCase):
    def test_glass_goes_grey_and_brass_red_green_stay(self) -> None:
        a = subject((145, 61, 99), (180, 140, 70), (210, 30, 40), (40, 120, 90), (120, 120, 120))
        out, info = key_spill.despill(a, KEY)
        glass = out[5, 5, :3].astype(int)
        self.assertLess(glass.max() - glass.min(), 12, glass)
        for row in (15, 25, 35, 45):  # brass, a red comb, green feathers, grey
            np.testing.assert_array_equal(out[row, 5], a[row, 5])
        self.assertEqual(info["pixels"], 200)
        np.testing.assert_array_equal(out[-1], a[-1])  # the ground is not touched

    def test_a_key_coloured_subject_is_left_alone(self) -> None:
        a = subject((200, 70, 150), (190, 60, 130), (180, 140, 70))  # a pink flamingo, mostly
        out, info = key_spill.despill(a, KEY)
        np.testing.assert_array_equal(out, a)
        self.assertEqual(info["skipped"], "subject is key-coloured")

    def test_no_hue_key_does_nothing(self) -> None:
        a = subject((145, 61, 99))
        out, info = key_spill.despill(a, (128, 128, 128))
        np.testing.assert_array_equal(out, a)
        self.assertEqual(info["pixels"], 0)


if __name__ == "__main__":
    unittest.main()
