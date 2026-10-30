"""psx_lowpoly.close_up: the cut's holes capped and its faces turned outward (TRELLIS.2's pig had open hoof
soles and ears, and faces wound both ways)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import psx_lowpoly as P  # noqa: E402


class CloseUpTests(unittest.TestCase):
    def test_a_hole_is_capped_and_an_inside_out_piece_turned(self) -> None:
        ball = trimesh.creation.icosphere(subdivisions=3)
        top = (ball.faces == int(np.argmax(ball.vertices[:, 1]))).any(axis=1)  # the fan round the top vertex
        holed = trimesh.Trimesh(ball.vertices, ball.faces[~top][:, ::-1])  # a hole at the top, wound inward
        out, info = P.close_up(holed)
        self.assertEqual(info["holes"], 1)
        self.assertEqual(info["flipped_pieces"], 1)
        self.assertTrue(out.is_watertight)
        self.assertGreater(out.volume, 0.0)

    def test_a_closed_outward_mesh_is_left_alone(self) -> None:
        ball = trimesh.creation.icosphere(subdivisions=2)
        out, info = P.close_up(ball)
        self.assertEqual((info["holes"], info["flipped_pieces"]), (0, 0))
        self.assertEqual(len(out.faces), len(ball.faces))
        self.assertGreater(out.volume, 0.0)


if __name__ == "__main__":
    unittest.main()
