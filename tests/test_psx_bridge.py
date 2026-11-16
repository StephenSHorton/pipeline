"""psx_lowpoly.bridge_pieces: a part the cut left floating (the pig's curly tail, two specks beside the rump) is
joined to the body, so a creature that skins as one mesh has one piece."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import psx_lowpoly as P  # noqa: E402
from workbench import eyes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def body_and(offsets: list[float]) -> trimesh.Trimesh:
    body = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    parts = [body]
    for x in offsets:
        s = trimesh.creation.icosphere(subdivisions=0, radius=0.08)
        s.apply_translation([x, 0, 0])
        parts.append(s)
    return trimesh.util.concatenate(parts)


class BridgeTests(unittest.TestCase):
    def test_floating_parts_beside_the_body_are_joined(self) -> None:
        low = body_and([1.12, 1.3])  # 0.12 and 0.3 from the sphere: both within 10% of a 2.6-wide model
        self.assertEqual(eyes.pieces(low), 3)
        out, info = P.bridge_pieces(low)
        self.assertEqual((info["pieces"], info["bridged"], info["left"]), (3, 2, 0))
        self.assertEqual(eyes.pieces(out), 1)
        self.assertEqual(len(out.faces), len(low.faces))

    def test_a_part_too_far_away_stays_loose_and_is_reported(self) -> None:
        low = body_and([1.12, 3.0])
        out, info = P.bridge_pieces(low)
        self.assertEqual((info["bridged"], info["left"]), (1, 1))
        self.assertEqual(eyes.pieces(out), 2)

    def test_one_piece_is_returned_unchanged(self) -> None:
        ball = trimesh.creation.icosphere(subdivisions=2)
        out, info = P.bridge_pieces(ball)
        self.assertEqual((info["pieces"], info["bridged"]), (1, 0))
        np.testing.assert_array_equal(out.faces, ball.faces)


class CutTests(unittest.TestCase):
    """psx_lowpoly.py end to end: a creature's loose part is joined, a prop's (a wheel, a post) is left alone."""

    def cut(self, subject: str) -> tuple[dict, int]:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            body = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
            tail = trimesh.creation.icosphere(subdivisions=2, radius=0.15)
            tail.apply_translation([0.0, 0.0, -0.72])  # a wheel-sized part 0.07 clear of the body
            mesh = trimesh.util.concatenate([body, tail])
            V = np.asarray(mesh.vertices)
            lo, hi = V.min(0), V.max(0)
            uv = np.stack([(V[:, 2] - lo[2]) / (hi[2] - lo[2]), (V[:, 1] - lo[1]) / (hi[1] - lo[1])], axis=1)
            from PIL import Image
            image = Image.fromarray(np.full((16, 16, 3), 180, np.uint8), "RGB")
            src = d / "in.glb"
            trimesh.Trimesh(V, mesh.faces, process=False,
                            visual=trimesh.visual.TextureVisuals(uv=uv, image=image)).export(src, file_type="glb")
            (d / "intent.json").write_text(json.dumps({"subject": subject, "output": "ps1"}))
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "psx_lowpoly.py"), str(src), "--out",
                                   str(d / "o"), "--name", "m", "--height", "30", "--faces", "700", "--atlas", "64",
                                   "--facing-check", "off", "--glb", str(d / "m.glb")],
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
            report = json.loads((d / "o" / "psx.json").read_text())
            return report, eyes.pieces(trimesh.load(d / "m.glb", force="mesh"))

    def test_a_creatures_loose_part_is_joined(self) -> None:
        report, pieces = self.cut("animal")
        self.assertEqual(pieces, 1)
        self.assertGreaterEqual(report["bridged"]["bridged"], 1)

    def test_a_props_separate_part_stays_separate(self) -> None:
        report, pieces = self.cut("prop")
        self.assertNotIn("bridged", report)
        self.assertGreaterEqual(pieces, 2)


if __name__ == "__main__":
    unittest.main()
