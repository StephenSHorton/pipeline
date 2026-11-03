#!/usr/bin/env python3
"""GLBs shaped like a Three.js export (#360): parts nested under transformed groups, meshes named apart from their
nodes, and some primitives with no index buffer."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_mesh  # noqa: E402
import inspect_glb  # noqa: E402
import sprite_stack  # noqa: E402


def nested_scene() -> trimesh.Scene:
    """A unit box under a group moved 3 up, and another under a group moved 5 along x and turned; neither mesh
    shares its node's name."""
    scene = trimesh.Scene()
    up = trimesh.transformations.translation_matrix([0, 3, 0])
    side = trimesh.transformations.translation_matrix([5, 0, 0]) @ trimesh.transformations.rotation_matrix(
        np.pi / 2, [0, 1, 0])
    scene.graph.update(frame_to="lift", frame_from=scene.graph.base_frame, matrix=up)
    scene.graph.update(frame_to="swing", frame_from=scene.graph.base_frame, matrix=side)
    scene.add_geometry(trimesh.creation.box((1, 1, 1)), geom_name="BoxGeometry_1", node_name="crate",
                       parent_node_name="lift")
    scene.add_geometry(trimesh.creation.box((2, 1, 1)), geom_name="BoxGeometry_2", node_name="plank",
                       parent_node_name="swing")
    return scene


class NestedNodes(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.glb = self.tmp / "nested.glb"
        nested_scene().export(self.glb)
        self.want = trimesh.load(self.glb, force="mesh").bounds

    def test_reference_places_both_parts(self) -> None:
        # The crate's top at y 3.5; the plank turned onto z, so it spans z -1..1 at x 5.
        np.testing.assert_allclose(self.want, [[-0.5, -0.5, -1.0], [5.5, 3.5, 1.0]], atol=1e-6)

    def test_as_mesh_applies_each_node_transform(self) -> None:
        for mod in (clean_mesh, sprite_stack):
            with self.subTest(mod.__name__):
                got = mod._as_mesh(trimesh.load(self.glb, process=False))
                np.testing.assert_allclose(got.bounds, self.want, atol=1e-6)
                self.assertEqual(len(got.faces), 24)


def glb_bytes(indexed: bool) -> bytes:
    """One triangle, with or without an index buffer."""
    import json

    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32).tobytes()
    idx = np.array([0, 1, 2, 0], np.uint16).tobytes()  # padded to 4 bytes
    buf = pos + (idx if indexed else b"")
    prim = {"attributes": {"POSITION": 0}, "mode": 4}
    accessors = [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0, 0, 0], "max": [1, 1, 0]}]
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)}]
    if indexed:
        prim["indices"] = 1
        accessors.append({"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"})
        views.append({"buffer": 0, "byteOffset": len(pos), "byteLength": 6})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [prim]}], "accessors": accessors, "bufferViews": views,
           "buffers": [{"byteLength": len(buf)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(buf), 0x004E4942) + buf
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


class InspectCounts(unittest.TestCase):
    def test_non_indexed_primitive_counts_its_triangles(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        for indexed in (True, False):
            with self.subTest(indexed=indexed):
                p = tmp / f"tri_{indexed}.glb"
                p.write_bytes(glb_bytes(indexed))
                out = inspect_glb.inspect(str(p))
                self.assertTrue(out["ok"], out["errors"])
                self.assertEqual(out["faces"], 1)


if __name__ == "__main__":
    unittest.main()
