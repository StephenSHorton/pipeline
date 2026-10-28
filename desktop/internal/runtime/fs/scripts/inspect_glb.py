#!/usr/bin/env python3
"""Inspect a GLB and print JSON. Exit 0 only if it has a mesh with faces."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def inspect(path: str) -> dict:
    p = Path(path)
    out: dict = {
        "path": str(p),
        "exists": p.is_file(),
        "bytes": p.stat().st_size if p.is_file() else 0,
        "ok": False,
        "meshes": 0,
        "primitives": 0,
        "vertices": 0,
        "faces": 0,
        "materials": 0,
        "textures": 0,
        "has_uvs": False,
        "has_base_color_texture": False,
        "has_metallic_roughness_texture": False,
        "errors": [],
    }
    if not p.is_file():
        out["errors"].append("file not found")
        return out
    if out["bytes"] < 64:
        out["errors"].append("file too small to be a GLB")
        return out
    try:
        from pygltflib import GLTF2
    except ImportError:
        out["errors"].append("pygltflib not installed")
        return out

    try:
        gltf = GLTF2().load(path)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"parse failed: {e}")
        return out

    meshes = gltf.meshes or []
    accessors = gltf.accessors or []
    materials = gltf.materials or []
    textures = gltf.textures or []
    out["meshes"] = len(meshes)
    out["materials"] = len(materials)
    out["textures"] = len(textures)

    verts = 0
    faces = 0
    prims = 0
    has_uvs = False
    for mesh in meshes:
        for prim in mesh.primitives or []:
            prims += 1
            attrs = prim.attributes
            pos = getattr(attrs, "POSITION", None)
            if pos is not None and 0 <= pos < len(accessors):
                verts += accessors[pos].count or 0
            tex = getattr(attrs, "TEXCOORD_0", None)
            if tex is not None:
                has_uvs = True
            indices = prim.indices
            if indices is not None and 0 <= indices < len(accessors):
                count = accessors[indices].count or 0
                faces += count // 3
    out["primitives"] = prims
    out["vertices"] = verts
    out["faces"] = faces
    out["has_uvs"] = has_uvs

    for mat in materials:
        pbr = getattr(mat, "pbrMetallicRoughness", None)
        if pbr is None:
            continue
        if getattr(pbr, "baseColorTexture", None) is not None:
            out["has_base_color_texture"] = True
        if getattr(pbr, "metallicRoughnessTexture", None) is not None:
            out["has_metallic_roughness_texture"] = True

    if verts > 0 and faces > 0:
        out["ok"] = True
    else:
        out["errors"].append("no triangulated mesh")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: inspect_glb.py <file.glb> [out.json]", file=sys.stderr)
        return 2
    report = inspect(sys.argv[1])
    text = json.dumps(report, indent=2)
    print(text)
    if len(sys.argv) >= 3:
        Path(sys.argv[2]).write_text(text + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
