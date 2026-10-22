#!/usr/bin/env python3
"""Give a GLB the UVs and base texture retexture.py needs, without a paint model.

`generate.sh --paint off` skips hy3d paint (a clay asset should never load a model), but retexture refuses a
mesh without UVs and bakes the photo over an existing base texture. This fills that gap: unwrap with xatlas
and bake whatever colour the mesh already has into a base texture.

Colour comes from, in order: the mesh's own texture (sampled through its old UVs on the same triangle),
its vertex colours, its material's base colour, else mid grey. A mesh that already has UVs and a texture
(clay.glb after reduce_mesh, which re-unwraps whatever it decimates) is written back unchanged unless
--force, so the common path costs nothing and loses nothing.

    unwrap_cli.py in.glb -o out.glb [--texture 1024] [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench import mesher  # noqa: E402

GREY = (128, 128, 128)


def _texture(mesh: trimesh.Trimesh) -> Image.Image | None:
    vis = mesh.visual
    if getattr(vis, "uv", None) is None:
        return None
    mat = getattr(vis, "material", None)
    image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if image is None:
        return None
    return image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))


def _flat_colour(mesh: trimesh.Trimesh) -> tuple:
    mat = getattr(mesh.visual, "material", None)
    for attr in ("baseColorFactor", "main_color", "diffuse"):
        c = getattr(mat, attr, None)
        if c is not None:
            c = np.asarray(c, dtype=np.float64)[:3]
            return tuple((c * 255 if c.max() <= 1.0 else c).astype(int))
    return GREY


def colour_source(mesh: trimesh.Trimesh):
    """(name, colour(face, bary) -> sRGB 0..255) for faces in the input mesh's order."""
    faces = np.asarray(mesh.faces)
    image = _texture(mesh)
    if image is not None:
        tex = np.asarray(image.convert("RGB"))
        th, tw = tex.shape[:2]
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)

        def from_texture(f, bary):
            st = np.einsum("ni,nij->nj", bary, uv[faces[f]])
            x = np.clip((st[:, 0] % 1.0) * tw, 0, tw - 1).astype(int)
            y = np.clip((1.0 - st[:, 1] % 1.0) * th, 0, th - 1).astype(int)
            return tex[y, x]
        return "texture", from_texture
    if mesh.visual.kind == "vertex":
        vc = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3]

        def from_vertices(f, bary):
            return np.einsum("ni,nij->nj", bary, vc[faces[f]])
        return "vertex colours", from_vertices
    flat = np.asarray(_flat_colour(mesh), dtype=np.float64)
    return "flat", lambda f, bary: np.broadcast_to(flat, (len(f), 3))


def unwrap_glb(src: str | Path, out: str | Path, texture: int = 1024, force: bool = False) -> dict:
    mesh = trimesh.load(src, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("not a triangle mesh")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if not force and _texture(mesh) is not None:
        if Path(src).resolve() != Path(out).resolve():
            shutil.copyfile(src, out)
        return {"path": str(out), "source": "kept", "faces": int(len(mesh.faces)), "texture": None}
    name, colour = colour_source(mesh)
    normals = np.asarray(mesh.vertex_normals)
    result = mesher.textured(mesh.vertices, mesh.faces, normals, texture, lambda p, f, b: colour(f, b))
    result.export(out)
    return {"path": str(out), "source": name, "faces": int(len(result.faces)), "texture": int(texture)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Unwrap a GLB with xatlas and bake its colour into a base texture.")
    ap.add_argument("glb")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--texture", type=int, default=1024, help="base texture size in texels")
    ap.add_argument("--force", action="store_true", help="unwrap again even if the mesh already has UVs and a texture")
    args = ap.parse_args()
    try:
        info = unwrap_glb(args.glb, args.out, texture=args.texture, force=args.force)
    except ValueError as exc:
        print(f"unwrap: {exc}", file=sys.stderr)
        return 1
    print(f"unwrap: wrote {args.out}  colour from {info['source']}")
    print("UNWRAP_JSON " + json.dumps(info, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
