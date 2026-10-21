#!/usr/bin/env python3
"""Save inspector paint: stack sheet PNG and/or mesh albedo PNG."""
from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

from factory_env import pick_glb, repo_root
from sprite_stack import composite_preview


def save_stack_sheet(asset: Path, png: bytes) -> dict:
    dest = asset / "stack"
    dest.mkdir(parents=True, exist_ok=True)
    img = Image.open(BytesIO(png)).convert("RGBA")
    img.save(dest / "sheet.png", optimize=True)
    meta_path = dest / "stack.json"
    layers = 1
    res = img.height
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            layers = int(meta.get("layers") or (img.width // max(1, res)))
            res = int(meta.get("resolution") or res)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    else:
        layers = max(1, img.width // max(1, res))
    slices = []
    for i in range(layers):
        box = (i * res, 0, (i + 1) * res, res)
        if box[2] > img.width:
            break
        slices.append(img.crop(box))
    if slices:
        preview = composite_preview([np_rgba(s) for s in slices], spacing=1)
        Image.fromarray(preview, "RGBA").save(dest / "preview.png", optimize=True)
    return {"ok": True, "file": "stack/sheet.png", "layers": len(slices), "resolution": res}


def np_rgba(im: Image.Image):
    import numpy as np

    return np.array(im.convert("RGBA"), dtype="uint8")


def save_mesh_albedo(asset: Path, png: bytes) -> dict:
    import trimesh

    src = None
    for name in ("textured_paint.glb", "textured.glb", "pbr.glb", "textured_photo.glb"):
        cand = asset / name
        if cand.is_file():
            src = cand
            break
    if src is None:
        src = pick_glb(repo_root(), asset.name)
    if src is None:
        raise FileNotFoundError(f"no GLB in {asset}")
    img = Image.open(BytesIO(png)).convert("RGBA")
    loaded = trimesh.load(str(src), force="scene")
    if isinstance(loaded, trimesh.Scene):
        for geom in loaded.geometry.values():
            _apply_tex(geom, img)
        dest = asset / "textured_paint.glb"
        loaded.export(str(dest))
    else:
        _apply_tex(loaded, img)
        dest = asset / "textured_paint.glb"
        loaded.export(str(dest))
    _patch_lods(asset, "textured_paint.glb")
    return {"ok": True, "file": "textured_paint.glb", "source": src.name}


def _apply_tex(mesh, img: Image.Image) -> None:
    vis = getattr(mesh, "visual", None)
    if vis is None:
        return
    mat = getattr(vis, "material", None)
    if mat is not None:
        if hasattr(mat, "baseColorTexture"):
            mat.baseColorTexture = img
        elif hasattr(mat, "image"):
            mat.image = img
    elif hasattr(vis, "material") is False and hasattr(vis, "image"):
        vis.image = img


def _patch_lods(asset: Path, filename: str) -> None:
    path = asset / "lods.json"
    item = {"id": "paint", "file": filename, "label": "Painted"}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"lods": []}
    else:
        data = {"lods": []}
    lods = [x for x in data.get("lods") or [] if x.get("id") != "paint"]
    lods.insert(0, item)
    data["lods"] = lods
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stem", required=True)
    p.add_argument("--kind", choices=("stack", "mesh"), required=True)
    p.add_argument("--png", required=True, type=Path)
    args = p.parse_args()
    asset = root / "out" / args.stem
    data = args.png.read_bytes()
    if args.kind == "stack":
        info = save_stack_sheet(asset, data)
    else:
        info = save_mesh_albedo(asset, data)
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
