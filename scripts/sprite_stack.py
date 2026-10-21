#!/usr/bin/env python3
"""Cut a textured GLB into a horizontal sprite stack.

Each layer is a top-down XZ slab of the mesh, rasterized into a square PNG
strip (bottom → top). Interior pixels on a slice are filled from the nearest
surface texel so the stack reads as volume, not a hollow rind.

Writes:
  out/<stem>/stack/sheet.png
  out/<stem>/stack/preview.png
  out/<stem>/stack/stack.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt

GLB_CANDIDATES = (
    "textured_photo.glb",
    "pbr.glb",
    "textured.glb",
    "pixel.glb",
    "mesh.glb",
    "mesh_hero.glb",
    "source.glb",
)

FACTORY_DEFAULTS = {
    "texels": 120,
    "spacing": 1,
    "height": 1,
    "brightness": 1.0,
    "contrast": 1.2,
    "saturation": 1.2,
}


def pick_glb(asset: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        glb = explicit.expanduser().resolve()
        if not glb.is_file():
            raise FileNotFoundError(glb)
        return glb
    for name in GLB_CANDIDATES:
        p = asset / name
        if p.is_file() and p.stat().st_size >= 64:
            return p
    raise FileNotFoundError(f"no mesh GLB in {asset}")


def _as_mesh(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Scene):
        parts = []
        for name, geom in obj.geometry.items():
            g = geom.copy()
            try:
                t, _ = obj.graph.get(name)
                g.apply_transform(t)
            except Exception:
                pass
            parts.append(g)
        obj = trimesh.util.concatenate(parts)
    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"not a triangle mesh: {type(obj)}")
    return obj


def _albedo(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    vis = mesh.visual
    mat = getattr(vis, "material", None)
    factor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    if mat is not None:
        raw = getattr(mat, "baseColorFactor", None)
        if raw is not None:
            factor[: min(4, len(raw))] = np.clip(np.asarray(raw, dtype=np.float32)[:4], 0, 1)

    img = None
    if mat is not None:
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None:
        img = getattr(vis, "image", None)

    uv = None
    if hasattr(vis, "uv") and vis.uv is not None:
        uv = np.asarray(vis.uv, dtype=np.float32)

    if img is None:
        rgb = (factor[:3] * 255).astype(np.uint8)
        solid = np.zeros((4, 4, 4), dtype=np.uint8)
        solid[..., :3] = rgb
        solid[..., 3] = 255
        return solid, None, factor

    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    return rgba, uv, factor


def _sample_tex(tex: np.ndarray, uv: np.ndarray, factor: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
    y = np.clip(((1.0 - v) * (h - 1)).astype(np.int32), 0, h - 1)
    px = tex[y, x].astype(np.float32)
    px[:, :3] *= factor[:3]
    px[:, 3] *= factor[3]
    return np.clip(px, 0, 255).astype(np.uint8)


def _stamp(volume: np.ndarray, occ: np.ndarray, ijk: np.ndarray, colors: np.ndarray) -> None:
    ny, nz, nx, _ = volume.shape
    x = ijk[:, 0]
    y = ijk[:, 1]
    z = ijk[:, 2]
    keep = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & (z >= 0) & (z < nz)
    if not np.any(keep):
        return
    x, y, z = x[keep], y[keep], z[keep]
    c = colors[keep]
    volume[y, z, x] = c
    occ[y, z, x] = True


def rasterize(
    mesh: trimesh.Trimesh,
    tex: np.ndarray,
    uv: np.ndarray | None,
    factor: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ny, nz, nx = shape
    volume = np.zeros((ny, nz, nx, 4), dtype=np.uint8)
    occ = np.zeros((ny, nz, nx), dtype=bool)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    fallback = (factor[:4] * 255.0).astype(np.uint8)
    fallback[3] = 255

    for face in faces:
        p0, p1, p2 = verts[face]
        e = max(
            float(np.linalg.norm(p1 - p0)),
            float(np.linalg.norm(p2 - p1)),
            float(np.linalg.norm(p0 - p2)),
        )
        n = int(np.ceil(e / max(pitch * 0.42, 1e-6)))
        n = max(2, min(n, 80))
        s, t = np.mgrid[0 : n + 1, 0 : n + 1]
        mask = (s + t) <= n
        a = s[mask].astype(np.float32) / n
        b = t[mask].astype(np.float32) / n
        c = 1.0 - a - b
        pts = a[:, None] * p0 + b[:, None] * p1 + c[:, None] * p2
        ijk = np.floor((pts - origin) / pitch).astype(np.int32)
        if uv is not None:
            uv0, uv1, uv2 = uv[face]
            uvs = a[:, None] * uv0 + b[:, None] * uv1 + c[:, None] * uv2
            colors = _sample_tex(tex, uvs, factor)
        else:
            colors = np.repeat(fallback[None, :], len(pts), axis=0)
        _stamp(volume, occ, ijk, colors)
    return volume, occ


def fill_slices(volume: np.ndarray, occ: np.ndarray) -> np.ndarray:
    ny = volume.shape[0]
    filled_any = 0
    for y in range(ny):
        slice_occ = occ[y]
        if not slice_occ.any():
            continue
        closed = binary_closing(slice_occ, iterations=1)
        filled = binary_fill_holes(closed)
        rgba = volume[y]
        need = filled & (rgba[:, :, 3] == 0)
        if need.any() and slice_occ.any():
            _, idx = distance_transform_edt(~slice_occ, return_indices=True)
            rgba[need] = rgba[idx[0][need], idx[1][need]]
            rgba[need, 3] = 255
        rgba[~filled] = 0
        volume[y] = rgba
        filled_any += 1
    return volume


def composite_preview(slices: list[np.ndarray], spacing: int = 1) -> np.ndarray:
    res = slices[0].shape[0]
    n = len(slices)
    h = res + spacing * (n - 1)
    w = res
    out = np.zeros((h, w, 4), dtype=np.float32)
    for i, sl in enumerate(slices):
        y0 = h - res - i * spacing
        src = sl.astype(np.float32)
        a = src[:, :, 3:4] / 255.0
        dst = out[y0 : y0 + res]
        out[y0 : y0 + res] = src * a + dst * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def stack_mesh(mesh: trimesh.Trimesh, res: int, fill: bool) -> dict:
    tex, uv, factor = _albedo(mesh)
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    extents = bounds[1] - bounds[0]
    cx = float((bounds[0, 0] + bounds[1, 0]) * 0.5)
    cz = float((bounds[0, 2] + bounds[1, 2]) * 0.5)
    pad = 1.10
    half = float(max(extents[0], extents[2]) * 0.5 * pad)
    pitch = (2.0 * half) / float(res)
    origin = np.array(
        [cx - half, float(bounds[0, 1]) - pitch * 0.25, cz - half],
        dtype=np.float32,
    )
    ny = int(np.ceil((extents[1] + pitch * 0.5) / pitch)) + 1
    ny = max(8, ny)
    print(f"sprite_stack: res={res} layers={ny} pitch={pitch:.5f} faces={len(mesh.faces)}", flush=True)
    volume, occ = rasterize(mesh, tex, uv, factor, origin, pitch, (ny, res, res))
    surface_voxels = int(occ.sum())
    print(f"sprite_stack: surface voxels={surface_voxels}", flush=True)
    if fill:
        volume = fill_slices(volume, occ)
    slices = [volume[y] for y in range(ny)]
    while len(slices) > 2 and slices[0][:, :, 3].max() == 0:
        slices.pop(0)
        origin[1] += pitch
    while len(slices) > 2 and slices[-1][:, :, 3].max() == 0:
        slices.pop()
    return {
        "slices": slices,
        "pitch": float(pitch),
        "origin": origin.tolist(),
        "resolution": int(res),
        "layers": len(slices),
        "bounds": bounds.tolist(),
        "extents": extents.tolist(),
        "surface_voxels": surface_voxels,
        "filled": bool(fill),
    }


def write_outputs(result: dict, dest: Path, source_name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    slices: list[np.ndarray] = result["slices"]
    res = result["resolution"]
    n = len(slices)
    sheet = np.zeros((res, res * n, 4), dtype=np.uint8)
    for i, sl in enumerate(slices):
        sheet[:, i * res : (i + 1) * res] = sl
    Image.fromarray(sheet, "RGBA").save(dest / "sheet.png", optimize=True)
    preview = composite_preview(slices, spacing=1)
    Image.fromarray(preview, "RGBA").save(dest / "preview.png", optimize=True)
    meta = {
        "name": source_name,
        "stem": result.get("stem"),
        "source": result.get("source"),
        "sheet": "sheet.png",
        "preview": "preview.png",
        "resolution": res,
        "layers": n,
        "pitch": result["pitch"],
        "origin": result["origin"],
        "bounds": result["bounds"],
        "extents": result["extents"],
        "filled": result["filled"],
        "surface_voxels": result["surface_voxels"],
        "axis": {"right": "x", "up": "y", "forward": "z", "slice": "y"},
        "sheet_layout": "horizontal-bottom-to-top",
        "defaults": dict(FACTORY_DEFAULTS),
        "kind": "stack",
    }
    (dest / "stack.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"sprite_stack: wrote {n} layers -> {dest}", flush=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glb", type=Path, default=None, help="source GLB (else pick from out/<stem>/)")
    p.add_argument("--stem", required=True, help="out/<stem>/ folder")
    p.add_argument("--res", type=int, default=160, help="XZ pixel resolution")
    p.add_argument("--no-fill", action="store_true", help="keep hollow surface voxels only")
    args = p.parse_args()
    asset = root / "out" / args.stem
    asset.mkdir(parents=True, exist_ok=True)
    try:
        glb = pick_glb(asset, args.glb)
    except FileNotFoundError as e:
        print(f"sprite_stack: {e}", file=sys.stderr)
        return 2
    print(f"sprite_stack: source {glb}", flush=True)
    mesh = _as_mesh(trimesh.load(str(glb), force="mesh"))
    result = stack_mesh(mesh, res=max(16, int(args.res)), fill=not args.no_fill)
    try:
        rel = str(glb.relative_to(asset)).replace("\\", "/")
    except ValueError:
        rel = glb.name
    result["stem"] = args.stem
    result["source"] = rel
    write_outputs(result, asset / "stack", glb.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
