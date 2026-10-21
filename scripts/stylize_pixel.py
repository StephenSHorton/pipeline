#!/usr/bin/env python3
"""Commit to low-poly texel art: faceted mesh, tiny atlas, concept palette, nearest.

Takes a painted (or clay) GLB plus the concept still. Writes pixel.glb with a
NEAREST sampler so the viewer crunch-filters instead of smearing Hunyuan soup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import xatlas
from PIL import Image
from pygltflib import GLTF2, Sampler
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


def _as_mesh(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError("not a triangle mesh")
    return obj


def _barycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=1)


def explode_flat(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """One unique vertex per corner so faces don't share normals (true facets)."""
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)[faces.reshape(-1)]
    uvs = np.asarray(mesh.visual.uv)[faces.reshape(-1)]
    new_faces = np.arange(len(verts), dtype=np.int64).reshape((-1, 3))
    out = trimesh.Trimesh(vertices=verts, faces=new_faces, process=False)
    out.visual = TextureVisuals(uv=uvs, material=mesh.visual.material)
    return out


def unwrap(mesh: trimesh.Trimesh, resolution: int = 128) -> trimesh.Trimesh:
    pack = xatlas.PackOptions()
    pack.padding = 3
    pack.bilinear = False
    pack.resolution = int(resolution)
    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.uint32),
    )
    try:
        atlas.generate(pack_options=pack)
    except TypeError:
        atlas.generate()
    vmapping, indices, uvs = atlas.get_mesh(0)
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    uvs = np.asarray(uvs, dtype=np.float32)
    # Pull islands in a texel so nearest filtering doesn't bleed.
    pad = 1.5 / max(int(resolution), 8)
    uvs = uvs * (1.0 - 2 * pad) + pad
    mesh.visual = TextureVisuals(uv=uvs)
    return mesh


def connected_remesh(mesh: trimesh.Trimesh, target_faces: int, voxels: int = 16) -> trimesh.Trimesh:
    """Keep one solid: largest shell, coarse simplify, voxel-fill, marching cubes.

    Quadric-decimating Hunyuan soup to 1.4k in one jump tears slivers and holes.
    """
    m = mesh.copy()
    m.merge_vertices()
    # Do not drop the 600+ Hunyuan shells — voxel occupancy reunites them.
    print(f"stylize_pixel: input {len(m.faces)} faces", flush=True)
    try:
        m.fill_holes()
    except Exception:
        pass
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    voxels = max(8, min(int(voxels), 96))
    # Don't pre-crush the source when the grid is fine enough to use it.
    if voxels < 24 and len(m.faces) > 6000:
        m = m.simplify_quadric_decimation(face_count=6000)
        m.merge_vertices()
        print(f"stylize_pixel: pre-simplify -> {len(m.faces)} faces", flush=True)
    longest = float(np.max(m.extents) or 1.0)
    pitch = longest / voxels
    print(f"stylize_pixel: voxelize pitch={pitch:.4f} voxels={voxels}", flush=True)
    try:
        from scipy.ndimage import binary_closing, binary_dilation
        from trimesh.voxel.base import VoxelGrid

        grid = m.voxelized(pitch)
        try:
            grid = grid.fill()
        except Exception:
            pass
        occ = np.array(grid.matrix, dtype=bool)
        # High-res grids must NOT inherit the 14-voxel close radius — 7 iters at
        # 48 cells melts the extra fidelity back into a potato (11k coplanar tris).
        close_iter = 1 if voxels >= 24 else 2
        occ = binary_closing(occ, iterations=close_iter)
        if voxels < 24:
            occ = binary_dilation(occ, iterations=1)
        grid = VoxelGrid(occ, transform=np.array(grid.transform))
        solid = grid.marching_cubes
        solid.apply_transform(np.array(grid.transform))
        if len(solid.faces) > 32:
            m = solid
            m.merge_vertices()
            print(
                f"stylize_pixel: marching cubes -> {len(m.faces)} faces "
                f"components={len(m.split(only_watertight=False))}",
                flush=True,
            )
    except Exception as e:
        print(f"stylize_pixel: voxel remesh skipped ({e})", flush=True)
    try:
        m.fill_holes()
    except Exception:
        pass
    if target_faces and len(m.faces) > int(target_faces * 1.6):
        m = m.simplify_quadric_decimation(face_count=int(target_faces))
        m.merge_vertices()
        print(f"stylize_pixel: final simplify -> {len(m.faces)} faces", flush=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    if not len(m.faces):
        raise RuntimeError("remesh produced an empty mesh")
    return m


def dilate(img: np.ndarray, filled: np.ndarray, rounds: int = 2) -> np.ndarray:
    out = img.copy()
    mask = filled.copy()
    for _ in range(rounds):
        nxt = out.copy()
        nmask = mask.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(out, (dy, dx), axis=(0, 1))
            smask = np.roll(mask, (dy, dx), axis=(0, 1))
            take = (~mask) & smask
            nxt[take] = shifted[take]
            nmask |= take
        out, mask = nxt, nmask
    return out


def pixelate_concept(photo: Image.Image, px: int = 96) -> Image.Image:
    """Box-downsample the cutout so projection samples are already chunky pixels."""
    rgba = photo.convert("RGBA")
    arr = np.array(rgba)
    ys, xs = np.where(arr[:, :, 3] > 16)
    if len(xs):
        rgba = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = rgba.size
    scale = px / max(w, h)
    nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    return rgba.resize((nw, nh), Image.Resampling.BOX)


def _fill_color(photo: Image.Image) -> np.ndarray:
    arr = np.array(photo.convert("RGBA"))
    rgb = arr[:, :, :3][arr[:, :, 3] > 16]
    rgb = rgb[rgb.mean(axis=1) < 245]
    if len(rgb) == 0:
        return np.array([90, 100, 70], dtype=np.float32)
    return np.median(rgb, axis=0).astype(np.float32)


def look_at(eye, target, up=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up)
    n = np.linalg.norm(s)
    if n < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        s = np.cross(f, up)
        n = np.linalg.norm(s)
    s = s / (n + 1e-8)
    u = np.cross(s, f)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def project_atlas(dest: trimesh.Trimesh, photo: Image.Image, atlas: int) -> np.ndarray:
    """Project a pixelated concept still onto the UV atlas. Unseen = fill olive.

    Does not sample Hunyuan's soup atlas — that was the camo noise.
    """
    src = pixelate_concept(photo, px=max(64, atlas * 3 // 4))
    rgba = np.array(src.convert("RGBA"), dtype=np.float32)
    rgb, alpha = rgba[:, :, :3], rgba[:, :, 3]
    ph, pw = rgb.shape[:2]
    fill = _fill_color(src)
    h = w = atlas
    out = np.broadcast_to(fill, (h, w, 3)).copy()
    verts = np.asarray(dest.vertices, dtype=np.float32)
    uvs = np.asarray(dest.visual.uv, dtype=np.float32)
    center = (verts.min(0) + verts.max(0)) * 0.5
    radius = float(np.linalg.norm(verts - center, axis=1).max() + 1e-6)
    dist = radius / np.tan(np.deg2rad(35.0) / 2.0) * 1.22
    direction = np.array([0.78, 0.42, 1.0], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    eye = center + direction * dist
    view = look_at(eye, center)
    f = 1.0 / np.tan(np.deg2rad(35.0) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f
    proj[1, 1] = f
    proj[2, 2] = -1.0005
    proj[2, 3] = -0.1
    proj[3, 2] = -1.0
    vp = proj @ view
    print(f"stylize_pixel: project {pw}x{ph} concept -> {atlas} atlas", flush=True)
    for i0, i1, i2 in dest.faces:
        tri = verts[[i0, i1, i2]]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nlen = np.linalg.norm(n) + 1e-8
        n = n / nlen
        if np.dot(n, eye - tri.mean(axis=0)) < 0.22:
            continue
        uv = uvs[[i0, i1, i2]]
        px = uv * np.array([w - 1, h - 1], dtype=np.float32)
        px[:, 1] = (h - 1) - px[:, 1]
        mn = np.floor(px.min(0)).astype(int)
        mx = np.ceil(px.max(0)).astype(int)
        x0, y0 = max(mn[0], 0), max(mn[1], 0)
        x1, y1 = min(mx[0], w - 1), min(mx[1], h - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        xx, yy = np.meshgrid(xs, ys)
        p = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
        bary = _barycentric(p, px[0:1], px[1:2], px[2:3])
        inside = bary.min(axis=1) >= -0.01
        if not np.any(inside):
            continue
        bary = bary[inside]
        pos = bary[:, 0:1] * tri[0] + bary[:, 1:2] * tri[1] + bary[:, 2:3] * tri[2]
        homo = np.concatenate([pos, np.ones((len(pos), 1), dtype=np.float32)], 1)
        clip = (vp @ homo.T).T
        ww = clip[:, 3]
        ndc = clip[:, :3] / np.clip(ww[:, None], 1e-5, None)
        sx = ((ndc[:, 0] * 0.5 + 0.5) * (pw - 1)).astype(np.int32)
        sy = ((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (ph - 1)).astype(np.int32)
        ok = (
            (ww > 1e-5)
            & (sx >= 0) & (sx < pw)
            & (sy >= 0) & (sy < ph)
            & (ndc[:, 2] > -1) & (ndc[:, 2] < 1)
        )
        pix = np.stack([xx.ravel()[inside], yy.ravel()[inside]], axis=1)
        for k, (x, y) in enumerate(pix):
            if not ok[k]:
                continue
            if alpha[sy[k], sx[k]] < 16:
                continue
            sample = rgb[sy[k], sx[k]]
            if sample.mean() > 245:
                continue
            out[y, x] = sample
    covered = np.any(np.abs(out - fill) > 1.0, axis=2)
    out = dilate(out, covered, rounds=4)
    return out.clip(0, 255).astype(np.uint8)


def palette_quantize(img: np.ndarray, photo: Image.Image, colors: int) -> np.ndarray:
    arr = np.array(photo.convert("RGBA"))
    opaque = arr[:, :, 3] > 16
    rgb = arr[:, :, :3][opaque]
    rgb = rgb[rgb.mean(axis=1) < 245]
    if len(rgb) < 64:
        src = photo.convert("RGB")
    else:
        n = min(len(rgb), 128 * 128)
        side = max(8, int(np.floor(np.sqrt(n))))
        src = Image.fromarray(rgb[: side * side].reshape(side, side, 3), mode="RGB")
    q = src.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = np.unique(np.array(q.convert("RGB"), dtype=np.int16).reshape(-1, 3), axis=0)
    pal = pal[pal.mean(axis=1) < 250]
    if len(pal) == 0:
        pal = np.array([[80, 90, 60]], dtype=np.int16)
    pix = img.reshape(-1, 3).astype(np.int16)
    d = ((pix[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    nearest = pal[d.argmin(axis=1)]
    return nearest.reshape(img.shape).astype(np.uint8)


def nearest_sampler(path: Path) -> None:
    gltf = GLTF2().load(str(path))
    if not gltf.samplers:
        gltf.samplers = [Sampler(magFilter=9728, minFilter=9728, wrapS=33071, wrapT=33071)]
    else:
        for s in gltf.samplers:
            s.magFilter = 9728
            s.minFilter = 9728
            s.wrapS = 33071
            s.wrapT = 33071
    if gltf.textures:
        for t in gltf.textures:
            t.sampler = 0
    gltf.save(str(path))


def run(args: argparse.Namespace) -> int:
    orig = _as_mesh(trimesh.load(args.mesh))
    dest = connected_remesh(orig, args.faces, voxels=args.voxels)
    photo_path = Path(args.photo) if args.photo else None
    if photo_path is None or not photo_path.is_file():
        raise FileNotFoundError("stylize_pixel needs --photo for the concept still")
    photo = Image.open(photo_path).convert("RGBA")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dest = unwrap(dest, resolution=args.atlas)
    atlas = project_atlas(dest, photo, args.atlas)
    atlas = palette_quantize(atlas, photo, args.colors)
    tex = Image.fromarray(atlas, mode="RGB")
    mat = PBRMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=1.0)
    dest.visual = TextureVisuals(uv=dest.visual.uv, material=mat)
    tex.save(out.with_suffix(".png"))
    dest = explode_flat(dest)
    dest.export(str(out))
    nearest_sampler(out)
    payload = {
        "ok": True,
        "style": "pixel",
        "faces": int(len(dest.faces)),
        "vertices": int(len(dest.vertices)),
        "atlas": args.atlas,
        "colors": args.colors,
        "voxels": args.voxels,
        "watertight": bool(dest.is_watertight),
        "glb": str(args.out),
    }
    Path(args.out).parent.joinpath("pixel.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"PIXEL_RESULT {args.out}")
    print(json.dumps(payload))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesh")
    ap.add_argument("--photo", default=None)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--faces", type=int, default=100000)
    ap.add_argument("--atlas", type=int, default=512)
    ap.add_argument("--colors", type=int, default=0, help="0 = keep Hunyuan colors")
    ap.add_argument("--lock", type=float, default=1.0)
    ap.add_argument("--voxels", type=int, default=96, help="voxels along the longest axis")
    args = ap.parse_args()
    try:
        return run(args)
    except Exception as e:
        print(f"stylize_pixel: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
