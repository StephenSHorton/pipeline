#!/usr/bin/env python3
"""Mesh clay into a textured GLB the rest of the pipeline accepts.

Surface nets rather than marching cubes: one vertex per surface cell gives an even, quad-like mesh that
decimates cleanly to a PS1 budget, and it vectorises in plain numpy. Vertices are then pulled onto the
true surface along the SDF gradient (the cell average sits slightly inside curves, which would shrink the
volume), and the same gradient gives the normals.

retexture.py and psx_lowpoly.py need UVs and a base texture (retexture refuses a mesh without UVs), so the
mesh is unwrapped with xatlas and each texel takes the colour of the nearest blob at its point on the
surface. xatlas splits vertices along seams; anything per-vertex goes through its vertex map.

    mesher.py clay.json -o clay.glb [--grid 96] [--height 1.2] [--texture 1024]
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench import clay  # noqa: E402

PAD_CELLS = 3


def _grid(blobs, grid: int):
    lo, hi = clay.bounds(blobs)
    h = float((hi - lo).max()) / max(int(grid), 8)
    lo = lo - PAD_CELLS * h
    hi = hi + PAD_CELLS * h
    n = np.ceil((hi - lo) / h).astype(int) + 1
    if any(b.mirror for b in blobs):
        # Lattice symmetric about X = 0, so a mirrored clay meshes to a mirrored mesh, not one a cell off.
        k = int(np.ceil(max(-lo[0], hi[0], 0.0) / h))
        lo[0] = -k * h
        n[0] = 2 * k + 1
    return lo, h, n


def gradient(fn, p: np.ndarray, eps: float) -> np.ndarray:
    e = np.eye(3) * eps
    g = np.stack([fn(p + e[i]) - fn(p - e[i]) for i in range(3)], axis=1) / (2 * eps)
    return g


def surface_nets(fn, lo: np.ndarray, h: float, n: np.ndarray):
    """Vertices (V, 3) and outward triangles (F, 3) of fn's zero set on the grid lo + h * index."""
    axes = [lo[a] + h * np.arange(n[a]) for a in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    d = fn(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)).reshape(tuple(n))
    inside = d < 0
    cells = tuple(n - 1)
    total = np.zeros(cells + (3,))
    count = np.zeros(cells)
    corner = np.stack([X, Y, Z], axis=-1)
    crossings = []
    for a in range(3):
        lo_sl = [slice(None)] * 3
        hi_sl = [slice(None)] * 3
        lo_sl[a] = slice(0, -1)
        hi_sl[a] = slice(1, None)
        d0, d1 = d[tuple(lo_sl)], d[tuple(hi_sl)]
        cross = inside[tuple(lo_sl)] != inside[tuple(hi_sl)]
        t = np.where(cross, d0 / np.where(cross, d0 - d1, 1.0), 0.0)
        pos = corner[tuple(lo_sl)].copy()
        pos[..., a] += t * h
        pos *= cross[..., None]
        crossings.append((a, cross, inside[tuple(lo_sl)]))
        # Each edge touches the four cells around it; sum it into each by slicing the other two axes.
        b, c = (a + 1) % 3, (a + 2) % 3
        for ob in (0, 1):
            for oc in (0, 1):
                sl = [slice(None)] * 3
                sl[b] = slice(ob, ob + cells[b])
                sl[c] = slice(oc, oc + cells[c])
                total += pos[tuple(sl)]
                count += cross[tuple(sl)]
    active = count > 0
    index = np.full(cells, -1, dtype=np.int64)
    index[active] = np.arange(int(active.sum()))
    verts = total[active] / count[active][:, None]

    quads = []
    for a, cross, low_inside in crossings:
        b, c = (a + 1) % 3, (a + 2) % 3
        # Interior edges only; padding keeps the surface off the grid boundary.
        valid = cross.copy()
        for ax in (b, c):
            sl = [slice(None)] * 3
            sl[ax] = 0
            valid[tuple(sl)] = False
            sl[ax] = -1
            valid[tuple(sl)] = False
        e = np.argwhere(valid)
        if not len(e):
            continue
        ring = []
        # Walk (b-1, c-1) -> (b, c-1) -> (b, c) -> (b-1, c): counter-clockwise seen from +a, since e_b x e_c = e_a.
        for ob, oc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            cc = e.copy()
            cc[:, b] += ob
            cc[:, c] += oc
            ring.append(index[cc[:, 0], cc[:, 1], cc[:, 2]])
        q = np.stack(ring, axis=1)
        # Inside at the low end means the surface faces +a; otherwise reverse the ring.
        flip = ~low_inside[e[:, 0], e[:, 1], e[:, 2]]
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    q = np.concatenate(quads) if quads else np.zeros((0, 4), np.int64)
    # Split each quad along its shorter diagonal.
    d02 = np.linalg.norm(verts[q[:, 0]] - verts[q[:, 2]], axis=1)
    d13 = np.linalg.norm(verts[q[:, 1]] - verts[q[:, 3]], axis=1)
    use02 = d02 <= d13
    tris = np.concatenate([
        np.where(use02[:, None], q[:, [0, 1, 2]], q[:, [0, 1, 3]]),
        np.where(use02[:, None], q[:, [0, 2, 3]], q[:, [1, 2, 3]]),
    ])
    return verts, tris


def mesh_clay(blobs, grid: int = 96, project: int = 3) -> trimesh.Trimesh:
    """Untextured mesh of the clay, in authored coordinates, with SDF-gradient vertex normals."""
    if not blobs:
        raise ValueError("no blobs to mesh")
    fn = lambda p: clay.sdf(blobs, p)  # noqa: E731
    lo, h, n = _grid(blobs, grid)
    verts, faces = surface_nets(fn, lo, h, n)
    if not len(faces):
        raise ValueError("clay has no surface (everything carved away, or blobs too small for the grid)")
    eps = h * 0.25
    for _ in range(project):
        g = gradient(fn, verts, eps)
        step = fn(verts)[:, None] * g / np.maximum((g * g).sum(1, keepdims=True), 1e-12)
        # Clamp to a cell so a vertex near a crease never jumps across thin geometry.
        length = np.linalg.norm(step, axis=1, keepdims=True)
        verts = verts - step * np.minimum(1.0, h / np.maximum(length, 1e-12))
    g = gradient(fn, verts, eps)
    normals = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:
        mesh.invert()
    mesh.vertex_normals = normals[_match(mesh.vertices, verts)]
    return mesh


def _match(points: np.ndarray, source: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    return cKDTree(source).query(points)[1]


def unwrap(vertices: np.ndarray, faces: np.ndarray, size: int):
    """xatlas atlas at `size` texels: (vmap, faces, uvs). Face order is kept."""
    pack = xatlas.PackOptions()
    pack.resolution = int(size)
    pack.padding = 2
    chart = xatlas.ChartOptions()
    chart.max_iterations = 2
    atl = xatlas.Atlas()
    atl.add_mesh(np.asarray(vertices, np.float32), np.asarray(faces, np.uint32))
    atl.generate(chart, pack)
    vmap, indices, uvs = atl[0]
    return np.asarray(vmap, np.int64), np.asarray(indices, np.int64), np.asarray(uvs, np.float64)


def texels(uvs: np.ndarray, faces: np.ndarray, size: int, budget: int = 4_000_000):
    """Every texel centre inside a UV triangle: (face index, barycentric (N, 3), column, row).

    Rows count from the top, matching how retexture samples (row = (1 - v) * size).
    """
    px = uvs[faces] * size                      # (F, 3, 2)
    mn = np.floor(px.min(1)).astype(np.int64)
    mx = np.ceil(px.max(1)).astype(np.int64)
    wh = np.maximum(mx - mn, 1)
    order = np.argsort(wh[:, 0] * wh[:, 1], kind="stable")
    a = px[:, 0]
    m = np.stack([px[:, 1] - a, px[:, 2] - a], axis=2)   # columns: b - a, c - a
    det = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]
    ok = np.abs(det) > 1e-12
    inv = np.zeros_like(m)
    inv[ok, 0, 0] = m[ok, 1, 1] / det[ok]
    inv[ok, 0, 1] = -m[ok, 0, 1] / det[ok]
    inv[ok, 1, 0] = -m[ok, 1, 0] / det[ok]
    inv[ok, 1, 1] = m[ok, 0, 0] / det[ok]
    out_f, out_b, out_x, out_y = [], [], [], []
    start = 0
    while start < len(order):
        # Faces sorted by box area, so a chunk's largest box is its last; grow it until the budget is spent.
        end = start + 1
        while end < len(order):
            w, hh = wh[order[end]]
            if (end - start + 1) * w * hh > budget:
                break
            end += 1
        idx = order[start:end]
        idx = idx[ok[idx]]
        start = end
        if not len(idx):
            continue
        W, H = wh[idx].max(0)
        ox, oy = np.meshgrid(np.arange(W), np.arange(H))
        x = mn[idx, 0, None] + ox.ravel()[None]          # (k, W*H)
        y = mn[idx, 1, None] + oy.ravel()[None]
        rel = np.stack([x + 0.5 - a[idx, 0, None], y + 0.5 - a[idx, 1, None]], axis=2)
        lam = np.einsum("kij,knj->kni", inv[idx], rel)
        u, w = lam[..., 0], lam[..., 1]
        # A little slack so edge texels are covered; bilinear lookups at seams then never see the bleed.
        inside = (u >= -0.02) & (w >= -0.02) & (u + w <= 1.02) & (x >= 0) & (x < size) & (y >= 0) & (y < size)
        k, j = np.nonzero(inside)
        uu, ww = np.clip(u[k, j], 0, 1), np.clip(w[k, j], 0, 1)
        s = np.maximum(uu + ww, 1.0)
        uu, ww = uu / s, ww / s
        out_f.append(idx[k])
        out_b.append(np.stack([1 - uu - ww, uu, ww], axis=1))
        out_x.append(x[k, j])
        out_y.append(size - 1 - y[k, j])
    if not out_f:
        z = np.zeros(0, np.int64)
        return z, np.zeros((0, 3)), z, z
    return np.concatenate(out_f), np.concatenate(out_b), np.concatenate(out_x), np.concatenate(out_y)


def bake(uvs: np.ndarray, faces: np.ndarray, size: int, colour) -> Image.Image:
    """Base texture whose texels are colour(face, bary) (sRGB 0..1 or 0..255), bled into the gaps."""
    f, bary, x, y = texels(uvs, faces, size)
    image = np.zeros((size, size, 3), np.uint8)
    filled = np.zeros((size, size), bool)
    if len(f):
        rgb = np.asarray(colour(f, bary), dtype=np.float64)
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        image[y, x] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        filled[y, x] = True
        # Bleed into empty texels so seams and mip levels never sample black.
        _, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
        image = image[iy, ix]
    return Image.fromarray(image, "RGB")


def textured(vertices, faces, normals, size: int, colour) -> trimesh.Trimesh:
    """Unwrap and bake: colour(points (N, 3), face, bary) gives sRGB per texel, in the input's face order."""
    vmap, nfaces, uvs = unwrap(vertices, faces, size)
    verts = np.asarray(vertices)[vmap]
    tri = verts[nfaces]

    def at(f, bary):
        return colour(np.einsum("ni,nij->nj", bary, tri[f]), f, bary)

    image = bake(uvs, nfaces, size, at)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0.0,
                                                   roughnessFactor=1.0)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    out = trimesh.Trimesh(vertices=verts, faces=nfaces, visual=visual, process=False)
    if normals is not None:
        out.vertex_normals = np.asarray(normals)[vmap]
    return out


def normalise_height(mesh: trimesh.Trimesh, height: float) -> float:
    """Scale about the origin so the mesh is `height` tall, then drop it onto y = 0. Returns the scale.

    Scaling about the origin keeps X = 0 (the mirror plane) where it is.
    """
    lo, hi = mesh.bounds
    span = float(hi[1] - lo[1])
    s = float(height) / span if span > 0 else 1.0
    v = np.asarray(mesh.vertices) * s
    v[:, 1] -= v[:, 1].min()
    normals = np.asarray(mesh.vertex_normals).copy()
    mesh.vertices = v
    mesh.vertex_normals = normals
    return s


def clay_glb(blobs, out: str | Path, grid: int = 96, height: float | None = None,
             texture: int = 1024) -> dict:
    mesh = mesh_clay(blobs, grid=grid)
    tex = textured(mesh.vertices, mesh.faces, mesh.vertex_normals, texture,
                   lambda p, f, b: clay.colour_at(blobs, p))
    scale = normalise_height(tex, height) if height else 1.0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tex.export(out)
    lo, hi = tex.bounds
    return {
        "path": str(out),
        "grid": int(grid),
        "vertices": int(len(tex.vertices)),
        "faces": int(len(tex.faces)),
        "volume_m3": round(float(mesh.volume) * scale ** 3, 6),
        "watertight": bool(mesh.is_watertight),
        "bounds": [[round(float(x), 4) for x in lo], [round(float(x), 4) for x in hi]],
        "scale": round(scale, 6),
        "texture": int(texture),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Mesh a clay JSON into a textured GLB.")
    ap.add_argument("clay")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--grid", type=int, default=96, help="cells along the longest side")
    ap.add_argument("--height", type=float, default=None, help="scale to this height (m), feet on y = 0")
    ap.add_argument("--texture", type=int, default=1024)
    args = ap.parse_args()
    try:
        blobs = clay.load(args.clay)
        info = clay_glb(blobs, args.out, grid=args.grid, height=args.height, texture=args.texture)
    except ValueError as exc:
        print(f"mesher: {exc}", file=sys.stderr)
        return 1
    print(f"mesher: wrote {args.out}  {info['faces']} faces")
    print("MESHER_JSON " + json.dumps(info, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
