#!/usr/bin/env python3
"""Reduce a mesh to a named LOD or an exact face count.

Presets
  lookdev  80,000  paint / UV unwrap
  game     12,000  realtime
  proxy     2,000  collision / thumbnails

`--octree` is documented here so generate.sh and this file share one table;
this script only decimates. Native low-poly generation is hy3d --octree.

Pieces smaller than --min-piece (default 2%) of the surface area are dropped first: shape models leave
floating crumbs, and every triangle spent on one is lost. A small piece that is long (at least MIN_EXTENT of the
mesh's largest dimension, e.g. a lantern's frame rod the shape model left unjoined) is part of the subject, not a
crumb, and stays. A textured mesh keeps a texture: quadric decimation
drops UVs, so the reduced mesh is unwrapped again (xatlas) and its atlas filled from the textured source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

PRESETS = {
    "hero": {"faces": None, "octree": 384, "label": "High"},
    "lookdev": {"faces": 80_000, "octree": 256, "label": "Medium"},
    "game": {"faces": 12_000, "octree": 128, "label": "Game"},
    "proxy": {"faces": 2_000, "octree": 128, "label": "Low"},
}
PRESET_ALIASES = {
    "high": "hero",
    "medium": "lookdev",
    "low": "proxy",
}


MIN_PIECE = 0.02
# A piece this long (share of the mesh's largest bounding-box side) stays at any area from MIN_LONG_AREA up. Measured
# on every saved hero mesh: crumbs reach 0.13 at most; qmv-new-lantern-r2's dropped side rod was 0.38 (1.5% area).
MIN_EXTENT = 0.25
MIN_LONG_AREA = 0.002
# A piece touching the biggest one (within this share of its largest side) is a part of it the generator meshed on
# its own, not a crumb floating off it: TRELLIS.2's pig tail came as five such pieces (0.07% to 0.4% of the area,
# gap 0) and the cut dropped them all. From MIN_TOUCH_AREA up it stays.
TOUCH_GAP = 0.005
MIN_TOUCH_AREA = 0.0005


def drop_small_pieces(mesh: trimesh.Trimesh, share: float = MIN_PIECE) -> tuple[trimesh.Trimesh, dict]:
    """Keep the connected pieces with at least `share` of the total area, or long ones (MIN_EXTENT of the mesh's
    largest side and MIN_LONG_AREA of its area); the biggest one always stays.

    Pieces are found on welded positions, so UV and normal seams do not split a piece; the faces are then
    masked on the original mesh, which keeps its UVs and material."""
    info = {"pieces": 1, "kept": 1, "dropped_faces": 0}
    if share <= 0 or not len(mesh.faces):
        return mesh, info
    welded = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    labels = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=len(welded.faces))
    n = int(labels.max()) + 1 if len(labels) else 0
    info["pieces"] = n
    if n <= 1:
        return mesh, info
    area = np.bincount(labels, weights=welded.area_faces, minlength=n)
    keep = area >= share * area.sum()
    size = float(np.ptp(welded.vertices, axis=0).max()) or 1.0
    corners = welded.vertices[welded.faces]  # (F, 3, 3)
    lo = np.full((n, 3), np.inf)
    hi = np.full((n, 3), -np.inf)
    np.minimum.at(lo, labels, corners.min(1))
    np.maximum.at(hi, labels, corners.max(1))
    long_ = ((hi - lo).max(1) >= MIN_EXTENT * size) & (area >= MIN_LONG_AREA * area.sum())
    info["kept_long"] = int((long_ & ~keep).sum())
    keep |= long_
    big = int(np.argmax(area))
    keep[big] = True
    cand = np.where(~keep & (area >= MIN_TOUCH_AREA * area.sum()))[0]
    if len(cand):
        from scipy.spatial import cKDTree
        main = cKDTree(welded.vertices[np.unique(welded.faces[labels == big])])
        # vertex to vertex, so at least an edge apart: a coarse body's nearest vertex can be that far from the seam
        gap = max(TOUCH_GAP * size, float(welded.edges_unique_length.mean()))
        touch = [c for c in cand if main.query(welded.vertices[np.unique(welded.faces[labels == c])],
                                                distance_upper_bound=gap)[0].min() <= gap]
        keep[touch] = True
        info["kept_touching"] = len(touch)
    mask = keep[labels]
    info["kept"] = int(keep.sum())
    info["dropped_faces"] = int((~mask).sum())
    if mask.all():
        return mesh, info
    out = mesh.copy()
    out.update_faces(mask)
    out.remove_unreferenced_vertices()
    return out, info


def texture_of(mesh: trimesh.Trimesh) -> Image.Image | None:
    vis = getattr(mesh, "visual", None)
    if getattr(vis, "uv", None) is None:
        return None
    mat = getattr(vis, "material", None)
    image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if image is None:
        return None
    return image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))


def rebake(high: trimesh.Trimesh, texture: Image.Image, low: trimesh.Trimesh, atlas: int) -> trimesh.Trimesh:
    """Unwrap the low mesh and fill its atlas with the colour of the nearest point on the textured high mesh."""
    import xatlas

    pack = xatlas.PackOptions()
    pack.resolution = atlas
    pack.padding = 2
    chart = xatlas.ChartOptions()
    chart.max_iterations = 2
    atl = xatlas.Atlas()
    atl.add_mesh(np.asarray(low.vertices, np.float32), np.asarray(low.faces, np.uint32))
    atl.generate(chart, pack)
    vmap, indices, uvs = atl[0]
    verts = np.asarray(low.vertices)[vmap]
    faces = np.asarray(indices, dtype=np.int64)

    # Texel centres inside each triangle, as points on the low mesh.
    px = uvs * atlas
    points, texels = [], []
    for f in faces:
        t = px[f]
        x0, y0 = np.floor(t.min(0)).astype(int)
        x1, y1 = np.ceil(t.max(0)).astype(int)
        xs, ys = np.meshgrid(np.arange(max(x0, 0), min(x1, atlas)), np.arange(max(y0, 0), min(y1, atlas)))
        if xs.size == 0:
            continue
        p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
        a, b, c = t
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        lam = np.linalg.solve(m, (p - a).T).T
        u, w = lam[:, 0], lam[:, 1]
        inside = (u >= -0.02) & (w >= -0.02) & (u + w <= 1.02)
        if not inside.any():
            continue
        bary = np.stack([1 - u - w, u, w], axis=1)[inside]
        points.append(bary @ verts[f])
        texels.append(p[inside].astype(int))
    tex = np.asarray(texture.convert("RGB"), dtype=np.uint8)
    image = np.zeros((atlas, atlas, 3), np.uint8)
    if points:
        points = np.concatenate(points)
        texels = np.concatenate(texels)
        # Dense colour samples of the high surface; nearest one per texel.
        n = int(min(4_000_000, max(500_000, 4 * len(points))))
        spts, sface = trimesh.sample.sample_surface(high, n, seed=0)
        bary = trimesh.triangles.points_to_barycentric(high.triangles[sface], spts)
        huv = np.einsum("ij,ijk->ik", bary, np.asarray(high.visual.uv)[high.faces[sface]])
        th, tw = tex.shape[:2]
        sx = np.clip((huv[:, 0] % 1.0) * tw, 0, tw - 1).astype(int)
        sy = np.clip((1.0 - huv[:, 1] % 1.0) * th, 0, th - 1).astype(int)
        colour = tex[sy, sx]
        _, nearest = cKDTree(spts).query(points, workers=-1)
        filled = np.zeros((atlas, atlas), bool)
        ty = atlas - 1 - texels[:, 1]
        image[ty, texels[:, 0]] = colour[nearest]
        filled[ty, texels[:, 0]] = True
        # Bleed into empty texels so seams never sample black.
        _, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
        image = image[iy, ix]
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(image, "RGB"),
                                                   metallicFactor=0.0, roughnessFactor=1.0)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    return trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)


WELD = 1e-4  # near-duplicate vertices closer than this share of the mesh size are one vertex before decimating
NUDGES = (1e-6, 1e-5, 1e-4, 1e-3)  # retry nudges, as shares of the mesh size; 1e-3 of a 20 in TV is 0.02 in


def simplify(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """Quadric decimation to about target faces. Two things stall the decimator well above target:
    - near-duplicate vertices along creases (the CRT: 379 pairs under 1e-4 of its size) leave sliver triangles its
      flip and degenerate tests refuse to collapse, so it stopped at 650 whether asked for 450 or 340. They are
      welded first.
    - exactly flat areas (a clay cylinder's caps) give every edge zero error. A retry with the vertices nudged
      (seeded, so repeatable) breaks the tie; the nudge grows until the count lands within 1.1x of target (a
      millionth of the size alone never cleared the CRT's slivers), and the closest result is kept."""
    scale = float(mesh.scale or 1.0)
    welded = mesh.copy()
    welded.merge_vertices(digits_vertex=max(0, int(-np.floor(np.log10(WELD * scale)))))
    welded.update_faces(welded.nondegenerate_faces())
    welded.remove_unreferenced_vertices()
    best = welded.simplify_quadric_decimation(face_count=target)
    rng = np.random.default_rng(0)
    for k in NUDGES:
        if len(best.faces) <= 1.1 * target:
            break
        nudged = welded.copy()
        nudged.vertices = welded.vertices + rng.normal(scale=k * scale, size=welded.vertices.shape)
        low = nudged.simplify_quadric_decimation(face_count=target)
        if abs(len(low.faces) - target) < abs(len(best.faces) - target):
            best = low
    return best


def reduce_mesh(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    if target <= 0 or len(mesh.faces) <= target:
        return mesh
    return simplify(mesh, target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--preset", choices=sorted({*PRESETS, *PRESET_ALIASES}), default=None)
    ap.add_argument("--faces", type=int, default=None)
    ap.add_argument("--report", default=None, help="write a JSON sidecar")
    ap.add_argument("--min-piece", type=float, default=MIN_PIECE,
                    help="drop pieces under this share of the surface area (0 keeps every piece)")
    ap.add_argument("--atlas", type=int, default=1024, help="atlas size when a textured mesh is re-baked")
    args = ap.parse_args()

    target = args.faces
    label = "custom"
    preset = PRESET_ALIASES.get(args.preset, args.preset) if args.preset else None
    if preset:
        spec = PRESETS[preset]
        label = spec["label"]
        if target is None:
            target = spec["faces"]
        args.preset = preset

    mesh = trimesh.load(args.inp, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print("reduce: not a triangle mesh", file=sys.stderr)
        return 1

    before_v, before_f = len(mesh.vertices), len(mesh.faces)
    has_uv = bool(getattr(getattr(mesh, "visual", None), "uv", None) is not None)
    mesh, pieces = drop_small_pieces(mesh, args.min_piece)
    if pieces["dropped_faces"]:
        print(f"reduce: dropped {pieces['pieces'] - pieces['kept']} of {pieces['pieces']} pieces "
              f"({pieces['dropped_faces']} faces) under {args.min_piece:.0%} of the area")
    rebaked = False
    if target is None:
        out_mesh = mesh
        target_note = "keep"
    else:
        target = min(target, len(mesh.faces))
        texture = texture_of(mesh)
        out_mesh = reduce_mesh(mesh, target)
        if texture is not None and out_mesh is not mesh and getattr(out_mesh.visual, "uv", None) is None:
            out_mesh = rebake(mesh, texture, out_mesh, args.atlas)
            rebaked = True
        target_note = str(target)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(args.out)
    after_v, after_f = len(out_mesh.vertices), len(out_mesh.faces)
    still_uv = bool(getattr(getattr(out_mesh, "visual", None), "uv", None) is not None)

    report = {
        "ok": True,
        "preset": args.preset,
        "label": label,
        "path": str(Path(args.out)),
        "src": str(Path(args.inp)),
        "before_vertices": before_v,
        "before_faces": before_f,
        "vertices": after_v,
        "faces": after_f,
        "target": target_note,
        "had_uvs": has_uv,
        "has_uvs": still_uv,
        "rebaked": rebaked,
        "pieces": pieces,
    }
    print(
        f"reduce: {label} {before_v}v/{before_f}f -> {after_v}v/{after_f}f "
        f"(target {target_note}) uvs {has_uv}->{still_uv}"
    )
    print(f"reduce: wrote {args.out}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print("REDUCE_JSON " + json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
