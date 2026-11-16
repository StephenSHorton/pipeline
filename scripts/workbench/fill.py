#!/usr/bin/env python3
"""Fill the texels no photo reached with the photo's own material, so no part of a model stays flat base colour.

retexture.py bakes the photo where its fitted camera saw the surface (and, with --mirror, the opposite side or
end); everything else keeps the base atlas, which on a clay asset is one flat colour. A hay bale's photo shows its
round end, so the rolled side came out plain tan: "the texture didn't make it all across it". This pass takes a
swatch of the subject from the photo itself (the most textured, least shadowed square inside its cut-out), makes
it tile, and paints it onto every texel whose coverage (retexture's <out>.cover.png) is below keep:

- triplanar in world space, each axis weighted by |normal|^k, so the swatch runs on across UV seams instead of
  restarting at every chart edge;
- scaled to the photo's own pixel size on the model, so the straw is as fine as it is on the photographed end;
- gained per channel to the mean of the photo texels along the boundary, so the filled side is the same colour
  and brightness as the baked one;
- feathered across the coverage ramp (and a few texels past it), replacing the base colour retexture faded in
  there rather than mixing with it. The feather and the base it replaces come only from the texel's own UV chart:
  charts sit a few texels apart in the atlas, and a neighbour there is some unrelated part of the model. Texels at
  or above keep are left exactly as they were.

With --delight (as the bake had it) the swatch comes from the photo with its lighting taken out (scripts/delight.py),
the same pixels retexture --delight baked, so a shadowed side cannot darken the fill either.

Atlas gutters (texels in no triangle) take the nearest filled colour, so bilinear and mip lookups at seams never
reach the old base. No model is loaded.

    fill.py textured_photo.glb --photo lifted.png -o textured_photo.glb [--cover textured_photo.cover.png] [--delight [1]]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retexture  # noqa: E402
from workbench import mesher  # noqa: E402

KEEP = 0.18       # retexture's --keep: below this the bake left (or faded towards) the base atlas
SHARP = 4.0       # triplanar blend exponent: higher is a crisper hand-off between the three projections
FEATHER = 6.0     # texels past the coverage ramp over which the photo's edge colour still leans into the fill
BAND = 12         # texels of photo along the boundary whose mean the fill is matched to


class FillError(RuntimeError):
    pass


def _albedo(mesh: trimesh.Trimesh) -> np.ndarray:
    mat = mesh.visual.material
    image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if image is None:
        raise FillError("the mesh has no base texture to fill")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def smooth_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals over vertices merged by position. A UV'd mesh splits its vertices along every
    seam; normals of the split copies would each see one side only, and the triplanar weights would jump there."""
    key = np.round(vertices / (np.ptp(vertices, axis=0).max() + 1e-12) * 1e6).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length is twice the area: the weight
    acc = np.zeros((inv.max() + 1, 3))
    for c in range(3):
        np.add.at(acc, inv[faces[:, c]], fn)
    n = acc[inv]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def uv_charts(vertices: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The UV chart of each face: connected components of faces that share a vertex with the same position and UV.

    Keyed by (position, UV) rather than by vertex index, so a mesh stored as a triangle soup still joins up inside
    its charts, while copies split along a seam (same position, different UV) stay apart."""
    if not len(faces):
        return np.zeros(0, np.int64)
    span = np.ptp(vertices, axis=0).max() + 1e-12
    key = np.concatenate([np.round(vertices / span * 1e6), np.round(uvs * 1e6)], axis=1).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    fv = inv.reshape(-1)[faces]
    n = int(fv.max()) + 1
    rows = np.concatenate([fv[:, 0], fv[:, 1]])
    cols = np.concatenate([fv[:, 1], fv[:, 2]])
    _, label = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    return label[fv[:, 0]].astype(np.int64)


def pick_swatch(rgb: np.ndarray, mask: np.ndarray, side: int | None = None,
                shade_rgb: np.ndarray | None = None) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The square of the subject with the most fine detail and the least shadow: (swatch, (x, y, side)).

    Score = local spread of the high-passed luma (straw, not shading) times relative brightness (a shadowed corner
    would darken the whole fill), over windows that lie wholly inside the cut-out. The swatch's slow shading is
    then divided out, so tiling it shows the material and not a repeating light ramp. shade_rgb: the photo as it was
    lit, when rgb is a de-lit copy: shadow is judged on it (a lifted shadow is still sky-lit grey, not straw) while
    the swatch's pixels come from rgb."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise FillError("the photo has no subject pixels to take a swatch from")
    short = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    side = int(side or np.clip(round(short * 0.4), 8, 512))
    side = max(4, min(side, short))
    luma = rgb.mean(axis=2)
    m = mask.astype(np.float32)
    H, W = mask.shape
    while True:
        h = side // 2
        # uniform_filter centres its window at c - side // 2 .. c - side // 2 + side - 1; keep it on the image.
        valid = ndimage.uniform_filter(m, side, mode="constant") > 0.995
        valid[:h] = False
        valid[H - side + h + 1:] = False
        valid[:, :h] = False
        valid[:, W - side + h + 1:] = False
        if valid.any():
            break
        if side <= 4:
            raise FillError("no square of the subject fits inside its cut-out")
        side = max(4, side // 2)  # a cut-out too thin for the square: shrink until one fits
    hp = luma - ndimage.gaussian_filter(luma, side / 6.0)
    # Clip the high-pass so one hard edge (a shadow's border, the cut-out's rim) cannot outscore a field of straw.
    lim = 3.0 * max(float(np.median(np.abs(hp[mask]))), 1e-3)
    hp = np.clip(hp, -lim, lim)
    mean = ndimage.uniform_filter(luma * m, side)
    spread = np.sqrt(np.maximum(ndimage.uniform_filter(hp * hp * m, side) - ndimage.uniform_filter(hp * m, side) ** 2,
                                0.0))
    ref = max(float(np.median(luma[mask])), 1.0)
    # Any shade in the window costs steeply: its dark patch would repeat across the whole fill.
    if shade_rgb is not None:
        lit = shade_rgb.mean(axis=2)
        dark = (lit < 0.5 * max(float(np.median(lit[mask])), 1.0)) & mask
    else:
        dark = (luma < 0.5 * ref) & mask
    shade = ndimage.uniform_filter(dark.astype(np.float32), side)
    score = np.where(valid, spread * np.clip(mean / ref, 0.0, 1.5) * (1.0 - shade) ** 8, -1.0)
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
    y0, x0 = cy - h, cx - h
    sw = rgb[y0:y0 + side, x0:x0 + side].astype(np.float32)
    low = ndimage.gaussian_filter(sw, (side / 3.0, side / 3.0, 0), mode="reflect")
    sw = sw / np.maximum(low, 1.0) * sw.reshape(-1, 3).mean(0)
    return np.clip(sw, 0, 255), (int(x0), int(y0), side)


def tileable(swatch: np.ndarray) -> np.ndarray:
    """Offset-blend: the swatch rolled by half its size wraps without a seam at its borders (they are the swatch's
    middle) but has one through its centre, where the plain swatch has none; cross-fade from one to the other.
    Mirror-padding would also tile, but repeats as a kaleidoscope of diamonds that reads as a pattern, not straw."""
    n, m = swatch.shape[:2]
    rolled = np.roll(swatch, (n // 2, m // 2), axis=(0, 1))
    wy = np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
    wx = np.sin(np.pi * (np.arange(m) + 0.5) / m) ** 2
    w = np.outer(wy, wx)[..., None] if swatch.ndim == 3 else np.outer(wy, wx)
    out = swatch * w + rolled * (1.0 - w)
    # A cross-fade averages two images and flattens their contrast where both weigh in; restore the swatch's spread.
    mu = swatch.reshape(-1, *swatch.shape[2:]).mean(0)
    return np.clip(mu + (out - mu) / np.sqrt(w ** 2 + (1.0 - w) ** 2), 0, 255)


def _bilinear_wrap(tile: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    th, tw = tile.shape[:2]
    x = (u % 1.0) * tw - 0.5
    y = (v % 1.0) * th - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 %= tw
    y0 %= th
    x1 = (x0 + 1) % tw
    y1 = (y0 + 1) % th
    top = tile[y0, x0] * (1 - fx) + tile[y0, x1] * fx
    bot = tile[y1, x0] * (1 - fx) + tile[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def triplanar(tile: np.ndarray, pos: np.ndarray, nrm: np.ndarray, period: float, sharp: float = SHARP) -> np.ndarray:
    """The tile projected along x, y and z in world space (period = the tile's world size), blended by |n|^sharp."""
    w = np.abs(nrm) ** sharp
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    p = pos / period
    # Each axis gets its own offset so the three projections do not line up their repeats.
    cx = _bilinear_wrap(tile, p[:, 2], p[:, 1] + 0.37)
    cy = _bilinear_wrap(tile, p[:, 0] + 0.21, p[:, 2] + 0.53)
    cz = _bilinear_wrap(tile, p[:, 0] + 0.71, p[:, 1])
    return cx * w[:, 0:1] + cy * w[:, 1:2] + cz * w[:, 2:3]


def _load_cover(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != (shape[1], shape[0]):
            im = im.resize((shape[1], shape[0]), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def fill_unseen(glb_in, cover_png, photo_rgba, glb_out, keep: float = KEEP, sharp: float = SHARP,
                delight: float | None = None) -> dict:
    """Paint the photo's swatch onto every texel of glb_in whose coverage is below keep; write glb_out (and its
    .albedo.png and .fill.json beside it). Returns the report that .fill.json holds. delight: the strength the bake
    de-lit the photo with (retexture --delight); the swatch is taken from the same de-lit pixels."""
    glb_in, glb_out = Path(glb_in), Path(glb_out)
    mesh = trimesh.load(glb_in, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or getattr(mesh.visual, "uv", None) is None:
        raise FillError(f"{glb_in.name}: not a triangle mesh with UVs")
    albedo = _albedo(mesh)
    th, tw = albedo.shape[:2]
    if th != tw:
        raise FillError(f"{glb_in.name}: the atlas is {tw}x{th}; fill handles square atlases (as retexture writes)")
    cover = _load_cover(Path(cover_png), (th, tw))
    uvs = np.asarray(mesh.visual.uv, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Every texel inside a UV triangle, with the surface point and smooth normal it shows.
    f, bary, tx, ty = mesher.texels(uvs, faces, tw)
    pos = np.einsum("ni,nij->nj", bary, verts[faces[f]])
    vn = smooth_normals(verts, faces)
    nrm = np.einsum("ni,nij->nj", bary, vn[faces[f]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12

    in_tri = np.zeros((th, tw), bool)
    in_tri[ty, tx] = True
    chart = np.full((th, tw), -1, np.int64)   # -1: a gutter texel, in no chart
    chart[ty, tx] = uv_charts(verts, uvs, faces)[f]
    hit = cover >= keep
    seen = hit[ty, tx]
    before = float(seen.mean()) if len(seen) else 0.0

    rgb, alpha_fit, alpha_bake = retexture.load_photo(photo_rgba, delight)
    # De-lit, the shadow is judged on the photo as it was lit: lifted, it is still not the material.
    shade_rgb = retexture.load_photo(photo_rgba)[0] if delight else None
    swatch, box = pick_swatch(rgb, ndimage.binary_erosion(alpha_bake, iterations=2) & alpha_bake, shade_rgb=shade_rgb)
    tile = tileable(swatch)
    # Photo pixels to metres: the subject's box in the photo against the model's. Rough (the view is oblique), but
    # it keeps the straw at the scale it has on the photographed end rather than a guess.
    ys, xs = np.where(alpha_fit)
    photo_diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    model_diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    period = tile.shape[0] * model_diag / max(photo_diag, 1.0)

    colour = triplanar(tile, pos, nrm, period, sharp)

    # Gain to the photo texels along the boundary (or every photo texel, or the swatch as is when none).
    fillmap = np.zeros((th, tw, 3), np.float32)
    fillmap[ty, tx] = colour
    unseen_map = in_tri & ~hit
    lit = hit & in_tri
    if lit.any():
        # Cast shadow in the photo (under a bale, in a fold) is not the material's colour: leave it out of the match,
        # as the swatch leaves it out, or the whole fill comes out as dark as the shade.
        luma = albedo.mean(-1)
        lit &= luma >= 0.5 * float(np.median(luma[lit]))
    band = lit & ndimage.binary_dilation(unseen_map, iterations=BAND)
    if band.sum() < 16:
        band = lit
    gain = np.ones(3, np.float32)
    if band.sum() >= 16:
        gain = np.clip(albedo[band].mean(0) / np.maximum(fillmap[band].mean(0), 1.0), 0.5, 2.0).astype(np.float32)
    fillmap *= gain

    out = albedo.copy()
    if unseen_map.any():
        # Per chart, inside its own box: in the atlas the nearest texel (or the blur's reach) is often across the
        # 2-4 texel gap xatlas packs charts with, on another part of the model. Leaning on that would paint a
        # stripe of unrelated photo along the seam on the unseen side, the restart-at-every-chart-edge look the
        # triplanar fill exists to avoid.
        base = albedo.copy()          # a chart with no bare texel: the texel itself, as with none anywhere
        edge = np.zeros_like(albedo)
        lean = np.zeros((th, tw), np.float32)
        bare = in_tri & (cover <= 0.0)
        boxes = ndimage.find_objects(chart + 1)  # chart c is label c + 1; gutters (-1) are 0, skipped
        sig = FEATHER / 2.0
        for c in np.unique(chart[unseen_map]):
            cb = boxes[c]
            mine = chart[cb] == c
            need = unseen_map[cb] & mine
            own = albedo[cb]
            # The base under retexture's fade: the nearest texel of this chart the photo did not touch at all.
            b = bare[cb] & mine
            if b.any():
                _, (by, bx) = ndimage.distance_transform_edt(~b, return_indices=True)
                base[cb][need] = own[by[need], bx[need]]
            h = hit[cb] & mine
            if h.any():
                # A few texels past the ramp the photo's edge colour still leans in, so the hand-off is no line.
                # Normalised blur of this chart's photo texels: a smooth colour field, not a nearest copy's streaks.
                hw = ndimage.gaussian_filter(h.astype(np.float32), sig, mode="constant")
                e = np.stack([ndimage.gaussian_filter(own[..., k] * h, sig, mode="constant") for k in range(3)], -1)
                e /= np.maximum(hw, 1e-6)[..., None]
                d = ndimage.distance_transform_edt(~h)   # every photo texel of the chart is inside its box
                lc = 0.6 * np.exp(-0.5 * (d / FEATHER) ** 2) * (hw > 1e-3)
                edge[cb][need] = e[need]
                lean[cb][need] = lc[need]
        t = np.clip(cover / keep, 0.0, 1.0)[..., None]
        # On the ramp the texel is t * photo + (1 - t) * base: swap the base share for the fill.
        mixed = albedo + (1.0 - t) * (fillmap - base)
        lean = lean[..., None] * (t < 1.0)
        mixed = mixed * (1.0 - lean) + edge * lean
        out[unseen_map] = mixed[unseen_map]

    # Gutters: the nearest texel that is in a triangle or baked, so no lookup at a seam reaches the old base.
    done = in_tri | hit
    gutter = ~done
    if gutter.any() and done.any():
        _, (gy, gx) = ndimage.distance_transform_edt(gutter, return_indices=True)
        out[gutter] = out[gy[gutter], gx[gutter]]

    image = Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")
    mesh.visual.material.baseColorTexture = image
    mesh.visual.material.image = image
    glb_out.parent.mkdir(parents=True, exist_ok=True)
    image.save(glb_out.with_suffix(".albedo.png"))
    report = {
        "glb": str(glb_out),
        "photo": str(photo_rgba),
        "keep": keep,
        "texels": int(len(seen)),
        "covered_before": round(before, 4),
        "covered_after": 1.0,  # every texel in a triangle is now the photo or its swatch
        "filled": int((~seen).sum()),
        "swatch": {"x": box[0], "y": box[1], "side": box[2]},
        "period_m": round(period, 4),
        "gain": [round(float(g), 3) for g in gain],
    }
    if delight:
        report["delight"] = float(delight)
    glb_out.with_suffix(".fill.json").write_text(json.dumps(report, indent=2) + "\n")
    # The GLB last, under a hidden name and renamed into place, as retexture does: fill may rewrite its own input.
    tmp = glb_out.with_name(f".{glb_out.name}.tmp")
    mesh.export(tmp, file_type="glb")
    os.replace(tmp, glb_out)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill texels no photo reached with a triplanar swatch of the photo.")
    ap.add_argument("glb", type=Path, help="retexture's output (textured_photo.glb)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="the filled GLB (may be the input)")
    ap.add_argument("--photo", type=Path, required=True, help="the cut-out the bake used (lifted.png)")
    ap.add_argument("--cover", type=Path, default=None, help="retexture's coverage image (default <glb>.cover.png)")
    ap.add_argument("--keep", type=float, default=KEEP, help="fill texels whose coverage is below this (retexture's)")
    ap.add_argument("--sharp", type=float, default=SHARP, help="triplanar blend exponent on |normal|")
    ap.add_argument("--delight", type=float, nargs="?", const=1.0, default=None, metavar="STRENGTH",
                    help="take the swatch from the photo with its lighting taken out, as retexture --delight baked it")
    args = ap.parse_args()
    cover = args.cover or args.glb.with_suffix(".cover.png")
    if not cover.is_file():
        print(f"fill: no coverage image at {cover} (retexture.py writes it)", file=sys.stderr)
        return 1
    try:
        report = fill_unseen(args.glb, cover, args.photo, args.out, args.keep, args.sharp, args.delight)
    except FillError as e:
        print(f"fill: {e}", file=sys.stderr)
        return 1
    print(f"fill: {report['filled']} texels filled ({report['covered_before']*100:.1f}% had the photo); "
          f"swatch {report['swatch']['side']}px, period {report['period_m']} m, gain {report['gain']}")
    print(f"fill: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
