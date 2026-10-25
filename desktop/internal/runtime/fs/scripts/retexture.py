#!/usr/bin/env python3
"""Photo-consistent albedo: fit a camera to each reference view, project into UVs.

Hunyuan/AniGen paint is a hallucination in a fragmented atlas. This pass overwrites
texels the camera actually saw with the input photo. Occluded texels keep the
generated atlas. Multiple --view images (front/back/top) stack. --mirror also bakes a side-on
photo, flipped, onto the far side; with --mirror-ends a front-on (or back-on) one goes onto the opposite end. --first-wins puts the first --view (a real photo) over the
others instead of averaging it in: the rest (drawn views) fill only where it did not reach.

--delight [strength] takes the photo's own lighting out of every --view before it is baked (scripts/delight.py: the
shade lifted toward the lit level), drawn views as well as the photo; off, the bake is as before. The silhouette fit
and the bake mask come from the untouched alpha, so the cameras are the same. Deep shade that needed a lift near
delight's cap (sky-lit, grey-blue once brightened) splats with its trust as weight, so where it is all the photo had
the texel's coverage drops below --keep and fill.py paints the material there instead.

Next to the albedo it writes <out>.cover.png: each texel's photo confidence (splat weight, mirror included, clipped
to 1) as 0..255, so a later pass (workbench/fill.py) knows which texels no photo reached (below --keep * 255).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delight as _delight  # noqa: E402
import seam_bake  # noqa: E402


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


def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / np.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def eye_on_sphere(center, radius, az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    d = np.array(
        [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)],
        dtype=np.float32,
    )
    return center + d * radius


def project(points, vp, w, h):
    homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], 1)
    clip = (vp @ homo.T).T
    wclip = clip[:, 3]
    ok = wclip > 1e-5
    ndc = np.zeros((len(points), 3), dtype=np.float32)
    ndc[ok] = clip[ok, :3] / wclip[ok, None]
    px = ((ndc[:, 0] * 0.5 + 0.5) * (w - 1)).astype(np.int32)
    py = ((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (h - 1)).astype(np.int32)
    inb = ok & (px >= 0) & (px < w) & (py >= 0) & (py < h) & (ndc[:, 2] > -1) & (ndc[:, 2] < 1)
    return px, py, ndc[:, 2], inb


def pack_sil(mask, size=128):
    ys, xs = np.where(mask)
    out = np.zeros((size, size), dtype=np.uint8)
    if len(xs) == 0:
        return out
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return (np.array(im) > 127).astype(np.uint8)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-8)


def sample_surface(mesh, n, with_fid: bool = False):
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=0) if with_fid else trimesh.sample.sample_surface(mesh, n)
    faces = mesh.faces[fid]
    tri = mesh.vertices[faces]
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    v2 = pts - tri[:, 0]
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
    bary = np.stack([u, v, w], axis=1)
    nrm = np.cross(v0, v1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
    if with_fid:
        return pts.astype(np.float32), faces, bary.astype(np.float32), nrm.astype(np.float32), fid
    return pts.astype(np.float32), faces, bary.astype(np.float32), nrm.astype(np.float32)


def aspect(mask) -> float:
    """A silhouette box's width over height."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 1.0
    return float(xs.max() - xs.min() + 1) / float(ys.max() - ys.min() + 1)


def fit_camera(pts, center, radius, ref_sil, fovy=np.deg2rad(40.0), az_hint=None, el_range=None, ref_aspect=None,
               spread=50):
    """(score, az, el, radius). Silhouettes are compared cropped and stretched square, so a camera turned away
    from a side view (the body foreshortened) can score almost as well; ref_aspect (the photo silhouette's
    width / height) scales the score by how well the box shapes agree, which tells those apart."""
    sh = sw = 256
    best = (-1.0, 40.0, 15.0, radius)
    if az_hint is None:
        azs = list(range(0, 360, 20))
    else:
        step = 10 if spread > 20 else 5
        azs = [((az_hint + d) % 360) for d in range(-spread, spread + 1, step)]
    els = el_range if el_range is not None else [-10, 5, 18, 32, 48]
    for az in azs:
        for el in els:
            eye = eye_on_sphere(center, radius, az, el)
            vp = perspective(fovy, 1.0, 0.05, 40.0) @ look_at(eye, center)
            px, py, _, inb = project(pts, vp, sw, sh)
            sil = np.zeros((sh, sw), dtype=np.uint8)
            if inb.any():
                sil[py[inb], px[inb]] = 1
            score = iou(pack_sil(sil), ref_sil)
            if ref_aspect is not None and inb.any():
                score *= min(aspect(sil), ref_aspect) / max(aspect(sil), ref_aspect)
            if score > best[0]:
                best = (score, float(az), float(el), radius)
    az0, el0 = best[1], best[2]
    reach = 16 if spread > 20 else 8
    for az in np.linspace(az0 - reach, az0 + reach, 7):
        for el in np.linspace(el0 - 10, el0 + 10, 5):
            for rad in (radius * 0.85, radius, radius * 1.2):
                eye = eye_on_sphere(center, rad, az, el)
                vp = perspective(fovy, 1.0, 0.05, 40.0) @ look_at(eye, center)
                px, py, _, inb = project(pts, vp, sw, sh)
                sil = np.zeros((sh, sw), dtype=np.uint8)
                if inb.any():
                    sil[py[inb], px[inb]] = 1
                score = iou(pack_sil(sil), ref_sil)
                if ref_aspect is not None and inb.any():
                    score *= min(aspect(sil), ref_aspect) / max(aspect(sil), ref_aspect)
                if score > best[0]:
                    best = (score, float(az), float(el), float(rad))
    return best


def load_photo(path, delight: float | None = None, with_trust: bool = False):
    """(rgb, alpha_fit, alpha_bake) of a cut-out. delight > 0: rgb with the photo's shading taken out (delight.py);
    both masks still come from the photo as it is, so the fit and the bake mask are the same as without.
    with_trust: a fourth item, delight's per-pixel trust (all ones without delight)."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    rgb = arr[:, :, :3].astype(np.float32)
    a = arr[:, :, 3]
    luma = rgb.max(axis=2)
    alpha_fit = a > 32
    solid = a > 220
    # Drop the Vision white halo from the bake, not from the silhouette fit. The halo is a rim: only bright
    # pixels near the cutout's edge go, so a white or bright-yellow subject still bakes its inside.
    ring = max(2, int(round(0.008 * max(a.shape))))
    rim = solid & ~ndimage.binary_erosion(solid, iterations=ring)
    alpha_bake = solid & ~(rim & (luma >= 245))
    trust = np.ones(a.shape, np.float32)
    if delight:
        # After the masks: a lifted shadow on the rim must not read as the white halo dropped above.
        out, _, trust = _delight.delight_full(arr, strength=float(delight), report=False)
        rgb = out[:, :, :3].astype(np.float32)
    if with_trust:
        return rgb, alpha_fit, alpha_bake, trust
    return rgb, alpha_fit, alpha_bake


def register(pts, vp, alpha_fit):
    """Photo pixel coordinates for each point. fit_camera scores silhouettes cropped to their boxes and
    stretched square, so position and per-axis scale are free: map the projected silhouette's box onto the
    photo's alpha box the same way."""
    ph, pw = alpha_fit.shape
    homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], 1)
    clip = (vp @ homo.T).T
    ok = clip[:, 3] > 1e-5
    ndc = np.zeros((len(pts), 3), dtype=np.float32)
    ndc[ok] = clip[ok, :3] / clip[ok, 3:4]
    ok &= (np.abs(ndc) < 1).all(axis=1)
    ys, xs = np.where(alpha_fit)
    nx0, nx1 = ndc[ok, 0].min(), ndc[ok, 0].max()
    ny0, ny1 = ndc[ok, 1].min(), ndc[ok, 1].max()
    fx = (ndc[:, 0] - nx0) / max(nx1 - nx0, 1e-6)
    fy = (ny1 - ndc[:, 1]) / max(ny1 - ny0, 1e-6)
    px = np.round(xs.min() + fx * (xs.max() - xs.min())).astype(np.int32)
    py = np.round(ys.min() + fy * (ys.max() - ys.min())).astype(np.int32)
    inb = ok & (px >= 0) & (px < pw) & (py >= 0) & (py < ph)
    return px, py, ndc[:, 2], inb


def splat(pts, nrm, uv_pts, eye, center, rgb, alpha_fit, alpha_bake, accum, weight, name, trust=None):
    """Project one photo from one camera into the atlas accumulators. Returns visible sample count. trust (per photo
    pixel, 0..1, from delight) scales each sample's weight."""
    th, tw = weight.shape
    ph, pw = rgb.shape[:2]
    vp = perspective(np.deg2rad(40.0), 1.0, 0.05, 40.0) @ look_at(eye, center)
    px, py, depth, inb = register(pts, vp, alpha_fit)
    view_dir = eye - pts
    view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8
    ndotv = np.einsum("ij,ij->i", nrm, view_dir)
    vis_mask = inb & (ndotv > 0.12) & alpha_bake[np.clip(py, 0, ph - 1), np.clip(px, 0, pw - 1)]

    # screen z-buffer at photo resolution, unique winners
    pix = py.astype(np.int64) * pw + px.astype(np.int64)
    # Farthest first: with repeated pixels the last write wins, so the nearest sample keeps the pixel.
    order = np.argsort(depth)[::-1]
    keep = np.full(ph * pw, -1, dtype=np.int32)
    sel = order[vis_mask[order]]
    keep[pix[sel]] = sel.astype(np.int32)
    winners = keep[keep >= 0]
    print(f"retexture: {len(winners)} visible samples from {name}")
    if len(winners) == 0:
        return 0

    tx = np.clip(uv_pts[winners, 0], 0, 1) * (tw - 1)
    ty = np.clip(1.0 - uv_pts[winners, 1], 0, 1) * (th - 1)
    tx = tx.astype(np.int32)
    ty = ty.astype(np.int32)
    conf = np.clip((ndotv[winners] - 0.12) / 0.55, 0.0, 1.0)
    if trust is not None:
        conf = conf * trust[py[winners], px[winners]]
    col = rgb[py[winners], px[winners]]
    for ox in range(-2, 3):
        for oy in range(-2, 3):
            falloff = float(np.exp(-0.35 * (ox * ox + oy * oy)))
            x = np.clip(tx + ox, 0, tw - 1)
            y = np.clip(ty + oy, 0, th - 1)
            w = conf * falloff
            np.add.at(accum, (y, x, 0), col[:, 0] * w)
            np.add.at(accum, (y, x, 1), col[:, 1] * w)
            np.add.at(accum, (y, x, 2), col[:, 2] * w)
            np.add.at(weight, (y, x), w)
    return len(winners)


def splat_colours(uv_pts, col, accum, weight):
    """Splat ready colours (one per surface sample, full weight) into the atlas accumulators (retexture --select)."""
    th, tw = weight.shape
    tx = (np.clip(uv_pts[:, 0], 0, 1) * (tw - 1)).astype(np.int32)
    ty = (np.clip(1.0 - uv_pts[:, 1], 0, 1) * (th - 1)).astype(np.int32)
    for ox in range(-2, 3):
        for oy in range(-2, 3):
            w = float(np.exp(-0.35 * (ox * ox + oy * oy)))
            x = np.clip(tx + ox, 0, tw - 1)
            y = np.clip(ty + oy, 0, th - 1)
            for ch in range(3):
                np.add.at(accum, (y, x, ch), col[:, ch] * w)
            np.add.at(weight, (y, x), w)


def blend(base, accum, weight, keep):
    """Projected colour where weight clears keep, faded into base on the ring below it."""
    out = base.copy()
    projected = accum / np.maximum(weight, 1e-6)[:, :, None]
    hit = weight > keep
    out[hit] = projected[hit]
    # soften the seam
    ring = (weight > 0) & ~hit
    if ring.any():
        t = (weight[ring] / keep)[:, None]
        out[ring] = projected[ring] * t + base[ring] * (1.0 - t)
    return out, hit


# Camera azimuth of each named view relative to the front camera (fit_camera's azimuth, eye_on_sphere).
# 'left' / 'right' follow Hunyuan3D-2mv's views: the subject's nose points toward image left / right. With the
# front camera at F, a head-toward-image-right view is the camera at F - 90 and head-left at F + 90 (checked on
# chicken-canon: lifted.png, head right, fits az 14.7 with the head along +x, so the front camera is at ~105).
# Front and back silhouettes are near mirror images, so a free fit cannot tell them apart; anchor on a side view.
FACING_OFFSET = {"camera": 0.0, "right": -90.0, "away": 180.0, "left": 90.0}


def _thumb(path, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """The cutout cropped to its subject and fitted, aspect kept, centred in a size square: its mask and its colours
    on grey. The aspect is half the clue: a profile is wide where a front view is narrow."""
    rgb, alpha, _ = load_photo(path)
    m = np.zeros((size, size), bool)
    c = np.full((size, size, 3), 128.0, np.float32)
    ys, xs = np.where(alpha)
    if len(xs) == 0:
        return m, c
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    h, w = alpha[box].shape
    k = size / max(h, w)
    th, tw = max(1, round(h * k)), max(1, round(w * k))
    y0, x0 = (size - th) // 2, (size - tw) // 2
    a = np.array(Image.fromarray(alpha[box].astype(np.uint8) * 255).resize((tw, th), Image.BILINEAR)) > 127
    col = np.where(alpha[box][..., None], rgb[box], 128.0).astype(np.uint8)
    m[y0:y0 + th, x0:x0 + tw] = a
    c[y0:y0 + th, x0:x0 + tw] = np.asarray(Image.fromarray(col).resize((tw, th), Image.BILINEAR), np.float32)
    return m, c


def photo_offset(photo, named: list[tuple[str, float]]) -> tuple[float, str]:
    """The offset of the named view (path, offset) the photo looks most like: outline (which way the head points)
    plus colour (a face from a rump, where front and back outlines mirror). The views were drawn from this photo,
    so the nearest one is roughly its angle; the fit then searches within 50 degrees of it."""
    m, c = _thumb(photo)
    best = None
    for path, off in named:
        vm, vc = _thumb(path)
        score = (1.0 - iou(m, vm)) + float(np.abs(c - vc).mean()) / 255.0
        if best is None or score < best[0]:
            best = (score, off, path)
    return best[1], best[2]


def mirror_az(az: float, off: float | None, ends: bool = False) -> float:
    """Azimuth of the camera reflected through the mid-plane that --mirror bakes across (elevation and radius stay).

    Without ends (plain --mirror, as it always was): the glTF z mid-plane for every view, the sagittal plane of a
    subject standing along x. With ends (--mirror-ends): side-on views (offset +-90) still use z; any other view
    the axis-aligned vertical mid-plane most across its line of sight, z when the camera looks mostly along z (az
    near 0 or 180), else x. Reflecting through z maps az to 180 - az, through x to -az."""
    if not ends or (off is not None and abs(off) == 90.0):
        return (180.0 - az) % 360
    a = np.deg2rad(az)
    return (180.0 - az) % 360 if abs(np.cos(a)) >= abs(np.sin(a)) else (-az) % 360


def view_offset(stem: str) -> float | None:
    if "front" in stem:
        return FACING_OFFSET["camera"]
    if "back" in stem:
        return FACING_OFFSET["away"]
    if "handle" in stem or "side" in stem:
        return -90.0
    if "left" in stem:
        return FACING_OFFSET["left"]
    if "right" in stem:
        return FACING_OFFSET["right"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--view", action="append", required=True, help="reference photo (repeatable)")
    ap.add_argument("--samples", type=int, default=280_000)
    ap.add_argument("--keep", type=float, default=0.18, help="leave generated albedo where photo confidence is below this")
    ap.add_argument("--mirror", action="store_true",
                    help="also bake each photo flipped onto the side it cannot see: through the glTF z mid-plane, "
                         "the far side of a side-on one")
    # Its own flag so a plain --mirror bake (generate.sh without --fill photo) stays byte-identical: the end rule
    # reflects a view that looks along x through x, where --mirror alone always reflected through z.
    ap.add_argument("--mirror-ends", action="store_true",
                    help="with --mirror: a view that is not side-on mirrors through the vertical mid-plane across "
                         "its line of sight instead, so a front-on or back-on photo lands on the opposite end")
    ap.add_argument("--erode", type=float, default=0.0,
                    help="pull the bake mask in by this fraction of the photo's longer side (drops cutout halos)")
    ap.add_argument("--first-facing", choices=sorted(FACING_OFFSET), default=None,
                    help="which way the subject faces in the first --view when its name does not say "
                         "(camera = front view; right/left = side-on, head toward image right/left; away = back)")
    ap.add_argument("--front-az", type=float, default=None, metavar="DEG",
                    help="where the mesh's front faces (camera azimuth, degrees) when the shape step knows it "
                         "(mv_orient turned the multi-view shape so its canonical view looks from az 0): the anchor "
                         "is searched around it instead of all the way round")
    ap.add_argument("--first-wins", action="store_true",
                    help="the first --view bakes over the others, which fill only where it did not land "
                         "(a real photo over drawn views); a weak camera fit for it is averaged in as usual")
    ap.add_argument("--delight", type=float, nargs="?", const=1.0, default=None, metavar="STRENGTH",
                    help="take the photo's lighting out of every --view before baking it (delight.py; 1 when given "
                         "bare, 0 or absent: off): its shade comes up to the lit level instead of baking in as a "
                         "dark patch, and shade too deep to lift splats with less weight (fill.py covers it). The "
                         "fit and bake mask use the untouched alpha, so the cameras do not change")
    ap.add_argument("--select", action="store_true",
                    help="photogrammetry bake (scripts/seam_bake.py): each triangle takes the one view that sees it "
                         "best, seams placed where the views agree, every view's colour evened out to the others "
                         "and the seams levelled; the base texture (the paint) takes what no view sees well, "
                         "colour-matched to the views. Off: every view averaged in wherever it lands")
    args = ap.parse_args()
    # Each view's azimuth offset from the front camera (None: unknown). The anchor (fitted freely) is the first
    # side-on view: its silhouette shows which way the head points, while front and back silhouettes are a
    # coin-flip, so a front-first free fit can land on the back. Splats accumulate, so order only moves fit.json.
    # A --first-wins photo of unknown angle is searched around the named view it looks most like (photo_offset):
    # fitted all the way round, a front photo of a left/right symmetric subject matches the back as well.
    offsets = [view_offset(Path(v).stem.lower()) for v in args.view]
    if args.first_facing is not None:
        offsets[0] = FACING_OFFSET[args.first_facing]
    side = next((i for i, o in enumerate(offsets) if o is not None and abs(o) == 90.0), 0)
    order = [side] + [i for i in range(len(args.view)) if i != side]
    top = order.index(0) if args.first_wins else -1  # where the first --view landed in the fit order
    views = [args.view[i] for i in order]
    offsets = [offsets[i] for i in order]
    first_off = offsets[0] or 0.0

    mesh = trimesh.load(args.glb, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print("retexture: not a triangle mesh", file=sys.stderr)
        return 1
    vis = mesh.visual
    if getattr(vis, "uv", None) is None:
        print("retexture: mesh has no UVs", file=sys.stderr)
        return 1
    mat = vis.material
    image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if image is None:
        image = Image.new("RGB", (2048, 2048), (128, 128, 128))
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    albedo = np.array(image.convert("RGB"), dtype=np.float32)
    th, tw = albedo.shape[:2]
    uvs = np.asarray(vis.uv, dtype=np.float32)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    center = verts.mean(0)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    radius0 = diag * 1.45

    print(f"retexture: sampling {args.samples} surface points")
    pts, faces, bary, nrm, fid = sample_surface(mesh, args.samples, with_fid=True)
    uv_pts = (
        bary[:, 0:1] * uvs[faces[:, 0]]
        + bary[:, 1:2] * uvs[faces[:, 1]]
        + bary[:, 2:3] * uvs[faces[:, 2]]
    )

    accum = np.zeros_like(albedo)
    weight = np.zeros((th, tw), dtype=np.float32)
    accum_m = np.zeros_like(albedo)
    weight_m = np.zeros((th, tw), dtype=np.float32)
    accum_t = np.zeros_like(albedo)
    weight_t = np.zeros((th, tw), dtype=np.float32)
    reports = []
    observed = []  # --select: what each view saw, baked together after the loop
    side_on = args.select and args.first_facing in ("right", "left")  # a creature's turnaround
    base_col = seam_bake.sample_atlas(albedo, uv_pts) if args.select else None
    tol = 0.02 * diag

    for i, (view, off) in enumerate(zip(views, offsets)):
        rgb, alpha_fit, alpha_bake, trust = load_photo(view, args.delight, with_trust=True)
        trust = trust if args.delight else None
        ph, pw = rgb.shape[:2]
        if args.erode > 0:
            r = max(1, int(round(args.erode * max(ph, pw))))
            alpha_bake = ndimage.binary_erosion(alpha_bake, iterations=r)
        ref_sil = pack_sil(alpha_fit)
        stem = Path(view).stem.lower()
        az_hint, el_range = None, None
        if not reports and args.front_az is not None:
            # a flat subject's side silhouette fits as well from behind: free, the scarecrow's right view landed at
            # az 75 and its front texture went on its back (front at 175 where the oriented shape's front is 0)
            az_hint = (args.front_az + first_off) % 360
        if reports:
            # Hints are offsets from the FRONT camera, found from the anchor's fit and its known offset
            # (its name, or --first-facing for the pipeline's unnamed lifted.png).
            front_az = (reports[0]["az"] - first_off) % 360
            if off is None and i == top:
                named = [(v, o) for v, o in zip(views, offsets) if o is not None]
                if named:
                    off, like = photo_offset(view, named)
                    print(f"retexture: {Path(view).name} looks most like {Path(like).name}; searched around it")
            if off is not None:
                az_hint = (front_az + off) % 360
        if "top" in stem:
            el_range = [60, 72, 82, 88]
        # --select (drawn turnaround views on a shape the shape step oriented): every view's angle is known up to
        # a few degrees, so the search stays near it and the box shape counts (a 40 degree turn fit 0.87 before)
        tight = args.select and args.front_az is not None and az_hint is not None
        score, az, el, rad = fit_camera(pts, center, radius0, ref_sil, az_hint=az_hint, el_range=el_range,
                                        ref_aspect=aspect(alpha_fit) if args.select else None,
                                        spread=15 if tight else 50)
        print(f"retexture: {Path(view).name}  IoU={score:.3f}  az={az:.1f} el={el:.1f} r={rad:.3f}")
        reports.append({"view": view, "iou": score, "az": az, "el": el, "radius": rad})
        on_top = i == top
        if score < 0.25:
            print("retexture: weak camera fit, still projecting")
            if on_top:
                # A photo the shape does not match would paste a wrong-angle picture over everything.
                print("retexture: weak fit for the first view: averaged in, not on top")
                on_top = False
        if args.first_wins:
            reports[-1]["on_top"] = on_top

        eye = eye_on_sphere(center, rad, az, el)
        if args.select:
            def look(eye_, rgb_, fit_, bake_, trust_, name_, boost):
                vp = perspective(np.deg2rad(40.0), 1.0, 0.05, 40.0) @ look_at(eye_, center)
                px, py, depth, inb = register(pts, vp, fit_)
                seen, ndotv = seam_bake.observe(pts, nrm, px, py, depth, inb, eye_, bake_, tol)
                stray = None
                if os.environ.get("RETEXTURE_ALIGN", "1") != "0":
                    # nudge the picture onto the mesh: the paint (which sits on the mesh exactly) as this camera
                    # sees it is the target, so a drawn ear lands on the mesh's ear, not on the neck behind it
                    front = inb & (ndotv > 0)
                    zb_seen, _ = seam_bake.observe(pts, nrm, px, py, depth, front, eye_, np.ones_like(bake_), tol)
                    ren, rmask = seam_bake.render_base(px, py, zb_seen, base_col, bake_.shape)
                    dy, dx = seam_bake.align(ren, rmask, rgb_, fit_)
                    print(f"retexture: aligned {name_} {seam_bake.align.report}")
                    cy, cx = np.clip(py, 0, ph - 1), np.clip(px, 0, pw - 1)
                    px = np.clip(np.round(px + dx[cy, cx]), 0, pw - 1).astype(np.int32)
                    py = np.clip(np.round(py + dy[cy, cx]), 0, ph - 1).astype(np.int32)
                    seen = zb_seen & (ndotv > seam_bake.ANGLE_MIN) & bake_[py, px]
                    stray = seam_bake.stray_contours(rgb_, fit_, ren, rmask, cx, cy, px, py)
                    print(f"retexture: {int((stray & seen).sum())} samples of {name_} draw a contour the mesh lacks")
                conf = seam_bake.confidence(ndotv, px, py, bake_, seam_bake.fit_quality(score), trust_)
                col = rgb_[np.clip(py, 0, ph - 1), np.clip(px, 0, pw - 1)].astype(np.float64)
                print(f"retexture: {int(seen.sum())} samples seen from {name_}")
                observed.append({"name": name_, "seen": seen, "conf": conf, "colour": col, "boost": boost,
                                 "anchor": boost >= 1.0, "stray": stray})
            look(eye, rgb, alpha_fit, alpha_bake, trust, Path(view).name, 3.0 if on_top else 1.0)
            if side_on and off is not None and abs(off) == 90.0:
                # A side-on creature's side views never put the head where the mesh has it (the pig's ears and
                # eyes 20-40 px off, beyond what align can move): they keep off the head, which the front view
                # and the paint (one set of eyes and ears, on the mesh's own) cover.
                front_dir = eye_on_sphere(center, 1.0, (az - off) % 360, 0.0) - center
                front_dir[1] = 0.0
                front_dir /= np.linalg.norm(front_dir) + 1e-9
                along = (pts - center) @ front_dir
                t = (along - along.min()) / max(float(np.ptp(along)), 1e-9)
                y = (pts[:, 1] - pts[:, 1].min()) / max(float(np.ptp(pts[:, 1])), 1e-9)
                head = (t > 1.0 - seam_bake.HEAD_SHARE) & (y > seam_bake.HEAD_ABOVE)
                observed[-1]["seen"] = observed[-1]["seen"] & ~head
                print(f"retexture: {Path(view).name} keeps off the head ({int(head.sum())} samples)")
            if args.mirror:
                meye = eye_on_sphere(center, rad, mirror_az(az, off, args.mirror_ends), el)
                look(meye, rgb[:, ::-1].copy(), alpha_fit[:, ::-1].copy(), alpha_bake[:, ::-1].copy(),
                     None if trust is None else trust[:, ::-1].copy(), Path(view).name + " (mirrored)", 0.6)
            continue
        acc, wgt = (accum_t, weight_t) if on_top else (accum, weight)
        splat(pts, nrm, uv_pts, eye, center, rgb, alpha_fit, alpha_bake, acc, wgt, Path(view).name, trust)
        if args.mirror:
            # Side-on photo of a near-symmetric subject: the far side is the near side reflected through the
            # sagittal plane (glTF z = centre), so the reflected camera sees it as the photo flipped left-right.
            # A front-on or back-on photo: the far END is the near one reflected through the mid-plane across the
            # line of sight (a bale's two round ends, a crate's front and back), the same flipped photo.
            meye = eye_on_sphere(center, rad, mirror_az(az, off, args.mirror_ends), el)
            splat(pts, nrm, uv_pts, meye, center, rgb[:, ::-1].copy(), alpha_fit[:, ::-1].copy(),
                  alpha_bake[:, ::-1].copy(),
                  accum_m, weight_m, Path(view).name + " (mirrored)",
                  None if trust is None else trust[:, ::-1].copy())

    base = albedo
    if args.select:
        mfaces, nv = seam_bake.merged_faces(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        res = seam_bake.bake(observed, albedo, uv_pts, fid, bary, mfaces, nv)
        print(f"retexture: select {json.dumps(res['report'])}")
        if os.environ.get("RETEXTURE_LABELS"):  # debug: paint each face its label's colour
            pal = np.array([[230, 60, 60], [60, 200, 60], [60, 90, 230], [230, 200, 40], [170, 170, 170],
                            [200, 60, 200], [60, 200, 200]], np.float32)
            res["colour"] = pal[np.minimum(res["label"][fid], len(pal) - 1)]
        splat_colours(uv_pts, res["colour"], accum, weight)
        # texels between samples (an atlas finer than the samples): the colour around them, not the paint
        # underneath, which showed as dark specks
        thin = weight < 4.0 * args.keep
        acc_b = np.stack([ndimage.gaussian_filter(accum[:, :, ch], 2.0) for ch in range(3)], axis=2)
        wgt_b = ndimage.gaussian_filter(weight, 2.0)
        near = thin & (wgt_b > 1e-3)
        accum[near] += acc_b[near] / wgt_b[near, None] * (4.0 * args.keep - weight[near, None])
        weight[near] = 4.0 * args.keep
        base = res["base"]
        # every face has a source now (the paint where no view sees it): the whole atlas counts as covered
        weight_m[:] = 0.0
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_suffix(".select.json").write_text(json.dumps(res["report"], indent=2) + "\n")
    elif args.mirror:
        # Mirrored texels only where the real photo did not land; the real view wins everywhere it reached.
        base, _ = blend(albedo, accum_m, weight_m, args.keep)
    out, hit = blend(base, accum, weight, args.keep)
    if weight_t.any():
        # The first view over everything: the views under it show only where it did not land.
        out, hit_t = blend(out, accum_t, weight_t, args.keep)
        hit |= hit_t

    locked = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    # Confidence the photo (or its mirror) reached each texel: what fill.py reads to find the texels left as base.
    cover = np.maximum(np.maximum(weight, weight_m), weight_t)
    vis.material.baseColorTexture = locked
    vis.material.image = locked
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(args.out).with_suffix(".albedo.png")
    locked.save(sidecar)
    Image.fromarray(np.rint(np.clip(cover, 0.0, 1.0) * 255.0).astype(np.uint8), "L").save(
        Path(args.out).with_suffix(".cover.png"))
    Path(args.out).with_suffix(".fit.json").write_text(json.dumps(reports, indent=2) + "\n")
    # The GLB last, under a hidden name and renamed into place: whoever reads it (generate.sh --mesh, a take) gets
    # the old file or the whole new one, and its sidecars are already there.
    tmp = Path(args.out).with_name(f".{Path(args.out).name}.tmp")
    mesh.export(tmp, file_type=Path(args.out).suffix.lstrip(".").lower() or "glb")
    os.replace(tmp, args.out)
    print(f"retexture: wrote {args.out}  covered {hit.mean()*100:.1f}% of atlas")
    if args.mirror:
        print(f"retexture: with the mirror {(cover > args.keep).mean()*100:.1f}% of atlas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
