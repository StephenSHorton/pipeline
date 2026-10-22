#!/usr/bin/env python3
"""The workbench's eyes: look at a GLB, measure what is wrong with it, compare it with its reference photo.

    .venv/bin/python scripts/workbench/eyes.py look out/S/psx.glb -o sheet.png [--views side,front,top] [--size 384]
    .venv/bin/python scripts/workbench/eyes.py measure out/S/psx.glb
    .venv/bin/python scripts/workbench/eyes.py compare out/S/psx.glb out/S/lifted.png --facing right -o cmp.png

A single photo leaves Hunyuan to guess depth, and the guesses fail in a few repeatable ways: a head that sits off the
body's centre line, a left side that isn't the right side mirrored, a round prop that comes out a rounded box. Each
of those is a number here, so a fix can be checked by measuring again instead of by eye.

- look: textured renders from fixed ortho cameras (rig_render's views and contact sheet).
- measure: bounds, triangles, pieces, head offset and yaw, mirror IoU, circularity. Everything is measured on
  area-weighted surface samples, not vertices: PSX meshes are unwelded, flat-shaded triangle soup, and their vertices
  bunch where the triangles are small. Silhouettes are rasterised triangles. The body frame (forward, lateral,
  centre line) is rest_check's, with the head end taken from the pipeline's facing (glTF +Z forward) rather than
  rest_check's highest-end rule, which picks the rump of a grazing sheep or pig.
- compare_reference: fits retexture's camera to the photo's silhouette, renders the model's silhouette from it, and
  reports the IoU plus the regions the model is missing or has extra.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mv_orient  # noqa: E402
import rest_check  # noqa: E402
import retexture  # noqa: E402
import rig_render  # noqa: E402

UP = rest_check.UP
FORWARD = np.array([0.0, 0.0, 1.0])  # psx --facing and rig_render: the creature faces glTF +Z
LOOK_VIEWS = ("side", "xside", "front", "back", "top", "q")
HEAD = 0.15  # the front share of the body length that counts as the head
SAMPLES = 200_000
RES = 256
FOV = np.deg2rad(40.0)  # retexture's camera


def load(glb) -> trimesh.Trimesh:
    """Rest-pose world-space triangles of every mesh in a GLB (skinned or static), as rig_render sees them."""
    it = rig_render.Item(Path(glb))
    Vs, _ = it.pose(None, 0.0)
    if not Vs:
        raise ValueError(f"{glb}: no triangle meshes")
    F, off = [], 0
    for m, V in zip(it.meshes, Vs):
        F.append(m["F"] + off)
        off += len(V)
    return trimesh.Trimesh(np.concatenate(Vs), np.concatenate(F), process=False)


def samples(mesh: trimesh.Trimesh, n: int = SAMPLES) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.asarray(pts, float)


# ---------------------------------------------------------------- look

def look(glb, views=LOOK_VIEWS, size: int = 384, out=None) -> str:
    """Contact sheet of textured renders, three tiles a row. Returns the sheet's path."""
    views = list(views)
    out = Path(out) if out else Path(tempfile.mkdtemp(prefix="eyes_")) / f"{Path(glb).stem}_look.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    tiles = rig_render.textured_views(glb, views, size, out.parent / f"{out.stem}_views")
    cols = min(3, len(views))
    rig_render.contact_sheet([tiles[v] for v in views], -(-len(views) // cols), cols, captions=views, out=out)
    return str(out)


# ---------------------------------------------------------------- measure

def pieces(mesh: trimesh.Trimesh) -> int:
    """Connected parts, with vertices welded by position first (otherwise every PSX triangle is its own piece)."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if not len(F):
        return 0
    tol = max(float(np.linalg.norm(np.ptp(V, axis=0))), 1e-9) * 1e-6
    _, ids = np.unique(np.round(V / tol).astype(np.int64), axis=0, return_inverse=True)
    G = ids.ravel()[F]
    a = np.concatenate([G[:, 0], G[:, 1], G[:, 2]])
    b = np.concatenate([G[:, 1], G[:, 2], G[:, 0]])
    n = int(G.max()) + 1
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    return int(len(np.unique(lab[np.unique(G)])))


def body_line(P: np.ndarray, toward: np.ndarray | None = FORWARD) -> dict:
    """rest_check's frame plus the barrel's centre line (lateral = k[0] * forward + k[1], forward about c)."""
    c, fwd, lat, f = rest_check._frame(P, toward)
    for i in range(4):
        body = rest_check._centres(P, f)[4:12]
        k = np.polyfit((body - c) @ fwd, body @ lat, 1)
        if i == 3:
            break
        # A head held to one side tilts the principal axis off the barrel, and slices square to a tilted axis cut
        # the barrel obliquely, so their centres understate the tilt. Turn forward onto the fitted line and refit.
        fwd = fwd + k[0] * lat
        fwd /= np.linalg.norm(fwd)
        lat = np.cross(UP, fwd)
        f = (P - c) @ fwd
    return {"c": c, "fwd": fwd, "lat": lat, "k": k}


def body_coords(X: np.ndarray, fr: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each point's offset from the centre line and its forward coordinate."""
    f = (X - fr["c"]) @ fr["fwd"]
    return X @ fr["lat"] - np.polyval(fr["k"], f), f


def head_offset(P: np.ndarray, fr: dict) -> tuple[float, str]:
    """Sideways offset of the head (the front HEAD share of the length, upper 60%) from the body's centre line, as %
    of the barrel's width, and which side of the creature it sits on (lateral +: its left, glTF +X facing +Z)."""
    off, f = body_coords(P, fr)
    h = P @ UP
    L = float(np.ptp(f))
    head = f >= f.max() - HEAD * L
    head &= h >= np.quantile(h[head], 0.4)
    slab = (f >= f.min() + 0.2 * L) & (f <= f.min() + 0.65 * L)  # rest_check's body slices 4..12
    lo, hi = np.percentile(off[slab], [1, 99])
    pct = float(off[head].mean() / max(hi - lo, 1e-9) * 100)
    return abs(pct), "left" if pct > 0 else "right"


def silhouette(mesh: trimesh.Trimesh, u: np.ndarray, v: np.ndarray, centre_u: float | None = None,
               res: int = RES) -> np.ndarray:
    """Orthographic silhouette of the triangles with vertex image coordinates (u right, v up), aspect kept. With
    centre_u, that u lands exactly on the image's vertical mid-line, so a left-right flip mirrors about it.
    Triangles, not surface samples (mv_orient.silhouette): a sparse cap or a thin part never leaves holes."""
    u0 = (u.min() + u.max()) / 2 if centre_u is None else centre_u
    v0 = (v.min() + v.max()) / 2
    span = max(2 * np.abs(u - u0).max(), np.ptp(v), 1e-9) / 0.96
    x = res / 2 + (u - u0) / span * res
    y = res / 2 - (v - v0) / span * res
    face, _ = rig_render.raster(np.stack([x, y, np.zeros_like(x)], 1), np.asarray(mesh.faces), res)
    return ndimage.binary_fill_holes(ndimage.binary_closing(face >= 0))


def mirror_iou(mesh: trimesh.Trimesh, fr: dict) -> float:
    """Top-view silhouette in the body frame against its own flip about the centre line."""
    off, f = body_coords(np.asarray(mesh.vertices, float), fr)
    m = silhouette(mesh, off, f, centre_u=0.0)
    return mv_orient.iou(m, m[:, ::-1])


def ellipse_fill(mask: np.ndarray) -> float:
    """Silhouette area over the area of its enclosing ellipse: the moment ellipse's shape, grown until it holds every
    pixel. A disc scores ~1, a square ~0.64 (a moment-matched ellipse alone would score a square 0.95)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0.0
    mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ys, xs = np.nonzero(mask)
    X = np.stack([xs, ys], 1).astype(float)
    X -= X.mean(0)
    cov = np.cov(X.T) + np.eye(2) * 1e-9
    d2 = np.einsum("ij,jk,ik->i", X, np.linalg.inv(cov), X)
    # Pixel centres sit half a pixel inside the edge; grow the bound by that so a disc doesn't score above 1.
    r = np.sqrt(d2.max()) + 0.5 / np.sqrt(np.linalg.eigvalsh(cov).min())
    return float(min(1.0, len(xs) / (np.pi * r * r * np.sqrt(np.linalg.det(cov)))))


def circularity(mesh: trimesh.Trimesh, fr: dict) -> tuple[float, str]:
    """The best ellipse fill of the top view and the two end views (along the body's forward and lateral axes), so a
    log, a wheel or a bale on its side counts as well as one standing up."""
    V = np.asarray(mesh.vertices, float)
    fwd, lat = fr["fwd"], fr["lat"]
    views = {"top": (V @ lat, V @ fwd), "end": (V @ lat, V @ UP), "side": (V @ fwd, V @ UP)}
    best = ("top", 0.0)
    for name, (u, v) in views.items():
        score = ellipse_fill(silhouette(mesh, u, v))
        if score > best[1]:
            best = (name, score)
    return round(best[1], 4), best[0]


def measure(glb, n: int = SAMPLES, toward: np.ndarray | None = FORWARD) -> dict:
    """Bounds, triangles, pieces, head offset (% of body width) and yaw, mirror IoU, circularity. The head numbers
    are None, with head_reason, for an upright mesh (taller than long), where rest_check doesn't apply. toward
    None finds the head end by rest_check's highest-end rule instead of the +Z facing."""
    mesh = load(glb)
    V = np.asarray(mesh.vertices)
    P = samples(mesh, n)
    fr = body_line(P, toward)
    out = {"bounds": {"min": np.round(V.min(0), 4).tolist(), "max": np.round(V.max(0), 4).tolist(),
                      "size": np.round(np.ptp(V, axis=0), 4).tolist()},
           "tris": int(len(mesh.faces)), "pieces": pieces(mesh)}
    rest = rest_check.measure(P[:: max(1, len(P) // 20_000)], toward)
    if rest is None:
        out.update(head_offset_pct=None, head_offset_side=None, head_yaw_deg=None,
                   head_reason="upright (taller than long): no body axis along the ground to measure the head against")
    else:
        pct, side = head_offset(P, fr)
        out.update(head_offset_pct=round(pct, 1), head_offset_side=side, head_yaw_deg=rest["head_yaw_deg"])
    circ, view = circularity(mesh, fr)
    out.update(mirror_iou=round(mirror_iou(mesh, fr), 4), circularity=circ, circularity_view=view)
    return out


# ---------------------------------------------------------------- compare_reference

def subject_mask(path) -> np.ndarray:
    """The subject: alpha where the image has it, else everything unlike the corners' flat background."""
    m = mv_orient.image_mask(Path(path))
    if m.mean() > 0.98:  # an opaque RGBA (a photo saved as PNG): fall back to the background rule
        rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
        ground = np.median(np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]), 0)
        m = np.abs(rgb - ground).sum(-1) > 60
    return m


def _shrink(mask: np.ndarray, res: int) -> tuple[np.ndarray, float]:
    s = res / max(mask.shape)
    if s >= 1:
        return mask, 1.0
    w, h = max(1, round(mask.shape[1] * s)), max(1, round(mask.shape[0] * s))
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)
    return np.asarray(im) > 127, s


def _ndc(X: np.ndarray, vp: np.ndarray) -> np.ndarray:
    homo = np.concatenate([X, np.ones((len(X), 1))], 1)
    clip = (vp @ homo.T).T
    return clip[:, :2] / np.maximum(clip[:, 3:4], 1e-6)


def _align(mesh: trimesh.Trimesh, ndc: np.ndarray, ref: np.ndarray, steps: int = 7) -> tuple[float, np.ndarray, tuple]:
    """Place the projected model on the photo with one scale and a shift (pixel = t + k * ndc, y flipped) that
    maximise the silhouette IoU, starting from the boxes matched. Unlike retexture.register's per-axis stretch, a
    missing part shows as missing instead of the rest stretching to fill its box. The model is rasterised once at
    twice the photo's scale; each placement only resamples that."""
    ys, xs = np.nonzero(ref)
    rw, rh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    lo, hi = ndc.min(0), ndc.max(0)
    k0 = float(np.sqrt(rw / max(hi[0] - lo[0], 1e-9) * rh / max(hi[1] - lo[1], 1e-9)))
    tx0 = (xs.min() + xs.max() + 1) / 2 - k0 * (lo[0] + hi[0]) / 2
    ty0 = (ys.min() + ys.max() + 1) / 2 + k0 * (lo[1] + hi[1]) / 2

    K, pad = 2 * k0, 4
    x, y = (ndc[:, 0] - lo[0]) * K + pad, (hi[1] - ndc[:, 1]) * K + pad
    size = int(np.ceil(max(x.max(), y.max()))) + pad
    face, _ = rig_render.raster(np.stack([x, y, np.zeros_like(x)], 1), np.asarray(mesh.faces), size)
    M = ndimage.binary_fill_holes(ndimage.binary_closing(face >= 0, iterations=2))

    def place(k, tx, ty):
        # photo pixel centre (X + 0.5) -> ndc -> M's pixel centre (m + 0.5) -> M's index m
        s = K / k
        off = ((pad - 0.5) - K * (tx / k + lo[0]) + 0.5 * s, (pad - 0.5) + K * (hi[1] - ty / k) + 0.5 * s)
        return ndimage.affine_transform(M, [s, s], offset=(off[1], off[0]), output_shape=ref.shape, order=0,
                                        cval=0) > 0

    best = (-1.0, None, None)
    d = 0.12 * max(rw, rh)
    for k in k0 * np.linspace(0.85, 1.15, steps):
        for dx in np.linspace(-d, d, steps):
            for dy in np.linspace(-d, d, steps):
                m = place(k, tx0 + dx, ty0 + dy)
                score = mv_orient.iou(m, ref)
                if score > best[0]:
                    best = (score, m, (float(k), float(tx0 + dx), float(ty0 + dy)))
    return best


def compare_reference(glb, image, facing: str | None = None, out=None, n: int = SAMPLES, res: int = RES,
                      min_area_pct: float = 1.0) -> dict:
    """IoU of the model's silhouette against the photo's, from retexture's fitted camera, plus the regions the model
    is missing (in the photo, not the model) or has extra. facing is retexture's --first-facing: camera (a front
    view), right / left (side-on, head toward image right / left) or away, for a model facing glTF +Z; None fits the
    camera freely. bbox is in photo pixels; area_pct is % of the photo's subject area; world_hint is where the region
    lies on the model (the mean of the model's surface there for extra, the photo point carried onto the plane
    through the model's centre for missing). camera says which camera won: retexture's fit or the facing's own."""
    mesh = load(glb)
    V = np.asarray(mesh.vertices, float)
    full = subject_mask(image)
    ref, s = _shrink(full, res)
    if not ref.any():
        raise ValueError(f"{image}: no subject found")
    centre = V.mean(0).astype(np.float32)
    radius = float(np.linalg.norm(np.ptp(V, axis=0))) * 1.45
    hint = None if facing is None else retexture.FACING_OFFSET[facing] % 360
    pts = samples(mesh, min(n, 20_000)).astype(np.float32)
    fit_iou, az, el, rad = retexture.fit_camera(pts, centre, radius, retexture.pack_sil(full), az_hint=hint)
    # fit_camera scores silhouettes stretched to their boxes, so a model missing a head or a leg (a different box)
    # can win at the wrong angle. Score the facing's own angle too, aligned without stretching, and keep the better.
    cams = [("fitted", az, el, rad)]
    if hint is not None:
        cams += [("facing", hint, el, rad), ("facing", hint, 5.0, rad)]
    best = None
    for cam in cams:
        view = retexture.look_at(retexture.eye_on_sphere(centre, cam[3], cam[1], cam[2]), centre)
        vp = retexture.perspective(FOV, 1.0, 0.05, 40.0) @ view
        score, model, place = _align(mesh, _ndc(V, vp), ref)
        if best is None or score > best[0]:
            best = (score, model, place, cam, view, vp)
    score, model, (k, tx, ty), (source, az, el, rad), view, vp = best
    half = rad * np.tan(FOV / 2)

    def on_plane(cx, cy):
        """Photo pixel -> NDC -> the plane through the model's centre, square to the camera."""
        nx, ny = (cx + 0.5 - tx) / k, (ty - cy - 0.5) / k
        return centre + view[0, :3] * nx * half + view[1, :3] * ny * half

    P = samples(mesh, n)
    q = _ndc(P, vp)
    ppx, ppy = np.floor(tx + k * q[:, 0]).astype(int), np.floor(ty - k * q[:, 1]).astype(int)
    inb = (ppx >= 0) & (ppx < ref.shape[1]) & (ppy >= 0) & (ppy < ref.shape[0])
    regions = []
    ref_area = int(ref.sum())
    for kind, diff in (("missing", ref & ~model), ("extra", model & ~ref)):
        lab, cnt = ndimage.label(ndimage.binary_opening(diff))
        for i in range(1, cnt + 1):
            r = lab == i
            area = float(r.sum() / ref_area * 100)
            if area < min_area_pct:
                continue
            ry, rx = np.nonzero(r)
            where = on_plane(rx.mean(), ry.mean())
            if kind == "extra":
                hit = inb.copy()
                hit[inb] = r[ppy[inb], ppx[inb]]
                if hit.any():
                    where = P[hit].mean(0)
            regions.append({"kind": kind, "bbox": [int(rx.min() / s), int(ry.min() / s), int((rx.max() + 1) / s),
                                                   int((ry.max() + 1) / s)],
                            "area_pct": round(area, 2), "world_hint": np.round(where, 4).tolist()})
    regions.sort(key=lambda g: -g["area_pct"])

    out = Path(out) if out else Path(tempfile.mkdtemp(prefix="eyes_")) / f"{Path(glb).stem}_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    photo = np.asarray(Image.open(image).convert("RGB").resize(ref.shape[::-1], Image.BILINEAR)).copy()
    photo[~ref] = rig_render.BG
    sil = np.full(ref.shape + (3,), rig_render.BG, np.uint8)
    sil[model] = (90, 90, 90)
    lap = np.full(ref.shape + (3,), 255, np.uint8)
    lap[ref & model] = (150, 150, 150)
    lap[ref & ~model] = (220, 40, 40)
    lap[model & ~ref] = (40, 90, 230)
    tiles = [Image.fromarray(a) for a in (photo, sil, lap)]
    sq = max(ref.shape)
    tiles = [_square(t, sq) for t in tiles]
    rig_render.contact_sheet(tiles, 1, 3, captions=["reference", "model", "red missing / blue extra"], out=out)
    return {"iou": round(score, 4), "regions": regions, "png": str(out),
            "camera": {"source": source, "fit_iou": round(float(fit_iou), 4), "az": round(float(az), 1),
                       "el": round(float(el), 1), "radius": round(float(rad), 4)}}


def _square(im: Image.Image, side: int) -> Image.Image:
    S = Image.new("RGB", (side, side), (255, 255, 255))
    S.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
    return S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("look")
    a.add_argument("glb")
    a.add_argument("-o", "--out")
    a.add_argument("--views", default=",".join(LOOK_VIEWS))
    a.add_argument("--size", type=int, default=384)
    a = sub.add_parser("measure")
    a.add_argument("glb")
    a.add_argument("--auto-head", action="store_true",
                   help="find the head end by rest_check's highest-end rule, not the +Z facing")
    a = sub.add_parser("compare")
    a.add_argument("glb")
    a.add_argument("image")
    a.add_argument("--facing", choices=sorted(retexture.FACING_OFFSET))
    a.add_argument("-o", "--out")
    args = ap.parse_args()
    if args.cmd == "look":
        print(look(args.glb, args.views.split(","), args.size, args.out))
    elif args.cmd == "measure":
        print(json.dumps(measure(args.glb, toward=None if args.auto_head else FORWARD), indent=1))
    else:
        print(json.dumps(compare_reference(args.glb, args.image, args.facing, args.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
