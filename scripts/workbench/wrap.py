#!/usr/bin/env python3
"""Wrap one lengthwise picture round a round prop: every point takes the picture's colour for where it sits along
the long axis and how far round it sits, so stripes that run end to end (a watermelon) meet nowhere.

    .venv/bin/python scripts/workbench/wrap.py in.glb picture.png -o out.glb

A camera bake of a round prop covers the side the picture shows and leaves the rest to a mirror and a fill swatch;
on the watermelon the swatch's stripes ran across where the picture's ran along, a seam round the middle (the
patchwork Stephen called out on the pig). Here nothing is projected from a camera: the picture is read as an
side view of the prop lying along its longest axis, its top-to-bottom shading taken out. A point at angle t round
that axis (0 facing the picture, +-90 deg top and bottom) reads the picture's column for where it sits along, at a
height even in t across BAND of the outline; the far half mirrors the near one, so the colour is the same on both
sides of the two mirror lines (top and bottom) and neither shows as a seam.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retexture  # noqa: E402

SAMPLES = 600_000
ENDS = 0.07  # the share of the outline's length left out at each end
BAND = 0.85  # the share of the outline's height the wrap reads (the rim is left out)


def outline(alpha: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    """The picture's subject: its first and last column, and each column's top and bottom row (nan where empty)."""
    cols = np.where(alpha.any(axis=0))[0]
    if len(cols) < 4:
        raise ValueError("the picture has no subject")
    h = alpha.shape[0]
    rows = np.arange(h)[:, None]
    top = np.where(alpha.any(axis=0), np.where(alpha, rows, h).min(axis=0), np.nan).astype(float)
    bot = np.where(alpha.any(axis=0), np.where(alpha, rows, -1).max(axis=0), np.nan).astype(float)
    return int(cols[0]), int(cols[-1]), top, bot


def flat_rows(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """The picture's top-to-bottom shading out: each row scaled to the subject's mean lightness. The stripes run
    along the rows, so a row's mean is its light, not its pattern (the melon's underside drawn in shade came out
    near black)."""
    lum = rgb.mean(axis=2)
    n = alpha.sum(axis=1)
    row = np.where(n > 0, (lum * alpha).sum(axis=1) / np.maximum(n, 1), np.nan)
    ok = np.isfinite(row)
    row = np.interp(np.arange(len(row)), np.where(ok)[0], row[ok])
    row = ndimage.gaussian_filter1d(row, max(2.0, 0.03 * len(row)))
    target = float((lum * alpha).sum() / max(alpha.sum(), 1))
    return np.clip(rgb * (target / np.maximum(row, 1.0))[:, None, None], 0, 255)


def wrap_colours(pts: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, flip: bool = False) -> np.ndarray:
    """Each point's colour from the picture, wrapped round the points' longest axis. flip turns the picture end
    for end (which end the stem is at)."""
    lo, hi = pts.min(0), pts.max(0)
    axis = int(np.argmax(hi - lo))
    a, b = [i for i in range(3) if i != axis]
    centre = (lo + hi) / 2
    along = (pts[:, axis] - lo[axis]) / max(hi[axis] - lo[axis], 1e-9)
    if flip:
        along = 1.0 - along
    # angle round the axis, 0 toward +a; the far half folds onto the near one (a mirror at the top and bottom)
    t = np.arctan2(pts[:, b] - centre[b], pts[:, a] - centre[a])
    t = np.where(np.abs(t) > np.pi / 2, np.sign(t) * (np.pi - np.abs(t)), t)
    c0, c1, top, bot = outline(alpha)
    rgb = flat_rows(rgb, alpha)
    # the ends read from ENDS in from the outline's ends: its first columns are the stem and the cutout's fringe,
    # which wrapped round the melon's end as a pale ring with a dark blue centre
    x = c0 + (ENDS + along * (1.0 - 2.0 * ENDS)) * (c1 - c0)
    xi = np.clip(np.rint(x).astype(int), c0, c1)
    # columns at the very ends can be empty or one pixel tall: take the nearest column with a real height
    good = np.isfinite(top) & (bot - top > 2)
    idx = np.where(good)[0]
    near = idx[np.clip(np.searchsorted(idx, xi), 0, len(idx) - 1)]
    mid = (top[near] + bot[near]) / 2
    half = (bot[near] - top[near]) / 2
    # Evenly round the angle over BAND of the outline's height: the picture's rim is squashed (read as sin(t) it
    # smeared a mirror line along the melon's top). Folding at the middle instead paired the stripes up like a
    # kaleidoscope; this way the only mirror lines are at the top and bottom.
    y = mid - (2.0 * t / np.pi) * half * BAND
    out = np.stack([ndimage.map_coordinates(rgb[..., c], [y, x], order=1, mode="nearest") for c in range(3)], 1)
    return out


def wrap(glb, picture, out, flip: bool = False, samples: int = SAMPLES) -> dict:
    mesh = trimesh.load(glb, force="mesh")
    vis = mesh.visual
    if getattr(vis, "uv", None) is None:
        raise ValueError(f"{glb}: no UVs")
    image = getattr(vis.material, "baseColorTexture", None) or getattr(vis.material, "image", None)
    size = image.size if isinstance(image, Image.Image) else (1024, 1024)
    tw, th = size
    rgb, _, alpha_bake = retexture.load_photo(picture)
    pts, faces, bary, _, _ = retexture.sample_surface(mesh, samples, with_fid=True)
    uvs = np.asarray(vis.uv, np.float32)
    uv_pts = sum(bary[:, i:i + 1] * uvs[faces[:, i]] for i in range(3))
    col = wrap_colours(pts, rgb, alpha_bake, flip)
    accum = np.zeros((th, tw, 3), np.float32)
    weight = np.zeros((th, tw), np.float32)
    retexture.splat_colours(uv_pts, col, accum, weight)
    # texels between samples take the colour around them (the atlas is finer than the samples)
    blur = lambda x: ndimage.gaussian_filter(x, (3, 3, 0) if x.ndim == 3 else 3)  # noqa: E731
    filled = blur(accum) / np.maximum(blur(weight), 1e-6)[..., None]
    tex = np.where((weight > 1e-3)[..., None], accum / np.maximum(weight, 1e-6)[..., None], filled)
    locked = Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8), "RGB")
    vis.material.baseColorTexture = locked
    vis.material.image = locked
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    mesh.export(tmp, file_type="glb")
    os.replace(tmp, out)
    return {"glb": str(out), "picture": str(picture), "flip": flip, "texels": int((weight > 1e-3).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("glb")
    ap.add_argument("picture")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--flip", action="store_true")
    a = ap.parse_args()
    print(wrap(a.glb, a.picture, a.out, a.flip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
