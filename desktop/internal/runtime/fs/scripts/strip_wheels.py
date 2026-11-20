#!/usr/bin/env python3
"""Take the wheels off a side-view car picture, in code, with no image-edit model and no hand painting.

    .venv/bin/python scripts/strip_wheels.py <car.png> <body_nowheels.png> [--wheel-out <wheel.png>] [--debug <overlay.png>]

How: Vision cuts the car out (bin/lift-subject), the wheels are found in that cut-out, and then
  1. the lower edge of the car is read column by column: between the wheels it is the underside (the "sill line");
     where it drops clearly below that line, that is a tyre. Each tyre's lower arc is fitted with a circle;
  2. inside each circle the wheel is replaced: above the sill line by the dark of the wheel well (the car stays
     solid there, so the shape model reads a body, not a hole); below the sill line it is cut away (made see-through);
  3. the car is set on a flat plain ground of the colour the picture's own border had (no shadow, no floor).
Because everything below the sill line inside the tyre circle is transparent in the cut-out and the ground is one
flat colour, nothing is left to be mistaken for part of the car (the old hand-painted patch became slabs under the
bumpers). Writes the picture, plus <name>.wheels.json (the fitted circles: centre, radius, sill line) next to it.
The wheel circles are also what you would crop to get the wheel itself.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
GROW = 1.06  # the well is filled a touch wider than the fitted tyre so its rim goes too
MIN_WHEEL = 0.06  # a wheel is at least this share of the car's length across, or it is not a wheel


def lift_binary() -> Path:
    cands = []
    if os.environ.get("LIFT_SUBJECT"):
        cands.append(Path(os.environ["LIFT_SUBJECT"]))
    cands.append(ROOT / "bin" / "lift-subject")
    try:  # a git worktree has no bin/ of its own: use the main checkout's
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], capture_output=True, text=True).stdout.strip()
        if common:
            cands.append((ROOT / common).resolve().parent / "bin" / "lift-subject")
    except OSError:
        pass
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    sys.exit("strip_wheels: no bin/lift-subject (run scripts/bootstrap.sh, or set LIFT_SUBJECT)")


def cut_out(src: Path) -> np.ndarray:
    """The car's alpha (0..255) from Vision's subject lift."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "lifted.png"
        subprocess.run([str(lift_binary()), str(src), str(out)], check=True, capture_output=True)
        return np.asarray(Image.open(out).convert("RGBA"))[..., 3]


def ground_colour(im: Image.Image) -> tuple[int, int, int]:
    a = np.asarray(im.convert("RGB"))
    b = max(2, min(a.shape[:2]) // 40)
    edge = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(edge, axis=0))


def lower_edge(solid: np.ndarray) -> np.ndarray:
    """For each column, the y of the lowest car pixel (-1 where the column has none)."""
    h = solid.shape[0]
    has = solid.any(axis=0)
    low = h - 1 - np.argmax(solid[::-1], axis=0)
    return np.where(has, low, -1)


def fit_circle(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle (Kasa) through points."""
    A = np.c_[2 * xs, 2 * ys, np.ones_like(xs)]
    b = xs**2 + ys**2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(cx), float(cy), float(np.sqrt(max(c + cx**2 + cy**2, 1e-6)))


def find_wheels(solid: np.ndarray) -> tuple[list[dict], float]:
    low = lower_edge(solid)
    cols = np.where(low >= 0)[0]
    x0, x1 = int(cols[0]), int(cols[-1])
    length = x1 - x0 + 1
    mid = cols[(cols > x0 + length * 0.35) & (cols < x1 - length * 0.35)]
    sill = float(np.median(low[mid]))  # the underside between the wheels
    height = solid.any(axis=1).nonzero()[0]
    drop = max(4.0, 0.012 * length)
    below = (low > sill + drop) & (low >= 0)
    below = ndimage.binary_closing(below, iterations=max(2, int(length * 0.01)))
    lab, n = ndimage.label(below)
    wheels = []
    for k in range(1, n + 1):
        xs = np.where(lab == k)[0]
        if xs.size < MIN_WHEEL * length * 0.5:
            continue
        # The tyre's lower arc is the edge below the sill line; fit a circle to it, using the arc's own end points
        # to drop the sill-line columns at its two ends.
        ys = low[xs].astype(float)
        keep = ys > sill + drop * 0.5
        if keep.sum() < 8:
            continue
        px, py = xs[keep].astype(float), ys[keep]
        cx, cy, r = fit_circle(px, py)
        for _ in range(6):  # trim what is not the tyre (mud flap, bumper corner, shadow) and refit
            res = np.abs(np.hypot(px - cx, py - cy) - r)
            ok = res <= max(1.5, np.quantile(res, 0.7) * 1.5)
            if ok.sum() < 8 or ok.all():
                break
            px, py = px[ok], py[ok]
            cx, cy, r = fit_circle(px, py)
        if r < MIN_WHEEL * length / 2 or r > 0.35 * length:
            continue
        wheels.append({"cx": cx, "cy": cy, "r": r, "bottom": float(low[xs].max()), "x_lo": int(xs.min()), "x_hi": int(xs.max())})
    wheels.sort(key=lambda w: w["cx"])
    return wheels, sill


def refine(rgb: np.ndarray, w: dict) -> dict:
    """The mask arc is short and its circle fit loose (it ran 10 px wide of the tyre). Two things are exact: the
    tyre's lowest point (the car's lowest pixel there) and the hub. The rim and spokes are the bright, unpainted
    part of a wheel, so their centre is the hub; the radius is then the hub's height above the lowest point."""
    h, wd = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:wd]
    win = np.hypot(xx - w["cx"], yy - w["cy"]) < w["r"] * 0.8
    lum = rgb.astype(float).mean(axis=2)
    paint = (rgb[..., 0] > 50) & (rgb[..., 0] > 2.0 * np.maximum(rgb[..., 1], rgb[..., 2]))
    hub = win & (lum > 110) & ~paint
    if hub.sum() < 30:
        return {**w, "hub": False}
    cx, cy = float(xx[hub].mean()), float(yy[hub].mean())
    return {**w, "cx": cx, "cy": cy, "r": w["bottom"] - cy, "hub": True}


def well_colour(rgb: np.ndarray, solid: np.ndarray, w: dict, sill: float) -> tuple[int, int, int]:
    """The wheel well's dark: the darkest fifth of the car's own pixels in the ring just outside the tyre."""
    h, wd = solid.shape
    yy, xx = np.mgrid[0:h, 0:wd]
    d = np.hypot(xx - w["cx"], yy - w["cy"])
    ring = solid & (d > w["r"] * 1.0) & (d < w["r"] * 1.25) & (yy < sill)
    px = rgb[ring]
    if len(px) < 20:
        return (24, 20, 20)
    lum = px.mean(axis=1)
    dark = px[lum <= np.quantile(lum, 0.2)]
    return tuple(int(v) for v in np.median(dark, axis=0))


WHEEL_SHRINK = 1.0  # share of the fitted tyre radius the wheel crop keeps


def crop_wheel(im: Image.Image, w: dict, ground: tuple[int, int, int], size: int = 1024) -> Image.Image:
    """One wheel on its own, as a square picture: the tyre's disc from the original photo, upscaled, on the flat
    ground colour (the arch, the bumper and the road outside the tyre are replaced)."""
    r = w["r"] * WHEEL_SHRINK
    pad = r * 0.18
    half = r + pad
    box = (round(w["cx"] - half), round(w["cy"] - half), round(w["cx"] + half), round(w["cy"] + half))
    tile = im.crop(box).resize((size, size), Image.LANCZOS)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2
    rad = size / 2 * r / half
    soft = np.clip((rad - np.hypot(xx - c, yy - c)) / 3.0 + 0.5, 0, 1)[..., None]
    flat = np.empty((size, size, 3), np.float32)
    flat[:] = ground
    return Image.fromarray((np.asarray(tile, np.float32) * soft + flat * (1 - soft)).round().astype(np.uint8))


def strip(src: Path, dst: Path, debug: Path | None = None, wheel_out: Path | None = None) -> dict:
    im = Image.open(src).convert("RGB")
    rgb = np.asarray(im).copy()
    alpha = cut_out(src)
    if alpha.shape != rgb.shape[:2]:
        alpha = np.asarray(Image.fromarray(alpha).resize(im.size, Image.LANCZOS))
    solid = alpha > 127
    wheels, sill = find_wheels(solid)
    wheels = [refine(rgb, w) for w in wheels]
    if len(wheels) < 2:
        sys.exit(f"strip_wheels: found {len(wheels)} wheel(s) in {src}; this needs a side view with at least two visible wheels")
    h, wd = solid.shape
    yy, xx = np.mgrid[0:h, 0:wd]
    keep = alpha.astype(float) / 255.0
    for w in wheels:
        inside = np.hypot(xx - w["cx"], yy - w["cy"]) <= w["r"] * GROW
        w["well_rgb"] = well_colour(rgb, solid, w, sill)
        # Only what is not the car's paint: the fender lip that the circle may graze stays red.
        paint = (rgb[..., 0] > 50) & (rgb[..., 0] > 2.0 * np.maximum(rgb[..., 1], rgb[..., 2]))
        well = inside & (yy < sill) & ~paint
        rgb[well] = w["well_rgb"]
        keep[well] = np.maximum(keep[well], 1.0)  # the well is part of the car, solid
        # What hung below the underside is gone: the tyre's whole column span, not only its fitted circle.
        keep[(yy >= sill) & (xx >= w["x_lo"] - 3) & (xx <= w["x_hi"] + 3)] = 0.0
    # The underside line itself: a one-pixel clean edge (anything of a wheel left below it goes).
    ground = ground_colour(im)
    flat = np.empty_like(rgb)
    flat[:] = ground
    a3 = keep[..., None]
    out = (rgb * a3 + flat * (1 - a3)).round().astype(np.uint8)
    Image.fromarray(out).save(dst)
    info = {"ground_rgb": list(ground), "sill_y": sill, "wheels": [{k: (round(v, 1) if isinstance(v, float) else list(v) if isinstance(v, tuple) else v) for k, v in w.items()} for w in wheels]}
    dst.with_suffix(".wheels.json").write_text(json.dumps(info, indent=1))
    if wheel_out:
        crop_wheel(im, wheels[0], ground).save(wheel_out)
    if debug:
        ov = im.copy()
        dr = ImageDraw.Draw(ov)
        for w in wheels:
            dr.ellipse([w["cx"] - w["r"], w["cy"] - w["r"], w["cx"] + w["r"], w["cy"] + w["r"]], outline=(0, 255, 0), width=3)
        dr.line([0, sill, wd, sill], fill=(255, 255, 0), width=2)
        ov.save(debug)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--debug", default="")
    ap.add_argument("--wheel-out", default="", help="also write the front wheel, alone on the flat ground, as its own picture")
    a = ap.parse_args()
    info = strip(Path(a.src), Path(a.dst), Path(a.debug) if a.debug else None, Path(a.wheel_out) if a.wheel_out else None)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
