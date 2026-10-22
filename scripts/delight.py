#!/usr/bin/env python3
"""Take the light out of a cut-out photo before it is baked onto a model: lift its shadows toward its lit side.

A photo bake copies the photo's pixels into the albedo, the light included. The hay bale's photo had its right side
in deep shadow, and that shadow became a dark patch painted on the model (and could reach the fill's swatch). The
light on a subject changes slowly across it, its material (straw, grain, paint) quickly, so:

- log-luminance (linearised sRGB, Rec.709) of every subject pixel;
- its low-frequency part, the illumination: a gaussian blur normalised by the alpha mask (blur of alpha * L over blur
  of alpha), so the background never bleeds a halo into the subject's edge. sigma = scale * the diagonal of the
  subject's box. The blur is edge-aware (a bilateral range term, `edge` log units, on a lightly smoothed copy): a
  shadow with a hard border (the bale's right side, a tenth of its width) is averaged only with itself, so it comes
  up all the way and the lit straw beside it does not glow. edge 0 is the plain normalised blur;
- each pixel's log-luminance moves by strength * (reference - illumination), where the reference is a high
  percentile of the illumination over the subject: the shade comes up to the lit level, the lit side barely moves;
- one gain per pixel on all three linear channels, so hue and chroma stay; capped at MAX_GAIN, and rolled off near
  white so a lifted pixel never clips a channel.

Pixels outside the mask and the alpha channel are left exactly as they were. delight_full also returns each pixel's
trust: deep shade lifted near the cap is skylight's grey-blue, not the material, and a bake leans on it less.

    delight.py lifted.png -o lifted_delit.png [--strength 1] [--scale 0.08] [--edge 0.5] [--max-gain 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MAX_GAIN = 8.0     # the most a shadow is brightened (3 stops): the bale's shade is ~18x darker than its lit side
EDGE = 0.5         # bilateral range sigma, in log-luminance units: a step bigger than this is light, not material
TRUST_FULL = 2.0   # a lift up to this is trusted fully (see delight_full)...
TRUST_ZERO = 3.0   # ...and from this up not at all: on the bale, shade lifted past 3x came out sky-lit grey-blue
LEVELS = 16        # range levels the bilateral is sampled at (Durand-Dorsey piecewise-linear)
REF_PCT = 90.0     # percentile of the illumination taken as the lit level
KNEE = 0.9         # linear level above which a lifted pixel rolls off toward white instead of clipping
FLOOR = 1e-4       # linear luminance floor before the log (the shade has near-black pixels)
WORK_SIGMA = 8.0   # the blur runs on an image shrunk so sigma is about this many pixels


def _to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _luminance(lin: np.ndarray) -> np.ndarray:
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def illumination(logy: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """The mask-normalised gaussian blur of logy (normalised convolution): blur(w * logy) / blur(w). Computed on a
    shrunk copy (block means) and scaled back up, since sigma is a tenth of the subject and the blur is smooth."""
    f = max(1, int(sigma // WORK_SIGMA))
    h, wd = logy.shape
    ph, pw = (-h) % f, (-wd) % f
    num = np.pad(w * logy, ((0, ph), (0, pw)))
    den = np.pad(w, ((0, ph), (0, pw)))
    if f > 1:
        num = num.reshape(num.shape[0] // f, f, num.shape[1] // f, f).mean(axis=(1, 3))
        den = den.reshape(den.shape[0] // f, f, den.shape[1] // f, f).mean(axis=(1, 3))
    s = sigma / f
    num = ndimage.gaussian_filter(num, s, mode="constant")
    den = ndimage.gaussian_filter(den, s, mode="constant")
    small = num / np.maximum(den, 1e-6)
    if f > 1:
        # Block centres sit at (i + 0.5) * f - 0.5 in full-res pixels: sample the small map there, bilinearly.
        yy = (np.arange(h) + 0.5) / f - 0.5
        xx = (np.arange(wd) + 0.5) / f - 0.5
        gy, gx = np.meshgrid(yy, xx, indexing="ij")
        return ndimage.map_coordinates(small, [gy, gx], order=1, mode="nearest")
    return small[:h, :wd]


def illumination_edge(logy: np.ndarray, w: np.ndarray, sigma: float, edge: float = EDGE) -> np.ndarray:
    """Edge-aware illumination: the normalised blur, each neighbour weighted also by how close its lightly smoothed
    log-luminance (sigma / 4: the straw gone, a shadow's border kept) is to the pixel's own. Piecewise-linear over
    LEVELS range levels, each one plain normalised blur with the range weight folded into the mask."""
    if edge <= 0:
        return illumination(logy, w, sigma)
    guide = illumination(logy, w, sigma / 4.0)
    sel = w > 0.5 if (w > 0.5).any() else w > 0
    lo, hi = np.percentile(guide[sel], [0.5, 99.5])
    if hi - lo < 1e-6:
        return illumination(logy, w, sigma)
    levels = np.linspace(lo, hi, LEVELS)
    t = np.clip((guide - lo) / (hi - lo) * (LEVELS - 1), 0, LEVELS - 1)
    i0 = np.minimum(np.floor(t).astype(np.int64), LEVELS - 2)
    f = t - i0
    out = np.zeros_like(logy)
    for k, c in enumerate(levels):
        near = (i0 == k) | (i0 == k - 1)
        if not near.any():
            continue
        jk = illumination(logy, w * np.exp(-0.5 * ((guide - c) / edge) ** 2), sigma)
        out += np.where(i0 == k, jk * (1.0 - f), 0.0) + np.where(i0 == k - 1, jk * f, 0.0)
    return out


def _soft(m: np.ndarray) -> np.ndarray:
    """Values above KNEE rolled off so they approach 1 and never reach it."""
    t = (m - KNEE) / (1.0 - KNEE)
    return np.where(m > KNEE, KNEE + (1.0 - KNEE) * np.tanh(np.maximum(t, 0.0)), m)


def delight_full(rgba, strength: float = 1.0, scale: float = 0.08, edge: float = EDGE,
                 max_gain: float = MAX_GAIN) -> tuple[np.ndarray, dict, np.ndarray]:
    """delight(), a summary (the gain the darkest tenth of the subject's illumination got, the variance of its
    low-frequency log-luminance before and after) and each pixel's trust, 0..1.

    Trust is how far the lift can vouch for the colour: 1 where the shade asked for at most TRUST_FULL x, falling to
    0 where it asked for TRUST_ZERO x or more (log-linear between). Deep shade is lit by the sky, not the sun: brightened, it comes out the
    sky's grey-blue, not the material's colour, so a bake should lean on it less (retexture --delight does)."""
    arr = np.array(rgba.convert("RGBA") if isinstance(rgba, Image.Image) else rgba, dtype=np.uint8, copy=True)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("delight takes an RGBA image")
    alpha = arr[..., 3]
    mask = alpha > 0
    report = {"strength": float(strength), "scale": float(scale), "edge": float(edge), "pixels": int(mask.sum())}
    trust = np.ones(alpha.shape, np.float32)
    if not mask.any() or strength == 0:
        return arr, {**report, "shadow_lift": 1.0, "max_gain": 1.0, "capped": 0.0}, trust
    w = alpha.astype(np.float64) / 255.0
    ys, xs = np.where(alpha >= 128) if (alpha >= 128).any() else np.where(mask)
    diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    sigma = max(1.0, scale * diag)

    lin = _to_linear(arr[..., :3].astype(np.float64) / 255.0)
    logy = np.log(np.maximum(_luminance(lin), FLOOR))
    illum = illumination_edge(logy, w, sigma, edge)
    solid = w > 0.5
    ref = float(np.percentile(illum[solid if solid.any() else mask], REF_PCT))

    gain = np.exp(-strength * (illum - ref))
    if strength > 0:
        t = (np.log(TRUST_ZERO) - np.log(np.maximum(gain, 1e-12))) / (np.log(TRUST_ZERO) - np.log(TRUST_FULL))
        trust = np.where(mask, np.clip(t, 0.0, 1.0), 1.0).astype(np.float32)
    capped = gain > max_gain
    gain = np.clip(gain, 1.0 / max_gain, max_gain)
    # One scalar per pixel, so hue and chroma stay. A brightened pixel whose top channel would pass KNEE rolls off
    # toward 1 instead of clipping (never below where it was); pixels that were already bright and are not lifted
    # are not touched by the roll-off.
    top = lin.max(axis=2)
    lifted = top * gain
    roll = (gain > 1.0) & (lifted > KNEE)
    safe = np.where(roll, np.maximum(_soft(lifted), top) / np.maximum(lifted, 1e-12), 1.0)
    gain = gain * safe
    out_lin = lin * gain[..., None]
    rgb = np.clip(np.rint(_to_srgb(out_lin) * 255.0), 0, 255).astype(np.uint8)
    out = arr.copy()
    out[mask, :3] = rgb[mask]

    after_log = np.log(np.maximum(_luminance(_to_linear(out[..., :3].astype(np.float64) / 255.0)), FLOOR))
    illum_after = illumination_edge(after_log, w, sigma, edge)
    sel = solid if solid.any() else mask
    dark = sel & (illum <= np.percentile(illum[sel], 10.0))
    report.update({
        "sigma_px": round(sigma, 2),
        "shadow_lift": round(float(np.median(gain[dark])), 3),
        "max_gain": round(float(gain[mask].max()), 3),
        "capped": round(float(capped[mask].mean()), 4),
        "lowfreq_var_before": round(float(illum[sel].var()), 5),
        "lowfreq_var_after": round(float(illum_after[sel].var()), 5),
        "low_trust": round(float((trust[sel] < 0.5).mean()), 4),
    })
    return out, report, trust


def delight_report(rgba, strength: float = 1.0, scale: float = 0.08, edge: float = EDGE,
                   max_gain: float = MAX_GAIN) -> tuple[np.ndarray, dict]:
    """delight() and its summary (delight_full without the trust)."""
    out, report, _ = delight_full(rgba, strength, scale, edge, max_gain)
    return out, report


def delight(rgba, strength: float = 1.0, scale: float = 0.08, edge: float = EDGE, max_gain: float = MAX_GAIN):
    """The cut-out with its shading lifted toward its lit level (see the module doc). Takes an RGBA uint8 array or a
    PIL image and returns the same kind; strength 0 returns it unchanged."""
    out, _ = delight_report(rgba, strength, scale, edge, max_gain)
    return Image.fromarray(out, "RGBA") if isinstance(rgba, Image.Image) else out


def main() -> int:
    ap = argparse.ArgumentParser(description="Lift a cut-out photo's shadows toward its lit level before a bake.")
    ap.add_argument("image", type=Path, help="an RGBA cut-out (lifted.png)")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--strength", type=float, default=1.0, help="0 leaves it as is, 1 takes the shading out")
    ap.add_argument("--scale", type=float, default=0.08,
                    help="blur sigma as a fraction of the subject's box diagonal: what counts as light, not material")
    ap.add_argument("--edge", type=float, default=EDGE,
                    help="bilateral range in log-luminance: a step bigger than this is treated as a shadow's border "
                         "(0: plain normalised blur)")
    ap.add_argument("--max-gain", type=float, default=MAX_GAIN, help="the most any pixel is brightened")
    args = ap.parse_args()
    try:
        with Image.open(args.image) as im:
            out, report = delight_report(im.convert("RGBA"), args.strength, args.scale, args.edge, args.max_gain)
    except (OSError, ValueError) as e:
        print(f"delight: {e}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(args.out)
    print(json.dumps({"out": str(args.out), **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
