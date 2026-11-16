#!/usr/bin/env python3
"""Take the magenta key out of a cut-out's see-through parts (glass, vents, gaps the cut-out kept).

A drawn subject sits on a flat magenta key. Where it is transparent (a lantern's glass globe, the vents in its
cap) the key shows through, and Apple Vision keeps those pixels as subject, so paint and retexture bake magenta
onto the mesh. Every solid pixel whose colour leans toward the key's hue loses that lean: its chroma along the
key's chroma direction is removed (glass seen against magenta goes neutral grey; brass, which leans elsewhere,
is untouched). The removal fades in between cos 0.92 and 0.98 of the key's direction, so a pixel only partly
toward the key changes only partly.

A subject that is mostly key-coloured (a pink flamingo on magenta) is left alone: past MAX_SHARE of the subject the
lean is the subject's own colour, not spill. That case prints a note instead.

usage: key_spill.py <in.png> <out.png> --key R,G,B   (prints one JSON line: {"pixels", "share", "skipped"})
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

COS_LO, COS_HI = 0.92, 0.98
MIN_CHROMA = 12.0  # chroma (0-255 scale) below this is grey already
MAX_SHARE = 0.5  # more of the subject than this leaning to the key: the subject is key-coloured, leave it


def despill(rgba: np.ndarray, key: tuple[int, int, int]) -> tuple[np.ndarray, dict]:
    """rgba (H, W, 4) uint8 -> (despilled copy, {"pixels", "share", "skipped"})."""
    out = rgba.copy()
    solid = rgba[..., 3] >= 128
    k = np.asarray(key, dtype=np.float64)
    kc = k - k.mean()
    norm = float(np.linalg.norm(kc))
    if norm < 1e-6 or not solid.any():
        return out, {"pixels": 0, "share": 0.0, "skipped": "no key hue" if norm < 1e-6 else "empty"}
    kc /= norm
    px = rgba[..., :3].astype(np.float64)
    c = px - px.mean(-1, keepdims=True)
    mag = np.linalg.norm(c, axis=-1)
    along = (c * kc).sum(-1)
    cos = np.where(mag > 1e-6, along / np.maximum(mag, 1e-6), 0.0)
    weight = np.clip((cos - COS_LO) / (COS_HI - COS_LO), 0.0, 1.0) * (mag >= MIN_CHROMA) * solid
    hit = weight > 0
    share = float(hit.sum()) / float(solid.sum())
    info = {"pixels": int(hit.sum()), "share": round(share, 4), "skipped": ""}
    if share > MAX_SHARE:
        info["skipped"] = "subject is key-coloured"
        return out, info
    fixed = px - (weight * np.clip(along, 0, None))[..., None] * kc
    out[..., :3] = np.clip(np.rint(fixed), 0, 255).astype(np.uint8)
    return out, info


def parse_key(text: str) -> tuple[int, int, int]:
    parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError("--key wants R,G,B")
    return parts[0], parts[1], parts[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--key", required=True, help="the ground key R,G,B (flat/key.json rgb)")
    args = ap.parse_args(argv)
    rgba = np.asarray(Image.open(args.src).convert("RGBA"))
    out, info = despill(rgba, parse_key(args.key))
    Image.fromarray(out, "RGBA").save(args.dst)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
