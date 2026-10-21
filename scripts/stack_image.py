#!/usr/bin/env python3
"""Build a sprite stack straight from the concept picture. No mesh.

The picture is the front view. The subject's outline is puffed into a volume: each
pixel gets a thickness from how far it sits inside the outline, so a disc becomes a
ball and a blade stays thin. When a depth model is installed, its estimate of what
is nearer the camera thickens those parts. Each column of voxels takes the color of
its pixel, front to back, with the key color unmixed from the cutout's edge first.
The slices go bottom to top, the same as sprite_stack.py.

Writes the usual stack contract:
  out/<stem>/stack/sheet.png
  out/<stem>/stack/preview.png
  out/<stem>/stack/stack.json   ("maker": "picture")

The depth model is optional: torch and transformers in the factory venv turn it on
(STACK_DEPTH_MODEL picks the repo, STACK_DEPTH=off turns it off).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label, maximum_filter

from sprite_stack import write_outputs

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
FILL = 0.9  # share of the stack's width the subject spans
MIN_PIXELS = 24
EDGE = 3  # px of the cutout's rim checked for key color
SPILL = 0.15  # share of the key's hue that marks a rim pixel as part ground


class StackError(RuntimeError):
    pass


def depth_model() -> str:
    return (os.environ.get("STACK_DEPTH_MODEL") or DEPTH_MODEL).strip()


def depth_ready() -> bool:
    if (os.environ.get("STACK_DEPTH") or "").strip().lower() == "off":
        return False
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def ground(asset: Path) -> tuple[tuple[int, int, int], int] | None:
    """The key color the still was painted on, and its tolerance: flat/key.json, else
    sampled from the border of the 2D still or the 3D input picture."""
    from flat2d.parts import TOL, ground_key

    key_file = asset / "flat" / "key.json"
    key: dict = {}
    if key_file.is_file():
        try:
            key = json.loads(key_file.read_text(encoding="utf-8"))
        except ValueError:
            key = {}
    if not isinstance(key, dict) or not isinstance(key.get("rgb"), list) or len(key["rgb"]) != 3:
        pics = [asset / "flat" / "source.png", *sorted(asset.glob("input.*"))]
        pic = next((p for p in pics if p.is_file()), None)
        if pic is None:
            return None
        try:
            with Image.open(pic) as im:
                key = ground_key(im)
        except OSError:
            return None
    tol = key.get("tol") if isinstance(key.get("tol"), (int, float)) else TOL
    return tuple(int(c) for c in key["rgb"]), int(tol)


def subject(asset: Path) -> tuple[Image.Image, np.ndarray, str]:
    """The concept picture and its subject mask (True = subject), and which file it came from."""
    lifted = asset / "lifted.png"
    if lifted.is_file():
        im = Image.open(lifted).convert("RGBA")
        alpha = np.asarray(im.getchannel("A")) >= 128
        if alpha.any() and not alpha.all():
            return im, alpha, "lifted.png"
    src = asset / "flat" / "source.png"
    if not src.is_file():
        raise StackError("no concept picture yet")
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    from flat2d.store import vision_mask

    mask = vision_mask(src)
    if mask is None:
        from flat2d.parts import _foreground

        rgb, tol = ground(asset) or ((255, 0, 255), 36)
        mask, w, h = _foreground(im, rgb, tol)
    alpha = np.frombuffer(bytes(mask), dtype=np.uint8).reshape(h, w) > 0
    return im, alpha, "flat/source.png"


def clean_colors(im: Image.Image, mask: np.ndarray, key: tuple[tuple[int, int, int], int] | None) -> Image.Image:
    """The cutout's colors with the ground taken out of its edge.

    The edge of a cutout is a blend of the subject and the key (magenta), and voxels
    that take those colors show as pink specks. The ground is unmixed from each pixel
    the way 2D Split does it (see_through), and any pixel still close to the key, or
    mostly ground, takes the color of the nearest clean subject pixel instead."""
    if key is None:
        return im
    from flat2d.parts import see_through

    rgb, tol = key
    rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    rgba[..., 3] = np.where(mask, 255, 0)
    clean = np.asarray(see_through(Image.fromarray(rgba, "RGBA"), rgb), dtype=np.uint8)
    col = clean[..., :3].astype(np.int32)
    near_key = np.abs(col - np.array(rgb, dtype=np.int32)).sum(-1) <= tol
    dirty = mask & (near_key | (clean[..., 3] < 128))
    # On the rim, a pixel carrying a share of the key's hue is part ground.
    weak = min(range(3), key=lambda ch: rgb[ch])
    strong = [ch for ch in range(3) if rgb[ch] - rgb[weak] > 96]
    if strong:
        spill = (col[..., strong].min(-1) - col[..., weak]) / float(min(rgb[ch] for ch in strong) - rgb[weak])
        rim = mask & (distance_transform_edt(mask) <= EDGE)
        dirty |= rim & (spill > SPILL)
    good = mask & ~dirty
    out = clean[..., :3].copy()
    if dirty.any() and good.any():
        _, (iy, ix) = distance_transform_edt(~good, return_indices=True)
        out[dirty] = out[iy[dirty], ix[dirty]]
    return Image.fromarray(out, "RGB").convert("RGBA")


def main_piece(mask: np.ndarray) -> np.ndarray:
    """The largest connected piece, holes filled. Specks and stray marks go."""
    labels, n = label(mask)
    if n == 0:
        raise StackError("no subject found in the picture")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = labels == int(sizes.argmax())
    if keep.sum() < MIN_PIXELS:
        raise StackError("the subject is too small to stack")
    return binary_fill_holes(keep)


def estimate_depth(im: Image.Image) -> np.ndarray:
    """Relative nearness, 0..1 (1 = nearest), same size as the picture."""
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipeline("depth-estimation", model=depth_model(), device=device)
    out = pipe(im.convert("RGB"))
    depth = np.asarray(out["depth"].resize(im.size, Image.BILINEAR), dtype=np.float32)
    lo, hi = float(depth.min()), float(depth.max())
    return (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)


def half_thickness(mask: np.ndarray, near: np.ndarray | None) -> np.ndarray:
    """Half the front-to-back thickness of each pixel's column, in pixels.

    The outline is filled with balls: one on each point of its ridge (the pixels
    furthest from the edge locally), as wide as that point is deep. A pixel's
    thickness is the tallest ball over it, so a disc becomes a sphere, a blade stays
    as thin as it is wide, and a lantern's glass bulges past its cap. Nearness from a
    depth model then scales that between 0.6x and 1.4x inside the subject."""
    dist = distance_transform_edt(mask)
    ridge = mask & (dist >= maximum_filter(dist, size=3) - 0.5)
    h, w = mask.shape
    half = np.zeros(mask.shape, dtype=np.float64)
    for y, x in np.argwhere(ridge):
        r = float(dist[y, x])
        k = int(np.ceil(r))
        y0, y1, x0, x1 = max(0, y - k), min(h, y + k + 1), max(0, x - k), min(w, x + k + 1)
        dy = np.arange(y0, y1)[:, None] - y
        dx = np.arange(x0, x1)[None, :] - x
        ball = np.sqrt(np.maximum(r * r - dy * dy - dx * dx, 0.0))
        np.maximum(half[y0:y1, x0:x1], ball, out=half[y0:y1, x0:x1])
    if near is not None:
        inside = near[mask]
        lo, hi = float(inside.min()), float(inside.max())
        rel = (near - lo) / (hi - lo) if hi > lo else np.full_like(near, 0.5)
        half = half * (0.6 + 0.8 * np.clip(rel, 0.0, 1.0))
    half[~mask] = 0.0
    return half


def stack_picture(im: Image.Image, mask: np.ndarray, res: int, near: np.ndarray | None = None) -> dict:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    w, h = x1 - x0, y1 - y0
    # One pitch for x, y and z: the wider of width and height spans FILL of the grid,
    # so a tall subject keeps its proportions in layers.
    scale = (res * FILL) / float(max(w, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    # Paint the ground under the subject's nearest color first, so resizing does not
    # blend the key (magenta) into the edge voxels.
    px = np.asarray(im.convert("RGB"), dtype=np.uint8)[y0:y1, x0:x1].copy()
    sub = mask[y0:y1, x0:x1]
    _, (iy, ix) = distance_transform_edt(~sub, return_indices=True)
    px = px[iy, ix]
    crop = Image.fromarray(px, "RGB").resize((nw, nh), Image.LANCZOS)
    small = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    m = np.asarray(small) >= 128
    if not m.any():
        raise StackError("the subject is too small to stack")
    near_small = None
    if near is not None:
        n = Image.fromarray(near[y0:y1, x0:x1].astype(np.float32), mode="F").resize((nw, nh), Image.BILINEAR)
        near_small = np.asarray(n, dtype=np.float32)
    half = half_thickness(m, near_small)
    half = np.minimum(half, res / 2.0 - 1.0)
    rgb = np.asarray(crop, dtype=np.uint8)

    ox = (res - nw) // 2
    zc = (res - 1) / 2.0
    z = np.arange(res, dtype=np.float32)[:, None]  # slice rows; the last row faces the camera
    slices = []
    voxels = 0
    for layer in range(nh):
        row = nh - 1 - layer  # layer 0 is the bottom of the picture
        sl = np.zeros((res, res, 4), dtype=np.uint8)
        cols = np.nonzero(m[row])[0]
        if cols.size:
            # At least one voxel thick, so the outline itself is solid.
            hz = np.maximum(half[row, cols], 0.5)[None, :]
            fill = np.abs(z - zc) <= hz
            zz, ci = np.nonzero(fill)
            xx = cols[ci] + ox
            sl[zz, xx, :3] = rgb[row, cols[ci]]
            sl[zz, xx, 3] = 255
            voxels += int(zz.size)
        slices.append(sl)
    pitch = 1.0 / res
    depth = float(2.0 * half.max() * pitch)
    return {
        "slices": slices,
        "pitch": pitch,
        "origin": [-0.5, 0.0, -0.5],
        "resolution": int(res),
        "layers": len(slices),
        "bounds": [[-nw * pitch / 2, 0.0, -depth / 2], [nw * pitch / 2, nh * pitch, depth / 2]],
        "extents": [nw * pitch, nh * pitch, depth],
        "surface_voxels": voxels,
        "filled": True,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", required=True, help="out/<stem>/ folder")
    p.add_argument("--root", type=Path, default=root, help="factory root that holds out/")
    p.add_argument("--res", type=int, default=160, help="XZ pixel resolution")
    p.add_argument("--depth", choices=("auto", "off"), default="auto", help="use the depth model when installed")
    args = p.parse_args()
    asset = args.root / "out" / args.stem
    try:
        im, mask, source = subject(asset)
        mask = main_piece(mask)
        im = clean_colors(im, mask, ground(asset))
        near = None
        maker = "outline"
        if args.depth == "auto" and depth_ready():
            print(f"stack_image: depth from {depth_model()}", flush=True)
            near = estimate_depth(im)
            maker = "outline+depth"
        result = stack_picture(im, mask, max(16, int(args.res)), near)
    except StackError as e:
        print(f"stack_image: {e}", file=sys.stderr)
        return 2
    result["stem"] = args.stem
    result["source"] = source
    write_outputs(result, asset / "stack", Path(source).name)
    meta_path = asset / "stack" / "stack.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["maker"] = "picture"
    meta["volume"] = maker
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "layers": result["layers"], "volume": maker, "source": source}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
