#!/usr/bin/env python3
"""Carve a sprite stack from four flat pixel art views of one object. No mesh, no image model.

The views, each a small picture on a transparent (or flat key colour) ground:
  top    seen from above, the front of the object at the bottom of the picture
         (columns run across the width, rows run back -> front)
  side   seen from the side, the object's nose at the left
         (mirrored before use, so its columns run back -> front; rows run top -> bottom)
  front  seen from the front (columns run across the width, as in the top view)
  back   seen from behind (mirrored before use, so its columns line up with the front's)

Each view is cropped to its subject first. Then the top view is W wide and L long, the side
view L long and H high, and the front and back views W wide and H high.

The carve:
  - One layer per pixel row of the side view, bottom row first. Each layer is the top view's size.
  - A cell is solid where the top view, the side view (along the length) and the front and back
    views (across the width) all say solid.
  - The topmost solid cell of each column takes the top view's colour. A cell lower down takes the
    front or back view's colour where the front or back of its layer is open, else the side view's.
  - The 1 px dark outline the views are drawn with is recoloured from the neighbouring inside colour
    first, or every layer would draw a line through the object.
  - Holes and overhangs (an arch, a spoiler) come out solid: four silhouettes cannot see them.

Every colour in the stack is a colour of one of the views: nothing is blended or resampled.
The layers are padded to a square (the stack files hold one resolution), and stack.json asks the
app for texels equal to the resolution and no contrast or saturation boost, so the pixels stay as
drawn.

Writes the usual stack files:
  out/<stem>/stack/sheet.png
  out/<stem>/stack/preview.png
  out/<stem>/stack/stack.json   ("maker": "views")

The views are read from out/<stem>/flat/pixel/{top,side,front,back}.png (pipeline#9 writes them there) unless given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from sprite_stack import write_outputs

VIEWS = ("top", "side", "front", "back")
DARK = 80  # the outline colour is darker than this (luma, 0-255)
SHARE = 0.4  # and draws at least this share of the edge
NEAR = 2.0  # how far (px) an outline pixel looks for the inside colour it takes
PALETTE_WARN = 32  # more distinct colours than this across the views is not one small palette


class ViewsError(RuntimeError):
    pass


def _rgba(im: Image.Image | np.ndarray) -> tuple[np.ndarray, bool]:
    """The view as RGBA, and whether it came with an alpha channel (else its ground is a key colour)."""
    if isinstance(im, np.ndarray):
        a = im.astype(np.uint8)
        if a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ViewsError(f"a view must be RGB or RGBA, got shape {a.shape}")
        if a.shape[2] == 3:
            return np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=2), False
        return a, True
    alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    return np.array(im.convert("RGBA"), dtype=np.uint8), alpha


def subject_mask(rgba: np.ndarray, key: tuple[int, int, int] | None = None, alpha: bool = True) -> np.ndarray:
    """Where the object is: the alpha channel when the picture has one (and no key is given), else everything but
    the key colour (given, or the most common colour on the picture's border)."""
    if key is None and alpha:
        return rgba[:, :, 3] >= 128
    rgb = rgba[:, :, :3]
    if key is None:
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        cols, counts = np.unique(border, axis=0, return_counts=True)
        key = tuple(int(v) for v in cols[counts.argmax()])
    return ~np.all(rgb == np.array(key, np.uint8), axis=2) & (rgba[:, :, 3] >= 128)


def crop(rgba: np.ndarray, mask: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ViewsError(f"the {name} view is empty")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgba[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def _luma(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def outline_color(rgba: np.ndarray, edge: np.ndarray) -> np.ndarray | None:
    """The colour the view's outline is drawn in: the most common edge colour, when it is dark and draws much of the
    edge. None when the view has no outline."""
    cols, counts = np.unique(rgba[edge][:, :3], axis=0, return_counts=True)
    if not len(cols):
        return None
    c = cols[counts.argmax()]
    if counts.max() < SHARE * edge.sum() or _luma(c) >= DARK:
        return None
    return c


def strip_outline(rgba: np.ndarray, mask: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Recolour the view's outline from the nearest inside pixel within NEAR px. The edge is every solid pixel with a
    4-neighbour outside the subject. "auto" recolours the edge pixels drawn in the outline colour (outline_color);
    "on" recolours every edge pixel; "off" leaves the view alone. The silhouette does not change."""
    out = rgba.copy()
    if mode == "off":
        return out
    pad = np.pad(mask, 1, constant_values=False)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    edge = mask & ~inner
    if mode == "auto":
        ink = outline_color(rgba, edge)
        if ink is None:
            return out
        edge &= np.all(rgba[:, :, :3] == ink, axis=2)
        inside = mask & ~np.all(rgba[:, :, :3] == ink, axis=2)
    else:
        inside = mask & inner
    if not inside.any() or not edge.any():
        return out
    dist, (iy, ix) = distance_transform_edt(~inside, return_indices=True)
    ey, ex = np.nonzero(edge & (dist <= NEAR))
    sy, sx = iy[ey, ex], ix[ey, ex]
    # Prefer the pixel straight in from the outline (away from the ground next to it), so a thin strip under the
    # outline keeps its own colour instead of the region beside it; else the nearest inside pixel.
    h, w = mask.shape
    dy = np.zeros(ey.size, np.int64)
    dx = np.zeros(ey.size, np.int64)
    for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ey + oy, ex + ox
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        outside = ~ok
        outside[ok] = ~mask[ny[ok], nx[ok]]
        dy -= oy * outside
        dx -= ox * outside
    cy, cx = ey + np.sign(dy), ex + np.sign(dx)
    ok = (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w) & ((dy != 0) | (dx != 0))
    ok[ok] = inside[cy[ok], cx[ok]]
    sy[ok], sx[ok] = cy[ok], cx[ok]
    out[ey, ex, :3] = rgba[sy, sx, :3]
    return out


def _fit(rgba: np.ndarray, mask: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour resize (no new colours) to w x h."""
    if rgba.shape[:2] == (h, w):
        return rgba, mask
    ys = (np.arange(h) * rgba.shape[0] // h).clip(0, rgba.shape[0] - 1)
    xs = (np.arange(w) * rgba.shape[1] // w).clip(0, rgba.shape[1] - 1)
    return rgba[ys][:, xs], mask[ys][:, xs]


def prepare(
    views: dict[str, Image.Image | np.ndarray],
    key: tuple[int, int, int] | None = None,
    outline: str = "auto",
    fit: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each view cropped to its subject, sizes checked, outline stripped, the side and back mirrored: name -> (rgba, mask)."""
    missing = [v for v in VIEWS if v not in views]
    if missing:
        raise ViewsError(f"missing view(s): {', '.join(missing)}")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in VIEWS:
        rgba, alpha = _rgba(views[name])
        rgba, mask = crop(rgba, subject_mask(rgba, key, alpha), name)
        out[name] = (rgba, mask)
    tw, tl = out["top"][0].shape[1], out["top"][0].shape[0]
    sh = out["side"][0].shape[0]
    want = {"side": (tl, sh), "front": (tw, sh), "back": (tw, sh)}
    for name, (w, h) in want.items():
        got = out[name][0].shape[1], out[name][0].shape[0]
        if got != (w, h):
            if not fit:
                raise ViewsError(
                    f"the {name} view is {got[0]}x{got[1]} but must be {w}x{h} to match the top view "
                    f"({tw} wide, {tl} long) and the side view ({sh} high); --fit resizes it"
                )
            out[name] = _fit(*out[name], w, h)
    for name in VIEWS:
        rgba, mask = out[name]
        out[name] = (strip_outline(rgba, mask, outline), mask)
    for name in ("side", "back"):
        rgba, mask = out[name]
        out[name] = (rgba[:, ::-1].copy(), mask[:, ::-1].copy())
    return out


def carve(prepared: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The solid cells and their colours: (solid[H, L, W], rgba[H, L, W, 4]), layer 0 at the bottom, row 0 of each
    layer the back of the object."""
    top, top_m = prepared["top"]
    side, side_m = prepared["side"]
    front, front_m = prepared["front"]
    back, back_m = prepared["back"]
    length, width = top_m.shape
    height = side_m.shape[0]
    rows = np.arange(height)[::-1]  # layer h is side/front/back row H-1-h
    s = side_m[rows]  # [H, L]
    f = front_m[rows] & back_m[rows]  # [H, W]
    solid = top_m[None, :, :] & s[:, :, None] & f[:, None, :]

    color = np.zeros((height, length, width, 4), np.uint8)
    # Side colour first, everywhere; front/back where the layer's front/back is open; top where it shows from above.
    color[:] = side[rows][:, :, None, :]
    pad = np.pad(solid, ((0, 0), (1, 1), (0, 0)), constant_values=False)
    open_back = solid & ~pad[:, :-2, :]  # nothing behind it (row - 1)
    open_front = solid & ~pad[:, 2:, :]  # nothing in front of it (row + 1)
    fr = np.broadcast_to(front[rows][:, None, :, :], color.shape)
    bk = np.broadcast_to(back[rows][:, None, :, :], color.shape)
    color[open_back] = bk[open_back]
    color[open_front] = fr[open_front]
    any_solid = solid.any(axis=0)
    top_layer = height - 1 - np.argmax(solid[::-1], axis=0)  # highest solid layer per (row, col)
    lz, lx = np.nonzero(any_solid)
    color[top_layer[lz, lx], lz, lx] = top[lz, lx]
    color[..., 3] = np.where(solid, 255, 0)
    color[~solid] = 0
    return solid, color


def palette(prepared: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    px = [rgba[mask][:, :3] for rgba, mask in prepared.values()]
    return np.unique(np.concatenate(px), axis=0)


def stack_views(prepared: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    solid, color = carve(prepared)
    height, length, width = solid.shape
    res = max(length, width)
    oz, ox = (res - length) // 2, (res - width) // 2
    slices = []
    for h in range(height):
        sl = np.zeros((res, res, 4), np.uint8)
        sl[oz : oz + length, ox : ox + width] = color[h]
        slices.append(sl)
    pitch = 1.0 / res
    ext = [width * pitch, height * pitch, length * pitch]
    return {
        "slices": slices,
        "pitch": pitch,
        "origin": [-0.5, 0.0, -0.5],
        "resolution": int(res),
        "layers": height,
        "bounds": [[-ext[0] / 2, 0.0, -ext[2] / 2], [ext[0] / 2, ext[1], ext[2] / 2]],
        "extents": ext,
        "surface_voxels": int(solid.sum()),
        "filled": True,
        "size": {"width": int(width), "length": int(length), "height": int(height)},
        "colors": int(len(palette(prepared))),
    }


def read_views(paths: dict[str, Path]) -> dict[str, Image.Image]:
    out = {}
    for name in VIEWS:
        p = paths[name]
        if not p.is_file():
            raise ViewsError(f"no {name} view at {p}")
        out[name] = Image.open(p)
    return out


def write_stack(result: dict, dest: Path, source: str, stem: str) -> dict:
    """write_outputs, then the views' own stack.json fields (as stack_image.py does)."""
    result["stem"] = stem
    result["source"] = source
    write_outputs(result, dest, Path(source).name or source)
    meta_path = dest / "stack.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["maker"] = "views"
    meta["size"] = result["size"]
    meta["colors"] = result["colors"]
    meta["defaults"] = {
        **meta.get("defaults", {}),
        "texels": result["resolution"],
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _key(text: str | None) -> tuple[int, int, int] | None:
    if not text:
        return None
    parts = [int(v) for v in text.replace("#", "").split(",")] if "," in text else None
    if parts is None:
        h = text.lstrip("#")
        parts = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    if len(parts) != 3:
        raise ViewsError(f"--key wants r,g,b or #rrggbb, got {text}")
    return tuple(parts)  # type: ignore[return-value]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", required=True, help="out/<stem>/ folder")
    p.add_argument("--root", type=Path, default=root, help="factory root that holds out/")
    for name in VIEWS:
        p.add_argument(f"--{name}", type=Path, default=None, help=f"the {name} view (default out/<stem>/flat/pixel/{name}.png)")
    p.add_argument("--key", default=None, help="ground colour of views without transparency, r,g,b or #rrggbb")
    p.add_argument("--outline", choices=("auto", "on", "off"), default="auto", help="strip the views' dark outline")
    p.add_argument("--fit", action="store_true", help="resize side/front/back (nearest) to match the top view")
    args = p.parse_args()
    asset = args.root / "out" / args.stem
    paths = {n: (getattr(args, n) or asset / "flat" / "pixel" / f"{n}.png").expanduser() for n in VIEWS}
    try:
        prepared = prepare(read_views(paths), key=_key(args.key), outline=args.outline, fit=args.fit)
        result = stack_views(prepared)
    except ViewsError as e:
        print(f"stack_views: {e}", file=sys.stderr)
        return 2
    if result["colors"] > PALETTE_WARN:
        print(f"stack_views: warning: {result['colors']} colours across the views; pixel art views share a small palette",
              file=sys.stderr)
    try:
        source = str(paths["top"].resolve().parent.relative_to(asset.resolve())).replace("\\", "/") or "."
    except ValueError:
        source = str(paths["top"].resolve().parent)
    write_stack(result, asset / "stack", source, args.stem)
    print(json.dumps({"ok": True, "layers": result["layers"], "resolution": result["resolution"], **result["size"],
                      "colors": result["colors"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
