"""The pixel art sheet: split it into its four views, check them, and clean them into true small pixel art.

The local image model draws "pixel art" as a soft 1152 px picture: big fake pixels (cells) a dozen screen pixels
across, with blurred rims, a few tints inside each flat colour and the magenta ground bleeding into the outline.
A sprite stack wants the real thing: one pixel per cell, one small palette, hard edges. Cleaning:

1. Cut each view out on the magenta key and crop it to its subject.
2. One palette for all four views: median cut over the views' inner pixels (their rims are part ground), about
   PALETTE colours, never dithered.
3. Snap every pixel to that palette, then find the drawing's cell grid: the cell size (shared by the four views,
   the model draws them at one pixel size) and each view's own offset, as the grid whose lines catch the most
   colour changes beyond what lines placed at random would.
4. One colour per cell, never blended: the most common palette colour in the cell's middle. A cell is ground when
   most of its middle is ground.
5. Sizes: the top view is the longest drawing top to bottom, the side view the widest, and the views must agree on
   length (top height, side width), width (top, front and back widths) and height (side, front and back heights).
   A view a cell or two off is fitted to the agreed size, nearest-neighbour (pixels kept, never blended).

When no grid stands out (the model drew a smooth picture) the cell size falls back to the subject's length over
TARGET, and pixel.json says so.

Writes flat/pixel/{top,side,front,back}.png (RGBA, transparent ground), flat/pixel/preview.png (the four views
enlarged, for people) and flat/pixel/pixel.json. scripts/stack_views.py carves the stack from the four PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from flat2d.pose import PANELS, PIXEL_PANELS

PALETTE = 24
SEED_COLOURS = 256  # colour bins kept as candidates before merging down to PALETTE
BIN = 16  # levels per channel in one candidate colour bin
BIN_SHARE = 0.0003  # a candidate bin holds at least this share of the views' inner pixels
MERGE_RGB = 48  # palette colours closer than this (summed RGB) are one colour
TARGET = 32  # cells along the subject's length when no grid is found
MIN_CELL, MAX_CELL = 3.0, 64.0
CELL_STEP = 0.25
GRID_MIN_SCORE = 0.2  # a grid must catch this much more of the colour changes than random lines
MAX_LONG = 72  # a "grid" finer than this many cells along the length (or 1.5x the target) is noise, not pixels
EDGE_MIN = 60  # summed RGB step that counts as a colour change for the grid
OUTLINE_SHARE = 0.4  # share of the drawing's rim in the darkest colour that means it has an outline
CORE = 0.2  # share of a cell cut from each side before its colour is voted
SPECK = 0.002  # blobs smaller than this share of a panel are dust, not the subject
PIECE_SHARE = 0.15
AGREE = 0.12  # sizes this far apart (share) are a flaw before fitting
FIT_CELLS = 2  # ... and a view within this many cells of the agreed size is fitted without a note


# ---------------------------------------------------------------- split and check the sheet

def ground_mask(rgb: np.ndarray, key: dict) -> np.ndarray:
    from flat2d.views import ground_of

    return ground_of(rgb, key)


def magenta_shade(rgb: np.ndarray) -> np.ndarray:
    """The ground's own hue at any brightness: the soft shadow the model draws under the subject despite being told
    not to (a darker magenta band under the car) is ground too. Red and blue stay subject: magenta has red and blue
    both well above green and close to each other."""
    a = rgb.astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lo, hi = np.minimum(r, b), np.maximum(r, b)
    return (lo >= 40) & (g * 10 < lo * 6) & ((hi - lo) * 100 < hi * 40)


def subject_mask(rgb: np.ndarray, key: dict) -> np.ndarray:
    """Not ground (nor the ground's hue in shadow), without dust: blobs under SPECK of the panel."""
    from scipy import ndimage

    fg = ~ground_mask(rgb, key) & ~magenta_shade(rgb)
    labels, n = ndimage.label(fg)
    if n == 0:
        return fg
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(4, SPECK * fg.size)
    keep[0] = False
    return keep[labels]


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def split(image: Image.Image, key: dict) -> dict[str, list[int]]:
    """{view id: [x0, y0, x1, y1]} on the sheet, at its magenta gaps (or in quarters), in PIXEL_PANELS order."""
    from flat2d.views import split_sheet

    rgb = np.asarray(image.convert("RGB"))
    got = split_sheet(image, key, "grid", subject_mask(rgb, key))
    return {pid: got["boxes"][old] for pid, old in zip(PIXEL_PANELS, PANELS)} | {"_how": got["how"]}


def check_sheet(image: Image.Image, key: dict) -> dict[str, Any]:
    """Split the sheet and check its views: each holds one subject; the top view is the tallest drawing and the
    side view the widest; the front and back are within AGREE of the proportion the side and top views allow
    (height over width: side height / side length x top length / top width). Returns {boxes, how, sizes {id: [w, h]},
    stretch {front, back}, flaws, notes, panels {id: crop}}."""
    from scipy import ndimage

    from flat2d.views import _close_gaps

    boxes = split(image, key)
    how = boxes.pop("_how")
    rgb = np.asarray(image.convert("RGB"))
    flaws: list[str] = []
    notes: list[str] = [] if how == "gaps" else ["quarters"]
    sizes: dict[str, list[int]] = {}
    panels: dict[str, Image.Image] = {}
    from flat2d.parts import ground_key

    keys: dict[str, dict] = {}
    for pid, (x0, y0, x1, y1) in boxes.items():
        crop = rgb[y0:y1, x0:x1]
        panels[pid] = Image.fromarray(crop)
        # Each panel's own ground: a graded ground is darker on the bottom row than the sheet's border median.
        keys[pid] = ground_key(panels[pid])
        fg = subject_mask(crop, keys[pid])
        box = bbox(fg)
        if box is None or fg.mean() < 0.01:
            flaws.append(f"empty:{pid}")
            sizes[pid] = [0, 0]
            continue
        labels, n = ndimage.label(_close_gaps(fg, max(2, int(0.02 * crop.shape[0]))))
        big = sorted(np.bincount(labels.ravel())[1:], reverse=True)
        if len(big) > 1 and big[1] > PIECE_SHARE * big[0]:
            flaws.append(f"pieces:{pid}")
        sizes[pid] = [box[2] - box[0], box[3] - box[1]]
    if not any(f.startswith("empty") for f in flaws):
        if sizes["top"][1] < max(s[1] for s in sizes.values()) * 0.97:
            flaws.append("top_short")
        if sizes["side"][0] < max(s[0] for s in sizes.values()) * 0.97:
            flaws.append("side_narrow")
        # A top view drawn nose-left still passes both of those (a car is wider than it is tall): the top view
        # runs top to bottom, the side view left to right.
        if sizes["top"][1] <= sizes["top"][0]:
            flaws.append("top_sideways")
        if sizes["side"][0] <= sizes["side"][1]:
            flaws.append("side_upright")
    stretch = sheet_stretch(sizes)
    for pid, v in stretch.items():
        if abs(v) > AGREE:
            flaws.append(f"stretch:{pid}")
    return {"boxes": boxes, "how": how, "sizes": sizes, "stretch": stretch, "flaws": flaws, "notes": notes,
            "panels": panels, "keys": keys}


def sheet_stretch(sizes: dict[str, list[int]]) -> dict[str, float]:
    """How much taller (+) or shorter (-) for its width the front and back views are drawn than the side and top
    views allow, as a share ({} when a view is empty)."""
    (tw, tl), (sl, sh) = sizes["top"], sizes["side"]
    if min(tw, tl, sl, sh) <= 0:
        return {}
    allowed = (sh / sl) * (tl / tw)
    return {pid: round((sizes[pid][1] / sizes[pid][0]) / allowed - 1.0, 3)
            for pid in ("front", "back") if min(sizes[pid]) > 0}


REDRAW_FLAWS = ("empty", "pieces", "top_short", "side_narrow", "top_sideways", "side_upright", "stretch")


def redrawable(flaws: list[str]) -> list[str]:
    return [f for f in flaws if f.split(":", 1)[0] in REDRAW_FLAWS]


# ---------------------------------------------------------------- palette and grid

def build_palette(views: dict[str, tuple[np.ndarray, np.ndarray]], colours: int = PALETTE) -> np.ndarray:
    """One palette (at most `colours` x 3 uint8) for every view, from their inner pixels (two pixels in from the
    ground, which the rim is blended with). Median cut first finds SEED_COLOURS candidates; then the two closest
    are merged, the rarer giving way to the commoner (never averaged), until `colours` are left and none are closer
    than MERGE_RGB. Merging by distance, not by count, keeps a small distinct colour (a tail light) that a plain
    median cut gives up for the many tints of a big flat body."""
    from scipy import ndimage

    picks = []
    for rgb, fg in views.values():
        inner = ndimage.binary_erosion(fg, iterations=2)
        if inner.sum() < 16:
            inner = fg
        picks.append(rgb[inner])
    px = np.concatenate(picks) if picks else np.zeros((1, 3), np.uint8)
    # Candidates: colour bins BIN levels wide per channel holding at least BIN_SHARE of the pixels (the blurred
    # in-betweens on rims fall under it), each its pixels' mean (the model's noise inside one flat colour).
    q = px.astype(np.int64) // BIN
    code = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    uniq, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)[:SEED_COLOURS]
    floor = max(4, int(BIN_SHARE * len(px)))
    order = [i for i in order if counts[i] >= floor] or [int(order[0])]
    sums = np.zeros((len(uniq), 3), np.float64)
    np.add.at(sums, inv.ravel(), px.astype(np.float64))
    cols = np.array([(sums[i] / counts[i]).round() for i in order], np.float64)
    count = np.array([float(counts[i]) for i in order])
    live = np.ones(len(order), bool)
    d = np.abs(cols[:, None, :] - cols[None, :, :]).sum(-1)
    np.fill_diagonal(d, np.inf)
    while live.sum() > 1:
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        if live.sum() <= colours and d[i, j] >= MERGE_RGB:
            break
        keep, drop = (i, j) if count[i] >= count[j] else (j, i)
        count[keep] += count[drop]
        live[drop] = False
        d[drop, :] = np.inf
        d[:, drop] = np.inf
    return cols[live].astype(np.uint8)


def snap(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Each pixel's nearest palette index (squared RGB distance)."""
    flat = rgb.reshape(-1, 3).astype(np.int32)
    p = pal.astype(np.int32)
    best = np.zeros(len(flat), np.int32)
    dist = np.full(len(flat), np.iinfo(np.int32).max, np.int64)
    for i, c in enumerate(p):
        d = ((flat - c) ** 2).sum(1)
        better = d < dist
        best[better], dist[better] = i, d[better]
    return best.reshape(rgb.shape[:2])


def edges(rgb: np.ndarray, fg: np.ndarray, axis: int) -> np.ndarray:
    """Colour change between position i-1 and i along axis (1: columns, 0: rows), summed across the view; index 0
    is 0. Ground counts as black so the step off the ground is a change; steps under EDGE_MIN (the model's tints
    and noise inside one flat colour) count for nothing."""
    a = np.where(fg[..., None], rgb.astype(np.int32), 0)
    d = np.abs(np.diff(a, axis=axis)).sum(-1)
    d = np.where(d >= EDGE_MIN, d, 0)
    prof = d.sum(0) if axis == 1 else d.sum(1)
    return np.concatenate([[0], prof]).astype(np.float64)


def _line_fit(e: np.ndarray, s: float) -> tuple[float, float]:
    """(score, offset) of the best grid of period s along one edge profile: the share of colour changes within
    0.75 px of a line, less the share lines at random would catch."""
    total = e.sum()
    if total <= 0:
        return 0.0, 0.0
    pos = np.arange(len(e), dtype=np.float64)
    offs = np.arange(0.0, s, 0.5)
    d = np.mod(pos[None, :] - offs[:, None], s)
    near = np.minimum(d, s - d) <= 0.75
    caught = (near * e[None, :]).sum(1) / total
    i = int(np.argmax(caught))
    return float(caught[i] - min(1.0, 1.5 / s)), float(offs[i])


def find_grid(views: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """The cell size shared by the views and each view's (x, y) offset: {cell, score, offsets {id: [ox, oy]}}.
    views: {id: (rgb, subject mask)}, cropped to the subject."""
    profiles = {pid: (edges(rgb, fg, 1), edges(rgb, fg, 0)) for pid, (rgb, fg) in views.items()}
    best = None
    for s in np.arange(MIN_CELL, MAX_CELL + 1e-9, CELL_STEP):
        total, offs = 0.0, {}
        for pid, (ex, ey) in profiles.items():
            sx, ox = _line_fit(ex, s)
            sy, oy = _line_fit(ey, s)
            total += sx + sy
            offs[pid] = [ox, oy]
        score = total / (2 * max(1, len(profiles)))
        if best is None or score > best["score"] + 1e-9:
            best = {"cell": float(s), "score": float(score), "offsets": offs}
    return best or {"cell": 0.0, "score": 0.0, "offsets": {}}


# ---------------------------------------------------------------- one colour per cell

def to_cells(idx: np.ndarray, fg: np.ndarray, cell: float | tuple[float, float], off: tuple[float, float],
             shape: tuple[int, int] | None = None) -> np.ndarray:
    """The view as cells: an int array of palette indices, -1 for ground. Each cell votes on its middle (CORE cut
    from each side): ground when most of it is ground, else its most common palette colour.
    cell: one size, or (width, height) of a cell; shape (rows, cols): exactly that many cells from the offset."""
    h, w = idx.shape
    cx, cy = (cell, cell) if np.isscalar(cell) else cell
    ox, oy = off
    if shape is None:
        xs = np.arange(ox - cx * np.ceil(ox / cx), w + cx, cx)
        ys = np.arange(oy - cy * np.ceil(oy / cy), h + cy, cy)
    else:
        xs = ox + cx * np.arange(shape[1] + 1)
        ys = oy + cy * np.arange(shape[0] + 1)
    out = np.full((len(ys) - 1, len(xs) - 1), -1, np.int32)
    mx, my = cx * CORE, cy * CORE
    for r in range(len(ys) - 1):
        a, b = int(round(ys[r] + my)), int(round(ys[r + 1] - my))
        a, b = max(0, a), min(h, max(b, a + 1))
        if a >= h:
            continue
        for c in range(len(xs) - 1):
            l, rr = int(round(xs[c] + mx)), int(round(xs[c + 1] - mx))
            l, rr = max(0, l), min(w, max(rr, l + 1))
            if l >= w:
                continue
            f = fg[a:b, l:rr]
            if f.size == 0 or f.mean() < 0.5:
                continue
            vals = idx[a:b, l:rr][f]
            out[r, c] = int(np.bincount(vals).argmax())
    return out


def drawn_outline(views: dict[str, tuple[np.ndarray, np.ndarray]], idx: dict[str, np.ndarray],
                  pal: np.ndarray) -> int | None:
    """The palette index of the drawing's dark outline, or None when it has none. The outline is a few screen pixels
    wide, so at one colour per cell it mostly loses the vote; it is put back as one cell on the silhouette
    (add_outline) when most of the drawing's rim is the darkest palette colour."""
    from scipy import ndimage

    lum = pal.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    dark = int(np.argmin(lum))
    if lum[dark] > 70:
        return None
    rim_hits, rim_all = 0, 0
    for pid, (_, fg) in views.items():
        rim = fg & ~ndimage.binary_erosion(fg, iterations=3)
        rim_hits += int((idx[pid][rim] == dark).sum())
        rim_all += int(rim.sum())
    return dark if rim_all and rim_hits / rim_all >= OUTLINE_SHARE else None


def add_outline(cells: np.ndarray, colour: int) -> np.ndarray:
    """Every subject cell next to ground (or the edge) takes the outline colour."""
    on = cells >= 0
    pad = np.pad(on, 1, constant_values=False)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    out = cells.copy()
    out[on & ~inner] = colour
    return out


def trim(cells: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(cells >= 0)
    if len(xs) == 0:
        return cells[:0, :0]
    return cells[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def fit(cells: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbour resize of a cell array to w x h (rows or columns repeated or dropped, never blended)."""
    ch, cw = cells.shape
    if (cw, ch) == (w, h) or cw == 0 or ch == 0:
        return cells
    rows = np.minimum((np.arange(h) + 0.5) * ch / h, ch - 1).astype(int)
    cols = np.minimum((np.arange(w) + 0.5) * cw / w, cw - 1).astype(int)
    return cells[rows][:, cols]


def rgba(cells: np.ndarray, pal: np.ndarray) -> Image.Image:
    h, w = cells.shape
    out = np.zeros((h, w, 4), np.uint8)
    on = cells >= 0
    out[on, :3] = pal[cells[on]]
    out[on, 3] = 255
    return Image.fromarray(out, "RGBA")


def agree(sizes: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """The agreed length, width and height (cells) and how far each view's measure is from it."""
    L = [sizes["top"][1], sizes["side"][0]]
    W = [sizes["top"][0], sizes["front"][0], sizes["back"][0]]
    H = [sizes["side"][1], sizes["front"][1], sizes["back"][1]]
    want = {"length": int(round(float(np.median(L)))), "width": int(round(float(np.median(W)))),
            "height": int(round(float(np.median(H))))}
    spread = {k: round((max(v) - min(v)) / max(1, want[k]), 3) for k, v in (("length", L), ("width", W),
                                                                           ("height", H))}
    return {"agreed": want, "measured": {"length": L, "width": W, "height": H}, "spread": spread}


def clean(panels: dict[str, Image.Image], key: dict | None = None, *, colours: int = PALETTE,
          target: int = TARGET) -> dict:
    """The four drawn views (crops of the sheet on its magenta ground, PIXEL_PANELS ids) as true pixel art.
    key: one ground key for every panel; None reads each panel's own (flat2d.parts.ground_key).
    Returns {views {id: RGBA image}, palette, grid, sizes, checks, flaws, notes, ok}. ok is False when the views
    disagree by more than AGREE on a size: the views are still written, fitted, but must not be carved."""
    from flat2d.parts import ground_key

    views: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid in PIXEL_PANELS:
        rgb = np.asarray(panels[pid].convert("RGB"))
        fg = subject_mask(rgb, key or ground_key(panels[pid]))
        box = bbox(fg)
        if box is None:
            raise ValueError(f"the {pid} view is empty")
        x0, y0, x1, y1 = box
        views[pid] = (rgb[y0:y1, x0:x1], fg[y0:y1, x0:x1])
    pal = build_palette(views, colours)
    idx = {pid: snap(rgb, pal) for pid, (rgb, _) in views.items()}
    grid = find_grid(views)
    px = {pid: (int(v[0].shape[1]), int(v[0].shape[0])) for pid, v in views.items()}
    length_px = max(px["top"][1], px["side"][0])
    notes: list[str] = []
    flaws: list[str] = []
    fitted: dict[str, Any] = {}
    max_long = max(MAX_LONG, int(round(1.5 * target)))
    if grid["score"] >= GRID_MIN_SCORE and length_px / max(grid["cell"], 1e-6) <= max_long:
        how = "found"
        cells = {pid: trim(to_cells(idx[pid], views[pid][1], grid["cell"], tuple(grid["offsets"][pid])))
                 for pid in PIXEL_PANELS}
        raw_sizes = {pid: (int(c.shape[1]), int(c.shape[0])) for pid, c in cells.items()}
        sizes = agree(raw_sizes)
        for k, v in sizes["spread"].items():
            if v > AGREE:
                flaws.append(f"disagree:{k}")
        a = sizes["agreed"]
        want = {"top": (a["width"], a["length"]), "side": (a["length"], a["height"]),
                "front": (a["width"], a["height"]), "back": (a["width"], a["height"])}
        for pid in PIXEL_PANELS:
            if raw_sizes[pid] != want[pid]:
                fitted[pid] = {"from": list(raw_sizes[pid]), "to": list(want[pid])}
                off = max(abs(raw_sizes[pid][0] - want[pid][0]), abs(raw_sizes[pid][1] - want[pid][1]))
                if off > FIT_CELLS:
                    notes.append(f"the {pid} view was {raw_sizes[pid][0]}x{raw_sizes[pid][1]} cells, fitted to "
                                 f"{want[pid][0]}x{want[pid][1]}")
                cells[pid] = fit(cells[pid], *want[pid])
        cell_px: Any = round(grid["cell"], 2)
    else:
        # No grid worth keeping (or one far finer than a sprite): cut each view straight into the agreed cells.
        # The model draws every view at its own scale, so only proportions are compared: length and height from
        # the side view, width from the top view; front and back are cut to width x height, and how far their
        # drawn proportions are from that is their stretch.
        how = "fallback"
        L = int(target)
        Hc = max(1, int(round(L * px["side"][1] / px["side"][0])))
        Wc = max(1, int(round(L * px["top"][0] / px["top"][1])))
        want = {"top": (Wc, L), "side": (L, Hc), "front": (Wc, Hc), "back": (Wc, Hc)}
        cells = {pid: to_cells(idx[pid], views[pid][1], (px[pid][0] / want[pid][0], px[pid][1] / want[pid][1]),
                               (0.0, 0.0), shape=(want[pid][1], want[pid][0])) for pid in PIXEL_PANELS}
        raw_sizes = {pid: want[pid] for pid in PIXEL_PANELS}
        stretch = {}
        for pid in ("front", "back"):
            drawn = px[pid][1] / px[pid][0]
            stretch[pid] = round(drawn / (Hc / Wc) - 1.0, 3)
            if abs(stretch[pid]) > AGREE:
                flaws.append(f"disagree:{pid}")
                notes.append(f"the {pid} view is drawn {abs(stretch[pid]):.0%} {'taller' if stretch[pid] > 0 else 'shorter'}"
                             f" for its width than the side and top views allow; squeezed to {Wc}x{Hc}")
        sizes = {"agreed": {"length": L, "width": Wc, "height": Hc}, "drawn_px": {k: list(v) for k, v in px.items()},
                 "stretch": stretch}
        found = length_px / max(grid["cell"], 1e-6)
        why = (f"its pixels are finer than a sprite (about {found:.0f} along the length)"
               if grid["score"] >= GRID_MIN_SCORE else f"no clear pixel grid (score {grid['score']:.2f})")
        notes.insert(0, f"{why}; each view cut straight into {L} cells along the length")
        grid = {**grid, "found_cell": grid["cell"]}
        cell_px = {pid: [round(px[pid][0] / want[pid][0], 2), round(px[pid][1] / want[pid][1], 2)]
                   for pid in PIXEL_PANELS}
        grid["offsets"] = {pid: [0.0, 0.0] for pid in PIXEL_PANELS}
    outline = drawn_outline(views, idx, pal)
    if outline is not None:
        cells = {pid: add_outline(c, outline) for pid, c in cells.items()}
    used = sorted({int(v) for c in cells.values() for v in np.unique(c) if v >= 0})
    remap = np.full(len(pal) + 1, -1, np.int32)
    remap[used] = np.arange(len(used))
    pal_used = pal[used]
    cells = {pid: np.where(c >= 0, remap[np.maximum(c, 0)], -1) for pid, c in cells.items()}
    images = {pid: rgba(cells[pid], pal_used) for pid in PIXEL_PANELS}
    return {"views": images, "palette": ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in pal_used],
            "grid": {"how": how, "cell_px": cell_px, "score": round(grid["score"], 3),
                     "offsets": {k: [round(v[0], 2), round(v[1], 2)] for k, v in grid["offsets"].items()},
                     **({"found_cell_px": round(grid["found_cell"], 2)} if "found_cell" in grid else {})},
            "sizes": {"drawn_cells": {k: list(v) for k, v in raw_sizes.items()}, **sizes, "fitted": fitted,
                      "final": {pid: list(images[pid].size) for pid in PIXEL_PANELS}},
            "outline": outline is not None, "flaws": flaws, "notes": notes, "ok": not flaws}


PREVIEW_SCALE = 8


def preview(images: dict[str, Image.Image], scale: int = PREVIEW_SCALE) -> Image.Image:
    """The four cleaned views enlarged (nearest) in the sheet's 2x2 order on a light ground, for people."""
    big = {pid: im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST) for pid, im in
           images.items()}
    pad = 2 * scale
    cw = max(big["top"].width, big["front"].width) + max(big["side"].width, big["back"].width) + 3 * pad
    ch = max(big["top"].height, big["side"].height) + max(big["front"].height, big["back"].height) + 3 * pad
    out = Image.new("RGBA", (cw, ch), (233, 238, 248, 255))
    col = max(big["top"].width, big["front"].width) + 2 * pad
    row = max(big["top"].height, big["side"].height) + 2 * pad
    for pid, (x, y) in {"top": (pad, pad), "side": (col, pad), "front": (pad, row), "back": (col, row)}.items():
        out.alpha_composite(big[pid], (x, y))
    return out


def write(folder: Path, result: dict, meta: dict | None = None) -> dict:
    """flat/pixel/<id>.png, preview.png and pixel.json under the asset's flat/ folder. Returns pixel.json."""
    out = Path(folder) / "pixel"
    out.mkdir(parents=True, exist_ok=True)
    for pid, im in result["views"].items():
        im.save(out / f"{pid}.png", format="PNG")
    preview(result["views"]).save(out / "preview.png", format="PNG")
    doc = {"version": 1, "views": {pid: f"{pid}.png" for pid in PIXEL_PANELS},
           "orientation": {"top": "looking down, front toward the bottom edge",
                           "side": "front toward the left edge", "front": "facing the viewer",
                           "back": "seen from behind"},
           **{k: result[k] for k in ("ok", "outline", "palette", "grid", "sizes", "flaws", "notes")}, **(meta or {})}
    (out / "pixel.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def clean_folder(folder: Path, *, colours: int = PALETTE, target: int = TARGET) -> dict:
    """Clean flat/views/<id>.png (written by optional.generate_pixel_views) into flat/pixel/. Returns pixel.json."""
    folder = Path(folder)
    panels = {pid: Image.open(folder / "views" / f"{pid}.png").convert("RGB") for pid in PIXEL_PANELS}
    result = clean(panels, None, colours=colours, target=target)
    return write(folder, result, {"colours_asked": colours})
