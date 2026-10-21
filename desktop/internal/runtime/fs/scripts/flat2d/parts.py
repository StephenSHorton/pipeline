"""Split a flat-background still into part crops. No model."""
from __future__ import annotations

import array
from pathlib import Path

from PIL import Image

MIN_AREA = 24
PAD = 4
TOL = 36
MAX_TOL = 96
BORDER = 8
EDGE = 3  # px of a cut edge that get the key color pulled out of them
SHADOW = 0.25  # darkest shadow of a saturated key that still counts as ground
BLOTCH_SHARE = 50
MISSED_SHARE = 200
SPECK_SHARE = 400  # with a segmenter mask, a piece under 1/400 of the biggest is a speck
SPILL_SLACK = 24  # how far outside 0..255 an unmixed color may land and still count
SPILL_LIGHT = 110  # mean brightness the unmixed color needs to read as glass
GLASS_FIT = 30  # how close a pixel must be to "gray plus key" to read as glass
GLASS_SHARE = 0.08  # share of glass-like pixels with bright highlights that marks a piece as glass
GLASS_THROUGH = 0.9  # median ground brightness, as a share of the key, of glass seen straight through  # a keyed piece the segmenter skipped is kept from 1/200 of the image  # a ground-colored piece under 1/50 of the biggest is ground


def ground_key(im: Image.Image) -> dict:
    """The ground a generated still actually painted: the border's median color, and a
    tolerance wide enough for its noise. Models rarely paint the exact #ff00ff asked for."""
    rgb = im.convert("RGB")
    w, h = rgb.size
    px = rgb.load()
    b = max(1, min(BORDER, w // 4, h // 4))
    cols = list(range(b)) + list(range(w - b, w))
    rows = list(range(b)) + list(range(h - b, h))
    pts = [px[x, y] for y in range(h) for x in cols] + [px[x, y] for x in range(b, w - b) for y in rows]
    med = tuple(sorted(c[i] for c in pts)[len(pts) // 2] for i in range(3))
    spread = sorted(abs(c[0] - med[0]) + abs(c[1] - med[1]) + abs(c[2] - med[2]) for c in pts)
    p90 = spread[int(len(spread) * 0.9)]
    return {"rgb": list(med), "tol": max(TOL, min(MAX_TOL, round(p90 * 1.5)))}


def _corner_bg(raw: bytes, w: int, h: int) -> tuple[int, int, int]:
    pts = (0, w - 1, (h - 1) * w, (h - 1) * w + (w - 1))
    acc = [0, 0, 0]
    for i in pts:
        o = i * 3
        acc[0] += raw[o]
        acc[1] += raw[o + 1]
        acc[2] += raw[o + 2]
    return acc[0] // 4, acc[1] // 4, acc[2] // 4


def _foreground(
    im: Image.Image, key: tuple[int, int, int] | None = None, tol: int = TOL
) -> tuple[bytearray, int, int]:
    rgba = im.convert("RGBA")
    w, h = rgba.size
    raw = rgba.tobytes()
    alpha = bytearray(w * h)
    if key is None:
        useful = False
        for i in range(w * h):
            if raw[i * 4 + 3] < 250:
                useful = True
                break
        if useful:
            for i in range(w * h):
                if raw[i * 4 + 3] > 16:
                    alpha[i] = 1
            return alpha, w, h
    rgb = im.convert("RGB").tobytes()
    bg = key if key is not None else _corner_bg(rgb, w, h)
    # A saturated key also casts darker copies of itself: shadows on the ground.
    # Those sit on the line from the key toward black.
    shade = key is not None and max(bg) - min(bg) > 96
    kk = float(bg[0] * bg[0] + bg[1] * bg[1] + bg[2] * bg[2]) or 1.0
    for i in range(w * h):
        o = i * 3
        r, g, b = rgb[o], rgb[o + 1], rgb[o + 2]
        if abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2]) <= tol:
            continue
        if shade:
            # Held to the base tolerance: a wide one would reach dark subjects too.
            t = (r * bg[0] + g * bg[1] + b * bg[2]) / kk
            t = SHADOW if t < SHADOW else (1.0 if t > 1.0 else t)
            if abs(r - t * bg[0]) + abs(g - t * bg[1]) + abs(b - t * bg[2]) <= TOL:
                continue
        alpha[i] = 1
    return alpha, w, h


def _near_ground(fg: bytearray, w: int, h: int, depth: int) -> bytearray:
    """Foreground pixels within depth steps of the ground."""
    near = bytearray(w * h)
    ring = []
    for i in range(w * h):
        if not fg[i]:
            continue
        x = i % w
        if (x > 0 and not fg[i - 1]) or (x + 1 < w and not fg[i + 1]) or (i >= w and not fg[i - w]) or (i + w < w * h and not fg[i + w]):
            near[i] = 1
            ring.append(i)
    for _ in range(depth - 1):
        nxt = []
        for i in ring:
            x = i % w
            for n in (i - 1 if x > 0 else -1, i + 1 if x + 1 < w else -1, i - w, i + w):
                if 0 <= n < w * h and fg[n] and not near[n]:
                    near[n] = 1
                    nxt.append(n)
        ring = nxt
    return near


def _components(
    fg: bytearray, w: int, h: int, min_area: int
) -> tuple[array.array, list[tuple[int, int, int, int, int, int]]]:
    seen = bytearray(w * h)
    labels = array.array("I", [0]) * (w * h)
    found: list[tuple[int, int, int, int, int, int]] = []
    label_id = 0
    for start in range(w * h):
        if not fg[start] or seen[start]:
            continue
        label_id += 1
        stack = [start]
        seen[start] = 1
        labels[start] = label_id
        area = 0
        minx = maxx = start % w
        miny = maxy = start // w
        while stack:
            i = stack.pop()
            area += 1
            x, y = i % w, i // w
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y
            for n in (i - 1, i + 1, i - w, i + w):
                if n < 0 or n >= w * h:
                    continue
                if (n == i - 1 and x == 0) or (n == i + 1 and x + 1 == w):
                    continue
                if fg[n] and not seen[n]:
                    seen[n] = 1
                    labels[n] = label_id
                    stack.append(n)
        if area >= min_area:
            found.append((area, minx, miny, maxx, maxy, label_id))
    found.sort(key=lambda item: item[0], reverse=True)
    return labels, found


def crops_from_masks(im: Image.Image, masks: list) -> list[Image.Image]:
    """Crop RGBA parts from boolean masks (row-major, true = keep). Largest first."""
    src = im.convert("RGBA")
    w, h = src.size
    scored = []
    for mask in masks:
        area = 0
        minx, miny, maxx, maxy = w, h, 0, 0
        for y in range(h):
            row = mask[y]
            for x in range(w):
                if not row[x]:
                    continue
                area += 1
                if x < minx:
                    minx = x
                if x > maxx:
                    maxx = x
                if y < miny:
                    miny = y
                if y > maxy:
                    maxy = y
        if area >= MIN_AREA:
            scored.append((area, minx, miny, maxx, maxy, mask))
    scored.sort(key=lambda item: item[0], reverse=True)
    crops: list[Image.Image] = []
    for _area, minx, miny, maxx, maxy, mask in scored:
        x0 = max(0, minx - PAD)
        y0 = max(0, miny - PAD)
        x1 = min(w, maxx + 1 + PAD)
        y1 = min(h, maxy + 1 + PAD)
        crop = src.crop((x0, y0, x1, y1))
        pix = crop.load()
        for y in range(crop.size[1]):
            for x in range(crop.size[0]):
                if not mask[y0 + y][x0 + x]:
                    r, g, b, _a = pix[x, y]
                    pix[x, y] = (r, g, b, 0)
        crops.append(crop)
    return crops


def split_sam(im: Image.Image, checkpoint: Path) -> list[Image.Image]:
    """SAM 2 automatic masks. Raises if the package or checkpoint cannot run."""
    import numpy as np
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2("configs/sam2/sam2_hiera_s.yaml", str(checkpoint))
    gen = SAM2AutomaticMaskGenerator(model)
    rgb = np.array(im.convert("RGB"))
    records = gen.generate(rgb)
    masks = []
    for rec in records:
        seg = rec["segmentation"]
        masks.append([list(map(bool, row)) for row in seg])
    return crops_from_masks(im, masks)


def split_image(
    im: Image.Image,
    min_area: int = MIN_AREA,
    key: tuple[int, int, int] | None = None,
    tol: int = TOL,
    subject: bytearray | None = None,
) -> list[Image.Image]:
    """Return RGBA crops, largest first. Background is key, else alpha, else the corner color.
    With a key, the key color is pulled out of the pixels along each cut edge (the glow
    a model paints where the subject meets the ground).

    subject is a foreground mask from a segmenter (Apple Vision). It wins over the key,
    which only adds back big pieces the segmenter skipped entirely (a thin sword)."""
    fg, w, h = _foreground(im, key=key, tol=tol)
    if subject is not None and len(subject) == w * h and any(subject):
        fg = _with_missed(im, subject, fg, w, h, min_area, key, tol)
    labels, boxes = _components(fg, w, h, min_area)
    near = _near_ground(fg, w, h, EDGE) if key is not None else None
    if subject is not None and len(boxes) > 1:
        # Specks the segmenter leaves around a subject are not parts.
        boxes = boxes[:1] + [box for box in boxes[1:] if box[0] * SPECK_SHARE >= boxes[0][0]]
    if key is not None and len(boxes) > 1:
        px = im.convert("RGB").load()
        boxes = boxes[:1] + [box for box in boxes[1:] if not _ground_blotch(px, labels, w, box, boxes[0][0], key, tol)]
    crops: list[Image.Image] = []
    src = im.convert("RGBA")
    for _area, minx, miny, maxx, maxy, label_id in boxes:
        x0 = max(0, minx - PAD)
        y0 = max(0, miny - PAD)
        x1 = min(w, maxx + 1 + PAD)
        y1 = min(h, maxy + 1 + PAD)
        crop = src.crop((x0, y0, x1, y1))
        pix = crop.load()
        for y in range(crop.size[1]):
            for x in range(crop.size[0]):
                i = (y0 + y) * w + (x0 + x)
                r, g, b, a = pix[x, y]
                if labels[i] != label_id:
                    pix[x, y] = (r, g, b, 0)
                elif near is not None and near[i]:
                    pix[x, y] = (*_despill((r, g, b), key), a)
        if key is not None:
            crop = see_through(crop, key)
        crop.info["box"] = (x0, y0, x1, y1)
        crops.append(crop)
    return crops


def reading_order(crops: list[Image.Image]) -> list[Image.Image]:
    """Left to right, then the next row down. A piece joins the row when its middle is
    level with the row's first piece. Crops without a box keep their place at the end."""
    boxed = [c for c in crops if "box" in c.info]
    rest = [c for c in crops if "box" not in c.info]
    rows: list[list[Image.Image]] = []
    bottom = None
    for crop in sorted(boxed, key=lambda c: c.info["box"][1]):
        x0, y0, x1, y1 = crop.info["box"]
        if bottom is None or (y0 + y1) / 2 > bottom:
            rows.append([])
            bottom = y1
        rows[-1].append(crop)
    ordered = [c for row in rows for c in sorted(row, key=lambda c: c.info["box"][0] + c.info["box"][2])]
    return ordered + rest


def see_through(crop: Image.Image, key) -> Image.Image:
    """Turn ground that shows through a piece back into transparency.

    A see-through pixel is a mix of the piece and the ground behind it,
    C = a*F + (1 - a)*K, and K is known. Two passes:
    - spill: how much of the key's hue a pixel carries says how much ground is in it;
      kept only where the piece color that leaves is a real color (a red potion stays red).
    - glass: a pixel that is a plain gray plus some amount of the key is clear glass over
      the ground, in a piece that has bright colorless highlights (glass catching light) or
      that is mostly the ground at full brightness (seen straight through). A shaded
      pink subject on a pink ground has neither and is left alone.
    """
    import numpy as np

    weak = min(range(3), key=lambda ch: key[ch])
    strong = [ch for ch in range(3) if key[ch] - key[weak] > 96]
    if not strong:
        return crop
    rgba = np.asarray(crop.convert("RGBA"), dtype=float)
    c, inside = rgba[..., :3], rgba[..., 3] > 0
    if not inside.any():
        return crop
    k_rgb = np.array(key, dtype=float)

    spill = np.min(c[..., strong], axis=-1) - c[..., weak]
    alpha = np.clip(1.0 - spill / (min(key[ch] for ch in strong) - key[weak]), 0.0, 1.0)
    piece = (c - (1.0 - alpha)[..., None] * k_rgb) / np.maximum(alpha, 1e-3)[..., None]
    real = (piece >= -SPILL_SLACK).all(-1) & (piece <= 255 + SPILL_SLACK).all(-1)
    # What glass leaves once the ground is out is light (clear glass, a highlight). A dark
    # leftover means the pixel was just a darker copy of the ground: a pink subject, kept.
    light = piece.mean(-1) >= SPILL_LIGHT
    seen = inside & real & light & (alpha > 0.04) & (alpha < 0.97)

    basis = np.stack([np.ones(3), k_rgb], axis=1)
    gray, ground = np.moveaxis(c @ np.linalg.pinv(basis).T, -1, 0)
    fit = np.abs(c - np.stack([gray, ground], -1) @ basis.T).sum(-1)
    glassy = inside & (fit < GLASS_FIT) & (ground > 0.35)
    lit = glassy.any() and (gray[glassy] > 60).mean() >= GLASS_SHARE
    # Or most of the piece is the ground at full brightness: the ground seen straight
    # through clear glass. A solid pink subject is shaded, so it reads darker than that.
    clear_view = glassy.sum() >= 0.5 * inside.sum() and glassy.any() and np.median(ground[glassy]) >= GLASS_THROUGH
    if glassy.sum() >= 0.2 * inside.sum() and (lit or clear_view):
        clear = np.clip(1.0 - ground, 0.0, 1.0)
        alpha = np.where(glassy, clear, alpha)
        behind = (c - ground[..., None] * k_rgb) / np.maximum(clear, 1e-3)[..., None]
        piece = np.where(glassy[..., None], behind, piece)
        seen |= glassy

    out = rgba.copy()
    out[..., :3] = np.where(seen[..., None], np.clip(piece, 0, 255), c)
    out[..., 3] = np.where(seen, alpha * rgba[..., 3], rgba[..., 3])
    return Image.fromarray(out.round().astype(np.uint8), "RGBA")


def _with_missed(im, subject: bytearray, keyed: bytearray, w: int, h: int, min_area: int, key, tol: int) -> bytearray:
    """The segmenter's mask, plus keyed pieces of at least 1/MISSED_SHARE of the image
    that it did not touch at all and that are not the ground's color. Keyed pieces it
    did touch were its call (shadows)."""
    fg = bytearray(subject)
    labels, boxes = _components(keyed, w, h, max(min_area, (w * h) // MISSED_SHARE))
    px = im.convert("RGB").load()
    for box in boxes:
        _area, minx, miny, maxx, maxy, label_id = box
        cells = [y * w + x for y in range(miny, maxy + 1) for x in range(minx, maxx + 1) if labels[y * w + x] == label_id]
        if any(subject[i] for i in cells):
            continue
        if key is not None and _mean_dist(px, cells, w, key) <= 2 * tol:
            continue
        for i in cells:
            fg[i] = 1
    return fg


def _mean_dist(px, cells: list[int], w: int, key) -> float:
    acc = [0, 0, 0]
    for i in cells:
        r, g, b = px[i % w, i // w]
        acc[0] += r
        acc[1] += g
        acc[2] += b
    return sum(abs(acc[c] / len(cells) - key[c]) for c in range(3))


def _ground_blotch(px, labels, w: int, box, biggest: int, key, tol: int) -> bool:
    """A small piece that is, on average, the ground's color: a patch of an uneven
    ground the key missed, not a part. The biggest piece is never passed here."""
    area, minx, miny, maxx, maxy, label_id = box
    if area * BLOTCH_SHARE > biggest:
        return False
    cells = [y * w + x for y in range(miny, maxy + 1) for x in range(minx, maxx + 1) if labels[y * w + x] == label_id]
    return _mean_dist(px, cells, w, key) <= 2 * tol


def _despill(rgb: tuple[int, int, int], key: tuple[int, int, int]) -> tuple[int, int, int]:
    """Remove the key's hue from an edge pixel: the channels the key is strong in
    come down to the one it is weak in, so a pink rim goes neutral."""
    weak = min(range(3), key=lambda c: key[c])
    strong = [c for c in range(3) if key[c] - key[weak] > 96]
    if not strong:
        return rgb
    floor = rgb[weak]
    excess = min(rgb[c] for c in strong) - floor
    if excess <= 0:
        return rgb
    out = list(rgb)
    for c in strong:
        out[c] -= excess
    return out[0], out[1], out[2]
