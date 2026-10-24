"""The four-view sheet: split it into panels, cut each one out, and check they show one subject (plan 1.2, 1.3).

A sheet is drawn as one picture, front | right | back | left in a row (layout 'strip') or row by row in a 2x2 grid
(layout 'grid'), on a magenta ground. split_sheet cuts it at the magenta gaps nearest the panel borders, or in equal
parts when a gap is missing. Each panel is cut out with Apple Vision (store.vision_mask; the magenta key when Vision
is unavailable), with thin full-width lines (a drawn baseline or frame) dropped first.

check_panels runs the hard checks. Any failure is a flaw:
    one subject per panel, at least AREA_MIN of it;
    Lab colour distance to the median at most COLOUR_DE; right against mirrored left IoU at least SIDE_IOU;
    side-on creatures: ONE facing.detect_image call on the right panel. Head left on a clean profile mirrors every
    panel in place (the turnaround of the mirrored creature, labels unchanged). An unavailable vision model is a
    hard flaw (checks.facing.how = 'unavailable'), never a pass.
A cloudy ground or a shadow is a note: matting and despill handle it. So is a height spread over HEIGHT_SPREAD (one
view drawn bigger than another): both readers of the views frame each one on its own (Hunyuan3D-2mv's recenter crops
every view to its box and scales it to fill; retexture fits each silhouette cropped to its box and stretched square),
so a view's size on the sheet never reaches the mesh or the paint. As a flaw it threw away three of five good photo
sheets of one cow and the lantern's views.

write_views writes flat/sheet.png, flat/views/<id>.png, flat/views.json and flat/source.png (the canonical panel).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from flat2d.pose import PANELS

AREA_MIN = 0.02
HEIGHT_SPREAD = 0.06
# A prop's height is also measured without thin rows at its top and bottom (a bail handle, a wire loop, a wick: at
# most THIN_ROW of the cell height wide). The vision mask keeps such a wire in one view and drops it in the next,
# and a lantern's handle stands taller in its side views; neither is a different subject. A prop fails the height
# check only when both measures spread. Creatures keep the plain measure (thin legs are part of their height).
THIN_ROW = 0.035
COLOUR_DE = 18.0
SIDE_IOU = 0.6
PIECE_SHARE = 0.15  # a second blob this big next to the largest is a second subject
# ...unless it sits within this share of the cell's height of the rest: one subject's parts that the cut-out split
# (the tractor's drawbar was taken for a line, cutting its back and side views in two, and the sheet was thrown out
# for one view: the bake covered 8.6% of the atlas). Two subjects stand further apart than that
PIECE_GAP = 0.02
LINE_FILL = 0.8  # a row this full of non-ground ...
LINE_THIN = 0.015  # ... and at most this thick (share of the side) is a drawn line, not the subject
CLOUDY_TOL = 45  # flat2d.optional.CLEAN_TOL
SHADOW_SHARE = 0.005

# Flaws a new seed can fix. A missing vision model is a flaw too, but drawing again cannot fix it.
NO_REDRAW = {"facing_unavailable"}
NOTE_TEXT = {"cloudy": "The ground is cloudy; matting handles it.", "shadow": "There is a shadow on the ground; "
             "matting handles it.", "quarters": "No clear gaps between the views; cut in equal parts.",
             "mask_key": "Apple Vision was unavailable; the views were cut out on the magenta key.",
             "height": "The views are drawn at different sizes; the shape and paint steps frame each one on its own."}


# ---------------------------------------------------------------- masks

def ground_of(rgb: np.ndarray, key: dict) -> np.ndarray:
    """True where the picture is the painted ground (flat2d.parts.ground_key's colour and tolerance)."""
    k = np.asarray(key["rgb"], dtype=np.int32)
    return np.abs(rgb.astype(np.int32) - k).sum(-1) <= int(key["tol"])


def drop_lines(fg: np.ndarray) -> np.ndarray:
    """Clear thin rows that are almost all non-ground: a drawn baseline or floor line. Rows only: a column can be
    'full' honestly (two tall subjects stacked in a grid), and a cut there would halve them."""
    fg = fg.copy()
    h = fg.shape[0]
    full = fg.mean(axis=1) >= LINE_FILL
    thin = max(2, int(round(h * LINE_THIN)))
    for a, b in _runs(full):
        if b - a <= thin:
            fg[a:b, :] = False
    return fg


def _runs(empty: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    out, i, n = [], 0, len(empty)
    while i < n:
        if empty[i]:
            j = i
            while j < n and empty[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _cut(fg: np.ndarray, axis: int, target: float, reach: float) -> int | None:
    """The middle of the empty band (columns for axis 0, rows for axis 1) nearest target, within reach."""
    occupied = fg.sum(axis=axis)
    across = fg.shape[axis]
    empty = occupied <= max(1, int(across * 0.003))
    best = None
    for a, b in _runs(empty):
        if a == 0 or b == len(empty):
            continue  # the margin outside the views, not a gap between them
        mid = (a + b) / 2
        d = abs(mid - target)
        if d <= reach and (best is None or d < best[0]):
            best = (d, int(round(mid)))
    return None if best is None else best[1]


def split_sheet(image: Image.Image, key: dict, layout: str = "strip", subject: np.ndarray | None = None) -> dict:
    """{how: 'gaps'|'quarters', layout, boxes: {id: [x0, y0, x1, y1]}} in PANELS order. subject is the sheet's
    subject mask (Apple Vision); without it, everything off the magenta key counts (a graded ground defeats that)."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    fg = drop_lines(subject if subject is not None else ~ground_of(rgb, key))
    if layout == "grid":
        row = _cut(fg, 1, h / 2, h / 4)
        cells = []
        for (y0, y1) in ((0, row or h // 2), (row or h // 2, h)):
            col = _cut(fg[y0:y1], 0, w / 2, w / 4)
            cells.append((y0, y1, col))
        found = row is not None and all(c[2] is not None for c in cells)
        if not found:
            cells = [(0, h // 2, w // 2), (h // 2, h, w // 2)]
        (a0, a1, ac), (b0, b1, bc) = cells
        boxes = [[0, a0, ac, a1], [ac, a0, w, a1], [0, b0, bc, b1], [bc, b0, w, b1]]
    else:
        cuts = [_cut(fg, 0, k * w / 4, w / 8) for k in (1, 2, 3)]
        found = all(c is not None for c in cuts) and cuts == sorted(cuts)
        if not found:
            cuts = [k * w // 4 for k in (1, 2, 3)]
        edges = [0, *cuts, w]
        boxes = [[edges[i], 0, edges[i + 1], h] for i in range(4)]
    return {"how": "gaps" if found else "quarters", "layout": layout,
            "boxes": {pid: [int(v) for v in box] for pid, box in zip(PANELS, boxes)}}


def _vision(crop: Image.Image) -> np.ndarray | None:
    from flat2d.store import vision_mask

    with tempfile.TemporaryDirectory(prefix="views_") as tmp:
        path = Path(tmp) / "panel.png"
        crop.save(path, format="PNG")
        mask = vision_mask(path)
    if mask is None:
        return None
    return np.frombuffer(bytes(mask), np.uint8).reshape(crop.size[1], crop.size[0]).astype(bool)


def sheet_mask(image: Image.Image, key: dict, vision: Callable | None = None) -> tuple[np.ndarray, str]:
    """The subjects on the whole sheet: Apple Vision's mask (it finds every view at once), with thin full-width
    lines dropped; the magenta key alone when Vision is unavailable. Returns (mask, 'vision'|'key')."""
    got = (vision or _vision)(image)
    if got is None or not got.any():
        return drop_lines(~ground_of(np.asarray(image.convert("RGB")), key)), "key"
    return drop_lines(got), "vision"


def blobs(mask: np.ndarray) -> list[int]:
    """Blob sizes, largest first."""
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel())[1:]
    return sorted((int(s) for s in sizes), reverse=True)


def _close_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """The mask with gaps up to `gap` pixels bridged (grown by half the gap and shrunk back), for counting subjects."""
    from scipy import ndimage

    r = max(1, (gap + 1) // 2)
    st = ndimage.generate_binary_structure(2, 1)
    return ndimage.binary_erosion(ndimage.binary_dilation(mask, st, iterations=r), st, iterations=r, border_value=0)


def body_height(mask: np.ndarray, thin: float) -> int:
    """Rows from the first to the last row with more than `thin` subject pixels: the height without thin wires
    at the ends. The plain box height when no row is that full."""
    w = mask.sum(1)
    rows = np.nonzero(w > thin)[0]
    if not len(rows):
        rows = np.nonzero(w)[0]
    return int(rows[-1] - rows[0] + 1) if len(rows) else 0


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, ...x3) to CIE Lab (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def side_iou(right: np.ndarray, left: np.ndarray, size: int = 256) -> float:
    """IoU of the right panel's silhouette against the mirrored left one. Each is cropped to its box and scaled to
    the same HEIGHT (keeping its aspect; size on the sheet is only a note), then centred on one canvas."""
    a, b = right[_slices(right)], np.fliplr(left[_slices(left)])
    if a.size == 0 or b.size == 0:
        return 0.0

    def scaled(m: np.ndarray) -> np.ndarray:
        w = max(1, int(round(m.shape[1] * size / m.shape[0])))
        return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, size), Image.NEAREST)) > 127

    ra, rb = scaled(a), scaled(b)
    width = max(ra.shape[1], rb.shape[1])
    ca, cb = np.zeros((size, width), bool), np.zeros((size, width), bool)
    oa, ob = (width - ra.shape[1]) // 2, (width - rb.shape[1]) // 2
    ca[:, oa:oa + ra.shape[1]] = ra
    cb[:, ob:ob + rb.shape[1]] = rb
    union = (ca | cb).sum()
    return float((ca & cb).sum() / union) if union else 0.0


def _slices(mask: np.ndarray):
    x0, y0, x1, y1 = bbox(mask)
    return slice(y0, y1), slice(x0, x1)


def ground_notes(image: Image.Image, key: dict, subject: np.ndarray) -> list[str]:
    """'cloudy' or 'shadow' on the sheet's ground. Notes only: they never cost a redraw."""
    if key["tol"] > CLOUDY_TOL:
        return ["cloudy"]
    rgb = np.asarray(image.convert("RGB"), dtype=float)
    k = np.array(key["rgb"], dtype=float)
    ground = np.abs(rgb - k).sum(-1) <= key["tol"]
    t = np.clip((rgb @ k) / max(float(k @ k), 1.0), 0.0, 1.0)
    darker = np.abs(rgb - t[..., None] * k).sum(-1) <= 36
    shadow = ~subject & ~ground & darker & (t < 0.85)
    return ["shadow"] if shadow.mean() > SHADOW_SHARE else []


# ---------------------------------------------------------------- checks

def check_panels(image: Image.Image, key: dict, kind: str | None, layout: str = "strip",
                 vision: Callable | None = None, detect: Callable | None = None,
                 phase: Callable[[str], None] | None = None) -> dict:
    """Split, cut out and check one sheet. Returns {flaws, notes, checks, views, split, panels, masks, mirrored}:
    panels and masks are per id (PIL crops and bool arrays), already mirrored when the head pointed left.
    phase(name) is called as each part starts: 'views_split' (split and cut out), then 'views_check'."""
    from flat2d.pose import FACING

    if phase:
        phase("views_split")
    subject, how_mask = sheet_mask(image, key, vision)
    split = split_sheet(image, key, layout, subject)
    panels: dict[str, Image.Image] = {}
    masks: dict[str, np.ndarray] = {}
    flaws: list[str] = []
    notes: list[str] = ["quarters"] if split["how"] == "quarters" else []
    rgb = image.convert("RGB")
    for pid, (x0, y0, x1, y1) in split["boxes"].items():
        panels[pid], masks[pid] = rgb.crop((x0, y0, x1, y1)), subject[y0:y1, x0:x1].copy()
    if how_mask == "key":
        notes.append("mask_key")
    if phase:
        phase("views_check")
    views, areas, heights, labs, bodies = [], {}, {}, {}, {}
    nominal = image.size[1] / (2 if split["layout"] == "grid" else 1)
    for pid in PANELS:
        mask = masks[pid]
        area = float(mask.mean())
        sizes = blobs(_close_gaps(mask, max(2, round(PIECE_GAP * nominal))))
        areas[pid] = round(area, 4)
        if area < AREA_MIN:
            flaws.append(f"empty:{pid}")
        elif len([s for s in sizes if s >= PIECE_SHARE * sizes[0]]) > 1:
            flaws.append(f"pieces:{pid}")
        box = bbox(mask)
        # Against the nominal cell height (the sheet's for a strip, half of it for a grid), never the cut cell's:
        # the cuts sit mid-gap, so two cells of one sheet can differ in height while their subjects do not.
        heights[pid] = round((box[3] - box[1]) / nominal, 4)
        bodies[pid] = round(body_height(mask, THIN_ROW * nominal) / nominal, 4)
        if mask.any():
            labs[pid] = to_lab(np.asarray(panels[pid])[mask]).mean(0)
        views.append({"id": pid, "file": f"flat/views/{pid}.png", "bbox": box, "height_frac": heights[pid],
                      "area": areas[pid], "cell": split["boxes"][pid]})
    spread = round(max(heights.values()) - min(heights.values()), 4)
    body_spread = round(max(bodies.values()) - min(bodies.values()), 4)
    creature = bool(kind and FACING.get(kind) == "right")
    if spread > HEIGHT_SPREAD and (creature or body_spread > HEIGHT_SPREAD):
        notes.append("height")
    de = 0.0
    if len(labs) >= 2:
        med = np.median(np.stack(list(labs.values())), axis=0)
        de = round(float(max(np.linalg.norm(v - med) for v in labs.values())), 2)
    if de > COLOUR_DE:
        flaws.append("colour")
    iou = round(side_iou(masks["right"], masks["left"]), 4)
    if iou < SIDE_IOU:
        flaws.append("side")
    checks: dict[str, Any] = {"height_spread": spread, "colour_de": de, "side_iou": iou, "areas": areas}
    if not creature:
        checks["body_height_spread"] = body_spread
    mirrored = False
    if kind and FACING.get(kind) == "right":
        fc = facing_check(panels["right"], detect)
        checks["facing"] = fc
        if fc["how"] == "unavailable":
            flaws.append("facing_unavailable")
        elif fc.get("mirror"):
            mirrored = True
            panels = {pid: ImageOps.mirror(im) for pid, im in panels.items()}
            masks = {pid: np.fliplr(m) for pid, m in masks.items()}
            for v in views:
                w = v["cell"][2] - v["cell"][0]
                x0, y0, x1, y1 = v["bbox"]
                v["bbox"] = [w - x1, y0, w - x0, y1]
        elif not fc.get("ok"):
            flaws.extend(f for f in fc.get("flaws") or ["facing"] if f not in flaws)
    else:
        checks["facing"] = {"how": "not_needed", "why": "a prop or a front-canonical figure"}
    union = np.zeros(image.size[::-1], dtype=bool)
    for pid, (x0, y0, x1, y1) in split["boxes"].items():
        m = masks[pid]
        union[y0:y1, x0:x1] |= np.fliplr(m) if mirrored else m
    notes.extend(n for n in ground_notes(image, key, union) if n not in notes)
    return {"flaws": flaws, "notes": notes, "checks": checks, "views": views, "split": split, "panels": panels,
            "masks": masks, "mirrored": mirrored}


def facing_check(panel: Image.Image, detect: Callable | None = None) -> dict:
    """One facing.detect_image call on the right panel (the model is unloaded after it: keep_alive 0).
    {how: 'vlm'|'mirrored'|'unavailable', head, profile, confidence, ok, mirror, flaws}."""
    import facing

    with tempfile.TemporaryDirectory(prefix="views_facing_") as tmp:
        path = Path(tmp) / "right.png"
        panel.save(path, format="PNG")
        det = (detect or facing.detect_image)(path)
    verdict = facing.judge(det, "right")
    out = {k: det.get(k) for k in ("head", "profile", "confidence") if k in det}
    if det.get("how") == "unavailable":
        return {**out, "how": "unavailable", "why": det.get("why") or "vision model unavailable", "ok": False,
                "mirror": False, "flaws": ["facing_unavailable"]}
    if verdict.get("mirror"):
        return {**out, "how": "mirrored", "head_before": det.get("head"), "head": "right", "ok": True,
                "mirror": True, "flaws": []}
    return {**out, "how": "vlm", "ok": bool(verdict.get("ok")), "mirror": False,
            "flaws": list(verdict.get("flaws") or [])}


def redrawable(flaws: list[str]) -> list[str]:
    """The flaws a new seed may fix (all hard flaws but a missing vision model)."""
    return [f for f in flaws if f not in NO_REDRAW]


def score(result: dict) -> tuple:
    """Lower is better: fewer flaws, then a flatter sheet (smaller height spread and colour distance)."""
    c = result["checks"]
    return (len(result["flaws"]), c["height_spread"] + c["colour_de"] / 100.0 - c["side_iou"])


# ---------------------------------------------------------------- files

def cutout(panel: Image.Image, mask: np.ndarray) -> Image.Image:
    """The panel with the subject's mask as alpha (RGBA, the panel's own size)."""
    rgba = panel.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
    return rgba


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_views(folder: Path, sheet: Image.Image, result: dict, *, kind: str | None, canonical: str,
                meta: dict, degraded: list[str]) -> dict:
    """Write flat/sheet.png (as drawn), flat/views/<id>.png (each panel on its own ground, mirrored when the head
    pointed left: generate.sh isolates each view itself) and flat/views.json, which is also returned.
    flat/source.png (the canonical panel) must be written first: views.json records its sha256."""
    from flat2d.parts import ground_key
    from flat2d.pose import FACING

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "views").mkdir(exist_ok=True)
    sheet.convert("RGB").save(folder / "sheet.png", format="PNG")
    for v in result["views"]:
        result["panels"][v["id"]].convert("RGB").save(folder / "views" / f"{v['id']}.png", format="PNG")
    facing = FACING.get(kind or "", "camera") if kind else "camera"
    doc = {
        "version": 1,
        "naming": "hunyuan",
        "canonical": canonical,
        "facing": facing,
        "kind": kind,
        "source_sha256": sha256_file(folder / "source.png"),
        # Each view's own ground key: a graded ground is darker on the bottom row than key.json's (the canonical
        # panel's), so the despill of back/left wants its own.
        "views": [{**{k: v[k] for k in ("id", "file", "bbox", "height_frac", "area", "cell")},
                   "key": ground_key(result["panels"][v["id"]])} for v in result["views"]],
        "consistent": not result["flaws"],
        "flaws": list(result["flaws"]),
        "notes": list(result["notes"]),
        "degraded": list(degraded),
        "checks": result["checks"],
        "mirrored": bool(result.get("mirrored")),
        "sheet": dict(meta),
    }
    (folder / "views.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


VIEWS_KEYS = ("version", "naming", "canonical", "facing", "source_sha256", "views", "consistent", "flaws", "notes",
              "degraded", "checks", "sheet")


def read_views(folder: Path) -> dict | None:
    try:
        doc = json.loads((Path(folder) / "views.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None
