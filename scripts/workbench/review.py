#!/usr/bin/env python3
"""The review: what Stephen's critiques taught, run on every model before he sees it.

    .venv/bin/python scripts/workbench/review.py out/S            # writes out/S/workbench/review/{review.json,sheet.png}
    .venv/bin/python scripts/workbench/review.py out/S --bake B.glb --paint P.glb

Each thing Stephen has called out on a model is either a number here (a check with a limit) or a row of the sheet
that has to be looked at. CRITIQUES lists them all and says which. Add to it whenever a critique catches something
the review missed: a number if one separates the bad model from the good one, else a row of the sheet.

- head_offset, mirror, pieces, tris: eyes.measure on the deliverable (psx.glb when there is one).
- paint: only for a bake made with retexture --select (a <bake>.select.json beside it). That bake is pulled toward
  Hunyuan's paint, which is painted on the model and so has one set of eyes and ears in the right places; a large
  patch where the bake still disagrees with it is a drawing's feature in the wrong place (the pig's ghost ear on its
  neck) or a drawing's own lighting (patchwork). Renders of the bake and the paint share the mesh, so they line up
  pixel for pixel. The number is the largest such patch as a share of the subject, worst view. On the pig: bakes
  with the ghost ear 0.12-0.22, the fixed bake 0.054. An older bake (fill photo, no select) is not pulled toward the
  paint and disagrees with it everywhere (the approved sheep: 0.41), so the check does not apply to it.
- spill: the chroma key's magenta in the deliverable, pixels leaning both red and blue over green. Reported with
  no limit yet: the casts seen so far were too faint for it (the wheelbarrow's first bake 0.009 against 0.006
  after the fix), so it gets a limit when a bad model gives it one.
- degraded: whatever generate.sh recorded in degraded.json.

The sheet: the head up close from both sides, three-quarter and front, painted above and grey below (the same
render's shading alone, so a painted ear with no ear under it shows), then the whole model, then where the bake
disagrees with the paint. The verdict is never a pass: the numbers catch what has been seen before, the sheet is
for everything else, and it must be looked at (count the ears and eyes on the painted row against the grey row).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rig_render  # noqa: E402
import seam_bake  # noqa: E402
from workbench import eyes  # noqa: E402

HEAD_OFFSET_MAX = 5.0  # % of body width; the sheep came out at 21, the fixed sheep and pig under 2
MIRROR_MIN = 0.9  # the approved sheep and pig 0.97
PSX_TRIS_MAX = 1300  # the 1,200 budget plus slack; the tractor needed 1,622 and that was said
PAINT_MAX = 0.08  # largest disagreeing patch / subject; the pig's ghost-ear bakes 0.12+, the fixed bake 0.054
PAINT_TONE = 25.0  # lightness difference (0-255) after gain matching that counts as disagreeing
SPILL_LEAN = 10.0  # red - green and blue - green both over this counts as leaning magenta
TILE = 384
PAINT_SIZE = 512
PAINT_VIEWS = ("side", "xside", "front", "back", "top")
HEAD_VIEWS = ("side", "xside", "q", "front")
BODY_VIEWS = ("side", "back", "top", "q")
CREATURES = ("animal", "person")

# Every critique so far, and what catches it now: a check by name, or "sheet" (and which row to read).
CRITIQUES = [
    {"critique": "Patchwork: different angles of the drawings plastered on (pig)", "caught_by": "paint; sheet body row"},
    {"critique": "Two sets of ears and eyes (pig)", "caught_by": "paint; sheet head rows: count ears and eyes, "
                                                                  "painted against grey"},
    {"critique": "Head off the body's centre line (sheep, chicken)", "caught_by": "head_offset"},
    {"critique": "Left side not the right side mirrored", "caught_by": "mirror"},
    {"critique": "Magenta cast from the key on grey or white props", "caught_by": "spill"},
    {"critique": "Hunyuan's paint showing through (the pig's maroon back stripe)", "caught_by": "sheet body row: top"},
    {"critique": "One-sided detail drawn on both sides (the mailbox flag)", "caught_by": "sheet body row: side "
                                                                                       "against q"},
    {"critique": "A round prop came out a box (hay bale)", "caught_by": "circularity (reported, no limit); sheet"},
    {"critique": "Legs pinched off or jagged at the PS1 cut (sheep, tractor)", "caught_by": "pieces, tris; sheet"},
    {"critique": "Views inconsistent, built from one picture (tractor)", "caught_by": "degraded"},
]


def _check(name: str, value, limit, ok, why: str) -> dict:
    return {"name": name, "value": value, "limit": limit, "ok": ok, "why": why}


def deliverable(folder: Path) -> Path | None:
    for name in ("psx.glb", "textured_photo.glb", "textured.glb", "mesh.glb"):
        if (folder / name).is_file():
            return folder / name
    return None


def subject(folder: Path) -> str:
    """intent.json's subject. Older assets have none: a PS1 facing of right or left is an animal drawn side-on,
    camera a person, no facing a prop."""
    try:
        intent = json.loads((folder / "intent.json").read_text())
    except (OSError, ValueError):
        return "prop"
    facing = (intent.get("psx") or {}).get("facing")
    return intent.get("subject") or {"right": "animal", "left": "animal", "camera": "person"}.get(facing, "prop")


# ---------------------------------------------------------------- renders

class Scene:
    """One GLB posed once, with a camera every render of it shares."""

    def __init__(self, glb, size: int, cam: dict | None = None) -> None:
        self.it = rig_render.Item(Path(glb))
        self.Vs, self.jp = self.it.pose(None, 0.0)
        if not self.Vs:
            raise ValueError(f"{glb}: no triangle meshes")
        self.pts = np.concatenate(self.Vs)
        self.cam = cam or rig_render.camera(self.pts, size)

    def render(self, view: str, mode: str = "texture_px:unlit") -> np.ndarray:
        im = rig_render.draw([self.it], [(self.Vs, self.jp)], view, mode, self.cam, False)
        return np.asarray(im.convert("RGB")).astype(float)


def subject_mask(im: np.ndarray) -> np.ndarray:
    return np.abs(im - np.asarray(rig_render.BG, float)).sum(2) > 12


def grey(lit: np.ndarray, unlit: np.ndarray) -> np.ndarray:
    """The shading alone: the lit render over the unlit one is each pixel's light, whatever the paint."""
    shade = (lit.sum(2) / np.maximum(unlit.sum(2), 1.0)).clip(0, 1.4)
    out = np.repeat((205.0 * shade)[..., None], 3, 2)
    out[~subject_mask(unlit)] = rig_render.BG
    return out


def head_box(sc: Scene, view: str, kind: str) -> tuple[int, int, int, int] | None:
    """The head's pixel box in a view of a PSX-oriented model (facing +Z): the front share of an animal above its
    legs, the top quarter of a person. None for a prop."""
    P = sc.pts
    lo, hi = P.min(0), P.max(0)
    H = hi[1] - lo[1]
    if kind == "animal":
        # a long animal's head is above its legs; a tall one's (a chicken: its front is all chest) at the top
        above = 0.3 if hi[2] - lo[2] > H else 0.55
        sel = (P[:, 2] > hi[2] - 0.38 * (hi[2] - lo[2])) & (P[:, 1] > lo[1] + above * H)
    elif kind == "person":
        sel = P[:, 1] > hi[1] - 0.25 * H
    else:
        return None
    if not sel.any():
        return None
    q = rig_render.project(P[sel], view, sc.cam)
    pad = 0.06 * sc.cam["size"]
    n = sc.cam["size"]
    x0, y0 = max(0, int(q[:, 0].min() - pad)), max(0, int(q[:, 1].min() - pad))
    x1, y1 = min(n, int(q[:, 0].max() + pad)), min(n, int(q[:, 1].max() + pad))
    return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None


def _crop(im: np.ndarray, box) -> Image.Image:
    x0, y0, x1, y1 = box
    return Image.fromarray(im[y0:y1, x0:x1].clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------- checks

def paint_check(bake_glb, paint_glb, size: int = PAINT_SIZE, key=None) -> tuple[dict, list[Image.Image]]:
    """The largest patch where a --select bake disagrees with the paint it was pulled toward, per view, and an
    overlay of each view (the bake dimmed, disagreeing pixels blue, the largest patch red). Paint the key's colour
    (the drawing sheet's key leaking into it, which the bake refills on purpose) is left out."""
    paint = Scene(paint_glb, size)
    bake = Scene(bake_glb, size, paint.cam)
    if len(paint.pts) != len(bake.pts):
        raise ValueError("the bake and the paint are not the same mesh")
    per, tiles = {}, []
    for v in PAINT_VIEWS:
        p, b = paint.render(v), bake.render(v)
        subj = ndimage.binary_erosion(subject_mask(p), iterations=4)
        if subj.sum() < 50:
            continue
        gain = np.median(p[subj], 0) / np.maximum(np.median(b[subj], 0), 1.0)
        lum = np.array([0.299, 0.587, 0.114])
        diff = np.abs(ndimage.gaussian_filter((b * gain) @ lum, 6) - ndimage.gaussian_filter(p @ lum, 6))
        judged = subj
        if key is not None:
            judged = subj & ~(np.linalg.norm(p - np.asarray(key, float), axis=2) < seam_bake.KEY_TOL)
        off = ndimage.binary_opening((diff > PAINT_TONE) & judged, iterations=2)
        lab, n = ndimage.label(off)
        sizes = ndimage.sum(off, lab, range(1, n + 1)) if n else np.zeros(0)
        share = float(sizes.max() / subj.sum()) if n else 0.0
        per[v] = round(share, 3)
        over = b * 0.45
        over[off] = [40, 90, 255]
        if n:
            over[lab == int(np.argmax(sizes)) + 1] = [255, 40, 40]
        over[~subject_mask(b)] = rig_render.BG
        tiles.append(_label(Image.fromarray(over.astype(np.uint8)), f"{v} {share:.3f}"))
    worst = max(per, key=per.get) if per else None
    return {"per_view": per, "worst_view": worst, "worst": per.get(worst, 0.0)}, tiles


def spill(sc: Scene, views=("side", "xside", "front", "back")) -> float:
    lean = n = 0
    for v in views:
        im = sc.render(v)
        m = subject_mask(im)
        r, g, b = im[..., 0], im[..., 1], im[..., 2]
        lean += int((m & (r - g > SPILL_LEAN) & (b - g > SPILL_LEAN)).sum())
        n += int(m.sum())
    return round(lean / max(n, 1), 4)


# ---------------------------------------------------------------- sheet

def _label(im: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 7 * len(text) + 6, 14], fill="white")
    d.text((3, 2), text, fill="black", font=ImageFont.load_default())
    return im


def _row(title: str, tiles: list[Image.Image], height: int = 260) -> Image.Image:
    scaled = [t.resize((max(1, round(t.width * height / t.height)), height), Image.LANCZOS) for t in tiles]
    w = sum(t.width for t in scaled) + 8 * max(0, len(scaled) - 1)
    row = Image.new("RGB", (max(w, 300), height + 22), "white")
    ImageDraw.Draw(row).text((4, 4), title, fill="black", font=ImageFont.load_default())
    x = 0
    for t in scaled:
        row.paste(t, (x, 22))
        x += t.width + 8
    return row


def sheet(sc: Scene, kind: str, paint_tiles: list[Image.Image], out: Path) -> Path:
    rows = []
    boxes = {v: head_box(sc, v, kind) for v in HEAD_VIEWS}
    if all(boxes.values()):
        painted, plain = [], []
        for v in HEAD_VIEWS:
            unlit, lit = sc.render(v), sc.render(v, "texture_px")
            painted.append(_label(_crop(lit, boxes[v]), v))
            plain.append(_label(_crop(grey(lit, unlit), boxes[v]), v))
        rows += [_row("head, painted: count the ears and eyes", painted),
                 _row("head, grey: the ears and eyes the mesh really has", plain)]
    full = []
    for v in BODY_VIEWS:
        im = sc.render(v, "texture_px")
        ys, xs = np.nonzero(subject_mask(im))
        pad = 8
        box = (max(0, xs.min() - pad), max(0, ys.min() - pad), xs.max() + pad, ys.max() + pad)
        full.append(_label(_crop(im, box), v))
    rows.append(_row("whole model: patchwork, paint showing through, one-sided detail", full))
    if paint_tiles:
        rows.append(_row("bake against Hunyuan's paint, in the bake's own frame (the names are its cameras, not the "
                         "PS1 facing): blue disagrees, red is the largest patch", paint_tiles))
    W = max(r.width for r in rows)
    S = Image.new("RGB", (W, sum(r.height for r in rows) + 6 * len(rows)), "white")
    y = 0
    for r in rows:
        S.paste(r, (0, y))
        y += r.height + 6
    out.parent.mkdir(parents=True, exist_ok=True)
    S.save(out)
    return out


# ---------------------------------------------------------------- review

def review(folder, bake=None, paint=None, out_dir=None, glb=None) -> dict:
    """Run every check on an asset folder and draw the sheet. bake/paint default to textured_photo.glb and
    textured.glb; glb (the model looked at) to deliverable(). Writes review.json and sheet.png into out_dir
    (default <folder>/workbench/review) and returns the review."""
    folder = Path(folder)
    out_dir = Path(out_dir) if out_dir else folder / "workbench" / "review"
    kind = subject(folder)
    model = Path(glb) if glb else deliverable(folder)
    if model is None:
        raise FileNotFoundError(f"{folder}: no model to review")
    checks = []
    m = eyes.measure(model)
    if kind in CREATURES:
        off = m["head_offset_pct"]
        checks.append(_check("head_offset", off, HEAD_OFFSET_MAX, None if off is None else off <= HEAD_OFFSET_MAX,
                             "head's sideways offset, % of body width" if off is not None else
                             "no head found to measure (body longer than wide only)"))
        checks.append(_check("mirror", m["mirror_iou"], MIRROR_MIN, m["mirror_iou"] >= MIRROR_MIN,
                             "top view against its own mirror image"))
    checks.append(_check("pieces", m["pieces"], 1, m["pieces"] <= 1, "a leg as its own piece is pinched: weight it"))
    if model.name == "psx.glb":
        checks.append(_check("tris", m["tris"], PSX_TRIS_MAX, m["tris"] <= PSX_TRIS_MAX, "the PS1 budget"))
    checks.append(_check("circularity", m["circularity"], None, None, f"ellipse fill, {m['circularity_view']} view; "
                                                                       "near 1 for a round prop"))
    sc = Scene(model, TILE)
    s = spill(sc)
    checks.append(_check("spill", s, None, None, "share of the subject leaning magenta (the key); no limit yet"))

    bake = Path(bake) if bake else folder / "textured_photo.glb"
    paint = Path(paint) if paint else folder / "textured.glb"
    paint_tiles = []
    if not bake.with_suffix(".select.json").is_file():
        checks.append(_check("paint", None, PAINT_MAX, None, "not a --select bake: the check does not apply"))
    elif not paint.is_file():
        checks.append(_check("paint", None, PAINT_MAX, None, f"no paint to compare with ({paint.name})"))
    else:
        try:
            try:
                key = json.loads((folder / "flat" / "key.json").read_text()).get("rgb")
            except (OSError, ValueError):
                key = None
            pc, paint_tiles = paint_check(bake, paint, key=key)
            checks.append(_check("paint", pc["worst"], PAINT_MAX, pc["worst"] <= PAINT_MAX,
                                 f"largest patch disagreeing with the paint, worst view {pc['worst_view']}; "
                                 f"per view {pc['per_view']}"))
        except ValueError as exc:
            checks.append(_check("paint", None, PAINT_MAX, None, str(exc)))
    try:
        for d in json.loads((folder / "degraded.json").read_text()).get("degraded", []):
            checks.append(_check("degraded", d.get("code") or d.get("stage"), None, False,
                                 d.get("message") or json.dumps(d)))
    except (OSError, ValueError):
        pass

    sheet_path = sheet(sc, kind, paint_tiles, out_dir / "sheet.png")
    failed = [c["name"] for c in checks if c["ok"] is False]
    res = {"model": str(model), "subject": kind, "failed": failed, "checks": checks, "sheet": str(sheet_path),
           "verdict": ("fix: " + ", ".join(failed)) if failed else "look",
           "look": "Read the sheet before showing anyone: count the ears and eyes on the painted head row against "
                   "the grey row, then read the body row for patchwork, paint showing through and one-sided detail.",
           "critiques": CRITIQUES}
    (out_dir / "review.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--glb")
    ap.add_argument("--bake")
    ap.add_argument("--paint")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = review(a.folder, a.bake, a.paint, a.out, a.glb)
    print(json.dumps({k: v for k, v in res.items() if k != "critiques"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
