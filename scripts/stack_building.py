#!/usr/bin/env python3
"""A building as a sprite stack from three drawn walls. No image model in this module.

Why: a four-view sheet of a building fails as a stack, because the image model draws views that disagree. Walls
drawn one at a time agree, so the method (proven by hand on 2026-11-21, a brick building and a blue cafe) is:

1. The text-to-image model draws ONE plain wall, straight on, on magenta (base.png). The edit model ADDS the front
   feature to a copy (front.png: a shop front and an awning) and, optionally, the back feature to another
   (back.png: a door). flat2d/building.py makes these pictures; this module only reads them.
2. walls_to_views: the plain wall's colours are fitted to the edited back's (the edit model brightens what it
   redraws), everything is cut to the plain wall's box (so an awning hanging past the corners is trimmed, and a
   pavement strip drawn under the wall goes), the roof plan is made in code, and the four views, the sheet and
   views.json (layout "walls") are written under out/<stem>/flat/. The checks are real: see wall_checks.
3. clean_walls: the views cleaned to cells in flat/pixel/, in one of two looks: "crisp" (pixelart.clean_folder: a
   small palette, hard outline) or "smooth" (each cell the area average of its pixels, no colour limit; the crisp
   clean's cell outline is kept). The stack is always read from flat/pixel/; pixel.json records the look.
4. set_bays, then build_shape: the window bays found on the plain wall are dropped or repeated so the front can
   be wider or narrower than the side is deep (one drawing makes a wide shop or a narrow, deep townhouse); then the
   block in numpy. The footprint is the front's width by the side's width (the side walls are the plain wall too), the front and back faces painted from the front and back pictures, a deck sunk inside a
   parapet, and an awning wedge sticking out of the front, found as the rows where the front and back differ
   across nearly the whole width. No rooftop pieces: those come later as their own feature.
5. write_building: the usual stack files (stack_views.write_stack), stack.json "maker": "walls" and "look".

Orientation inside build_shape: solid[h, y, x], layer h from the bottom, y from the back (0) to the front, x across
the width as seen from the front. The back picture is mirrored so its columns line up with the front's.

CLI (no model; builds from walls already drawn):
  python scripts/stack_building.py --root R --stem S --walls DIR [--look crisp|smooth] [--size N]
      [--awning auto|yes|no] [--wide N] [--deep N] [--render PNG]
DIR holds base.png, front.png and (optional) back.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from flat2d import pixelart  # noqa: E402
from flat2d.parts import ground_key  # noqa: E402

VIEWS = ("top", "side", "front", "back")
WALLS = ("base", "front", "back")
LOOKS = ("crisp", "smooth")
LOOK = "crisp"
SIZE = 160  # cells across the wall: both proven buildings were made at 160
COLOURS = 24  # the crisp look's palette (pixelart.clean_folder)
KEY = (255, 0, 255)
SHEET_PX = 1152  # flat/sheet.png: the four views 2x2 (front, side / back, top), as the pixel sheet is

# walls_to_views
CORNICE_SHARE = 0.7  # rows narrower than this share of the widest row are above the cornice (a chimney stub)
CORNICE_BAND = (4, 14)  # px below the wall's top edge where the cornice colour is read for the roof's parapet
ROOF_RIM_SHARE = 1 / 28  # the roof plan's parapet: this share of the wall's width ...
ROOF_RIM_MIN = 6  # ... and at least this many px
ROOF_LINE = 3  # px of dark line drawn around the parapet and the deck
ROOF_LINE_RGB = (30, 30, 34)
DECK_RGB = (128, 134, 140)  # the roof deck's grey: nothing in an elevation says what the deck looks like
BASE_BAND = 0.06  # a strip under the wall (pavement, ground line) is at most this share of its height
BASE_REF = 0.2  # ... judged against the wall rows in this share of the height above it
BASE_TOL = 0.02  # a row reaching this share of the width past those walls (either side) is strip, not wall
# wall_checks (a flaw is recorded in views.json; only an empty wall stops the build)
SHAPE_AGREE = 0.9  # the plain wall and the back (and the plain wall and its mirror) overlap at least this much
FRONT_AGREE = 0.85  # the front adds a feature, so its outline may differ a little more
UPPER_DIFF = 60.0  # mean summed-RGB difference above which the front's upper half is not the plain wall's
FIT_SLOPE = (0.7, 1.4)  # a colour fit steeper or flatter than this is not a brightness shift
EMPTY_SHARE = 0.2  # a wall covering less than this share of the plain wall's box is missing
# build_shape
AWNING_OUT_SHARE = 14 / 160  # how far the awning sticks out, as a share of the wall's width (14 cells at 160)
PARAPET_SHARE = 4 / 160  # how high the parapet stands over the deck, as a share of the width (4 layers at 160)
AWNING_DIFF = 90  # summed-RGB difference that counts as "the front differs from the back" in a cell
AWNING_TOP = 0.5  # the awning's top edge: the highest layer where this share of the width differs ...
AWNING_MIN_SHARE = 0.03  # ... in a run of at least this share of the height
AWNING_BAND = 0.76  # the awning runs down while one of the AWNING_LOOK layers below differs over this share
AWNING_LOOK = 4
AWNING_BAND_OWN = 0.9  # ... or over this share of how much its own top layers differ, when that is less ...
AWNING_MAX_SHARE = 0.16  # ... while the band stays under this share of the height
AWNING_COLUMN = 0.5  # a column is the awning's when it differs over this share of the awning's layers
AWNING_QUIET = 0.75  # an awning layer differs over this share of the quiet columns (plain wall below it) ...
AWNING_QUIET_SHARE = 0.04  # ... in at least this share of the height (about 6 layers at 147, 8 at 199)
AWNING_QUIET_MIN = 3  # fewer quiet columns than this and the shop front leaves no plain wall to test on
# plain_wall_door (the plain wall is every side: a door drawn on it shows on all four)
DOOR_ROWS = (0.03, 0.05, 0.07, 0.09, 0.11, 0.13)  # rows probed, as shares of the wall's height above its bottom
DOOR_EDGE = 0.12  # the corner trim each side, as a share of the width, is left out of the probe
DOOR_COMMON = 0.2  # a colour seen in both this share of a probed row's left end and its right end is the wall's
DOOR_RUN = 0.06  # a run of other colours this wide (share of the width) is an opening that reaches the ground ...
DOOR_HITS = 3  # ... and one in at least this many probed rows is a door (the 2026-11-21 walls: plain walls 0 under 0.03, doors 6 of 6)
DECK_TOL = 30  # summed-RGB distance from the top view's centre colour that is still deck
CRISP_DISTINCT = 64  # a face with at most this many colours takes its parapet colour by vote, else by median
BAY_ROWS = (0.08, 0.55)  # the window bays are read in these rows of the plain wall (top floors; under the cornice)
BAY_DARK = 0.55  # a cell darker than this share of the wall's median brightness is glass ...
BAY_COLUMN = 0.15  # ... and a column with glass in over this share of those rows is a window column
BAY_OUTLINE = 0.9
BAY_GAP = 3  # window columns this close (mullions, frames) are one window
BAY_MIN = 0.05  # a window narrower than this share of the width is not one
BAY_NARROW = 0.6  # ... and one narrower than this share of the typical window is trim (dark corner stones)
BAY_MAX = 12  # at most this many bays on a face


class WallsError(RuntimeError):
    """The walls cannot make a building (missing, empty, the wrong size, no awning where one was asked for)."""


# ------------------------------------------------------------------------------------------------ pictures -> views


def _rgb(pic: Image.Image | np.ndarray | Path | str) -> np.ndarray:
    if isinstance(pic, (str, Path)):
        if not Path(pic).is_file():
            raise WallsError(f"no wall picture at {pic}")
        pic = Image.open(pic)
    if isinstance(pic, Image.Image):
        pic = getattr(pic, "image", pic)
        return np.asarray(pic.convert("RGB")).copy()
    return np.asarray(pic, np.uint8)[..., :3].copy()


def _mask(rgb: np.ndarray) -> np.ndarray:
    return pixelart.subject_mask(rgb, ground_key(Image.fromarray(rgb)))


def _below_cornice(fg: np.ndarray) -> np.ndarray:
    """The wall without what stands above the cornice (a chimney stub): rows above the first row as wide as
    CORNICE_SHARE of the widest row are dropped."""
    widths = fg.sum(axis=1)
    if widths.max() == 0:
        return fg
    top = int(np.argmax(widths > CORNICE_SHARE * widths.max()))
    out = fg.copy()
    out[:top] = False
    return out


def base_rows(fg: np.ndarray) -> int:
    """How many of the wall's bottom rows are a strip drawn under it (a grey pavement, a ground line) despite the
    prompt: rows that reach past the walls above them on either side, at most BASE_BAND of the height. The walls
    are the median ends of the rows in the BASE_REF band above (a cornice or awning higher up does not count)."""
    rows = np.nonzero(fg.any(1))[0]
    if len(rows) < 10:
        return 0
    top, bottom = int(rows[0]), int(rows[-1])
    h = bottom - top + 1
    cap = max(2, int(round(BASE_BAND * h)))
    ref = [r for r in range(bottom - int(round(BASE_REF * h)), bottom - cap + 1) if fg[r].any()]
    if len(ref) < 3:
        return 0
    left = float(np.median([int(np.argmax(fg[r])) for r in ref]))
    right = float(np.median([fg.shape[1] - 1 - int(np.argmax(fg[r][::-1])) for r in ref]))
    tol = max(2.0, BASE_TOL * (right - left + 1))
    cut = 0
    for r in range(bottom, bottom - cap, -1):
        xs = np.nonzero(fg[r])[0]
        if not len(xs) or xs[0] < left - tol or xs[-1] > right + tol:
            cut += 1
            continue
        break
    return cut if cut and any(fg[r].any() for r in range(bottom - cut + 1, bottom + 1)) else 0


def plain_wall_door(pic) -> dict[str, Any]:
    """Whether a plain wall picture has an opening that reaches the ground (a door, a shop front): a run of colours
    that are not the wall's, the plinth's or the trim's, across the bottom rows of the wall. Windows stand on sills
    above those rows. Returns {door, hits, runs}."""
    rgb = _rgb(pic)
    fg = _below_cornice(_mask(rgb))
    rows = np.nonzero(fg.any(1))[0]
    if len(rows) < 10:
        raise WallsError("the plain wall is empty")
    bottom = int(rows[-1]) - base_rows(fg)
    h = bottom - int(rows[0]) + 1
    xs = np.nonzero(fg[bottom - h // 3])[0]
    x0, x1 = int(xs[0]), int(xs[-1])
    w = x1 - x0 + 1
    edge = int(DOOR_EDGE * w)
    runs = []
    for share in DOOR_ROWS:
        px = rgb[bottom - int(share * h), x0 + edge : x1 - edge + 1].astype(int) // 32
        keys = px[:, 0] * 64 + px[:, 1] * 8 + px[:, 2]
        # The wall's own colours (its material, the plinth, the trim) run along the whole row, so they show at both
        # ends of it; a door in between does not, however flat its colour.
        end = max(1, int(DOOR_COMMON * len(keys)))
        odd = ~(np.isin(keys, keys[:end]) & np.isin(keys, keys[-end:]))
        best = run = 0
        for o in odd:
            run = run + 1 if o else 0
            best = max(best, run)
        runs.append(round(best / w, 3))
    hits = sum(r >= DOOR_RUN for r in runs)
    return {"door": hits >= DOOR_HITS, "hits": hits, "runs": runs}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def fit_colours(base: np.ndarray, target: np.ndarray, mask: np.ndarray, robust: bool = False) -> tuple[np.ndarray, list]:
    """base with each channel fitted (a * v + b) to target's on mask. robust: fit again on the 70% of pixels that
    fit best (the front's shop front is not the plain wall, so it must not pull the fit)."""
    out = base.copy()
    fits = []
    for c in range(3):
        x, y = base[..., c][mask].astype(float), target[..., c][mask].astype(float)
        a, b = np.polyfit(x, y, 1)
        if robust:
            res = np.abs(a * x + b - y)
            keep = res <= np.quantile(res, 0.7)
            a, b = np.polyfit(x[keep], y[keep], 1)
        out[..., c] = np.clip(a * base[..., c] + b, 0, 255).astype(np.uint8)
        fits.append([round(float(a), 3), round(float(b), 1)])
    return out, fits


def wall_checks(base_fg: np.ndarray, front_fg: np.ndarray, back_fg: np.ndarray | None, base: np.ndarray,
                front: np.ndarray, box: tuple[int, int, int, int], fits: list) -> tuple[list[str], dict]:
    """Flaws of the three walls, all inside the plain wall's box: the back is the plain wall's shape (drift:back),
    the front keeps the plain wall's outline (drift:front) and its upper half (upper:front), the plain wall is
    symmetric (asymmetric: one picture is both side walls), and the colour fit is a brightness shift (colour)."""
    y0, y1, x0, x1 = box
    flaws: list[str] = []
    m: dict[str, Any] = {}
    m["front_iou"] = round(_iou(base_fg, front_fg), 3)
    if m["front_iou"] < FRONT_AGREE:
        flaws.append("drift:front")
    if back_fg is not None:
        m["back_iou"] = round(_iou(base_fg, back_fg), 3)
        if m["back_iou"] < SHAPE_AGREE:
            flaws.append("drift:back")
    inner = base_fg[y0:y1, x0:x1]
    m["mirror_iou"] = round(_iou(inner, inner[:, ::-1]), 3)
    if m["mirror_iou"] < SHAPE_AGREE:
        flaws.append("asymmetric")
    mid = (y0 + y1) // 2
    both = base_fg[y0:mid] & front_fg[y0:mid]
    if both.any():
        d = np.abs(base[y0:mid].astype(int) - front[y0:mid].astype(int)).sum(-1)[both]
        m["upper_diff"] = round(float(d.mean()), 1)
        if m["upper_diff"] > UPPER_DIFF:
            flaws.append("upper:front")
    if any(not FIT_SLOPE[0] <= a <= FIT_SLOPE[1] for a, _ in fits):
        flaws.append("colour")
    return flaws, m


def roof_plan(width: int, cornice: np.ndarray, side_px: int) -> np.ndarray:
    """The roof seen from above, made in code: a square as wide as the walls, a parapet in the cornice colour with
    a dark line round it, and a grey deck."""
    top = np.full((side_px, side_px, 3), KEY, np.uint8)
    x0 = (side_px - width) // 2
    rim = max(ROOF_RIM_MIN, int(width * ROOF_RIM_SHARE))
    L = ROOF_LINE
    top[x0 : x0 + width, x0 : x0 + width] = ROOF_LINE_RGB
    top[x0 + L : x0 + width - L, x0 + L : x0 + width - L] = cornice.astype(np.uint8)
    top[x0 + rim : x0 + width - rim, x0 + rim : x0 + width - rim] = ROOF_LINE_RGB
    top[x0 + rim + L : x0 + width - rim - L, x0 + rim + L : x0 + width - rim - L] = DECK_RGB
    return top


def _write(path: Path, data: Image.Image | str) -> None:
    """Write a picture (PNG) or text, leaving the file alone when it already holds the same bytes: a Version is told
    apart by its files' size and time, so building again from the same walls must not make a new concept Version."""
    import io

    if isinstance(data, Image.Image):
        buf = io.BytesIO()
        data.save(buf, format="PNG")
        raw = buf.getvalue()
    else:
        raw = data.encode("utf-8")
    try:
        if path.read_bytes() == raw:
            return
    except OSError:
        pass
    path.write_bytes(raw)


def walls_to_views(root: Path, stem: str, base, front, back=None, meta: dict | None = None) -> dict:
    """Turn the drawn walls (paths, PIL images or arrays; back may be None: the plain wall is the back too) into
    out/<stem>/flat/views/{top,side,front,back}.png, flat/sheet.png, flat/source.png (the front), flat/key.json,
    flat/views.json (layout "walls") and flat/walls/ (the pictures as drawn). Writes only those files.
    meta: extra fields for views.json (the prompts, seeds and models that drew the walls). Returns views.json."""
    pics = {"base": _rgb(base), "front": _rgb(front)}
    if back is not None:
        pics["back"] = _rgb(back)
    shapes = {k: v.shape for k, v in pics.items()}
    if len(set(shapes.values())) != 1:
        raise WallsError(f"the walls must be one size: {', '.join(f'{k} {v[1]}x{v[0]}' for k, v in shapes.items())}")
    masks = {k: _mask(v) for k, v in pics.items()}
    for k, fg in masks.items():
        if fg.sum() < 0.01 * fg.size:
            raise WallsError(f"the {k} wall is empty")
    # The edit model brightens what it redraws. The plain wall and the back are the same picture but for the door,
    # so the plain wall's colours are fitted to the back's, channel by channel, and the sides match the front and
    # back. With no back the fit is to the front, on the pixels that fit best (the shop front is not the wall).
    if "back" in pics:
        side, fits = fit_colours(pics["base"], pics["back"], masks["base"] & masks["back"])
    else:
        side, fits = fit_colours(pics["base"], pics["front"], masks["base"] & masks["front"], robust=True)
    wall = _below_cornice(masks["base"])
    ys, xs = np.nonzero(wall)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    cut = base_rows(wall)
    y1 -= cut
    box = np.zeros(wall.shape, bool)
    box[y0:y1, x0:x1] = True
    width = x1 - x0
    if width < 16 or y1 - y0 < 16:
        raise WallsError("the plain wall is too small to build from")
    panels: dict[str, np.ndarray] = {}
    fgs: dict[str, np.ndarray] = {}
    for name, rgb, fg in (("side", side, masks["base"]), ("front", pics["front"], masks["front"]),
                          ("back", pics.get("back", side), masks.get("back", masks["base"]))):
        # Every wall is cut to the plain wall's box: what an edit hangs past the corners (an awning) is trimmed.
        fg = _below_cornice(fg) & box
        if fg.sum() < EMPTY_SHARE * box.sum():
            raise WallsError(f"the {name} wall does not cover the plain wall's box")
        out = rgb.copy()
        out[~fg] = KEY
        panels[name], fgs[name] = out, fg
    flaws, measures = wall_checks(fgs["side"], fgs["front"], fgs["back"] if "back" in pics else None,
                                  side, pics["front"], (y0, y1, x0, x1), fits)
    band = slice(y0 + CORNICE_BAND[0], y0 + CORNICE_BAND[1])
    cornice = np.median(side[band][fgs["side"][band]], axis=0)
    side_px = max(wall.shape)
    panels["top"] = roof_plan(width, cornice, side_px)

    flat = Path(root) / "out" / stem / "flat"
    views = flat / "views"
    views.mkdir(parents=True, exist_ok=True)
    for name in VIEWS:
        _write(views / f"{name}.png", Image.fromarray(panels[name]))
    sheet = Image.new("RGB", (2 * side_px, 2 * side_px), KEY)
    for i, name in enumerate(("front", "side", "back", "top")):
        im = Image.fromarray(panels[name])
        sheet.paste(im, ((i % 2) * side_px, (i // 2) * side_px))
    _write(flat / "sheet.png", sheet.resize((SHEET_PX, SHEET_PX), Image.LANCZOS))
    _write(flat / "source.png", Image.fromarray(panels["front"]))
    key = ground_key(Image.fromarray(pics["front"]))
    _write(flat / "key.json", json.dumps({**key, "views": "flat/views.json"}, indent=2) + "\n")
    drawn = flat / "walls"
    drawn.mkdir(exist_ok=True)
    for name, rgb in pics.items():
        _write(drawn / f"{name}.png", Image.fromarray(rgb))
    if "back" not in pics:
        (drawn / "back.png").unlink(missing_ok=True)
    notes = ["walls: one plain wall drawn by the local model is both sides; the front"
             + (" and back are" if "back" in pics else " is")
             + " that wall edited by the local edit model; the roof is made in code; sheet order front, side, back, "
             "roof"]
    if "back" not in pics:
        notes.append("no back feature: the back is the plain wall")
    if cut:
        notes.append(f"a strip {cut} px high under the plain wall was cut off")
    doc = {"version": 1, "layout": "walls", "canonical": "front",
           "source_sha256": hashlib.sha256((flat / "source.png").read_bytes()).hexdigest(),
           "views": [{"id": n, "file": f"views/{n}.png"} for n in VIEWS],
           "consistent": not flaws, "flaws": flaws, "notes": notes,
           "walls": {"files": {n: f"walls/{n}.png" for n in pics}, "box": [x0, y0, x1, y1], "base_cut": cut,
                     "colour_fit": {"to": "back" if "back" in pics else "front", "rgb": fits},
                     "checks": measures},
           **(meta or {})}
    _write(flat / "views.json", json.dumps(doc, indent=2) + "\n")
    return doc


# ----------------------------------------------------------------------------------------------------- the clean


def smooth_folder(flat: Path) -> None:
    """The smooth look: each cell of the crisp clean (flat/pixel/) recoloured with the area average of the drawn
    view's own pixels in it (the magenta ground never tints an edge), no colour limit. The crisp clean's cells
    (which are solid, the outline) are kept, so both looks have the same size and silhouette."""
    for name in VIEWS:
        cleaned = Image.open(flat / "pixel" / f"{name}.png").convert("RGBA")
        w, h = cleaned.size
        panel = Image.open(flat / "views" / f"{name}.png").convert("RGB")
        rgb = np.asarray(panel)
        fg = pixelart.subject_mask(rgb, ground_key(panel))
        x0, y0, x1, y1 = pixelart.bbox(fg)
        rgb, fg = rgb[y0:y1, x0:x1].astype(np.float32), fg[y0:y1, x0:x1].astype(np.float32)
        cover = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).resize((w, h), Image.BOX), np.float32) / 255
        cols = [np.asarray(Image.fromarray(rgb[..., c] * fg).convert("F").resize((w, h), Image.BOX), np.float32)
                for c in range(3)]
        avg = np.stack(cols, axis=-1) / np.maximum(cover[..., None], 1e-3)
        a = np.asarray(cleaned)
        solid = a[..., 3] > 127
        res = np.zeros((h, w, 4), np.uint8)
        res[..., :3] = np.where((cover > 0.05)[..., None], np.clip(avg, 0, 255), a[..., :3])
        res[..., 3] = np.where(solid, 255, 0)
        res[~solid] = 0
        Image.fromarray(res).save(flat / "pixel" / f"{name}.png")


def clean_walls(root: Path, stem: str, look: str = LOOK, size: int = SIZE) -> dict:
    """flat/views/ cleaned into flat/pixel/ in the given look, size cells across the wall. Returns pixel.json."""
    if look not in LOOKS:
        raise WallsError(f"look must be one of {', '.join(LOOKS)}")
    flat = Path(root) / "out" / stem / "flat"
    try:
        doc = pixelart.clean_folder(flat, colours=COLOURS, target=int(size), grid=False)
    except ValueError as e:
        raise WallsError(f"could not clean the walls: {e}") from None
    if look == "smooth":
        smooth_folder(flat)
        imgs = {n: Image.open(flat / "pixel" / f"{n}.png").convert("RGBA") for n in VIEWS}
        pixelart.preview(imgs).save(flat / "pixel" / "preview.png", format="PNG")
        doc["notes"] = list(doc.get("notes") or []) + [
            "smooth: each cell is the area average of the drawing; the palette is the crisp clean's, before averaging"]
    doc["look"] = look
    (flat / "pixel" / "pixel.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


# ------------------------------------------------------------------------------------------------------ the bays


def find_bays(wall: np.ndarray) -> list[tuple[int, int]]:
    """The window bays of a cleaned wall (RGBA, flat/pixel/), left to right: one per window column, each running
    from the middle of the pier on its left to the middle of the pier on its right (the outer bays as wide as the
    others). What lies outside the bays is the corner trim. Windows are found as dark glass in the top floors."""
    H, W = wall.shape[:2]
    solid = wall[..., 3] > 127
    lum = wall[..., :3].astype(np.float32) @ np.array([0.3, 0.59, 0.11], np.float32)
    rows = slice(int(BAY_ROWS[0] * H), int(BAY_ROWS[1] * H))
    ref = float(np.median(lum[rows][solid[rows]])) if solid[rows].any() else 0.0
    share = ((lum[rows] < BAY_DARK * ref) & solid[rows]).mean(axis=0)
    dark = (share > BAY_COLUMN) & (share < BAY_OUTLINE)  # a column dark nearly all the way is an outline, not glass
    runs: list[list[int]] = []
    for x in np.flatnonzero(dark):
        if runs and x - runs[-1][1] <= BAY_GAP:
            runs[-1][1] = int(x) + 1
        else:
            runs.append([int(x), int(x) + 1])
    runs = [r for r in runs if r[1] - r[0] >= BAY_MIN * W and r[0] > 0 and r[1] < W]
    if not runs:
        return []
    # Dark corner stones (quoins) are narrower than the windows: keep the runs near the widest ones.
    typical = float(np.median([r[1] - r[0] for r in runs]))
    runs = [r for r in runs if r[1] - r[0] >= BAY_NARROW * typical]
    if len(runs) == 1:
        return [(runs[0][0], runs[0][1])]
    piers = [b[0] - a[1] for a, b in zip(runs, runs[1:])]
    half = int(round(float(np.mean(piers)) / 2))
    cuts = [max(0, runs[0][0] - half)] + [(a[1] + b[0]) // 2 for a, b in zip(runs, runs[1:])] + \
           [min(W, runs[-1][1] + half)]
    return list(zip(cuts, cuts[1:]))


def bay_order(bays: list[tuple[int, int]], width: int, n: int) -> list[int]:
    """Which drawn bays make a face of n bays, left to right. The bays at the middle of the wall (the door, under a
    shop front or a back door) are kept once; bays beside them are dropped from the outside in, or repeated (a copy
    of the bay next to the middle), one side then the other, so the face stays as near symmetric as it was."""
    m = len(bays)
    if not 1 <= n <= BAY_MAX:
        raise WallsError(f"a face is 1 to {BAY_MAX} bays")
    mid = width / 2
    centre = [i for i, (a, b) in enumerate(bays) if a - 2 <= mid <= b + 2]
    if not centre:
        centre = [min(range(m), key=lambda i: abs((bays[i][0] + bays[i][1]) / 2 - mid))]
    if n < len(centre):
        raise WallsError(f"the middle of this wall is {len(centre)} bays (the door), so a face needs at least "
                         f"{len(centre)}")
    left, right = list(range(centre[0])), list(range(centre[-1] + 1, m))
    side = 0 if len(left) >= len(right) else 1
    while len(left) + len(centre) + len(right) > n:
        (left.pop(0) if (side == 0 and left) or not right else right.pop())
        side ^= 1
    while len(left) + len(centre) + len(right) < n:
        if side == 0:
            left.append(left[-1] if left else centre[0])
        else:
            right.insert(0, right[0] if right else centre[-1])
        side ^= 1
    return left + centre + right


def regroup(wall: np.ndarray, bays: list[tuple[int, int]], order: list[int]) -> np.ndarray:
    """The wall with its bays put back in the given order, the corner trim each side kept."""
    parts = [wall[:, : bays[0][0]]] + [wall[:, bays[i][0] : bays[i][1]] for i in order] + [wall[:, bays[-1][1] :]]
    return np.concatenate(parts, axis=1)


def set_bays(view: dict[str, np.ndarray], wide: int | None = None, deep: int | None = None) -> dict[str, Any]:
    """The cleaned views with the front and back cut or grown to `wide` bays and the side to `deep` bays (None:
    as drawn). The bays are read once, on the plain wall (the side), and used on all three: the edit model keeps
    the plain wall's windows where they were. Returns {view, bays, wide, deep, front_order, side_order}."""
    side = view["side"]
    bays = find_bays(side)
    out = dict(view)
    info: dict[str, Any] = {"bays": [list(b) for b in bays], "drawn": len(bays)}
    if (wide is not None or deep is not None) and not bays:
        raise WallsError("no window bays were found on the plain wall, so its width and depth cannot be changed")
    W = side.shape[1]
    for key, n, faces in (("wide", wide, ("front", "back")), ("deep", deep, ("side",))):
        if n is None:
            info[key] = len(bays)
            continue
        order = bay_order(bays, W, int(n))
        for f in faces:  # every wall is drawn from the plain wall, so its columns line up with the bays as drawn
            out[f] = regroup(view[f], bays, order)
        info[key], info[f"{key}_order"] = int(n), order
    info["view"] = out
    return info


# ----------------------------------------------------------------------------------------------------- the shape


def _paint(px: np.ndarray, look: str) -> np.ndarray:
    """One colour for a set of RGBA pixels: the most common in the crisp look (a palette colour stays one), the
    per-channel median in the smooth look (no colour is common there)."""
    px = px.reshape(-1, 4)
    if look == "crisp" or len(np.unique(px, axis=0)) <= CRISP_DISTINCT:
        cols, n = np.unique(px, axis=0, return_counts=True)
        return cols[n.argmax()].astype(np.uint8)
    return np.median(px, axis=0).astype(np.uint8)


def deck_inset(top: np.ndarray) -> tuple[int, int, int, int]:
    """Cells from the roof plan's left, back, right and front edge to its deck: the deck is the region of the
    plan's centre colour that holds the centre."""
    from scipy import ndimage

    solid = top[..., 3] > 127
    ys, xs = np.nonzero(solid)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    near = solid & (np.abs(top[..., :3].astype(int) - top[cy, cx, :3].astype(int)).sum(-1) <= DECK_TOL)
    lab, _ = ndimage.label(near)
    dy, dx = np.nonzero(lab == lab[cy, cx])
    return int(dx.min() - x0), int(dy.min() - y0), int(x1 - dx.max()), int(y1 - dy.max())


def find_awning(front: np.ndarray, back: np.ndarray, fm: np.ndarray, bm: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    """The awning's layers (h0 lowest, h1 highest) and its columns, or None. The awning is what the front has that
    the back does not right across the wall: the layers where the two pictures differ over nearly the whole width
    (shop windows and a door leave gaps of plain wall; an awning does not). Inputs are layer-ordered (row 0 the
    bottom), the back already mirrored."""
    H, W = fm.shape
    diff = fm & bm & (np.abs(front[..., :3].astype(int) - back[..., :3].astype(int)).sum(-1) > AWNING_DIFF)
    cover = diff.sum(axis=1) / max(1, W)
    # The awning's top edge: the top of the highest run of at least AWNING_MIN_SHARE of the height of layers that
    # differ over AWNING_TOP of the width (a cornice layer or two the cleaner snapped differently is not an awning).
    need = max(2, int(round(AWNING_MIN_SHARE * H)))
    h1, run = -1, 0
    for h, over in enumerate(cover > AWNING_TOP):
        run = run + 1 if over else 0
        if run >= need:
            h1 = h  # a run that reaches the length keeps raising h1 to its top as it goes on
    if h1 < 0:
        return None
    def walk_down(band: float) -> int:
        h = h1
        while h > AWNING_LOOK and cover[h - AWNING_LOOK : h].max() > band:
            h -= 1
        return h

    h0 = walk_down(AWNING_BAND)
    # An awning set inside the corner trim spans less of the wall, so its band may also be measured against its own
    # top layers; that is only an awning while it stays a band (a shop front with no awning runs down the whole floor).
    own = walk_down(min(AWNING_BAND, AWNING_BAND_OWN * float(cover[h1 - need + 1 : h1 + 1].mean())))
    if h1 - own <= AWNING_MAX_SHARE * H:
        h0 = min(h0, own)
    # The awning's columns: those that differ over at least AWNING_COLUMN of its layers (a corner trim beside it, which
    # an edit may touch in a layer or two, is wall and must not stick out).
    cols = np.nonzero(diff[h0 : h1 + 1].mean(axis=0) >= AWNING_COLUMN)[0]
    if not len(cols):
        return None
    # A shop front with no awning differs over most of the width too (big windows, a sign band). What only an awning
    # does is cover the quiet columns: those, inside its span, that are plain wall in most layers below it (the wall
    # between the windows and the door). The band must differ over AWNING_QUIET of them in enough of its layers.
    quiet = np.zeros(W, bool)
    quiet[cols.min() : cols.max() + 1] = True
    if h0 > 0:
        quiet &= diff[:h0].mean(axis=0) < AWNING_COLUMN
    if quiet.sum() >= AWNING_QUIET_MIN:
        covered = (diff[h0 : h1 + 1][:, quiet].mean(axis=1) >= AWNING_QUIET).sum()
        if covered < max(2, int(round(AWNING_QUIET_SHARE * H))):
            return None
    return h0, h1, cols


def awning_of(view: dict[str, np.ndarray]) -> tuple[int, int, np.ndarray] | None:
    """find_awning on cleaned views as build_shape reads them (layers from the bottom, the back mirrored)."""
    rows = np.arange(view["front"].shape[0])[::-1]
    fr, bk = view["front"][rows], view["back"][:, ::-1][rows]
    return find_awning(fr, bk, fr[..., 3] > 127, bk[..., 3] > 127)


def build_shape(view: dict[str, np.ndarray], awning: str = "auto", look: str = LOOK, found: Any = "find",
                scale: int | None = None) -> dict[str, Any]:
    """The building's solid cells and colours from the cleaned views (RGBA arrays, flat/pixel/). awning: "auto"
    builds the wedge when the pictures show one, "yes" requires one (WallsError when none is found: no stack is
    written), "no" never builds one. found: the awning (h0, h1, columns) or None, when it was looked for already
    (on the walls as drawn: repeated shop windows can look like an awning); "find" looks here. scale: the width the
    awning's reach and the parapet are measured against (default the front's; set_bays passes the drawn one).
    Returns {solid [H, L, W], color [H, L, W, 4], depth, awning, front_face and
    back_face (each layer's front and back wall row, y), ...}."""
    if awning not in ("auto", "yes", "no"):
        raise WallsError("awning must be auto, yes or no")
    front, back, side = view["front"], view["back"][:, ::-1], view["side"]
    H, W = front.shape[:2]
    D = side.shape[1]  # the block's depth, back to front: the plain wall is the side walls
    if back.shape[:2] != (H, W) or side.shape[0] != H:
        raise WallsError(f"the cleaned walls disagree on size: front {W}x{H}, back {back.shape[1]}x{back.shape[0]}, "
                         f"side {side.shape[1]}x{side.shape[0]}")
    A = max(1, int(round(AWNING_OUT_SHARE * (scale or W))))
    P = max(1, int(round(PARAPET_SHARE * (scale or W))))
    rows = np.arange(H)[::-1]  # layer h is picture row H-1-h
    fr, bk, sd = front[rows], back[rows], side[rows]
    fm, bm, sm = fr[..., 3] > 127, bk[..., 3] > 127, sd[..., 3] > 127
    if awning == "no":
        found = None
    elif isinstance(found, str):
        found = find_awning(fr, bk, fm, bm)
    if awning == "yes" and found is None:
        raise WallsError("an awning was asked for, but no band where the front differs from the back right across "
                         "the wall (over the plain wall between the windows too) was found; no stack was written")
    L = D + (A if found else 0)
    wall = bm & bm[:, ::-1]  # the block's outline across the width: the back has no awning on it
    solid = np.zeros((H, L, W), bool)
    color = np.zeros((H, L, W, 4), np.uint8)
    # 1. The block: each layer is the side wall's extent (depth) times the back wall's extent (width).
    solid[:, :D, :] = sm[:, :, None] & wall[:, None, :]
    color[:, :D, :] = sd[:, :, None, :]
    ys = np.arange(D)[None, :, None]
    y_front = np.where(sm.any(axis=1), D - 1 - np.argmax(sm[:, ::-1], axis=1), 0)[:, None, None]
    y_back = np.argmax(sm, axis=1)[:, None, None]
    is_front = np.broadcast_to(ys == y_front, (H, D, W)) & solid[:, :D]
    is_back = np.broadcast_to(ys == y_back, (H, D, W)) & solid[:, :D]
    color[:, :D][is_back] = np.broadcast_to(bk[:, None], (H, D, W, 4))[is_back]
    color[:, :D][is_front] = np.broadcast_to(fr[:, None], (H, D, W, 4))[is_front]
    # 2. The roof: the deck sunk P layers inside a parapet, inset as the roof plan's deck is, in the cornice colour.
    top_h = H - 1
    parapet = _paint(fr[H - 6 : H - 1][fm[H - 6 : H - 1]], look)
    il, ib, ir, ifr = deck_inset(view["top"])
    tw, tl = view["top"].shape[1], view["top"].shape[0]
    foot = solid[top_h - P]
    fy, fx = np.nonzero(foot)
    if not len(fy):
        raise WallsError("the block has no roof layer")
    # The roof plan's insets, scaled from its size to the footprint's.
    sy, sx = (fy.max() - fy.min() + 1) / tl, (fx.max() - fx.min() + 1) / tw
    inner = np.zeros((L, W), bool)
    inner[fy.min() + int(round(ib * sy)) : fy.max() + 1 - int(round(ifr * sy)),
          fx.min() + int(round(il * sx)) : fx.max() + 1 - int(round(ir * sx))] = True
    deck = view["top"][tl // 2, tw // 2]  # the roof plan's centre is deck
    for h in range(top_h - P + 1, H):
        solid[h][inner] = False
        color[h][solid[h]] = parapet
    color[top_h - P][inner & solid[top_h - P]] = deck
    # 3. The awning: a wedge out of the front wall, widest at its bottom edge, painted from the front picture.
    info: dict[str, Any] = {"found": bool(found), "asked": awning}
    if found:
        h0, h1, cols = found
        for h in range(h0, h1 + 1):
            out = int(round(A * (h1 - h + 1) / (h1 - h0 + 1)))
            yf = int(y_front[h, 0, 0])
            for x in range(int(cols.min()), int(cols.max()) + 1):
                if fm[h, x]:  # never a cell the front picture has no colour for
                    solid[h, yf + 1 : yf + 1 + out, x] = True
                    color[h, yf + 1 : yf + 1 + out, x] = fr[h, x]
        info.update({"layers": [h0, h1], "columns": [int(cols.min()), int(cols.max())], "out": A})
    color[..., 3] = np.where(solid, 255, 0)
    color[~solid] = 0
    return {"solid": solid, "color": color, "depth": D, "awning": info, "parapet": P,
            "deck_inset": [il, ib, ir, ifr], "front_face": y_front[:, 0, 0].astype(int), "back_face": y_back[:, 0, 0].astype(int)}


def stack_result(solid: np.ndarray, color: np.ndarray) -> dict[str, Any]:
    """The shape as stack_views.write_stack takes it: square slices, bottom first."""
    H, L, W = solid.shape
    res = max(L, W)
    oz, ox = (res - L) // 2, (res - W) // 2
    slices = []
    for h in range(H):
        sl = np.zeros((res, res, 4), np.uint8)
        sl[oz : oz + L, ox : ox + W] = color[h]
        slices.append(sl)
    pitch = 1.0 / res
    ext = [W * pitch, H * pitch, L * pitch]
    return {"slices": slices, "pitch": pitch, "origin": [-0.5, 0.0, -0.5], "resolution": int(res), "layers": H,
            "bounds": [[-ext[0] / 2, 0.0, -ext[2] / 2], [ext[0] / 2, ext[1], ext[2] / 2]], "extents": ext,
            "surface_voxels": int(solid.sum()), "filled": True,
            "size": {"width": int(W), "length": int(L), "height": int(H)},
            "colors": int(len(np.unique(color[solid][:, :3], axis=0)))}


def read_pixel(root: Path, stem: str) -> dict[str, np.ndarray]:
    folder = Path(root) / "out" / stem / "flat" / "pixel"
    out = {}
    for n in VIEWS:
        p = folder / f"{n}.png"
        if not p.is_file():
            raise WallsError(f"no cleaned {n} view at {p}")
        out[n] = np.asarray(Image.open(p).convert("RGBA"))
    return out


def write_building(root: Path, stem: str, look: str = LOOK, awning: str = "auto", wide: int | None = None,
                   deep: int | None = None) -> dict[str, Any]:
    """Build the shape from flat/pixel/ and write out/<stem>/stack/ (maker "walls"). wide, deep: window bays across
    the front and along the side (None: as drawn; see set_bays). Returns {meta (stack.json), result, shape}.
    Nothing is written when the shape cannot be built."""
    from stack_views import write_stack

    drawn = read_pixel(root, stem)
    bays = set_bays(drawn, wide, deep)
    found = None if awning == "no" else awning_of(drawn)
    if found is not None and "wide_order" in bays:  # the awning's columns, carried through the new bay order
        cols = np.zeros((1, drawn["front"].shape[1]), bool)
        cols[0, found[2]] = True
        found = (found[0], found[1], np.flatnonzero(regroup(cols, bays["bays"], bays["wide_order"])[0]))
    shape = build_shape(bays.pop("view"), awning=awning, look=look, found=found, scale=drawn["front"].shape[1])
    shape["bays"] = bays
    result = stack_result(shape["solid"], shape["color"])
    dest = Path(root) / "out" / stem / "stack"
    meta = write_stack(result, dest, "flat/pixel", stem)
    meta.update({"maker": "walls", "look": look, "shape": "code", "awning": shape["awning"], "bays": bays})
    (dest / "stack.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"meta": meta, "result": result, "shape": shape}


def build(root: Path, stem: str, walls: dict[str, Any], look: str = LOOK, size: int = SIZE, awning: str = "auto",
          meta: dict | None = None, line=print, check=None, carving=None, wide: int | None = None,
          deep: int | None = None) -> dict[str, Any]:
    """The whole pure part, saving Versions as the pixel stack draw does: the concept and stack on screen first (if
    never saved), then the concept (views), the flat (cleaned walls) and the stack, made from the concept.
    walls: {base, front, back?}. line(text): progress; check(): raises to cancel between steps; carving(): called
    when the walls are cleaned and the shape is next."""
    from flow import autosave

    if look not in LOOKS:
        raise WallsError(f"look must be one of {', '.join(LOOKS)}")
    if (Path(root) / "out" / stem).is_dir():
        autosave(root, stem, ("concept", "stack"))
    views = walls_to_views(root, stem, walls["base"], walls["front"], walls.get("back"), meta=meta)
    line("Walls: " + (", ".join(views["flaws"]) if views["flaws"] else "they agree") + ".")
    autosave(root, stem, ("concept",))
    if check:
        check()
    pixel = clean_walls(root, stem, look, size)
    s = pixel["sizes"]["final"]
    line(f"Cleaned ({look}): front {s['front'][0]}x{s['front'][1]}, side {s['side'][0]}x{s['side'][1]}.")
    autosave(root, stem, ("flat",))
    if check:
        check()
    if carving:
        carving()
    built = write_building(root, stem, look, awning, wide, deep)
    size_ = built["result"]["size"]
    aw = built["shape"]["awning"]
    bays = built["meta"]["bays"]
    line(f"Bays: {bays['wide']} across, {bays['deep']} deep (drawn {bays['drawn']}).")
    line(f"Stack built: {size_['width']} wide, {size_['length']} long, {size_['height']} layers, "
         f"{built['result']['colors']} colours; " + ("awning layers {}-{}.".format(*aw["layers"]) if aw["found"]
                                                     else "no awning."))
    autosave(root, stem, ("stack",), made_from=("concept",))
    return {"views": views, "pixel": pixel, "stack": built["meta"], "size": size_,
            "colors": built["result"]["colors"], "layers": built["result"]["layers"], "awning": aw,
            "flaws": views["flaws"]}


# ------------------------------------------------------------------------------------------------------ a look


def render_turned(slices: list[np.ndarray], path: Path, scale: int = 3, angles=(35, 145, 215, 325)) -> Path:
    """A turned look at a stack, the way a game draws one: every layer rotated, squashed and drawn a step higher,
    four angles in a 2x2 picture. For agents and people to look at a build (the app's Stack stage is the real view)."""
    res = slices[0].shape[0]
    H = len(slices)
    frames = []
    for ang in angles:
        side_px = int(res * 1.5)
        canvas = Image.new("RGBA", (side_px * scale, (int(side_px * 0.6) + H) * scale), (44, 46, 56, 255))
        for h, sl in enumerate(slices):
            im = Image.fromarray(sl).rotate(ang, Image.NEAREST, expand=True)
            im = im.resize((im.width * scale, max(1, int(im.height * 0.6)) * scale), Image.NEAREST)
            x = (canvas.width - im.width) // 2
            y = canvas.height - im.height - h * scale - 6
            for dy in range(scale):
                canvas.alpha_composite(im, (x, y - dy))
        frames.append(canvas.convert("RGB"))
    out = Image.new("RGB", (frames[0].width * 2, frames[0].height * 2))
    for i, f in enumerate(frames):
        out.paste(f, ((i % 2) * f.width, (i // 2) * f.height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


def read_sheet_slices(stack_dir: Path) -> list[np.ndarray]:
    """The slices of a written stack (stack/sheet.png, bottom first)."""
    meta = json.loads((Path(stack_dir) / "stack.json").read_text(encoding="utf-8"))
    sheet = np.asarray(Image.open(Path(stack_dir) / meta["sheet"]).convert("RGBA"))
    res = int(meta["resolution"])
    return [sheet[:, i * res : (i + 1) * res] for i in range(int(meta["layers"]))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=SCRIPTS.parent, help="factory root that holds out/")
    p.add_argument("--stem", required=True, help="out/<stem>/ folder")
    p.add_argument("--walls", type=Path, required=True, help="folder with base.png, front.png and (optional) back.png")
    p.add_argument("--look", choices=LOOKS, default=LOOK)
    p.add_argument("--size", type=int, default=SIZE, help=f"cells across the wall (default {SIZE})")
    p.add_argument("--awning", choices=("auto", "yes", "no"), default="auto",
                   help="yes: fail when no awning is found; no: never build one")
    p.add_argument("--wide", type=int, default=None, help="window bays across the front (default: as drawn)")
    p.add_argument("--deep", type=int, default=None, help="window bays along the side (default: as drawn)")
    p.add_argument("--render", type=Path, default=None, help="also write a turned 4-angle look to this PNG")
    args = p.parse_args()
    from factory_env import check_stem

    try:
        stem = check_stem(args.stem)
        from flat2d.pose import pixel_size

        size = pixel_size(args.size)
    except ValueError as e:
        print(f"stack_building: {e}", file=sys.stderr)
        return 2
    walls: dict[str, Any] = {n: args.walls / f"{n}.png" for n in ("base", "front")}
    if (args.walls / "back.png").is_file():
        walls["back"] = args.walls / "back.png"
    try:
        out = build(args.root, stem, walls, look=args.look, size=size, awning=args.awning,
                    line=lambda t: print(t, file=sys.stderr), wide=args.wide, deep=args.deep)
    except WallsError as e:
        print(f"stack_building: {e}", file=sys.stderr)
        return 2
    if args.render:
        render_turned(read_sheet_slices(args.root / "out" / stem / "stack"), args.render)
    print(json.dumps({"ok": True, "look": args.look, **out["size"], "layers": out["layers"], "colors": out["colors"],
                      "awning": out["awning"], "flaws": out["flaws"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
