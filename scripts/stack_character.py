#!/usr/bin/env python3
"""A sprite-stack character made of part stacks, one per body part (pipeline#2).

The method follows loktar00/sprite-stack-animation (MIT, see NOTICE): a character is a few small stacks, one per
body part, put on a skeleton every frame by code. This module makes the parts; the walk is code in the app
(desktop/frontend/src/screens/asset/stack/character/).

How the parts are drawn so they match: they are never drawn one by one. The local image model draws the WHOLE
character once, as the usual four-view pixel art sheet (top, side, front, back), standing straight with its arms
hanging down clear of the body and its legs a little apart. That sheet is cleaned and carved into one stack
(stack_views.py, the same route as Draw views & stack). Then the carved body is cut into parts in code, at joint
heights measured on the front view, so every part has the same palette, the same scale, and joints that meet.

Subcommands:
  draw   --stem S --prompt TEXT [--size N]   draw the sheet (image model, a few minutes), clean it, carve the body,
                                             save each as a Version (concept, flat, stack)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

ROOT = SCRIPTS.parent

# What the drawing must show so the body can be cut into parts: limbs clear of the body and of each other.
POSE = ("Standing straight and still in a neutral pose, facing forward, both arms hanging straight down at the sides "
        "with a clear gap between each arm and the body, both legs straight and slightly apart with a clear gap "
        "between them, feet flat.")


def draw(root: Path, stem: str, prompt: str, size: int = 96, style: str = "clean") -> dict:
    """Draw the character's sheet (image model), then cut it into parts and save the whole body and the parts
    pulled apart as two new Stack Versions. style 'clean' (the default) asks for clean game art with no pixel
    wording; 'pixel' asks for retro pixel art, which the model draws with big chunky pixels (Stephen, pipeline#2:
    "too heavy of a pixel art"). size: the character's height in cells, from the soles to the top of the head."""
    from flat2d import optional
    from flow import autosave

    asset = root / "out" / stem
    asset.mkdir(parents=True, exist_ok=True)
    intent = asset / "intent.json"
    if not intent.is_file():
        intent.write_text(json.dumps({"subject": "person", "output": "stack", "kind": "character",
                                      "made": int(time.time())}, indent=2) + "\n")
    text = " ".join(prompt.split()).rstrip(".") + ". " + POSE
    if (asset / "flat").is_dir():
        autosave(root, stem, ("concept",))
    t0 = time.time()
    saved = optional.draw(root, stem, text, views="pixel", size=min(size, 256), style=style)
    print(f"drawn in {time.time() - t0:.0f} s, seed {saved.get('seed')}, flaws {saved.get('flaws')}")
    (asset / "character").mkdir(parents=True, exist_ok=True)
    (asset / "character" / "draw.json").write_text(json.dumps(
        {"prompt": text, "style": style, "height": size, "seed": saved.get("seed")}, indent=2) + "\n")
    autosave(root, stem, ("concept",))
    if saved.get("pixel"):
        autosave(root, stem, ("flat",))
    return make_parts(root, stem, show=True)


# ---------------------------------------------------------------- cutting the carved body into parts

class PartsError(RuntimeError):
    pass


CHAR_VIEWS = ("front", "side", "back")  # the top view of a standing person is not used: the model cannot draw it
HEIGHT = 40  # cells from the soles to the top of the head when no pixel grid is found
OUTLINE_LUMA = 40  # darker than this is the drawing's outline (stack_views.DARK is 80: too dark-hair friendly)
KEEP = 0.05  # a blob under this share of the biggest is not the character (a panel's divider line, dust)


def clean_views(asset: Path, height: int = HEIGHT, grid: bool = True, colours: int = 24, despeckle: int = 0) -> dict:
    """The front, side and back views of the drawn sheet (flat/views/) as true pixel art at ONE scale: one palette
    for the three, the drawing's own pixel grid when one stands out (else the front's height cut into `height`
    cells), one colour per cell. The side and back are fitted to the front's height. pixelart.clean is not used:
    it fits every view to the length of a long object (a car), which squashes a standing person.
    colours: the palette size for the three views. despeckle: passes of a cleanup in which a cell whose colour
    hardly any of its eight neighbours share takes its neighbours' commonest colour (the drawing's soft shading
    otherwise comes out as speckle, one cell light, the next dark).
    Writes character/views/{front,side,back}.png and returns {id: (rgba, mask)}."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    from flat2d import pixelart as PA
    from flat2d.parts import ground_key

    views = {}
    for pid in CHAR_VIEWS:
        p = asset / "flat" / "views" / f"{pid}.png"
        if not p.is_file():
            raise PartsError(f"no drawn {pid} view at {p}")
        im = Image.open(p).convert("RGB")
        rgb = np.asarray(im)
        fg = PA.subject_mask(rgb, ground_key(im))
        labels, n = ndimage.label(fg)
        if n == 0:
            raise PartsError(f"the {pid} view is empty")
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        big = int(sizes.argmax())
        bx0, by0, bx1, by1 = PA.bbox(labels == big)
        keep = np.zeros(n + 1, bool)
        for k, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None or sizes[k] < KEEP * sizes[big]:
                continue
            y, x = sl
            # the character, and pieces of it inside its box; never a divider line running past it
            if x.start >= bx0 - 4 and x.stop <= bx1 + 4 and y.start >= by0 - 4 and y.stop <= by1 + 4:
                keep[k] = True
        fg = keep[labels]
        x0, y0, x1, y1 = PA.bbox(fg)
        views[pid] = (rgb[y0:y1, x0:x1], fg[y0:y1, x0:x1])
    pal = PA.build_palette(views, colours)
    idx = {pid: PA.snap(rgb, pal) for pid, (rgb, _) in views.items()}
    use_grid = grid
    grid = PA.find_grid(views)
    tall_px = views["front"][0].shape[0]
    cell = grid["cell"] if use_grid and grid["score"] >= PA.GRID_MIN_SCORE and 24 <= tall_px / max(grid["cell"], 1) <= 96 \
        else tall_px / height
    offs = grid["offsets"] if cell == grid["cell"] else {pid: [0.0, 0.0] for pid in CHAR_VIEWS}
    cells = {pid: PA.trim(PA.to_cells(idx[pid], views[pid][1], cell, tuple(offs[pid]))) for pid in CHAR_VIEWS}
    H = cells["front"].shape[0]
    for pid in ("side", "back"):
        c = cells[pid]
        w = cells["front"].shape[1] if pid == "back" else max(1, round(c.shape[1] * H / c.shape[0]))
        cells[pid] = PA.fit(c, w, H)
    for pid in CHAR_VIEWS:
        cells[pid] = _despeckle(cells[pid], pal, despeckle)
    out_dir = asset / "character" / "views"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for pid in CHAR_VIEWS:
        im = PA.rgba(cells[pid], pal)
        im.save(out_dir / f"{pid}.png")
        a = np.asarray(im)
        out[pid] = (a, a[..., 3] >= 128)
    print(f"views: cell {cell:.2f} px ({'grid' if cell == grid['cell'] else 'height'}), "
          + ", ".join(f"{k} {v[0].shape[1]}x{v[0].shape[0]}" for k, v in out.items()))
    return out


def _despeckle(cells, pal, passes: int, near: int = 80):
    """passes of: a subject cell sharing its colour with at most two of its 8 neighbours takes the commonest colour
    among them, when the two are near (shading: one cell light, the next dark). A cell far from its neighbours (an
    eye, a button) is a drawn feature and stays, unless it is alone on the rim. Ground (-1) never spreads and never
    changes."""
    import numpy as np
    from scipy.ndimage import convolve

    n = len(pal)
    p = pal.astype(np.int32)
    dist = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    k = np.ones((3, 3), int)
    k[1, 1] = 0
    for _ in range(passes):
        on = cells >= 0
        counts = np.stack([convolve((cells == i).astype(int), k, mode="constant") for i in range(n)])
        own = np.take_along_axis(counts, np.maximum(cells, 0)[None], 0)[0]
        best = counts.argmax(0)
        rim = on & (convolve((~on).astype(int), k, mode="constant", cval=1) > 0)
        # a lone cell on the rim is the ground's colour bled into the drawing's edge, near or not
        change = on & (counts.max(0) >= 4) & (own <= 2) & (dist[np.maximum(cells, 0), best] < near) | rim & (own <= 1) & (counts.max(0) >= 2)
        if not change.any():
            break
        cells = np.where(change, best, cells)
    return cells


def _round_fill(w: int, d: int, p: float = 2.5):
    """A rounded cross-section filling a w x d box (a superellipse); a box 2 or less across stays a box."""
    import numpy as np

    if w <= 2 or d <= 2:
        return np.ones((d, w), bool)
    v = (np.arange(d) + 0.5 - d / 2) / (d / 2)
    u = (np.arange(w) + 0.5 - w / 2) / (w / 2)
    return (np.abs(u)[None, :] ** p + np.abs(v)[:, None] ** p) <= 1.0


def carve_character(views: dict):
    """The body from the front and side silhouettes: (solid[H, L, W], rgba[H, L, W, 4]), layer 0 at the bottom,
    row 0 the back, column 0 the character's right (the viewer's left on the front view).

    Each layer: every run of the front silhouette (an arm, the body, a leg) gets the depth of the side silhouette
    at that height and a rounded cross-section. When a layer has three runs or more, its outer runs are arms, and the side view cannot tell an arm's depth from the body's
    behind it, so an arm is made about as deep as it is wide, in the middle of the side view's depth.
    Colours: a cell open to the front takes the front view's colour, open to the back the back view's, else the
    side view's (the side is drawn facing left and is mirrored so its columns run back -> front)."""
    import numpy as np

    import stack_views as SV

    from scipy.ndimage import distance_transform_edt

    prep = {}
    for pid, (rgba, mask) in views.items():
        # Every outline pixel, on the rim and inside (between an arm and the body, around the hair), takes the
        # colour of the nearest pixel that is not outline: drawn lines are for a flat picture, not for a volume.
        rgba = rgba.copy()
        line = mask & (SV._luma(rgba[..., :3]) < OUTLINE_LUMA)
        keep = mask & ~line
        if line.any() and keep.any():
            _, (iy, ix) = distance_transform_edt(~keep, return_indices=True)
            rgba[line] = rgba[iy[line], ix[line]]
        prep[pid] = (rgba, mask)
    front, fm = prep["front"]
    side, sm = prep["side"]
    back, bm = prep["back"]
    side, sm = side[:, ::-1], sm[:, ::-1]
    back, bm = back[:, ::-1], bm[:, ::-1]
    H, W = fm.shape
    L = sm.shape[1]
    rows = np.arange(H)[::-1]  # layer h is view row H-1-h
    front2d = fm[rows]  # [H, W], layer 0 at the bottom
    m = measure(front2d, sm[rows])
    t0, t1 = m["torso_cols"]
    solid = np.zeros((H, L, W), bool)
    trust = np.zeros((H, W), bool)  # columns whose sides the side view shows (not the body behind an arm)
    arm_mid = np.full((H, W), -1)  # an arm's cells: the arm's middle column in that layer
    for h in range(H):
        fr = _runs(front2d[h])
        sr = _runs(sm[rows[h]])
        if not fr or not sr:
            continue
        d0, d1 = sr[0][0], sr[-1][1]
        arms_here = m["hand"] <= h <= m["shoulder"]
        pieces = []
        for c0, c1 in fr:
            if arms_here and not c0 <= m["mid"] < c1:  # a run beside the body's middle: all arm
                pieces.append((c0, c1, True))
            elif arms_here:  # cut the run at the torso's columns: the outer pieces are arms
                for a_, b_, arm in ((c0, min(c1, t0), True), (max(c0, t0), min(c1, t1), False),
                                    (max(c0, t1), c1, True)):
                    if b_ > a_:
                        pieces.append((a_, b_, arm))
            else:
                pieces.append((c0, c1, False))
        for c0, c1, arm in pieces:
            w = c1 - c0
            a_, b_ = d0, d1
            if arm:
                dd = min(d1 - d0, max(w, m["arm_w"]) + 1)
                a_ = int(round((d0 + d1) / 2 - dd / 2))
                b_ = a_ + dd
                arm_mid[h, c0:c1] = (c0 + c1) // 2
            elif not arms_here:
                trust[h, c0:c1] = True
            solid[h, a_:b_, c0:c1] |= _round_fill(w, b_ - a_)
        solid[h] &= sm[rows[h]][:, None]
    # Colours, by which way each surface cell faces (from a smoothed copy of the layer): facing front, the front
    # view; facing back, the back view; facing sideways, the side view. On a rounded body nearly every cell is open
    # to the front somewhere, so "open to the front" smears the front view's edge column round the flank.
    # The side view is used only where it shows that part (an arm, a leg, the head): on the body's flank behind an
    # arm it paints the arm, so there the flank takes the front or back view from inside the body's edge.
    from scipy.ndimage import gaussian_filter

    color = np.zeros((H, L, W, 4), np.uint8)
    fr_c = np.broadcast_to(front[rows][:, None, :, :], color.shape)
    bk_c = np.broadcast_to(back[rows][:, None, :, :], color.shape)
    sd_c = np.broadcast_to(side[rows][:, :, None, :], color.shape)
    ok_f = np.broadcast_to(fm[rows][:, None, :], solid.shape)
    ok_b = np.broadcast_to(bm[rows][:, None, :], solid.shape)
    ok_s = np.broadcast_to(sm[rows][:, :, None], solid.shape)
    soft = gaussian_filter(solid.astype(np.float32), sigma=(0, 1.5, 1.5))
    gl = np.gradient(soft, axis=1)  # + towards the front
    gw = np.gradient(soft, axis=2)
    # the gradient points into the body; a cell facing front has the body behind it (gl < 0)
    faces_side = np.abs(gw) > np.abs(gl) * 1.2
    faces_front = ~faces_side & (gl <= 0)
    half = np.broadcast_to((np.arange(L) >= L / 2)[None, :, None], solid.shape)
    color[:] = np.where(half[..., None], fr_c, bk_c)
    miss = solid & (color[..., 3] < 128)
    color[miss] = np.where(half[miss][:, None], bk_c[miss], fr_c[miss])
    use_b = solid & ~faces_side & ~faces_front & ok_b
    color[use_b] = bk_c[use_b]
    use_f = solid & faces_front & ok_f
    color[use_f] = fr_c[use_f]
    trusted = np.broadcast_to(trust[:, None, :], solid.shape)
    use_s = solid & faces_side & trusted & ok_s
    color[use_s] = sd_c[use_s]
    # an arm's sides: the side view cannot say where in the body's depth the arm hangs, so it takes the arm's own
    # colour, the front (or back) view down the arm's middle
    on_arm = solid & faces_side & np.broadcast_to((arm_mid >= 0)[:, None, :], solid.shape)
    if on_arm.any():
        hh, ll, ww = np.nonzero(on_arm)
        mid = arm_mid[hh, ww]
        fr_r, bk_r = front[rows], back[rows]
        take_f = (ll >= L / 2) & fm[rows][hh, mid] | ~bm[rows][hh, mid]
        color[hh, ll, ww] = np.where(take_f[:, None], fr_r[hh, mid], bk_r[hh, mid])
    # the flank behind an arm: the front (or back) view a few cells inside the body's edge, where it is shirt
    flank = solid & faces_side & ~trusted & np.broadcast_to((arm_mid < 0)[:, None, :], solid.shape)
    if flank.any():
        hh, ll, ww = np.nonzero(flank)
        inset = max(2, m["arm_w"] // 2)
        wi = np.clip(ww, t0 + inset, t1 - 1 - inset)
        fr_r, bk_r = front[rows], back[rows]
        f_ok, b_ok = fm[rows], bm[rows]
        take_f = (ll >= L / 2) & f_ok[hh, wi] | ~b_ok[hh, wi]
        color[hh, ll, ww] = np.where(take_f[:, None], fr_r[hh, wi], bk_r[hh, wi])
    color[..., 3] = np.where(solid, 255, 0)
    color[~solid] = 0
    return solid, color, m


def body_stack(solid, rgba) -> dict:
    """A carved volume as a stack result for stack_views.write_stack (layers padded to a square)."""
    import numpy as np

    height, length, width = solid.shape
    res = max(length, width)
    oz, ox = (res - length) // 2, (res - width) // 2
    slices = []
    for h in range(height):
        sl = np.zeros((res, res, 4), np.uint8)
        sl[oz:oz + length, ox:ox + width] = rgba[h]
        slices.append(sl)
    pitch = 1.0 / res
    ext = [width * pitch, height * pitch, length * pitch]
    return {"slices": slices, "pitch": pitch, "origin": [-0.5, 0.0, -0.5], "resolution": int(res),
            "layers": height, "bounds": [[-ext[0] / 2, 0.0, -ext[2] / 2], [ext[0] / 2, ext[1], ext[2] / 2]],
            "extents": ext, "surface_voxels": int(solid.sum()), "filled": True,
            "size": {"width": int(width), "length": int(length), "height": int(height)},
            "colors": int(len(np.unique(rgba[solid][:, :3], axis=0)))}


def body_volume(asset: Path, source: str = "carved", layers: int | None = None):
    """(solid, rgba, measure) of the character, at the height and style it was drawn with (draw.json).

    source 'carved': carved from the drawn front, side and back views (carve_character). source 'mesh': sliced
    from the 3D model made from the drawing (character/mesh/model.glb, mesh_volume): every layer a true
    cross-section of the model, coloured from its surface. layers: shrink the body to this many layers (a flat
    sprite stack is 32 or 64), every part cut again at that size."""
    meta = asset / "character" / "draw.json"
    d = json.loads(meta.read_text()) if meta.is_file() else {"style": "pixel", "height": HEIGHT}
    height = int(d.get("height") or HEIGHT)
    m = None
    if source == "mesh":
        solid, rgba = mesh_volume(asset, height, colours=int(d.get("colours") or 24))
    elif source == "carved":
        solid, rgba, m = carve_character(clean_views(asset, height, grid=d.get("style") != "clean",
                                                     colours=int(d.get("colours") or 24),
                                                     despeckle=int(d.get("despeckle") or 0)))
    else:
        raise PartsError(f"unknown body source {source!r} (carved or mesh)")
    if layers and layers < solid.shape[0]:
        solid, rgba = shrink(solid, rgba, solid.shape[0] / layers)
        m = None
    return solid, rgba, m or measure(solid.any(axis=1), solid.any(axis=2))


# ---------------------------------------------------------------- a body sliced from a 3D model (pipeline#6)

MESH_DIR = "character/mesh"
MESH_GLB = "model.glb"


def _palette_index(rgb, colours: int):
    """(palette[k, 3], index per colour): at most `colours` colours, by flat2d's palette (median cut, then the
    closest merged, so a small distinct colour such as an eye survives)."""
    import numpy as np

    from flat2d import pixelart as PA

    side = int(np.ceil(np.sqrt(len(rgb))))
    img = np.zeros((side * side, 3), np.uint8)
    img[: len(rgb)] = rgb
    mask = np.zeros(side * side, bool)
    mask[: len(rgb)] = True
    pal = PA.build_palette({"v": (img.reshape(side, side, 3), mask.reshape(side, side))}, colours)
    return pal, PA.snap(rgb.reshape(-1, 1, 3), pal).ravel()


def _front_picture(solid, rgba):
    """The body seen from the front: for each layer and column the colour of the frontmost cell (row L-1 is the
    front), and where there is one. Layer 0 at the bottom."""
    import numpy as np

    H, L, W = solid.shape
    any_ = solid.any(axis=1)
    first = L - 1 - np.argmax(solid[:, ::-1, :], axis=1)
    hh, ww = np.nonzero(any_)
    pic = np.zeros((H, W, 3), np.uint8)
    pic[hh, ww] = rgba[hh, first[hh, ww], ww, :3]
    return pic, any_


def _face_front(solid, rgba, drawn) -> tuple:
    """Turn the volume about its upright axis (0, 90, 180 or 270 degrees) so the character faces the front
    (row L-1), the way the drawn front view (rgba[h, w, 4], top row first) shows it. A model made from a picture
    faces wherever its maker put it: each turn's front picture is compared with the drawing, colour for colour."""
    import numpy as np

    from flat2d import pixelart as PA

    want = drawn[::-1]  # layer 0 at the bottom
    wm = want[..., 3] >= 128
    best = None
    for k in range(4):
        s = np.rot90(solid, k, axes=(1, 2))  # (L, W) plane, seen from above
        c = np.rot90(rgba, k, axes=(1, 2))
        pic, on = _front_picture(s, c)
        cols = np.flatnonzero(on.any(axis=0))
        if not len(cols):
            continue
        pic, on = pic[:, cols[0]:cols[-1] + 1], on[:, cols[0]:cols[-1] + 1]
        Hd, Wd = want.shape[:2]
        rows = np.minimum((np.arange(Hd) + 0.5) * pic.shape[0] / Hd, pic.shape[0] - 1).astype(int)
        cc = np.minimum((np.arange(Wd) + 0.5) * pic.shape[1] / Wd, pic.shape[1] - 1).astype(int)
        pic, on = pic[rows][:, cc], on[rows][:, cc]
        both = on & wm
        if not both.any():
            continue
        diff = float(np.abs(pic[both].astype(int) - want[both][:, :3].astype(int)).sum(1).mean())
        print(f"mesh: turn {90 * k:3d} deg, front differs from the drawing by {diff:.1f}")
        if best is None or diff < best[0]:
            best = (diff, k)
    if best is None:
        raise PartsError("the model's front matches no side of the drawing")
    k = best[1]
    return np.ascontiguousarray(np.rot90(solid, k, axes=(1, 2))), np.ascontiguousarray(np.rot90(rgba, k, axes=(1, 2)))


TRELLIS_DIR = ROOT / "vendor" / "trellis-mac"


def front_cutout(asset: Path, size: int = 1024) -> Path:
    """The drawn front view (flat/views/front.png) cut from its key-coloured ground, square, with real
    transparency: character/mesh/cutout.png, what the 3D model is made from."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    from flat2d import pixelart as PA
    from flat2d.parts import ground_key

    im = Image.open(asset / "flat" / "views" / "front.png").convert("RGB")
    rgb = np.asarray(im)
    fg = PA.subject_mask(rgb, ground_key(im))
    labels, n = ndimage.label(fg)
    if n == 0:
        raise PartsError("the front view is empty")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    big = int(sizes.argmax())
    bx0, by0, bx1, by1 = PA.bbox(labels == big)
    keep = np.zeros(n + 1, bool)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None or sizes[k] < 0.002 * sizes[big]:
            continue
        y, x = sl  # the character and pieces of it inside its box; never the panel's divider line
        keep[k] = x.start >= bx0 - 4 and x.stop <= bx1 + 4 and y.start >= by0 - 4 and y.stop <= by1 + 4
    mask = keep[labels]
    x0, y0, x1, y1 = PA.bbox(mask)
    cut = np.dstack([rgb, (mask * 255).astype(np.uint8)])[y0:y1, x0:x1]
    h, w = cut.shape[:2]
    side = int(max(h, w) * 1.15)
    sq = np.zeros((side, side, 4), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = cut
    dest = asset / MESH_DIR
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "cutout.png"
    Image.fromarray(sq, "RGBA").resize((size, size), Image.LANCZOS).save(out)
    return out


def make_mesh(root: Path, stem: str, seed: int = 42) -> Path:
    """The character's 3D model from its drawn front view, by TRELLIS.2-4B (Microsoft, MIT; vendor/trellis-mac):
    character/mesh/model.glb. One picture in, so the back is the model's own guess. A big model (about 7 GB on
    the GPU, refuses to start under 60 GB free): one at a time on the Mac."""
    import subprocess

    asset = root / "out" / stem
    py = TRELLIS_DIR / ".venv" / "bin" / "python"
    runner = TRELLIS_DIR / "pipeline_runner.py"
    if not py.is_file() or not runner.is_file():
        raise PartsError(f"TRELLIS.2 is not set up at {TRELLIS_DIR} (its .venv and pipeline_runner.py)")
    cut = front_cutout(asset)
    out = asset / MESH_DIR / MESH_GLB
    subprocess.run([str(py), str(runner), str(cut), str(out), "--seed", str(seed), "--save-cond"],
                   cwd=TRELLIS_DIR, check=True)
    (asset / MESH_DIR / "mesh.json").write_text(json.dumps(
        {"maker": "TRELLIS.2-4B", "licence": "MIT", "from": "flat/views/front.png", "seed": seed,
         "made": int(time.time())}, indent=2) + "\n")
    return out


def _outside_colours(solid, rgba):
    """Every cell inside the body takes the colour of the nearest cell on its outside (one with open air beside
    it). A model made from a picture has surfaces inside it too (the head under the hair, the inside of a sleeve):
    cubes hide them, but flat slices show the cells just behind each layer's edge, so the hair would show skin."""
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt

    outside = solid & ~binary_erosion(solid, border_value=0)
    inner = solid & ~outside
    if inner.any() and outside.any():
        _, idx = distance_transform_edt(~outside, return_indices=True)
        rgba = rgba.copy()
        rgba[inner] = rgba[idx[0][inner], idx[1][inner], idx[2][inner]]
    return rgba


def mesh_volume(asset: Path, height: int, colours: int = 24, glb: Path | None = None):
    """(solid[H, L, W], rgba[H, L, W, 4]) sliced from the character's 3D model, laid out as carve_character's:
    layer 0 at the soles, row 0 the back, column 0 the character's right. height: cells from the soles to the top.

    Each layer is a true cross-section: the model's surface is sampled densely, every sample takes its texture
    colour (snapped to a palette of `colours`), each surface cell the colour most of its samples have, and then
    every layer is closed and filled, its inside taking the nearest surface colour (sprite_stack.fill_slices).
    A model made from a picture can come with holes (open soles) and faces wound both ways, so the inside is
    filled layer by layer, never by which side of the surface a cell is on."""
    import numpy as np
    import trimesh
    from PIL import Image

    import sprite_stack as SS

    path = glb or asset / MESH_DIR / MESH_GLB
    if not path.is_file():
        raise PartsError(f"no 3D model of the character at {path}: make it first (stack_character.py mesh)")
    mesh = SS._as_mesh(trimesh.load(path, force="scene"))
    tex, uv, factor = SS._albedo(mesh)
    lo, hi = mesh.bounds
    pitch = float(hi[1] - lo[1]) / height
    # two empty cells round the model: a layer's closing (fill_slices) erodes whatever touches the grid's edge
    nx = int(np.ceil((hi[0] - lo[0]) / pitch)) + 5
    nz = int(np.ceil((hi[2] - lo[2]) / pitch)) + 5
    origin = np.array([lo[0] - 2 * pitch, lo[1], lo[2] - 2 * pitch])
    # about 24 samples a cell of surface
    count = int(min(12_000_000, max(400_000, 24 * mesh.area / pitch ** 2)))
    pts, fi = trimesh.sample.sample_surface(mesh, count, seed=0)  # the same model slices the same every time
    if uv is not None:
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fi], pts)
        uvs = (uv[mesh.faces[fi]] * bary[:, :, None]).sum(1)
        cols = SS._sample_tex(tex, uvs, factor)[:, :3]
    else:
        cols = np.repeat((factor[None, :3] * 255).astype(np.uint8), len(pts), 0)
    pal, idx = _palette_index(cols, colours)
    ijk = np.floor((pts - origin) / pitch).astype(np.int64)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, height - 1)  # the soles and the crown sit on the first and last layer
    ok = (ijk[:, 0] >= 0) & (ijk[:, 0] < nx) & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
    ijk, idx = ijk[ok], idx[ok]
    k = len(pal)
    cell = (ijk[:, 1] * nz + ijk[:, 2]) * nx + ijk[:, 0]
    votes = np.bincount(cell * k + idx, minlength=height * nz * nx * k).reshape(-1, k)
    hit = votes.sum(1) > 0
    vol = np.zeros((height * nz * nx, 4), np.uint8)
    vol[hit, :3] = pal[_pick_colour(votes[hit], pal)]
    vol[hit, 3] = 255
    volume = vol.reshape(height, nz, nx, 4)
    occ = hit.reshape(height, nz, nx)
    volume = SS.fill_slices(volume, occ)
    solid = volume[..., 3] > 0
    volume[~solid] = 0
    volume = _outside_colours(solid, volume)
    drawn = np.asarray(Image.open(asset / "character" / "views" / "front.png").convert("RGBA"))
    solid, volume = _face_front(solid, volume, drawn)
    # trim to the body, one empty cell round it
    rr = np.flatnonzero(solid.any(axis=(0, 2)))
    cc = np.flatnonzero(solid.any(axis=(0, 1)))
    solid = solid[:, rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1]
    volume = volume[:, rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1]
    print(f"mesh: {path.name} sliced to {solid.shape[0]} layers of {solid.shape[2]} x {solid.shape[1]} cells, "
          f"{len(pal)} colours, {count} surface samples")
    return solid, volume


NEAR_RGB = 70  # colours closer than this vote together (shades of one shirt), so a thin dark line never outvotes them


def _pick_colour(votes, cols):
    """Each row's colour from its votes (rows x colours): the colour whose near colours (NEAR_RGB) have the most
    votes between them, so three shades of red beat one black line; ties go to the colour voted most itself."""
    import numpy as np

    c = cols.astype(np.int32)
    near = (np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)) < NEAR_RGB).astype(np.float64)
    score = votes @ near + votes * 1e-3
    return score.argmax(1)


def shrink(solid, rgba, f: float):
    """The body at 1/f of its size in every direction (a flat sprite stack is 32 to 64 layers, not 192): a small
    cell is filled when about two fifths of the big cells in it are, and takes the colour most of its big
    SURFACE cells have (near colours voting together, _pick_colour), so the inside's fill never shows through on a
    small cell's face. Colours are only picked, never blended: the palette stays the body's."""
    import numpy as np
    from scipy.ndimage import binary_erosion

    H, L, W = solid.shape
    h, l, w = (max(1, int(round(n / f))) for n in (H, L, W))
    zi = np.minimum((np.arange(H) * h) // H, h - 1)
    ri = np.minimum((np.arange(L) * l) // L, l - 1)
    ci = np.minimum((np.arange(W) * w) // W, w - 1)
    block = ((zi[:, None, None] * l + ri[None, :, None]) * w + ci[None, None, :])
    n_small = h * l * w
    full = np.bincount(block.ravel(), minlength=n_small)
    have = np.bincount(block[solid], minlength=n_small)
    keep = have >= 0.4 * np.maximum(full, 1)
    surface = solid & ~binary_erosion(solid, border_value=0)
    cols, inv = np.unique(rgba[solid][:, :3], axis=0, return_inverse=True)
    k = len(cols)
    cidx = np.full(solid.shape, -1, np.int64)
    cidx[solid] = inv.ravel()
    out = np.zeros((n_small, 4), np.uint8)
    for which in (surface, solid):  # the surface's colour, else any (a small cell all inside)
        b = block[which]
        votes = np.bincount(b * k + cidx[which], minlength=n_small * k).reshape(n_small, k)
        got = keep & (votes.sum(1) > 0) & (out[:, 3] == 0)
        out[got, :3] = cols[_pick_colour(votes[got], cols)]
        out[got, 3] = 255
    small = out.reshape(h, l, w, 4)
    return small[..., 3] > 0, small


def _runs(row) -> list[tuple[int, int]]:
    """The solid runs of a 1D mask, as [start, end) pairs."""
    import numpy as np

    r = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(r[::2], r[1::2])]


def measure(front, side=None) -> dict:
    """Joint heights and columns from the front silhouette (front[H, W], layer 0 at the bottom; side[H, L] the
    side silhouette, only for the depth L).

    Legs: the layers from the bottom up where the silhouette is two runs side by side; the crotch is the first
    layer where it is one, and its run is the hips, whose columns are the torso's all the way up. Arms: everything
    outside the torso's columns, from the layer with the most of it, down to the hands and up to the shoulders
    (the drawn arms may touch the body: they are told apart by column, not by a gap). The neck is the narrowest
    layer above the shoulders. Knees and elbows sit halfway along their limbs; the ankle is the top of the foot."""
    import numpy as np

    H, W = front.shape
    L = side.shape[1] if side is not None else W
    runs = [_runs(front[h]) for h in range(H)]
    filled = [h for h in range(H) if runs[h]]
    if not filled:
        raise PartsError("the drawing is empty")
    top = max(filled) + 1
    sole = min(filled)
    # The crotch: the first layer from the bottom where the middle of the body (between the feet) is solid.
    # Below it the gap between the legs is there; hands hanging beside the hips do not count (they are outside).
    cols = np.flatnonzero(front[sole:sole + 3].any(axis=0))
    c_mid = (cols.min() + cols.max()) / 2
    win = max(3, int(round((cols.max() - cols.min()) / 6)))
    lo, hi = int(c_mid - win), int(c_mid + win) + 1

    m0, m1 = int(round(c_mid)) - 2, int(round(c_mid)) + 3

    h = sole
    while h < top and not front[h, m0:m1].all():  # the gap between the legs (or the ground between the boots)
        h += 1
    if h == sole or h >= top:
        raise PartsError("the legs are not apart at the bottom of the front view: draw them a little apart")
    crotch = h
    # the hips: from the outer edge of one leg to the other's, just under the crotch
    legs = [r for r in runs[crotch - 1] if r[1] > lo and r[0] < hi]
    t0, t1 = legs[0][0], legs[-1][1]

    def outside(h):
        n = int(front[h, :t0].sum() + front[h, t1:].sum())
        # under the crotch, cells outside the hips are an arm only beside the legs (a boot may be wider)
        return n if h >= crotch or len(runs[h]) >= 3 else 0

    out = np.array([outside(h) for h in range(H)])
    out[:sole] = 0
    if out.max() == 0:
        raise PartsError("no arms outside the body's width in the front view")
    start = int(out[crotch:].argmax()) + crotch
    hand = start
    while hand - 1 >= sole and out[hand - 1] > 0:
        hand -= 1
    shoulder = start
    while shoulder + 1 < top and out[shoulder + 1] > 0:
        shoulder += 1
    widths = front.sum(axis=1)
    above = range(shoulder + 1, shoulder + 1 + max(1, (top - shoulder) // 2))  # the neck is low between shoulders and crown
    neck = min(above, key=lambda h: (widths[h], h)) if len(above) else shoulder + 1
    arm_w = int(np.median([max(front[h, :t0].sum(), front[h, t1:].sum()) for h in range(hand, shoulder + 1)]))
    arm_w = max(arm_w, 2)
    leg_len = crotch - sole
    ankle = sole + max(2, round(leg_len / 6))
    knee = ankle + round((crotch - ankle) / 2)
    elbow = hand + round((shoulder + 1 - hand) / 2)
    mid = (t0 + t1 - 1) / 2
    return {"H": H, "L": L, "W": W, "sole": sole, "ankle": ankle, "knee": knee, "crotch": crotch, "hand": hand,
            "elbow": elbow, "shoulder": shoulder, "neck": neck, "top": top, "torso_cols": [int(t0), int(t1)],
            "mid": mid, "arm_w": arm_w}


def assign(solid, m: dict):
    """Each solid cell's part, as a label volume: 0 empty, then PART_NAMES order (1-based). By layer and column:
    above the neck the head; outside the torso's columns between the hands and the shoulders an arm; below the
    crotch a leg; the rest the torso. Right is the character's right (low columns)."""
    import numpy as np

    H, L, W = solid.shape
    idx = {n: i + 1 for i, n in enumerate(PART_NAMES)}
    t0, t1 = m["torso_cols"]
    mid = m["mid"]
    lab = np.zeros(solid.shape, np.int8)
    for h in range(H):
        for c in range(W):
            col = solid[h, :, c]
            if not col.any():
                continue
            side = "R" if c < mid else "L"
            arm = m["hand"] <= h <= m["shoulder"] and (c < t0 or c >= t1)
            if h >= m["neck"]:
                name = "head"
            elif arm:
                name = ("forearm_" if h < m["elbow"] else "upperarm_") + side
            elif h < m["crotch"]:
                name = ("foot_" if h < m["ankle"] else "shin_" if h < m["knee"] else "thigh_") + side
            else:
                name = "torso"
            lab[h, col, c] = idx[name]
    return lab


PART_NAMES = ("head", "torso", "upperarm_R", "upperarm_L", "forearm_R", "forearm_L", "thigh_R", "thigh_L",
              "shin_R", "shin_L", "foot_R", "foot_L")
LIMBS = ("upperarm", "forearm", "thigh", "shin")


def cut_parts(solid, rgba, m: dict) -> dict:
    """Each part as its own small stack: {name: {"cells": rgba[n, cw, ch, 4] bottom slice first, "z0": the
    bottom slice's layer in the body, "axis": (x, y) of the part's turning axis in body cells}}.

    Every slice is cut to the same cell, centred on the part's axis, so a part turns about its own joint: the
    torso and head about the body's middle, a foot about its ankle, a limb about its bone. A limb's slices are
    centred on the bone's line through its two joints, so the bone runs through the middle of every slice."""
    import numpy as np

    lab = assign(solid, m)
    H, L, W = solid.shape
    out = {}
    for i, name in enumerate(PART_NAMES):
        mask = lab == i + 1
        if not mask.any():
            raise PartsError(f"no cells for {name}")
        hs = np.flatnonzero(mask.any(axis=(1, 2)))
        z0, z1 = int(hs[0]), int(hs[-1]) + 1
        # the axis: the mean of the part's cells (x along the length, back -> front; y across)
        rr, cc = np.nonzero(mask.any(axis=0))
        ax = float(rr.mean()), float(cc.mean())
        kind = name.split("_")[0]
        if kind in ("shin", "foot", "forearm"):
            # one line down a whole limb: a shin, foot or forearm hangs on the axis of the part above it (thigh,
            # upper arm), so in the standing pose, with the joints straight below each other, the parts meet as
            # drawn instead of each sitting on its own middle
            top = {"shin": "thigh_", "foot": "thigh_", "forearm": "upperarm_"}[kind] + name[-1]
            ax = tuple(float(v) for v in out[top]["axis"])
        if name in ("torso", "head"):
            ax = (L - 1) / 2, m["mid"]
        # cell: big enough that the axis is the centre and every cell fits
        span_r = max(abs(rr.min() - ax[0]), abs(rr.max() - ax[0]))
        span_c = max(abs(cc.min() - ax[1]), abs(cc.max() - ax[1]))
        cw = int(2 * np.ceil(span_r + 0.5) + 1)
        ch = int(2 * np.ceil(span_c + 0.5) + 1)
        cells = np.zeros((z1 - z0, cw, ch, 4), np.uint8)
        r_off = int(round(ax[0])) - cw // 2
        c_off = int(round(ax[1])) - ch // 2
        for h in range(z0, z1):
            r, c = np.nonzero(mask[h])
            cells[h - z0, r - r_off, c - c_off] = rgba[h, r, c]
        out[name] = {"cells": cells, "z0": z0, "axis": (r_off + cw // 2, c_off + ch // 2)}
    return out


def write_strip(cells, path: Path) -> dict:
    """One part as a strip PNG: its slices side by side, bottom slice first. In a cell, image x is the
    character's forward (+x) and image y its left (+y), so the cell is cw wide and ch tall, as in
    sprite-stack-animation's strips."""
    import numpy as np
    from PIL import Image

    n, cw, ch, _ = cells.shape
    strip = np.zeros((ch, cw * n, 4), np.uint8)
    for i in range(n):
        strip[:, i * cw:(i + 1) * cw] = cells[i].transpose(1, 0, 2)
    Image.fromarray(strip, "RGBA").save(path, optimize=True)
    return {"cw": int(cw), "ch": int(ch), "n": int(n)}


def parts_sheet(parts: dict, path: Path, scale: int = 8) -> None:
    """Every part's strip enlarged, one row each, on a dark ground: a picture for people."""
    import numpy as np
    from PIL import Image, ImageDraw

    rows = []
    for name in PART_NAMES:
        cells = parts[name]["cells"]
        n, cw, ch, _ = cells.shape
        rows.append((name, cells, cw, ch, n))
    pad, label_w = 6, 110
    width = label_w + max(n * (cw + 1) for _, _, cw, _, n in rows) * scale + pad * 2
    height = sum((ch + 1) * scale + pad for *_, ch, _ in [(r[0], r[1], r[2], r[3], r[4]) for r in rows]) + pad
    im = Image.new("RGB", (width, height), (24, 22, 30))
    d = ImageDraw.Draw(im)
    y = pad
    for name, cells, cw, ch, n in rows:
        d.text((pad, y + 2), name.replace("_R", " right").replace("_L", " left"), fill=(220, 220, 230))
        for i in range(n):
            x = label_w + i * (cw + 1) * scale
            d.rectangle([x, y, x + cw * scale - 1, y + ch * scale - 1], fill=(36, 34, 44))
            sl = Image.fromarray(cells[i].transpose(1, 0, 2), "RGBA").resize((cw * scale, ch * scale), Image.NEAREST)
            im.paste(sl, (x, y), sl)
        y += (ch + 1) * scale + pad
    im.save(path, optimize=True)


# How far apart the exploded view pulls the parts, in cells: (across, up) for each part.
SPREAD = {"head": (0, 4), "torso": (0, 0), "upperarm": (4, 0), "forearm": (6, -3), "thigh": (2, -3),
          "shin": (3, -6), "foot": (4, -9)}


def exploded(parts: dict, m: dict) -> dict:
    """The parts back where they were cut from, pulled apart (SPREAD), as one stack result for write_stack:
    a picture of the parts that turns in the Stack stage like any stack."""
    import numpy as np

    k = max(1.0, m["top"] / 40)  # SPREAD is for a body 40 cells tall
    spread = {n: (int(round(dx * k)), int(round(dz * k))) for n, (dx, dz) in SPREAD.items()}
    lift = -min(dz for _, dz in spread.values())
    reach = max(dx for dx, _ in spread.values())
    H = m["top"] + lift + spread["head"][1] + 1
    L, W = m["L"] + 2, m["W"] + 2 * reach + 2
    vol = np.zeros((H, L, W, 4), np.uint8)
    for name, p in parts.items():
        kind = name.split("_")[0]
        dx, dz = spread[kind]
        side = name[-1] if "_" in name else ""
        shift = -dx if side == "R" else dx
        cells = p["cells"]
        n, cw, ch, _ = cells.shape
        r0 = p["axis"][0] - cw // 2 + 1
        c0 = p["axis"][1] - ch // 2 + reach + 1 + shift
        for i in range(n):
            z = p["z0"] + i + lift + dz
            sl = cells[i]
            on = sl[..., 3] > 0
            rr, cc = np.nonzero(on)
            vol[z, rr + r0, cc + c0] = sl[rr, cc]
    return body_stack(vol[..., 3] > 0, vol)


def make_parts(root: Path, stem: str, show: bool = True) -> dict:
    """Cut the drawn body into parts: out/<stem>/character/parts/<name>.png and parts.json, parts.png (the strips
    enlarged), and, with show, the exploded parts as a new Stack Version."""
    import stack_views as SV

    asset = root / "out" / stem
    solid, rgba, m = body_volume(asset)
    parts = cut_parts(solid, rgba, m)
    dest = asset / "character"
    (dest / "parts").mkdir(parents=True, exist_ok=True)
    listing = []
    for name in PART_NAMES:
        info = write_strip(parts[name]["cells"], dest / "parts" / f"{name}.png")
        listing.append({"name": name, "strip": f"parts/{name}.png", **info, "z0": parts[name]["z0"],
                        "axis": [round(v, 2) for v in parts[name]["axis"]]})
    (dest / "parts.json").write_text(json.dumps({"measure": {k: v for k, v in m.items()}, "parts": listing},
                                                indent=2) + "\n")
    parts_sheet(parts, dest / "parts.png")
    if show:
        from flow import autosave

        SV.write_stack(body_stack(solid, rgba), asset / "stack", "character/views", stem)
        autosave(root, stem, ("stack",), made_from=("concept",))
        result = exploded(parts, m)
        SV.write_stack(result, asset / "stack", "character/parts", stem)
        autosave(root, stem, ("stack",), made_from=("concept",))
    print("measured:", {k: m[k] for k in ("sole", "ankle", "knee", "crotch", "hand", "elbow", "shoulder", "neck", "top")})
    return {"measure": m, "parts": listing}

CHARACTER_FORMAT = "pipeline-sprite-character"
# How each part hangs on the rig: the joint at its top (where z_local = 0) and, for a limb, the joint it reaches.
PART_JOINTS = {
    "torso": ("pelvis", None), "head": ("neck", None),
    "thigh": ("hip", "knee"), "shin": ("knee", "ankle"), "foot": ("ankle", None),
    "upperarm": ("shoulder", "elbow"), "forearm": ("elbow", "hand"),
}


def rig_numbers(m: dict, parts: dict) -> dict:
    """The rig in cells, from the measured joints and the parts' axes (see rig.ts RigConfig)."""
    mid_r = (m["L"] - 1) / 2
    thigh = m["crotch"] - m["knee"]
    shin = m["knee"] - m["ankle"]
    leg = thigh + shin + m["ankle"]
    ax = {n: parts[n]["axis"] for n in PART_NAMES}
    return {
        "thigh": thigh, "shin": shin, "ankle": m["ankle"],
        "hipHalf": round((ax["thigh_L"][1] - ax["thigh_R"][1]) / 2, 2),
        "hipFwd": round((ax["thigh_L"][0] + ax["thigh_R"][0]) / 2 - mid_r, 2),
        "kneeHint": 1,
        "standHip": round(m["ankle"] + 0.96 * (thigh + shin), 2),
        "stride": round(1.3 * leg, 2), "duty": 0.6, "lift": round(0.17 * (thigh + shin), 2),
        "bob": round(0.025 * leg, 2), "twist": 0.14,
        "spine": m["shoulder"] - m["crotch"], "neck": m["neck"] - m["crotch"],
        "arms": True,
        "shoulderHalf": round((ax["upperarm_L"][1] - ax["upperarm_R"][1]) / 2, 2),
        "shoulderFwd": round((ax["upperarm_L"][0] + ax["upperarm_R"][0]) / 2 - mid_r, 2),
        "upperArm": m["shoulder"] - m["elbow"], "foreArm": m["elbow"] - m["hand"],
        "armSwing": 0.45,
    }


def check_cut(solid, rgba, parts: dict) -> None:
    """The parts put back at their z0 and axis are the carved body, cell for cell (nothing lost, nothing doubled)."""
    import numpy as np

    back = np.zeros_like(rgba)
    hits = np.zeros(solid.shape, np.int16)
    for name in PART_NAMES:
        p = parts[name]
        n, cw, ch, _ = p["cells"].shape
        r_off, c_off = p["axis"][0] - cw // 2, p["axis"][1] - ch // 2
        on = p["cells"][..., 3] > 0
        z, r, c = np.nonzero(on)
        back[z + p["z0"], r + r_off, c + c_off] = p["cells"][z, r, c]
        hits[z + p["z0"], r + r_off, c + c_off] += 1
    if hits.max() > 1 or not np.array_equal(hits > 0, solid) or not np.array_equal(back[solid], rgba[solid]):
        raise PartsError("the parts do not put back together into the body")


LOOKS = ("cubes", "slices")
SOURCE_NOTE = {"carved": "Carved body (from drawn views)", "mesh": "Sliced body (from a 3D model)"}
LOOK_NOTE = {"cubes": "3D cubes", "slices": "flat sprite-stack slices"}


def make_character(root: Path, stem: str, source: str = "carved", layers: int | None = None,
                   look: str | None = None, note: str | None = None) -> dict:
    """The walking character as one Stack Version: the whole body (sheet.png, stack.json, as any stack) plus
    stack/character.json (rig, walk, how each part hangs on it) and one strip per part, stack/part_<name>.png.
    The Stack stage plays the walk when stack.json names a character; Export copies it all.

    source: 'carved' (from the drawn views) or 'mesh' (sliced from the 3D model). layers: the body's height in
    layers (default: as drawn). look: how the Stack stage first shows this Version, 'cubes' (lit 3D cubes) or
    'slices' (flat pixel slices, the sprite-stack look); stack.json "look". note: the Version's note."""
    import stack_views as SV
    from flow import snapshot

    if look is not None and look not in LOOKS:
        raise PartsError(f"unknown look {look!r} ({' or '.join(LOOKS)})")
    asset = root / "out" / stem
    solid, rgba, m = body_volume(asset, source, layers)
    parts = cut_parts(solid, rgba, m)
    check_cut(solid, rgba, parts)
    dest = asset / "stack"
    SV.write_stack(body_stack(solid, rgba), dest, "character/mesh" if source == "mesh" else "character/views", stem)
    for old in dest.glob("part_*.png"):
        old.unlink()
    listing = []
    for name in PART_NAMES:
        kind = name.split("_")[0]
        top, reach = PART_JOINTS[kind]
        info = write_strip(parts[name]["cells"], dest / f"part_{name}.png")
        joint_layer = {"torso": m["crotch"], "head": m["neck"], "thigh": m["crotch"], "shin": m["knee"],
                       "foot": m["ankle"], "upperarm": m["shoulder"], "forearm": m["elbow"]}[kind]
        side = name[-1] if "_" in name else None
        listing.append({
            "name": name, "file": f"part_{name}.png", **info, "z0": parts[name]["z0"],
            "joint": top, "reaches": reach, "side": side, "joint_layer": joint_layer,
            "axis_in_body": [round(float(v), 2) for v in parts[name]["axis"]],
        })
    rig = rig_numbers(m, parts)
    doc = {
        "format": CHARACTER_FORMAT, "version": 1, "stem": stem,
        "units": "cells: one cell is one cube of the part stacks",
        "frame": {"forward": "+x (image x in a strip cell)", "left": "+y (image y in a strip cell)", "up": "+z (slice)"},
        "height": int(m["top"]),
        "measure": {k: m[k] for k in ("sole", "ankle", "knee", "crotch", "hand", "elbow", "shoulder", "neck", "top")},
        "rig": rig,
        "walk": {"stride": rig["stride"], "duty": rig["duty"], "speed": rig["stride"], "fps": 60,
                 "lap": {"strides": 6, "frames": 360}},
        "parts": listing,
    }
    (dest / "character.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    meta_path = dest / "stack.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["character"] = "character.json"
    meta["body"] = source
    if look:
        meta["look"] = look
    else:
        meta.pop("look", None)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if note is None:
        note = f"{SOURCE_NOTE[source]}, {m['top']} layers" + (f", {LOOK_NOTE[look]}" if look else "")
    snapshot(root, stem, "stack", note=note, made_from=("concept",))
    print("character:", {k: rig[k] for k in ("thigh", "shin", "hipHalf", "standHip", "stride", "spine")})
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--root", default=str(ROOT))
    d.add_argument("--stem", required=True)
    d.add_argument("--prompt", required=True)
    d.add_argument("--size", type=int, default=96, help="height in cells (default 96)")
    d.add_argument("--style", choices=("clean", "pixel"), default="clean")
    p = sub.add_parser("parts")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--stem", required=True)
    p.add_argument("--no-show", action="store_true", help="do not save the exploded parts as a Stack Version")
    w = sub.add_parser("walk", help="save the walking character (body, part strips, rig, walk) as a Stack Version")
    w.add_argument("--root", default=str(ROOT))
    w.add_argument("--stem", required=True)
    w.add_argument("--body", choices=("carved", "mesh"), default="carved",
                   help="carved from the drawn views, or sliced from the 3D model (make it with: mesh)")
    w.add_argument("--layers", type=int, default=None, help="shrink the body to this many layers (32, 64)")
    w.add_argument("--look", choices=LOOKS, default=None, help="how the Stack stage first shows this Version")
    w.add_argument("--note", default=None, help="the Version's note")
    g = sub.add_parser("mesh", help="make the character's 3D model from its drawn front view (TRELLIS.2, MIT)")
    g.add_argument("--root", default=str(ROOT))
    g.add_argument("--stem", required=True)
    g.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.cmd == "draw":
        draw(Path(a.root), a.stem, a.prompt, a.size, a.style)
    elif a.cmd == "parts":
        make_parts(Path(a.root), a.stem, show=not a.no_show)
    elif a.cmd == "walk":
        make_character(Path(a.root), a.stem, a.body, a.layers, a.look, a.note)
    elif a.cmd == "mesh":
        make_mesh(Path(a.root), a.stem, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
