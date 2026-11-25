#!/usr/bin/env python3
"""Join buildings already made (their current stacks) into one stack: a row of buildings, or a corner building
with a wing. No image model; nothing is redrawn. Each part is read from out/<part>/stack/ as its slices, cut to
what it holds, turned if asked, and placed on one ground grid; the joined stack is written to out/<stem>/stack/
(stack.json "maker": "walls", "joined": the parts and where they went) and saved as a new Stack version.

Layouts:
  row     the parts left to right, walls touching, fronts in one line (a terrace; a corner shop beside a taller
          neighbour). A part may be turned (part:turn) first; turns are quarter turns, 1 faces the front right.
  corner  two parts: the main building, and a wing turned to face right, set behind the main one with its front
          in line with the main building's right wall: an L-shaped corner building with two street fronts.

The joined stack keeps the first part's cell size (pitch), so it shows as big as its parts together.

Where parts meet, the one placed first keeps its cells (a roof's eave never cuts into a neighbour's wall face:
the faces that meet are inside the building).

A corner is one building, so it gets one roof (one_roof): where every part has a flat roof with its deck at the same
height (within ROOF_STEP layers), the parapets where the parts meet are taken off and one deck runs over the whole
L, with one parapet round its outside edge. Rooftop pieces (a chimney, a vent) stay where they stood. A row keeps
each part's own roof: a terrace is buildings side by side, each behind its own parapet.

Orientation, as in stack_building: a layer is [y, x], y from the back (0) to the front, x across the front.

CLI:
  python scripts/stack_join.py --root R --stem S --layout row|corner PART[:TURN] PART[:TURN] ... [--render PNG]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import stack_building as SB  # noqa: E402

LAYOUTS = ("row", "corner")
WALL_LAYER = 0.5  # a part's walls are measured in this share of its height (over an awning, under a roof's eave)
MAX_PARTS = 8
ROOF_STEP = 2  # decks this many layers apart or fewer are one roof (the lower part's walls are raised to meet)
RING_TOL = 60  # summed-RGB distance from the parapet's outside colours that is still parapet (not a chimney)
DECK_SHARE = 0.8  # a part's deck is its highest layer covering this share of its walls' layer


class JoinError(RuntimeError):
    """The parts cannot be joined (missing, not a stack, a bad layout)."""


def read_part(root: Path, stem: str, turn: int = 0) -> dict[str, Any]:
    """A part's solid and colours [H, L, W], cut to what it holds and turned `turn` quarter turns (1: the front
    faces right, +x), with its wall box (y0, y1, x0, x1 inclusive) measured at WALL_LAYER of its height."""
    folder = Path(root) / "out" / stem / "stack"
    if not (folder / "stack.json").is_file():
        raise JoinError(f"{stem} has no stack to join")
    color = np.stack(SB.read_sheet_slices(folder))  # [H, res, res, 4]
    solid = color[..., 3] > 127
    if not solid.any():
        raise JoinError(f"{stem}'s stack is empty")
    ys, xs = np.nonzero(solid.any(axis=0))
    color = color[:, ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    solid = solid[:, ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    k = int(turn) % 4
    if k:
        color, solid = np.rot90(color, k, axes=(1, 2)), np.rot90(solid, k, axes=(1, 2))
    h = int(WALL_LAYER * solid.shape[0])
    wy, wx = np.nonzero(solid[h])
    if not len(wy):
        raise JoinError(f"{stem} has no walls at half its height")
    pitch = float(json.loads((folder / "stack.json").read_text(encoding="utf-8")).get("pitch") or 0) or None
    return {"stem": stem, "turn": k, "pitch": pitch, "solid": np.ascontiguousarray(solid), "color": np.ascontiguousarray(color),
            "wall": (int(wy.min()), int(wy.max()), int(wx.min()), int(wx.max()))}


def place(parts: list[dict[str, Any]], layout: str) -> list[tuple[int, int]]:
    """Where each part's array goes (y, x offsets; may be negative, join() shifts them)."""
    if layout == "row":
        at, x = [], 0
        front = max(p["wall"][1] for p in parts)  # every front on one line
        for p in parts:
            y0, y1, x0, x1 = p["wall"]
            at.append((front - y1, x - x0))
            x += x1 - x0 + 1
        return at
    if layout == "corner":
        if len(parts) != 2:
            raise JoinError("a corner is two parts: the main building and its wing")
        main, wing = parts
        my0, my1, mx0, mx1 = main["wall"]
        wy0, wy1, wx0, wx1 = wing["wall"]
        # The wing's front (its right wall once turned) in line with the main building's right wall, and the wing's
        # front edge against the main building's back wall.
        return [(0, 0), (my0 - 1 - wy1, mx1 - wx1)]
    raise JoinError(f"layout must be {' or '.join(LAYOUTS)}")


def join(parts: list[dict[str, Any]], layout: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """The joined solid and colours [H, L, W] and where each part went (its offset in the joined grid). A corner
    gets one roof (one_roof); where it cannot, the first entry of where's roof note says why."""
    at = place(parts, layout)
    oy = -min(y for y, _ in at)
    ox = -min(x for _, x in at)
    H = max(p["solid"].shape[0] for p in parts)
    L = max(y + oy + p["solid"].shape[1] for (y, _), p in zip(at, parts))
    W = max(x + ox + p["solid"].shape[2] for (_, x), p in zip(at, parts))
    solid = np.zeros((H, L, W), bool)
    color = np.zeros((H, L, W, 4), np.uint8)
    where = []
    for (y, x), p in zip(at, parts):
        h, l, w = p["solid"].shape
        y, x = y + oy, x + ox
        free = p["solid"] & ~solid[:h, y : y + l, x : x + w]  # the part placed first keeps its cells
        solid[:h, y : y + l, x : x + w] |= free
        color[:h, y : y + l, x : x + w][free] = p["color"][free]
        where.append({"stem": p["stem"], "turn": p["turn"], "at": [int(y), int(x)], "size": [h, l, w]})
    color[~solid] = 0
    if layout == "corner":
        solid, color, roof = one_roof(solid, color, parts, [(y + oy, x + ox) for y, x in at])
        where[0]["roof"] = roof
    return solid, color, where


def roof_of(p: dict[str, Any]) -> dict[str, Any] | None:
    """A part's flat roof: {deck (layer), parapet (layers over the deck), rim (cells from the edge to the deck),
    ring (bools [L, W]: the parapet's cells), deck_rgba, parapet_rgba}; None when its roof is not flat (a gable)."""
    from scipy import ndimage

    S, C = p["solid"], p["color"]
    H = S.shape[0]
    counts = S.sum(axis=(1, 2))
    walls = counts[int(WALL_LAYER * H)]
    over = np.flatnonzero(counts >= DECK_SHARE * walls)
    d = int(over.max())
    if d + 1 >= H:
        return None
    foot = S[d]
    ys, xs = np.nonzero(foot)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    if S[d + 1, cy, cx] or counts[d + 1] > 0.5 * counts[d]:
        return None  # the layer over the top of the walls is not a hollow parapet: a pitched roof
    P = 0
    while d + 1 + P < H and counts[d + 1 + P] >= 0.5 * counts[d + 1]:
        P += 1
    row = S[d + 1, cy, xs.min() : cx + 1]
    rim = int(np.argmin(row)) if not row.all() else 1
    # The parapet: what stands round the deck's open middle (its sides may differ in thickness). A piece standing
    # inside the open middle (a vent) is a hole in it and stays a piece; one against the parapet (a chimney) is told
    # from it by its colour: not one of the colours the parapet shows on its outside edge.
    lab, _ = ndimage.label(foot & ~S[d + 1])
    hollow = ndimage.binary_fill_holes(lab == lab[cy, cx]) if lab[cy, cx] else np.zeros_like(foot)
    ring = foot & ~hollow
    dist = ndimage.distance_transform_cdt(np.pad(foot, 1), metric="chessboard")[1:-1, 1:-1]
    top = C[d + P][S[d + P] & ring]
    cols, n = np.unique(top.reshape(-1, 4), axis=0, return_counts=True)
    edge = np.unique(C[d + 1 : d + 1 + P][S[d + 1 : d + 1 + P] & (dist == 1)[None]].reshape(-1, 4), axis=0)
    ref = edge[:, :3].astype(int)
    for k in range(P):
        ys, xs = np.nonzero(ring & S[d + 1 + k] & (dist > 1))
        px = C[d + 1 + k][ys, xs, :3].astype(int)
        far = np.array([(np.abs(ref - c).sum(-1) > RING_TOL).all() for c in px], bool)
        ring[ys[far], xs[far]] = False
    return {"deck": d, "parapet": P, "rim": max(1, rim), "ring": ring, "deck_rgba": C[d, cy, cx].copy(),
            "parapet_rgba": cols[n.argmax()].copy()}


def one_roof(solid: np.ndarray, color: np.ndarray, parts: list[dict[str, Any]],
             at: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One roof over joined parts (in place on solid and color; at: each part's offset in the joined grid): the
    parapets where parts meet go, one deck covers the joined footprint at the highest part's deck (a part up to
    ROOF_STEP layers lower has its walls raised to it), and one parapet, as high and as wide as the first part's,
    runs round the outside edge. What stood on a part's deck (a chimney, a vent) is kept, lifted with the deck and
    stood on it. Returns (solid, color, {one: True, deck, parapet, rim}), or the arrays unchanged and {one: False,
    why}."""
    from scipy import ndimage

    roofs = [roof_of(p) for p in parts]
    if any(r is None for r in roofs):
        return solid, color, {"one": False, "why": "a part's roof is not flat"}
    ds = [r["deck"] for r in roofs]
    if max(ds) - min(ds) > ROOF_STEP:
        return solid, color, {"one": False, "why": "the parts' roofs are at different heights"}
    D, P, rim = max(ds), roofs[0]["parapet"], roofs[0]["rim"]
    H, L, W = solid.shape
    if D + P >= H:
        grow = D + P + 1 - H
        solid = np.concatenate([solid, np.zeros((grow, L, W), bool)])
        color = np.concatenate([color, np.zeros((grow, L, W, 4), np.uint8)])
        H += grow
    else:
        solid, color = solid.copy(), color.copy()
    S0, C0 = solid.copy(), color.copy()
    foot = np.zeros((L, W), bool)
    dmap = np.full((L, W), -1, int)
    pieces = []  # (layer, y, x, rgba) of what stood on a deck, lifted to the one deck
    was_ring = np.zeros((L, W), bool)
    for (y, x), p, r in zip(at, parts, roofs):
        _, l, w = p["solid"].shape
        mine = p["solid"][r["deck"]]
        sub = dmap[y : y + l, x : x + w]
        sub[mine & (sub < 0)] = r["deck"]
        foot[y : y + l, x : x + w] |= mine
        was_ring[y : y + l, x : x + w] |= r["ring"]
        up = p["solid"][r["deck"] + 1 :] & ~r["ring"][None]
        for h, yy, xx in zip(*np.nonzero(up)):
            pieces.append((D + 1 + int(h), y + int(yy), x + int(xx), p["color"][r["deck"] + 1 + h, yy, xx]))
    lo = min(ds)
    for h in range(lo + 1, H):  # everything over each part's own deck goes
        above = foot & (dmap < h)
        solid[h][above] = False
        color[h][above] = 0
    # A lower part's walls raised to the deck: its top wall layer repeated.
    for h in range(lo + 1, D + 1):
        rise = foot & (dmap < h)
        top = np.take_along_axis(C0, np.clip(dmap, 0, H - 1)[None, :, :, None], axis=0)[0]
        solid[h][rise] = True
        color[h][rise] = top[rise]
    dist = ndimage.distance_transform_cdt(np.pad(foot, 1), metric="chessboard")[1:-1, 1:-1]
    ring = foot & (dist <= rim)
    color[D][foot & ~ring] = roofs[0]["deck_rgba"]
    for k in range(1, P + 1):
        solid[D + k][ring] = True
        layer = np.clip(dmap + k, 0, H - 1)
        old_s = np.take_along_axis(S0, layer[None], axis=0)[0]
        old_c = np.take_along_axis(C0, layer[None, :, :, None], axis=0)[0]
        keep = ring & was_ring & old_s  # a part's own parapet where it still is the edge keeps its colours
        color[D + k][ring] = roofs[0]["parapet_rgba"]
        color[D + k][keep] = old_c[keep]
    low: dict[tuple[int, int], tuple[int, Any]] = {}
    for h, y, x, rgba in pieces:
        if h >= H or solid[h, y, x]:
            continue
        solid[h, y, x] = True
        color[h, y, x] = rgba
        if (y, x) not in low or h < low[(y, x)][0]:
            low[(y, x)] = (h, rgba)
    for (y, x), (h, rgba) in low.items():  # a piece that stood on a parapet taken away stands on the deck
        for z in range(D + 1, h):
            if not solid[z, y, x]:
                solid[z, y, x] = True
                color[z, y, x] = rgba
    color[~solid] = 0
    # Pieces lifted past the old top: trim empty layers off the top.
    filled = np.flatnonzero(solid.any(axis=(1, 2)))
    solid, color = solid[: filled.max() + 1], color[: filled.max() + 1]
    return solid, color, {"one": True, "deck": int(D), "parapet": int(P), "rim": int(rim)}


def write_joined(root: Path, stem: str, specs: list[tuple[str, int]], layout: str = "row",
                 line=print) -> dict[str, Any]:
    """Join the parts' current stacks and write out/<stem>/stack/, saved as a new Stack version. specs: (part
    stem, quarter turns). Returns {meta, size, layers, colors, parts}."""
    from factory_env import check_stem
    from flow import autosave
    from stack_views import write_stack

    if layout not in LAYOUTS:
        raise JoinError(f"layout must be {' or '.join(LAYOUTS)}")
    if not 2 <= len(specs) <= MAX_PARTS:
        raise JoinError(f"join 2 to {MAX_PARTS} parts")
    try:
        stem = check_stem(stem)
        names = [check_stem(s) for s, _ in specs]
    except ValueError as e:
        raise JoinError(str(e)) from None
    if stem in names:
        raise JoinError("a joined building cannot be one of its own parts")
    turns = [t for _, t in specs]
    if layout == "corner":
        if len(turns) != 2:
            raise JoinError("a corner is two parts: the main building and its wing")
        turns = [turns[0], 1 if turns[1] == 0 else turns[1]]  # the wing faces right unless told otherwise
    parts = [read_part(root, n, t) for n, t in zip(names, turns)]
    solid, color, where = join(parts, layout)
    asset = Path(root) / "out" / stem
    if asset.is_dir():
        autosave(root, stem, ("stack",))
    # One cell is one size across a joined building: the first part's (buildings share 1 / the drawn wall's cells).
    result = SB.stack_result(solid, color, pitch=parts[0]["pitch"])
    meta = write_stack(result, asset / "stack", "joined", stem)
    roof = where[0].pop("roof", None)
    meta.update({"maker": "walls", "shape": "joined", "joined": {"layout": layout, "parts": where,
                                                                 **({"roof": roof} if roof else {})}})
    (asset / "stack" / "stack.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    autosave(root, stem, ("stack",))
    size = result["size"]
    line(f"Joined {len(parts)} parts ({layout}): {size['width']} wide, {size['length']} long, {size['height']} layers.")
    if roof:
        line("One roof over the corner." if roof["one"] else f"Each part keeps its roof: {roof['why']}.")
    return {"meta": meta, "size": size, "layers": result["layers"], "colors": result["colors"], "parts": where,
            "roof": roof}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=SCRIPTS.parent, help="factory root that holds out/")
    p.add_argument("--stem", required=True, help="the joined building's out/<stem>/ folder")
    p.add_argument("--layout", choices=LAYOUTS, default="row")
    p.add_argument("parts", nargs="+", help="part stems, each optionally :TURN (quarter turns)")
    p.add_argument("--render", type=Path, default=None, help="also write a turned 4-angle look to this PNG")
    args = p.parse_args()
    specs = []
    for raw in args.parts:
        name, _, turn = raw.partition(":")
        try:
            specs.append((name, int(turn or 0)))
        except ValueError:
            print(f"stack_join: {raw}: a turn is a whole number of quarter turns", file=sys.stderr)
            return 2
    try:
        out = write_joined(args.root, args.stem, specs, args.layout, line=lambda t: print(t, file=sys.stderr))
    except (JoinError, SB.WallsError) as e:
        print(f"stack_join: {e}", file=sys.stderr)
        return 2
    if args.render:
        SB.render_turned(SB.read_sheet_slices(args.root / "out" / args.stem / "stack"), args.render)
    print(json.dumps({"ok": True, **out["size"], "layers": out["layers"], "colors": out["colors"],
                      "parts": out["parts"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
