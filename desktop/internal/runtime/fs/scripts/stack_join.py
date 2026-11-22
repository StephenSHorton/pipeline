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

Where parts meet, the one placed first keeps its cells (a roof's eave never cuts into a neighbour's wall face:
the faces that meet are inside the building).

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
    return {"stem": stem, "turn": k, "solid": np.ascontiguousarray(solid), "color": np.ascontiguousarray(color),
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
    """The joined solid and colours [H, L, W] and where each part went (its offset in the joined grid)."""
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
    return solid, color, where


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
    result = SB.stack_result(solid, color)
    meta = write_stack(result, asset / "stack", "joined", stem)
    meta.update({"maker": "walls", "shape": "joined", "joined": {"layout": layout, "parts": where}})
    (asset / "stack" / "stack.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    autosave(root, stem, ("stack",))
    size = result["size"]
    line(f"Joined {len(parts)} parts ({layout}): {size['width']} wide, {size['length']} long, {size['height']} layers.")
    return {"meta": meta, "size": size, "layers": result["layers"], "colors": result["colors"], "parts": where}


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
