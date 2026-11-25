#!/usr/bin/env python3
"""A tower with setbacks from a building already made: blocks stacked one on another, each upper block narrower
than the one under it, all from the same drawn walls. No image model; nothing is redrawn.

The building's cleaned walls (out/<from>/flat/pixel/) are read once. Each block (a tier) is the drawn wall with its
floors and window bays cut or repeated (stack_building.set_floors, set_bays), built as stack_building builds a
block (build_shape). The bottom block keeps the ground floor (the shop front, its awning, the back door); the blocks
above it are upper floors only. Each upper block stands on the deck of the block under it, in its middle, so the
lower block's roof is a terrace with its parapet round the edge. Only the top block takes the roof asked for (flat
or gable); the others are flat. Every cell is 1 / the drawn wall's cells (pitch), as in every building.

The tower is written to out/<stem>/stack/ (stack.json "maker": "walls", "shape": "tower", "tower": the source and
each tier) and saved as a new Stack version.

CLI:
  python scripts/stack_tower.py --root R --stem S --from BUILDING TIER TIER ... [--roof flat|gable]
      [--ridge across|along] [--roof-colour C] [--render PNG]
  TIER is FLOORS:WIDE:DEEP (floors, window bays across the front, window bays along the side), bottom first,
  for example 3:5:3 6:3:2 (a three-floor base five bays wide, then six floors three bays wide set back over it).
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

MAX_TIERS = 5


def parse_tier(raw: str) -> tuple[int, int, int]:
    """FLOORS:WIDE:DEEP as three whole numbers."""
    parts = str(raw).split(":")
    try:
        if len(parts) != 3:
            raise ValueError
        f, w, d = (int(p) for p in parts)
    except ValueError:
        raise SB.WallsError(f"a tier is FLOORS:WIDE:DEEP, three whole numbers: {raw}") from None
    return f, w, d


def tier_shape(drawn: dict[str, np.ndarray], floors: int, wide: int, deep: int, ground: bool, look: str,
               roof: str = "flat", ridge: str = "across", roof_colour: Any = None) -> dict[str, Any]:
    """One block: the drawn walls cut to `floors` floors (with the ground floor or not) and wide x deep bays."""
    fl = SB.set_floors(drawn, floors, ground)
    view = fl.pop("view")
    found = SB.awning_of(view) if ground else None
    bays = SB.set_bays(view, wide, deep)
    if found is not None and "wide_order" in bays:  # the awning's columns, carried through the new bay order
        cols = np.zeros((1, view["front"].shape[1]), bool)
        cols[0, found[2]] = True
        found = (found[0], found[1], np.flatnonzero(SB.regroup(cols, bays["bays"], bays["wide_order"])[0]))
    shape = SB.build_shape(bays.pop("view"), awning="auto" if ground else "no", look=look, found=found,
                           scale=drawn["front"].shape[1], roof=roof, ridge=ridge, roof_colour=roof_colour)
    shape["floors"], shape["bays"] = fl, bays
    return shape


def tower_shape(drawn: dict[str, np.ndarray], tiers: list[tuple[int, int, int]], look: str = SB.LOOK,
                roof: str = "flat", ridge: str = "across", roof_colour: Any = None) -> dict[str, Any]:
    """The tower's solid [H, L, W] and colours, and each tier: {floors, wide, deep, at [layer, y, x] of its block's
    back left bottom corner, size [layers, depth, width] of its block, deck (the layer its roof deck is on)}."""
    if not 1 <= len(tiers) <= MAX_TIERS:
        raise SB.WallsError(f"a tower is 1 to {MAX_TIERS} tiers")
    shapes = []
    for i, (f, w, d) in enumerate(tiers):
        top = i == len(tiers) - 1
        try:
            shapes.append(tier_shape(drawn, f, w, d, i == 0, look, roof if top else "flat", ridge, roof_colour))
        except SB.WallsError as e:  # which tier could not be cut from the drawn walls, and why
            raise SB.WallsError(f"tier {i + 1} ({w} by {d} window bays) cannot be made: {e}") from None
    # Where each block goes: on the deck of the block under it, in its middle.
    places = []  # (layer, y, x) of the block's back left corner, and the eave its arrays carry round it
    for i, s in enumerate(shapes):
        e = int(s["roof"].get("eave", 0)) if s["roof"]["kind"] == "gable" else 0
        W = s["solid"].shape[2] - 2 * e
        if i == 0:
            places.append((0, 0, 0, e, s["depth"], W))
            continue
        z0, y0, x0, _, D0, W0 = places[-1]
        below = shapes[i - 1]
        if s["depth"] > D0 or W > W0 or (s["depth"] == D0 and W == W0):
            raise SB.WallsError(f"tier {i + 1} must be narrower than the tier under it ({W}x{s['depth']} cells on "
                                f"{W0}x{D0}): fewer bays across or along")
        deck = z0 + below["solid"].shape[0] - 1 - below["parapet"]
        places.append((deck + 1, y0 + (D0 - s["depth"]) // 2, x0 + (W0 - W) // 2, e, s["depth"], W))
    oy = -min(y - e for _, y, _, e, _, _ in places)
    ox = -min(x - e for _, _, x, e, _, _ in places)
    H = max(z + s["solid"].shape[0] for (z, *_), s in zip(places, shapes))
    L = max(y - e + oy + s["solid"].shape[1] for (_, y, _, e, _, _), s in zip(places, shapes))
    Wt = max(x - e + ox + s["solid"].shape[2] for (_, _, x, e, _, _), s in zip(places, shapes))
    solid = np.zeros((H, L, Wt), bool)
    color = np.zeros((H, L, Wt, 4), np.uint8)
    info = []
    for (z, y, x, e, D, W), s, (f, w, d) in zip(places, shapes, tiers):
        h, l, wd = s["solid"].shape
        ys, xs = y - e + oy, x - e + ox
        mine = s["solid"]
        # The block above wins in its own cells: a lower block's parapet under its wall is the wall now.
        solid[z : z + h, ys : ys + l, xs : xs + wd] |= mine
        color[z : z + h, ys : ys + l, xs : xs + wd][mine] = s["color"][mine]
        info.append({"floors": f, "wide": w, "deep": d, "at": [int(z), int(y + oy), int(x + ox)],
                     "size": [int(h), int(D), int(W)], "deck": int(z + h - 1 - s["parapet"]) if s["parapet"] else None,
                     "floor_order": s["floors"]["order"], "awning": s["awning"], "roof": s["roof"]})
    color[~solid] = 0
    return {"solid": solid, "color": color, "tiers": info, "drawn_floors": shapes[0]["floors"]["floors"]}


def write_tower(root: Path, stem: str, source: str, tiers: list[tuple[int, int, int]], roof: str = "flat",
                ridge: str = "across", roof_colour: Any = None, line=print) -> dict[str, Any]:
    """Build the tower from out/<source>/flat/pixel/ and write out/<stem>/stack/, saved as a new Stack version.
    Returns {meta, size, layers, colors, tiers}. Nothing is written when the tower cannot be built."""
    from factory_env import check_stem
    from flow import autosave
    from stack_views import write_stack

    try:
        stem, source = check_stem(stem), check_stem(source)
    except ValueError as e:
        raise SB.WallsError(str(e)) from None
    if stem == source:
        raise SB.WallsError("a tower is written as a new building, not over the one it is made from")
    if roof not in SB.ROOFS or ridge not in SB.RIDGES:
        raise SB.WallsError(f"roof must be {' or '.join(SB.ROOFS)} and ridge {' or '.join(SB.RIDGES)}")
    drawn = SB.read_pixel(root, source)
    pix = Path(root) / "out" / source / "flat" / "pixel" / "pixel.json"
    look = json.loads(pix.read_text(encoding="utf-8")).get("look", SB.LOOK) if pix.is_file() else SB.LOOK
    shape = tower_shape(drawn, [tuple(t) for t in tiers], look, roof, ridge, roof_colour)
    asset = Path(root) / "out" / stem
    if asset.is_dir():
        autosave(root, stem, ("stack",))
    result = SB.stack_result(shape["solid"], shape["color"], pitch=1.0 / drawn["front"].shape[1])
    meta = write_stack(result, asset / "stack", "tower", stem)
    meta.update({"maker": "walls", "look": look, "shape": "tower",
                 "tower": {"from": source, "drawn_floors": shape["drawn_floors"], "tiers": shape["tiers"]}})
    (asset / "stack" / "stack.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    autosave(root, stem, ("stack",))
    size = result["size"]
    line(f"Tower of {len(tiers)} tiers from {source}: {size['width']} wide, {size['length']} long, "
         f"{size['height']} layers.")
    return {"meta": meta, "size": size, "layers": result["layers"], "colors": result["colors"],
            "tiers": shape["tiers"]}


# ------------------------------------------------------------------------------------- a tower asked for by the app


def building_bays(root: Path, source: str) -> tuple[int, int]:
    """The window bays across the front and along the side of a building made from drawn walls: as it was made
    (its stack.json "bays", wide and deep), or the plain wall's drawn bays when its stack does not say."""
    meta_path = Path(root) / "out" / source / "stack" / "stack.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    if meta and (meta.get("maker") != "walls" or meta.get("shape") not in (None, "code")):
        raise SB.WallsError(f"{source} is not a building made from drawn walls, so it cannot be made a tower")
    bays = meta.get("bays") if isinstance(meta.get("bays"), dict) else {}
    wide, deep = bays.get("wide"), bays.get("deep")
    if isinstance(wide, int) and isinstance(deep, int) and wide > 0 and deep > 0:
        return wide, deep
    drawn = len(SB.find_bays(SB.read_pixel(root, source)["side"]))
    if not drawn:
        raise SB.WallsError(f"no window bays were found on {source}'s plain wall, so its tiers cannot step in")
    return drawn, drawn


def building_roof(root: Path, source: str) -> dict[str, Any]:
    """The roof the building was made with: {roof, ridge, roof_colour}, flat when its stack does not say."""
    try:
        roof = json.loads((Path(root) / "out" / source / "stack" / "stack.json").read_text(encoding="utf-8")).get("roof")
    except (OSError, ValueError, AttributeError):
        roof = None
    if not isinstance(roof, dict) or roof.get("kind") != "gable":
        return {"roof": "flat", "ridge": "across", "roof_colour": None}
    colour = roof.get("colour")
    return {"roof": "gable", "ridge": roof.get("ridge") if roof.get("ridge") in SB.RIDGES else "across",
            "roof_colour": colour if isinstance(colour, list) and len(colour) == 3 else None}


def stepped_tiers(wide: int, deep: int, tiers: int, floors: int) -> list[tuple[int, int, int]]:
    """Tiers bottom first as (floors, wide, deep): the bottom tier the building's own bays, each tier above one
    window bay narrower on every side (2 fewer across, 2 fewer along), every tier `floors` floors. When the bays run
    out before the tiers do, it says so: it never gives fewer tiers than asked."""
    if not 1 <= tiers <= MAX_TIERS:
        raise SB.WallsError(f"a tower is 1 to {MAX_TIERS} tiers")
    if not 1 <= floors <= SB.FLOOR_MAX:
        raise SB.WallsError(f"a tier is 1 to {SB.FLOOR_MAX} floors")
    room = 1 + (min(wide, deep) - 1) // 2
    if tiers > room:
        raise SB.WallsError(
            f"this building is {wide} window bay{'s' if wide != 1 else ''} across and {deep} along; each tier steps "
            f"in one bay on every side, so it has room for {room} tier{'s' if room != 1 else ''}, not {tiers}")
    return [(floors, wide - 2 * i, deep - 2 * i) for i in range(tiers)]


def _whole(body: dict[str, Any], key: str, default: int | None = None) -> int:
    v = body.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{key} must be a whole number")
    return v


def parse_request(body: Any) -> dict[str, Any]:
    """A tower asked for by the app (POST /2d/tower), checked: {stem (the new tower), from (a building already
    made), tiers (how many), floors (in each tier), roof?, ridge?, roof_colour? (the top tier's; left out: the
    building's own), project?}. Raises ValueError with a plain line."""
    from factory_env import check_stem

    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    stem = check_stem(str(body.get("stem") or ""))
    source = str(body.get("from") or "")
    if not source:
        raise ValueError("from: name the building the tower is made from")
    source = check_stem(source)
    if stem == source:
        raise ValueError("a tower is written as a new building, not over the one it is made from")
    out: dict[str, Any] = {"stem": stem, "from": source, "tiers": _whole(body, "tiers"),
                           "floors": _whole(body, "floors")}
    if not 1 <= out["tiers"] <= MAX_TIERS:
        raise ValueError(f"tiers must be 1 to {MAX_TIERS}")
    if not 1 <= out["floors"] <= SB.FLOOR_MAX:
        raise ValueError(f"floors must be 1 to {SB.FLOOR_MAX}")
    if body.get("roof") is not None:
        roof = str(body["roof"]).strip().lower()
        if roof not in SB.ROOFS:
            raise ValueError(f"roof must be {' or '.join(SB.ROOFS)}")
        out["roof"] = roof
    if body.get("ridge") is not None:
        ridge = str(body["ridge"]).strip().lower()
        if ridge not in SB.RIDGES:
            raise ValueError(f"ridge must be {' or '.join(SB.RIDGES)}")
        out["ridge"] = ridge
    if body.get("roof_colour") is not None:
        try:
            SB.roof_rgb(body["roof_colour"])
        except SB.WallsError as e:
            raise ValueError(str(e)) from None
        out["roof_colour"] = body["roof_colour"]
    project = body.get("project")
    if project is not None and not isinstance(project, str):
        raise ValueError("project must be a project's id or null")
    return out


def write_requested(root: Path, req: dict[str, Any], line=print) -> dict[str, Any]:
    """The tower parse_request checked: tiers stepped in from the building's own bays, the top tier's roof the
    building's unless asked, written by write_tower (a new Stack version)."""
    wide, deep = building_bays(root, req["from"])
    tiers = stepped_tiers(wide, deep, req["tiers"], req["floors"])
    roof = building_roof(root, req["from"])
    if "roof" in req:  # a roof asked for starts from the building's ridge and colour, not a flat one's
        roof["roof"] = req["roof"]
    for k in ("ridge", "roof_colour"):
        if k in req:
            roof[k] = req[k]
    return write_tower(root, req["stem"], req["from"], tiers, roof["roof"], roof["ridge"], roof["roof_colour"],
                       line=line)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=SCRIPTS.parent, help="factory root that holds out/")
    p.add_argument("--stem", required=True, help="the tower's out/<stem>/ folder")
    p.add_argument("--from", dest="source", required=True, help="the building whose drawn walls make the tower")
    p.add_argument("tiers", nargs="+", help="FLOORS:WIDE:DEEP for each block, bottom first")
    p.add_argument("--roof", choices=SB.ROOFS, default="flat", help="the top block's roof")
    p.add_argument("--ridge", choices=SB.RIDGES, default="across")
    p.add_argument("--roof-colour", default=None)
    p.add_argument("--render", type=Path, default=None, help="also write a turned 4-angle look to this PNG")
    args = p.parse_args()
    try:
        tiers = [parse_tier(t) for t in args.tiers]
        out = write_tower(args.root, args.stem, args.source, tiers, args.roof, args.ridge, args.roof_colour,
                          line=lambda t: print(t, file=sys.stderr))
    except SB.WallsError as e:
        print(f"stack_tower: {e}", file=sys.stderr)
        return 2
    if args.render:
        SB.render_turned(SB.read_sheet_slices(args.root / "out" / args.stem / "stack"), args.render)
    print(json.dumps({"ok": True, **out["size"], "layers": out["layers"], "colors": out["colors"],
                      "tiers": [{k: t[k] for k in ("floors", "wide", "deep", "at", "size")} for t in out["tiers"]]}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
