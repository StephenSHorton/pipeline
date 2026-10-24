#!/usr/bin/env python3
"""The workbench's eyes for a rig: where it stretches, bone by bone, and why.

    .venv/bin/python scripts/workbench/rigeyes.py out/mlg-sheep [--threshold 2] [--clips Walk,Run] [--heat flagged]

The rig check (rig/inspect.json) says how much a rig stretches; this says where and points at the cause. It reads the
rigged GLB the app's viewer shows (rig/rigged_allclips.glb, else rig.json's rigged file, else rigged.glb):

- stretch: per clip, each edge's worst length over its rest length across sampled frames (UV-seam copies of a
  corner welded first, so a seam is not an edge of length 0). Edges past the threshold (2x) are charged to the top
  bone of both ends, as the check's stretch_owners are. These numbers are per edge (its worst frame), so they are
  not the check's p99, which is over every edge at every frame; both are in the answer, labelled.
- weights: each bone's share of the welded points' weight.
- pivots: per leg (rig_territory.classify), the xz distance from the leg's root joint (the horse template's
  back_leg_pelvis, front_scapula) to where the leg's bones cross the leg line, over the body length. The sheep's
  hind pivot sat 0.39 of a body length ahead of its leg: Walk swung the leg about a point mid-body, and weight edits
  only moved its stretch a little. Moving the pivot back over the leg (0.13) is what fixed it.
- diagnosis: the above in plain words, each with what to try.
- heat (optional): for the bones named (or the flagged ones), rig_render's heat:<bone> from side, front, q and back,
  at rest and at the frame where that bone stretches most, tiled into one sheet. The frames render into a temp folder
  that is removed afterwards (keep_images keeps it); nothing is written into out/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rig  # noqa: E402
import rig_inspect  # noqa: E402
import rig_motion  # noqa: E402
import rig_territory  # noqa: E402

THRESHOLD = 2.0  # an edge this many times its rest length is a visible tear-like stretch on a PS1 mesh
SAMPLES = 24  # frames sampled per clip (rig_inspect's motion metrics sample len // 24 too)
SHARE_MIN = 0.005  # weight shares under this are left out of the answer
PIVOT_MAX = 0.25  # x body length a leg's root joint may sit from its leg (sheep: 0.39 broken, 0.13 fixed)
OWNS_NOTHING = 0.001  # a leg's upper bone with less than this share of the weight moves nothing
SEAM_MIN = 5  # a clip needs this many edges past the threshold before it is worth a diagnosis line
HEAT_MAX = 4  # bones in a heat sheet ("flagged" picks at most this many)
HEAT_VIEWS = ("side", "front", "q", "back")
HEAT_SIZE = 256
HEAT_TILE = 192


class RigError(ValueError):
    """No rig to review, or a rig this cannot read."""


@dataclass
class Rig:
    glb: Path
    doc: dict
    binary: bytes
    P: np.ndarray  # rest positions per corner
    F: np.ndarray
    J: np.ndarray
    W: np.ndarray
    names: list[str]
    parents: list[int]
    jpos: np.ndarray  # rest joint positions, from the inverse binds
    joints: list[int]  # node index per bone
    IBM: np.ndarray  # (bones, 4, 4) row-major


def rig_glb(folder: Path) -> Path:
    """The rigged GLB the viewer shows (candidates.ts riggedFiles): all clips first, then rig.json's, then rigged.glb."""
    folder = Path(folder)
    cand = [folder / "rig" / "rigged_allclips.glb"]
    try:
        named = json.loads((folder / "rig" / "rig.json").read_text(encoding="utf-8")).get("rigged")
    except (OSError, ValueError, AttributeError):
        named = None
    if isinstance(named, str) and named.endswith(".glb"):
        cand.append(folder / named.lstrip("/"))
    cand.append(folder / "rigged.glb")
    for p in cand:
        if p.is_file():
            return p
    raise RigError(f"{folder.name} has no rig yet (no rig/rigged_allclips.glb or rigged.glb); rig it first")


def load(glb: Path) -> Rig:
    doc, binary = rig.read_glb(Path(glb))
    if not doc.get("skins"):
        raise RigError(f"{Path(glb).name} has no skin")
    node = next((n for n in doc.get("nodes", []) if "skin" in n and "mesh" in n), None)
    meshes = [doc["meshes"][node["mesh"]]] if node else doc.get("meshes", [])
    prim = next((p for m in meshes for p in m["primitives"] if "JOINTS_0" in p["attributes"]), None)
    if prim is None:
        raise RigError(f"{Path(glb).name} has no skinned mesh")
    a = prim["attributes"]
    P = rig.accessor(doc, binary, a["POSITION"]).astype(float)
    F = (rig.accessor(doc, binary, prim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in prim
         else np.arange(len(P)).reshape(-1, 3))
    J = rig.accessor(doc, binary, a["JOINTS_0"]).astype(np.int64)
    W = rig.accessor(doc, binary, a["WEIGHTS_0"]).astype(float)
    skin = doc["skins"][node["skin"] if node else 0]
    joints = list(skin["joints"])
    names = [doc["nodes"][j].get("name") or f"bone_{k}" for k, j in enumerate(joints)]
    IBM = rig.accessor(doc, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    jpos = np.array([np.linalg.inv(m)[:3, 3] for m in IBM])
    pnode = {c: i for i, n in enumerate(doc["nodes"]) for c in n.get("children", [])}
    idx = {j: k for k, j in enumerate(joints)}
    parents = [idx.get(pnode.get(j, -1), -1) for j in joints]
    return Rig(Path(glb), doc, binary, P, F, J, W, names, parents, jpos, joints, IBM)


def welded(r: Rig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One point per group of corners at the same place: (first corner of each group, edges between groups, their
    rest lengths). Zero-length edges (a corner and its seam copy) are dropped."""
    inv = rig_motion.groups(r.P)
    first = np.zeros(int(inv.max()) + 1, np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    Fw = inv[r.F]
    E = np.unique(np.sort(np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]], 1), axis=0)
    E = E[E[:, 0] != E[:, 1]]
    H = max(float(np.ptp(r.P[:, 1])), 1e-9)
    l0 = np.linalg.norm(r.P[first[E[:, 0]]] - r.P[first[E[:, 1]]], axis=1)
    ok = l0 > 1e-6 * H
    return first, E[ok], l0[ok]


def stretch(r: Rig, threshold: float = THRESHOLD, clips: list[str] | None = None, samples: int = SAMPLES) -> dict:
    """Per clip: p99 and max of each edge's worst stretch, how many edges pass threshold, the bones that own them
    (top bone of both ends, once when they agree), and per bone the frame (fraction of the clip) where it has the
    most such edges."""
    first, E, l0 = welded(r)
    Jg, Wg = r.J[first], r.W[first]
    top = Jg[np.arange(len(Jg)), Wg.argmax(1)]
    Ph = np.c_[r.P[first], np.ones(len(first))]
    out = {}
    for name, times, _jp, M, _Q in rig_inspect._clip_frames(r.doc, r.binary, r.joints, r.IBM):
        if name == "Rest" or (clips and name not in clips):
            continue
        dur = float(times[-1]) or 1.0
        worst = np.ones(len(E))
        per_frame: list[tuple[float, Counter]] = []
        for f in range(0, len(times), max(1, len(times) // samples)):
            V = rig_motion.skin(Ph, Jg, Wg, M[f])
            ratio = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) / l0
            worst = np.maximum(worst, ratio)
            per_frame.append((float(times[f]) / dur, _owners(E[ratio > threshold], top)))
        bones = _owners(E[worst > threshold], top)
        at = {}
        for b in bones:
            frac, c = max(per_frame, key=lambda fc: fc[1][b])
            at[r.names[b]] = round(frac, 3)
        busiest = max(per_frame, key=lambda fc: sum(fc[1].values()))
        out[name] = {"p99": round(float(np.percentile(worst, 99)), 2), "max": round(float(worst.max()), 2),
                     "over": int((worst > threshold).sum()), "edges": int(len(E)),
                     "bones": {r.names[b]: n for b, n in bones.most_common(6)},
                     "worst_frame": at, "busiest_frame": round(busiest[0], 3)}
    return out


def _owners(E: np.ndarray, top: np.ndarray) -> Counter:
    c: Counter = Counter()
    for a, b in E:
        ta, tb = int(top[a]), int(top[b])
        c[ta] += 1
        if tb != ta:
            c[tb] += 1
    return c


def shares(r: Rig) -> dict[str, float]:
    """Each bone's share of the weight over the welded points (seam copies counted once)."""
    first, _, _ = welded(r)
    M = rig.to_dense(r.J[first], r.W[first], len(r.names))
    s = M.sum(0) / max(float(M.sum()), 1e-12)
    return {r.names[k]: float(s[k]) for k in range(len(r.names))}


def chains(r: Rig) -> dict | None:
    """rig_territory.classify on the rest skeleton, plus a part name per bone: "hind legs" / "front legs" (legs
    grouped by the spine joint they hang from, back to front), "neck", "tail", or "body"."""
    try:
        ch = rig_territory.classify(r.names, r.parents, r.jpos)
    except ValueError:
        return None
    fwd = np.asarray(ch["forward"], float)
    att = sorted({lg["attach"] for lg in ch["legs"]}, key=lambda k: float(r.jpos[k] @ fwd))
    label = {k: f"legs off {r.names[k]}" for k in att}
    if len(att) >= 2:
        label[att[0]], label[att[-1]] = "hind legs", "front legs"
    elif att:
        label[att[0]] = "legs"
    part = {n: "body" for n in r.names}
    for key in ("neck", "tail"):
        for k in (ch.get(key) or {}).get("bones") or []:
            part[r.names[k]] = key
    for lg in ch["legs"]:
        lg["part"] = label[lg["attach"]]
        for k in lg["bones"]:
            part[r.names[k]] = lg["part"]
    ch["part"] = part
    return ch


def pivots(r: Rig, ch: dict | None, limit: float = PIVOT_MAX) -> dict:
    """Per leg: how far (xz, over the body length along the rig's forward) its root joint sits from where its bones
    cross the leg line (the mesh's leg band top; halfway from the floor to the lowest leg root without one)."""
    if not ch or not ch["legs"]:
        return {"legs": [], "note": "no legs found in the skeleton"}
    line = rig_territory.mesh_leg_top(r.P, r.F)
    source = "mesh leg band"
    if line is None:
        floor = float(r.P[:, 1].min())
        line = floor + 0.5 * (min(float(r.jpos[lg["root"], 1]) for lg in ch["legs"]) - floor)
        source = "halfway up to the lowest leg root (no leg band in the mesh)"
    fwd = np.asarray(ch["forward"], float)
    length = max(float(np.ptp(r.P @ fwd)), 1e-9)
    height = max(float(np.ptp(r.P[:, 1])), 1e-9)
    legs = []
    for lg in ch["legs"]:
        root = lg["root"]
        col = rig_territory._column(lg, r.parents, r.jpos, line)
        d = float(np.linalg.norm((r.jpos[root] - col)[[0, 2]])) / length
        upper = [k for k in lg["bones"] if r.parents[k] == root]
        # how far the thigh's lower end (its child joint) sits above the leg line, x height: above it, the whole
        # thigh is where the territory fades legs out
        knee = [k for k in lg["bones"] if upper and r.parents[k] == upper[0]]
        above = (float(r.jpos[knee[0], 1]) - line) / height if knee else None
        legs.append({"root": r.names[root], "upper": r.names[upper[0]] if upper else None, "part": lg["part"],
                     "attach": r.names[lg["attach"]], "off_leg": round(d, 3), "ok": d <= limit,
                     "upper_above_line": None if above is None else round(above, 3)})
    return {"leg_line": round(float(line), 4), "leg_line_from": source, "body_length": round(length, 4),
            "measure": f"xz distance from the leg's root joint to its leg at the leg line, / body length; over {limit} "
                       "is flagged", "legs": legs}


def check_summary(folder: Path, glb: Path) -> dict | None:
    """The rig check's own summary (rig/inspect.json), and whether the rig changed after it was made."""
    p = Path(folder) / "rig" / "inspect.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    s = doc.get("summary") or {}
    slide = next((c.get("label") for c in doc.get("checks") or [] if c.get("key") == "slide"), None)
    return {"line": s.get("line"), "level": s.get("level"), "tears": s.get("tears"),
            "clips": {c: {k: v.get(k) for k in ("stretch_p99", "stretch_max", "tears") if k in v}
                      for c, v in (s.get("motion") or {}).items()},
            "foot_slide": slide, "problems": [q.get("label") for q in s.get("problems") or []],
            "stale": glb.stat().st_mtime > p.stat().st_mtime + 1}


def _buried(lg: dict) -> bool:
    """The leg's thigh lies wholly above the leg line, where the territory fades legs out: it owns nothing by design."""
    return lg.get("upper_above_line") is not None and lg["upper_above_line"] > 0


def diagnose(st: dict, sh: dict[str, float], pv: dict, ch: dict | None, threshold: float) -> tuple[list[str], list[str]]:
    """Plain-words findings and the bones worth a heat look, most telling first."""
    notes: list[str] = []
    flagged: list[str] = []
    part = (ch or {}).get("part") or {}
    legs = pv.get("legs") or []
    nothing = [lg for lg in legs if lg["upper"] and sh.get(lg["upper"], 0.0) < OWNS_NOTHING]
    # Group clips by the parts the two bones owning most of their stretched edges belong to: one seam (the body
    # against the hind legs) usually stretches in several clips, on one side in one and the other in the next.
    seams: dict[tuple, list[str]] = {}
    for clip, c in st.items():
        if c["over"] < SEAM_MIN or not c["bones"]:
            continue
        key = tuple(sorted({part.get(b, "body") for b in list(c["bones"])[:2]}))
        seams.setdefault(key, []).append(clip)
    for key, clips in sorted(seams.items(), key=lambda kv: -max(st[c]["over"] for c in kv[1])):
        worst = max(clips, key=lambda c: st[c]["over"])
        tally: Counter = Counter()
        for c in clips:
            tally.update(dict(list(st[c]["bones"].items())[:2]))
        bones = [b for b, _ in tally.most_common()]
        why = (f"{', '.join(bones)} ({' / '.join(key)}) stretch past {threshold:g}x in {', '.join(clips)} "
               f"({st[worst]['over']} edges in {worst})")
        mine = [lg for lg in legs if lg["part"] in key]
        off = [lg for lg in mine if not lg["ok"]]
        empty = [lg for lg in nothing if lg["part"] in key]
        if off:
            why += f": check the {off[0]['part']}' pivot ({off[0]['root']} is {off[0]['off_leg']:.2f} of a body " \
                   "length from its leg)"
        elif empty and all(_buried(lg) for lg in empty):
            why += f": the lower leg meets the body here ({', '.join(lg['upper'] for lg in empty)} lie above the leg " \
                   "line and own nothing by design). Soften the seam (the territory's leg_band, smoothing), don't move " \
                   "weight onto the thigh: on the sheep that made every clip worse"
        elif empty:
            why += f": {', '.join(lg['upper'] for lg in empty)} own nothing, so the lower leg meets the body at this " \
                   "seam; look at the heat of both sides of it"
        else:
            why += ": look at these bones' heat (weight bleeding across the seam, or a joint in the wrong place)"
        notes.append(why)
        flagged += [b for b in bones if b not in flagged]
    for lg in legs:
        if not lg["ok"]:
            notes.append(f"{lg['root']} sits {lg['off_leg']:.2f} of a body length from its leg (over {PIVOT_MAX}): the "
                         "leg swings about a point off the leg. Fix the joint (a re-rig with reuse_skin true), not "
                         "the weights")
            if lg["upper"] and lg["upper"] not in flagged:
                flagged.append(lg["upper"])
    buried = [lg for lg in nothing if _buried(lg)]
    if buried:
        notes.append(f"{', '.join(lg['upper'] for lg in buried)} own almost nothing because the whole bone sits above the "
                     f"leg line (its lower joint {min(lg['upper_above_line'] for lg in buried):.2f}x height up or more, "
                     "where the territory fades legs out): expected, not a weights bug")
    loose = [lg for lg in nothing if not _buried(lg)]
    if loose:
        notes.append(f"{', '.join(lg['upper'] for lg in loose)} own almost nothing (under {100 * OWNS_NOTHING:g}% "
                     "of the weight each): the leg's top bends with whatever owns it instead")
        flagged += [lg["upper"] for lg in nothing if lg["upper"] not in flagged]
    if not notes:
        notes.append(f"no clip has {SEAM_MIN} or more edges past {threshold:g}x and every leg's pivot is over its leg")
    return notes, flagged


def heat_sheet(folder: Path, glb: Path, bones: list[str], st: dict, frames_dir: Path, out: Path) -> dict:
    """rig_render's heat:<bone> for each bone, side/front/q/back at rest and at its worst frame, one row a bone."""
    import rig_render

    worst: dict[str, tuple[str, float]] = {}
    for b in bones:
        best = None
        for clip, c in st.items():
            n = c["bones"].get(b, 0)
            if b in c["worst_frame"] and (best is None or n > best[0]):
                best = (n, clip, c["worst_frame"][b])
        if best is None and st:  # a flagged pivot's leg with no stretched edges: the clip that stretches most
            clip = max(st, key=lambda k: st[k]["over"])
            best = (0, clip, st[clip]["busiest_frame"])
        worst[b] = (best[1], best[2]) if best else ("Rest", 0.0)
    frames: dict[str, list[float]] = {}
    for clip, frac in worst.values():
        if clip != "Rest" and frac not in frames.setdefault(clip, []):
            frames[clip].append(frac)
    res = rig_render.render(folder, {"glbs": [str(glb)], "views": list(HEAT_VIEWS),
                                     "modes": [f"heat:{b}" for b in bones], "frames": frames or {},
                                     "size": HEAT_SIZE, "sheet_tile": HEAT_TILE, "out": str(frames_dir)})
    recs = res["frames"]

    def png(clip: str, frac: float, view: str, bone: str) -> str | None:
        cand = [f for f in recs if f["clip"] == clip]
        if not cand:
            return None
        f = min(cand, key=lambda f: abs(f["frac"] - frac))
        return f["pngs"].get(f"{view}:heat:{bone}")

    cells, caps = [], []
    for b in bones:
        clip, frac = worst[b]
        for view in HEAT_VIEWS:
            cells.append(png("Rest", 0.0, view, b) or "-")
            caps.append(f"{b} rest {view}")
        for view in HEAT_VIEWS:
            cells.append(png(clip, frac, view, b) or "-")
            caps.append(f"{b} {clip}@{frac:.2f} {view}")
    rig_render.contact_sheet(cells, len(bones), 2 * len(HEAT_VIEWS), caps, out, tile=HEAT_TILE)
    return {"bones": {b: {"clip": worst[b][0], "frac": worst[b][1]} for b in bones},
            "columns": ["rest " + v for v in HEAT_VIEWS] + ["worst " + v for v in HEAT_VIEWS]}


def review(folder: Path, threshold: float = THRESHOLD, clips: list[str] | None = None,
           heat: list[str] | str | None = None, keep_images: bool = False, sheet_dir: Path | None = None) -> dict:
    """The whole review of out/<stem>'s rig. heat: bone names, or "flagged". The sheet goes to sheet_dir (a temp
    folder when not given); the frames it is tiled from go to a temp folder removed afterwards unless keep_images."""
    folder = Path(folder)
    glb = rig_glb(folder)
    r = load(glb)
    if clips:
        have = [a.get("name") for a in r.doc.get("animations", [])]
        missing = [c for c in clips if c not in have]
        if missing:
            raise RigError(f"no clip {missing[0]!r} in {glb.name} (has: {', '.join(n for n in have if n) or 'none'})")
    st = stretch(r, threshold, clips)
    sh = shares(r)
    ch = chains(r)
    pv = pivots(r, ch)
    notes, flagged = diagnose(st, sh, pv, ch, threshold)
    out = {"glb": glb, "check": check_summary(folder, glb), "threshold": threshold,
           "stretch_note": "p99/max here are of each edge's worst frame (the check's p99 is over every edge at every "
                           "frame); over = edges past threshold; bones = top bone of each end of those edges",
           "clips": st, "weights": {b: round(100 * s, 1) for b, s in sorted(sh.items(), key=lambda kv: -kv[1])
                                    if s > SHARE_MIN},
           "weights_unit": "% of the weight", "pivots": pv, "diagnosis": notes, "flagged": flagged[:HEAT_MAX]}
    if heat:
        bones = flagged[:HEAT_MAX] if heat == "flagged" else list(dict.fromkeys(heat))
        unknown = [b for b in bones if b not in r.names]
        if unknown:
            raise RigError(f"no bone {unknown[0]!r} in {glb.name} (bones: {', '.join(r.names)})")
        if not bones:
            out["heat"] = {"skipped": "nothing flagged"}
            return out
        frames_dir = Path(tempfile.mkdtemp(prefix="rigeyes_"))
        sheet_dir = Path(sheet_dir) if sheet_dir else Path(tempfile.mkdtemp(prefix="rigeyes_sheet_"))
        sheet = sheet_dir / f"{folder.name}_heat.png"
        try:
            out["heat"] = heat_sheet(folder, glb, bones, st, frames_dir, sheet)
        finally:
            if keep_images:
                out["heat_frames"] = frames_dir
            else:
                shutil.rmtree(frames_dir, ignore_errors=True)
        out["sheet"] = sheet
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset", help="an out/<stem> folder")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--clips", default="", help="comma-separated clip names (default: every clip)")
    ap.add_argument("--heat", default="", help='comma-separated bone names, or "flagged"')
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--sheet-dir", default=None)
    a = ap.parse_args(argv)
    heat = a.heat if a.heat in ("", "flagged") else [b for b in a.heat.split(",") if b]
    res = review(Path(a.asset), a.threshold, [c for c in a.clips.split(",") if c] or None, heat or None,
                 a.keep_images, Path(a.sheet_dir) if a.sheet_dir else None)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
