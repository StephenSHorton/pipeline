#!/usr/bin/env python3
"""Skeleton builder: a body plan's skeleton built for THIS mesh (design §2.2), instead of a vendor template posed onto it.

    build(source, plan, pred, out_dir) -> BuildInfo (dict)          raises BuildError(code, msg)
    wing_weight_repair(P, J, W, skel, plan) -> (J, W, info)
    wing_weight_smooth(P, faces, J, W, skel, plan) -> (J, W, info)

Everything is in the psx mesh's own coordinates: +Y up, +Z forward (head and toes), +X the creature's LEFT, floor at
the mesh's min y (0 for a psx mesh). H = mesh height. Rest rotations are identity: the skeleton is positions only.

Steps
1. Chains from the skin-tokens prediction (rig_pred.label with forward +Z), named by geometry, never by bone_N.
   Legs split by side (+X = L). A wrong leg count (or no prediction) falls back to rig_fit.floor_clusters: each leg
   is a vertical line from its floor cluster up to 0.45H. A facing cross-check (an independent label: red patch, else
   the top of the mesh, plus the toes) that puts the head at -Z raises BuildError('facing'): the pipeline never
   builds backwards.
2. hips = the predicted branch joint moved to the cross-section centre at its z; root = (hips.x, floor, hips.z).
   spine_0/1 at 1/3, 2/3 of the cross-section-centre line from hips to the neck base; spine_2, spine_3, head,
   mouth_upper resampled along the predicted head chain; the beak tip by a surface ray (rig_pred._surface_ext).
3. Tail resampled along the predicted tail chain; tail_tip by a surface ray.
4. Digitigrade legs: femur inset into the body, Foot/Toes/Toes_tip on the floor cluster (toes toward +Z), the
   intertarsal AnkleLeg apex toward -Z, the true knee LowerLeg turned forward and pulled inside.
5. Folded wings on the body sides (place_folded_wings: rig_pred.register's wing placement, lifted out).
6. Symmetrise about cx (the midpoint of the two foot clusters).
7. Self-check: joints inside (tips may be 0.02H out; root sits on the floor), rig_fit.pose_gate, plan.ratios,
   mirror error. A failure raises BuildError('check').
8. rig/template/skeleton.json and build.json (build.json is written on a BuildError too, with ok:false).

CLI: rig_build.py --stem S [--plan ground_bird] [--pred-mode sampled|beam] [--out DIR] [--no-pred]
"""
from __future__ import annotations

import argparse
import inspect as _inspect
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import rig
import rig_fit
import rig_metrics
import rig_plan
import rig_pred

FWD = np.array([0.0, 0.0, 1.0])
TIP_BASES = ("mouth_upper_tip", "mouth_lower_tip", "Toes_tip", "tail_tip", "wing_tip")
CONTACT_BASES = ("Foot", "Toes")  # floor joints: kept on the floor; on a paper-thin foot they may sit 0.02H outside
TIP_OUT_H = 0.02          # tips may sit this far outside the mesh
MIRROR_MAX_H = 0.01
FOOT_CLUSTER_MAX_H = 0.02


class BuildError(Exception):
    """code: 'facing' | 'pred' | 'wings' | 'check' | 'mesh' | 'plan'."""

    def __init__(self, code: str, msg: str, info: dict | None = None) -> None:
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg
        self.info = info or {}


# ---------------------------------------------------------------- names

def _split_side(name: str, suffixes=("_L", "_R")) -> tuple[str, str]:
    for s in suffixes:
        if name.endswith(s):
            return name[: -len(s)], s[1:]
    return name, ""


def is_tip(name: str) -> bool:
    return _split_side(name)[0] in TIP_BASES


def is_contact(name: str) -> bool:
    return _split_side(name)[0] in CONTACT_BASES


def mirror_name(name: str) -> str | None:
    base, side = _split_side(name)
    return None if not side else base + ("_R" if side == "L" else "_L")


def _role(base: str) -> str:
    if base in ("root", "hips"):
        return base
    if base.startswith("spine_") and base in ("spine_0", "spine_1"):
        return "spine"
    if base in ("spine_2", "spine_3"):
        return "neck"
    if base in ("head", "mouth_upper", "mouth_upper_tip"):
        return "head"
    if base.startswith("mouth_lower"):
        return "jaw"
    if base.startswith("tail"):
        return "tail"
    if base.startswith("wing_feather"):
        return "feather"
    if base.startswith("wing"):
        return "wing"
    if base in ("Foot",):
        return "foot"
    if base.startswith("Toes"):
        return "toe"
    if base in ("UpperLeg", "LowerLeg", "AnkleLeg"):
        return "leg"
    return "other"


# ---------------------------------------------------------------- mesh

class Mesh:
    """The source mesh, welded, with winding-number, ray and surface helpers (mesh coordinates)."""

    def __init__(self, source: Path) -> None:
        import trimesh
        self.doc, self.bin = rig.read_glb(source)
        prim = self.doc["meshes"][0]["primitives"][0]
        self.P = rig.accessor(self.doc, self.bin, prim["attributes"]["POSITION"]).astype(float)
        F = rig.accessor(self.doc, self.bin, prim["indices"]).reshape(-1) if "indices" in prim else np.arange(len(self.P))
        _, first, inv = np.unique(np.round(self.P / 1e-5).astype(np.int64), axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        self.V = self.P[first]
        F = inv[F.reshape(-1, 3)]
        self.F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        if len(self.V) < 8 or len(self.F) < 4:
            raise BuildError("mesh", f"{source.name} has {len(self.V)} points and {len(self.F)} triangles")
        self.y0 = float(self.V[:, 1].min())
        self.H = float(np.ptp(self.V[:, 1]))
        self.tm = trimesh.Trimesh(self.V, self.F, process=False)
        self.surface = np.r_[rig_fit._sample(self.V, self.F, 4000), self.V]
        self.tree = cKDTree(self.surface)

    def wn(self, Q: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, float))
        return rig_fit.winding_number(self.V, self.F, Q, chunk=4000)

    def inside(self, Q: np.ndarray) -> np.ndarray:
        return np.abs(self.wn(Q)) > 0.5

    def surf_d(self, Q: np.ndarray) -> np.ndarray:
        return self.tree.query(np.atleast_2d(np.asarray(Q, float)))[0]

    def ray(self, origin: np.ndarray, d: np.ndarray, cap_H: float = 0.4) -> np.ndarray | None:
        """Where a ray from origin along d finally leaves the mesh (rig_pred._surface_ext, re-cast past inner
        surfaces of overlapping parts), within cap_H x H; None on a miss."""
        o = np.asarray(origin, float)
        d = np.asarray(d, float) / max(np.linalg.norm(d), 1e-12)
        hit, at = None, o
        for _ in range(6):
            h = rig_pred._surface_ext(self.tm, at, d, self.H)
            if h is None or np.linalg.norm(h - o) > cap_H * self.H:
                break
            hit = np.asarray(h, float)
            at = hit + d * 2e-3 * self.H
            if not self.inside(at[None])[0]:
                break
        return hit

    def slab(self, z: float, y_lo: float = -np.inf, y_hi: float = np.inf, side: float = 0.0, cx: float = 0.0,
             min_pts: int = 6) -> np.ndarray:
        """Mesh points in a z slab around z (widened until it has min_pts), optionally one side of cx and a y band."""
        V = self.V
        band = (V[:, 1] >= y_lo) & (V[:, 1] <= y_hi)
        if side:
            band &= side * (V[:, 0] - cx) > 0
        w = 0.03 * self.H
        while True:
            S = V[band & (np.abs(V[:, 2] - z) < w)]
            if len(S) >= min_pts or w > 0.25 * self.H:
                return S
            w *= 1.5


def _mid(S: np.ndarray, z: float) -> np.ndarray:
    """A cross-section's centre: the middle of its x and y extent (robust to low-poly vertex density)."""
    return np.array([(S[:, 0].min() + S[:, 0].max()) / 2, (S[:, 1].min() + S[:, 1].max()) / 2, z])


def _arc(poly: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()) if len(poly) > 1 else 0.0


def _resample(poly: np.ndarray, fracs) -> list[np.ndarray]:
    L = _arc(poly)
    return [rig._at_arclen(poly, f * L) if L > 0 else poly[0].copy() for f in fracs]


def _rotate(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    a = axis / max(np.linalg.norm(axis), 1e-12)
    return R.from_rotvec(np.radians(deg) * a).apply(v)


def _pull(mesh: Mesh, p: np.ndarray, anchors: list[np.ndarray], mirror_cx: float | None = None, steps: int = 24
          ) -> tuple[np.ndarray, bool]:
    """p moved the least distance toward each anchor in turn until it (and its mirror about cx, when given) is
    inside. Returns (point, moved)."""
    S = np.array([-1.0, 1.0, 1.0])
    def ok(Q):
        good = mesh.inside(Q)
        if mirror_cx is not None:
            M = Q * S
            M[:, 0] += 2 * mirror_cx
            good &= mesh.inside(M)
        return good
    if ok(p[None])[0]:
        return p, False
    for a in anchors:
        t = np.linspace(0, 1, steps + 1)[1:, None]
        Q = p[None] + t * (a - p)[None]
        good = ok(Q)
        if good.any():
            return Q[int(np.argmax(good))], True
    return p, False


def _seek(mesh: Mesh, p: np.ndarray, r: float, mirror_cx: float | None = None, y_min: float | None = None,
          n: int = 7, fix_x: float | None = None) -> np.ndarray | None:
    """The inside point (and inside mirrored about cx, when given) nearest p within a cube of half-size r (a square
    in the x = fix_x plane for a midline joint); None."""
    g = np.linspace(-r, r, n)
    Q = p[None] + np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    if fix_x is not None:
        Q[:, 0] = fix_x
        Q = np.unique(Q, axis=0)
    if y_min is not None:
        Q = Q[Q[:, 1] >= y_min]
    Q = Q[np.argsort(np.linalg.norm(Q - p, axis=1))]
    good = mesh.inside(Q)
    if mirror_cx is not None:
        M = Q * np.array([-1.0, 1.0, 1.0])
        M[:, 0] += 2 * mirror_cx
        good &= mesh.inside(M)
    return Q[int(np.argmax(good))] if good.any() else None


# ---------------------------------------------------------------- facing

def _toes_vote(mesh: Mesh, legs_xz: list[np.ndarray], cx: float) -> dict:
    """Toes point forward: per leg, how far the floor points reach ahead of the leg's floor point vs behind it."""
    V, H = mesh.V, mesh.H
    low = V[V[:, 1] < mesh.y0 + 0.05 * H]
    out = []
    for xz in legs_xz:
        side = np.sign(xz[0] - cx) or 1.0
        S = low[(side * (low[:, 0] - cx) > 0) & (np.abs(low[:, 0] - xz[0]) < 0.15 * H)]
        if len(S) < 3:
            continue
        ahead, behind = float(S[:, 2].max() - xz[1]), float(xz[1] - S[:, 2].min())
        out.append((ahead - behind) / H)
    v = float(np.mean(out)) if out else 0.0
    return {"value": round(v, 3), "vote": int(np.sign(v)) if abs(v) > 0.03 else 0, "legs": len(out)}


def facing_check(mesh: Mesh, pred: rig_pred.Pred | None, red: np.ndarray | None, legs_xz: list[np.ndarray], cx: float,
                 prior: dict | None = None) -> dict:
    """Does the head point to +Z? Independent cues only (the +Z label always calls the +Z-most chain the head).

    prior: the psx pass's facing_check (psx.json; its own toes + VLM votes on the textured model). When it found the
    head at +Z with votes behind it, it is one more cue (weight 1), so a correctly faced bird whose tail stands taller
    than its head (label -1) is not refused on that single weak cue."""
    cues = {}
    if isinstance(prior, dict) and prior.get("head_axis") in ("+Z", "-Z"):
        backed = any(int(v or 0) != 0 for v in (prior.get("votes") or {}).values())
        vote = (1 if prior["head_axis"] == "+Z" else -1) if backed else 0
        cues["psx"] = {"head_axis": prior["head_axis"], "agrees": prior.get("agrees"), "vote": vote, "weight": 1}
    if pred is not None:
        lab = rig_pred.label(pred, None, red if red is not None and len(red) == len(pred.P) else None)
        yaw = rig_pred.head_chain_yaw(pred, lab)
        vote = 0 if yaw is None else (1 if yaw == 0 else -1 if yaw == 180 else 0)
        cues["label"] = {"how": lab["how"], "yaw": lab["yaw"], "head_chain_yaw": yaw, "vote": vote,
                         "weight": 2 if lab["how"] == "red patch" else 1}
    else:
        top = mesh.V[mesh.V[:, 1] > mesh.y0 + 0.9 * mesh.H]
        dz = float(top[:, 2].mean() - mesh.V[:, 2].mean()) if len(top) else 0.0
        cues["top"] = {"value": round(dz / mesh.H, 3), "vote": int(np.sign(dz)) if abs(dz) > 0.03 * mesh.H else 0, "weight": 1}
    toes = _toes_vote(mesh, legs_xz, cx)
    cues["toes"] = {**toes, "weight": 1}
    score = sum(c["vote"] * c["weight"] for c in cues.values())
    return {"cues": cues, "score": score, "head": "+z" if score >= 0 else "-z", "agrees": score >= 0}


# ---------------------------------------------------------------- wings

def place_folded_wings(mesh: Mesh, cx: float, side: float, z_from: float, z_to: float, body_lo: float, spec: dict,
                       names: list[str]) -> dict[str, np.ndarray]:
    """Folded wings on the body side (a copy of rig_pred.register's wing placement, rig_pred.py:407-424, applied to
    a chain laid from spine_1's z back to the tail base): each joint snaps (y, z) onto the side surface when it is
    more than 0.03H off it, and sits at `lateral` of the surface's lateral reach there (wing_1) or `inset_H` inside it
    (the rest). names: the wing chain names for this side, wing_1 .. wing_tip."""
    H = mesh.H
    lateral = float(spec.get("lateral", 0.85))
    height = float(spec.get("height", 0.6))
    inset = float(spec.get("inset_H", 0.03)) * H
    S = mesh.V[side * (mesh.V[:, 0] - cx) > 0]
    tr = cKDTree(S[:, 1:])
    out: dict[str, np.ndarray] = {}
    n = len(names)
    for k, nm in enumerate(names):
        f = k / max(n - 1, 1)
        z = z_from + f * (z_to - z_from)
        sl = mesh.slab(z, y_lo=body_lo, side=side, cx=cx)
        ylo, yhi = (float(sl[:, 1].min()), float(sl[:, 1].max())) if len(sl) else (body_lo, body_lo + 0.3 * H)
        y = ylo + (height - 0.15 * f) * (yhi - ylo)          # back and slightly down
        p_ = np.array([cx, y, z])
        d, j = tr.query(p_[1:])
        if d > 0.03 * H:
            p_[1:] = S[j, 1:]
        near = S[np.linalg.norm(S[:, 1:] - p_[1:], axis=1) < 0.06 * H]
        ext = float((side * (near[:, 0] - cx)).max()) if len(near) else 0.1 * H
        hit = mesh.ray(np.array([cx, p_[1], p_[2]]), np.array([side, 0.0, 0.0]), cap_H=1.0)
        if hit is not None:
            ext = min(ext, float(side * (hit[0] - cx))) if ext > 0 else float(side * (hit[0] - cx))
        off = lateral * ext if k == 0 else max(ext - inset, 0.5 * ext)
        p_[0] = cx + side * off
        out[nm] = p_
    return out


def _feathers(plan_joints: list[tuple[str, str | None]], pos: dict, side_sfx: str, H: float, feather_H: float) -> None:
    d = np.array([0.0, -0.5, -1.0])
    d /= np.linalg.norm(d)
    for name, parent in plan_joints:
        if not name.startswith("wing_feather") or not name.endswith(side_sfx):
            continue
        pos[name] = pos[parent] + feather_H * H * d


# ---------------------------------------------------------------- build

def _floor_points(V: np.ndarray, C: np.ndarray, seeds: np.ndarray, share: float = 0.08) -> list[np.ndarray]:
    """rig_fit.floor_clusters' final assignment, giving each cluster's points."""
    y = V[:, 1]
    low = V[y <= np.quantile(y, share)]
    cx = float(np.median(V[:, 0]))
    side_low = np.sign(low[:, 0] - cx)
    side_seed = np.sign(seeds[:, 0] - cx)
    cost = np.abs(low[:, None, 2] - C[None, :, 1]) + 1e3 * (side_low[:, None] != side_seed[None])
    lab = cost.argmin(1)
    return [low[lab == k] for k in range(len(C))]


def _feet(C: np.ndarray, cpts: list[np.ndarray], cx: float, y0: float, H: float, foot_y: float, toes_max: float
          ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Foot, Toes, Toes_tip per side (L, R), already mirror-symmetric about cx: Foot at the clusters' mean lateral
    offset and mean z (clamped into the z range both clusters cover, so a staggered stance keeps each foot on its own
    floor points), Toes at the forward third of that range, Toes_tip at its +Z end (at most toes_max ahead)."""
    off = float(np.mean([abs(C[0][0] - cx), abs(C[1][0] - cx)]))
    rng = [(float(P_[:, 2].min()), float(P_[:, 2].max())) if len(P_) >= 3 else (c[1] - 0.04 * H, c[1] + 0.06 * H)
           for P_, c in zip(cpts, C)]
    z_lo, z_hi = max(r[0] for r in rng), min(r[1] for r in rng)
    if z_hi - z_lo < 0.03 * H:  # the clusters share no z range: use the mean range
        z_lo, z_hi = float(np.mean([r[0] for r in rng])), float(np.mean([r[1] for r in rng]))
    fz = float(np.clip(np.mean(C[:, 1]), z_lo, z_hi))
    if all(len(P_) for P_ in cpts):  # a foot over 0.015H from its floor points: move the pair (±0.03H) to the best worst

        def worst(o, z):
            return max(float(np.linalg.norm(P_[:, [0, 2]] - [cx + sg * o, z], axis=1).min())
                       for P_, sg in zip(cpts, (1.0, -1.0)))
        best = (worst(off, fz), off, fz)
        for o in [] if best[0] <= 0.015 * H else np.linspace(max(off - 0.03 * H, 0.0), off + 0.03 * H, 13):
            for z in np.linspace(max(z_lo, fz - 0.03 * H), min(z_hi, fz + 0.03 * H), 13):
                w = worst(float(o), float(z))
                if w < best[0] - 1e-9:
                    best = (w, float(o), float(z))
        off, fz = best[1], best[2]
    tz = max(z_lo + (5 / 6) * (z_hi - z_lo), fz + 0.02 * H)
    ttz = float(np.clip(z_hi, tz + 0.02 * H, fz + max(toes_max, 0.04 * H)))
    if ttz <= tz:
        ttz = tz + 0.02 * H
    fronts = [P_[P_[:, 2] >= z_lo + (2 / 3) * (z_hi - z_lo)] for P_ in cpts]
    toff = [float(np.abs(f[:, 0] - cx).mean()) for f in fronts if len(f)]
    toff = float(np.mean(toff)) if toff else off
    out = []
    for s in (1.0, -1.0):
        out.append((np.array([cx + s * off, foot_y, fz]), np.array([cx + s * toff, foot_y, tz]),
                    np.array([cx + s * toff, y0 + 0.01 * H, ttz])))
    return out


def _segs(names: list[str], parents: list[str | None]) -> list[tuple[int, int]]:
    k = {n: i for i, n in enumerate(names)}
    return [(k[p], i) for i, p in enumerate(parents) if p is not None]


def build(source: Path, plan: dict, pred: rig_pred.Pred | None, out_dir: Path, pred_mode: str | None = None) -> dict:
    """Build the plan's skeleton for the mesh at `source` (psx.glb). Writes out_dir/skeleton.json and build.json and
    returns the BuildInfo. Raises BuildError(code, msg); build.json is still written then (ok: false)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag: dict = {"source": str(source), "plan": plan.get("id"), "pred_mode": pred_mode or "default",
                  "pred_joints": None if pred is None else len(pred.names)}
    try:
        info = _build(Path(source), plan, pred, out_dir, diag)
    except BuildError as e:
        rec = {"ok": False, "error": {"code": e.code, "msg": e.msg}, **diag, **e.info}
        (out_dir / "build.json").write_text(json.dumps(_jsonable(rec), indent=1))
        raise
    return info


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _build(source: Path, plan: dict, pred: rig_pred.Pred | None, out_dir: Path, diag: dict) -> dict:
    if not plan.get("joints"):
        raise BuildError("plan", f"plan {plan.get('id')} declares no joints (it wraps a fixed template)")
    mesh = Mesh(source)
    H, y0, V = mesh.H, mesh.y0, mesh.V
    caps = plan.get("caps", {})
    chains = plan.get("chains", {})
    legspec = chains.get("legs", {}) if isinstance(chains.get("legs"), dict) else {}
    wingspec = chains.get("wings", {}) if isinstance(chains.get("wings"), dict) else {}
    pj = rig_plan.joints(plan)
    names = [n for n, _ in pj]
    parents = [p for _, p in pj]
    pos: dict[str, np.ndarray] = {}
    src: dict[str, str] = {}
    cx_med = float(np.median(V[:, 0]))
    red = None
    try:
        red = rig_pred._red(mesh.doc, mesh.bin)
    except Exception:  # noqa: BLE001
        red = None

    # ---- 1. chains from the prediction
    lab = None
    head_c = tail_c = None
    legs_pred: dict[float, list[int]] = {}
    legs_source = "fallback"
    why_fallback = "no prediction"
    if pred is not None:
        lab = rig_pred.label(pred, FWD, red if red is not None and len(red) == len(pred.P) else None)
        diag["label"] = [{"label": c["label"], "pred": c["names"]} for c in lab["chains"]]
        head_c = next((c["joints"] for c in lab["chains"] if c["label"] == "head"), None)
        tail_c = next((c["joints"] for c in lab["chains"] if c["label"] == "tail"), None)
        legs = [c["joints"] for c in lab["chains"] if c["label"] == "leg"]
        want = int(caps.get("legs", 2))
        if len(legs) != want:
            why_fallback = f"the prediction has {len(legs)} legs, the plan {want}"
        else:
            for c in legs:
                legs_pred.setdefault(float(np.sign(pred.jp[c[0]][0] - cx_med) or 1.0), []).append(c)
            if want == 2 and sorted(legs_pred) == [-1.0, 1.0] and all(len(v) == 1 for v in legs_pred.values()):
                legs_source = "pred"
                legs_pred = {k: v[0] for k, v in legs_pred.items()}
            else:
                why_fallback = "the predicted legs are not one per side"
                legs_pred = {}
    if legs_source == "fallback":
        legs_pred = {}
    diag["legs_source"] = legs_source
    if legs_source == "fallback":
        diag["legs_fallback_why"] = why_fallback

    # floor clusters (both branches: the feet always sit on the mesh's own floor points)
    if legs_source == "pred":
        seeds = np.array([pred.jp[legs_pred[s][-1]] for s in (1.0, -1.0)])
    else:
        low = V[V[:, 1] <= np.quantile(V[:, 1], 0.08)]
        seeds = []
        for s in (1.0, -1.0):
            S = low[s * (low[:, 0] - cx_med) > 0]
            seeds.append(S.mean(0) if len(S) else np.array([cx_med + s * 0.1 * H, y0, float(np.median(low[:, 2]))]))
        seeds = np.array(seeds)
    C, counts = rig_fit.floor_clusters(V, seeds)
    cpts = _floor_points(V, C, seeds)
    cx = float(C[:, 0].mean())
    diag["floor_clusters"] = {"L": C[0].tolist(), "R": C[1].tolist(), "n": counts.tolist()}
    diag["cx"] = cx

    # facing (independent cues)
    fc = facing_check(mesh, pred, red, [C[0], C[1]], cx, prior=plan.get("facing_check"))
    diag["facing_check"] = fc
    if not fc["agrees"]:
        raise BuildError("facing", "the head points to -Z (cues: " + ", ".join(
            f"{k} {v['vote']:+d}" for k, v in fc["cues"].items()) + "); fix the psx facing, the builder never builds backwards")
    if head_c is not None:
        tipz = pred.jp[head_c[-1]][2]
        if tipz < pred.jp[pred.root[0]][2]:
            raise BuildError("facing", "the predicted head chain points to -Z")

    # leg tops (the body bottom, for cross-sections)
    leg_root = {}
    for k, s in enumerate((1.0, -1.0)):
        if legs_source == "pred":
            leg_root[s] = pred.jp[legs_pred[s][0]].copy()
        else:
            leg_root[s] = np.array([C[k][0], y0 + 0.45 * H, C[k][1]])
    body_lo = float(min(p[1] for p in leg_root.values())) - 0.05 * H

    # ---- 2. hips, root, spine, neck, head, jaw
    if pred is not None:
        hb = pred.jp[pred.root[0]].copy()
        src_h = f"pred:{pred.names[pred.root[0]]}"
    else:
        hb = np.array([cx, y0 + 0.45 * H, float(C[:, 1].mean())])
        src_h = "fallback"
    S = mesh.slab(hb[2], y_lo=body_lo)
    hips = _mid(S, hb[2]) if len(S) else hb
    hips[0] = cx
    if not mesh.inside(hips[None])[0]:
        hips = hb
    pos["hips"], src["hips"] = hips, "centroid"
    diag["hips_from"] = src_h
    pos["root"], src["root"] = np.array([hips[0], y0, hips[2]]), "plan"

    if head_c is not None:
        hline = np.array([pred.jp[k] for k in head_c])
        hsrc = "pred:head"
    else:
        # fallback: the mesh's upper centreline toward +Z (rig._centreline), from above the hips
        start = hips + np.array([0, 0.1 * H, 0.15 * H])
        hline, _ = rig._centreline(V, start, hips[1])
        hline = hline[:: max(1, len(hline) // 6)]
        hsrc = "fallback"
    neck_base = hline[0]
    zs = np.linspace(hips[2], neck_base[2], 9)[1:-1]
    cl = [hips]
    for z in zs:
        f = (z - hips[2]) / (neck_base[2] - hips[2]) if abs(neck_base[2] - hips[2]) > 1e-9 else 0.5
        yy = hips[1] + f * (neck_base[1] - hips[1])
        Sz = mesh.slab(z, y_lo=max(body_lo, yy - 0.25 * H), y_hi=yy + 0.25 * H)
        c = _mid(Sz, z) if len(Sz) else np.array([cx, yy, z])
        c[0] = cx
        cl.append(c if mesh.inside(c[None])[0] else np.array([cx, yy, z]))
    cl.append(neck_base)
    sp0, sp1 = _resample(np.array(cl), (1 / 3, 2 / 3))
    pos["spine_0"], pos["spine_1"] = sp0, sp1
    src["spine_0"] = src["spine_1"] = "centroid"
    s2, s3, hd, mu = _resample(hline, (0, 1 / 3, 2 / 3, 1))
    for n, p in (("spine_2", s2), ("spine_3", s3), ("head", hd), ("mouth_upper", mu)):
        pos[n], src[n] = p, hsrc if head_c is None else ("pred:" + pred.names[head_c[0]] if n == "spine_2" else
                                                            "pred:" + pred.names[head_c[-1]] if n == "mouth_upper" else hsrc)
    neck = chains.get("neck", {}) if isinstance(chains.get("neck"), dict) else {}
    cap = float(neck.get("tip_cap_H", 0.4))
    # a predicted head chain that ends in a comb or crest climbs steeply from head to mouth_upper (passing
    # chickens rise ~50 deg, comb-tipped ones 59-72 deg): swing mouth_upper toward the head's forward-most surface,
    # keeping the segment's length, so the beak base sits in the head and not in the comb
    seg = mu - hd
    rise = float(neck.get("beak_rise_max_deg", 55))
    if np.degrees(np.arctan2(seg[1], max(seg[2], 1e-9))) > rise:
        near = V[np.linalg.norm(V - hd, axis=1) < 0.3 * H]
        if len(near):
            f = near[np.argmax(near[:, 2])] - hd
            if f[2] > 0:
                # the forward-most point is usually the beak tip; the base is the furthest point toward it that
                # is still inside the (low-poly, thin-beaked) head
                fn = float(np.linalg.norm(f))
                for k in (0.6, 0.5, 0.4, 0.3, 0.2):
                    cand = hd + f / fn * min(float(np.linalg.norm(seg)), k * fn)
                    if mesh.inside(cand[None])[0]:
                        mu = pos["mouth_upper"] = cand
                        src["mouth_upper"] = "head_forward"
                        break
    # the beak: a ray from mouth_upper toward the head's forward-most surface (the head direction, bent forward)
    hd_dir = mu - hd
    headpts = V[np.linalg.norm(V - mu, axis=1) < 0.25 * H]
    if len(headpts):
        fwd_pt = headpts[np.argmax(headpts[:, 2])]
        d = fwd_pt - mu
        if np.linalg.norm(d) > 1e-6 and d[2] > 0:
            hd_dir = d
    tip = mesh.ray(mu, hd_dir, cap)
    if tip is None:
        tip = mu + 0.05 * H * hd_dir / max(np.linalg.norm(hd_dir), 1e-9)
        src["mouth_upper_tip"] = "fallback"
    else:
        src["mouth_upper_tip"] = "surface_ray"
    pos["mouth_upper_tip"] = tip
    jaw = chains.get("jaw", {}) if isinstance(chains.get("jaw"), dict) else {}
    ml = mu - np.array([0, float(jaw.get("drop_H", 0.02)) * H, 0])
    pos["mouth_lower"] = ml
    pos["mouth_lower_tip"] = ml + float(jaw.get("len_frac", 0.85)) * (tip - ml)
    src["mouth_lower"] = src["mouth_lower_tip"] = "plan"

    # ---- 3. tail
    tailspec = chains.get("tail", {}) if isinstance(chains.get("tail"), dict) else {}
    if tail_c is not None:
        tline = np.array([pred.jp[k] for k in tail_c])
        tsrc = "pred:tail"
    else:
        back = V[V[:, 2] < np.quantile(V[:, 2], 0.03)]
        end = back.mean(0) if len(back) else hips - np.array([0, 0, 0.3 * H])
        a = hips + 0.35 * (end - hips)
        tline = np.array([a, a + 0.5 * (end - a), end])
        tsrc = "fallback"
    t1, t2, t3 = _resample(tline, (0, 0.5, 1))
    if len(tline) < 2 or _arc(tline) < 1e-6:
        t2, t3 = t1 - np.array([0, 0, 0.1 * H]), t1 - np.array([0, 0, 0.2 * H])
    pos.update(tail_1=t1, tail_2=t2, tail_3=t3)
    src["tail_1"] = "pred:" + pred.names[tail_c[0]] if tail_c is not None else tsrc
    src["tail_2"] = tsrc
    src["tail_3"] = "pred:" + pred.names[tail_c[-1]] if tail_c is not None else tsrc
    tdir = t3 - t2
    tt = mesh.ray(t3, tdir, float(tailspec.get("tip_cap_H", 0.4)))
    if tt is None:
        tt = t3 + 0.05 * H * tdir / max(np.linalg.norm(tdir), 1e-9)
        src["tail_tip"] = "fallback"
    else:
        src["tail_tip"] = "surface_ray"
    pos["tail_tip"] = tt

    # ---- 4. legs (digitigrade)
    inset = float(legspec.get("femur_inset", 0.3))
    foot_y = y0 + float(legspec.get("foot_y_H", 0.02)) * H
    toes_max = float(legspec.get("toes_max_H", 0.12)) * H
    turn = float(legspec.get("knee_turn_deg", 35))
    back = float(legspec.get("ankle_back_H", 0.06)) * H
    feet = _feet(C, cpts, cx, y0, H, foot_y, toes_max)
    for k, (s, sfx) in enumerate(((1.0, "L"), (-1.0, "R"))):
        lsrc = f"pred:{pred.names[legs_pred[s][0]]}" if legs_source == "pred" else "fallback"
        r0 = leg_root[s]
        Ss = mesh.slab(r0[2], y_lo=r0[1] - 0.05 * H, y_hi=r0[1] + 0.35 * H, side=s, cx=cx)
        lat = _mid(Ss, r0[2]) if len(Ss) else r0
        ul = r0 + inset * (lat - r0)
        if not mesh.inside(ul[None])[0]:
            ul, _ = _pull(mesh, ul, [lat, hips])
        foot, toes, toes_tip = feet[k]
        # AnkleLeg (intertarsal): the predicted middle joint when its bend apex points back
        ank = None
        if legs_source == "pred":
            ch = legs_pred[s]
            if len(ch) >= 3:
                mid = pred.jp[ch[len(ch) // 2]].copy()
                u = foot - ul
                t = float(np.clip((mid - ul) @ u / max(u @ u, 1e-12), 0, 1))
                # ... and sits in the leg's middle height band: a chain that runs on into the toes has its middle
                # joint near the floor, which would collapse the tarsus onto the foot
                band = 0.2 * (ul[1] - foot[1]) < mid[1] - foot[1] < 0.8 * (ul[1] - foot[1])
                if (mid - (ul + t * u))[2] < 0 and band:
                    ank = mid
                    src[f"AnkleLeg_{sfx}"] = f"pred:{pred.names[ch[len(ch) // 2]]}"
        if ank is None:
            ank = ul + 0.55 * (foot - ul) - np.array([0, 0, back])
            src[f"AnkleLeg_{sfx}"] = "plan"
        if not mesh.inside(ank[None])[0]:
            ank, _ = _pull(mesh, ank, [ul + 0.55 * (foot - ul), ul])
        # LowerLeg (the true knee): along hip->ankle turned forward and down, 0.45 of the way, inside
        u = ank - ul
        L = float(np.linalg.norm(u))
        u = u / max(L, 1e-12)
        ax = np.cross(u, FWD)
        if np.linalg.norm(ax) < 1e-6:
            ax = np.array([1.0, 0, 0])
        knee = ul + 0.45 * L * _rotate(u, ax, turn)
        knee, _ = _pull(mesh, knee, [(ul + ank) / 2, ul])
        pos[f"UpperLeg_{sfx}"], src[f"UpperLeg_{sfx}"] = ul, lsrc
        pos[f"LowerLeg_{sfx}"], src[f"LowerLeg_{sfx}"] = knee, "plan"
        pos[f"AnkleLeg_{sfx}"] = ank
        pos[f"Foot_{sfx}"], src[f"Foot_{sfx}"] = foot, "floor_cluster"
        pos[f"Toes_{sfx}"], src[f"Toes_{sfx}"] = toes, "floor_cluster"
        pos[f"Toes_tip_{sfx}"], src[f"Toes_tip_{sfx}"] = toes_tip, "floor_cluster"
        if legs_source == "fallback":
            src[f"UpperLeg_{sfx}"] = "fallback"

    # ---- 5. wings
    wings_kind = caps.get("wings")
    wnames = {"L": [], "R": []}
    for lst in rig_plan.chain_names(plan, "wings") if "wings" in chains else []:
        if lst:
            wnames[_split_side(lst[0])[1]] = lst
    if wings_kind == "folded" and any(wnames.values()):
        for s, sfx in ((1.0, "L"), (-1.0, "R")):
            got = place_folded_wings(mesh, cx, s, float(pos["spine_1"][2]), float(pos["tail_1"][2]), body_lo, wingspec, wnames[sfx])
            for n, p in got.items():
                pos[n], src[n] = p, "body_side"
            _feathers(pj, pos, "_" + sfx, H, float(wingspec.get("feather_H", 0.04)))
            for n, _ in pj:
                if n.startswith("wing_feather") and n.endswith("_" + sfx):
                    src[n] = "plan"
    elif wings_kind == "spread":
        raise BuildError("wings", "spread wings need a predicted wing chain; this builder only places folded wings")
    missing = [n for n in names if n not in pos]
    if missing:
        raise BuildError("plan", f"the builder placed no position for {missing[:6]}")

    # ---- 6. symmetrise about cx
    Sx = np.array([-1.0, 1.0, 1.0])

    def refl(p):
        q = p * Sx
        q[0] += 2 * cx
        return q

    pairs = [(n, mirror_name(n)) for n in names if _split_side(n)[1] == "L" and mirror_name(n) in pos]
    pre = max((float(np.linalg.norm(pos[a] - refl(pos[b]))) for a, b in pairs), default=0.0) / H
    for a, b in pairs:
        m = (pos[a] + refl(pos[b])) / 2
        if m[0] < cx:
            m = refl(m)
        pos[a], pos[b] = m, refl(m)
    for n in names:
        if not _split_side(n)[1]:
            pos[n][0] = cx
    diag["mirror_err_before_H"] = pre

    # inside pass (parents first; sided joints move as a mirrored pair)
    moved = []
    for n, par in pj:
        side = _split_side(n)[1]
        if n == "root" or side == "R" or is_tip(n):
            continue
        if mesh.inside(pos[n][None])[0] and (side != "L" or mesh.inside(refl(pos[n])[None])[0]):
            continue
        mc = cx if side == "L" else None
        q = _seek(mesh, pos[n], 0.03 * H, mc, y_min=y0 + 0.004 * H, fix_x=None if side else cx)
        if q is None and is_contact(n):
            q = _seek(mesh, pos[n], 0.05 * H, mc, y_min=y0 + 0.004 * H, n=11)
        if q is not None or is_contact(n):
            if q is not None:
                pos[n] = q
                moved.append(n)
                if side == "L":
                    pos[mirror_name(n)] = refl(q)
            continue
        anchors = [np.array([cx, pos[n][1], pos[n][2]])] if side else []
        anchors.append(pos[par] if par else pos["hips"])
        anchors.append(pos["hips"])
        p, did = _pull(mesh, pos[n], anchors, cx if side == "L" else None)
        if did:
            pos[n] = p
            moved.append(n)
            if side == "L":
                pos[mirror_name(n)] = refl(p)
    for n, par in pj:
        side = _split_side(n)[1]
        if not is_tip(n) or side == "R":
            continue
        for _ in range(12):
            pts = [pos[n]] + ([refl(pos[n])] if side == "L" else [])
            ins = mesh.inside(np.array(pts))
            dd = mesh.surf_d(np.array(pts))
            if all(i or d <= TIP_OUT_H * H * 0.9 for i, d in zip(ins, dd)):
                break
            pos[n] = pos[n] + 0.15 * (pos[par] - pos[n])
            if n not in moved:
                moved.append(n)
        if side == "L":
            pos[mirror_name(n)] = refl(pos[n])
    diag["pulled_inside"] = moved

    # ---- 7. self-check
    jpos = np.array([pos[n] for n in names])
    ins = mesh.inside(jpos)
    dsurf = mesh.surf_d(jpos)
    bad_inside = [n for n, i, d in zip(names, ins, dsurf)
                  if not i and n != "root" and (not (is_tip(n) or is_contact(n)) or d > TIP_OUT_H * H)]
    contact_out = [n for n, i in zip(names, ins) if not i and is_contact(n)]
    segs = _segs(names, parents)
    segs_body = [(a, c) for a, c in segs if names[a] != "root"]
    gate = rig_fit.pose_gate(jpos, segs_body, V, mesh.F, mesh.surface)

    def arc(ns):
        return _arc(np.array([pos[n] for n in ns])) / H
    lengths = {
        "leg": arc(["UpperLeg_L", "LowerLeg_L", "AnkleLeg_L", "Foot_L"]),
        "neck": arc([n for n in (neck.get("names") or ["spine_2", "spine_3", "head", "mouth_upper", "mouth_upper_tip"])]),
        "tail": arc([n for n in (tailspec.get("names") or ["tail_1", "tail_2", "tail_3", "tail_tip"])]),
    }
    ratios = plan.get("ratios", {})
    bad_ratio = [f"{k} {v:.3f}H outside {ratios[k]}" for k, v in lengths.items()
                 if k in ratios and not (ratios[k][0] <= v <= ratios[k][1])]
    mirror = max((float(np.linalg.norm(pos[a] - refl(pos[b]))) for a, b in pairs), default=0.0) / H
    mid_off = max((abs(pos[n][0] - cx) for n in names if not _split_side(n)[1]), default=0.0) / H
    mirror = max(mirror, mid_off)
    feet = {}
    for k, sfx in enumerate(("L", "R")):
        f = pos[f"Foot_{sfx}"]
        dc = float(np.linalg.norm(f[[0, 2]] - C[k])) / H
        dp = float(np.linalg.norm(cpts[k][:, [0, 2]] - f[[0, 2]], axis=1).min()) / H if len(cpts[k]) else dc
        feet[sfx] = {"to_centroid_H": dc, "to_points_H": dp, "y_H": float(f[1] - y0) / H}
    axes = {"head_z_gt_hips": bool(pos["head"][2] > pos["hips"][2]),
            "toes_tip_z_gt_foot": bool(all(pos[f"Toes_tip_{s}"][2] > pos[f"Foot_{s}"][2] for s in "LR")),
            "L_at_plus_x": bool(all(pos[n][0] > cx for n in names if _split_side(n)[1] == "L"))}
    # cd_j2b both ways (/H): predicted joints to the built bones, built joints (no wings, no root) to predicted bones
    cd = {}
    if pred is not None:
        bsegs = [(jpos[a], jpos[c]) for a, c in segs_body]
        a2b_all = rig_pred.cd_j2b(pred.jp, bsegs) / H
        # judged on the predicted joints the plan maps (the labelled chains and the predicted root): a prediction's
        # extra parts (the drawn chicken's three-toe fans and comb: 18 of its 43 joints, 0.056H off a one-toe plan)
        # are not the built skeleton missing the model
        used = set(int(j) for c in (lab or {}).get("chains", []) for j in c.get("joints", []))
        used |= {i for i in range(len(pred.names)) if pred.par[i] < 0}
        a2b = a2b_all[sorted(used)] if used else a2b_all
        psegs = [(pred.jp[int(pred.par[c])], pred.jp[c]) for c in range(len(pred.names)) if pred.par[c] >= 0]
        keep = [i for i, n in enumerate(names) if n != "root" and _role(_split_side(n)[0]) not in ("wing", "feather")]
        b2a = rig_pred.cd_j2b(jpos[keep], psegs) / H if psegs else np.zeros(1)
        cd = {"pred_to_built_H": {"mean": float(a2b.mean()), "max": float(a2b.max()), "joints": int(len(a2b))},
              "pred_all_to_built_H": {"mean": float(a2b_all.mean()), "max": float(a2b_all.max()), "joints": int(len(a2b_all))},
              "built_to_pred_H": {"mean": float(b2a.mean()), "max": float(b2a.max())}}
    pf = rig_metrics.pose_flags(jpos, segs_body, V, mesh.F, mesh.surface, H, names=names)
    checks = {"inside_share": float(ins.mean()), "outside": bad_inside, "contact_outside": contact_out, "gate": gate, "lengths_H": lengths,
              "ratio_problems": bad_ratio, "mirror_err_H": mirror, "feet": feet, "axes": axes,
              "flagged_joints": [j["joint"] for j in pf.get("joints", []) if j.get("flags")]}
    problems = []
    if bad_inside:
        problems.append(f"joints outside the mesh: {bad_inside}")
    if gate.get("level") == "fail":
        problems.append(f"pose gate fail ({gate})")
    problems += bad_ratio
    if mirror > MIRROR_MAX_H:
        problems.append(f"mirror error {mirror:.4f}H > {MIRROR_MAX_H}H")
    if not all(axes.values()):
        problems.append(f"axes: {axes}")
    checks["problems"] = problems

    sides = {"L": "L", "R": "R", "": "mid"}
    joints = [{"name": n, "parent": p, "pos": [round(float(x), 6) for x in pos[n]], "role": _role(_split_side(n)[0]),
               "side": sides[_split_side(n)[1]], "source": src.get(n, "plan")} for n, p in pj]
    skel = {"plan": plan.get("id"), "joints": joints, "H": round(H, 6), "cx": round(cx, 6), "floor_y": round(y0, 6)}
    rec = {"ok": not problems, **diag, "checks": checks, "cd_j2b": cd, "sources": {n: src.get(n, "plan") for n in names},
           "joints": len(names)}
    if problems:
        raise BuildError("check", "; ".join(problems), {"checks": checks, "cd_j2b": cd, "skeleton": skel})
    (out_dir / "skeleton.json").write_text(json.dumps(skel, indent=1))
    (out_dir / "build.json").write_text(json.dumps(_jsonable(rec), indent=1))
    return {**_jsonable(rec), "names": names, "parents": parents,
            "parent_index": [names.index(p) if p else -1 for p in parents],
            "pos": jpos.tolist(), "skeleton": skel, "H": H, "cx": cx,
            "skeleton_json": str(out_dir / "skeleton.json"), "build_json": str(out_dir / "build.json")}


# ---------------------------------------------------------------- wing weight repair

def _skel_arrays(skel: dict) -> tuple[list[str], np.ndarray, list[int]]:
    """Accepts skeleton.json ({joints:[{name,parent,pos}]}) or the synthesiser's {names, parent, rest_t|pos|jpos}
    (parent: indices or names; rest_t: identity-rest local translations, accumulated to world)."""
    if "joints" in skel:
        names = [j["name"] for j in skel["joints"]]
        k = {n: i for i, n in enumerate(names)}
        par = [k[j["parent"]] if j.get("parent") is not None else -1 for j in skel["joints"]]
        return names, np.array([j["pos"] for j in skel["joints"]], float), par
    names = list(skel["names"])
    k = {n: i for i, n in enumerate(names)}
    par = [(-1 if p is None or (isinstance(p, (int, np.integer)) and p < 0) else k[p] if isinstance(p, str) else int(p))
           for p in skel["parent"]]
    for key in ("pos", "jpos", "rest_world"):
        if key in skel:
            return names, np.asarray(skel[key], float), par
    t = np.asarray(skel["rest_t"], float)
    W = np.zeros_like(t)
    done = [False] * len(names)

    def world(i):
        if not done[i]:
            W[i] = t[i] + (world(par[i]) if par[i] >= 0 else 0.0)
            done[i] = True
        return W[i]
    for i in range(len(names)):
        world(i)
    return names, W, par


def wing_weight_repair(P: np.ndarray, J: np.ndarray, W: np.ndarray, skel: dict, plan: dict
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """When the skin gives the folded wing bones under 3% of the lateral body-side vertices (top bone), reassign the
    lateral vertices (|x-cx| > 0.7 x the local half-width, y between the leg tops and 0.2H above wing_1, z along the
    wings) to the nearest wing bone of their side, blended linearly into their old weights over 0.05H from the
    region's edge. Returns (J, W, info); J/W are returned unchanged when nothing is applied."""
    info = {"applied": False}
    if not (plan.get("caps") or {}).get("wings"):
        return J, W, {**info, "reason": "the plan has no wings"}
    P = np.asarray(P, float)
    names, jpos, par = _skel_arrays(skel)
    wing = [i for i, n in enumerate(names) if n.startswith("wing_")]
    if not wing:
        return J, W, {**info, "reason": "the skeleton has no wing bones"}
    H = float(skel.get("H") or np.ptp(P[:, 1]))
    pairs = [(i, names.index(mirror_name(n))) for i, n in enumerate(names) if _split_side(n)[1] == "L" and mirror_name(n) in names]
    cx = float(skel["cx"]) if skel.get("cx") is not None else (
        float(np.mean([(jpos[a][0] + jpos[b][0]) / 2 for a, b in pairs])) if pairs else float(np.median(P[:, 0])))
    legtop = [jpos[i][1] for i, n in enumerate(names) if n.startswith("UpperLeg")]
    y_lo = float(np.mean(legtop)) if legtop else float(P[:, 1].min() + 0.3 * H)
    w1 = [jpos[i] for i, n in enumerate(names) if _split_side(n)[0] == "wing_1"]
    y_hi = float(max(p[1] for p in w1)) + 0.2 * H if w1 else float(P[:, 1].max())
    wz = jpos[wing][:, 2]
    z_lo, z_hi = float(wz.min()) - 0.05 * H, float(wz.max()) + 0.05 * H
    ax = np.abs(P[:, 0] - cx)
    tr = cKDTree(P[:, 1:])
    hw = np.array([float(ax[nb].max()) if nb else 0.0 for nb in tr.query_ball_point(P[:, 1:], 0.05 * H)])
    margin = np.minimum.reduce([ax - 0.7 * hw, P[:, 1] - y_lo, y_hi - P[:, 1], P[:, 2] - z_lo, z_hi - P[:, 2]])
    lat = (margin > 0) & (hw > 0)
    n_lat = int(lat.sum())
    if n_lat == 0:
        return J, W, {**info, "reason": "no lateral body-side vertices", "verts": 0}
    top = rig_metrics.top_bone(J, W)
    wset = set(wing)
    share = float(np.mean([t in wset for t in top[lat]]))
    info.update(share_before=round(share, 4), verts=n_lat, cx=round(cx, 5))
    if share >= 0.03:
        return J, W, {**info, "reason": f"wing bones already own {share:.1%} of the lateral vertices"}
    segs = [(par[c], c) for c in range(len(names)) if par[c] >= 0]
    idx = np.nonzero(lat)[0]
    D = rig_metrics.seg_dist(P[idx], jpos, segs)
    side_v = np.sign(P[idx, 0] - cx)
    for b in range(len(names)):
        if b not in wset:
            D[:, b] = np.inf
        else:
            sb = np.sign(jpos[b][0] - cx)
            D[sb != side_v, b] = np.inf
    best = D.argmin(1)
    okb = np.isfinite(D[np.arange(len(idx)), best])
    a = np.clip(margin[idx] / (0.05 * H), 0, 1) * okb
    M = rig.to_dense(J, W, len(names))
    one = np.zeros((len(idx), len(names)))
    one[np.arange(len(idx)), best] = 1.0
    M[idx] = (1 - a[:, None]) * M[idx] + a[:, None] * one
    J2, W2 = rig.to_top4(M)
    top2 = rig_metrics.top_bone(J2, W2)
    share2 = float(np.mean([t in wset for t in top2[lat]]))
    from collections import Counter
    cnt = Counter(names[b] for b, w in zip(best, a) if w > 0)
    info.update(applied=True, share_after=round(share2, 4), reassigned=int((a > 0).sum()),
                bones=dict(cnt.most_common()), reason="wing bones owned under 3% of the lateral vertices")
    return J2, W2, info


def _wing_side_margin(P: np.ndarray, names: list[str], jpos: np.ndarray, cx: float, H: float) -> np.ndarray:
    """Signed distance (world units, > 0 inside) of each point into the folded wings' box on the body sides: lateral
    (|x-cx| over 0.7 x the local half-width within 0.05H in y-z), above the leg tops, at most 0.2H above wing_1, and
    in z within 0.05H of the wing chain (wing_1 .. wing_tip; the feathers hang behind and below it and are left out).
    The rump, the vent and the belly between the legs are outside it."""
    ax = np.abs(P[:, 0] - cx)
    tr = cKDTree(P[:, 1:])
    hw = np.array([float(ax[nb].max()) if nb else 0.0 for nb in tr.query_ball_point(P[:, 1:], 0.05 * H)])
    legtop = [jpos[i][1] for i, n in enumerate(names) if n.startswith("UpperLeg")]
    y_lo = float(np.mean(legtop)) if legtop else float(P[:, 1].min() + 0.3 * H)
    w1 = [jpos[i] for i, n in enumerate(names) if _split_side(n)[0] == "wing_1"]
    y_hi = float(max(p[1] for p in w1)) + 0.2 * H if w1 else float(P[:, 1].max())
    chain = [i for i, n in enumerate(names) if n.startswith("wing_") and not n.startswith("wing_feather")]
    wz = jpos[chain][:, 2] if chain else jpos[[i for i, n in enumerate(names) if n.startswith("wing_")]][:, 2]
    z_lo, z_hi = float(wz.min()) - 0.05 * H, float(wz.max()) + 0.05 * H
    m = np.minimum.reduce([ax - 0.7 * hw, P[:, 1] - y_lo, y_hi - P[:, 1], P[:, 2] - z_lo, z_hi - P[:, 2]])
    m[hw <= 0] = -np.inf
    return m


def wing_region_repair(P: np.ndarray, J: np.ndarray, W: np.ndarray, skel: dict, plan: dict, min_share: float = 0.8,
                       far_H: float = 0.1, ramp_H: float = 0.05) -> tuple[np.ndarray, np.ndarray, dict]:
    """The `parts` view of the folded wings: the wing region is the points whose nearest bone (rig_metrics.seg_dist)
    is a wing bone. When that region puts under `min_share` of its weight on the wing bones (the drawn chicken: 18%,
    the rest on the body and tail), each region point is blended onto the nearest wing bone of its own side, by
    (distance to the nearest other bone - distance to the wing) / ramp_H, clipped to [0, 1], so the region's edge stays
    as skinned. Wing weight on points well away from every wing bone (over far_H, and nearer another bone: chest
    points the skin gave to wing_1) moves to that point's nearest other bone. The region is also held to the wings' box on
    the body sides (_wing_side_margin): the feather bones hang behind and below the wing, so the rump, vent and belly
    points between the legs are nearest a feather on a round body (the drawn chicken: 221 of 665 region points) and a
    wing stroke dragged them off the tail and hips (Flap p99 stretch 2.9x on tail_1 / wing_feather_4_1 edges). Wing
    weight on points outside the box moves to their nearest other bone, ramped over ramp_H past the box edge, and
    the points inside it blend onto the wing ramped the same way in from the edge. Returns (J, W, info); J/W are the
    same objects when nothing is applied."""
    info = {"applied": False}
    if not (plan.get("caps") or {}).get("wings"):
        return J, W, {**info, "reason": "the plan has no wings"}
    P = np.asarray(P, float)
    names, jpos, par = _skel_arrays(skel)
    wing = [i for i, n in enumerate(names) if n.startswith("wing_")]
    if not wing:
        return J, W, {**info, "reason": "the skeleton has no wing bones"}
    H = float(skel.get("H") or np.ptp(P[:, 1]))
    pairs = [(i, names.index(mirror_name(n))) for i, n in enumerate(names) if _split_side(n)[1] == "L" and mirror_name(n) in names]
    cx = float(skel["cx"]) if skel.get("cx") is not None else (
        float(np.mean([(jpos[a][0] + jpos[b][0]) / 2 for a, b in pairs])) if pairs else float(np.median(P[:, 0])))
    segs = [(par[c], c) for c in range(len(names)) if par[c] >= 0]
    D = rig_metrics.seg_dist(P, jpos, segs)
    other = [i for i in range(len(names)) if i not in set(wing)]
    dw, dn = D[:, wing].min(1), D[:, other].min(1)
    M = rig.to_dense(J, W, len(names))
    wmask = np.zeros(len(names), bool)
    wmask[wing] = True
    margin = _wing_side_margin(P, names, jpos, cx, H)
    inbox = np.clip(margin / (ramp_H * H), 0.0, 1.0)
    outbox = np.clip(-margin / (ramp_H * H), 0.0, 1.0)
    reg = (dw < dn) & (margin > 0)
    share = float(M[reg][:, wmask].sum() / max(M[reg].sum(), 1e-12)) if reg.any() else 1.0
    far = (dw > far_H * H) & (dn < dw) & (M[:, wmask].sum(1) > 0)
    off = (outbox > 0) & ~far & (M[:, wmask].sum(1) > 0)
    leak = float(M[far][:, wmask].sum()) + float((M[off][:, wmask].sum(1) * outbox[off]).sum())
    info.update(share_before=round(share, 4), region=int(reg.sum()), leak_mass=round(leak, 3), cx=round(cx, 5),
                off_side=int(off.sum()))
    if share >= min_share and leak < 1e-3 * len(P):
        return J, W, {**info, "reason": f"the wing region already puts {share:.0%} of its weight on the wings"}
    from collections import Counter
    cnt: Counter = Counter()
    if share < min_share:
        side_w = np.array([np.sign(jpos[i][0] - cx) for i in wing])
        sv = np.sign(P[:, 0] - cx)
        for v in np.nonzero(reg)[0]:
            cand = [w for w, s in zip(wing, side_w) if s == sv[v]] or wing
            b = cand[int(np.argmin(D[v, cand]))]
            a = float(np.clip((dn[v] - dw[v]) / (ramp_H * H), 0.0, 1.0)) * float(inbox[v])
            if a <= 0:
                continue
            M[v] *= 1 - a
            M[v, b] += a
            cnt[names[b]] += 1
    near_other = np.array(other)[D[:, other].argmin(1)]
    for v in np.nonzero(far)[0]:
        s = M[v, wmask].sum()
        M[v, wmask] = 0.0
        M[v, near_other[v]] += s
    for v in np.nonzero(off)[0]:  # off the body side (rump, vent, belly): the wing weight ramps onto the body
        s = M[v, wmask].sum() * outbox[v]
        M[v, wmask] *= 1.0 - outbox[v]
        M[v, near_other[v]] += s
    J2, W2 = rig.to_top4(M)
    M2 = rig.to_dense(J2, W2, len(names))
    share2 = float(M2[reg][:, wmask].sum() / max(M2[reg].sum(), 1e-12)) if reg.any() else 1.0
    info.update(applied=True, share_after=round(share2, 4), reassigned=int(sum(cnt.values())), stripped=int(far.sum()) + int(off.sum()),
                bones=dict(cnt.most_common()), reason=(f"the wing region put {share:.0%} of its weight on the wings"
                                                       if share < min_share else "wing weight on points far from the wings"))
    return J2, W2, info


def wing_weight_smooth(P: np.ndarray, faces, J: np.ndarray, W: np.ndarray, skel: dict, plan: dict,
                       iters: int | None = None, rings: int = 2) -> tuple[np.ndarray, np.ndarray, dict]:
    """Soften the folded wings' weight edge. The wing region's weights end in one ring of edges on the flank, so a wing
    stroke tears that ring (Flap p99 stretch 3.1x at 70 deg elevation on the drawn chicken). `iters` rounds of
    neighbour averaging (half old, half the mean of the mesh neighbours, coincident points welded) on the points with
    any wing weight plus `rings` rings around them spread the edge over a band (p99 1.9x on the same stroke), except
    the points nearest a wing or neck bone, which keep their weights (the parts shares). Weight sums are kept; the result is cut back to the top 4. iters defaults to the plan's chains.wings.smooth_iters (0: off).
    Returns (J, W, info); J/W are the same objects when nothing is applied."""
    info = {"applied": False}
    if not (plan.get("caps") or {}).get("wings"):
        return J, W, {**info, "reason": "the plan has no wings"}
    wspec = (plan.get("chains") or {}).get("wings") or {}
    iters = int(wspec.get("smooth_iters", 0) if iters is None else iters)
    if iters <= 0 or faces is None:
        return J, W, {**info, "reason": "smoothing off"}
    import scipy.sparse as sp
    P = np.asarray(P, float)
    Fc = np.asarray(faces).reshape(-1, 3).astype(int)
    names, jpos, par = _skel_arrays(skel)
    wing = [i for i, n in enumerate(names) if n.startswith("wing_")]
    if not wing or not len(Fc):
        return J, W, {**info, "reason": "no wing bones or no faces"}
    # the wing region (points nearest a wing bone) and the neck's keep their weights, so the parts row's shares
    # (wings >= 50%, neck >= 95%) stand: the edge softens outward, onto the flank around the wing
    segs = [(par[c], c) for c in range(len(names)) if par[c] >= 0]
    nearest = rig_metrics.seg_dist(P, jpos, segs).argmin(1) if segs else np.zeros(len(P), int)
    import rig_fit
    neck = set(((rig_fit.CHAINS.get(plan.get("class") or "avian") or {}).get("neck")) or [])
    keep_b = np.array([n.startswith("wing_") or n in neck for n in names])
    frozen_v = keep_b[nearest]
    H = float(skel.get("H") or np.ptp(P[:, 1]))
    key = np.round(P / (1e-6 * max(H, 1e-9))).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n = int(inv.max()) + 1
    r = inv[np.concatenate([Fc[:, 0], Fc[:, 1], Fc[:, 2], Fc[:, 1], Fc[:, 2], Fc[:, 0]])]
    c = inv[np.concatenate([Fc[:, 1], Fc[:, 2], Fc[:, 0], Fc[:, 0], Fc[:, 1], Fc[:, 2]])]
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    M = rig.to_dense(J, W, len(names))
    Mw = np.zeros((n, M.shape[1]))
    np.add.at(Mw, inv, M)
    Mw /= np.bincount(inv, minlength=n)[:, None]
    band = Mw[:, wing].sum(1) > 0.02
    for _ in range(int(rings)):
        band = band | (A @ band.astype(float) > 0)
    frozen = np.zeros(n, bool)
    np.logical_or.at(frozen, inv, frozen_v)
    band &= ~frozen
    for _ in range(iters):
        Ms = (A @ Mw) / deg[:, None]
        Mw[band] = 0.5 * Mw[band] + 0.5 * Ms[band]
    J2, W2 = rig.to_top4(Mw[inv])
    info.update(applied=True, iters=iters, rings=int(rings), verts=int(band[inv].sum()))
    return J2, W2, info


def sole_weight_repair(P: np.ndarray, J: np.ndarray, W: np.ndarray, skel: dict, plan: dict | None = None,
                       full_H: float = 0.02, none_H: float = 0.04) -> tuple[np.ndarray, np.ndarray, dict]:
    """Points on the floor belong to the foot. Weight a sole point (under none_H above the floor, fully under full_H)
    carries on the leg bones above the foot (the femur, knee or tarsus) moves to that side's foot bone. A chicken's
    hind toe lies on the floor behind the foot, and the skin bound it to the tarsus: in a stride the tarsus swings and
    takes it 0.04H through the floor, so the floor guard lifts the planted foot (Run stance clearance 0.05-0.07H).
    The plan's leg_ik names the bones (femur/A/B above K = the foot). Returns (J, W, info); same J/W objects when
    nothing moves."""
    P = np.asarray(P, float)
    plan = plan or {}
    names, jpos, par = _skel_arrays(skel)
    k = {n: i for i, n in enumerate(names)}
    lk = plan.get("leg_ik") or {}
    keys = [lk.get(a, d) for a, d in (("femur", "UpperLeg"), ("A", "LowerLeg"), ("B", "AnkleLeg"))]
    foot_key = lk.get("K", "Foot")
    sfx = tuple(plan.get("side_suffix") or ("_L", "_R"))
    move = {}
    for s in sfx:
        if foot_key + s not in k:
            continue
        for b in keys:
            if b + s in k:
                move[k[b + s]] = k[foot_key + s]
    info = {"applied": False}
    if not move:
        return J, W, {**info, "reason": "the skeleton has no leg chain above a foot"}
    H = float(skel.get("H") or np.ptp(P[:, 1]))
    floor = float(P[:, 1].min())
    h = P[:, 1] - floor
    a = np.clip((none_H * H - h) / max((none_H - full_H) * H, 1e-12), 0.0, 1.0)
    Ji = np.asarray(J).astype(int)
    W = np.asarray(W)
    src = np.isin(Ji, list(move)) & (W > 0) & (a[:, None] > 0)
    if not src.any():
        return J, W, {**info, "reason": "the floor points are already on the feet", "moved_mass": 0.0}
    M = rig.to_dense(Ji, W, len(names)).astype(float)
    from collections import Counter
    cnt: Counter = Counter()
    moved = 0.0
    for v in np.nonzero(src.any(1))[0]:
        for b, f in move.items():
            w = M[v, b] * a[v]
            if w > 0:
                M[v, b] -= w
                M[v, f] += w
                moved += float(w)
                cnt[names[b]] += 1
    J2, W2 = rig.to_top4(M)
    info.update(applied=True, reason="floor points moved onto the feet", verts=int(src.any(1).sum()),
                moved_mass=round(moved, 4), bones=dict(cnt.most_common()))
    return J2, W2, info


def side_weight_repair(P: np.ndarray, J: np.ndarray, W: np.ndarray, skel: dict, plan: dict | None = None,
                       mid_H: float = 0.01, margin_H: float = 0.02) -> tuple[np.ndarray, np.ndarray, dict]:
    """Move weight a vertex carries on the OTHER side's sided bones (Toes_R on a vertex of the left foot) onto the
    mirrored bone of its own side. A vertex counts as one side's when it is off the midline (|x-cx| > mid_H) on that
    side and that side's sided bones are nearer to it than the other side's by more than margin_H. Skin-tokens gives
    a staggered stance's inner toes a share of the far foot (0.35 Toes_R on the near foot's inner toe on the drawn
    chicken), so the far foot's toe curl drags the planted foot's skin through the floor. Returns (J, W, info); J/W are
    the same objects when nothing moves."""
    P = np.asarray(P, float)
    sfx = tuple((plan or {}).get("side_suffix") or ("_L", "_R"))
    names, jpos, par = _skel_arrays(skel)
    info = {"applied": False}
    k = {n: i for i, n in enumerate(names)}
    side = np.array([1 if n.endswith(sfx[0]) else -1 if n.endswith(sfx[1]) else 0 for n in names])
    mirror = {i: k.get(n[: -len(sfx[0])] + sfx[1]) if side[i] == 1 else k.get(n[: -len(sfx[1])] + sfx[0])
              for i, n in enumerate(names) if side[i] != 0}
    mirror = {i: m for i, m in mirror.items() if m is not None}
    if not mirror:
        return J, W, {**info, "reason": "the skeleton has no sided bone pairs"}
    H = float(skel.get("H") or np.ptp(P[:, 1]))
    pairs = [(i, m) for i, m in mirror.items() if side[i] == 1]
    cx = float(skel["cx"]) if skel.get("cx") is not None else float(np.mean([(jpos[a][0] + jpos[b][0]) / 2 for a, b in pairs]))
    J = np.asarray(J)
    W = np.asarray(W)
    Ji = J.astype(int)
    on_side = side[Ji] * (W > 0)
    cand = np.nonzero((on_side == 1).any(1) & (on_side == -1).any(1)
                      | ((on_side == 1).any(1) & (P[:, 0] < cx)) | ((on_side == -1).any(1) & (P[:, 0] > cx)))[0]
    if len(cand) == 0:
        return J, W, {**info, "reason": "no vertex carries weight on the wrong side", "moved_mass": 0.0}
    segs = [(par[c], c) for c in range(len(names)) if par[c] >= 0]
    D = rig_metrics.seg_dist(P[cand], jpos, segs)
    dL = np.where(side[None] == 1, D, np.inf).min(1)
    dR = np.where(side[None] == -1, D, np.inf).min(1)
    off = P[cand, 0] - cx
    own = np.zeros(len(cand), int)
    own[(off > mid_H * H) & (dL + margin_H * H < dR)] = 1
    own[(off < -mid_H * H) & (dR + margin_H * H < dL)] = -1
    J2 = Ji.copy()
    moved = 0.0
    from collections import Counter
    cnt: Counter = Counter()
    for r, v in enumerate(cand):
        if own[r] == 0:
            continue
        for c in range(J.shape[1]):
            b = Ji[v, c]
            if W[v, c] > 0 and side[b] == -own[r] and b in mirror:
                J2[v, c] = mirror[b]
                moved += float(W[v, c])
                cnt[names[b]] += 1
    if moved <= 0:
        return J, W, {**info, "reason": "the wrong-side weights sit on the midline or between the sides",
                      "candidates": int(len(cand)), "moved_mass": 0.0}
    J3, W3 = rig.to_top4(rig.to_dense(J2, W, len(names)))
    info.update(applied=True, reason="moved weight off the other side's bones", candidates=int(len(cand)),
                verts=int(sum(1 for r in range(len(cand)) if own[r] != 0 and any(
                    W[cand[r], c] > 0 and side[Ji[cand[r], c]] == -own[r] for c in range(J.shape[1])))),
                moved_mass=round(moved, 4), bones=dict(cnt.most_common()), cx=round(cx, 5))
    return J3, W3, info


# ---------------------------------------------------------------- CLI

def _predict(source: Path, work: Path, mode: str | None, plan: dict | None = None):
    """Today's rig_pred.predict, called unchanged when it takes no mode. Once predict grows `mode` (or the design's
    `beams` / `temperature`), the mode is passed through: beam -> beams=plan.pred.beams (default 4), sampled ->
    temperature=plan.pred.temperature (default 0.7)."""
    params = _inspect.signature(rig_pred.predict).parameters
    pp = (plan or {}).get("pred") or {}
    kw = {}
    if mode and "mode" in params:
        kw["mode"] = mode
    elif mode == "beam" and "beams" in params:
        kw["beams"] = int(pp.get("beams") or 4) if int(pp.get("beams") or 1) > 1 else 4
    elif mode == "sampled" and "temperature" in params:
        kw["temperature"] = float(pp.get("temperature", 0.7))
    return rig_pred.predict(source, work, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stem", required=True)
    ap.add_argument("--plan", default="ground_bird")
    ap.add_argument("--pred-mode", choices=["sampled", "beam"], default=None)
    ap.add_argument("--source", default="psx.glb", help="the mesh, relative to the asset (default psx.glb)")
    ap.add_argument("--out", default=None, help="output dir (default out/<stem>/rig/template)")
    ap.add_argument("--no-pred", action="store_true", help="build without the skin-tokens prediction (fallbacks)")
    a = ap.parse_args(argv)
    asset = rig.OUT / a.stem
    source = asset / a.source
    if not source.is_file():
        print(f"rig_build: no {source}", file=sys.stderr)
        return 1
    plan = rig_plan.load(a.plan)
    out = Path(a.out) if a.out else asset / "rig" / "template"
    pred = None
    if not a.no_pred:
        work = asset / "rig"
        work.mkdir(parents=True, exist_ok=True)
        glb, secs, cached = _predict(source, work, a.pred_mode, plan)
        pred = rig_pred.Pred(glb)
        print(f"pred: {len(pred.names)} joints ({'cached' if cached else f'{secs:.0f}s'})")
    try:
        info = build(source, plan, pred, out, pred_mode=a.pred_mode)
    except BuildError as e:
        print(f"rig_build: couldn't build a skeleton ({e.code}: {e.msg})")
        print(f"wrote {out / 'build.json'}")
        return 2
    c = info["checks"]
    print(f"built {info['joints']} joints for {a.stem} ({plan['id']}); legs from {info['legs_source']}")
    print(f"  inside {c['inside_share']:.0%}, mirror {c['mirror_err_H']:.4f}H, gate {c['gate']['level']}, "
          f"lengths/H " + ", ".join(f"{k} {v:.2f}" for k, v in c["lengths_H"].items()))
    if info.get("cd_j2b"):
        print(f"  cd_j2b pred->built mean {info['cd_j2b']['pred_to_built_H']['mean']:.3f}H, "
              f"built->pred mean {info['cd_j2b']['built_to_pred_H']['mean']:.3f}H")
    print(f"wrote {info['skeleton_json']} and {info['build_json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
