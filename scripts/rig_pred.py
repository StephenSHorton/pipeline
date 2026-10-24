#!/usr/bin/env python3
"""A learned skeleton as the fitting target for the template (used by rig.py's posed_template when RIG_PRED is on).

1. Predict: skin-tokens' own `rig` mode (vendor/skin-tokens.cpp) predicts a skeleton + weights for the mesh
   (bones named bone_N, no clips). Cached in rig/pred_rig.glb by the source's size and time.
2. Label: its limb chains (between branch joints) are labelled leg / head / tail from geometry; which end is the
   head comes from the template facing rig_fit already chose (the chain that reaches farthest that way), and an
   independent label (a compact red patch, else the top of the mesh) is kept to cross-check it.
3. Register (template frame): hips onto the predicted root; the template's trunk+neck+head, tail and each leg are
   laid along the matching predicted chain by arc length (length clamped 0.7-1.6x). Head and tail lines run on to
   the mesh surface; a leg line runs on down to the floor (onto the floor cluster) when its tip stops > 0.03H up.
   Each leg's pivot joint (the horse's pelvis, scapula) then goes back over its leg, as far from it as the
   template's, scaled like the leg (place_pivots): left with the hips it sat mid-body.
   Template chains with no predicted counterpart (a chicken's wings) are folded onto the body sides and flagged
   unsupported; their clips are not shipped.
4. Binder "pred": the predicted weights moved onto the template bones (each influence goes to the template bone
   of the same chain nearest its point on the predicted bone), instead of running skin-tokens' `skin` again.
5. Floor pass on the baked clips: the body is grounded the way the template is (when the template touches its
   floor, ours touches ours; otherwise it keeps the template's clearance, scaled); each leg is bent / stretched
   (two-bone IK, foot kept level) so its foot keeps the template foot's lift; a head that still goes through the
   floor is lifted at the neck.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import rig
import rig_fit

LEN_CLAMP = (0.7, 1.6)
LEG_CLAMP = (0.35, 1.6)  # a chicken's legs are 0.41x the bird template's (after proportioning): 0.7 put the feet under the floor

# Template paths per class (template joint names, root first). "body": hips -> spine -> neck -> head tip;
# "tail": hips -> tail tip; legs from the hip/shoulder joint down to the floor joint (the key; its toes ride along).
PATHS = {
    "avian": {
        "body": ["hips", "spine_0", "spine_1", "spine_2", "spine_3", "head", "mouth_upper", "mouth_upper_tip"],
        "tail": ["hips", "tail_1", "tail_2", "tail_3", "tail_tip"],
        "legs": [["UpperLeg_L", "LowerLeg_L", "AnkleLeg_L", "Foot_L"], ["UpperLeg_R", "LowerLeg_R", "AnkleLeg_R", "Foot_R"]],
        "groups": {"wings": ["wing_1_L", "wing_1_R"]},
        "needs": {"wings": ["Flap", "Glide"]},  # clips that need a chain: not shipped when it is unsupported
        "neck_lift": "spine_2",
    },
    "quadruped": {
        "body": ["hips", "spine_1", "spine_2", "spine_3", "spine_4", "spine_5", "spine_6", "head", "nose", "head_leaf"],
        "tail": ["hips", "tail_1", "tail_2", "tail_3", "tail_4", "tail_leaf"],
        "legs": [["front_humerus_l", "front_leg_upper_l", "front_leg_lower_l", "front_leg_ankle_l", "front_leg_foot_l"],
                 ["front_humerus_r", "front_leg_upper_r", "front_leg_lower_r", "front_leg_ankle_r", "front_leg_foot_r"],
                 ["back_leg_upper_l", "back_leg_lower_l", "back_leg_ankle_l", "back_leg_foot_l"],
                 ["back_leg_upper_r", "back_leg_lower_r", "back_leg_ankle_r", "back_leg_foot_r"]],
        "groups": {},
        "needs": {},
        "neck_lift": "spine_5",
    },
}
ANCHOR_CLAMP = (0.25, 4.0)  # lay_runs: a run between two predicted anchors is laid to reach them
ANCHOR_GAP = 0.03  # x the body line's length: an anchor must sit this far past the one before it
LYING = {"Sleep", "Death", "lay_to_idle"}


# ---------------------------------------------------------------- predict

def _mode(mode: str | None, beams: int | None, temperature: float | None) -> tuple[str, list[str]]:
    """Cache tag and extra CLI flags for a prediction mode. The default (greedy) mode has tag '' and no flags."""
    if mode == "beam" and not beams:
        beams = 4
    if mode == "sampled" and temperature is None:
        temperature = 0.7
    flags, tag = [], []
    if beams and int(beams) > 1:
        flags += ["--beams", str(int(beams))]
        tag.append(f"beam{int(beams)}")
    if temperature is not None and float(temperature) > 0:
        flags += ["--temperature", f"{float(temperature):g}"]
        tag.append(f"t{float(temperature):g}")
    return "-".join(tag), flags


def predict(source: Path, work: Path, beams: int | None = None, temperature: float | None = None,
            mode: str | None = None) -> tuple[Path, float, bool]:
    """skin-tokens `rig` on the source mesh -> rig/pred_rig.glb (cached by the source's content).

    beams / temperature (or mode 'beam' / 'sampled') ask for a beam-searched or sampled prediction; those land in
    rig/pred_rig.<mode>.glb with the mode in the cache key, so the default prediction and its cache are unchanged."""
    tag, flags = _mode(mode, beams, temperature)
    stem = f"pred_rig.{tag}" if tag else "pred_rig"
    out = work / f"{stem}.glb"
    key_f = work / f"{stem}.json"
    import hashlib
    key = hashlib.md5(source.read_bytes()).hexdigest()  # a straightened copy is rewritten every run: key on content
    if tag:
        key += ":" + tag
    if out.is_file() and key_f.is_file():
        got = json.loads(key_f.read_text())
        if got.get("key") == key:
            return out, float(got.get("seconds", 0)), True
    cli = rig.SKIN / "build" / "release" / "bin" / "skintokens-cli"
    model = rig.SKIN / "models" / "SkinTokens-GGUF" / "F16"
    t = time.time()
    proc = subprocess.run([str(cli), "rig", str(model), str(source), str(out), "--device", "auto"] + flags,
                          capture_output=True, text=True)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError("skin-tokens rig failed: " + (proc.stderr or proc.stdout)[-600:])
    secs = time.time() - t
    key_f.write_text(json.dumps({"key": key, "seconds": round(secs, 1), **({"mode": tag} if tag else {})}))
    return out, secs, False


def paths_for(cls: str, plan: dict | None = None) -> dict | None:
    """The template paths for a class: the body plan's (rig_plan.spec) when given, else PATHS[cls]."""
    if plan:
        import rig_plan
        got = rig_plan.spec(plan, cls, "paths", None)
        if got:
            return got
    return PATHS.get(cls)


# ---------------------------------------------------------------- the predicted skeleton

class Pred:
    def __init__(self, glb: Path) -> None:
        self.doc, self.b = rig.read_glb(glb)
        pr = self.doc["meshes"][0]["primitives"][0]
        a = pr["attributes"]
        self.P = rig.accessor(self.doc, self.b, a["POSITION"]).astype(float)
        self.J = rig.accessor(self.doc, self.b, a["JOINTS_0"]).astype(int)
        W = rig.accessor(self.doc, self.b, a["WEIGHTS_0"]).astype(float)
        if W.max() > 1.5:
            W = W / 255.0
        self.W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
        sk = rig.Skeleton(self.doc)
        self.joints = self.doc["skins"][0]["joints"]
        k_of = {j: k for k, j in enumerate(self.joints)}
        self.names = [self.doc["nodes"][j].get("name", f"n{j}") for j in self.joints]
        w = sk.world({})
        self.jp = np.array([w[j][0] for j in self.joints])
        self.par = np.array([k_of.get(sk.parent.get(j), -1) for j in self.joints])
        self.kids = {k: [c for c in range(len(self.joints)) if self.par[c] == k] for k in range(len(self.joints))}
        self.root = [k for k in range(len(self.joints)) if self.par[k] < 0]
        self.lo, self.hi = self.P.min(0), self.P.max(0)
        self.H = float(self.hi[1] - self.lo[1])

    def subtree(self, k: int) -> list[int]:
        out, st = [], [k]
        while st:
            x = st.pop()
            out.append(x)
            st += self.kids[x]
        return out

    def path_to(self, k: int) -> list[int]:
        """Joints from the root down to k (inclusive)."""
        out = [k]
        while self.par[out[-1]] >= 0:
            out.append(int(self.par[out[-1]]))
        return out[::-1]

    def minor(self, c: int) -> bool:
        """A branch is minor (ear, toe, jaw) when its whole subtree stays within 0.15H of its branch joint."""
        return float(np.linalg.norm(self.jp[self.subtree(c)] - self.jp[self.par[c]], axis=1).max()) < 0.15 * self.H

    def chains(self) -> list[list[int]]:
        """Limb chains: from each non-minor child of a branch joint (>= 2 non-minor children) to a leaf, following the
        farthest-reaching child; chains that end at the next branch joint are trunk."""
        def major(k: int) -> bool:
            return sum(not self.minor(c) for c in self.kids[k]) >= 2
        starts = []
        for r in self.root:
            starts += [c for c in self.kids[r] if not self.minor(c)] if major(r) else [r]
        out, seen = [], set()
        while starts:
            c = starts.pop(0)
            if c in seen:
                continue
            ch = [c]
            while True:
                x = ch[-1]
                seen.add(x)
                if major(x):
                    starts += list(self.kids[x])
                    ch = None
                    break
                main = [y for y in self.kids[x] if not self.minor(y)]
                rest = [y for y in self.kids[x] if self.minor(y)]
                if not main and rest:
                    main = [max(rest, key=lambda y: float(np.linalg.norm(self.jp[self.subtree(y)] - self.jp[x], axis=1).max()))]
                if not main:
                    break
                ch.append(main[0])
            if ch is not None:
                out.append(ch)
        return out


def _red(doc: dict, b: bytes) -> np.ndarray | None:
    c = rig_fit._texel_colours(doc, b)
    if c is None:
        return None
    return (c[:, 0] > 150) & (c[:, 1] < 90) & (c[:, 2] < 90)


def label(p: Pred, forward: np.ndarray | None, red: np.ndarray | None = None) -> dict:
    """Chains labelled leg / head / tail / other. forward: unit vector the head is toward (from the template facing);
    None: an independent guess (a compact red patch's side, else the top 10% of the mesh). Returns the labels and
    the facing they imply."""
    H, lo = p.H, p.lo[1]
    info = []
    for c in p.chains():
        anchor = p.jp[p.par[c[0]]] if p.par[c[0]] >= 0 else p.jp[c[0]]
        tip = p.jp[c[-1]]
        drop = float((max(anchor[1], p.jp[c[0]][1]) - tip[1]) / H)
        tip_gap = float((tip[1] - lo) / H)
        info.append({"joints": c, "names": [p.names[k] for k in c], "tip_gap_H": round(tip_gap, 3), "drop_H": round(drop, 3),
                     "label": "leg" if tip_gap < 0.30 and drop > 0.15 else None})
    cen = p.P.mean(0)
    how = "template facing"
    if forward is None:
        hp = None
        if red is not None and red.sum() >= 20 and red.mean() < 0.05:
            hp, how = p.P[red].mean(0), "red patch"
        if hp is None:
            hp, how = p.P[p.P[:, 1] > lo + 0.9 * H].mean(0), "top of the mesh"
        d = hp - cen
        d[1] = 0
        ax = int(np.argmax(np.abs(d[[0, 2]]))) * 2
        forward = np.zeros(3)
        forward[ax] = np.sign(d[ax]) or 1.0
    f = forward / np.linalg.norm(forward)
    # a leg is off the mid-plane (a hanging cow's tail drops to the floor down the middle)
    lat = np.cross([0.0, 1.0, 0.0], f)
    mid = float(p.jp[p.root[0]] @ lat)
    for d in info:
        if d["label"] == "leg" and abs(float(p.jp[d["joints"][0]] @ lat) - mid) < 0.04 * H and abs(float(p.jp[d["joints"][-1]] @ lat) - mid) < 0.04 * H:
            d["label"] = None
            d["mid_plane"] = True
    nonleg = [d for d in info if d["label"] is None]
    if nonleg:
        # the head chain reaches farthest toward the facing; the tail is the one reaching farthest the other way
        head = max(nonleg, key=lambda d: float(((p.jp[d["joints"]] - cen) @ f).max()))
        head["label"] = "head"
        rest = [d for d in nonleg if d is not head]
        if rest:
            t = min(rest, key=lambda d: float(((p.jp[d["joints"]] - cen) @ f).min()))
            if float(((p.jp[t["joints"]] - cen) @ f).min()) < -0.15 * float(np.ptp(p.P @ f)):
                t["label"] = "tail"
    for d in info:
        d["label"] = d["label"] or "other"
    yaw = float(np.degrees(np.arctan2(f[0], f[2]))) % 360
    return {"chains": info, "forward": f.round(3).tolist(), "yaw": round(yaw), "how": how,
            "legs": sum(d["label"] == "leg" for d in info)}


def head_chain_yaw(p: Pred, lab: dict) -> int | None:
    """The yaw (0/90/180/270) the labelled head chain points along: its tip relative to the root, in XZ."""
    head = next((d for d in lab["chains"] if d["label"] == "head"), None)
    if head is None:
        return None
    v = p.jp[head["joints"][-1]] - p.jp[p.root[0]]
    return int(round(np.degrees(np.arctan2(v[0], v[2])) / 90.0) * 90) % 360


# ---------------------------------------------------------------- target lines (world frame)

def _surface_ext(mesh, tip: np.ndarray, d: np.ndarray, H: float) -> np.ndarray | None:
    """Where a ray from `tip` along `d` leaves the mesh (within 0.4H)."""
    d = d / max(np.linalg.norm(d), 1e-12)
    try:
        locs, _, _ = mesh.ray.intersects_location([tip], [d], multiple_hits=True)
    except Exception:  # noqa: BLE001
        return None
    if len(locs) == 0:
        return None
    t = (locs - tip) @ d
    ok = t > 1e-4
    if not ok.any():
        return None
    k = int(np.argmin(np.where(ok, t, np.inf)))
    return locs[k] if t[k] < 0.4 * H else None


def target_lines(p: Pred, lab: dict, mesh, floor: float) -> dict:
    """World-frame polylines: body (root .. head tip .. surface), tail (root .. tail tip .. surface), legs (each
    leg chain .. floor). Leg lines whose tip stops > 0.03H above the floor go on down onto the floor cluster."""
    H = p.H
    root = p.root[0]
    out = {"legs": []}
    for kind in ("head", "tail"):
        d = next((c for c in lab["chains"] if c["label"] == kind), None)
        if d is None:
            continue
        ks = p.path_to(d["joints"][-1])
        line = [p.jp[k] for k in ks]
        ext = _surface_ext(mesh, line[-1], line[-1] - line[-2], H) if len(line) > 1 else None
        if ext is not None:
            line.append(ext)
        out["body" if kind == "head" else "tail"] = {"line": np.array(line), "pred": [p.names[k] for k in ks],
                                                       "surface_ext": ext is not None}
        if kind == "head" and d["joints"][0] in ks:  # where the predicted head chain (its neck) leaves the body
            out["body"]["neck_index"] = ks.index(d["joints"][0])
    legs = [c for c in lab["chains"] if c["label"] == "leg"]
    if legs:
        low = p.P[p.P[:, 1] <= floor + 0.04 * H]
        tips = np.array([p.jp[c["joints"][-1]] for c in legs])
        for c, tip in zip(legs, tips):
            line = [p.jp[k] for k in c["joints"]]
            gap = float(tip[1] - floor)
            ext = None
            if gap > 0.03 * H:
                # the floor points nearer this tip than any other tip; their centre is the foot's floor point
                if len(low):
                    own = np.linalg.norm(low[:, None, [0, 2]] - tips[None, :, [0, 2]], axis=2).argmin(1) == list(map(tuple, tips)).index(tuple(tip))
                    near = low[own & (np.linalg.norm(low[:, [0, 2]] - tip[[0, 2]], axis=1) < 0.2 * H)]
                else:
                    near = low
                xz = near[:, [0, 2]].mean(0) if len(near) >= 3 else tip[[0, 2]]
                ext = np.array([xz[0], floor, xz[1]])
                line.append(ext)
            par = int(p.par[c["joints"][0]])
            out["legs"].append({"line": np.array(line), "pred": [p.names[k] for k in c["joints"]], "tip_gap_H": round(gap / H, 3),
                                "floor_ext": ext is not None,
                                # the joint the leg hangs from (a pelvis, a shoulder): where the trunk carries it
                                **({"attach": p.jp[par]} if par >= 0 else {})})
    return out


# ---------------------------------------------------------------- registration (template frame)

def _at(poly: np.ndarray, s: float) -> np.ndarray:
    return rig._at_arclen(poly, s)


def _shift_subtree(P: dict, children: dict, name: str, delta: np.ndarray) -> None:
    for n in [name] + rig_fit._descendants(children, name):
        P[n] = P[n] + delta


def _move_chain(P: dict, children: dict, chain: list[str], new: dict) -> None:
    """rig_fit._move_chain, except the chain's first joint (hips, shared by several paths) doesn't swing its other
    branches: only joints hanging off chain[1:] follow their chain joint's bone."""
    old = {n: P[n].copy() for n in P}
    inchain = set(chain)
    for n in chain:
        if n in new:
            P[n] = new[n]
    for k, n in enumerate(chain):
        if k == 0:
            continue
        nxt = chain[k + 1] if k + 1 < len(chain) else None
        u, v = (old[nxt] - old[n], P[nxt] - P[n]) if nxt is not None else (old[n] - old[chain[k - 1]], P[n] - P[chain[k - 1]])
        Rm, _ = rig_fit._rot_between(u, v, None)
        s = np.linalg.norm(v) / max(np.linalg.norm(u), 1e-12)
        for d in rig_fit._descendants(children, n):
            if d not in inchain:
                P[d] = P[n] + s * (Rm @ (old[d] - old[n]))


def lay(P: dict, children: dict, chain: list[str], line: np.ndarray, keep_tail_rigid: bool = False,
        clamp: tuple[float, float] = LEN_CLAMP) -> dict:
    """Lay `chain` along `line` by arc length: the chain's length scaled to the line's (clamped 0.7-1.6x), each joint
    at its scaled arc length (no bend cap: the line is a skeleton that sits in the mesh). Joints off the chain follow
    their chain ancestor (rig_fit._move_chain). keep_tail_rigid: the last joint's own subtree (toes) only shifts."""
    seg = [float(np.linalg.norm(P[chain[k + 1]] - P[chain[k]])) for k in range(len(chain) - 1)]
    tl = sum(seg)
    line = np.asarray(line, float).copy()
    line[0] = P[chain[0]]
    ml = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    ratio = ml / max(tl, 1e-12)
    s = float(np.clip(ratio, *clamp))
    tail = {d: P[d] - P[chain[-1]] for d in rig_fit._descendants(children, chain[-1])} if keep_tail_rigid else {}
    new = {chain[0]: P[chain[0]].copy()}
    acc, turns = 0.0, []
    for k in range(len(chain) - 1):
        acc += seg[k] * s
        new[chain[k + 1]] = _at(line, acc)
        u = P[chain[k + 1]] - P[chain[k]]
        v = new[chain[k + 1]] - new[chain[k]]
        turns.append(float(np.degrees(np.arccos(np.clip(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12), -1, 1)))))
    _move_chain(P, children, chain, new)
    for d, off in tail.items():
        P[d] = P[chain[-1]] + off
    return {"length_ratio": round(ratio, 3), "length_clamped": bool(ratio != s), "max_turn_deg": round(max(turns, default=0.0), 1),
            "tip_residual": round(float(np.linalg.norm(P[chain[-1]] - line[-1])), 4)}


def place_pivots(P: dict, P0: dict, children: dict, spec: dict, laid: dict[str, float], tV: np.ndarray | None = None,
                 tF: np.ndarray | None = None, H: float = 1.0) -> list[dict]:
    """Put each laid leg's pivot joint (the parent of its chain root that is on no body / tail path: the horse's
    back_leg_pelvis, front_scapula) back where the template has it over that leg: chain root + the template's own
    pivot offset (P0, before the hips moved onto the predicted root), scaled like the leg. register moves the hips
    onto the predicted root (mid-body) and each leg onto its predicted leg, but the pivot in between stayed with the
    hips: the sheep's hind pivot sat half a body length ahead of the thigh, so Walk swung the thigh a third of the
    height and tore the rump from the leg. A pivot that lands outside the mesh is pulled back toward the leg until
    it is inside. Its other children (none on the horse) move with it. Distances in the report are / H."""
    parent = {k: p for p, ks in children.items() for k in ks}
    trunk = set(spec.get("body") or []) | set(spec.get("tail") or [])
    out = []
    for root, s in laid.items():
        pv = parent.get(root)
        if pv is None or pv in trunk or pv not in P0:
            continue
        off = s * (P0[pv] - P0[root])
        new, t = P[root] + off, 1.0
        if tV is not None and tF is not None:
            ts = np.linspace(1.0, 0.0, 11)
            wn = rig_fit.winding_number(tV, tF, P[root][None] + ts[:, None] * off[None], chunk=4000)
            inside = np.where(np.abs(wn) > 0.5)[0]
            t = float(ts[inside[0]]) if len(inside) else 1.0
            new = P[root] + t * off
        delta = new - P[pv]
        P[pv] = new
        leg = {root} | set(rig_fit._descendants(children, root))
        for d in rig_fit._descendants(children, pv):
            if d not in leg:
                P[d] = P[d] + delta
        out.append({"joint": pv, "over": root, "moved_H": round(float(np.linalg.norm(delta)) / H, 3),
                    "to_leg_H": round(float(np.linalg.norm(new - P[root])) / H, 3), "pulled_in": round(1.0 - t, 2)})
    return out


def over_legs(P0: dict, joint: str, spec: dict, plegs: list[dict], root: np.ndarray, hind: bool = True,
              tV: np.ndarray | None = None, tF: np.ndarray | None = None) -> np.ndarray | None:
    """Where a trunk joint goes on a four-legged rig: over the predicted hind (or front) legs as the template has it
    over its own (the template joint minus the mean of its two hind/front leg chain roots, added to the mean of the
    two hindmost/foremost predicted leg roots; template frame, +Z forward), on the predicted root's centre line. A
    target outside the mesh is pulled toward the predicted root until it is inside. None without four legs on each."""
    legs = [c for c in spec.get("legs") or [] if c and c[0] in P0]
    if joint not in P0 or len(legs) != 4 or len(plegs) != 4:
        return None
    sign = 1.0 if hind else -1.0
    pair_t = sorted(legs, key=lambda c: sign * P0[c[0]][2])[:2]
    pair_p = sorted(plegs, key=lambda q: sign * q["line"][0][2])[:2]
    off = P0[joint] - np.mean([P0[c[0]] for c in pair_t], 0)
    new = np.mean([q["line"][0] for q in pair_p], 0) + off
    new[0] = root[0]
    if tV is not None and tF is not None:
        ts = np.linspace(0.0, 1.0, 11)
        pts = new[None] + ts[:, None] * (root - new)[None]
        inside = np.where(np.abs(rig_fit.winding_number(tV, tF, pts, chunk=4000)) > 0.5)[0]
        if len(inside):
            new = pts[inside[0]]
    return new


def hips_target(P0: dict, spec: dict, plegs: list[dict], root: np.ndarray, tV: np.ndarray | None = None,
                tF: np.ndarray | None = None) -> np.ndarray | None:
    """The hips over the hind legs (over_legs). The predicted root sits mid-body: the sheep's hips landed 0.36 of
    its length ahead of the hind thighs (the horse's sit 0.08 ahead), the rump behind them one rigid block owning 45%
    of the weight."""
    return over_legs(P0, "hips", spec, plegs, root, True, tV, tF)


def shoulder_joint(spec: dict, children: dict) -> str | None:
    """The body joint the front legs hang from (the horse's spine_4, through front_scapula): the first leg chain's
    root (PATHS lists the front legs first), walked up to the body path."""
    parent = {k: p for p, ks in children.items() for k in ks}
    body = set(spec.get("body") or [])
    legs = spec.get("legs") or []
    if len(legs) != 4:
        return None
    n = legs[0][0]
    while n is not None and n not in body:
        n = parent.get(n)
    return n if n != "hips" else None


def trunk_anchors(spec: dict, children: dict, lines_t: dict, root: np.ndarray) -> dict[str, np.ndarray]:
    """Where the predicted skeleton puts the trunk's joints, template joint -> point (template frame, +Z forward):
    the hips where the two hindmost predicted legs hang, the shoulder joint where the two foremost hang, and the
    neck root (the plan's neck_lift) where the predicted head chain leaves the body, all on the root's centre line.
    Laid by arc length from a mid-body root with the horse's proportions, the sheep's hips sat 0.36 of its length
    ahead of the hind thighs and, laid from the hips, its neck root sat mid-chest (Eating's head went through the
    floor and the floor pass bent the neck 20 deg back up)."""
    plegs = lines_t.get("legs") or []
    out: dict[str, np.ndarray] = {}
    if len(plegs) != 4 or len(spec.get("legs") or []) != 4 or not all("attach" in q for q in plegs):
        return out
    by_z = sorted(plegs, key=lambda q: q["attach"][2])
    for joint, pair in (("hips", by_z[:2]), (shoulder_joint(spec, children), by_z[2:])):
        if joint:
            at = np.mean([q["attach"] for q in pair], 0)
            at[0] = root[0]
            out[joint] = at
    body = lines_t.get("body") or {}
    neck = spec.get("neck_lift")
    if neck and body.get("neck_index") is not None:
        at = np.asarray(body["line"][body["neck_index"]], float).copy()
        at[0] = root[0]
        out[neck] = at
    return out


def _arc_of(line: np.ndarray, pt: np.ndarray) -> float:
    """Arc length along the polyline of its nearest point to pt."""
    best, at, acc = np.inf, 0.0, 0.0
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        L = float(np.linalg.norm(ab))
        t = float(np.clip((pt - a) @ ab / max(L * L, 1e-12), 0, 1)) if L > 0 else 0.0
        d = float(np.linalg.norm(a + t * ab - pt))
        if d < best:
            best, at = d, acc + t * L
        acc += L
    return at


def ordered_anchors(line: np.ndarray, anchors: list[tuple[str, np.ndarray]], gap: float = ANCHOR_GAP) -> list[tuple[str, np.ndarray, float]]:
    """The anchors (in chain order) that advance along the line, each with its arc length: each must sit at least
    `gap` of the line's length past the one before (the line's start for the first) and short of its end. The first
    that doesn't ends the list: a neck anchor without its shoulder would squeeze the whole spine into one run."""
    line = np.asarray(line, float)
    total = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    out, prev = [], 0.0
    for joint, pt in anchors:
        a = _arc_of(line, np.asarray(pt, float))
        if a < prev + gap * total or a > total - gap * total:
            break
        out.append((joint, np.asarray(pt, float), a))
        prev = a
    return out


def _cut(line: np.ndarray, s0: float, s1: float, p0: np.ndarray, p1: np.ndarray | None) -> np.ndarray:
    """The polyline between arc lengths s0 and s1, starting at p0 and ending at p1 (the line's end when None)."""
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    mid = [x for x, c in zip(line, cum) if s0 < c < s1]
    return np.array([p0, *mid, *([p1] if p1 is not None else [])])


def lay_runs(P: dict, children: dict, chain: list[str], line: np.ndarray, anchors: list[tuple[str, np.ndarray]]) -> dict:
    """lay() in runs: the chain from one anchor joint to the next along the line between them (split by position
    along the line, so a neck that rises or leans back stays in its own run), the last run on to the line's end.
    The joints after each anchor are put back to the template's shape before their run (the run before swung and
    scaled them with its last bone). Anchors that don't advance along the line are dropped (ordered_anchors)."""
    line = np.asarray(line, float)
    kept = [(j, pt, a) for j, pt, a in ordered_anchors(line, anchors) if j in chain]
    runs, start, p0, s0 = [], chain[0], line[0], 0.0
    for joint, pt, a in [*kept, (None, None, np.inf)]:
        i0 = chain.index(start)
        sub = chain[i0:] if joint is None else chain[i0:chain.index(joint) + 1]
        seg = _cut(line, s0, a, p0, pt)
        if len(sub) >= 2 and len(seg) >= 2:
            rel = {d: P[d] - P[sub[-1]] for d in rig_fit._descendants(children, sub[-1])} if joint else {}
            # a run ending on an anchor reaches it (the sheep's shoulder-to-neck bone is 1.8x the horse's); the
            # open run to the head tip keeps the usual clamp
            runs.append(lay(P, children, sub, seg, clamp=ANCHOR_CLAMP if joint else LEN_CLAMP))
            for d, off in rel.items():
                P[d] = P[sub[-1]] + off
        if joint is not None:
            start, p0, s0 = joint, pt, a
    if not kept or not runs:
        return lay(P, children, chain, line)
    return {"length_ratio": runs[0]["length_ratio"], "length_ratios": [r["length_ratio"] for r in runs],
            "length_clamped": any(r["length_clamped"] for r in runs), "max_turn_deg": max(r["max_turn_deg"] for r in runs),
            "tip_residual": runs[-1]["tip_residual"], "anchors": [j for j, _, _ in kept]}


def register(cls: str, P: dict, children: dict, lines_t: dict, mesh_t: np.ndarray, tV: np.ndarray, tF: np.ndarray,
             H_t: float, plan: dict | None = None) -> dict:
    """Template joints (name -> template-frame position, changed in place) onto the predicted lines (template frame).
    plan: the body plan whose paths to use (rig_plan.spec); PATHS[cls] without one."""
    spec = paths_for(cls, plan)
    rep = {"paths": [], "unsupported": []}
    P0 = {n: p.copy() for n, p in P.items()}  # the template's own proportions, for place_pivots
    root_t = (lines_t.get("body") or lines_t.get("tail") or {}).get("line")
    lines_t = dict(lines_t)
    anchors = trunk_anchors(spec, children, lines_t, root_t[0]) if root_t is not None else {}
    hips_at = anchors.pop("hips", None)
    if hips_at is None and root_t is not None:  # no attach joints on the predicted legs: the template's offset
        hips_at = hips_target(P0, spec, lines_t.get("legs", []), root_t[0], tV, tF)
    runs: list[tuple[str, np.ndarray]] = []
    if hips_at is not None:
        # lay the body from the hips forward and the tail from them back: the predicted lines start at the root
        rep["hips_over_legs"] = {"from_root": round(float(np.linalg.norm(hips_at - root_t[0])), 4)}
        z = hips_at[2]
        runs = sorted(((j, a) for j, a in anchors.items() if j in spec["body"]), key=lambda ja: spec["body"].index(ja[0]))
        for kind, ahead in (("body", True), ("tail", False)):
            q = lines_t.get(kind)
            if q is None:
                continue
            pts = [x for x in q["line"] if (x[2] > z) == ahead and np.linalg.norm(x - hips_at) > 1e-6]
            # no predicted line on that side of the hips (a tail hanging straight down from where the hind legs hang):
            # as if there were no predicted chain (a tail: unsupported)
            lines_t[kind] = {**q, "line": np.array([hips_at, *pts])} if pts else None
        root_t = np.array([hips_at])
    if root_t is not None and "hips" in P:
        delta = root_t[0] - P["hips"]
        for n in P:
            P[n] = P[n] + delta
        if "root" in P:  # the root joint stays on the floor under the hips (its weight is folded away anyway)
            P["root"] = np.array([P["hips"][0], float(mesh_t[:, 1].min()), P["hips"][2]])
        rep["hips_moved"] = round(float(np.linalg.norm(delta)), 4)
    tree = cKDTree(mesh_t)
    for kind in ("body", "tail"):
        tgt = lines_t.get(kind)
        chain = [n for n in spec[kind] if n in P]
        if tgt is None or len(chain) < 2:
            rep["paths"].append({"kind": kind, "skipped": "no predicted chain"})
            if kind == "tail" and tgt is None:
                rep["unsupported"].append("tail")
            continue
        before = {n: P[n].copy() for n in P}
        s0 = rig_fit.outside_score(P, chain, tV, tF, tree, H_t)
        if kind == "body" and runs:
            r = lay_runs(P, children, chain, tgt["line"], runs)
            if r.get("anchors"):
                rep["trunk_anchors"] = {j: np.round(a, 4).tolist() for j, a in runs if j in r["anchors"]}
        else:
            r = lay(P, children, chain, tgt["line"])
        s1 = rig_fit.outside_score(P, chain, tV, tF, tree, H_t)
        kept = s1 <= s0 + 0.01
        if not kept:
            P.update(before)
        rep["paths"].append({"kind": kind, "pred": tgt["pred"], **r, "outside_before": round(s0, 4), "outside_after": round(s1, 4),
                             "kept": bool(kept), "surface_ext": tgt["surface_ext"]})
    # legs: match predicted legs to template legs by side (template +X is left) and front/back (template +Z)
    tlegs = [c for c in spec["legs"] if all(n in P for n in c)]
    plegs = lines_t.get("legs", [])
    cx = float(np.median(mesh_t[:, 0]))
    used = set()
    laid: dict[str, float] = {}  # leg chain root -> the length scale it was laid at
    for c in tlegs:
        side = np.sign(P[c[0]][0] - cx)
        cand = [i for i, q in enumerate(plegs) if i not in used and np.sign(q["line"][0][0] - cx) == side]
        if not cand:
            rep["paths"].append({"kind": "leg", "chain": c[0], "skipped": "no predicted leg on that side"})
            continue
        i = min(cand, key=lambda i: abs(plegs[i]["line"][0][2] - P[c[0]][2]))
        used.add(i)
        q = plegs[i]
        _shift_subtree(P, children, c[0], q["line"][0] - P[c[0]])
        # the key joint sits above the floor as much as the template's does: lift the floor point by that
        line = q["line"].copy()
        low = min(P[n][1] for n in [c[-1]] + rig_fit._descendants(children, c[-1]))
        key_y = float(mesh_t[:, 1].min()) + (P[c[-1]][1] - low)  # its toes reach the floor, not below it
        if q["floor_ext"] or line[-1, 1] < key_y:
            line[-1, 1] = key_y
        r = lay(P, children, c, line, keep_tail_rigid=True, clamp=LEG_CLAMP)
        rep["paths"].append({"kind": "leg", "chain": c[0], "pred": q["pred"], **r, "pred_tip_gap_H": q["tip_gap_H"],
                             "floor_ext": q["floor_ext"]})
        laid[c[0]] = float(np.clip(r["length_ratio"], *LEG_CLAMP))
    rep["pivots"] = place_pivots(P, P0, children, spec, laid, tV, tF, H_t)
    # chains with no predicted counterpart: folded onto the body sides (rig_fit's wings), unsupported
    for g in spec["groups"]:
        rep["unsupported"].append(g)
    if "wings" in spec["groups"]:
        wings = [c for c in rig_fit.CHAINS[cls]["wings"] if all(n in P for n in c)]
        S0 = mesh_t
        for c in wings:
            side = 1.0 if P[c[-1]][0] > cx else -1.0
            S = S0[side * (S0[:, 0] - cx) > 0]
            tr = cKDTree(S[:, 1:])
            new = {}
            for n in c:
                p_ = P[n].copy()
                d, k = tr.query(p_[1:])
                if d > 0.03 * H_t:
                    p_[1:] = S[k, 1:]
                near = S[np.linalg.norm(S[:, 1:] - p_[1:], axis=1) < 0.06 * H_t]
                if len(near):
                    p_[0] = cx + side * 0.85 * float((side * (near[:, 0] - cx)).max())
                new[n] = p_
            rig_fit._move_chain(P, children, c, new)
    return rep


def cd_j2b(A: np.ndarray, segs_B: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Each point of A to its nearest segment of B."""
    out = np.full(len(A), np.inf)
    for a, b in segs_B:
        ab = b - a
        t = np.clip(((A - a) @ ab) / max(ab @ ab, 1e-12), 0, 1) if ab @ ab > 1e-12 else np.zeros(len(A))
        out = np.minimum(out, np.linalg.norm(A - (a + t[:, None] * ab), axis=1))
    return out


# ---------------------------------------------------------------- binder "pred": predicted weights onto the template

def ours_from_posed(posed: Path) -> tuple[dict, bytes]:
    """A skeleton doc like skin-tokens' output from the posed template: the same joints, identity rotations, plain
    translations, one SkinnedMesh node, inverse binds of the rest positions."""
    pdoc, _ = rig.read_glb(posed)
    sk = rig.Skeleton(pdoc)
    w = sk.world({})
    joints = pdoc["skins"][0]["joints"]
    order = [i for i in sk.order if i in joints]
    new_of = {i: k for k, i in enumerate(order)}
    nodes = []
    for i in order:
        p = sk.parent.get(i)
        t = w[i][0] - (w[p][0] if p in joints else 0)
        nd = {"name": pdoc["nodes"][i]["name"], "translation": np.asarray(t, float).tolist()}
        kids = [new_of[c] for c in pdoc["nodes"][i].get("children", []) if c in new_of]
        if kids:
            nd["children"] = kids
        nodes.append(nd)
    nodes.append({"name": "SkinnedMesh", "mesh": 0, "skin": 0})
    roots = [new_of[i] for i in order if sk.parent.get(i) not in joints]
    g = rig.GlbWriter()
    ibm = []
    for i in order:
        M = np.eye(4)
        M[:3, 3] = w[i][0]
        ibm.append(np.linalg.inv(M).T.reshape(-1))
    acc = g.array(np.array(ibm, "<f4"), "MAT4")
    g.doc.update(nodes=nodes, scenes=[{"nodes": roots + [len(nodes) - 1]}], scene=0,
                 skins=[{"joints": list(range(len(order))), "inverseBindMatrices": acc}], meshes=[{"primitives": []}])
    body = b"".join(g.chunks)
    return g.doc, body


def transfer(p: Pred, lab: dict, P_mesh: np.ndarray, ours: dict, cls: str, unsupported: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per vertex and predicted influence: the point on the predicted bone nearest the vertex goes to the nearest
    template bone of the same chain group (leg / body+head / tail); unsupported groups get nothing."""
    sk = rig.Skeleton(ours)
    joints = ours["skins"][0]["joints"]
    names = [ours["nodes"][j]["name"] for j in joints]
    w = sk.world({})
    jp = np.array([w[j][0] for j in joints])
    kid = {k: [joints.index(c) for c in ours["nodes"][j].get("children", []) if c in joints] for k, j in enumerate(joints)}
    spec = PATHS[cls]
    idx = {n: k for k, n in enumerate(names)}

    def desc(k: int) -> list[int]:
        out, st = [], [k]
        while st:
            x = st.pop()
            out.append(x)
            st += kid[x]
        return out
    group_t = {}
    banned = set()
    for g, roots in spec["groups"].items():
        if g in unsupported:
            for r_ in roots:
                if r_ in idx:
                    banned |= set(desc(idx[r_]))
    for k in range(len(names)):
        group_t[k] = "body"
    if "tail_1" in idx:
        for k in desc(idx["tail_1"]):
            group_t[k] = "tail"
    for li, c in enumerate(spec["legs"]):
        if c[0] in idx:
            for k in desc(idx[c[0]]):
                group_t[k] = f"leg{li}"
    # predicted joint -> group: legs matched to template legs by nearest leg root
    gp = {k: "body" for k in range(len(p.names))}
    leg_roots = {li: jp[idx[c[0]]] for li, c in enumerate(spec["legs"]) if c[0] in idx}
    for d in lab["chains"]:
        if d["label"] == "tail":
            for k in d["joints"]:
                gp[k] = "tail"
        elif d["label"] == "leg" and leg_roots:
            li = min(leg_roots, key=lambda i: float(np.linalg.norm(leg_roots[i] - p.jp[d["joints"][0]])))
            for k in d["joints"]:
                gp[k] = f"leg{li}"
    # template bone segments (joint k -> each child); a leaf bone is its point
    segs = {k: [(jp[k], jp[c]) for c in kid[k]] or [(jp[k], jp[k])] for k in range(len(names))}
    cand = {g: [k for k in range(len(names)) if group_t[k] == g and k not in banned and names[k] != "root"] for g in set(group_t.values())}
    pkid = p.kids
    M = np.zeros((len(P_mesh), len(names)))
    for slot in range(p.J.shape[1]):
        wv = p.W[:, slot]
        jv = p.J[:, slot]
        for k in np.unique(jv[wv > 0]):
            sel = (jv == k) & (wv > 0)
            X = P_mesh[sel]
            a = p.jp[k]
            best = np.full(len(X), np.inf)
            q = np.repeat(a[None], len(X), 0)
            for c in pkid[k] or [k]:
                ab = p.jp[c] - a
                t = np.clip(((X - a) @ ab) / max(ab @ ab, 1e-12), 0, 1) if ab @ ab > 1e-12 else np.zeros(len(X))
                y = a + t[:, None] * ab
                dd = np.linalg.norm(X - y, axis=1)
                better = dd < best
                q[better], best[better] = y[better], dd[better]
            ks = cand.get(gp[k]) or cand["body"]
            D = np.stack([np.min([np.linalg.norm(q - (s0 + np.clip(((q - s0) @ (s1 - s0)) / max((s1 - s0) @ (s1 - s0), 1e-12), 0, 1)[:, None] * (s1 - s0)), axis=1)
                                  for s0, s1 in segs[t_]], axis=0) for t_ in ks], 1)
            tgt = np.array(ks)[D.argmin(1)]
            np.add.at(M, (np.where(sel)[0], tgt), wv[sel])
    J, W = rig.to_top4(M)
    info = {"vertices_on_banned": int((M[:, sorted(banned)].sum(1) > 0).sum()) if banned else 0,
            "bones_used": int((M.sum(0) > 0).sum())}
    return J, W, info


# ---------------------------------------------------------------- floor pass

class FK:
    """Forward kinematics of our skeleton (identity rest rotations; node translations) with per-node local quats."""

    def __init__(self, ours: dict) -> None:
        self.nodes = ours["nodes"]
        self.joints = ours["skins"][0]["joints"]
        sk = rig.Skeleton(ours)
        self.parent = sk.parent
        self.order = [i for i in sk.order if i in self.joints]
        self.t0 = {i: np.array(self.nodes[i].get("translation", [0, 0, 0]), float) for i in self.order}
        w = sk.world({})
        self.rest = {i: w[i][0] for i in self.order}

    def world(self, quats: dict[int, np.ndarray], trans: dict[int, np.ndarray]) -> tuple[dict, dict]:
        pos, rot = {}, {}
        for i in self.order:
            q = R.from_quat(quats[i]) if i in quats else R.identity()
            t = trans.get(i, self.t0[i])
            p = self.parent.get(i)
            if p in pos:
                pos[i] = pos[p] + rot[p].apply(t)
                rot[i] = rot[p] * q
            else:
                pos[i], rot[i] = t.copy(), q
        return pos, rot


def _skin(P: np.ndarray, J: np.ndarray, W: np.ndarray, fk: FK, pos: dict, rot: dict) -> np.ndarray:
    js = fk.joints
    out = np.zeros_like(P)
    for slot in range(J.shape[1]):
        for k in np.unique(J[:, slot]):
            sel = J[:, slot] == k
            j = js[k]
            out[sel] += W[sel, slot][:, None] * (rot[j].apply(P[sel] - fk.rest[j]) + pos[j])
    return out


def template_floor_track(template: dict, names: list[str], legs_keys: list[str], times: dict,
                         trunk_cut: tuple[str, ...] = ()) -> dict:
    """The template's own body per frame (template units): its lowest point above its reference floor ("low"), and
    for each leg key joint the lowest point of that foot above the body's lowest point ("foot": ground-relative
    clearance, 0 in stance); plus the template's height at the reference frame. trunk_cut: the joints whose subtrees
    are not trunk (neck, leg roots, tail); given, the lowest trunk point above the reference floor per frame
    ("trunk") and at the reference frame ("trunk_rest") too, and the body's height over its reference height
    ("span": about 1 standing, about 0.5 lying)."""
    import rig_motion
    doc, b = rig.read_glb(rig.M2M / template["clips"][0])
    nodes = doc["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    mi = next(i for i, n in enumerate(nodes) if "mesh" in n and "skin" in n)
    pr = doc["meshes"][nodes[mi]["mesh"]]["primitives"][0]
    a = pr["attributes"]
    P = rig.accessor(doc, b, a["POSITION"]).astype(float)
    J = rig.accessor(doc, b, a["JOINTS_0"]).astype(int)
    W = rig.accessor(doc, b, a["WEIGHTS_0"]).astype(float)
    skn = doc["skins"][nodes[mi]["skin"]]
    IBM = rig.accessor(doc, b, skn["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    Ph = np.c_[P, np.ones(len(P))]
    ni = {n.get("name"): i for i, n in enumerate(nodes)}
    top_node = np.array(skn["joints"])[J[np.arange(len(J)), W.argmax(1)]]

    def sub(i: int) -> set[int]:
        out, st = set(), [i]
        while st:
            x = st.pop()
            out.add(x)
            st += nodes[x].get("children", [])
        return out
    feet = {k: np.isin(top_node, sorted(sub(ni[k]) | {parent[ni[k]]})) for k in legs_keys if k in ni}
    cut = [ni[n] for n in trunk_cut if n in ni]
    # every cut joint must be found: a trunk that keeps a leg would read about 0 standing, i.e. always lying
    trunk = ~np.isin(top_node, sorted(set().union(*[sub(i) for i in cut]))) if cut and len(cut) == len(trunk_cut) else None
    chans = {}
    for f in template["clips"]:
        d, bb = (doc, b) if f == template["clips"][0] else rig.read_glb(rig.M2M / f)
        for an in d.get("animations", []):
            _, ch = rig_motion.channels(d, bb, an)
            if d is not doc:  # another clip file: map its channels onto this file's nodes by name
                ch = [(ni[d["nodes"][x[0]].get("name")],) + tuple(x[1:]) for x in ch if d["nodes"][x[0]].get("name") in ni]
            chans[an["name"]] = ch

    def at(name: str | None, t: float) -> np.ndarray:
        local = rig_motion.sample(nodes, chans[name], t) if name else {}
        wm = rig_motion.world_mats(nodes, parent, local)
        M = np.stack([wm[j] @ IBM[k] for k, j in enumerate(skn["joints"])])
        return rig_motion.skin(Ph, J, W, M)
    ref = template["ref"] if isinstance(template["ref"], str) else None
    V0 = at(ref, 0.0)
    fl0 = float(V0[:, 1].min())
    out = {}
    for nm in names:
        if nm not in chans:
            continue
        lows, fts, tks, sps = [], {k: [] for k in feet}, [], []
        for t in times[nm]:
            V = at(nm, float(t))
            lo = float(V[:, 1].min())
            lows.append(lo - fl0)
            if trunk is not None and trunk.any():
                tks.append(float(V[trunk, 1].min()) - fl0)
                sps.append(float(np.ptp(V[:, 1])) / max(float(np.ptp(V0[:, 1])), 1e-9))
            for k, m in feet.items():
                fts[k].append(float(V[m, 1].min()) - lo if m.any() else 0.0)
        out[nm] = {"low": np.array(lows), "foot": {k: np.array(v) for k, v in fts.items()},
                   **({"trunk": np.array(tks), "span": np.array(sps)} if tks else {})}
    return {"clips": out, "H": float(np.ptp(V0[:, 1])),
            **({"trunk_rest": float(V0[trunk, 1].min()) - fl0} if trunk is not None and trunk.any() else {})}


def template_head_track(template: dict, names: list[str], head_root: str, times: dict,
                        joint_names: tuple[str, ...] = ()) -> dict:
    """The template's head per frame (template units): the lowest point of the vertices whose top bone is head_root or
    below it, above the template's reference floor ("head"), and above the template body's lowest point that frame
    ("head_rel"), and the height of the rest of the body over its reference height ("body_h": about 1 standing, about
    0.5 lying), and the world positions of joint_names ("joints": {name: (frames, 3)}); plus the template's height
    at the reference frame and the head's rest clearance ("rest")."""
    import rig_motion
    doc, b = rig.read_glb(rig.M2M / template["clips"][0])
    nodes = doc["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    mi = next(i for i, n in enumerate(nodes) if "mesh" in n and "skin" in n)
    a = doc["meshes"][nodes[mi]["mesh"]]["primitives"][0]["attributes"]
    P = rig.accessor(doc, b, a["POSITION"]).astype(float)
    J = rig.accessor(doc, b, a["JOINTS_0"]).astype(int)
    W = rig.accessor(doc, b, a["WEIGHTS_0"]).astype(float)
    skn = doc["skins"][nodes[mi]["skin"]]
    IBM = rig.accessor(doc, b, skn["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    Ph = np.c_[P, np.ones(len(P))]
    ni = {n.get("name"): i for i, n in enumerate(nodes)}
    top_node = np.array(skn["joints"])[J[np.arange(len(J)), W.argmax(1)]]
    if head_root not in ni:
        return {"clips": {}, "H": 1.0, "rest": None}
    sub, st = set(), [ni[head_root]]
    while st:
        x = st.pop()
        sub.add(x)
        st += nodes[x].get("children", [])
    head = np.isin(top_node, sorted(sub))
    chans = {}
    for f in template["clips"]:
        d, bb = (doc, b) if f == template["clips"][0] else rig.read_glb(rig.M2M / f)
        for an in d.get("animations", []):
            _, ch = rig_motion.channels(d, bb, an)
            if d is not doc:
                ch = [(ni[d["nodes"][x[0]].get("name")],) + tuple(x[1:]) for x in ch if d["nodes"][x[0]].get("name") in ni]
            chans[an["name"]] = ch

    jn = [n for n in joint_names if n in ni]

    def at(name: str | None, t: float) -> tuple[np.ndarray, dict]:
        local = rig_motion.sample(nodes, chans[name], t) if name else {}
        wm = rig_motion.world_mats(nodes, parent, local)
        M = np.stack([wm[j] @ IBM[k] for k, j in enumerate(skn["joints"])])
        return rig_motion.skin(Ph, J, W, M), {n: wm[ni[n]][:3, 3].copy() for n in jn}
    ref = template["ref"] if isinstance(template["ref"], str) and template["ref"] in chans else None
    V0 = at(ref, 0.0)[0]
    fl0 = float(V0[:, 1].min())
    out = {}
    for nm in names:
        if nm not in chans or not head.any():
            continue
        hs, rel, bh, jp = [], [], [], {n: [] for n in jn}
        for t in times[nm]:
            V, js = at(nm, float(t))
            for n in jn:
                jp[n].append(js[n])
            hs.append(float(V[head, 1].min()) - fl0)
            rel.append(float(V[head, 1].min()) - float(V[:, 1].min()))
            bh.append(float(np.ptp(V[~head, 1])) / max(float(np.ptp(V0[~head, 1])), 1e-9))
        out[nm] = {"head": np.array(hs), "head_rel": np.array(rel), "body_h": np.array(bh),
                   "joints": {n: np.array(v) for n, v in jp.items()}}
    return {"clips": out, "H": float(np.ptp(V0[:, 1])), "rest": float(V0[head, 1].min()) - fl0 if head.any() else None}


HEAD_REACH = (0.15, 0.5)  # template head clearance / its rest clearance: full reach below the first, none above the second
HEAD_STAND = 0.75  # the template's body (all but the head) at least this share of its rest height: standing, not lying
HEAD_BEND_MAX = 60.0  # degrees of extra neck bend, shared along the neck chain
HEAD_SHARE = (0.4, 0.35, 0.25)  # of the bend, from the neck's root toward the head (another length: even shares)
# Share of the template's turn at the plan's neck_from joint (the horse's spine_4, its neck's base) that head_reach
# keeps; the trunk joints behind it go back to rest. On the horse that joint carries 38 vertices and its neck rises
# out of it; on the sheep it and its child ribcage_front carry 543 (the withers and the brisket), so the horse's 63
# degree Eating turn there folded the sheep's chest down onto its front legs. 0 keeps the body standing and lowers the
# head with the neck alone (on the sheep its mouth stops 0.38 of its height above the floor); a share keeps that much
# of the turn, its off-path children (ribcage, shoulders) holding their standing world turn: 0.5 stops at 0.25 with
# the brisket shifted 0.10 of the height, 1.0 at 0.19 with it shifted 0.19. Stephen chose half on the sheep (version 11:
# the check's Eating p99 1.73, the chest bent 32 degrees to the horse's 73).
NECK_BASE_KEEP = 0.5


def _smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def head_bend(fk: "FK", quats: dict, trans: dict, chain: list[int], lateral: np.ndarray, Ph: np.ndarray, Jh: np.ndarray,
              Wh: np.ndarray, want: float, max_deg: float = HEAD_BEND_MAX) -> tuple[dict, float]:
    """Bend the neck chain (joint nodes, root first) about the lateral axis, the bend shared along it, so the lowest
    of the head points (Ph, skinned by Jh/Wh) comes down to `want`, or as near as max_deg allows. A 1-D search on
    the total angle (the head's lowest point falls steadily as the neck bends down). Returns new quats and degrees."""
    share = np.asarray(HEAD_SHARE if len(chain) == len(HEAD_SHARE) else [1.0] * len(chain), float)
    share /= share.sum()

    def bent(theta: float) -> dict:
        q = dict(quats)
        for j, sh in zip(chain, share):
            pos, rot = fk.world(q, trans)
            pj = fk.parent.get(j)
            prot = rot[pj] if pj in rot else R.identity()
            q[j] = (prot.inv() * R.from_rotvec(lateral * theta * sh) * rot[j]).as_quat()
        return q

    def low(q: dict) -> float:
        pos, rot = fk.world(q, trans)
        return float(_skin(Ph, Jh, Wh, fk, pos, rot)[:, 1].min())
    l0 = low(quats)
    if l0 <= want:
        return quats, 0.0
    # the lowest point isn't monotonic in the bend (a head already pointing down rises either way): walk out from 0
    # both ways in 2.5 degree steps to the first angle that reaches, else take the lowest angle found
    steps = np.radians(np.arange(2.5, max_deg + 1e-6, 2.5))
    lowest, hits = (l0, 0.0), []
    for sgn in (1.0, -1.0):
        prev = 0.0
        for a in steps:
            lv = low(bent(sgn * a))
            if lv < lowest[0]:
                lowest = (lv, sgn * a)
            if lv <= want:
                lo_a, hi_a = prev, a  # refine between the last miss and this hit
                for _ in range(10):
                    mid = 0.5 * (lo_a + hi_a)
                    if low(bent(sgn * mid)) > want:
                        lo_a = mid
                    else:
                        hi_a = mid
                hits.append(sgn * hi_a)
                break
            prev = a
    theta = min(hits, key=abs) if hits else lowest[1]
    if theta == 0.0:
        return quats, 0.0
    return bent(theta), float(np.degrees(abs(theta)))


LIE = (0.25, 0.5)  # the template trunk's clearance / its rest clearance: lying below the first, standing above the second
LIE_SPAN = 0.75  # and the template's height under this share of its rest height (lying), blended over 0.1 below it
LIE_HOLD = 0.1  # seconds: lying must last about twice this; shorter (a flapping bird dropping through its floor) is not

PITCH_MAX = 20.0  # degrees the trunk may tip forward about the hind legs' tops for the head to reach


def trunk_pitch(fk: "FK", quats: dict, trans: dict, spec: dict, lateral: np.ndarray, Ph: np.ndarray, Jh: np.ndarray,
                Wh: np.ndarray, want: float, max_deg: float = PITCH_MAX) -> tuple[dict, dict, float]:
    """Tip the trunk (spec["hips"]) about the lateral axis through the hind legs' tops (spec["pivot"] joints) so the
    lowest head point comes down to `want`, or as near as max_deg allows; every leg root (spec["keep"]) keeps its
    world rotation, so the legs only move with their sockets (the caller plants the feet again). Returns quats,
    trans and the angle in degrees (0 when tipping doesn't help)."""
    hips, rootn = spec["hips"], spec["root"]
    pos0, rot0 = fk.world(quats, trans)
    pivot = np.mean([pos0[k] for k in spec["pivot"]], 0)
    keep_w = {k: rot0[k] for k in spec["keep"]}

    def tipped(phi: float) -> tuple[dict, dict]:
        q, t = dict(quats), dict(trans)
        Rp = R.from_rotvec(lateral * phi)
        ph = fk.parent.get(hips)
        prot = rot0[ph] if ph in rot0 else R.identity()
        q[hips] = (prot.inv() * Rp * rot0[hips]).as_quat()
        # the hips turn about their own joint: move the root so the pivot stays where it was
        t[rootn] = t[rootn] + (pivot - pos0[hips]) - Rp.apply(pivot - pos0[hips])
        pos, rot = fk.world(q, t)
        for k in spec["keep"]:
            pk = fk.parent[k]
            q[k] = (rot[pk].inv() * keep_w[k]).as_quat()
        return q, t

    def low(phi: float) -> float:
        q, t = tipped(phi)
        pos, rot = fk.world(q, t)
        return float(_skin(Ph, Jh, Wh, fk, pos, rot)[:, 1].min())
    l0 = low(0.0)
    sgn = 1.0 if low(np.radians(2)) < low(np.radians(-2)) else -1.0
    best, prev = (l0, 0.0), 0.0
    for a in np.radians(np.arange(1.0, max_deg + 1e-6, 1.0)):
        lv = low(sgn * a)
        if lv < best[0]:
            best = (lv, sgn * a)
        if lv <= want:  # refine between the last miss and this hit, so the tip follows the target smoothly
            lo_a, hi_a = prev, a
            for _ in range(10):
                mid = 0.5 * (lo_a + hi_a)
                if low(sgn * mid) > want:
                    lo_a = mid
                else:
                    hi_a = mid
            best = (want, sgn * hi_a)
            break
        prev = a
    if best[1] == 0.0:
        return quats, trans, 0.0
    q, t = tipped(best[1])
    return q, t, float(np.degrees(abs(best[1])))


def _two_bone(pos: dict, rot: dict, A: int, B: int, K: int, target: np.ndarray, lateral: np.ndarray) -> tuple[R, R]:
    """World rotations at B (bend) then at A (aim) that put K at target (as near as the leg reaches)."""
    a, b, k = pos[A], pos[B], pos[K]
    la, lb = np.linalg.norm(b - a), np.linalg.norm(k - b)
    d0 = np.linalg.norm(k - a)
    d1 = float(np.clip(np.linalg.norm(target - a), abs(la - lb) + 1e-6, la + lb - 1e-6))
    n = np.cross(b - a, k - b)
    if np.linalg.norm(n) < 1e-9:
        n = lateral.copy()
    n /= np.linalg.norm(n)

    def ang(d: float) -> float:  # interior angle at B for a given A-K distance
        return float(np.arccos(np.clip((la * la + lb * lb - d * d) / (2 * la * lb), -1, 1)))
    qB = R.from_rotvec(n * (ang(d0) - ang(d1)))
    k2 = b + qB.apply(k - b)
    u, v = k2 - a, target - a
    ax = np.cross(u, v)
    if np.linalg.norm(ax) < 1e-12:
        return qB, R.identity()
    th = float(np.arctan2(np.linalg.norm(ax), u @ v))
    return qB, R.from_rotvec(ax / np.linalg.norm(ax) * th)


def floor_pass(clips: list[dict], ours: dict, P: np.ndarray, J: np.ndarray, W: np.ndarray, template: dict, cls: str,
               scale_y: float, lateral: np.ndarray, plan: dict | None = None, prefer_track: bool = False,
               head_reach: bool = False) -> dict:
    """Changes the baked clips in place (local quats of leg / neck joints, root translation), per frame:
    1. the body (all but the head) is grounded like the template's: its lowest point at the template's clearance
       above its floor, scaled (0 whenever the template touches);
    2. each foot is bent / stretched (two-bone IK at hip and knee, the foot kept level) to the template foot's
       clearance above the template body's lowest point, scaled (0 in stance), then 1. again (three rounds);
    2b. head_reach: while the standing template brings its head near its floor (Eating: its head's clearance under
       HEAD_REACH[1] of its rest clearance, its body over HEAD_STAND of its rest height), the neck chain (neck_lift
       to the head) is bent down about the lateral axis (head_bend: the bend shared along the chain, at most
       HEAD_BEND_MAX degrees) until the head's lowest point has the template head's clearance, scaled; blended in and
       out with the template's clearance (a short neck on the horse's clip otherwise stops in the air); with a plan
       neck_from the trunk is held standing first (NECK_BASE_KEEP) and is not tipped, so a short neck stops short;
    3. a head that still goes through the floor is lifted at the neck (about the lateral axis, up to 45 degrees).
    plan: the body plan whose leg / neck_lift paths to use (PATHS[cls] without one). prefer_track: a clip that carries
    its own contacts (c['track']['contact'], rig_clips) plants each foot on its own stance frames (clearance 0 there)
    and is never neck-lifted on its beak-contact frames (a Peck touches the floor on purpose).
    Returns per clip, before and after: lowest point vs the floor and floating in the template's contact frames (/H)."""
    fk = FK(ours)
    names = [ours["nodes"][j]["name"] for j in fk.joints]
    node_of = {n: j for n, j in zip(names, fk.joints)}
    floor, H = float(P[:, 1].min()), float(np.ptp(P[:, 1]))
    if len(P) > 12000:  # a dense mesh (a 52k-vertex person): every k-th vertex, plus the lowest ones, is plenty for a floor
        keep = np.unique(np.r_[np.arange(0, len(P), len(P) // 8000), np.argsort(P[:, 1])[:500]])
        P, J, W = P[keep], J[keep], W[keep]
    spec = paths_for(cls, plan)
    fix = spec is not None  # a class without leg paths here (people) is only grounded (step 1), no IK
    spec = spec or {"legs": [], "neck_lift": None}
    if not clips:
        return {}
    legs = [c for c in spec["legs"] if all(n in node_of for n in c)]
    J_top = np.array(fk.joints)[J[np.arange(len(J)), W.argmax(1)]]

    def sub(n: str) -> set[int]:
        out, st = set(), [node_of[n]]
        while st:
            x = st.pop()
            out.add(x)
            st += [c for c in ours["nodes"][x].get("children", []) if c in fk.order]
        return out
    foot_v = [np.isin(J_top, sorted(sub(c[-1]) | {node_of[c[-2]]})) for c in legs]
    neck = spec.get("neck_lift")
    head_v = np.isin(J_top, sorted(sub(neck))) if neck in node_of else np.zeros(len(P), bool)
    body_v = ~head_v
    times = {c["name"]: c["times"] for c in clips}
    # the trunk: what's left of the body without the neck, the legs (from each leg's root under a body joint) and the
    # tail; a lying frame grounds it, not the folded legs under it
    body_path = [n for n in spec.get("body") or [] if n in node_of]
    cut = [neck] if neck in node_of else []
    for cc in legs:
        k = node_of[cc[0]]
        while fk.parent.get(k) is not None and ours["nodes"][fk.parent[k]].get("name") not in body_path:
            k = fk.parent[k]
        cut.append(ours["nodes"][k]["name"] if fk.parent.get(k) is not None else cc[0])
    tail = [n for n in spec.get("tail") or [] if n in node_of]
    cut += tail[1:2]
    trunk_v = body_v & ~np.isin(J_top, sorted(set().union(*[sub(n) for n in cut]))) if fix and body_path and cut \
        else np.zeros(len(P), bool)
    track = template_floor_track(template, list(times), [c[-1] for c in legs], times,
                                 tuple(cut) if trunk_v.any() else ())
    k_scale = H / max(track["H"], 1e-9)
    rootn = clips[0]["root"]
    reach = None
    if head_reach and fix and neck in node_of:
        body_names = [n for n in spec.get("body") or [] if n in node_of]
        hname = "head" if "head" in node_of else neck
        if neck in body_names and hname in body_names and body_names.index(hname) >= body_names.index(neck):
            chain = body_names[body_names.index(neck):body_names.index(hname) + 1]
        else:
            chain = [neck]
        reach_v = np.isin(J_top, sorted(sub(hname)))
        # each neck bone (and the head, toward its first child on the plan's body path) is aimed the way the
        # template's points: the same world rotation on a neck whose rest runs level (a sheep's) instead of up (the
        # horse's) curls it back under the chest
        tips = chain[1:] + [body_names[body_names.index(hname) + 1]] if hname in body_names and \
            body_names.index(hname) + 1 < len(body_names) else chain[1:]
        aims = list(zip(chain, tips))
        htrack = template_head_track(template, list(times), hname, times, tuple(dict.fromkeys(chain + tips)))
        # the trunk to hold standing while the head reaches: the body joints after the first up to the neck's base
        # (neck_from, before neck_lift), the legs hanging from the neck's base (the front legs, which the horse plants
        # forward to reach: on the sheep that pose stretched 16 shoulder edges past 2x), and neck_from's children off
        # the body path (ribcage, shoulders)
        nf = spec.get("neck_from")
        hold = None
        if nf in body_names and neck in body_names and 0 < body_names.index(nf) < body_names.index(neck):
            hold = {"trunk": [node_of[n] for n in body_names[1:body_names.index(nf)]], "base": node_of[nf],
                    "off": [k for k in ours["nodes"][node_of[nf]].get("children", [])
                            if k in fk.order and ours["nodes"][k].get("name") not in body_names]}
            for cc in legs:  # each leg from its root (the child of a body joint) to the joint above the foot
                k, run = node_of[cc[0]], [node_of[n] for n in cc[:-1]]
                while fk.parent.get(k) is not None and ours["nodes"][fk.parent[k]].get("name") not in body_names:
                    k = fk.parent[k]
                    run.append(k)
                if fk.parent.get(k) == node_of[nf]:
                    hold["trunk"] += run
        if reach_v.any() and htrack.get("rest"):
            yaw = float(np.arctan2(-lateral[2], lateral[0]))
            reach = {"chain": [node_of[n] for n in chain], "v": reach_v, "track": htrack, "k": H / max(htrack["H"], 1e-9),
                     "aims": [(node_of[a], node_of[b], a, b) for a, b in aims], "yaw": R.from_rotvec([0, yaw, 0]),
                     "hold": hold}
            if body_names and legs and node_of.get(body_names[0]) is not None:
                fwd = fk.rest[node_of[hname]] - fk.rest[node_of[body_names[0]]]
                tops = sorted(legs, key=lambda cc: float(fk.rest[node_of[cc[0]]] @ fwd))
                keep = []
                for cc in legs:  # each leg's root: up from its first joint to the child of a body joint
                    k = node_of[cc[0]]
                    while fk.parent.get(k) is not None and ours["nodes"][fk.parent[k]].get("name") not in body_names:
                        k = fk.parent[k]
                    keep.append(k)
                if fk.parent.get(node_of[body_names[0]]) == rootn:
                    reach["pitch"] = {"hips": node_of[body_names[0]], "root": rootn, "keep": sorted(set(keep)),
                                      "pivot": [node_of[cc[0]] for cc in tops[:2]]}
    out = {}
    for c in clips:
        tr = track["clips"].get(c["name"])
        if tr is None:
            continue
        before, after, flo_b, flo_a, neck_deg = [], [], [], [], []
        reach_deg, reach_gap_b, reach_gap_a, pitch_deg = [], [], [], []
        hb = None
        if reach is not None and c["name"] in reach["track"]["clips"]:
            ht = reach["track"]["clips"][c["name"]]
            rel = ht["head"] / reach["track"]["rest"]
            hb = (1.0 - _smooth01((rel - HEAD_REACH[0]) / (HEAD_REACH[1] - HEAD_REACH[0]))) \
                * _smooth01((ht["body_h"] - HEAD_STAND + 0.1) / 0.1)
            if hb.max() < 0.05:
                hb = None
        contact = tr["low"] / track["H"] < 0.02
        own = ((c.get("track") or {}).get("contact") or {}) if prefer_track else {}
        F_ = len(c["times"])

        def own_mask(key: str) -> np.ndarray | None:
            m = own.get(key)
            if m is None:
                return None
            m = np.asarray(m, bool).reshape(-1)
            return m if len(m) == F_ else None
        foot_own = {cch[-1]: own_mask(cch[-1]) for cch in legs}
        beak = own_mask("beak")
        tracked = bool(own) and any(m is not None for m in foot_own.values())
        # lying: the template's trunk down on its floor, both near the floor and near the template's own lowest point
        # (under LIE[1] of its rest clearance, blended in from there to LIE[0]; a flying bird whose clip carries it far
        # below its floor with its legs tucked is not lying); the trunk is grounded at the template trunk's clearance,
        # scaled, whenever the lowest-point grounding leaves it higher (a sheep's folded hooves would otherwise prop
        # its belly off the floor). Only ever lower, so the legs fold under it rather than lift it. The template must be
        # short too (its height under LIE_SPAN of its rest height), and stay down (an opening over LIE_HOLD each side
        # keeps the ramps in and out and drops blips): a flapping bird's trunk passes its floor mid-air
        lw = None
        if trunk_v.any() and "trunk" in tr and track.get("trunk_rest", 0.0) > 0.05 * track["H"] and not tracked:
            share = np.maximum(np.abs(tr["trunk"]), tr["trunk"] - tr["low"]) / track["trunk_rest"]
            lw = (1.0 - _smooth01((share - LIE[0]) / (LIE[1] - LIE[0]))) \
                * (1.0 - _smooth01((tr["span"] - LIE_SPAN + 0.1) / 0.1))
            tt = np.asarray(c["times"], float)
            near = np.abs(tt[:, None] - tt[None, :]) <= LIE_HOLD + 1e-6
            lw = np.where(near, lw[None, :], 1.0).min(1)  # erode, then dilate: a dip shorter than the window goes
            lw = np.where(near, lw[None, :], 0.0).max(1)
            if lw.max() < 0.01:
                lw = None
        for f in range(len(c["times"])):
            quats = {j: c["rot"][j][f].astype(float) for j in c["rot"]}
            trans = {rootn: c["root_t"][f].astype(float)}
            pos, rot = fk.world(quats, trans)
            V = _skin(P, J, W, fk, pos, rot)
            before.append((V[:, 1].min() - floor) / H)
            if contact[f]:
                flo_b.append((V[:, 1].min() - floor) / H)
            if hb is not None and reach["hold"] and hb[f] > 0.01 and not (beak is not None and beak[f]):
                # the head reaches with the neck, not the chest: the trunk and front legs blend back to rest (identity
                # local turns) and the neck's base keeps NECK_BASE_KEEP of its turn, before the body is grounded and
                # the feet planted, so the legs stand under a standing trunk
                hold, w = reach["hold"], float(hb[f])
                for j in hold["trunk"]:
                    quats[j] = R.from_rotvec(R.from_quat(quats[j]).as_rotvec() * (1.0 - w)).as_quat()
                q_base = R.from_quat(quats[hold["base"]])
                q_stand = R.from_rotvec(q_base.as_rotvec() * (1.0 - w))
                q_kept = R.from_rotvec(q_base.as_rotvec() * (1.0 - w * (1.0 - NECK_BASE_KEEP)))
                quats[hold["base"]] = q_kept.as_quat()
                for j in hold["off"]:
                    quats[j] = (q_kept.inv() * q_stand * R.from_quat(quats[j])).as_quat()
                pos, rot = fk.world(quats, trans)
                V = _skin(P, J, W, fk, pos, rot)
            want = floor + max(0.0, float(tr["low"][f])) * k_scale

            def ground(V: np.ndarray) -> None:
                dy = want - float(V[body_v, 1].min())
                if lw is not None and lw[f] > 0.0:
                    dt = floor + max(0.0, float(tr["trunk"][f])) * k_scale - float(V[trunk_v, 1].min())
                    dy += float(lw[f]) * min(0.0, dt - dy)
                trans[rootn] = trans[rootn] + np.array([0, dy, 0])
            def legs_ik() -> None:
                nonlocal quats
                pos, rot = fk.world(quats, trans)
                for li, cch in enumerate(legs):
                    if not foot_v[li].any():
                        continue
                    A, B, K = node_of[cch[0]], node_of[cch[1]], node_of[cch[-1]]
                    Vf = _skin(P[foot_v[li]], J[foot_v[li]], W[foot_v[li]], fk, pos, rot)
                    fo = foot_own.get(cch[-1])
                    if fo is not None and not fo[f]:
                        continue  # a swing frame of a clip with its own contacts: its IK already lifts the foot
                    clear = 0.0 if fo is not None else max(0.0, float(tr["foot"][cch[-1]][f]))
                    need = (want + clear * k_scale) - float(Vf[:, 1].min())
                    # no dead band while lying: a folded hoof hovering about the floor would switch the IK on and off
                    if abs(need) < (0.002 * H if lw is None or lw[f] <= 0.0 else 1e-6 * H):
                        continue
                    qB, qA = _two_bone(pos, rot, A, B, K, pos[K] + np.array([0, need, 0]), lateral)
                    oldK = rot[K]
                    quats[B] = (rot[fk.parent[B]].inv() * qB * rot[B]).as_quat()
                    pA = fk.parent.get(A)
                    quats[A] = ((rot[pA] if pA in rot else R.identity()).inv() * qA * rot[A]).as_quat()
                    pos, rot = fk.world(quats, trans)
                    quats[K] = (rot[fk.parent[K]].inv() * oldK).as_quat()
                    pos, rot = fk.world(quats, trans)
            for _ in range(3 if fix else 0):
                ground(V)
                legs_ik()
                pos, rot = fk.world(quats, trans)
                V = _skin(P, J, W, fk, pos, rot)
            ground(V)
            pos, rot = fk.world(quats, trans)
            V = _skin(P, J, W, fk, pos, rot)
            if hb is not None and hb[f] > 0.01 and not (beak is not None and beak[f]):
                target = floor + max(0.0, float(reach["track"]["clips"][c["name"]]["head"][f])) * reach["k"]
                reach_gap_b.append((float(V[reach["v"], 1].min()) - target) / H)
                tj = reach["track"]["clips"][c["name"]]["joints"]
                for ja, jb, na, nb in reach["aims"]:
                    pos, rot = fk.world(quats, trans)
                    v = pos[jb] - pos[ja]
                    u = reach["yaw"].apply(tj[nb][f] - tj[na][f])
                    if np.linalg.norm(v) < 1e-9 or np.linalg.norm(u) < 1e-9:
                        continue
                    swing = R.align_vectors([u], [v])[0]
                    turn = R.from_rotvec(swing.as_rotvec() * float(hb[f]))
                    pj = fk.parent.get(ja)
                    quats[ja] = ((rot[pj] if pj in rot else R.identity()).inv() * turn * rot[ja]).as_quat()
                pos, rot = fk.world(quats, trans)
                V = _skin(P, J, W, fk, pos, rot)
                now = float(V[reach["v"], 1].min())
                want_h = now + float(hb[f]) * (target - now)
                if now - want_h > 0.002 * H and reach.get("pitch") and not reach["hold"]:
                    # the neck alone is too short: tip the trunk forward about the hind legs' tops (the legs keep their
                    # world turn), then plant the feet again (not with a held trunk: tipping it folds the front knees
                    # to 27-65 degrees on the sheep; the head stops short instead)
                    quats, trans, pdeg = trunk_pitch(fk, quats, trans, reach["pitch"], lateral, P[reach["v"]],
                                                     J[reach["v"]], W[reach["v"]], want_h)
                    pitch_deg.append(pdeg)
                    if pdeg > 0:
                        legs_ik()
                        pos, rot = fk.world(quats, trans)
                        V = _skin(P, J, W, fk, pos, rot)
                        ground(V)
                        pos, rot = fk.world(quats, trans)
                        V = _skin(P, J, W, fk, pos, rot)
                        now = float(V[reach["v"], 1].min())
                if now - want_h > 0.002 * H:
                    quats, ang = head_bend(fk, quats, trans, reach["chain"], lateral, P[reach["v"]], J[reach["v"]],
                                           W[reach["v"]], want_h)
                    reach_deg.append(ang)
                    pos, rot = fk.world(quats, trans)
                    V = _skin(P, J, W, fk, pos, rot)
                reach_gap_a.append((float(V[reach["v"], 1].min()) - target) / H)
            lifted = 0.0
            if fix and head_v.any() and not (beak is not None and beak[f]) and floor - float(V[head_v, 1].min()) > 0.005 * H:
                nj = node_of[neck]
                pn = fk.parent[nj]
                for ang in np.radians(np.arange(2, 46, 2)):
                    done = False
                    for sgn in (1, -1):
                        q2 = dict(quats)
                        q2[nj] = (rot[pn].inv() * R.from_rotvec(lateral * sgn * ang) * rot[nj]).as_quat()
                        p2, r2 = fk.world(q2, trans)
                        if _skin(P[head_v], J[head_v], W[head_v], fk, p2, r2)[:, 1].min() >= floor - 0.005 * H:
                            quats, done, lifted = q2, True, float(np.degrees(ang))
                            break
                    if done:
                        break
                pos, rot = fk.world(quats, trans)
                V = _skin(P, J, W, fk, pos, rot)
            neck_deg.append(lifted)
            after.append((V[:, 1].min() - floor) / H)
            if contact[f]:
                flo_a.append((V[:, 1].min() - floor) / H)
            for j in c["rot"]:
                c["rot"][j][f] = np.asarray(quats[j], np.float32)
            c["root_t"][f] = np.asarray(trans[rootn], np.float32)
        out[c["name"]] = {"low_min_before": round(float(min(before)), 3), "low_min_after": round(float(min(after)), 3),
                          "float_max_before": round(float(max(flo_b)), 3) if flo_b else None,
                          "float_max_after": round(float(max(flo_a)), 3) if flo_a else None,
                          "contact_frames": int(contact.sum()), "frames": len(c["times"]),
                          "neck_lift_max_deg": round(max(neck_deg), 1), "fixed": fix,
                          **({"head_reach": {"frames": int((hb > 0.01).sum()),
                                             "bend_max_deg": round(max(reach_deg, default=0.0), 1),
                                             "pitch_max_deg": round(max(pitch_deg, default=0.0), 1),
                                             "gap_max_before": round(max(reach_gap_b, default=0.0), 3),
                                             "gap_max_after": round(max(reach_gap_a, default=0.0), 3)}} if hb is not None else {}),
                          **({"lying_frames": int((lw > 0.01).sum())} if lw is not None else {}),
                          **({"own_contacts": True} if tracked else {})}
    return out


# ---------------------------------------------------------------- fallback: the predicted rig itself, with mapped clips

def _arc(pts: np.ndarray) -> np.ndarray:
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return s / max(s[-1], 1e-9)


def fallback_clips(p: Pred, lab: dict, template: dict, cls: str, names: list[str], yaw: float,
                   plan: dict | None = None) -> tuple[list[dict], dict]:
    """Template clips on the predicted skeleton, chain by chain: each predicted joint takes the world rotation change
    of the template joint at the same arc-length fraction of the matching chain (turned by the facing yaw); joints on
    no chain follow their parent. For when the template can't be fitted (the pose gate failed)."""
    tdoc, _ = rig.read_glb(rig.M2M / template["rig"])
    tsk = rig.Skeleton(tdoc)
    ref_w = tsk.world(rig.ref_local(template, tsk))
    tpos = {n: ref_w[i][0] for n, i in tsk.index.items()}
    spec = paths_for(cls, plan)
    Ry = R.from_euler("y", yaw, degrees=True)
    import rig_plan
    neck_from = rig_plan.spec(plan, cls, "neck_from", None) or {"avian": "spine_2", "quadruped": "spine_4"}.get(cls)
    if neck_from not in spec["body"]:
        neck_from = spec["body"][min(2, len(spec["body"]) - 1)]
    plan: dict[int, str] = {}
    head_t = spec["body"][spec["body"].index(neck_from):]
    legs_t = [c for c in spec["legs"] if all(n in tpos for n in c)]
    legs_p = [d for d in lab["chains"] if d["label"] == "leg"]
    cen_p = np.mean([p.jp[d["joints"][0]] for d in legs_p], 0) if legs_p else p.jp[p.root[0]]
    cen_t = np.mean([tpos[c[0]] for c in legs_t], 0)
    span_t = max(float(np.ptp([tpos[c[0]][2] for c in legs_t])), 1e-6)
    span_p = max(float(np.ptp([Ry.inv().apply(p.jp[d["joints"][0]] - cen_p)[2] for d in legs_p])) if legs_p else 0.0, 1e-6)
    for d in lab["chains"]:
        if d["label"] == "head":
            tn = head_t
        elif d["label"] == "tail":
            tn = spec["tail"][1:]
        elif d["label"] == "leg":
            v = Ry.inv().apply(p.jp[d["joints"][0]] - cen_p)  # into the template frame (+X left, +Z forward)
            tn = min(legs_t, key=lambda c: 10.0 * (np.sign(v[0]) != np.sign(tpos[c[0]][0] - cen_t[0]))
                     + abs(v[2] / span_p - (tpos[c[0]][2] - cen_t[2]) / span_t))
        else:
            continue
        tn = [n for n in tn if n in tpos]
        ts = _arc(np.array([tpos[n] for n in tn]))
        ps = _arc(p.jp[d["joints"]])
        for k, s in zip(d["joints"], ps):
            plan[k] = tn[int(np.clip(np.searchsorted(ts, s, side="right") - 1, 0, len(tn) - 1))]
    for k in p.root:
        plan.setdefault(k, "hips")
    sources = [rig.read_glb(rig.M2M / f) for f in template["clips"]]
    anims = {a["name"]: (d, b, a) for d, b in sources for a in d.get("animations", [])}
    order = sorted(range(len(p.joints)), key=lambda k: len(p.path_to(k)))
    sk = rig.Skeleton(p.doc)
    w0 = sk.world({})
    rest_rot = [w0[j][1] for j in p.joints]
    th = float(np.ptp([ref_w[i][0][1] for i in tsk.index.values()]))
    scale = p.H / max(th, 1e-9)
    hips = tsk.index["hips"]
    out = []
    for name in names:
        if name not in anims:
            continue
        d, b, a = anims[name]
        ch = rig.prepare_clip(d, b, a, tsk)
        times = np.arange(0, rig.clip_duration(d, b, a) + 1e-6, 1.0 / rig.FPS)
        rot = {p.joints[k]: [] for k in order}
        root_t = []
        rnode = p.joints[p.root[0]]
        for t in times:
            tw = tsk.world(rig.sample_clip(ch, tsk, float(t)))
            eff, want = {}, {}
            for k in order:
                n = plan.get(k)
                dl = Ry * tw[tsk.index[n]][1] * ref_w[tsk.index[n]][1].inv() * Ry.inv() if n else (eff[p.par[k]] if p.par[k] >= 0 else R.identity())
                eff[k] = dl
                want[k] = dl * rest_rot[k]
            for k in order:
                pk = p.par[k]
                if pk >= 0:
                    pw = want[pk]
                else:
                    pj = sk.parent.get(p.joints[k])
                    pw = w0[pj][1] if pj is not None else R.identity()
                rot[p.joints[k]].append((pw.inv() * want[k]).as_quat())
            root_t.append(np.array(p.doc["nodes"][rnode].get("translation", [0, 0, 0]), float)
                          + Ry.apply((tw[hips][0] - ref_w[hips][0]) * scale))
        out.append({"name": name, "times": times.astype(np.float32), "rot": {j: np.array(v, np.float32) for j, v in rot.items()},
                    "root": rnode, "root_t": np.array(root_t, np.float32)})
    return out, {"plan": {p.names[k]: n for k, n in plan.items()}, "scale": round(scale, 3)}
