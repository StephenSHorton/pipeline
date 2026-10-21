#!/usr/bin/env python3
"""Bind and motion metrics shared by the clip synthesiser (rig_clips), the renderer (rig_render) and rig_inspect.

numpy only (scipy's cKDTree in pose_flags). Everything takes arrays, never files, so a caller can gate clips on
in-memory LBS before anything is written.

Conventions
- P (N,3) rest positions, J (N,K) int joint indices, W (N,K) weights (rows need not sum to 1).
- jpos (B,3) rest joint positions in skin-joint order.
- segs: list of (parent_k, child_k) pairs in skin-joint order. Bone k is the set of segments from joint k to each of
  its children; a leaf joint is a point. This is the all-children distance used by rig.far_motion and bindcheck.
- H: model height (ptp of rest y).
- Skinning matrices are world @ inverse-bind, (B,4,4), row-major acting on column vectors.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

FAR_REL = 0.15  # an influence is "far" when its bone is this much (x H) farther than the vertex's nearest bone


# ---------------------------------------------------------------- geometry helpers

def seg_dist(P: np.ndarray, jpos: np.ndarray, segs) -> np.ndarray:
    """Distance (N,B) from each point to each bone (joint -> each child; a leaf is a point). Copied from rig_inspect /
    rig.far_motion's segd (all-children form)."""
    P = np.asarray(P, float)
    jpos = np.asarray(jpos, float)
    kids: dict[int, list[int]] = {}
    for a, c in segs:
        kids.setdefault(int(a), []).append(int(c))
    D = np.empty((len(P), len(jpos)))
    for k in range(len(jpos)):
        out = np.linalg.norm(P - jpos[k], axis=1)
        for c in kids.get(k, []):
            ab = jpos[c] - jpos[k]
            den = float(ab @ ab)
            if den <= 1e-12:
                continue
            t = np.clip(((P - jpos[k]) @ ab) / den, 0, 1)
            out = np.minimum(out, np.linalg.norm(P - (jpos[k] + t[:, None] * ab), axis=1))
        D[:, k] = out
    return D


def segs_from_parent(parent) -> list[tuple[int, int]]:
    """(parent_k, child_k) pairs from a parent list in skin-joint order (-1 = no parent)."""
    return [(int(p), k) for k, p in enumerate(parent) if p is not None and int(p) >= 0]


def groups(P: np.ndarray) -> np.ndarray:
    """Which welded point each corner is a copy of (same rule as rig_motion.groups)."""
    return np.unique(np.round(np.asarray(P, float) / 1e-5).astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)


def edges_from_faces(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Unique mesh edges (E,2) as corner indices, with UV-seam copies welded and zero-length edges dropped."""
    P = np.asarray(P, float)
    F = np.asarray(F).reshape(-1, 3).astype(int)
    inv = groups(P)
    ng = int(inv.max()) + 1 if len(inv) else 0
    first = np.zeros(ng, int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    Fw = inv[F]
    E = np.unique(np.sort(np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]], 1), axis=0)
    E = E[E[:, 0] != E[:, 1]]
    E = first[E]
    H = max(float(np.ptp(P[:, 1])), 1e-9)
    l0 = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    return E[l0 > 1e-6 * H]


def top_bone(J: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.asarray(J)[np.arange(len(J)), np.asarray(W).argmax(1)]


def _chain_set(chain, names) -> set[int]:
    """Joint indices of a chain given as names / indices, or a list of such chains."""
    out: set[int] = set()
    if chain is None:
        return out
    if isinstance(chain, (str, int, np.integer)):
        chain = [chain]
    for x in chain:
        if isinstance(x, (list, tuple)):
            out |= _chain_set(x, names)
        elif isinstance(x, str):
            if names is not None and x in names:
                out.add(list(names).index(x))
        else:
            out.add(int(x))
    return out


# ---------------------------------------------------------------- skinning

def skin_frames(P: np.ndarray, J: np.ndarray, W: np.ndarray, world_mats_per_frame: np.ndarray,
                ibm: np.ndarray | None = None) -> np.ndarray:
    """LBS positions V (F,N,3). world_mats_per_frame is (F,B,4,4) or (B,4,4) skinning matrices (world @ IBM); pass
    `ibm` (B,4,4) to have them multiplied here when the input is plain joint world matrices."""
    M = np.asarray(world_mats_per_frame, float)
    if M.ndim == 3:
        M = M[None]
    if ibm is not None:
        M = M @ np.asarray(ibm, float)[None]
    P = np.asarray(P, float)
    J = np.asarray(J).astype(int)
    W = np.asarray(W, float)
    Ph = np.c_[P, np.ones(len(P))]
    out = np.zeros((len(M), len(P), 3))
    for f in range(len(M)):
        X = out[f]
        for k in range(J.shape[1]):
            X += W[:, k:k + 1] * np.einsum("nij,nj->ni", M[f][J[:, k], :3], Ph)
    return out


# ---------------------------------------------------------------- bind metrics

def a2_far_weight(P, J, W, jpos, segs, H, names=None, D=None) -> dict:
    """Share of weight on bones more than 0.15H farther than the vertex's nearest bone, and the worst bones.
    share = far weight mass / total weight mass; verts_mostly_far = share of vertices with > half their weight far;
    worst = [(bone, share of all weight mass that is far weight on that bone)] top 5."""
    J = np.asarray(J).astype(int)
    W = np.asarray(W, float)
    if D is None:
        D = seg_dist(P, jpos, segs)
    H = max(float(H), 1e-9)
    Db = np.take_along_axis(D, J, 1)
    far = (Db - D.min(1)[:, None]) / H > FAR_REL
    fw = W * far
    tot = max(float(W.sum()), 1e-12)
    per = np.zeros(D.shape[1])
    np.add.at(per, J.reshape(-1), fw.reshape(-1))
    order = [int(k) for k in np.argsort(-per) if per[k] > 0][:5]
    rowsum = np.maximum(W.sum(1), 1e-12)
    return {"share": round(float(fw.sum() / tot), 5),
            "verts_mostly_far": round(float((fw.sum(1) / rowsum > 0.5).mean()), 5),
            "worst": [(names[k] if names is not None else k, round(float(per[k] / tot), 5)) for k in order]}


def foot_off_chain(P, J, W, names, legs, floor_rel: float = 0.08) -> dict:
    """Floor vertices (y < lo + 0.08H) whose strongest bone is NOT on a leg chain (ports bindcheck FOOT).
    legs: list of chains (lists of joint names or indices). Returns {share, floor_verts, owners [(name, n)]}."""
    P = np.asarray(P, float)
    H = max(float(np.ptp(P[:, 1])), 1e-9)
    fv = P[:, 1] < P[:, 1].min() + floor_rel * H
    legset = _chain_set(legs, names)
    top = top_bone(J, W)
    bad = fv & ~np.isin(top, list(legset))
    own = Counter((names[t] if names is not None else int(t)) for t in top[bad])
    return {"share": round(float(bad.sum() / max(int(fv.sum()), 1)), 4), "floor_verts": int(fv.sum()),
            "owners": own.most_common(5)}


def part_shares(P, J, W, chains, jpos=None, segs=None, names=None, D=None, masks=None) -> dict:
    """Per chain label, the weight share the region puts on its own bones. A label's region is the vertices whose
    geometrically nearest bone (seg_dist) is on that chain, so regions come from the chain labels, not boxes.
    masks: optional {label: (N,) bool}; a label's region is also held to its mask (a built ground bird's folded
    wings: the points on the body sides, rig_build._wing_side_margin > 0, not the rump under the feathers).
    chains: {label: names | indices | list of chains} (rig_fit.CHAINS[cls] shape; None/[] labels are skipped).
    Returns {label: {n, share, others {label: share}}}; 'body' collects bones on no chain."""
    if D is None:
        if jpos is None or segs is None:
            raise ValueError("part_shares needs jpos and segs (or a precomputed D) to find each chain's region")
        D = seg_dist(P, jpos, segs)
    J = np.asarray(J).astype(int)
    W = np.asarray(W, float)
    B = D.shape[1]
    label = np.array(["body"] * B, dtype=object)
    sets = {}
    for lab, ch in (chains or {}).items():
        s = _chain_set(ch, names)
        if s:
            sets[lab] = s
            for k in s:
                if k < B:
                    label[k] = lab
    nearest = D.argmin(1)
    LJ = label[J]
    out = {}
    for lab, s in sets.items():
        reg = np.isin(nearest, list(s))
        if masks and masks.get(lab) is not None:
            reg &= np.asarray(masks[lab], bool)
        if not reg.any():
            continue
        Wm, Lm = W[reg], LJ[reg]
        tot = max(float(Wm.sum()), 1e-12)
        others = {}
        for o in set(label):
            v = float((Wm * (Lm == o)).sum()) / tot
            if v > 0:
                others[o] = round(v, 4)
        out[lab] = {"n": int(reg.sum()), "share": round(float((Wm * (Lm == lab)).sum()) / tot, 4),
                    "others": dict(sorted(others.items(), key=lambda kv: -kv[1]))}
    return out


def pose_flags(jpos, segs, V, F, surface, H, names=None) -> dict:
    """Per joint BOX (outside the mesh box + 2% margin) / FAR (outside the mesh, |winding| < 0.5, and > 0.08H from
    the surface samples); ports tools_posecheck. Also per bone, the share of its 9 samples that are far outside
    (rig_fit.pose_gate's rule). Returns {joints: [{joint, pos, wn, d_rel, flags}], flagged: [...], bones: [...],
    joints_outside_box, bone_far_outside_0p08H}."""
    from scipy.spatial import cKDTree
    from rig_fit import winding_number
    jpos = np.asarray(jpos, float)
    V = np.asarray(V, float)
    F = np.asarray(F).reshape(-1, 3).astype(int)
    H = max(float(H), 1e-9)
    lo, hi = V.min(0), V.max(0)
    m = 0.02 * (hi - lo)
    tree = cKDTree(np.asarray(surface, float))
    box_out = ~np.all((jpos >= lo - m) & (jpos <= hi + m), axis=1)
    wn = winding_number(V, F, jpos)
    d = tree.query(jpos)[0] / H
    rows = []
    for k in range(len(jpos)):
        fl = (["BOX"] if box_out[k] else []) + (["FAR"] if abs(wn[k]) < 0.5 and d[k] > 0.08 else [])
        rows.append({"joint": names[k] if names is not None else k, "pos": jpos[k].round(4).tolist(),
                     "wn": round(float(wn[k]), 3), "d_rel": round(float(d[k]), 4), "flags": fl})
    bones, far_all = [], []
    if segs:
        t = np.linspace(0, 1, 9)
        pts = np.concatenate([jpos[a][None] * (1 - t[:, None]) + jpos[c][None] * t[:, None] for a, c in segs])
        step = max(1, int(np.ceil(len(F) * len(pts) / 4e7)))
        idx = np.arange(0, len(pts), step)
        q = pts[idx]
        far = (np.abs(winding_number(V, F, q)) < 0.5) & (tree.query(q)[0] / H > 0.08)
        far_all = far
        seg_of = idx // len(t)
        for i, (a, c) in enumerate(segs):
            s = seg_of == i
            if s.any():
                bones.append({"bone": names[c] if names is not None else int(c),
                              "far_share": round(float(far[s].mean()), 3)})
    return {"joints": rows, "flagged": [r for r in rows if r["flags"]], "bones": bones,
            "joints_outside_box": round(float(box_out.mean()), 3) if len(jpos) else 0.0,
            "bone_far_outside_0p08H": round(float(np.mean(far_all)), 3) if len(far_all) else 0.0}


# ---------------------------------------------------------------- motion metrics

def _frames(Vs) -> np.ndarray:
    Vs = np.asarray(Vs, float)
    return Vs[None] if Vs.ndim == 2 else Vs


def c_shape(Vs, V0) -> dict:
    """Bounding-box extent ratio vs rest, per axis. ext_ratio = [x, y, z] max over frames (growth),
    ext_ratio_min = [x, y, z] min over frames, ext_ratio_max = the worst axis, height_ratio_min / _max, and
    worst_frame (the frame of ext_ratio_max). Exactly 1 everywhere at rest."""
    Vs = _frames(Vs)
    e0 = np.maximum(np.ptp(np.asarray(V0, float), 0), 1e-9)
    r = np.ptp(Vs, 1) / e0[None]  # (F,3)
    mx = r.max(0)
    return {"ext_ratio": [round(float(x), 4) for x in mx], "ext_ratio_min": [round(float(x), 4) for x in r.min(0)],
            "ext_ratio_max": round(float(mx.max()), 4), "height_ratio_min": round(float(r[:, 1].min()), 4),
            "height_ratio_max": round(float(r[:, 1].max()), 4), "worst_frame": int(r.max(1).argmax())}


def foot_slide(foot_pos, contact, v_expected=0.0, fps: float = 30.0, H: float | None = None) -> dict:
    """Horizontal (xz) slide of a foot while it is in contact. foot_pos (F,3) world, contact (F,) bool, v_expected the
    foot's expected world velocity in units/second ((3,), (F,3) or scalar 0: a planted foot on a moving root has
    v = 0; on an in-place clip it is -walk speed). Per contiguous contact interval, the residual is
    pos - start - integral(v_expected); slide = the max residual distance from the interval start, over intervals.
    Units: divided by H when H is given (H-relative), raw units otherwise."""
    X = np.asarray(foot_pos, float)
    c = np.asarray(contact, bool).reshape(-1)
    v = np.broadcast_to(np.asarray(v_expected, float), X.shape)
    dt = 1.0 / float(fps)
    expect = np.r_[np.zeros((1, 3)), np.cumsum(v[:-1] * dt, 0)] if len(X) else np.zeros((0, 3))
    worst, span, n = 0.0, None, 0
    f = 0
    while f < len(c):
        if not c[f]:
            f += 1
            continue
        g = f
        while g + 1 < len(c) and c[g + 1]:
            g += 1
        n += 1
        res = (X[f:g + 1] - X[f]) - (expect[f:g + 1] - expect[f])
        d = float(np.linalg.norm(res[:, [0, 2]], axis=1).max())
        if d > worst:
            worst, span = d, [f, g]
        f = g + 1
    scale = max(float(H), 1e-9) if H is not None else 1.0
    return {"slide": round(worst / scale, 5), "intervals": n, "worst": span, "relative_to_H": H is not None}


def _qangle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    dot = np.clip(np.abs((a * b).sum(-1)), 0, 1)
    return np.degrees(2 * np.arccos(dot))


def loop_seam(clip: dict) -> float:
    """Degrees: the largest rotation between frame 0 and the last frame over all joints of a clip dict
    (bake_clips shape: rot {joint_index: (F,4) local xyzw}). Computed whatever clip['loop'] says; the caller gates."""
    worst = 0.0
    for q in (clip.get("rot") or {}).values():
        q = np.asarray(q, float)
        if len(q) >= 2:
            worst = max(worst, float(_qangle_deg(q[0], q[-1])))
    return round(worst, 4)


def stretch_owners(V0, Vt, edges, J_top, names, k: int = 5) -> dict:
    """Which bones own the edges above the p99 |log strain| (ports tools_stretchwho). Vt (F,N,3) or (N,3); edges
    (E,2) corner indices (edges_from_faces). Each bad edge is charged to the top bone of BOTH ends (once if they
    agree). Returns {p99 (strain ratio exp(thr)), max, stretch, squash, owners [(name, n)] top k,
    stretch_owners, squash_owners}."""
    V0 = np.asarray(V0, float)
    Vt = _frames(Vt)
    E = np.asarray(edges).astype(int).reshape(-1, 2)
    J_top = np.asarray(J_top).astype(int)
    if not len(E):
        return {"p99": 1.0, "max": 1.0, "stretch": 0, "squash": 0, "owners": [], "stretch_owners": [], "squash_owners": []}
    l0 = np.maximum(np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1), 1e-12)
    lt = np.linalg.norm(Vt[:, E[:, 0]] - Vt[:, E[:, 1]], axis=2)
    R_ = np.log(np.maximum(lt, 1e-12) / l0[None])
    S = np.abs(R_)
    thr = float(np.percentile(S, 99))
    bad = S > thr
    own, so, qo = Counter(), Counter(), Counter()
    n_st = n_sq = 0
    nm = (lambda b: names[b]) if names is not None else (lambda b: int(b))
    for f, e in zip(*np.nonzero(bad)):
        bs = {int(J_top[E[e, 0]]), int(J_top[E[e, 1]])}
        st = R_[f, e] > 0
        n_st += st
        n_sq += not st
        for b in bs:
            own[nm(b)] += 1
            (so if st else qo)[nm(b)] += 1
    return {"p99": round(float(np.exp(thr)), 4), "max": round(float(np.exp(S.max())), 4), "stretch": int(n_st),
            "squash": int(n_sq), "owners": own.most_common(k), "stretch_owners": so.most_common(k),
            "squash_owners": qo.most_common(k)}


def frame_stats(Vs, V0, H) -> dict:
    """Per frame (lists of length F): height_rel (ptp y / H), long_axis_tilt_deg (angle between the first principal
    axis of the frame and of rest, sign-free), min_y_rel ((min y - rest min y) / H), max_disp_rel (max |V - V0| / H)."""
    Vs = _frames(Vs)
    V0 = np.asarray(V0, float)
    H = max(float(H), 1e-9)

    def axis(X):
        Xc = X - X.mean(0)
        return np.linalg.svd(Xc, full_matrices=False)[2][0]
    a0 = axis(V0)
    tilt = [float(np.degrees(np.arccos(np.clip(abs(axis(X) @ a0), 0, 1)))) for X in Vs]
    return {"height_rel": [round(float(x), 4) for x in np.ptp(Vs[:, :, 1], 1) / H],
            "long_axis_tilt_deg": [round(x, 2) for x in tilt],
            "min_y_rel": [round(float(x), 4) for x in (Vs[:, :, 1].min(1) - V0[:, 1].min()) / H],
            "max_disp_rel": [round(float(x), 4) for x in np.linalg.norm(Vs - V0[None], axis=2).max(1) / H]}
