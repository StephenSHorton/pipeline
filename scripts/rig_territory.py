#!/usr/bin/env python3
"""Weight territory: each limb owns only its own part of the mesh, so a swinging thigh can't carry the rump.

    from rig_territory import territory
    J, W, info = territory(P, F, J, W, names, parents, joint_pos)          # top-4 in, top-4 out

Learned weights (skin-tokens) on a template skeleton spill: on a sheep rigged with the horse template the hind
pelvis/thigh bones owned 22% of the mesh, most of it wool body above the legs, and the tail chain (fitted inside the
rump) owned the rump. Played, Walk lifted the rump by half the height and grew the body 17%.

1. Chains (classify): the spine line runs from the hips to the head; every branch off it is a leg (it reaches the
   floor, or its names say leg/foot/toe/hoof/paw), a tail (named tail, or hanging back from the hips), or other (a
   ribcage helper, a wing). The neck is the spine line after the shoulder (the last spine joint a leg hangs from, or
   the plan's neck_lift), with the head's whole subtree. The trunk (the receivers) is the spine line before the neck.
2. Removal (0..1 per vertex and chain, a function of rest position only, so copies of a corner agree):
   - legs: above the leg line (the mesh's leg band top: the lowest horizontal section that is body-wide, as the
     workbench's leg finder reads it; measured on the sheep the upper-leg joints sit 0.1-0.2x height above it,
     inside the wool, so the joints are only the fallback) the leg's weight fades out over LEG_BAND x height
     (smoothstep). Near the line, points far (in xz) from that leg's own column lose it too (a belly between the
     legs), from LEG_REACH[0] to LEG_REACH[1] x height. Below the line a leg keeps only points nearer its own
     column than any other leg's (legs don't grab each other).
   - neck (with the head): points behind the neck's root joint along the body's forward axis lose it over
     NECK_BAND x height (the chest stays with the chest).
   - tail: points in front of the tail's root joint lose it over TAIL_BAND, and points farther than TAIL_REACH x
     height from every tail bone lose it outright (a tail chain fitted inside the rump moves nothing).
   Removed weight goes to the trunk influences the point already has (in proportion), else to its nearest trunk
   bone.
3. Smoothing (smooth_weights): `smooth` iterations of Laplacian smoothing of the weight vectors over the welded
   mesh, then the removal is applied again, so it never bleeds past a limit or from one leg onto another.
Top-4, normalised; corners at the same point get the same weights (they are smoothed as one point).
"""
from __future__ import annotations

import re

import numpy as np

LEG_BAND = 0.06  # x height: leg weight fades out over this band above the leg line
LEG_REACH = (0.08, 0.16)  # x height, xz from the leg's column at the line: kept nearer, stripped farther (near the line)
NECK_BAND = 0.06  # x height behind the neck root joint
TAIL_BAND = 0.05
TAIL_REACH = 0.10  # x height from the tail bones
SMOOTH_ITERS = 6  # measured on the sheep: 0, 3, 6, 9, 12; 6 lowered every clip's stretch p99 below the unlimited rig
SMOOTH_ALPHA = 0.5
WELD_TOL = 0.002  # x height: corners this close are one point for smoothing
LEG_NAME = re.compile(r"leg|foot|toe|hoof|paw|ankle|knee|thigh|shin|calf|humerus|scapula|pelvis", re.I)
TAIL_NAME = re.compile(r"tail", re.I)
HEAD_NAME = re.compile(r"^head$", re.I)


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _children(parents: list[int]) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {k: [] for k in range(len(parents))}
    for k, p in enumerate(parents):
        if p is not None and p >= 0:
            ch[p].append(k)
    return ch


def _subtree(ch: dict[int, list[int]], k: int) -> list[int]:
    out, st = [], [k]
    while st:
        x = st.pop()
        out.append(x)
        st += ch[x]
    return out


def classify(names: list[str], parents: list[int], jpos: np.ndarray, neck_root: str | None = None) -> dict:
    """Chains of a skeleton by topology (names as hints). parents[k] is k's parent index (-1 for none).
    Returns {"hips", "head", "spine" (hips..head), "trunk" (receivers), "legs": [{"bones", "root", "attach"}],
    "neck": {"bones", "root", "attach"} | None, "tail": {...} | None, "other": [bones], "forward": unit xz vector}."""
    jpos = np.asarray(jpos, float)
    n = len(names)
    ch = _children(parents)
    roots = [k for k in range(n) if parents[k] is None or parents[k] < 0]
    if not roots:
        raise ValueError("no root joint")
    low = {k.lower(): i for i, k in enumerate(names)}
    hips = low.get("hips")
    if hips is None:
        hips = roots[0]
        while len(ch[hips]) == 1:
            hips = ch[hips][0]
    sub_h = set(_subtree(ch, hips))
    head = next((k for k in sub_h if HEAD_NAME.match(names[k])), None)
    if head is None:
        head = next((k for k in sub_h if "head" in names[k].lower() and not re.search(r"leaf|end|tip", names[k], re.I)), None)
    spine = [hips]
    if head is not None:
        path = [head]
        while path[-1] != hips:
            path.append(parents[path[-1]])
        spine = path[::-1]
    ylo, Hj = float(jpos[:, 1].min()), max(float(np.ptp(jpos[:, 1])), 1e-9)
    on_line = set(spine)
    fwd = jpos[head] - jpos[hips] if head is not None else np.array([0.0, 0.0, 1.0])
    fwd = np.array([fwd[0], 0.0, fwd[2]])
    fwd = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 1e-9 else np.array([0.0, 0.0, 1.0])
    legs, tails, other = [], [], []
    for si, s in enumerate(spine[:-1] if head is not None else spine):
        for c in ch[s]:
            if c in on_line:
                continue
            bones = _subtree(ch, c)
            reach_floor = (float(jpos[bones, 1].min()) - ylo) < 0.15 * Hj
            leg_named = any(LEG_NAME.search(names[b]) for b in bones)
            tail_named = any(TAIL_NAME.search(names[b]) for b in bones)
            behind = float((jpos[bones].mean(0) - jpos[hips]) @ fwd) < 0
            if tail_named or (not reach_floor and not leg_named and si == 0 and behind):
                tails.append({"bones": bones, "root": c, "attach": s})
            elif reach_floor or leg_named:
                legs.append({"bones": bones, "root": c, "attach": s, "spine_i": si})
            else:
                other += bones
    neck = None
    nr = low.get(neck_root.lower()) if neck_root else None
    if nr is None and head is not None:
        att = sorted({lg["spine_i"] for lg in legs})
        if len(att) >= 2:  # legs hang from two places on the spine: the neck starts after the front ones
            nr = spine[att[-1] + 1]
    if nr is not None and nr in on_line and nr != hips:
        i = spine.index(nr)
        bones = sorted(set(spine[i:]) | set(_subtree(ch, head)) | {b for s in spine[i:-1] for c in ch[s] if c not in on_line
                                                                  for b in _subtree(ch, c)})
        neck = {"bones": bones, "root": nr, "attach": spine[i - 1]}
        trunk = spine[:i]
    else:
        trunk = spine[:-1] if head is not None and len(spine) > 1 else spine
    tail = None
    if tails:
        tail = {"bones": sorted({b for t in tails for b in t["bones"]}), "root": tails[0]["root"], "attach": tails[0]["attach"]}
    for lg in legs:
        lg.pop("spine_i", None)
    return {"hips": hips, "head": head, "spine": spine, "trunk": trunk, "legs": legs, "neck": neck, "tail": tail,
            "other": sorted(other), "forward": fwd}


def mesh_leg_top(P: np.ndarray, F: np.ndarray | None) -> float | None:
    """The mesh's leg band top (absolute y), as the workbench leg finder reads it; None when it finds no leg band."""
    try:
        import trimesh
        try:
            from workbench import edit
        except ImportError:
            import edit  # type: ignore
        m = trimesh.Trimesh(np.asarray(P, float), np.asarray(F).reshape(-1, 3), process=True) if F is not None else None
        S = edit._surface(m) if m is not None and len(m.faces) else np.asarray(P, float)
        return float(edit.leg_top(m, S))
    except Exception:  # noqa: BLE001 - no leg band (a blob down to the floor), or no faces
        return None


def _seg_dist(X: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    L = float(ab @ ab)
    t = np.clip(((X - a) @ ab) / L, 0, 1) if L > 1e-12 else np.zeros(len(X))
    return np.linalg.norm(X - (a + t[:, None] * ab), axis=1)


def _bones_dist(X: np.ndarray, bones: list[int], parents: list[int], jpos: np.ndarray, within: set[int] | None = None) -> np.ndarray:
    """Distance from each point to the nearest of `bones`, each bone a segment from its joint to each child in
    `within` (or a point when it has none)."""
    within = set(bones) if within is None else within
    ch = _children(parents)
    out = np.full(len(X), np.inf)
    for k in bones:
        kids = [c for c in ch[k] if c in within]
        if not kids:
            out = np.minimum(out, np.linalg.norm(X - jpos[k], axis=1))
        for c in kids:
            out = np.minimum(out, _seg_dist(X, jpos[k], jpos[c]))
    return out


def _column(lg: dict, parents: list[int], jpos: np.ndarray, line: float) -> np.ndarray:
    """The leg's centre where it crosses the leg line (xyz on the line), from its bones; its lowest joint below."""
    ch = _children(parents)
    best = None
    for k in lg["bones"]:
        for c in ch[k]:
            a, b = jpos[k], jpos[c]
            if (a[1] - line) * (b[1] - line) <= 0 and abs(a[1] - b[1]) > 1e-9:
                t = (line - a[1]) / (b[1] - a[1])
                best = a + t * (b - a)
                break
        if best is not None:
            break
    if best is None:
        k = min(lg["bones"], key=lambda b: abs(jpos[b, 1] - line))
        best = np.array([jpos[k, 0], line, jpos[k, 2]])
    return best


def removal(P: np.ndarray, chains: dict, parents: list[int], jpos: np.ndarray, H: float, leg_line: float,
            leg_band: float = LEG_BAND, leg_reach: tuple[float, float] = LEG_REACH, neck_band: float = NECK_BAND,
            tail_band: float = TAIL_BAND, tail_reach: float = TAIL_REACH) -> list[tuple[str, list[int], np.ndarray]]:
    """Per chain (label, bones, removal 0..1 per point). Legs also carry the 'not mine' rule below the line."""
    P = np.asarray(P, float)
    out = []
    cols = [_column(lg, parents, jpos, leg_line) for lg in chains["legs"]]
    # below the line each point belongs to the leg whose part below the line (column down to the foot) is nearest
    if chains["legs"]:
        dl = []
        ch = _children(parents)
        for lg, c in zip(chains["legs"], cols):
            lowb = [k for k in lg["bones"] if jpos[k, 1] <= leg_line]
            d = np.full(len(P), np.inf)
            for k in lowb:
                d = np.minimum(d, np.linalg.norm(P - jpos[k], axis=1))
                for cc in ch[k]:
                    d = np.minimum(d, _seg_dist(P, jpos[k], jpos[cc]))
            top = min(lowb, key=lambda k: -jpos[k, 1]) if lowb else None
            if top is not None:
                d = np.minimum(d, _seg_dist(P, c, jpos[top]))
            dl.append(d)
        owner = np.argmin(np.stack(dl, 1), 1)
    near_line = smoothstep((P[:, 1] - (leg_line - leg_band * H)) / max(leg_band * H, 1e-9))
    for li, (lg, c) in enumerate(zip(chains["legs"], cols)):
        up = smoothstep((P[:, 1] - leg_line) / max(leg_band * H, 1e-9))
        dxz = np.linalg.norm((P - c)[:, [0, 2]], axis=1)
        far = smoothstep((dxz - leg_reach[0] * H) / max((leg_reach[1] - leg_reach[0]) * H, 1e-9)) * near_line
        r = np.maximum(up, far)
        if len(chains["legs"]) > 1:
            r = np.where((owner != li) & (P[:, 1] < leg_line + leg_band * H), 1.0, r)
        out.append(("leg", lg["bones"], r))
    fwd = np.asarray(chains["forward"], float)
    if chains.get("neck"):
        nk = chains["neck"]
        s = (P - jpos[nk["root"]]) @ fwd
        out.append(("neck", nk["bones"], smoothstep(-s / max(neck_band * H, 1e-9))))
    if chains.get("tail"):
        tl = chains["tail"]
        s = (P - jpos[tl["root"]]) @ fwd
        r = smoothstep(s / max(tail_band * H, 1e-9))
        d = _bones_dist(P, tl["bones"], parents, jpos)
        r = np.maximum(r, smoothstep((d - tail_reach * H) / max(0.5 * tail_reach * H, 1e-9)))
        out.append(("tail", tl["bones"], r))
    return out


def blocked(P: np.ndarray, names: list[str], parents: list[int], jpos: np.ndarray, leg_line: float,
            neck_root: str | None = None, **bands) -> tuple[np.ndarray, dict]:
    """(points, joints) bool: the joints each point may not follow (its removal for that joint's chain is at least
    a half), for checks that judge weights against the nearest bone (rig_inspect) and for a weight editor's limits.
    Returns it with the chains."""
    P = np.asarray(P, float)
    jpos = np.asarray(jpos, float)
    parents = [int(p) if p is not None else -1 for p in parents]
    chains = classify(names, parents, jpos, neck_root)
    H = max(float(np.ptp(P[:, 1])), 1e-9)
    out = np.zeros((len(P), len(names)), bool)
    for _, bones, r in removal(P, chains, parents, jpos, H, float(leg_line), **bands):
        out[:, bones] |= (r >= 0.5)[:, None]
    return out, chains


def _apply(M: np.ndarray, rem: list, trunk: list[int], near_trunk: np.ndarray) -> dict:
    """Strip each chain's weight by its removal and hand it to the point's trunk influences (or nearest trunk bone)."""
    moved = {}
    take = np.zeros(len(M))
    for label, bones, r in rem:
        x = M[:, bones] * r[:, None]
        M[:, bones] -= x
        take += x.sum(1)
        moved[label] = moved.get(label, 0.0) + float(x.sum())
    T = M[:, trunk].sum(1)
    has = T > 1e-6
    M[np.ix_(has, trunk)] += M[np.ix_(has, trunk)] * (take[has] / T[has])[:, None]
    rows = np.nonzero(~has & (take > 0))[0]
    M[rows, near_trunk[rows]] += take[rows]
    return moved


def weld(P: np.ndarray, F: np.ndarray | None, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Each corner's welded point (corners within tol of each other, chained) and the welded edges (unique pairs).
    A decimated mesh keeps near-copies of a point (0.00001-0.0002x height apart) that a 1e-5 grid doesn't join: as
    separate points the smallest weight difference between them reads as a 100x stretch."""
    P = np.asarray(P, float)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(P).query_pairs(max(tol, 1e-9), output_type="ndarray")
    n = len(P)
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        inv = connected_components(g, directed=False)[1]
    else:
        inv = np.arange(n)
    if F is None or not len(F):
        return inv, np.zeros((0, 2), int)
    Fw = inv[np.asarray(F).reshape(-1, 3)]
    E = np.sort(np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]], 1)
    E = np.unique(E[E[:, 0] != E[:, 1]], axis=0)
    return inv, E


def smooth_weights(M: np.ndarray, inv: np.ndarray, E: np.ndarray, iters: int = SMOOTH_ITERS, alpha: float = SMOOTH_ALPHA) -> np.ndarray:
    """Laplacian smoothing of per-corner dense weights over the welded mesh (uniform neighbours); copies of a point
    come out equal."""
    ng = int(inv.max()) + 1
    G = np.zeros((ng, M.shape[1]))
    np.add.at(G, inv, M)
    G /= np.bincount(inv, minlength=ng)[:, None]
    if len(E) and iters > 0:
        deg = np.bincount(E.ravel(), minlength=ng).astype(float)
        ok = deg > 0
        for _ in range(int(iters)):
            S = np.zeros_like(G)
            np.add.at(S, E[:, 0], G[E[:, 1]])
            np.add.at(S, E[:, 1], G[E[:, 0]])
            G[ok] = (1 - alpha) * G[ok] + alpha * S[ok] / deg[ok, None]
    return G[inv]


def _top4(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    J = np.argsort(-M, axis=1)[:, :4]
    W = np.take_along_axis(M, J, 1)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    J[W <= 0] = 0
    return J.astype(np.int64), W.astype(np.float32)


def territory(P: np.ndarray, F: np.ndarray | None, J: np.ndarray, W: np.ndarray, names: list[str], parents: list[int],
              jpos: np.ndarray, leg_top: float | None = None, neck_root: str | None = None, smooth: int = SMOOTH_ITERS,
              alpha: float = SMOOTH_ALPHA, chains: dict | None = None, weld_tol: float = WELD_TOL, **bands) -> tuple[np.ndarray, np.ndarray, dict]:
    """Limit each chain's weights to its territory, smooth, limit again. P (n,3) rest positions, F (m,3) faces or
    None (no smoothing), J/W top-k joint indices / weights, names/parents/jpos per joint (parents[k] = -1 for a
    root). leg_top: the leg line (absolute y; default read from the mesh, else the joints). neck_root: the plan's
    neck_lift joint name, if any. bands: leg_band, leg_reach, neck_band, tail_band, tail_reach (x height).
    Returns top-4 J, W and what was done."""
    P = np.asarray(P, float)
    jpos = np.asarray(jpos, float)
    parents = [int(p) if p is not None else -1 for p in parents]
    H = max(float(np.ptp(P[:, 1])), 1e-9)
    chains = chains or classify(names, parents, jpos, neck_root)
    info: dict = {"applied": False, "legs": len(chains["legs"]), "neck": bool(chains["neck"]), "tail": bool(chains["tail"])}
    if not chains["legs"] and not chains["neck"] and not chains["tail"]:
        info["why"] = "no limb chains"
        return np.asarray(J), np.asarray(W, np.float32), info
    how = "given"
    if leg_top is None:
        leg_top, how = mesh_leg_top(P, F), "mesh leg band"
    if leg_top is None and chains["legs"]:  # fallback: the lowest joint of each leg's root bone's child, averaged
        ch = _children(parents)
        tops = [jpos[ch[lg["root"]][0], 1] if ch[lg["root"]] else jpos[lg["root"], 1] for lg in chains["legs"]]
        leg_top, how = float(np.mean(tops)), "upper-leg joints"
    leg_top = float(leg_top if leg_top is not None else P[:, 1].min())
    n = max(len(names), int(np.max(J)) + 1)
    M = np.zeros((len(J), n))
    np.add.at(M, (np.repeat(np.arange(len(J)), J.shape[1]), np.asarray(J).reshape(-1)), np.asarray(W, float).reshape(-1))
    M /= np.maximum(M.sum(1, keepdims=True), 1e-12)
    trunk = list(chains["trunk"])
    near_trunk = np.array(trunk)[np.argmin(np.stack([_bones_dist(P, [t], parents, jpos, set(trunk)) for t in trunk], 1), 1)]
    rem = removal(P, chains, parents, jpos, H, leg_top, **bands)
    limb = sorted({b for _, bones, _ in rem for b in bones})
    before = float(M[:, limb].sum())
    moved = _apply(M, rem, trunk, near_trunk)
    inv, E = weld(P, F, weld_tol * H)
    if smooth and len(E):
        M = smooth_weights(M, inv, E, smooth, alpha)
        again = _apply(M, rem, trunk, near_trunk)
        moved = {k: moved.get(k, 0.0) + again.get(k, 0.0) for k in set(moved) | set(again)}
    else:  # still give copies of a point the same weights
        M = smooth_weights(M, inv, E, 0)
    J2, W2 = _top4(M)
    info.update(applied=True, leg_line=round(leg_top, 5), leg_line_rel=round((leg_top - float(P[:, 1].min())) / H, 4),
                leg_line_from=how, smooth=int(smooth) if len(E) else 0, alpha=alpha,
                moved={k: round(v, 2) for k, v in moved.items()},
                limb_share_before=round(before / len(P), 4), limb_share_after=round(float(M[:, limb].sum()) / len(P), 4),
                trunk=[names[k] for k in trunk], neck_root=names[chains["neck"]["root"]] if chains["neck"] else None,
                tail_root=names[chains["tail"]["root"]] if chains["tail"] else None,
                leg_roots=[names[lg["root"]] for lg in chains["legs"]], bands={k: v for k, v in bands.items()})
    return J2, W2, info
