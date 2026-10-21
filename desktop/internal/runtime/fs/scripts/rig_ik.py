#!/usr/bin/env python3
"""IK and curve maths for synthesised clips (numpy and scipy only).

Conventions (the same as rig.py): positions are world, rotations are scipy `Rotation`, quaternions are xyzw.
Built rigs have identity rest rotations, so a joint's "world delta" is its world rotation: a clip's local rotation
is `local_i = delta_parent^-1 * delta_i` (rig.py bake_clips).

    fk(parent, rest_t, local_q, root_t)              -> (pos (J,3), rot Rotation of length J)
    two_bone_3d(pA, pB, pK, target, pole)            -> (qA, qB) world deltas (Rotation) for joints A and B
    fabrik(points, target, iters=12, plane_normal)   -> points (n,3), lengths kept, points[0] fixed
    aim(rot_world, from_dir, to_dir)                 -> Rotation: rot_world turned so its local from_dir points along to_dir
    min_jerk(x), ease_io(x)                          -> 0..1 easing of x in [0, 1] (clipped)
    fourier(phi, coeffs)                             -> sum_k a_k cos(2 pi k phi) + b_k sin(2 pi k phi)
    mirror_delta(q)                                  -> S R S with S = diag(-1, 1, 1)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

REACH = 0.98  # two_bone_3d never straightens a limb past this share of its length
EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def _any_perp(u: np.ndarray) -> np.ndarray:
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    return _unit(a - (a @ u) * u)


def _as_rot(q) -> R:
    if isinstance(q, R):
        return q
    q = np.asarray(q, float)
    if q.shape[-2:] == (3, 3):
        return R.from_matrix(q)
    return R.from_quat(q)


# ---------------------------------------------------------------- forward kinematics

def fk(parent, rest_t, local_q, root_t=None) -> tuple[np.ndarray, R]:
    """World positions and rotations of a joint tree.

    parent: sequence (or dict) with parent[i] = index of i's parent, or -1/None for a root.
    rest_t: (J,3) local translations (the rest offsets from the parent; for a root, its rest position).
    local_q: (J,4) local rotations xyzw (identity rests: these are the clip's local rotations), or a stacked Rotation.
    root_t: optional (3,) translation that replaces rest_t of every root joint (a clip's root channel).
    Returns pos (J,3) and rot, a stacked Rotation of length J (rot[i] is joint i's world rotation).
    pos_i = pos_p + rot_p(rest_t_i), rot_i = rot_p * local_i, as rig.Skeleton.world (scale 1)."""
    rest_t = np.asarray(rest_t, float).reshape(-1, 3)
    n = len(rest_t)
    par = [(parent.get(i) if isinstance(parent, dict) else parent[i]) for i in range(n)]
    par = [-1 if p is None or p < 0 else int(p) for p in par]
    loc = local_q if isinstance(local_q, R) else R.from_quat(np.asarray(local_q, float).reshape(-1, 4))
    lm = loc.as_matrix()
    pos = np.zeros((n, 3))
    rot = np.zeros((n, 3, 3))
    done = np.zeros(n, bool)

    def visit(i: int, depth: int = 0) -> None:
        if done[i]:
            return
        if depth > n:
            raise ValueError("fk: the parent table has a cycle")
        p = par[i]
        if p < 0:
            pos[i] = rest_t[i] if root_t is None else np.asarray(root_t, float)
            rot[i] = lm[i]
        else:
            visit(p, depth + 1)
            pos[i] = pos[p] + rot[p] @ rest_t[i]
            rot[i] = rot[p] @ lm[i]
        done[i] = True

    for i in range(n):
        visit(i)
    return pos, R.from_matrix(rot)


# ---------------------------------------------------------------- two-bone IK

def _frame(d: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """Orthonormal frame (columns d, n, d x n) from a bone direction and the bend normal."""
    d = _unit(d)
    nrm = _unit(nrm - (nrm @ d) * d)
    if np.linalg.norm(nrm) < 0.5:
        nrm = _any_perp(d)
    return np.stack([d, nrm, np.cross(d, nrm)], 1)


def two_bone_solve(pA, pB, pK, target, pole) -> tuple[np.ndarray, np.ndarray]:
    """The solved positions (B', K') of a two-bone chain A-B-K with A fixed (see two_bone_3d)."""
    pA, pB, pK, target = (np.asarray(v, float) for v in (pA, pB, pK, target))
    la, lb = np.linalg.norm(pB - pA), np.linalg.norm(pK - pB)
    to = target - pA
    dist = np.linalg.norm(to)
    u = _unit(to) if dist > EPS else _unit(pK - pA)
    if np.linalg.norm(u) < 0.5:
        u = _unit(pB - pA)
    lo = abs(la - lb) * (1 + 1e-9) + 1e-12
    d = float(np.clip(dist, lo, REACH * (la + lb)))
    # bend direction: the pole projected off the reach axis; else the rest apex offset; else any perpendicular
    v = np.asarray(pole, float) - (np.asarray(pole, float) @ u) * u
    if np.linalg.norm(v) < 1e-6 * max(np.linalg.norm(pole), 1.0):
        ak = _unit(pK - pA)
        off = (pB - pA) - ((pB - pA) @ ak) * ak
        v = off - (off @ u) * u
    v = _unit(v)
    if np.linalg.norm(v) < 0.5:
        v = _any_perp(u)
    a = (la * la - lb * lb + d * d) / (2 * d)
    h = np.sqrt(max(la * la - a * a, 0.0))
    return pA + a * u + h * v, pA + d * u


def two_bone_3d(pA, pB, pK, target, pole) -> tuple[R, R]:
    """Analytic two-bone IK (law of cosines). A is the upper joint (fixed), B the middle (the apex), K the end.

    pA, pB, pK: rest world positions; target: where K should go; pole: a DIRECTION (e.g. [0, 0, -1] for a bird's
    intertarsal joint pointing back). The bend plane contains the A->target axis and the pole, and the apex B lands
    on the pole side. Reach is clamped to 0.98 (la + lb) (and to |la - lb| from below), so a target out of reach
    is approached along the same line.

    Returns (qA, qB): world rotation deltas from the rest pose (scipy Rotation). qA turns bone A-B onto A-B';
    qB is B's TOTAL world delta (not relative to qA) turning B-K onto B'-K'. Both keep a consistent roll: each
    maps the rest (bone, bend-normal) frame onto the solved one, so the bend normal is the hinge axis."""
    pA, pB, pK, target = (np.asarray(v, float) for v in (pA, pB, pK, target))
    nB, nK = two_bone_solve(pA, pB, pK, target, pole)
    n_rest = np.cross(pB - pA, pK - pB)
    if np.linalg.norm(n_rest) < 1e-9 * max(np.linalg.norm(pB - pA) * np.linalg.norm(pK - pB), EPS):
        # a straight rest chain bends toward the pole: same sign as the solved normal (v x u)
        u0 = _unit(pK - pA)
        v0 = np.asarray(pole, float) - (np.asarray(pole, float) @ u0) * u0
        if np.linalg.norm(v0) < 1e-9:
            v0 = _any_perp(u0)
        n_rest = np.cross(_unit(v0), u0)
    n_new = np.cross(nB - pA, nK - nB)
    if np.linalg.norm(n_new) < EPS:
        n_new = n_rest
    qA = R.from_matrix(_frame(nB - pA, n_new) @ _frame(pB - pA, n_rest).T)
    qB = R.from_matrix(_frame(nK - nB, n_new) @ _frame(pK - pB, n_rest).T)
    return qA, qB


# ---------------------------------------------------------------- FABRIK

def fabrik(points, target, iters: int = 12, plane_normal=None) -> np.ndarray:
    """FABRIK on a chain (n,3) with points[0] fixed. Bone lengths are kept exactly (the last pass runs from the
    root). With plane_normal, the chain and the target are first put in the plane through points[0] with that
    normal (directions projected, lengths kept), and the solve stays in it."""
    P = np.array(points, float).reshape(-1, 3)
    if len(P) < 2:
        return P
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)
    T = np.asarray(target, float).copy()
    base = P[0].copy()
    if plane_normal is not None:
        nrm = _unit(plane_normal)
        if np.linalg.norm(nrm) > 0.5:
            T = T - ((T - base) @ nrm) * nrm
            Q = [base]
            for k in range(len(L)):
                d = P[k + 1] - P[k]
                d = d - (d @ nrm) * nrm
                if np.linalg.norm(d) < EPS:
                    d = _any_perp(nrm)
                Q.append(Q[-1] + _unit(d) * L[k])
            P = np.array(Q)
    if np.linalg.norm(T - base) >= L.sum():
        d = _unit(T - base)
        if np.linalg.norm(d) < 0.5:
            d = _unit(P[-1] - base)
        return base + np.r_[0, np.cumsum(L)][:, None] * d[None]
    for _ in range(max(int(iters), 1)):
        P[-1] = T
        for k in range(len(P) - 2, -1, -1):
            P[k] = P[k + 1] + _unit(P[k] - P[k + 1]) * L[k] if np.linalg.norm(P[k] - P[k + 1]) > EPS else P[k]
        P[0] = base
        for k in range(len(P) - 1):
            d = P[k + 1] - P[k]
            P[k + 1] = P[k] + (_unit(d) if np.linalg.norm(d) > EPS else _unit(P[-1] - P[0])) * L[k]
        if np.linalg.norm(P[-1] - T) < 1e-9:
            break
    return P


# ---------------------------------------------------------------- aim, easing, curves

def aim(rot_world, from_dir, to_dir) -> R:
    """rot_world (Rotation, xyzw quat or 3x3) turned by the shortest arc so its LOCAL from_dir points along to_dir.
    Antiparallel directions turn 180 degrees about an axis perpendicular to them."""
    rw = _as_rot(rot_world)
    a = _unit(rw.apply(np.asarray(from_dir, float)))
    b = _unit(to_dir)
    c = np.cross(a, b)
    s, cos = np.linalg.norm(c), float(np.clip(a @ b, -1, 1))
    if s < 1e-9:
        if cos > 0:
            return rw
        return R.from_rotvec(_any_perp(a) * np.pi) * rw
    return R.from_rotvec(c / s * np.arctan2(s, cos)) * rw


def min_jerk(x):
    """Minimum-jerk 0 -> 1: 10x^3 - 15x^4 + 6x^5, x clipped to [0, 1]."""
    x = np.clip(np.asarray(x, float), 0, 1)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def ease_io(x):
    """Sine ease-in-out 0 -> 1: (1 - cos(pi x)) / 2, x clipped to [0, 1]."""
    x = np.clip(np.asarray(x, float), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def fourier(phi, coeffs: Sequence) -> np.ndarray:
    """A periodic curve in phase phi (cycles; period 1): sum over k of a_k cos(2 pi k phi) + b_k sin(2 pi k phi).

    coeffs[k] = (a_k, b_k) for k = 0, 1, 2, ... (b_0 is ignored, a_0 is the mean); a bare number at index k means
    (a_k, 0). A dict {k: (a_k, b_k)} also works. f(phi + 1) == f(phi) exactly, so loops close."""
    phi = np.asarray(phi, float)
    items = coeffs.items() if isinstance(coeffs, dict) else enumerate(coeffs)
    out = np.zeros_like(phi)
    for k, c in items:
        a, b = (float(c), 0.0) if np.ndim(c) == 0 else (float(c[0]), float(c[1]) if len(c) > 1 else 0.0)
        w = 2 * np.pi * int(k) * phi
        out = out + a * np.cos(w) + (b * np.sin(w) if k else 0.0)
    return out


S = np.diag([-1.0, 1.0, 1.0])


def mirror_delta(q):
    """Reflect a world rotation delta through the sagittal plane (x -> -x): S R S, S = diag(-1, 1, 1).
    The rotation about lateral X is kept; the rotations about up Y and forward Z are negated
    (quaternion (x, y, z, w) -> (x, -y, -z, w)). Takes a Rotation (returns a Rotation) or xyzw quats (..., 4)
    (returns the same shape)."""
    if isinstance(q, R):
        return R.from_quat(q.as_quat() * np.array([1.0, -1.0, -1.0, 1.0]))
    q = np.asarray(q, float)
    if q.shape[-2:] == (3, 3):
        return S @ q @ S
    return q * np.array([1.0, -1.0, -1.0, 1.0])
