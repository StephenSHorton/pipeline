#!/usr/bin/env python3
"""Clip generators for built templates (design sections 2.6 and 6): the kinematic core they share (a joint tree with
identity rests, a per-frame pose track, leg IK, neck FABRIK) plus one generator per plan `gen`.

Conventions
- Template space: +Y up, +Z forward (head), +X the creature's LEFT, floor at y = 0. Built rigs have identity rest
  rotations, so a joint's local frame at rest IS the body frame: a local rotation "about +X" is a pitch about the
  lateral axis in the parent's frame.
- pitch: about +X, positive = nose down. yaw: about +Y (positive turns the head toward +X). roll: about +Z, positive
  rolls onto the creature's right (its left side rises).
- A Track holds local rotation matrices (F, J, 3, 3) and the root joint's offset from rest (F, 3). Generators turn
  joints (`turn`: local = R * local, i.e. about body axes in the parent's frame) and then call the per-frame solvers,
  which set world rotations through local_i = world_parent^-1 * want_i (rig.py bake_clips).
- R side curves are the L curves mirrored (rig_ik.mirror_delta: S R S, S = diag(-1, 1, 1)), at phase + 0.5 for gaits.
- Looping clips are periodic in phi = t / T with T snapped to whole frames (gaits: an even count, so phase + 0.5 lands
  on a frame), so frame F == frame 0.
- Every amplitude is multiplied by the energy e; timing and contacts are not.

GENERATORS maps each gen name to fn(ctx, spec, e) -> Track.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as R

import rig_ik

FPS = 30
X, Y, Z = np.eye(3)
S = np.diag([-1.0, 1.0, 1.0])
EYE = np.eye(3)

# Gait phase tables (per-leg phase offsets, as fractions of the cycle). The biped walk/run use L 0, R 0.5; the
# quadruped tables are for later plans (lateral-sequence walk, trot, pace).
GAITS = {
    "biped": {"L": 0.0, "R": 0.5},
    "walk_ls": {"LH": 0.0, "LF": 0.25, "RH": 0.5, "RF": 0.75},
    "trot": {"LH": 0.0, "RF": 0.0, "RH": 0.5, "LF": 0.5},
    "pace": {"LH": 0.0, "LF": 0.0, "RH": 0.5, "RF": 0.5},
}


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def rot_axis(axis, deg) -> np.ndarray:
    """(F,3,3) rotations about a fixed axis by deg (scalar or (F,))."""
    a = np.atleast_1d(np.radians(np.asarray(deg, float)))
    return R.from_rotvec(np.outer(a, unit(axis))).as_matrix()


def mirror_m(M: np.ndarray) -> np.ndarray:
    """S M S on (...,3,3): the sagittal mirror of a rotation (lateral component kept, up/forward negated)."""
    return S @ M @ S


def flex_deg(a, b, c) -> float:
    """Angle (degrees) between bone a->b and bone b->c: 0 = straight."""
    u, v = unit(np.subtract(b, a)), unit(np.subtract(c, b))
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def ease_out(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    return 1 - (1 - x) ** 2


def pulse(t, t0, dur):
    """A sine pulse: 0 outside [t0, t0+dur], sin(pi u) inside (0 at both ends)."""
    u = (np.asarray(t, float) - t0) / dur
    return np.where((u > 0) & (u < 1), np.sin(np.pi * np.clip(u, 0, 1)), 0.0)


def bump(u, peak: float = 0.5):
    """0 -> 1 -> 0 over u in [0, 1], peaking at u = peak, with zero slope at both ends and at the peak."""
    u = np.clip(np.asarray(u, float), 0, 1)
    p = float(np.clip(peak, 0.05, 0.95))
    return np.where(u < p, rig_ik.ease_io(u / p), rig_ik.ease_io((1 - u) / (1 - p)))


def hermite(u, p0, p1, m0, m1):
    u = np.asarray(u, float)
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def snap_frames(T: float, even: bool = False) -> int:
    n = max(2, int(round(float(T) * FPS)))
    if even and n % 2:
        n += 1
    return n


# ---------------------------------------------------------------- the joint tree

class Skel:
    """A built skeleton with identity rests: names, parent indices, local rest translations and world rest positions.

    Accepts {names, parent (indices or names), rest_t | pos | jpos} or skeleton.json ({joints: [{name, parent, pos}]}).
    `order` lists parents before children."""

    def __init__(self, skel: dict, H: float | None = None, cx: float | None = None) -> None:
        if "joints" in skel:
            names = [j["name"] for j in skel["joints"]]
            k = {n: i for i, n in enumerate(names)}
            par = [k[j["parent"]] if j.get("parent") is not None else -1 for j in skel["joints"]]
            world = np.array([j["pos"] for j in skel["joints"]], float)
            rest_t = None
        else:
            names = list(skel["names"])
            k = {n: i for i, n in enumerate(names)}
            par = []
            for p in skel["parent"]:
                if p is None or (not isinstance(p, str) and int(p) < 0):
                    par.append(-1)
                else:
                    par.append(k[p] if isinstance(p, str) else int(p))
            world = None
            for key in ("pos", "jpos", "rest_world"):
                if key in skel:
                    world = np.asarray(skel[key], float).reshape(-1, 3)
            rest_t = None if world is not None else np.asarray(skel["rest_t"], float).reshape(-1, 3)
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        self.parent = par
        n = len(names)
        order, seen = [], set()

        def visit(i: int) -> None:
            if i in seen:
                return
            if par[i] >= 0:
                visit(par[i])
            seen.add(i)
            order.append(i)
        for i in range(n):
            visit(i)
        self.order = order
        if world is None:
            world = np.zeros((n, 3))
            for i in order:
                world[i] = rest_t[i] + (world[par[i]] if par[i] >= 0 else 0.0)
        self.rest = world
        self.rest_t = np.array([world[i] - (world[par[i]] if par[i] >= 0 else 0.0) for i in range(n)])
        self.children = {i: [c for c in range(n) if par[c] == i] for i in range(n)}
        self.root = next(i for i in order if par[i] < 0)
        self.H = float(H if H is not None else (skel.get("H") or np.ptp(world[:, 1])))
        self.cx = float(cx if cx is not None else (skel.get("cx") if skel.get("cx") is not None else world[self.root][0]))

    def has(self, name: str) -> bool:
        return name in self.idx

    def i(self, name: str) -> int:
        return self.idx[name]

    def subtree(self, name: str) -> list[int]:
        out, st = [], [self.idx[name]]
        while st:
            x = st.pop()
            out.append(x)
            st += self.children[x]
        return out

    def chain_len(self, names: list[str]) -> float:
        p = self.rest[[self.idx[n] for n in names]]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


# ---------------------------------------------------------------- a clip being authored

class Track:
    """F = n + 1 frames at 30 fps (the last frame is t = T). local (F,J,3,3), root_off (F,3) (root translation minus
    its rest), contact {key: bool (F,)}, info {..} for the gates (v, hold masks, ...)."""

    def __init__(self, sk: Skel, n: int, loop: bool = True) -> None:
        self.sk = sk
        self.n = int(n)
        self.times = np.arange(self.n + 1) / FPS
        self.T = self.n / FPS
        self.phi = self.times / self.T
        self.loop = loop
        F, J = self.n + 1, len(sk.names)
        self.local = np.broadcast_to(EYE, (F, J, 3, 3)).copy()
        self.root_off = np.zeros((F, 3))
        self.contact: dict[str, np.ndarray] = {}
        self.info: dict = {}

    @property
    def F(self) -> int:
        return self.n + 1

    def turn(self, name: str, axis, deg, frames=slice(None)) -> None:
        """local = R(axis, deg) * local for one joint (no-op when the joint is missing)."""
        if not self.sk.has(name):
            return
        i = self.sk.i(name)
        M = rot_axis(axis, deg)
        if len(M) == 1:
            M = np.broadcast_to(M[0], self.local[frames, i].shape)
        self.local[frames, i] = M @ self.local[frames, i]

    def turn_m(self, name: str, M: np.ndarray) -> None:
        if self.sk.has(name):
            i = self.sk.i(name)
            self.local[:, i] = M @ self.local[:, i]

    def sided(self, base: str, sides: tuple[str, str], axis_L, deg_fn, offset: float = 0.0) -> None:
        """L: about axis_L by deg_fn(phi); R: the mirror of L's rotation at phi + offset (gaits: 0.5)."""
        L, Rr = base + sides[0], base + sides[1]
        self.turn_m(L, rot_axis(axis_L, deg_fn(self.phi)))
        self.turn_m(Rr, mirror_m(rot_axis(axis_L, deg_fn((self.phi + offset) % 1.0 if self.loop else self.phi))))

    def fk(self) -> tuple[np.ndarray, np.ndarray]:
        sk = self.sk
        F, J = self.F, len(sk.names)
        pos = np.zeros((F, J, 3))
        rot = np.zeros((F, J, 3, 3))
        for i in sk.order:
            p = sk.parent[i]
            if p < 0:
                pos[:, i] = sk.rest[i] + self.root_off
                rot[:, i] = self.local[:, i]
            else:
                pos[:, i] = pos[:, p] + np.einsum("fij,j->fi", rot[:, p], sk.rest_t[i])
                rot[:, i] = rot[:, p] @ self.local[:, i]
        return pos, rot

    def fk_frame(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        sk = self.sk
        J = len(sk.names)
        pos = np.zeros((J, 3))
        rot = np.zeros((J, 3, 3))
        L = self.local[f]
        for i in sk.order:
            p = sk.parent[i]
            if p < 0:
                pos[i] = sk.rest[i] + self.root_off[f]
                rot[i] = L[i]
            else:
                pos[i] = pos[p] + rot[p] @ sk.rest_t[i]
                rot[i] = rot[p] @ L[i]
        return pos, rot

    def quats(self) -> dict[int, np.ndarray]:
        F, J = self.F, len(self.sk.names)
        q = R.from_matrix(self.local.reshape(-1, 3, 3)).as_quat().reshape(F, J, 4)
        # a consistent hemisphere per joint (LINEAR sampling slerps the short way either way, but this keeps files tidy)
        for j in range(J):
            for f in range(1, F):
                if q[f, j] @ q[f - 1, j] < 0:
                    q[f, j] = -q[f, j]
        return {j: q[:, j].astype(np.float32) for j in range(J)}


def aim_m(W: np.ndarray, from_local, to_world) -> np.ndarray:
    """W turned (shortest arc) so its local from_local points along to_world (3x3 in, 3x3 out)."""
    if np.linalg.norm(to_world) < 1e-12 or np.linalg.norm(from_local) < 1e-12:
        return W
    return rig_ik.aim(W, from_local, to_world).as_matrix()


# ---------------------------------------------------------------- the creature: legs, neck, wings, vertex sets

class Ctx:
    """Everything a generator needs about the creature: the skeleton, the plan's chains and the bound mesh."""

    def __init__(self, plan: dict, sk: Skel, P: np.ndarray, Fc: np.ndarray, J: np.ndarray, W: np.ndarray, H: float) -> None:
        import rig_metrics
        self.plan = plan
        self.sk = sk
        self.P = np.asarray(P, float)
        self.Fc = np.asarray(Fc).reshape(-1, 3).astype(int) if Fc is not None else np.zeros((0, 3), int)
        self.J = np.asarray(J).astype(int)
        self.W = np.asarray(W, float)
        self.H = float(H)
        self.floor = float(self.P[:, 1].min()) if len(self.P) else 0.0
        self.top = rig_metrics.top_bone(self.J, self.W) if len(self.P) else np.zeros(0, int)
        self.sides = tuple(plan.get("side_suffix") or ("_L", "_R"))
        lk = plan.get("leg_ik") or {}
        self.leg_keys = {k: lk.get(k, d) for k, d in (("femur", "UpperLeg"), ("A", "LowerLeg"), ("B", "AnkleLeg"),
                                                       ("K", "Foot"))}
        self.pole = unit(lk.get("pole", [0, 0, -1]))
        flat = list(lk.get("flat") or ["Foot", "Toes"])
        self.toes_key = flat[1] if len(flat) > 1 else "Toes"
        self.legs = []
        for s, sgn in zip(self.sides, (1.0, -1.0)):
            nm = {k: v + s for k, v in self.leg_keys.items()}
            nm["T"] = self.toes_key + s
            nm["tip"] = self.toes_key + "_tip" + s
            if all(sk.has(nm[k]) for k in ("femur", "A", "B", "K")):
                ix = {k: sk.i(v) for k, v in nm.items() if sk.has(v)}
                la = float(np.linalg.norm(sk.rest[ix["B"]] - sk.rest[ix["A"]]))
                lb = float(np.linalg.norm(sk.rest[ix["K"]] - sk.rest[ix["B"]]))
                # the bend direction: the leg's own rest apex offset (so a rest target solves to the rest pose), when
                # it agrees with the plan's pole (-Z: the intertarsal joint points back)
                pa, pb, pk = sk.rest[ix["A"]], sk.rest[ix["B"]], sk.rest[ix["K"]]
                ax = unit(pk - pa)
                off = (pb - pa) - ((pb - pa) @ ax) * ax
                pole = unit(off) if np.linalg.norm(off) > 1e-6 * max(la, 1e-9) and unit(off) @ self.pole > 0.3 else self.pole
                self.legs.append({"side": s[-1], "sfx": s, "sign": sgn, "names": nm, "ix": ix, "la": la, "lb": lb,
                                  "pole": pole})
        neck = list(plan.get("neck_ik") or ["spine_2", "spine_3", "head", "mouth_upper_tip"])
        self.neck = [n for n in neck if sk.has(n)] if all(sk.has(n) for n in neck) else []
        ch = plan.get("chains") or {}
        self.spine = [n for n in ((ch.get("spine") or {}).get("names") or ["spine_0", "spine_1"]) if sk.has(n)]
        tail = ch.get("tail") or {}
        self.tail = [n for n in (tail.get("names") if isinstance(tail, dict) else tail or ["tail_1", "tail_2", "tail_3", "tail_tip"])
                     if sk.has(n)]
        jaw = ch.get("jaw") or {}
        jn = jaw.get("names") if isinstance(jaw, dict) else jaw
        self.jaw = (jn or ["mouth_lower"])[0] if sk.has((jn or ["mouth_lower"])[0]) else None
        wings = ch.get("wings") or {}
        wn = wings.get("names") if isinstance(wings, dict) else None
        self.wing = [n for n in (wn or ["wing_1", "wing_2", "wing_3", "wing_4", "wing_5", "wing_tip"])
                     if sk.has(n + self.sides[0]) and sk.has(n + self.sides[1])]
        self.hips = "hips" if sk.has("hips") else sk.names[sk.children[sk.root][0]]
        self.head = "head" if sk.has("head") else (self.neck[-2] if len(self.neck) > 1 else None)
        self.tip = self.neck[-1] if self.neck else None
        # vertex sets. A bird's thighs sit inside the body contour, so the skin gives the femurs the lower belly: the
        # belly (what Sleep rests on) is the torso's AND the femurs' points (the drawn chicken's torso-only underside
        # is 0.30H off the floor; its femur-owned underside is the real belly).
        torso = {sk.i(n) for n in [sk.names[sk.root], self.hips] + self.spine}
        torso |= {lg["ix"]["femur"] for lg in self.legs}
        self.belly_v = np.isin(self.top, sorted(torso))
        self.foot_v = {}
        for lg in self.legs:
            ids = sk.subtree(lg["names"]["K"])
            self.foot_v[lg["names"]["K"]] = np.isin(self.top, ids)
            lg["verts"] = np.isin(self.top, sk.subtree(lg["names"]["A"]))

    # ------------------------------------------------ axes

    def raise_axis(self, names: list[str]) -> np.ndarray:
        """The axis whose positive rotation lifts the chain's tip toward +Y (a back-pointing tail or wing: +X)."""
        p = self.sk.rest[[self.sk.i(n) for n in names]]
        a = np.cross(unit(p[-1] - p[0]), Y)
        return unit(a) if np.linalg.norm(a) > 1e-6 else X

    def abduct_axis(self, names: list[str]) -> np.ndarray:
        """Wing abduction is about the body's forward axis (+Z; positive lifts the L wing's lower edge out and up), as
        for an arm hanging at the side. On a folded wing (bones along -Z) this swings the wing surface out from the
        flank about the shoulder; on a spread wing (bones along +X) it raises the wing. Swinging a folded wing's tip
        out sideways (about Y) instead tears the flank: 8x stretch at 60 degrees on the synthetic bird."""
        return Z.copy()

    # ------------------------------------------------ geometry helpers

    def knee_reach(self, lg: dict, r: float = 0.95) -> float:
        return r * (lg["la"] + lg["lb"])

    def d_for_flex(self, lg: dict, flex: float) -> float:
        """Distance A..K when the B joint is flexed by `flex` degrees (0 = straight)."""
        la, lb = lg["la"], lg["lb"]
        return float(math.sqrt(max(la * la + lb * lb + 2 * la * lb * math.cos(math.radians(flex)), 1e-12)))

    def rest_flex(self, lg: dict) -> float:
        ix = lg["ix"]
        return flex_deg(self.sk.rest[ix["A"]], self.sk.rest[ix["B"]], self.sk.rest[ix["K"]])

    # ------------------------------------------------ per-frame solvers

    def crouch_for(self, tr: Track, f: int, targets: dict[str, np.ndarray], r: float = 0.96) -> float:
        """How far the root must drop at frame f so every leg's A joint is within r x reach of its foot target."""
        pos, _ = tr.fk_frame(f)
        need = 0.0
        for lg in self.legs:
            t = targets.get(lg["side"])
            if t is None:
                continue
            d = pos[lg["ix"]["A"]] - t
            reach = r * (lg["la"] + lg["lb"])
            h2 = d[0] ** 2 + d[2] ** 2
            if h2 >= reach * reach:
                continue
            dy_ok = math.sqrt(reach * reach - h2)
            if d[1] > dy_ok:
                need = max(need, float(d[1] - dy_ok))
        return need

    def solve_legs(self, tr: Track, f: int, legs: dict[str, dict], crouch: bool = True, guard: bool = True) -> float:
        """Two-bone IK per leg at frame f: legs[side] = {target (3,), foot (3x3 world), toes (3x3 world | None),
        pole (3,) | None}. Returns the crouch applied (root drop) so the targets stay in reach. With guard, a leg whose
        skin goes below the floor has its target raised by the overlap and is solved again (up to 3 rounds)."""
        drop = 0.0
        if crouch:
            drop = self.crouch_for(tr, f, {s: v["target"] for s, v in legs.items()})
            if drop > 0:
                tr.root_off[f, 1] -= drop
        for lg in self.legs:
            spec = legs.get(lg["side"])
            if spec is None:
                continue
            target = np.asarray(spec["target"], float).copy()
            for _ in range(4 if guard else 1):
                self._solve_leg(tr, f, lg, spec, target)
                if not guard or not lg["verts"].any():
                    break
                pos, rot = tr.fk_frame(f)
                lo = float(self.skin(pos, rot, lg["verts"])[:, 1].min()) - self.floor
                if lo >= -1e-4 * self.H:
                    break
                target[1] += -lo + 1e-4 * self.H
        return drop

    def _solve_leg(self, tr: Track, f: int, lg: dict, spec: dict, target: np.ndarray) -> None:
        pos, rot = tr.fk_frame(f)
        ix = lg["ix"]
        A, B, K = ix["A"], ix["B"], ix["K"]
        pole = spec.get("pole")
        pole = lg["pole"] if pole is None else pole
        qA, qB = rig_ik.two_bone_3d(pos[A], pos[B], pos[K], target, pole)
        WA = qA.as_matrix() @ rot[A]
        WB = qB.as_matrix() @ rot[B]
        pa = self.sk.parent[A]
        tr.local[f, A] = rot[pa].T @ WA
        tr.local[f, B] = WA.T @ WB
        footW = spec.get("foot")
        if footW is not None:
            tr.local[f, K] = WB.T @ footW
            if spec.get("toes") is not None and "T" in ix:
                tr.local[f, ix["T"]] = footW.T @ spec["toes"]

    def place_leg(self, tr: Track, f: int, lg: dict, ankle: np.ndarray, foot: np.ndarray, footW, toesW=None) -> None:
        """Aim A->B at `ankle` and B->K at `foot` (exact placement, for lying poses)."""
        pos, rot = tr.fk_frame(f)
        ix = lg["ix"]
        A, B, K = ix["A"], ix["B"], ix["K"]
        WA = aim_m(rot[A], self.sk.rest_t[B], ankle - pos[A])
        WB = aim_m(WA @ tr.local[f, B], self.sk.rest_t[K], foot - (pos[A] + WA @ self.sk.rest_t[B]))
        tr.local[f, A] = rot[self.sk.parent[A]].T @ WA
        tr.local[f, B] = WA.T @ WB
        if footW is not None:
            tr.local[f, K] = WB.T @ footW
            if toesW is not None and "T" in ix:
                tr.local[f, ix["T"]] = footW.T @ toesW

    def solve_neck(self, tr: Track, f: int, tip_target=None, head_target=None, head_world=None,
                   plane=X) -> np.ndarray:
        """FABRIK on the plan's neck_ik chain (spine_2, spine_3, head, mouth_upper_tip) at frame f.
        tip_target: where the beak tip goes; with head_world the head is set to that world rotation and FABRIK runs on
        the chain up to the head (to tip_target minus the head's own tip offset, or to head_target). Returns the solved
        points."""
        ch = self.neck
        if len(ch) < 3:
            return np.zeros((0, 3))
        sk = self.sk
        pos, rot = tr.fk_frame(f)
        idx = [sk.i(n) for n in ch]
        pts = pos[idx].copy()
        offl = [rot[idx[k]].T @ (pts[k + 1] - pts[k]) for k in range(len(idx) - 1)]
        if head_world is not None:
            if tip_target is not None:
                head_target = np.asarray(tip_target, float) - head_world @ offl[-1]
            work, goal = pts[:-1], head_target
        else:
            work, goal = pts, tip_target
        if goal is None:
            return pts
        base_axis = unit(work[-1] - work[0])
        bulge = unit(np.cross(base_axis, X if plane is None else plane))
        if bulge @ Z > 0:
            bulge = -bulge
        L = float(np.linalg.norm(np.diff(work, axis=0), axis=1).sum())
        init = work.copy()
        for k in range(1, len(init) - 1):
            init[k] = init[k] + bulge * 0.04 * L * math.sin(math.pi * k / (len(init) - 1))
        new = rig_ik.fabrik(init, goal, iters=24, plane_normal=plane)
        # FABRIK keeps the perturbed lengths; re-run from the true lengths starting at the solved shape
        new = rig_ik.fabrik(_relength(new, work), goal, iters=24, plane_normal=plane)
        Wpar = rot[sk.parent[idx[0]]]
        last = len(idx) - 2 if head_world is None else len(idx) - 3
        for k in range(len(idx) - 1):
            i = idx[k]
            Wcur = Wpar @ tr.local[f, i]
            if k <= last:
                Wnew = aim_m(Wcur, offl[k], new[k + 1] - new[k])
            elif head_world is not None and k == len(idx) - 2:
                Wnew = np.asarray(head_world, float)
            else:
                Wnew = Wcur
            tr.local[f, i] = Wpar.T @ Wnew
            Wpar = Wnew
        return new

    # ------------------------------------------------ skinning

    def skin(self, pos: np.ndarray, rot: np.ndarray, sel=None) -> np.ndarray:
        """LBS of the bound mesh (identity rests: M = [R | p - R p_rest]). pos (F,J,3) / rot (F,J,3,3) or one frame."""
        one = pos.ndim == 2
        if one:
            pos, rot = pos[None], rot[None]
        P, J, W = (self.P, self.J, self.W) if sel is None else (self.P[sel], self.J[sel], self.W[sel])
        t = pos - np.einsum("fbij,bj->fbi", rot, self.sk.rest)
        out = np.zeros((len(pos), len(P), 3))
        step = max(1, int(4e6 // max(len(P) * 9, 1)))
        for a in range(0, len(pos), step):
            b = min(a + step, len(pos))
            for k in range(J.shape[1]):
                jk = J[:, k]
                out[a:b] += W[None, :, k, None] * (np.einsum("fnij,nj->fni", rot[a:b][:, jk], P) + t[a:b][:, jk])
        return out[0] if one else out


def _relength(new: np.ndarray, ref: np.ndarray) -> np.ndarray:
    L = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    out = [new[0]]
    for k in range(len(L)):
        out.append(out[-1] + unit(new[k + 1] - new[k]) * L[k])
    return np.array(out)


# ---------------------------------------------------------------- shared leg machinery (Walk, Run)

def gait_legs(ctx: Ctx, tr: Track, p: dict, e: float, *, beta: float, stride_H: float, lift_H: float, lift_peak: float,
              femur: float, toe_curl: float, toe_off_pitch: float, crouch_H: float) -> dict:
    """Feet-on-the-floor gait for the two legs (L phase 0, R phase 0.5). Scripts the femur, then per frame solves the
    foot targets with two-bone IK (pole -Z) holding the foot world-flat in stance (toe-off pitch over the last 10% of
    stance with the toes flat, toes curled in swing). Stride is capped to what the IK chain can reach at the gait's
    crouch. Returns the gait's info (v, stride used, contact masks)."""
    sk, H = ctx.sk, ctx.H
    T = tr.T
    # constant posture first (crouch); the reach geometry is read from it
    tr.root_off[:, 1] -= crouch_H * H * e
    pos0, _ = tr.fk_frame(0)
    caps = []
    for lg in ctx.legs:
        ix = lg["ix"]
        dy = pos0[ix["A"]][1] - sk.rest[ix["K"]][1]
        r = 0.93 * (lg["la"] + lg["lb"])
        caps.append(2 * math.sqrt(max(r * r - dy * dy, 0.0)))
    s_want = stride_H * H * e
    s = min([s_want] + [0.85 * c for c in caps]) if caps else s_want
    v = s / (beta * T)
    lift = lift_H * H * e
    info = {"stride_H_wanted": round(s_want / H, 4), "stride_H": round(s / H, 4), "v": v, "beta": beta,
            "capped": bool(s < s_want - 1e-9)}
    # femur pitch in phase with the leg: forward (negative pitch) at touchdown
    fem = ctx.leg_keys["femur"]
    tr.sided(fem, ctx.sides, X, lambda ph: -femur * e * np.cos(2 * np.pi * ph), offset=0.5)
    offs = {"L": 0.0, "R": 0.5}
    plans = {}
    for lg in ctx.legs:
        ix = lg["ix"]
        foot0 = sk.rest[ix["K"]].copy()
        toe0 = sk.rest[ix["T"]].copy() if "T" in ix else foot0 + np.array([0, 0, 0.1 * H])
        off = toe0 - foot0
        zc = pos0[ix["A"]][2]  # stance sweep centred under the A joint
        ph = (tr.phi + offs[lg["side"]]) % 1.0
        st = ph < beta
        u_st = np.clip(ph / beta, 0, 1)
        u_sw = np.clip((ph - beta) / (1 - beta), 0, 1)
        zt = np.where(st, zc + off[2] + s / 2 - s * u_st,
                      hermite(u_sw, zc + off[2] - s / 2, zc + off[2] + s / 2, -0.5 * s * (1 - beta) / beta,
                              -0.5 * s * (1 - beta) / beta))
        yt = toe0[1] + np.where(st, 0.0, lift * bump(u_sw, lift_peak))
        # foot pitch: 0 in stance, up to toe_off over the last 10% of stance, back to 0 over the first 30% of swing
        toff = np.where(st, rig_ik.ease_io((u_st - 0.9) / 0.1), 1 - rig_ik.ease_io(u_sw / 0.3)) * toe_off_pitch * e
        curl = np.where(st, 0.0, np.sin(np.pi * u_sw) ** 2) * toe_curl * e
        toes = np.c_[np.full(tr.F, toe0[0]), yt, zt]
        plans[lg["side"]] = (toes, off, toff, curl)
        tr.contact[lg["names"]["K"]] = st & (u_st < 0.9)
    return {"info": info, "plans": plans}


def apply_gait_legs(ctx: Ctx, tr: Track, g: dict, f: int, crouch: bool = True) -> float:
    legs = {}
    for lg in ctx.legs:
        toes, off, toff, curl = g["plans"][lg["side"]]
        # toe-off lifts the heel: the Foot->Toes vector (+Z) turns nose-down (positive pitch) about the toes joint
        footW = rot_axis(X, toff[f])[0]
        toesW = rot_axis(X, curl[f])[0] if curl[f] > 0 else EYE
        target = toes[f] - footW @ off
        legs[lg["side"]] = {"target": target, "foot": footW, "toes": toesW}
    return ctx.solve_legs(tr, f, legs, crouch=crouch)


def lock_feet(ctx: Ctx, tr: Track, f: int, lift: float = 0.0, crouch: bool = True) -> float:
    """Feet IK-planted at their rest positions (raised by `lift`), feet and toes world-flat."""
    legs = {}
    for lg in ctx.legs:
        t = ctx.sk.rest[lg["ix"]["K"]] + np.array([0, lift, 0])
        legs[lg["side"]] = {"target": t, "foot": EYE, "toes": EYE}
    return ctx.solve_legs(tr, f, legs, crouch=crouch)


def _P(spec: dict) -> dict:
    return dict(spec.get("params") or {})


# ---------------------------------------------------------------- generators

def idle_bird(ctx: Ctx, spec: dict, e: float) -> Track:
    p = _P(spec)
    sk, H = ctx.sk, ctx.H
    tr = Track(sk, snap_frames(spec.get("T", 3.2)), loop=True)
    t, ph = tr.times, tr.phi
    breath = p.get("breath_deg", 1.5) * e
    for n in ctx.spine:
        tr.turn(n, X, breath * np.sin(2 * np.pi * ph))
    shift = p.get("shift_H", 0.01) * H * e
    tr.root_off[:, 0] += shift * np.sin(2 * np.pi * ph)
    tr.turn(ctx.hips, Z, -p.get("shift_roll", 2) * e * np.sin(2 * np.pi * ph))  # roll onto the loaded (L at +X) leg
    # head looks: discrete targets, minimum-jerk moves, then holds; the sequence ends at (0, 0)
    looks = [tuple(x) for x in (p.get("looks") or [[30, 5], [-25, -8], [10, 10], [0, 0]])]
    if looks[-1] != (0, 0):
        looks.append((0, 0))
    mv = float(p.get("look_move_s", 0.08))
    slot = tr.T / len(looks)
    yaw = np.zeros(tr.F)
    pit = np.zeros(tr.F)
    prev = (0.0, 0.0)
    for k, (yw, pt) in enumerate(looks):
        t0 = 0.2 * slot + k * slot
        a = rig_ik.min_jerk((t - t0) / mv)
        yaw = np.where(t >= t0, prev[0] + (yw - prev[0]) * a, yaw)
        pit = np.where(t >= t0, prev[1] + (pt - prev[1]) * a, pit)
        prev = (yw, pt)
    yaw *= e
    pit *= e
    if ctx.neck:
        tr.turn(ctx.neck[1], Y, 0.4 * yaw)
        tr.turn(ctx.neck[1], X, 0.4 * pit)
    if ctx.head:
        tr.turn(ctx.head, Y, 0.6 * yaw)
        tr.turn(ctx.head, X, 0.6 * pit)
    if ctx.tail:
        tr.turn(ctx.tail[0], Y, p.get("tail_flick", 8) * e * pulse(t, min(2.0, 0.6 * tr.T), 0.25))
    if ctx.jaw:
        tr.turn(ctx.jaw, X, p.get("jaw_twitch", 5) * e * pulse(t, min(1.1, 0.35 * tr.T), 0.15))
    for f in range(tr.F):
        lock_feet(ctx, tr, f)
    for lg in ctx.legs:
        tr.contact[lg["names"]["K"]] = np.ones(tr.F, bool)
    tr.info.update(v=0.0, looks=looks)
    return tr


def walk_biped(ctx: Ctx, spec: dict, e: float) -> Track:
    p = _P(spec)
    sk, H = ctx.sk, ctx.H
    tr = Track(sk, snap_frames(spec.get("T", 0.55), even=True), loop=True)
    ph = tr.phi
    beta = float(p.get("beta", 0.62))
    g = gait_legs(ctx, tr, p, e, beta=beta, stride_H=p.get("stride_H", 0.55), lift_H=p.get("lift_H", 0.12),
                  lift_peak=p.get("lift_peak", 0.45), femur=p.get("femur", 8), toe_curl=p.get("toe_curl", 40),
                  toe_off_pitch=p.get("toe_off_pitch", 20), crouch_H=p.get("crouch_H", 0.03))
    v = g["info"]["v"]
    # pelvis: bob at 2/T lowest at mid-stance, sway toward the stance leg, roll and yaw at 1/T
    mid = beta / 2
    bob = p.get("bob_H", 0.015) * H * e
    bob_y = -bob * np.cos(4 * np.pi * (ph - mid))
    tr.root_off[:, 1] += bob_y
    tr.root_off[:, 0] += p.get("sway_H", 0.008) * H * e * np.cos(2 * np.pi * (ph - mid))
    tr.turn(ctx.hips, Z, -p.get("hip_roll", 4) * e * np.cos(2 * np.pi * (ph - mid)))
    tr.turn(ctx.hips, Y, p.get("hip_yaw", 5) * e * np.sin(2 * np.pi * ph))
    # tail yaw with a lag per segment; wing jitter at 2/T
    lag = float(p.get("tail_lag", 0.08))
    for k, n in enumerate(ctx.tail[:-1] or ctx.tail):
        tr.turn(n, Y, p.get("tail_yaw", 6) * e / 2 * np.sin(2 * np.pi * (ph - lag * (k + 1))))
    if ctx.wing:
        ax = ctx.raise_axis([w + ctx.sides[0] for w in ctx.wing])
        tr.sided(ctx.wing[0], ctx.sides, ax, lambda q: p.get("wing_jitter", 2) * e * np.sin(4 * np.pi * q), offset=0.5)
    for f in range(tr.F):
        apply_gait_legs(ctx, tr, g, f)
    # head: one bob per step: thrust (30%) from the contralateral toe-off, then hold the beak still in the world
    hold = float(p.get("head_hold", 0.70))
    t_hold = hold * tr.T / 2
    D = v * t_hold
    step = ((ph - (beta - 0.5)) % 0.5) / 0.5
    th = 1 - hold
    zoff = np.where(step < th, -D + D * rig_ik.ease_io(step / th), -D * (step - th) / hold)
    holdm = step >= th
    if ctx.neck and ctx.head:
        pos, rot = tr.fk()
        tip0 = sk.rest[sk.i(ctx.tip)]
        bob_scale = 0.3
        crouch = -float(p.get("crouch_H", 0.03)) * H * e
        for f in range(tr.F):
            tgt = np.array([tip0[0], tip0[1] + crouch + bob_scale * bob_y[f], tip0[2] + zoff[f]])
            ctx.solve_neck(tr, f, tip_target=tgt, head_world=EYE, plane=None)
    tr.info.update(g["info"])
    tr.info.update(head_hold=holdm, head_D=D, v=v)
    return tr


def run_biped(ctx: Ctx, spec: dict, e: float) -> Track:
    p = _P(spec)
    sk, H = ctx.sk, ctx.H
    tr = Track(sk, snap_frames(spec.get("T", 0.34), even=True), loop=True)
    ph = tr.phi
    beta = float(p.get("beta", 0.40))
    tr.turn(ctx.hips, X, p.get("trunk_pitch", 12) * e)
    g = gait_legs(ctx, tr, p, e, beta=beta, stride_H=p.get("stride_H", 1.0), lift_H=p.get("lift_H", 0.14),
                  lift_peak=p.get("lift_peak", 0.5), femur=p.get("femur", 10), toe_curl=p.get("toe_curl", 50),
                  toe_off_pitch=p.get("toe_off_pitch", 25), crouch_H=p.get("crouch_H", 0.06))
    bob = p.get("bob_H", 0.02) * H * e
    tr.root_off[:, 1] += -bob * np.cos(4 * np.pi * (ph - beta / 2))
    if ctx.neck:
        tr.turn(ctx.neck[0], X, -p.get("neck_extend", 15) * e)
    if ctx.wing:
        ax = ctx.abduct_axis([w + ctx.sides[0] for w in ctx.wing])
        tr.sided(ctx.wing[0], ctx.sides, ax, lambda q: p.get("wing_abduct", 8) * e + 0 * q, offset=0.0)
    for f in range(tr.F):
        apply_gait_legs(ctx, tr, g, f)
    if ctx.head:
        _, rot = tr.fk()
        hi = sk.i(ctx.head)
        pa = sk.parent[hi]
        for f in range(tr.F):
            tr.local[f, hi] = rot[f, pa].T  # head world-level
    tr.info.update(g["info"])
    return tr


def peck(ctx: Ctx, spec: dict, e: float) -> Track:
    p = _P(spec)
    sk, H = ctx.sk, ctx.H
    tr = Track(sk, snap_frames(spec.get("T", 1.6)), loop=True)
    t = tr.times
    n_pk = max(1, int(p.get("pecks", 2)))
    slot = tr.T / n_pk
    dn, hd, up = float(p.get("down_s", 0.12)), float(p.get("hold_s", 0.05)), float(p.get("up_s", 0.20))
    s = t % slot
    s = np.where(np.isclose(t, tr.T), 0.0, s)
    prog = np.where(s < dn, rig_ik.ease_io(s / dn),
                    np.where(s < dn + hd, 1.0, np.where(s < dn + hd + up, 1 - rig_ik.ease_io((s - dn - hd) / up), 0.0)))
    hold = (s >= dn - 1e-9) & (s <= dn + hd + 1e-9)
    # the goal: in front of the toes, just above the floor
    tips = [sk.rest[lg["ix"]["T"]] if "T" in lg["ix"] else sk.rest[lg["ix"]["K"]] for lg in ctx.legs]
    tip_names = [lg["names"]["tip"] for lg in ctx.legs if sk.has(lg["names"]["tip"])]
    toes_z = max([sk.rest[sk.i(n)][2] for n in tip_names] or [p_[2] for p_ in tips] or [0.0])
    goal = np.array([ctx.sk.cx, ctx.floor + p.get("beak_gap_H", 0.01) * H, toes_z + p.get("reach_H", 0.12) * H])
    # hips pitch: the plan's amount, raised (up to hips_pitch_max) until the neck can reach the goal
    hp_base = p.get("hips_pitch", 15) * e
    hp_max = max(p.get("hips_pitch_max", 30), hp_base)
    Lc = sk.chain_len(ctx.neck) if ctx.neck else 0.0
    hp = hp_base
    if ctx.neck:
        probe = Track(sk, 1)
        for cand in np.linspace(hp_base, hp_max, 31):
            probe.local[:] = EYE
            probe.root_off[:] = 0
            probe.turn(ctx.hips, X, cand)
            lock_feet(ctx, probe, 0)
            pos, _ = probe.fk_frame(0)
            hp = float(cand)
            if np.linalg.norm(goal - pos[sk.i(ctx.neck[0])]) <= 0.97 * Lc:
                break
    tr.turn(ctx.hips, X, hp * prog)
    if ctx.jaw:
        jaw = pulse(s, dn * 0.5, dn * 0.5 + hd + up * 0.6)
        tr.turn(ctx.jaw, X, p.get("jaw_open", 12) * e * jaw)
    if ctx.tail:
        tr.turn(ctx.tail[0], ctx.raise_axis(ctx.tail), p.get("tail_up", 10) * e * prog)
    # between pecks the bird breathes (one cycle a loop, so frame F == frame 0) instead of freezing at rest
    breath = float(p.get("breath_deg", 1.5)) * e * np.sin(2 * np.pi * tr.phi)
    for n in ctx.spine:
        tr.turn(n, X, breath)
    for f in range(tr.F):
        lock_feet(ctx, tr, f)
    if ctx.neck:
        for f in range(tr.F):
            if prog[f] <= 1e-9:
                continue
            pos, _ = tr.fk_frame(f)
            cur = pos[sk.i(ctx.tip)]
            ctx.solve_neck(tr, f, tip_target=(1 - prog[f]) * cur + prog[f] * goal)
    for lg in ctx.legs:
        tr.contact[lg["names"]["K"]] = np.ones(tr.F, bool)
    tr.contact["beak"] = hold
    tr.info.update(v=0.0, goal=goal, hips_pitch=hp, beak_hold=hold)
    return tr


def flap_folded(ctx: Ctx, spec: dict, e: float) -> Track:
    """Wing beats from the folded rest. The main stroke is ELEVATION: the shoulder lifts the folded wing about the
    lateral axis, so the wrist and the tip (back along -Z) come up above the back, as a chicken's wings do when it
    flaps on the ground. Abduction about +Z (the lower edge out from the flank) and spread (the tip out sideways) ride
    on the same stroke so the wing separates from the body. Abduction alone only hinges a folded wing's lower edge
    out (the flank flares like a skirt: no wing rises; seen on the drawn chicken)."""
    p = _P(spec)
    sk, H = ctx.sk, ctx.H
    tr = Track(sk, snap_frames(spec.get("T", 0.75)), loop=True)
    t = tr.times
    beats = max(1, int(round(float(p.get("hz", 8)) * tr.T)))
    hz = beats / tr.T
    es = float(p.get("ease_s", 0.15))
    env = rig_ik.ease_io(t / es) * rig_ik.ease_io((tr.T - t) / es)
    abd = float(p.get("abduct", 60)) * e
    elev = float(p.get("elevate", 25)) * e
    spread = float(p.get("spread", 0.0)) * e
    lag = float(p.get("lag", 0.1))
    amps = list(p.get("lag_amp") or [0.6, 0.35])
    rise = np.zeros(tr.F)
    if ctx.wing:
        L = [w + ctx.sides[0] for w in ctx.wing]
        ax_abd = ctx.abduct_axis(L)
        ax_el = ctx.raise_axis(L)
        # spread: the L wing's tip swings out to +X about the axis normal to the wing and +X (about -Y for a wing
        # folded back along -Z)
        wp = sk.rest[[sk.i(n) for n in L]]
        ax_sp = unit(np.cross(unit(wp[-1] - wp[0]), X)) if np.linalg.norm(np.cross(unit(wp[-1] - wp[0]), X)) > 1e-6 else -Y
        joints = [w for w in ctx.wing if not w.endswith("tip")]
        # the stroke travels down the wing: joint k's world angles blend toward the stroke lagged k * lag (amplitude
        # amps), so the distal joints trail without adding up to more than the shoulder's angle
        prev = None
        for k, w in enumerate(joints):
            cur = 0.5 + 0.5 * np.sin(2 * np.pi * (hz * t - lag * k))
            if k == 0:
                a = env * cur
                M = rot_axis(ax_el, elev * a) @ rot_axis(ax_sp, spread * a) @ rot_axis(ax_abd, abd * a)
            else:
                amp = amps[0] if k == 1 else amps[min(1, len(amps) - 1)]
                d = env * (cur - prev)
                M = rot_axis(ax_el, elev * amp * d) @ rot_axis(ax_abd, abd * amp * d)
            prev = cur
            tr.turn_m(w + ctx.sides[0], M)
            tr.turn_m(w + ctx.sides[1], mirror_m(M))
    # a small hop over the middle 0.45 s; the chest comes up
    hop_d = min(0.45, tr.T)
    t0 = (tr.T - hop_d) / 2
    hop = float(p.get("hop_H", 0.03)) * H * e * pulse(t, t0, hop_d)
    tr.root_off[:, 1] += hop
    tr.turn(ctx.hips, X, float(p.get("trunk_pitch", -8)) * e * env)
    for f in range(tr.F):
        lift = max(0.0, hop[f] - 0.01 * H)
        lock_feet(ctx, tr, f, lift=lift, crouch=False)
    if ctx.head and p.get("neck_level", True):
        _, rot = tr.fk()
        hi = sk.i(ctx.head)
        pa = sk.parent[hi]
        for f in range(tr.F):
            tr.local[f, hi] = rot[f, pa].T
    for lg in ctx.legs:
        tr.contact[lg["names"]["K"]] = hop <= 0.01 * H
    if ctx.wing:  # how far the wing-tip JOINTS rise over their rest height, in H (reported only: the gate reads the skin, rig_clips.wing_skin)
        pos, _ = tr.fk()
        tips = [sk.i(ctx.wing[-1] + s_) for s_ in ctx.sides]
        rise = (pos[:, tips, 1] - (sk.rest[tips, 1] + tr.root_off[:, 1:2])).min(1) / H
    tr.info.update(v=0.0, hz=hz, beats=beats, abduct_peak=abd, spread_peak=spread, elevate_peak=elev,
                   tip_rise_H=float(rise.max()), tip_rise=rise)
    return tr


def _sleep_pose(ctx: Ctx, tr: Track, f: int, drop: float, p: dict, e: float) -> None:
    """Lying base pose at frame f: the root dropped by `drop`; each tarsus laid back from the knee to an intertarsal
    joint on the floor and the foot flat forward, raised where its skin would go through the floor."""
    sk = ctx.sk
    tr.root_off[f, 1] = -drop
    for lg in ctx.legs:
        ix = lg["ix"]
        lift = 0.0
        for _ in range(4):
            pos, _ = tr.fk_frame(f)
            knee = pos[ix["A"]]
            fy = sk.rest[ix["K"]][1] + lift
            ay = max(fy, knee[1] - 0.999 * lg["la"])
            az = knee[2] - math.sqrt(max(lg["la"] ** 2 - (knee[1] - ay) ** 2, 0.0))
            ankle = np.array([knee[0], ay, az])
            fz = az + math.sqrt(max(lg["lb"] ** 2 - (ay - fy) ** 2, 0.0))
            foot = np.array([knee[0], fy, fz])
            ctx.place_leg(tr, f, lg, ankle, foot, EYE, EYE)
            if not lg["verts"].any():
                break
            pos, rot = tr.fk_frame(f)
            lo = float(ctx.skin(pos, rot, lg["verts"])[:, 1].min()) - ctx.floor
            if lo >= -1e-4 * ctx.H:
                break
            lift += -lo + 1e-4 * ctx.H


def sleep_lie(ctx: Ctx, spec: dict, e: float) -> Track:
    p = _P(spec)
    sk, H = ctx.sk, ctx.H
    tr = Track(sk, snap_frames(spec.get("T", 4.0)), loop=True)
    ph = tr.phi
    # static base: neck pulled in, head re-levelled, tail droops; then the root drops until the belly rests
    if ctx.neck:
        tr.turn(ctx.neck[0], X, p.get("head_in", -30) * e / 2)
        tr.turn(ctx.neck[1], X, p.get("head_in", -30) * e / 2)
    if ctx.tail:
        tr.turn(ctx.tail[0], ctx.raise_axis(ctx.tail), -5 * e)
    breath = p.get("breath_deg", 1.0) * e * np.sin(2 * np.pi * ph)
    for n in ctx.spine:
        tr.turn(n, X, breath)
    # the thighs swing forward and up so they tuck into the body instead of pressing through the floor
    tuck = float(p.get("femur_tuck", 40)) * e
    for lg in ctx.legs:
        tr.turn(lg["names"]["femur"], X, -tuck)
    gap = p.get("belly_gap_H", 0.01) * H
    belly = ctx.belly_v
    if not belly.any():
        belly = np.ones(len(ctx.P), bool)
    y_b = float(ctx.P[belly, 1].min()) - ctx.floor
    probe = Track(sk, 1)
    probe.local[0] = tr.local[0]
    best = None
    for drop in np.linspace(max(y_b - gap, 0.0), 0.0, 41):
        probe.root_off[:] = 0
        probe.local[0] = tr.local[0]
        _sleep_pose(ctx, probe, 0, drop, p, e)
        pos, rot = probe.fk_frame(0)
        V = ctx.skin(pos, rot)
        lo = float(V[:, 1].min()) - ctx.floor
        bg = float(V[belly, 1].min()) - ctx.floor
        if best is None or lo > best[1] + 1e-9:
            best = (drop, lo, bg)
        if lo >= -0.005 * H:
            best = (drop, lo, bg)
            break
    drop = best[0]
    for f in range(tr.F):
        _sleep_pose(ctx, tr, f, drop, p, e)
    if ctx.head:
        _, rot = tr.fk()
        hi = sk.i(ctx.head)
        pa = sk.parent[hi]
        for f in range(tr.F):
            tr.local[f, hi] = rot[f, pa].T
    # contact only where the foot's skin is on the floor: tucked feet the floor guard lifted are not planted (the
    # stance gate and floor_pass(prefer_track) read this mask)
    pos_all, rot_all = tr.fk()
    Vs = ctx.skin(pos_all, rot_all)
    for lg in ctx.legs:
        fv = ctx.foot_v.get(lg["names"]["K"])
        if fv is not None and fv.any():
            fy = (Vs[:, fv, 1].min(1) - ctx.floor) / H
            tr.contact[lg["names"]["K"]] = fy <= float((ctx.plan.get("gates") or {}).get("stance_clear_H", 0.02))
        else:
            tr.contact[lg["names"]["K"]] = np.ones(tr.F, bool)
        pos, _ = tr.fk_frame(0)
        ix = lg["ix"]
        tr.info.setdefault("ankle_flex_deg", {})[lg["side"]] = round(flex_deg(pos[ix["A"]], pos[ix["B"]], pos[ix["K"]]), 1)
    tr.info.update(v=0.0, drop_H=round(drop / H, 4), lying=True)
    return tr


def death_fall(ctx: Ctx, spec: dict, e: float) -> Track:
    p = _P(spec)
    sk, H = ctx.sk, ctx.H
    tr = Track(sk, snap_frames(spec.get("T", 1.5)), loop=False)
    t = tr.times
    fall = float(p.get("fall_s", 1.2))
    u = rig_ik.ease_io(t / fall)
    side = str(p.get("roll_side", "L")).upper()
    roll = -float(p.get("roll", 80)) * e if side == "L" else float(p.get("roll", 80)) * e
    r_t0 = min(0.3, 0.25 * fall)
    ur = rig_ik.ease_io((t - r_t0) / max(fall - r_t0, 1e-6))
    tr.turn(ctx.hips, Z, roll * ur)
    # the hips come down to the body's half-width at the hips (the height of a bird lying on its side)
    hz = sk.rest[sk.i(ctx.hips)]
    near = ctx.belly_v & (np.abs(ctx.P[:, 2] - hz[2]) < 0.08 * H)
    if near.sum() < 10:
        near = ctx.belly_v if ctx.belly_v.any() else np.ones(len(ctx.P), bool)
    halfw = float(np.ptp(ctx.P[near, 0])) / 2 if near.any() else 0.3 * H
    target_y = ctx.floor + halfw
    tr.root_off[:, 1] += -(hz[1] - target_y) * u * min(1.0, e / 0.7 if e < 0.7 else 1.0)
    # neck goes limp: +40 deg in total, ease-out, 0.1 s lag per segment
    limp = float(p.get("neck_limp", 40)) * e
    segs = ctx.neck[:-1] if ctx.neck else []
    lag = float(p.get("seg_lag_s", 0.1))
    for k, n in enumerate(segs):
        a = ease_out((t - r_t0 - k * lag) / max(fall - r_t0 - k * lag, 1e-6))
        tr.turn(n, X, limp / max(len(segs), 1) * a)
    if ctx.wing:
        ax = ctx.raise_axis([w + ctx.sides[0] for w in ctx.wing])
        tr.sided(ctx.wing[0], ctx.sides, ax, lambda q: -p.get("wing_relax", 5) * e * u, offset=0.0)
    # legs fold (intertarsal to ankle_fold deg) through an IK target moving up and back in the hips' frame
    fold = float(p.get("ankle_fold", 100))
    curl = float(p.get("toe_curl", 45)) * e
    for f in range(tr.F):
        pos, rot = tr.fk_frame(f)
        Rh = rot[sk.i(ctx.hips)]
        legs = {}
        for lg in ctx.legs:
            ix = lg["ix"]
            fl0 = ctx.rest_flex(lg)
            flex = fl0 + (fold - fl0) * e * u[f]
            d = ctx.d_for_flex(lg, flex)
            rest_dir = unit(sk.rest[ix["K"]] - sk.rest[ix["A"]])
            dir_ = rot_axis(X, 35 * e * u[f])[0] @ rest_dir  # swing the foot back (and so up)
            tgt = pos[ix["A"]] + Rh @ (d * dir_)
            footW = Rh @ rot_axis(X, 0.5 * curl * u[f])[0]
            legs[lg["side"]] = {"target": tgt, "foot": footW, "toes": footW @ rot_axis(X, curl * u[f])[0],
                                "pole": Rh @ lg["pole"]}
        ctx.solve_legs(tr, f, legs, crouch=False)
    # nothing goes through the floor; the final pose rests on it
    pos, rot = tr.fk()
    V = ctx.skin(pos, rot)
    lo = V[:, :, 1].min(1) - ctx.floor
    tr.root_off[:, 1] += -lo[-1] * u
    pos, rot = tr.fk()
    V = ctx.skin(pos, rot)
    lo = V[:, :, 1].min(1) - ctx.floor
    tr.root_off[:, 1] += np.maximum(0.0, -lo)
    for lg in ctx.legs:
        tr.contact[lg["names"]["K"]] = np.zeros(tr.F, bool)  # the legs fold from t = 0: no planted foot
    tr.info.update(v=0.0, lying=True, hold_s=float(p.get("hold_s", 0.3)), fall_s=fall, roll_deg=roll)
    return tr


GENERATORS = {
    "idle_bird": idle_bird,
    "walk_biped": walk_biped,
    "run_biped": run_biped,
    "peck": peck,
    "flap_folded": flap_folded,
    "sleep_lie": sleep_lie,
    "death_fall": death_fall,
}
