#!/usr/bin/env python3
"""Weight territory (scripts/rig_territory.py) and head reach (rig_pred.head_bend / trunk_pitch) on a synthetic
quadruped: a box body on four box legs, a head box on a short neck, a tail chain."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig  # noqa: E402
import rig_pred  # noqa: E402
import rig_territory as rt  # noqa: E402

# name, parent, position (glTF frame: +Y up, +Z forward, +X left); legs top out inside the body, as on the sheep
JOINTS = [
    ("root", None, (0, 0, -0.7)),
    ("hips", "root", (0, 0.8, -0.7)),
    ("spine_1", "hips", (0, 0.8, 0.0)),
    ("spine_2", "spine_1", (0, 0.8, 0.7)),
    ("neck", "spine_2", (0, 0.85, 0.95)),
    ("head", "neck", (0, 0.9, 1.15)),
    ("nose", "head", (0, 0.8, 1.45)),
    ("tail_1", "hips", (0, 0.85, -0.95)),
    ("tail_2", "tail_1", (0, 0.7, -1.2)),
]
for side, x in (("l", 0.22), ("r", -0.22)):
    JOINTS += [(f"back_leg_pelvis_{side}", "hips", (x * 0.5, 0.78, -0.6)),
               (f"back_leg_upper_{side}", f"back_leg_pelvis_{side}", (x, 0.45, -0.75)),
               (f"back_leg_lower_{side}", f"back_leg_upper_{side}", (x, 0.2, -0.75)),
               (f"back_leg_foot_{side}", f"back_leg_lower_{side}", (x, 0.0, -0.75)),
               (f"front_scapula_{side}", "spine_2", (x * 0.5, 0.85, 0.6)),
               (f"front_leg_upper_{side}", f"front_scapula_{side}", (x, 0.45, 0.7)),
               (f"front_leg_lower_{side}", f"front_leg_upper_{side}", (x, 0.2, 0.7)),
               (f"front_leg_foot_{side}", f"front_leg_lower_{side}", (x, 0.0, 0.7))]
NAMES = [j[0] for j in JOINTS]
PARENTS = [NAMES.index(p) if p else -1 for _, p, _ in JOINTS]
JPOS = np.array([j[2] for j in JOINTS], float)


def box(lo, hi, n=4) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=np.subtract(hi, lo), transform=trimesh.transformations.translation_matrix(np.add(lo, hi) / 2))
    for _ in range(n):
        m = m.subdivide()
    return m


def quadruped() -> tuple[np.ndarray, np.ndarray]:
    """Flat-shaded (every triangle its own corners) body + legs + head; returns P, F."""
    parts = [box((-0.35, 0.5, -1.0), (0.35, 1.0, 1.0)), box((-0.12, 0.75, 1.0), (0.12, 1.0, 1.5), 3)]
    for x in (0.22, -0.22):
        for z in (-0.75, 0.7):
            parts.append(box((x - 0.05, 0.0, z - 0.05), (x + 0.05, 0.52, z + 0.05), 3))
    m = trimesh.util.concatenate(parts)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    return V[F].reshape(-1, 3), np.arange(len(F) * 3).reshape(-1, 3)


def nearest_bone_weights(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each point wholly on its nearest bone (joint to child segment)."""
    kids = {k: [c for c, p in enumerate(PARENTS) if p == k] for k in range(len(NAMES))}
    D = np.full((len(P), len(NAMES)), np.inf)
    for k in range(len(NAMES)):
        for c in kids[k]:
            D[:, k] = np.minimum(D[:, k], rt._seg_dist(P, JPOS[k], JPOS[c]))
    D[:, NAMES.index("root")] = np.inf
    J = np.zeros((len(P), 4), np.int64)
    W = np.zeros((len(P), 4), np.float32)
    J[:, 0] = D.argmin(1)
    W[:, 0] = 1.0
    return J, W


def spill(P: np.ndarray, J: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The skin-tokens mistakes: the left thigh grabs the rump above it, the neck grabs the chest behind it."""
    J, W = J.copy(), W.copy()
    rump = (P[:, 1] > 0.7) & (P[:, 2] < -0.4) & (P[:, 0] > 0)
    J[rump, 1] = NAMES.index("back_leg_pelvis_l")
    W[rump] = [0.4, 0.6, 0, 0]
    chest = (P[:, 1] > 0.6) & (P[:, 2] > 0.5) & (P[:, 2] < 0.8)
    J[chest, 1] = NAMES.index("neck")
    W[chest] = [0.5, 0.5, 0, 0]
    return J, W


def dense(J: np.ndarray, W: np.ndarray) -> np.ndarray:
    return rig.to_dense(J, W, len(NAMES))


class TerritoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.P, cls.F = quadruped()
        cls.J0, cls.W0 = spill(cls.P, *nearest_bone_weights(cls.P))

    def test_classify(self) -> None:
        ch = rt.classify(NAMES, PARENTS, JPOS)
        self.assertEqual(len(ch["legs"]), 4)
        self.assertEqual({NAMES[lg["root"]] for lg in ch["legs"]},
                         {"back_leg_pelvis_l", "back_leg_pelvis_r", "front_scapula_l", "front_scapula_r"})
        self.assertEqual(NAMES[ch["neck"]["root"]], "neck")  # after the last spine joint a leg hangs from
        self.assertEqual(NAMES[ch["tail"]["root"]], "tail_1")
        self.assertEqual([NAMES[k] for k in ch["trunk"]], ["hips", "spine_1", "spine_2"])
        self.assertGreater(ch["forward"][2], 0.9)

    def test_leg_line_is_the_mesh_leg_band_top(self) -> None:
        self.assertAlmostEqual(rt.mesh_leg_top(self.P, self.F), 0.5, delta=0.04)

    def test_thigh_weight_above_the_line_goes_to_the_hips(self) -> None:
        J, W, info = rt.territory(self.P, self.F, self.J0, self.W0, NAMES, PARENTS, JPOS, smooth=0)
        self.assertTrue(info["applied"])
        M0, M = dense(self.J0, self.W0), dense(J, W)
        rump = (self.P[:, 1] > 0.7) & (self.P[:, 2] < -0.4) & (self.P[:, 0] > 0)
        pel = NAMES.index("back_leg_pelvis_l")
        self.assertGreater(M0[rump, pel].sum(), 50)
        self.assertLess(M[rump, pel].max(), 1e-6)
        hips = NAMES.index("hips")
        self.assertGreater(M[rump, hips].mean(), M0[rump, hips].mean() + 0.3)
        # the legs below the line keep their own bones
        leg = (self.P[:, 1] < 0.4) & (np.abs(self.P[:, 0] - 0.22) < 0.06) & (np.abs(self.P[:, 2] + 0.75) < 0.06)
        own = [NAMES.index(n) for n in ("back_leg_upper_l", "back_leg_lower_l", "back_leg_foot_l")]
        self.assertGreater(M[leg][:, own].sum(1).min(), 0.99)

    def test_neck_weight_on_the_chest_goes_to_the_spine(self) -> None:
        J, W, _ = rt.territory(self.P, self.F, self.J0, self.W0, NAMES, PARENTS, JPOS, smooth=0)
        M0, M = dense(self.J0, self.W0), dense(J, W)
        neck = NAMES.index("neck")
        behind = (self.P[:, 1] > 0.6) & (self.P[:, 2] > 0.5) & (self.P[:, 2] < 0.95 - 0.06 * 1.5)
        self.assertGreater(M0[behind, neck].sum(), 20)
        self.assertLess(M[behind, neck].max(), 1e-6)
        trunk = [NAMES.index(n) for n in ("hips", "spine_1", "spine_2")]
        self.assertGreater(M[behind][:, trunk].sum(1).min(), 0.99)

    def test_normalised_top4_and_no_seams(self) -> None:
        J, W, _ = rt.territory(self.P, self.F, self.J0, self.W0, NAMES, PARENTS, JPOS)
        self.assertEqual(J.shape[1], 4)
        np.testing.assert_allclose(W.sum(1), 1.0, atol=1e-5)
        self.assertTrue((W >= 0).all())
        M = dense(J, W)
        inv = np.unique(np.round(self.P / 1e-5).astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)
        for g in np.unique(inv)[:4000:7]:
            rows = M[inv == g]
            self.assertLess(float(np.abs(rows - rows[0]).max()), 1e-5)

    def test_smoothing_softens_a_sharp_edge_and_stays_in_bounds(self) -> None:
        J, W, _ = rt.territory(self.P, self.F, self.J0, self.W0, NAMES, PARENTS, JPOS, smooth=0)
        J6, W6, info = rt.territory(self.P, self.F, self.J0, self.W0, NAMES, PARENTS, JPOS, smooth=6)
        self.assertEqual(info["smooth"], 6)
        inv, E = rt.weld(self.P, self.F, 1e-5)

        def edge_jump(J_, W_) -> float:
            G = np.zeros((int(inv.max()) + 1, len(NAMES)))
            G[inv] = dense(J_, W_)
            return float(np.abs(G[E[:, 0]] - G[E[:, 1]]).sum(1).max())
        self.assertLess(edge_jump(J6, W6), edge_jump(J, W))
        # smoothing never brings a leg's weight back above the line, nor one leg's weight onto another
        M = dense(J6, W6)
        legs = [k for k, n in enumerate(NAMES) if "leg" in n or "scapula" in n]
        self.assertLess(M[self.P[:, 1] > 0.5 + 0.06 * 1.0 + 0.01][:, legs].max(), 1e-6)
        right = (self.P[:, 1] < 0.4) & (self.P[:, 0] < -0.15)
        left_bones = [k for k, n in enumerate(NAMES) if n.endswith("_l")]
        self.assertLess(M[right][:, left_bones].max(), 1e-6)

    def test_smooth_weights_alone(self) -> None:
        # a strip of points 0..9 with a 1/0 step between 4 and 5: smoothing spreads it, sums stay 1
        n = 10
        P = np.c_[np.arange(n, dtype=float), np.zeros(n), np.zeros(n)]
        E = np.c_[np.arange(n - 1), np.arange(1, n)]
        M = np.zeros((n, 2))
        M[:5, 0] = 1
        M[5:, 1] = 1
        S = rt.smooth_weights(M, np.arange(n), E, iters=3)
        self.assertLess(abs(S[4, 0] - S[5, 0]), 0.6)
        np.testing.assert_allclose(S.sum(1), 1.0)
        np.testing.assert_allclose(rt.smooth_weights(M, np.arange(n), E, iters=0), M)
        _ = P

    def test_tail_inside_the_rump_moves_nothing_of_it(self) -> None:
        J0, W0 = self.J0.copy(), self.W0.copy()
        rump = (self.P[:, 2] < -0.85) & (self.P[:, 1] > 0.55)
        J0[rump, 2] = NAMES.index("tail_1")
        W0[rump, :3] = [0.3, 0.3, 0.4]
        J, W, _ = rt.territory(self.P, self.F, J0, W0, NAMES, PARENTS, JPOS, smooth=0)
        M = dense(J, W)
        far = rump & (rt._bones_dist(self.P, [NAMES.index("tail_1"), NAMES.index("tail_2")], PARENTS, JPOS) > 0.2 * 1.5)
        self.assertTrue(far.any())
        self.assertLess(M[far][:, [NAMES.index("tail_1"), NAMES.index("tail_2")]].max(), 1e-6)

    def test_blocked_matches_the_limits(self) -> None:
        B, ch = rt.blocked(self.P, NAMES, PARENTS, JPOS, 0.5)
        pel = NAMES.index("back_leg_pelvis_l")
        self.assertTrue(B[self.P[:, 1] > 0.6, pel].all())
        self.assertFalse(B[:, NAMES.index("hips")].any())
        self.assertEqual(len(ch["legs"]), 4)

    def test_no_chains_is_a_no_op(self) -> None:
        names, parents, jp = ["root", "a"], [-1, 0], np.array([[0, 0, 0], [0, 1, 0]], float)
        J = np.zeros((3, 4), np.int64)
        W = np.zeros((3, 4), np.float32)
        W[:, 0] = 1
        J2, W2, info = rt.territory(np.eye(3), None, J, W, names, parents, jp)
        self.assertFalse(info["applied"])
        np.testing.assert_array_equal(W2, W)


def chain_rig(neck_len: float = 0.15) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """A skin-tokens-like skeleton (identity rests, node translations): root -> hips -> shoulder -> neck -> head,
    plus a front leg off the shoulder and a hind leg off the hips; and the head's points (a block in front)."""
    spec = [("root", None, (0, 0, -0.5)), ("hips", "root", (0, 0.8, -0.5)), ("shoulder", "hips", (0, 0.8, 0.5)),
            ("neck", "shoulder", (0, 0.8, 0.6)), ("head", "neck", (0, 0.8, 0.6 + neck_len)),
            ("nose", "head", (0, 0.75, 0.6 + neck_len + 0.2)),
            ("fleg", "shoulder", (0, 0.5, 0.5)), ("ffoot", "fleg", (0, 0.0, 0.5)),
            ("hleg", "hips", (0, 0.5, -0.5)), ("hfoot", "hleg", (0, 0.0, -0.5))]
    names = [s[0] for s in spec]
    pos = {s[0]: np.array(s[2], float) for s in spec}
    nodes = []
    for n, p, _ in spec:
        t = pos[n] - (pos[p] if p else 0)
        nodes.append({"name": n, "translation": t.tolist(), "children": [names.index(c) for c, pp, _ in spec if pp == n]})
    ours = {"nodes": nodes, "skins": [{"joints": list(range(len(spec)))}]}
    z0 = 0.6 + neck_len
    Ph = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (0.65, 0.85) for z in (z0, z0 + 0.25)], float)
    Jh = np.zeros((len(Ph), 4), np.int64)
    Jh[:, 0] = names.index("head")
    Wh = np.zeros((len(Ph), 4))
    Wh[:, 0] = 1.0
    return ours, Ph, Jh, Wh


class HeadReachTests(unittest.TestCase):
    def low(self, fk, q, t, Ph, Jh, Wh) -> float:
        pos, rot = fk.world(q, t)
        return float(rig_pred._skin(Ph, Jh, Wh, fk, pos, rot)[:, 1].min())

    def test_neck_bend_reaches_the_clearance(self) -> None:
        ours, Ph, Jh, Wh = chain_rig(0.3)
        fk = rig_pred.FK(ours)
        q = {j: np.array([0, 0, 0, 1.0]) for j in fk.joints}
        t = {0: fk.t0[0].copy()}
        lateral = np.array([1.0, 0, 0])
        l0 = self.low(fk, q, t, Ph, Jh, Wh)
        want = 0.45
        q2, ang = rig_pred.head_bend(fk, q, t, [3, 4], lateral, Ph, Jh, Wh, want)
        self.assertGreater(ang, 5)
        self.assertLess(self.low(fk, q2, t, Ph, Jh, Wh), want + 0.01)
        self.assertGreater(self.low(fk, q2, t, Ph, Jh, Wh), want - 0.03)
        self.assertGreater(l0, want)
        # already low enough: untouched
        q3, ang3 = rig_pred.head_bend(fk, q, t, [3, 4], lateral, Ph, Jh, Wh, l0 + 0.1)
        self.assertEqual(ang3, 0.0)

    def test_short_neck_tips_the_trunk_to_reach_the_floor(self) -> None:
        ours, Ph, Jh, Wh = chain_rig(0.08)
        fk = rig_pred.FK(ours)
        q = {j: np.array([0, 0, 0, 1.0]) for j in fk.joints}
        t = {0: fk.t0[0].copy()}
        lateral = np.array([1.0, 0, 0])
        want = 0.05  # the template's head grazes the floor
        q2, ang = rig_pred.head_bend(fk, q, t, [3, 4], lateral, Ph, Jh, Wh, want, max_deg=60)
        self.assertGreater(self.low(fk, q2, t, Ph, Jh, Wh), want + 0.05)  # the short neck can't
        spec = {"hips": 1, "root": 0, "keep": [6, 8], "pivot": [8]}
        q3, t3, pdeg = rig_pred.trunk_pitch(fk, q2, t, spec, lateral, Ph, Jh, Wh, want, max_deg=30)
        self.assertGreater(pdeg, 0)
        self.assertLess(self.low(fk, q3, t3, Ph, Jh, Wh), self.low(fk, q2, t, Ph, Jh, Wh) - 0.05)
        pos0, _ = fk.world(q2, t)
        pos3, rot3 = fk.world(q3, t3)
        np.testing.assert_allclose(pos3[8], pos0[8], atol=1e-6)  # the hind leg's top is the pivot
        np.testing.assert_allclose(rot3[6].as_quat(), [0, 0, 0, 1], atol=1e-6)  # the legs keep their world turn


if __name__ == "__main__":
    unittest.main()
