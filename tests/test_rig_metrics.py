#!/usr/bin/env python3
"""scripts/rig_metrics.py on a synthetic stick rig: bind metrics, motion metrics, stretch owners, pose flags."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig_metrics as rm  # noqa: E402

NAMES = ["root", "hips", "spine", "head", "leg_L", "foot_L"]
PARENT = [-1, 0, 1, 2, 1, 4]
JPOS = np.array([[0, 0.0, 0], [0, 0.5, 0], [0, 1.0, 0], [0, 1.5, 0], [0.3, 0.5, 0], [0.3, 0.0, 0]])
H = 1.6


def tube(a, b, n=6, ring=8, r=0.04):
    """Corner positions and faces of a closed-ish tube from a to b (a few rings)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    u = np.cross(ax, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(ax, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ax / np.linalg.norm(ax), u)
    P = []
    for i in range(n):
        c = a + ax * (i + 0.5) / n
        for j in range(ring):
            t = 2 * np.pi * j / ring
            P.append(c + r * (np.cos(t) * u + np.sin(t) * v))
    F = []
    for i in range(n - 1):
        for j in range(ring):
            p0, p1 = i * ring + j, i * ring + (j + 1) % ring
            q0, q1 = p0 + ring, p1 + ring
            F += [[p0, p1, q1], [p0, q1, q0]]
    return np.array(P), np.array(F)


def synth():
    """Vertices along each bone, bound 100% to the bone they sit on (clean)."""
    segs = rm.segs_from_parent(PARENT)
    Ps, Fs, Js = [], [], []
    off = 0
    for a, c in segs:
        if a == 0:
            continue  # root -> hips carries no mesh
        P, F = tube(JPOS[a], JPOS[c])
        Ps.append(P)
        Fs.append(F + off)
        Js.append(np.full(len(P), a))
        off += len(P)
    P = np.concatenate(Ps)
    F = np.concatenate(Fs)
    J = np.zeros((len(P), 4), int)
    J[:, 0] = np.concatenate(Js)
    W = np.zeros((len(P), 4))
    W[:, 0] = 1.0
    return P, F, J, W, segs


def mats_identity(B, frames=1):
    return np.tile(np.eye(4), (frames, B, 1, 1))


class BindMetrics(unittest.TestCase):
    def test_a2_clean_is_zero_and_injected_far_bone_raises_it(self):
        P, F, J, W, segs = synth()
        clean = rm.a2_far_weight(P, J, W, JPOS, segs, H, names=NAMES)
        self.assertEqual(clean["share"], 0.0)
        self.assertEqual(clean["worst"], [])
        # inject: head-tube vertices (bound to spine) get 60% of their weight on foot_L, ~1.3H away
        sel = J[:, 0] == NAMES.index("spine")
        J2, W2 = J.copy(), W.copy()
        J2[sel, 1] = NAMES.index("foot_L")
        W2[sel, 0], W2[sel, 1] = 0.4, 0.6
        bad = rm.a2_far_weight(P, J2, W2, JPOS, segs, H, names=NAMES)
        self.assertGreater(bad["share"], 0.0)
        self.assertGreater(bad["verts_mostly_far"], 0.0)
        self.assertEqual(bad["worst"][0][0], "foot_L")

    def test_seg_dist_uses_every_child(self):
        D = rm.seg_dist(np.array([[0.3, 0.25, 0.0], [0.0, 0.75, 0.0]]), JPOS, rm.segs_from_parent(PARENT))
        self.assertAlmostEqual(D[0, NAMES.index("leg_L")], 0.0, places=9)  # leg_L -> foot_L
        self.assertAlmostEqual(D[1, NAMES.index("hips")], 0.0, places=9)   # hips -> spine
        self.assertAlmostEqual(D[0, NAMES.index("foot_L")], 0.25, places=9)  # a leaf is a point
        # hips -> leg_L is horizontal at y=0.5: the point (0.3, 0.25) is 0.25 below its end
        self.assertAlmostEqual(D[0, NAMES.index("hips")], 0.25, places=9)

    def test_foot_off_chain(self):
        P, F, J, W, segs = synth()
        legs = [["leg_L", "foot_L"]]
        ok = rm.foot_off_chain(P, J, W, NAMES, legs)
        self.assertGreater(ok["floor_verts"], 0)
        self.assertEqual(ok["share"], 0.0)
        floor = P[:, 1] < P[:, 1].min() + 0.08 * np.ptp(P[:, 1])
        J2 = J.copy()
        J2[floor, 0] = NAMES.index("spine")
        bad = rm.foot_off_chain(P, J2, W, NAMES, legs)
        self.assertEqual(bad["share"], 1.0)
        self.assertEqual(bad["owners"][0][0], "spine")

    def test_part_shares(self):
        P, F, J, W, segs = synth()
        chains = {"neck": ["spine", "head"], "legs": [["leg_L", "foot_L"]], "tail": None}
        s = rm.part_shares(P, J, W, chains, jpos=JPOS, segs=segs, names=NAMES)
        # only the corners at a joint, equally near the parent's bone, can sit on another chain
        self.assertGreater(s["neck"]["share"], 0.9)
        self.assertGreater(s["legs"]["share"], 0.9)
        self.assertNotIn("tail", s)
        J2 = J.copy()
        J2[J[:, 0] == NAMES.index("leg_L"), 0] = NAMES.index("hips")
        s2 = rm.part_shares(P, J2, W, chains, jpos=JPOS, segs=segs, names=NAMES)
        self.assertLess(s2["legs"]["share"], 0.8)
        self.assertIn("body", s2["legs"]["others"])
        with self.assertRaises(ValueError):
            rm.part_shares(P, J, W, chains)

    def test_pose_flags(self):
        P, F, J, W, segs = synth()
        jp = JPOS.copy()
        jp[NAMES.index("head")] = [2.0, 1.5, 0]  # far out of the box
        r = rm.pose_flags(jp, segs, P, F, P, H, names=NAMES)
        flagged = {x["joint"]: x["flags"] for x in r["flagged"]}
        self.assertIn("head", flagged)
        self.assertIn("BOX", flagged["head"])
        self.assertIn("FAR", flagged["head"])
        self.assertNotIn("spine", flagged)
        self.assertGreater(r["bone_far_outside_0p08H"], 0)
        self.assertEqual(len(r["bones"]), len(segs))


class MotionMetrics(unittest.TestCase):
    def test_skin_frames_identity_and_translation(self):
        P, F, J, W, segs = synth()
        V = rm.skin_frames(P, J, W, mats_identity(len(NAMES), 2))
        self.assertEqual(V.shape, (2, len(P), 3))
        np.testing.assert_allclose(V[1], P, atol=1e-12)
        M = np.tile(np.eye(4), (len(NAMES), 1, 1))
        M[:, 1, 3] = 0.5
        np.testing.assert_allclose(rm.skin_frames(P, J, W, M)[0], P + [0, 0.5, 0], atol=1e-12)
        ibm = np.tile(np.eye(4), (len(NAMES), 1, 1))
        ibm[:, 1, 3] = -0.5
        np.testing.assert_allclose(rm.skin_frames(P, J, W, M, ibm=ibm)[0], P, atol=1e-12)

    def test_c_shape_is_one_at_rest(self):
        P, *_ = synth()
        c = rm.c_shape(np.stack([P, P, P]), P)
        self.assertEqual(c["ext_ratio"], [1.0, 1.0, 1.0])
        self.assertEqual(c["ext_ratio_max"], 1.0)
        self.assertEqual(c["height_ratio_min"], 1.0)
        self.assertEqual(c["height_ratio_max"], 1.0)
        sq = P * [2.0, 0.5, 1.0]
        c2 = rm.c_shape(sq, P)
        self.assertAlmostEqual(c2["ext_ratio"][0], 2.0, places=4)
        self.assertAlmostEqual(c2["height_ratio_min"], 0.5, places=4)

    def test_foot_slide_planted_is_zero(self):
        pos = np.tile([0.3, 0.0, 0.1], (31, 1))
        contact = np.ones(31, bool)
        self.assertEqual(rm.foot_slide(pos, contact, 0.0)["slide"], 0.0)
        # in place: the foot moves back at the expected speed, so no slide either
        t = np.arange(31) / 30
        back = np.c_[np.full(31, 0.3), np.zeros(31), 0.1 - 0.6 * t]
        self.assertAlmostEqual(rm.foot_slide(back, contact, [0, 0, -0.6])["slide"], 0.0, places=9)
        # the same foot with v_expected 0 slides 0.6 units = 0.375H
        r = rm.foot_slide(back, contact, 0.0, H=H)
        self.assertAlmostEqual(r["slide"], 0.6 / H, places=4)
        self.assertTrue(r["relative_to_H"])
        # swing frames don't count
        swing = contact.copy()
        swing[5:] = False
        self.assertAlmostEqual(rm.foot_slide(back, swing, 0.0)["slide"], 0.6 * 4 / 30, places=6)
        self.assertEqual(rm.foot_slide(back, np.zeros(31, bool), 0.0)["intervals"], 0)

    def test_loop_seam(self):
        q = np.tile([0, 0, 0, 1.0], (31, 1))
        q[10] = [0, np.sin(0.3), 0, np.cos(0.3)]
        self.assertEqual(rm.loop_seam({"rot": {0: q, 1: q}, "loop": True}), 0.0)
        q2 = q.copy()
        a = np.radians(2.0) / 2
        q2[-1] = [np.sin(a), 0, 0, np.cos(a)]
        self.assertAlmostEqual(rm.loop_seam({"rot": {0: q, 3: q2}}), 2.0, places=3)
        q3 = q.copy()
        q3[-1] = -q3[0]  # the same rotation, opposite sign
        self.assertEqual(rm.loop_seam({"rot": {0: q3}}), 0.0)

    def test_stretch_owners_names_the_injected_bone(self):
        P, F, J, W, segs = synth()
        E = rm.edges_from_faces(P, F)
        self.assertGreater(len(E), 0)
        top = rm.top_bone(J, W)
        frames = []
        for s in np.linspace(0, 1, 6):
            V = P.copy()
            sel = top == NAMES.index("leg_L")
            c = P[sel].mean(0)
            V[sel] = c + (P[sel] - c) * (1 + 1.5 * s)  # the leg tube balloons
            frames.append(V)
        r = rm.stretch_owners(P, np.stack(frames), E, top, NAMES)
        self.assertEqual(r["owners"][0][0], "leg_L")
        self.assertEqual(r["stretch_owners"][0][0], "leg_L")
        self.assertGreater(r["max"], 2.0)
        self.assertEqual(r["squash"], 0)

    def test_frame_stats(self):
        P, *_ = synth()
        s = rm.frame_stats(np.stack([P, P + [0, 0.1, 0]]), P, H)
        self.assertEqual(s["min_y_rel"][0], 0.0)
        self.assertAlmostEqual(s["min_y_rel"][1], 0.1 / H, places=4)
        self.assertAlmostEqual(s["max_disp_rel"][1], 0.1 / H, places=4)
        self.assertEqual(s["long_axis_tilt_deg"][0], 0.0)
        self.assertAlmostEqual(s["height_rel"][0], np.ptp(P[:, 1]) / H, places=4)
        # lying on its side: rotate 90 degrees about Z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        lie = rm.frame_stats(P @ R.T, P, H)
        self.assertAlmostEqual(lie["long_axis_tilt_deg"][0], 90.0, places=1)


if __name__ == "__main__":
    unittest.main()
