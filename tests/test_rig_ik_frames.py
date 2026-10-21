#!/usr/bin/env python3
"""Math core for built rigs: scripts/rig_ik.py (IK, easing, mirroring) and scripts/rig_frames.py (designed frames,
reframe with identical skinning)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig  # noqa: E402
import rig_frames  # noqa: E402
import rig_ik  # noqa: E402

S = np.diag([-1.0, 1.0, 1.0])
GROUND_BIRD = ROOT / "scripts" / "rig_plans" / "ground_bird.json"


class TwoBoneTests(unittest.TestCase):
    def test_reaches_in_reach_targets_with_apex_on_pole_side(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            pA = rng.normal(size=3)
            pB = pA + rng.normal(size=3)
            pK = pB + rng.normal(size=3)
            la, lb = np.linalg.norm(pB - pA), np.linalg.norm(pK - pB)
            d = rng.uniform(abs(la - lb) + 0.02 * (la + lb), 0.97 * (la + lb))
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            target = pA + d * u
            pole = rng.normal(size=3)
            if np.linalg.norm(np.cross(pole, u)) < 0.2 * np.linalg.norm(pole):
                continue
            qA, qB = rig_ik.two_bone_3d(pA, pB, pK, target, pole)
            nB = pA + qA.apply(pB - pA)
            nK = nB + qB.apply(pK - pB)
            self.assertLess(np.linalg.norm(nK - target), 1e-6)
            self.assertAlmostEqual(np.linalg.norm(nB - pA), la, places=9)
            self.assertAlmostEqual(np.linalg.norm(nK - nB), lb, places=9)
            # the apex sits on the pole side of the A-target axis, in the plane of axis and pole
            off = (nB - pA) - ((nB - pA) @ u) * u
            v = pole - (pole @ u) * u
            self.assertGreater(off @ v, 0)
            self.assertLess(abs(off @ np.cross(u, v / np.linalg.norm(v))), 1e-8)

    def test_reach_is_clamped(self):
        pA, pB, pK = np.zeros(3), np.array([0, -1.0, 0]), np.array([0, -2.0, 0])
        qA, qB = rig_ik.two_bone_3d(pA, pB, pK, np.array([0, -5.0, 0]), [0, 0, -1])
        nB = pA + qA.apply(pB - pA)
        nK = nB + qB.apply(pK - pB)
        self.assertAlmostEqual(np.linalg.norm(nK - pA), 0.98 * 2, places=9)
        self.assertLess(nB[2], 0)  # bends back toward the pole
        self.assertLess(np.linalg.norm(np.cross(nK, [0, -1, 0])), 1e-9)

    def test_rest_target_is_identity(self):
        pA, pB, pK = np.zeros(3), np.array([0, -1.0, -0.4]), np.array([0, -1.8, 0.1])
        qA, qB = rig_ik.two_bone_3d(pA, pB, pK, pK, [0, 0, -1])
        self.assertLess(qA.magnitude(), 1e-7)
        self.assertLess(qB.magnitude(), 1e-7)


class FabrikTests(unittest.TestCase):
    def test_keeps_lengths_and_root(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            pts = np.cumsum(rng.normal(size=(5, 3)), axis=0)
            L = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            target = pts[0] + rng.normal(size=3) * L.sum() * 0.6
            out = rig_ik.fabrik(pts, target)
            np.testing.assert_allclose(np.linalg.norm(np.diff(out, axis=0), axis=1), L, atol=1e-9)
            np.testing.assert_allclose(out[0], pts[0], atol=1e-12)
        far = rig_ik.fabrik(pts, pts[0] + np.array([100.0, 0, 0]))
        np.testing.assert_allclose(np.linalg.norm(np.diff(far, axis=0), axis=1), L, atol=1e-9)

    def test_reaches_and_plane(self):
        pts = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0.0]])
        out = rig_ik.fabrik(pts, [0, 1.5, 1.5], plane_normal=[1, 0, 0])
        self.assertLess(np.linalg.norm(out[-1] - [0, 1.5, 1.5]), 1e-3)
        self.assertLess(np.abs(out[:, 0]).max(), 1e-12)
        np.testing.assert_allclose(np.linalg.norm(np.diff(out, axis=0), axis=1), 1.0, atol=1e-9)


class CurveTests(unittest.TestCase):
    def test_mirror_delta(self):
        rng = np.random.default_rng(3)
        for rv in rng.normal(size=(20, 3)):
            q = R.from_rotvec(rv)
            m = rig_ik.mirror_delta(q)
            np.testing.assert_allclose(m.as_matrix(), S @ q.as_matrix() @ S, atol=1e-12)
            np.testing.assert_allclose(m.as_matrix(), R.from_rotvec(rv * [1, -1, -1]).as_matrix(), atol=1e-9)
            np.testing.assert_allclose(rig_ik.mirror_delta(q.as_quat()), m.as_quat(), atol=1e-12)
        # pitch about lateral is kept; yaw and roll flip
        self.assertAlmostEqual(rig_ik.mirror_delta(R.from_euler("x", 20, degrees=True)).as_euler("xyz", degrees=True)[0], 20)
        self.assertAlmostEqual(rig_ik.mirror_delta(R.from_euler("y", 20, degrees=True)).as_euler("xyz", degrees=True)[1], -20)
        self.assertAlmostEqual(rig_ik.mirror_delta(R.from_euler("z", 20, degrees=True)).as_euler("xyz", degrees=True)[2], -20)

    def test_easing_and_fourier(self):
        x = np.linspace(0, 1, 11)
        for f in (rig_ik.min_jerk, rig_ik.ease_io):
            y = f(x)
            self.assertAlmostEqual(float(y[0]), 0.0)
            self.assertAlmostEqual(float(y[-1]), 1.0)
            self.assertAlmostEqual(float(f(0.5)), 0.5)
            self.assertTrue(np.all(np.diff(y) >= 0))
        c = [0.1, (0.5, 0.2), (0.0, -0.3)]
        phi = np.linspace(0, 1, 31)
        y = rig_ik.fourier(phi, c)
        self.assertAlmostEqual(float(y[0]), float(y[-1]), places=12)
        self.assertAlmostEqual(float(rig_ik.fourier(0.25, c)), 0.1 + 0.2 - 0.5 * 0 + 0.0 * -1, places=12)

    def test_aim_and_fk(self):
        r = rig_ik.aim(R.identity(), [0, 1, 0], [0, 0, 1])
        np.testing.assert_allclose(r.apply([0, 1, 0]), [0, 0, 1], atol=1e-12)
        r = rig_ik.aim(R.identity(), [0, 1, 0], [0, -1, 0])
        np.testing.assert_allclose(r.apply([0, 1, 0]), [0, -1, 0], atol=1e-12)
        parent = [-1, 0, 1]
        rest_t = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0.0]])
        q = np.array([[0, 0, 0, 1], R.from_euler("x", 90, degrees=True).as_quat(), [0, 0, 0, 1]])
        pos, rot = rig_ik.fk(parent, rest_t, q, root_t=[0, 2, 0])
        np.testing.assert_allclose(pos, [[0, 2, 0], [0, 3, 0], [0, 3, 1]], atol=1e-12)
        np.testing.assert_allclose(rot[2].as_matrix(), R.from_euler("x", 90, degrees=True).as_matrix(), atol=1e-12)


# ---------------------------------------------------------------- frames + reframe

def bird_skeleton() -> tuple[dict, dict]:
    """A small symmetric bird-like skeleton (skeleton.json shape) and its plan-style joint list."""
    joints = [("root", None, [0, 0, 0]), ("hips", "root", [0, 0.5, 0]), ("spine_0", "hips", [0, 0.55, 0.1]),
              ("head", "spine_0", [0, 0.8, 0.3]), ("head_tip", "head", [0, 0.82, 0.42]),
              ("tail_1", "hips", [0, 0.55, -0.25]), ("tail_tip", "tail_1", [0, 0.6, -0.4])]
    plan_joints = [["root", None], ["hips", "root"], ["spine_0", "hips"], ["head", "spine_0"], ["head_tip", "head"],
                   ["tail_1", "hips"], ["tail_tip", "tail_1"]]
    for s, sx in (("L", 1), ("R", -1)):
        joints += [(f"UpperLeg_{s}", "hips", [0.1 * sx, 0.45, 0.0]), (f"LowerLeg_{s}", f"UpperLeg_{s}", [0.12 * sx, 0.3, 0.06]),
                   (f"AnkleLeg_{s}", f"LowerLeg_{s}", [0.12 * sx, 0.18, -0.05]), (f"Foot_{s}", f"AnkleLeg_{s}", [0.12 * sx, 0.02, 0.0]),
                   (f"Toes_tip_{s}", f"Foot_{s}", [0.12 * sx, 0.0, 0.1]),
                   (f"wing_1_{s}", "spine_0", [0.15 * sx, 0.6, 0.05]), (f"wing_tip_{s}", f"wing_1_{s}", [0.17 * sx, 0.55, -0.2])]
    plan_joints += [["UpperLeg_{s}", "hips"], ["LowerLeg_{s}", "UpperLeg_{s}"], ["AnkleLeg_{s}", "LowerLeg_{s}"],
                    ["Foot_{s}", "AnkleLeg_{s}"], ["Toes_tip_{s}", "Foot_{s}"], ["wing_1_{s}", "spine_0"],
                    ["wing_tip_{s}", "wing_1_{s}"]]
    skel = {"joints": [{"name": n, "parent": p, "pos": pos} for n, p, pos in joints], "H": 0.82, "cx": 0.0}
    return skel, {"joints": plan_joints, "side_suffix": ["_L", "_R"]}


def rig_doc(skel: dict) -> tuple[dict, bytes]:
    """A GLB doc (Armature holder, TRS joints with identity rests, packed IBM) from skeleton.json."""
    names = [j["name"] for j in skel["joints"]]
    idx = {n: k + 1 for k, n in enumerate(names)}  # node 0 = Armature
    pos = {j["name"]: np.asarray(j["pos"], float) for j in skel["joints"]}
    nodes = [{"name": "Armature", "children": [idx["root"]]}]
    for j in skel["joints"]:
        p = j["parent"]
        t = pos[j["name"]] - (pos[p] if p else 0)
        nodes.append({"name": j["name"], "translation": t.tolist(),
                      "children": [idx[c["name"]] for c in skel["joints"] if c["parent"] == j["name"]]})
    for n in nodes:
        if not n["children"]:
            n.pop("children")
    g = rig.GlbWriter()
    ibm = np.array([np.linalg.inv(np.block([[np.eye(3), pos[n][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])).T.reshape(-1)
                    for n in names], "<f4")
    acc = g.array(ibm, "MAT4")
    g.doc.update(nodes=nodes, skins=[{"joints": [idx[n] for n in names], "inverseBindMatrices": acc}],
                 scenes=[{"nodes": [0]}], scene=0)
    with tempfile.TemporaryDirectory() as td:
        g.write(Path(td) / "r.glb")
        return rig.read_glb(Path(td) / "r.glb")


class FrameTests(unittest.TestCase):
    def check_frames(self, skel: dict, plan: dict) -> dict:
        Q = rig_frames.frames(skel, plan)
        pos = {j["name"]: np.asarray(j["pos"], float) for j in skel["joints"]}
        for n, q in Q.items():
            np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-9, err_msg=n)
            self.assertAlmostEqual(np.linalg.det(q), 1.0, places=9, msg=n)
            if n.endswith("_L"):
                np.testing.assert_allclose(Q[n[:-2] + "_R"], S @ q @ S, atol=1e-4, err_msg=n)
        np.testing.assert_allclose(Q["root"], np.eye(3), atol=1e-12)
        return Q, pos

    def test_synthetic_bird(self):
        skel, plan = bird_skeleton()
        Q, pos = self.check_frames(skel, plan)
        # +Y along the bone to the PRIMARY child (plan order: hips -> spine_0, not tail_1 or legs), both sides
        def ydir(a, b):
            v = pos[b] - pos[a]
            return v / np.linalg.norm(v)
        for a, b in (("hips", "spine_0"), ("UpperLeg_L", "LowerLeg_L"), ("UpperLeg_R", "LowerLeg_R"),
                     ("AnkleLeg_R", "Foot_R"), ("wing_1_R", "wing_tip_R"), ("tail_1", "tail_tip")):
            np.testing.assert_allclose(Q[a][:, 1], ydir(a, b), atol=1e-9, err_msg=a)
        # +X lateral (creature left) on midline and both sides; leaves copy the parent
        for n in ("hips", "spine_0", "UpperLeg_L", "UpperLeg_R", "Foot_L", "Foot_R"):
            self.assertGreater(Q[n][0, 0], 0.5, n)
        np.testing.assert_allclose(Q["Toes_tip_R"], Q["Foot_R"], atol=1e-12)
        np.testing.assert_allclose(Q["head_tip"], Q["head"], atol=1e-12)
        # a leg hinge (rotation about local X) flexes with the same sign on both sides: R = S L S
        flexL = Q["LowerLeg_L"] @ R.from_euler("x", 30, degrees=True).as_matrix() @ Q["LowerLeg_L"].T
        flexR = Q["LowerLeg_R"] @ R.from_euler("x", 30, degrees=True).as_matrix() @ Q["LowerLeg_R"].T
        np.testing.assert_allclose(flexR, S @ flexL @ S, atol=1e-9)

    def test_vertical_bone_uses_forward_reference(self):
        skel = {"joints": [{"name": "root", "parent": None, "pos": [0, 0, 0]},
                           {"name": "a", "parent": "root", "pos": [0, 0, 0]},
                           {"name": "b", "parent": "a", "pos": [1, 0, 0]}]}
        Q = rig_frames.frames(skel, None)
        np.testing.assert_allclose(Q["a"][:, 1], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(Q["a"][:, 0], [0, 0, 1], atol=1e-12)
        # a lateral bone on both sides (the forward fallback) still mirrors: R = S L S
        skel = {"joints": [{"name": "root", "parent": None, "pos": [0, 0, 0]},
                           {"name": "a_L", "parent": "root", "pos": [0.1, 0, 0]}, {"name": "b_L", "parent": "a_L", "pos": [1, 0.1, 0.05]},
                           {"name": "a_R", "parent": "root", "pos": [-0.1, 0, 0]}, {"name": "b_R", "parent": "a_R", "pos": [-1, 0.1, 0.05]}]}
        Q = rig_frames.frames(skel, None)
        np.testing.assert_allclose(Q["a_R"], S @ Q["a_L"] @ S, atol=1e-12)
        np.testing.assert_allclose(Q["b_R"], S @ Q["b_L"] @ S, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(Q["a_R"]), 1.0)

    @unittest.skipUnless(GROUND_BIRD.is_file(), "scripts/rig_plans/ground_bird.json not present")
    def test_ground_bird_plan_order(self):
        plan = json.loads(GROUND_BIRD.read_text())
        order = rig_frames.plan_joint_order(plan)
        self.assertEqual(len(order), 55)
        self.assertEqual(len(set(order)), 55)
        self.assertLess(order.index("spine_0"), order.index("tail_1"))
        self.assertIn("wing_feather_4_2_R", order)

    def test_reframe_keeps_lbs(self):
        skel, plan = bird_skeleton()
        Q = rig_frames.frames(skel, plan)
        doc, binary = rig_doc(skel)
        joints = doc["skins"][0]["joints"]
        rng = np.random.default_rng(4)
        jp = np.array([j["pos"] for j in skel["joints"]], float)
        H = float(np.ptp(jp[:, 1]))
        P = jp[rng.integers(0, len(jp), 400)] + rng.normal(scale=0.05, size=(400, 3))
        J = rng.integers(0, len(joints), (400, 4))
        W = rng.random((400, 4))
        W /= W.sum(1, keepdims=True)
        F = 9
        clips = [{"name": "Wiggle", "times": np.arange(F, dtype=np.float32) / 30,
                  "rot": {i: R.from_rotvec(rng.normal(scale=0.5, size=(F, 3))).as_quat().astype(np.float32) for i in joints},
                  "root": joints[0], "root_t": (np.array([0, 0, 0.0]) + rng.normal(scale=0.05, size=(F, 3))).astype(np.float32)}]
        d2, b2, c2 = rig_frames.reframe(doc, binary, clips, Q)
        delta = rig_frames.check_identical((doc, binary, clips), (d2, b2, c2), P, J, W)
        self.assertLess(delta / H, 1e-5)
        # the new rests carry the frames; IBMs are inv(world rest); nothing else moved
        skel2 = rig.Skeleton(d2)
        w2 = skel2.world({})
        IBM = rig.accessor(d2, b2, d2["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        for k, i in enumerate(joints):
            n = d2["nodes"][i]["name"]
            np.testing.assert_allclose(w2[i][1].as_matrix(), Q[n], atol=1e-6, err_msg=n)
            np.testing.assert_allclose(w2[i][0], skel["joints"][k]["pos"], atol=1e-6, err_msg=n)
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = w2[i][1].as_matrix(), w2[i][0]
            np.testing.assert_allclose(IBM[k] @ M, np.eye(4), atol=1e-5)
            self.assertNotIn("matrix", d2["nodes"][i])
        self.assertEqual(d2["nodes"][0], doc["nodes"][0])  # Armature untouched
        np.testing.assert_allclose(c2[0]["root_t"], clips[0]["root_t"])
        # the input is not mutated
        self.assertNotIn("rotation", doc["nodes"][1])
        # the reframed rig writes out (write_rigged's layout reads it) and reads back with frames.json
        with tempfile.TemporaryDirectory() as td:
            rig_frames.write_frames(Path(td) / "frames.json", Q)
            back = rig_frames.read_frames(Path(td) / "frames.json")
            for n in Q:
                np.testing.assert_allclose(back[n], Q[n], atol=1e-6)

    def test_reframe_glb_animations_and_template_frames(self):
        skel, plan = bird_skeleton()
        doc, binary = rig_doc(skel)
        joints = doc["skins"][0]["joints"]
        Q = rig_frames.frames(skel, plan)
        rng = np.random.default_rng(5)
        F = 5
        clip = {"name": "A", "times": np.arange(F, dtype=np.float32) / 30,
                "rot": {i: R.from_rotvec(rng.normal(scale=0.4, size=(F, 3))).as_quat().astype(np.float32) for i in joints},
                "root": joints[0], "root_t": rng.normal(scale=0.05, size=(F, 3)).astype(np.float32)}
        # store the clip as a real glTF animation, then reframe the GLB (no clip dicts)
        g = rig.GlbWriter()
        g.doc["bufferViews"], g.doc["accessors"] = [], []
        tacc = g.array(clip["times"].reshape(-1, 1), "SCALAR")
        samplers, channels = [], []
        for i, q in clip["rot"].items():
            samplers.append({"input": tacc, "output": g.array(q, "VEC4"), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": i, "path": "rotation"}})
        samplers.append({"input": tacc, "output": g.array(clip["root_t"], "VEC3"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": clip["root"], "path": "translation"}})
        ibm = rig.accessor(doc, binary, doc["skins"][0]["inverseBindMatrices"]).astype("<f4")
        g.doc.update(nodes=doc["nodes"], scenes=doc["scenes"], scene=0,
                     skins=[{"joints": joints, "inverseBindMatrices": g.array(ibm, "MAT4")}],
                     animations=[{"name": "A", "samplers": samplers, "channels": channels}])
        with tempfile.TemporaryDirectory() as td:
            g.write(Path(td) / "a.glb")
            adoc, abin = rig.read_glb(Path(td) / "a.glb")
            d2, b2, _ = rig_frames.reframe(adoc, abin, [], Q)
            c_before = rig_frames.clips_from_doc(adoc, abin)
            c_after = rig_frames.clips_from_doc(d2, b2)
            self.assertEqual(len(c_after), 1)
            P = np.array([j["pos"] for j in skel["joints"]], float) + 0.01
            J = np.tile(np.arange(4), (len(P), 1)) % len(joints)
            W = np.full((len(P), 4), 0.25)
            delta = rig_frames.check_identical(({**adoc, "animations": []}, abin, c_before),
                                               ({**d2, "animations": []}, b2, c_after), P, J, W)
            self.assertLess(delta, 1e-5)
            # frames_from_template on the reframed GLB returns the designed frames by name
            g2 = rig.GlbWriter()
            g2.doc["bufferViews"], g2.doc["accessors"] = [], []
            g2.doc.update(nodes=d2["nodes"], scenes=d2["scenes"], scene=0,
                          skins=[{"joints": joints, "inverseBindMatrices": g2.array(
                              rig.accessor(d2, b2, d2["skins"][0]["inverseBindMatrices"]).astype("<f4"), "MAT4")}])
            g2.write(Path(td) / "t.glb")
            T = rig_frames.frames_from_template(Path(td) / "t.glb", list(Q) + ["nope"])
            self.assertNotIn("nope", T)
            for n in Q:
                np.testing.assert_allclose(T[n], Q[n], atol=1e-6, err_msg=n)


if __name__ == "__main__":
    unittest.main()
