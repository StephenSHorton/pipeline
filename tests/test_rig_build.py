#!/usr/bin/env python3
"""scripts/rig_build.py on the synthetic capsule bird (tests/fixtures/synth_bird.py)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import rig  # noqa: E402
import rig_build  # noqa: E402
import rig_fit  # noqa: E402
import rig_metrics  # noqa: E402
import rig_plan  # noqa: E402
import rig_pred  # noqa: E402
from fixtures import synth_bird  # noqa: E402

PLAN = rig_plan.load("ground_bird")


def _pos(info: dict) -> dict:
    return {j["name"]: np.array(j["pos"]) for j in info["skeleton"]["joints"]}


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "bird"
        cls.info, cls.fx = synth_bird.build(cls.dir)
        cls.pos = _pos(cls.info)
        cls.H = cls.fx["H"]
        cls.cx = cls.info["cx"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_outputs_and_names(self) -> None:
        sk = json.loads((self.dir / "template" / "skeleton.json").read_text())
        bj = json.loads((self.dir / "template" / "build.json").read_text())
        self.assertTrue(bj["ok"])
        self.assertEqual([j["name"] for j in sk["joints"]], rig_plan.joint_names(PLAN))
        self.assertEqual(len(sk["joints"]), 55)
        for j in sk["joints"]:
            self.assertEqual(set(j), {"name", "parent", "pos", "role", "side", "source"})
            self.assertNotRegex(j["source"], r"^bone_")
        self.assertEqual([j["name"] for j in sk["joints"] if j["parent"] is None], ["root"])
        self.assertEqual(next(j for j in sk["joints"] if j["name"] == "hips")["parent"], "root")
        for k in ("H", "cx"):
            self.assertIn(k, sk)
        for k in ("checks", "cd_j2b", "sources", "pred_mode", "pred_joints", "legs_source"):
            self.assertIn(k, bj)
        self.assertEqual(bj["pred_joints"], 15)
        self.assertEqual(bj["legs_source"], "pred")

    def test_every_joint_inside(self) -> None:
        V, F = self.fx["V"], self.fx["F"]
        names = list(self.pos)
        wn = rig_fit.winding_number(V, F, np.array([self.pos[n] for n in names]))
        surf = rig_fit._sample(V, F, 20000)
        from scipy.spatial import cKDTree
        d = cKDTree(np.r_[surf, V]).query(np.array([self.pos[n] for n in names]))[0]
        for n, w, dd in zip(names, wn, d):
            if n == "root":
                self.assertAlmostEqual(self.pos[n][1], 0.0, places=5)  # on the floor, under the hips
                continue
            if rig_build.is_tip(n):
                self.assertTrue(abs(w) > 0.5 or dd <= 0.02 * self.H, f"{n} {dd / self.H:.3f}H outside")
            else:
                self.assertGreater(abs(w), 0.5, f"{n} is outside the mesh")

    def test_frame(self) -> None:
        p, cx = self.pos, self.cx
        for n in p:
            if n.endswith("_L"):
                self.assertGreater(p[n][0], cx, n)
            if n.endswith("_R"):
                self.assertLess(p[n][0], cx, n)
        self.assertGreater(p["head"][2], p["hips"][2])
        self.assertGreater(p["mouth_upper_tip"][2], p["head"][2])
        self.assertLess(p["tail_tip"][2], p["hips"][2])
        for s in "LR":
            self.assertGreater(p[f"Toes_tip_{s}"][2], p[f"Foot_{s}"][2])
            self.assertGreater(p[f"Toes_{s}"][2], p[f"Foot_{s}"][2])
            # digitigrade: the intertarsal apex points back, the true knee forward of it
            self.assertLess(p[f"AnkleLeg_{s}"][2], p[f"LowerLeg_{s}"][2])
        self.assertAlmostEqual(p["root"][0], p["hips"][0], places=6)
        self.assertAlmostEqual(p["root"][2], p["hips"][2], places=6)

    def test_feet_on_floor_clusters(self) -> None:
        V = self.fx["V"]
        seeds = np.array([[synth_bird.LEG_X, 0, 0.0], [-synth_bird.LEG_X, 0, 0.0]])
        C, _ = rig_fit.floor_clusters(V, seeds)
        for k, s in enumerate("LR"):
            f = self.pos[f"Foot_{s}"]
            self.assertLess(np.linalg.norm(f[[0, 2]] - C[k]), 0.02 * self.H, s)
            self.assertLessEqual(f[1], 0.02 * self.H + 1e-6)

    def test_mirror(self) -> None:
        p, cx = self.pos, self.cx
        err = 0.0
        for n in p:
            if n.endswith("_L"):
                m = p[n[:-2] + "_R"].copy()
                m[0] = 2 * cx - m[0]
                err = max(err, float(np.linalg.norm(p[n] - m)))
            elif not n.endswith("_R"):
                err = max(err, abs(p[n][0] - cx))
        self.assertLess(err, 0.01 * self.H)
        self.assertLess(self.info["checks"]["mirror_err_H"], 0.01)

    def test_checks_and_ratios(self) -> None:
        c = self.info["checks"]
        self.assertEqual(c["problems"], [])
        self.assertEqual(c["gate"]["level"], "pass")
        for k, (lo, hi) in PLAN["ratios"].items():
            self.assertTrue(lo <= c["lengths_H"][k] <= hi, (k, c["lengths_H"][k]))
        self.assertLessEqual(self.info["cd_j2b"]["pred_to_built_H"]["mean"], 0.03)

    def test_three_leg_prediction_falls_back_to_floor_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            info, fx = synth_bird.build(Path(t), legs=3)
            self.assertEqual(info["legs_source"], "fallback")
            self.assertIn("3 legs", info["legs_fallback_why"])
            self.assertEqual(info["sources"]["UpperLeg_L"], "fallback")
            self.assertEqual(info["sources"]["Foot_L"], "floor_cluster")
            p = _pos(info)
            C, _ = rig_fit.floor_clusters(fx["V"], np.array([[0.1, 0, 0.0], [-0.1, 0, 0.0]]))
            for k, s in enumerate("LR"):
                self.assertLess(np.linalg.norm(p[f"Foot_{s}"][[0, 2]] - C[k]), 0.02 * fx["H"])
            self.assertEqual(info["checks"]["problems"], [])

    def test_reversed_mesh_raises_facing(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            fx = synth_bird.write(Path(t), reverse=True)
            with self.assertRaises(rig_build.BuildError) as cm:
                rig_build.build(fx["psx"], PLAN, rig_pred.Pred(fx["pred"]), Path(t) / "template")
            self.assertEqual(cm.exception.code, "facing")
            bj = json.loads((Path(t) / "template" / "build.json").read_text())
            self.assertFalse(bj["ok"])
            self.assertEqual(bj["error"]["code"], "facing")
            self.assertFalse((Path(t) / "template" / "skeleton.json").exists())

    def test_psx_facing_check_is_a_cue(self) -> None:
        # a mesh whose top leans back (tail taller than head): the top cue alone says -Z; the psx pass's facing_check
        # (toes + VLM: head +Z) is one more cue, so the build is not refused on that single weak cue
        class M:
            pass
        rng = np.random.default_rng(0)
        m = M()
        m.V = rng.uniform([-0.2, 0, -0.3], [0.2, 1.0, 0.3], (800, 3))
        m.V[m.V[:, 1] > 0.9, 2] -= 0.2
        m.y0, m.H = 0.0, 1.0
        legs = [np.array([0.1, 0.0]), np.array([-0.1, 0.0])]
        fc = rig_build.facing_check(m, None, None, legs, 0.0)
        self.assertEqual(fc["cues"]["top"]["vote"], -1)
        self.assertFalse(fc["agrees"])
        prior = {"head_axis": "+Z", "agrees": True, "votes": {"toes": 1, "vlm": 1}}
        fc = rig_build.facing_check(m, None, None, legs, 0.0, prior=prior)
        self.assertEqual(fc["cues"]["psx"]["vote"], 1)
        self.assertTrue(fc["agrees"])
        fc = rig_build.facing_check(m, None, None, legs, 0.0, prior={**prior, "votes": {"toes": 0, "vlm": 0}})
        self.assertEqual(fc["cues"]["psx"]["vote"], 0)  # a check with no votes behind it says nothing

    def test_no_prediction_builds_from_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            fx = synth_bird.write(Path(t))
            info = rig_build.build(fx["psx"], PLAN, None, Path(t) / "template")
            self.assertEqual(info["legs_source"], "fallback")
            self.assertEqual(info["checks"]["problems"], [])
            p = _pos(info)
            self.assertGreater(p["head"][2], p["hips"][2])

    def test_wrapper_plan_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            fx = synth_bird.write(Path(t))
            with self.assertRaises(rig_build.BuildError) as cm:
                rig_build.build(fx["psx"], rig_plan.load("flying_bird"), None, Path(t) / "template")
            self.assertEqual(cm.exception.code, "plan")

    def test_cli(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            asset = Path(t) / "out" / "synth"
            fx = synth_bird.write(asset)
            (asset / "rig").mkdir()
            fx["pred"].rename(asset / "rig" / "pred_rig.glb")  # rig_pred.predict's cache, keyed by the source's md5
            key = hashlib.md5((asset / "psx.glb").read_bytes()).hexdigest()
            (asset / "rig" / "pred_rig.json").write_text(json.dumps({"key": key, "seconds": 0}))
            out = Path(t) / "built"
            env = {**os.environ, "FACTORY_ROOT": t}
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "rig_build.py"), "--stem", "synth", "--out", str(out)],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("built 55 joints", r.stdout)
            self.assertTrue(json.loads((out / "build.json").read_text())["ok"])
            self.assertTrue((out / "skeleton.json").is_file())


class WingRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.info, cls.fx = synth_bird.build(Path(cls.tmp.name))
        cls.skel = cls.info["skeleton"]
        cls.names = cls.info["names"]
        cls.jpos = np.array(cls.info["pos"])
        cls.P = cls.fx["V"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _weights(self, no_wings: bool):
        jpos = self.jpos.copy()
        parent = self.info["parent_index"]
        if no_wings:  # wing bones pushed far away: the skin gives them nothing (a skin-tokens-like failure)
            for i, n in enumerate(self.names):
                if n.startswith("wing_"):
                    jpos[i] = jpos[i] + np.array([0, 50.0, 0])
        return synth_bird.distance_weights(self.P, jpos, parent)

    def test_repairs_when_wings_own_nothing(self) -> None:
        J, W = self._weights(no_wings=True)
        J2, W2, info = rig_build.wing_weight_repair(self.P, J, W, self.skel, PLAN)
        self.assertTrue(info["applied"], info)
        self.assertLess(info["share_before"], 0.03)
        self.assertGreater(info["share_after"], 0.3)
        self.assertEqual(J2.shape, J.shape)
        np.testing.assert_allclose(W2.sum(1), 1.0, atol=1e-5)
        wing = {i for i, n in enumerate(self.names) if n.startswith("wing_")}
        top = rig_metrics.top_bone(J2, W2)
        lat = np.abs(self.P[:, 0] - self.info["cx"]) > 0.15
        # every reassigned vertex went to a wing bone of its own side
        for v in np.nonzero(lat & np.isin(top, list(wing)))[0]:
            self.assertEqual(np.sign(self.P[v, 0] - self.info["cx"]), np.sign(self.jpos[top[v]][0] - self.info["cx"]))

    def test_no_change_when_wings_already_skinned(self) -> None:
        J, W = self._weights(no_wings=False)
        J2, W2, info = rig_build.wing_weight_repair(self.P, J, W, self.skel, PLAN)
        self.assertFalse(info["applied"], info)
        self.assertIs(J2, J)
        self.assertIs(W2, W)

    def test_accepts_the_synthesiser_skeleton_shape(self) -> None:
        J, W = self._weights(no_wings=True)
        par = self.info["parent_index"]
        rest_t = np.array([self.jpos[i] - (self.jpos[par[i]] if par[i] >= 0 else 0) for i in range(len(par))])
        skel = {"names": self.names, "parent": par, "rest_t": rest_t}
        _, _, a = rig_build.wing_weight_repair(self.P, J, W, self.skel, PLAN)
        _, _, b = rig_build.wing_weight_repair(self.P, J, W, skel, PLAN)
        self.assertTrue(b["applied"])
        self.assertAlmostEqual(a["share_after"], b["share_after"], delta=0.02)

    def test_plan_without_wings_is_a_no_op(self) -> None:
        J, W = self._weights(no_wings=True)
        plan = {**PLAN, "caps": {**PLAN["caps"], "wings": False}}
        _, _, info = rig_build.wing_weight_repair(self.P, J, W, self.skel, plan)
        self.assertFalse(info["applied"])


class SideRepairTests(WingRepairTests):  # shares the fixture (and re-runs the cheap wing tests)
    """side_weight_repair: weight on the other side's sided bones moves to the mirrored bone of the vertex's side."""

    def _inject(self):
        J, W = self._weights(no_wings=False)
        J, W = J.copy(), W.copy()
        k = {n: i for i, n in enumerate(self.names)}
        cx = self.info["cx"]
        # left-foot points (near Toes_L) get 0.35 of Toes_R in their second slot, as skin-tokens did on the chicken
        toe_l = self.jpos[k["Toes_L"]]
        near = np.nonzero((np.linalg.norm(self.P - toe_l, axis=1) < 0.08) & (self.P[:, 0] - cx > 0.03))[0][:20]
        self.assertGreater(len(near), 0)
        for v in near:
            J[v, 1] = k["Toes_R"]
            W[v, 1] = 0.35
            W[v, 0] = 0.65
            W[v, 2:] = 0
        return J, W, near, k

    def test_moves_wrong_side_weight_to_the_mirror_bone(self) -> None:
        J, W, near, k = self._inject()
        J2, W2, info = rig_build.side_weight_repair(self.P, J, W, self.skel, PLAN)
        self.assertTrue(info["applied"], info)
        self.assertIn("Toes_R", info["bones"])
        np.testing.assert_allclose(W2.sum(1), 1.0, atol=1e-5)
        right = {i for i, n in enumerate(self.names) if n.endswith("_R")}
        for v in near:
            self.assertFalse(any(W2[v, c] > 0 and J2[v, c] in right for c in range(J2.shape[1])), v)
        # the moved weight landed on Toes_L
        self.assertTrue(all((W2[v][J2[v] == k["Toes_L"]].sum() > 0.3) for v in near))

    def test_clean_weights_are_returned_unchanged(self) -> None:
        J, W = self._weights(no_wings=False)
        J2, W2, info = rig_build.side_weight_repair(self.P, J, W, self.skel, PLAN)
        if not info["applied"]:
            self.assertIs(J2, J)
            self.assertIs(W2, W)
        else:  # distance weights may bleed across the midline gap a little; never by much
            self.assertLess(info["moved_mass"], 0.02 * len(self.P))

    def test_midline_points_are_left_alone(self) -> None:
        J, W = self._weights(no_wings=False)
        J, W = J.copy(), W.copy()
        k = {n: i for i, n in enumerate(self.names)}
        v = int(np.argmin(np.abs(self.P[:, 0] - self.info["cx"])))
        J[v, 1] = k["Toes_R"]
        W[v, :] = [0.6, 0.4, 0, 0]
        J2, W2, _ = rig_build.side_weight_repair(self.P, J, W, self.skel, PLAN)
        self.assertIn(k["Toes_R"], set(np.asarray(J2[v])[np.asarray(W2[v]) > 0].tolist()))


class WingRegionRepairTests(WingRepairTests):  # shares the fixture (and re-runs the cheap wing tests)
    def test_region_goes_to_the_wings(self) -> None:
        J, W = self._weights(no_wings=True)
        J2, W2, info = rig_build.wing_region_repair(self.P, J, W, self.skel, PLAN)
        self.assertTrue(info["applied"], info)
        self.assertLess(info["share_before"], 0.5)
        self.assertGreaterEqual(info["share_after"], 0.5)
        np.testing.assert_allclose(W2.sum(1), 1.0, atol=1e-5)
        again = rig_build.wing_region_repair(self.P, J2, W2, self.skel, PLAN)[2]
        self.assertFalse(again["applied"] and again["share_before"] < 0.5, again)

    def test_wing_weight_off_the_body_side_goes_to_the_body(self) -> None:
        # points nearest a wing bone (within far_H) but off the wings' box on the body side (the rump, vent and belly
        # under a round body's feathers) lose their wing weight; on the drawn chicken a wing stroke dragged such
        # points off the hips and tail (Flap p99 stretch 2.9x)
        J, W = self._weights(no_wings=False)
        J, W = np.asarray(J).copy(), np.asarray(W, float).copy()
        H = float(self.skel.get("H") or np.ptp(self.P[:, 1]))
        par = self.info["parent_index"]
        segs = [(par[c], c) for c in range(len(self.names)) if par[c] >= 0]
        D = rig_metrics.seg_dist(self.P, self.jpos, segs)
        wing = [i for i, n in enumerate(self.names) if n.startswith("wing_")]
        other = [i for i in range(len(self.names)) if i not in wing]
        dw, dn = D[:, wing].min(1), D[:, other].min(1)
        m = rig_build._wing_side_margin(self.P, self.names, self.jpos, self.info["cx"], H)
        vs = np.nonzero((dw < dn) & (dw < 0.1 * H) & (m < -0.05 * H))[0]
        self.assertTrue(len(vs))
        for v in vs:  # bound wholly to their nearest wing bone (as the skin bound the chicken's rump to a feather)
            J[v] = [wing[int(np.argmin(D[v, wing]))], 0, 0, 0]
            W[v] = [1.0, 0, 0, 0]
        J2, W2, info = rig_build.wing_region_repair(self.P, J, W, self.skel, PLAN)
        self.assertTrue(info["applied"], info)
        M = rig.to_dense(J2, W2, len(self.names))
        self.assertLess(float(M[vs][:, wing].sum(1).max()), 1e-6)
        np.testing.assert_allclose(W2.sum(1), 1.0, atol=1e-5)

    def test_skinned_wings_are_left_alone_and_plan_without_wings_is_a_no_op(self) -> None:
        J, W = self._weights(no_wings=False)
        J2, W2, info = rig_build.wing_region_repair(self.P, J, W, self.skel, PLAN)
        if not info["applied"]:
            self.assertIs(J2, J)
        plan = {**PLAN, "caps": {**PLAN["caps"], "wings": False}}
        self.assertFalse(rig_build.wing_region_repair(self.P, J, W, self.skel, plan)[2]["applied"])


class SoleRepairTests(WingRepairTests):  # shares the fixture (and re-runs the cheap wing tests)
    def test_floor_points_on_the_tarsus_move_to_the_foot(self) -> None:
        J, W = self._weights(no_wings=False)
        J, W = J.copy(), W.copy()
        k = {n: i for i, n in enumerate(self.names)}
        floor = self.P[:, 1].min()
        sole = np.nonzero((self.P[:, 1] - floor < 0.01) & (self.P[:, 0] > self.info["cx"]))[0][:10]
        self.assertGreater(len(sole), 0)
        for v in sole:  # a hind toe bound to the tarsus
            J[v] = [k["AnkleLeg_L"], k["Foot_L"], 0, 0]
            W[v] = [0.9, 0.1, 0, 0]
        J2, W2, info = rig_build.sole_weight_repair(self.P, J, W, self.skel, PLAN)
        self.assertTrue(info["applied"], info)
        for v in sole:
            self.assertAlmostEqual(float(W2[v][J2[v] == k["Foot_L"]].sum()), 1.0, places=4)
        high = self.P[:, 1] - floor > 0.1
        np.testing.assert_allclose(rig.to_dense(J2, W2, len(self.names))[high],
                                   rig.to_dense(J, W, len(self.names))[high], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
