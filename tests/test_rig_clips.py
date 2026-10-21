#!/usr/bin/env python3
"""The clip synthesiser (scripts/rig_clips.py + rig_clips_gaits.py) on the synthetic bird (tests/fixtures/synth_bird.py)
with distance weights folded the way rig.py's clean stage folds them (root and leaf weight onto parents)."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import rig  # noqa: E402
import rig_build  # noqa: E402
import rig_clips  # noqa: E402
import rig_clips_gaits as gaits  # noqa: E402
import rig_ik  # noqa: E402
import rig_metrics  # noqa: E402
import rig_plan  # noqa: E402
import rig_pred  # noqa: E402
import rig_template  # noqa: E402
from fixtures import synth_bird  # noqa: E402

PLAN = rig_plan.load("ground_bird")
S = np.diag([-1.0, 1.0, 1.0])
STEM = "synth-bird-clips"


def bird(base: Path):
    """Build the fixture skeleton into <base>/<STEM>/rig/template and bind it with folded distance weights."""
    asset = base / STEM
    tdir = asset / "rig" / "template"
    d = synth_bird.write(asset)
    info = rig_build.build(d["psx"], PLAN, rig_pred.Pred(d["pred"]), tdir)
    pos = np.array(info["pos"])
    par = info["parent_index"]
    J, W = synth_bird.distance_weights(d["V"], pos, par)
    J, W, _ = rig.fold_tips(J, W, {c: p for c, p in enumerate(par) if p >= 0}, par.index(-1), info["names"].index("hips"))
    skel = {"names": info["names"], "parent": par, "pos": pos, "H": info["H"], "cx": info["cx"]}
    return asset, tdir, d, info, skel, J, W


def track_of(clip: dict, sk: gaits.Skel) -> gaits.Track:
    return rig_clips._track_from_clip(clip, sk)


class SynthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.asset, cls.tdir, cls.d, cls.info, cls.skel, cls.J, cls.W = bird(cls.base)
        cls.H = cls.d["H"]
        cls.sk = gaits.Skel(cls.skel)
        cls.clips, cls.rep = rig_clips.synthesise(PLAN, cls.skel, cls.d["V"], cls.d["F"], cls.J, cls.W, cls.H)
        cls.by = {c["name"]: c for c in cls.clips}
        cls.keys_before = set(rig.TEMPLATES)

    @classmethod
    def tearDownClass(cls) -> None:
        for k in [k for k in rig.TEMPLATES if k.startswith("built:")]:
            rig.TEMPLATES.pop(k, None)
        cls.tmp.cleanup()

    # ------------------------------------------------ shape and contract

    def test_generators_match_the_plan_format(self) -> None:
        self.assertEqual(set(rig_clips.GENERATORS), {"idle_bird", "walk_biped", "run_biped", "peck", "flap_folded",
                                                     "sleep_lie", "death_fall"})
        self.assertEqual(rig_plan._gens(), set(rig_clips.GENERATORS))
        for p in rig_plan.all_plans():
            self.assertEqual(rig_plan.validate(p), [], p["id"])

    def test_every_plan_clip_is_made_and_all_pass_on_the_fixture(self) -> None:
        self.assertEqual([c["name"] for c in self.clips], [c["name"] for c in PLAN["clips"]])
        self.assertEqual(self.rep["clips_failed"], [], self.rep["clips_failed"])
        for c in self.clips:
            self.assertTrue(c["passed"], (c["name"], c["reasons"]))
            self.assertGreaterEqual(c["energy"], next(s for s in PLAN["clips"] if s["name"] == c["name"])["min_energy"])
            F = len(c["times"])
            self.assertEqual(c["times"].dtype, np.float32)
            np.testing.assert_allclose(np.diff(c["times"]), 1 / 30, atol=1e-5)
            self.assertEqual(set(c["rot"]), set(range(len(self.sk.names))))
            for q in c["rot"].values():
                self.assertEqual(q.shape, (F, 4))
                np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-5)
            self.assertEqual(c["root"], self.sk.names.index("root"))
            self.assertEqual(c["root_t"].shape, (F, 3))
            for k in ("loop", "track", "params", "energy", "gates"):
                self.assertIn(k, c)
            self.assertIn("Foot_L", c["track"]["contact"])
        rj = rig_clips.report_json(self.rep)
        json.dumps(rj)
        for name, r in rj["clips"].items():
            self.assertEqual(set(r) >= {"params", "energy", "attempts", "gates", "passed", "reasons"}, True, name)

    def test_loop_seams_close(self) -> None:
        for c in self.clips:
            if c["loop"]:
                self.assertLessEqual(rig_metrics.loop_seam(c), 0.5, c["name"])
                np.testing.assert_allclose(c["root_t"][-1], c["root_t"][0], atol=1e-4 * self.H, err_msg=c["name"])

    def test_idle_starts_at_rest(self) -> None:
        c = self.by["Idle"]
        q0 = np.array([c["rot"][j][0] for j in range(len(self.sk.names))])
        ang = np.degrees(2 * np.arccos(np.clip(np.abs(q0[:, 3]), 0, 1)))
        self.assertLess(ang.max(), 0.05)
        np.testing.assert_allclose(c["root_t"][0], self.sk.rest[self.sk.root], atol=1e-6)

    def test_r_leg_is_the_mirrored_l_leg_half_a_cycle_later(self) -> None:
        for name in ("Walk", "Run"):
            c = self.by[name]
            tr = track_of(c, self.sk)
            pos, rot = tr.fk()
            n = tr.n
            self.assertEqual(n % 2, 0, name)
            cx = self.sk.cx
            for base in ("UpperLeg", "LowerLeg", "AnkleLeg", "Foot", "Toes"):
                L, Rr = self.sk.i(base + "_L"), self.sk.i(base + "_R")
                for f in range(n):
                    g = (f + n // 2) % n
                    want = S @ rot[g, L] @ S
                    err = np.degrees(np.linalg.norm(R.from_matrix(want.T @ rot[f, Rr]).as_rotvec()))
                    self.assertLess(err, 0.5, (name, base, f, err))
                    pm = pos[g, L].copy()
                    pm[0] = 2 * cx - pm[0]
                    self.assertLess(np.linalg.norm(pm - pos[f, Rr]) / self.H, 0.005, (name, base, f))
            # and the curve is not trivially symmetric: the L foot moves through the cycle
            self.assertGreater(np.ptp(pos[:, self.sk.i("Foot_L"), 2]) / self.H, 0.1, name)

    def test_mirror_delta_matches_the_r_wing_in_flap(self) -> None:
        tr = track_of(self.by["Flap"], self.sk)
        _, rot = tr.fk()
        for base in ("wing_1", "wing_3"):
            L, Rr = self.sk.i(base + "_L"), self.sk.i(base + "_R")
            for f in range(tr.F):
                np.testing.assert_allclose(rot[f, Rr], rig_ik.mirror_delta(rot[f, L]), atol=1e-5)

    # ------------------------------------------------ motion criteria

    def test_walk_feet_do_not_slide_and_the_head_holds(self) -> None:
        g = self.by["Walk"]["gates"]
        self.assertLessEqual(g["foot_slide"]["value"], 0.01)
        self.assertLessEqual(g["head_hold_drift_H"]["value"], 0.012)
        self.assertLessEqual(self.by["Run"]["gates"]["foot_slide"]["value"], 0.03)
        # the stance foot really is on the floor and really moves back at the walk speed
        c = self.by["Walk"]
        tr = track_of(c, self.sk)
        pos, _ = tr.fk()
        foot = pos[:, self.sk.i("Foot_L")]
        con = c["track"]["contact"]["Foot_L"]
        self.assertGreater(con.sum(), 3)
        v = c["params"]["v"]
        f = np.nonzero(con)[0]
        f = f[: (np.nonzero(np.diff(f) > 1)[0][0] + 1) if (np.diff(f) > 1).any() else len(f)]  # the first stance
        self.assertGreater(len(f), 2)
        dz = foot[f[-1], 2] - foot[f[0], 2]
        self.assertAlmostEqual(dz, -v * (f[-1] - f[0]) / 30, delta=0.002 * self.H)
        self.assertLess(np.abs(foot[f, 1] - self.sk.rest[self.sk.i("Foot_L")][1]).max() / self.H, 0.02)  # the floor guard may lift it

    def test_peck_beak_touches_down_with_feet_planted(self) -> None:
        c = self.by["Peck"]
        g = c["gates"]["beak_gap_H"]
        self.assertTrue(g["ok"])
        self.assertEqual(len(g["per_peck"]), 2)
        for x in g["per_peck"]:
            self.assertGreaterEqual(x, -0.005)
            self.assertLessEqual(x, 0.02)
        self.assertTrue(c["track"]["contact"]["beak"].any())
        tr = track_of(c, self.sk)
        pos, _ = tr.fk()
        for k in ("Foot_L", "Foot_R"):
            p = pos[:, self.sk.i(k)]
            self.assertLess(np.linalg.norm(p - p[0], axis=1).max() / self.H, 0.005)

    def test_sleep_rests_its_belly_on_the_floor(self) -> None:
        g = self.by["Sleep"]["gates"]
        self.assertLessEqual(g["belly_gap_H"]["value"], 0.03)
        self.assertGreaterEqual(g["floor_min_H"]["value"], -0.01)

    def test_death_falls_on_its_side_and_holds(self) -> None:
        c = self.by["Death"]
        self.assertFalse(c["loop"])
        k = int(round(0.3 * 30))
        last = len(c["times"]) - 1
        for q in c["rot"].values():
            np.testing.assert_allclose(q[-k:], np.repeat(q[-1:], k, 0), atol=1e-6)
        np.testing.assert_allclose(c["root_t"][-k:], np.repeat(c["root_t"][-1:], k, 0), atol=1e-6)
        self.assertTrue(c["gates"]["holds"]["ok"])
        tr = track_of(c, self.sk)
        _, rot = tr.fk()
        up = rot[last, self.sk.i("hips")] @ np.array([1.0, 0, 0])  # the creature's left side faces down
        self.assertLess(up[1], -0.9)
        self.assertGreaterEqual(c["gates"]["floor_min_H"]["value"], -0.01)

    def test_flap_spreads_the_wings_past_the_floor_angle(self) -> None:
        c = self.by["Flap"]
        self.assertGreaterEqual(c["gates"]["abduct_deg"]["value"], 20)
        self.assertGreater(c["gates"]["cshape"]["ext_ratio"][0], 1.03)
        self.assertLessEqual(c["gates"]["stretch_p99"]["value"], 2.5)

    def test_flap_lifts_the_wing_tips(self) -> None:
        # the main stroke is elevation: the folded wings' tips come up (a flank flare alone is not a flap)
        c = self.by["Flap"]
        self.assertIn("wing_rise_H", c["gates"])
        self.assertTrue(c["gates"]["wing_rise_H"]["ok"])
        self.assertGreaterEqual(c["gates"]["wing_rise_H"]["value"], 0.1)
        # measured on the wing SKIN (wing-owned points), not the tip joints; the skin shows above the back
        self.assertIn("tip_rise_H", c["gates"]["wing_rise_H"])
        self.assertIn("wing_clear_H", c["gates"])
        self.assertTrue(c["gates"]["wing_clear_H"]["ok"], c["gates"]["wing_clear_H"])

    def test_wing_skin_reads_the_skin_not_the_joints(self) -> None:
        # a stroke whose joints move but whose skin carries no wing weight rises nothing
        import rig_clips_gaits as gaits
        ctx = gaits.Ctx(PLAN, self.sk, self.d["V"], self.d["F"], self.J, np.asarray(self.W, float), self.H)
        spec = next(c for c in PLAN["clips"] if c["name"] == "Flap")
        tr = gaits.flap_folded(ctx, spec, 1.0)
        pos, rot = tr.fk()
        still = np.repeat(np.asarray(ctx.P, float)[None], tr.F, 0) + tr.root_off[:, None]
        w = rig_clips.wing_skin(ctx, still, tr)
        self.assertIsNotNone(w)
        self.assertGreater(tr.info["tip_rise_H"], 0.1)
        self.assertLess(abs(w["rise_H"]), 1e-6)

    def test_wing_weight_smooth_keeps_weight_sums_and_softens_the_edge(self) -> None:
        import rig_build
        J2, W2, info = rig_build.wing_weight_smooth(self.d["V"], self.d["F"], self.J, self.W, self.skel, PLAN)
        self.assertTrue(info["applied"])
        self.assertEqual(info["iters"], PLAN["chains"]["wings"]["smooth_iters"])
        np.testing.assert_allclose(np.asarray(W2).sum(1), 1.0, atol=1e-5)
        names = self.skel["names"]
        wing = [i for i, n in enumerate(names) if n.startswith("wing_")]
        M0 = rig.to_dense(self.J, self.W, len(names))[:, wing].sum(1)
        M1 = rig.to_dense(J2, W2, len(names))[:, wing].sum(1)
        self.assertGreater(int(((M1 > 0.01) & (M1 < 0.99)).sum()), int(((M0 > 0.01) & (M0 < 0.99)).sum()))
        off = rig_build.wing_weight_smooth(self.d["V"], self.d["F"], self.J, self.W, self.skel, PLAN, iters=0)
        self.assertFalse(off[2]["applied"])
        self.assertIs(off[0], self.J)

    # ------------------------------------------------ energy search

    def test_energy_search_lowers_energy_then_fails_below_min_energy(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["gates"]["stretch_p99"] = 1.0  # nothing can pass
        plan["clips"] = [c for c in plan["clips"] if c["name"] in ("Walk", "Flap")]
        clips, rep = rig_clips.synthesise(plan, self.skel, self.d["V"], self.d["F"], self.J, self.W, self.H)
        walk = rep["clips"]["Walk"]
        self.assertFalse(walk["passed"])
        self.assertEqual([a["energy"] for a in walk["attempts"]], [1.0, 0.8, 0.64])
        self.assertTrue(any("stretch" in r for r in walk["reasons"]))
        self.assertTrue(any("min_energy" in r for r in walk["reasons"]))
        self.assertEqual([f["name"] for f in rep["clips_failed"]], ["Walk", "Flap"])
        self.assertEqual([c["name"] for c in clips], ["Walk", "Flap"])  # failed clips are returned, never dropped
        self.assertFalse(clips[0]["passed"])
        flap = rep["clips"]["Flap"]
        es = [a["energy"] for a in flap["attempts"]]
        self.assertLessEqual(len(es), rig_clips.FLAP_TRIES)
        self.assertGreaterEqual(min(es) * 60, 20 - 1e-6)  # never tried under the abduct_min floor
        self.assertTrue(any("abduct_min" in r or "min_energy" in r or "tries" in r for r in flap["reasons"]))
        self.assertIn(0.8, es)
        self.assertIn("wing_weight_repair", flap["attempts"][es.index(0.8)])  # the repair is tried once, at e = 0.8

    def test_flap_repairs_wing_weights_once_and_returns_them(self) -> None:
        plan = copy.deepcopy(PLAN)
        plan["clips"] = [c for c in plan["clips"] if c["name"] == "Flap"]
        plan["gates"]["stretch_p99"] = 1.0
        # strip all weight off the wing bones so the repair triggers
        names = self.skel["names"]
        wing = [i for i, n in enumerate(names) if n.startswith("wing_")]
        M = rig.to_dense(self.J, self.W, len(names))
        for i in wing:
            M[:, names.index("spine_1")] += M[:, i]
            M[:, i] = 0
        J0, W0 = rig.to_top4(M)
        _, rep = rig_clips.synthesise(plan, self.skel, self.d["V"], self.d["F"], J0, W0, self.H)
        att = rep["clips"]["Flap"]["attempts"]
        rep_at = [a for a in att if "wing_weight_repair" in a]
        self.assertEqual(len(rep_at), 1)
        self.assertTrue(rep["wing_weight_repair"]["applied"])
        self.assertFalse(np.array_equal(rep["W"], W0))
        es = [a["energy"] for a in att]
        self.assertEqual(es.count(0.8), 2)  # retried at the same energy on the repaired weights

    def test_names_index_space(self) -> None:
        """J and the rot keys follow `names` when given (e.g. skinned_raw.glb's own order)."""
        names = list(reversed(self.skel["names"]))
        perm = np.array([self.skel["names"].index(n) for n in names])  # new index -> skel index
        inv = np.argsort(perm)  # skel index -> new index
        J2 = inv[self.J]
        plan = copy.deepcopy(PLAN)
        plan["clips"] = [c for c in plan["clips"] if c["name"] == "Peck"]
        clips, rep = rig_clips.synthesise(plan, self.skel, self.d["V"], self.d["F"], J2, self.W, self.H, names=names)
        c = clips[0]
        self.assertTrue(c["passed"], c["reasons"])
        self.assertEqual(c["root"], names.index("root"))
        ref = self.by["Peck"]
        for k_new, q in c["rot"].items():
            np.testing.assert_allclose(np.abs(q), np.abs(ref["rot"][int(perm[k_new])]), atol=1e-5)
        self.assertEqual(rep["names"], names)


class TemplateRoundTrip(unittest.TestCase):
    """clips.glb written by rig_template, re-baked by rig.bake_clips, and read by rig_pred.template_floor_track."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.asset, cls.tdir, cls.d, cls.info, cls.skel, cls.J, cls.W = bird(cls.base)
        cls.H = cls.d["H"]
        cls.sk = gaits.Skel(cls.skel)
        cls.clips, cls.rep = rig_clips.synthesise(PLAN, cls.skel, cls.d["V"], cls.d["F"], cls.J, cls.W, cls.H)
        cls.rig_glb = cls.tdir / "rig.glb"
        rig_template.write_rig(cls.tdir / "skeleton.json", cls.rig_glb)
        cls.src, cls.sbin = rig.read_glb(cls.d["psx"])
        cls.ok = [c for c in cls.clips if c["passed"]]
        rig_template.write_clips(cls.rig_glb, cls.src, cls.sbin, cls.J, cls.W, cls.ok, cls.tdir / "clips.glb",
                                 names=cls.skel["names"])
        rig_clips.write_report(cls.tdir / "clips_report.json", cls.rep)
        rig_template.write_template_json(cls.tdir, PLAN, [c["name"] for c in cls.ok])
        cls.template = rig_template.load_template(cls.asset)

    @classmethod
    def tearDownClass(cls) -> None:
        for k in [k for k in rig.TEMPLATES if k.startswith("built:")]:
            rig.TEMPLATES.pop(k, None)
        cls.tmp.cleanup()

    def test_verify_passes(self) -> None:
        probs = rig_template.verify(self.rig_glb, self.tdir / "clips.glb", PLAN, out=False)
        self.assertEqual([p for p in probs if p["level"] == "error"], [])
        self.assertEqual(self.template["default"], PLAN["default"])

    def test_bake_clips_round_trip(self) -> None:
        rdoc, _ = rig.read_glb(self.rig_glb)
        skel = rig.Skeleton(rdoc)
        store = rig_template.ref_store(self.rig_glb, {})
        names = [c["name"] for c in self.ok]
        baked, missing = rig.bake_clips(self.template, names, rdoc, skel, store)
        self.assertEqual(missing, [])
        node_of = {n.get("name"): i for i, n in enumerate(rdoc["nodes"])}
        par = self.sk.parent
        for b, c in zip(baked, self.ok):
            self.assertEqual(b["name"], c["name"])
            self.assertEqual(len(b["times"]), len(c["times"]))
            mine = track_of(c, self.sk).fk()[0]
            F = len(c["times"])
            q = np.zeros((F, len(self.sk.names), 4))
            for j, n in enumerate(self.sk.names):
                q[:, j] = b["rot"][node_of[n]]
            for f in range(F):
                pos, _ = rig_ik.fk(par, self.sk.rest_t, q[f], root_t=b["root_t"][f])
                err = np.linalg.norm(pos - mine[f], axis=1).max() / self.H
                self.assertLess(err, 1e-4, (c["name"], f, err))

    def test_template_floor_track_reads_peck(self) -> None:
        times = {"Peck": next(c for c in self.ok if c["name"] == "Peck")["times"]}
        tr = rig_pred.template_floor_track(self.template, ["Peck"], ["Foot_L", "Foot_R"], times)
        self.assertIn("Peck", tr["clips"])
        t = tr["clips"]["Peck"]
        self.assertEqual(len(t["low"]), len(times["Peck"]))
        self.assertEqual(set(t["foot"]), {"Foot_L", "Foot_R"})
        self.assertGreater(tr["H"], 0.5 * self.H)
        self.assertLess(np.abs(t["low"]).max(), 0.03 * self.H)

    def test_cli_remakes_one_clip_and_keeps_the_rest(self) -> None:
        old_out = rig.OUT
        rig.OUT = self.base
        try:
            before = rig.read_glb(self.tdir / "clips.glb")[0]
            names_before = [a["name"] for a in before["animations"]]
            code = rig_clips.main(["--stem", STEM, "--clips", "Peck"])
        finally:
            rig.OUT = old_out
        self.assertEqual(code, 0)
        after = rig.read_glb(self.tdir / "clips.glb")[0]
        self.assertEqual([a["name"] for a in after["animations"]], names_before)
        rep = json.loads((self.tdir / "clips_report.json").read_text())
        self.assertEqual(set(rep["clips"]), {c["name"] for c in PLAN["clips"]})
        self.assertEqual(rep["clips_failed"], [])
        probs = rig_template.verify(self.rig_glb, self.tdir / "clips.glb", PLAN, out=False)
        self.assertEqual([p for p in probs if p["level"] == "error"], [])


if __name__ == "__main__":
    unittest.main()
