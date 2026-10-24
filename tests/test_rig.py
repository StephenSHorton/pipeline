#!/usr/bin/env python3
"""Rigging maths in scripts/rig.py: weight locality, template placement, clip retargeting; the /rig route."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import factory_http  # noqa: E402
import rig  # noqa: E402
import rest_check  # noqa: E402
import rig_inspect  # noqa: E402
import rig_motion  # noqa: E402
import rig_fit  # noqa: E402
import rig_pred  # noqa: E402
import flow  # noqa: E402

HORSE = rig.M2M / "rigs" / "rig-horse.glb"


@unittest.skipUnless(HORSE.is_file(), "vendor/mesh2motion-app is not checked out")
class RigMathTests(unittest.TestCase):
    def test_far_influences_are_dropped(self) -> None:
        joints = np.array([[0, 0, 0], [0, 1, 0], [5, 0, 0]], float)  # bone 0->1 runs up the y axis; joint 2 is far away
        child = {0: 1}
        P = np.array([[0.05, 0.5, 0]])
        J = np.array([[0, 2, 0, 0]])
        W = np.array([[0.7, 0.3, 0, 0]])
        J2, W2, dropped = rig.local_weights(P, J, W, joints, child, height=1.0)
        self.assertEqual(dropped, 1)
        self.assertAlmostEqual(float(W2[0, 0]), 1.0)
        self.assertAlmostEqual(float(W2[0, 1]), 0.0)

    def test_template_is_placed_on_the_mesh(self) -> None:
        # A box 1.2 m tall, centred 3 m away: the posed template must end up the same height, on its floor, over it.
        rng = np.random.default_rng(0)
        mesh = rng.uniform([-0.3, 0.0, -1.0], [0.3, 1.2, 1.0], (4000, 3)) + np.array([3.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "posed.glb"
            doc, ref, _ = rig.posed_template(rig.TEMPLATES["quadruped"], out, mesh)
            posed, _ = rig.read_glb(out)
        skel = rig.Skeleton(posed)
        world = skel.world({})
        pts = np.array([world[i][0] for i in posed["skins"][0]["joints"]])
        self.assertAlmostEqual(float(pts[:, 1].min()), 0.0, places=3)
        self.assertAlmostEqual(float(np.ptp(pts[:, 1])), 1.2, places=3)
        root = next(i for i in posed["skins"][0]["joints"] if skel.parent.get(i) not in posed["skins"][0]["joints"])
        self.assertAlmostEqual(float(world[root][0][0]), 3.0, places=1)
        self.assertIn("hips", ref)

    def test_reference_clip_retargets_to_the_rest_pose(self) -> None:
        # Baking the reference clip onto a skeleton bound in that pose must give back its own rest rotations.
        template = rig.TEMPLATES["quadruped"]
        rng = np.random.default_rng(1)
        mesh = rng.uniform([-0.4, 0.0, -1.1], [0.4, 1.5, 1.1], (3000, 3))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "posed.glb"
            _, ref, _ = rig.posed_template(template, out, mesh)
            ours, _ = rig.read_glb(out)
        skel = rig.Skeleton(ours)
        clips, missing = rig.bake_clips(template, [template["ref"]], ours, skel, ref)
        self.assertEqual(missing, [])
        first = clips[0]
        for node, quats in first["rot"].items():
            rest = rig.R.from_quat(skel.rest[node][1])
            got = rig.R.from_quat(quats[0])
            self.assertLess(float((rest.inv() * got).magnitude()), 1e-3, ours["nodes"][node].get("name"))


class FitTests(unittest.TestCase):
    def test_a_far_top_influence_is_dropped(self) -> None:
        # bone 0->1 runs up y; joint 2 is far. A vertex on bone 0 whose every influence is on joint 2 goes to bone 0.
        joints = np.array([[0, 0, 0], [0, 1, 0], [5, 0, 0]], float)
        P = np.array([[0.05, 0.5, 0]])
        J2, W2, dropped = rig.local_weights(P, np.array([[2, 0, 0, 0]]), np.array([[1.0, 0, 0, 0]]), joints, {0: 1}, height=1.0)
        self.assertEqual(int(J2[0, 0]), 0)
        self.assertAlmostEqual(float(W2[0, 0]), 1.0)

    def test_close_yaws_go_to_the_cues(self) -> None:
        import rig_fit
        reg = {0: {"chamfer": 0.058}, 90: {"chamfer": 0.078}, 180: {"chamfer": 0.050}, 270: {"chamfer": 0.077}}
        head_minus_z = {"z": {"toes": {"vote": -1}}, "x": {}}
        self.assertEqual(rig_fit.choose_yaw(reg, head_minus_z, None)["yaw"], 180)
        head_plus_z = {"z": {"toes": {"vote": 1}, "narrow": {"vote": 1}}, "x": {}}
        got = rig_fit.choose_yaw(reg, head_plus_z, {"vote": 1})
        self.assertEqual((got["yaw"], got["decided_by"]), (0, "geometric+vlm"))
        far = {0: {"chamfer": 0.05}, 90: {"chamfer": 0.09}, 180: {"chamfer": 0.08}, 270: {"chamfer": 0.09}}
        self.assertEqual(rig_fit.choose_yaw(far, head_minus_z, None)["decided_by"], "registration")

    def test_floor_clusters_split_by_side(self) -> None:
        import rig_fit
        rng = np.random.default_rng(4)
        body = rng.uniform([-0.3, 0.3, -0.5], [0.3, 1.0, 0.5], (3000, 3))
        feet = np.vstack([rng.normal([x, 0.01, z], 0.02, (200, 3)) for x, z in ((0.15, 0.4), (-0.15, 0.4), (0.15, -0.4), (-0.15, -0.4))])
        seeds = np.array([[0.2, 0, 0.6], [-0.2, 0, 0.5], [0.2, 0, -0.8], [-0.2, 0, -0.3]])  # a staggered stance
        C, n = rig_fit.floor_clusters(np.vstack([body, feet]), seeds)
        np.testing.assert_allclose(C, [[0.15, 0.4], [-0.15, 0.4], [0.15, -0.4], [-0.15, -0.4]], atol=0.03)

    def test_winding_number_inside_and_out(self) -> None:
        import rig_fit
        import trimesh
        box = trimesh.creation.box((1, 1, 1))
        w = rig_fit.winding_number(np.asarray(box.vertices), np.asarray(box.faces), np.array([[0, 0, 0], [2, 0, 0]], float))
        np.testing.assert_allclose(np.abs(w), [1, 0], atol=1e-6)


class WeightCleanupTests(unittest.TestCase):
    def test_corners_at_one_point_share_weights(self) -> None:
        # Two triangles' corners at the same spot with different bones would tear apart when the mesh moves.
        P = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], float)
        J = np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]])
        W = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], float)
        J2, W2, split = rig.share_corners(P, J, W)
        self.assertEqual(split, 1)
        a = rig.to_dense(J2, W2, 4)
        np.testing.assert_allclose(a[0], a[1])
        np.testing.assert_allclose(a[0], [0, 0.5, 0.5, 0], atol=1e-6)
        np.testing.assert_allclose(a[2], [0, 0, 0, 1], atol=1e-6)

    def test_root_and_tip_weight_moves_to_parents(self) -> None:
        # root(0) -> hips(1) -> leg(2) -> leg_leaf(3)
        parent = {1: 0, 2: 1, 3: 2}
        J = np.array([[0, 3, 0, 0], [2, 0, 0, 0]])
        W = np.array([[0.5, 0.5, 0, 0], [1, 0, 0, 0]], float)
        J2, W2, moved = rig.fold_tips(J, W, parent, root=0, hips=1)
        a = rig.to_dense(J2, W2, 4)
        np.testing.assert_allclose(a[0], [0, 0.5, 0.5, 0], atol=1e-6)
        np.testing.assert_allclose(a[1], [0, 0, 1, 0], atol=1e-6)
        self.assertAlmostEqual(moved, 0.5)


class InspectTests(unittest.TestCase):
    def test_split_corners_are_counted(self) -> None:
        P = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], float)
        M = np.array([[1, 0], [0, 1], [1, 0]], float)
        seam, split = rig_inspect.corner_split(P, M)
        self.assertEqual(seam["positions"], 2)
        self.assertEqual(seam["positions_with_conflicting_top_bone"], 1)
        np.testing.assert_allclose(split, [0.5, 0.5, 0])

    def test_head_turn_is_measured_from_above(self) -> None:
        # neck at the origin, head bone pointing +z; the head's vertices sit 30 degrees off to the side.
        names = ["neck", "head", "nose"]
        parent = [-1, 0, 1]
        jp = np.array([[0, 1, 0], [0, 1, 0.2], [0, 1, 0.5]], float)
        a = np.radians(30)
        head = np.array([[np.sin(a) * 0.5, 1, np.cos(a) * 0.5]] * 20) + np.random.default_rng(0).normal(0, 1e-3, (20, 3))
        P = np.vstack([head, [[0, 0, 0]]])
        top = np.array([1] * 20 + [0])
        got = rig_inspect.head_turn(names, parent, jp, P, top)
        self.assertAlmostEqual(got["off_deg"], 30, delta=1)

    def test_bone_names_read_as_words(self) -> None:
        for name, words in (("back_leg_foot_r", "right back foot"), ("front_leg_lower_l", "left front lower leg"),
                            ("ear_2_l", "left ear"), ("spine_4", "spine"), ("mouth_tip", "mouth tip"), ("head", "head")):
            self.assertEqual(rig_inspect.plain_bone(name), words)

    def test_summary_line(self) -> None:
        info = {"class": "quadruped", "made": 1, "rest": {"measured": {"head_yaw_deg": -21.9, "turned": True},
                                                           "straightened": {"head_yaw_deg": 0.4}}}
        checks = [{"level": "pass", "key": "seam", "step": "skin", "label": "Corners move together"},
                  {"level": "warn", "key": "fit", "step": "skeleton", "label": "Right back foot 0.33 m off"},
                  {"level": "pass", "key": "head", "step": "skeleton", "label": "Head on its bone", "off_deg": 3.0}]
        motion = {"clips": {"Walk": {"tears": 0, "stretch_max": 4.2}, "Kick": {"tears": 0, "stretch_max": 6.47}}}
        got = rig_inspect.summarize(info, checks, motion, 0)
        self.assertEqual(got["line"], "No tears · head straightened 22° · worst stretch 6.5× (Kick)")
        self.assertEqual(got["level"], "warn")
        self.assertEqual(got["head"], {"before": -21.9, "after": 0.4, "straightened": True, "off_deg": 3.0, "made_before": False})
        self.assertEqual(got["worst_stretch"], {"clip": "Kick", "x": 6.5, "metric": "max"})
        self.assertEqual(got["problems"], [{"level": "warn", "key": "fit", "label": "Right back foot 0.33 m off"}])
        old = rig_inspect.summarize({"class": "quadruped"}, [
            {"level": "fail", "key": "seam", "step": "skin", "label": "Tears at 51 points"},
            {"level": "fail", "key": "head", "step": "skeleton", "label": "Head 24° off its bone", "off_deg": 24.0}], None, 51)
        self.assertEqual(old["line"], "Tears at 51 points · head 24° off")
        self.assertIsNone(old["motion"])
        self.assertIsNone(old["worst_stretch"])
        kept = rig_inspect.summarize({"class": "quadruped", "rest": {"measured": {"head_yaw_deg": 22.2, "turned": True}}},
                                     [{"level": "fail", "key": "motion", "step": "motion", "label": "", "clip": "Walk", "points": 37}],
                                     {"clips": {"Walk": {"tears": 37, "stretch_max": 3.0}}}, 0)
        self.assertEqual(kept["line"], "Walk tears at 37 points · head kept turned 22° · worst stretch 3.0× (Walk)")
        before = rig_inspect.summarize({"class": "quadruped", "rest": {"measured": {"head_yaw_deg": -22.0, "turned": True},
                                                                     "straightened": None, "before_straightening": True}},
                                       [{"level": "fail", "key": "seam", "step": "skin", "label": "Tears at 51 points"}], None, 51)
        self.assertEqual(before["line"], "Tears at 51 points · head turned 22°")


class MotionTests(unittest.TestCase):
    """Two triangles of a flat-shaded quad (no shared corners); bone 1 slides up out of the page."""
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = np.arange(6).reshape(2, 3)

    def run_clip(self, bone: np.ndarray) -> dict:
        J = np.c_[bone, np.zeros((6, 3), int)]
        W = np.c_[np.ones(6), np.zeros((6, 3))]
        moved = np.eye(4)
        moved[2, 3] = 0.5
        poses = [(0.0, np.stack([np.eye(4), np.eye(4)])), (1.0, np.stack([np.eye(4), moved]))]
        return rig_motion.measure_arrays(self.P, self.F, J, W, bone, ["a", "b"], poses)

    def test_corners_on_different_bones_tear(self) -> None:
        got = self.run_clip(np.array([0, 0, 0, 1, 1, 1]))  # each triangle on its own bone
        self.assertEqual(got["tears"], 2)
        self.assertEqual(got["tear_t"], 1.0)
        self.assertAlmostEqual(got["worst_gap_pct"], 50.0, places=1)

    def test_shared_weights_do_not_tear(self) -> None:
        got = self.run_clip(np.array([0, 1, 0, 1, 1, 0]))  # by point: x = 1 follows bone 1
        self.assertEqual(got["tears"], 0)
        self.assertIsNone(got["tear_t"])
        self.assertGreater(got["stretch_max"], 1.0)
        self.assertGreaterEqual(got["stretch_p99"], 1.0)
        self.assertEqual(got["worst_t"], 1.0)
        self.assertIn(got["worst_bone"], ("a", "b"))


def toy_animal(head_yaw_deg: float) -> np.ndarray:
    """Points of a box body along +z with a raised neck and head swung sideways by head_yaw_deg."""
    rng = np.random.default_rng(2)
    body = rng.uniform([-0.3, 0.6, -1.0], [0.3, 1.2, 0.6], (6000, 3))
    legs = rng.uniform([-0.25, 0.0, -0.9], [0.25, 0.6, 0.5], (1500, 3))
    t = rng.uniform(0, 1, 3000)
    a = np.radians(head_yaw_deg) * t
    neck = np.c_[np.sin(a) * 0.6 * t, 1.1 + 0.4 * t, 0.6 + np.cos(a) * 0.6 * t] + rng.normal(0, 0.06, (3000, 3))
    return np.vstack([body, legs, neck])


class RestCheckTests(unittest.TestCase):
    def test_a_turned_head_is_found_and_straightened(self) -> None:
        turned = rest_check.measure(toy_animal(25))
        self.assertTrue(turned["turned"])
        self.assertGreater(abs(turned["head_yaw_deg"]), 8)
        P, _, _ = rest_check.straighten_points(toy_animal(25))
        self.assertLess(abs(rest_check.measure(P)["head_yaw_deg"]), 2)

    def test_a_straight_head_is_left_alone(self) -> None:
        self.assertFalse(rest_check.measure(toy_animal(0))["turned"])

    def test_upright_meshes_are_not_checked(self) -> None:
        person = np.random.default_rng(3).uniform([-0.3, 0, -0.15], [0.3, 1.8, 0.15], (3000, 3))
        self.assertIsNone(rest_check.measure(person))


class RigRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "out" / "cow").mkdir(parents=True)
        self.factory = factory_http.Factory(self.root)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), factory_http.make_handler(self.factory))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._tmp.cleanup()

    def post(self, path: str, data: dict) -> tuple[int, dict]:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(self.base + path, data=json.dumps(data).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_rejects_unknown_asset_and_class(self) -> None:
        self.assertEqual(self.post("/rig/missing", {})[0], 404)
        self.assertEqual(self.post("/rig/cow", {"class": "dragon"})[0], 400)

    def test_straighten_and_asset_root_reach_rig_py(self) -> None:
        specs = []
        self.factory._run_rig_job = specs.append
        self.assertEqual(self.post("/rig/cow", {"class": "quadruped", "straighten": "off"})[0], 202)
        self.assertEqual(specs[0]["argv"][-4:], ["--class", "quadruped", "--straighten", "off"])
        self.factory._job = None
        self.post("/rig/cow", {})
        self.assertNotIn("--straighten", specs[1]["argv"])
        self.factory._job = None
        status, body = self.post("/rig/cow", {"straighten": "sideways"})
        self.assertEqual((status, body["error"]), (400, "straighten must be auto or off"))
        self.assertEqual(factory_http.rig_env(self.root)["FACTORY_ROOT"], str(self.root))

    def test_rig_py_writes_under_factory_root(self) -> None:
        import os
        import subprocess

        env = {**os.environ, "FACTORY_ROOT": str(self.root)}
        run = lambda stem: subprocess.run([sys.executable, str(ROOT / "scripts" / "rig.py"), "--stem", stem, "--class", "quadruped"],  # noqa: E731
                                          env=env, capture_output=True, text=True, timeout=120)
        self.assertIn("no textured mesh", run("cow").stderr)  # found out/cow under FACTORY_ROOT
        self.assertIn("no asset", run("radio").stderr)  # the repo's own out/radio isn't looked at

    def test_inspect_route(self) -> None:
        self.assertEqual(self.post("/rig/missing/inspect", {})[0], 404)
        status, body = self.post("/rig/cow/inspect", {})
        self.assertEqual((status, body["error"]), (400, "this asset has no rig yet"))
        asset = self.root / "out" / "cow"
        (asset / "rig").mkdir()
        (asset / "rig" / "rig.json").write_text('{"rigged": "rigged.glb"}')
        for rel in ("rigged.glb", "rig/template_posed.glb"):
            (asset / rel).write_bytes(b"glb")
        status, body = self.post("/rig/cow/inspect", {})
        self.assertEqual((status, body["error"]), (400, "this version is missing rig/skinned_raw.glb; Re-rig instead"))
        (asset / "rig" / "skinned_raw.glb").write_bytes(b"glb")
        flow.snapshot(self.root, "cow", "rig")

        def rebuild(folder: Path) -> dict:
            (folder / "rig" / "inspect.json").write_text('{"version": 2, "checks": []}')
            (folder / "rig" / "summary.json").write_text('{"line": "No tears"}')
            return {"checks": [], "summary": {"line": "No tears"}}
        real, rig_inspect.rebuild = rig_inspect.rebuild, rebuild
        try:
            self.factory._job = {"state": "running", "stem": "other"}
            self.assertEqual(self.post("/rig/cow/inspect", {})[0], 409)
            self.factory._job = None
            status, body = self.post("/rig/cow/inspect", {})
        finally:
            rig_inspect.rebuild = real
        self.assertEqual((status, body["take"], body["summary"]), (200, "t1", {"line": "No tears"}))
        rig = next(s for s in flow.state(self.root, "cow")["stages"] if s["stage"] == "rig")
        self.assertEqual([t["id"] for t in rig["takes"]], ["t1"])
        self.assertFalse(rig["unsaved"])
        self.assertEqual(rig["takes"][0]["summary"], {"line": "No tears"})

    def test_versions_hold_still_while_a_rig_is_checked(self) -> None:
        asset = self.root / "out" / "cow"
        (asset / "rig").mkdir()
        (asset / "rig" / "rig.json").write_text('{"rigged": "rigged.glb"}')
        for rel in ("rig/skinned_raw.glb", "rig/template_posed.glb"):
            (asset / rel).write_bytes(b"glb")
        (asset / "rigged.glb").write_bytes(b"glb one")
        flow.snapshot(self.root, "cow", "rig")
        (asset / "rigged.glb").write_bytes(b"glb two-longer")
        flow.snapshot(self.root, "cow", "rig")
        flow.pick(self.root, "cow", "rig", "t1")
        seen = []

        def rebuild(folder: Path) -> dict:
            seen.append(self.post("/flow/cow/pick", {"stage": "rig", "take": "t2"})[0])  # a version picked mid-check
            (folder / "rigged.glb").write_bytes(b"glb two-longer")  # as if it had gone through anyway
            (folder / "rig" / "inspect.json").write_text("{}")
            return {"checks": [], "summary": {}}
        real, rig_inspect.rebuild = rig_inspect.rebuild, rebuild
        try:
            status, body = self.post("/rig/cow/inspect", {})
        finally:
            rig_inspect.rebuild = real
        self.assertEqual(seen, [409])
        self.assertEqual((status, body["take"]), (200, None))  # the rig on disk changed: no version is overwritten
        self.assertEqual((asset / "takes" / "rig" / "t1" / "rigged.glb").read_bytes(), b"glb one")


def fake_pred(joints: list[tuple[str, int, tuple[float, float, float]]], P: np.ndarray) -> rig_pred.Pred:
    """A predicted skeleton without a GLB: (name, parent index or -1, position) per joint, and mesh points."""
    p = object.__new__(rig_pred.Pred)
    p.names = [n for n, _, _ in joints]
    p.par = np.array([k for _, k, _ in joints])
    p.jp = np.array([x for _, _, x in joints], float)
    p.kids = {k: [c for c in range(len(joints)) if p.par[c] == k] for k in range(len(joints))}
    p.root = [k for k in range(len(joints)) if p.par[k] < 0]
    p.P = P
    p.lo, p.hi = P.min(0), P.max(0)
    p.H = float(p.hi[1] - p.lo[1])
    return p


class PredTests(unittest.TestCase):
    """The predicted skeleton as the fitting target (scripts/rig_pred.py)."""
    BOX = np.random.default_rng(0).uniform([-0.25, 0, -0.7], [0.25, 1, 0.7], (4000, 3))

    def bird(self, hanging_tail: bool = False) -> rig_pred.Pred:
        j = [("root", -1, (0, 0.5, 0)),
             ("neck", 0, (0, 0.7, 0.3)), ("head", 1, (0, 0.9, 0.5)), ("beak", 2, (0, 0.95, 0.65)),
             ("tail1", 0, (0, 0.6, -0.35)), ("tail2", 4, (0, 0.75, -0.6)),
             ("legL1", 0, (0.15, 0.35, 0)), ("legL2", 6, (0.15, 0.2, 0.02)), ("legL3", 7, (0.15, 0.05, 0.05)),
             ("legR1", 0, (-0.15, 0.35, 0)), ("legR2", 9, (-0.15, 0.2, 0.02)), ("legR3", 10, (-0.15, 0.05, 0.05))]
        if hanging_tail:  # a cow's tail drops to near the floor down the middle: not a third leg
            j[4] = ("tail1", 0, (0, 0.5, -0.5))
            j[5] = ("tail2", 4, (0, 0.1, -0.55))
        return fake_pred(j, self.BOX)

    def test_chains_are_labelled_by_the_facing(self) -> None:
        p = self.bird()
        lab = rig_pred.label(p, np.array([0, 0, 1.0]))
        got = {d["names"][0]: d["label"] for d in lab["chains"]}
        self.assertEqual(got, {"neck": "head", "tail1": "tail", "legL1": "leg", "legR1": "leg"})
        self.assertEqual(rig_pred.head_chain_yaw(p, lab), 0)
        back = rig_pred.label(p, np.array([0, 0, -1.0]))  # facing the other way swaps head and tail
        self.assertEqual({d["names"][0]: d["label"] for d in back["chains"]}["tail1"], "head")
        self.assertEqual(rig_pred.head_chain_yaw(p, back), 180)

    def test_a_tail_down_the_middle_is_not_a_leg(self) -> None:
        lab = rig_pred.label(self.bird(hanging_tail=True), np.array([0, 0, 1.0]))
        got = {d["names"][0]: d["label"] for d in lab["chains"]}
        self.assertEqual(got["tail1"], "tail")
        self.assertEqual(lab["legs"], 2)

    def test_a_chain_is_laid_along_a_line_by_arc_length(self) -> None:
        P = {"a": np.zeros(3), "b": np.array([1.0, 0, 0]), "c": np.array([2.0, 0, 0]), "toe": np.array([2.0, 0, 0.3])}
        children = {"a": ["b"], "b": ["c"], "c": ["toe"]}
        line = np.array([[0, 0, 0], [0, -1.2, 0], [1.2, -1.2, 0]], float)  # 2.4 long: the chain stretches 1.2x
        r = rig_pred.lay(P, children, ["a", "b", "c"], line, keep_tail_rigid=True)
        self.assertAlmostEqual(r["length_ratio"], 1.2, places=6)
        np.testing.assert_allclose(P["b"], [0, -1.2, 0], atol=1e-9)
        np.testing.assert_allclose(P["c"], [1.2, -1.2, 0], atol=1e-9)
        np.testing.assert_allclose(P["toe"] - P["c"], [0, 0, 0.3], atol=1e-9)  # the toe only shifts
        P2 = {"a": np.zeros(3), "b": np.array([1.0, 0, 0])}
        r2 = rig_pred.lay(P2, {"a": ["b"]}, ["a", "b"], np.array([[0, 0, 0], [5.0, 0, 0]]))
        self.assertTrue(r2["length_clamped"])
        np.testing.assert_allclose(P2["b"], [1.6, 0, 0], atol=1e-9)  # clamped at 1.6x

    def test_two_bone_ik_reaches_the_target(self) -> None:
        from scipy.spatial.transform import Rotation as R
        pos = {0: np.array([0, 1.0, 0]), 1: np.array([0, 0.5, 0.1]), 2: np.array([0, 0.0, 0])}
        rot = {k: R.identity() for k in pos}
        target = np.array([0, 0.25, 0.15])
        qB, qA = rig_pred._two_bone(pos, rot, 0, 1, 2, target, np.array([1.0, 0, 0]))
        k = pos[1] + qB.apply(pos[2] - pos[1])
        k = pos[0] + qA.apply(k - pos[0])
        np.testing.assert_allclose(k, target, atol=1e-6)

    @staticmethod
    def quadruped() -> tuple[dict, dict, dict, np.ndarray, np.ndarray]:
        """A small horse-shaped skeleton (template frame, +Z forward) in a 1.4-long, 1-tall box body on legs: the
        template (P0) has its hips and hind pivot in the rump over the thigh; P is after register moved the hips onto
        a mid-body predicted root (the whole skeleton shifted forward) and laid the hind leg back on its predicted leg."""
        import trimesh
        P0 = {"hips": np.array([0, 0.8, -0.5]), "spine_1": np.array([0, 0.8, 0.0]), "spine_2": np.array([0, 0.8, 0.5]),
              "tail_1": np.array([0, 0.85, -0.65]),
              "pelvis_r": np.array([-0.1, 0.8, -0.4]), "thigh_r": np.array([-0.2, 0.6, -0.5]), "shin_r": np.array([-0.2, 0.05, -0.5]),
              "scapula_r": np.array([-0.1, 0.85, 0.45]), "humerus_r": np.array([-0.2, 0.6, 0.55]), "fore_r": np.array([-0.2, 0.05, 0.55])}
        children = {"hips": ["spine_1", "tail_1", "pelvis_r"], "spine_1": ["spine_2"], "spine_2": ["scapula_r"],
                    "pelvis_r": ["thigh_r"], "thigh_r": ["shin_r"], "scapula_r": ["humerus_r"], "humerus_r": ["fore_r"]}
        P = {n: p + np.array([0, 0, 0.45]) for n, p in P0.items()}  # the hips went mid-body
        for n in ("thigh_r", "shin_r", "humerus_r", "fore_r"):  # the legs laid back on the predicted legs
            P[n] = P0[n].copy()
        box = trimesh.creation.box(extents=(0.6, 0.7, 1.4))
        box.apply_translation((0, 0.65, 0))
        return P0, P, children, np.asarray(box.vertices, float), np.asarray(box.faces)

    def test_a_hind_pivot_goes_back_over_its_leg(self) -> None:
        P0, P, children, V, F = self.quadruped()
        spec = {"body": ["hips", "spine_1", "spine_2"], "tail": ["hips", "tail_1"]}
        self.assertGreater(np.linalg.norm(P["pelvis_r"] - P["thigh_r"]), 0.4)  # mid-body: the bug
        rep = rig_pred.place_pivots(P, P0, children, spec, {"thigh_r": 1.0, "humerus_r": 1.0}, V, F, H=1.0)
        self.assertEqual([r["joint"] for r in rep], ["pelvis_r", "scapula_r"])
        np.testing.assert_allclose(P["pelvis_r"], P0["pelvis_r"], atol=1e-9)  # the template's pivot over the thigh
        np.testing.assert_allclose(P["pelvis_r"] - P["thigh_r"], P0["pelvis_r"] - P0["thigh_r"], atol=1e-9)
        self.assertGreater(P["pelvis_r"][1], P["thigh_r"][1])  # above the leg, in the rump
        np.testing.assert_allclose(P["scapula_r"] - P["humerus_r"], P0["scapula_r"] - P0["humerus_r"], atol=1e-9)
        np.testing.assert_allclose(P["thigh_r"], P0["thigh_r"], atol=1e-9)  # the legs and trunk are not touched
        np.testing.assert_allclose(P["hips"], P0["hips"] + [0, 0, 0.45], atol=1e-9)
        self.assertTrue(np.all(np.abs(rig_fit.winding_number(V, F, np.array([P["pelvis_r"], P["scapula_r"]]))) > 0.5))
        self.assertEqual(rep[0]["pulled_in"], 0.0)

    def test_a_pivot_scales_with_its_leg_and_stays_inside(self) -> None:
        P0, P, children, V, F = self.quadruped()
        spec = {"body": ["hips", "spine_1", "spine_2"], "tail": ["hips", "tail_1"]}
        rig_pred.place_pivots(P, P0, children, spec, {"thigh_r": 0.5}, V, F)
        np.testing.assert_allclose(P["pelvis_r"] - P["thigh_r"], 0.5 * (P0["pelvis_r"] - P0["thigh_r"]), atol=1e-9)
        P0, P, children, V, F = self.quadruped()
        P0["pelvis_r"] = P0["thigh_r"] + np.array([0, 0.55, 0])  # a template pivot that would poke out of the back
        rep = rig_pred.place_pivots(P, P0, children, spec, {"thigh_r": 1.0}, V, F)
        self.assertGreater(rep[0]["pulled_in"], 0)
        self.assertLess(P["pelvis_r"][1], float(V[:, 1].max()))

    def test_the_hips_go_over_the_hind_legs(self) -> None:
        P0 = {"hips": np.array([0, 0.8, -0.5]), "fl": np.array([0.2, 0.6, 0.55]), "fr": np.array([-0.2, 0.6, 0.55]),
              "bl": np.array([0.2, 0.6, -0.55]), "br": np.array([-0.2, 0.6, -0.55])}
        spec = {"legs": [["fl"], ["fr"], ["bl"], ["br"]]}
        # predicted legs of a longer body, its root mid-body
        plegs = [{"line": np.array([[x, 0.6, z], [x, 0.0, z]])} for x in (0.2, -0.2) for z in (0.6, -0.7)]
        root = np.array([0.01, 0.75, 0.0])
        at = rig_pred.hips_target(P0, spec, plegs, root)
        np.testing.assert_allclose(at, [0.01, 0.8, -0.65], atol=1e-9)  # 0.05 ahead of the hind roots, as on P0
        _, _, _, V, F = self.quadruped()  # a box 0.7 tall from y 0.3: y 1.2 pokes out of the back
        P0["hips"] = np.array([0, 1.2, -0.5])
        at = rig_pred.hips_target(P0, spec, plegs, root, V, F)
        self.assertLess(at[1], float(V[:, 1].max()))
        self.assertGreater(at[1], root[1])  # pulled toward the root only as far as the mesh needs
        self.assertIsNone(rig_pred.hips_target(P0, {"legs": [["fl"], ["fr"]]}, plegs[:2], root))  # a bird

    def test_the_trunk_joints_go_where_the_predicted_legs_and_neck_hang(self) -> None:
        spec = {"body": ["hips", "spine_1", "spine_2", "neck", "head"], "neck_lift": "neck",
                "legs": [["humerus_l"], ["humerus_r"], ["thigh_l"], ["thigh_r"]]}
        children = {"hips": ["spine_1", "pelvis_l", "pelvis_r"], "spine_1": ["spine_2"], "spine_2": ["neck", "scapula_l", "scapula_r"],
                    "neck": ["head"], "scapula_l": ["humerus_l"], "scapula_r": ["humerus_r"], "pelvis_l": ["thigh_l"], "pelvis_r": ["thigh_r"]}
        self.assertEqual(rig_pred.shoulder_joint(spec, children), "spine_2")
        line = np.array([[0, 1.0, -0.2], [0, 1.0, 0.3], [0, 1.1, 0.6], [0, 1.3, 0.9], [0, 1.5, 1.3], [0, 1.7, 1.6]])
        legs = [{"line": np.array([[x, 0.6, z], [x, 0, z]]), "attach": np.array([x / 2, 1.0, z + 0.1])}
                for x in (0.2, -0.2) for z in (0.2, -0.7)]
        lines_t = {"body": {"line": line, "neck_index": 2}, "legs": legs}
        an = rig_pred.trunk_anchors(spec, children, lines_t, line[0])
        np.testing.assert_allclose(an["hips"], [0, 1.0, -0.6])  # the hind legs' attach, on the centre line
        np.testing.assert_allclose(an["spine_2"], [0, 1.0, 0.3])  # the front legs'
        np.testing.assert_allclose(an["neck"], line[2])  # where the head chain leaves the body
        P = {"hips": np.array([0, 1.0, -0.6]), "spine_1": np.array([0, 1.0, -0.3]), "spine_2": np.array([0, 1.0, 0.0]),
             "neck": np.array([0, 1.5, 0.9]), "head": np.array([0, 1.8, 1.6])}
        for side, x in (("l", 0.2), ("r", -0.2)):
            P[f"scapula_{side}"], P[f"humerus_{side}"] = np.array([x, 1.0, 0.1]), np.array([x, 0.6, 0.2])
            P[f"pelvis_{side}"], P[f"thigh_{side}"] = np.array([x, 1.0, -0.9]), np.array([x, 0.6, -1.0])
        r = rig_pred.lay_runs(P, children, spec["body"], np.r_[[an["hips"]], line],
                              [("spine_2", an["spine_2"]), ("neck", an["neck"])])
        np.testing.assert_allclose(P["spine_2"], an["spine_2"], atol=1e-6)
        np.testing.assert_allclose(P["neck"], an["neck"], atol=1e-6)
        np.testing.assert_allclose(P["head"], line[-1], atol=1e-6)
        self.assertEqual(r["anchors"], ["spine_2", "neck"])
        self.assertEqual(len(r["length_ratios"]), 3)
        lines_t["legs"] = [{k: v for k, v in q.items() if k != "attach"} for q in legs]
        self.assertEqual(rig_pred.trunk_anchors(spec, children, lines_t, line[0]), {})  # no attach: the template's way

    @staticmethod
    def spine() -> tuple[list[str], dict, dict]:
        chain = ["hips", "s1", "s2", "s3", "neck", "n1", "head"]
        children = {a: [b] for a, b in zip(chain, chain[1:])}
        P = {n: np.array([0, 1.0, -0.6 + 0.3 * k]) for k, n in enumerate(chain)}
        return chain, children, P

    def test_runs_split_by_place_on_the_line_not_by_depth(self) -> None:
        # a neck that rises and leans back past its root: by z its points would fall in the shoulder run
        chain, children, P = self.spine()
        line = np.array([[0, 1, -0.6], [0, 1, 0.3], [0, 1.1, 0.6], [0, 1.5, 0.5], [0, 1.9, 0.55], [0, 1.95, 0.9]])
        r = rig_pred.lay_runs(P, children, chain, line, [("s3", line[1]), ("neck", line[2])])
        np.testing.assert_allclose(P["neck"], line[2], atol=1e-6)
        neck = line[2:]
        segs = list(zip(neck[:-1], neck[1:]))
        self.assertLess(float(rig_pred.cd_j2b(P["head"][None], segs)[0]), 0.02)  # on the predicted neck
        self.assertGreater(P["head"][1], 1.8)
        self.assertGreater(P["n1"][1], 1.3)  # on the rising neck, not on a chord from the root to the tip
        self.assertEqual(r["anchors"], ["s3", "neck"])

    def test_an_anchor_that_doesnt_advance_drops_the_rest(self) -> None:
        chain, children, P = self.spine()
        line = np.array([[0, 1, -0.6], [0, 1, -0.3], [0, 1, 0.0], [0, 1, 0.3], [0, 1.3, 0.6], [0, 1.6, 0.9]])
        # the shoulder sits on the hips: it and the neck after it are dropped, the chain is laid in one run
        self.assertEqual(rig_pred.ordered_anchors(line, [("s3", line[0]), ("neck", line[1])]), [])
        r = rig_pred.lay_runs(P, children, chain, line, [("s3", line[0]), ("neck", line[1])])
        self.assertNotIn("anchors", r)
        np.testing.assert_allclose(P["head"], line[-1], atol=1e-6)

    def test_a_leg_hung_straight_off_the_trunk_has_no_pivot(self) -> None:
        P0, P, children, V, F = self.quadruped()
        spec = {"body": ["hips", "spine_1", "spine_2", "pelvis_r"], "tail": ["hips", "tail_1"]}  # a bird's leg off the hips
        before = {n: p.copy() for n, p in P.items()}
        self.assertEqual(rig_pred.place_pivots(P, P0, children, spec, {"thigh_r": 1.0}, V, F), [])
        for n in P:
            np.testing.assert_allclose(P[n], before[n])


def _glb(doc: dict, binary: bytes) -> bytes:
    import rig_template
    return rig_template._glb_bytes(doc, binary)


def _stub_skin(P_fn):
    """A run_skin stand-in: skin-tokens' skeleton shape (rig_pred.ours_from_posed) with nearest-bone distance weights."""
    sys.path.insert(0, str(ROOT / "tests"))
    from fixtures import synth_bird

    def run_skin(mesh: Path, skeleton: Path, out: Path, fit: str = "none") -> float:
        bdoc, bbin = rig_pred.ours_from_posed(skeleton)
        src, sbin = rig.read_glb(mesh)
        P = rig.accessor(src, sbin, src["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).astype(float)
        sk = rig.Skeleton(bdoc)
        w = sk.world({})
        joints = bdoc["skins"][0]["joints"]
        jpos = np.array([w[j][0] for j in joints])
        par = [joints.index(sk.parent[j]) if sk.parent.get(j) in joints else -1 for j in joints]
        J, W = synth_bird.distance_weights(P, jpos, par)
        rig.write_rigged(out, bdoc, bbin, src, sbin, J, W, [])
        P_fn.append(len(P))
        return 0.0
    return run_skin


class SeamTests(unittest.TestCase):
    def test_a_wrap_implied_vendor_loop_has_no_seam_but_a_pop_does(self) -> None:
        from scipy.spatial.transform import Rotation as R
        n = 30
        ang = np.arange(n) * (360.0 / n)  # one full turn, the last key one step before the first (wrap implied)
        Q = R.from_euler("x", ang[:, None], degrees=True).as_quat()[:, None, :]
        self.assertLess(rig_inspect.seam_excess_deg(Q), 1e-6)
        pop = Q.copy()
        pop[-1, 0] = R.from_euler("x", [ang[-1] - 40], degrees=True).as_quat()
        self.assertGreater(rig_inspect.seam_excess_deg(pop), 20)


class PlanModeTests(unittest.TestCase):
    """--template auto|fixed|build and the fixed path's plan (design section 4, 9.3)."""

    def test_auto_builds_only_plans_with_a_builder_and_no_fixed_template(self) -> None:
        import rig_plan
        self.assertEqual(rig.decide(rig_plan.load("ground_bird"), "auto"), "build")
        for pid in ("flying_bird", "horse", "biped", "spider"):
            self.assertEqual(rig.decide(rig_plan.load(pid), "auto"), "fixed", pid)
        self.assertEqual(rig.decide(rig_plan.load("ground_bird"), "fixed"), "fixed")
        with self.assertRaises(SystemExit):
            rig.decide(rig_plan.load("horse"), "build")

    def test_cli_stops_on_a_plan_for_another_class_and_on_no_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            saved = rig.OUT
            rig.OUT = Path(tmp)
            try:
                asset = Path(tmp) / "thing"
                asset.mkdir()
                with self.assertRaises(SystemExit) as cm:  # no mesh: said before the planner is asked for a class
                    rig.main(["--stem", "thing", "--no-fbx"])
                self.assertIn("no textured mesh", str(cm.exception))
                (asset / "textured.glb").write_bytes(b"x")
                with self.assertRaises(SystemExit) as cm:
                    rig.main(["--stem", "thing", "--class", "quadruped", "--plan", "ground_bird", "--no-fbx"])
                self.assertIn("is for class avian, not quadruped", str(cm.exception))
            finally:
                rig.OUT = saved

    def test_the_fixed_path_reads_the_class_wrapper_plan(self) -> None:
        import rig_plan
        gb = rig_plan.load("ground_bird")
        fp = rig.fixed_plan(gb, "avian")
        self.assertEqual(fp["id"], "flying_bird")
        self.assertEqual(rig_plan.spec(fp, "avian", "needs", {}), rig_pred.PATHS["avian"]["needs"])  # Flap and Glide
        self.assertEqual(rig.fixed_plan(rig_plan.load("horse"), "quadruped")["id"], "horse")

    def test_the_trial_runs_on_a_fixed_upright_avian_and_never_on_a_built_template(self) -> None:
        import rig_plan
        tall = np.random.default_rng(0).uniform([-0.2, 0, -0.3], [0.2, 1.0, 0.3], (500, 3))
        long_ = np.random.default_rng(0).uniform([-0.2, 0, -1.0], [0.2, 0.5, 1.0], (500, 3))
        names = rig.TEMPLATES["avian"]["default"]
        fixed = rig.fixed_plan(rig_plan.load("ground_bird"), "avian")
        self.assertEqual(rig.ground_bird_trial(fixed, "avian", tall, names, False, False), ["Flap", "Glide"])
        self.assertEqual(rig.ground_bird_trial(fixed, "avian", long_, names, False, False), [])
        self.assertEqual(rig.ground_bird_trial(fixed, "avian", tall, names, True, False), [])
        self.assertEqual(rig.ground_bird_trial(rig_plan.load("ground_bird"), "avian", tall, names, False, True), [])
        self.assertEqual(rig.ground_bird_trial(rig_plan.load("horse"), "quadruped", tall, ["Idle", "Kick"], False, False), [])

    def test_class_lists_skip_built_templates(self) -> None:
        rig.TEMPLATES["built:x"] = {"class": "avian", "built": True}
        try:
            self.assertEqual(rig.CLASSES, ("humanoid", "quadruped", "multileg", "avian"))
            self.assertEqual(rig._template_class(rig.TEMPLATES["built:x"]), "avian")
            self.assertEqual(rig._template_class(rig.TEMPLATES["quadruped"]), "quadruped")
        finally:
            del rig.TEMPLATES["built:x"]

    def test_a_facing_prior_needs_a_checked_psx(self) -> None:
        self.assertIsNone(rig.facing_prior_yaw({"facing": "right", "canonical": False}, "psx.glb"))
        self.assertEqual(rig.facing_prior_yaw({"canonical": True}, "psx.glb")["yaw"], 0)
        self.assertEqual(rig.facing_prior_yaw({"facing_check": {"agrees": True}}, "psx.glb")["yaw"], 0)
        self.assertIsNone(rig.facing_prior_yaw({"canonical": True}, "textured.glb"))

    def test_a_prior_turns_the_template_and_the_registration_only_verifies(self) -> None:
        import rig_fit
        reg = {0: {"chamfer": 0.09}, 90: {"chamfer": 0.2}, 180: {"chamfer": 0.05}, 270: {"chamfer": 0.2}}
        got = rig_fit.choose_yaw(reg, {}, lambda: self.fail("no VLM with a prior"), prior={"yaw": 0})
        self.assertEqual((got["yaw"], got["decided_by"]), (0, "declared"))
        self.assertTrue(got["cue_conflict"])  # best 180 by 1.8x: the facing row warns
        ok = rig_fit.choose_yaw({0: {"chamfer": 0.05}, 90: {"chamfer": 0.2}, 180: {"chamfer": 0.06}, 270: {"chamfer": 0.2}},
                                {}, None, prior={"yaw": 0})
        self.assertNotIn("cue_conflict", ok)

    def test_predict_keeps_the_default_cache_and_keys_modes_apart(self) -> None:
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "psx.glb"
            src.write_bytes(b"mesh")
            key = hashlib.md5(b"mesh").hexdigest()
            (Path(tmp) / "pred_rig.glb").write_bytes(b"glb")
            (Path(tmp) / "pred_rig.json").write_text(json.dumps({"key": key, "seconds": 3}))
            self.assertEqual(rig_pred.predict(src, Path(tmp))[2], True)
            (Path(tmp) / "pred_rig.t0.7.glb").write_bytes(b"glb")
            (Path(tmp) / "pred_rig.t0.7.json").write_text(json.dumps({"key": key + ":t0.7", "seconds": 5}))
            out, secs, cached = rig_pred.predict(src, Path(tmp), temperature=0.7)
            self.assertEqual((out.name, secs, cached), ("pred_rig.t0.7.glb", 5.0, True))
            self.assertEqual(rig_pred.predict(src, Path(tmp), mode="sampled")[0].name, "pred_rig.t0.7.glb")


@unittest.skipUnless((ROOT / "scripts" / "rig_plans" / "ground_bird.json").is_file(), "no body plans")
class BuiltPathTests(unittest.TestCase):
    """rig.py's build branch end to end on the synthetic bird (tests/fixtures/synth_bird.py), skin-tokens stubbed with
    distance weights and the prediction with the fixture's: template mode built, all 7 clips, nothing dropped."""
    STEM = "synth-bird"

    @classmethod
    def setUpClass(cls) -> None:
        import contextlib
        import io
        sys.path.insert(0, str(ROOT / "tests"))
        from fixtures import synth_bird
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.asset = cls.root / "out" / cls.STEM
        cls.d = synth_bird.write(cls.asset)
        import rig_frames
        cls.saved = (rig.OUT, rig.run_skin, rig_pred.predict, rig_pred.floor_pass, rig.far_motion, rig_frames.reframe)
        cls.skinned, cls.calls = [], []
        rig.OUT = cls.root / "out"
        rig.run_skin = _stub_skin(cls.skinned)
        rig_pred.predict = lambda source, work, **kw: (Path(cls.d["pred"]), 0.0, True)

        def record(name, fn):
            def wrapped(*a, **k):
                cls.calls.append(name)
                return fn(*a, **k)
            return wrapped
        rig_pred.floor_pass = record("floor_pass", cls.saved[3])
        rig.far_motion = record("far_motion", cls.saved[4])
        rig_frames.reframe = record("reframe", cls.saved[5])
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                cls.code = rig.main(["--stem", cls.STEM, "--class", "avian", "--no-fbx", "--render", "0", "--reframe", "1"])
        finally:
            rig.OUT, rig.run_skin, rig_pred.predict, rig_pred.floor_pass, rig.far_motion, rig_frames.reframe = cls.saved
            cls.log = buf.getvalue()
        cls.info = json.loads((cls.asset / "rig" / "rig.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        for k in [k for k in rig.TEMPLATES if str(k).startswith("built:")]:
            del rig.TEMPLATES[k]
        cls.tmp.cleanup()

    def test_the_rig_is_built_for_the_model(self) -> None:
        import rig_plan
        info = self.info
        self.assertEqual(self.code, 0, self.log[-2000:])
        self.assertEqual(self.skinned, [len(self.d["V"])])
        self.assertEqual(info["template"]["mode"], "built", self.log[-2000:])
        self.assertTrue(info["template"]["built"])
        self.assertEqual((info["plan"], info["template"]["plan"]), ("ground_bird", "ground_bird"))
        self.assertFalse(info["fit"]["registration"]["poor_match"])
        self.assertEqual((info["fit"]["yaw"], info["fit"]["facing"]["yaw"]), (0, 0))
        self.assertEqual(info["fit"]["unsupported"], [])
        self.assertNotIn("pred_register", info["fit"])
        self.assertNotIn("first_gate", info["fit"])
        self.assertNotIn("fallback", info)
        self.assertEqual(info["clips"], rig_plan.load("ground_bird")["default"])
        self.assertEqual(info["clips"], ["Idle", "Walk", "Run", "Peck", "Flap", "Sleep", "Death"])
        self.assertEqual((info["unsupported_clips"], info["dropped_clips"], info["clips_failed"]), ([], [], []))
        self.assertTrue(json.loads((self.asset / "rig" / "template" / "validate.json").read_text())["ok"])
        rep = json.loads((self.asset / "rig" / "template" / "clips_report.json").read_text())
        self.assertTrue(all(c["passed"] for c in rep["clips"].values()))

    def test_rigged_glb_carries_every_clip_on_our_joints(self) -> None:
        doc, _ = rig.read_glb(self.asset / "rigged.glb")
        self.assertEqual([a["name"] for a in doc["animations"]], self.info["clips"])
        tj = json.loads((self.asset / "rig" / "template" / "template.json").read_text())
        self.assertEqual(tj["default"], self.info["clips"])
        names = {doc["nodes"][j]["name"] for j in doc["skins"][0]["joints"]}
        import rig_plan
        self.assertEqual(names, set(rig_plan.joint_names(rig_plan.load("ground_bird"))))

    def test_the_check_rows_cover_the_built_template(self) -> None:
        insp = json.loads((self.asset / "rig" / "inspect.json").read_text())
        keys = {c["key"]: c for c in insp["checks"]}
        self.assertNotIn("match", keys)
        self.assertNotIn("unsupported", keys)
        for k in ("template", "built", "a2", "cshape", "clips"):
            self.assertIn(k, keys, sorted(keys))
        self.assertEqual(keys["template"]["level"], "pass")
        self.assertEqual(keys["clips"]["level"], "pass")
        self.assertTrue(insp["template"]["built"])
        self.assertEqual(len(insp["roles"]), len(insp["bones"]))

    def test_reuse_skin_skips_skin_tokens_on_the_same_mesh_and_skeleton(self) -> None:
        import contextlib
        import io
        saved = (rig.OUT, rig.run_skin, rig_pred.predict)
        calls: list[int] = []
        rig.OUT = self.root / "out"
        rig.run_skin = _stub_skin(calls)
        rig_pred.predict = lambda source, work, **kw: (Path(self.d["pred"]), 0.0, True)
        rj = self.asset / "rig" / "rig.json"
        before = rj.read_bytes()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = rig.main(["--stem", self.STEM, "--class", "avian", "--no-fbx", "--render", "0", "--reframe", "1",
                                 "--reuse-skin"])
            info = json.loads(rj.read_text())
        finally:
            rig.OUT, rig.run_skin, rig_pred.predict = saved
            rj.write_bytes(before)
        self.assertEqual(code, 0, buf.getvalue()[-2000:])
        self.assertEqual(calls, [], buf.getvalue()[-2000:])
        self.assertTrue(info["reused_skin"])
        self.assertIn("Reused the skin weights", buf.getvalue())

    def moved_copy(self, tmp: Path, name: str = "", shift=(0.0, 0.05, 0.02)) -> tuple[Path, Path, str]:
        """A copy of rig/skinned_raw.glb and a skeleton like it with its first joint off the body (a leg, as a pivot
        fix moves) moved (or renamed to `name`)."""
        import shutil
        raw = Path(tmp) / "raw.glb"
        shutil.copy2(self.asset / "rig" / "skinned_raw.glb", raw)
        doc, b = rig.read_glb(raw)
        trunk = rig._trunk(rig._joint_rest(doc))
        j = next(i for i in doc["skins"][0]["joints"] if doc["nodes"][i].get("name") not in trunk)
        doc["nodes"][j]["translation"] = (np.array(doc["nodes"][j].get("translation", [0, 0, 0])) + shift).tolist()
        if name:
            doc["nodes"][j]["name"] = name
        moved = Path(tmp) / "moved.glb"
        moved.write_bytes(_glb(doc, b))
        return raw, moved, doc["nodes"][j]["name"]

    def test_moved_joints_reuse_the_weights_on_the_moved_skeleton(self) -> None:
        mesh = self.asset / self.info["source"]
        with tempfile.TemporaryDirectory() as tmp:
            raw, moved, jn = self.moved_copy(Path(tmp))
            self.assertEqual(rig.skin_reusable(raw, mesh, moved), f"joint {jn} moved")  # the strict rule as before
            self.assertIsNone(rig.skin_reusable(raw, mesh, moved, moved_ok=True))
            d0, b0 = rig.read_glb(raw)
            a0 = d0["meshes"][0]["primitives"][0]["attributes"]
            W0 = rig.accessor(d0, b0, a0["WEIGHTS_0"])
            self.assertGreater(rig.rejoint_skin(raw, moved), 0)
            self.assertIsNone(rig.skin_reusable(raw, mesh, moved))  # raw's joints are now the skeleton's
            d1, b1 = rig.read_glb(raw)
            np.testing.assert_array_equal(rig.accessor(d1, b1, a0["WEIGHTS_0"]), W0)  # the weights are untouched
            sk = rig.Skeleton(d1)
            w = sk.world({})
            IBM = rig.accessor(d1, b1, d1["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
            for k, i in enumerate(d1["skins"][0]["joints"]):  # the inverse binds follow the joints (rest = identity)
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = w[i][1].as_matrix(), w[i][0]
                np.testing.assert_allclose(IBM[k] @ M, np.eye(4), atol=1e-5)
            raw2, renamed, _ = self.moved_copy(Path(tmp), name="not_a_joint")
            self.assertEqual(rig.skin_reusable(raw2, mesh, renamed, moved_ok=True), "the skeleton's joints changed")

    def turned_copy(self, tmp: Path) -> Path:
        """rig/skinned_raw.glb turned 180 degrees about up: the same joint names, order and parents."""
        from scipy.spatial.transform import Rotation as R
        doc, b = rig.read_glb(self.asset / "rig" / "skinned_raw.glb")
        n = doc["nodes"][doc["skins"][0]["joints"][0]]
        q = R.from_euler("y", 180, degrees=True)
        n["rotation"] = (q * R.from_quat(n.get("rotation", [0, 0, 0, 1]))).as_quat().tolist()
        n["translation"] = q.apply(n.get("translation", [0, 0, 0])).tolist()
        turned = Path(tmp) / "turned.glb"
        turned.write_bytes(_glb(doc, b))
        return turned

    def test_a_turned_or_far_moved_skeleton_does_not_reuse_the_weights(self) -> None:
        mesh = self.asset / self.info["source"]
        with tempfile.TemporaryDirectory() as tmp:
            raw = self.asset / "rig" / "skinned_raw.glb"
            turned = self.turned_copy(Path(tmp))
            self.assertIsNotNone(rig.skin_reusable(raw, mesh, turned))
            why = rig.skin_reusable(raw, mesh, turned, moved_ok=True)
            self.assertIsNotNone(why)
            self.assertIn("on the body", why)
            md, mb = rig.read_glb(mesh)
            H = float(np.ptp(rig.accessor(md, mb, md["meshes"][0]["primitives"][0]["attributes"]["POSITION"])[:, 1]))
            _, far, jn = self.moved_copy(Path(tmp), shift=(0.0, (rig.MOVED_MAX + 0.1) * H, 0.0))
            why = rig.skin_reusable(raw, mesh, far, moved_ok=True)  # off the body, but too far to be a pivot fix
            self.assertTrue(why and why.startswith(f"joint {jn} moved") and "over" in why, why)

    def test_reuse_skin_with_moved_joints_skips_skin_tokens_and_says_so(self) -> None:
        import contextlib
        import io
        rawp = self.asset / "rig" / "skinned_raw.glb"
        saved = (rig.OUT, rig.run_skin, rig_pred.predict)
        calls: list[int] = []
        rig.OUT = self.root / "out"
        rig.run_skin = _stub_skin(calls)
        rig_pred.predict = lambda source, work, **kw: (Path(self.d["pred"]), 0.0, True)
        rj = self.asset / "rig" / "rig.json"
        before, raw_before = rj.read_bytes(), rawp.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            _, moved, _ = self.moved_copy(Path(tmp))
            rawp.write_bytes(moved.read_bytes())  # learned on joints that have since moved
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = rig.main(["--stem", self.STEM, "--class", "avian", "--no-fbx", "--render", "0", "--reframe", "1",
                                 "--reuse-skin"])
            info = json.loads(rj.read_text())
            now = rig._joint_rest(rig.read_glb(rawp)[0])
            posed = rig._joint_rest(rig.read_glb(self.asset / "rig" / "template_posed.glb")[0])
        finally:
            rig.OUT, rig.run_skin, rig_pred.predict = saved
            rj.write_bytes(before)
            rawp.write_bytes(raw_before)
        self.assertEqual(code, 0, buf.getvalue()[-2000:])
        self.assertEqual(calls, [], buf.getvalue()[-2000:])
        self.assertIn("Reused the skin weights learned before; joints moved", buf.getvalue())
        self.assertTrue(info["reused_skin"])
        self.assertEqual(info["skin_reuse"]["note"], "Reused the skin weights learned before; joints moved")
        for n, (p, _) in posed.items():  # the clips were baked on the moved joints, not the old ones
            np.testing.assert_allclose(now[n][0], p, atol=1e-5)

    def test_reuse_skin_on_a_turned_skeleton_learns_the_weights_again(self) -> None:
        import contextlib
        import io
        rawp = self.asset / "rig" / "skinned_raw.glb"
        saved = (rig.OUT, rig.run_skin, rig_pred.predict)
        calls: list[int] = []
        rig.OUT = self.root / "out"
        rig.run_skin = _stub_skin(calls)
        rig_pred.predict = lambda source, work, **kw: (Path(self.d["pred"]), 0.0, True)
        rj = self.asset / "rig" / "rig.json"
        before, raw_before = rj.read_bytes(), rawp.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            rawp.write_bytes(self.turned_copy(Path(tmp)).read_bytes())  # learned on a skeleton facing the other way
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = rig.main(["--stem", self.STEM, "--class", "avian", "--no-fbx", "--render", "0", "--reframe", "1",
                                 "--reuse-skin"])
            info = json.loads(rj.read_text())
        finally:
            rig.OUT, rig.run_skin, rig_pred.predict = saved
            rj.write_bytes(before)
            rawp.write_bytes(raw_before)
        self.assertEqual(code, 0, buf.getvalue()[-2000:])
        self.assertEqual(len(calls), 1, buf.getvalue()[-2000:])
        self.assertRegex(buf.getvalue(), r"Couldn't reuse rig/skinned_raw\.glb \(joint \S+ moved [\d.]+× height, "
                                         r"and it is on the body\); learning the weights again")
        self.assertNotIn("joints moved", buf.getvalue())
        self.assertFalse(info["reused_skin"])

    def test_reframe_stays_last(self) -> None:
        # the designed frames go on after the floor pass and the far-bone check (FK / _skin assume identity rests)
        self.assertEqual(self.calls[-1], "reframe", self.calls)
        self.assertIn("floor_pass", self.calls)
        self.assertTrue(self.info["reframe"]["applied"], self.info["reframe"])
        self.assertLess(self.info["reframe"]["max_delta_H"], 1e-5)
        doc, _ = rig.read_glb(self.asset / "rigged.glb")
        hips = next(n for n in doc["nodes"] if n.get("name") == "hips")
        self.assertGreater(1 - abs(hips.get("rotation", [0, 0, 0, 1])[3]), 1e-3)  # a real rest frame, not identity

    def test_a_rebuild_in_a_fresh_process_uses_template_json(self) -> None:
        import os
        import subprocess
        env = {**os.environ, "FACTORY_ROOT": str(self.root)}
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "rig_inspect.py"), "--stem", self.STEM],
                              env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        insp = json.loads((self.asset / "rig" / "inspect.json").read_text())
        self.assertEqual(Path(insp["template"]["rig"]), (self.asset / "rig" / "template" / "rig.glb").resolve())
        self.assertEqual(insp["template"]["ref_clip"], "Rest")
        self.assertEqual(next(c for c in insp["checks"] if c["key"] == "template")["level"], "pass")

    def test_a_built_rig_without_its_template_json_fails_the_check_instead_of_using_the_vendor_one(self) -> None:
        tj = self.asset / "rig" / "template" / "template.json"
        saved = tj.read_bytes()
        for k in [k for k in rig.TEMPLATES if str(k).startswith("built:")]:
            del rig.TEMPLATES[k]
        tj.unlink()
        try:
            b = rig_inspect.rebuild(self.asset)
        finally:
            tj.write_bytes(saved)
        row = next(c for c in b["checks"] if c["key"] == "template")
        self.assertEqual(row["level"], "fail")


@unittest.skipUnless((ROOT / "scripts" / "rig_plans" / "ground_bird.json").is_file() and
                     (rig.M2M / "rigs" / "rig-bird.glb").is_file(), "no body plans or no vendor bird template")
class BuildFallbackTests(unittest.TestCase):
    """auto mode, the build raises: the vendor-template fallback binds the build's own (sampled) prediction, so
    rig_pred.predict runs exactly once, and the fallback is loud (DEGRADED lines + rig.json degraded)."""
    STEM = "synth-bird-fallback"

    @classmethod
    def setUpClass(cls) -> None:
        import contextlib
        import io
        import rig_build
        import rig_fit
        sys.path.insert(0, str(ROOT / "tests"))
        from fixtures import synth_bird
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.asset = cls.root / "out" / cls.STEM
        cls.d = synth_bird.write(cls.asset)
        cls.saved = (rig.OUT, rig.run_skin, rig_pred.predict, rig_build.build, rig_fit.vlm_cue)
        cls.predicts, cls.skinned = [], []
        rig.OUT = cls.root / "out"
        rig.run_skin = _stub_skin(cls.skinned)

        def predict(source, work, **kw):
            cls.predicts.append(kw)
            return Path(cls.d["pred"]), 116.0, False
        rig_pred.predict = predict

        def build(*a, **k):
            raise rig_build.BuildError("check", "neck 0.713H outside [0.25, 0.6]")
        rig_build.build = build
        rig_fit.vlm_cue = lambda *a, **k: None
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                cls.code = rig.main(["--stem", cls.STEM, "--class", "avian", "--no-fbx", "--render", "0"])
        finally:
            rig.OUT, rig.run_skin, rig_pred.predict, rig_build.build, rig_fit.vlm_cue = cls.saved
            cls.log = buf.getvalue()
        cls.info = json.loads((cls.asset / "rig" / "rig.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_one_prediction_reused_by_the_fallback(self) -> None:
        self.assertEqual(self.code, 0, self.log[-2000:])
        self.assertEqual(len(self.predicts), 1, self.predicts)
        self.assertEqual(self.predicts[0].get("temperature"), 0.7)  # the build's sampled prediction
        self.assertEqual(self.log.count("Predicted a skeleton"), 1, self.log[-3000:])
        self.assertEqual(self.info["template"]["mode"], "fixed")
        self.assertEqual(self.info["build_error"]["code"], "check")
        pred = self.info["pred"]
        self.assertEqual(pred["reused_from"], "build")
        self.assertEqual((pred["mode"], pred["file"]), ("sampled", "rig/" + Path(self.d["pred"]).name))

    def test_every_fallback_is_degraded_and_loud(self) -> None:
        deg = {d["code"]: d for d in self.info["degraded"]}
        self.assertIn("build_failed", deg, self.info["degraded"])
        self.assertTrue(all(d["stage"] == "rig" for d in self.info["degraded"]))
        self.assertIn("neck 0.713H", deg["build_failed"]["message"])
        self.assertIn("DEGRADED rig build_failed: check: neck 0.713H", self.log)
        lines = [ln for ln in self.log.splitlines() if ln.startswith("DEGRADED rig ")]
        self.assertEqual(len(lines), len(self.info["degraded"]), lines)
        import re
        parsed = [dict(zip(("stage", "code", "message"), m.groups())) for m in
                  (re.match(r"^DEGRADED (\S+) (\S+): ?(.*)$", ln) for ln in lines) if m]
        self.assertEqual(parsed, self.info["degraded"])
        # the synthetic bird on the flying-bird template is the qmv chicken's case: poor match, no wings to flap
        self.assertEqual(set(deg), {"build_failed", "poor_match", "clips_dropped"}, self.info["degraded"])
        self.assertEqual(deg["clips_dropped"]["message"], "Flap,Glide")
        self.assertEqual(self.info["unsupported_clips"], ["Flap", "Glide"])
        self.assertTrue(deg["poor_match"]["message"].startswith("scale 2."), deg["poor_match"])
        self.assertIn("DEGRADED rig clips_dropped: Flap,Glide", self.log)

    def test_a_failed_build_prediction_is_not_retried(self) -> None:
        import contextlib
        import io
        calls = []

        def predict(source, work, **kw):
            calls.append(kw)
            raise RuntimeError("skin-tokens rig failed: no GPU")
        saved = (rig.OUT, rig.run_skin, rig_pred.predict)
        rig.OUT, rig.run_skin, rig_pred.predict = self.root / "out", _stub_skin([]), predict
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                try:
                    rig.main(["--stem", self.STEM, "--class", "avian", "--no-fbx", "--render", "0"])
                except SystemExit:
                    pass  # with no prediction the template may not fit; only the call count matters here
        finally:
            rig.OUT, rig.run_skin, rig_pred.predict = saved
        log = buf.getvalue()
        self.assertEqual(len(calls), 1, log[-2000:])
        self.assertIn("DEGRADED rig build_failed: RuntimeError: skin-tokens rig failed", log)
        self.assertIn("DEGRADED rig pred_failed: no predicted skeleton (the build's prediction failed", log)


if __name__ == "__main__":
    unittest.main()
