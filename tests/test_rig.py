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
        self.assertEqual(got["worst_stretch"], {"clip": "Kick", "x": 6.5})
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


if __name__ == "__main__":
    unittest.main()
