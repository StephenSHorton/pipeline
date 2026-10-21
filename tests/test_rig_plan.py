#!/usr/bin/env python3
"""Body plans as data: scripts/rig_plan.py and scripts/rig_plans/*.json."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig  # noqa: E402
import rig_fit  # noqa: E402
import rig_plan  # noqa: E402
import rig_pred  # noqa: E402

BIRD = rig.M2M / "rigs" / "rig-bird.glb"
CLASS_OF = {"ground_bird": "avian", "flying_bird": "avian", "horse": "quadruped", "biped": "humanoid", "spider": "multileg"}


def box(x: float, y: float, z: float, n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.uniform([-x / 2, 0, -z / 2], [x / 2, y, z / 2], (n, 3))


class PlanFileTests(unittest.TestCase):
    def test_every_plan_validates(self) -> None:
        plans = rig_plan.all_plans()
        self.assertEqual(sorted(p["id"] for p in plans), sorted(CLASS_OF))
        for p in plans:
            self.assertEqual(rig_plan.validate(p), [], p["id"])
            self.assertEqual(p["class"], CLASS_OF[p["id"]])

    def test_cli_validate_all_exits_0(self) -> None:
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "rig_plan.py"), "validate", "--all"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_ground_bird_clips_are_the_plan_set(self) -> None:
        p = rig_plan.load("ground_bird")
        names = ["Idle", "Walk", "Run", "Peck", "Flap", "Sleep", "Death"]
        self.assertEqual([c["name"] for c in p["clips"]], names)
        self.assertEqual(p["default"], names)
        self.assertNotIn("Glide", p["default"])
        self.assertEqual(p["builder"], "rig_build")
        self.assertIsNone(p["fixed_template"])
        self.assertEqual(p["ref"], "Rest")

    def test_wrapper_plans_match_todays_templates(self) -> None:
        for pid, cls in CLASS_OF.items():
            if pid == "ground_bird":
                continue
            p = rig_plan.load(pid)
            self.assertIsNone(p["builder"], pid)
            self.assertEqual(p["fixed_template"], rig.TEMPLATES[cls], pid)
            self.assertEqual(p["default"], rig.TEMPLATES[cls]["default"], pid)
        self.assertTrue(rig_plan.load("flying_bird")["trial_when_upright"])
        self.assertEqual(rig_plan.load("flying_bird")["paths"], rig_pred.PATHS["avian"])
        horse = rig_plan.load("horse")["paths"]
        self.assertEqual(horse["neck_from"], "spine_4")
        self.assertEqual({k: v for k, v in horse.items() if k != "neck_from"}, rig_pred.PATHS["quadruped"])

    def test_load_by_alias_and_unknown(self) -> None:
        self.assertEqual(rig_plan.load("avian-ground")["id"], "ground_bird")
        self.assertEqual(rig_plan.load("quadruped")["id"], "horse")
        with self.assertRaises(KeyError):
            rig_plan.load("dragon")
        a = rig_plan.load("ground_bird")
        a["clips"].clear()
        self.assertTrue(rig_plan.load("ground_bird")["clips"], "load must hand out copies")


class JointTreeTests(unittest.TestCase):
    def test_side_expansion(self) -> None:
        self.assertEqual(rig_plan.sided("UpperLeg_{s}", "_L"), "UpperLeg_L")
        self.assertEqual(rig_plan.sided("wing_feather_1_2_{s}", "_R"), "wing_feather_1_2_R")
        tree = dict(rig_plan.joints(rig_plan.load("ground_bird")))
        self.assertEqual(tree["LowerLeg_R"], "UpperLeg_R")
        self.assertEqual(tree["wing_1_L"], "spine_1")
        self.assertIsNone(tree["root"])
        self.assertEqual(rig_plan.chain_names(rig_plan.load("ground_bird"), "legs")[1][0], "UpperLeg_R")

    @unittest.skipUnless(BIRD.is_file(), "vendor/mesh2motion-app is not checked out")
    def test_ground_bird_is_the_rig_bird_tree(self) -> None:
        doc, _ = rig.read_glb(BIRD)
        nodes = doc["nodes"]
        skin = doc["skins"][0]["joints"]
        parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
        vendor = {nodes[j]["name"]: (nodes[parent[j]]["name"] if parent.get(j) in skin else None) for j in skin}
        ours = rig_plan.joints(rig_plan.load("ground_bird"))
        self.assertEqual(len(ours), 55)
        self.assertEqual(len(vendor), 55)
        self.assertEqual({n for n, _ in ours}, set(vendor))
        self.assertEqual(dict(ours), vendor)


class ValidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p = rig_plan.load("ground_bird")

    def problems(self, mutate) -> list[str]:
        p = copy.deepcopy(self.p)
        mutate(p)
        return rig_plan.validate(p)

    def test_catches_bad_plans(self) -> None:
        self.assertTrue(any("'hips'" in e for e in self.problems(
            lambda p: p.update(joints=[j for j in p["joints"] if j[0] != "hips"]))))
        self.assertTrue(any("twice" in e for e in self.problems(lambda p: p["joints"].append(["head", "spine_3"]))))
        self.assertTrue(any("mirror" in e for e in self.problems(lambda p: p["joints"].append(["ear_L", "head"]))))
        self.assertTrue(any("undeclared joint" in e for e in self.problems(
            lambda p: p["chains"]["tail"]["names"].append("tail_9"))))
        self.assertTrue(any("generator" in e for e in self.problems(lambda p: p["clips"][0].update(gen="moonwalk"))))
        self.assertTrue(any("default clip Glide" in e for e in self.problems(lambda p: p["default"].append("Glide"))))
        self.assertTrue(any("undeclared parent" in e for e in self.problems(lambda p: p["joints"].append(["x", "nope"]))))


class ResolveTests(unittest.TestCase):
    def test_upright_avian_is_a_ground_bird(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp)
            p = rig_plan.resolve(asset, "avian", P=box(0.5, 1.0, 0.8))
            self.assertEqual(p["id"], "ground_bird")
            saved = json.loads((asset / "rig" / "plan.json").read_text())
            self.assertEqual(saved["id"], "ground_bird")
            for k in ("facing", "facing_source", "canonical"):
                self.assertIn(k, saved)

    def test_other_avian_is_a_flying_bird(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rig_plan.resolve(Path(tmp), "avian", P=box(2.0, 0.5, 1.0))["id"], "flying_bird")
            # the same rule as rig.py: 0.9 x max(x, z) is the line
            self.assertEqual(rig_plan.resolve(Path(tmp), "avian", P=box(1.0, 0.85, 1.0))["id"], "flying_bird")

    def test_other_classes_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp)
            self.assertEqual(rig_plan.resolve(asset, "quadruped")["id"], "horse")
            self.assertEqual(rig_plan.resolve(asset, "humanoid")["id"], "biped")
            self.assertEqual(rig_plan.resolve(asset, "multileg")["id"], "spider")
            (asset / "intent.json").write_text(json.dumps({"rig": {"plan": "flying_bird"}}))
            p = rig_plan.resolve(asset, "avian", P=box(0.5, 1.0, 0.8))
            self.assertEqual((p["id"], p["resolved"]["by"]), ("flying_bird", "intent"))
            p = rig_plan.resolve(asset, "avian", override="ground_bird", P=box(2, 0.5, 1))
            self.assertEqual((p["id"], p["resolved"]["by"]), ("ground_bird", "override"))
            with self.assertRaises(ValueError):
                rig_plan.resolve(Path(tmp) / "x", "rigid")

    def test_no_write_and_class_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp)
            rig_plan.resolve(asset, "avian", P=box(0.5, 1.0, 0.8), write=False)
            self.assertFalse((asset / "rig" / "plan.json").exists())
            # an intent plan for another class is ignored; an explicit override is flagged (rig.py stops on it)
            (asset / "intent.json").write_text(json.dumps({"rig": {"plan": "ground_bird"}}))
            p = rig_plan.resolve(asset, "quadruped", write=False)
            self.assertEqual((p["id"], p["resolved"]["by"]), ("horse", "rule"))
            self.assertIn("ignored_intent", p["resolved"])
            p = rig_plan.resolve(asset, "quadruped", override="ground_bird", write=False)
            self.assertIn("class_mismatch", p["resolved"])
            rig_plan.write_plan(asset / "rig", {"id": "x"})
            self.assertEqual(json.loads((asset / "rig" / "plan.json").read_text())["id"], "x")

    def test_facing_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp)
            (asset / "flat").mkdir()
            (asset / "flat" / "key.json").write_text(json.dumps({"canonical": True, "facing": "right"}))
            (asset / "psx").mkdir()
            (asset / "psx" / "psx.json").write_text(json.dumps({"facing": "right"}))
            p = rig_plan.resolve(asset, "quadruped")
            self.assertEqual((p["facing"], p["facing_source"], p["canonical"]), ("right", "drawn", True))


class SpecTests(unittest.TestCase):
    def test_falls_back_to_the_class_tables(self) -> None:
        self.assertEqual(rig_plan.spec(None, "avian", "paths", None), rig_pred.PATHS["avian"])
        self.assertEqual(rig_plan.spec(None, "avian", "chains", None), rig_fit.CHAINS["avian"])
        self.assertEqual(rig_plan.spec(None, "avian", "needs", {}), {"wings": ["Flap", "Glide"]})
        self.assertEqual(rig_plan.spec(None, "quadruped", "neck_lift", None), "spine_5")
        self.assertEqual(rig_plan.spec(None, "quadruped", "neck_from", "x"), "x")
        self.assertEqual(rig_plan.spec(None, "humanoid", "paths", "none"), "none")
        self.assertFalse(rig_plan.spec(None, "avian", "trial_when_upright", False))

    def test_reads_the_plan_first(self) -> None:
        g = rig_plan.load("ground_bird")
        self.assertEqual(rig_plan.spec(g, "avian", "needs", {}), {"wings": ["Flap"]})
        self.assertEqual(rig_plan.spec(g, "avian", "neck_from", None), "spine_2")
        self.assertEqual(rig_plan.spec(g, "avian", "chains", None), g["chains"])
        self.assertEqual(rig_plan.spec(rig_plan.load("horse"), "quadruped", "neck_from", None), "spine_4")
        self.assertTrue(rig_plan.spec(rig_plan.load("flying_bird"), "avian", "trial_when_upright", False))
        self.assertEqual(rig_plan.spec(rig_plan.load("biped"), "humanoid", "groups", {}), {})


if __name__ == "__main__":
    unittest.main()
