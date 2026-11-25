#!/usr/bin/env python3
"""The Animate stage with the real model: the operator (in/operator.jpg) driven by Pipeline's walk skeleton
(scripts/skeletons/walk_side.json) through LTX-2.3 and the Union-Control IC-LoRA, then the result's own skeleton
(Apple Vision, bin/bodypose) checked against the control: within 3% of the picture's width on average.

About 40 GB at its peak and a minute or two, so it only runs when asked, on a Mac set up by
scripts/bootstrap.sh animate:

    PIPELINE_LTX_TEST=1 .venv/bin/python -m pytest -q tests/test_animate_ltx.py -s

PIPELINE_LTX_REMOVE=1 as well runs the walk with --remove backpack (two passes and Qwen-Image-Edit, about 8 min).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import animate as A  # noqa: E402

LIMIT_PCT = 3.0
PROMPT = ("The soldier in a green flight suit turns to face right and walks to the right with a natural walking "
          "stride, legs alternating and arms swinging, side view, staying in the centre of the frame. Static "
          "locked-off camera, full body visible from head to boots, plain studio background, even soft lighting.")


def ready() -> str | None:
    if os.environ.get("PIPELINE_LTX_TEST") != "1":
        return "set PIPELINE_LTX_TEST=1 to run LTX-2.3 (about 40 GB)"
    try:
        A.check_ltx(A.ltx_paths(ROOT))
    except A.AnimateError as e:
        return str(e)
    if not A.bodypose_tool(ROOT):
        return "no bin/bodypose (scripts/bootstrap.sh animate)"
    if not shutil.which("ffmpeg"):
        return "no ffmpeg"
    return None


@unittest.skipIf(ready(), ready() or "")
class WalkTheOperator(unittest.TestCase):
    def test_walk_follows_the_skeleton(self):
        self.walk()

    @unittest.skipIf(os.environ.get("PIPELINE_LTX_REMOVE") != "1", "set PIPELINE_LTX_REMOVE=1 too (about 8 min)")
    def test_walk_with_the_backpack_erased(self):
        # Two LTX passes and the photo-edit model between them: still within the limit, the kit named in anim.json.
        doc = self.walk(remove="backpack", keep="chest rig, radio and pouches on his chest")
        self.assertEqual(doc["remove"]["frames"], [24, 48, 72])
        self.assertTrue(all(k["pose_pct"] is not None for k in doc["remove"]["keys"]))

    def walk(self, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").symlink_to(ROOT / "scripts")
            (root / "out" / "operator").mkdir(parents=True)
            shutil.copy(ROOT / "in" / "operator.jpg", root / "out" / "operator" / "input.jpg")
            paths = A.ltx_paths(ROOT)
            env = {"PIPELINE_LTX_DIR": str(paths["clone"]), "PIPELINE_LTX_PYTHON": str(paths["python"]),
                   "PIPELINE_LTX_MODEL": str(paths["model"]), "PIPELINE_LTX_LORA": str(paths["lora"]),
                   "PIPELINE_LTX_GEMMA": str(paths["gemma"]), "PIPELINE_BODYPOSE": str(A.bodypose_tool(ROOT))}
            lift = A.lift_tool(ROOT)
            if lift:
                (root / "bin").mkdir()
                (root / "bin" / "lift-subject").symlink_to(lift)
            with mock.patch.dict(os.environ, env):
                doc = A.animate(root, "operator", prompt=PROMPT, skeleton="walk_side", action="walk", seed=42, **kw)
            check = doc["pose_check"]
            print(json.dumps({"pose_check": check, "ltx_seconds": doc["ltx_seconds"], "seconds": doc["seconds"],
                              "cutouts": doc["cutouts"]}, indent=1))
            self.assertEqual(doc["frames"], 73)
            self.assertGreaterEqual(check["frames_compared"], 60, "Vision found too few skeletons in the result")
            self.assertLessEqual(check["mean_joint_error_pct"], LIMIT_PCT)
            work = root / "out" / "operator" / "flat"
            self.assertEqual(len(list((work / "anim" / "walk" / "cutouts").glob("*.png"))), 73)
            self.assertTrue((work / "sheets" / "walk.png").is_file())
            return doc


if __name__ == "__main__":
    unittest.main()
