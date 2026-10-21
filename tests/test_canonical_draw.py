#!/usr/bin/env python3
"""The canonical creature draw (design section 7): the pose prompt, key.json, the facing redraw and mirror."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import facing  # noqa: E402
from flat2d import drawjob, optional, pose  # noqa: E402
from flat2d.optional import still_prompt  # noqa: E402

TODAY_RADIO = (
    "radio. A single object only, centered, floating, isolated on a solid flat magenta #FF00FF background. "
    "Flat even lighting from the front. No shadow, no drop shadow, no cast shadow, no contact shadow, "
    "no reflection, no floor, no ground plane, no glow, no smoke, no particles, no effects, nothing else in the image."
)

STUB_ENV = {"FLAT2D_STILL": "stub", "FLAT2D_STUB_STEP": "0", "FLAT2D_STUB_LOAD": "0", "FLAT2D_STUB_FLAWS": "0",
            "FLAT2D_PREVIEW": "off"}


def det(head, profile=True, confidence=0.9):
    return {"head": head, "profile": profile, "confidence": confidence, "how": "vlm"}


class PromptTests(unittest.TestCase):
    def test_creature_prompt(self) -> None:
        kind = pose.pose_for({"output": "ps1", "subject": "creature"}, "A PS1 style graphics of a chicken")
        self.assertEqual(kind, "avian")
        p = still_prompt("A PS1 style graphics of a chicken", kind)
        self.assertIn("Head and beak pointing toward the RIGHT edge", p)
        self.assertIn("pointing toward the RIGHT edge", p)
        self.assertIn("strict profile", p)
        self.assertNotIn("floating", p)
        self.assertTrue(p.startswith("A PS1 style graphics of a chicken. Full body"))
        self.assertIn("#FF00FF", p)
        q = still_prompt("a cow", "quadruped")
        self.assertIn("all four legs visible and straight", q)
        self.assertNotIn("floating", q)
        h = still_prompt("a knight", "humanoid")
        self.assertIn("A-pose", h)
        self.assertIn("facing the camera", h)

    def test_prop_prompt_unchanged(self) -> None:
        self.assertEqual(still_prompt("radio"), TODAY_RADIO)
        self.assertEqual(still_prompt("radio", None), TODAY_RADIO)
        kind = pose.pose_for({"output": "ps1"}, "radio")
        self.assertIsNone(kind)
        self.assertEqual(still_prompt("radio", kind), TODAY_RADIO)

    def test_pose_for(self) -> None:
        cases = [
            (None, "a chicken", None),
            ({"output": "sprite"}, "a chicken", None),  # 2D-only: never posed
            ({"output": "ps1"}, "A PS1 style graphics of a chicken", "avian"),
            ({"output": "model"}, "two brown horses", "quadruped"),
            ({"output": "stack"}, "a farmer's straw hat", None),
            ({"output": "ps1"}, "a chicken drumstick", None),
            ({"output": "ps1", "subject": "creature"}, "a mech spider", "creature"),
            ({"output": "ps1", "subject": "humanoid"}, "a knight", "humanoid"),
            ({"output": "ps1", "subject": "prop"}, "a chicken", None),
            ({"output": "ps1"}, "a farmer", None),  # people only by subject
            ({"output": "ps1", "name": "cow"}, "PS1 style", "quadruped"),
            ({"output": "ps1", "kind": "avian"}, "a thing", "avian"),
        ]
        for intent, prompt, want in cases:
            with self.subTest(intent=intent, prompt=prompt):
                self.assertEqual(pose.pose_for(intent, prompt), want)
                self.assertEqual(pose.is_creature(intent, prompt), want is not None)


class GenerateStillTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "out").mkdir()
        self.env = mock.patch.dict("os.environ", {**STUB_ENV, "FLAT2D_LIFT": str(self.root / "no-lift")})
        self.env.start()
        optional._STUB_LOADED = False

    def tearDown(self) -> None:
        self.env.stop()
        self._tmp.cleanup()

    def key(self, stem: str) -> dict:
        return json.loads((self.root / "out" / stem / "flat" / "key.json").read_text())

    def draw(self, answers, intent=None, prompt="A PS1 style graphics of a chicken"):
        seq = list(answers)
        calls = []

        def detect(png):
            calls.append(Path(png).name)
            return seq.pop(0) if len(seq) > 1 else seq[0]

        with mock.patch.object(facing, "detect_image", side_effect=detect):
            saved = optional.generate_still(self.root, "chick", prompt, intent=intent or {"output": "ps1"})
        return saved, calls

    def test_canonical_key(self) -> None:
        saved, calls = self.draw([det("right")])
        self.assertEqual(len(calls), 1)
        k = self.key("chick")
        self.assertEqual((k["pose"], k["facing"], k["canonical"]), ("profile_right", "right", True))
        self.assertEqual(k["facing_check"]["how"], "vlm")
        self.assertIn("rgb", k)  # the ground key is still there
        self.assertEqual(saved["redrawn"], [])
        self.assertEqual(saved["kind"], "avian")
        prompt = (self.root / "out" / "chick" / "flat" / "prompt.txt").read_text().strip()
        self.assertEqual(prompt, "A PS1 style graphics of a chicken")

    def test_redraw_then_accept(self) -> None:
        saved, calls = self.draw([det("left"), det("right")])
        self.assertEqual(len(calls), 2)
        self.assertEqual(saved["redrawn"], ["facing"])
        k = self.key("chick")
        self.assertEqual((k["canonical"], k["facing_check"]["how"]), (True, "vlm"))

    def test_mirrored_fallback(self) -> None:
        from PIL import Image, ImageOps

        drawn = []
        real = optional.still_image

        def still(prompt, seed=1):
            im = real(prompt, seed=seed)
            im = getattr(im, "image", im).copy()
            im.putpixel((0, 0), (1, 2, 3))  # mark the top-left corner so the mirror is visible
            drawn.append(im)
            return im

        with mock.patch.object(optional, "still_image", side_effect=still):
            saved, calls = self.draw([det("left")])
        self.assertEqual(len(calls), optional.REDRAWS + 1)  # every try had the head left: redraws spent
        self.assertEqual(saved["redrawn"], ["facing"] * optional.REDRAWS)
        k = self.key("chick")
        self.assertEqual((k["facing"], k["canonical"], k["pose"]), ("right", True, "profile_right"))
        self.assertEqual(k["facing_check"]["how"], "mirrored")
        self.assertEqual(k["facing_check"]["head_before"], "left")
        src = Image.open(self.root / "out" / "chick" / "flat" / "source.png").convert("RGB")
        w = src.width
        self.assertEqual(src.getpixel((w - 1, 0)), (1, 2, 3))
        self.assertTrue(any(ImageOps.mirror(d.convert("RGB")).tobytes() == src.tobytes() for d in drawn))

    def test_three_quarter_is_not_canonical(self) -> None:
        saved, _ = self.draw([det("right", profile=False)])
        k = self.key("chick")
        self.assertFalse(k["canonical"])
        self.assertEqual(k["facing_check"]["how"], "not_canonical")
        self.assertIn("profile", k["facing_check"]["why"])

    def test_vlm_unavailable_keeps_canonical(self) -> None:
        saved, calls = self.draw([{"head": None, "profile": None, "confidence": 0.0, "how": "unavailable",
                                   "why": "down"}])
        self.assertEqual(len(calls), 1)  # no redraws burned on a missing model
        k = self.key("chick")
        self.assertEqual((k["canonical"], k["facing"], k["facing_check"]["how"]), (True, "right", "unchecked"))

    def test_posed_idea_keeps_its_facing(self) -> None:
        from flat2d import ideas

        with mock.patch.object(facing, "detect_image") as detect:
            got = ideas.draw_idea(self.root, "hen", "a chicken", 5, intent={"output": "ps1"})
            plain = ideas.draw_idea(self.root, "hen", "a chicken", 6)
        detect.assert_not_called()  # Ideas never run the vision check
        self.assertEqual(got["item"]["pose"], "profile_right")
        self.assertNotIn("pose", plain["item"])
        ideas.use_idea(self.root, "hen", got["item"]["file"])
        k = self.key("hen")
        self.assertEqual((k["pose"], k["facing"], k["facing_source"], k["canonical"]),
                         ("profile_right", "right", "drawn", False))

    def test_prop_draw_unchanged(self) -> None:
        with mock.patch.object(facing, "detect_image") as detect:
            optional.generate_still(self.root, "radio", "a radio", intent={"output": "ps1"})
            optional.generate_still(self.root, "radio2", "a chicken")  # no intent: an old caller
        detect.assert_not_called()
        for stem in ("radio", "radio2"):
            k = self.key(stem)
            self.assertNotIn("canonical", k)
            self.assertNotIn("pose", k)


class FakeFactory:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._job = {"job_id": "j1", "state": "running"}


class DrawJobCanonicalTests(unittest.TestCase):
    """drawjob threads the payload intent through and writes intent.json psx.facing and rig.canonical."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "out").mkdir()
        self.env = mock.patch.dict("os.environ", {**STUB_ENV, "FLAT2D_LIFT": str(self.root / "no-lift")})
        self.env.start()
        optional._STUB_LOADED = False

    def tearDown(self) -> None:
        self.env.stop()
        self._tmp.cleanup()

    def run_job(self, answers, intent):
        import factory_http

        seq = list(answers)
        factory = FakeFactory(self.root)
        spec = {"job_id": "j1", "kind": "flat2d", "stem": "chicken-canon", "backend": "stub", "steps": []}
        payload = {"prompt": "A PS1 style graphics of a chicken", "intent": intent}
        drawjob.fresh_log(self.root / "out" / "chicken-canon" / "draw.log")
        with mock.patch.object(facing, "detect_image", side_effect=lambda p: seq.pop(0) if len(seq) > 1 else seq[0]):
            drawjob.run(factory, spec, payload, write_intent=factory_http.write_intent)
        self.assertEqual(factory._job["state"], "ok", factory._job.get("error"))
        return factory._job["result"]

    def test_key_and_intent_are_canonical(self) -> None:
        result = self.run_job([det("right")], {"output": "ps1", "psx": {"height": "48"}})
        asset = self.root / "out" / "chicken-canon"
        k = json.loads((asset / "flat" / "key.json").read_text())
        self.assertEqual((k["pose"], k["facing"], k["canonical"]), ("profile_right", "right", True))
        intent = json.loads((asset / "intent.json").read_text())
        self.assertEqual(intent["psx"], {"height": "48", "facing": "right"})
        self.assertEqual(intent["rig"], {"canonical": True, "plan_hint": "ground_bird"})
        self.assertEqual(intent["facing_source"], "drawn")
        self.assertEqual((result["facing"], result["canonical"]), ("right", True))

    def test_mirrored_is_recorded_in_the_job(self) -> None:
        result = self.run_job([det("left")], {"output": "ps1"})
        self.assertEqual(result["facing_check"]["how"], "mirrored")
        self.assertTrue(any("mirrored" in n for n in result["notes"]))
        log = (self.root / "out" / "chicken-canon" / "draw.log").read_text()
        self.assertIn("The head points the wrong way.", log)
        intent = json.loads((self.root / "out" / "chicken-canon" / "intent.json").read_text())
        self.assertEqual(intent["psx"]["facing"], "right")


if __name__ == "__main__":
    unittest.main()
