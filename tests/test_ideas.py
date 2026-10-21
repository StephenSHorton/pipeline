#!/usr/bin/env python3
"""Ideas: draw several stills, keep some, use one as the concept."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import factory_http  # noqa: E402
import flow  # noqa: E402
from flat2d import ideas, optional  # noqa: E402


def fake_still(prompt: str, seed: int = 1) -> Image.Image:
    return Image.new("RGB", (32, 32), (seed % 256, 0, 255))


class IdeasTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.factory = factory_http.Factory(self.root)
        patches = [
            mock.patch.object(optional, "still_image", side_effect=fake_still),
            mock.patch.object(ideas, "still_backend", return_value="z-image"),
            mock.patch.dict("os.environ", {"FLAT2D_LIFT": str(self.root / "no-lift")}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def post(self, action: str, payload: dict) -> tuple[int, dict]:
        body = json.dumps({"stem": "lamp", **payload}).encode("utf-8")
        status, raw, _ = factory_http.dispatch(self.factory, "POST", "/2d/" + action, {}, body)
        return status, json.loads(raw)

    def test_draw_keep_clear_use(self) -> None:
        for seed in (11, 22, 33):
            status, data = self.post("idea", {"prompt": "brass lamp", "seed": seed})
            self.assertEqual(status, 201, data)
        self.assertEqual([i["file"] for i in data["items"]], ["idea-01.png", "idea-02.png", "idea-03.png"])
        self.assertEqual(data["items"][1]["seed"], 22)

        self.post("keep", {"file": "idea-02.png", "keep": True})
        status, data = self.post("clear", {})
        self.assertEqual([i["file"] for i in data["items"]], ["idea-02.png"])
        self.assertFalse((self.root / "out" / "lamp" / "ideas" / "idea-01.png").exists())

        status, data = self.post("idea", {"prompt": "brass lamp", "seed": 44})
        self.assertEqual(data["item"]["file"], "idea-03.png")

        status, data = self.post("use", {"file": "idea-02.png"})
        self.assertEqual(status, 200, data)
        source = Image.open(self.root / "out" / "lamp" / "flat" / "source.png")
        self.assertEqual(source.getpixel((0, 0))[:3], (22, 0, 255))
        key = json.loads((self.root / "out" / "lamp" / "flat" / "key.json").read_text())
        self.assertEqual(key["rgb"], [22, 0, 255])
        stages = {s["stage"]: s for s in flow.state(self.root, "lamp")["stages"]}
        self.assertEqual(stages["concept"]["pick"], "t1")
        self.assertIn("flat/prompt.txt", stages["concept"]["files"])

        status, raw, _ = factory_http.dispatch(self.factory, "GET", "/2d/lamp/ideas", {}, b"")
        listed = json.loads(raw)
        self.assertEqual(listed["prompt"], "brass lamp")
        self.assertTrue(next(i for i in listed["items"] if i["file"] == "idea-02.png")["used"])

        assets = self.factory.list_assets()["assets"]
        self.assertTrue(assets[0]["has_ideas"])

    def test_cloudy_ground_is_redrawn_on_a_new_seed(self) -> None:
        import random

        rnd = random.Random(3)

        def cloudy_then_flat(prompt: str, seed: int = 1) -> Image.Image:
            if seed == 5:
                im = Image.new("RGB", (32, 32))
                im.putdata([(255 - rnd.randint(0, 90), rnd.randint(0, 90), 255 - rnd.randint(0, 90)) for _ in range(32 * 32)])
                return im
            return fake_still(prompt, seed)

        with mock.patch.object(optional, "still_image", side_effect=cloudy_then_flat) as drawn:
            status, data = self.post("idea", {"prompt": "brass lamp", "seed": 5})
        self.assertEqual(status, 201, data)
        self.assertEqual(drawn.call_count, 2)
        self.assertNotEqual(data["item"]["seed"], 5)
        self.assertEqual(data["item"]["redrawn"], ["cloudy"])
        self.assertIn("A single object only", drawn.call_args.args[0])

    def test_all_cloudy_keeps_the_flattest(self) -> None:
        import random

        draws = []

        def cloudy(prompt: str, seed: int = 1) -> Image.Image:
            spread = [120, 60, 90][len(draws)]
            rnd = random.Random(len(draws))
            im = Image.new("RGB", (32, 32))
            im.putdata([(255 - rnd.randint(0, spread), rnd.randint(0, spread // 2), 255 - rnd.randint(0, spread)) for _ in range(32 * 32)])
            draws.append(seed)
            return im

        with mock.patch.object(optional, "still_image", side_effect=cloudy):
            image, seed, redrawn = optional.clean_still("lamp", 7)
        self.assertEqual(len(draws), 3)
        self.assertEqual(seed, draws[1])  # the second try had the flattest ground
        self.assertEqual(redrawn, ["cloudy", "cloudy"])

    def test_shadow_on_the_ground_is_a_flaw(self) -> None:
        im = Image.new("RGB", (40, 40), (255, 0, 255))
        for y in range(30, 36):
            for x in range(8, 32):
                im.putpixel((x, y), (120, 0, 120))  # the ground, darker: a cast shadow
        for y in range(8, 30):
            for x in range(12, 28):
                im.putpixel((x, y), (60, 70, 50))
        subject = bytearray(40 * 40)
        for y in range(8, 30):
            for x in range(12, 28):
                subject[y * 40 + x] = 1
        with mock.patch("flat2d.store.vision_mask", return_value=subject):
            self.assertEqual(optional.ground_flaws(im), ["shadow"])
        with mock.patch("flat2d.store.vision_mask", return_value=None):
            self.assertEqual(optional.ground_flaws(im), [])
        self.assertEqual(optional.ground_flaws(Image.new("RGB", (40, 40), (255, 0, 255))), [])

    def test_empty_prompt_and_unknown_idea(self) -> None:
        status, data = self.post("idea", {"prompt": "  "})
        self.assertEqual(status, 400)
        status, data = self.post("use", {"file": "idea-09.png"})
        self.assertEqual(status, 400)

    def test_no_model(self) -> None:
        with mock.patch.object(ideas, "still_backend", return_value=""):
            status, data = self.post("idea", {"prompt": "lamp"})
        self.assertEqual(status, 400)
        self.assertFalse(data["available"])


if __name__ == "__main__":
    unittest.main()
