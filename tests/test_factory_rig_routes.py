#!/usr/bin/env python3
"""The factory's rig routes: POST /rig/<stem> body checks and flags, GET /rig/plans, POST /rig/<stem>/render,
POST /facing/detect (stubbed VLM), and facing errors as 400s."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "tests"))

import factory_http  # noqa: E402
import flow  # noqa: E402
from test_factory_http import http_json, write_triangle_glb  # noqa: E402


class RigArgvTests(unittest.TestCase):
    def argv(self, body: dict) -> list[str]:
        return factory_http.rig_argv("py", Path("/r"), "hen", body)[4:]

    def test_flags_pass_through(self) -> None:
        got = self.argv({"class": "avian", "plan": "ground_bird", "template": "build", "binder": "heat",
                         "pred": "never", "floor": 0, "render": True, "reframe": "1", "clips": ["Idle", "Peck"],
                         "straighten": "off"})
        self.assertEqual(got, ["--class", "avian", "--plan", "ground_bird", "--template", "build", "--binder", "heat",
                               "--pred", "never", "--floor", "0", "--render", "1", "--reframe", "1",
                               "--clips", "Idle,Peck", "--straighten", "off"])
        self.assertEqual(self.argv({}), [])
        self.assertEqual(self.argv({"class": "quadruped", "reuse_skin": True}), ["--class", "quadruped", "--reuse-skin"])
        self.assertEqual(self.argv({"reuse_skin": "0"}), [])

    def test_plan_sets_the_class(self) -> None:
        self.assertEqual(self.argv({"plan": "ground_bird"}), ["--class", "avian", "--plan", "ground_bird"])
        self.assertEqual(self.argv({"plan": "quadruped"}), ["--class", "quadruped", "--plan", "quadruped"])  # an alias

    def test_bad_values(self) -> None:
        for body in ({"template": "vendor"}, {"binder": "bone"}, {"pred": "sometimes"}, {"plan": "dragon"},
                     {"class": "fish"}, {"class": "humanoid", "plan": "ground_bird"}, {"floor": "2"},
                     {"render": "maybe"}, {"reframe": 3}, {"clips": ["Walk;rm -rf"]}, {"clips": 5},
                     {"straighten": "yes"}, {"reuse_skin": "maybe"}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                factory_http.rig_argv("py", Path("/r"), "hen", body)

    def test_render_argv(self) -> None:
        base = ["py", "/r/scripts/rig_render.py", "--stem", "hen"]
        self.assertEqual(factory_http.render_argv("py", Path("/r"), "hen", "default"), base)
        self.assertEqual(factory_http.render_argv("py", Path("/r"), "hen", None), base)
        got = factory_http.render_argv("py", Path("/r"), "hen", {"views": ["side", "xside"], "modes": "texture,heat:hips",
                                                               "frames": {"Walk": [0.25, 0.5], "Peck": "auto"},
                                                               "size": 256, "floor": False})
        self.assertEqual(got[4:], ["--views", "side,xside", "--modes", "texture,heat:hips", "--clips",
                                   "Walk@0.25,0.5;Peck", "--size", "256", "--no-floor"])
        for job in ({"views": ["under"]}, {"modes": ["paint"]}, {"size": 9000}, {"clips": "Walk$(x)"}, "fancy",
                    {"frames": {"Walk": [2]}}):
            with self.subTest(job=job), self.assertRaises(ValueError):
                factory_http.render_argv("py", Path("/r"), "hen", job)

    def test_intent_keeps_subject_and_rig(self) -> None:
        got = factory_http.clean_intent({"output": "ps1", "facing": "right", "subject": "creature", "kind": "avian",
                                         "name": "chicken", "facing_source": "user", "evil": 1,
                                         "rig": {"plan": "ground_bird", "canonical": True, "plan_hint": "ground_bird",
                                                 "x": "y"}})
        self.assertEqual(got, {"output": "ps1", "psx": {"facing": "right", "name": "chicken"}, "subject": "creature",
                               "kind": "avian", "name": "chicken", "facing_source": "user",
                               "rig": {"plan": "ground_bird", "plan_hint": "ground_bird", "canonical": True}})
        with self.assertRaises(ValueError):
            factory_http.clean_intent({"output": "ps1", "facing": "up"})


class RigRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "in").mkdir()
        from PIL import Image

        Image.new("RGB", (32, 24), (200, 120, 40)).save(self.root / "in" / "hen.png")
        (self.root / "in" / "junk.png").write_bytes(b"not a png")
        write_triangle_glb(self.root / "out" / "hen" / "psx.glb")
        (self.root / "out" / "hen" / "flat").mkdir()
        Image.new("RGB", (32, 24), (200, 120, 40)).save(self.root / "out" / "hen" / "flat" / "source.png")
        self.factory = factory_http.Factory(self.root)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), factory_http.make_handler(self.factory))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._tmp.cleanup()

    def wait(self, job_id: str) -> dict:
        deadline = time.time() + 10
        last: dict = {}
        while time.time() < deadline:
            _, last = http_json(self.base + f"/status/{job_id}")
            if last.get("state") != "running":
                break
            time.sleep(0.05)
        return last

    def test_plans_list_ground_bird(self) -> None:
        status, payload = http_json(self.base + "/rig/plans")
        self.assertEqual(status, 200)
        ids = [p["id"] for p in payload["plans"]]
        self.assertIn("ground_bird", ids)
        gb = next(p for p in payload["plans"] if p["id"] == "ground_bird")
        self.assertEqual(gb["class"], "avian")
        self.assertTrue(gb["built"])
        self.assertEqual(gb["clips"], ["Idle", "Walk", "Run", "Peck", "Flap", "Sleep", "Death"])
        _, root = http_json(self.base + "/")
        for r in ("GET /rig/plans", "POST /rig/<stem>/render", "POST /facing/detect"):
            self.assertIn(r, root["routes"])

    def test_rig_body_checked(self) -> None:
        for body in ({"template": "nope"}, {"binder": "x"}, {"plan": "dragon"}, {"pred": "x"}, {"render": "x"}):
            with self.subTest(body=body):
                status, payload = http_json(self.base + "/rig/hen", method="POST", data=body)
                self.assertEqual(status, 400)
                self.assertFalse(payload["ok"])
        self.assertIsNone(self.factory.status()["job_id"])  # nothing started
        status, _ = http_json(self.base + "/rig/nobody", method="POST", data={"template": "fixed"})
        self.assertEqual(status, 404)

    def test_rig_job_runs_the_flags(self) -> None:
        seen = {}

        class Proc:
            pid = 4242

            def __init__(self, argv, **kw):
                seen["argv"] = argv
                (self_root / "out" / "hen" / "rigged.glb").write_bytes(b"glb")

            def wait(self):
                return 0

        self_root = self.root
        with mock.patch.object(factory_http.subprocess, "Popen", Proc):
            status, job = http_json(self.base + "/rig/hen", method="POST",
                                    data={"plan": "ground_bird", "template": "build", "binder": "heat", "render": 0})
            self.assertEqual(status, 202)
            self.assertEqual(job["kind"], "rig")
            last = self.wait(job["job_id"])
        self.assertEqual(last["state"], "ok")
        self.assertEqual(seen["argv"][2:], ["--stem", "hen", "--class", "avian", "--plan", "ground_bird",
                                            "--template", "build", "--binder", "heat", "--render", "0"])

    def test_render_job(self) -> None:
        status, payload = http_json(self.base + "/rig/hen/render", method="POST", data={"job": "default"})
        self.assertEqual(status, 400)  # no rig yet
        self.assertIn("no rig", payload["error"])
        asset = self.root / "out" / "hen"
        (asset / "rigged.glb").write_bytes(b"glb")
        (asset / "rig").mkdir()
        (asset / "rig" / "rig.json").write_text("{}")
        flow.snapshot(self.root, "hen", "rig")
        seen = {}

        class Proc:
            pid = 4243

            def __init__(self, argv, **kw):
                seen["argv"] = argv
                (asset / "rig" / "preview").mkdir(parents=True, exist_ok=True)
                (asset / "rig" / "preview" / "contact_sheet.png").write_bytes(b"sheet")
                (asset / "rig" / "preview" / "frames.json").write_text("{}")

            def wait(self):
                return 0

        status, _ = http_json(self.base + "/rig/hen/render", method="POST", data={"job": {"views": ["moon"]}})
        self.assertEqual(status, 400)
        with mock.patch.object(factory_http.subprocess, "Popen", Proc):
            status, job = http_json(self.base + "/rig/hen/render", method="POST", data={"job": {"views": ["side"]}})
            self.assertEqual(status, 202)
            self.assertEqual(job["kind"], "render")
            last = self.wait(job["job_id"])
        self.assertEqual(last["state"], "ok", last)
        self.assertEqual(seen["argv"][1:], [str(self.factory.root / "scripts" / "rig_render.py"), "--stem", "hen",
                                            "--views", "side"])
        # The sheet joined the picked rig version without making a new one.
        rig = flow.read_manifest(asset)["stages"]["rig"]
        self.assertEqual(len(rig["takes"]), 1)
        self.assertIn("rig/preview/contact_sheet.png", rig["takes"][0]["files"])

    def test_facing_detect_stubbed(self) -> None:
        import facing

        calls = []

        def ask(png, question, keep=False):
            calls.append(Path(png).name)
            mirrored = "mirrored" in Path(png).name
            if question == facing.PROFILE_Q:
                return "YES"
            return "LEFT" if mirrored else "RIGHT"

        with mock.patch.object(facing, "ask", ask):
            status, det = http_json(self.base + "/facing/detect", method="POST", data={"image": "in/hen.png"},
                                    timeout=20)
            self.assertEqual(status, 200, det)
            self.assertEqual(det["head"], "right")
            self.assertTrue(det["profile"])
            self.assertGreaterEqual(det["confidence"], facing.ACCEPT)
            self.assertFalse(det["low"])
            self.assertEqual(det["image"], "in/hen.png")
            self.assertEqual(len(calls), 3)
            status, det = http_json(self.base + "/facing/detect", method="POST", data={"stem": "hen"}, timeout=20)
            self.assertEqual(status, 200)
            self.assertEqual(det["image"], "out/hen/flat/source.png")

        def unsure(png, question, keep=False):
            return "YES" if question == facing.PROFILE_Q else "RIGHT"  # no swap on the mirror: low confidence

        with mock.patch.object(facing, "ask", unsure):
            _, det = http_json(self.base + "/facing/detect", method="POST", data={"image": "in/hen.png"}, timeout=20)
            self.assertTrue(det["low"])
        status, _ = http_json(self.base + "/facing/detect", method="POST", data={})
        self.assertEqual(status, 400)
        status, _ = http_json(self.base + "/facing/detect", method="POST", data={"image": "../../etc/passwd.png"})
        self.assertEqual(status, 400)
        status, _ = http_json(self.base + "/facing/detect", method="POST", data={"image": "in/missing.png"})
        self.assertEqual(status, 404)
        with mock.patch.object(facing, "ask", ask):
            status, _ = http_json(self.base + "/facing/detect", method="POST", data={"image": "in/junk.png"})
        self.assertEqual(status, 400)
        self.assertEqual(self.factory._sync_draws, 0)  # the slot is given back

    def test_facing_detect_waits_for_a_job(self) -> None:
        with self.factory._lock:
            self.factory._job = {"job_id": "x", "state": "running", "stem": "hen"}
        status, payload = http_json(self.base + "/facing/detect", method="POST", data={"image": "in/hen.png"})
        self.assertEqual(status, 409)
        self.assertTrue(payload["busy"])

    def test_bad_facing_is_a_400_everywhere(self) -> None:
        status, payload = http_json(self.base + "/intent/hen", method="POST", data={"output": "ps1", "facing": "up"})
        self.assertEqual(status, 400)
        self.assertIn("facing", payload["error"])
        status, _ = http_json(self.base + "/generate", method="POST",
                              data={"image": "in/hen.png", "quality": "psx", "facing": "up", "stem": "hen"})
        self.assertEqual(status, 400)
        status, _ = http_json(self.base + "/generate", method="POST",
                              data={"image": "in/hen.png", "stem": "hen", "intent": {"output": "ps1", "facing": "up"}})
        self.assertEqual(status, 400)
        with mock.patch.object(factory_http, "still_backend", return_value="mflux"):
            status, _ = http_json(self.base + "/2d/draw", method="POST",
                                  data={"kind": "flat2d", "stem": "hen", "prompt": "a hen",
                                        "intent": {"output": "ps1", "facing": "sideways"}})
        self.assertEqual(status, 400)
        self.assertIsNone(self.factory.status()["job_id"])

    def test_intent_keeps_the_draws_rig_hint(self) -> None:
        path = self.root / "out" / "hen" / "intent.json"
        path.write_text(json.dumps({"output": "ps1", "psx": {"facing": "right"}, "facing_source": "drawn",
                                    "rig": {"canonical": True, "plan_hint": "ground_bird"}}))
        status, _ = http_json(self.base + "/intent/hen", method="POST", data={"output": "ps1", "facing": "right",
                                                                              "height": 48})
        self.assertEqual(status, 200)
        doc = json.loads(path.read_text())
        self.assertEqual(doc["rig"], {"canonical": True, "plan_hint": "ground_bird"})
        self.assertEqual(doc["facing_source"], "drawn")
        self.assertEqual(doc["psx"], {"facing": "right", "height": "48"})


if __name__ == "__main__":
    unittest.main()
