#!/usr/bin/env python3
"""A 2D draw and Ideas as factory jobs (flat2d/drawjob.py), on the stub still model."""
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
sys.path.insert(0, str(ROOT / "scripts"))

import factory_http  # noqa: E402
from flat2d import drawjob, optional  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_factory_http import http_bytes, http_json  # noqa: E402


def real_jpeg(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (200, 40, 40)).save(path, format="JPEG")


class LabelTests(unittest.TestCase):
    def test_draw_stage_names_never_collide_with_rig_or_generate_labels(self) -> None:
        # factory_http merges drawjob.LABELS after the generate and rig labels: a shared name relabels theirs.
        theirs = {"isolate", "shape", "reduce", "paint", "retexture", "psx", "lods", "inspect", "stack", "preview",
                  "classify", "rest", "pose", "skin", "clean", "clips", "check", "fbx"}
        self.assertEqual(set(drawjob.LABELS) & theirs, set())
        self.assertEqual(factory_http.STAGE_LABELS["check"], "Check the rig")
        self.assertIn("views_split", drawjob.LABELS)
        self.assertIn("views_check", drawjob.LABELS)
        self.assertEqual(factory_http.STAGE_LABELS["views_check"], drawjob.LABELS["views_check"])

    def test_a_sheet_plan(self) -> None:
        self.assertEqual(drawjob.plan_for("flat2d", loaded=False, views=True),
                         ["load", "draw", "views_split", "views_check"])
        self.assertEqual(drawjob.plan_for("flat2d", loaded=True), ["draw", "ground"])


class DrawJobTests(unittest.TestCase):
    STEP = "0.02"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "in").mkdir()
        (self.root / "out").mkdir()
        self.env = mock.patch.dict(
            "os.environ",
            {
                "FLAT2D_STILL": "stub",
                "FLAT2D_STUB_STEP": self.STEP,
                "FLAT2D_STUB_LOAD": "0.05",
                "FLAT2D_STUB_FLAWS": "0",
                "FLAT2D_LIFT": str(self.root / "no-lift"),
                "FLAT2D_SFX_DIR": str(self.root / "no-sfx"),
                "FLAT2D_PREVIEW": "on",
            },
        )
        self.env.start()
        optional._STUB_LOADED = False
        self.factory = factory_http.Factory(self.root)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), factory_http.make_handler(self.factory))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        job = self.factory._snapshot()
        if job and job.get("state") == "running":
            self.factory.cancel(job["job_id"])
            self.wait(job["job_id"], timeout=10)
        self.httpd.shutdown()
        self.httpd.server_close()
        self.env.stop()
        self._tmp.cleanup()

    def start(self, body: dict) -> dict:
        status, payload = http_json(self.base + "/2d/draw", method="POST", data=body)
        self.assertEqual(status, 202, payload)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["status_url"], f"/status/{payload['job_id']}")
        return payload

    def wait(self, job_id: str, timeout: float = 15) -> dict:
        deadline = time.time() + timeout
        last: dict = {}
        while time.time() < deadline:
            _, last = http_json(self.base + f"/status/{job_id}")
            if last.get("state") != "running":
                return last
            time.sleep(0.03)
        self.fail(f"job still running: {last}")

    def keys(self, st: dict) -> list[str]:
        return [s["key"] for s in (st.get("progress") or {}).get("stages") or []]

    def test_a_photo_draw_runs_the_sheet_from_the_photo(self) -> None:
        import facing
        from PIL import Image

        (self.root / "in").mkdir(exist_ok=True)
        Image.new("RGB", (64, 48), (200, 10, 10)).save(self.root / "in" / "cow.jpg")
        answers = iter(["a red cow", "YES"])
        with mock.patch.dict("os.environ", {"FLAT2D_SHEET": "1152x1152"}), mock.patch.object(
            facing, "detect_image", return_value={"head": "right", "profile": True, "confidence": 0.9, "how": "vlm"}
        ), mock.patch.object(facing, "ask", side_effect=lambda *a, **k: next(answers)):
            job = self.start({"kind": "flat2d", "stem": "cow", "reference": "in/cow.jpg",
                              "intent": {"output": "ps1", "kind": "quadruped"}})
            st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "ok", st)
        self.assertEqual(self.keys(st)[-3:], ["draw", "views_split", "views_check"])
        self.assertIn("Your photo shows a red cow.", st.get("log_tail") or "")
        self.assertIn("They show your photo's subject.", st.get("log_tail") or "")
        doc = json.loads((self.root / "out" / "cow" / "flat" / "views.json").read_text())
        self.assertEqual(doc["sheet"]["reference"]["file"], "in/cow.jpg")
        self.assertEqual(st["result"]["views"], "flat/views.json")

    def test_a_3d_bound_draw_is_a_four_view_sheet_in_its_own_stages(self) -> None:
        import facing

        with mock.patch.dict("os.environ", {"FLAT2D_SHEET": "1152x1152"}), mock.patch.object(
            facing, "detect_image", return_value={"head": "right", "profile": True, "confidence": 0.9, "how": "vlm"}
        ):
            job = self.start({"kind": "flat2d", "stem": "hen", "prompt": "a chicken", "intent": {"output": "ps1"}})
            st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "ok", st)
        self.assertEqual(self.keys(st), ["load", "draw", "views_split", "views_check"])
        labels = {s["key"]: s["label"] for s in st["progress"]["stages"]}
        self.assertEqual(labels["views_check"], "Check the views agree")
        r = st["result"]
        self.assertEqual((r["views"], r["sheet"], r["consistent"], r["canonical_view"]),
                         ("flat/views.json", "flat/sheet.png", True, "right"))
        flat = self.root / "out" / "hen" / "flat"
        for name in ("sheet.png", "views.json", "views/front.png", "views/right.png", "views/back.png",
                     "views/left.png", "source.png", "key.json"):
            self.assertTrue((flat / name).is_file(), name)
        import flow

        take = flow.read_manifest(self.root / "out" / "hen")["stages"]["concept"]["takes"][-1]
        for name in ("flat/sheet.png", "flat/views.json", "flat/views/right.png"):
            self.assertIn(name, take["files"])
        times = json.loads((self.root / "out" / "draw-times.json").read_text())
        for key in ("views_split", "views_check", "attempt"):
            self.assertIn(key, times["stub"])

    def test_draw_job_runs_stages_saves_still_and_remembers_times(self) -> None:
        first = self.start({"kind": "flat2d", "stem": "lamp", "prompt": "a brass lamp", "intent": {"output": "sprite"}})
        self.assertEqual(first["kind"], "flat2d")
        st = self.wait(first["job_id"])
        self.assertEqual(st["state"], "ok", st)
        self.assertEqual(self.keys(st), ["load", "draw", "ground"])
        self.assertEqual(st["progress"]["stage"], "ground")
        self.assertEqual(st["backend"], "stub")
        self.assertIsNone(st["estimate_s"])
        flat = self.root / "out" / "lamp" / "flat"
        self.assertTrue((flat / "source.png").is_file())
        self.assertTrue((flat / "key.json").is_file())
        self.assertTrue((self.root / "out" / "lamp" / "intent.json").is_file())
        import flow

        self.assertTrue(flow.read_manifest(self.root / "out" / "lamp")["stages"]["concept"]["takes"])
        self.assertTrue(st["preview_url"].startswith(f"/2d/lamp/drawing?j={first['job_id']}&v="))
        code, jpeg = http_bytes(self.base + st["preview_url"])
        self.assertEqual(code, 200)
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        for key in ("load", "draw", "ground"):
            self.assertIsNotNone(st["stage_times"][key]["ended"], key)
        self.assertEqual(st["result"]["source"], "flat/source.png")
        self.assertEqual(st["result"]["redrawn"], [])
        self.assertIsInstance(st["result"]["seed"], int)
        self.assertIn("Step 8 of 8 [100%]", (self.root / "out" / "lamp" / "draw.log").read_text())
        times = json.loads((self.root / "out" / "draw-times.json").read_text())
        for key in ("load", "attempt", "ground"):
            self.assertIn(key, times["stub"])

        _, s2 = http_json(self.base + "/2d/status")
        self.assertTrue(s2["loaded"])
        self.assertIsNotNone(s2["estimate_s"]["draw"])
        self.assertIsNotNone(s2["estimate_s"]["draw_first"])

        second = self.start({"kind": "flat2d", "stem": "lamp", "prompt": "a brass lamp, taller"})
        st2 = self.wait(second["job_id"])
        self.assertEqual(st2["state"], "ok", st2)
        self.assertEqual(self.keys(st2), ["draw", "ground"])
        self.assertIsNotNone(st2["estimate_s"])
        self.assertTrue((self.root / "out" / "lamp" / "draw.prev.log").is_file())

    def test_parse_progress_reads_the_draw_log(self) -> None:
        log = "\n".join(
            [
                "STAGES draw ground pieces",
                "STAGE draw Draw the picture",
                "Step 4 of 8 [ 50%]",
            ]
        )
        p = factory_http.parse_progress(log)
        self.assertEqual(p["stage_index"], 0)
        self.assertEqual(p["sub"], 0.5)
        self.assertEqual(p["stages"][0]["label"], "Draw the picture")
        log += "\nStep 8 of 8 [100%]\nSTAGE ground Check the ground\nChecking the ground."
        log += "\nSTAGES draw ground redraw pieces\nSTAGE redraw Draw it again\nStep 2 of 8 [ 25%]"
        p = factory_http.parse_progress(log)
        self.assertEqual(p["stage"], "redraw")
        self.assertEqual(p["stage_index"], 2)
        self.assertEqual(p["stage_count"], 4)
        self.assertEqual(p["sub"], 0.25)

    def test_cloudy_ground_redraws_in_its_own_stage(self) -> None:
        with mock.patch.dict("os.environ", {"FLAT2D_STUB_FLAWS": "1"}):
            job = self.start({"kind": "flat2d", "stem": "crate", "prompt": "a wooden crate"})
            st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "ok", st)
        self.assertEqual(self.keys(st), ["load", "draw", "ground", "redraw"])
        self.assertEqual(st["result"]["redrawn"], ["cloudy"])
        log = (self.root / "out" / "crate" / "draw.log").read_text()
        self.assertIn("Ground is cloudy. Drawing again on a new seed (2 of 3).", log)
        self.assertIsNotNone(st["stage_times"]["redraw"]["ended"])

    def test_picked_steps_run_as_stages(self) -> None:
        job = self.start(
            {
                "kind": "flat2d",
                "stem": "drum",
                "prompt": "a war drum",
                "steps": ["sound", "pieces"],
                "sound": "a deep boom",
                "pieces": "drum",
            }
        )
        st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "ok", st)
        self.assertEqual(self.keys(st), ["load", "draw", "ground", "sound", "pieces"])
        self.assertEqual(st["steps"], ["sound", "pieces"])
        self.assertIn("Sound: stand-in. The effect runner is not set.", st["result"]["notes"])
        self.assertTrue(any(n.startswith("Pieces") for n in st["result"]["notes"]))
        self.assertEqual(st["result"]["hear"], "sounds/asset.wav")
        self.assertIn("sound", st["result"]["ran"])
        self.assertIsNotNone(st["stage_times"]["pieces"]["ended"])
        times = json.loads((self.root / "out" / "draw-times.json").read_text())["stub"]
        self.assertIn("sound", times)
        self.assertIn("pieces", times)

    def test_drop_source_runs_only_the_picked_steps(self) -> None:
        status, body = http_json(self.base + "/2d/draw", method="POST",
                                 data={"kind": "flat2d", "stem": "none", "source": "drop", "steps": ["sound"]})
        self.assertEqual(status, 400)
        optional.stub_still("x", seed=3).save(self.root / "in" / "pic.png")
        flat = self.root / "out" / "drop" / "flat"
        flat.mkdir(parents=True)
        (flat / "source.png").write_bytes((self.root / "in" / "pic.png").read_bytes())
        job = self.start({"kind": "flat2d", "stem": "drop", "source": "drop", "steps": ["sound"], "sound": "a click"})
        st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "ok", st)
        self.assertEqual(self.keys(st), ["sound"])
        self.assertIsNone(st["preview_url"])

    def test_cancel_stops_the_draw(self) -> None:
        with mock.patch.dict("os.environ", {"FLAT2D_STUB_STEP": "0.3"}):
            job = self.start({"kind": "flat2d", "stem": "slow", "prompt": "a slow thing"})
            time.sleep(0.5)
            status, res = http_json(self.base + f"/cancel/{job['job_id']}", method="POST")
            self.assertEqual(status, 200)
            self.assertEqual(res["state"], "cancelling")
            asked = time.time()
            st = self.wait(job["job_id"], timeout=5)
        self.assertLess(time.time() - asked, 2.0)
        self.assertEqual(st["state"], "cancelled")
        self.assertEqual(st["error"], "Cancelled")
        self.assertFalse((self.root / "out" / "slow" / "flat" / "source.png").exists())
        self.assertFalse((self.root / "out" / "draw-times.json").exists())
        self.assertTrue((self.root / "out" / "slow" / "draw.log").read_text().rstrip().endswith("Cancelled."))

    def test_one_job_at_a_time(self) -> None:
        (self.root / "out" / "radio").mkdir()
        real_jpeg(self.root / "in" / "radio.jpg")  # a real picture: /generate refuses anything else
        with mock.patch.dict("os.environ", {"FLAT2D_STUB_STEP": "0.3"}):
            job = self.start({"kind": "flat2d", "stem": "busy", "prompt": "a busy thing"})
            self.assertTrue(self.factory.health()["busy"])
            _, health = http_json(self.base + "/health")
            self.assertEqual(health["job_id"], job["job_id"])
            status, body = http_json(self.base + "/generate", method="POST", data={"image": "in/radio.jpg"})
            self.assertEqual(status, 409, body)
            status, body = http_json(self.base + "/rig/radio", method="POST", data={})
            self.assertEqual(status, 409, body)
            status, body = http_json(self.base + "/2d/draw", method="POST",
                                     data={"kind": "flat2d", "stem": "other", "prompt": "x"})
            self.assertEqual(status, 409, body)
            self.assertTrue(body["busy"])
            for action in ("generate", "idea"):
                status, body = http_json(self.base + f"/2d/{action}", method="POST", data={"stem": "other", "prompt": "x"})
                self.assertEqual(status, 409, body)
                self.assertEqual(body, {"ok": False, "busy": True, "error": "the factory is busy with another job"})
            self.assertFalse((self.root / "out" / "other").exists())
            self.factory.cancel(job["job_id"])
            self.wait(job["job_id"])

        release = threading.Event()
        self.factory._runner = lambda spec: (release.wait(5), 1)[1]
        status, gen = http_json(self.base + "/generate", method="POST", data={"image": "in/radio.jpg", "stem": "radio"})
        self.assertEqual(status, 202, gen)
        status, body = http_json(self.base + "/2d/draw", method="POST", data={"kind": "flat2d", "stem": "lamp", "prompt": "x"})
        self.assertEqual(status, 409, body)
        release.set()
        self.wait(gen["job_id"])

    def test_ideas_job_draws_count_pictures(self) -> None:
        job = self.start({"kind": "ideas", "stem": "fern", "prompt": "a fern in a pot", "count": 2})
        self.assertEqual(job["kind"], "ideas")
        st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "ok", st)
        self.assertEqual(self.keys(st), ["load", "idea1", "idea2"])
        self.assertEqual(st["progress"]["stages"][1]["label"], "Picture 1")
        self.assertEqual(len(st["result"]["items"]), 2)
        self.assertEqual(len(st["result"]["ideas"]["items"]), 2)
        data = json.loads((self.root / "out" / "fern" / "ideas" / "ideas.json").read_text())
        self.assertEqual(len(data["items"]), 2)
        self.assertTrue(st["preview_url"])
        self.assertIn("idea", json.loads((self.root / "out" / "draw-times.json").read_text())["stub"])
        status, body = http_json(self.base + "/2d/draw", method="POST",
                                 data={"kind": "ideas", "stem": "fern", "prompt": "x", "count": 9})
        self.assertEqual(status, 400, body)

    def test_payload_never_reaches_the_job(self) -> None:
        job = self.start(
            {
                "kind": "flat2d",
                "stem": "clip",
                "prompt": "a runner",
                "steps": ["motion"],
                "motion": "run",
                "frames": ["aGVsbG8="],
                "video": "",
            }
        )
        seen = [self.factory._snapshot()]
        st = self.wait(job["job_id"])
        for doc in seen + [st]:
            self.assertNotIn("video", doc)
            self.assertNotIn("frames", doc)
            self.assertNotIn("aGVsbG8=", json.dumps(doc))
        self.assertEqual(st["state"], "ok", st)

    def test_bad_bodies_are_400(self) -> None:
        for body in (
            {"kind": "flat2d", "stem": "x", "prompt": ""},
            {"kind": "paint", "stem": "x", "prompt": "a"},
            {"kind": "flat2d", "stem": "../x", "prompt": "a"},
            {"kind": "flat2d", "stem": "x", "prompt": "a", "steps": ["dance"]},
        ):
            status, payload = http_json(self.base + "/2d/draw", method="POST", data=body)
            self.assertEqual(status, 400, (body, payload))
        with mock.patch.dict("os.environ", {"FLAT2D_STILL": "qwen-image"}), mock.patch(
            "flat2d.sheet.mflux_ready", return_value=False
        ):
            status, payload = http_json(self.base + "/2d/draw", method="POST",
                                        data={"kind": "flat2d", "stem": "x", "prompt": "a"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "No local image model. Install mflux.")
        status, _ = http_json(self.base + "/2d/x/drawing")
        self.assertEqual(status, 404)

    def test_estimate_needs_every_planned_stage(self) -> None:
        drawjob.write_times(self.root, "stub", {"attempt": 3.0, "ground": 0.5})
        self.assertEqual(drawjob.estimate(self.root, "stub", ["draw", "ground"]), 3.5)
        self.assertIsNone(drawjob.estimate(self.root, "stub", ["load", "draw", "ground"]))
        self.assertIsNone(drawjob.estimate(self.root, "qwen-image", ["draw"]))
        drawjob.write_times(self.root, "stub", {"load": 10.0})
        # A redraw is one more draw and ground check.
        self.assertEqual(drawjob.estimate(self.root, "stub", ["load", "draw", "ground", "redraw"]), 17.0)
        self.assertEqual(drawjob.stage_estimates({"attempt": 3.0, "ground": 0.5}, ["draw", "redraw"], redraws=2),
                         {"draw": 3.0, "redraw": 7.0})

    def test_estimate_grows_with_a_redraw(self) -> None:
        optional._STUB_LOADED = True
        drawjob.write_times(self.root, "stub", {"attempt": 3.0, "ground": 0.5})
        seen: list[tuple] = []
        with mock.patch.dict("os.environ", {"FLAT2D_STUB_FLAWS": "1"}):
            job = self.start({"kind": "flat2d", "stem": "crate", "prompt": "a wooden crate"})
            self.assertEqual(self.factory.status(job["job_id"])["estimate_s"], 3.5)
            deadline = time.time() + 15
            while time.time() < deadline:
                st = self.factory.status(job["job_id"])
                seen.append((st["estimate_s"], dict(st.get("stage_est") or {})))
                if st["state"] != "running":
                    break
                time.sleep(0.02)
        self.assertIn((7.0, {"draw": 3.0, "ground": 0.5, "redraw": 3.5}), seen)

    def test_a_new_job_never_shows_the_last_draw_log(self) -> None:
        folder = self.root / "out" / "lamp"
        folder.mkdir(parents=True)
        (folder / "draw.log").write_text("STAGES draw ground\nSTAGE ground Check the ground\nDone.\n")
        with mock.patch.object(drawjob, "run", lambda *a, **k: None):
            job = self.factory.start_flat({"kind": "flat2d", "stem": "lamp", "prompt": "a lamp"})
            st = self.factory.status(job["job_id"])
        self.assertNotIn("Done.", st.get("log_tail") or "")
        self.assertIn("Done.", (folder / "draw.prev.log").read_text())
        with self.factory._lock:
            self.factory._job["state"] = "ok"

    def test_views_reaches_the_draw_payload(self) -> None:
        seen: list[dict] = []
        with mock.patch.object(drawjob, "run", lambda factory, spec, payload, **k: seen.append(payload)):
            self.factory.start_flat({"kind": "flat2d", "stem": "lamp", "prompt": "a lamp", "views": 1,
                                     "intent": {"output": "ps1"}})
            with self.factory._lock:
                self.factory._job["state"] = "ok"
            with self.assertRaises(ValueError):
                self.factory.start_flat({"kind": "flat2d", "stem": "lamp", "prompt": "a lamp", "views": 3})
        self.assertEqual(seen[0]["views"], 1)
        self.assertFalse(optional.wants_views(seen[0]["intent"], seen[0]["views"]))

    def test_a_photo_reference_reaches_the_draw_payload(self) -> None:
        from PIL import Image

        (self.factory.root / "in").mkdir(exist_ok=True)
        Image.new("RGB", (64, 48), (200, 10, 10)).save(self.factory.root / "in" / "cow.jpg")
        (self.factory.root / "in" / "notes.png").write_text("not a picture")
        seen: list[tuple[dict, dict]] = []
        with mock.patch.object(drawjob, "run", lambda factory, spec, payload, **k: seen.append((spec, payload))):
            # No words needed with a photo: the vision model names it.
            self.factory.start_flat({"kind": "flat2d", "stem": "cow", "prompt": "", "reference": "in/cow.jpg",
                                     "intent": {"output": "ps1"}})
            with self.factory._lock:
                self.factory._job["state"] = "ok"
            for bad, err in ((dict(reference="../cow.jpg"), ValueError), (dict(reference="in/notes.png"), ValueError),
                             (dict(reference="in/gone.jpg"), FileNotFoundError),
                             (dict(reference="in/cow.jpg", intent={"output": "sprite"}), ValueError),
                             (dict(reference="in/cow.jpg", views=1), ValueError),
                             (dict(reference="in/cow.jpg", kind="ideas"), ValueError)):
                body = {"kind": "flat2d", "stem": "cow", "prompt": "a cow", "intent": {"output": "ps1"}, **bad}
                with self.assertRaises(err, msg=bad):
                    self.factory.start_flat(body)
        spec, payload = seen[0]
        self.assertEqual(payload["reference"], str((self.factory.root / "in" / "cow.jpg").resolve()))
        self.assertEqual(spec["reference"], "in/cow.jpg")
        self.assertEqual(spec["backend"], "stub")  # sheet.edit_backend(): the photo draw keeps its own times
        self.assertEqual(payload["prompt"], "")

    def test_2d_writes_wait_for_the_job_on_that_asset(self) -> None:
        with mock.patch.dict("os.environ", {"FLAT2D_STUB_STEP": "0.3"}):
            job = self.start({"kind": "flat2d", "stem": "lamp", "prompt": "a lamp"})
            for action in ("steps", "split", "save", "use", "sound"):
                status, body = http_json(self.base + f"/2d/{action}", method="POST", data={"stem": "lamp"})
                self.assertEqual(status, 409, (action, body))
                self.assertTrue(body["busy"])
            status, body = http_json(self.base + "/2d/split", method="POST", data={"stem": "other"})
            self.assertNotEqual(status, 409, body)
            self.factory.cancel(job["job_id"])
            self.wait(job["job_id"])

    def test_a_plain_draw_holds_off_generate_and_rig(self) -> None:
        (self.root / "out" / "radio").mkdir()
        real_jpeg(self.root / "in" / "radio.jpg")  # a real picture: /generate refuses anything else
        self.factory.sync_draw(True)
        try:
            status, body = http_json(self.base + "/generate", method="POST", data={"image": "in/radio.jpg"})
            self.assertEqual(status, 409, body)
            status, body = http_json(self.base + "/rig/radio", method="POST", data={})
            self.assertEqual(status, 409, body)
        finally:
            self.factory.sync_draw(False)

    def test_an_idea_drawn_again_starts_its_step_bar_over(self) -> None:
        log = "STAGES idea1 idea2\nSTAGE idea1 Picture 1\nStep 8 of 8 [100%]\nChecking the ground.\n"
        log += "Ground is cloudy. Drawing again on a new seed (2 of 3). [  0%]"
        self.assertEqual(factory_http.parse_progress(log)["sub"], 0.0)
        log += "\nAgain, ground was cloudy · Step 2 of 8 [ 25%]"
        self.assertEqual(factory_http.parse_progress(log)["sub"], 0.25)

    def test_stub_backend_is_opt_in(self) -> None:
        from flat2d.sheet import still_backend

        self.assertEqual(still_backend(), "stub")
        with mock.patch("flat2d.sheet.mflux_ready", return_value=False):
            self.assertEqual(still_backend(), "stub")
        with mock.patch.dict("os.environ", {"FLAT2D_STILL": ""}), mock.patch("flat2d.sheet.mflux_ready", return_value=False):
            self.assertEqual(still_backend(), "")


class RelayTests(unittest.TestCase):
    """The mflux callback, without mflux: fake config and latents."""

    class Hook:
        wants_preview = True

        def __init__(self, cancel=False):
            self.calls = []
            self.cancel = cancel

        def cancelled(self):
            return self.cancel

        def step(self, k, n):
            self.calls.append(("step", k, n))

        def loading(self, on):
            self.calls.append(("loading", on))

        def preview(self, img):
            self.calls.append(("preview",))

    def config(self):
        return mock.Mock(num_inference_steps=8, height=1024, width=1024)

    def test_relay_reports_steps_and_ends_loading_after_step_one(self) -> None:
        relay = optional._Relay(model=None, preview=False)
        relay.loading = True
        hook = self.Hook()
        cfg = self.config()
        with optional.drawing(hook):
            relay.call_before_loop(1, "p", "lat0", cfg)
            for t in range(8):
                relay.call_in_loop(t, 1, "p", f"lat{t + 1}", cfg, None)
            relay.call_after_loop(1, "p", "lat8", cfg)
        self.assertEqual(hook.calls[0], ("loading", False))
        self.assertEqual([c[1] for c in hook.calls if c[0] == "step"], list(range(1, 9)))
        self.assertIsNone(optional.DRAW)
        relay.call_in_loop(0, 1, "p", "x", cfg, None)  # no hook: nothing happens

    def test_cancel_while_loading_stops_once_the_weights_are_in(self) -> None:
        relay = optional._Relay(model=None, preview=False)
        relay.loading = True
        hook = self.Hook(cancel=True)
        cfg = self.config()
        with optional.drawing(hook):
            relay.call_before_loop(1, "p", "lat0", cfg)
            relay.call_in_loop(0, 1, "p", "lat1", cfg, None)  # the weights load after this: no stop yet
            self.assertFalse(relay.warm)
            with self.assertRaises(optional.DrawCancelled):
                relay.call_in_loop(1, 1, "p", "lat2", cfg, None)
        self.assertTrue(relay.warm)
        self.assertFalse(relay.loading)
        self.assertEqual(hook.calls, [("loading", False)])

    def test_model_loaded_means_the_weights_ran_once(self) -> None:
        relay = optional._Relay(model=None, preview=False)
        with mock.patch.object(optional, "still_backend", return_value="qwen-image"), \
                mock.patch.object(optional, "_QWEN", object()), mock.patch.object(optional, "_QWEN_RELAY", relay):
            self.assertFalse(optional.model_loaded())  # built, never stepped (a load that was stopped)
            hook = self.Hook()
            with optional.drawing(hook):
                optional._cold(relay)
            self.assertTrue(relay.loading)  # the next draw shows the load again
            self.assertEqual(hook.calls, [("loading", True)])
            relay.call_in_loop(1, 1, "p", "x", self.config(), None)
            self.assertTrue(optional.model_loaded())
            with optional.drawing(self.Hook()):
                optional._cold(relay)
            self.assertFalse(relay.loading)

    def test_relay_raises_when_cancelled(self) -> None:
        relay = optional._Relay(model=None, preview=False)
        with optional.drawing(self.Hook(cancel=True)):
            with self.assertRaises(optional.DrawCancelled):
                relay.call_in_loop(2, 1, "p", "x", self.config(), None)

    def test_a_failed_preview_never_breaks_the_draw(self) -> None:
        relay = optional._Relay(model=None, preview=True)
        hook = self.Hook()
        with optional.drawing(hook):
            relay.call_before_loop(1, "p", "lat0", self.config())
            relay.call_in_loop(1, 1, "p", "lat1", self.config(), None)
        self.assertEqual(hook.calls, [("step", 1, 8)])


if __name__ == "__main__":
    unittest.main()
