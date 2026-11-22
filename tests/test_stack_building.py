#!/usr/bin/env python3
"""A building as a sprite stack from three drawn walls (scripts/stack_building.py) and its job (POST /2d/building,
flat2d/building.py). No image model runs: the two model calls are patched to return wall pictures, and the stub
backend stands behind them in case a call is missed."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import factory_http  # noqa: E402
import stack_building as SB  # noqa: E402
from flat2d import building, drawjob, optional  # noqa: E402
from test_factory_http import http_json  # noqa: E402

# The walls the method was proven on (2026-11-21). Too large to commit (~2.7 MB a set): skipped when absent.
EXAMPLES = Path("/Users/stephenhorton/projects/ai-asset-pipeline/out/_experiments")
BRICK, CAFE = EXAMPLES / "walls2", EXAMPLES / "walls3"
SMALL = 48  # cells across the synthetic wall: quick


def _have(folder: Path) -> bool:
    return all((folder / f"{n}.png").is_file() for n in SB.WALLS)


def synthetic_walls(awning: bool = True, px: int = 320) -> dict[str, Image.Image]:
    """A plain wall (textured boards, a cornice, four windows a floor, a dark outline) on magenta, and the same
    wall with a shop front and a striped awning (front) or with a door (back), as an edit model would add them."""
    rng = np.random.default_rng(7)
    a = np.zeros((px, px, 3), np.uint8)
    a[:] = (255, 0, 255)
    x0, x1, y0, y1 = 40, px - 40, 60, px - 40
    noise = rng.normal(0, 14, (y1 - y0, (x1 - x0) // 2, 1))
    noise = np.concatenate([noise, noise[:, ::-1]], axis=1)  # the plain wall is symmetric, as the method assumes
    wall = np.array([90, 140, 200], float) + noise  # boards: many shades
    a[y0:y1, x0:x1] = np.clip(wall, 0, 255).astype(np.uint8)
    a[y0 : y0 + 14, x0:x1] = (235, 235, 225)  # the cornice
    for floor in (0, 1):
        for i in range(4):
            wx = (x0 + 16, x0 + 70, x1 - 104, x1 - 50)[i]  # mirrored about the middle
            wy = y0 + 40 + floor * 110
            a[wy : wy + 50, wx : wx + 34] = (30, 50, 90)
    a[y0:y1, x0 : x0 + 3] = a[y0:y1, x1 - 3 : x1] = (20, 20, 30)  # outline
    a[y0 : y0 + 3, x0:x1] = a[y1 - 3 : y1, x0:x1] = (20, 20, 30)
    base = a.copy()
    back = a.copy()
    mid = (x0 + x1) // 2
    back[y1 - 80 : y1 - 3, mid - 18 : mid + 18] = (240, 240, 240)  # a door
    front = a.copy()
    # Two shop windows and a door, with plain wall between them (an awning is what differs across the whole width).
    front[y1 - 80 : y1 - 10, x0 + 30 : x0 + 64] = (150, 200, 220)
    front[y1 - 80 : y1 - 10, x1 - 64 : x1 - 30] = (150, 200, 220)
    front[y1 - 80 : y1 - 3, mid - 16 : mid + 16] = (160, 90, 40)
    if awning:
        for x in range(x0 - 10, x1 + 10):  # the awning hangs past the corners; the box trims it
            front[y1 - 104 : y1 - 82, x] = (220, 30, 50) if (x // 12) % 2 else (250, 250, 250)
    return {"base": Image.fromarray(base), "front": Image.fromarray(front), "back": Image.fromarray(back)}


def save_walls(folder: Path, walls: dict[str, Image.Image]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, im in walls.items():
        im.save(folder / f"{name}.png")
    return folder


def sticks_out(shape: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per layer: is anything in front of that layer's front wall, and is anything behind its back wall."""
    solid = shape["solid"]
    ys = np.arange(solid.shape[1])[None, :]
    occupied = solid.any(axis=2)  # [H, L]
    front = (occupied & (ys > shape["front_face"][:, None])).any(axis=1)
    back = (occupied & (ys < shape["back_face"][:, None])).any(axis=1)
    return front, back


class BuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "out").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self, stem: str, walls, look: str = "crisp", size: int = SMALL, awning: str = "auto") -> dict:
        return SB.build(self.root, stem, walls, look=look, size=size, awning=awning, line=lambda t: None)

    def assert_contract(self, stem: str, look: str, out: dict) -> dict:
        asset = self.root / "out" / stem
        meta = json.loads((asset / "stack" / "stack.json").read_text())
        self.assertEqual((meta["maker"], meta["look"], meta["kind"], meta["source"]), ("walls", look, "stack", "flat/pixel"))
        res, layers = meta["resolution"], meta["layers"]
        self.assertEqual(Image.open(asset / "stack" / "sheet.png").size, (res * layers, res))
        self.assertEqual(layers, out["layers"])
        self.assertEqual(res, max(out["size"]["width"], out["size"]["length"]))
        self.assertEqual(meta["defaults"]["texels"], res)
        self.assertEqual((meta["defaults"]["contrast"], meta["defaults"]["saturation"]), (1.0, 1.0))
        views = json.loads((asset / "flat" / "views.json").read_text())
        self.assertEqual((views["layout"], views["canonical"]), ("walls", "front"))
        for name in SB.VIEWS:
            self.assertTrue((asset / "flat" / "views" / f"{name}.png").is_file())
            self.assertTrue((asset / "flat" / "pixel" / f"{name}.png").is_file())
        self.assertEqual(json.loads((asset / "flat" / "pixel" / "pixel.json").read_text())["look"], look)
        return meta

    def assert_awning_only_at_front(self, stem: str, look: str) -> None:
        shape = SB.build_shape(SB.read_pixel(self.root, stem), look=look)
        self.assertTrue(shape["awning"]["found"])
        h0, h1 = shape["awning"]["layers"]
        front, back = sticks_out(shape)
        self.assertTrue(front[h0])  # the awning's bottom edge sticks out furthest
        self.assertFalse(front[:h0].any() or front[h1 + 1 :].any())
        self.assertFalse(back.any())

    def test_synthetic_walls_make_a_stack_in_both_looks(self) -> None:
        walls = save_walls(self.root / "walls", synthetic_walls())
        files = {n: walls / f"{n}.png" for n in SB.WALLS}
        crisp = self.build("crisp", files, "crisp")
        smooth = self.build("smooth", files, "smooth")
        self.assert_contract("crisp", "crisp", crisp)
        self.assert_contract("smooth", "smooth", smooth)
        self.assertEqual(crisp["size"], smooth["size"])
        self.assertEqual(crisp["size"]["width"], SMALL)
        self.assertGreater(smooth["colors"], 5 * crisp["colors"])
        self.assertLessEqual(crisp["colors"], SB.COLOURS + 2)
        for stem, look in (("crisp", "crisp"), ("smooth", "smooth")):
            self.assert_awning_only_at_front(stem, look)
        self.assertEqual(crisp["flaws"], [])

    def test_versions_are_saved_and_the_stack_is_made_from_the_concept(self) -> None:
        import flow

        walls = save_walls(self.root / "walls", synthetic_walls())
        self.build("b", {n: walls / f"{n}.png" for n in SB.WALLS})
        data = flow.read_manifest(self.root / "out" / "b")
        for stage in ("concept", "flat", "stack"):
            self.assertTrue(data["stages"][stage]["takes"], stage)
        self.assertIn("flat/walls/front.png", data["stages"]["concept"]["takes"][-1]["files"])
        take = data["stages"]["stack"]["takes"][-1]
        self.assertEqual(set(take["from"]), {"concept"})

    def test_no_awning_asked_for_and_none_drawn_builds_without_a_wedge(self) -> None:
        walls = synthetic_walls(awning=False)
        out = self.build("plain", walls)
        self.assertFalse(out["awning"]["found"])
        self.assertEqual(out["size"]["length"], out["size"]["width"])  # the plain wall's width both ways

    def test_an_awning_asked_for_and_not_found_fails_and_writes_no_stack(self) -> None:
        with self.assertRaises(SB.WallsError) as e:
            self.build("plain", synthetic_walls(awning=False), awning="yes")
        self.assertIn("awning", str(e.exception))
        self.assertFalse((self.root / "out" / "plain" / "stack").exists())

    def test_no_back_picture_uses_the_plain_wall(self) -> None:
        walls = synthetic_walls()
        walls.pop("back")
        out = self.build("noback", walls)
        self.assertIn("no back feature: the back is the plain wall", out["views"]["notes"])
        self.assertFalse((self.root / "out" / "noback" / "flat" / "walls" / "back.png").exists())

    def test_walls_of_different_sizes_are_refused(self) -> None:
        walls = synthetic_walls()
        walls["front"] = walls["front"].resize((200, 200))
        with self.assertRaises(SB.WallsError):
            self.build("odd", walls)

    def test_a_front_that_redraws_the_upper_floors_is_a_flaw(self) -> None:
        walls = synthetic_walls()
        front = np.asarray(walls["front"]).copy()
        front[70:150, 50:270] = (200, 120, 40)  # the upper floor painted over
        walls["front"] = Image.fromarray(front)
        out = self.build("drift", walls, awning="no")
        self.assertIn("upper:front", out["flaws"])
        self.assertFalse(out["views"]["consistent"])

    def test_unrelated_files_are_kept(self) -> None:
        keep = self.root / "out" / "k" / "flat" / "views" / "lifted.png"
        keep.parent.mkdir(parents=True)
        keep.write_bytes(b"x")
        self.build("k", synthetic_walls())
        self.assertTrue(keep.is_file())

    def test_render_turned_writes_a_picture(self) -> None:
        self.build("r", synthetic_walls())
        path = SB.render_turned(SB.read_sheet_slices(self.root / "out" / "r" / "stack"), self.root / "r.png", scale=1)
        self.assertGreater(Image.open(path).size[0], 0)


@unittest.skipUnless(_have(BRICK) and _have(CAFE), "the example walls (out/_experiments/walls2, walls3) are not here")
class ExampleTests(unittest.TestCase):
    """The two buildings the owner approved, at the size they were made at."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        (cls.root / "out").mkdir()
        cls.out = {}
        for name, src in (("brick", BRICK), ("cafe", CAFE)):
            walls = cls.root / "walls" / name
            walls.mkdir(parents=True)
            for n in SB.WALLS:
                shutil.copy2(src / f"{n}.png", walls / f"{n}.png")
            for look in SB.LOOKS:
                cls.out[(name, look)] = SB.build(cls.root, f"{name}-{look}", {n: walls / f"{n}.png" for n in SB.WALLS},
                                                 look=look, size=SB.SIZE, awning="yes", line=lambda t: None)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_sizes_awnings_and_colours(self) -> None:
        # The cafe is the hand run's size exactly; the brick one layer taller (198 by hand).
        for name, layers in (("brick", 199), ("cafe", 147)):
            crisp, smooth = self.out[(name, "crisp")], self.out[(name, "smooth")]
            self.assertEqual((crisp["layers"], smooth["layers"]), (layers, layers))
            self.assertEqual(crisp["size"]["width"], SB.SIZE)
            self.assertEqual(crisp["size"]["length"], SB.SIZE + round(SB.AWNING_OUT_SHARE * SB.SIZE))
            for look in SB.LOOKS:
                aw = self.out[(name, look)]["awning"]
                self.assertTrue(aw["found"])
                self.assertTrue(40 <= aw["layers"][0] < aw["layers"][1] <= 70, aw)  # the ground floor's top
            self.assertGreater(smooth["colors"], 100 * crisp["colors"])
            self.assertEqual(crisp["flaws"], [])

    def test_the_awning_sticks_out_only_at_the_front(self) -> None:
        for name in ("brick", "cafe"):
            for look in SB.LOOKS:
                shape = SB.build_shape(SB.read_pixel(self.root, f"{name}-{look}"), look=look)
                h0, h1 = shape["awning"]["layers"]
                front, back = sticks_out(shape)
                self.assertTrue(front[h0 : h0 + (h1 - h0) // 2].all())  # the wedge's wide lower half
                self.assertFalse(front[:h0].any() or front[h1 + 1 :].any())
                self.assertFalse(back.any())


class PromptTests(unittest.TestCase):
    def test_prompts_and_request(self) -> None:
        p = building.prompts("a red brick wall, three floors", "Add a shop front and a green awning", "")
        self.assertTrue(p["base"].startswith(building.WALL_LEAD + "a red brick wall, three floors."))
        self.assertIn("A plain wall: no shop front", p["base"])
        self.assertTrue(p["front"].startswith("Add a shop front and a green awning. Keep everything else"))
        self.assertNotIn("back", p)
        req = building.parse({"stem": "shop", "wall": "w", "front": "an awning"})
        self.assertEqual((req["look"], req["size"], req["seeds"], req["awning"]), ("crisp", 160, building.SEEDS, "yes"))
        req = building.parse({"stem": "shop", "wall": "w", "front": "a door", "look": "smooth", "size": 96, "seed": 5})
        self.assertEqual((req["look"], req["size"], req["seeds"], req["awning"]),
                         ("smooth", 96, {"base": 5, "front": 16, "back": 26}, "auto"))
        for bad in ({"front": "x"}, {"wall": "x"}, {"wall": "x", "front": "y", "look": "soft"},
                    {"wall": "x", "front": "y", "size": 4}, {"wall": "x", "front": "y", "seed": "z"},
                    {"wall": 3, "front": "y"}):
            with self.assertRaises(ValueError, msg=bad):
                building.parse({"stem": "shop", **bad})

    def test_stage_names_never_collide_with_rig_or_generate_labels(self) -> None:
        theirs = {"isolate", "shape", "reduce", "paint", "retexture", "psx", "lods", "inspect", "stack", "preview",
                  "classify", "rest", "pose", "skin", "clean", "clips", "check", "fbx"}
        ours = set(building.plan_for(True, False, True))
        self.assertEqual(ours & theirs, set())
        for key in ours:
            self.assertIn(key, factory_http.STAGE_LABELS)


class RouteTests(unittest.TestCase):
    """POST /2d/building with the two model calls patched to return the synthetic walls."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "in").mkdir()
        (self.root / "out").mkdir()
        self.env = mock.patch.dict("os.environ", {"FLAT2D_STILL": "stub", "FLAT2D_STUB_STEP": "0.01",
                                                  "FLAT2D_STUB_LOAD": "0.01", "FLAT2D_PREVIEW": "on"})
        self.env.start()
        optional._STUB_LOADED = False
        self.walls = synthetic_walls()
        self.calls: list[tuple[str, str]] = []

        def still(prompt, seed=1, width=1024, height=1024):
            self.calls.append(("still", prompt, seed))
            return self.walls["base"]

        def edit(prompt, reference, seed=1, width=1024, height=1024):
            self.calls.append(("edit", prompt, seed))
            return self.walls["front"] if "awning" in prompt else self.walls["back"]

        self.patches = [mock.patch.object(optional, "still_image", still), mock.patch.object(optional, "edit_image", edit)]
        for p in self.patches:
            p.start()
        self.factory = factory_http.Factory(self.root)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), factory_http.make_handler(self.factory))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        job = self.factory._snapshot()
        if job and job.get("state") == "running":
            self.factory.cancel(job["job_id"])
        self.httpd.shutdown()
        self.httpd.server_close()
        for p in self.patches:
            p.stop()
        self.env.stop()
        self._tmp.cleanup()

    def wait(self, job_id: str, timeout: float = 60) -> dict:
        deadline = time.time() + timeout
        last: dict = {}
        while time.time() < deadline:
            _, last = http_json(self.base + f"/status/{job_id}")
            if last.get("state") != "running":
                return last
            time.sleep(0.05)
        self.fail(f"job still running: {last}")

    BODY = {"stem": "shop", "wall": "a plain wall of blue boards", "front": "Add a shop front and a striped awning",
            "back": "Add a white door", "look": "smooth", "size": SMALL}

    def test_the_route_draws_the_walls_and_builds_the_stack(self) -> None:
        status, job = http_json(self.base + "/2d/building", method="POST", data=self.BODY)
        self.assertEqual(status, 202, job)
        self.assertEqual(job["kind"], "building")
        st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "ok", st)
        self.assertEqual([c[0] for c in self.calls], ["still", "edit", "edit"])  # the plain wall first, then edits
        self.assertEqual([c[2] for c in self.calls], [12, 23, 33])
        asset = self.root / "out" / "shop"
        meta = json.loads((asset / "stack" / "stack.json").read_text())
        self.assertEqual((meta["maker"], meta["look"]), ("walls", "smooth"))
        self.assertEqual(st["result"]["stack"]["layers"], meta["layers"])
        self.assertTrue(st["result"]["awning"]["found"])
        keys = [s["key"] for s in st["progress"]["stages"]]
        self.assertEqual(keys[-5:], ["wall", "front_add", "back_add", "walls_fit", "carve"])
        self.assertTrue(set(st["stage_times"]) >= {"wall", "front_add", "back_add", "walls_fit", "carve"})
        times = json.loads((self.root / "out" / "draw-times.json").read_text())
        self.assertIn("carve", times[building.time_key("stub", "stub")])
        self.assertNotIn("stub", times)  # the plain draw's row is never written by a building
        views = json.loads((asset / "flat" / "views.json").read_text())
        self.assertEqual(views["sheet"]["seeds"], building.SEEDS)
        self.assertIn("awning", views["sheet"]["prompts"]["front"])
        self.assertFalse((asset / "flat" / ".walls-draw").exists())
        self.assertTrue((asset / "flat" / "walls" / "back.png").is_file())

    def test_bad_requests_and_a_busy_slot(self) -> None:
        for bad in ({"stem": "shop"}, {**self.BODY, "look": "soft"}, {**self.BODY, "size": 1}):
            status, payload = http_json(self.base + "/2d/building", method="POST", data=bad)
            self.assertEqual(status, 400, payload)
        with self.factory._lock:
            self.factory._job = {"job_id": "x", "state": "running"}
        status, payload = http_json(self.base + "/2d/building", method="POST", data=self.BODY)
        self.assertEqual((status, payload.get("busy")), (409, True))
        with self.factory._lock:
            self.factory._job = None

    def test_an_awning_asked_for_and_not_drawn_is_an_error_with_no_stack(self) -> None:
        self.walls = synthetic_walls(awning=False)
        status, job = http_json(self.base + "/2d/building", method="POST", data=self.BODY)
        self.assertEqual(status, 202, job)
        # The patched edit returns the front only for an awning prompt; this front has no awning painted.
        st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "error", st)
        self.assertIn("awning", st["error"])
        self.assertFalse((self.root / "out" / "shop" / "stack").exists())

    def test_cancel_stops_the_job(self) -> None:
        gate = threading.Event()

        def slow_still(prompt, seed=1, width=1024, height=1024):
            gate.wait(5)
            return self.walls["base"]

        with mock.patch.object(optional, "still_image", slow_still):
            status, job = http_json(self.base + "/2d/building", method="POST", data=self.BODY)
            self.assertEqual(status, 202, job)
            http_json(self.base + f"/cancel/{job['job_id']}", method="POST", data={})
            gate.set()
            st = self.wait(job["job_id"])
        self.assertEqual(st["state"], "cancelled", st)
        self.assertFalse((self.root / "out" / "shop" / "stack").exists())


if __name__ == "__main__":
    unittest.main()
