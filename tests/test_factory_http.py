#!/usr/bin/env python3
"""Drive the shipped factory HTTP adapter (health / inspect / generate lock / GLB)."""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import factory_env  # noqa: E402
import factory_http  # noqa: E402
import inspect_glb  # noqa: E402
import trimesh  # noqa: E402


def write_triangle_glb(path: Path) -> None:
    mesh = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def http_json(url: str, method: str = "GET", data: dict | None = None, timeout: float = 5) -> tuple[int, dict]:
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {"ok": False, "error": raw.decode("utf-8", errors="replace")}
        return e.code, payload


def http_bytes(url: str, timeout: float = 5) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


class FactoryEnvTests(unittest.TestCase):
    def test_factory_python_is_real_exe(self) -> None:
        py = factory_env.factory_python(ROOT)
        self.assertTrue(py.is_file(), f"missing factory python {py}")
        self.assertTrue(py.name.lower().startswith("python"))

    def test_generate_argv_runs_script_directly(self) -> None:
        image = ROOT / "in" / "radio-concept.jpg"
        argv = factory_env.generate_argv(
            ROOT,
            image=image,
            quality="stack",
            stem="radio",
            poly="medium",
        )
        self.assertTrue(argv[0].endswith("generate.sh"))

    def test_poly_aliases_map_to_canonical(self) -> None:
        self.assertEqual(factory_env.normalize_poly("high"), "hero")
        self.assertEqual(factory_env.normalize_poly("medium"), "lookdev")
        self.assertEqual(factory_env.normalize_poly("low"), "proxy")
        self.assertEqual(factory_env.normalize_poly("hero"), "hero")
        self.assertEqual(factory_env.poly_label("high"), "High")
        self.assertEqual(factory_env.poly_label("medium"), "Medium")
        self.assertEqual(factory_env.poly_label("low"), "Low")

    def test_generate_argv_poly_and_refs(self) -> None:
        image = ROOT / "in" / "radio-concept.jpg"
        extra = ROOT / "in" / "radio-side.jpg"
        argv = factory_env.generate_argv(
            ROOT,
            image=image,
            quality="draft",
            stem="radio",
            poly="high",
            refs=[extra],
            prompt="a radio",
        )
        self.assertIn("--poly", argv)
        self.assertIn("high", argv)
        self.assertIn("--ref", argv)
        self.assertIn(str(extra), argv)
        self.assertIn("--prompt", argv)
        self.assertIn("a radio", argv)

    def test_generate_argv_hits_shipped_script(self) -> None:
        image = ROOT / "in" / "radio-concept.jpg"
        argv = factory_env.generate_argv(
            ROOT,
            image=image,
            quality="draft",
            stem="radio",
            poly="proxy",
        )
        joined = " ".join(argv)
        self.assertIn("generate.sh", joined)
        self.assertIn("--quality", argv)
        self.assertIn("draft", argv)

    def test_auto_stem_from_prompt_filename_and_fallback(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "out" / "radio").mkdir(parents=True)
            prompt = "A weathered handheld radio, scuffed paint"
            self.assertEqual(
                factory_env.auto_stem(root, prompt, []),
                "weathered-handheld-radio-scuffed-paint",
            )
            self.assertEqual(
                factory_env.slug_stem(
                    "A weathered handheld radio, scuffed paint, palm-isle light"
                ),
                "weathered-handheld-radio-scuffed-paint",
            )
            self.assertTrue(factory_env.is_generic_stem("IMG_1234"))
            self.assertTrue(factory_env.is_generic_stem("screenshot-2"))
            self.assertFalse(factory_env.is_generic_stem("weathered-radio"))
            generic = root / "in" / "IMG_0042.jpg"
            generic.parent.mkdir(parents=True, exist_ok=True)
            generic.write_bytes(b"not-a-real-jpeg")
            self.assertEqual(factory_env.auto_stem(root, "", [generic]), "asset-1")
            (root / "out" / "asset-1").mkdir()
            self.assertEqual(factory_env.auto_stem(root, "", [generic]), "asset-2")
            named = root / "in" / "coral-lantern.png"
            named.write_bytes(b"not-a-real-png")
            self.assertEqual(factory_env.auto_stem(root, "", [named]), "coral-lantern")
            self.assertEqual(
                factory_env.auto_stem(root, "radio", []),
                "radio-2",
            )
            still = root / "out" / "a-top-down-view-of-a-spider-mech" / "flat" / "source.png"
            still.parent.mkdir(parents=True)
            still.write_bytes(b"\x89PNG\r\n")
            self.assertTrue(factory_env.is_generic_stem("source"))
            self.assertEqual(factory_env.auto_stem(root, "", [still]), "a-top-down-view-of-a-spider-mech-2")
            loose = root / "in" / "source.png"
            loose.write_bytes(b"\x89PNG\r\n")
            self.assertEqual(factory_env.auto_stem(root, "", [loose]), "asset-2")

    def test_caption_from_png_text_and_jpeg_comment(self) -> None:
        import struct
        import tempfile
        import zlib

        def png_chunk(typ: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(typ + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", crc)

        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "upload.png"
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
            raw = zlib.compress(b"\x00\x00")
            text = b"Description\x00brass lantern with coral"
            png.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + png_chunk(b"IHDR", ihdr)
                + png_chunk(b"tEXt", text)
                + png_chunk(b"IDAT", raw)
                + png_chunk(b"IEND", b"")
            )
            self.assertEqual(factory_env.caption_from_image(png), "brass-lantern-with-coral")
            jpg = Path(td) / "IMG_99.jpg"
            comment = b"wooden crate with stencil"
            jpg.write_bytes(
                b"\xff\xd8\xff\xfe"
                + (len(comment) + 2).to_bytes(2, "big")
                + comment
                + b"\xff\xd9"
            )
            self.assertEqual(factory_env.caption_from_image(jpg), "wooden-crate-with-stencil")

    def test_generate_argv_accepts_stack_quality(self) -> None:
        image = ROOT / "in" / "radio-concept.jpg"
        argv = factory_env.generate_argv(
            ROOT,
            image=image,
            quality="stack",
            stem="radio",
            poly="proxy",
        )
        self.assertIn("stack", argv)

    def test_inspect_glb_function_on_triangle(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            glb = Path(td) / "tri.glb"
            write_triangle_glb(glb)
            report = inspect_glb.inspect(str(glb))
            self.assertTrue(report["ok"])
            self.assertGreater(report["vertices"], 0)
            self.assertGreater(report["faces"], 0)
            self.assertGreaterEqual(report["bytes"], 64)


class FactoryHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "in").mkdir()
        write_triangle_glb(self.root / "out" / "radio" / "mesh.glb")
        (self.root / "in" / "radio-concept.jpg").write_bytes(b"not-a-real-jpeg")
        self.factory = factory_http.Factory(self.root)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), factory_http.make_handler(self.factory))
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self._thr = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thr.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._tmp.cleanup()

    def test_health_json_ok(self) -> None:
        status, payload = http_json(self.base + "/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        self.assertEqual(payload.get("service"), "ai-asset-pipeline")
        self.assertTrue(payload.get("python"))
        self.assertEqual(payload.get("backend"), "mlx")
        feats = payload.get("features") or {}
        self.assertIn("shape_mv", feats)
        self.assertIn("unirig", feats)
        self.assertFalse(feats.get("unirig"))
        self.assertEqual(feats.get("shape"), "mini-mlx")
        self.assertFalse(feats.get("shape_mv"))
        self.assertTrue(feats.get("paint_pbr"))

    def test_factory_features_mlx_never_claims_2mv(self) -> None:
        feats = factory_http.factory_features(self.root)
        self.assertEqual(feats["shape"], "mini-mlx")
        self.assertFalse(feats["shape_mv"])
        self.assertFalse(feats["unirig"])

    def test_inspect_radio_ok(self) -> None:
        status, payload = http_json(self.base + "/inspect/radio")
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        self.assertGreater(payload.get("vertices") or 0, 0)
        self.assertGreater(payload.get("faces") or 0, 0)
        self.assertEqual(payload.get("stem"), "radio")

    def test_glb_download_magic(self) -> None:
        status, data = http_bytes(self.base + "/glb/radio")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(data), 64)
        self.assertEqual(data[:4], b"glTF")

    def test_stack_404_without_sidecar(self) -> None:
        status, payload = http_json(self.base + "/stack/radio")
        self.assertEqual(status, 404)
        self.assertFalse(payload.get("ok"))

    def test_upload_json_base64(self) -> None:
        import base64

        raw = b"\xff\xd8\xffmech"
        payload = json.dumps(
            {"name": "pipeline-mech.jpg", "data": base64.b64encode(raw).decode("ascii")}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/upload",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(status, 201)
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("path"), "in/pipeline-mech.jpg")
        self.assertEqual((self.root / "in" / "pipeline-mech.jpg").read_bytes(), raw)

    def test_upload_then_path_under_in(self) -> None:
        req = urllib.request.Request(
            self.base + "/upload?name=Concept%20Shot.PNG",
            data=b"\x89PNG\r\n",
            method="POST",
            headers={"Content-Type": "image/png"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(status, 201)
        self.assertTrue(payload.get("ok"))
        self.assertTrue(str(payload.get("path") or "").startswith("in/"))
        self.assertTrue((self.root / payload["path"]).is_file())

    def test_list_and_delete_assets(self) -> None:
        status, payload = http_json(self.base + "/assets")
        self.assertEqual(status, 200)
        ids = [a["id"] for a in payload.get("assets") or []]
        self.assertIn("radio", ids)
        status, payload = http_json(self.base + "/assets/radio", method="DELETE")
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("deleted"))
        self.assertFalse((self.root / "out" / "radio").exists())
        status, payload = http_json(self.base + "/assets")
        ids = [a["id"] for a in payload.get("assets") or []]
        self.assertNotIn("radio", ids)

    def test_list_mesh_preview_png(self) -> None:
        folder = self.root / "out" / "spider-mech"
        write_triangle_glb(folder / "mesh.glb")
        (folder / "preview.png").write_bytes(b"\x89PNG\r\n")
        (folder / "input.png").write_bytes(b"\x89PNG\r\n")
        status, payload = http_json(self.base + "/assets")
        self.assertEqual(status, 200)
        row = next(a for a in payload["assets"] if a["id"] == "spider-mech")
        self.assertEqual(row["preview"], "preview.png")
        self.assertEqual(row["input"], "input.png")
        self.assertTrue(row["has_mesh"])

    def test_stack_json_ok(self) -> None:
        p = self.root / "out" / "radio" / "stack" / "stack.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kind": "stack", "layers": 12, "resolution": 160}), encoding="utf-8")
        status, payload = http_json(self.base + "/stack/radio")
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        self.assertEqual(payload.get("layers"), 12)
        self.assertEqual(payload.get("stem"), "radio")

    def test_generate_error_from_missing_module(self) -> None:
        log = self.root / "out" / "broken" / "generate.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "generate: lifting subject via Vision\n"
            "ModuleNotFoundError: No module named 'trimesh'\n",
            encoding="utf-8",
        )
        err = factory_http._generate_error(1, {"ok": False, "error": "no GLB"}, str(log))
        self.assertEqual(err, "missing python module trimesh")

    def test_generate_auto_stem_prompt_and_default_stack(self) -> None:
        started = threading.Event()
        release = threading.Event()
        seen: dict = {}

        def runner(spec: dict) -> int:
            seen.update(spec)
            started.set()
            release.wait(timeout=5)
            write_triangle_glb(self.root / "out" / spec["stem"] / "mesh.glb")
            return 0

        self.factory._runner = runner
        status, first = http_json(
            self.base + "/generate",
            method="POST",
            data={
                "image": "in/radio-concept.jpg",
                "prompt": "A weathered handheld radio",
                "poly": "low",
            },
        )
        self.assertEqual(status, 202)
        self.assertEqual(first.get("stem"), "weathered-handheld-radio")
        self.assertTrue(started.wait(timeout=3))
        self.assertEqual(seen.get("quality"), "stack")
        self.assertEqual(seen.get("stem"), "weathered-handheld-radio")
        release.set()
        deadline = time.time() + 5
        last = first
        while time.time() < deadline:
            status, last = http_json(self.base + f"/status/{first['job_id']}")
            if last.get("state") != "running":
                break
            time.sleep(0.05)
        self.assertEqual(last.get("state"), "ok")

    def test_generate_auto_stem_generic_fallback(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def runner(spec: dict) -> int:
            started.set()
            release.wait(timeout=5)
            write_triangle_glb(self.root / "out" / spec["stem"] / "mesh.glb")
            return 0

        cam = self.root / "in" / "IMG_0042.jpg"
        cam.write_bytes(b"not-a-real-jpeg")
        self.factory._runner = runner
        status, first = http_json(
            self.base + "/generate",
            method="POST",
            data={"image": "in/IMG_0042.jpg", "poly": "low"},
        )
        self.assertEqual(status, 202)
        self.assertEqual(first.get("stem"), "asset-1")
        self.assertTrue(started.wait(timeout=3))
        release.set()
        deadline = time.time() + 5
        last = first
        while time.time() < deadline:
            status, last = http_json(self.base + f"/status/{first['job_id']}")
            if last.get("state") != "running":
                break
            time.sleep(0.05)
        self.assertEqual(last.get("state"), "ok")

    def test_rename_asset_moves_folder(self) -> None:
        status, payload = http_json(
            self.base + "/assets/radio",
            method="PATCH",
            data={"name": "Brass Lantern"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("renamed"))
        self.assertEqual(payload.get("stem"), "brass-lantern")
        self.assertFalse((self.root / "out" / "radio").exists())
        self.assertTrue((self.root / "out" / "brass-lantern" / "mesh.glb").is_file())
        status, again = http_json(
            self.base + "/assets/brass-lantern",
            method="PATCH",
            data={"stem": "brass-lantern"},
        )
        self.assertEqual(status, 200)
        self.assertFalse(again.get("renamed"))

    def test_rename_conflict_and_missing(self) -> None:
        write_triangle_glb(self.root / "out" / "other" / "mesh.glb")
        status, payload = http_json(
            self.base + "/assets/radio",
            method="PATCH",
            data={"stem": "other"},
        )
        self.assertEqual(status, 400)
        self.assertIn("exists", str(payload.get("error") or "").lower())
        status, payload = http_json(
            self.base + "/assets/missing",
            method="PATCH",
            data={"stem": "new-name"},
        )
        self.assertEqual(status, 404)

    def test_generate_unknown_poly_400(self) -> None:
        status, payload = http_json(
            self.base + "/generate",
            method="POST",
            data={"image": "in/radio-concept.jpg", "quality": "draft", "stem": "nope", "poly": "ultra"},
        )
        self.assertEqual(status, 400)
        self.assertIn("poly", str(payload.get("error") or "").lower())

    def test_generate_accepts_high_and_images(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def runner(spec: dict) -> int:
            started.set()
            self.assertEqual(spec.get("poly"), "hero")
            self.assertEqual(spec.get("poly_label"), "High")
            self.assertTrue(spec.get("refs"))
            self.assertEqual(spec.get("prompt"), "handheld radio")
            release.wait(timeout=5)
            write_triangle_glb(self.root / "out" / spec["stem"] / "mesh.glb")
            return 0

        extra = self.root / "in" / "radio-side.jpg"
        extra.write_bytes(b"not-a-real-jpeg")
        self.factory._runner = runner
        status, first = http_json(
            self.base + "/generate",
            method="POST",
            data={
                "image": "in/radio-concept.jpg",
                "images": ["in/radio-side.jpg"],
                "prompt": "handheld radio",
                "quality": "draft",
                "stem": "multi-ref",
                "poly": "high",
            },
        )
        self.assertEqual(status, 202)
        self.assertTrue(started.wait(timeout=3))
        release.set()
        deadline = time.time() + 5
        last = first
        while time.time() < deadline:
            status, last = http_json(self.base + f"/status/{first['job_id']}")
            if last.get("state") != "running":
                break
            time.sleep(0.05)
        self.assertEqual(last.get("state"), "ok")

    def test_generate_missing_image_400(self) -> None:
        status, payload = http_json(
            self.base + "/generate",
            method="POST",
            data={"image": "in/nope.jpg", "quality": "draft", "stem": "nope", "poly": "proxy"},
        )
        self.assertEqual(status, 400)
        self.assertFalse(payload.get("ok"))

    def test_generate_lock_and_status(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def runner(spec: dict) -> int:
            started.set()
            release.wait(timeout=5)
            write_triangle_glb(self.root / "out" / spec["stem"] / "mesh.glb")
            return 0

        self.factory._runner = runner
        status, first = http_json(
            self.base + "/generate",
            method="POST",
            data={
                "image": "in/radio-concept.jpg",
                "quality": "draft",
                "stem": "locked",
                "poly": "proxy",
            },
        )
        self.assertEqual(status, 202)
        self.assertTrue(first.get("ok"))
        self.assertEqual(first.get("state"), "running")
        self.assertTrue(started.wait(timeout=3))

        status, busy = http_json(
            self.base + "/generate",
            method="POST",
            data={
                "image": "in/radio-concept.jpg",
                "quality": "draft",
                "stem": "other",
                "poly": "proxy",
            },
        )
        self.assertEqual(status, 409)
        self.assertTrue(busy.get("busy"))

        job_id = first["job_id"]
        status, running = http_json(self.base + f"/status/{job_id}")
        self.assertEqual(status, 200)
        self.assertEqual(running.get("state"), "running")

        release.set()
        deadline = time.time() + 5
        last = running
        while time.time() < deadline:
            status, last = http_json(self.base + f"/status/{job_id}")
            if last.get("state") != "running":
                break
            time.sleep(0.05)
        self.assertEqual(status, 200)
        self.assertEqual(last.get("state"), "ok")
        self.assertTrue((last.get("inspect") or {}).get("ok"))
        self.assertGreater((last.get("inspect") or {}).get("faces") or 0, 0)


    def test_progress_from_stage_markers(self) -> None:
        log = "\n".join([
            "generate: root=/x quality=textured octree=128",
            "STAGES isolate shape reduce paint retexture psx lods inspect",
            "STAGE isolate Cutting out the subject",
            "STAGE shape Building the shape",
            "STAGE reduce Reducing the mesh",
            "STAGE paint Painting the texture",
            "  [ 50%] Painting (7/15)",
        ])
        p = factory_http.parse_progress(log)
        self.assertEqual([s["key"] for s in p["stages"]][-3:], ["psx", "lods", "inspect"])
        self.assertEqual(p["stage"], "paint")
        self.assertEqual(p["stage_index"], 3)
        self.assertEqual(p["sub"], 0.5)
        self.assertAlmostEqual(p["fraction"], (3 + 0.5) / 8, places=2)
        self.assertEqual(p["stages"][5]["label"], "PS1 low-poly")
        # The config line and a prompt that says "paint" no longer move anything.
        self.assertIsNone(factory_http.parse_progress("generate: prompt=scuffed paint octree=128"))

    def test_intent_is_checked_and_written(self) -> None:
        self.assertIsNone(factory_http.clean_intent({"output": "rocket"}))
        intent = factory_http.clean_intent({"output": "ps1", "facing": "right", "height": 58, "name": "cow", "up": "q"})
        self.assertEqual(intent, {"output": "ps1", "psx": {"facing": "right", "height": "58", "name": "cow"}})
        status, payload = http_json(self.base + "/intent/radio", method="POST", data={"output": "stack"})
        self.assertEqual(status, 200)
        doc = json.loads((self.root / "out" / "radio" / "intent.json").read_text())
        self.assertEqual(doc["output"], "stack")
        status, assets = http_json(self.base + "/assets")
        self.assertEqual(next(a for a in assets["assets"] if a["id"] == "radio")["output"], "stack")
        status, _ = http_json(self.base + "/intent/radio", method="POST", data={"output": "nope"})
        self.assertEqual(status, 400)
        status, _ = http_json(self.base + "/intent/missing", method="POST", data={"output": "ps1"})
        self.assertEqual(status, 404)

    def test_cancel_stops_the_job(self) -> None:
        import subprocess as sp

        started = threading.Event()

        def runner(spec: dict) -> int:
            proc = sp.Popen(["sleep", "30"], start_new_session=True)
            with self.factory._lock:
                self.factory._job["pid"] = proc.pid
            started.set()
            return proc.wait()

        self.factory._runner = runner
        status, first = http_json(self.base + "/generate", method="POST", data={"image": "in/radio-concept.jpg", "stem": "radio"})
        self.assertEqual(status, 202)
        self.assertTrue(started.wait(timeout=3))
        status, payload = http_json(self.base + f"/cancel/{first['job_id']}", method="POST", data={})
        self.assertEqual(status, 200)
        deadline = time.time() + 5
        last = {}
        while time.time() < deadline:
            _, last = http_json(self.base + f"/status/{first['job_id']}")
            if last.get("state") != "running":
                break
            time.sleep(0.05)
        self.assertEqual(last.get("state"), "cancelled")
        status, _ = http_json(self.base + "/cancel/nope", method="POST", data={})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
