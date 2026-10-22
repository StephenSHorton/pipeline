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


def write_picture(path: Path, size: tuple[int, int] = (32, 24)) -> Path:
    """A real, decodable picture (the factory refuses 3D from anything else), in the format its suffix names."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}.get(path.suffix.lower(), "PNG")
    Image.new("RGB", size, (200, 120, 40)).save(path, fmt)
    return path


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

    def test_generate_argv_mesh_and_paint_off(self) -> None:
        image = ROOT / "in" / "radio-concept.jpg"
        mesh = ROOT / "out" / "radio" / "workbench" / "clay.glb"
        argv = factory_env.generate_argv(ROOT, image=image, quality="psx", stem="radio", poly="medium", mesh=mesh,
                                         paint=False)
        self.assertEqual(argv[argv.index("--mesh") + 1], str(mesh))
        self.assertEqual(argv[argv.index("--paint") + 1], "off")
        # The defaults add neither flag: a plain make's argv is as it was.
        plain = factory_env.generate_argv(ROOT, image=image, quality="psx", stem="radio", poly="medium")
        self.assertNotIn("--mesh", plain)
        self.assertNotIn("--paint", plain)
        # fill: only "photo" reaches generate.sh; mirror_ends only when stated.
        self.assertNotIn("--fill", plain)
        self.assertNotIn("--mirror-ends", plain)
        none = factory_env.generate_argv(ROOT, image=image, quality="psx", stem="radio", poly="medium", fill="none")
        self.assertEqual(none, plain)
        filled = factory_env.generate_argv(ROOT, image=image, quality="psx", stem="radio", poly="medium",
                                           fill="photo", mirror_ends=False)
        self.assertEqual(filled[filled.index("--fill") + 1], "photo")
        self.assertEqual(filled[filled.index("--mirror-ends") + 1], "off")
        # delight: only when stated, on or off.
        self.assertNotIn("--delight", plain)
        delit = factory_env.generate_argv(ROOT, image=image, quality="psx", stem="radio", poly="medium",
                                          fill="photo", delight=True)
        self.assertEqual(delit[delit.index("--delight") + 1], "on")
        off = factory_env.generate_argv(ROOT, image=image, quality="psx", stem="radio", poly="medium", delight=False)
        self.assertEqual(off[off.index("--delight") + 1], "off")

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
        write_picture(self.root / "in" / "radio-concept.jpg")
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
        # A bare temp root has no binaries, weights or vendor venv: every probe says so.
        self.assertEqual(feats.get("shape"), "none")
        self.assertFalse(feats.get("shape_mv"))
        self.assertFalse(feats.get("remesh"))
        self.assertIn("licence", feats.get("draw") or {})

    def test_factory_features_are_probed(self) -> None:
        import os

        feats = factory_http.factory_features(self.root)
        self.assertEqual((feats["shape"], feats["shape_mv"], feats["unirig"], feats["remesh"]), ("none", False, False, False))
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        # A dangling autoremesher link is no remesh; a link to a runnable file is; a plain non-executable file is not.
        (bin_dir / "autoremesher").symlink_to(self.root / "nowhere" / "autoremesher")
        self.assertFalse(factory_http.factory_features(self.root)["remesh"])
        target = self.root / "remesher-bin"
        target.write_text("#!/bin/sh\n")
        (bin_dir / "autoremesher").unlink()
        (bin_dir / "autoremesher").symlink_to(target)
        self.assertFalse(factory_http.factory_features(self.root)["remesh"])
        os.chmod(target, 0o755)
        self.assertTrue(factory_http.factory_features(self.root)["remesh"])
        # Shape: bin/hy3d runnable and weights/shape-small holding a config and safetensors.
        hy3d = bin_dir / "hy3d"
        hy3d.write_text("#!/bin/sh\n")
        os.chmod(hy3d, 0o755)
        self.assertEqual(factory_http.factory_features(self.root)["shape"], "none")
        weights = self.root / "weights" / "shape-small"
        weights.mkdir(parents=True)
        (weights / "config.yaml").write_text("x")
        (weights / "model.fp16.safetensors").write_bytes(b"x")
        feats = factory_http.factory_features(self.root)
        self.assertEqual(feats["shape"], "hy3d-mlx")
        self.assertTrue(feats["shape_weights"])
        # 2mv weights alone are not 2mv: the hunyuan3d-2 venv must import too.
        mv = self.root / "weights" / "tencent" / "Hunyuan3D-2mv" / "hunyuan3d-dit-v2-mv-turbo"
        mv.mkdir(parents=True)
        (mv / "config.yaml").write_text("x")
        (mv / "model.fp16.safetensors").write_bytes(b"x")
        feats = factory_http.factory_features(self.root)
        self.assertTrue(feats["shape_mv_weights"])
        self.assertFalse(feats["shape_mv_venv"])
        self.assertFalse(feats["shape_mv"])

    def test_draw_feature_names_the_licence(self) -> None:
        from unittest import mock

        def draw_feature():
            out = factory_http.draw_feature()
            out.pop("photo", None)
            return out

        with mock.patch.object(factory_http, "still_backend", return_value="z-image"):
            self.assertEqual(draw_feature(),
                             {"backend": "z-image", "model": "z-image-turbo", "licence": "apache-2.0"})
        with mock.patch.object(factory_http, "still_backend", return_value="qwen-2512"):
            self.assertEqual(draw_feature(), {"backend": "qwen-2512", "model": "qwen-image-2512+lightning8",
                                              "licence": "apache-2.0"})
        with mock.patch.object(factory_http, "still_backend", return_value="qwen-image"):
            self.assertIn("non-commercial", draw_feature()["licence"])
        with mock.patch.object(factory_http, "still_backend", return_value="brand-new"):
            self.assertEqual(draw_feature()["licence"], "unknown")
        with mock.patch.object(factory_http, "still_backend", return_value=""):
            self.assertEqual(draw_feature(), {"backend": "", "licence": "none"})
        with mock.patch("flat2d.sheet.edit_backend", return_value="qwen-edit"):
            self.assertEqual(factory_http.draw_feature()["photo"], {"backend": "qwen-edit",
                                                                    "model": "qwen-image-edit-2511+lightning8",
                                                                    "licence": "apache-2.0"})
        with mock.patch("flat2d.sheet.edit_backend", return_value=""):
            self.assertEqual(factory_http.draw_feature()["photo"]["backend"], "")

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

        import io
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, "JPEG")
        raw = buf.getvalue()
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
            data=write_picture(self.root / "scratch.png").read_bytes(),
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
        write_picture(cam)
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
        write_picture(extra)
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
        self.assertIsNone(factory_http.clean_intent(None))
        for bad in ({"output": "rocket"}, {"output": "ps1", "up": "q"}, {"output": "ps1", "height": "tall"},
                    {"output": "ps1", "faces": "0"}, {"output": "model", "poly": "ultra"},
                    {"output": "model", "steps": ["fly"]}, {"output": "model", "kind": "bad/kind"},
                    {"output": "model", "facing_source": "guess"}, {"output": "model", "rig": {"canonical": "yes"}},
                    "ps1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                factory_http.clean_intent(bad)
        intent = factory_http.clean_intent({"output": "ps1", "facing": "right", "height": 58, "name": "cow", "up": ""})
        # name is both the psx file name and what the subject is (the canonical draw reads it).
        self.assertEqual(intent, {"output": "ps1", "psx": {"facing": "right", "height": "58", "name": "cow"}, "name": "cow"})
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

    # -- No 3D without a picture ------------------------------------------------------------------------------

    def assert_no_picture(self, status: int, payload: dict) -> None:
        self.assertEqual(status, 400, payload)
        self.assertEqual(payload.get("error"), "3D needs a picture. Draw or add one first.")
        # Refused before the job slot: no job was started (idle, or still the last finished one).
        _, now = http_json(self.base + "/status")
        self.assertIn(now.get("state"), ("idle", "ok"))
        if now.get("state") == "idle":
            self.assertIsNone(now.get("job_id"))

    def test_generate_refuses_without_a_picture(self) -> None:
        ran = []
        self.factory._runner = lambda spec: ran.append(spec) or 0
        (self.root / "in" / "notes.jpg").write_text("this is a text file, renamed\n")
        (self.root / "in" / "radio.glb").write_bytes((self.root / "out" / "radio" / "mesh.glb").read_bytes())
        for body in ({}, {"stem": "x"}, {"image": "out/radio/mesh.glb"}, {"image": "in/notes.jpg"},
                     {"image": "in/radio.glb"}, {"image": "in/missing.png"},
                     {"image": "in/radio-concept.jpg", "images": ["in/notes.jpg"]},
                     {"image": "in/radio-concept.jpg", "images": ["out/radio/mesh.glb"]}):
            with self.subTest(body=body):
                self.assert_no_picture(*http_json(self.base + "/generate", method="POST", data=body))
        self.assertEqual(ran, [])
        with self.assertRaises(ValueError) as caught:
            factory_env.require_picture(self.root / "in" / "notes.jpg")
        self.assertEqual(str(caught.exception), factory_env.NO_PICTURE)
        for ext in (".png", ".webp", ".jpeg"):
            self.assertTrue(factory_env.require_picture(write_picture(self.root / "in" / f"ok{ext}")).is_file())

    def test_upload_refuses_what_is_not_a_picture(self) -> None:
        import base64

        cases = [("model.glb", (self.root / "out" / "radio" / "mesh.glb").read_bytes()),
                 ("notes.txt", b"hello"), ("notes.jpg", b"just text, renamed"), ("noext", b"\x89PNG\r\n")]
        for name, data in cases:
            with self.subTest(name=name):
                body = {"name": name, "data": base64.b64encode(data).decode("ascii")}
                self.assert_no_picture(*http_json(self.base + "/upload", method="POST", data=body))
        self.assertEqual(sorted(p.name for p in (self.root / "in").iterdir()), ["radio-concept.jpg"])

    def test_generate_bad_quality_and_psx_are_400(self) -> None:
        for body, word in (({"quality": "ultra"}, "quality"), ({"quality": "psx", "psx_height": "tall"}, "height"),
                           ({"quality": "psx", "psx_up": "x"}, "up"), ({"quality": "psx", "psx_faces": "-3"}, "faces"),
                           ({"intent": {"output": "nope"}}, "output")):
            with self.subTest(body=body):
                status, payload = http_json(self.base + "/generate", method="POST",
                                            data={"image": "in/radio-concept.jpg", "stem": "q", **body})
                self.assertEqual(status, 400, payload)
                self.assertIn(word, payload.get("error") or "")
        _, idle = http_json(self.base + "/status")
        self.assertEqual(idle.get("state"), "idle")

    # -- Honest status: DEGRADED lines, views handoff, unload ---------------------------------------------------

    def run_generate(self, body: dict, log_lines: tuple[str, ...] = ()) -> tuple[dict, dict]:
        seen: dict = {}

        def runner(spec: dict) -> int:
            seen.update(spec)
            with open(spec["log"], "a", encoding="utf-8") as fh:
                for line in log_lines:
                    fh.write(line + "\n")
            write_triangle_glb(self.root / "out" / spec["stem"] / "mesh.glb")
            return 0

        self.factory._runner = runner
        status, first = http_json(self.base + "/generate", method="POST", data=body)
        self.assertEqual(status, 202, first)
        deadline = time.time() + 5
        last = first
        while time.time() < deadline:
            _, last = http_json(self.base + f"/status/{first['job_id']}")
            if last.get("state") != "running":
                break
            time.sleep(0.05)
        self.assertEqual(last.get("state"), "ok", last)
        return seen, last

    def test_degraded_lines_reach_the_status(self) -> None:
        _, last = self.run_generate({"image": "in/radio-concept.jpg", "stem": "deg"},
                                    ("generate: shape", "DEGRADED shape mv_failed: x", "  DEGRADED nope", "DEGRADED shape mv_failed: x"))
        self.assertIn({"stage": "shape", "code": "mv_failed", "message": "x"}, last["degraded"])
        self.assertEqual(sum(1 for d in last["degraded"] if d["code"] == "mv_failed"), 1)
        # An uploaded photo has no views: an honest note, not an error.
        self.assertIn({"stage": "shape", "code": "single_view", "message": "one picture"}, last["degraded"])
        self.assertEqual(last["result"]["degraded"], last["degraded"])
        self.assertEqual(factory_env.parse_degraded("DEGRADED rig clips_dropped: Flap,Glide\nDEGRADED bad"),
                         [{"stage": "rig", "code": "clips_dropped", "message": "Flap,Glide"}])

    def test_generate_always_unloads_the_draw_model(self) -> None:
        from unittest import mock
        from flat2d import optional

        calls = []
        with mock.patch.object(optional, "unload", lambda: calls.append(1), create=True):
            _, last = self.run_generate({"image": "in/radio-concept.jpg", "stem": "unl"})
        self.assertEqual(calls, [1])
        log = (self.root / "out" / "unl" / "generate.log").read_text()
        self.assertIn("unloaded the draw model", log)
        # No unload() (or one that raises) is a warning in the job log; the job still runs.
        with mock.patch.object(optional, "unload", None, create=True):
            self.assertIn("WARNING", self.factory._unload_draw())

        def boom() -> None:
            raise RuntimeError("stuck")

        with mock.patch.object(optional, "unload", boom, create=True):
            self.assertIn("stuck", self.factory._unload_draw())

    def make_views(self, stem: str, *, sha: str | None = None, consistent: bool = True) -> Path:
        flat = self.root / "out" / stem / "flat"
        src = write_picture(flat / "source.png")
        views = []
        for vid in ("front", "right", "back", "left"):
            write_picture(flat / "views" / f"{vid}.png")
            views.append({"id": vid, "file": f"flat/views/{vid}.png"})
        doc = {"version": 1, "canonical": "right", "source_sha256": sha or factory_env.file_sha256(src),
               "views": views, "consistent": consistent, "flaws": [] if consistent else ["height spread"],
               "degraded": []}
        (flat / "views.json").write_text(json.dumps(doc))
        return flat / "views.json"

    def test_matching_views_pass_to_generate(self) -> None:
        vj = self.make_views("hen")
        seen, last = self.run_generate({"image": "out/hen/flat/source.png", "stem": "hen"})
        self.assertEqual(seen["views_json"], str(vj.resolve()))
        argv = factory_env.generate_argv(self.root, image=Path(seen["image"]), quality="stack", stem="hen",
                                         poly="medium", views_json=Path(seen["views_json"]))
        self.assertEqual(argv[argv.index("--views-json") + 1], str(vj.resolve()))
        self.assertEqual(last["degraded"], [])

    def test_stale_views_are_not_passed(self) -> None:
        self.make_views("hen", sha="0" * 64)
        seen, last = self.run_generate({"image": "out/hen/flat/source.png", "stem": "hen"})
        self.assertIsNone(seen["views_json"])
        self.assertEqual([(d["stage"], d["code"]) for d in last["degraded"]], [("views", "stale")])
        self.assertIn("DEGRADED views stale:", (self.root / "out" / "hen" / "generate.log").read_text())
        argv = factory_env.generate_argv(self.root, image=Path(seen["image"]), quality="stack", stem="hen", poly="medium",
                                         degraded=last["degraded"])
        self.assertNotIn("--views-json", argv)
        # generate.sh records the factory's note in degraded.json and inspect.json (the model card), not only the log.
        self.assertTrue(argv[argv.index("--degraded") + 1].startswith("views stale: "), argv)

    def test_a_drawing_without_views_says_so(self) -> None:
        # e.g. an older concept take picked: the take carries no views, so none are left beside its source.png
        write_picture(self.root / "out" / "hen" / "flat" / "source.png")
        seen, last = self.run_generate({"image": "out/hen/flat/source.png", "stem": "hen"})
        self.assertIsNone(seen["views_json"])
        self.assertEqual(last["degraded"], [{"stage": "shape", "code": "single_view",
                                             "message": "this drawing has no turnaround views; one picture"}])

    def test_inconsistent_views_and_bad_view_files(self) -> None:
        self.make_views("hen", consistent=False)
        seen, last = self.run_generate({"image": "out/hen/flat/source.png", "stem": "hen"})
        self.assertIsNone(seen["views_json"])
        self.assertEqual([(d["stage"], d["code"]) for d in last["degraded"]], [("views", "inconsistent")])
        self.make_views("hen")
        (self.root / "out" / "hen" / "flat" / "views" / "back.png").write_text("not a picture")
        before = self.factory._snapshot()["job_id"]
        self.assert_no_picture(*http_json(self.base + "/generate", method="POST",
                                          data={"image": "out/hen/flat/source.png", "stem": "hen"}))
        self.assertEqual(self.factory._snapshot()["job_id"], before)

    def test_rig_and_draw_sidecars_feed_degraded(self) -> None:
        folder = self.root / "out" / "radio"
        (folder / "rig").mkdir()
        (folder / "rig" / "rig.json").write_text(json.dumps(
            {"degraded": ["rig build_failed: no spine", {"stage": "rig", "code": "poor_match", "message": "scale 2.3x"}]}))
        (folder / "rig" / "rig.log").write_text("DEGRADED rig clips_dropped: Flap,Glide\n")
        job = {"job_id": "r", "kind": "rig", "stem": "radio", "state": "ok", "log": str(folder / "rig" / "rig.log"),
               "started": time.time() - 5}
        self.factory._job = dict(job)
        _, st = http_json(self.base + "/status/r")
        self.assertEqual([d["code"] for d in st["degraded"]], ["clips_dropped", "build_failed", "poor_match"])
        # A rig.json older than the job is the last rig's news, not this one's.
        self.factory._job = {**job, "started": time.time() + 60}
        _, st = http_json(self.base + "/status/r")
        self.assertEqual([d["code"] for d in st["degraded"]], ["clips_dropped"])
        (folder / "flat").mkdir()
        (folder / "flat" / "views.json").write_text(json.dumps({"degraded": ["sheet backend: z-image"]}))
        (folder / "draw.log").write_text("DEGRADED draw facing_unavailable: no VLM\n")
        self.factory._job = {"job_id": "d", "kind": "flat2d", "stem": "radio", "state": "ok", "started": time.time() - 5,
                             "log": str(folder / "draw.log"), "result": {"ok": True}}
        _, st = http_json(self.base + "/status/d")
        self.assertEqual([d["code"] for d in st["degraded"]], ["facing_unavailable", "backend"])
        self.assertEqual(st["result"]["degraded"], st["degraded"])

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

    def test_stop_signal_reaps_the_job_then_stops_serving(self) -> None:
        import signal
        import subprocess as sp

        started = threading.Event()
        procs = []

        def runner(spec: dict) -> int:
            proc = sp.Popen(["sleep", "30"], start_new_session=True)
            procs.append(proc)
            with self.factory._lock:
                self.factory._job["pid"] = proc.pid
            started.set()
            return proc.wait()

        class Server:
            stopped = threading.Event()

            def shutdown(self) -> None:
                self.stopped.set()

        self.factory._runner = runner
        try:
            status, first = http_json(self.base + "/generate", method="POST", data={"image": "in/radio-concept.jpg", "stem": "radio"})
            self.assertEqual(status, 202)
            self.assertTrue(started.wait(timeout=3))
            server = Server()
            handle = factory_http.stop_handler(self.factory, server)
            handle(signal.SIGTERM, None)
            # The job's own session is gone by the time the handler returns its shutdown thread.
            self.assertEqual(procs[0].wait(timeout=5), -signal.SIGTERM)
            self.assertTrue(server.stopped.wait(timeout=3))
            handle(signal.SIGINT, None)  # a second signal is harmless
            deadline = time.time() + 5
            while time.time() < deadline and (self.factory._snapshot() or {}).get("state") == "running":
                time.sleep(0.05)
            self.assertEqual(self.factory._snapshot().get("state"), "cancelled")
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()

    # -- A supplied mesh and no paint (the workbench) -------------------------------------------------------------

    def test_generate_takes_a_mesh_under_the_asset_and_paint_off(self) -> None:
        clay = self.root / "out" / "radio" / "workbench" / "clay.glb"
        write_triangle_glb(clay)
        seen, last = self.run_generate({"image": "in/radio-concept.jpg", "mesh": "out/radio/workbench/clay.glb",
                                        "paint": False, "quality": "psx"})
        self.assertEqual(seen["stem"], "radio")  # the mesh's asset, when no stem is given
        self.assertEqual(seen["mesh"], str(clay.resolve()))
        self.assertIs(seen["paint"], False)
        # The shape never ran from one picture: no "one picture" fallback for it.
        self.assertFalse([d for d in last["degraded"] if d["code"] == "single_view"], last["degraded"])
        argv = factory_env.generate_argv(self.root, image=Path(seen["image"]), quality="psx", stem=seen["stem"],
                                         poly="medium", mesh=Path(seen["mesh"]), paint=seen["paint"])
        self.assertEqual(argv[argv.index("--mesh") + 1], str(clay.resolve()))
        self.assertIn("off", argv)
        # Without them a make is as before: paint on, no mesh. The query string's "false" is paint off too.
        seen, _ = self.run_generate({"image": "in/radio-concept.jpg", "stem": "plain"})
        self.assertIsNone(seen["mesh"])
        self.assertIs(seen["paint"], True)
        seen, _ = self.run_generate({"image": "in/radio-concept.jpg", "stem": "q", "paint": "false"})
        self.assertIs(seen["paint"], False)

    def test_generate_takes_fill_and_mirror_ends(self) -> None:
        seen, last = self.run_generate({"image": "in/radio-concept.jpg", "stem": "f", "quality": "textured",
                                        "fill": "photo"})
        self.assertEqual(seen["fill"], "photo")
        self.assertIsNone(seen["mirror_ends"])
        self.assertEqual(last.get("fill"), "photo")
        seen, _ = self.run_generate({"image": "in/radio-concept.jpg", "stem": "f", "fill": "photo",
                                     "mirror_ends": "off"})
        self.assertIs(seen["mirror_ends"], False)
        # Absent or "none": as before.
        seen, _ = self.run_generate({"image": "in/radio-concept.jpg", "stem": "f"})
        self.assertEqual(seen["fill"], "none")
        seen, _ = self.run_generate({"image": "in/radio-concept.jpg", "stem": "f", "fill": "none"})
        self.assertEqual(seen["fill"], "none")

    def test_generate_takes_delight(self) -> None:
        seen, last = self.run_generate({"image": "in/radio-concept.jpg", "stem": "f", "quality": "textured"})
        self.assertIsNone(seen["delight"])
        seen, last = self.run_generate({"image": "in/radio-concept.jpg", "stem": "f", "quality": "textured",
                                        "fill": "photo", "delight": True})
        self.assertIs(seen["delight"], True)
        self.assertIs(last.get("delight"), True)
        argv = factory_env.generate_argv(self.root, image=Path(seen["image"]), quality="textured", stem="f",
                                         poly="medium", delight=seen["delight"])
        self.assertEqual(argv[argv.index("--delight") + 1], "on")
        # The query string's "off" is false.
        seen, _ = self.run_generate({"image": "in/radio-concept.jpg", "stem": "f", "delight": "off"})
        self.assertIs(seen["delight"], False)

    def test_generate_refuses_a_bad_fill(self) -> None:
        for body, word in (
            ({"fill": "swatch"}, "fill"),
            ({"fill": ["photo"]}, "fill"),
            ({"fill": "photo", "quality": "draft"}, "draft"),
            ({"fill": "photo", "mirror_ends": "maybe"}, "mirror_ends"),
            ({"fill": "photo", "mirror_ends": 1}, "mirror_ends"),
            ({"delight": "maybe"}, "delight"),
            ({"delight": 1}, "delight"),
            ({"delight": [True]}, "delight"),
        ):
            with self.subTest(body=body):
                status, payload = http_json(self.base + "/generate", method="POST",
                                            data={"image": "in/radio-concept.jpg", "stem": "f", **body})
                self.assertEqual(status, 400, payload)
                self.assertIn(word, payload.get("error") or "")
        _, idle = http_json(self.base + "/status")
        self.assertEqual(idle.get("state"), "idle")

    def test_fill_is_a_labelled_stage(self) -> None:
        prog = factory_http.parse_progress("STAGES isolate shape reduce unwrap retexture fill lods inspect\n"
                                           "STAGE fill x\n")
        self.assertEqual(prog["stages"][5], {"key": "fill", "label": "Fill what the photo missed"})
        self.assertEqual(prog["stage_index"], 5)

    def test_generate_refuses_a_mesh_outside_the_asset(self) -> None:
        write_triangle_glb(self.root / "in" / "loose.glb")
        write_triangle_glb(self.root / "out" / "other" / "mesh.glb")
        write_triangle_glb(self.root.parent / f"{self.root.name}-outside.glb")
        (self.root / "out" / "radio" / "notes.txt").write_text("x")
        for body, word in (
            ({"mesh": f"../{self.root.name}-outside.glb"}, "escapes"),
            ({"mesh": str(self.root.parent / f"{self.root.name}-outside.glb")}, "escapes"),
            ({"mesh": "in/loose.glb"}, "out/<stem>/"),
            ({"mesh": "out/mesh.glb"}, "out/<stem>/"),
            ({"mesh": "out/radio/notes.txt"}, ".glb"),
            ({"mesh": "out/other/mesh.glb", "stem": "radio"}, "out/radio/"),
            ({"mesh": "out/radio/missing.glb"}, "no mesh"),
            ({"mesh": ["out/radio/mesh.glb"]}, "mesh"),
            ({"mesh": "out/radio/mesh.glb", "paint": "maybe"}, "paint"),
            ({"mesh": "out/radio/mesh.glb", "paint": False, "quality": "pbr"}, "pbr"),
            # The picture is still required: isolate and the photo bake read it.
            ({"mesh": "out/radio/mesh.glb", "image": ""}, "picture"),
        ):
            with self.subTest(body=body):
                status, payload = http_json(self.base + "/generate", method="POST",
                                            data={"image": "in/radio-concept.jpg", **body})
                self.assertEqual(status, 400, payload)
                self.assertIn(word, payload.get("error") or "")
        _, idle = http_json(self.base + "/status")
        self.assertEqual(idle.get("state"), "idle")

    def test_unwrap_is_a_labelled_stage(self) -> None:
        prog = factory_http.parse_progress("STAGES isolate shape reduce unwrap retexture inspect\nSTAGE unwrap x\n")
        self.assertEqual(prog["stages"][3], {"key": "unwrap", "label": "Unwrap the mesh"})
        self.assertEqual(prog["stage_index"], 3)

    def test_reap_without_a_running_job_does_nothing(self) -> None:
        self.factory.reap()
        self.factory._job = {"job_id": "x", "state": "ok", "pid": 999999}
        self.factory.reap()
        self.assertFalse(self.factory._job.get("cancel"))


class ServeShutdownTests(unittest.TestCase):
    """The app stops the sidecar with SIGTERM to its process group (desktop/internal/sidecar/proc.go); a job runs
    in its own session, so serve() has to take it down itself."""

    def test_sigterm_kills_the_job_group_and_exits(self) -> None:
        import os
        import signal
        import socket
        import subprocess as sp
        import tempfile

        with tempfile.TemporaryDirectory() as tmp, socket.socket() as s:
            root = Path(tmp)
            (root / "in").mkdir()
            write_picture(root / "in" / "radio-concept.jpg")
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            pid_file = root / "job.pid"
            code = (
                "import subprocess, sys\n"
                f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
                "import factory_http\n"
                "def runner(self, spec):\n"
                "    proc = subprocess.Popen(['sleep', '30'], start_new_session=True)\n"
                "    with self._lock:\n"
                "        self._job['pid'] = proc.pid\n"
                f"    open({str(pid_file)!r} + '.tmp', 'w').write(str(proc.pid))\n"
                f"    __import__('os').replace({str(pid_file)!r} + '.tmp', {str(pid_file)!r})\n"
                "    return proc.wait()\n"
                "factory_http.Factory._run_generate_sh = runner\n"
                f"factory_http.serve('127.0.0.1', {port}, factory_http.Path({tmp!r}))\n"
            )
            side = sp.Popen([sys.executable, "-c", code], stdout=sp.DEVNULL, stderr=sp.PIPE, start_new_session=True)
            job_pid = 0
            try:
                base = f"http://127.0.0.1:{port}"
                deadline = time.time() + 20
                while time.time() < deadline:
                    try:
                        http_json(base + "/health", timeout=1)
                        break
                    except OSError:
                        time.sleep(0.1)
                status, _ = http_json(base + "/generate", method="POST", data={"image": "in/radio-concept.jpg", "stem": "radio"})
                self.assertEqual(status, 202)
                deadline = time.time() + 5
                while time.time() < deadline and not pid_file.is_file():
                    time.sleep(0.05)
                job_pid = int(pid_file.read_text())
                os.killpg(side.pid, signal.SIGTERM)  # what proc.go sends; the job is not in this group
                self.assertEqual(side.wait(timeout=10), 0, side.stderr.read().decode())
                deadline = time.time() + 5
                while time.time() < deadline:
                    try:
                        os.kill(job_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                with self.assertRaises(ProcessLookupError):
                    os.kill(job_pid, 0)
                self.assertFalse((root / "out" / "factory_http.pid").exists())
            finally:
                if side.poll() is None:
                    side.kill()
                    side.wait()
                if job_pid:
                    try:
                        os.killpg(job_pid, signal.SIGKILL)
                    except OSError:
                        pass


if __name__ == "__main__":
    unittest.main()
