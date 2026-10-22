#!/usr/bin/env python3
"""The workbench's MCP server (scripts/workbench/mcp_server.py and tools.py), in process and over loopback HTTP.

The factory is a stub (FactoryClient with its HTTP call replaced), so nothing here reaches :8766 or loads a model.
Each test's repo root is a scratch folder under out/_wbtest_mcp_*/, removed afterwards.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from workbench import mcp_server, tools  # noqa: E402
from workbench.factory_client import FactoryClient, FactoryDown  # noqa: E402

PLAN_TOOLS = {"asset_info", "look", "measure", "compare_reference", "clay_get", "clay_set", "clay_edit", "clay_mesh",
              "mesh_edit", "bake_photo", "psx", "draw_views", "generate", "job_status", "cancel", "pick_take"}


class StubFactory(FactoryClient):
    """Answers the workbench's factory calls from a table, and records them."""

    def __init__(self, status=None, down=False):
        super().__init__("http://127.0.0.1:1")
        self.state = status or {"ok": True, "state": "idle", "job_id": None}
        self.down = down
        self.calls: list[tuple[str, str, dict | None]] = []

    def _call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if self.down:
            raise FactoryDown("factory not reachable (stub)")
        if path.startswith("/status"):
            return dict(self.state)
        if path.endswith("/snapshot"):
            return {"ok": True, "stage": body["stage"], "take": {"id": "t1"}, "created": True}
        if path == "/generate":
            return {"ok": True, "accepted": True, "job_id": "abcd1234", "state": "running"}
        return {"ok": True}

    def snapshots(self):
        return [c for c in self.calls if c[1].endswith("/snapshot")]


def quadruped(head_dx: float = 0.0) -> trimesh.Trimesh:
    body = trimesh.creation.icosphere(subdivisions=3)
    body.apply_scale([0.4, 0.35, 1.0])
    body.apply_translation([0, 0.8, 0])
    parts = [body, trimesh.creation.icosphere(subdivisions=3, radius=0.25).apply_translation([head_dx, 1.0, 1.15])]
    for x in (-0.22, 0.22):
        for z in (-0.6, 0.6):
            leg = trimesh.creation.cylinder(radius=0.08, height=0.6, sections=16)
            leg.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            parts.append(leg.apply_translation([x, 0.3, z]))
    return trimesh.util.concatenate(parts)


class Base(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / "out").mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="_wbtest_mcp_", dir=ROOT / "out"))
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / "out").mkdir()
        self.factory = StubFactory()
        self.bench = tools.Workbench(self.root, self.factory)
        self.ids = iter(range(1, 1000))

    def rpc(self, method, params=None):
        msg = {"jsonrpc": "2.0", "id": next(self.ids), "method": method}
        if params is not None:
            msg["params"] = params
        return mcp_server.handle_rpc(self.bench, msg)

    def call(self, name, args):
        res = self.rpc("tools/call", {"name": name, "arguments": args})
        self.assertIn("result", res, res)
        return res["result"]

    def ok(self, name, args):
        res = self.call(name, args)
        self.assertFalse(res["isError"], res["content"][0]["text"])
        return json.loads(res["content"][0]["text"]), res

    def asset(self, stem: str, mesh: trimesh.Trimesh, name: str = "psx.glb") -> str:
        folder = self.root / "out" / stem
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(folder / name)
        return f"out/{stem}/{name}"


class ProtocolTests(Base):
    def test_initialize_echoes_version_and_lists_tools(self) -> None:
        res = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "t", "version": "0"}})["result"]
        self.assertEqual(res["protocolVersion"], "2025-06-18")
        self.assertEqual(res["capabilities"], {"tools": {}})
        self.assertEqual(res["serverInfo"]["name"], "workbench")
        self.assertIsNone(mcp_server.handle_rpc(self.bench, {"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(self.rpc("ping")["result"], {})
        self.assertEqual(self.rpc("nope")["error"]["code"], mcp_server.METHOD_NOT_FOUND)

    def test_tool_schemas_are_valid(self) -> None:
        listed = self.rpc("tools/list")["result"]["tools"]
        self.assertEqual({t["name"] for t in listed}, PLAN_TOOLS)
        for t in listed:
            s = t["inputSchema"]
            self.assertTrue(t["description"], t["name"])
            self.assertEqual(s["type"], "object", t["name"])
            self.assertIsInstance(s["properties"], dict)
            self.assertFalse(s.get("additionalProperties", True), t["name"])
            self.assertLessEqual(set(s["required"]), set(s["properties"]), t["name"])
            for key, prop in s["properties"].items():
                self.assertTrue("type" in prop or "enum" in prop, f"{t['name']}.{key}")
            json.dumps(s)

    def test_bad_arguments(self) -> None:
        self.assertEqual(self.rpc("tools/call", {"name": "nope", "arguments": {}})["error"]["code"],
                         mcp_server.INVALID_PARAMS)
        self.assertEqual(self.rpc("tools/call", {"name": "measure", "arguments": [1]})["error"]["code"],
                         mcp_server.INVALID_PARAMS)
        for args, says in (({"stem": 5}, "must be string"), ({"stem": "x", "colour": 1}, "not an argument"),
                           ({"glb": "../../etc/hosts.glb"}, "escapes"), ({}, "give stem or glb"),
                           ({"stem": "../x"}, ""), ({"stem": "missing"}, "no asset")):
            res = self.call("measure", args)
            self.assertTrue(res["isError"], args)
            self.assertIn(says, res["content"][0]["text"])
        res = self.call("clay_set", {"stem": "bale", "blobs": [{"name": "a", "form": {"type": "Cone"}}]})
        self.assertTrue(res["isError"])
        self.assertIn("Cone", res["content"][0]["text"])
        self.assertFalse((self.root / "out" / "bale").exists())


class ToolTests(Base):
    def test_measure_round_trip(self) -> None:
        glb = self.asset("synth", quadruped(head_dx=0.16))
        out, _ = self.ok("measure", {"glb": glb})
        self.assertEqual(out["glb"], glb)
        self.assertEqual(out["pieces"], 6)
        self.assertAlmostEqual(out["head_offset_pct"], 20, delta=3)
        by_stem, _ = self.ok("measure", {"stem": "synth"})
        self.assertEqual(by_stem["tris"], out["tris"])
        self.assertEqual(self.factory.calls, [])  # reading never asks the factory

    def test_clay_set_edit_mesh(self) -> None:
        blobs = [{"name": "bale", "form": {"type": "Cylinder", "radius": 0.6, "height": 1.2},
                  "rotation": [0, 0, 90], "color": "#c8a050"},
                 {"name": "dent", "op": "carve", "form": {"type": "Sphere", "radius": 0.2},
                  "translation": [0, 0.6, 0]}]
        out, _ = self.ok("clay_set", {"stem": "bale", "blobs": blobs})
        self.assertEqual([b["name"] for b in out["blobs"]], ["bale", "dent"])
        self.assertIn("skipped", out["take"])  # a few-KB JSON change does not copy the whole mesh stage
        self.ok("clay_edit", {"stem": "bale", "name": "dent", "patch": {"remove": True}})
        self.ok("clay_edit", {"stem": "bale", "name": "bale", "patch": {"form": {"radius": 0.5}}})
        got, _ = self.ok("clay_get", {"stem": "bale"})
        self.assertEqual([b["name"] for b in got["blobs"]], ["bale"])
        self.assertEqual(got["blobs"][0]["form"]["radius"], 0.5)
        self.assertEqual(got["blobs"][0]["form"]["height"], 1.2)
        mesh, _ = self.ok("clay_mesh", {"stem": "bale", "grid": 32, "height": 1.0, "texture": 128})
        glb = self.root / "out" / "bale" / "workbench" / "clay.glb"
        self.assertTrue(glb.is_file())
        self.assertEqual(mesh["path"], "out/bale/workbench/clay.glb")
        self.assertGreater(mesh["faces"], 100)
        loaded = trimesh.load(glb, force="mesh")
        self.assertIsNotNone(loaded.visual.uv)
        self.assertAlmostEqual(float(np.ptp(loaded.vertices[:, 1])), 1.0, places=2)
        scene = json.loads((glb.parent / "scene.json").read_text())
        self.assertEqual(scene["objects"][0]["kind"], "clay")
        # Character fields go through the schema, and a noise patch keeps the seed it does not mention.
        self.ok("clay_edit", {"stem": "bale", "name": "bale", "patch": {
            "form": {"rounding": 0.1, "bulge": 0.05}, "noise": {"amp": 0.01, "scale": 0.15, "seed": 4}}})
        self.ok("clay_edit", {"stem": "bale", "name": "bale", "patch": {"noise": {"amp": 0.012}}})
        got, _ = self.ok("clay_get", {"stem": "bale"})
        self.assertEqual(got["blobs"][0]["noise"]["seed"], 4)
        self.assertEqual(got["blobs"][0]["noise"]["amp"], 0.012)
        self.assertEqual(got["blobs"][0]["form"]["bulge"], 0.05)
        self.ok("clay_edit", {"stem": "bale", "name": "bale", "patch": {"noise": None}})
        got, _ = self.ok("clay_get", {"stem": "bale"})
        self.assertNotIn("noise", got["blobs"][0])
        self.assertEqual(scene["objects"][0]["mesh"], "clay.glb")
        self.assertEqual([p.name for p in glb.parent.iterdir() if p.name.startswith(".")], [])  # no temp left
        snaps = self.factory.snapshots()
        self.assertEqual(len(snaps), 1)  # clay_mesh only
        self.assertTrue(all(c[1] == "/flow/bale/snapshot" and c[2]["stage"] == "mesh" for c in snaps))
        measured, _ = self.ok("measure", {"stem": "bale", "lod": "clay"})
        self.assertEqual(measured["pieces"], 1)

    def test_mesh_edit(self) -> None:
        stray = trimesh.creation.icosphere(subdivisions=1, radius=0.05).apply_translation([2, 2, 2])
        src = self.asset("junk", trimesh.util.concatenate([quadruped(), stray]), "mesh.glb")
        out, _ = self.ok("mesh_edit", {"stem": "junk", "src": src, "op": "remove_floaters", "params": {"keep": 6}})
        self.assertEqual(out["glb"], "out/junk/workbench/edit_1.glb")
        self.assertEqual(out["components"], 7)
        self.assertEqual(out["kept"], 6)
        out, _ = self.ok("mesh_edit", {"stem": "junk", "src": out["glb"], "op": "delete_sphere",
                                       "params": {"center": [0, 1.0, 1.15], "radius": 0.3}})
        self.assertEqual(out["glb"], "out/junk/workbench/edit_2.glb")
        self.assertGreater(out["faces_deleted"], 0)
        res = self.call("mesh_edit", {"stem": "junk", "src": src, "op": "symmetrize"})
        self.assertTrue(res["isError"])
        self.assertIn("not yet", res["content"][0]["text"])
        res = self.call("mesh_edit", {"stem": "junk", "src": src, "op": "delete_sphere", "params": {"radius": 1}})
        self.assertTrue(res["isError"])
        self.assertIn("center is required", res["content"][0]["text"])

    def fake_psx(self, faces: int = 1200, fail: bool = False):
        """Stand in for psx_lowpoly (bake.psx): write what it writes, record where, and answer its report."""
        seen = []

        def psx(root, glb, out_dir, glb_out, name, height, facing, faces_=None, atlas=None):
            seen.append({"out_dir": out_dir, "glb_out": glb_out, "name": name, "facing": facing})
            if fail:
                raise tools.bake.BakeError("psx_lowpoly exited 1: boom")
            out_dir.mkdir(parents=True, exist_ok=True)
            for ext in ("obj", "mtl", "png"):
                (out_dir / f"{name}.{ext}").write_text(ext)
            report = {"name": name, "obj": f"{name}.obj", "texture": f"{name}.png", "glb": glb_out.name,
                      "triangles": faces, "atlas": 256, "height": height, "facing": facing}
            (out_dir / "psx.json").write_text(json.dumps(report))
            quadruped().export(glb_out)
            return report

        self.addCleanup(setattr, tools.bake, "psx", tools.bake.psx)
        tools.bake.psx = psx
        return seen

    def test_psx_publishes_like_generate(self) -> None:
        src = self.asset("bale", quadruped(), "textured_photo.glb")
        folder = self.root / "out" / "bale"
        (folder / "psx").mkdir()
        (folder / "psx" / "old_name.obj").write_text("stale")  # another name's cut: cleared, as generate.sh does
        (folder / "workbench" / "psx").mkdir(parents=True)  # an earlier workbench run's copy: removed
        (folder / "workbench" / "psx.glb").write_bytes(b"old")
        (folder / "workbench" / "psx" / "bale.obj").write_text("old")
        (folder / "lods.json").write_text(json.dumps({"stem": "bale", "lods": [
            {"id": "paint", "file": "textured_paint.glb", "label": "Painted"},
            {"id": "photo", "file": "textured_photo.glb", "label": "Photo bake"},
            {"id": "hero", "file": "mesh.glb", "label": "Hero"}]}))
        seen = self.fake_psx()
        out, _ = self.ok("psx", {"stem": "bale", "glb": src, "height": 60, "facing": "camera"})
        self.assertEqual(out["glb"], "out/bale/psx.glb")
        self.assertEqual(out["obj"], "out/bale/psx/bale.obj")
        self.assertTrue(out["take"]["ok"])
        self.assertEqual(len(self.factory.snapshots()), 1)
        # psx_lowpoly wrote into a hidden folder, gone now; the files moved into the published places
        self.assertTrue(seen[0]["out_dir"].parent.name.startswith(".psx_"))
        self.assertEqual([p.name for p in folder.iterdir() if p.name.startswith(".")], [])
        self.assertEqual(sorted(p.name for p in (folder / "psx").iterdir()),
                         ["bale.mtl", "bale.obj", "bale.png", "psx.json"])
        self.assertGreater((folder / "psx.glb").stat().st_size, 64)
        self.assertFalse((folder / "workbench" / "psx.glb").exists())
        self.assertFalse((folder / "workbench" / "psx").exists())
        want = {"id": "psx", "file": "psx.glb", "label": "PSX", "faces": 1200, "vertices": None, "has_uvs": True}
        self.assertEqual(out["lod"], want)
        doc = json.loads((folder / "lods.json").read_text())
        self.assertEqual([x["id"] for x in doc["lods"]], ["paint", "photo", "psx", "hero"])
        self.assertEqual(doc["lods"][2], want)
        # Again with another budget: the entry is replaced where it stands, not added twice.
        self.fake_psx(faces=800)
        self.ok("psx", {"stem": "bale", "glb": src, "height": 60, "facing": "camera"})
        doc = json.loads((folder / "lods.json").read_text())
        self.assertEqual([x["id"] for x in doc["lods"]], ["paint", "photo", "psx", "hero"])
        self.assertEqual(doc["lods"][2]["faces"], 800)
        # The stem's model is now this PSX: look and measure by stem see it.
        self.assertEqual(self.bench.target({"stem": "bale"}), folder / "psx.glb")

    def test_psx_without_lods_json_and_a_failed_cut(self) -> None:
        src = self.asset("cart", quadruped(), "textured_photo.glb")
        folder = self.root / "out" / "cart"
        self.fake_psx(fail=True)
        res = self.call("psx", {"stem": "cart", "glb": src, "height": 40, "facing": "right"})
        self.assertTrue(res["isError"])
        self.assertFalse((folder / "lods.json").exists())
        self.assertFalse((folder / "psx.glb").exists())
        self.assertEqual([p.name for p in folder.iterdir() if p.name.startswith(".")], [])
        self.fake_psx()
        self.ok("psx", {"stem": "cart", "glb": src, "height": 40, "facing": "right"})
        doc = json.loads((folder / "lods.json").read_text())
        self.assertEqual(doc["stem"], "cart")
        # No lods.json: the folder's own models are listed too, as generate.sh's lods stage would list them.
        self.assertEqual([x["id"] for x in doc["lods"]], ["photo", "psx"])

    def test_psx_dropped_when_a_job_starts_during_the_cut(self) -> None:
        src = self.asset("bale", quadruped(), "textured_photo.glb")
        folder = self.root / "out" / "bale"
        (folder / "psx.glb").write_bytes(b"published")
        (folder / "psx").mkdir()
        (folder / "psx" / "bale.obj").write_text("published")
        seen = self.fake_psx()
        cut = tools.bake.psx

        def cut_then_job_starts(*args, **kw):
            report = cut(*args, **kw)
            self.factory.state = {"ok": True, "state": "running", "stem": "bale", "job_id": "j2"}
            return report

        tools.bake.psx = cut_then_job_starts
        res = self.call("psx", {"stem": "bale", "glb": src, "height": 60, "facing": "camera"})
        self.assertTrue(res["isError"])
        self.assertIn("started writing bale during the PSX cut", res["content"][0]["text"])
        self.assertEqual(len(seen), 1)
        self.assertEqual((folder / "psx.glb").read_bytes(), b"published")
        self.assertEqual((folder / "psx" / "bale.obj").read_text(), "published")
        self.assertFalse((folder / "lods.json").exists())
        self.assertEqual([p.name for p in folder.iterdir() if p.name.startswith(".")], [])

    def test_psx_refused_while_a_job_writes_the_stem(self) -> None:
        src = self.asset("bale", quadruped(), "textured_photo.glb")
        seen = self.fake_psx()
        self.factory.state = {"ok": True, "state": "running", "stem": "bale", "job_id": "j1"}
        res = self.call("psx", {"stem": "bale", "glb": src, "height": 60, "facing": "camera"})
        self.assertTrue(res["isError"])
        self.assertIn("a factory job is writing bale", res["content"][0]["text"])
        self.assertEqual(seen, [])
        self.assertFalse((self.root / "out" / "bale" / "psx.glb").exists())

    def test_look_samples_per_pixel_by_default(self) -> None:
        glb = self.asset("synth", quadruped())
        out, _ = self.ok("look", {"glb": glb, "views": ["top", "bottom"], "size": 64})
        self.assertEqual(out["sampling"], "pixel")
        self.assertEqual(out["views"], ["top", "bottom"])
        out, _ = self.ok("look", {"glb": glb, "views": ["q"], "size": 64, "sampling": "face"})
        self.assertEqual(out["sampling"], "face")
        res = self.call("look", {"glb": glb, "sampling": "smooth"})
        self.assertTrue(res["isError"])

    def test_clay_edits_read_under_the_lock(self) -> None:
        # Another client's edit lands between this call's start and its write (here: during the busy check, which
        # runs under the lock). Read outside the lock, the second save would drop it.
        self.ok("clay_set", {"stem": "bale", "blobs": [{"name": "a", "form": {"type": "Sphere", "radius": 0.5}}]})
        clay_json = self.root / "out" / "bale" / "workbench" / "clay.json"
        real = self.factory._call

        def racing(method, path, body=None):
            if path.startswith("/status") and not getattr(racing, "done", False):
                racing.done = True
                doc = json.loads(clay_json.read_text())
                blobs = doc["blobs"] if isinstance(doc, dict) else doc
                blobs.append({**blobs[0], "name": "b"})
                clay_json.write_text(json.dumps(doc))
            return real(method, path, body)

        self.factory._call = racing
        self.ok("clay_edit", {"stem": "bale", "name": "a", "patch": {"color": "#ff0000"}})
        got, _ = self.ok("clay_get", {"stem": "bale"})
        self.assertEqual([b["name"] for b in got["blobs"]], ["a", "b"])

    def test_pick_waits_for_a_workbench_write(self) -> None:
        (self.root / "out" / "bale").mkdir(parents=True)
        done = threading.Event()
        with self.bench._lock("bale"):
            t = threading.Thread(target=lambda: (self.call("pick_take", {"stem": "bale", "stage": "mesh",
                                                                         "take": "t1"}), done.set()))
            t.start()
            self.assertFalse(done.wait(0.3))
        t.join(5)
        self.assertTrue(done.is_set())

    def test_bake_photo_wants_a_cutout(self) -> None:
        from PIL import Image

        glb = self.asset("bale", trimesh.creation.box(), "mesh.glb")
        Image.new("RGB", (32, 32), (200, 160, 80)).save(self.root / "out" / "bale" / "input.jpg")
        Image.new("RGBA", (32, 32), (200, 160, 80, 255)).save(self.root / "out" / "bale" / "flat.png")
        for pic in ("out/bale/input.jpg", "out/bale/flat.png"):
            res = self.call("bake_photo", {"stem": "bale", "glb": glb, "views": [{"path": pic}]})
            self.assertTrue(res["isError"], pic)
            self.assertIn("lifted.png", res["content"][0]["text"])
        self.assertEqual(self.factory.calls, [])

    def test_bake_photo_mirrors_the_ends_and_fills_by_default(self) -> None:
        # A drum lying along z with a plain tan atlas, and a head-on photo of its round end: by default the far end
        # gets the photo flipped (mirror_ends) and the barrel a swatch of it (fill), so no texel stays tan.
        from PIL import Image

        from workbench import mesher

        drum = trimesh.creation.cylinder(radius=0.5, height=0.6, sections=24)
        vmap, faces, uvs = mesher.unwrap(drum.vertices, drum.faces, 256)
        mesh = trimesh.Trimesh(np.asarray(drum.vertices)[vmap], faces, process=False)
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (256, 256), (200, 160, 80)))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)
        glb = self.asset("bale", mesh, "textured.glb")
        rng = np.random.default_rng(0)
        a = np.zeros((128, 128, 4), np.uint8)
        yy, xx = np.mgrid[:128, :128]
        a[..., :3] = np.clip(rng.normal((120, 150, 60), 25, (128, 128, 3)), 0, 255).astype(np.uint8)
        a[..., 3] = np.where((yy - 64) ** 2 + (xx - 64) ** 2 <= 54 ** 2, 255, 0)
        Image.fromarray(a, "RGBA").save(self.root / "out" / "bale" / "lifted.png")
        view = [{"path": "out/bale/lifted.png", "facing": "camera"}]
        out, _ = self.ok("bake_photo", {"stem": "bale", "glb": glb, "views": view})
        self.assertTrue(out["mirror"])
        self.assertTrue(out["mirror_ends"])
        self.assertEqual(out["fill"]["covered_after"], 1.0)
        self.assertLess(out["fill"]["covered_before"], 1.0)
        baked = trimesh.load(self.root / out["glb"], force="mesh").visual.material.baseColorTexture
        self.assertEqual(int((np.asarray(baked.convert("RGB")) == (200, 160, 80)).all(-1).sum()), 0)
        # fill none and mirror_ends false: the plain bake, as before.
        out, _ = self.ok("bake_photo", {"stem": "bale", "glb": glb, "views": view, "fill": "none",
                                        "mirror_ends": False})
        self.assertFalse(out["mirror"])
        self.assertNotIn("fill", out)
        # A side-on first view is not end-on: mirror_ends leaves it alone.
        out, _ = self.ok("bake_photo", {"stem": "bale", "glb": glb, "fill": "none",
                                        "views": [{"path": "out/bale/lifted.png", "facing": "right"}]})
        self.assertFalse(out["mirror"])
        # Asked for on a side-on view, the mirror is the plain one (through z): no --mirror-ends reaches retexture.
        argvs = []
        real = tools.bake._run
        self.addCleanup(setattr, tools.bake, "_run", real)
        tools.bake._run = lambda root, argv, env=None: (argvs.append(argv), real(root, argv, env))[1]
        out, _ = self.ok("bake_photo", {"stem": "bale", "glb": glb, "fill": "none", "mirror": True,
                                        "views": [{"path": "out/bale/lifted.png", "facing": "right"}]})
        self.assertEqual((out["mirror"], out["mirror_ends"]), (True, False))
        self.assertIn("--mirror", argvs[-1])
        self.assertNotIn("--mirror-ends", argvs[-1])
        out, _ = self.ok("bake_photo", {"stem": "bale", "glb": glb, "views": view, "fill": "none"})
        self.assertIn("--mirror-ends", argvs[-1])

    def test_old_render_folders_are_removed(self) -> None:
        glb = self.asset("synth", quadruped())
        first, _ = self.ok("look", {"glb": glb, "views": ["side"], "size": 64})
        old = Path(first["png"]).parent
        self.assertTrue(old.is_dir())
        keep, tools.RENDER_KEEP_S = tools.RENDER_KEEP_S, -1
        self.addCleanup(setattr, tools, "RENDER_KEEP_S", keep)
        second, _ = self.ok("look", {"glb": glb, "views": ["side"], "size": 64})
        self.assertFalse(old.exists())
        self.assertTrue(Path(second["png"]).is_file())

    def test_generate_is_a_factory_job(self) -> None:
        self.asset("bale", trimesh.creation.box(), "workbench/clay.glb")
        from PIL import Image

        Image.new("RGB", (32, 32), (200, 160, 80)).save(self.root / "out" / "bale" / "photo.png")
        out, _ = self.ok("generate", {"stem": "bale", "image": "out/bale/photo.png", "mesh": "out/bale/workbench/clay.glb",
                                      "paint": False, "quality": "textured", "psx": {"facing": "right", "height": 40}})
        self.assertEqual(out["job_id"], "abcd1234")
        method, path, body = self.factory.calls[-1]
        self.assertEqual((method, path), ("POST", "/generate"))
        self.assertEqual(body, {"image": "out/bale/photo.png", "quality": "textured", "stem": "bale",
                                "mesh": "out/bale/workbench/clay.glb", "paint": False, "fill": "photo",
                                "facing": "right", "psx_height": "40"})
        # The workbench fills by default; fill none and mirror_ends pass through, and a draft (no bake) sends no fill.
        self.ok("generate", {"image": "out/bale/photo.png", "fill": "none", "mirror_ends": False})
        body = self.factory.calls[-1][2]
        self.assertEqual((body["fill"], body["mirror_ends"]), ("none", False))
        self.ok("generate", {"image": "out/bale/photo.png", "quality": "draft"})
        self.assertNotIn("fill", self.factory.calls[-1][2])
        self.assertTrue(self.call("generate", {"image": "out/bale/photo.png", "fill": "swatch"})["isError"])
        res = self.call("generate", {"image": "out/bale/photo.png", "psx": {"facing": "up"}})
        self.assertTrue(res["isError"])


class BusyStemTests(Base):
    BLOBS = [{"name": "b", "form": {"type": "Sphere", "radius": 0.5}}]

    def test_refuses_while_a_job_writes_the_stem(self) -> None:
        self.factory.state = {"ok": True, "state": "running", "stem": "bale", "job_id": "j1"}
        res = self.call("clay_set", {"stem": "bale", "blobs": self.BLOBS})
        self.assertTrue(res["isError"])
        self.assertIn("a factory job is writing bale", res["content"][0]["text"])
        self.assertFalse((self.root / "out" / "bale").exists())
        self.assertEqual(self.factory.snapshots(), [])

    MESH = {"stem": "bale", "grid": 16, "texture": 64}

    def test_another_stems_job_or_a_finished_one_does_not_block(self) -> None:
        for state in ({"state": "running", "stem": "sheep"}, {"state": "done", "stem": "bale"}):
            self.factory.state = {"ok": True, "job_id": "j1", **state}
            self.ok("clay_set", {"stem": "bale", "blobs": self.BLOBS})
            out, _ = self.ok("clay_mesh", self.MESH)
            self.assertTrue(out["take"]["ok"])

    def test_factory_down_writes_without_a_take(self) -> None:
        self.factory.down = True
        self.ok("clay_set", {"stem": "bale", "blobs": self.BLOBS})
        out, _ = self.ok("clay_mesh", self.MESH)
        self.assertFalse(out["take"]["ok"])
        self.assertIn("kept no take", out["take"]["error"])
        self.assertTrue((self.root / "out" / "bale" / "workbench" / "clay.json").is_file())
        res = self.call("job_status", {})
        self.assertTrue(res["isError"])


class HttpTests(Base):
    def setUp(self) -> None:
        super().setUp()
        self.httpd = mcp_server.make_server("127.0.0.1", 0, self.bench)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/mcp"

    def post(self, payload, headers=None, method="POST", raw=None):
        data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode())
        req = urllib.request.Request(self.url, data=data, method=method, headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_http_round_trip(self) -> None:
        status, body = self.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["result"]["tools"]), len(PLAN_TOOLS))
        status, body = self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual((status, body), (202, b""))
        status, body = self.post(None, raw=b"{nope")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["code"], mcp_server.PARSE_ERROR)
        status, _ = self.post(None, method="GET")
        self.assertEqual(status, 405)

    def test_non_local_origin_is_rejected(self) -> None:
        ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        status, _ = self.post(ping, {"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.post(ping, {"Origin": "http://127.0.0.1.evil.example:8768"})
        self.assertEqual(status, 403)
        status, body = self.post(ping, {"Origin": "http://localhost:5173"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["result"], {})


if __name__ == "__main__":
    unittest.main()
