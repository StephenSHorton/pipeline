#!/usr/bin/env python3
"""scripts/workbench/rigeyes.py (the rig_review tool) on a small four-joint-chain creature built here: stretch is
charged to the bones that own it, the pivot check tells a mid-body hind pivot from one over its leg, and the heat
sheet leaves no files behind. Plus the rig job tool, the playbook digest the MCP server sends, and the narration."""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig  # noqa: E402
from workbench import mcp_server, narrate, rigeyes, tools  # noqa: E402
from workbench.factory_client import FactoryClient  # noqa: E402

# A creature 1.35 long (z), facing +Z: hips at the back, a spine joint at the front, a head, and two hind legs that
# hang from the hips through a pivot (pelvis_*) like the horse template's. PIVOT_Z moves the pivots: over the legs
# (z = -0.4, where the legs are) or mid-body (z = +0.2), the sheep's broken rig.
LEG_Z = -0.4
NAMES = ["root", "hips", "spine", "head", "pelvis_L", "leg_L", "foot_L", "pelvis_R", "leg_R", "foot_R"]
PARENT = [-1, 0, 1, 2, 1, 4, 5, 1, 7, 8]


def joints(pivot_z: float) -> np.ndarray:
    return np.array([[0, 0, 0], [0, 0.6, -0.4], [0, 0.62, 0.4], [0, 0.85, 0.75],
                     [0.15, 0.6, pivot_z], [0.15, 0.55, LEG_Z], [0.15, 0.0, LEG_Z],
                     [-0.15, 0.6, pivot_z], [-0.15, 0.55, LEG_Z], [-0.15, 0.0, LEG_Z]], float)


def tube(a, b, rings: int, r: float, ring: int = 8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    u = np.cross(ax, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(ax, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ax / np.linalg.norm(ax), u)
    P = [a + ax * i / (rings - 1) + r * (np.cos(t) * u + np.sin(t) * v)
         for i in range(rings) for t in 2 * np.pi * np.arange(ring) / ring]
    F = []
    for i in range(rings - 1):
        for k in range(ring):
            p0, p1 = i * ring + k, i * ring + (k + 1) % ring
            F += [[p0, p1, p0 + ring], [p1, p1 + ring, p0 + ring]]
    return np.array(P), np.array(F), np.repeat(np.arange(rings), ring)


def write_creature(path: Path, pivot_z: float) -> None:
    """Rigid weights: the body tube to hips (back half) and spine (front half), the head to head, each leg tube to
    its leg bone except its top ring, which the hips own (the seam). One 'Walk' clip swings pelvis_L 40 degrees."""
    JP = joints(pivot_z)
    parts = []
    P, F, ringno = tube([0, 0.6, -0.55], [0, 0.6, 0.6], 12, 0.15)
    parts.append((P, F, np.where(P[:, 2] < 0, 1, 2)))
    P, F, _ = tube([0, 0.62, 0.55], [0, 0.85, 0.85], 4, 0.08)
    parts.append((P, F, np.full(len(P), 3)))
    for leg in (5, 8):
        P, F, ringno = tube(JP[leg], JP[leg] * [1, 0, 1], 12, 0.05)  # down to the floor
        parts.append((P, F, np.where(ringno == 0, 1, leg)))
    Ps, Fs, Js, off = [], [], [], 0
    for P, F, bone in parts:
        Ps.append(P)
        Fs.append(F + off)
        Js.append(bone)
        off += len(P)
    P = np.concatenate(Ps).astype("<f4")
    F = np.concatenate(Fs).astype("<u4")
    J = np.zeros((len(P), 4), "u1")
    J[:, 0] = np.concatenate(Js)
    W = np.zeros((len(P), 4), "<f4")
    W[:, 0] = 1
    g = rig.GlbWriter()
    d = g.doc
    buf = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), (200, 200, 190), np.uint8)).save(buf, "PNG")
    d["images"] = [{"bufferView": g.blob(buf.getvalue()), "mimeType": "image/png"}]
    d["samplers"] = [{"magFilter": 9728, "minFilter": 9728}]
    d["textures"] = [{"source": 0, "sampler": 0}]
    d["materials"] = [{"name": "m", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0}}]
    attrs = {"POSITION": g.array(P, "VEC3", 34962, minmax=True),
             "TEXCOORD_0": g.array(np.full((len(P), 2), 0.25, "<f4"), "VEC2", 34962),
             "JOINTS_0": g.array(J, "VEC4", 34962), "WEIGHTS_0": g.array(W, "VEC4", 34962)}
    d["meshes"] = [{"name": "Body", "primitives": [{"attributes": attrs, "indices": g.array(F.reshape(-1), "SCALAR", 34963),
                                                    "material": 0}]}]
    nodes = []
    for k, n in enumerate(NAMES):
        t = JP[k] - (JP[PARENT[k]] if PARENT[k] >= 0 else 0)
        node = {"name": n, "translation": [float(x) for x in t]}
        kids = [c for c in range(len(NAMES)) if PARENT[c] == k]
        if kids:
            node["children"] = kids
        nodes.append(node)
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})
    d["nodes"] = nodes
    ibm = np.array([np.linalg.inv(np.block([[np.eye(3), JP[k][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])).T.reshape(16)
                    for k in range(len(NAMES))], "<f4")
    d["skins"] = [{"joints": list(range(len(NAMES))), "inverseBindMatrices": g.array(ibm, "MAT4"), "skeleton": 0}]
    times = np.array([0, 0.25, 0.5, 0.75, 1.0], "<f4")[:, None]
    q = np.array([R.from_euler("x", a, degrees=True).as_quat() for a in [0, 40, 0, -40, 0]], "<f4")
    ti = g.array(times, "SCALAR")
    d["accessors"][ti]["min"], d["accessors"][ti]["max"] = [0.0], [1.0]
    d["animations"] = [{"name": "Walk", "samplers": [{"input": ti, "output": g.array(q, "VEC4"), "interpolation": "LINEAR"}],
                        "channels": [{"sampler": 0, "target": {"node": 4, "path": "rotation"}}]}]
    d["scenes"] = [{"nodes": [0, len(NAMES)]}]
    d["scene"] = 0
    g.write(path)


def files(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


class Creature(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="rigeyes_test_"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def asset(self, stem: str, pivot_z: float) -> Path:
        folder = self.root / "out" / stem
        (folder / "rig").mkdir(parents=True)
        write_creature(folder / "rig" / "rigged_allclips.glb", pivot_z)
        return folder


class Stretch(Creature):
    def test_stretch_goes_to_the_bones_that_own_it(self) -> None:
        res = rigeyes.review(self.asset("mid", 0.2))
        walk = res["clips"]["Walk"]
        self.assertGreater(walk["over"], 0)
        self.assertEqual(set(walk["bones"]), {"hips", "leg_L"})  # the seam ring (hips) against the swung leg
        self.assertIn("leg_L", walk["worst_frame"])
        self.assertGreater(walk["max"], rigeyes.THRESHOLD)

    def test_weights_and_glb(self) -> None:
        folder = self.asset("w", LEG_Z)
        (folder / "rigged.glb").write_bytes((folder / "rig" / "rigged_allclips.glb").read_bytes())
        self.assertEqual(rigeyes.rig_glb(folder), folder / "rig" / "rigged_allclips.glb")  # what the viewer shows
        res = rigeyes.review(folder)
        self.assertGreater(res["weights"]["hips"], res["weights"]["spine"])
        self.assertNotIn("root", res["weights"])  # owns nothing: under the share shown
        self.assertIsNone(res["check"])  # no rig/inspect.json
        with self.assertRaises(rigeyes.RigError):
            rigeyes.review(folder, clips=["Gallop"])


class Pivot(Creature):
    def test_mid_body_pivot_is_flagged(self) -> None:
        res = rigeyes.review(self.asset("mid", 0.2))
        legs = {lg["root"]: lg for lg in res["pivots"]["legs"]}
        self.assertEqual(set(legs), {"pelvis_L", "pelvis_R"})
        self.assertFalse(legs["pelvis_L"]["ok"])
        self.assertGreater(legs["pelvis_L"]["off_leg"], rigeyes.PIVOT_MAX)
        self.assertTrue(any("pelvis_L" in n and "pivot" in n for n in res["diagnosis"]), res["diagnosis"])
        self.assertIn("leg_L", res["flagged"])

    def test_pivot_over_its_leg_passes(self) -> None:
        good = rigeyes.review(self.asset("good", LEG_Z))
        self.assertTrue(all(lg["ok"] for lg in good["pivots"]["legs"]), good["pivots"])
        self.assertTrue(all(lg["off_leg"] < 0.05 for lg in good["pivots"]["legs"]))
        self.assertFalse(any("from its leg" in n for n in good["diagnosis"]), good["diagnosis"])
        # the same swing about a pivot over the leg stretches far less than about one mid-body
        bad = rigeyes.review(self.asset("bad", 0.2))
        self.assertLess(good["clips"]["Walk"]["over"], bad["clips"]["Walk"]["over"])


class Diagnosis(unittest.TestCase):
    """A thigh above the leg line owns nothing by design: the diagnosis says so instead of asking for its weight."""

    def legs(self, above: float) -> dict:
        return {"legs": [{"root": "pelvis", "upper": "thigh", "part": "hind legs", "attach": "hips", "off_leg": 0.1,
                          "ok": True, "upper_above_line": above}]}

    def stretch(self) -> dict:
        return {"Walk": {"over": 12, "bones": {"hips": 9, "shin": 4}}}

    def test_thigh_above_the_line_is_by_design(self) -> None:
        ch = {"part": {"thigh": "hind legs", "shin": "hind legs", "pelvis": "hind legs"}}
        notes, _ = rigeyes.diagnose(self.stretch(), {"hips": 50.0}, self.legs(0.13), ch, 2.0)
        self.assertTrue(any("by design" in n and "don't move weight onto the thigh" in n for n in notes), notes)
        self.assertTrue(any("expected, not a weights bug" in n for n in notes), notes)

    def test_thigh_below_the_line_owning_nothing_is_a_finding(self) -> None:
        ch = {"part": {"thigh": "hind legs", "shin": "hind legs", "pelvis": "hind legs"}}
        notes, flagged = rigeyes.diagnose(self.stretch(), {"hips": 50.0}, self.legs(-0.05), ch, 2.0)
        self.assertTrue(any("bends with whatever owns it" in n for n in notes), notes)
        self.assertIn("thigh", flagged)


class Heat(Creature):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = self.root / "tmp"
        self.tmp.mkdir()
        patch = mock.patch.object(tempfile, "tempdir", str(self.tmp))  # every mkdtemp lands where we can see it
        patch.start()
        self.addCleanup(patch.stop)

    def test_no_heat_writes_nothing(self) -> None:
        folder = self.asset("mid", 0.2)
        before = files(folder)
        res = rigeyes.review(folder)
        self.assertNotIn("sheet", res)
        self.assertEqual(files(folder), before)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_heat_leaves_only_the_sheet(self) -> None:
        folder = self.asset("mid", 0.2)
        before = files(folder)
        sheets = self.root / "sheets"
        sheets.mkdir()
        res = rigeyes.review(folder, heat="flagged", sheet_dir=sheets)
        self.assertTrue(Path(res["sheet"]).is_file())
        self.assertEqual(Path(res["sheet"]).parent, sheets)
        self.assertEqual(set(res["heat"]["bones"]), set(res["flagged"]))
        self.assertEqual(res["heat"]["bones"]["leg_L"]["clip"], "Walk")
        with Image.open(res["sheet"]) as im:
            self.assertEqual(im.size[0], 2 * len(rigeyes.HEAT_VIEWS) * rigeyes.HEAT_TILE)
        self.assertEqual(files(folder), before)  # nothing into the asset
        self.assertEqual(list(self.tmp.iterdir()), [])  # the frames are gone
        self.assertNotIn("heat_frames", res)

    def test_keep_images_keeps_the_frames(self) -> None:
        res = rigeyes.review(self.asset("mid", 0.2), heat=["leg_L"], keep_images=True, sheet_dir=self.root)
        frames = Path(res["heat_frames"])
        self.assertTrue(any(p.suffix == ".png" for p in frames.iterdir()))
        with self.assertRaises(rigeyes.RigError):
            rigeyes.review(self.asset("x", 0.2), heat=["tail_9"])


class StubFactory(FactoryClient):
    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:1")
        self.calls: list[tuple[str, str, dict | None]] = []

    def _call(self, method, path, body=None, timeout=None):
        self.calls.append((method, path, body))
        if path.startswith("/rig/"):
            return {"ok": True, "accepted": True, "job_id": "r1g00001", "state": "running", "kind": "rig"}
        if path.startswith("/status"):
            return {"ok": True, "state": "idle"}
        return {"ok": True}


class Tools(Creature):
    def setUp(self) -> None:
        super().setUp()
        self.factory = StubFactory()
        self.bench = tools.Workbench(self.root, self.factory)

    def test_rig_review_tool(self) -> None:
        folder = self.asset("sheepish", 0.2)
        before = files(folder)
        out = self.bench.call("rig_review", {"stem": "sheepish", "heat": "flagged"})
        self.assertEqual(out["glb"], "out/sheepish/rig/rigged_allclips.glb")
        self.assertTrue(Path(out["_image"]).is_file())
        self.assertNotIn("png", out)  # the narrator copies a png into out/_logs; the heat sheet stays in temp
        self.assertEqual(files(folder), before)
        self.assertFalse((self.root / "out" / "_logs").exists())
        res = mcp_server.call_tool(self.bench, {"name": "rig_review", "arguments": {"stem": "sheepish"}})
        self.assertFalse(res["isError"])
        self.assertIn("diagnosis", res["content"][0]["text"])
        bad = mcp_server.call_tool(self.bench, {"name": "rig_review", "arguments": {"stem": "sheepish", "heat": "all"}})
        self.assertTrue(bad["isError"])
        (self.root / "out" / "bare").mkdir()
        none = mcp_server.call_tool(self.bench, {"name": "rig_review", "arguments": {"stem": "bare"}})
        self.assertTrue(none["isError"])
        self.assertIn("no rig yet", none["content"][0]["text"])

    def test_rig_job(self) -> None:
        self.asset("sheepish", LEG_Z)
        out = self.bench.call("rig", {"stem": "sheepish", "class": "quadruped", "reuse_skin": True})
        self.assertEqual(out["job_id"], "r1g00001")
        post = [c for c in self.factory.calls if c[1].startswith("/rig/")]
        self.assertEqual(post, [("POST", "/rig/sheepish", {"class": "quadruped", "reuse_skin": True})])
        self.assertEqual(self.bench.tools["rig"].kind, "job")

    def test_narration(self) -> None:
        for tool in ("rig", "rig_review", "playbook"):
            self.assertIn(tool, narrate.TITLES)
        line = narrate.detail_of("rig_review", {"threshold": 2.0, "clips": {"Walk": {"over": 13}, "Idle": {"over": 0}},
                                                "pivots": {"legs": [{"root": "pelvis_L", "ok": False}]},
                                                "flagged": ["hips", "leg_L"]})
        self.assertEqual(line, "Walk: 13 edges past 2x · pivot off its leg: pelvis_L · look at hips, leg_L")
        self.assertEqual(narrate.detail_of("rig", {"job_id": "r1g00001"}), "job r1g00001")


class Playbook(unittest.TestCase):
    def test_instructions_carry_the_digest(self) -> None:
        digest = tools.playbook_digest()
        self.assertTrue(digest)
        self.assertEqual(mcp_server.INSTRUCTIONS, digest)
        self.assertLess(len(mcp_server.INSTRUCTIONS), 3800)  # Claude Code cuts server instructions at 4,096
        for word in ("rig_review", "playbook", "ONE model", "take", "Compare"):
            self.assertIn(word, digest)
        init = mcp_server.handle_rpc(tools.Workbench(Path(tempfile.gettempdir()), StubFactory()),
                                     {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(init["result"]["instructions"], digest)

    def test_playbook_tool_returns_the_file(self) -> None:
        bench = tools.Workbench(Path(tempfile.gettempdir()), StubFactory())
        text = bench.call("playbook", {})["playbook"]
        self.assertEqual(text, tools.PLAYBOOK.read_text(encoding="utf-8"))
        self.assertIn(tools.playbook_digest(), text)
        self.assertLess(len(text.split()), 2400)

    def test_digest_needs_its_markers(self) -> None:
        self.assertEqual(tools.playbook_digest("no markers here"), "")
        self.assertEqual(tools.playbook_digest("a <!-- digest --> rules <!-- /digest --> b"), "rules")
        with mock.patch.object(tools, "PLAYBOOK", Path(os.devnull + "_missing.md")):
            self.assertEqual(tools.playbook_digest(), "")
            with self.assertRaises(FileNotFoundError):
                tools.playbook_text()

    def test_mirror_ships_the_playbook(self) -> None:
        js = (ROOT / "desktop" / "frontend" / "scripts" / "copy-factory.mjs").read_text()
        self.assertIn('".md"', js)


if __name__ == "__main__":
    unittest.main()
