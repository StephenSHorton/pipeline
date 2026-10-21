#!/usr/bin/env python3
"""Template IO, loader, verifier (scripts/rig_template.py) and the rig dump (scripts/rig_dump.py)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig  # noqa: E402
import rig_dump  # noqa: E402
import rig_plan  # noqa: E402
import rig_pred  # noqa: E402
import rig_template  # noqa: E402

PY = sys.executable
FIXED = {
    "root": (0, 0, 0), "hips": (0, 0.5, 0), "spine_0": (0, 0.55, 0.1), "spine_1": (0, 0.6, 0.2), "spine_2": (0, 0.7, 0.3),
    "spine_3": (0, 0.8, 0.35), "head": (0, 0.9, 0.4), "mouth_upper": (0, 0.9, 0.45), "mouth_upper_tip": (0, 0.9, 0.5),
    "mouth_lower": (0, 0.88, 0.45), "mouth_lower_tip": (0, 0.88, 0.49),
    "tail_1": (0, 0.55, -0.1), "tail_2": (0, 0.58, -0.2), "tail_3": (0, 0.6, -0.3), "tail_tip": (0, 0.62, -0.4),
}
LEG = {"UpperLeg": (0.1, 0.45, 0), "LowerLeg": (0.1, 0.3, 0.03), "AnkleLeg": (0.1, 0.15, -0.03), "Foot": (0.1, 0.02, 0),
       "Toes": (0.1, 0.01, 0.05), "Toes_tip": (0.1, 0.0, 0.1)}


def synth_skeleton(plan: dict) -> dict:
    """skeleton.json for the plan's joints: a stick bird in the build frame (+Z head, +X left, floor 0)."""
    pos: dict[str, np.ndarray] = {}
    joints = []
    for name, parent in rig_plan.joints(plan):
        side = rig_template.side_of(name)
        s = 1.0 if side == "L" else -1.0
        base = name[:-2] if side else name
        if name in FIXED:
            p = np.array(FIXED[name], float)
        elif base in LEG:
            p = np.array(LEG[base], float) * [s, 1, 1]
        elif base == "wing_1":
            p = pos[parent] + [0.15 * s, 0, 0]
        else:
            p = pos[parent] + [0.02 * s, -0.005, -0.03]
        pos[name] = p
        joints.append({"name": name, "parent": parent, "pos": p.tolist(), "role": "leg" if base in LEG else None,
                       "side": side, "source": "plan"})
    return {"joints": joints, "H": 1.0, "cx": 0.0}


def synth_mesh() -> tuple[dict, bytes]:
    """A box (x ±0.3, y 0..1, z ±0.5) GLB doc with POSITION, NORMAL, TEXCOORD_0 and indices."""
    P = np.array([[x, y, z] for x in (-0.3, 0.3) for y in (0.0, 1.0) for z in (-0.5, 0.5)], "<f4")
    N = (P - P.mean(0)) / np.linalg.norm(P - P.mean(0), axis=1, keepdims=True)
    UV = np.c_[P[:, 0] + 0.5, P[:, 2] + 0.5].astype("<f4")
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], "<u4")
    g = rig.GlbWriter()
    a = {"POSITION": g.array(P, "VEC3", 34962, True), "NORMAL": g.array(N.astype("<f4"), "VEC3", 34962),
         "TEXCOORD_0": g.array(UV, "VEC2", 34962)}
    g.doc["meshes"] = [{"primitives": [{"attributes": a, "indices": g.array(F.reshape(-1), "SCALAR", 34963)}]}]
    g.doc["nodes"] = [{"mesh": 0}]
    g.doc["scenes"] = [{"nodes": [0]}]
    return g.doc, b"".join(g.chunks)


def nearest_weights(P: np.ndarray, jp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(P[:, None] - jp[None], axis=2)
    J = np.zeros((len(P), 4), int)
    W = np.zeros((len(P), 4))
    J[:, 0] = d.argmin(1)
    W[:, 0] = 1.0
    return J, W


def synth_clips(plan: dict, nj: int, root: int, root_t0: np.ndarray, hips: int) -> list[dict]:
    out = []
    for c in plan["clips"]:
        T = float(c["T"])
        times = np.arange(0, T + 1e-6, 1 / 30).astype(np.float32)
        ph = 2 * np.pi * times / times[-1]
        rot = {}
        amp = np.radians(10)
        q = R.from_rotvec(np.outer(np.sin(ph) * amp if c["loop"] else np.minimum(times / T, 1) * amp, [1, 0, 0])).as_quat()
        rot[hips] = q.astype(np.float32)
        root_t = np.tile(root_t0, (len(times), 1)) + np.outer(np.sin(ph), [0, 0.01, 0])
        out.append({"name": c["name"], "times": times, "rot": rot, "root": root, "root_t": root_t.astype(np.float32),
                    "loop": c["loop"], "params": {"amp_deg": 10}, "energy": 1.0})
    return out


class Built:
    """A built template written into <tmp>/<stem>/rig/template/."""

    def __init__(self, base: Path, stem: str = "synth-bird") -> None:
        self.plan = rig_plan.load("ground_bird")
        self.asset = base / stem
        self.dir = self.asset / "rig" / "template"
        self.dir.mkdir(parents=True)
        self.skel = synth_skeleton(self.plan)
        (self.dir / "skeleton.json").write_text(json.dumps(self.skel))
        self.rig = self.dir / "rig.glb"
        rig_template.write_rig(self.dir / "skeleton.json", self.rig)
        self.src, self.sbin = synth_mesh()
        (self.asset / "psx.glb").write_bytes(b"psx")
        names = [j["name"] for j in self.skel["joints"]]
        self.names = names
        jp = np.array([j["pos"] for j in self.skel["joints"]])
        P = rig.accessor(self.src, self.sbin, 0).astype(float)
        self.J, self.W = nearest_weights(P, jp)
        self.clips = synth_clips(self.plan, len(names), names.index("root"), jp[names.index("root")], names.index("hips"))
        self.clips_glb = self.dir / "clips.glb"
        rig_template.write_clips(self.rig, self.src, self.sbin, self.J, self.W, self.clips, self.clips_glb)
        self.tj = rig_template.write_template_json(self.dir, self.plan, [c["name"] for c in self.clips])


class TemplateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.b = Built(self.base)

    def tearDown(self) -> None:
        for k in [k for k in rig.TEMPLATES if str(k).startswith("built:")]:
            del rig.TEMPLATES[k]
        self.tmp.cleanup()

    def test_rig_layout(self) -> None:
        doc, b = rig.read_glb(self.b.rig)
        self.assertEqual(len(doc["skins"]), 1)
        self.assertEqual(len(doc.get("meshes", [])), 0)
        self.assertEqual(len(doc.get("animations", [])), 0)
        joints = doc["skins"][0]["joints"]
        self.assertEqual(joints, list(range(55)))
        self.assertEqual(set(rig_plan.joint_names(self.b.plan)), {doc["nodes"][i]["name"] for i in joints})
        sk = rig.Skeleton(doc)
        roots = [i for i in joints if sk.parent.get(i) not in joints]
        self.assertEqual([doc["nodes"][i]["name"] for i in roots], ["root"])
        self.assertEqual(sk.parent[sk.index["hips"]], sk.index["root"])
        arm = doc["nodes"][sk.index["Armature"]]
        self.assertFalse({"translation", "rotation", "scale", "matrix"} & set(arm))
        for n in doc["nodes"]:
            self.assertNotIn("matrix", n)
            self.assertNotEqual(n.get("name"), "fit")
            self.assertNotIn("rotation", n)
        self.assertLess(rig_template.ibm_error(doc, b), 1e-5)
        acc = doc["accessors"][doc["skins"][0]["inverseBindMatrices"]]
        self.assertEqual((acc["componentType"], acc["type"]), (5126, "MAT4"))
        # world positions are the skeleton's
        w = sk.world({})
        for j in self.b.skel["joints"]:
            np.testing.assert_allclose(w[sk.index[j["name"]]][0], j["pos"], atol=1e-6)

    def test_verify_passes_and_writes_validate(self) -> None:
        probs = rig_template.verify(self.b.rig, self.b.clips_glb, self.b.plan)
        self.assertEqual([p for p in probs if p["level"] == "error"], [], probs)
        v = json.loads((self.b.dir / "validate.json").read_text())
        self.assertTrue(v["ok"])
        for k in ("ok", "problems", "joints", "ibm_err", "axes", "floor_y", "ref_present", "body"):
            self.assertIn(k, v)
        self.assertEqual(v["joints"], 55)
        self.assertTrue(v["ref_present"])
        self.assertEqual(v["body"], {"verts": 8, "skinned": True})
        self.assertLess(v["ibm_err"], 1e-5)

    def test_verify_catches_problems(self) -> None:
        # a broken loop, a missing default clip and a backwards head
        bad = [dict(c) for c in self.b.clips if c["name"] != "Death"]
        walk = next(c for c in bad if c["name"] == "Walk")
        hips = self.b.names.index("hips")
        walk["rot"] = {hips: R.from_rotvec(np.outer(np.linspace(0, 0.3, len(walk["times"])), [1, 0, 0])).as_quat().astype(np.float32)}
        out = self.b.dir / "bad_clips.glb"
        rig_template.write_clips(self.b.rig, self.b.src, self.b.sbin, self.b.J, self.b.W, bad, out)
        probs = rig_template.verify(self.b.rig, out, self.b.plan, out=False)
        msgs = " | ".join(p["msg"] for p in probs if p["level"] == "error")
        self.assertIn("Walk loops", msgs)
        self.assertIn("Death is missing", msgs)
        sk = json.loads(json.dumps(self.b.skel))
        for j in sk["joints"]:
            j["pos"][2] = -j["pos"][2]
            j["pos"][0] = -j["pos"][0]
        rig2 = self.b.dir / "rig_back.glb"
        rig_template.write_rig(sk, rig2)
        rig_template.write_clips(rig2, self.b.src, self.b.sbin, self.b.J, self.b.W, self.b.clips, self.b.dir / "c2.glb")
        keys = {p["key"] for p in rig_template.verify(rig2, self.b.dir / "c2.glb", self.b.plan, out=False) if p["level"] == "error"}
        self.assertIn("axes", keys)
        # a matrix node and a fit node
        posed = self.b.dir.parent / "template_posed.glb"
        rig_template.write_posed(self.b.rig, posed)
        keys = {p["key"] for p in rig_template.verify(posed, self.b.clips_glb, self.b.plan, out=False) if p["level"] == "error"}
        self.assertIn("fit", keys)

    def test_clips_glb(self) -> None:
        doc, b = rig.read_glb(self.b.clips_glb)
        rdoc, rb = rig.read_glb(self.b.rig)
        self.assertEqual([doc["nodes"][i]["name"] for i in doc["skins"][0]["joints"]],
                         [rdoc["nodes"][i]["name"] for i in rdoc["skins"][0]["joints"]])
        np.testing.assert_array_equal(rig.accessor(doc, b, doc["skins"][0]["inverseBindMatrices"]),
                                      rig.accessor(rdoc, rb, rdoc["skins"][0]["inverseBindMatrices"]))
        body = [n for n in doc["nodes"] if "mesh" in n and "skin" in n]
        self.assertEqual([n["name"] for n in body], ["Body"])
        pr = doc["meshes"][0]["primitives"][0]
        self.assertIn("indices", pr)
        self.assertTrue({"POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"} <= set(pr["attributes"]))
        names = [a["name"] for a in doc["animations"]]
        self.assertEqual(names, ["Rest"] + self.b.plan["default"])
        rest = doc["animations"][0]
        self.assertAlmostEqual(rig.clip_duration(doc, b, rest), 1 / 30, places=6)
        self.assertEqual(len(rig.accessor(doc, b, rest["samplers"][0]["input"])), 2)
        root = rig.Skeleton(doc).index["root"]
        for an in doc["animations"]:
            rot = {c["target"]["node"] for c in an["channels"] if c["target"]["path"] == "rotation"}
            tr = {c["target"]["node"] for c in an["channels"] if c["target"]["path"] == "translation"}
            self.assertEqual(rot, set(doc["skins"][0]["joints"]))
            self.assertEqual(tr, {root})
            self.assertFalse(any(c["target"]["path"] == "scale" for c in an["channels"]))
            self.assertTrue(all(s["interpolation"] == "LINEAR" for s in an["samplers"]))

    def test_ref_store_and_bake_round_trip(self) -> None:
        store = rig_template.ref_store(self.b.rig, {"flags": "abcdef", "gate": {"level": "ok"}})
        fit = store["__fit__"]
        self.assertEqual((fit["yaw"], fit["scale"], fit["offset"], fit["unsupported"]), (0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], []))
        self.assertEqual(fit["registration"], {"built": True, "scale_ratio": 1.0, "poor_match": False})
        self.assertEqual(fit["gate"], {"level": "ok"})
        self.assertEqual(store["hips"][1], [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(all(n in store for n in self.b.names))
        tpl = rig_template.load_template(self.b.asset)
        # ref_local samples frame 0 of 'Rest' from clips[0]
        ours, obin = rig.read_glb(self.b.clips_glb)
        sk = rig.Skeleton(ours)
        local = rig.ref_local(tpl, rig.Skeleton(rig.read_glb(Path(tpl["rig"]))[0]))
        self.assertTrue(all(np.allclose(local[i][1], [0, 0, 0, 1]) for i in range(55)))
        names = tpl["default"]
        baked, missing = rig.bake_clips(tpl, names, ours, sk, store)
        self.assertEqual(missing, [])
        by = {c["name"]: c for c in self.b.clips}
        hips = sk.index["hips"]
        for c in baked:
            src = by[c["name"]]
            self.assertEqual(len(c["times"]), len(src["times"]))
            np.testing.assert_allclose(np.abs(np.sum(c["rot"][hips] * src["rot"][hips], 1)), 1.0, atol=1e-4)
            np.testing.assert_allclose(c["root_t"], src["root_t"], atol=1e-4)
            for i in range(55):
                if i != hips:
                    np.testing.assert_allclose(np.abs(c["rot"][i][:, 3]), 1.0, atol=1e-4)

    def test_write_clips_remaps_by_name(self) -> None:
        # the caller's (skinned_raw) order is reversed: J and rot keys follow it
        order = self.b.names[::-1]
        k_of = {n: k for k, n in enumerate(order)}
        J2 = np.vectorize(lambda j: k_of[self.b.names[j]])(self.b.J)
        clips = []
        for c in self.b.clips:
            c2 = dict(c)
            c2["rot"] = {k_of[self.b.names[k]]: v for k, v in c["rot"].items()}
            c2["root"] = k_of["root"]
            clips.append(c2)
        out = self.b.dir / "remap.glb"
        rig_template.write_clips(self.b.rig, self.b.src, self.b.sbin, J2, self.b.W, clips, out, names=order)
        d1, b1 = rig.read_glb(self.b.clips_glb)
        d2, b2 = rig.read_glb(out)
        a1 = d1["meshes"][0]["primitives"][0]["attributes"]
        a2 = d2["meshes"][0]["primitives"][0]["attributes"]
        np.testing.assert_array_equal(rig.accessor(d1, b1, a1["JOINTS_0"]), rig.accessor(d2, b2, a2["JOINTS_0"]))
        walk1 = rig_dump.dump(self.b.clips_glb, clip="Walk")["clips"][0]["rotation_range_deg"]
        walk2 = rig_dump.dump(out, clip="Walk")["clips"][0]["rotation_range_deg"]
        self.assertEqual(walk1, walk2)

    def test_write_posed(self) -> None:
        posed = self.b.dir.parent / "template_posed.glb"
        rig_template.write_posed(self.b.rig, posed)
        doc, b = rig.read_glb(posed)
        fit = doc["nodes"][-1]
        self.assertEqual(fit["name"], "fit")
        self.assertEqual(fit["extras"], {"yaw_deg": 0, "scale": [1.0, 1.0, 1.0], "offset": [0.0, 0.0, 0.0], "built": True})
        self.assertEqual(doc["scenes"][0]["nodes"], [len(doc["nodes"]) - 1])
        sk = rig.Skeleton(doc)
        self.assertNotIn(sk.index["root"], [i for i in range(len(doc["nodes"])) if i not in sk.parent])
        self.assertLess(rig_template.ibm_error(doc, b), 1e-5)
        ours, _ = rig_pred.ours_from_posed(posed)  # the consumer that rebuilds a skin-tokens-like doc
        self.assertEqual(ours["nodes"][0]["name"], "root")
        self.assertEqual(len(ours["skins"][0]["joints"]), 55)

    def test_template_json_and_loader_identity(self) -> None:
        tj = json.loads((self.b.dir / "template.json").read_text())
        self.assertEqual(tj["ref"], "Rest")
        self.assertTrue(tj["built"])
        self.assertEqual(tj["plan"], "ground_bird")
        self.assertEqual(tj["class"], "avian")
        self.assertEqual(tj["default"], self.b.plan["default"])
        self.assertTrue(Path(tj["rig"]).is_absolute() and Path(tj["clips"][0]).is_file())
        self.assertEqual(rig.M2M / tj["rig"], Path(tj["rig"]))
        self.assertEqual(len(tj["built_for"]), 32)
        t1 = rig_template.load_template(self.b.asset)
        self.assertIs(rig.TEMPLATES["built:synth-bird"], t1)
        self.assertIs(rig_template.load_template(self.b.asset), t1)
        self.assertEqual(next(k for k, v in rig.TEMPLATES.items() if v is t1), "built:synth-bird")  # _template_class-style lookup
        self.assertIsNone(rig_template.load_template(self.base / "nothing-here"))
        # template_for
        self.assertIs(rig_template.template_for({"class": "avian", "template": {"mode": "built", "built": True}}, self.b.asset), t1)
        self.assertIs(rig_template.template_for({"class": "avian", "template": {"mode": "fixed"}}, self.b.asset), rig.TEMPLATES["avian"])
        self.assertIs(rig_template.template_for({"class": "quadruped"}, self.b.asset), rig.TEMPLATES["quadruped"])
        with self.assertRaises(FileNotFoundError):
            rig_template.template_for({"class": "avian", "template": {"built": True}}, self.base / "nothing-here")

    def test_floor_track_reads_built_clips(self) -> None:
        tpl = rig_template.load_template(self.b.asset)
        times = {"Peck": self.b.clips[3]["times"]}
        tr = rig_pred.template_floor_track(tpl, ["Peck"], ["Foot_L", "Foot_R"], times)
        self.assertIn("Peck", tr["clips"])
        self.assertAlmostEqual(tr["H"], 1.0, places=5)

    def test_dump_built(self) -> None:
        d = rig_dump.dump(self.b.rig, self.b.clips_glb)
        self.assertEqual(d["errors"], [])
        self.assertEqual(d["roots"], ["root"])
        self.assertEqual(d["joint_count"], 55)
        self.assertLess(d["mirror"]["max_H"], 1e-6)
        self.assertLess(d["ibm"]["max_err"], 1e-5)
        by = {j["name"]: j for j in d["joints"]}
        self.assertEqual(by["Foot_L"]["label"], "leg")
        self.assertEqual(by["Foot_L"]["label_from"], "extras")
        self.assertEqual(by["wing_1_L"]["side"], "L")
        clips = {c["name"]: c for c in d["clips"]}
        self.assertEqual(set(clips), {"Rest"} | set(self.b.plan["default"]))
        self.assertAlmostEqual(clips["Walk"]["fps"], 30.0, places=1)
        self.assertLess(clips["Walk"]["loop_gap_deg"], 0.5)
        self.assertGreater(clips["Death"]["loop_gap_deg"], 5)
        self.assertIn(rig_dump.text(d).splitlines()[0], str(self.b.rig))


class CliAndVendorTest(unittest.TestCase):
    def test_vendor_rigs_dump_clean(self) -> None:
        for rig_f, anim in (("rig-bird.glb", "bird-animations.glb"), ("rig-horse.glb", "horse-animations.glb")):
            d = rig_dump.dump(rig.M2M / "rigs" / rig_f, rig.M2M / "animations" / anim)
            self.assertEqual(d["errors"], [], rig_f)
            self.assertEqual(d["roots"], ["root"])
            self.assertGreater(len(d["clips"]), 3)
            self.assertTrue(rig_dump.text(d))
        d = rig_dump.dump(rig.M2M / "rigs" / "rig-bird.glb")
        self.assertEqual(d["joint_count"], 55)
        self.assertLess(d["mirror"]["max_H"], 0.02)

    def test_cli(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            b = Built(Path(tmp.name), "cli-bird")
            r = subprocess.run([PY, str(ROOT / "scripts" / "rig_dump.py"), str(rig.M2M / "rigs" / "rig-bird.glb"), "--json"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["joint_count"], 55)
            r = subprocess.run([PY, str(ROOT / "scripts" / "rig_dump.py"), str(b.rig), "--clips", str(b.clips_glb), "--clip", "Walk"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("clip Walk", r.stdout)
            (b.asset / "rig" / "plan.json").write_text(json.dumps(b.plan))
            r = subprocess.run([PY, str(ROOT / "scripts" / "rig_template.py"), "verify", "--dir", str(b.dir)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(json.loads((b.dir / "validate.json").read_text())["ok"])
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
