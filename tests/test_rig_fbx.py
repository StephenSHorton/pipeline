#!/usr/bin/env python3
"""scripts/rig_fbx.py: the FBX export + re-import check (Blender tests are skipped without Blender)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig  # noqa: E402
import rig_fbx  # noqa: E402

BLENDER = rig.blender_path()

# A tiny rigged GLB: a 2-bone armature skinning a subdivided column, with two actions (Idle, Walk) at 30 fps.
MAKE_GLB = """
import bpy, sys, math
out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = 30
bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.1, depth=1.0, location=(0, 0, 0.5))
me = bpy.context.object
me.name = "Body"
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.subdivide(number_cuts=6)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.armature_add(location=(0, 0, 0))
arm = bpy.context.object
arm.name = "Armature"
bpy.ops.object.mode_set(mode="EDIT")
eb = arm.data.edit_bones
b0 = eb[0]; b0.name = "root"; b0.head = (0, 0, 0); b0.tail = (0, 0, 0.5)
b1 = eb.new("hips"); b1.head = (0, 0, 0.5); b1.tail = (0, 0, 1.0); b1.parent = b0
bpy.ops.object.mode_set(mode="OBJECT")
g0 = me.vertex_groups.new(name="root"); g1 = me.vertex_groups.new(name="hips")
for v in me.data.vertices:
    t = max(0.0, min(1.0, v.co.z + 0.5))
    g0.add([v.index], 1.0 - t, "REPLACE"); g1.add([v.index], t, "REPLACE")
mod = me.modifiers.new("Armature", "ARMATURE"); mod.object = arm
me.parent = arm
arm.animation_data_create()
for name, n, ang in (("Idle", 96, 0.1), ("Walk", 12, 0.4)):  # Idle 3.2 s: a 24 fps import would show
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    arm.animation_data.action = act
    pb = arm.pose.bones["hips"]
    pb.rotation_mode = "XYZ"
    for f, a in ((1, 0.0), (1 + n // 2, ang), (1 + n, 0.0)):
        pb.rotation_euler = (a, 0, 0)
        pb.keyframe_insert("rotation_euler", frame=f)
    pr = arm.pose.bones["root"]; pr.rotation_mode = "XYZ"
    pr.rotation_euler = (0, 0, 0); pr.keyframe_insert("rotation_euler", frame=1)
    tr = arm.animation_data.nla_tracks.new(); tr.name = name
    tr.strips.new(name, 1, act)
    arm.animation_data.action = None
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_animations=True,
                          export_animation_mode="NLA_TRACKS")
print("GLB_OK")
"""


class EvaluateTests(unittest.TestCase):
    """The pass/fail rules, on hand-written FBX_CHECK reports (no Blender)."""

    def report(self, **kw):
        r = {"armatures": 1, "meshes": 1, "scene_fps": 30.0, "bones": 2, "bone_names": ["root", "hips"],
             "mesh_vertices": 100, "max_influences": 2,
             "actions": [{"name": "rig|rig|Idle", "frame_start": 1, "frame_end": 31, "height_first": 1.0, "height_mid": 1.0},
                         {"name": "rig|rig|Death", "frame_start": 1, "frame_end": 40, "height_first": 1.0, "height_mid": 0.2}]}
        r.update(kw)
        return r

    def test_clean_report_passes(self) -> None:
        problems, warnings = rig_fbx.evaluate(self.report(), ["Idle", "Death"], ["root", "hips"])
        self.assertEqual(problems, [])
        self.assertEqual(warnings, [])  # Death lies down: its height drop is not a warning

    def test_failures(self) -> None:
        r = self.report(scene_fps=24.0, max_influences=6, bone_names=["root", "spine"])
        problems, _ = rig_fbx.evaluate(r, ["Idle", "Walk", "Death"], ["root", "hips"])
        text = " | ".join(problems)
        self.assertIn("2 actions in the FBX, 3 clips shipped", text)
        self.assertIn("missing Walk", text)
        self.assertIn("bone names differ", text)
        self.assertIn("6 weights", text)
        self.assertIn("scene fps 24", text)

    def test_duration_mismatch(self) -> None:
        # Idle 3.2 s imported at 24 fps but declared 30: frame_end 77 -> 2.53 s in the FBX
        r = self.report(actions=[{"name": "rig|Idle", "frame_start": 1, "frame_end": 77, "height_first": 1.0, "height_mid": 1.0}])
        problems, _ = rig_fbx.evaluate(r, ["Idle"], ["root", "hips"], {"Idle": 3.2})
        self.assertTrue(any("wrong frame rate" in p for p in problems), problems)
        r = self.report(actions=[{"name": "rig|Idle", "frame_start": 1, "frame_end": 97, "height_first": 1.0, "height_mid": 1.0}])
        problems, _ = rig_fbx.evaluate(r, ["Idle"], ["root", "hips"], {"Idle": 3.2})
        self.assertEqual(problems, [])

    def test_height_warning_for_standing_clip(self) -> None:
        r = self.report(actions=[{"name": "Walk", "frame_start": 1, "frame_end": 13, "height_first": 1.0, "height_mid": 0.3}])
        problems, warnings = rig_fbx.evaluate(r, ["Walk"], ["root", "hips"])
        self.assertEqual(problems, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("Walk", warnings[0])

    def test_note(self) -> None:
        self.assertEqual(rig_fbx.note_for({"ok": True, "actions": [1] * 7, "bones": 55, "max_influences": 4, "scene_fps": 30.0}),
                         "FBX checked: 7 clips, 55 bones, 4 weights max, 30 fps")
        self.assertTrue(rig_fbx.note_for({"ok": False, "problems": ["x"]}).startswith("FBX check failed: x"))

    def test_no_blender(self) -> None:
        import unittest.mock as mock
        with mock.patch.object(rig, "blender_path", return_value=None):
            note, check = rig_fbx.export_and_check(Path("nope.glb"), Path("nope.fbx"), ["Idle"])
        self.assertIsNone(check)
        self.assertIn("Blender not found", note)


@unittest.skipUnless(BLENDER, "Blender is not installed")
class BlenderRoundTripTests(unittest.TestCase):
    def test_export_and_check_tiny_rig(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            glb = tmp / "rigged.glb"
            script = tmp / "make.py"
            script.write_text(MAKE_GLB)
            proc = subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(script), "--", str(glb)],
                                  capture_output=True, text=True, timeout=300)
            self.assertTrue(glb.is_file(), proc.stdout[-800:] + proc.stderr[-800:])
            joints, anims = rig_fbx.glb_facts(glb)
            self.assertEqual(sorted(anims), ["Idle", "Walk"])
            fbx = tmp / "rig" / "tiny.fbx"
            note, check = rig_fbx.export_and_check(glb, fbx, ["Idle", "Walk"])
            self.assertTrue(fbx.is_file(), note)
            self.assertTrue(check["ok"], check.get("problems"))
            self.assertEqual(len(check["actions"]), 2)
            self.assertEqual(sorted(rig_fbx.clip_name(a["name"]) for a in check["actions"]), ["Idle", "Walk"])
            self.assertEqual(check["scene_fps"], 30.0)
            durs = rig_fbx.glb_durations(glb)
            for act in check["actions"]:  # the frames are 30 a second too, not the importer's default 24
                want = durs[rig_fbx.clip_name(act["name"])]
                self.assertAlmostEqual((act["frame_end"] - act["frame_start"]) / 30.0, want, delta=1 / 30 + 1e-6)
            self.assertLessEqual(check["max_influences"], 4)
            self.assertEqual(set(check["bone_names"]), set(joints))
            saved = json.loads((tmp / "rig" / "fbx_check.json").read_text())
            self.assertTrue(saved["ok"])
            self.assertTrue(note.startswith("FBX checked: 2 clips"), note)
            # --check-only re-reads the same FBX without exporting
            note2, check2 = rig_fbx.export_and_check(glb, fbx, ["Idle", "Walk"], check_only=True)
            self.assertTrue(check2["ok"], check2.get("problems"))
            self.assertTrue(check2["check_only"])


if __name__ == "__main__":
    unittest.main()
