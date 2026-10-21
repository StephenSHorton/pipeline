"""Blender side of scripts/rig_fbx.py: write the s&box FBX from rigged.glb, then re-import it and report what shipped.

Run as:  blender -b --factory-startup -P scripts/blender/fbx_check.py -- <rigged.glb> <out.fbx> [--check-only]

Export (skipped with --check-only) reproduces rig.export_fbx: import the GLB, drop meshes with no armature modifier
(the glTF importer's bone display Icosphere), name the armature 'rig', strip "<object>|" from the action names, set the
scene to 30 fps before the import (clips are baked at 30; the importer converts key times at the scene rate) and export with add_leaf_bones off, every action, no NLA, FACE smoothing and
path_mode STRIP. It prints "FBX_OK <n actions>".

Check: reset the scene, import the FBX and print "FBX_CHECK <json>" with bones, bone_names, mesh_vertices,
max_influences, actions [{name, frame_start, frame_end, height_first, height_mid}], scene_fps, up_axis, unit_scale,
rest_height. Ported from the e3 tools_fbxcheck.py.
"""
import json
import sys

import bpy

FPS = 30


def export(glb, fbx):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # The glTF importer turns key times into frames at the scene rate in force during the import: set 30 fps first,
    # or the actions keep the factory 24 frames a second while the file says 30 (every clip ~25% fast in the engine).
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=glb)
    for o in list(bpy.data.objects):
        if o.type == "MESH" and not any(m.type == "ARMATURE" for m in o.modifiers):
            bpy.data.objects.remove(o)
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            o.name = o.data.name = "rig"
    for act in bpy.data.actions:  # "SkinTokens learned binding|Walk" -> "Walk"; the FBX take becomes "rig|Walk"
        act.name = act.name.split("|")[-1]
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    bpy.ops.export_scene.fbx(filepath=fbx, add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=True,
                             bake_anim_use_nla_strips=False, mesh_smooth_type="FACE", path_mode="STRIP")
    print("FBX_OK", len(bpy.data.actions), flush=True)


def _extent(me_obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = me_obj.evaluated_get(dg)
    m = ev.to_mesh()
    try:
        if not len(m.vertices):
            return [0.0, 0.0, 0.0]
        mw = me_obj.matrix_world
        pts = [mw @ v.co for v in m.vertices]
        return [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)]
    finally:
        ev.to_mesh_clear()


def _global_settings(fbx):
    """UpAxis / UnitScaleFactor as written in the file's GlobalSettings (the importer converts them away)."""
    try:
        from io_scene_fbx import parse_fbx
        root, _ver = parse_fbx.parse(fbx)
        props = {}
        for el in root.elems:
            if el.id != b"GlobalSettings":
                continue
            for sub in el.elems:
                if sub.id == b"Properties70":
                    for p in sub.elems:
                        key = p.props[0].decode() if isinstance(p.props[0], bytes) else str(p.props[0])
                        props[key] = p.props[4] if len(p.props) > 4 else None
        sign = props.get("UpAxisSign", 1)
        up = props.get("UpAxis")
        return {"up_axis": (("-" if sign and sign < 0 else "+") + "XYZ"[up]) if up in (0, 1, 2) else None,
                "unit_scale": float(props["UnitScaleFactor"]) if props.get("UnitScaleFactor") is not None else None,
                "file_fps": float(props["CustomFrameRate"]) if props.get("CustomFrameRate") is not None else None}
    except Exception as exc:  # noqa: BLE001 - the report must still print
        return {"up_axis": None, "unit_scale": None, "global_settings_error": str(exc)[:200]}


def check(fbx):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    scene = bpy.context.scene
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    out = {"armatures": len(arms), "meshes": len(meshes), "scene_fps": round(scene.render.fps / (scene.render.fps_base or 1), 3),
           "bones": 0, "bone_names": [], "mesh_vertices": 0, "max_influences": 0, "actions": [],
           "up_axis": None, "unit_scale": None, "rest_height": None}
    if not arms:
        print("FBX_CHECK " + json.dumps(out), flush=True)
        return
    arm = arms[0]
    bones = [b.name for b in arm.data.bones]
    out.update(bones=len(bones), bone_names=bones, armature_scale=round(float(arm.matrix_world.to_scale()[0]), 6))
    out.update(_global_settings(fbx))
    me = next((o for o in meshes if any(m.type == "ARMATURE" for m in o.modifiers)), meshes[0] if meshes else None)
    if me is not None:
        names = {g.index: g.name for g in me.vertex_groups}
        bone_set = set(bones)
        out["mesh_vertices"] = len(me.data.vertices)
        out["max_influences"] = max((sum(1 for g in v.groups if g.weight > 1e-6 and names.get(g.group) in bone_set)
                                     for v in me.data.vertices), default=0)
    if arm.animation_data is None:
        arm.animation_data_create()
    rest_action = arm.animation_data.action
    arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
    if me is not None:
        ext = _extent(me)
        axis = max(range(3), key=lambda i: ext[i])
        out["rest_extent"] = [round(e, 4) for e in ext]
        out["rest_height"] = round(ext[2], 4)
        # Blender is Z-up after the importer's axis conversion: a creature stood the right way up is tallest on Z
        # unless it is long and low, so up_axis reports which axis the importer put up (Z) and the tallest one.
        out["tallest_axis"] = "XYZ"[axis]
    for act in bpy.data.actions:
        arm.animation_data.action = act
        try:  # Blender 4.4+: slotted actions need a slot bound to the armature
            if getattr(arm.animation_data, "action_slot", None) is None and getattr(act, "slots", None):
                arm.animation_data.action_slot = act.slots[0]
        except Exception:
            pass
        s, e = (float(x) for x in act.frame_range)
        row = {"name": act.name, "frame_start": s, "frame_end": e}
        if me is not None:
            hs = []
            for fr in (s, (s + e) / 2):
                scene.frame_set(int(round(fr)))
                hs.append(round(_extent(me)[2], 4))
            row["height_first"], row["height_mid"] = hs
        out["actions"].append(row)
    arm.animation_data.action = rest_action
    print("FBX_CHECK " + json.dumps(out), flush=True)


def main():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    flags = {x for x in a if x.startswith("--")}
    pos = [x for x in a if not x.startswith("--")]
    glb, fbx = pos[0], pos[1]
    if "--check-only" not in flags:
        export(glb, fbx)
    check(fbx)


main()
