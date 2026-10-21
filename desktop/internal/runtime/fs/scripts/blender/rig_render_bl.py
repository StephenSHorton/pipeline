"""Blender Workbench side of rig_render.py --blender (a parity cross-check of the CPU renderer only).

    Blender -b --factory-startup -P scripts/blender/rig_render_bl.py -- jobs.json

Each job (written by rig_render.render_blender):
    {glbs: [...], clip: name | null, t: seconds, rest: bool, mode: texture|groups|sticks|heat:<bone>, views: [...],
     cams: {view: {right, up, fwd}} (glTF frame), centre: [x, y, z] (Blender frame), ortho, size, fps, floor,
     floor_thick, floor_box: [[x0, y0], [x1, y1]] (Blender XY), partmap: {bone: role}, role_rgb: {role: [r, g, b] 0..1}, pose: {bone: [axis, degrees]},
     out_prefix: ".../Walk_0.25_{view}_texture"}
writes <out_prefix with {view}>.png per view on a transparent film (rig_render composites the background).

Ported from the research brender.py (partmap and pose options, x-ray sticks by role, vertex-colour groups) and
brender_floor.py (the red floor slab seen edge-on). The importer converts glTF (+Y up, +Z forward) to Blender
(+Z up, -Y forward): glTF (x, y, z) -> Blender (x, -z, y).
"""
import json
import sys

import bpy
from mathutils import Matrix, Vector

jobs = json.load(open(sys.argv[sys.argv.index("--") + 1]))


def g2b(v):
    return Vector((v[0], -v[2], v[1]))


def mk(name, rgb):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*rgb, 1)
    return m


def stick(a, b, r, mat):
    d = b - a
    if d.length < 1e-6:
        return
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, location=(a + b) / 2, vertices=8)
    o = bpy.context.active_object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
    o.data.materials.append(mat)


def ball(p, r, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=p, segments=8, ring_count=6)
    bpy.context.active_object.data.materials.append(mat)


def ramp(x):
    stops = [(20, 20, 90), (0, 200, 255), (40, 220, 40), (255, 230, 0), (255, 30, 0)]
    x = min(max(x, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    return tuple((stops[i][k] * (1 - f) + stops[i + 1][k] * f) / 255 for k in range(3))


def find_action(clip):
    for ac in bpy.data.actions:
        n = ac.name
        if n == clip or n.split("|")[-1] == clip or n.startswith(clip + "_") or n.startswith(clip + "."):
            return ac
    return None


for job in jobs:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = int(job.get("fps", 30))
    arms = []
    for g in job["glbs"]:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=g)
        new = [o for o in bpy.data.objects if o not in before]
        arms += [o for o in new if o.type == "ARMATURE"]
        for o in new:  # the importer's bone-shape icospheres
            if o.type == "MESH" and o.name.startswith("Icosphere"):
                o.hide_render = True
    sc.render.fps = int(job.get("fps", 30))
    partmap = job.get("partmap", {})
    rgb = {k: tuple(v) for k, v in job["role_rgb"].items()}

    for a in arms:
        ad = a.animation_data or a.animation_data_create()
        for tr in ad.nla_tracks:
            tr.mute = True
        act = find_action(job["clip"]) if job.get("clip") else None
        if job.get("clip") and act is None:
            print("NO ACTION", job["clip"], [ac.name for ac in bpy.data.actions])
        ad.action = act
        if act is not None and getattr(act, "slots", None) and hasattr(ad, "action_slot") and ad.action_slot is None:
            ad.action_slot = act.slots[0]
        if act is None:
            for pb in a.pose.bones:
                pb.matrix_basis = Matrix()
    f = float(job.get("t", 0.0)) * sc.render.fps
    sc.frame_set(int(f), subframe=f - int(f))
    for a in arms:
        for bn, (ax, deg) in job.get("pose", {}).items():
            pb = a.pose.bones.get(bn)
            if pb:
                import math
                pb.rotation_mode = "XYZ"
                r = list(pb.rotation_euler)
                r["XYZ".index(ax)] += math.radians(deg)
                pb.rotation_euler = r
    bpy.context.view_layer.update()
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render]
    mode = job.get("mode", "texture")
    sh = sc.display.shading
    sc.render.engine = "BLENDER_WORKBENCH"
    sh.light = "STUDIO"
    try:
        sc.view_settings.view_transform = "Standard"
    except TypeError:
        pass

    if mode == "groups" or mode.startswith("heat:"):
        bone = mode.split(":", 1)[1] if mode.startswith("heat:") else None
        for o in meshes:
            if not o.vertex_groups:
                continue
            gname = {g.index: g.name for g in o.vertex_groups}
            ca = o.data.color_attributes.new("rr", "FLOAT_COLOR", "POINT")
            for v in o.data.vertices:
                if bone is None:
                    best = max(v.groups, key=lambda g: g.weight, default=None)
                    c = rgb.get(partmap.get(gname[best.group], "other"), rgb["other"]) if best else (0.3, 0.3, 0.3)
                else:
                    c = ramp(sum(g.weight for g in v.groups if gname[g.group] == bone))
                ca.data[v.index].color = (*c, 1)
            o.data.color_attributes.active_color = ca
        sh.color_type = "VERTEX"
    elif mode == "sticks":
        mats = {k: mk(k, v) for k, v in rgb.items()}
        body = mk("body", (0.75, 0.72, 0.68))
        for o in meshes:
            o.data.materials.clear()
            o.data.materials.append(body)
        r = float(job["ortho"]) / 260.0
        for a in arms:
            for pb in a.pose.bones:
                h = a.matrix_world @ pb.head
                k = partmap.get(pb.name, "other")
                for c in pb.children:
                    stick(h, a.matrix_world @ c.head, r, mats[k])
                ball(h, r * 1.8, mats[k])
        sh.color_type = "MATERIAL"
        sh.show_xray = True
        sh.xray_alpha = 0.5
    else:
        sh.color_type = "TEXTURE"

    slabs = {}
    if job.get("floor"):
        th = float(job.get("floor_thick", 0.004))
        fm = mk("floorline", (1.0, 0.0, 0.0))
        (x0, y0), (x1, y1) = job.get("floor_box", [[-1, -1], [1, 1]])
        big = float(job["ortho"]) * 4
        # edge-on views: a slab wider than the frame; top and q: the CPU outline's footprint
        for key, (cx, cy, sx, sy) in (("big", (0, 0, big, big)), ("box", ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))):
            bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, -th / 2))
            fl = bpy.context.active_object
            fl.scale = (sx, sy, th)
            fl.data.materials.append(fm)
            if sh.color_type == "VERTEX":
                ca = fl.data.color_attributes.new("rr", "FLOAT_COLOR", "POINT")
                for v in fl.data.vertices:
                    ca.data[v.index].color = (1, 0, 0, 1)
            slabs[key] = fl

    sc.render.resolution_x = sc.render.resolution_y = int(job["size"])
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    cam = bpy.data.cameras.new("c")
    cam.type = "ORTHO"
    cam.ortho_scale = float(job["ortho"])
    dist = float(job["ortho"]) * 10 + 10
    cam.clip_start = 0.01
    cam.clip_end = dist * 2
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    cen = Vector(job["centre"])
    for view in job["views"]:
        b = job["cams"][view]
        rt, up, fwd = g2b(b["right"]), g2b(b["up"]), g2b(b["fwd"])
        co.location = cen - fwd * dist
        co.rotation_mode = "QUATERNION"
        co.rotation_quaternion = Matrix((rt, up, -fwd)).transposed().to_quaternion()
        edge = abs(b["fwd"][1]) < 1e-6
        if slabs:
            slabs["big"].hide_render = not edge
            slabs["box"].hide_render = edge
        sc.render.filepath = job["out_prefix"].replace("{view}", view) + ".png"
        bpy.ops.render.render(write_still=True)
    print("JOB_DONE", job["out_prefix"])
print("RIG_RENDER_BL_OK")
