#!/usr/bin/env python3
"""Headless GLB turnaround via Blender. Usage: blender --background --python preview_glb.py -- in.glb out.png"""
import math
import sys
from pathlib import Path

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1 :]
if len(argv) < 2:
    print("usage: blender --background --python preview_glb.py -- <in.glb> <out.png>", file=sys.stderr)
    sys.exit(2)

glb = Path(argv[0]).resolve()
out = Path(argv[1]).resolve()
out.parent.mkdir(parents=True, exist_ok=True)

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.12, 0.12, 0.13, 1)
bg.inputs[1].default_value = 1.0

bpy.ops.import_scene.gltf(filepath=str(glb))
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not objs:
    print("no mesh in glb", file=sys.stderr)
    sys.exit(1)

# Frame the mesh
import mathutils

bbox_min = mathutils.Vector((1e9, 1e9, 1e9))
bbox_max = mathutils.Vector((-1e9, -1e9, -1e9))
for o in objs:
    for v in o.bound_box:
        w = o.matrix_world @ mathutils.Vector(v)
        bbox_min.x = min(bbox_min.x, w.x)
        bbox_min.y = min(bbox_min.y, w.y)
        bbox_min.z = min(bbox_min.z, w.z)
        bbox_max.x = max(bbox_max.x, w.x)
        bbox_max.y = max(bbox_max.y, w.y)
        bbox_max.z = max(bbox_max.z, w.z)
center = (bbox_min + bbox_max) / 2
diag = (bbox_max - bbox_min).length
dist = max(diag * 1.35, 0.5)

cam = bpy.data.cameras.new("Cam")
cam_obj = bpy.data.objects.new("Cam", cam)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (center.x + dist * 0.7, center.y - dist * 0.9, center.z + dist * 0.45)
direction = center - cam_obj.location
cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
cam.lens = 50

light_data = bpy.data.lights.new("Key", "AREA")
light_data.energy = 400
light_data.size = diag
key = bpy.data.objects.new("Key", light_data)
scene.collection.objects.link(key)
key.location = (center.x + dist, center.y - dist * 0.4, center.z + dist)
fill_data = bpy.data.lights.new("Fill", "AREA")
fill_data.energy = 120
fill_data.size = diag
fill = bpy.data.objects.new("Fill", fill_data)
scene.collection.objects.link(fill)
fill.location = (center.x - dist * 0.8, center.y - dist * 0.2, center.z + dist * 0.3)

# Simple clay material so an untextured mesh reads
mat = bpy.data.materials.new("Clay")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.42, 0.48, 0.38, 1)
    bsdf.inputs["Roughness"].default_value = 0.55
for o in objs:
    if o.data.materials:
        continue
    o.data.materials.append(mat)

scene.render.filepath = str(out)
bpy.ops.render.render(write_still=True)
print(f"preview: {out}")
