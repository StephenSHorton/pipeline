"""Blender bone-heat weights for a skinned GLB (run by scripts/rig_heat.py; not imported by the pipeline).

  blender -b --factory-startup -P scripts/blender/heat_bind.py -- in.glb out.json [--weld 0.002]

in.glb is a GLB with one skinned mesh and its armature (its own weights are ignored). The script:
  1. imports it, drops the mesh's vertex groups and armature modifier and unparents it (keeping its world place);
  2. prints the import sanity numbers (HEAT_SANITY <json>): scales, dimensions, non-manifold edges, zero-area faces,
     minimum edge, before and after the weld;
  3. welds points closer than --weld (2 mm by default; heat weighting fails on split corners);
  4. parents the mesh to the armature with Automatic Weights (ARMATURE_AUTO) in the rest pose;
  5. writes out.json: every welded point's world position in glTF axes (+Y up) and its weights by bone name.
     rig_heat.py moves these onto the unwelded vertices by position (a KD-tree), so the vertex order stays the
     source's. Prints HEAT_OK <json> with the counts, or HEAT_FAIL <reason>.
Ports scratch skinres/tools/bl_heat.py and bl_dbg2.py.
"""
import json
import sys

import bmesh
import bpy

a = sys.argv[sys.argv.index("--") + 1:]
src, out = a[0], a[1]
weld = float(a[a.index("--weld") + 1]) if "--weld" in a else 0.002


def sanity(o, bm) -> dict:
    lens = [e.calc_length() for e in bm.edges]
    return {"verts": len(bm.verts), "faces": len(bm.faces), "edges": len(bm.edges),
            "nonmanifold_edges": sum(1 for e in bm.edges if not e.is_manifold),
            "zero_area_faces": sum(1 for f in bm.faces if f.calc_area() < 1e-10),
            "min_edge": min(lens) if lens else None}


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not arms:
    print("HEAT_FAIL no armature in " + src, flush=True)
    sys.exit(1)
arm = arms[0]
# The importer adds an unparented Icosphere as the bones' display shape: drop every mesh not bound to the armature.
me = [o for o in bpy.data.objects if o.type == "MESH" and (o.parent is not None or any(m.type == "ARMATURE" for m in o.modifiers))]
for o in [o for o in bpy.data.objects if o.type == "MESH" and o not in me]:
    bpy.data.objects.remove(o)
if len(me) != 1:
    print(f"HEAT_FAIL expected one skinned mesh, found {len(me)}", flush=True)
    sys.exit(1)
o = me[0]
o.vertex_groups.clear()
for m in list(o.modifiers):
    if m.type == "ARMATURE":
        o.modifiers.remove(m)
mw = o.matrix_world.copy()
o.parent = None
o.matrix_world = mw

info = {"arm_scale": list(arm.matrix_world.to_scale()), "mesh_scale": list(o.matrix_world.to_scale()),
        "mesh_dims": list(o.dimensions), "weld": weld}
bm = bmesh.new()
bm.from_mesh(o.data)
info["imported"] = sanity(o, bm)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
info["welded"] = sanity(o, bm)
bm.to_mesh(o.data)
bm.free()
o.data.update()
print("HEAT_SANITY " + json.dumps(info), flush=True)

if arm.animation_data:
    arm.animation_data_clear()
bpy.context.scene.frame_set(0)
for x in bpy.data.objects:
    x.select_set(False)
o.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
arm.data.pose_position = "REST"  # heat on the bind pose
try:
    r = bpy.ops.object.parent_set(type="ARMATURE_AUTO")
except RuntimeError as e:  # Blender raises with the heat solver's message on a hard failure
    print("HEAT_FAIL parent_set: " + str(e).strip(), flush=True)
    sys.exit(1)
arm.data.pose_position = "POSE"

names = [g.name for g in o.vertex_groups]
deform = {b.name for b in arm.data.bones if b.use_deform}
M = o.matrix_world
pos, wts = [], []
used = [0] * len(names)
for v in o.data.vertices:
    w = M @ v.co
    pos.append([round(w.x, 7), round(w.z, 7), round(-w.y, 7)])  # Blender Z-up -> glTF Y-up
    row = sorted(((g.group, g.weight) for g in v.groups if g.weight > 1e-4), key=lambda t: -t[1])
    for gi, _ in row:
        used[gi] += 1
    wts.append([[gi, round(wt, 6)] for gi, wt in row])
empty = [n for n, u in zip(names, used) if u == 0 and n in deform]
with open(out, "w") as fh:
    json.dump({"groups": names, "positions": pos, "weights": wts, "sanity": info, "empty_groups": empty,
               "parent_set": list(r)}, fh)
print("HEAT_OK " + json.dumps({"groups": len(names), "verts": len(pos), "empty_groups": empty,
                               "unweighted": sum(1 for w in wts if not w)}), flush=True)
