#!/usr/bin/env python3
"""Prototype humanoid auto-rig + idle/walk clips. Not Mixamo — a local stand-in
until UniRig/AniGen is on the machine. Writes a skinned GLB Three.js can play.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from pygltflib import (
    ANIM_LINEAR,
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    GLTF2,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Asset,
    Buffer,
    BufferView,
    Mesh,
    Node,
    Primitive,
    Scene,
    Skin,
)


BONES = [
    "Hips", "Spine", "Chest", "Neck", "Head",
    "UpLeg_L", "LowLeg_L", "Foot_L",
    "UpLeg_R", "LowLeg_R", "Foot_R",
    "Shoulder_L", "UpArm_L", "LowArm_L", "Hand_L",
    "Shoulder_R", "UpArm_R", "LowArm_R", "Hand_R",
]
PARENT = {
    "Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    "UpLeg_L": "Hips", "LowLeg_L": "UpLeg_L", "Foot_L": "LowLeg_L",
    "UpLeg_R": "Hips", "LowLeg_R": "UpLeg_R", "Foot_R": "LowLeg_R",
    "Shoulder_L": "Chest", "UpArm_L": "Shoulder_L", "LowArm_L": "UpArm_L", "Hand_L": "LowArm_L",
    "Shoulder_R": "Chest", "UpArm_R": "Shoulder_R", "LowArm_R": "UpArm_R", "Hand_R": "LowArm_R",
}


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)], dtype=np.float32)


def rest_positions(mn, mx):
    c = (mn + mx) * 0.5
    h = mx[1] - mn[1]
    w = mx[0] - mn[0]
    d = mx[2] - mn[2]
    y = mn[1]
    def p(xf, yf, zf):
        return np.array([c[0] + xf * w, y + yf * h, c[2] + zf * d], dtype=np.float32)
    return {
        "Hips": p(0.00, 0.52, 0.00),
        "Spine": p(0.00, 0.62, 0.00),
        "Chest": p(0.00, 0.74, 0.02),
        "Neck": p(0.00, 0.84, 0.01),
        "Head": p(0.00, 0.95, 0.00),
        "UpLeg_L": p(-0.09, 0.50, 0.00),
        "LowLeg_L": p(-0.09, 0.27, 0.01),
        "Foot_L": p(-0.09, 0.03, 0.06),
        "UpLeg_R": p(0.09, 0.50, 0.00),
        "LowLeg_R": p(0.09, 0.27, 0.01),
        "Foot_R": p(0.09, 0.03, 0.06),
        "Shoulder_L": p(-0.16, 0.80, 0.00),
        "UpArm_L": p(-0.28, 0.70, 0.00),
        "LowArm_L": p(-0.36, 0.56, 0.02),
        "Hand_L": p(-0.40, 0.46, 0.03),
        "Shoulder_R": p(0.16, 0.80, 0.00),
        "UpArm_R": p(0.28, 0.70, 0.00),
        "LowArm_R": p(0.36, 0.56, 0.02),
        "Hand_R": p(0.40, 0.46, 0.03),
    }


def dist_seg(points, a, b):
    ab = b - a
    den = float(np.dot(ab, ab)) + 1e-8
    t = np.clip(((points - a) @ ab) / den, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def skin_weights(verts, rest, child_of):
    n = len(verts)
    k = len(BONES)
    dists = np.zeros((n, k), dtype=np.float32)
    for i, name in enumerate(BONES):
        a = rest[name]
        kids = [c for c, p in child_of.items() if p == name]
        b = rest[kids[0]] if kids else a + np.array([0, 0.04, 0], dtype=np.float32)
        # average if multiple children
        if len(kids) > 1:
            b = np.mean([rest[c] for c in kids], axis=0)
        dists[:, i] = dist_seg(verts, a, b)
        # keep head weights off the feet
        if name == "Head":
            dists[:, i] += np.clip((rest["Hips"][1] + 0.05 - verts[:, 1]) * 8.0, 0, None)
        if "Leg" in name or "Foot" in name:
            dists[:, i] += np.clip((verts[:, 1] - rest["Chest"][1]) * 6.0, 0, None)
    order = np.argpartition(dists, 4, axis=1)[:, :4]
    nearest = np.take_along_axis(dists, order, axis=1)
    w = 1.0 / np.square(nearest + 1e-3)
    w /= w.sum(axis=1, keepdims=True)
    return order.astype(np.uint16), w.astype(np.float32)


def world_from_local(rest, child_of):
    """Local translation = rest[child] - rest[parent]; rotation identity."""
    local = {}
    for name in BONES:
        p = PARENT.get(name)
        local[name] = rest[name] - rest[p] if p else rest[name]
    return local


def ibm_for(rest):
    # inverse bind: translate by -rest
    mats = []
    for name in BONES:
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -rest[name]
        mats.append(m.T.reshape(-1))  # glTF column-major via flatten of transpose of row-major...
        # row-major eye with translation: glTF wants column-major
    # Standard: column-major of matrix that maps world->joint
    out = []
    for name in BONES:
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -rest[name]
        out.append(np.ascontiguousarray(m.T).reshape(-1))  # store column-major
    return np.stack(out, axis=0).astype(np.float32)


def pack_blob(chunks):
    buf = bytearray()
    views = []
    for data, target in chunks:
        raw = data if isinstance(data, (bytes, bytearray)) else data.tobytes()
        pad = (4 - (len(buf) % 4)) % 4
        buf.extend(b"\x00" * pad)
        start = len(buf)
        buf.extend(raw)
        views.append((start, len(raw), target))
    pad = (4 - (len(buf) % 4)) % 4
    buf.extend(b"\x00" * pad)
    return bytes(buf), views


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    mesh = trimesh.load(args.inp, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print("autobiped: not a triangle mesh", file=sys.stderr)
        return 1
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    norms = np.asarray(mesh.vertex_normals, dtype=np.float32)
    uvs = None
    if getattr(mesh.visual, "uv", None) is not None:
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)

    # Make the tallest axis Y.
    ext = np.ptp(verts, axis=0)
    up = int(np.argmax(ext))
    if up != 1:
        swap = [0, 1, 2]
        swap[1], swap[up] = swap[up], swap[1]
        verts = verts[:, swap]
        norms = norms[:, swap]
        print(f"autobiped: swapped axis {up} -> Y")

    mn, mx = verts.min(0), verts.max(0)
    rest = rest_positions(mn, mx)
    joints, weights = skin_weights(verts, rest, PARENT)
    local = world_from_local(rest, PARENT)
    ibms = ibm_for(rest)

    # Animation samples: idle (2s) and walk (1s)
    def clip_rot(times, per_bone_angles):
        # per_bone_angles: name -> list of (axis, angle) per key
        return times, per_bone_angles

    idle_t = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    walk_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    ident = np.array([0, 0, 0, 1], dtype=np.float32)

    def series(times, fn):
        return np.stack([fn(t) for t in times], axis=0)

    idle_map = {
        "Chest": series(idle_t, lambda t: quat_from_axis_angle([1, 0, 0], 0.04 * np.sin(np.pi * t))),
        "Spine": series(idle_t, lambda t: quat_from_axis_angle([1, 0, 0], 0.02 * np.sin(np.pi * t))),
    }
    walk_map = {
        "UpLeg_L": series(walk_t, lambda t: quat_from_axis_angle([1, 0, 0], 0.55 * np.sin(2 * np.pi * t))),
        "UpLeg_R": series(walk_t, lambda t: quat_from_axis_angle([1, 0, 0], 0.55 * np.sin(2 * np.pi * t + np.pi))),
        "LowLeg_L": series(walk_t, lambda t: quat_from_axis_angle([1, 0, 0], 0.35 * max(0.0, -np.sin(2 * np.pi * t)))),
        "LowLeg_R": series(walk_t, lambda t: quat_from_axis_angle([1, 0, 0], 0.35 * max(0.0, -np.sin(2 * np.pi * t + np.pi)))),
        "UpArm_L": series(walk_t, lambda t: quat_from_axis_angle([1, 0, 0], -0.4 * np.sin(2 * np.pi * t))),
        "UpArm_R": series(walk_t, lambda t: quat_from_axis_angle([1, 0, 0], -0.4 * np.sin(2 * np.pi * t + np.pi))),
        "Hips": series(walk_t, lambda t: quat_from_axis_angle([1, 0, 0], 0.04 * np.sin(4 * np.pi * t))),
    }

    chunks = []
    def add_f32(arr, target=ARRAY_BUFFER):
        arr = np.ascontiguousarray(arr)
        chunks.append((arr, target))
        return len(chunks) - 1

    pos_i = add_f32(verts)
    nrm_i = add_f32(norms)
    jnt_i = add_f32(joints)  # uint16
    # joints stored as uint16 — handle separately
    chunks[-1] = (joints.astype(np.uint16), ARRAY_BUFFER)
    wgt_i = add_f32(weights)
    idx_i = add_f32(faces.reshape(-1), ELEMENT_ARRAY_BUFFER)
    chunks[-1] = (faces.reshape(-1).astype(np.uint32), ELEMENT_ARRAY_BUFFER)
    ibm_i = add_f32(ibms)
    uv_i = None
    if uvs is not None and len(uvs) == len(verts):
        uv_i = add_f32(uvs)

    idle_t_i = add_f32(idle_t)
    walk_t_i = add_f32(walk_t)
    rot_accessors = {}
    for name, keys in {**{f"idle:{k}": v for k, v in idle_map.items()}, **{f"walk:{k}": v for k, v in walk_map.items()}}.items():
        rot_accessors[name] = add_f32(keys)

    blob, views_meta = pack_blob(chunks)

    gltf = GLTF2(asset=Asset(generator="ai-asset-pipeline autobiped"))
    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.bufferViews = []
    gltf.accessors = []
    for start, length, target in views_meta:
        kwargs = {"buffer": 0, "byteOffset": start, "byteLength": length}
        if target is not None:
            kwargs["target"] = target
        gltf.bufferViews.append(BufferView(**kwargs))

    def acc(view, ctype, count, component, mn=None, mx=None):
        a = Accessor(bufferView=view, byteOffset=0, componentType=ctype, count=count, type=component)
        if mn is not None:
            a.min = np.asarray(mn).tolist()
            a.max = np.asarray(mx).tolist()
        gltf.accessors.append(a)
        return len(gltf.accessors) - 1

    a_pos = acc(pos_i, FLOAT, len(verts), "VEC3", verts.min(0), verts.max(0))
    a_nrm = acc(nrm_i, FLOAT, len(verts), "VEC3")
    a_jnt = acc(jnt_i, UNSIGNED_SHORT, len(verts), "VEC4")
    a_wgt = acc(wgt_i, FLOAT, len(verts), "VEC4")
    a_idx = acc(idx_i, UNSIGNED_INT, int(faces.size), "SCALAR")
    a_ibm = acc(ibm_i, FLOAT, len(BONES), "MAT4")
    attrs = {"POSITION": a_pos, "NORMAL": a_nrm, "JOINTS_0": a_jnt, "WEIGHTS_0": a_wgt}
    if uv_i is not None:
        attrs["TEXCOORD_0"] = acc(uv_i, FLOAT, len(verts), "VEC2")

    prim = Primitive(attributes=attrs, indices=a_idx, mode=4)
    gltf.meshes = [Mesh(primitives=[prim])]

    # Nodes: 0 = mesh, 1.. = bones
    gltf.nodes = [Node(name="Mesh", mesh=0, skin=0)]
    bone_node = {}
    for name in BONES:
        bone_node[name] = len(gltf.nodes)
        t = local[name].tolist()
        gltf.nodes.append(Node(name=name, translation=t, children=[]))
    for name in BONES:
        p = PARENT.get(name)
        if p:
            gltf.nodes[bone_node[p]].children.append(bone_node[name])
    # drop empty children lists
    for n in gltf.nodes:
        if not n.children:
            n.children = None

    hips_idx = bone_node["Hips"]
    gltf.skins = [Skin(
        name="Biped",
        skeleton=hips_idx,
        joints=[bone_node[n] for n in BONES],
        inverseBindMatrices=a_ibm,
    )]
    gltf.nodes[0].children = [hips_idx]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0

    a_idle_t = acc(idle_t_i, FLOAT, len(idle_t), "SCALAR", [float(idle_t.min())], [float(idle_t.max())])
    a_walk_t = acc(walk_t_i, FLOAT, len(walk_t), "SCALAR", [float(walk_t.min())], [float(walk_t.max())])

    def make_clip(clip_name, times_acc, mmap, prefix):
        samplers = []
        channels = []
        for bname, _ in mmap.items():
            arot = acc(rot_accessors[f"{prefix}:{bname}"], FLOAT, mmap[bname].shape[0], "VEC4")
            sid = len(samplers)
            samplers.append(AnimationSampler(input=times_acc, output=arot, interpolation="LINEAR"))
            channels.append(AnimationChannel(
                sampler=sid,
                target=AnimationChannelTarget(node=bone_node[bname], path="rotation"),
            ))
        return Animation(name=clip_name, samplers=samplers, channels=channels)

    gltf.animations = [
        make_clip("idle", a_idle_t, idle_map, "idle"),
        make_clip("walk", a_walk_t, walk_map, "walk"),
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.set_binary_blob(blob)
    gltf.save_binary(str(out))
    print(f"autobiped: {len(verts)} verts, {len(BONES)} bones, clips idle+walk -> {out}")
    meta = {"bones": BONES, "clips": ["idle", "walk"], "path": str(out)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
