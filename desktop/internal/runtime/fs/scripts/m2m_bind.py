#!/usr/bin/env python3
"""Bind a static mesh to the Mesh2Motion human rig and keep their CC0 clips.

Fits the mesh to the mannequin bind pose, assigns Mesh2Motion-style closest-bone
weights (skip root + leaf bones), and writes a skinned GLB that still contains
Idle_A, Walk, Jog, and the rest of human-base-animations.glb.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    GLTF2,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    Accessor,
    BufferView,
    Image,
    Material,
    PbrMetallicRoughness,
    Texture,
    TextureInfo,
)
import trimesh


def q_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def node_local(n):
    t = np.array(n.translation or [0, 0, 0], dtype=np.float32)
    r = np.array(n.rotation or [0, 0, 0, 1], dtype=np.float32)
    s = np.array(n.scale or [1, 1, 1], dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = q_to_mat(r) * s
    m[:3, 3] = t
    return m


def world_mats(gltf):
    locals_ = [node_local(n) for n in gltf.nodes]
    worlds = [np.eye(4, dtype=np.float32) for _ in gltf.nodes]
    children = {i: (n.children or []) for i, n in enumerate(gltf.nodes)}
    roots = list(gltf.scenes[gltf.scene].nodes)
    stack = [(r, np.eye(4, dtype=np.float32)) for r in roots]
    while stack:
        i, parent = stack.pop()
        w = parent @ locals_[i]
        worlds[i] = w
        for c in children[i]:
            stack.append((c, w))
    return worlds


def fit_mesh(src_verts, target_min, target_max):
    mn, mx = src_verts.min(0), src_verts.max(0)
    src_h = float(mx[1] - mn[1]) or 1.0
    tgt_h = float(target_max[1] - target_min[1]) or 1.0
    scale = tgt_h / src_h
    # feet on target ground, xz centered
    src_c = (mn + mx) * 0.5
    tgt_c = (target_min + target_max) * 0.5
    out = (src_verts - np.array([src_c[0], mn[1], src_c[2]])) * scale
    out += np.array([tgt_c[0], target_min[1], tgt_c[2]])
    return out.astype(np.float32), scale


def skin_weights(verts, joint_world, joints, names, children):
    n = len(verts)
    mids = []
    valid = []
    for ji, node_i in enumerate(joints):
        name = names[node_i] or ""
        if name in ("root",) or name.endswith("_leaf") or "leaf" in name:
            continue
        kids = children.get(node_i, [])
        p = joint_world[node_i][:3, 3]
        if kids:
            q = joint_world[kids[0]][:3, 3]
            mid = (p + q) * 0.5
        else:
            mid = p
        mids.append(mid)
        valid.append(ji)
    mids = np.stack(mids, axis=0)
    d = np.linalg.norm(verts[:, None, :] - mids[None, :, :], axis=2)
    # hip gate: pelvis shouldn't own vertices below crotch
    pelvis = None
    for local_i, ji in enumerate(valid):
        if names[joints[ji]] in ("pelvis", "hips"):
            pelvis = local_i
            hip_y = mids[local_i][1]
            below = verts[:, 1] < hip_y - 0.02
            d[below, local_i] += 10.0
    k = min(4, d.shape[1])
    order = np.argpartition(d, k, axis=1)[:, :k]
    nearest = np.take_along_axis(d, order, axis=1)
    w = 1.0 / np.square(nearest + 1e-4)
    w /= w.sum(axis=1, keepdims=True)
    joint_idx = np.array(valid, dtype=np.uint16)[order]
    return joint_idx.astype(np.uint16), w.astype(np.float32)


def append_f32(blob, gltf, data, ctype, typ, target, mn=None, mx=None):
    raw = np.ascontiguousarray(data).tobytes()
    pad = (4 - (len(blob) % 4)) % 4
    blob.extend(b"\x00" * pad)
    start = len(blob)
    blob.extend(raw)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=start, byteLength=len(raw), target=target))
    acc = Accessor(
        bufferView=len(gltf.bufferViews) - 1,
        byteOffset=0,
        componentType=ctype,
        count=int(np.asarray(data).shape[0] if typ != "SCALAR" else np.asarray(data).size),
        type=typ,
    )
    if typ == "SCALAR":
        acc.count = int(np.asarray(data).size)
    if mn is not None:
        acc.min = np.asarray(mn, dtype=float).tolist()
        acc.max = np.asarray(mx, dtype=float).tolist()
    gltf.accessors.append(acc)
    return len(gltf.accessors) - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--m2m", default=None, help="mesh2motion-app root")
    ap.add_argument("--keep", default="Idle_A,Walk,Jog,Walk_Formal,Idle_Talking")
    ap.add_argument("--all-clips", action="store_true")
    args = ap.parse_args()

    root = Path(args.m2m) if args.m2m else Path(__file__).resolve().parents[1] / "vendor" / "mesh2motion-app"
    template = root / "static" / "animations" / "human-base-animations.glb"
    if not template.is_file():
        print(f"m2m_bind: missing {template}", file=sys.stderr)
        return 1

    src = trimesh.load(args.mesh, force="mesh")
    if not isinstance(src, trimesh.Trimesh):
        print("m2m_bind: not a triangle mesh", file=sys.stderr)
        return 1
    verts = np.asarray(src.vertices, dtype=np.float32)
    faces = np.asarray(src.faces, dtype=np.uint32)
    try:
        norms = np.asarray(src.vertex_normals, dtype=np.float32)
    except Exception:
        src.fix_normals()
        norms = np.asarray(src.vertex_normals, dtype=np.float32)
    uvs = None
    if getattr(src.visual, "uv", None) is not None:
        uvs = np.asarray(src.visual.uv, dtype=np.float32)
    albedo = getattr(getattr(src.visual, "material", None), "baseColorTexture", None) or getattr(
        getattr(src.visual, "material", None), "image", None
    )

    mannequin = trimesh.load(str(template), force="mesh")
    if isinstance(mannequin, trimesh.Scene):
        mannequin = trimesh.util.concatenate(tuple(mannequin.geometry.values()))
    tmin, tmax = np.asarray(mannequin.vertices).min(0), np.asarray(mannequin.vertices).max(0)
    fitted, scale = fit_mesh(verts, tmin, tmax)
    print(f"m2m_bind: fit scale {scale:.4f}  {len(fitted)} verts -> mannequin height {tmax[1]-tmin[1]:.3f}")

    gltf = GLTF2().load(str(template))
    worlds = world_mats(gltf)
    skin = gltf.skins[0]
    joints = list(skin.joints)
    names = [n.name or "" for n in gltf.nodes]
    children = {i: list(n.children or []) for i, n in enumerate(gltf.nodes)}
    jidx, weights = skin_weights(fitted, worlds, joints, names, children)

    blob = bytearray(gltf.binary_blob() or b"")
    a_pos = append_f32(blob, gltf, fitted, FLOAT, "VEC3", ARRAY_BUFFER, fitted.min(0), fitted.max(0))
    a_nrm = append_f32(blob, gltf, norms, FLOAT, "VEC3", ARRAY_BUFFER)
    a_jnt = append_f32(blob, gltf, jidx, UNSIGNED_SHORT, "VEC4", ARRAY_BUFFER)
    a_wgt = append_f32(blob, gltf, weights, FLOAT, "VEC4", ARRAY_BUFFER)
    a_idx = append_f32(blob, gltf, faces.reshape(-1).astype(np.uint32), UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)
    attrs = gltf.meshes[0].primitives[0].attributes
    attrs.POSITION = a_pos
    attrs.NORMAL = a_nrm
    attrs.JOINTS_0 = a_jnt
    attrs.WEIGHTS_0 = a_wgt
    if uvs is not None and len(uvs) == len(fitted):
        attrs.TEXCOORD_0 = append_f32(blob, gltf, uvs, FLOAT, "VEC2", ARRAY_BUFFER)
    gltf.meshes[0].primitives[0].indices = a_idx

    if albedo is not None:
        if not isinstance(albedo, PILImage.Image):
            albedo = PILImage.fromarray(np.asarray(albedo))
        buf = io.BytesIO()
        albedo.convert("RGB").save(buf, format="PNG")
        png = buf.getvalue()
        pad = (4 - (len(blob) % 4)) % 4
        blob.extend(b"\x00" * pad)
        start = len(blob)
        blob.extend(png)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=start, byteLength=len(png)))
        gltf.images = gltf.images or []
        gltf.images.append(Image(bufferView=len(gltf.bufferViews) - 1, mimeType="image/png"))
        gltf.textures = gltf.textures or []
        gltf.textures.append(Texture(source=len(gltf.images) - 1))
        gltf.materials = [Material(
            name="operator",
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture=TextureInfo(index=len(gltf.textures) - 1),
                metallicFactor=0.08,
                roughnessFactor=0.62,
            ),
        )]
        gltf.meshes[0].primitives[0].material = 0

    if not args.all_clips:
        keep = {s.strip() for s in args.keep.split(",") if s.strip()}
        kept = [a for a in (gltf.animations or []) if a.name in keep]
        if kept:
            gltf.animations = kept
            print("m2m_bind: kept clips", [a.name for a in kept])
        else:
            print("m2m_bind: keep list missed, leaving all", len(gltf.animations or []), "clips")
    else:
        print("m2m_bind: keeping all", len(gltf.animations or []), "clips")

    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(out))
    print(f"m2m_bind: wrote {out} ({out.stat().st_size} bytes)")
    meta = {
        "src": str(Path(args.mesh)),
        "template": str(template),
        "scale": float(scale),
        "clips": [a.name for a in (gltf.animations or [])],
        "verts": int(len(fitted)),
        "faces": int(len(faces)),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
