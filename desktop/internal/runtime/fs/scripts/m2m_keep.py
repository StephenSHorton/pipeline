#!/usr/bin/env python3
"""Retarget Mesh2Motion clips onto an already-skinned mesh, keeping its weights.

Maps the target skeleton to Mesh2Motion body bones by rest-pose proximity after
a height fit, then copies rest-relative rotations (and scaled root/pelvis
translation). JOINTS_0, WEIGHTS_0, and inverseBindMatrices are left untouched.

This is the factory stand-in for Mesh2Motion's experimental 'upload your rigged
mesh' path, which is a browser UI rather than a CLI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pygltflib import (
    ARRAY_BUFFER,
    FLOAT,
    GLTF2,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    BufferView,
)

BODY = (
    "root",
    "pelvis",
    "spine_01",
    "spine_02",
    "spine_03",
    "neck_01",
    "head",
    "clavicle_l",
    "upperarm_l",
    "lowerarm_l",
    "hand_l",
    "clavicle_r",
    "upperarm_r",
    "lowerarm_r",
    "hand_r",
    "thigh_l",
    "calf_l",
    "foot_l",
    "ball_l",
    "thigh_r",
    "calf_r",
    "foot_r",
    "ball_r",
)


def q_to_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def qinv(q):
    x, y, z, w = q
    n = float(x * x + y * y + z * z + w * w) or 1.0
    return np.array([-x, -y, -z, w], dtype=np.float32) / n


def qnorm(q):
    n = float(np.linalg.norm(q)) or 1.0
    out = q / n
    if out[3] < 0:
        out = -out
    return out.astype(np.float32)


def node_local(n):
    t = np.array(n.translation or [0, 0, 0], dtype=np.float32)
    r = np.array(n.rotation or [0, 0, 0, 1], dtype=np.float32)
    s = np.array(n.scale or [1, 1, 1], dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = q_to_mat(r) * s
    m[:3, 3] = t
    return m, t, qnorm(r)


def worlds(gltf):
    locals_ = []
    rest_t = []
    rest_q = []
    for n in gltf.nodes:
        m, t, q = node_local(n)
        locals_.append(m)
        rest_t.append(t)
        rest_q.append(q)
    W = [np.eye(4, dtype=np.float32) for _ in gltf.nodes]
    children = {i: list(n.children or []) for i, n in enumerate(gltf.nodes)}
    roots = list(gltf.scenes[gltf.scene].nodes)
    stack = [(r, np.eye(4, dtype=np.float32)) for r in roots]
    while stack:
        i, p = stack.pop()
        w = p @ locals_[i]
        W[i] = w
        for c in children[i]:
            stack.append((c, w))
    return W, np.stack(rest_t), rest_q, children


def accessor_np(gltf, blob, acc_i):
    acc = gltf.accessors[acc_i]
    bv = gltf.bufferViews[acc.bufferView]
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc.type]
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[acc.componentType]
    count = int(acc.count)
    nbytes = count * ncomp * np.dtype(dtype).itemsize
    arr = np.frombuffer(blob[start : start + nbytes], dtype=dtype).copy()
    if ncomp == 1:
        return arr
    return arr.reshape(count, ncomp)


def append_f32(blob, gltf, data, typ):
    raw = np.ascontiguousarray(data, dtype=np.float32).tobytes()
    pad = (4 - (len(blob) % 4)) % 4
    blob.extend(b"\x00" * pad)
    start = len(blob)
    blob.extend(raw)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=start, byteLength=len(raw), target=ARRAY_BUFFER))
    acc = Accessor(
        bufferView=len(gltf.bufferViews) - 1,
        byteOffset=0,
        componentType=FLOAT,
        count=int(np.asarray(data).shape[0]),
        type=typ,
    )
    if typ == "SCALAR":
        acc.min = [float(np.min(data))]
        acc.max = [float(np.max(data))]
    gltf.accessors.append(acc)
    return len(gltf.accessors) - 1


def fit_pts(src, tgt):
    smin, smax = src.min(0), src.max(0)
    tmin, tmax = tgt.min(0), tgt.max(0)
    src_h = float(smax[1] - smin[1]) or 1.0
    tgt_h = float(tmax[1] - tmin[1]) or 1.0
    scale = tgt_h / src_h
    sc = (smin + smax) * 0.5
    tc = (tmin + tmax) * 0.5
    out = (src - np.array([sc[0], smin[1], sc[2]])) * scale
    out += np.array([tc[0], tmin[1], tc[2]])
    return out.astype(np.float32), scale


def hierarchy_map(tgt_joints, children, W, src_name_to_node):
    """Map an unlabeled humanoid (AniGen joint_N) by parent/child chains.

    Position matching fails across A-pose vs T-pose. Chains do not:
    pelvis → spine → chest(≥3 kids) → neck/head and two arms; two down-legs.
    Laterality is rest-pose X (AniGen +X = Mesh2Motion left).
    """
    joint_set = set(tgt_joints)
    parent_of = {}
    for i, kids in children.items():
        for c in kids:
            parent_of[c] = i
    roots = [j for j in tgt_joints if parent_of.get(j) not in joint_set]
    if not roots:
        return {}
    root = min(roots, key=lambda n: abs(float(W[n][0, 3])))

    def kids_of(n):
        return [c for c in children.get(n, []) if c in joint_set]

    def chain(start):
        seq = [start]
        n = start
        while True:
            k = kids_of(n)
            if len(k) != 1:
                break
            seq.append(k[0])
            n = k[0]
        return seq

    def x_of(n):
        return float(W[n][0, 3])

    def y_of(n):
        return float(W[n][1, 3])

    mapping = {root: "pelvis"}
    rkids = kids_of(root)
    if not rkids:
        return mapping
    spine_kid = max(rkids, key=y_of)
    legs = [c for c in rkids if c != spine_kid]

    for start in legs:
        seq = chain(start)
        side = "_l" if x_of(start) >= 0 else "_r"
        for i, n in enumerate(seq):
            mapping[n] = ("thigh", "calf", "foot", "ball")[min(i, 3)] + side

    spine_seq = [spine_kid]
    n = spine_kid
    chest = spine_kid
    while True:
        k = kids_of(n)
        if len(k) >= 2:
            chest = n
            break
        if not k:
            chest = n
            break
        n = k[0]
        spine_seq.append(n)
        chest = n

    before = [b for b in spine_seq if b != chest]
    if before:
        mapping[before[0]] = "spine_01"
        for extra in before[1:]:
            mapping[extra] = "spine_02"
    mapping[chest] = "spine_03"

    ckids = kids_of(chest)
    if not ckids:
        return {t: src_name_to_node[name] for t, name in mapping.items() if name in src_name_to_node}

    neck = min(ckids, key=lambda c: abs(x_of(c)))
    arms = [c for c in ckids if c != neck]
    nseq = chain(neck)
    mapping[nseq[0]] = "neck_01"
    for extra in nseq[1:]:
        mapping[extra] = "head"

    for start in arms:
        seq = chain(start)
        side = "_l" if x_of(start) >= 0 else "_r"
        for i, n in enumerate(seq):
            mapping[n] = ("clavicle", "upperarm", "lowerarm", "hand")[min(i, 3)] + side

    out = {}
    for tnode, name in mapping.items():
        if name in src_name_to_node:
            out[tnode] = src_name_to_node[name]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("skinned", help="already-skinned GLB (AniGen, UniRig, …)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--m2m", default=None)
    ap.add_argument("--keep", default="Idle_A,Walk,Jog,Walk_Formal,Idle_Talking")
    ap.add_argument(
        "--rest-clip",
        default="Idle_A",
        help="clip whose first frame is the source rest (AniGen is A-pose; M2M bind is T-pose)",
    )
    ap.add_argument("--all-clips", action="store_true")
    args = ap.parse_args()

    root = Path(args.m2m) if args.m2m else Path(__file__).resolve().parents[1] / "vendor" / "mesh2motion-app"
    template = root / "static" / "animations" / "human-base-animations.glb"
    if not template.is_file():
        print(f"m2m_keep: missing {template}", file=sys.stderr)
        return 1

    tgt = GLTF2().load(args.skinned)
    src = GLTF2().load(str(template))
    if not tgt.skins or not src.skins:
        print("m2m_keep: both files need a skin", file=sys.stderr)
        return 1

    tgt_W, tgt_rest_t, tgt_rest_q, children_t = worlds(tgt)
    src_W, src_rest_t, src_rest_q, _ = worlds(src)
    tgt_joints = list(tgt.skins[0].joints)
    src_joints = list(src.skins[0].joints)
    tgt_names = [n.name or f"node{i}" for i, n in enumerate(tgt.nodes)]
    src_names = [n.name or f"node{i}" for i, n in enumerate(src.nodes)]

    src_body = [j for j in src_joints if src_names[j] in BODY]
    src_body_pts = np.stack([src_W[j][:3, 3] for j in src_body])
    tgt_pts = np.stack([tgt_W[j][:3, 3] for j in tgt_joints])
    _, height_scale = fit_pts(tgt_pts, src_body_pts)
    src_name_to_node = {src_names[j]: j for j in src_body}
    # tgt node -> src node  (then invert to src->tgt for the clip copy)
    tgt_to_src = hierarchy_map(tgt_joints, children_t, tgt_W, src_name_to_node)
    node_map = {}
    print(f"m2m_keep: height fit {1.0 / height_scale:.4f} (src is {height_scale:.3f}x taller)")
    for tnode, snode in tgt_to_src.items():
        node_map[snode] = tnode
        print(f"  {tgt_names[tnode]:20s} <- {src_names[snode]}")
    if len(node_map) < 8:
        print("m2m_keep: hierarchy map too thin, refusing to retarget", file=sys.stderr)
        return 1

    src_blob = bytearray(src.binary_blob() or b"")
    tgt_blob = bytearray(tgt.binary_blob() or b"")
    motion_scale = 1.0 / height_scale  # M2M metres -> AniGen units

    def pose_from_clip(anim):
        q = list(src_rest_q)
        t = src_rest_t.copy()
        for ch in anim.channels or []:
            samp = anim.samplers[ch.sampler]
            values = accessor_np(src, src_blob, samp.output)
            node = ch.target.node
            if ch.target.path == "rotation":
                q[node] = qnorm(values.reshape(-1, 4)[0])
            elif ch.target.path == "translation":
                t[node] = values.reshape(-1, 3)[0].astype(np.float32)
        return q, t

    keep = None if args.all_clips else {s.strip() for s in args.keep.split(",") if s.strip()}
    src_anims = list(src.animations or [])
    rest_anim = next((a for a in src_anims if a.name == args.rest_clip), None)
    if rest_anim is not None:
        src_rest_q, src_rest_t = pose_from_clip(rest_anim)
        print(f"m2m_keep: source rest = first frame of {args.rest_clip}")
    if keep:
        src_anims = [a for a in src_anims if a.name in keep]
    if not src_anims:
        print("m2m_keep: no clips matched", file=sys.stderr)
        return 1

    new_anims = []
    for anim in src_anims:
        channels = []
        samplers = []
        for ch in anim.channels or []:
            snode = ch.target.node
            path = ch.target.path
            if snode not in node_map:
                continue
            if path == "scale":
                continue
            tnode = node_map[snode]
            samp = anim.samplers[ch.sampler]
            times = accessor_np(src, src_blob, samp.input)
            values = accessor_np(src, src_blob, samp.output)
            interp = samp.interpolation or "LINEAR"

            if path == "rotation":
                # cubic-spline packs 3x VEC4 per key; treat rows as consecutive quats
                qs = values.reshape(-1, 4).astype(np.float32)
                q0 = src_rest_q[snode]
                qt = tgt_rest_q[tnode]
                out = np.empty_like(qs)
                prev = None
                for i, q in enumerate(qs):
                    rel = qmul(qinv(q0), qnorm(q))
                    o = qnorm(qmul(qt, rel))
                    if prev is not None and float(np.dot(prev, o)) < 0:
                        o = -o
                    out[i] = o
                    prev = o
                values = out.reshape(values.shape)
            elif path == "translation":
                if src_names[snode] not in ("root", "pelvis"):
                    continue
                ts = values.reshape(-1, 3).astype(np.float32)
                rest = src_rest_t[snode]
                t_rest = tgt_rest_t[tnode]
                values = (t_rest + (ts - rest) * motion_scale).reshape(values.shape)
            else:
                continue

            a_in = append_f32(tgt_blob, tgt, times.astype(np.float32), "SCALAR")
            typ = "VEC4" if path == "rotation" else "VEC3"
            a_out = append_f32(tgt_blob, tgt, values.astype(np.float32), typ)
            samplers.append(AnimationSampler(input=a_in, output=a_out, interpolation=interp))
            channels.append(
                AnimationChannel(
                    sampler=len(samplers) - 1,
                    target=AnimationChannelTarget(node=tnode, path=path),
                )
            )
        if not channels:
            continue
        new_anims.append(Animation(name=anim.name, samplers=samplers, channels=channels))
        print(f"m2m_keep: clip {anim.name}  {len(channels)} channels")

    if not new_anims:
        print("m2m_keep: produced no animation channels", file=sys.stderr)
        return 1

    tgt.animations = new_anims
    tgt.buffers[0].byteLength = len(tgt_blob)
    tgt.set_binary_blob(bytes(tgt_blob))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tgt.save_binary(str(out))
    meta = {
        "src": str(Path(args.skinned)),
        "template": str(template),
        "clips": [a.name for a in new_anims],
        "map": {tgt_names[t]: src_names[s] for s, t in node_map.items()},
        "height_scale": float(motion_scale),
        "kept_weights": True,
        "joints": int(len(tgt_joints)),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"m2m_keep: wrote {out} ({out.stat().st_size} bytes) clips={meta['clips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
