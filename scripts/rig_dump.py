#!/usr/bin/env python3
"""Skeleton and clip dump of any rig GLB (vendor templates, rig/template/*, rigged.glb, pred_rig.glb): what an
agent reads instead of writing a dump script.

  rig_dump.py <glb> [--clips <glb>] [--json] [--clip Walk]

dump(glb, clips=None) -> dict with:
  nodes        every node (index, name, TRS/matrix/mesh/skin keys, children)
  joints       the joint tree: parent, depth, world position (and /H), local and world axes (X/Y/Z columns),
               side (L/R/None), chain label (node extras, else rig_pred.label for bone_N rigs, else by name)
  mirror       L/R mirror error about x = cx (/H), worst pairs
  ibm          IBM vs inv(rest world) error
  meshes       mesh+skin nodes: verts, indexed, attributes, influences, weight sums, bbox
  clips        per animation: duration, keys, inferred fps, loop gap (deg), per-joint rotation range about the
               body axes (deg, world delta vs frame 0), root translation range
  errors       structural problems (a skin joint missing, NaNs, a broken IBM accessor); [] when it reads clean
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rig  # noqa: E402

SIDES = (("_L", "_R"), ("_l", "_r"), (".L", ".R"), ("Left", "Right"), ("left", "right"))
BONE_N = re.compile(r"^(bone|joint|j)_?\d+$", re.I)
NAME_LABELS = (("wing", "wing"), ("feather", "wing"), ("tail", "tail"), ("toe", "leg"), ("foot", "leg"), ("leg", "leg"),
               ("ankle", "leg"), ("thigh", "leg"), ("calf", "leg"), ("ball", "leg"), ("scapula", "leg"),
               ("humerus", "leg"), ("mouth", "head"), ("head", "head"), ("nose", "head"), ("jaw", "head"), ("ear", "head"),
               ("eye", "head"), ("neck", "neck"), ("spine", "spine"), ("hips", "spine"), ("pelvis", "spine"), ("ribcage", "spine"), ("root", "root"),
               ("clavicle", "arm"), ("arm", "arm"), ("hand", "arm"), ("finger", "arm"), ("thumb", "arm"),
               ("shoulder", "arm"), ("elbow", "arm"), ("wrist", "arm"), ("knee", "leg"), ("hip", "leg"))


def side_of(name: str) -> str | None:
    for l, r in SIDES:
        if name.endswith(l):
            return "L"
        if name.endswith(r):
            return "R"
    return None


def mirror_name(name: str) -> str | None:
    for l, r in SIDES:
        if name.endswith(l):
            return name[: -len(l)] + r
        if name.endswith(r):
            return name[: -len(r)] + l
    return None


def _decompose(m: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = np.array(m, float).reshape(4, 4).T
    t = M[:3, 3].copy()
    s = np.linalg.norm(M[:3, :3], axis=0)
    rot = M[:3, :3] / np.maximum(s, 1e-12)
    if np.linalg.det(rot) < 0:
        s[0] = -s[0]
        rot[:, 0] = -rot[:, 0]
    return t, R.from_matrix(rot).as_quat(), s


def _locals(doc: dict) -> dict[int, tuple]:
    """Local TRS per node, reading a `matrix` properly (rig.Skeleton reads it as identity)."""
    out = {}
    for i, n in enumerate(doc["nodes"]):
        out[i] = _decompose(n["matrix"]) if "matrix" in n else rig.node_local(n)
    return out


def _label_by_name(name: str) -> str:
    low = name.lower()
    for key, lab in NAME_LABELS:
        if key in low:
            return lab
    return "other"


def _pred_labels(glb: Path, doc: dict) -> dict[str, str]:
    """rig_pred.label chain labels for an unnamed (bone_N) skinned rig; {} when it can't say."""
    try:
        import rig_pred
        p = rig_pred.Pred(glb)
        lab = rig_pred.label(p, None)
        out = {}
        for c in lab["chains"]:
            for k in c["joints"]:
                out[p.names[k]] = c["label"]
        return out
    except Exception:  # noqa: BLE001
        return {}


def _r(x, k: int = 4):
    return np.round(np.asarray(x, float), k).tolist()


def dump(glb, clips=None, clip: str | None = None) -> dict:
    glb = Path(glb)
    doc, b = rig.read_glb(glb)
    errors: list[str] = []
    warnings: list[str] = []
    nodes = doc.get("nodes", [])
    loc = _locals(doc)
    skel = rig.Skeleton(doc)
    skel.rest = loc  # matrix-aware
    world = skel.world({})
    node_list = []
    for i, n in enumerate(nodes):
        node_list.append({"i": i, "name": n.get("name"), "keys": [k for k in ("translation", "rotation", "scale", "matrix", "mesh", "skin") if k in n],
                          "children": n.get("children", []), **({"extras": n["extras"]} if n.get("extras") else {})})
        if "matrix" in n:
            warnings.append(f"node {n.get('name', i)} has a matrix (rig.Skeleton reads it as identity)")
    skins = doc.get("skins", [])
    if not skins:
        errors.append("no skin")
        joints = [i for i, n in enumerate(nodes) if "mesh" not in n]
    else:
        joints = skins[0]["joints"]
        if len(skins) > 1:
            warnings.append(f"{len(skins)} skins (the pipeline reads skins[0])")
        bad = [j for j in joints if j >= len(nodes)]
        if bad:
            errors.append(f"skin joints past the node list: {bad[:5]}")
            joints = [j for j in joints if j < len(nodes)]
    jset = set(joints)
    names = {j: nodes[j].get("name", f"n{j}") for j in joints}
    if len(set(names.values())) != len(names):
        errors.append("duplicate joint names")
    jp = np.array([world[j][0] for j in joints]) if joints else np.zeros((0, 3))
    if joints and not np.isfinite(jp).all():
        errors.append("non-finite joint positions")
    # meshes
    meshes = []
    Pall = []
    for i, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        m = doc["meshes"][n["mesh"]]
        info = {"node": i, "name": n.get("name"), "skin": n.get("skin"), "primitives": len(m["primitives"])}
        pr = m["primitives"][0]
        a = pr["attributes"]
        info["attributes"] = sorted(a)
        info["indexed"] = "indices" in pr
        try:
            P = rig.accessor(doc, b, a["POSITION"]).astype(float)
            Pall.append(P)
            info["verts"] = int(len(P))
            info["bbox"] = [_r(P.min(0)), _r(P.max(0))]
            if "indices" in pr:
                info["tris"] = int(doc["accessors"][pr["indices"]]["count"] // 3)
            if "JOINTS_0" in a and "WEIGHTS_0" in a:
                J = rig.accessor(doc, b, a["JOINTS_0"]).astype(int)
                W = rig.accessor(doc, b, a["WEIGHTS_0"]).astype(float)
                if W.max() > 1.5:
                    W = W / 255.0
                ws = W.sum(1)
                info["influences_max"] = int((W > 1e-6).sum(1).max())
                info["weight_sum"] = [round(float(ws.min()), 4), round(float(ws.max()), 4)]
                nsk = len(doc["skins"][n["skin"]]["joints"]) if "skin" in n and skins else 0
                info["joints_used"] = int(len(np.unique(J[W > 1e-6])))
                if nsk and J.max() >= nsk:
                    errors.append(f"mesh {n.get('name', i)}: JOINTS_0 refers past the skin's {nsk} joints")
                if np.abs(ws - 1).max() > 0.01:
                    warnings.append(f"mesh {n.get('name', i)}: weights sum to {ws.min():.3f}..{ws.max():.3f}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"mesh {n.get('name', i)}: {e}")
        meshes.append(info)
    if Pall:
        PP = np.concatenate(Pall)
        H = float(np.ptp(PP[:, 1]))
        H_from = "mesh"
    else:
        H = float(np.ptp(jp[:, 1])) if len(jp) else 1.0
        H_from = "joints"
    H = max(H, 1e-9)
    # labels
    labels: dict[str, str] = {}
    if any(BONE_N.match(nm) for nm in names.values()) and meshes:
        labels = _pred_labels(glb, doc)
    # joint tree
    depth = {}
    for i in skel.order:
        p = skel.parent.get(i)
        depth[i] = depth.get(p, -1) + 1 if p in depth else 0
    mids = [j for j in joints if side_of(names[j]) is None]
    cx = float(np.mean([world[j][0][0] for j in mids])) if mids else (float(jp[:, 0].mean()) if len(jp) else 0.0)
    jt = []
    for j in joints:
        n = nodes[j]
        p = skel.parent.get(j)
        lr = R.from_quat(loc[j][1]).as_matrix()
        wr = world[j][1].as_matrix()
        ex = n.get("extras") or {}
        lab = ex.get("role") or ex.get("chain") or labels.get(names[j]) or _label_by_name(names[j])
        jt.append({"i": j, "name": names[j], "parent": nodes[p].get("name") if p is not None else None,
                   "parent_is_joint": p in jset, "depth": depth.get(j, 0), "side": side_of(names[j]), "label": lab,
                   "label_from": "extras" if (ex.get("role") or ex.get("chain")) else "rig_pred" if names[j] in labels else "name",
                   "pos": _r(world[j][0]), "pos_H": _r((world[j][0] - [cx, 0, 0]) / H, 3),
                   "local_t": _r(loc[j][0]), "local_q": _r(loc[j][1]),
                   "local_axes": {"x": _r(lr[:, 0], 3), "y": _r(lr[:, 1], 3), "z": _r(lr[:, 2], 3)},
                   "world_axes": {"x": _r(wr[:, 0], 3), "y": _r(wr[:, 1], 3), "z": _r(wr[:, 2], 3)},
                   "children": [nodes[c].get("name") for c in n.get("children", []) if c in jset]})
    roots = [names[j] for j in joints if skel.parent.get(j) not in jset]
    # mirror
    pos = {names[j]: world[j][0] for j in joints}
    pairs = []
    for nm in pos:
        if side_of(nm) == "L":
            o = mirror_name(nm)
            if o in pos:
                a = pos[nm].copy()
                a[0] = 2 * cx - a[0]
                pairs.append((float(np.linalg.norm(a - pos[o]) / H), nm, o))
    pairs.sort(reverse=True)
    unpaired = sorted(nm for nm in pos if side_of(nm) and mirror_name(nm) not in pos)
    mirror = {"cx": round(cx, 5), "pairs": len(pairs), "max_H": round(pairs[0][0], 5) if pairs else 0.0,
              "mean_H": round(float(np.mean([p[0] for p in pairs])), 5) if pairs else 0.0,
              "worst": [{"L": p[1], "R": p[2], "err_H": round(p[0], 5)} for p in pairs[:5]], "unpaired": unpaired}
    # IBM
    ibm = {"present": False}
    if skins and "inverseBindMatrices" in skins[0]:
        try:
            I = rig.accessor(doc, b, skins[0]["inverseBindMatrices"])
            if len(I) != len(skins[0]["joints"]):
                errors.append(f"IBM count {len(I)} != joints {len(skins[0]['joints'])}")
            I = I.reshape(-1, 4, 4).transpose(0, 2, 1)
            errs = []
            for k, j in enumerate(joints[: len(I)]):
                M = np.eye(4)
                M[:3, :3] = world[j][1].as_matrix() * 1.0
                M[:3, 3] = world[j][0]
                errs.append(float(np.abs(I[k] @ M - np.eye(4)).max()))
            acc = doc["accessors"][skins[0]["inverseBindMatrices"]]
            view = doc["bufferViews"][acc["bufferView"]]
            k = int(np.argmax(errs)) if errs else 0
            ibm = {"present": True, "max_err": float(max(errs) if errs else 0.0), "worst": names[joints[k]] if errs else None,
                   "packed_f32": acc["componentType"] == 5126 and view.get("byteStride", 64) == 64}
            if not np.isfinite(I).all():
                errors.append("non-finite IBMs")
        except Exception as e:  # noqa: BLE001
            errors.append(f"IBM accessor: {e}")
    elif skins:
        warnings.append("no inverseBindMatrices")
    out = {"file": str(glb), "nodes": node_list, "joint_count": len(joints), "roots": roots, "H": round(H, 6), "H_from": H_from,
           "joints": jt, "mirror": mirror, "ibm": ibm, "meshes": meshes, "skins": len(skins),
           "scene_nodes": [nodes[i].get("name") for sc in doc.get("scenes", []) for i in sc.get("nodes", [])],
           "errors": errors, "warnings": warnings}
    # clips
    sources = [(glb, doc, b)]
    if clips:
        cdoc, cb = rig.read_glb(Path(clips))
        sources.append((Path(clips), cdoc, cb))
    cl = []
    for path, d, bb in sources:
        if not d.get("animations"):
            continue
        cskel = rig.Skeleton(d)
        cskel.rest = _locals(d)
        cnames = {n.get("name"): i for i, n in enumerate(d["nodes"]) if n.get("name")}
        cj = [cnames[names[j]] for j in joints if names[j] in cnames]
        miss = [names[j] for j in joints if names[j] not in cnames]
        if miss and path != glb:
            warnings.append(f"{path.name}: {len(miss)} of the joints are not in the clip file ({miss[:3]})")
        root_i = cnames.get(roots[0]) if roots else None
        for an in d["animations"]:
            if clip and an.get("name") != clip:
                continue
            try:
                cl.append(_clip_stats(d, bb, an, cskel, cj, root_i, H, path.name, full=bool(clip)))
            except Exception as e:  # noqa: BLE001
                errors.append(f"clip {an.get('name')}: {e}")
    if clip and not cl:
        errors.append(f"no clip named {clip}")
    out["clips"] = cl
    return out


def _clip_stats(d: dict, bb: bytes, an: dict, skel: rig.Skeleton, cj: list[int], root_i: int | None, H: float,
                src: str, full: bool = False) -> dict:
    inputs = [rig.accessor(d, bb, s["input"])[:, 0] for s in an["samplers"]]
    keys = max(len(t) for t in inputs)
    dur = max(float(t[-1]) for t in inputs)
    t0 = min(float(t[0]) for t in inputs)
    fps = (keys - 1) / max(dur - t0, 1e-9) if keys > 1 else None
    paths = {}
    for ch in an["channels"]:
        paths[ch["target"]["path"]] = paths.get(ch["target"]["path"], 0) + 1
    gap = 0.0
    for ch in an["channels"]:
        if ch["target"]["path"] == "rotation":
            v = rig.accessor(d, bb, an["samplers"][ch["sampler"]]["output"])
            if len(v) > 1:
                dd = abs(float(np.dot(v[0], v[-1]))) / max(float(np.linalg.norm(v[0]) * np.linalg.norm(v[-1])), 1e-12)
                gap = max(gap, float(np.degrees(2 * np.arccos(min(1.0, dd)))))
    chans = rig.prepare_clip(d, bb, an, skel)
    n = int(min(max(keys, 2), 61))
    times = np.linspace(t0, dur, n)
    W0 = skel.world(rig.sample_clip(chans, skel, float(times[0])))
    rv = {j: [] for j in cj}
    rootp = []
    for t in times:
        Wt = skel.world(rig.sample_clip(chans, skel, float(t)))
        for j in cj:
            rv[j].append((Wt[j][1] * W0[j][1].inv()).as_rotvec())
        if root_i is not None:
            rootp.append(Wt[root_i][0])
    rng = {}
    for j in cj:
        a = np.degrees(np.array(rv[j]))
        rng[skel.nodes[j].get("name")] = _r(a.max(0) - a.min(0), 1)
    top = sorted(rng.items(), key=lambda kv: -max(kv[1]))
    out = {"name": an.get("name"), "source": src, "duration": round(dur, 4), "start": round(t0, 4), "keys": keys,
           "fps": round(fps, 2) if fps else None, "channels": paths, "loop_gap_deg": round(gap, 3),
           "most_moving": [{"joint": k, "range_deg_xyz": v} for k, v in top[:8]]}
    if an.get("extras"):
        out["extras"] = an["extras"]
    if rootp:
        rp = np.array(rootp)
        out["root_range"] = _r(np.ptp(rp, 0), 5)
        out["root_range_H"] = _r(np.ptp(rp, 0) / H, 4)
    if full:
        out["rotation_range_deg"] = rng
    return out


def text(d: dict) -> str:
    L = [f"{d['file']}", f"  {d['joint_count']} joints, roots {d['roots']}, H {d['H']:.4f} (from {d['H_from']}), "
         f"{d['skins']} skin(s), scene {d['scene_nodes']}"]
    if d["ibm"].get("present"):
        L.append(f"  IBM err {d['ibm']['max_err']:.2e} (worst {d['ibm']['worst']}), packed f32 {d['ibm']['packed_f32']}")
    m = d["mirror"]
    L.append(f"  mirror about x={m['cx']}: {m['pairs']} pairs, max {m['max_H']:.4f}H" + (f" ({m['worst'][0]['L']})" if m["worst"] else "")
             + (f", unpaired {m['unpaired'][:4]}" if m["unpaired"] else ""))
    for me in d["meshes"]:
        L.append(f"  mesh {me.get('name')}: {me.get('verts')} verts, {me.get('tris', '?')} tris, indexed {me.get('indexed')}, "
                 f"attrs {','.join(me.get('attributes', []))}" + (f", <= {me['influences_max']} influences, {me['joints_used']} bones used" if "influences_max" in me else ""))
    L.append("  joints (name  side  label  pos/H [x-cx, y, z]  local Y axis):")
    for j in d["joints"]:
        L.append("    " + "  " * j["depth"] + f"{j['name']}  {j['side'] or '-'}  {j['label']}  "
                 f"[{j['pos_H'][0]:+.3f} {j['pos_H'][1]:+.3f} {j['pos_H'][2]:+.3f}]  y={j['local_axes']['y']}")
    for c in d.get("clips", []):
        L.append(f"  clip {c['name']} ({c['source']}): {c['duration']}s, {c['keys']} keys, {c['fps']} fps, loop gap {c['loop_gap_deg']}°, "
                 f"channels {c['channels']}" + (f", root range/H {c['root_range_H']}" if "root_range_H" in c else ""))
        for mm in c["most_moving"][:5]:
            L.append(f"      {mm['joint']}: {mm['range_deg_xyz']}°")
        if "rotation_range_deg" in c:
            for k, v in c["rotation_range_deg"].items():
                L.append(f"      . {k}: {v}°")
    for e in d["errors"]:
        L.append(f"  ERROR {e}")
    for w in d["warnings"]:
        L.append(f"  warn {w}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb")
    ap.add_argument("--clips", help="a clip GLB to read animations from (joints matched by name)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--clip", help="only this clip, with every joint's rotation range")
    a = ap.parse_args(argv)
    d = dump(Path(a.glb), a.clips, a.clip)
    print(json.dumps(d, indent=1) if a.json else text(d))
    return 1 if d["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
