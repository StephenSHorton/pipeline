#!/usr/bin/env python3
"""A template built for one asset: write its rig and clip GLBs, verify them, and load them as a template.

The contract is design §1: a template is a dict `{rig, clips[], ref, default[]}` (paths joined as `M2M / path`;
absolute paths survive the join). A built one adds `{class, plan, built: true, dir, built_for}` and lives as an
object inside `rig.TEMPLATES` under `built:<stem>`, so identity lookups (`rig._template_class`) work.

Layout (both GLBs):
- nodes 0..N-1 are the joints in skeleton order (parents before children, `root` first), so a joint's skin index
  is its node index, the same as skin-tokens' `skinned_raw.glb`;
- node N is the `Armature` holder (no TRS), children [root];
- clips.glb adds node N+1 `Body` (mesh 0, skin 0).
Joints carry translations only (identity rest rotations). The IBMs are a packed float32 MAT4 accessor equal to
inv(rest world).

CLI:
  rig_template.py verify --stem S        writes rig/template/validate.json
  rig_template.py show --stem S          prints template.json and the TEMPLATES key
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rig  # noqa: E402

FPS = rig.FPS
REF = "Rest"
SIDES = (("_L", "_R"), ("_l", "_r"))


# ---------------------------------------------------------------- small helpers

def _skel_dict(skel_json) -> dict:
    if isinstance(skel_json, (str, Path)):
        return json.loads(Path(skel_json).read_text())
    return skel_json


def _ordered(joints: list[dict]) -> list[dict]:
    """Joints parents-first, keeping the given order otherwise (root first)."""
    by = {j["name"]: j for j in joints}
    out, seen = [], set()

    def visit(j: dict, stack: tuple = ()) -> None:
        if j["name"] in seen:
            return
        if j["name"] in stack:
            raise ValueError(f"skeleton: a cycle through {j['name']}")
        p = j.get("parent")
        if p is not None:
            if p not in by:
                raise ValueError(f"skeleton: {j['name']}'s parent {p} is not a joint")
            visit(by[p], stack + (j["name"],))
        seen.add(j["name"])
        out.append(j)

    for j in joints:
        visit(j)
    return out


def _ibm(world_pos: np.ndarray, world_rot: list | None = None) -> np.ndarray:
    """Packed column-major inverse binds (the convention posed_template and ours_from_posed use)."""
    out = []
    for k, p in enumerate(world_pos):
        M = np.eye(4)
        if world_rot is not None:
            M[:3, :3] = world_rot[k]
        M[:3, 3] = p
        out.append(np.linalg.inv(M).T.reshape(-1))
    return np.array(out, "<f4")


def _glb_bytes(doc: dict, binary: bytes) -> bytes:
    import struct
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    body = binary + b"\0" * ((4 - len(binary) % 4) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(body))
    return out + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(body), 0x004E4942) + body


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


def _names_in(x) -> set[str]:
    """Every string in a nested chains/paths structure."""
    if x is None:
        return set()
    if isinstance(x, str):
        return {x}
    if isinstance(x, dict):
        return set().union(*(_names_in(v) for v in x.values())) if x else set()
    if isinstance(x, (list, tuple)):
        return set().union(*(_names_in(v) for v in x)) if x else set()
    return set()


def required_names(plan: dict | None, cls: str | None) -> set[str]:
    """Joint names the class tables (rig_fit.CHAINS, rig_pred.PATHS, or the plan's own paths) refer to."""
    import rig_fit
    import rig_pred
    need = _names_in(rig_fit.CHAINS.get(cls or "", {}))
    paths = dict(rig_pred.PATHS.get(cls or "", {}))
    if plan and isinstance(plan.get("paths"), dict):
        paths.update(plan["paths"])
    for k in ("body", "tail", "legs", "groups", "neck_lift", "neck_from"):
        need |= _names_in(paths.get(k))
    return need


# ---------------------------------------------------------------- writers

def write_rig(skel_json, path: Path) -> dict:
    """skeleton.json ({joints:[{name, parent, pos, role?, side?, source?}], H, cx}) -> rig GLB: TRS-only joints with
    identity rests (translations only), an `Armature` holder, root `root`, packed float32 IBMs. Returns the doc."""
    sk = _skel_dict(skel_json)
    joints = _ordered(sk["joints"])
    roots = [j for j in joints if j.get("parent") is None]
    if len(roots) != 1 or roots[0]["name"] != "root":
        raise ValueError(f"skeleton: needs exactly one root joint named 'root' (got {[j['name'] for j in roots]})")
    names = [j["name"] for j in joints]
    if len(set(names)) != len(names):
        raise ValueError("skeleton: duplicate joint names")
    if "fit" in names or "Armature" in names:
        raise ValueError("skeleton: 'fit' and 'Armature' are reserved node names")
    k_of = {n: k for k, n in enumerate(names)}
    pos = np.array([j["pos"] for j in joints], float)
    nodes = []
    for k, j in enumerate(joints):
        p = j.get("parent")
        t = pos[k] - (pos[k_of[p]] if p is not None else 0.0)
        nd = {"name": j["name"], "translation": np.asarray(t, float).tolist()}
        kids = [c for c, jj in enumerate(joints) if jj.get("parent") == j["name"]]
        if kids:
            nd["children"] = kids
        ex = {key: j[key] for key in ("role", "side", "source") if j.get(key) is not None}
        if ex:
            nd["extras"] = ex
        nodes.append(nd)
    N = len(nodes)
    nodes.append({"name": "Armature", "children": [0]})
    g = rig.GlbWriter()
    g.doc["asset"]["generator"] = "ai-asset-pipeline rig_template.py"
    acc = g.array(_ibm(pos), "MAT4")
    g.doc.update(nodes=nodes, scenes=[{"name": "Scene", "nodes": [N]}], scene=0,
                 skins=[{"name": "Armature", "joints": list(range(N)), "inverseBindMatrices": acc, "skeleton": 0}])
    ex = {k: sk[k] for k in ("H", "cx") if k in sk}
    if ex:
        g.doc["extras"] = ex
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.write(path)
    return g.doc


def write_posed(rig_glb: Path, out: Path) -> dict:
    """rig/template_posed.glb: the rig plus a top `fit` node (extras yaw 0 / scale 1 / offset 0 / built true), the
    layout posed_template produces. Every joint gets explicit identity rotation and unit scale."""
    doc, b = rig.read_glb(Path(rig_glb))
    skel = rig.Skeleton(doc)
    joints = set(doc["skins"][0]["joints"])
    for i in joints:
        n = doc["nodes"][i]
        n.setdefault("translation", [0.0, 0.0, 0.0])
        n.setdefault("rotation", [0.0, 0.0, 0.0, 1.0])
        n.setdefault("scale", [1.0, 1.0, 1.0])
    tops = [i for i in range(len(doc["nodes"])) if i not in skel.parent]
    doc["nodes"].append({"name": "fit", "children": tops, "extras": {"yaw_deg": 0, "scale": [1.0, 1.0, 1.0],
                                                                      "offset": [0.0, 0.0, 0.0], "built": True}})
    fit = len(doc["nodes"]) - 1
    scenes = doc.get("scenes") or [{"nodes": tops}]
    for sc in scenes:
        sc["nodes"] = [fit] + [n for n in sc.get("nodes", []) if n not in tops]
    doc["scenes"] = scenes
    doc["scene"] = doc.get("scene", 0)
    doc.pop("animations", None)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_glb_bytes(doc, b))
    return doc


def ref_store(rig_glb: Path, fitinfo: dict) -> dict:
    """{joint_name: [t, q, s] of the rest TRS} + {"__fit__": fitinfo}, the shape bake_clips reads.

    fitinfo is completed with the built defaults: yaw 0, scale [1,1,1], offset [0,0,0], unsupported [] (it MUST be
    empty: otherwise local_weights' skip list and the unsupported-clip block fire on the wing bones),
    registration {built, scale_ratio 1, poor_match false}, facing {yaw 0, decided_by 'built'} unless given."""
    doc, _ = rig.read_glb(Path(rig_glb))
    skel = rig.Skeleton(doc)
    store = {}
    for i, (t, q, s) in skel.rest.items():
        nm = skel.nodes[i].get("name")
        if nm and nm != "fit":
            store[nm] = [t.tolist(), q.tolist(), s.tolist()]
    fi = dict(fitinfo or {})
    fi["yaw"] = 0
    fi["scale"] = [1.0, 1.0, 1.0]
    fi["offset"] = [0.0, 0.0, 0.0]
    fi["unsupported"] = []
    reg = dict(fi.get("registration") or {})
    reg.setdefault("built", True)
    reg.setdefault("scale_ratio", 1.0)
    reg.setdefault("poor_match", False)
    fi["registration"] = reg
    fac = dict(fi.get("facing") or {})
    fac.setdefault("yaw", 0)
    fac.setdefault("decided_by", "built")
    fi["facing"] = fac
    fi.setdefault("built", True)
    store["__fit__"] = fi
    return store


def write_skinned_glb(out: Path, ours: dict, obin: bytes, src: dict, sbin: bytes, J: np.ndarray, W: np.ndarray,
                      clips: list[dict], anim_extras: dict | None = None) -> None:
    """The rig.write_rigged layout (a copy; rig.py is not edited): `ours`' nodes, scenes and skin + the source
    mesh's first primitive (POSITION, NORMAL, TEXCOORD_0, indices, material, texture) with J/W on the node(s) that
    carry `mesh` + clips as LINEAR animations (rotation per entry of c["rot"], translation on c["root"])."""
    g = rig.GlbWriter()
    g.doc["asset"]["generator"] = "ai-asset-pipeline rig_template.py"
    sp = src["meshes"][0]["primitives"][0]
    attrs = {}
    for key in ("POSITION", "NORMAL", "TEXCOORD_0"):
        if key in sp["attributes"]:
            arr = rig.accessor(src, sbin, sp["attributes"][key]).astype("<f4")
            attrs[key] = g.array(arr, {"POSITION": "VEC3", "NORMAL": "VEC3", "TEXCOORD_0": "VEC2"}[key], 34962, key == "POSITION")
    attrs["JOINTS_0"] = g.array(np.asarray(J).astype("<u2"), "VEC4", 34962)
    attrs["WEIGHTS_0"] = g.array(np.asarray(W).astype("<f4"), "VEC4", 34962)
    prim = {"attributes": attrs}
    if "indices" in sp:
        idx = rig.accessor(src, sbin, sp["indices"]).reshape(-1).astype("<u4")
        prim["indices"] = g.array(idx, "SCALAR", 34963)
    if "material" in sp and src.get("materials"):
        mat = json.loads(json.dumps(src["materials"][sp["material"]]))
        g.doc["materials"] = [mat]
        prim["material"] = 0
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        if "metallicRoughnessTexture" not in pbr:
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = 1.0
        tex_ref = pbr.get("baseColorTexture")
        if tex_ref is not None and src.get("textures"):
            tex = src["textures"][tex_ref["index"]]
            img = src["images"][tex["source"]]
            if "bufferView" in img:
                v = src["bufferViews"][img["bufferView"]]
                data = sbin[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
                g.doc["images"] = [{"bufferView": g.blob(data), "mimeType": img.get("mimeType", "image/png")}]
                g.doc["samplers"] = [src["samplers"][tex["sampler"]]] if "sampler" in tex and src.get("samplers") else [{}]
                g.doc["textures"] = [{"source": 0, "sampler": 0}]
                mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
            else:
                pbr.pop("baseColorTexture", None)
    g.doc["meshes"] = [{"name": "model", "primitives": [prim]}]
    nodes = json.loads(json.dumps(ours["nodes"]))
    for n in nodes:
        if "mesh" in n:
            n["mesh"] = 0
            n["skin"] = 0
    g.doc["nodes"] = nodes
    g.doc["scenes"] = ours["scenes"]
    g.doc["scene"] = ours.get("scene", 0)
    skin = json.loads(json.dumps(ours["skins"][0]))
    skin["inverseBindMatrices"] = g.array(rig.accessor(ours, obin, skin["inverseBindMatrices"]).astype("<f4"), "MAT4")
    g.doc["skins"] = [skin]
    anims = []
    for c in clips:
        times = np.asarray(c["times"], "<f4")
        tacc = g.array(times.reshape(-1, 1), "SCALAR")
        g.doc["accessors"][tacc]["min"] = [float(times[0])]
        g.doc["accessors"][tacc]["max"] = [float(times[-1])]
        samplers, channels = [], []
        for node, q in c["rot"].items():
            samplers.append({"input": tacc, "output": g.array(np.asarray(q, "<f4"), "VEC4"), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": int(node), "path": "rotation"}})
        samplers.append({"input": tacc, "output": g.array(np.asarray(c["root_t"], "<f4"), "VEC3"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": int(c["root"]), "path": "translation"}})
        an = {"name": c["name"], "samplers": samplers, "channels": channels}
        ex = (anim_extras or {}).get(c["name"])
        if ex:
            an["extras"] = ex
        anims.append(an)
    if anims:
        g.doc["animations"] = anims
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.write(out)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else None
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return None


def write_clips(rig_glb: Path, src_doc: dict, src_bin: bytes, J: np.ndarray, W: np.ndarray, clips: list[dict],
                path: Path, names: list[str] | None = None) -> dict:
    """clips.glb: the rig's joints (same names, order and IBMs) + a `Body` node (the source mesh with J/W) + a real
    2-key `Rest` animation (t = 0 and 1/30, both rest) + every clip, LINEAR at 30 fps, a rotation channel per joint
    (rest-filled where a clip has none) and translation on root only.

    Clip dicts have the bake_clips shape {name, times, rot {joint: (F,4) local xyzw}, root, root_t (F,3), loop?, ...}.
    `rot` keys and `root` are node indices of the rig GLB (== skin indices), or joint names. `names`, when given,
    lists the joint names in the caller's index order (e.g. skinned_raw.glb's skin order): J and integer rot/root
    keys are then remapped onto the rig's order by name. Returns {clips:[names], joints, verts}."""
    rdoc, rbin = rig.read_glb(Path(rig_glb))
    joints = rdoc["skins"][0]["joints"]
    rnames = [rdoc["nodes"][i]["name"] for i in joints]
    node_of = {n: joints[k] for k, n in enumerate(rnames)}
    skin_of = {n: k for k, n in enumerate(rnames)}
    J = np.asarray(J).astype(int)
    W = np.asarray(W, float)
    if names is not None:
        missing = [n for n in names if n not in skin_of]
        if missing:
            raise ValueError(f"write_clips: joints not in the rig: {missing[:5]}")
        remap = np.array([skin_of[n] for n in names])
        J = remap[J]

    def node(key) -> int:
        if isinstance(key, str):
            return node_of[key]
        k = int(key)
        if names is not None:
            return node_of[names[k]]
        return k

    root_node = node_of["root"]
    skel = rig.Skeleton(rdoc)
    rest_root_t = skel.rest[root_node][0]
    rest = {"name": REF, "times": np.array([0.0, 1.0 / FPS], "<f4"),
            "rot": {i: np.tile(skel.rest[i][1], (2, 1)).astype("<f4") for i in joints},
            "root": root_node, "root_t": np.tile(rest_root_t, (2, 1)).astype("<f4")}
    out_clips = [rest]
    extras = {REF: {"loop": True, "ref": True}}
    for c in clips:
        if c["name"] == REF:
            raise ValueError("write_clips: 'Rest' is reserved for the reference clip")
        times = np.asarray(c["times"], "<f4")
        F = len(times)
        rot = {}
        for k, q in c["rot"].items():
            rot[node(k)] = np.asarray(q, "<f4").reshape(F, 4)
        for i in joints:
            rot.setdefault(i, np.tile(skel.rest[i][1], (F, 1)).astype("<f4"))
        rn = node(c.get("root", "root"))
        if rn != root_node:
            raise ValueError(f"write_clips: {c['name']}'s root translation is not on 'root'")
        rt = c.get("root_t")
        rt = np.tile(rest_root_t, (F, 1)) if rt is None else np.asarray(rt, "<f4").reshape(F, 3)
        out_clips.append({"name": c["name"], "times": times, "rot": rot, "root": root_node, "root_t": rt})
        ex = {k: _jsonable(c[k]) for k in ("loop", "params", "energy") if k in c}
        if ex:
            extras[c["name"]] = ex
    ours = json.loads(json.dumps(rdoc))
    ours["nodes"].append({"name": "Body", "mesh": 0, "skin": 0})
    body = len(ours["nodes"]) - 1
    for sc in ours.get("scenes", []):
        sc["nodes"] = list(sc.get("nodes", [])) + [body]
    write_skinned_glb(Path(path), ours, rbin, src_doc, src_bin, J, W, out_clips, extras)
    return {"clips": [c["name"] for c in out_clips], "joints": len(joints), "verts": int(len(J))}


# ---------------------------------------------------------------- verify

def _quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(np.sum(np.asarray(a, float) * np.asarray(b, float), axis=-1))
    na = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.degrees(2 * np.arccos(np.clip(d / np.maximum(na, 1e-12), -1, 1)))


def ibm_error(doc: dict, binary: bytes) -> float:
    """max |IBM_k - inv(rest world_k)| over the skin's joints."""
    skel = rig.Skeleton(doc)
    w = skel.world({})
    skn = doc["skins"][0]
    if "inverseBindMatrices" not in skn:
        return float("inf")
    I = rig.accessor(doc, binary, skn["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    err = 0.0
    for k, j in enumerate(skn["joints"]):
        M = np.eye(4)
        M[:3, :3] = w[j][1].as_matrix()
        M[:3, 3] = w[j][0]
        err = max(err, float(np.abs(I[k] - np.linalg.inv(M)).max()))
    return err


def _verify(rig_glb: Path, clips_glb: Path, plan: dict | None) -> tuple[list[dict], dict]:
    probs: list[dict] = []
    facts: dict = {}

    def bad(key: str, msg: str, level: str = "error") -> None:
        probs.append({"level": level, "key": key, "msg": msg})

    plan = plan or {}
    cls = plan.get("class")
    rdoc, rbin = rig.read_glb(Path(rig_glb))
    cdoc, cbin = rig.read_glb(Path(clips_glb))

    # -- rig GLB
    if len(rdoc.get("skins", [])) != 1:
        bad("skin", f"the rig has {len(rdoc.get('skins', []))} skins (needs exactly 1)")
        return probs, facts
    rj = rdoc["skins"][0]["joints"]
    rnames = [rdoc["nodes"][i].get("name") for i in rj]
    facts["joints"] = len(rj)
    for which, doc in (("rig", rdoc), ("clips", cdoc)):
        for i, n in enumerate(doc["nodes"]):
            if "matrix" in n:
                bad("matrix", f"{which}: node {n.get('name', i)} has a matrix (TRS only)")
            if n.get("name") == "fit":
                bad("fit", f"{which}: a node is named 'fit'")
    rskel = rig.Skeleton(rdoc)
    roots = [i for i in rj if rskel.parent.get(i) not in rj]
    if len(roots) != 1 or rdoc["nodes"][roots[0]].get("name") != "root":
        bad("root", "needs exactly one root joint named 'root' (got " + ", ".join(str(rdoc["nodes"][i].get("name")) for i in roots) + ")")
    elif "hips" in rnames and rskel.parent.get(rskel.index["hips"]) != roots[0]:
        bad("root", "hips is not root's child", "warn")
    arm = [i for i, n in enumerate(rdoc["nodes"]) if n.get("name") == "Armature"]
    if not arm:
        bad("armature", "no Armature holder node", "warn")
    elif any(k in rdoc["nodes"][arm[0]] for k in ("translation", "rotation", "scale", "matrix")):
        bad("armature", "the Armature holder has a transform")
    if rdoc.get("meshes") or rdoc.get("animations"):
        bad("rig", f"the rig GLB has {len(rdoc.get('meshes', []))} meshes and {len(rdoc.get('animations', []))} animations (needs 0 and 0)")
    nonid = [rdoc["nodes"][i].get("name") for i in rj if np.abs(rskel.rest[i][1] - [0, 0, 0, 1]).max() > 1e-6
             or np.abs(rskel.rest[i][2] - 1).max() > 1e-6]
    if nonid:
        bad("rests", "built rigs keep identity rest rotations and unit scale; not on " + ", ".join(nonid[:6]))
    err = ibm_error(rdoc, rbin)
    facts["ibm_err"] = err
    if not err < 1e-5:
        bad("ibm", f"rig IBM error {err:.2e} (needs < 1e-5)")
    acc = rdoc["accessors"][rdoc["skins"][0]["inverseBindMatrices"]] if "inverseBindMatrices" in rdoc["skins"][0] else None
    if acc is not None:
        view = rdoc["bufferViews"][acc["bufferView"]]
        if acc["componentType"] != 5126 or acc["type"] != "MAT4" or view.get("byteStride", 64) != 64:
            bad("ibm", "the IBM accessor is not a packed float32 MAT4")

    # -- clips GLB vs rig
    if len(cdoc.get("skins", [])) != 1:
        bad("skin", f"clips.glb has {len(cdoc.get('skins', []))} skins (needs exactly 1)")
        return probs, facts
    cj = cdoc["skins"][0]["joints"]
    cnames = [cdoc["nodes"][i].get("name") for i in cj]
    if cnames != rnames:
        bad("names", "joint names/order differ between rig.glb and clips.glb")
    if cj != rj:
        bad("names", "joint node indices differ between rig.glb and clips.glb")
    cerr = ibm_error(cdoc, cbin)
    if not cerr < 1e-5:
        bad("ibm", f"clips.glb IBM error {cerr:.2e}")
    if cnames == rnames:
        I1 = rig.accessor(rdoc, rbin, rdoc["skins"][0]["inverseBindMatrices"])
        I2 = rig.accessor(cdoc, cbin, cdoc["skins"][0]["inverseBindMatrices"])
        if I1.shape != I2.shape or np.abs(I1 - I2).max() > 1e-6:
            bad("ibm", "IBMs differ between rig.glb and clips.glb")
    need = required_names(plan, cls)
    miss = sorted(need - set(rnames))
    if miss:
        bad("names", f"joints the {cls} tables need are missing: " + ", ".join(miss[:10]))
    if plan.get("joints"):
        try:
            import rig_plan
            want = set(rig_plan.joint_names(plan))
            if want != set(rnames):
                bad("names", f"joints differ from the plan's: missing {sorted(want - set(rnames))[:6]}, extra {sorted(set(rnames) - want)[:6]}")
        except Exception as e:  # noqa: BLE001
            bad("names", f"couldn't expand the plan's joints ({e})", "warn")

    # -- body
    body = [i for i, n in enumerate(cdoc["nodes"]) if "mesh" in n and "skin" in n]
    facts["body"] = {"verts": 0, "skinned": False}
    if not body:
        bad("body", "no node with both mesh and skin in clips.glb")
        Pm = None
    else:
        bn = cdoc["nodes"][body[0]]
        if bn.get("name") != "Body":
            bad("body", f"the mesh+skin node is named {bn.get('name')!r} (expected 'Body')", "warn")
        pr = cdoc["meshes"][bn["mesh"]]["primitives"][0]
        a = pr["attributes"]
        if "indices" not in pr:
            bad("body", "the Body primitive is not indexed")
        for key in ("POSITION", "JOINTS_0", "WEIGHTS_0"):
            if key not in a:
                bad("body", f"the Body primitive has no {key}")
        for key in ("NORMAL", "TEXCOORD_0"):
            if key not in a:
                bad("body", f"the Body primitive has no {key}", "warn")
        Pm = rig.accessor(cdoc, cbin, a["POSITION"]).astype(float) if "POSITION" in a else None
        skinned = "JOINTS_0" in a and "WEIGHTS_0" in a
        if skinned:
            Jb = rig.accessor(cdoc, cbin, a["JOINTS_0"]).astype(int)
            Wb = rig.accessor(cdoc, cbin, a["WEIGHTS_0"]).astype(float)
            if Jb.max() >= len(cj):
                bad("body", "JOINTS_0 refers past the skin's joints")
            ws = Wb.sum(1)
            if np.abs(ws - 1).max() > 1e-3:
                bad("body", f"weights don't sum to 1 (worst {np.abs(ws - 1).max():.3f})", "warn")
        facts["body"] = {"verts": 0 if Pm is None else int(len(Pm)), "skinned": bool(skinned)}

    # -- frame axes and floor
    w = rskel.world({})
    pos = {rdoc["nodes"][i]["name"]: w[i][0] for i in rj}
    jp = np.array([pos[n] for n in rnames])
    H = float(np.ptp(Pm[:, 1])) if Pm is not None and len(Pm) else float(np.ptp(jp[:, 1]))
    H = max(H, 1e-9)
    mids = [n for n in rnames if side_of(n) is None]
    cx = float(np.mean([pos[n][0] for n in mids])) if mids else float(jp[:, 0].mean())
    axes = {"cx": round(cx, 6)}
    if "head" in pos and "hips" in pos:
        axes["head_z_minus_hips_z"] = round(float(pos["head"][2] - pos["hips"][2]), 6)
        if pos["head"][2] <= pos["hips"][2]:
            bad("axes", "head is not ahead of hips in +Z (the template must face +Z)")
    for l, r in SIDES:
        for s in (l, r):
            tt, ft = f"Toes_tip{s}", f"Foot{s}"
            if tt in pos and ft in pos:
                axes[f"toes{s}"] = round(float(pos[tt][2] - pos[ft][2]), 6)
                if pos[tt][2] <= pos[ft][2]:
                    bad("axes", f"{tt} is not ahead of {ft} in +Z")
    wrong = [n for n in rnames if side_of(n) == "L" and pos[n][0] <= cx] + [n for n in rnames if side_of(n) == "R" and pos[n][0] >= cx]
    axes["wrong_side"] = wrong
    if wrong:
        bad("axes", "joints on the wrong side of x = cx (+X is the creature's left): " + ", ".join(wrong[:6]))
    facts["axes"] = axes
    fy = float(jp[:, 1].min())
    facts["floor_y"] = {"joints": round(fy, 6), "mesh": None if Pm is None else round(float(Pm[:, 1].min()), 6), "H": round(H, 6)}
    if abs(fy) > 0.01 * H:
        bad("floor", f"the lowest joint is at y = {fy:.4f} ({fy / H:+.3f}H; needs within 0.01H of 0)")
    if Pm is not None and abs(float(Pm[:, 1].min())) > 0.01 * H:
        bad("floor", f"the mesh's lowest point is at y = {float(Pm[:, 1].min()):.4f} (needs within 0.01H of 0)")

    # -- clips
    anims = {a.get("name"): a for a in cdoc.get("animations", [])}
    facts["clips"] = list(anims)
    ref = anims.get(REF)
    facts["ref_present"] = ref is not None
    if ref is None:
        bad("ref", "no 'Rest' animation in clips.glb")
    else:
        dur = rig.clip_duration(cdoc, cbin, ref)
        keys = len(rig.accessor(cdoc, cbin, ref["samplers"][0]["input"]))
        if keys < 2 or abs(dur - 1.0 / FPS) > 1e-4:
            bad("ref", f"'Rest' has {keys} keys over {dur:.4f}s (needs 2 keys at 0 and 1/30)")
        ch = rig.prepare_clip(cdoc, cbin, ref, rig.Skeleton(cdoc))
        cskel = rig.Skeleton(cdoc)
        off = 0.0
        for i, p, _, vals in ch:
            r0 = cskel.rest[i][0 if p == "translation" else 1 if p == "rotation" else 2]
            off = max(off, float(np.abs(vals - r0).max()))
        if off > 1e-5:
            bad("ref", f"'Rest' is not the rest pose (off by {off:.2e})")
    loops = {c["name"]: bool(c.get("loop")) for c in plan.get("clips", []) if isinstance(c, dict) and "name" in c}
    seams = {}
    root_n = rskel.index.get("root")
    for nm, an in anims.items():
        if nm == REF:
            continue
        rot_nodes = {ch["target"]["node"] for ch in an["channels"] if ch["target"]["path"] == "rotation"}
        tr_nodes = {ch["target"]["node"] for ch in an["channels"] if ch["target"]["path"] == "translation"}
        if any(ch["target"]["path"] == "scale" for ch in an["channels"]):
            bad("clips", f"{nm} has scale channels")
        if tr_nodes - {root_n}:
            bad("clips", f"{nm} translates joints other than root", "warn")
        if set(cj) - rot_nodes:
            bad("clips", f"{nm} has no rotation channel on {len(set(cj) - rot_nodes)} joints", "warn")
        for s in an["samplers"]:
            if s.get("interpolation", "LINEAR") != "LINEAR":
                bad("clips", f"{nm} has a non-LINEAR sampler")
                break
        t = rig.accessor(cdoc, cbin, an["samplers"][0]["input"])[:, 0]
        if len(t) > 2:
            dt = np.diff(t)
            if np.abs(dt - 1.0 / FPS).max() > 1e-4:
                bad("clips", f"{nm} is not sampled at 30 fps", "warn")
        is_loop = loops.get(nm, (an.get("extras") or {}).get("loop", False))
        if is_loop:
            gap = 0.0
            for ch in an["channels"]:
                if ch["target"]["path"] != "rotation":
                    continue
                v = rig.accessor(cdoc, cbin, an["samplers"][ch["sampler"]]["output"])
                gap = max(gap, float(_quat_angle(v[0], v[-1])))
            seams[nm] = round(gap, 4)
            if gap > 0.5:
                bad("clips", f"{nm} loops but its last frame is {gap:.2f}° off frame 0 (needs ≤ 0.5°)")
    facts["loop_seams"] = seams
    for nm in plan.get("default", []):
        if nm not in anims:
            bad("clips", f"default clip {nm} is missing from clips.glb")
    return probs, facts


def verify(rig_glb: Path, clips_glb: Path, plan: dict | None, out: Path | None | bool = None) -> list[dict]:
    """Checks a template against the contract (design §1/§2.3) -> [{level: error|warn, key, msg}].
    Writes validate.json {ok, problems[], joints, ibm_err, axes, floor_y, ref_present, body:{verts, skinned}} to
    `out`; by default next to the rig GLB (rig/template/validate.json); out=False writes nothing."""
    probs, facts = _verify(Path(rig_glb), Path(clips_glb), plan)
    if out is not False:
        path = Path(out) if out else Path(rig_glb).parent / "validate.json"
        rep = {"ok": not any(p["level"] == "error" for p in probs), "problems": probs,
               "joints": facts.get("joints"), "ibm_err": facts.get("ibm_err"), "axes": facts.get("axes"),
               "floor_y": facts.get("floor_y"), "ref_present": facts.get("ref_present", False),
               "body": facts.get("body", {"verts": 0, "skinned": False}), "clips": facts.get("clips", []),
               "loop_seams": facts.get("loop_seams", {}), "plan": (plan or {}).get("id")}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(rep), indent=1))
    return probs


# ---------------------------------------------------------------- template.json and loading

def _md5(p: Path) -> str | None:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None


def write_template_json(dir: Path, plan: dict, clips_ok: list[str]) -> dict:
    """rig/template/template.json: {rig, clips:[...], ref:'Rest', default, class, plan, built:true, dir, built_for,
    frames}. `default` is the plan's default clips that are in clips_ok, in the plan's order (then any other ok
    clips). Paths are absolute so `M2M / path` keeps them."""
    d = Path(dir).resolve()
    asset = d.parent.parent
    want = list(plan.get("default") or [])
    ok = [c for c in clips_ok if c != REF]
    default = [c for c in want if c in ok] + [c for c in ok if c not in want]
    tj = {"rig": str(d / "rig.glb"), "clips": [str(d / "clips.glb")], "ref": REF, "default": default,
          "class": plan.get("class"), "plan": plan.get("id"), "built": True, "dir": str(d),
          "built_for": _md5(asset / "psx.glb"), "frames": "frames.json"}
    d.mkdir(parents=True, exist_ok=True)
    (d / "template.json").write_text(json.dumps(tj, indent=1))
    return tj


def _asset(asset) -> Path:
    p = Path(asset)
    if not p.is_absolute() and len(p.parts) == 1 and not p.exists():
        return rig.OUT / p
    return p


def key_for(asset) -> str:
    return f"built:{_asset(asset).name}"


def load_template(asset) -> dict | None:
    """Reads <asset>/rig/template/template.json and inserts it into rig.TEMPLATES['built:<stem>']; returns that same
    object (so identity lookups work). A second call returns the existing entry (refreshed in place if the file
    changed). None when there is no template.json."""
    a = _asset(asset)
    f = a / "rig" / "template" / "template.json"
    if not f.is_file():
        return None
    tj = json.loads(f.read_text())
    tj.setdefault("dir", str(f.parent.resolve()))
    key = key_for(a)
    cur = rig.TEMPLATES.get(key)
    if cur is not None:
        if cur != tj:
            cur.clear()
            cur.update(tj)
        return cur
    rig.TEMPLATES[key] = tj
    return tj


def template_for(info: dict, asset) -> dict:
    """The template a rig run used: the built one (load_template) when rig.json says template.built, else
    rig.TEMPLATES[info['class']]. A built run without its template.json raises FileNotFoundError rather than
    silently comparing against the class's vendor template."""
    if (info.get("template") or {}).get("built"):
        t = load_template(asset)
        if t is None:
            raise FileNotFoundError(f"{_asset(asset) / 'rig' / 'template' / 'template.json'} is missing; the rig was built for this model, re-run the rig")
        return t
    return rig.TEMPLATES[info["class"]]


# ---------------------------------------------------------------- CLI

def _plan_for(asset: Path, tdir: Path) -> dict | None:
    for f in (asset / "rig" / "plan.json",):
        if f.is_file():
            return json.loads(f.read_text())
    tj = tdir / "template.json"
    if tj.is_file():
        pid = json.loads(tj.read_text()).get("plan")
        if pid:
            import rig_plan
            return rig_plan.load(pid)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="check rig/template/{rig,clips}.glb; writes rig/template/validate.json")
    v.add_argument("--stem")
    v.add_argument("--dir", help="a template directory (default out/<stem>/rig/template)")
    v.add_argument("--plan", help="plan id (default rig/plan.json, else template.json's plan)")
    s = sub.add_parser("show", help="print template.json and load it")
    s.add_argument("--stem", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        if not (args.stem or args.dir):
            ap.error("verify needs --stem or --dir")
        tdir = Path(args.dir) if args.dir else rig.OUT / args.stem / "rig" / "template"
        asset = tdir.parent.parent
        if args.plan:
            import rig_plan
            plan = rig_plan.load(args.plan)
        else:
            plan = _plan_for(asset, tdir)
        probs = verify(tdir / "rig.glb", tdir / "clips.glb", plan)
        rep = json.loads((tdir / "validate.json").read_text())
        for p in probs:
            print(f"{p['level']:5s} {p['key']}: {p['msg']}")
        print(f"rig_template: {'ok' if rep['ok'] else 'FAILED'} ({rep['joints']} joints, IBM error {rep['ibm_err']:.1e}, "
              f"{rep['body']['verts']} verts, clips {', '.join(rep['clips'])}) -> {tdir / 'validate.json'}")
        return 0 if rep["ok"] else 1
    t = load_template(args.stem)
    if t is None:
        print(f"rig_template: no template.json for {args.stem}", file=sys.stderr)
        return 1
    print(json.dumps({"key": key_for(args.stem), **t}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
