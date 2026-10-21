#!/usr/bin/env python3
"""Designed per-joint frames (bone rolls) and the reframe that applies them without moving a single vertex.

Inside the pipeline every rig has identity rest rotations; designed frames are applied once, last, before the final
write_rigged (design section 5).

Convention (Blender / rig-bird style):
- local +Y runs along the bone toward its primary child (the first child in the plan's joint list);
- local +X is the creature's lateral axis (+X = creature's left), Gram-Schmidt-projected off Y; when |Y . X| > 0.95
  the reference is forward (+Z) instead;
- Z = X x Y; leaf/tip joints copy their parent's frame; `root` keeps identity;
- R side frames are the mirror of their L partner: Q_R = S Q_L S, S = diag(-1, 1, 1). R joints are solved by the
  mirrored rule (same +X reference, -Z fallback), which gives exactly S Q_L S on a symmetric skeleton and still
  keeps +Y on the bone where the sides differ (an unsymmetrised fixed-path rig).

Reframe (Q_i = designed world rest rotation; old rests identity, but any old rest works):
    local'_i(t) = Q_parent^-1 local_i(t) Q_i,   t'_i = Q_parent^-1 t_i,   IBM'_i = inv(world'_rest_i)
so world'_i(t) = world_i(t) Q_i and every skinning matrix world' IBM' is unchanged: LBS is identical at every frame.

API:
    frames(skel_json, plan=None)                  -> {name: 3x3 Q_world (columns X, Y, Z)}
    reframe(doc, bin, clips, Q)                   -> (doc', bin', clips')
    frames_from_template(template_rig_glb, our_names, yaw_deg=0.0) -> {name: 3x3}
    check_identical(before, after, P, J, W)       -> max vertex delta (mesh units) over rest + every clip frame
    write_frames(path, Q, meta=None) / read_frames(path)
    clips_from_doc(doc, bin)                      -> bake_clips-shaped clip dicts from a GLB's animations

CLI:
    .venv/bin/python scripts/rig_frames.py check --stem S [--write-frames] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig  # noqa: E402  (read-only: read_glb, accessor, GlbWriter, TEMPLATES, M2M, OUT)

S = np.diag([-1.0, 1.0, 1.0])
LATERAL = np.array([1.0, 0.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])
CONVENTION = "+Y along the bone to its primary child, +X lateral (creature left), Z = X x Y; R = S L S"


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def _mat(q) -> np.ndarray:
    """A rotation as a 3x3 matrix from a Rotation, an xyzw quaternion or a 3x3."""
    if isinstance(q, R):
        return q.as_matrix()
    q = np.asarray(q, float)
    return q if q.shape == (3, 3) else R.from_quat(q).as_matrix()


# ---------------------------------------------------------------- frames

def plan_joint_order(plan: dict | None) -> list[str]:
    """The plan's joint names in list order, `{s}` expanded per side (L entries, then R, at each line)."""
    if not plan or not plan.get("joints"):
        return []
    sides = [s.lstrip("_") for s in plan.get("side_suffix", ["_L", "_R"])]
    out: list[str] = []
    for e in plan["joints"]:
        name = e["name"] if isinstance(e, dict) else e[0]
        if "{s}" in name:
            out.extend(name.replace("{s}", s) for s in sides)
        else:
            out.append(name)
    return out


def _partner(name: str, suffix: tuple[str, str]) -> str | None:
    """The L name of an R joint (`Foot_R` -> `Foot_L`), or None when it is not an R joint."""
    left, right = suffix
    return name[: -len(right)] + left if name.endswith(right) else None


def frames(skel_json: dict, plan: dict | None = None) -> dict[str, np.ndarray]:
    """Designed world rest rotation per joint (3x3, columns X, Y, Z) from skeleton.json
    `{joints: [{name, parent, pos}], ...}` and the plan (for the primary child; None = skeleton.json order)."""
    joints = skel_json["joints"]
    pos = {j["name"]: np.asarray(j["pos"], float) for j in joints}
    parent = {j["name"]: j.get("parent") for j in joints}
    order = plan_joint_order(plan)
    rank = {n: k for k, n in enumerate(order)}
    base = len(order)
    names = [j["name"] for j in joints]
    kids: dict[str, list[str]] = {n: [] for n in names}
    for k, n in enumerate(names):
        p = parent[n]
        if p in kids:
            kids[p].append(n)
    for p in kids:
        kids[p].sort(key=lambda c: rank.get(c, base + names.index(c)))
    suffix = tuple((plan or {}).get("side_suffix", ["_L", "_R"]))[:2]
    Q: dict[str, np.ndarray] = {}

    def designed(n: str, mirror_ref: bool) -> np.ndarray | None:
        if not kids[n]:
            return None
        y = _unit(pos[kids[n][0]] - pos[n])
        if not y.any():
            return None
        ref = LATERAL if abs(y @ LATERAL) <= 0.95 else (-FORWARD if mirror_ref else FORWARD)
        x = _unit(ref - (ref @ y) * y)
        return np.stack([x, y, np.cross(x, y)], 1)

    def solve(n: str, depth: int = 0) -> np.ndarray:
        if n in Q:
            return Q[n]
        if depth > len(names):
            raise ValueError("frames: the joint tree has a cycle")
        p = parent[n]
        if n == "root" or p not in pos:
            # the top joint: root keeps identity; another top joint still gets a designed frame when it has a child
            Q[n] = np.eye(3) if n == "root" else (designed(n, False) if designed(n, False) is not None else np.eye(3))
            return Q[n]
        # an R joint is solved by the mirrored rule (lateral ref +X, forward fallback -Z), which equals S Q_L S
        # exactly on a symmetric skeleton and keeps +Y on the bone when the two sides differ
        f = designed(n, _partner(n, suffix) is not None)
        Q[n] = f if f is not None else solve(p, depth + 1).copy()  # a leaf / zero-length bone copies its parent
        return Q[n]

    for n in names:
        solve(n)
    return {n: Q[n] for n in names}


def frames_from_template(template_rig_glb, our_names, yaw_deg: float = 0.0) -> dict[str, np.ndarray]:
    """The fixed template's own rest world rotations by joint name (for our joints that it has), turned by the
    fit yaw (degrees about +Y) when the template was turned onto the mesh. A relative path is under vendor M2M."""
    path = Path(template_rig_glb)
    if not path.is_absolute() and not path.is_file():
        path = rig.M2M / path
    doc, _ = rig.read_glb(path)
    skel = rig.Skeleton(doc)
    world = skel.world({})
    yaw = R.from_euler("y", float(yaw_deg), degrees=True)
    out = {}
    for n in our_names:
        i = skel.index.get(n)
        if i is not None:
            out[n] = (yaw * world[i][1]).as_matrix()
    return out


def write_frames(path: Path, Q: dict, meta: dict | None = None) -> None:
    """frames.json: {convention, joints: {name: {quat: xyzw, x, y, z}}, ...meta}."""
    js = {"convention": CONVENTION, **(meta or {}), "joints": {}}
    for n, q in Q.items():
        m = _mat(q)
        js["joints"][n] = {"quat": np.round(R.from_matrix(m).as_quat(), 7).tolist(),
                           "x": np.round(m[:, 0], 6).tolist(), "y": np.round(m[:, 1], 6).tolist(),
                           "z": np.round(m[:, 2], 6).tolist()}
    Path(path).write_text(json.dumps(js, indent=1))


def read_frames(path: Path) -> dict[str, np.ndarray]:
    js = json.loads(Path(path).read_text())
    return {n: R.from_quat(v["quat"]).as_matrix() for n, v in js["joints"].items()}


# ---------------------------------------------------------------- reframe

def _parents(nodes: list[dict]) -> dict[int, int]:
    return {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}


def _local(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.asarray(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    M[:3, :3] = R.from_quat(n.get("rotation", [0, 0, 0, 1])).as_matrix() * np.asarray(n.get("scale", [1, 1, 1]), float)
    M[:3, 3] = n.get("translation", [0, 0, 0])
    return M


def world_mats(nodes: list[dict], parent: dict[int, int], override: dict | None = None) -> list[np.ndarray]:
    """4x4 world matrix per node; override {node: (t or None, q or None)} replaces the local TRS parts."""
    loc = []
    for i, n in enumerate(nodes):
        M = _local(n)
        if override and i in override:
            t, q = override[i]
            if q is not None:
                M[:3, :3] = R.from_quat(q).as_matrix() * np.asarray(n.get("scale", [1, 1, 1]), float)
            if t is not None:
                M[:3, 3] = t
        loc.append(M)
    out: list = [None] * len(nodes)

    def w(i: int) -> np.ndarray:
        if out[i] is None:
            out[i] = w(parent[i]) @ loc[i] if i in parent else loc[i]
        return out[i]

    for i in range(len(nodes)):
        w(i)
    return out


def _append(doc: dict, binary: bytearray, arr: np.ndarray, kind: str) -> int:
    """A new tightly packed float32 accessor at the end of the buffer."""
    binary.extend(b"\0" * ((4 - len(binary) % 4) % 4))
    data = np.ascontiguousarray(arr, "<f4").tobytes()
    doc.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
    binary.extend(data)
    doc.setdefault("accessors", []).append({"bufferView": len(doc["bufferViews"]) - 1, "componentType": 5126,
                                            "count": int(arr.shape[0]), "type": kind})
    return len(doc["accessors"]) - 1


def reframe(doc: dict, binary: bytes, clips: list[dict], Q: dict) -> tuple[dict, bytes, list[dict]]:
    """Apply designed frames Q {joint name: world rest rotation (3x3 / xyzw / Rotation)} to a skinned rig.

    doc/binary: a GLB (skin-tokens' skeleton, or a rigged.glb) with TRS joint nodes; clips: bake_clips-shaped dicts
    {name, times, rot {node: (F,4) local xyzw}, root, root_t (F,3), ...}. Joints not in Q, and non-joint nodes
    (Armature, Body), keep identity. Returns copies: the nodes' rest rotation/translation rewritten, the skin's
    IBMs rebuilt (patched in place when tightly packed float32, else appended), any GLB animations rewritten onto
    new accessors, and the clips' rotations (and non-root translations) rewritten. LBS is unchanged."""
    doc2 = json.loads(json.dumps(doc))
    nodes = doc2["nodes"]
    skin = doc2["skins"][0]
    joints = list(skin["joints"])
    jset = set(joints)
    parent = _parents(nodes)
    Qm = {i: np.eye(3) for i in range(len(nodes))}
    for i in joints:
        n = nodes[i].get("name")
        if n in Q:
            Qm[i] = _mat(Q[n])
    for i in joints:
        if "matrix" in nodes[i]:
            raise ValueError(f"reframe: joint {nodes[i].get('name')!r} stores a matrix; TRS only")

    def qp(i: int) -> np.ndarray:
        p = parent.get(i)
        return Qm[p] if p in jset else np.eye(3)

    for i in joints:
        n = nodes[i]
        Pinv = qp(i).T
        r = R.from_quat(n.get("rotation", [0, 0, 0, 1])).as_matrix()
        n["rotation"] = [float(v) for v in R.from_matrix(Pinv @ r @ Qm[i]).as_quat()]
        n["translation"] = [float(v) for v in Pinv @ np.asarray(n.get("translation", [0, 0, 0]), float)]
    # IBMs from the new rest
    wm = world_mats(nodes, parent)
    ibm = np.array([np.linalg.inv(wm[i]).T.reshape(-1) for i in joints], "<f4")
    bin2 = bytearray(binary)
    acc = doc2["accessors"][skin["inverseBindMatrices"]] if "inverseBindMatrices" in skin else None
    view = doc2["bufferViews"][acc["bufferView"]] if acc and "bufferView" in acc else None
    if (acc and view and acc["componentType"] == 5126 and acc["type"] == "MAT4" and acc["count"] == len(joints)
            and view.get("byteStride", 64) == 64):
        o = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        bin2[o:o + ibm.nbytes] = ibm.tobytes()
    else:
        skin["inverseBindMatrices"] = _append(doc2, bin2, ibm, "MAT4")
    # animations stored in the GLB itself
    for anim in doc2.get("animations", []):
        for ch in anim["channels"]:
            i, path = ch["target"].get("node"), ch["target"]["path"]
            if i not in jset or path not in ("rotation", "translation"):
                continue
            smp = anim["samplers"][ch["sampler"]]
            vals = rig.accessor(doc2, bytes(bin2), smp["output"]).astype(float)
            if path == "rotation":
                if smp.get("interpolation") == "CUBICSPLINE":
                    raise ValueError("reframe: CUBICSPLINE rotations are not supported")
                new = R.from_matrix(qp(i).T @ R.from_quat(vals).as_matrix() @ Qm[i]).as_quat()
            else:
                new = vals @ qp(i)  # rows: (Qp^T v)^T = v^T Qp
            smp["output"] = _append(doc2, bin2, new, "VEC4" if path == "rotation" else "VEC3")
    if doc2.get("buffers"):
        doc2["buffers"][0]["byteLength"] = len(bin2)
    clips2 = []
    for c in clips:
        c2 = dict(c)
        # joints the clip leaves alone keep the (new) node rest, which reframe already rewrote
        c2["rot"] = {int(i): R.from_matrix(qp(int(i)).T @ R.from_quat(np.asarray(q, float)).as_matrix() @ Qm[int(i)])
                     .as_quat().astype(np.float32) for i, q in c["rot"].items()}
        if c.get("root") is not None and "root_t" in c:
            c2["root_t"] = (np.asarray(c["root_t"], float) @ qp(int(c["root"]))).astype(np.float32)
        clips2.append(c2)
    return doc2, bytes(bin2), clips2


# ---------------------------------------------------------------- checks

def _lbs(Pm: np.ndarray, J: np.ndarray, W: np.ndarray, M: np.ndarray) -> np.ndarray:
    Ph = np.c_[Pm, np.ones(len(Pm))]
    out = np.zeros((len(Pm), 3))
    for k in range(J.shape[1]):
        out += W[:, k:k + 1] * np.einsum("nij,nj->ni", M[J[:, k], :3], Ph)
    return out


def skin_mats(doc: dict, binary: bytes, override: dict | None = None) -> np.ndarray:
    """(J,4,4) skinning matrices world_j @ IBM_j (IBM read column-major, as rig.far_motion does)."""
    skin = doc["skins"][0]
    joints = skin["joints"]
    IBM = rig.accessor(doc, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    wm = world_mats(doc["nodes"], _parents(doc["nodes"]), override)
    return np.stack([wm[j] @ IBM[k] for k, j in enumerate(joints)])


def _clip_frames(c: dict):
    F = len(c["times"])
    for f in range(F):
        ov = {int(i): (None, np.asarray(q)[f]) for i, q in c["rot"].items()}
        if c.get("root") is not None and "root_t" in c:
            r = int(c["root"])
            ov[r] = (np.asarray(c["root_t"])[f], ov.get(r, (None, None))[1])
        yield ov


def check_identical(before, after, P, J, W) -> float:
    """Max vertex position delta (mesh units) between two rigs' LBS, at rest and at every frame of every clip.
    before/after: (doc, bin, clips) tuples (clips matched by order); P (N,3), J (N,4) skin-joint indices, W (N,4)."""
    (d0, b0, c0), (d1, b1, c1) = before, after
    P = np.asarray(P, float)
    J = np.asarray(J, int)
    W = np.asarray(W, float)
    worst = float(np.abs(_lbs(P, J, W, skin_mats(d0, b0)) - _lbs(P, J, W, skin_mats(d1, b1))).max())
    if len(c0) != len(c1):
        raise ValueError("check_identical: the two rigs have different clip counts")
    for ca, cb in zip(c0, c1):
        for oa, ob in zip(_clip_frames(ca), _clip_frames(cb)):
            Va = _lbs(P, J, W, skin_mats(d0, b0, oa))
            Vb = _lbs(P, J, W, skin_mats(d1, b1, ob))
            worst = max(worst, float(np.abs(Va - Vb).max()))
    return worst


def clips_from_doc(doc: dict, binary: bytes) -> list[dict]:
    """A GLB's animations as bake_clips-shaped dicts (rotation channels per joint, translation on the top joint),
    sampled on each animation's first rotation timeline. Animations with differing key times are resampled
    by rig.sample_clip."""
    skel = rig.Skeleton(doc)
    joints = doc["skins"][0]["joints"]
    jset = set(joints)
    top = next(i for i in joints if skel.parent.get(i) not in jset)
    out = []
    for anim in doc.get("animations", []):
        ch = rig.prepare_clip(doc, binary, anim, skel)
        if not ch:
            continue
        times = ch[0][2]
        rot = {i: [] for i in joints}
        root_t = []
        for t in times:
            loc = rig.sample_clip(ch, skel, float(t))
            for i in joints:
                rot[i].append(loc[i][1])
            root_t.append(loc[top][0])
        out.append({"name": anim.get("name"), "times": np.asarray(times, np.float32),
                    "rot": {i: np.array(v, np.float32) for i, v in rot.items()}, "root": top,
                    "root_t": np.array(root_t, np.float32)})
    return out


def skel_json_from_doc(doc: dict) -> dict:
    """skeleton.json-shaped joints from a GLB's rest pose (for rigs that have no rig/template/skeleton.json)."""
    skel = rig.Skeleton(doc)
    joints = doc["skins"][0]["joints"]
    jset = set(joints)
    w = skel.world({})
    return {"joints": [{"name": doc["nodes"][i].get("name"),
                        "parent": doc["nodes"][skel.parent[i]].get("name") if skel.parent.get(i) in jset else None,
                        "pos": w[i][0].tolist()} for i in joints]}


def _mesh_pjw(doc: dict, binary: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for n in doc["nodes"]:
        if "mesh" in n and "skin" in n:
            a = doc["meshes"][n["mesh"]]["primitives"][0]["attributes"]
            return (rig.accessor(doc, binary, a["POSITION"]).astype(float),
                    rig.accessor(doc, binary, a["JOINTS_0"]).astype(int),
                    rig.accessor(doc, binary, a["WEIGHTS_0"]).astype(float))
    raise ValueError("no skinned mesh in the GLB")


def _has_rest_rotations(doc: dict, tol: float = 1e-6) -> bool:
    """True when any skin joint stores a rest rotation other than identity."""
    for i in (doc.get("skins") or [{}])[0].get("joints", []):
        q = doc["nodes"][i].get("rotation")
        if q is not None and (abs(abs(float(q[3])) - 1.0) > tol or any(abs(float(x)) > tol for x in q[:3])):
            return True
    return False


def _built(info: dict) -> bool:
    t = info.get("template")
    return bool(isinstance(t, dict) and t.get("built"))


def check_stem(stem: str, write: bool = False) -> dict:
    """Reframe the asset's rig with its designed frames and measure rest + clip LBS before vs after."""
    d = rig.OUT / stem
    rdir = d / "rig"
    src = next((p for p in (d / "rigged.glb", rdir / "skinned_raw.glb") if p.is_file()), None)
    if src is None:
        raise SystemExit(f"{stem}: no rigged.glb or rig/skinned_raw.glb")
    doc, binary = rig.read_glb(src)
    P, J, W = _mesh_pjw(doc, binary)
    H = float(np.ptp(P[:, 1])) or 1.0
    names = [doc["nodes"][i].get("name") for i in doc["skins"][0]["joints"]]
    tdir = rdir / "template"
    info = json.loads((rdir / "rig.json").read_text()) if (rdir / "rig.json").is_file() else {}
    plan = None
    for p in (rdir / "plan.json", tdir / "plan.json"):
        if p.is_file():
            plan = json.loads(p.read_text())
            plan = plan.get("plan", plan) if isinstance(plan, dict) and "joints" not in plan else plan
            break
    if (tdir / "skeleton.json").is_file():
        Q, source = frames(json.loads((tdir / "skeleton.json").read_text()), plan), "skeleton.json"
    elif not _built(info) and (rdir / "template_posed.glb").is_file():
        # the fixed template as it was posed and fitted onto this mesh (yaw and proportions already applied)
        Q, source = frames_from_template(rdir / "template_posed.glb", names), "template_posed.glb"
    elif not _built(info) and (isinstance(info.get("template"), str) or info.get("class") in rig.TEMPLATES):
        fit = info.get("fit") if isinstance(info.get("fit"), dict) else {}
        trig = info["template"] if isinstance(info.get("template"), str) else rig.TEMPLATES[info["class"]]["rig"]
        Q = frames_from_template(trig, names, float(fit.get("yaw") or 0.0))
        source = f"template:{trig}"
    else:
        Q, source = frames(skel_json_from_doc(doc), plan), "rest geometry"
    clips = clips_from_doc(doc, binary)
    doc_nc = {**doc, "animations": []}
    # a rig that was already reframed (rig.py --reframe 1: rig.json reframe.applied, or rest rotations that are not
    # identity) is measured as it stands: reframing it again would stack the frames (116.6 deg on chicken-canon)
    already = bool((info.get("reframe") or {}).get("applied")) or _has_rest_rotations(doc)
    if already:
        d2, b2, c2 = doc_nc, binary, clips
        delta = 0.0
        source = "rigged.glb rests (already reframed)"
    else:
        d2, b2, c2 = reframe(doc_nc, binary, clips, Q)
        delta = check_identical((doc_nc, binary, clips), (d2, b2, c2), P, J, W)
    # bone-to-child alignment after reframe: angle between local +Y and the primary child
    skel2 = rig.Skeleton(d2)
    w2 = skel2.world({})
    jset = set(d2["skins"][0]["joints"])
    angles = []
    for i in d2["skins"][0]["joints"]:
        kids = [c for c in d2["nodes"][i].get("children", []) if c in jset]
        if kids:
            v = _unit(w2[kids[0]][0] - w2[i][0])
            if v.any():
                angles.append(float(np.degrees(np.arccos(np.clip(w2[i][1].apply([0, 1, 0]) @ v, -1, 1)))))
    out = {"stem": stem, "glb": str(src.relative_to(d)), "frames_from": source, "already_reframed": already,
           "joints": len(names),
           "framed": sum(1 for n in names if n in Q), "clips": [c["name"] for c in clips], "H": round(H, 6),
           "max_delta": delta, "max_delta_H": delta / H, "ok": delta / H < 1e-5,
           "median_bone_child_deg": round(float(np.median(angles)), 3) if angles else None}
    if write:
        fp = (tdir if tdir.is_dir() else rdir) / "frames.json"
        write_frames(fp, Q, {"from": source})
        out["frames_json"] = str(fp.relative_to(d))
    (rdir / "frames_check.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="rest + clip LBS before vs after reframe (writes rig/frames_check.json)")
    c.add_argument("--stem", required=True)
    c.add_argument("--write-frames", action="store_true", help="also write frames.json")
    c.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = check_stem(a.stem, a.write_frames)
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"{r['stem']}: {r['glb']} {r['joints']} joints ({r['framed']} framed from {r['frames_from']}), "
              f"{len(r['clips'])} clips, max LBS delta {r['max_delta_H']:.2e} H -> {'ok' if r['ok'] else 'FAIL'}; "
              f"median bone-to-child {r['median_bone_child_deg']} deg")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
