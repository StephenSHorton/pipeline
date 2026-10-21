#!/usr/bin/env python3
"""A synthetic chicken-like test subject, shared by the rig-tool tests (rig_build, rig_clips, rig_template, ...).

Import it with:
    sys.path.insert(0, str(ROOT / "tests"))
    from fixtures import synth_bird          # a namespace package: there is no fixtures/__init__.py

What it gives you
- make_mesh(reverse=False) -> (V, F): closed capped primitives (body ellipsoid, neck cylinder, head sphere, beak cone,
  tail cone, two legs of two cylinders each, two foot boxes with the toes toward +Z). Feet on y=0, head at +Z, +X is
  the bird's LEFT, the head is the highest point (so a "top of the mesh" facing guess points at it; there is no
  texture, so no red cue). H = 1.12. `reverse` turns everything 180 degrees about +Y (head at -Z).
- pred_joints(legs=2, reverse=False) -> (names, parent, jpos): a 15-joint skin-tokens-style prediction with the
  chicken topology seen on a-ps1-style-graphics-of-a-chicken-2 (bone_0 branch; head chain bone_1..5; tail
  bone_6..8; legs bone_9..11 and bone_12..14). legs=3 adds a third leg chain bone_15..17 (a wrong leg count).
- write(dir, legs=2, reverse=False) -> dict: writes psx.glb (POSITION, NORMAL, TEXCOORD_0, indices; the layout
  psx_lowpoly writes) and pred_rig.glb (mesh + JOINTS_0/WEIGHTS_0 + skin + bone_N nodes, nearest-joint weights)
  and returns {psx, pred, V, F, names, parent, jpos, H}. Load the prediction with rig_pred.Pred(d["pred"]).
- distance_weights(P, jpos, parent, k=4) -> (J, W): smooth inverse-distance-to-bone weights, top-k, for tests that
  need a plausible skin without skin-tokens.
- build(dir, legs=2) -> (BuildInfo, fixture dict): runs rig_build.build on the fixture (writes dir/template/).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

H = 1.12
LEG_X = 0.10


def _cyl(a, b, r, sections=12):
    import trimesh
    return trimesh.creation.cylinder(radius=r, segment=[a, b], sections=sections)


def _cone(base, apex, r, sections=12):
    import trimesh
    base, apex = np.asarray(base, float), np.asarray(apex, float)
    d = apex - base
    T = trimesh.geometry.align_vectors([0, 0, 1.0], d / np.linalg.norm(d))
    T[:3, 3] = base
    return trimesh.creation.cone(radius=r, height=float(np.linalg.norm(d)), sections=sections, transform=T)


def _sphere(c, r, sub=2, scale=(1.0, 1.0, 1.0)):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    m.vertices = m.vertices * (np.asarray(scale, float) * r) + np.asarray(c, float)
    return m


def _box(c, ext):
    import trimesh
    T = np.eye(4)
    T[:3, 3] = c
    return trimesh.creation.box(extents=ext, transform=T)


def _turn(X: np.ndarray, reverse: bool) -> np.ndarray:
    X = np.array(X, float)
    if reverse:
        X[..., 0] *= -1
        X[..., 2] *= -1
    return X


def make_mesh(reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    parts = [
        _sphere([0, 0.55, 0], 1.0, 3, (0.20, 0.25, 0.30)),          # body
        _cyl([0, 0.60, 0.20], [0, 0.97, 0.38], 0.07),                 # neck
        _sphere([0, 1.02, 0.42], 0.10),                               # head (top at 1.12)
        _cone([0, 1.00, 0.48], [0, 0.99, 0.57], 0.04),                # beak toward +Z
        _cone([0, 0.62, -0.20], [0, 0.85, -0.50], 0.10),              # tail, back and up
    ]
    for s in (1, -1):
        x = s * LEG_X
        parts += [
            _cyl([x, 0.38, 0.02], [x, 0.20, -0.05], 0.04),            # thigh + shank (hidden top inside the body)
            _sphere([x, 0.20, -0.05], 0.04, 1),                       # intertarsal joint
            _cyl([x, 0.20, -0.05], [x, 0.02, 0.0], 0.035),            # tarsus
            _box([x, 0.015, 0.07], [0.08, 0.03, 0.20]),               # foot, toes toward +Z
        ]
    m = trimesh.util.concatenate(parts)
    V = np.asarray(m.vertices, float)
    V[:, 1] -= V[:, 1].min()
    return _turn(V, reverse), np.asarray(m.faces, np.int64)


def pred_joints(legs: int = 2, reverse: bool = False) -> tuple[list[str], list[int], np.ndarray]:
    J = [
        ([0, 0.50, 0.00], -1),                                                     # 0 branch (hips)
        ([0, 0.62, 0.22], 0), ([0, 0.74, 0.27], 1), ([0, 0.86, 0.32], 2), ([0, 0.96, 0.37], 3), ([0, 1.03, 0.44], 4),
        ([0, 0.64, -0.24], 0), ([0, 0.72, -0.34], 6), ([0, 0.80, -0.44], 7),       # tail
        ([LEG_X, 0.37, 0.02], 0), ([LEG_X, 0.20, -0.05], 9), ([LEG_X, 0.04, 0.0], 10),      # leg (+X = L)
        ([-LEG_X, 0.37, 0.02], 0), ([-LEG_X, 0.20, -0.05], 12), ([-LEG_X, 0.04, 0.0], 13),  # leg (R)
    ]
    if legs == 3:
        J += [([0.06, 0.42, -0.12], 0), ([0.06, 0.22, -0.16], 15), ([0.06, 0.04, -0.13], 16)]
    names = [f"bone_{k}" for k in range(len(J))]
    parent = [p for _, p in J]
    jpos = _turn(np.array([p for p, _ in J], float), reverse)
    return names, parent, jpos


def distance_weights(P: np.ndarray, jpos: np.ndarray, parent, k: int = 4, falloff: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Weights ~ 1/d^falloff to each bone (joint -> its children; a leaf is a point), top-k, rows sum to 1."""
    import rig_metrics
    segs = [(int(p), c) for c, p in enumerate(parent) if p is not None and int(p) >= 0]
    D = rig_metrics.seg_dist(P, np.asarray(jpos, float), segs)
    Wd = 1.0 / np.maximum(D, 1e-4) ** falloff
    J = np.argsort(-Wd, axis=1)[:, :k]
    W = np.take_along_axis(Wd, J, 1)
    W /= W.sum(1, keepdims=True)
    return J.astype(np.int64), W.astype(np.float32)


def _normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    import trimesh
    return np.asarray(trimesh.Trimesh(V, F, process=False).vertex_normals, float)


def write_psx(path: Path, V: np.ndarray, F: np.ndarray) -> Path:
    import rig
    g = rig.GlbWriter()
    uv = np.c_[(V[:, 0] - V[:, 0].min()) / max(np.ptp(V[:, 0]), 1e-9), (V[:, 1] - V[:, 1].min()) / max(np.ptp(V[:, 1]), 1e-9)]
    attrs = {"POSITION": g.array(V.astype("<f4"), "VEC3", 34962, True),
             "NORMAL": g.array(_normals(V, F).astype("<f4"), "VEC3", 34962),
             "TEXCOORD_0": g.array(uv.astype("<f4"), "VEC2", 34962)}
    idx = g.array(F.reshape(-1).astype("<u4"), "SCALAR", 34963)
    g.doc.update(scene=0, scenes=[{"nodes": [0]}], nodes=[{"mesh": 0, "name": "synth_bird"}],
                 meshes=[{"primitives": [{"attributes": attrs, "indices": idx}]}])
    g.write(path)
    return path


def write_pred(path: Path, V: np.ndarray, F: np.ndarray, names, parent, jpos) -> Path:
    """A skin-tokens `rig`-style GLB: the mesh skinned to bone_N joints (nearest-joint weights), TRS translations."""
    import rig
    jpos = np.asarray(jpos, float)
    near = np.linalg.norm(V[:, None] - jpos[None], axis=2).argmin(1)
    Jt = np.zeros((len(V), 4), np.uint16)
    Jt[:, 0] = near
    Wt = np.zeros((len(V), 4), np.float32)
    Wt[:, 0] = 1.0
    g = rig.GlbWriter()
    attrs = {"POSITION": g.array(V.astype("<f4"), "VEC3", 34962, True),
             "JOINTS_0": g.array(Jt.astype("<u2"), "VEC4", 34962),
             "WEIGHTS_0": g.array(Wt, "VEC4", 34962)}
    idx = g.array(F.reshape(-1).astype("<u4"), "SCALAR", 34963)
    nb = len(names)
    nodes = []
    for k, n in enumerate(names):
        p = parent[k]
        t = jpos[k] - (jpos[p] if p >= 0 else 0.0)
        nd = {"name": n, "translation": [float(x) for x in t]}
        kids = [c for c in range(nb) if parent[c] == k]
        if kids:
            nd["children"] = kids
        nodes.append(nd)
    ibm = np.stack([np.linalg.inv(np.block([[np.eye(3), jpos[k][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
                    for k in range(nb)])
    ibm_acc = g.array(np.ascontiguousarray(ibm.transpose(0, 2, 1)).reshape(nb, 16).astype("<f4"), "MAT4")
    roots = [k for k in range(nb) if parent[k] < 0]
    nodes.append({"name": "pred_mesh", "mesh": 0, "skin": 0})
    g.doc.update(scene=0, scenes=[{"nodes": roots + [nb]}], nodes=nodes,
                 meshes=[{"primitives": [{"attributes": attrs, "indices": idx}]}],
                 skins=[{"joints": list(range(nb)), "inverseBindMatrices": ibm_acc}])
    g.write(path)
    return path


def write(out: Path, legs: int = 2, reverse: bool = False) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    V, F = make_mesh(reverse)
    names, parent, jpos = pred_joints(legs, reverse)
    psx = write_psx(out / "psx.glb", V, F)
    pred = write_pred(out / "pred_rig.glb", V, F, names, parent, jpos)
    return {"psx": psx, "pred": pred, "V": V, "F": F, "names": names, "parent": parent, "jpos": jpos,
            "H": float(np.ptp(V[:, 1]))}


def build(out: Path, legs: int = 2, plan_id: str = "ground_bird"):
    """rig_build.build on the fixture -> (BuildInfo, fixture dict). Writes out/template/{skeleton,build}.json."""
    import rig_build
    import rig_plan
    import rig_pred
    d = write(Path(out), legs=legs)
    info = rig_build.build(d["psx"], rig_plan.load(plan_id), rig_pred.Pred(d["pred"]), Path(out) / "template")
    return info, d
