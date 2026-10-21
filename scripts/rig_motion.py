#!/usr/bin/env python3
"""Play every clip of a rigged GLB on the CPU and measure it: where corners tear apart, how far edges stretch.

    .venv/bin/python scripts/rig_motion.py out/psx-cow/rigged.glb

Linear-blend skinning, as a game engine does it, at `samples` even times per clip. A flat-shaded mesh keeps a
separate copy of each corner per triangle; copies of one point (positions equal to 1e-5) that follow different
bones pull apart. A point tears when its copies are more than 0.5% of the model's height apart. Stretch is the
strain of each welded edge, |log(length / rest length)|, reported as a factor (1.0 = no stretch).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig import accessor, read_glb  # noqa: E402

TEAR_GAP = 0.005  # of the model's height


def _trs(t, q, s) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R.from_quat(q).as_matrix() * np.asarray(s, float)
    M[:3, 3] = t
    return M


def _rest(node: dict) -> tuple:
    return (np.asarray(node.get("translation", [0, 0, 0]), float), np.asarray(node.get("rotation", [0, 0, 0, 1]), float),
            np.asarray(node.get("scale", [1, 1, 1]), float))


def world_mats(nodes: list[dict], parent: dict[int, int], local: dict[int, tuple]) -> list[np.ndarray]:
    """World matrix of every node; `local` overrides a node's rest TRS."""
    out: list[np.ndarray | None] = [None] * len(nodes)

    def get(i: int) -> np.ndarray:
        if out[i] is None:
            n = nodes[i]
            M = _trs(*local[i]) if i in local else (np.asarray(n["matrix"], float).reshape(4, 4).T if "matrix" in n else _trs(*_rest(n)))
            out[i] = get(parent[i]) @ M if i in parent else M
        return out[i]
    return [get(i) for i in range(len(nodes))]


def channels(doc: dict, binary: bytes, anim: dict) -> tuple[float, list[tuple[int, int, np.ndarray, np.ndarray]]]:
    """Duration and (node, slot, times, values) per channel, read once. slot: 0 translation, 1 rotation, 2 scale."""
    slot = {"translation": 0, "rotation": 1, "scale": 2}
    out = []
    for ch in anim["channels"]:
        path = ch["target"].get("path")
        if path not in slot or "node" not in ch["target"]:
            continue
        sm = anim["samplers"][ch["sampler"]]
        out.append((ch["target"]["node"], slot[path], accessor(doc, binary, sm["input"])[:, 0].astype(float),
                    accessor(doc, binary, sm["output"]).astype(float)))
    return max((float(c[2][-1]) for c in out), default=0.0), out


def sample(nodes: list[dict], chans: list, t: float) -> dict[int, tuple]:
    """Local TRS of every animated node at time t (linear; rotations normalised)."""
    local: dict[int, list] = {}
    for node, k, T, V in chans:
        if len(T) == 1:
            v = V[0]
        else:
            j = int(np.clip(np.searchsorted(T, t), 1, len(T) - 1))
            a = float(np.clip((t - T[j - 1]) / max(T[j] - T[j - 1], 1e-9), 0, 1))
            v0, v1 = V[j - 1], V[j]
            if k == 1 and v0 @ v1 < 0:
                v1 = -v1
            v = v0 * (1 - a) + v1 * a
        if k == 1:
            v = v / np.linalg.norm(v)
        local.setdefault(node, list(_rest(nodes[node])))[k] = v
    return {i: tuple(v) for i, v in local.items()}


def skin(Ph: np.ndarray, J: np.ndarray, W: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Skinned positions: Ph homogeneous points, mats (bones, 4, 4) = world @ inverse bind."""
    X = np.zeros((len(Ph), 3))
    for k in range(J.shape[1]):
        X += W[:, k:k + 1] * np.einsum("nij,nj->ni", mats[J[:, k], :3], Ph)
    return X


def groups(P: np.ndarray) -> np.ndarray:
    """Which point each corner is a copy of."""
    return np.unique(np.round(P / 1e-5).astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)


def measure_arrays(P: np.ndarray, F: np.ndarray, J: np.ndarray, W: np.ndarray, top_bone: np.ndarray, names: list[str],
                   poses: list[tuple[float, np.ndarray]]) -> dict:
    """One clip from its sampled bone matrices: poses = [(t, mats)]."""
    inv = groups(P)
    ng = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=ng)[:, None]
    H = max(float(np.ptp(P[:, 1])), 1e-9)
    first = np.zeros(ng, int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    Fw = inv[F]
    E = np.unique(np.sort(np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]], 1), axis=0)
    l0 = np.linalg.norm(P[first[E[:, 0]]] - P[first[E[:, 1]]], axis=1)
    E, l0 = E[l0 > 1e-6 * H], l0[l0 > 1e-6 * H]
    Ph = np.c_[P, np.ones(len(P))]
    tears, tear_t, gap_max, strains = 0, None, 0.0, []
    best = (-1.0, 0.0, 0)  # strain, t, edge
    worst_c, worst_p99 = None, -1.0  # the frame whose 1% of edges stretch most: who owns those edges
    for t, mats in poses:
        X = skin(Ph, J, W, mats)
        c = np.zeros((ng, 3))
        np.add.at(c, inv, X)
        c /= cnt
        g = np.zeros(ng)
        np.maximum.at(g, inv, 2 * np.linalg.norm(X - c[inv], axis=1) / H)
        torn = int((g > TEAR_GAP).sum())
        if torn > tears:
            tears, tear_t = torn, t
        gap_max = max(gap_max, float(g.max()))
        if len(E):
            st = np.abs(np.log(np.maximum(np.linalg.norm(c[E[:, 0]] - c[E[:, 1]], axis=1), 1e-12) / l0))
            strains.append(st)
            p = float(np.percentile(st, 99))
            if p > worst_p99:
                worst_p99, worst_c = p, c
            k = int(st.argmax())
            if st[k] > best[0]:
                best = (float(st[k]), t, k)
    S = np.concatenate(strains) if strains else np.zeros(1)
    bone = names[int(top_bone[first[E[best[2], 0]]])] if len(E) and best[0] >= 0 else None
    owners = []
    if worst_c is not None and len(E):
        import rig_metrics
        P0 = np.zeros((ng, 3))
        np.add.at(P0, inv, P)
        P0 /= cnt
        own = rig_metrics.stretch_owners(P0, worst_c, E, np.asarray(top_bone)[first], names)
        owners = [[n, int(c_)] for n, c_ in own["owners"]]
    return {"stretch_owners": owners, "tears": tears, "tear_t": None if tear_t is None else round(float(tear_t), 2),
            "worst_gap_pct": round(100 * gap_max, 2), "stretch_max": round(float(np.exp(S.max())), 2),
            "stretch_p99": round(float(np.exp(np.percentile(S, 99))), 2), "worst_t": round(float(best[1]), 2),
            "worst_bone": bone}


def measure(glb: Path, top_bone: np.ndarray, names: list[str], samples: int = 16) -> dict:
    """Per glTF animation: duration, tears (most points torn at one sample) and when, worst corner gap (% of the
    height), stretch max and p99 (x), and when and on which bone (names[top_bone[v]]) the worst stretch is."""
    t0 = time.time()
    doc, binary = read_glb(Path(glb))
    pr = doc["meshes"][0]["primitives"][0]
    a = pr["attributes"]
    P = accessor(doc, binary, a["POSITION"]).astype(float)
    J = accessor(doc, binary, a["JOINTS_0"]).astype(int)
    W = accessor(doc, binary, a["WEIGHTS_0"]).astype(float)
    F = accessor(doc, binary, pr["indices"]).reshape(-1, 3).astype(int) if "indices" in pr else np.arange(len(P)).reshape(-1, 3)
    skn = doc["skins"][0]
    joints = skn["joints"]
    IBM = accessor(doc, binary, skn["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    nodes = doc["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    clips = {}
    for anim in doc.get("animations", []):
        dur, chans = channels(doc, binary, anim)
        poses = []
        for t in np.linspace(0, dur, samples):
            wm = world_mats(nodes, parent, sample(nodes, chans, float(t)))
            poses.append((float(t), np.array([wm[j] @ IBM[k] for k, j in enumerate(joints)])))
        clips[anim.get("name") or f"clip {len(clips) + 1}"] = {"duration": round(dur, 2), **measure_arrays(P, F, J, W, top_bone, names, poses)}
    return {"samples": samples, "tear_gap_rel_h": TEAR_GAP, "seconds": round(time.time() - t0, 2), "clips": clips}


def main() -> int:
    glb = Path(sys.argv[1])
    doc, binary = read_glb(glb)
    pr = doc["meshes"][0]["primitives"][0]["attributes"]
    J = accessor(doc, binary, pr["JOINTS_0"]).astype(int)
    W = accessor(doc, binary, pr["WEIGHTS_0"]).astype(float)
    names = [doc["nodes"][j].get("name", "") for j in doc["skins"][0]["joints"]]
    print(json.dumps(measure(glb, J[np.arange(len(J)), W.argmax(1)], names), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
