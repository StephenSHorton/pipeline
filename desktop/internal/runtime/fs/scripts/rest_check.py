#!/usr/bin/env python3
"""Rest-pose check for animal meshes (body along the ground): is the head straight in line with the body?

    .venv/bin/python scripts/rest_check.py out/psx-cow/psx.glb                    # measure
    .venv/bin/python scripts/rest_check.py out/psx-cow/psx.glb --straighten o.glb # write a straightened copy

Hunyuan3D turns a side-on animal's head 15-31 degrees toward the camera, even from a perfect profile picture, and
the template skeleton's neck runs straight, so every clip kept the turn. measure() finds the body's long axis (the
widest horizontal direction), the head end (the end with the highest point), the barrel's centreline, and the
neck-to-muzzle direction against it. straighten() swings the neck and head back about a pivot where the neck
starts, blending smoothly over 22% of the body length, and rewrites only POSITION and NORMAL in the GLB, so UVs,
flat-shaded corners and the texture stay byte-identical.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

UP = np.array([0.0, 1.0, 0.0])
TURNED = 4.0  # degrees of head yaw worth straightening


def _frame(P: np.ndarray, toward: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centre, forward (toward the head), lateral axis and each point's forward coordinate. The head end is the one
    with the highest point, unless `toward` (a known forward, e.g. glTF +Z for a faced mesh) lies within 60 degrees
    of the long axis: a grazing sheep or pig holds its head below its rump."""
    H = P - np.outer(P @ UP, UP)
    c = H.mean(0)
    _, v = np.linalg.eigh(np.cov((H - c).T))
    fwd = v[:, -1] - UP * (v[:, -1] @ UP)
    fwd /= np.linalg.norm(fwd)
    f = (P - c) @ fwd
    if toward is not None and abs(fwd @ toward) >= 0.5:
        flip = fwd @ toward < 0
    else:
        flip = (P @ UP)[f > np.quantile(f, 0.85)].max() < (P @ UP)[f < np.quantile(f, 0.15)].max()
    if flip:
        fwd, f = -fwd, -f
    return c, fwd, np.cross(UP, fwd), f


def _centres(P: np.ndarray, f: np.ndarray, n: int = 20) -> np.ndarray:
    """Per slice along the body, the centre of its upper 60% (legs left out)."""
    edges = np.linspace(f.min(), f.max(), n + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        S = P[(f >= a) & (f <= b)]
        if len(S) == 0:
            out.append(out[-1] if out else P.mean(0))
            continue
        out.append(S[(S @ UP) >= np.quantile(S @ UP, 0.4)].mean(0))
    return np.array(out)


def measure(P: np.ndarray, toward: np.ndarray | None = None) -> dict | None:
    """Head yaw/pitch against the body, and left/right mirror error (median, % of body length). None if the mesh
    is upright (taller than it is long), where this check doesn't apply. toward: see _frame."""
    P = np.asarray(P, float)
    if np.ptp(P @ UP) > 0.9 * np.ptp(P - np.outer(P @ UP, UP), axis=0).max():
        return None
    c, fwd, lat, f = _frame(P, toward)
    L = float(np.ptp(f))
    cs = _centres(P, f)
    body = cs[4:12]
    k = np.polyfit((body - c) @ fwd, body @ lat, 1)
    neck, tip = cs[13], cs[19]
    v = tip - neck
    side = v @ lat - (np.polyval(k, (tip - c) @ fwd) - np.polyval(k, (neck - c) @ fwd))
    yaw = float(np.degrees(np.arctan2(side, v @ fwd)))
    pitch = float(np.degrees(np.arctan2(v @ UP, np.hypot(v @ fwd, v @ lat))))
    off = (P @ lat) - np.polyval(k, (P - c) @ fwd)
    d = cKDTree(P).query(P - 2 * np.outer(off, lat))[0]
    head = f > np.quantile(f, 0.8)
    return {"head_yaw_deg": round(yaw, 1), "head_pitch_deg": round(pitch, 1),
            "body_yaw_deg": round(float(np.degrees(np.arctan(k[0]))), 1),
            "mirror_body_pct": round(float(np.median(d[~head]) / L * 100), 2),
            "mirror_head_pct": round(float(np.median(d[head]) / L * 100), 2),
            "turned": abs(yaw) > TURNED}


def straighten_points(P: np.ndarray, N: np.ndarray | None = None, iters: int = 3) -> tuple[np.ndarray, np.ndarray | None, list[float]]:
    """Swing the neck and head about the neck's start until the head yaw is under 1 degree."""
    P = np.asarray(P, float).copy()
    N = None if N is None else np.asarray(N, float).copy()
    yaws = []
    for _ in range(iters):
        m = measure(P)
        if m is None:
            break
        yaws.append(m["head_yaw_deg"])
        if abs(m["head_yaw_deg"]) < 1:
            break
        c, fwd, _, f = _frame(P)
        cs = _centres(P, f)
        pivot = cs[12]
        f0 = (pivot - c) @ fwd
        f1 = f0 + 0.22 * np.ptp(f)
        t = np.clip((f - f0) / (f1 - f0), 0, 1)
        w = t * t * (3 - 2 * t)
        sel = w > 0
        rot = R.from_rotvec(np.outer(w[sel], UP) * np.radians(-m["head_yaw_deg"]))
        P[sel] = pivot + rot.apply(P[sel] - pivot)
        if N is not None:
            N[sel] = rot.apply(N[sel])
    return P, N, yaws


def _glb(path: Path) -> tuple[bytearray, dict, int]:
    b = bytearray(path.read_bytes())
    n = struct.unpack("<I", b[12:16])[0]
    return b, json.loads(b[20:20 + n]), 20 + n + 8


def _vec3(b: bytearray, d: dict, base: int, ai: int) -> tuple[int, int, int]:
    a = d["accessors"][ai]
    v = d["bufferViews"][a["bufferView"]]
    if a["componentType"] != 5126 or a["type"] != "VEC3":
        raise ValueError("expected float VEC3")
    return base + v.get("byteOffset", 0) + a.get("byteOffset", 0), v.get("byteStride", 12), a["count"]


def _get(b: bytearray, d: dict, base: int, ai: int) -> np.ndarray:
    off, stride, n = _vec3(b, d, base, ai)
    raw = np.frombuffer(bytes(b[off:off + stride * (n - 1) + 12]), "u1")
    return np.lib.stride_tricks.as_strided(raw, (n, 12), (stride, 1)).copy().view("<f4").reshape(n, 3).astype(float)


def _put(b: bytearray, d: dict, base: int, ai: int, X: np.ndarray) -> None:
    off, stride, n = _vec3(b, d, base, ai)
    X32 = X.astype("<f4")
    for i in range(n):
        b[off + i * stride:off + i * stride + 12] = X32[i].tobytes()
    d["accessors"][ai]["min"] = X.min(0).tolist()
    d["accessors"][ai]["max"] = X.max(0).tolist()


def straighten(src: Path, dst: Path) -> dict:
    """Write a copy of src with the head straightened. Returns the before/after measurements."""
    b, d, base = _glb(src)
    attrs = d["meshes"][0]["primitives"][0]["attributes"]
    P = _get(b, d, base, attrs["POSITION"])
    N = _get(b, d, base, attrs["NORMAL"]) if "NORMAL" in attrs else None
    before = measure(P)
    P2, N2, _ = straighten_points(P, N)
    _put(b, d, base, attrs["POSITION"], P2)
    if N2 is not None:
        _put(b, d, base, attrs["NORMAL"], N2)
    js = json.dumps(d, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    body = bytes(b[base:])
    dst.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(body))
                    + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(body), 0x004E4942) + body)
    return {"before": before, "after": measure(P2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--straighten", type=Path, metavar="OUT.glb")
    args = ap.parse_args()
    if args.straighten:
        print(json.dumps(straighten(args.mesh, args.straighten), indent=1))
    else:
        b, d, base = _glb(args.mesh)
        print(json.dumps(measure(_get(b, d, base, d["meshes"][0]["primitives"][0]["attributes"]["POSITION"])), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
