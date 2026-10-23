#!/usr/bin/env python3
"""Hand-edited skin weights into a rigged GLB: the Rig stage's Weights panel saves through POST /rig/<stem>/weights.

The panel edits the four bones per point (JOINTS_0 / WEIGHTS_0) on the model on screen and sends all of them back:

    {"primitives": [{"mesh": 0, "primitive": 0, "count": 3582,
                     "joints": "<base64 of count*4 little-endian uint16>",
                     "weights": "<base64 of count*4 little-endian float32>"}],
     "note": "less thigh on the hind legs"}                       # optional, a few words for rig.json

`joints` and `weights` may also be plain lists of count*4 numbers. Every point's weights must be finite, not
negative and add up to 1 (within 1e-3; they are renormalised exactly on write), and every joint index must be a
joint of that mesh's skin. A primitive that is not in the GLB, has no skin, or has another point count is refused.

The write is a splice: the JOINTS_0 and WEIGHTS_0 bytes are overwritten where they sit in the binary chunk, in the
accessor's own component type (u8/u16 joints; float or normalised u8/u16 weights), so every other byte of the file,
the JSON chunk included, stays as it was. rigged.glb is written (rig.json `rigged`), and rig/rigged_allclips.glb and
rig/rigged_nofloor.glb too when they hold the same mesh, since the viewer can show either. rig/skinned_raw.glb (what
the weight model made) is never touched. Then the FBX is exported again (rig_fbx through Blender; without Blender
the GLB stands and the note says so) and rig.json gets an `edited_weights` note.

    .venv/bin/python scripts/rig_weights.py --stem mlg-sheep            # what the weights are now (a summary)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
COMP = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
SIBLINGS = ("rig/rigged_allclips.glb", "rig/rigged_nofloor.glb")
SUM_TOL = 1e-3
# The keys of rig_fbx's check that rig.json keeps (rig.slim_fbx_check).
FBX_KEEP = ("ok", "problems", "warnings", "scene_fps", "max_influences", "up_axis", "unit_scale", "clips")

Exporter = Callable[[Path, Path, "list[str] | None"], "tuple[str, dict | None]"]


class WeightsError(ValueError):
    """A payload or a GLB the weights can't go into (the route answers 400)."""


# ------------------------------------------------------------------ reading a GLB


class Glb:
    """A GLB's JSON and where its binary chunk sits in the file bytes (so a splice leaves the rest as it was)."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 20 or data[:4] != b"glTF":
            raise WeightsError("not a GLB")
        self.data = data
        jlen, jtype = struct.unpack("<II", data[12:20])
        if jtype != JSON_CHUNK:
            raise WeightsError("the GLB's first chunk is not JSON")
        self.doc: dict[str, Any] = json.loads(data[20:20 + jlen])
        at = 20 + jlen
        self.bin_start = -1
        self.bin_len = 0
        if at + 8 <= len(data):
            blen, btype = struct.unpack("<II", data[at:at + 8])
            if btype == BIN_CHUNK:
                self.bin_start, self.bin_len = at + 8, blen

    @classmethod
    def read(cls, path: Path) -> "Glb":
        return cls(Path(path).read_bytes())

    def _layout(self, acc_i: int) -> tuple[dict, np.dtype, int, int, int]:
        """(accessor, component dtype, file offset of the first element, element byte size, byte stride)."""
        acc = self.doc["accessors"][acc_i]
        if "bufferView" not in acc or acc.get("sparse"):
            raise WeightsError(f"accessor {acc_i} has no plain buffer view")
        view = self.doc["bufferViews"][acc["bufferView"]]
        if view.get("buffer", 0) != 0 or self.bin_start < 0:
            raise WeightsError(f"accessor {acc_i} is not in the GLB's binary chunk")
        dt = np.dtype(COMP[acc["componentType"]]).newbyteorder("<")
        size = dt.itemsize * 4
        stride = int(view.get("byteStride") or size)
        off = self.bin_start + int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        if off + stride * (acc["count"] - 1) + size > self.bin_start + self.bin_len:
            raise WeightsError(f"accessor {acc_i} runs past the binary chunk")
        return acc, dt, off, size, stride

    def read_vec4(self, acc_i: int) -> np.ndarray:
        acc, dt, off, size, stride = self._layout(acc_i)
        n = acc["count"]
        raw = np.frombuffer(self.data, np.uint8, count=stride * (n - 1) + size, offset=off) if n else np.zeros(0, np.uint8)
        if stride != size and n:
            raw = np.lib.stride_tricks.as_strided(raw, shape=(n, size), strides=(stride, 1)).copy()
        out = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dt).reshape(n, 4).astype(np.float64 if dt.kind == "f" else np.int64)
        if acc.get("normalized") and dt.kind == "u":
            out = out / float(np.iinfo(dt).max)
        return out

    def skinned(self) -> list[dict[str, Any]]:
        """Every skinned primitive: {mesh, primitive, count, joints acc, weights acc, joint count of its skin}."""
        doc = self.doc
        skin_of: dict[int, int] = {}
        for node in doc.get("nodes", []):
            if "mesh" in node and "skin" in node:
                skin_of.setdefault(int(node["mesh"]), int(node["skin"]))
        out = []
        for mi, mesh in enumerate(doc.get("meshes", [])):
            for pi, prim in enumerate(mesh.get("primitives", [])):
                a = prim.get("attributes", {})
                if "JOINTS_0" not in a or "WEIGHTS_0" not in a:
                    continue
                skin = skin_of.get(mi)
                joints = len(doc["skins"][skin]["joints"]) if skin is not None and doc.get("skins") else 0
                out.append({"mesh": mi, "primitive": pi, "count": int(doc["accessors"][a["JOINTS_0"]]["count"]),
                            "joints_acc": int(a["JOINTS_0"]), "weights_acc": int(a["WEIGHTS_0"]), "skin_joints": joints,
                            "name": mesh.get("name", "")})
        return out

    def weights(self) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """{(mesh, primitive): (joints int Nx4, weights float Nx4)} for every skinned primitive."""
        return {(p["mesh"], p["primitive"]): (self.read_vec4(p["joints_acc"]), self.read_vec4(p["weights_acc"]))
                for p in self.skinned()}


# ------------------------------------------------------------------ the payload


def _array(v: Any, n: int, dtype: Any, what: str) -> np.ndarray:
    if isinstance(v, str):
        try:
            raw = base64.b64decode(v, validate=True)
        except (ValueError, TypeError) as e:
            raise WeightsError(f"{what} is not base64: {e}") from None
        dt = np.dtype(dtype).newbyteorder("<")
        if len(raw) != n * 4 * dt.itemsize:
            raise WeightsError(f"{what} has {len(raw)} bytes, {n * 4 * dt.itemsize} expected ({n} points x 4)")
        return np.frombuffer(raw, dt).astype(np.float64 if dt.kind == "f" else np.int64).reshape(n, 4)
    if isinstance(v, list):
        if len(v) != n * 4:
            raise WeightsError(f"{what} has {len(v)} numbers, {n * 4} expected ({n} points x 4)")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            raise WeightsError(f"{what} must hold numbers only")
        arr = np.asarray(v, dtype=np.float64).reshape(n, 4)
        if np.dtype(dtype).kind != "f":
            if not np.all(np.isfinite(arr)) or np.any(arr != np.round(arr)):
                raise WeightsError(f"{what} must hold whole numbers")
            arr = arr.astype(np.int64)
        return arr
    raise WeightsError(f"{what} must be base64 or a list")


def parse_payload(raw: Any, glb: Glb) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """The checked weights, {(mesh, primitive): (joints Nx4, weights Nx4 renormalised)}; WeightsError when bad."""
    if not isinstance(raw, dict):
        raise WeightsError("the body must be a JSON object")
    prims = raw.get("primitives")
    if not isinstance(prims, list) or not prims:
        raise WeightsError("primitives must be a non-empty list")
    have = {(p["mesh"], p["primitive"]): p for p in glb.skinned()}
    if not have:
        raise WeightsError("this rig has no skinned mesh")
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k, p in enumerate(prims):
        if not isinstance(p, dict):
            raise WeightsError(f"primitives[{k}] must be an object")
        try:
            key = (int(p.get("mesh", 0)), int(p.get("primitive", 0)))
        except (TypeError, ValueError):
            raise WeightsError(f"primitives[{k}]: mesh and primitive must be numbers") from None
        if key in out:
            raise WeightsError(f"mesh {key[0]} primitive {key[1]} is sent twice")
        info = have.get(key)
        if info is None:
            raise WeightsError(f"mesh {key[0]} primitive {key[1]} is not a skinned part of this rig")
        n = info["count"]
        if p.get("count") is not None and p.get("count") != n:
            raise WeightsError(f"mesh {key[0]} primitive {key[1]} has {n} points, the edit has {p.get('count')}")
        joints = _array(p.get("joints"), n, np.uint16, f"primitives[{k}].joints")
        weights = _array(p.get("weights"), n, np.float32, f"primitives[{k}].weights")
        if not np.all(np.isfinite(weights)):
            raise WeightsError(f"primitives[{k}].weights has a value that is not a number")
        if np.any(weights < -1e-6):
            raise WeightsError(f"primitives[{k}].weights has a negative weight")
        weights = np.clip(weights, 0.0, None)
        sums = weights.sum(axis=1)
        bad = np.flatnonzero(np.abs(sums - 1.0) > SUM_TOL)
        if bad.size:
            raise WeightsError(f"primitives[{k}]: point {int(bad[0])}'s weights add up to {sums[bad[0]]:.4f}, not 1 "
                               f"({bad.size} points)")
        if np.any(joints < 0) or np.any(joints >= max(1, info["skin_joints"])):
            raise WeightsError(f"primitives[{k}].joints has an index outside the skin's {info['skin_joints']} joints")
        # Rows already at 1 to float32 precision go in as sent, so saving unchanged weights rewrites the same bytes.
        off = np.abs(sums - 1.0) > 2e-6
        weights[off] = weights[off] / sums[off, None]
        out[key] = (joints, weights)
    return out


# ------------------------------------------------------------------ writing


def _encode_weights(w: np.ndarray, dt: np.dtype, normalized: bool) -> np.ndarray:
    if dt.kind == "f":
        return w.astype(dt)
    if not normalized:
        raise WeightsError("WEIGHTS_0 is an integer accessor that is not normalised")
    top = float(np.iinfo(dt).max)
    q = np.round(w * top).astype(np.int64)
    # The quantised row adds up to exactly `top`: the rounding error goes on each row's biggest weight.
    big = np.argmax(w, axis=1)
    q[np.arange(len(q)), big] += int(top) - q.sum(axis=1)
    return np.clip(q, 0, top).astype(dt)


def splice(glb: Glb, edits: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]) -> bytes:
    """The GLB's bytes with each edited primitive's JOINTS_0 / WEIGHTS_0 overwritten in place; nothing else moves."""
    buf = bytearray(glb.data)
    have = {(p["mesh"], p["primitive"]): p for p in glb.skinned()}
    for key, (joints, weights) in edits.items():
        info = have.get(key)
        if info is None or info["count"] != len(joints):
            raise WeightsError(f"mesh {key[0]} primitive {key[1]} does not match this GLB")
        for acc_i, values, is_w in ((info["joints_acc"], joints, False), (info["weights_acc"], weights, True)):
            acc, dt, off, size, stride = glb._layout(acc_i)
            if is_w:
                enc = _encode_weights(values, dt, bool(acc.get("normalized")))
            else:
                if dt.kind == "f":
                    raise WeightsError("JOINTS_0 is a float accessor")
                if values.max(initial=0) > np.iinfo(dt).max:
                    raise WeightsError(f"a joint index does not fit JOINTS_0's {dt.itemsize * 8}-bit numbers")
                enc = values.astype(dt)
            rows = np.ascontiguousarray(enc).reshape(len(values), 4)
            if stride == size:
                buf[off:off + size * len(rows)] = rows.tobytes()
            else:
                for r in range(len(rows)):
                    buf[off + r * stride:off + r * stride + size] = rows[r].tobytes()
    return bytes(buf)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.weights.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def same_mesh(a: Glb, b: Glb) -> bool:
    """Both GLBs hold the same skinned primitives with the same point counts (the rig's sibling files)."""
    sa = [(p["mesh"], p["primitive"], p["count"], p["skin_joints"]) for p in a.skinned()]
    sb = [(p["mesh"], p["primitive"], p["count"], p["skin_joints"]) for p in b.skinned()]
    return bool(sa) and sa == sb


def changed_points(old: dict, new: dict, tol: float = 1e-4) -> int:
    """How many points' bones or weights differ (a zero weight's joint index doesn't count)."""
    n = 0
    for key, (j1, w1) in new.items():
        j0, w0 = old[key]
        for v in range(len(j1)):
            a = {int(j): float(w) for j, w in zip(j0[v], w0[v]) if w > tol}
            b = {int(j): float(w) for j, w in zip(j1[v], w1[v]) if w > tol}
            if a.keys() != b.keys() or any(abs(a[j] - b[j]) > tol for j in a):
                n += 1
    return n


def default_exporter(glb: Path, fbx: Path, clips: list[str] | None) -> tuple[str, dict | None]:
    import rig_fbx

    return rig_fbx.export_and_check(glb, fbx, clips)


def _fbx_path(folder: Path, info: dict) -> Path | None:
    if info.get("fbx"):
        return folder / str(info["fbx"])
    found = sorted((folder / "rig").glob("*.fbx"))
    return found[0] if len(found) == 1 else None


def save_weights(folder: Path, raw: Any, exporter: Exporter | None = default_exporter) -> dict[str, Any]:
    """Check the edit, splice it into the rig's GLBs, export the FBX again and note it in rig.json.

    Returns {files, changed, fbx, fbx_note, blender}. WeightsError for a bad edit (nothing is written then)."""
    folder = Path(folder)
    info_path = folder / "rig" / "rig.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    except (OSError, ValueError):
        raise WeightsError("rig/rig.json is unreadable; Re-rig instead") from None
    rigged_rel = str(info.get("rigged") or "rigged.glb")
    rigged = folder / rigged_rel
    if not rigged.is_file():
        raise WeightsError(f"this asset has no {rigged_rel}")
    main = Glb.read(rigged)
    edits = parse_payload(raw, main)
    note = " ".join(str(raw.get("note") or "").split())[:200]
    old = main.weights()
    changed = changed_points(old, edits)

    # Every file is spliced in memory before any is written, so a bad sibling never leaves the rig half saved.
    out: list[tuple[str, bytes]] = [(rigged_rel, splice(main, edits))]
    skipped: list[str] = []
    for rel in SIBLINGS:
        p = folder / rel
        if rel == rigged_rel or not p.is_file():
            continue
        try:
            sib = Glb.read(p)
            if not same_mesh(main, sib):
                skipped.append(rel)
                continue
            out.append((rel, splice(sib, edits)))
        except (WeightsError, OSError, ValueError, KeyError):
            skipped.append(rel)
    for rel, data in out:
        _write_atomic(folder / rel, data)
    files = [rel for rel, _ in out]

    fbx_note, check, fbx_rel = "", None, None
    fbx = _fbx_path(folder, info)
    clips = info.get("clips") if isinstance(info.get("clips"), list) else None
    if clips and isinstance(clips[0], dict):
        clips = [c.get("name") for c in clips]
    if exporter is not None and fbx is not None:
        try:
            fbx_note, check = exporter(rigged, fbx, clips)
        except Exception as e:  # noqa: BLE001 - the GLB is saved; the FBX failing is a note, not an error
            fbx_note, check = f"FBX not exported: {e}", None
        if fbx.is_file() and check is not None:
            fbx_rel = fbx.relative_to(folder).as_posix()
            files.append(fbx_rel)
    elif fbx is None:
        fbx_note = "No FBX for this rig; the GLB is saved."

    prev = info.get("edited_weights") if isinstance(info.get("edited_weights"), dict) else {}
    info["edited_weights"] = {
        "made": int(time.time()), "saves": int(prev.get("saves") or 0) + 1, "changed_points": changed,
        "files": files, "note": note or prev.get("note", ""),
        "about": "skin weights painted by hand in the Weights panel; rig/skinned_raw.glb keeps the weight model's own",
    }
    if skipped:
        info["edited_weights"]["skipped"] = skipped
    if fbx_note:
        info["fbx_note"] = fbx_note
    if check is not None:
        info["fbx_check"] = {k: check.get(k) for k in FBX_KEEP if k in check}
    info_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = info_path.with_name(".rig.json.weights.tmp")
    tmp.write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, info_path)
    files.append("rig/rig.json")
    return {"files": files, "changed": changed, "fbx": fbx_rel, "fbx_note": fbx_note,
            "blender": check is not None, "skipped": skipped}


# ------------------------------------------------------------------ the app's payload, from Python (tests, scripts)


def payload_from(glb: Glb, weights: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None,
                 note: str = "") -> dict[str, Any]:
    """The body the panel sends, for these weights (default: the GLB's own)."""
    weights = weights if weights is not None else glb.weights()
    prims = []
    for (m, p), (j, w) in weights.items():
        prims.append({"mesh": m, "primitive": p, "count": len(j),
                      "joints": base64.b64encode(np.asarray(j, "<u2").tobytes()).decode(),
                      "weights": base64.b64encode(np.asarray(w, "<f4").tobytes()).decode()})
    return {"primitives": prims, "note": note}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stem", required=True)
    ap.add_argument("--root", default=os.environ.get("FACTORY_ROOT") or str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    folder = Path(a.root) / "out" / a.stem
    info = json.loads((folder / "rig" / "rig.json").read_text()) if (folder / "rig" / "rig.json").is_file() else {}
    glb = Glb.read(folder / str(info.get("rigged") or "rigged.glb"))
    for p in glb.skinned():
        j, w = glb.read_vec4(p["joints_acc"]), glb.read_vec4(p["weights_acc"])
        used = int((w > 1e-4).sum(axis=1).max(initial=0))
        print(f"mesh {p['mesh']} primitive {p['primitive']}: {p['count']} points, {p['skin_joints']} joints, "
              f"up to {used} bones a point, weights sum {w.sum(axis=1).min():.4f}..{w.sum(axis=1).max():.4f}")
    print("edited:", json.dumps(info.get("edited_weights")) if info.get("edited_weights") else "no")
    return 0


if __name__ == "__main__":
    sys.exit(main())
