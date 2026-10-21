#!/usr/bin/env python3
"""Drop generator shrapnel from a GLB.

ops:
  floaters  keep the largest connected pieces
  sphere    delete vertices (and their faces) inside a sphere in mesh space
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

from factory_env import pick_glb, repo_root


def _as_mesh(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Scene):
        parts = []
        for name, geom in obj.geometry.items():
            g = geom.copy()
            try:
                t, _ = obj.graph.get(name)
                g.apply_transform(t)
            except Exception:
                pass
            parts.append(g)
        obj = trimesh.util.concatenate(parts)
    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"not a triangle mesh: {type(obj)}")
    return obj


def remove_floaters(mesh: trimesh.Trimesh, keep: int = 1, min_faces: int = 0) -> tuple[trimesh.Trimesh, dict]:
    parts = mesh.split(only_watertight=False)
    parts.sort(key=lambda m: len(m.faces), reverse=True)
    before = len(parts)
    kept: list[trimesh.Trimesh] = []
    for i, p in enumerate(parts):
        if i < max(1, keep) or (min_faces and len(p.faces) >= min_faces):
            kept.append(p)
    if not kept:
        kept = parts[:1]
    out = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
    return out, {
        "components": before,
        "kept": len(kept),
        "faces_in": int(len(mesh.faces)),
        "faces_out": int(len(out.faces)),
    }


def delete_sphere(mesh: trimesh.Trimesh, center: np.ndarray, radius: float) -> tuple[trimesh.Trimesh, dict]:
    center = np.asarray(center, dtype=np.float64).reshape(3)
    dist = np.linalg.norm(mesh.vertices - center, axis=1)
    inside = dist <= radius
    face_inside = inside[mesh.faces].any(axis=1)
    keep_faces = ~face_inside
    n_del = int(np.count_nonzero(face_inside))
    if not np.any(keep_faces):
        raise ValueError("sphere would delete the entire mesh")
    out = mesh.submesh([keep_faces], append=True, repair=False)
    if isinstance(out, list):
        out = trimesh.util.concatenate(out) if out else mesh
    return out, {
        "faces_in": int(len(mesh.faces)),
        "faces_out": int(len(out.faces)),
        "faces_deleted": n_del,
        "radius": float(radius),
        "center": center.tolist(),
    }


def patch_lods(asset: Path, filename: str = "mesh_clean.glb") -> None:
    path = asset / "lods.json"
    item = {"id": "clean", "file": filename, "label": "Clean"}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"lods": []}
    else:
        data = {"lods": []}
    lods = [x for x in data.get("lods") or [] if x.get("id") != "clean"]
    lods.insert(0, item)
    data["lods"] = lods
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stem", required=True)
    p.add_argument("--glb", type=Path, default=None)
    p.add_argument("--op", choices=("floaters", "sphere"), default="floaters")
    p.add_argument("--keep", type=int, default=1, help="largest components to keep")
    p.add_argument("--min-faces", type=int, default=0)
    p.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    p.add_argument("--radius", type=float, default=0.05)
    p.add_argument("--out-name", default="mesh_clean.glb")
    args = p.parse_args()
    asset = root / "out" / args.stem
    src = args.glb or pick_glb(root, args.stem)
    if src is None:
        print("clean_mesh: no GLB", file=sys.stderr)
        return 2
    mesh = _as_mesh(trimesh.load(str(src), force="mesh"))
    if args.op == "floaters":
        mesh, info = remove_floaters(mesh, keep=max(1, args.keep), min_faces=args.min_faces)
    else:
        if not args.center:
            print("clean_mesh: --center X Y Z required for sphere", file=sys.stderr)
            return 2
        mesh, info = delete_sphere(mesh, args.center, args.radius)
    dest = asset / args.out_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dest))
    patch_lods(asset, args.out_name)
    info.update({"ok": True, "source": str(src), "out": str(dest), "op": args.op})
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
