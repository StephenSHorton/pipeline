#!/usr/bin/env python3
"""Reduce a mesh to a named LOD or an exact face count.

Presets
  lookdev  80,000  paint / UV unwrap
  game     12,000  realtime
  proxy     2,000  collision / thumbnails

`--octree` is documented here so generate.sh and this file share one table;
this script only decimates. Native low-poly generation is hy3d --octree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import trimesh

PRESETS = {
    "hero": {"faces": None, "octree": 256, "label": "High"},
    "lookdev": {"faces": 80_000, "octree": 128, "label": "Medium"},
    "game": {"faces": 12_000, "octree": 64, "label": "Game"},
    "proxy": {"faces": 2_000, "octree": 64, "label": "Low"},
}
PRESET_ALIASES = {
    "high": "hero",
    "medium": "lookdev",
    "low": "proxy",
}


def reduce_mesh(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    if target <= 0 or len(mesh.faces) <= target:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--preset", choices=sorted({*PRESETS, *PRESET_ALIASES}), default=None)
    ap.add_argument("--faces", type=int, default=None)
    ap.add_argument("--report", default=None, help="write a JSON sidecar")
    args = ap.parse_args()

    target = args.faces
    label = "custom"
    preset = PRESET_ALIASES.get(args.preset, args.preset) if args.preset else None
    if preset:
        spec = PRESETS[preset]
        label = spec["label"]
        if target is None:
            target = spec["faces"]
        args.preset = preset

    mesh = trimesh.load(args.inp, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print("reduce: not a triangle mesh", file=sys.stderr)
        return 1

    before_v, before_f = len(mesh.vertices), len(mesh.faces)
    has_uv = bool(getattr(getattr(mesh, "visual", None), "uv", None) is not None)
    if target is None:
        out_mesh = mesh
        target_note = "keep"
    else:
        target = min(target, before_f)
        out_mesh = reduce_mesh(mesh, target)
        target_note = str(target)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(args.out)
    after_v, after_f = len(out_mesh.vertices), len(out_mesh.faces)
    still_uv = bool(getattr(getattr(out_mesh, "visual", None), "uv", None) is not None)

    report = {
        "ok": True,
        "preset": args.preset,
        "label": label,
        "path": str(Path(args.out)),
        "src": str(Path(args.inp)),
        "before_vertices": before_v,
        "before_faces": before_f,
        "vertices": after_v,
        "faces": after_f,
        "target": target_note,
        "had_uvs": has_uv,
        "has_uvs": still_uv,
    }
    print(
        f"reduce: {label} {before_v}v/{before_f}f -> {after_v}v/{after_f}f "
        f"(target {target_note}) uvs {has_uv}->{still_uv}"
    )
    print(f"reduce: wrote {args.out}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print("REDUCE_JSON " + json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
