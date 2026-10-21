#!/usr/bin/env python3
"""Watertight voxel remesh of Hunyuan soup so paint has one solid to texture.

This is the realism path: keep volume, do not explode faces or posterize.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stylize_pixel import _as_mesh, connected_remesh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesh")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--voxels", type=int, default=96)
    ap.add_argument("--faces", type=int, default=100000)
    ap.add_argument("--smooth", type=int, default=12, help="Taubin before paint only — never after UVs exist")
    args = ap.parse_args()
    src = _as_mesh(__import__("trimesh").load(args.mesh, force="mesh"))
    solid = connected_remesh(src, target_faces=args.faces, voxels=args.voxels)
    if args.smooth > 0:
        import trimesh

        print(f"solidify: taubin smooth {args.smooth} iters (kills voxel ridges)", flush=True)
        trimesh.smoothing.filter_taubin(solid, lamb=0.5, nu=-0.53, iterations=args.smooth)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    solid.export(args.out)
    payload = {
        "ok": True,
        "voxels": args.voxels,
        "smooth": args.smooth,
        "faces": int(len(solid.faces)),
        "vertices": int(len(solid.vertices)),
        "glb": str(args.out),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"SOLIDIFY_RESULT {args.out}")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
