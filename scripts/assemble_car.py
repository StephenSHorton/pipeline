#!/usr/bin/env python3
"""Assemble a car from separately made parts: one body GLB and one wheel GLB (each a psx.glb from generate.sh).

    .venv/bin/python scripts/assemble_car.py --body out/car_body/psx.glb --wheel out/car_wheel/psx.glb \
        --out out/car/car.glb [--length 3.8] [--wheel-d 0.62] [--wheel-w 0.34] [--axle 0.31] [--track 0.0] \
        [--drop 0.0]

Frame: glTF, Y up, the car looks down +Z, metres, wheels touch y=0. Both inputs are put in that frame first: the
body is scaled to --length along Z and set on the wheels (its underside --drop above the axle line); the wheel's
disc axis is turned to X and it is scaled to --wheel-d across and --wheel-w wide.

The output scene is: car (root) > body, wheel_FL, wheel_FR, wheel_RL, wheel_RR. Each wheel node sits at its axle
centre, so rotating it about its own X spins it. The right-hand wheels are the left one turned 180 degrees about Y so
the face the picture drew stays on the outside. All four wheels share one mesh and one texture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    s = trimesh.load(str(path))
    geoms = list(s.geometry.values()) if isinstance(s, trimesh.Scene) else [s]
    return trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]


def rot_y(deg: float) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--wheel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=float, default=3.8)
    ap.add_argument("--wheel-d", type=float, default=0.62)
    ap.add_argument("--wheel-w", type=float, default=0.34)
    ap.add_argument("--axle", type=float, default=0.31, help="axle distance from the middle, as a share of length")
    ap.add_argument("--track", type=float, default=0.0, help="push wheels out (+) or in (-), metres")
    ap.add_argument("--drop", type=float, default=0.0, help="body underside above the axle line, metres")
    ap.add_argument("--wheel-yaw", type=float, default=90.0, help="turn the wheel disc axis from Z to X")
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    body = load_mesh(Path(a.body))
    wheel = load_mesh(Path(a.wheel))

    # Body: centre on x/z, length along Z set to --length, underside at axle line + drop.
    lo, hi = body.bounds
    sc = a.length / (hi[2] - lo[2])
    body.apply_translation(-np.array([(lo[0] + hi[0]) / 2, 0, (lo[2] + hi[2]) / 2]))
    body.apply_scale(sc)
    r = a.wheel_d / 2
    body.apply_translation([0, r + a.drop - body.bounds[0][1], 0])

    # Wheel: disc axis to X, centred on its own centre, diameter/width set.
    wheel.apply_transform(rot_y(a.wheel_yaw))
    wlo, whi = wheel.bounds
    wheel.apply_translation(-(wlo + whi) / 2)
    wlo, whi = wheel.bounds
    ext = whi - wlo
    d_now = max(ext[1], ext[2])
    wheel.apply_scale(1.0)  # keep type
    S = np.diag([a.wheel_w / ext[0], a.wheel_d / ext[1], a.wheel_d / ext[2], 1.0])
    wheel.apply_transform(S)

    bw = body.bounds[1][0]  # body half-width
    x = bw - a.wheel_w / 2 + a.track
    z = a.axle * a.length
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body", geom_name="body")
    scene.add_geometry(wheel, node_name="wheel_FL", geom_name="wheel", transform=trimesh.transformations.translation_matrix([x, r, z]))
    scene.add_geometry(wheel, node_name="wheel_FR", geom_name="wheel", transform=trimesh.transformations.translation_matrix([-x, r, z]) @ rot_y(180))
    scene.add_geometry(wheel, node_name="wheel_RL", geom_name="wheel", transform=trimesh.transformations.translation_matrix([x, r, -z]))
    scene.add_geometry(wheel, node_name="wheel_RR", geom_name="wheel", transform=trimesh.transformations.translation_matrix([-x, r, -z]) @ rot_y(180))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    # PS1 look: nearest sampling, as the single-model psx.glb has (trimesh drops the sampler on export).
    import pygltflib
    g = pygltflib.GLTF2().load(str(out))
    g.samplers = [pygltflib.Sampler(magFilter=9728, minFilter=9728, wrapS=33071, wrapT=33071)]
    for t in g.textures:
        t.sampler = 0
    g.save(str(out))

    tris = len(body.faces) + 4 * len(wheel.faces)
    rep = {
        "glb": str(out),
        "triangles": {"body": int(len(body.faces)), "wheel_each": int(len(wheel.faces)), "total": int(tris)},
        "size_m": [float(v) for v in (scene.bounds[1] - scene.bounds[0])],
        "wheels": {"diameter": a.wheel_d, "width": a.wheel_w, "x": x, "z": z, "y": r},
        "nodes": sorted(scene.graph.nodes_geometry),
    }
    for name, m in (("body", body), ("wheel", wheel)):
        v = m.visual
        mat = getattr(v, "material", None)
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        rep.setdefault("textures", {})[name] = list(img.size) if img is not None else None
    print(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
