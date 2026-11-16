#!/usr/bin/env python3
"""Split a hard-surface mesh into named pieces with pivots (Recoil-style).

Heuristic, not a learned part segmenter: antenna = everything above the lid,
handle = thin side protrusion, knobs = two small front-face clusters.
Writes a scene GLB plus articulated.json for the viewer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


def submesh(mesh: trimesh.Trimesh, face_mask: np.ndarray, name: str) -> trimesh.Trimesh | None:
    idx = np.nonzero(face_mask)[0]
    if len(idx) < 8:
        return None
    part = mesh.submesh([idx], append=True, repair=False)
    if part is None or len(part.faces) < 8:
        return None
    part.metadata["name"] = name
    return part


def face_mask_from_verts(faces: np.ndarray, vert_mask: np.ndarray, how: str = "majority") -> np.ndarray:
    hits = vert_mask[faces].sum(axis=1)
    if how == "all":
        return hits == 3
    if how == "any":
        return hits >= 1
    return hits >= 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    mesh = trimesh.load(args.inp, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print("articulate: not a triangle mesh", file=sys.stderr)
        return 1
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    ymin, ymax = float(v[:, 1].min()), float(v[:, 1].max())
    y_lid = float(np.percentile(v[:, 1], 94))
    ant_cut = y_lid + 0.02 * (ymax - ymin)
    ant_v = v[:, 1] > ant_cut
    ant_f = face_mask_from_verts(f, ant_v)

    # Handle: extreme -X wall, middle height, not the full side panel — keep a mid-Z band.
    xlo = float(np.percentile(v[:, 0], 4))
    y_lo, y_hi = float(np.percentile(v[:, 1], 30)), float(np.percentile(v[:, 1], 72))
    z_lo, z_hi = float(np.percentile(v[:, 2], 30)), float(np.percentile(v[:, 2], 70))
    han_v = (v[:, 0] <= xlo) & (v[:, 1] >= y_lo) & (v[:, 1] <= y_hi) & (v[:, 2] >= z_lo) & (v[:, 2] <= z_hi)
    han_f = face_mask_from_verts(f, han_v) & ~ant_f

    # Knobs: front (+Z) protrusions below the lid, two X-clusters on the display face.
    z_front = float(np.percentile(v[:, 2], 93))
    kn_v = (v[:, 2] >= z_front) & (v[:, 1] < ant_cut) & (v[:, 1] > float(np.percentile(v[:, 1], 40)))
    kn_f_all = face_mask_from_verts(f, kn_v) & ~ant_f & ~han_f
    kn_faces = np.nonzero(kn_f_all)[0]
    knob_masks = []
    if len(kn_faces) >= 16:
        cents = v[f[kn_faces]].mean(axis=1)
        xmed = np.median(cents[:, 0])
        left = kn_faces[cents[:, 0] < xmed]
        right = kn_faces[cents[:, 0] >= xmed]
        if len(left) >= 8:
            m = np.zeros(len(f), dtype=bool)
            m[left] = True
            knob_masks.append(("KnobA", m))
        if len(right) >= 8:
            m = np.zeros(len(f), dtype=bool)
            m[right] = True
            knob_masks.append(("KnobB", m))

    used = ant_f | han_f
    for _, m in knob_masks:
        used |= m
    body_f = ~used

    parts = []
    for name, mask in [("Antenna", ant_f), ("Handle", han_f), *knob_masks, ("Body", body_f)]:
        p = submesh(mesh, mask, name)
        if p is None:
            print(f"articulate: skip {name} (too small)")
            continue
        parts.append((name, p))
        print(f"articulate: {name} {len(p.vertices)}v {len(p.faces)}f")

    if not any(n == "Body" for n, _ in parts):
        print("articulate: failed to keep a body", file=sys.stderr)
        return 1

    scene = trimesh.Scene()
    pivots = []
    for name, p in parts:
        node = scene.add_geometry(p, node_name=name, geom_name=name)
        c = p.vertices.mean(axis=0)
        if name == "Antenna":
            pivot = [float(c[0]), float(p.vertices[:, 1].min()), float(c[2])]
            axis, amp, hz = [0, 1, 0], 0.45, 0.35
        elif name.startswith("Knob"):
            pivot = [float(c[0]), float(c[1]), float(p.vertices[:, 2].min())]
            axis, amp, hz = [0, 0, 1], 2.8, 0.22
        elif name == "Handle":
            pivot = [float(p.vertices[:, 0].max()), float(c[1]), float(c[2])]
            axis, amp, hz = [0, 0, 1], 0.08, 0.15
        else:
            pivot = [float(c[0]), float(c[1]), float(c[2])]
            axis, amp, hz = [0, 1, 0], 0.0, 0.0
        pivots.append({
            "name": name,
            "node": name,
            "pivot": pivot,
            "axis": axis,
            "amp": amp,
            "hz": hz,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps({"parts": pivots, "src": str(Path(args.inp))}, indent=2) + "\n")
    print(f"articulate: wrote {out} and {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
