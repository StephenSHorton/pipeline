#!/usr/bin/env python3
"""Turn a multi-view shape so the canonical view lands at azimuth 0, and write shape.json.

Hunyuan3D-2mv puts its *front* view at azimuth 0 (camera on +Z, glTF Y up). The rest of the pipeline (hy3d paint,
retexture's anchor, psx --facing) expects the *canonical* picture there, which for a creature is the side-on
`right` view (head toward image right). This script tries the four quarter turns about +Y, renders an
orthographic silhouette of the turned mesh at each named view's azimuth, and keeps the turn whose silhouettes
best match the cut-out views (mean IoU). Below --min-iou the result is degraded (printed and recorded).

Azimuths follow retexture.py's cameras (eye_on_sphere): relative to the front camera, `left` is +90 (nose
toward image left), `back` 180 and `right` -90 (nose toward image right).

usage: mv_orient.py <mesh.glb> --view right=lifted/right.png [--view front=...] --canonical right
                    [-o out.glb] [--json shape.json] [--no-turn] [--meta k=v ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

AZ = {"front": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}
YAWS = (0, 90, 180, 270)
SIZE = 128
MIN_IOU = 0.5


def pack(mask: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Crop a boolean mask to its bbox and fit it, aspect kept, centred in a size x size square."""
    ys, xs = np.nonzero(mask)
    out = np.zeros((size, size), bool)
    if len(xs) == 0:
        return out
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = (size - 4) / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    im = Image.fromarray(crop.astype(np.uint8) * 255).resize((nw, nh), Image.BILINEAR)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = np.asarray(im) > 127
    return out


def image_mask(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        a = np.asarray(im.convert("RGBA"))[..., 3]
        return a > 127
    rgb = np.asarray(im.convert("RGB")).astype(np.int32)
    # No alpha: a flat ground taken from the corners.
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    ground = np.median(corners, 0)
    return np.abs(rgb - ground).sum(-1) > 60


def yaw_matrix(deg: float) -> np.ndarray:
    """Rotation about +Y by deg (right-handed: +X toward -Z at +90)."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def silhouette(points: np.ndarray, az_deg: float, res: int = 256) -> np.ndarray:
    """Orthographic silhouette from a camera at az (retexture's eye_on_sphere), image right = (cos, 0, -sin)."""
    a = np.deg2rad(az_deg)
    u = points[:, 0] * np.cos(a) - points[:, 2] * np.sin(a)
    v = points[:, 1]
    lo_u, hi_u, lo_v, hi_v = u.min(), u.max(), v.min(), v.max()
    span = max(hi_u - lo_u, hi_v - lo_v, 1e-9)
    px = ((u - lo_u) / span * (res - 1)).astype(int)
    py = ((hi_v - v) / span * (res - 1)).astype(int)
    m = np.zeros((res, res), bool)
    m[py, px] = True
    m = ndimage.binary_closing(m, iterations=2)
    return ndimage.binary_fill_holes(m)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def score(points: np.ndarray, refs: dict[str, np.ndarray], canonical: str) -> dict:
    """IoU of each named view after each quarter turn; the canonical view is placed at az 0."""
    out = {}
    for yaw in YAWS:
        p = points @ yaw_matrix(yaw).T
        per = {}
        for vid, ref in refs.items():
            az = AZ[vid] - AZ[canonical]
            per[vid] = round(iou(pack(silhouette(p, az)), ref), 4)
        out[yaw] = {"views": per, "mean": round(float(np.mean(list(per.values()))), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None, help="turned mesh (default: overwrite the input)")
    ap.add_argument("--view", action="append", default=[], help="id=path, id in front|left|back|right")
    ap.add_argument("--canonical", required=True, choices=sorted(AZ))
    ap.add_argument("--json", type=Path, default=None, help="write/merge the result into this shape.json")
    ap.add_argument("--no-turn", action="store_true", help="report the IoU at yaw 0 only (single-image shape)")
    ap.add_argument("--min-iou", type=float, default=MIN_IOU)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--meta", action="append", default=[], help="k=v recorded in shape.json")
    args = ap.parse_args()

    views = {}
    for item in args.view:
        vid, _, path = item.partition("=")
        if vid not in AZ or not path:
            print(f"mv_orient: bad --view {item!r} (id=path, id in {sorted(AZ)})", file=sys.stderr)
            return 2
        views[vid] = Path(path)
    if args.canonical not in views:
        print(f"mv_orient: the canonical view {args.canonical} is not among --view", file=sys.stderr)
        return 2

    mesh = trimesh.load(str(args.mesh), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        print("mv_orient: not a triangle mesh", file=sys.stderr)
        return 1
    pts, _ = trimesh.sample.sample_surface(mesh, args.samples, seed=0)
    pts = np.asarray(pts) - mesh.bounds.mean(0)
    refs = {vid: pack(image_mask(p)) for vid, p in views.items()}

    scores = score(pts, refs, args.canonical)
    if args.no_turn:
        yaw = 0
    else:
        # Ties (a symmetric prop) keep the model's own frame.
        yaw = max(YAWS, key=lambda y: (scores[y]["mean"], -YAWS.index(y)))
    best = scores[yaw]
    out_path = args.out or args.mesh
    if yaw:
        mesh.apply_transform(np.block([[yaw_matrix(yaw), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    if yaw or args.out:
        mesh.export(str(out_path))

    degraded = []
    if best["mean"] < args.min_iou:
        msg = f"silhouette IoU {best['mean']:.2f} < {args.min_iou:.2f} at yaw {yaw}"
        degraded.append({"stage": "shape", "code": "orient_low_iou", "message": msg})
    orient = {"yaw": yaw, "canonical": args.canonical, "iou": best["mean"],
              "iou_canonical": best["views"][args.canonical], "views": best["views"],
              "turned": not args.no_turn, "candidates": {str(k): v["mean"] for k, v in scores.items()}}
    print(f"mv_orient: yaw {yaw} iou {best['mean']:.3f} (canonical {args.canonical} "
          f"{best['views'][args.canonical]:.3f}) candidates "
          + " ".join(f"{k}:{v['mean']:.3f}" for k, v in scores.items()))
    if args.json:
        data = {}
        if args.json.is_file():
            try:
                data = json.loads(args.json.read_text())
            except ValueError:
                data = {}
        for kv in args.meta:
            k, _, v = kv.partition("=")
            data[k] = int(v) if v.lstrip("-").isdigit() else v
        data["orient"] = orient
        data["degraded"] = [d for d in data.get("degraded", []) if d.get("code") != "orient_low_iou"] + degraded
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n")
    print("MV_ORIENT " + json.dumps(orient, separators=(",", ":")))
    return 3 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
