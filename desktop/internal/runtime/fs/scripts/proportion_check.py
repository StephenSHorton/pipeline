#!/usr/bin/env python3
"""Give a multi-view shape the photo's length: stretch it along its length axis when a side-on photo says so.

The turnaround views a multi-view shape is built from are drawn from the user's photo, and the drawing can squash a
long animal: the sheep's photo is height/length 0.74 side-on, its drawn side view 1.04 and the shape 0.99, legs a
third of the height in all three, so the length alone was lost (each view is drawn in a square quarter of one
sheet). This script runs after mv_orient.py (the canonical view at azimuth 0) and puts the photo's proportion back.

Is the photo side-on? The mesh is rendered orthographically (mv_orient's silhouette) at azimuths every 15 degrees
from the side view (the canonical view when it is left/right, else az +-90) round to the end views, and at each
one the mesh is also stretched along its length axis over a range of factors; the best silhouette IoU against the
photo (and its mirror image: the opposite side, az + 180, is the exact mirror of an orthographic render) is that
angle's score. Searching the stretch in 3D, not just squaring both boxes, is what tells a 3/4 photo from a side
one: a squashed mesh seen at 45 degrees looks much like a side view once its box is stretched, but stretching the
mesh itself and then turning it does not. The photo counts as side-on only when
  - the side angles (|theta| <= 15) score at least --min-match (the photo is a good fit at all),
  - they beat the end angles (|theta| >= 75) by --end-margin (not a front/back photo), and
  - they trail the oblique angles (30..60) by no more than --oblique-tol (not a 3/4 photo; a real side-on photo
    and the 30-degree render are close, within render noise, so a near tie goes to the side).
Then factor = mesh height/length over photo height/length (bounding boxes), clamped to --min-factor..--max-factor;
under --min-change it is skipped. The mesh is scaled by it along the length axis (the horizontal axis in the side
view's image plane) about its centre, height and width fixed, and written over the input (or -o).

The result merges into shape.json under "proportion" (photo aspect, mesh aspect before/after, factor, the match
scores, applied, why). Exit 0 whether it applied or skipped; 1 when the mesh or photo cannot be read; 2 bad args.

usage: proportion_check.py <mesh.glb> --photo views/lifted/photo.png --canonical right
                           [-o out.glb] [--json shape.json] [--min-factor 0.75] [--max-factor 1.5]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mv_orient import AZ, image_mask, iou, pack, silhouette  # noqa: E402

THETAS = tuple(range(-90, 91, 15))
# Coarse stretch grid (~5% steps), then a fine one (~1%) round the best at the side angle.
COARSE = tuple(np.exp(np.linspace(np.log(0.6), np.log(1.8), 23)))
# Thresholds, from out/mlg-sheep (real side-on photo; the mesh rendered at 0..90 as stand-in photos of a known
# angle) and the synthetic tests: see tests/test_proportion_check.py and the commit message.
MIN_MATCH = 0.70
END_MARGIN = 0.05
OBLIQUE_TOL = 0.01
AGREE = 0.15          # the search's best stretch and the box ratio must agree within 15%
MIN_FACTOR, MAX_FACTOR, MIN_CHANGE = 0.75, 1.5, 0.05


def photo_mask(path: Path) -> np.ndarray:
    """The cut-out's alpha, minus specks under 2% of the largest piece (the halo of grass a cut-out keeps)."""
    m = image_mask(path)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 0.02 * sizes.max()])
    return m


def cut_off(mask: np.ndarray) -> list[str]:
    """The frame edges the subject touches: there its outline is the frame's, not the animal's (hooves cut off at the
    bottom, a tail out at the side), so its height/length is not to be trusted."""
    edges = {"top": mask[0].any(), "bottom": mask[-1].any(), "left": mask[:, 0].any(), "right": mask[:, -1].any()}
    return [k for k, hit in edges.items() if hit]


def body(mesh: trimesh.Trimesh, share: float = 0.02) -> trimesh.Trimesh:
    """The pieces of at least `share` of the area (reduce_mesh's rule): a floating crumb past the rump must not
    count as length."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    keep = [p for p in parts if p.area >= share * mesh.area] or [max(parts, key=lambda p: p.area)]
    return trimesh.util.concatenate(keep)


def aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    return float((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1))


def stretch(axis: np.ndarray, f: float) -> np.ndarray:
    """3x3: scale by f along the unit axis, nothing across it."""
    return np.eye(3) + (f - 1.0) * np.outer(axis, axis)


def length_axis(side_az: float) -> np.ndarray:
    """Image right at side_az (mv_orient's silhouette: right = (cos, 0, -sin))."""
    a = np.deg2rad(side_az)
    return np.array([np.cos(a), 0.0, -np.sin(a)])


def match(pts: np.ndarray, photo: np.ndarray, side_az: float, axis: np.ndarray,
          thetas=THETAS, factors=COARSE) -> dict[int, tuple[float, float]]:
    """Per angle from the side view: (best IoU over the length stretches, that stretch)."""
    p = pack(photo)
    pm = p[:, ::-1]
    out = {}
    for th in thetas:
        best = (-1.0, 1.0)
        for f in factors:
            s = pack(silhouette(pts @ stretch(axis, f).T, side_az + th))
            best = max(best, (max(iou(p, s), iou(pm, s)), float(f)))
        out[th] = best
    return out


def decide(scores: dict[int, tuple[float, float]], min_match=MIN_MATCH, end_margin=END_MARGIN,
           oblique_tol=OBLIQUE_TOL) -> tuple[bool, str, dict]:
    side = max(v[0] for t, v in scores.items() if abs(t) <= 15)
    oblique = max(v[0] for t, v in scores.items() if 30 <= abs(t) <= 60)
    end = max(v[0] for t, v in scores.items() if abs(t) >= 75)
    best = max(scores, key=lambda t: scores[t][0])
    m = {"side": round(side, 4), "oblique": round(oblique, 4), "end": round(end, 4), "best_angle": best}
    if side < min_match:
        return False, f"the photo matches no side of the shape well (best side IoU {side:.2f} < {min_match:.2f})", m
    if side - end < end_margin:
        return False, f"the photo looks front- or back-on (side {side:.3f} vs end {end:.3f})", m
    if oblique - side > oblique_tol:
        return False, f"the photo looks three-quarter (side {side:.3f} vs oblique {oblique:.3f} at {best} deg)", m
    return True, "side-on", m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--photo", type=Path, required=True, help="the photo's cut-out (alpha), views/lifted/photo.png")
    ap.add_argument("--canonical", required=True, choices=sorted(AZ), help="the view mv_orient put at az 0")
    ap.add_argument("-o", "--out", type=Path, default=None, help="scaled mesh (default: overwrite the input)")
    ap.add_argument("--json", type=Path, default=None, help="merge the result into this shape.json")
    ap.add_argument("--min-factor", type=float, default=MIN_FACTOR)
    ap.add_argument("--max-factor", type=float, default=MAX_FACTOR)
    ap.add_argument("--min-change", type=float, default=MIN_CHANGE)
    ap.add_argument("--min-match", type=float, default=MIN_MATCH)
    ap.add_argument("--end-margin", type=float, default=END_MARGIN)
    ap.add_argument("--oblique-tol", type=float, default=OBLIQUE_TOL)
    ap.add_argument("--samples", type=int, default=60_000)
    args = ap.parse_args()
    if not 0 < args.min_factor <= 1 <= args.max_factor:
        print("proportion_check: want 0 < --min-factor <= 1 <= --max-factor", file=sys.stderr)
        return 2

    result: dict = {"photo": str(args.photo), "canonical": args.canonical, "applied": False}
    try:
        mesh = trimesh.load(str(args.mesh), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("not a triangle mesh")
        photo = photo_mask(args.photo)
        if photo.sum() < 100:
            raise ValueError("the photo cut-out is empty")
    except Exception as exc:  # noqa: BLE001 - any unreadable input is the one failure this script reports
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"proportion_check: failed: {result['error']}", file=sys.stderr)
        write_json(args.json, result)
        return 1

    side_az = 0.0 if args.canonical in ("left", "right") else 90.0
    axis = length_axis(side_az)
    solid = body(mesh)
    centre = solid.bounds.mean(0)
    v = solid.vertices - centre
    height = float(np.ptp(v[:, 1]))
    length = float(np.ptp(v @ axis))
    pts, _ = trimesh.sample.sample_surface(solid, args.samples, seed=0)
    pts = np.asarray(pts) - centre

    photo_aspect = aspect(photo)
    mesh_aspect = height / max(length, 1e-9)
    scores = match(pts, photo, side_az, axis)
    side_on, why, m = decide(scores, args.min_match, args.end_margin, args.oblique_tol)
    raw = mesh_aspect / photo_aspect
    result.update({"photo_aspect": round(photo_aspect, 4), "mesh_aspect_before": round(mesh_aspect, 4),
                   "mesh_aspect_after": round(mesh_aspect, 4), "factor_raw": round(raw, 4), "factor": 1.0,
                   "side_on": side_on, "match": m,
                   "angles": {str(t): [round(s, 4), round(f, 3)] for t, (s, f) in scores.items()}})

    edges = cut_off(photo)
    if side_on and edges:
        why = f"the photo's subject runs off the frame ({', '.join(edges)}); its height/length is not trusted"
        side_on = False
        result["cut_off"] = edges
    if side_on:
        # The fine search at the side angle: the IoU's own best stretch must agree with the box ratio, else the
        # photo's box is off (a stick, a shadow, a cropped hoof) and the ratio is not to be trusted.
        c = scores[0][1]
        fine = match(pts, photo, side_az, axis, thetas=(0,), factors=tuple(c * np.exp(np.linspace(-0.06, 0.06, 13))))
        searched = fine[0][1]
        result["factor_search"] = round(searched, 4)
        if abs(np.log(raw / searched)) > np.log(1 + AGREE):
            why = f"the box ratio ({raw:.2f}) and the silhouette fit ({searched:.2f}) disagree; left alone"
            side_on = False
    factor = float(np.clip(raw, args.min_factor, args.max_factor))
    result["clamped"] = bool(factor != raw)
    if side_on and abs(factor - 1.0) < args.min_change:
        why = f"the length is already within {args.min_change:.0%} of the photo's (factor {raw:.3f})"
    elif side_on:
        T = np.eye(4)
        T[:3, :3] = stretch(axis, factor)
        T[:3, 3] = centre - T[:3, :3] @ centre
        mesh.apply_transform(T)
        mesh.export(str(args.out or args.mesh))
        solid = body(mesh)
        v = solid.vertices - solid.bounds.mean(0)
        after = float(np.ptp(v[:, 1]) / max(np.ptp(v @ axis), 1e-9))
        result.update({"applied": True, "factor": round(factor, 4), "mesh_aspect_after": round(after, 4)})
        why = "side-on photo; length scaled" + (f" (clamped from {raw:.3f})" if result["clamped"] else "")
    result["reason"] = why

    if result["applied"]:
        print(f"proportion_check: the photo is side-on, so the shape was made {factor:.2f}x as long "
              f"(height/length {mesh_aspect:.2f} -> {result['mesh_aspect_after']:.2f}, photo {photo_aspect:.2f})")
    else:
        print(f"proportion_check: left alone: {why} (shape height/length {mesh_aspect:.2f}, photo {photo_aspect:.2f})")
    print(f"proportion_check: match side {m['side']:.3f} oblique {m['oblique']:.3f} end {m['end']:.3f} "
          f"(best at {m['best_angle']} deg)")
    write_json(args.json, result)
    print("PROPORTION " + json.dumps({k: result[k] for k in ("applied", "factor", "photo_aspect",
                                      "mesh_aspect_before", "mesh_aspect_after", "reason")}, separators=(",", ":")))
    return 0


def write_json(path: Path | None, result: dict) -> None:
    if not path:
        return
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except ValueError:
            data = {}
    data["proportion"] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
