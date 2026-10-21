"""Optional heat binder: Blender's bone-heat (Automatic Weights) skin for a rig, as a second opinion to skin-tokens.

  .venv/bin/python scripts/rig_heat.py --stem S [--source psx.glb] [--skeleton rig/template_posed.glb]
                                       [--out rig/skinned_heat.glb] [--weld 0.002]

run_heat(mesh, skeleton, out) is a drop-in for rig.run_skin(mesh, skeleton, out):
  1. builds a skinned GLB from the skeleton (rig_pred.ours_from_posed: the same joints, identity rests, plain
     translations, the same shape skin-tokens returns) and the source mesh, with placeholder weights;
  2. Blender (scripts/blender/heat_bind.py) welds it at 2 mm, runs ARMATURE_AUTO in the rest pose and writes every
     welded point's position and weights by bone name, plus the import sanity numbers (non-manifold edges,
     zero-area faces, minimum edge, scale);
  3. a KD-tree moves those weights back onto the unwelded source vertices by position and onto the skeleton's
     joints by bone name (top 4, renormalised); points heat left empty take their nearest weighted point's weights;
  4. writes `out` like rig/skinned_raw.glb (skeleton + source mesh + JOINTS_0/WEIGHTS_0) in the source's vertex
     order, so rig.py's clean stage (local_weights, fold_tips, share_corners) runs on it unchanged.
Never the default binder: once the fit is right skin-tokens wins (docs research); opt in with --binder heat.
Ports scratch skinres/tools/bl_heat.py, transplant.py and bl_dbg2.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig  # noqa: E402

BL_SCRIPT = Path(__file__).resolve().parent / "blender" / "heat_bind.py"


class HeatError(RuntimeError):
    pass


def transplant(P: np.ndarray, Q: np.ndarray, groups: list[str], weights: list[list], joint_names: list[str],
               height: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Weights on points Q (by group name) -> top-4 (J, W) on points P over joint_names, by nearest position.
    Returns J (N,4) int, W (N,4) float (rows sum to 1) and a report."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    H = float(height if height is not None else max(np.ptp(P[:, 1]), 1e-9))
    jidx = {n: k for k, n in enumerate(joint_names)}
    gmap = np.array([jidx.get(g, -1) for g in groups] + [-1], int)
    unknown = sorted({g for g in groups if g not in jidx})
    nj = len(joint_names)
    D = np.zeros((len(Q), nj))
    for i, row in enumerate(weights):
        for gi, w in row:
            k = gmap[int(gi)]
            if k >= 0:
                D[i, k] += float(w)
    has = D.sum(1) > 1e-6
    if not has.any():
        raise HeatError("heat gave no point any weight")
    tree_all = cKDTree(Q)
    dist, near = tree_all.query(P)
    tree_w = cKDTree(Q[has])
    widx = np.flatnonzero(has)
    take = near.copy()
    empty = ~has[near]
    if empty.any():
        _, nn = tree_w.query(Q[near[empty]])
        take[empty] = widx[nn]
    M = D[take]
    order = np.argsort(-M, axis=1)[:, :4]
    W = np.take_along_axis(M, order, 1)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    J = np.where(W > 0, order, 0)
    used = sorted({joint_names[k] for k in np.unique(J[W > 0])})
    rep = {"points": int(len(P)), "welded_points": int(len(Q)),
           "dist_median_H": round(float(np.median(dist)) / H, 6), "dist_max_H": round(float(dist.max()) / H, 6),
           "unweighted_welded": int((~has).sum()), "filled_points": int(empty.sum()),
           "bones_used": len(used), "bones_unused": [n for n in joint_names if n not in used],
           "unknown_groups": unknown}
    return J.astype(np.int64), W, rep


def run_heat(mesh: Path, skeleton: Path, out: Path, weld: float = 0.002, blender: str | None = None,
             log: Path | None = None) -> dict:
    """Heat-bind `mesh` (a source GLB) to `skeleton` (a posed template GLB) and write `out`. Returns a report with
    `seconds`, `sanity`, `transplant` and `notes`; raises HeatError when Blender is missing or the bind fails."""
    import rig_pred
    t0 = time.time()
    blender = blender or rig.blender_path()
    if not blender:
        raise HeatError("Blender not found (set BLENDER to its binary)")
    bdoc, bbin = rig_pred.ours_from_posed(Path(skeleton))
    src, sbin = rig.read_glb(Path(mesh))
    P = rig.accessor(src, sbin, src["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).astype(float)
    names = [bdoc["nodes"][j]["name"] for j in bdoc["skins"][0]["joints"]]
    J0 = np.zeros((len(P), 4), np.int64)
    W0 = np.zeros((len(P), 4))
    W0[:, 0] = 1.0
    with tempfile.TemporaryDirectory(prefix="rig_heat_") as td:
        inp, res = Path(td) / "heat_in.glb", Path(td) / "heat.json"
        rig.write_rigged(inp, bdoc, bbin, src, sbin, J0, W0, [])
        proc = subprocess.run([blender, "-b", "--factory-startup", "-P", str(BL_SCRIPT), "--", str(inp), str(res),
                               "--weld", str(weld)], capture_output=True, text=True, timeout=1200)
        text = proc.stdout + "\n" + proc.stderr
        if log is not None:
            Path(log).write_text(text, encoding="utf-8")
        fail = [ln for ln in proc.stdout.splitlines() if ln.startswith("HEAT_FAIL")]
        if fail or proc.returncode != 0 or not res.is_file():
            raise HeatError((fail[0][10:] if fail else "Blender heat bind failed: ") + text[-600:])
        got = json.loads(res.read_text())
    notes = []
    if "failed to find solution" in text.lower():
        notes.append("Blender's heat solver failed on one or more bones (those points fall back to their neighbours)")
    Q = np.array(got["positions"], float)
    H = float(np.ptp(P[:, 1])) or 1.0
    J, W, rep = transplant(P, Q, got["groups"], got["weights"], names, H)
    if rep["dist_median_H"] > 1e-3 or rep["dist_max_H"] > 0.05:
        raise HeatError(f"Blender's points don't line up with the source (median {rep['dist_median_H']}H, "
                        f"max {rep['dist_max_H']}H off): a frame mismatch")
    if got.get("empty_groups"):
        notes.append(f"{len(got['empty_groups'])} bones got no heat weight: {', '.join(got['empty_groups'][:8])}")
    if rep["unknown_groups"]:
        notes.append(f"dropped weight on groups not in the skeleton: {', '.join(rep['unknown_groups'][:8])}")
    rig.write_rigged(Path(out), bdoc, bbin, src, sbin, J, W, [])
    return {"seconds": round(time.time() - t0, 1), "weld": weld, "sanity": got.get("sanity"),
            "empty_groups": got.get("empty_groups", []), "transplant": rep, "notes": notes,
            "out": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--source", default="", help="mesh in out/<stem>/ (default psx.glb, then textured_photo.glb, textured.glb)")
    ap.add_argument("--skeleton", default="rig/template_posed.glb", help="posed template, relative to out/<stem>/")
    ap.add_argument("--out", default="rig/skinned_heat.glb", help="relative to out/<stem>/")
    ap.add_argument("--weld", type=float, default=0.002, help="weld distance before heat, in mesh units (2 mm)")
    args = ap.parse_args()
    asset = Path(os.environ.get("FACTORY_ROOT") or rig.ROOT) / "out" / args.stem
    if not asset.is_dir():
        raise SystemExit(f"rig_heat: no asset {args.stem}")
    source = asset / args.source if args.source else next(
        (asset / n for n in ("psx.glb", "textured_photo.glb", "textured.glb") if (asset / n).is_file()), None)
    skel = asset / args.skeleton
    if source is None or not source.is_file():
        raise SystemExit("rig_heat: no source mesh")
    if not skel.is_file():
        raise SystemExit(f"rig_heat: no skeleton {args.skeleton} (rig the asset once first)")
    out = asset / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        rep = run_heat(source, skel, out, weld=args.weld, log=out.with_suffix(".log"))
    except HeatError as e:
        raise SystemExit(f"rig_heat: {e}")
    rep["source"], rep["skeleton"] = source.name, args.skeleton
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    s = rep["sanity"] or {}
    imp, wel = s.get("imported") or {}, s.get("welded") or {}
    print(f"rig_heat: import {imp.get('verts')} points -> {wel.get('verts')} welded at {args.weld}; "
          f"{wel.get('nonmanifold_edges')} non-manifold edges, {wel.get('zero_area_faces')} zero-area faces, "
          f"min edge {wel.get('min_edge')}, mesh scale {s.get('mesh_scale')}")
    t = rep["transplant"]
    print(f"rig_heat: {t['bones_used']} bones used, {t['filled_points']} points filled from neighbours, "
          f"transplant max {t['dist_max_H']}H; {rep['seconds']} s -> {out}")
    for n in rep["notes"]:
        print("rig_heat: " + n)
    print("HEAT_RESULT " + json.dumps({k: rep[k] for k in ("seconds", "out")} | {"bones_used": t["bones_used"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
