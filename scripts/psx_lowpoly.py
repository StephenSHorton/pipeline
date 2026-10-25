#!/usr/bin/env python3
"""PS1 budget pass: a photo-baked GLB cut down to a low-poly, flat-shaded model with a tiny crunched atlas.

    .venv/bin/python scripts/psx_lowpoly.py out/<stem>/textured_photo.glb --out out/<stem>/psx --name cow \
        --height 58 [--facing right] [--faces 1200] [--atlas 256] [--yaw 0] [--up y] [--glb out/<stem>/psx.glb]

What a PS1 artist did by hand: decimate the detailed mesh to --faces, unwrap again (xatlas) and re-bake the
texture from the detailed mesh onto the low one (each texel takes the colour of the closest point on the
original), crunch the atlas to --atlas px and 15-bit colour, and write unshared corners with one normal per face
so it shades flat.

Writes <out>/<name>.obj + <name>.png + psx.json, and with --glb the same model as a GLB (nearest sampler, flat
normals, glTF axes, metres) for the viewer and engines that take glTF.

Axes. Hunyuan's +Z is the side the photo saw. --facing says where the subject looks in the photo: camera (a
portrait, the default), right or left (side-on, like the cow), away. The model is turned so the subject looks
down glTF +Z. The .obj then follows --up: y (default) keeps glTF axes (Y up, +Z front), which is what s&box's .obj
importer wants: it converts glTF axes to Source itself. z is Z up with that front turned to +X (Source axes), for
tools that take Source axes as they are; s&box turns such a file a second time and it lands on its side unless it
is rotated at import. Units are inches either way (1 unit = 1 inch); --height is in .obj units, feet on 0, centred.

Facing gate. After the model is placed (head down +Z by --facing), facing.check_mesh votes on where the head
really is: the toes, the narrow end and a red comb (rig_fit.geometric_cues) plus the vision model on four textured
renders. If the toes AND the vision model both put the head at -Z, the model is turned a further 180 degrees (a
re-place at +180: x and z negated) and psx.json records facing_check {flipped: true, why} and facing_source
'cues'. --facing-check off (or PSX_FACING_CHECK=off) skips it; a failing check never fails the pass.

Legs. --legs clean (default keep) swaps a creature's decimated legs (Hunyuan's lumpy, slanted legs cut to jagged
facets with flat triangles splayed at the hooves) for straight columns. The legs are found on the detailed placed
mesh with workbench/edit.py's leg finder; the low faces wholly below the band top and within a leg's reach go, and
each leg gets an N-sided (--leg-sides) prism on the straight line through its centres, from the floor up into the
body, radius --leg-scale times the measured leg radius at the top tapering to --leg-taper of that at the floor. The
columns come out of the --faces budget (the body is decimated to the rest) and bake from their own leg's texels.
A shape with no legs (a hay bale) keeps its own, with a note in psx.json's legs and degraded.

--leg-weight W (default 1, off) keeps the decimated legs but gives them W times their share of the budget, as
--head-weight does the head: before the cut each leg's x,z are scaled W times about its centre line (fading out over
the top of the leg band, heights unchanged), and the low mesh is mapped back exactly after it. The legs stay part of
the mesh; tidy them on the detailed mesh first (workbench mesh_edit tidy_legs). Not with --legs clean.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))  # facing.py, rig*.py next to this file
from reduce_mesh import simplify  # noqa: E402

# Degrees about +Y (glTF up) that turn the subject to look down +Z, by where it looks in the photo.
FACING = {"camera": 0.0, "right": -90.0, "left": 90.0, "away": 180.0}
INCH = 0.0254  # the GLB is in metres; --height is in .obj units, inches for s&box


def load_textured(path: Path) -> tuple[trimesh.Trimesh, np.ndarray]:
    scene = trimesh.load(str(path), force="scene")
    meshes = [g for g in scene.dump(concatenate=False) if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise SystemExit(f"{path}: no triangle mesh")
    mesh = max(meshes, key=lambda m: len(m.faces))
    visual = mesh.visual
    if not hasattr(visual, "uv") or visual.uv is None:
        raise SystemExit(f"{path}: the mesh has no UVs (use the textured or textured_photo GLB)")
    material = visual.material
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is None:
        raise SystemExit(f"{path}: no base colour texture")
    return mesh, np.asarray(image.convert("RGB"), dtype=np.float32)


def place(mesh: trimesh.Trimesh, height: float, yaw: float) -> trimesh.Trimesh:
    """glTF axes: turned yaw degrees about +Y, scaled to height, feet on y = 0, centred on x and z."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if yaw:
        a = np.deg2rad(yaw)
        c, s = np.cos(a), np.sin(a)
        v = v @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])  # x' = c x + s z, z' = -s x + c z: +X -> -Z at +90
    lo, hi = v.min(0), v.max(0)
    v *= height / max(hi[1] - lo[1], 1e-6)
    lo, hi = v.min(0), v.max(0)
    centre = (lo + hi) / 2
    centre[1] = lo[1]
    out = mesh.copy()
    out.vertices = v - centre
    return out


def to_obj_axes(v: np.ndarray, up: str) -> np.ndarray:
    """glTF (Y up, front +Z) to Z up with the front on +X: (x, y, z) = (z, x, y), a rotation, keeps handedness."""
    if up == "y":
        return v
    return np.stack([v[:, 2], v[:, 0], v[:, 1]], axis=1)


CRUMB = 0.02  # a piece of the detailed mesh under this share of its area is a crumb Hunyuan left floating
SLIVER = 0.001  # after decimation, pieces under this share of the area (or of one face) are slivers


def _keep(mesh: trimesh.Trimesh, ok) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    keep = [p for p in parts if ok(p)] or [max(parts, key=lambda p: p.area)]
    return keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)


def decimate(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    plain = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if len(plain.faces) <= faces:
        return plain
    # Hunyuan leaves floating crumbs; at a PS1 budget every triangle on one counts. Drop them on the detailed
    # mesh, where a crumb really is a separate piece.
    plain = _keep(plain, lambda p: p.area >= CRUMB * plain.area)
    low = simplify(plain, faces)
    low.remove_unreferenced_vertices()
    # Decimation pinches thin parts (a sheep's leg below the hock, its ears) down to a vertex or two, and split()
    # joins by edges, so they come out as pieces of their own. They are the model: keep them, drop only slivers.
    return _keep(low, lambda p: len(p.faces) > 1 and p.area >= SLIVER * low.area)


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def decimate_head(mesh: trimesh.Trimesh, faces: int, weight: float, f0: float = 0.72, f1: float = 0.80) -> trimesh.Trimesh:
    """decimate() with the head given a bigger share of the budget: the front of the placed model (the head looks
    down +Z) is scaled up `weight` times across and up before decimating, then the low mesh is scaled back. The
    scale ramps in over the front f0..f1 of the length and leaves the depth alone, so the mapping stays exact."""
    if weight == 1.0:
        return decimate(mesh, faces)
    v = np.asarray(mesh.vertices, float)
    lo, hi = float(v[:, 2].min()), float(v[:, 2].max())
    span = max(hi - lo, 1e-9)
    centre = v[(v[:, 2] - lo) / span > f1].mean(axis=0)

    def scale(p: np.ndarray) -> np.ndarray:
        return 1.0 + (weight - 1.0) * smoothstep(((p[:, 2] - lo) / span - f0) / (f1 - f0))

    warped = mesh.copy()
    warped.vertices = v.copy()
    warped.vertices[:, :2] = centre[:2] + (v[:, :2] - centre[:2]) * scale(v)[:, None]
    low = decimate(warped, faces)
    back = np.asarray(low.vertices, float).copy()
    back[:, :2] = centre[:2] + (back[:, :2] - centre[:2]) / scale(back)[:, None]
    return trimesh.Trimesh(vertices=back, faces=low.faces, process=False)


# --leg-weight: the legs get a bigger share of the budget, the way --head-weight gives the head one.
LEG_CORE = 1.3  # each leg is scaled whole out to this many of its measured radii from its centre line
LEG_WEIGHT_RAMP = 0.15  # the scale fades to 1 over this top share of the leg band, where the legs meet the belly


def leg_warp(high: trimesh.Trimesh) -> dict:
    """The legs of the placed high mesh for --leg-weight, with workbench/edit.py's finder (the sturdy one trim_belly
    and tidy_legs use, which copes with a shortened leg whose hoof splits): band floor and top, and per leg its centre
    line (per-height centres) and core radius. Raises edit.EditError when there are no legs."""
    from workbench import edit
    P = edit._surface(high)
    floor = float(P[:, 1].min())
    top = edit.leg_top(high, P)
    legs, how = edit._sturdy_legs(P, floor, top)
    return {"floor": floor, "top": top, "ys": legs.ys, "C": legs.C, "R": LEG_CORE * legs.R, "from": how}


def _warp_legs(v: np.ndarray, warp: dict, weight: float, inverse: bool = False) -> np.ndarray:
    """Scale each leg's x,z about its centre line by s(y) = 1 + (weight - 1) w(y), w fading from 1 to 0 over the top
    LEG_WEIGHT_RAMP of the band. Per leg the distance d from the line maps piecewise linearly, (0, R, R + b) to
    (0, s R, R + b), identity past R + b, so it is one-to-one at every height and its inverse is exact; heights never
    change. The legs are warped one after another (each map a bijection of the plane), the inverse in reverse order,
    so legs whose scaled cores overlap still come back exactly."""
    out = np.asarray(v, float).copy()
    floor, top = warp["floor"], warp["top"]
    band = max(top - floor, 1e-9)
    y = out[:, 1]
    wy = 1.0 - smoothstep((y - (top - LEG_WEIGHT_RAMP * band)) / (LEG_WEIGHT_RAMP * band))
    s = 1.0 + (weight - 1.0) * wy
    order = range(len(warp["R"]))
    for i in (reversed(order) if inverse else order):
        R = float(warp["R"][i])
        b = 2.0 * np.abs(s - 1.0) * R + 0.5 * R  # the outer slope (R + b - s R) / b stays above 1/2
        c = np.stack([np.interp(y, warp["ys"], warp["C"][i][:, a]) for a in range(2)], axis=1)
        rel = out[:, [0, 2]] - c
        d = np.linalg.norm(rel, axis=1)
        k0, k1 = (np.full_like(s, R), s * R) if not inverse else (s * R, np.full_like(s, R))  # inner knot: from, to
        e = R + b
        d2 = np.where(d < k0, d * k1 / k0, np.where(d < e, k1 + (d - k0) * (e - k1) / (e - k0), d))
        out[:, [0, 2]] = c + rel * (d2 / np.maximum(d, 1e-12))[:, None]
    return out


def decimate_weighted(mesh: trimesh.Trimesh, faces: int, head_weight: float, leg_weight: float = 1.0,
                      warp: dict | None = None) -> trimesh.Trimesh:
    """decimate_head() with the legs also given leg_weight times their share of the budget: before decimating, each
    leg's x,z are scaled about its centre line (_warp_legs), the decimator spends triangles where the error is big,
    then the low mesh is mapped back exactly (the head warp is undone inside decimate_head, then the legs')."""
    if leg_weight == 1.0:
        return decimate_head(mesh, faces, head_weight)
    warp = warp or leg_warp(mesh)
    warped = mesh.copy()
    warped.vertices = _warp_legs(np.asarray(mesh.vertices, float), warp, leg_weight)
    low = decimate_head(warped, faces, head_weight)
    back = _warp_legs(np.asarray(low.vertices, float), warp, leg_weight, inverse=True)
    return trimesh.Trimesh(vertices=_feet_down(back, warp), faces=low.faces, process=False)


FOOT_SNAP = 0.05  # a decimated hoof left this share of the leg band (or less) above the floor is set back down on it


def _feet_down(v: np.ndarray, warp: dict) -> np.ndarray:
    """Put every leg back on the floor: the decimator can leave a hoof's lowest corner a little up (a third of an inch
    on the sheep). Per leg (vertices nearest its centre line, in the bottom of the band) whose lowest point m sits up to
    FOOT_SNAP of the band above the floor, heights below m + FOOT_SNAP band drop by m, tapering to nothing at the top
    of that span: order by height is kept, so nothing turns over, and the rest of the leg is as cut."""
    out = v.copy()
    floor, top = warp["floor"], warp["top"]
    span = FOOT_SNAP * (top - floor)
    low = np.nonzero(out[:, 1] < floor + 0.25 * (top - floor))[0]
    if not len(low):
        return out
    y = out[low, 1]
    d = np.stack([np.linalg.norm(out[low][:, [0, 2]] - np.stack([np.interp(y, warp["ys"], C[:, a]) for a in range(2)], 1),
                                 axis=1) for C in warp["C"]])
    lab = d.argmin(0)
    for i in range(len(warp["C"])):
        mine = low[lab == i]
        if not len(mine):
            continue
        m = float(out[mine, 1].min()) - floor
        if 0.0 < m <= span:
            u = out[mine, 1] - floor - m
            drop = m * np.clip(1.0 - u / span, 0.0, 1.0)
            out[mine, 1] -= drop
    return out


# --legs clean: the decimated legs (Hunyuan's lumpy, slanted legs cut to a few dozen facets, flat triangles splayed
# at the hooves) are swapped for one straight tapered prism per leg.
LEG_FIT = (0.1, 0.75)  # the leg's centre line and radius are read over this share of the band: above the hoof's
#                        forward bulge, below the wool hanging among the tops of the legs
LEG_TOP = (0.5, 0.85)  # the leg's top centre is the median of the finder's centres over this share of the band
LEG_MARGIN = 0.05  # low faces are removed only wholly below the band top less this share of the band
LEG_REACH = 1.6  # ... and within this many source radii of a leg's centre line
LEG_PUSH = 0.25  # the column runs this share of the band up into the body, so the hole it leaves is covered
LEG_BAKE_REACH = 2.5  # a column bakes from high faces within this many source radii of its leg, below the top


def column_faces(sides: int) -> int:
    """Triangles in one leg column: the sides (one band of quads) and a flat bottom cap; no top cap."""
    return 2 * sides + (sides - 2)


def measure_legs(high: trimesh.Trimesh) -> dict:
    """The legs of the placed high mesh with workbench/edit.py's leg finder: the band top, and per leg a straight
    centre line, its radius (the median over LEG_FIT of the band of each height's 80th percentile distance from the
    finder's centre line) and its reach for leg_faces. The line runs from the finder's centre at the floor (where the
    hoof stands) through its median centre over LEG_TOP of the band, on to the top: the centre right at the top drifts into the wool
    hanging there, and a line fitted through a zigzag shank (hoof forward, cannon back, forearm forward again) and
    stretched to the ends leans three or four times as far as the leg does. Raises edit.EditError when the shape has
    no legs to find (a hay bale, a crate)."""
    from workbench import edit
    P = edit._surface(high)
    floor = float(P[:, 1].min())
    top = edit.leg_top(high, P)
    found = edit._Legs(P, floor, top)
    band = top - floor
    sel = (P[:, 1] >= floor + LEG_FIT[0] * band) & (P[:, 1] <= floor + LEG_FIT[1] * band)
    Q = P[sel]
    lab, _, dist = found.nearest(Q)
    edges = np.linspace(floor + LEG_FIT[0] * band, floor + LEG_FIT[1] * band, 13)
    legs = []
    for i in range(found.n):
        mine = lab == i
        y, xz, d = Q[mine, 1], Q[mine][:, [0, 2]], dist[mine]
        if len(y) < 30:
            raise edit.EditError(f"leg {i} has too few points over the middle of the band")
        bottom = found.at(found.C[i], np.array([floor]))[0]
        upper = found.ys[(found.ys >= floor + LEG_TOP[0] * band) & (found.ys <= floor + LEG_TOP[1] * band)]
        if len(upper):  # the line through the floor centre and the upper centre (at its own height), up to the top
            mid = np.median(found.at(found.C[i], upper), axis=0)
            head = bottom + (mid - bottom) * band / max(float(np.median(upper)) - floor, 1e-9)
        else:
            head = found.at(found.C[i], np.array([top]))[0]
        k = np.digitize(y, edges) - 1
        per = [float(np.percentile(d[k == b], 80)) for b in range(len(edges) - 1) if (k == b).sum() >= 10]
        # How far the leg's surface reaches from the straight line at most (its lumps, its bend), for the cut.
        line = bottom[None, :] + ((y - floor) / max(band, 1e-9))[:, None] * (head - bottom)[None, :]
        off = np.linalg.norm(xz - line, axis=1)
        radius = float(np.median(per)) if per else float(found.R[i])
        legs.append({"bottom": bottom, "top": head,
                     "radius": radius, "reach": max(LEG_REACH * radius, 1.3 * float(np.percentile(off, 98)))})
    return {"floor": floor, "top": top, "legs": legs, "finder": found}


def leg_faces(low: trimesh.Trimesh, legs: dict) -> np.ndarray:
    """Which low faces are leg: every corner below the band top (less LEG_MARGIN of the band) and within a leg's
    reach of its straight centre line (LEG_REACH source radii, or further when the leg's own surface strays further
    from the line over LEG_FIT: lumps, a bend), so wool hanging between the legs stays. Then whatever is left
    wholly below that line and no longer joined to the body (a hoof's splayed flap whose far corner was outside the
    reach) goes too."""
    floor, top = legs["floor"], legs["top"]
    cut = top - LEG_MARGIN * (top - floor)
    V = np.asarray(low.vertices, float)
    F = np.asarray(low.faces)
    inleg = np.zeros(len(V), bool)
    for leg in legs["legs"]:
        t = (V[:, 1] - floor) / max(top - floor, 1e-9)
        c = leg["bottom"][None, :] + t[:, None] * (leg["top"] - leg["bottom"])[None, :]
        inleg |= np.linalg.norm(V[:, [0, 2]] - c, axis=1) <= leg["reach"]
    gone = (V[F, 1] < cut).all(1) & inleg[F].all(1)
    rest = np.nonzero(~gone)[0]
    if len(rest):
        part = trimesh.graph.connected_component_labels(trimesh.graph.face_adjacency(F[rest]), node_count=len(rest))
        top_of = np.full(part.max() + 1, -np.inf)
        np.maximum.at(top_of, part, V[F[rest], 1].max(1))
        gone[rest[top_of[part] < cut]] = True
    return gone


def leg_column(bottom: np.ndarray, top: np.ndarray, r_bottom: float, r_top: float, sides: int):
    """One leg: an N-sided prism with horizontal rings centred on the straight line bottom -> top (xyz), the bottom
    ring on the floor (bottom's y), radius r_bottom there and r_top at the top, a flat bottom cap, no top cap.
    A flat side faces +-x (the side view sees a face, not an edge). Outward winding."""
    a = np.pi / sides + 2 * np.pi * np.arange(sides) / sides
    ring = np.stack([np.cos(a), np.zeros(sides), np.sin(a)], axis=1)
    V = np.vstack([bottom + r_bottom * ring, top + r_top * ring])
    F = []
    for j in range(sides):
        p, q = j, (j + 1) % sides
        F += [[p, q + sides, q], [p, p + sides, q + sides]]
    F += [[0, j, j + 1] for j in range(1, sides - 1)]  # bottom fan, facing down
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    # Wind outward whatever the ring order: the sides' normals point away from the centre line.
    tri = V[F[0]]
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    mid = (bottom + top) / 2
    if np.dot(n, tri.mean(0) - mid) < 0:
        F = F[:, ::-1]
    return V, F


def inside(mesh: trimesh.Trimesh, points: np.ndarray) -> bool:
    """Whether every point is enclosed by the mesh as far as rays up and out along +-x and +-z can tell (Hunyuan's
    meshes are not watertight, so contains() can't be trusted): each of the five rays from each point hits it."""
    dirs = np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]], float)
    origins = np.repeat(np.asarray(points, float), len(dirs), axis=0)
    hit = mesh.ray.intersects_any(origins, np.tile(dirs, (len(points), 1)))
    return bool(np.all(hit))


def clean_legs(high: trimesh.Trimesh, faces: int, head_weight: float, sides: int = 6, scale: float = 1.0,
               taper: float = 0.8):
    """decimate_head() with the legs replaced by straight columns. Returns (low, info, regions): regions pairs each
    column's low faces with the high faces of its own leg, for bake() to sample. Raises edit.EditError when the
    shape has no legs (the caller keeps the decimated legs)."""
    legs = measure_legs(high)
    floor, top = legs["floor"], legs["top"]
    band = top - floor
    n = len(legs["legs"])
    cost = n * column_faces(sides)
    if cost >= faces:
        from workbench import edit
        raise edit.EditError(f"{n} leg columns of {sides} sides need {cost} triangles, the budget is {faces}")

    def body(target: int):
        low = decimate_head(high, target, head_weight)
        gone = leg_faces(low, legs)
        kept = trimesh.Trimesh(vertices=low.vertices, faces=np.asarray(low.faces)[~gone], process=False)
        kept.remove_unreferenced_vertices()
        return kept, int(gone.sum())

    target = faces - cost
    kept, removed = body(target)
    # The decimator spent triangles on legs that went: cut again with those given back to the body, until the total
    # is within a percent of the budget (never over it).
    for _ in range(6):
        short = faces - cost - len(kept.faces)
        if short <= max(4, faces // 100) or removed == 0:
            break
        target += short
        again, gone = body(target)
        if len(again.faces) + cost > faces:
            break
        kept, removed = again, gone
    Vs, Fs, regions, info_legs = [np.asarray(kept.vertices, float)], [np.asarray(kept.faces)], [], []
    off, foff = len(kept.vertices), len(kept.faces)
    PUSHES = (LEG_PUSH, 0.75 * LEG_PUSH, 0.5 * LEG_PUSH, 0.25 * LEG_PUSH, 0.0)

    def ends(leg):
        b = np.array([leg["bottom"][0], floor, leg["bottom"][1]])
        t0 = np.array([leg["top"][0], top, leg["top"][1]])
        return b, t0, leg["radius"] * scale

    def max_shift(b, t0, toward):
        """The lean may bring the top over the foot (a vertical column) but never past it: a pair leaning opposite
        ways crosses in a V, the crooked look the columns are for."""
        ahead = (t0[2] - b[2]) * toward  # where the top sits past the foot in the lean's direction (<0: short of it)
        return -ahead if ahead < 0 else float("inf")

    def column(leg, shift, push):
        b, t0, r_top = ends(leg)
        toward = -np.sign(t0[2]) if t0[2] else 0.0
        ts = t0 + np.array([0.0, 0.0, toward * shift])
        t = ts + (ts - b) * push  # along the line, so it stays straight
        V, F = leg_column(b, t, r_top * taper, r_top, sides)
        return b, t, r_top, V, F

    def fit(leg):
        """Up into the body by LEG_PUSH of the band, less where that would poke out of it: the open top ring has to
        stay inside the high mesh. A leg at the very end of the body (a hind leg under the rump's curve, made fat by
        --leg-scale) can stick out behind it even at the band top; its top then leans in along the length, a
        quarter of the column radius at a time (never past vertical), until it is inside. None: nothing covers it."""
        b, t0, r_top = ends(leg)
        toward = -np.sign(t0[2]) if t0[2] else 0.0
        cap = min(2.0 * r_top, max_shift(b, t0, toward)) if toward else 0.0
        for shift in [x for x in np.arange(0, 2.01, 0.25) * r_top if x <= cap + 1e-9]:
            for push in PUSHES:
                if inside(high, column(leg, shift, push)[3][sides:]):
                    return float(shift), push
        return None

    # Left and right legs of a pair (the nearest leg along the length) lean and push alike, or the two cross in a V.
    fits = [fit(leg) for leg in legs["legs"]]
    zs = np.array([leg["top"][1] for leg in legs["legs"]])
    plan: list[tuple[float, float, bool]] = []
    for i, leg in enumerate(legs["legs"]):
        others = [j for j in range(n) if j != i]
        j = min(others, key=lambda k: abs(zs[k] - zs[i])) if others else None
        r = leg["radius"] * scale
        paired = j is not None and abs(zs[j] - zs[i]) < 2.0 * max(r, legs["legs"][j]["radius"] * scale)
        mine, theirs = fits[i], (fits[j] if paired else fits[i])
        if mine is None or theirs is None:
            plan.append((0.0, LEG_PUSH, False))
            continue
        shift = max(mine[0], theirs[0])
        push = min(mine[1], theirs[1])
        b, t0, r_top = ends(leg)
        toward = -np.sign(t0[2]) if t0[2] else 0.0
        shift = min(shift, max_shift(b, t0, toward)) if toward else 0.0
        plan.append((shift, push, bool(inside(high, column(leg, shift, push)[3][sides:]))))
    for i, leg in enumerate(legs["legs"]):
        shift, push, done = plan[i]
        b, t, r_top, V, F = column(leg, shift, push)
        Vs.append(V)
        Fs.append(F + off)
        regions.append(np.arange(foff, foff + len(F)))
        off += len(V)
        foff += len(F)
        info_legs.append({"bottom": [round(float(x), 3) for x in b], "top": [round(float(x), 3) for x in t],
                          "source_radius": round(leg["radius"], 3), "radius_top": round(r_top, 3),
                          "radius_bottom": round(r_top * taper, 3), "push": round(push, 4), "lean_in": round(float(shift), 3), "covered": done,
                          "slant_deg": round(float(np.degrees(np.arctan2(np.linalg.norm((t - b)[[0, 2]]),
                                                                        (t - b)[1]))), 1)})
    low = trimesh.Trimesh(vertices=np.vstack(Vs), faces=np.vstack(Fs), process=False)
    # Each column bakes from its own leg on the high mesh: high faces below the band top, nearest that leg and
    # within LEG_BAKE_REACH radii, so a fat column's top texels take the leg and not belly wool beside it.
    hc = np.asarray(high.triangles_center)
    lab, _, d = legs["finder"].nearest(hc)
    src = []
    for i, leg in enumerate(legs["legs"]):
        mine = np.nonzero((hc[:, 1] < top) & (lab == i) & (d <= LEG_BAKE_REACH * leg["radius"]))[0]
        src.append(mine)
    pairs = [(r, s) for r, s in zip(regions, src) if len(s)]
    width = float(np.ptp(np.asarray(high.vertices)[:, 0]))
    info = {"mode": "clean", "legs_found": n, "sides": sides, "scale": scale, "taper": taper,
            "top_y": round(top, 3), "band": round(band, 3), "faces_removed": removed,
            "column_triangles": cost, "body_triangles": int(len(kept.faces)),
            "radius_vs_body_width": round(float(np.median([x["radius_top"] for x in info_legs])) / max(width, 1e-9), 3),
            "legs": info_legs}
    return low, info, pairs


def bake(high: trimesh.Trimesh, texture: np.ndarray, low: trimesh.Trimesh, atlas: int, regions=None):
    """Unwrap the low mesh and fill its atlas from the closest point on the textured high mesh. regions, a list of
    (low face indices, high face indices), makes those low faces take their closest point among those high faces
    only (clean_legs: a column bakes from its own leg)."""
    pack = xatlas.PackOptions()
    pack.resolution = atlas
    pack.padding = 2
    pack.bilinear = False
    pack.bruteForce = True
    chart = xatlas.ChartOptions()
    chart.max_iterations = 4
    atl = xatlas.Atlas()
    atl.add_mesh(np.asarray(low.vertices, np.float32), np.asarray(low.faces, np.uint32))
    atl.generate(chart, pack)
    vmap, indices, uvs = atl[0]
    verts = np.asarray(low.vertices)[vmap]
    faces = np.asarray(indices, dtype=np.int64)

    # Every texel centre that lands inside a triangle, as a point on the low mesh.
    px = uvs * atlas
    points, texels, owner = [], [], []
    for fi, f in enumerate(faces):
        t = px[f]
        x0, y0 = np.floor(t.min(0)).astype(int)
        x1, y1 = np.ceil(t.max(0)).astype(int)
        xs, ys = np.meshgrid(np.arange(max(x0, 0), min(x1, atlas)), np.arange(max(y0, 0), min(y1, atlas)))
        if xs.size == 0:
            continue
        p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
        a, b, c = t
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        lam = np.linalg.solve(m, (p - a).T).T
        u, w = lam[:, 0], lam[:, 1]
        inside = (u >= -0.02) & (w >= -0.02) & (u + w <= 1.02)
        if not inside.any():
            continue
        bary = np.stack([1 - u - w, u, w], axis=1)[inside]
        points.append(bary @ verts[f])
        texels.append(p[inside].astype(int))
        owner.append(np.full(len(bary), fi))
    points = np.concatenate(points)
    texels = np.concatenate(texels)

    closest, _, tri = trimesh.proximity.closest_point(high, points)
    if regions:
        # xatlas keeps the input's face order, so a face index here is the low mesh's.
        if len(faces) != len(low.faces):
            raise RuntimeError("xatlas changed the face count; can't map regions")
        owner = np.concatenate(owner)
        for lows, highs in regions:
            sel = np.isin(owner, lows)
            if not sel.any():
                continue
            part = trimesh.Trimesh(vertices=high.vertices, faces=np.asarray(high.faces)[highs], process=False)
            c, _, t = trimesh.proximity.closest_point(part, points[sel])
            closest[sel] = c
            tri[sel] = np.asarray(highs)[t]
    bary = trimesh.triangles.points_to_barycentric(high.triangles[tri], closest)
    huv = np.einsum("ij,ijk->ik", bary, np.asarray(high.visual.uv)[high.faces[tri]])
    th, tw = texture.shape[:2]
    sx = np.clip((huv[:, 0] % 1.0) * tw, 0, tw - 1).astype(int)
    sy = np.clip((1.0 - huv[:, 1] % 1.0) * th, 0, th - 1).astype(int)

    image = np.zeros((atlas, atlas, 3), np.float32)
    filled = np.zeros((atlas, atlas), bool)
    ty = atlas - 1 - texels[:, 1]
    image[ty, texels[:, 0]] = texture[sy, sx]
    filled[ty, texels[:, 0]] = True

    # Bleed colour into empty texels so seams never sample black.
    _, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
    image = image[iy, ix]
    return verts, faces, uvs, image, float(filled.mean())


def crunch(image: np.ndarray) -> Image.Image:
    rgb = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray((rgb >> 3) << 3, "RGB")  # 15-bit colour, like PS1 VRAM


def flat_corners(verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray):
    """Unshared corners, one normal per face: (positions, uvs, normals) per corner, 3 per triangle."""
    pos = verts[faces].reshape(-1, 3)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return pos, uvs[faces].reshape(-1, 2), np.repeat(n, 3, axis=0)


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray, name: str, up: str) -> None:
    pos, uv, nrm = flat_corners(to_obj_axes(verts, up), faces, uvs)
    lines = [f"# {name}: ai-asset-pipeline PSX low-poly", f"mtllib {name}.mtl", f"o {name}", f"usemtl {name}"]
    lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in pos]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in nrm[::3]]
    lines += [f"f {i + 1}/{i + 1}/{i // 3 + 1} {i + 2}/{i + 2}/{i // 3 + 1} {i + 3}/{i + 3}/{i // 3 + 1}"
              for i in range(0, len(pos), 3)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".mtl").write_text(f"newmtl {name}\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {name}.png\n",
                                        encoding="utf-8")


def write_glb(path: Path, verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray, png: Image.Image,
              name: str) -> None:
    """glTF axes, metres, unlit-friendly matte material, nearest filtering with no mips."""
    pos, uv, nrm = flat_corners(verts * INCH, faces, uvs)
    uv = uv.copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # glTF UV origin is top-left
    buf = BytesIO()
    png.save(buf, "PNG")
    count = len(pos)
    # Corners are unshared, so the index buffer is 0..n-1; importers and inspect_glb expect one anyway.
    wide = count > 65535
    index = np.arange(count, dtype="<u4" if wide else "<u2")
    blobs = [pos.astype("<f4").tobytes(), nrm.astype("<f4").tobytes(), uv.astype("<f4").tobytes(), buf.getvalue(),
             index.tobytes()]
    views, offset = [], 0
    for i, b in enumerate(blobs):
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(b)}
        if i < 3:
            view["target"] = 34962
        elif i == 4:
            view["target"] = 34963
        views.append(view)
        offset += (len(b) + 3) & ~3
    doc = {
        "asset": {"version": "2.0", "generator": "ai-asset-pipeline psx_lowpoly"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                                  "indices": 3, "material": 0}]}],
        "materials": [{"name": f"{name}_psx", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],  # NEAREST, clamp
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": count, "type": "VEC2"},
            {"bufferView": 4, "componentType": 5125 if wide else 5123, "count": count, "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    body = b"".join(b + b"\0" * (((len(b) + 3) & ~3) - len(b)) for b in blobs)
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (((len(js) + 3) & ~3) - len(js))
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(body))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(body), 0x004E4942) + body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


def turn_180(verts: np.ndarray) -> np.ndarray:
    """The placed model turned 180 degrees about +Y: the same as place() at yaw + 180 (place centres x and z)."""
    out = np.array(verts, dtype=np.float64, copy=True)
    out[:, 0] *= -1
    out[:, 2] *= -1
    return out


def drawn_facing(src: Path) -> dict:
    """{facing, canonical, facing_source} from the asset's flat/key.json (the source GLB sits in out/<stem>/), for
    when generate.sh does not pass --canonical / --facing-source. {} when there is none."""
    try:
        key = json.loads((Path(src).resolve().parent / "flat" / "key.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(key, dict) or not key.get("facing"):
        return {}
    canonical = bool(key.get("canonical"))
    return {"facing": key["facing"], "canonical": canonical,
            "facing_source": key.get("facing_source") or ("drawn" if canonical else None)}


def subject_kind(src: Path) -> tuple[bool, str | None]:
    """(is a creature, flat2d pose kind) from the asset's intent.json (the source GLB sits in out/<stem>/). With no
    intent.json the subject is unknown and the gate runs (True, None); an intent that names no creature (a crate, a
    tank, subject 'prop') is (False, None): its 'toes' and 'head' mean nothing, so the facing gate is skipped."""
    try:
        intent = json.loads((Path(src).resolve().parent / "intent.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True, None
    if not isinstance(intent, dict):
        return True, None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from flat2d.pose import pose_for
    except Exception:  # noqa: BLE001 - no pose table: keep the gate
        return True, None
    kind = pose_for({**intent, "output": intent.get("output") or "ps1"}, intent.get("prompt"))
    return kind is not None, kind


def facing_gate(verts, faces, uvs, png, name: str, cls: str, declared: str, work: Path, check=None) -> dict:
    """Write a temporary GLB of the placed low model, run facing.check_mesh on it. Returns facing_check."""
    if check is None:
        import facing
        check = facing.check_mesh
    tmp = work / f".{name}_facing.glb"
    try:
        write_glb(tmp, verts, faces, uvs, png, name)
        got = check(tmp, cls, declared)
    finally:
        tmp.unlink(missing_ok=True)
    return got


def degrade(stage: str, code: str, message: str) -> dict:
    """Print 'DEGRADED <stage> <code>: <message>' and record it in $DEGRADED_JSON when generate.sh sets it."""
    row = {"stage": stage, "code": code, "message": message}
    print(f"DEGRADED {stage} {code}: {message}", flush=True)
    path = os.environ.get("DEGRADED_JSON")
    if path:
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            data = {}
        rows = data.get("degraded") or []
        if row not in rows:
            rows.append(row)
        data["degraded"] = rows
        p.write_text(json.dumps(data, indent=2) + "\n")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="textured GLB (textured_photo.glb, textured.glb)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--name", required=True)
    ap.add_argument("--height", type=float, required=True, help="height in .obj units (s&box: inches)")
    ap.add_argument("--facing", choices=sorted(FACING), default="camera",
                    help="where the subject looks in the photo; it is turned to look down the model's front")
    ap.add_argument("--faces", type=int, default=1200, help="triangle budget (PS1 props ran a few hundred to ~1.5k)")
    ap.add_argument("--atlas", type=int, default=256, help="texture size; the PS1 topped out at 256")
    ap.add_argument("--head-weight", type=float, default=1.0,
                    help="give a creature's head this many times its share of the triangles (ears, face); 1 = off")
    ap.add_argument("--leg-weight", type=float, default=1.0,
                    help="give a creature's legs this many times their share of the triangles, so the cut does not "
                         "pinch them (composes with --head-weight; not with --legs clean); 1 = off")
    ap.add_argument("--yaw", type=float, default=0, help="extra turn about the up axis, degrees")
    ap.add_argument("--up", choices=("z", "y"), default="y", help=".obj axes: y = glTF, Y up, +Z front (s&box); z = Z up, +X front (Source)")
    ap.add_argument("--glb", dest="glb_out", type=Path, default=None, help="also write the model as a GLB here")
    ap.add_argument("--facing-source", choices=("drawn", "detected", "user", "cues"), default=None,
                    help="where --facing came from (recorded in psx.json)")
    ap.add_argument("--canonical", action="store_true", help="the picture is a pipeline-drawn canonical profile")
    ap.add_argument("--cls", default="animal", help="class for the facing cues (humanoid skips the narrow-end cue)")
    ap.add_argument("--facing-check", choices=("auto", "off"), default=None,
                    help="check the placed model's head is at +Z and turn it if the toes and the VLM disagree "
                         "(default auto; PSX_FACING_CHECK=off turns it off)")
    ap.add_argument("--legs", choices=("keep", "clean"), default="keep",
                    help="clean: replace each decimated leg with a straight tapered prism (creatures; a shape with no "
                         "legs keeps its own with a note); keep (default): the decimated legs as they are")
    ap.add_argument("--leg-sides", type=int, default=6, help="--legs clean: sides of each leg column")
    ap.add_argument("--leg-scale", type=float, default=1.0,
                    help="--legs clean: column radius at the top as a multiple of the measured leg radius")
    ap.add_argument("--leg-taper", type=float, default=0.8,
                    help="--legs clean: bottom radius over top radius")
    args = ap.parse_args()
    if args.leg_sides < 3 or args.leg_scale <= 0 or args.leg_taper <= 0:
        ap.error("--leg-sides is at least 3; --leg-scale and --leg-taper are positive")
    if args.leg_weight < 1.0:
        ap.error("--leg-weight is at least 1")
    if args.leg_weight != 1.0 and args.legs == "clean":
        ap.error("--leg-weight weights the legs the cut keeps; --legs clean replaces them: give one")

    high, texture = load_textured(args.src)
    high = place(high, args.height, FACING[args.facing] + args.yaw)
    degraded: list[dict] = []
    legs_info: dict | None = None
    regions = None
    low = None
    if args.legs == "clean":
        from workbench import edit
        try:
            low, legs_info, regions = clean_legs(high, args.faces, args.head_weight, args.leg_sides, args.leg_scale,
                                                 args.leg_taper)
        except edit.EditError as e:
            legs_info = {"mode": "keep", "asked": "clean", "why": str(e)}
            print(f"psx_lowpoly: --legs clean found no legs to replace ({e}); kept the decimated legs",
                  file=sys.stderr)
            degraded.append(degrade("psx", "legs_clean", f"no legs to replace ({e}); kept the decimated legs"))
    if low is None and args.leg_weight != 1.0:
        from workbench import edit
        try:
            warp = leg_warp(high)
            low = decimate_weighted(high, args.faces, args.head_weight, args.leg_weight, warp)
            legs_info = {"mode": "weighted", "leg_weight": args.leg_weight, "legs_found": len(warp["R"]),
                         "top_y": round(float(warp["top"]), 3), "found_from": warp["from"]}
        except edit.EditError as e:
            legs_info = {"mode": "keep", "asked": "weighted", "why": str(e)}
            print(f"psx_lowpoly: --leg-weight found no legs to weight ({e}); cut them as they are", file=sys.stderr)
            degraded.append(degrade("psx", "leg_weight", f"no legs to weight ({e}); cut as they are"))
    if low is None:
        low = decimate_head(high, args.faces, args.head_weight)
    verts, faces, uvs, image, used = bake(high, texture, low, args.atlas, regions)
    png = crunch(image)

    gate = args.facing_check or ("off" if (os.environ.get("PSX_FACING_CHECK") or "").lower() in {"off", "0", "no"}
                                 else "auto")
    facing_check: dict = {"skipped": True} if gate == "off" else {}
    drawn = drawn_facing(args.src)
    canonical = bool(args.canonical or (drawn.get("canonical") and drawn.get("facing") == args.facing))
    source = args.facing_source or (drawn.get("facing_source") if drawn.get("facing") == args.facing else None)
    creature, kind = subject_kind(args.src)
    cls = "humanoid" if kind == "humanoid" else args.cls
    if gate == "auto" and not creature and not canonical:
        gate = "off"
        facing_check = {"skipped": True, "why": "not a creature (intent.json): no head or toes to check"}
    if gate == "auto":
        try:
            import facing
            got = facing_gate(verts, faces, uvs, png, args.name, cls, args.facing, args.out)
            facing_check = {"flipped": False, "agrees": got["agrees"], "head_axis": got["head_axis"],
                            "votes": got["votes"], "vlm": got.get("vlm"), "declared": args.facing}
            if got["flipped_needed"]:
                verts = turn_180(verts)
                facing_check["flipped"] = True
                facing_check["why"] = facing.flip_why(args.facing)
                facing_check["decided_by"] = "toes+vlm"
                source = "cues"
                print("psx_lowpoly: warning: " + facing_check["why"], file=sys.stderr)
            else:
                facing_check["decided_by"] = "declared" if (canonical and got["agrees"]) else (
                    "agrees" if got["agrees"] else "declared (cues disagree, not both toes and VLM)")
        except Exception as e:  # noqa: BLE001 - the check is advisory; the pass still writes the model
            facing_check = {"error": f"{e.__class__.__name__}: {e}"}
            print(f"psx_lowpoly: facing check failed: {e}", file=sys.stderr)
            degraded.append(degrade("psx", "facing_check",
                                    f"facing check failed ({e.__class__.__name__}: {e}); kept the declared facing"))

    obj = args.out / f"{args.name}.obj"
    tex = args.out / f"{args.name}.png"
    write_obj(obj, verts, faces, uvs, args.name, args.up)
    png.save(tex)
    if args.glb_out:
        write_glb(args.glb_out, verts, faces, uvs, png, args.name)
    size = to_obj_axes(verts, args.up)
    report = {
        "name": args.name,
        "obj": obj.name,
        "texture": tex.name,
        "glb": args.glb_out.name if args.glb_out else None,
        "triangles": int(len(faces)),
        "faces_asked": int(args.faces),
        "source": args.src.name,
        "source_triangles": int(len(high.faces)),
        "atlas": args.atlas,
        "atlas_used": round(used, 3),
        "obj_axes": "z-up, +x front" if args.up == "z" else "y-up, +z front (glTF)",
        "glb_axes": "y-up, +z front (glTF), metres",
        "height": args.height,
        "facing": args.facing,
        "facing_source": source,
        "canonical": canonical,
        "facing_check": facing_check,
        "degraded": degraded,
        "size": [round(float(x), 1) for x in size.max(0) - size.min(0)],
    }
    if legs_info is not None:
        report["legs"] = legs_info
    (args.out / "psx.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PSX_RESULT " + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
