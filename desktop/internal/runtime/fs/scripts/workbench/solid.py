#!/usr/bin/env python3
"""Close holes and carve parts away in voxel space: the shape fixes the drawings won't make.

The image model draws some things however it is asked: a hole through a jerry can's handle, slots in a propane
tank's collar, see-through car windows, wing mirrors a brief said to leave off. Rewording never removed them (round 3).
Here the mesh is filled solid on a grid, changed there, and surfaced again (mesher.surface_nets), so the answer is
always one closed skin with no open cut faces:

- close: holes straight through the shape (the handle's) are filled slice by slice: looking down each axis, a hole
  ringed by solid in a slice is filled (a ball-shaped closing reaches into a ring's hole from above and below and
  never fills it). size > 0 also closes dents narrower than about 2 x size. Only inside the sphere (center, radius)
  when given, so the rest (an embossed panel, a spout's recess) is kept.
- carve: spheres set empty before closing (a wing mirror, an antenna), so what they cut away is not filled back.
- gaps: with neither, what a closing of `size` would fill, as clusters (centre, voxels, extent), largest first:
  where the holes are, before choosing one.

The answer has no UVs or texture: rebuild it from the drawn views with generate (mesh, paint off).
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

from workbench import mesher

GRID = 192  # voxels along the longest side: a 14 in can at about 0.07 in, far finer than the PS1 cut keeps
PAD = 3
CRUMB = 0.01


class SolidError(ValueError):
    pass


def voxels(mesh: trimesh.Trimesh, grid: int = GRID) -> tuple[np.ndarray, np.ndarray, float]:
    """The mesh filled solid: (inside mask, grid origin, pitch). Each voxel centre is tested for being inside by
    ray parity along x, so a mesh with small cracks still fills (trimesh's voxel fill floods through them)."""
    lo, hi = mesh.bounds
    h = float((hi - lo).max()) / grid
    n = np.ceil((hi - lo) / h).astype(int) + 2 * PAD
    origin = lo - PAD * h
    axes = [origin[a] + h * (np.arange(n[a]) + 0.5) for a in range(3)]
    inside = np.zeros(tuple(n), bool)
    # one ray per (y, z) column, all its x crossings at once: inside between the 1st and 2nd, 3rd and 4th, ...
    Y, Z = np.meshgrid(axes[1], axes[2], indexing="ij")
    origins = np.stack([np.full(Y.size, origin[0] - h), Y.ravel(), Z.ravel()], 1)
    dirs = np.tile([1.0, 0.0, 0.0], (len(origins), 1))
    locs, ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    xs = axes[0]
    for r in np.unique(ray):
        hits = np.sort(locs[ray == r, 0])
        hits = hits[np.concatenate([[True], np.diff(hits) > 1e-7 * h])]  # a ray through an edge hits twice
        j, k = divmod(int(r), len(axes[2]))
        for a, b in zip(hits[0::2], hits[1::2]):
            inside[(xs > a) & (xs < b), j, k] = True
    return inside, origin, h


def _sphere_mask(shape, origin, h, center, radius) -> np.ndarray:
    idx = np.indices(shape).reshape(3, -1).T
    p = origin + h * (idx + 0.5)
    return (np.linalg.norm(p - np.asarray(center, float), axis=1) <= radius).reshape(shape)


def _closed(inside: np.ndarray, r: float) -> np.ndarray:
    """Closing by a ball of r voxels, through distance transforms (a ball kernel took minutes at r 28)."""
    k = int(np.ceil(r)) + 1
    pad = np.pad(inside, k)
    grown = ndimage.distance_transform_edt(~pad) <= r
    out = ndimage.distance_transform_edt(grown) > r
    return out[(slice(k, -k),) * 3]


def _through(inside: np.ndarray) -> np.ndarray:
    """Holes ringed by solid in a slice along any axis: what a hole straight through the shape leaves."""
    add = np.zeros_like(inside)
    for a in range(3):
        st = np.zeros((3, 3, 3), bool)
        sl = [slice(None)] * 3
        sl[a] = 1
        st[tuple(sl)] = ndimage.generate_binary_structure(2, 1)
        add |= ndimage.binary_fill_holes(inside, structure=st) & ~inside
    return add


def gaps(mesh: trimesh.Trimesh, size: float = 0.0, grid: int = GRID, top: int = 8) -> list[dict]:
    """What close would fill (holes through the shape, and dents under `size` when > 0), as clusters, largest
    first."""
    inside, origin, h = voxels(mesh, grid)
    added = _through(inside)
    if size > 0:
        added |= _closed(inside, size / h) & ~inside
    lab, n = ndimage.label(added)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        idx = np.argwhere(m) + [s.start for s in sl]
        p = origin + h * (idx + 0.5)
        out.append({"center": [round(float(x), 4) for x in p.mean(0)], "voxels": int(m.sum()),
                    "extent": [round(float(x), 4) for x in np.ptp(p, 0) + h]})
    out.sort(key=lambda c: -c["voxels"])
    return out[:top]


def solidify(mesh: trimesh.Trimesh, close: bool = True, size: float = 0.0, center=None, radius: float | None = None,
             carve: list[dict] | None = None, grid: int = GRID) -> tuple[trimesh.Trimesh, dict]:
    """The mesh closed (within the sphere when given) and carved, surfaced again as one closed skin."""
    inside, origin, h = voxels(mesh, grid)
    before = int(inside.sum())
    if not before:
        raise SolidError("nothing inside the mesh: it has no closed volume to fill")
    removed = 0
    for c in carve or []:
        cut = _sphere_mask(inside.shape, origin, h, c["center"], float(c["radius"])) & inside
        removed += int(cut.sum())
        inside &= ~cut
    filled = 0
    if close:
        add = _through(inside)
        if size > 0:
            add |= _closed(inside, size / h) & ~inside
        if center is not None:
            add &= _sphere_mask(inside.shape, origin, h, center, float(radius))
        filled = int(add.sum())
        inside |= add
    # a carve can leave crumbs behind: drop pieces under CRUMB of the volume (a tractor's wheels are real pieces)
    lab, n = ndimage.label(inside)
    if n > 1:
        sizes = np.asarray(ndimage.sum(inside, lab, range(1, n + 1)))
        inside = np.isin(lab, 1 + np.where(sizes >= CRUMB * sizes.sum())[0])
    # signed distance (negative inside), softened by a voxel so the surface is not a staircase
    d = (ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)) * h
    d = ndimage.gaussian_filter(d, 0.8)
    lo = origin + 0.5 * h
    shape = np.array(d.shape)

    def fn(p):
        return ndimage.map_coordinates(d, ((p - lo) / h).T, order=1, mode="nearest")

    V, F = mesher.surface_nets(fn, lo, h, shape)
    out = trimesh.Trimesh(V, F, process=True)
    out.remove_unreferenced_vertices()
    info = {"pitch": round(h, 5), "voxels_before": before, "filled": filled, "carved": removed,
            "pieces": int(n), "faces": int(len(out.faces)), "watertight": bool(out.is_watertight),
            "extents_before": [round(float(x), 4) for x in mesh.extents],
            "extents_after": [round(float(x), 4) for x in out.extents]}
    return out, info
