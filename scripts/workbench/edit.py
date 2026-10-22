"""Proportion edits for mesh_edit: thicken_legs, shorten_legs, widen_head.

Pure functions on a mesh's vertex array (numpy, scipy, trimesh; no model loads). Only vertices move, each by a
function of its position alone, so UV-seam duplicates stay together and the faces, UVs and texture are kept as they
are: the edited GLB wears the same material.

The frame is the GLB's own: glTF Y up, the floor at the lowest vertex. The body's length is the longer horizontal
axis (x or z) and its lateral axis the other one. Where the legs end and where the head starts are read from the
shape when they are not given:

- leg band: horizontal cross-sections of the surface, filled; the legs are a few small discs, the belly a big one.
  The band runs from the floor to the lowest height whose section reaches BODY_SHARE of the widest section.
- legs: the band's footprint in xz, rasterised and labelled; the sub-band with the most separate blobs that come
  down near the floor gives the seeds (legs that touch at the hooves usually part higher up, or the other way round).
  A blob that pinches in the middle (two legs pressed together) is split at the waist; one too long for a leg that
  does not pinch is refused. Fewer than two legs is refused too. Every band vertex then goes to the leg whose centre
  line (per-height centroid, smoothed) is nearest.
- head end: an end whose tip stands above the body's mid-height and is not thin (a tail is low, or thin, or both);
  with two such ends, the one that sticks out further past the outermost legs (a head and neck overhang the front
  legs; the rump barely passes the hind ones). Otherwise refused: give head.
- head start: walking from the middle of the body toward the head, the first section narrower than NECK_SHARE of the
  body's width, measured above the leg band; refused (give start) when the end never narrows below NARROW_SHARE.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

BODY_SHARE = 0.2  # a section this share of the widest one is body, not legs
NECK_SHARE = 0.8  # a section this share of the body's width is where the neck begins
WAIST_SHARE = 0.55  # a blob whose middle is narrower than this share of both sides is two legs touching
LEG_ASPECT = 1.8  # kept for callers; a blob past SPLIT_ASPECT with no waist and MERGED_SIZE times the others is legs together
LEG_ASPECT_ALONE = 2.4  # this long is legs pressed together whatever the others are
MERGED_SIZE = 1.6
MERGED_ASPECT = 1.25  # with MERGED_SIZE: two legs overlapping heavily make a blob barely longer than wide
SPLIT_ASPECT = 1.5  # only a blob this much longer than wide is tried for a waist
WAIST_GRID = 80  # raster cells along a blob's long axis for the waist test
FOOT_SHARE = 0.35
HEAD_RISE = 0.35  # a head's top stands this share of the back's height above the body's mid-height
TAIL_SECTION = 0.05  # an end whose tip is thinner than this (height share x width share of the body) is a tail
NARROW_SHARE = 0.6  # toward the head the section narrows below this share of the body's width
END_SHARE = 0.12  # the tip of an end: this share of the length  # a leg comes down within this share of the band from the floor (a raised hoof too)
GRID = 128  # raster cells along the longest horizontal side
SAMPLES = 120_000
AXES = {"+x": (0, 1.0), "-x": (0, -1.0), "+z": (2, 1.0), "-z": (2, -1.0)}


class EditError(ValueError):
    pass


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _surface(mesh: trimesh.Trimesh, n: int = SAMPLES) -> np.ndarray:
    """Area-weighted surface points: vertices bunch where triangles are small, so profiles use samples."""
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.vstack([np.asarray(pts, float), np.asarray(mesh.vertices, float)])


class _Raster:
    """An xz grid over the mesh's footprint."""

    def __init__(self, P: np.ndarray, grid: int = GRID):
        lo, hi = P[:, [0, 2]].min(0), P[:, [0, 2]].max(0)
        self.cell = float(max(hi - lo)) / grid or 1.0
        self.lo = lo - 2 * self.cell
        self.shape = tuple(int(s) for s in np.ceil((hi - self.lo) / self.cell).astype(int) + 3)

    def ij(self, P: np.ndarray) -> np.ndarray:
        return np.clip(((P[:, [0, 2]] - self.lo) / self.cell).astype(int), 0, np.array(self.shape) - 1)

    def mask(self, P: np.ndarray, fill: bool = True) -> np.ndarray:
        m = np.zeros(self.shape, bool)
        if len(P):
            ij = self.ij(P)
            m[ij[:, 0], ij[:, 1]] = True
            m = ndimage.binary_closing(ndimage.binary_dilation(m), iterations=1)
            if fill:
                m = ndimage.binary_fill_holes(m)
        return m


def section_profile(P: np.ndarray, floor: float, top: float, bins: int = 40, grid: int = GRID):
    """Filled horizontal cross-section area per height bin (as a share of the widest), and the bin centres."""
    r = _Raster(P, grid)
    edges = np.linspace(floor, top, bins + 1)
    area = np.zeros(bins)
    for k in range(bins):
        sel = (P[:, 1] >= edges[k]) & (P[:, 1] < edges[k + 1])
        area[k] = r.mask(P[sel]).sum()
    area /= max(area.max(), 1.0)
    return (edges[:-1] + edges[1:]) / 2, area


def leg_top(mesh: trimesh.Trimesh, P: np.ndarray | None = None) -> float:
    """Height (absolute y) where the legs meet the belly: the lowest section that is body-wide."""
    P = _surface(mesh) if P is None else P
    floor, top = float(P[:, 1].min()), float(P[:, 1].max())
    ys, area = section_profile(P, floor, top)
    wide = np.nonzero(area >= BODY_SHARE)[0]
    if not len(wide) or wide[0] == 0:
        raise EditError("no leg band: the shape is body-wide right down to the floor")
    return float(ys[wide[0]])


def _top_abs(mesh, P, top_frac):
    floor, ceil = float(P[:, 1].min()), float(P[:, 1].max())
    if top_frac is None:
        return floor, leg_top(mesh, P), "auto"
    return floor, floor + float(top_frac) * (ceil - floor), "given"


def _split_waist(xz: np.ndarray, depth: int = 3) -> list[tuple[np.ndarray, float]]:
    """Split one footprint blob's points (xz) where it pinches: legs that touch (or nearly) make one blob shaped like
    a figure eight. The points are rasterised again, finer (WAIST_GRID cells along the blob), so the pinch is not
    filled in by the coarse raster's closing; walking along the long axis, a column whose outline is narrower than
    WAIST_SHARE of the widest column on both sides is a waist, and each side is split again. Returns the parts, each with its
    aspect (long over short axis of its points; a hoof is up to about 2, two touching legs at least 1.7)."""
    if len(xz) < 30:
        return [(xz, 1.0)]
    c0 = xz.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((xz - c0).T))
    aspect = float(np.sqrt(evals[1] / max(evals[0], 1e-12)))
    if depth == 0 or aspect < SPLIT_ASPECT:  # a round blob is one leg (two touching are at least sqrt 3 long)
        return [(xz, aspect)]
    uv = (xz - c0) @ evecs[:, ::-1]  # long axis first
    cell = float(np.ptp(uv[:, 0])) / WAIST_GRID or 1.0
    ij = np.floor((uv - uv.min(0)) / cell).astype(int) + 2
    m = np.zeros(tuple(ij.max(0) + 3), bool)
    m[ij[:, 0], ij[:, 1]] = True
    m = ndimage.binary_dilation(m)
    # Each column's outline width (first to last cell), so a hollow or patchy outline still reads as its span.
    any_ = m.any(1)
    first = np.where(any_, m.argmax(1), 0)
    last = np.where(any_, m.shape[1] - 1 - m[:, ::-1].argmax(1), -1)
    w = ndimage.gaussian_filter1d((last - first + 1).astype(float), 1.0, mode="constant")
    span = np.nonzero(w > 0.5)[0]
    lo, hi = span[0] + int(0.2 * len(span)), span[0] + int(np.ceil(0.8 * len(span)))
    best, cut = 1.0, None
    for j in range(max(lo, 1), min(hi, len(w) - 1)):
        share = w[j] / max(min(w[:j].max(), w[j + 1:].max()), 1e-9)
        if share < best:
            best, cut = share, j
    if cut is None or best >= WAIST_SHARE:
        return [(xz, aspect)]
    return [q for side in (ij[:, 0] < cut, ij[:, 0] >= cut) for q in _split_waist(xz[side], depth - 1)]


def _merged(parts: list[tuple[np.ndarray, float]]) -> bool:
    """Whether a blob left after the waist split still looks like more than one leg: very long for a leg, or long
    and clearly bigger than the other legs."""
    sizes = np.array([float(len(q)) for q, _ in parts])
    for i, (_, aspect) in enumerate(parts):
        others = np.delete(sizes, i)
        if aspect > LEG_ASPECT_ALONE or (aspect > MERGED_ASPECT and len(others) and
                                         sizes[i] > MERGED_SIZE * float(np.median(others))):
            return True
    return False


def find_legs(P: np.ndarray, floor: float, top: float, grid: int = GRID, slabs: int = 6):
    """Seed masks (xz rasters) of each leg: the sub-band of the lower 85% of the leg band with the most blobs, a blob
    that pinches in the middle split in two. When that sub-band has a blob that still looks merged but more blobs than
    the best sub-band whose blobs all read as single legs, the clean one is taken: merging makes fewer blobs, so the
    extra ones are a leg broken up (the sheep with shortened legs: wool among the legs at the top of the band shatters
    one leg into pieces there, while the five sub-bands below agree on four clean legs). Returns the seeds; whether that sub-band has a blob too long for one leg
    that could not be split (legs pressed together along their length); and the splits, each the merged blob's mask
    with its legs' (seed index, xz centre)."""
    r = _Raster(P, grid)
    edges = np.linspace(floor, floor + 0.85 * (top - floor), slabs + 1)
    best: list[np.ndarray] = []
    best_odd, best_groups = False, []
    clean: tuple[list, list] = ([], [])  # the clean sub-band with the most blobs: (masks, groups)
    for k in range(slabs):
        sel = (P[:, 1] >= edges[k]) & (P[:, 1] < edges[k + 1])
        lab, n = ndimage.label(r.mask(P[sel]))
        if not n:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = [i + 1 for i in range(n) if sizes[i] >= 0.15 * sizes.max()]
        # A leg reaches (near) the floor; a blob that only hangs into the band (chest wool, an udder) is not one.
        keep = [i for i in keep if _reaches_floor(P, r, lab == i, floor, top)]
        S = P[sel]
        ij = r.ij(S)
        masks, parts, groups = [], [], []
        for i in keep:
            mine = lab == i
            inside = mine[ij[:, 0], ij[:, 1]]
            xz = S[inside][:, [0, 2]]
            split = _split_waist(xz)
            if len(split) > 1:
                # Each coarse cell goes to the part with most of its points.
                count = np.zeros((len(split),) + mine.shape)
                for j, (q, _) in enumerate(split):
                    qi = r.ij(np.c_[q[:, 0], np.zeros(len(q)), q[:, 1]])
                    np.add.at(count[j], (qi[:, 0], qi[:, 1]), 1)
                win = count.argmax(0)
                ms = [(win == j) & (count[j] > 0) & mine for j in range(len(split))]
                ok = [j for j, q in enumerate(ms) if q.sum() and _reaches_floor(P, r, q, floor, top)]
                if len(ok) > 1:
                    groups.append((mine, [(len(masks) + n, split[j][0].mean(0)) for n, j in enumerate(ok)]))
                    masks += [ms[j] for j in ok]
                    parts += [split[j] for j in ok]
                    continue
                split = _split_waist(xz, depth=0)
            masks.append(mine)
            parts.append(split[0])
        odd = _merged(parts)
        if not odd and len(masks) > len(clean[0]):
            clean = (masks, groups)
        if len(masks) > len(best):
            best, best_odd, best_groups = masks, odd, groups
    # Legs pressed together make fewer blobs, not more. A winning sub-band that looks merged yet has more blobs than
    # the best clean one is a leg broken into pieces (wool hanging among the legs), so the clean one stands.
    if best_odd and clean[0] and len(clean[0]) < len(best):
        return clean[0], False, clean[1]
    return best, best_odd, best_groups


def _too_few(n: int, hint: str) -> EditError:
    return EditError(f"found {n} leg-like part{'' if n == 1 else 's'}; {hint}")


def _reaches_floor(P: np.ndarray, r: "_Raster", m: np.ndarray, floor: float, top: float) -> bool:
    low = P[P[:, 1] < floor + 0.85 * (top - floor)]
    ij = r.ij(low)
    inside = low[m[ij[:, 0], ij[:, 1]], 1]
    return bool(len(inside)) and float(inside.min()) < floor + FOOT_SHARE * (top - floor)


class _Legs:
    """Each leg's centre line through the band: per-height centroids, smoothed, held flat past the ends."""

    def __init__(self, P: np.ndarray, floor: float, top: float, bins: int = 24):
        self.floor, self.top = floor, top
        self.raster = _Raster(P, GRID)
        seeds, odd, self.groups = find_legs(P, floor, top)
        if len(seeds) < 2:
            raise _too_few(len(seeds), "check the shape (it needs at least two legs standing on the floor)")
        if odd:
            raise EditError("some legs touch along their length and can't be told apart; check the shape")
        band = P[(P[:, 1] <= floor + 0.85 * (top - floor))]
        ij = self.raster.ij(band)
        lab = np.full(len(band), -1)
        for i, m in enumerate(seeds):
            lab[m[ij[:, 0], ij[:, 1]]] = i
        self.edges = np.linspace(floor, top, bins + 1)
        self.ys = (self.edges[:-1] + self.edges[1:]) / 2
        self.n = len(seeds)
        # Legs split at a waist touch: their centres are the middle of each height's extent, not the median. Along
        # the contact the outlines crowd, so a median there runs off (a centre a little off moves a crowd of points
        # to the other leg, which pushes both centres further the same way); the extent's middle moves half as far
        # as the boundary does, so refitting closes in on the contact.
        self.touching = {i for _, legs in self.groups for i, _ in legs}
        for _ in range(6 if self.touching else 3):  # seed, then let every band point join its nearest centre line
            self._fit(band, lab)
            lab = self.nearest(band)[0]

    def _fit(self, band: np.ndarray, lab: np.ndarray) -> None:
        C = np.zeros((self.n, len(self.ys), 2))
        R = np.zeros(self.n)
        for i in range(self.n):
            pts = band[lab == i]
            if not len(pts):
                raise EditError("a leg lost all its points")
            k = np.digitize(pts[:, 1], self.edges) - 1
            got = [b for b in range(len(self.ys)) if (k == b).sum() >= 3]
            if not got:
                got = [int(np.median(k))]
                cen = [np.median(pts[:, [0, 2]], axis=0)]
            else:
                cen = [np.percentile(pts[k == b][:, [0, 2]], [2, 98], axis=0).mean(0) if i in self.touching
                       else np.median(pts[k == b][:, [0, 2]], axis=0) for b in got]
            cen = np.array(cen)
            for a in range(2):
                C[i, :, a] = np.interp(self.ys, self.ys[got], cen[:, a])
            C[i] = ndimage.gaussian_filter1d(C[i], 1.5, axis=0, mode="nearest")
            d = np.linalg.norm(pts[:, [0, 2]] - self.at(C[i], pts[:, 1]), axis=1)
            R[i] = float(np.percentile(d, 90))
        self.C, self.R = C, R

    def at(self, Ci: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(y, self.ys, Ci[:, 0]), np.interp(y, self.ys, Ci[:, 1])], axis=1)

    def nearest(self, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each point: its leg, its centre-line point (xz) and its distance from it."""
        cs = np.stack([self.at(self.C[i], V[:, 1]) for i in range(self.n)])  # (n, N, 2)
        d = np.linalg.norm(V[None, :, [0, 2]] - cs, axis=2)
        lab = d.argmin(0)
        return lab, cs[lab, np.arange(len(V))], d[lab, np.arange(len(V))]

    def info(self) -> list[dict]:
        mid = len(self.ys) // 3
        return [{"centre_xz": [round(float(c), 4) for c in self.C[i, mid]], "radius": round(float(self.R[i]), 4)}
                for i in range(self.n)]


def _extents(V: np.ndarray) -> list[float]:
    return [round(float(x), 4) for x in np.ptp(V, axis=0)]


def thicken_legs(mesh: trimesh.Trimesh, factor: float, top: float | None = None, ramp: float = 0.15):
    """Scale each leg's x,z about its own centre line by factor; y is untouched, so the feet stay on the floor.
    The scale ramps to 1 over the top ramp share of the band, so the belly join is continuous. Points further than
    the leg's radius are pushed out by the edge's displacement, fading over a width that keeps the map monotonic
    (no fold), so a belly or a neighbouring part near a leg moves aside instead of being swallowed."""
    V = np.asarray(mesh.vertices, float)
    P = _surface(mesh)
    floor, ytop, how = _top_abs(mesh, P, top)
    legs = _Legs(P, floor, ytop)
    out = V.copy()
    inband = V[:, 1] < ytop
    W = V[inband]
    lab, c, d = legs.nearest(W)
    R = legs.R[lab]
    wy = 1.0 - smoothstep((W[:, 1] - (ytop - ramp * (ytop - floor))) / max(ramp * (ytop - floor), 1e-9))
    s = (factor - 1.0) * wy  # extra scale
    # Inside R: r' = r (1 + s). Outside: the edge's push s R fading to 0 over a width that keeps dr'/dr > 0.
    width = np.maximum(1.8 * s * R, 1e-9) + 0.5 * R
    push = np.where(d <= R, s * d, s * R * (1.0 - smoothstep((d - R) / width)))
    u = (W[:, [0, 2]] - c) / np.maximum(d, 1e-12)[:, None]
    W2 = W.copy()
    W2[:, [0, 2]] += u * push[:, None]
    out[inband] = W2
    info = {"op": "thicken_legs", "factor": float(factor), "top": round(float((ytop - floor) / np.ptp(P[:, 1])), 4),
            "top_y": round(float(ytop), 4), "top_from": how, "legs": legs.info(), "legs_found": legs.n,
            "verts_moved": int((np.linalg.norm(out - V, axis=1) > 1e-9).sum()),
            "extents_before": _extents(V), "extents_after": _extents(out)}
    return out, info


def _ss_integral(u: np.ndarray) -> np.ndarray:
    """Integral of smoothstep from 0 to u (u clipped to [0, 1]; past 1 it grows linearly)."""
    uc = np.clip(u, 0.0, 1.0)
    return uc ** 3 - uc ** 4 / 2 + np.maximum(u - 1.0, 0.0)


def shorten_legs(mesh: trimesh.Trimesh, factor: float, top: float | None = None, blend: float = 0.1):
    """Compress the leg band toward the floor by factor and lower everything above by the height removed. The
    squash is a smooth height map whose slope goes from factor to 1 across top +- blend * band, so there is no
    crease; above that the body moves down unchanged, by exactly (1 - factor) * leg height."""
    V = np.asarray(mesh.vertices, float)
    P = _surface(mesh)
    floor, ytop, how = _top_abs(mesh, P, top)
    if how == "auto":
        seeds, odd, _ = find_legs(P, floor, ytop)
        if len(seeds) < 2 and not odd:
            raise _too_few(len(seeds), "give top or check the shape")
    band = ytop - floor
    b = max(blend * band, 1e-9)
    a = ytop - b
    y = V[:, 1]
    # y' = floor + k (y - floor) + (1 - k) * integral_floor^y S, S the smoothstep from a to a + 2b.
    k = float(factor)
    integ = 2 * b * _ss_integral((y - a) / (2 * b))
    out = V.copy()
    out[:, 1] = floor + k * (y - floor) + (1 - k) * integ
    drop = (1 - k) * band
    info = {"op": "shorten_legs", "factor": k, "top": round(float(band / np.ptp(P[:, 1])), 4),
            "top_y": round(float(ytop), 4), "top_from": how, "lowered_by": round(float(drop), 4),
            "extents_before": _extents(V), "extents_after": _extents(out)}
    return out, info


def _length_axes(V: np.ndarray) -> tuple[int, int]:
    ext = np.ptp(V, axis=0)
    return (0, 2) if ext[0] >= ext[2] else (2, 0)


def _width_profile(P: np.ndarray, li: int, sign: float, fi: int, bins: int = 40):
    """Lateral width per slice along the length, t from rump (0) to nose (1)."""
    F = sign * P[:, fi]
    lo, hi = F.min(), F.max()
    t = (F - lo) / max(hi - lo, 1e-12)
    k = np.minimum((t * bins).astype(int), bins - 1)
    w = np.zeros(bins)
    for b in range(bins):
        sel = k == b
        if sel.sum() >= 5:
            q = np.percentile(P[sel, li], [2, 98])
            w[b] = q[1] - q[0]
    return (np.arange(bins) + 0.5) / bins, w


def _end_shape(P: np.ndarray, fi: int, li: int, above: float) -> dict:
    """The body's middle (back line, belly, width) and each end's tip, all measured above the legs. An end's rise is
    its top above the body's mid-height, as a share of the back's height above it; thick and wide are the tip's
    height and lateral spread as shares of the body's."""
    F = P[:, fi]
    t = (F - F.min()) / max(float(np.ptp(F)), 1e-12)
    up = P[:, 1] >= above
    k = np.minimum((t * 20).astype(int), 19)
    tops = [np.percentile(P[(k == b) & up, 1], 98) for b in range(7, 13) if ((k == b) & up).sum() >= 5]
    bots = [np.percentile(P[(k == b) & up, 1], 2) for b in range(7, 13) if ((k == b) & up).sum() >= 5]
    mid = (t > 0.35) & (t < 0.65) & up
    if not tops or mid.sum() < 10:
        raise EditError("no body found above the legs")
    back, belly = float(np.median(tops)), float(np.median(bots))
    width = float(np.ptp(np.percentile(P[mid, li], [2, 98])))
    half = (back + belly) / 2
    ends = {}
    for sign, sel in ((-1.0, t < END_SHARE), (1.0, t > 1 - END_SHARE)):
        Q = P[sel & up]
        if len(Q) < 10:
            ends[sign] = None
            continue
        ends[sign] = {"rise": (float(np.percentile(Q[:, 1], 98)) - half) / max(back - half, 1e-9),
                      "thick": float(np.ptp(np.percentile(Q[:, 1], [2, 98]))) / max(back - belly, 1e-9),
                      "wide": float(np.ptp(np.percentile(Q[:, li], [2, 98]))) / max(width, 1e-9)}
    return ends


def head_end(mesh: trimesh.Trimesh, P: np.ndarray | None = None) -> tuple[str, str]:
    """Which end is the head (+x/-x/+z/-z) and why. An end can be the head when its tip stands clearly above the
    body's mid-height and is not a thin appendage (a level or hanging tail is both low and thin). One such end is
    the head; with two, the one that sticks out further past the outermost legs (a head and neck overhang the front
    legs; the rump barely passes the hind ones). Anything else is refused: a shape with no head end, or ends that
    look alike (a cylinder, a ball, a box on wheels), is not guessed at."""
    P = _surface(mesh) if P is None else P
    fi, li = _length_axes(P)
    ax = "xyz"[fi]
    floor = float(P[:, 1].min())
    try:
        ytop = leg_top(mesh, P)
    except EditError:
        ytop = None
    ends = _end_shape(P, fi, li, floor if ytop is None else ytop)
    can = [sign for sign, e in ends.items()
           if e is not None and e["rise"] >= HEAD_RISE and e["thick"] * e["wide"] >= TAIL_SECTION]
    if not can:
        raise EditError("no head end found: neither end rises above the body's mid-height as a head does; give head")
    if len(can) == 1:
        other = ends[-can[0]]
        why = "the other end is a thin tail" if other and other["rise"] >= HEAD_RISE else "the other end sits low"
        return (f"+{ax}" if can[0] > 0 else f"-{ax}"), f"only end rising above the body ({why})"
    try:
        if ytop is None:
            raise EditError("no leg band")
        legs = _Legs(P, floor, ytop)
        feet = legs.C[:, 0, 0 if fi == 0 else 1]
        over_pos = P[:, fi].max() - feet.max()
        over_neg = feet.min() - P[:, fi].min()
        if abs(over_pos - over_neg) > 0.05 * np.ptp(P[:, fi]):
            return (f"+{ax}" if over_pos > over_neg else f"-{ax}"), "overhang past the outermost legs"
    except EditError:
        pass
    raise EditError(f"can't tell the head end: both ends of {ax} look alike; give head")


def head_start(P: np.ndarray, fi: int, li: int, sign: float, above: float) -> float:
    """Fraction along the length (0 rump, 1 nose) where the body narrows into the neck, measured above the legs.
    The head end has to narrow for real (below NARROW_SHARE of the body's width) before the tip; a shape that keeps
    its width (a barrel, a box) has no neck to start from."""
    Q = P[P[:, 1] >= above]
    ts, w = _width_profile(Q if len(Q) > 100 else P, li, sign, fi)
    mid = (ts >= 0.35) & (ts <= 0.6)
    body = float(np.median(w[mid])) if mid.any() else float(w.max())
    toward = (ts > 0.5) & (ts <= 0.95)
    if not toward.any() or w[toward].min() >= NARROW_SHARE * body:
        raise EditError("no neck found: the body never narrows toward the head end; give start")
    for t, x in zip(ts, w):
        if t > 0.5 and x < NECK_SHARE * body:
            return float(t)
    raise EditError("no neck found: the body never narrows toward the head end; give start")


def widen_head(mesh: trimesh.Trimesh, factor: float, start: float | None = None, ramp: float = 0.1,
               head: str | None = None):
    """Scale the lateral axis about the head's own centre line by factor, ramping in (smoothstep) from start to
    start + ramp along the length, so the neck blends. Depth and height are unchanged."""
    V = np.asarray(mesh.vertices, float)
    P = _surface(mesh)
    if head is None:
        head, why = head_end(mesh, P)
    else:
        why = "given"
    if head not in AXES:
        raise EditError(f"head {head!r}; one of {sorted(AXES)}")
    fi, sign = AXES[head]
    li = 2 if fi == 0 else 0
    if start is None:
        try:
            above = leg_top(mesh, P)
        except EditError:
            above = float(P[:, 1].min())
        start, how = head_start(P, fi, li, sign, above), "auto"
    else:
        start, how = float(start), "given"
    F = sign * P[:, fi]
    lo, hi = float(F.min()), float(F.max())
    tP = (F - lo) / (hi - lo)
    # The head's centre line: per-slice lateral median of the points past the ramp, smoothed; held flat behind it.
    bins = 20
    edges = np.linspace(min(start + ramp, 0.99), 1.0, bins + 1)
    cs, got = [], []
    for b in range(bins):
        sel = (tP >= edges[b]) & (tP <= edges[b + 1])
        if sel.sum() >= 5:
            cs.append(float(np.median(P[sel, li])))
            got.append((edges[b] + edges[b + 1]) / 2)
    if not got:
        raise EditError("nothing past the head start")
    cs = ndimage.gaussian_filter1d(np.array(cs), 1.5, mode="nearest") if len(cs) > 2 else np.array(cs)
    tV = (sign * V[:, fi] - lo) / (hi - lo)
    c = np.interp(tV, got, cs)
    w = smoothstep((tV - start) / max(ramp, 1e-9))
    out = V.copy()
    out[:, li] = c + (V[:, li] - c) * (1.0 + (factor - 1.0) * w)
    hsel = tV > start + ramp
    info = {"op": "widen_head", "factor": float(factor), "head": head, "head_from": why,
            "start": round(start, 4), "start_from": how, "ramp": float(ramp), "lateral_axis": "xyz"[li],
            "head_centre": round(float(np.median(cs)), 4),
            "head_width_before": round(float(np.ptp(V[hsel, li])) if hsel.any() else 0.0, 4),
            "head_width_after": round(float(np.ptp(out[hsel, li])) if hsel.any() else 0.0, 4),
            "verts_moved": int((w > 0).sum()),
            "extents_before": _extents(V), "extents_after": _extents(out)}
    return out, info


OPS = {"thicken_legs": thicken_legs, "shorten_legs": shorten_legs, "widen_head": widen_head}


def apply(mesh: trimesh.Trimesh, op: str, **params) -> tuple[trimesh.Trimesh, dict]:
    """Run one op; the answer is a copy of the mesh with only its vertices moved (faces, UVs, material kept)."""
    V, info = OPS[op](mesh, **params)
    out = mesh.copy()
    out.vertices = V
    return out, info
