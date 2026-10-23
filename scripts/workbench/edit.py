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


MAX_LEGS = 8  # trim_belly: more parts than this under a body are not legs (a car's wheels break into crumbs)
LEG_FOOTPRINT_MAX = 2.2  # trim_belly: a leg's footprint seen from above is at most this much longer than wide


def _footprint_aspect(mask: np.ndarray) -> float:
    """How much longer than wide a footprint (a raster mask) is, from its second moments."""
    ij = np.argwhere(mask).astype(float)
    if len(ij) < 3:
        return 1.0
    ev = np.linalg.eigvalsh(np.cov(ij.T))
    return float(np.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12)))


class _UpperLegs(_Legs):
    """_Legs seeded from the upper half of the leg band only, with no waist splitting: where the legs meet the belly
    they are plain columns, while a hoof (a heel, a toe gap) can pinch and split into parts that each reach the floor,
    which find_legs then reads as legs pressed together and refuses. trim_belly falls back to this when _Legs refuses
    that way; it only needs the legs where they meet the belly."""

    def __init__(self, P: np.ndarray, floor: float, top: float, bins: int = 24, slabs: int = 4):
        self.floor, self.top = floor, top
        self.raster = r = _Raster(P, GRID)
        self.groups, self.touching = [], set()
        edges = np.linspace(floor + 0.4 * (top - floor), floor + 0.85 * (top - floor), slabs + 1)
        seeds: list[np.ndarray] = []
        for k in range(slabs):
            sel = (P[:, 1] >= edges[k]) & (P[:, 1] < edges[k + 1])
            lab, n = ndimage.label(r.mask(P[sel]))
            if not n:
                continue
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            ms = [lab == i + 1 for i in range(n) if sizes[i] >= 0.15 * sizes.max()]
            ms = [m for m in ms if _reaches_floor(P, r, m, floor, top)]
            if len(ms) > len(seeds):
                seeds = ms
        if len(seeds) < 2:
            raise _too_few(len(seeds), "check the shape (it needs at least two legs standing on the floor)")
        self.seeds = seeds
        band = P[(P[:, 1] <= floor + 0.85 * (top - floor))]
        ij = r.ij(band)
        lab = np.full(len(band), -1)
        for i, m in enumerate(seeds):
            lab[m[ij[:, 0], ij[:, 1]]] = i
        self.edges = np.linspace(floor, top, bins + 1)
        self.ys = (self.edges[:-1] + self.edges[1:]) / 2
        self.n = len(seeds)
        for _ in range(3):
            self._fit(band, lab)
            lab = self.nearest(band)[0]


def _tighten(legs: "_Legs", P: np.ndarray, seeds: list[np.ndarray], cap: float = 1.5, iters: int = 3) -> "_Legs":
    """Refit the legs' centre lines and radii from their seed footprints, then from band points within cap radii of
    their line only. _Legs gives every band point to its nearest leg, so a keel of wool hanging between the front
    legs drags a leg's line toward it and swells its radius, and the keel then counts as leg; here it stays out."""
    band = P[P[:, 1] <= legs.floor + 0.85 * (legs.top - legs.floor)]
    ij = legs.raster.ij(band)
    lab = np.full(len(band), -1)
    for i, m in enumerate(seeds):
        lab[m[ij[:, 0], ij[:, 1]]] = i
    legs._fit(band, lab)
    for _ in range(iters):
        lab, _, d = legs.nearest(band)
        lab = np.where(d <= cap * legs.R[lab], lab, -1)
        legs._fit(band, lab)
    return legs


def _sturdy_legs(P: np.ndarray, floor: float, ytop: float) -> tuple["_Legs", str]:
    """The legs as trim_belly and tidy_legs need them: _Legs, or _UpperLegs when a hoof splits oddly (find_legs's
    waist test reads a split hoof on a shortened leg as legs pressed together), refusing parts that are too many or
    slab-shaped (wheels), then _tighten'ed so wool hanging among the legs stays out. Returns the legs and where
    they were read from."""
    try:
        legs, how = _Legs(P, floor, ytop), "leg band"
        seeds = find_legs(P, floor, ytop)[0]
    except EditError as e:
        if "touch along" not in str(e):
            raise
        legs, how = _UpperLegs(P, floor, ytop), "upper leg band (the feet split oddly)"
        seeds = legs.seeds
    # Legs stand on round footprints; a wheel seen from above is a long thin slab. Lifting "belly" between
    # wheels would bite into them, so slab-shaped parts mean this is not a creature.
    if len(seeds) > MAX_LEGS:
        raise EditError(f"found {len(seeds)} parts under it, too many for legs; check the shape")
    big = max(int(m.sum()) for m in seeds)
    flat = max(_footprint_aspect(m) for m in seeds if m.sum() >= 0.25 * big)
    if flat > LEG_FOOTPRINT_MAX:
        raise EditError(f"the parts it stands on are not leg-shaped ({flat:.1f} times longer than wide, like "
                        "wheels); check the shape")
    return _tighten(legs, P, seeds), how


BELLY_PCT = 60  # the belly line: this percentile of the underside heights between the legs
LEG_KEEP = 1.3  # trim_belly leaves points within this many leg radii of a leg's centre line alone
LEG_FREE = 1.8  # and lifts fully from this many; smoothstep between
FOOT_KEEP = (0.2, 0.4)  # trim_belly lifts nothing below the first share of floor-to-line, fully above the second
BELLY_FADE = 0.05  # share of the length over which the lift fades out toward the head and past the hind legs


def _soft_floor(y: np.ndarray, line: float, b: float) -> np.ndarray:
    """A C1 soft max(y, line): line below line - b, y above line + b, (t + 1)^2 / 4 between (t = (y - line) / b).
    Never below the line, never below y; slope 0 at the bottom of the knee and 1 at its top, so no crease."""
    t = (y - line) / b
    return np.where(t >= 1.0, y, np.where(t <= -1.0, line, line + b * (t + 1.0) ** 2 / 4.0))


def _turned(V0: np.ndarray, V1: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Faces that turned over: their normal now points against the old one (faces squashed to no area aside)."""
    T0, T1 = V0[F], V1[F]
    n0 = np.cross(T0[:, 1] - T0[:, 0], T0[:, 2] - T0[:, 0])
    n1 = np.cross(T1[:, 1] - T1[:, 0], T1[:, 2] - T1[:, 0])
    return ((n0 * n1).sum(1) < 0) & (np.linalg.norm(n1, axis=1) > 1e-12 * max(float(np.ptp(V0)), 1e-9) ** 2)


def _unfold_up(V0: np.ndarray, V1: np.ndarray, F: np.ndarray, cap: np.ndarray,
               iters: int = 30) -> tuple[np.ndarray, int, int]:
    """Lay flat every face the lift turned over: its vertices (with their UV-seam duplicates) rise toward its
    highest, each no higher than its cap (the full lift for a vertex the trim moved, where it is for one it left).
    Moving heights only keeps a face's footprint, so a face whose corners share one height faces the way it faced
    before (up or down); raising can turn a neighbour, so repeat. In the lift's fade beside a leg a wall's lower
    corner is lifted further than its upper one and stands on its head; this is where that happens. Returns the
    vertices, how many faces had turned and how many still are."""
    _, inv = np.unique(np.round(V0, 9), axis=0, return_inverse=True)
    inv = inv.ravel()
    out = V1.copy()
    bad = _turned(V0, out, F)
    first = int(bad.sum())
    for _ in range(iters):
        if not bad.any():
            break
        top = np.full(inv.max() + 1, -np.inf)
        fb = F[bad]
        np.maximum.at(top, inv[fb].ravel(), np.repeat(out[fb, 1].max(1), 3))
        out[:, 1] = np.maximum(out[:, 1], np.minimum(top[inv], cap))
        bad = _turned(V0, out, F)
    return out, first, int(bad.sum())


def _smooth_y(V: np.ndarray, F: np.ndarray, moved: np.ndarray, weight: np.ndarray, iters: int = 4,
              lam: float = 0.5) -> np.ndarray:
    """Laplacian smoothing of the height only, on the moved vertices, over the welded mesh (UV-seam duplicates share
    a position, so they are one node and move together: no seam opens). Height only, so the footprint, and with it
    every face's facing up or down, is unchanged."""
    _, first, inv = np.unique(np.round(V, 9), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    E = trimesh.geometry.faces_to_edges(inv[F])
    E = E[E[:, 0] != E[:, 1]]
    E = np.unique(np.sort(E, axis=1), axis=0)
    n = len(first)
    y = V[first, 1].copy()
    mv = np.zeros(n, bool)
    np.logical_or.at(mv, inv, moved)
    wt = np.zeros(n)
    np.maximum.at(wt, inv, weight)
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    a = lam * wt * mv * (deg > 0)
    for _ in range(iters):
        s = np.bincount(E[:, 0], y[E[:, 1]], n) + np.bincount(E[:, 1], y[E[:, 0]], n)
        y = y + a * (s / np.maximum(deg, 1.0) - y)
    out = V.copy()
    out[:, 1] = y[inv]
    return out


def _head_by_overhang(P: np.ndarray, legs: "_Legs", fi: int, li: int, above: float) -> str:
    """head_end's rule with legs already found (head_end finds its own, and refuses when _Legs does): of the ends
    that rise like a head, the one that overhangs the outermost legs more."""
    ends = _end_shape(P, fi, li, above)
    can = [sg for sg, e in ends.items() if e is not None and e["rise"] >= HEAD_RISE and e["thick"] * e["wide"] >= TAIL_SECTION]
    ax = "xyz"[fi]
    if len(can) == 1:
        return f"+{ax}" if can[0] > 0 else f"-{ax}"
    if not can:
        raise EditError("no head end")
    feet = legs.C[:, 0, 0 if fi == 0 else 1]
    over_pos, over_neg = P[:, fi].max() - feet.max(), feet.min() - P[:, fi].min()
    if abs(over_pos - over_neg) <= 0.05 * np.ptp(P[:, fi]):
        raise EditError("ends look alike")
    return f"+{ax}" if over_pos > over_neg else f"-{ax}"


def _along_window(mesh, P: np.ndarray, legs: "_Legs | None", fi: int, li: int, above: float):
    """Weight along the length (1 where the belly may be trimmed): from the hind legs (outermost centre less 2R) to
    where the neck starts, fading over BELLY_FADE of the length; the head and what hangs past the hind legs are left.
    With no head end found, the front side ends past the front legs the same way as the hind side."""
    F = P[:, fi]
    lo, hi = float(F.min()), float(F.max())
    length = max(hi - lo, 1e-9)
    fade = BELLY_FADE * length
    a = 0 if fi == 0 else 1  # the leg centre lines are (x, z)
    try:
        try:
            head, _ = head_end(mesh, P)
        except EditError:
            if legs is None:
                raise
            head = _head_by_overhang(P, legs, fi, li, above)
        hfi, sign = AXES[head]
        if hfi != fi:
            raise EditError("head across the body")
        t0 = head_start(P, fi, li, sign, above)
        front = (lo + t0 * length) if sign > 0 else (hi - t0 * length)
        why = f"head {head}"
    except EditError:
        sign, front, why = 1.0, None, "no head"
    if legs is None:
        return (lambda x: np.ones_like(x)), {"along": "whole length (no legs)"}
    feet = legs.C[:, 0, a]
    R = float(legs.R.max())
    s = sign
    back = (feet.min() - 2 * R) if s > 0 else (feet.max() + 2 * R)  # the hind side, in world units
    if front is None:
        front = (feet.max() + 2 * R) if s > 0 else (feet.min() - 2 * R)

    def w(x: np.ndarray) -> np.ndarray:
        u = s * x
        f, bk = s * front, s * back
        return smoothstep((u - (bk - fade)) / fade) * (1.0 - smoothstep((u - f) / fade))

    return w, {"along": why, "front": round(float(front), 4), "back": round(float(back), 4)}


def belly_line(P: np.ndarray, legs: "_Legs", fi: int, li: int) -> float:
    """The belly's underside height between the legs: per xz cell, the lowest surface point clear of the legs (more
    than LEG_FREE radii from every centre line), over the rectangle the leg centres span; BELLY_PCT of those. Spikes
    and a keel are a minority of the cells, so the line sits on the belly itself, not on their tips."""
    a = 0 if fi == 0 else 1
    feet = legs.C[:, 0, :]
    lab, _, d = legs.nearest(P)
    clear = d > LEG_FREE * legs.R[lab]
    inside = ((P[:, fi] >= feet[:, a].min()) & (P[:, fi] <= feet[:, a].max()) &
              (P[:, li] >= feet[:, 1 - a].min()) & (P[:, li] <= feet[:, 1 - a].max()))
    Q = P[clear & inside & (P[:, 1] > legs.floor + 0.05 * (legs.top - legs.floor))]
    if len(Q) < 20:
        raise EditError("no belly between the legs to read a line from; give line")
    r = _Raster(P, GRID)
    ij = r.ij(Q)
    key = ij[:, 0] * r.shape[1] + ij[:, 1]
    order = np.lexsort((Q[:, 1], key))
    firsts = order[np.r_[True, key[order][1:] != key[order][:-1]]]
    return float(np.percentile(Q[firsts, 1], BELLY_PCT))


def trim_belly(mesh: trimesh.Trimesh, line: float | None = None, blend: float = 0.03, keep_legs: bool = True):
    """Lift what hangs under the belly (a keel of wool between the front legs, spikes along the underside) up to a
    line, leaving the legs, head and tail. Every vertex goes through a C1 soft max with the line (_soft_floor, knee
    +- blend of the height), weighted by its distance from the nearest leg's centre line (0 within LEG_KEEP radii, 1
    from LEG_FREE) and by its place along the length (_along_window); then a few height-only Laplacian passes on the
    moved vertices round off the plateaus, and what was fully lifted is held at or above the line. Only heights
    change, so faces, UVs and texture are kept and no face turns over.
    The line (a share of the height) defaults to the higher of the leg band's top and belly_line."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    P = _surface(mesh)
    floor, ceil = float(P[:, 1].min()), float(P[:, 1].max())
    h = max(ceil - floor, 1e-9)
    fi, li = _length_axes(P)
    info = {"op": "trim_belly"}
    try:
        ytop = leg_top(mesh, P)
        legs, info["legs_from"] = _sturdy_legs(P, floor, ytop)
    except EditError as e:
        if line is None:
            raise EditError(f"{e}; give line") from None
        ytop, legs = None, None
    if line is None:
        under = belly_line(P, legs, fi, li)
        yl = max(ytop, under)
        info.update(line_from="belly underside" if under >= ytop else "leg band top",
                    leg_top=round((ytop - floor) / h, 4), underside=round((under - floor) / h, 4))
    else:
        yl = floor + float(line) * h
        info["line_from"] = "given"
    b = max(float(blend) * h, 1e-9)
    w_along, along = _along_window(mesh, P, legs, fi, li, ytop if ytop is not None else floor)
    if along.get("along") == "no head" and line is None:
        # A creature has a head; with none, the "legs" are likely wheels or a table's, and the "belly" a chassis.
        raise EditError("found legs but no head, so this may not be a creature; give line")
    w = w_along(V[:, fi])
    if legs is not None and keep_legs:
        lab, _, d = legs.nearest(V)
        R = legs.R[lab]
        w = w * smoothstep((d - LEG_KEEP * R) / ((LEG_FREE - LEG_KEEP) * R))
    # Feet splay past their leg's radius (a toe, a heel) but sit on the floor: what hangs from the belly ends well
    # above it. A weight rising with height keeps the map monotonic in y (y + w(y) (line - y), w' >= 0).
    if legs is not None:
        k0, k1 = (floor + f * (yl - floor) for f in FOOT_KEEP)
        w = w * smoothstep((V[:, 1] - k0) / max(k1 - k0, 1e-9))
    out = V.copy()
    out[:, 1] = V[:, 1] + w * (_soft_floor(V[:, 1], yl, b) - V[:, 1])
    moved = np.abs(out[:, 1] - V[:, 1]) > 1e-9
    if moved.any():
        out = _smooth_y(out, F, moved, w)
        full = moved & (w >= 1.0 - 1e-9)
        out[full, 1] = np.maximum(out[full, 1], yl)
        out[:, 1] = np.maximum(out[:, 1], V[:, 1])  # a trim only lifts
        out[~moved] = V[~moved]
    cap = np.where(np.abs(out[:, 1] - V[:, 1]) > 1e-9, np.maximum(_soft_floor(V[:, 1], yl, b), out[:, 1]), out[:, 1])
    out, turned, left = _unfold_up(V, out, F, cap)
    info.update(turned_laid_flat=turned, turned_left=left)
    lift = out[:, 1] - V[:, 1]
    info.update({"line": round((yl - floor) / h, 4), "line_y": round(yl, 4), "blend": float(blend),
                 "keep_legs": bool(keep_legs), "legs": legs.info() if legs is not None else [],
                 "legs_found": legs.n if legs is not None else 0, **along,
                 "verts_moved": int((np.abs(lift) > 1e-9).sum()), "max_lift": round(float(lift.max(initial=0.0)), 4),
                 "max_lift_share": round(float(lift.max(initial=0.0)) / h, 4),
                 "extents_before": _extents(V), "extents_after": _extents(out)})
    return out, info


TIDY_BINS = 20  # tidy_legs: height bins over the leg band for each leg's centre and radius
TIDY_FIT = 0.85  # the centre line and radius are fitted below this share of the band (above, the belly crowds in)
TIDY_BLEND = 0.15  # the move fades out over this top share of the band, so the leg joins the belly with no crease
TIDY_CAP = 1.5  # surface points within this many leg radii of a leg's centre line are measured as that leg
HOOF = 0.15  # the hoof: the bottom share of the band, where the radius may flare
HOOF_FLARE = 0.2  # by at most this share over the shank's taper
TIDY_REACH = (1.3, 1.9)  # a vertex moves fully within the first many leg radii of its centre line, not past the second
TAUBIN = (0.5, -0.53)  # lambda, mu: smoothing that does not shrink
FLAT_NY = (0.5, 0.85)  # Taubin fades out from walls (|normal y| below the first) to flat faces (above the second)
TIDY_ITERS = 10  # Taubin passes at smooth 1


def _welded(V: np.ndarray, F: np.ndarray):
    """UV-seam duplicates welded by position: (a representative per node, node per vertex, unique node edges)."""
    _, first, inv = np.unique(np.round(V, 9), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    E = trimesh.geometry.faces_to_edges(inv[F])
    E = E[E[:, 0] != E[:, 1]]
    return first, inv, np.unique(np.sort(E, axis=1), axis=0)


def _taubin_xz(V: np.ndarray, F: np.ndarray, weight: np.ndarray, iters: int) -> np.ndarray:
    """Taubin (lambda/mu) smoothing of x and z only, each vertex's step scaled by its weight, over the welded mesh
    (UV-seam duplicates are one node and move together, so no seam opens). Heights stay, so feet stay on the floor;
    vertices of weight 0 are a fixed boundary and are returned exactly as they were."""
    if iters <= 0 or not (weight > 0).any():
        return V.copy()
    first, inv, E = _welded(V, F)
    n = len(first)
    X = V[first][:, [0, 2]].copy()
    wt = np.zeros(n)
    np.maximum.at(wt, inv, weight)
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    wt *= deg > 0
    for _ in range(iters):
        for f in TAUBIN:
            s = np.stack([np.bincount(E[:, 0], X[E[:, 1], a], n) + np.bincount(E[:, 1], X[E[:, 0], a], n)
                          for a in range(2)], axis=1)
            X = X + (f * wt)[:, None] * (s / np.maximum(deg, 1.0)[:, None] - X)
    out = V.copy()
    moved = wt[inv] > 0
    out[np.ix_(moved, [0, 2])] = X[inv[moved]]
    return out


def _leg_profile(pts: np.ndarray, floor: float, top: float, bins: int = TIDY_BINS):
    """One leg's surface points binned by height: bin heights, the centroid (xz) and mean radius per bin, filled
    across empty bins, and the point count per bin (the fits' weights)."""
    edges = np.linspace(floor, top, bins + 1)
    ys = (edges[:-1] + edges[1:]) / 2
    k = np.clip(np.digitize(pts[:, 1], edges) - 1, 0, bins - 1)
    n = np.bincount(k, minlength=bins).astype(float)
    got = np.nonzero(n >= 3)[0]
    if len(got) < 4:
        raise EditError("a leg has too few points to measure; check the shape")
    C = np.array([pts[k == b][:, [0, 2]].mean(0) for b in got])
    r = np.array([np.linalg.norm(pts[k == b][:, [0, 2]] - C[j], axis=1).mean() for j, b in enumerate(got)])
    Cf = np.stack([np.interp(ys, ys[got], C[:, a]) for a in range(2)], axis=1)
    return ys, Cf, np.interp(ys, ys[got], r), np.where(n >= 3, n, 0.0)


def _fit(t: np.ndarray, y: np.ndarray, w: np.ndarray, deg: int) -> np.ndarray:
    """Weighted least-squares polynomial coefficients of y (one or more columns) in t."""
    A = np.vander(t, deg + 1)
    sw = np.sqrt(w)[:, None]
    return np.linalg.lstsq(A * sw, (y.reshape(len(t), -1)) * sw, rcond=None)[0]


def _radius_at(taper: np.ndarray, flare: float, t: np.ndarray) -> np.ndarray:
    """The target radius at band share t: the straight taper, flared by up to `flare` toward the floor below HOOF."""
    return np.polyval(taper, t) * (1.0 + flare * (1.0 - smoothstep(t / HOOF)))


def _leg_shape(ys: np.ndarray, C: np.ndarray, r: np.ndarray, n: np.ndarray, floor: float, band: float) -> dict:
    """A leg's straight centre line (a line through its centroids: the slant stays, the zig-zag goes) and smooth
    radius profile (a quadratic through its radii: fuller at the top, the slim cannon, the hoof's flare), each
    fitted below TIDY_FIT of the band; and how far the leg strays from them (zig-zag, lumps), as shares of its
    radius, and its slant from vertical in degrees."""
    t = (ys - floor) / band
    w = n * (t <= TIDY_FIT)
    cl = _fit(t, C, w, 1)
    sel = w > 0
    # Radius: a straight taper over the shank (above the hoof), never fuller at the bottom than the top, and a
    # flare at the hoof of at most HOOF_FLARE, read from the hoof's own radius.
    shank = w * (t >= HOOF)
    rl = _fit(t, r, shank if (shank > 0).sum() >= 2 else w, 1)[:, 0]
    rl[0] = max(rl[0], 0.0)
    base = np.polyval(rl, t)
    hoof = sel & (t < HOOF)
    flare = float(np.clip(np.average(r[hoof] / np.maximum(base[hoof], 1e-9), weights=w[hoof]) - 1.0, 0.0,
                          HOOF_FLARE)) if hoof.any() else 0.0
    rfit = _radius_at(rl, flare, t)
    Cfit = np.vander(t, 2) @ cl
    rm = float(np.average(r[sel], weights=w[sel]))
    return {"line": cl, "taper": rl, "flare": flare, "rmin": float(r[sel].min()), "rmax": float(r[sel].max()),
            "zigzag": float(np.sqrt(np.average(np.sum((C - Cfit) ** 2, 1)[sel], weights=w[sel]))) / rm,
            "lumps": float(np.sqrt(np.average(((r - rfit) ** 2)[sel], weights=w[sel]))) / rm,
            "slant": float(np.degrees(np.arctan2(np.linalg.norm(cl[0]), band))), "mean_radius": rm}


def _legs_measured(P: np.ndarray, legs: "_Legs", floor: float, top: float, cap: float = TIDY_CAP):
    """Each leg's profile and fitted shape from surface points within cap leg radii of its centre line."""
    band = top - floor
    Q = P[P[:, 1] < top]
    lab, _, d = legs.nearest(Q)
    out = []
    for i in range(legs.n):
        prof = _leg_profile(Q[(lab == i) & (d <= cap * legs.R[i])], floor, top)
        out.append((prof, _leg_shape(*prof, floor, band)))
    return out


def tidy_legs(mesh: trimesh.Trimesh, thickness: float = 1.0, straighten: float = 1.0, smooth: float = 1.0,
              top: float | None = None):
    """Give each leg a straight centre line and a smooth radius profile, keeping it part of the mesh. Per leg, its
    surface's centroid c(y) and mean radius r(y) per height; the target centre line is a straight line fitted
    through c(y) (the leg's slant stays, the zig-zag goes), the target radius a quadratic fitted to r(y) (fuller at
    the top, slimmer at the cannon, the hoof's flare) times thickness. Each leg vertex moves in x,z only:
    xz' = c'(y) + (xz - c(y)) r'(y) / r(y), with c' = c + straighten (line - c) and r' = thickness (smooth fit +
    (1 - smooth) r). The move fades out over the top TIDY_BLEND of the band (so the leg joins the belly without a
    crease) and with distance from the leg's centre line (TIDY_REACH radii: the belly and a neighbouring part stay).
    Then Taubin smoothing of x,z on the leg vertices (UV-seam duplicates welded) takes off the lumps left. Heights
    never change, so the feet stay on the floor; faces, UVs and texture are kept."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    P = _surface(mesh)
    floor, ytop, how = _top_abs(mesh, P, top)
    band = max(ytop - floor, 1e-9)
    legs, legs_from = _sturdy_legs(P, floor, ytop)
    shapes = _legs_measured(P, legs, floor, ytop)
    out = V.copy()
    inband = np.nonzero(V[:, 1] < ytop)[0]
    W = V[inband]
    lab, _, d = legs.nearest(W)
    R = legs.R[lab]
    wy = 1.0 - smoothstep((W[:, 1] - (ytop - TIDY_BLEND * band)) / (TIDY_BLEND * band))
    wr = 1.0 - smoothstep((d - TIDY_REACH[0] * R) / ((TIDY_REACH[1] - TIDY_REACH[0]) * R))
    w = wy * wr
    t = (W[:, 1] - floor) / band
    weight = np.zeros(len(V))
    for i, ((ys, C, r, _), shape) in enumerate(shapes):
        m = lab == i
        if not m.any():
            continue
        y = W[m, 1]
        c = np.stack([np.interp(y, ys, C[:, a]) for a in range(2)], axis=1)
        rr = np.maximum(np.interp(y, ys, r), 1e-9)
        line = np.vander(t[m], 2) @ shape["line"]
        rfit = np.clip(_radius_at(shape["taper"], shape["flare"], t[m]), 0.6 * shape["rmin"], 1.2 * shape["rmax"])
        c2 = c + float(straighten) * (line - c)
        r2 = float(thickness) * (float(smooth) * rfit + (1.0 - float(smooth)) * rr)
        target = c2 + (W[m][:, [0, 2]] - c) * (r2 / rr)[:, None]
        xz = W[m][:, [0, 2]]
        out[inband[m][:, None], [0, 2]] = xz + w[m][:, None] * (target - xz)
    weight[inband] = w
    # Smoothing in x,z moves a face lying flat (the hoof's sole, the belly over the leg's top) within its own plane,
    # where it turns over easily; it is meant for the leg's walls, so it fades on faces that lie flat.
    ny = np.abs(np.asarray(trimesh.Trimesh(V, F, process=False).vertex_normals)[:, 1])
    flat = 1.0 - smoothstep((ny - FLAT_NY[0]) / (FLAT_NY[1] - FLAT_NY[0]))
    out = _taubin_xz(out, F, weight * flat * float(smooth), int(round(TIDY_ITERS * float(smooth))))
    turned = int(_turned(V, out, F).sum())
    after = _legs_measured(_surface(trimesh.Trimesh(out, F, process=False)), legs, floor, ytop,
                           cap=TIDY_CAP * max(float(thickness), 1.0))

    def rprof(shape, scale=1.0):
        u = np.array([0.0, 0.25, 0.5, 0.75])
        return [round(float(x) * scale, 4) for x in _radius_at(shape["taper"], shape["flare"], u)]

    per = []
    for i, ((_, b), (_, a)) in enumerate(zip(shapes, after)):
        per.append({"centre_xz": legs.info()[i]["centre_xz"],
                    "slant_before": round(b["slant"], 2), "slant_after": round(a["slant"], 2),
                    "zigzag_before": round(b["zigzag"], 4), "zigzag_after": round(a["zigzag"], 4),
                    "lumps_before": round(b["lumps"], 4), "lumps_after": round(a["lumps"], 4),
                    "radius_before": round(b["mean_radius"], 4), "radius_after": round(a["mean_radius"], 4),
                    "radius_profile": rprof(b, float(thickness)), "hoof_flare": round(b["flare"], 3)})
    info = {"op": "tidy_legs", "thickness": float(thickness), "straighten": float(straighten),
            "smooth": float(smooth), "top": round(float(band / np.ptp(P[:, 1])), 4), "top_y": round(float(ytop), 4),
            "top_from": how, "legs_from": legs_from, "legs_found": legs.n, "legs": per,
            "radius_profile_at": [0.0, 0.25, 0.5, 0.75],
            "verts_moved": int((np.linalg.norm(out - V, axis=1) > 1e-9).sum()), "turned_faces": turned,
            "extents_before": _extents(V), "extents_after": _extents(out)}
    return out, info


OPS = {"thicken_legs": thicken_legs, "shorten_legs": shorten_legs, "widen_head": widen_head,
       "trim_belly": trim_belly, "tidy_legs": tidy_legs}


def apply(mesh: trimesh.Trimesh, op: str, **params) -> tuple[trimesh.Trimesh, dict]:
    """Run one op; the answer is a copy of the mesh with only its vertices moved (faces, UVs, material kept)."""
    V, info = OPS[op](mesh, **params)
    out = mesh.copy()
    out.vertices = V
    return out, info
