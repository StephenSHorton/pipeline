"""retexture --select: one picture per triangle, colours evened out, seams levelled (photogrammetry texturing).

The plain bake averages every view wherever it lands. Drawn views never agree exactly (a three-quarter head on a
side view, each panel lit its own way), so the average shows as patches, double images and hard edges (Stephen on
the pig: "very obvious we plastered different angles of the pig images onto the 3D model"). This is the method
photogrammetry uses instead (Waechter, Moehrle and Goesele, "Let there be color!", 2014):

1. observe: which surface samples each view sees (a depth test, not first-come pixels), with a confidence from the
   viewing angle, the distance to the cut-out's edge (where a fit is off first) and the camera fit's IoU.
2. label_faces: each triangle takes the view that sees it best; a smoothness term (Potts, over shared edges) keeps
   neighbours on the same view so the seams are few and fall where views agree. The base texture (Hunyuan's paint)
   is one more label with a small constant score: it takes what no view sees well (under the belly, between legs).
3. solve_gains: one colour gain per view and channel, from triangles two sources both see, so the panels and the
   paint come to the same brightness and tint (the drawn views anchor the solve; the paint follows them).
4. level_seams: per vertex and label an additive offset, least squares: the two sides of a seam meet, offsets
   change smoothly inside each label's patch (the global adjustment of the paper).

bake() returns per-sample colours ready to splat and the base texture with the paint's gain applied (its gutters).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import splu

ANGLE_MIN = 0.12       # n.v below this is grazing: not seen
ANGLE_FULL = 0.55      # n.v above ANGLE_MIN + this counts fully
EDGE_FRAC = 0.025      # confidence ramps up over this fraction of the photo's longer side from the cut-out's edge
EDGE_FLOOR = 0.15
BASE_SCORE = 0.06      # the base texture's constant score: wins only where no view sees well
SMOOTH = 0.06          # Potts cost per neighbour on another label
ICM_ROUNDS = 12
SEAM_SMOOTH = 0.15     # weight of offset smoothness inside a patch against seam agreement (1)
MIN_SHARE = 0.15       # a face counts for the gain solve when both sources score at least this
GAIN_CAP = 0.15        # a drawn view's gain stays within 1 +- this: views of one sheet differ little, more is misfit
CONSIST_SAMPLE = 50.0   # a sample whose tone (0..255 lightness and redness) is this far from the paint's is a
                        # feature the mesh does not have there (a drawn eye or ear off its mesh position)
CONSIST_FRAC = 0.1      # a face whose samples are this fraction such outliers starts to lose the view
CONSIST_SPAN = 0.3      # ...and is down to CONSIST_FLOOR at CONSIST_FRAC + this
CONSIST_FLOOR = 0.05
CONSIST_DROP = 1        # outlier samples leave the view entirely (0: only its face scores drop)
CONSIST_GROW = 6        # rings of faces a penalty spreads over (fading 10% a ring)
HEAD_SHARE = 0.45       # a side-on creature's head: this front share of its length...
HEAD_ABOVE = 0.35       # ...above this share of its height (the legs below are the side views')
RATIO_CAP = 0.5         # the paint's colour moves at most e^0.5 (x1.65) toward the views'
RATIO_SMOOTH = 8       # rounds of neighbour averaging on the view/paint colour ratio before it is spread


def merged_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Faces over vertices merged by position (UV seams split them; the surface is one piece)."""
    key = np.round(vertices / (np.ptp(vertices, axis=0).max() * 1e-5 + 1e-12)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    return inv[faces], int(inv.max()) + 1


def face_adjacency(faces: np.ndarray) -> sparse.csr_matrix:
    """F x F, 1 where two faces share an edge."""
    nf = len(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    fid = np.tile(np.arange(nf), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[order], fid[order]
    same = (e[1:] == e[:-1]).all(axis=1)
    a, b = fid[:-1][same], fid[1:][same]
    m = sparse.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(nf, nf)).tocsr()
    m.data[:] = 1.0
    return m


def observe(pts, nrm, px, py, depth, inb, eye, alpha_bake, tol):
    """(visible mask, n.v) for one view: in the bake mask, facing the camera, and at the front of a depth buffer
    built from the samples themselves (half resolution, min-filtered so sparse samples leave no holes)."""
    ph, pw = alpha_bake.shape
    view_dir = eye - pts
    view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8
    ndotv = np.einsum("ij,ij->i", nrm, view_dir)
    cand = inb & (ndotv > 0)
    hy, hx = (ph + 1) // 2, (pw + 1) // 2
    zb = np.full(hy * hx, np.inf, np.float32)
    cell = (py[cand] // 2) * hx + px[cand] // 2
    np.minimum.at(zb, cell, depth[cand])
    zb = ndimage.minimum_filter(zb.reshape(hy, hx), size=3).reshape(-1)
    seen = np.zeros(len(pts), bool)
    c = np.where(cand)[0]
    seen[c] = depth[c] <= zb[(py[c] // 2) * hx + px[c] // 2] + tol
    seen &= ndotv > ANGLE_MIN
    seen &= alpha_bake[np.clip(py, 0, ph - 1), np.clip(px, 0, pw - 1)]
    return seen, ndotv


def confidence(ndotv, px, py, alpha_bake, quality, trust=None):
    """Per-sample confidence: viewing angle x distance from the cut-out's edge x the camera fit's quality."""
    ph, pw = alpha_bake.shape
    ang = np.clip((ndotv - ANGLE_MIN) / ANGLE_FULL, 0.0, 1.0)
    dist = ndimage.distance_transform_edt(alpha_bake)
    ramp = max(1.0, EDGE_FRAC * max(ph, pw))
    edge = np.clip(dist / ramp, EDGE_FLOOR, 1.0)[np.clip(py, 0, ph - 1), np.clip(px, 0, pw - 1)]
    conf = ang * edge * quality
    if trust is not None:
        conf = conf * trust[np.clip(py, 0, ph - 1), np.clip(px, 0, pw - 1)]
    return conf.astype(np.float32)


def fit_quality(iou: float) -> float:
    """A camera fit's IoU as a weight: 0.95 is a clean fit, 0.75 a view the shape does not quite match."""
    return float(np.clip((iou - 0.55) / 0.4, 0.1, 1.0))


def face_scores(fid, nf, seen, conf):
    """Mean confidence over each face's samples (unseen samples count as 0, so a half-hidden face scores less)."""
    count = np.bincount(fid, minlength=nf).astype(np.float32)
    total = np.bincount(fid, weights=np.where(seen, conf, 0.0), minlength=nf).astype(np.float32)
    return total / np.maximum(count, 1.0)


def label_faces(scores: np.ndarray, adj: sparse.csr_matrix, smooth: float = SMOOTH,
                rounds: int = ICM_ROUNDS) -> np.ndarray:
    """F x K scores (the last column the base) -> one label per face. Iterated conditional modes on
    -score + smooth * (neighbours on another label); a view that does not see a face is never its label."""
    nf, k = scores.shape
    allowed = scores > 0
    allowed[:, -1] = True
    label = np.argmax(np.where(allowed, scores, -1.0), axis=1)
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        onehot = sparse.csr_matrix((np.ones(nf), (np.arange(nf), label)), shape=(nf, k))
        agree = (adj @ onehot).toarray()
        cost = -scores + smooth * (deg[:, None] - agree)
        cost[~allowed] = np.inf
        best = np.argmin(cost, axis=1)
        move = (best != label) & (rng.random(nf) < 0.5)  # half at a time: synchronous updates oscillate
        if not move.any():
            break
        label[move] = best[move]
    return label


def _tone_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in lightness and redness (r - g): what an out-of-place ear or eye changes; a tint shift less."""
    la, lb = a.mean(axis=1), b.mean(axis=1)
    ra, rb = a[:, 0] - a[:, 1], b[:, 0] - b[:, 1]
    return np.hypot(la - lb, ra - rb)


def solve_gains(means: np.ndarray, scores: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """K x 3 multiplicative gains. means F x K x 3 (each source's mean colour over a face, nan where unseen),
    scores F x K. Least squares on log colour over faces two sources both see well; the anchor sources' log gains
    are held near 0 (their mean colour is the target), the rest are free."""
    nf, k, _ = means.shape
    logm = np.log(np.clip(means, 1.0, None))
    ata = np.zeros((k, k))
    atb = np.zeros((k, 3))
    for a in range(k):
        for b in range(a + 1, k):
            both = (scores[:, a] >= MIN_SHARE) & (scores[:, b] >= MIN_SHARE) if b < k - 1 else \
                (scores[:, a] >= MIN_SHARE)
            both &= np.isfinite(logm[:, a, 0]) & np.isfinite(logm[:, b, 0])
            if both.sum() < 20:
                continue
            w = np.minimum(scores[both, a], np.maximum(scores[both, b], MIN_SHARE))
            d = (logm[both, b] - logm[both, a])  # g_a - g_b should equal this
            sw = float(w.sum())
            ata[a, a] += sw
            ata[b, b] += sw
            ata[a, b] -= sw
            ata[b, a] -= sw
            atb[a] += (w[:, None] * d).sum(0)
            atb[b] -= (w[:, None] * d).sum(0)
    # every gain is pulled gently toward 1; the anchors' mean log gain is held at 0 (their average colour is the
    # target), so the drawn views even out among themselves and the free sources (the base) follow them
    scale = max(1.0, ata.trace() / max(k, 1))
    one = anchors.astype(float)
    g = np.linalg.solve(ata + np.eye(k) * scale * 0.01 + np.outer(one, one) * scale * 10.0, atb)
    cap = np.where(anchors, np.log1p(GAIN_CAP), 0.7)
    return np.exp(np.clip(g, -cap[:, None], cap[:, None]))


def vertex_adjacency(mfaces: np.ndarray, nv: int) -> sparse.csr_matrix:
    e = np.concatenate([mfaces[:, [0, 1]], mfaces[:, [1, 2]], mfaces[:, [2, 0]]])
    m = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(nv, nv)).tocsr()
    m.data[:] = 1.0
    return m


def harmonic_fill(adj: sparse.csr_matrix, value: np.ndarray, known: np.ndarray) -> np.ndarray:
    """value (nv x c) kept where known, the rest solved so each unknown vertex is the mean of its neighbours
    (Laplace interpolation over the surface): what the views say spreads smoothly into what they never saw."""
    nv = adj.shape[0]
    unk = np.where(~known)[0]
    out = value.copy()
    if len(unk) == 0 or not known.any():
        return out
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    lap = (sparse.diags(deg) - adj).tocsr()
    l_uu = lap[unk][:, unk].tocsc() + sparse.identity(len(unk)) * 1e-6
    l_uk = lap[unk][:, np.where(known)[0]]
    lu = splu(l_uu)
    for ch in range(value.shape[1]):
        out[unk, ch] = lu.solve(-(l_uk @ value[known, ch]))
    return out


def match_base(mfaces, nv, label, fid, bary, colour, base_col, k, used=None):
    """Per-sample gain for the base texture (N x 3): the ratio of the views' colour to the paint's, measured per
    vertex where views own the faces, smoothed, and spread over the faces no view owns. The paint keeps its detail
    and takes the views' local colour (a darker flank stays darker, the pink snout stays pink)."""
    corner = mfaces.reshape(-1)
    w = bary.reshape(-1, 3)
    vs = mfaces[fid]
    # only samples a view actually painted (used): a dropped outlier on a view's face carries no view colour
    view_s = (label[fid] != k - 1) if used is None else (used != k - 1)
    num = np.zeros((nv, 3))
    den = np.zeros((nv, 3))
    cnt = np.zeros(nv)
    for c in range(3):
        ww = np.where(view_s, w[:, c], 0.0)
        np.add.at(cnt, vs[:, c], ww)
        for ch in range(3):
            np.add.at(num[:, ch], vs[:, c], ww * colour[:, ch])
            np.add.at(den[:, ch], vs[:, c], ww * base_col[:, ch])
    known = cnt > 0.5
    ratio = np.ones((nv, 3))
    ratio[known] = np.log(np.clip(num[known], 1, None) / np.clip(den[known], 1, None))
    ratio[~known] = 0.0
    ratio = np.clip(ratio, -RATIO_CAP, RATIO_CAP)  # before smoothing: one stray dark pixel must not spread a smudge
    adj = vertex_adjacency(mfaces, nv)
    deg = np.maximum(np.asarray(adj.sum(axis=1)).reshape(-1), 1)
    for _ in range(int(RATIO_SMOOTH)):  # low-pass where known: the ratio carries colour, not the views' detail
        sm = (adj @ ratio + ratio) / (deg + 1)[:, None]
        ratio[known] = sm[known]
    ratio = harmonic_fill(adj, ratio, known)
    ratio = np.clip(ratio, -RATIO_CAP, RATIO_CAP)
    return np.exp((ratio[vs] * w[:, :, None]).sum(axis=1))


def level_seams(mfaces: np.ndarray, nv: int, flabel: np.ndarray, fid: np.ndarray, bary: np.ndarray,
                colour: np.ndarray, smooth: float = SEAM_SMOOTH) -> np.ndarray:
    """Per-sample additive offsets (N x 3): unknowns o[v, l] for each vertex v and label l of a face around it;
    seam vertices pull the labels' colours there to their mean, edges inside a patch keep offsets alike."""
    nf = len(mfaces)
    corner_v = mfaces.reshape(-1)
    corner_l = np.repeat(flabel, 3)
    key = corner_v.astype(np.int64) * (flabel.max() + 1) + corner_l
    ukey, var_of_corner = np.unique(key, return_inverse=True)
    var_of_corner = var_of_corner.reshape(nf, 3)
    nvar = len(ukey)
    uv_vert = ukey // (flabel.max() + 1)

    # Each label's colour at each (vertex, label): samples on that label's faces, weighted by their barycentric
    # weight toward the vertex.
    w = bary.astype(np.float64)
    vars_s = var_of_corner[fid]                      # N x 3
    acc = np.zeros((nvar, 3))
    wsum = np.zeros(nvar)
    for c in range(3):
        np.add.at(wsum, vars_s[:, c], w[:, c])
        for ch in range(3):
            np.add.at(acc[:, ch], vars_s[:, c], w[:, c] * colour[:, ch])
    known = wsum > 1e-6
    col = np.where(known[:, None], acc / np.maximum(wsum, 1e-9)[:, None], 0.0)

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    # seams: every pair of labels at a vertex should meet
    order = np.argsort(uv_vert, kind="stable")
    sv = uv_vert[order]
    starts = np.r_[0, np.where(np.diff(sv) != 0)[0] + 1, len(sv)]
    for i in range(len(starts) - 1):
        group = order[starts[i]:starts[i + 1]]
        group = group[known[group]]
        if len(group) < 2:
            continue
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                va, vb = group[a], group[b]
                rows += [r, r]
                cols += [va, vb]
                vals += [1.0, -1.0]
                rhs.append(col[vb] - col[va])
                r += 1
    seam_rows = r
    # smoothness along each face edge, inside the face's own label
    s = np.sqrt(smooth)
    e = np.concatenate([var_of_corner[:, [0, 1]], var_of_corner[:, [1, 2]], var_of_corner[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    ne = len(e)
    rows += list(np.repeat(np.arange(r, r + ne), 2))
    cols += list(e.reshape(-1))
    vals += [s, -s] * ne
    rhs += [np.zeros(3)] * ne
    r += ne
    a_mat = sparse.coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    b_mat = np.asarray(rhs).reshape(r, 3)
    ata = (a_mat.T @ a_mat + sparse.identity(nvar) * 1e-4).tocsc()
    lu = splu(ata)
    off = np.stack([lu.solve(a_mat.T @ b_mat[:, ch]) for ch in range(3)], axis=1)
    level_seams.report = {"variables": int(nvar), "seam_constraints": int(seam_rows)}
    return (off[vars_s] * w[:, :, None]).sum(axis=1)


def sample_atlas(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    th, tw = tex.shape[:2]
    tx = (np.clip(uv[:, 0], 0, 1) * (tw - 1)).astype(np.int32)
    ty = (np.clip(1.0 - uv[:, 1], 0, 1) * (th - 1)).astype(np.int32)
    return tex[ty, tx].astype(np.float64)


def bake(obs: list[dict], base_tex: np.ndarray, uv_pts: np.ndarray, fid: np.ndarray, bary: np.ndarray,
         mfaces: np.ndarray, nv: int, smooth: float = SMOOTH) -> dict:
    """obs: one dict per view {name, seen (N,), conf (N,), colour (N x 3), boost, anchor}. Returns {colour (N x 3),
    label (F,), gains (K x 3), base (the base texture with its gain), report}."""
    nf = len(mfaces)
    n = len(fid)
    k = len(obs) + 1
    base_col = sample_atlas(base_tex, uv_pts)
    scores = np.zeros((nf, k), np.float32)
    means = np.full((nf, k, 3), np.nan)
    count = np.bincount(fid, minlength=nf)
    for j, o in enumerate(obs):
        scores[:, j] = face_scores(fid, nf, o["seen"], o["conf"] * o.get("boost", 1.0))
        cnt = np.bincount(fid, weights=o["seen"].astype(float), minlength=nf)
        for ch in range(3):
            s = np.bincount(fid, weights=np.where(o["seen"], o["colour"][:, ch], 0.0), minlength=nf)
            means[:, j, ch] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    scores[:, -1] = BASE_SCORE
    for ch in range(3):
        s = np.bincount(fid, weights=base_col[:, ch], minlength=nf)
        means[:, -1, ch] = np.where(count > 0, s / np.maximum(count, 1), np.nan)
    adj = face_adjacency(mfaces)
    label = label_faces(scores, adj, smooth)
    anchors = np.array([bool(o.get("anchor", True)) for o in obs] + [False])
    gains = solve_gains(means, scores, anchors)

    def compose(label):
        """Each sample's colour from its face's label; a sample that label cannot see (a face half hidden from
        it) takes the best view that does, else the base, matched to the views around it."""
        colour = base_col * gains[-1]
        used = np.full(n, k - 1)
        best = np.full(n, -1.0)
        slabel = label[fid]
        for j, o in enumerate(obs):
            c = o["colour"] * gains[j]
            mine = o["seen"] & (slabel == j)
            colour[mine] = c[mine]
            used[mine] = j
            alt = o["seen"] & (slabel != j) & (used == k - 1) & (o["conf"] > best)
            alt &= slabel != k - 1
            colour[alt] = c[alt]
            best[alt] = o["conf"][alt]
        local = match_base(mfaces, nv, label, fid, bary, colour, base_col, k, used)
        on_base = used == k - 1
        colour[on_base] = (base_col * local)[on_base]
        return colour, base_col * local

    colour, matched = compose(label)
    # Photo-consistency: where a view's colour on a face is far from the paint there (matched to the views'
    # local colour), that view drew something the mesh does not have at that spot (a drawn ear on the neck behind
    # the mesh's ear): its score there drops and the faces are labelled again.
    # Checked per sample, not per face mean: an eye is a few samples, and a face mean washed it out (the pig kept
    # the side views' eyes on its cheeks beside the front view's, and their ears behind the mesh's ears).
    penalised = 0
    for j, o in enumerate(obs):
        seen = o["seen"]
        bad = np.zeros(n, bool)
        bad[seen] = _tone_distance(o["colour"][seen] * gains[j], matched[seen]) > CONSIST_SAMPLE
        if o.get("stray") is not None:
            bad |= seen & o["stray"]
        nseen = np.bincount(fid, weights=seen.astype(float), minlength=nf)
        frac = np.bincount(fid, weights=bad.astype(float), minlength=nf) / np.maximum(nseen, 1)
        factor = np.clip(1.0 - (frac - CONSIST_FRAC) / CONSIST_SPAN, CONSIST_FLOOR, 1.0)
        # a misplaced feature's surroundings go with it (the pale outline of a drawn ear around its dark inside)
        pen = 1.0 - factor
        for _ in range(int(CONSIST_GROW)):
            pen = np.maximum(pen, np.asarray(adj.multiply(pen[None, :]).max(axis=1).todense()).reshape(-1) * 0.9)
        factor = 1.0 - pen
        penalised += int(((factor < 0.99) & (label == j)).sum())
        scores[:, j] *= factor
        if CONSIST_DROP:
            # gone from every path, not only the face scores: the per-sample fallback in compose() handed a
            # penalised view's samples straight back
            o["seen"] = seen & ~bad
    label = label_faces(scores, adj, smooth)
    colour, _ = compose(label)
    off = level_seams(mfaces, nv, label, fid, bary, colour)
    colour = np.clip(colour + off, 0, 255)
    names = [o["name"] for o in obs] + ["base"]
    share = np.bincount(label, minlength=k) / max(nf, 1)
    report = {"faces": {names[j]: round(float(share[j]), 4) for j in range(k)}, "inconsistent_faces": penalised,
              "gains": {names[j]: [round(float(x), 3) for x in gains[j]] for j in range(k)},
              "offset_mean": round(float(np.abs(off).mean()), 2), **getattr(level_seams, "report", {})}
    return {"colour": colour.astype(np.float32), "label": label, "gains": gains,
            "base": np.clip(base_tex.astype(np.float32) * gains[-1], 0, 255), "report": report}


# ---------------------------------------------------------------- aligning a view to the mesh

CONTOUR_SIGMA = 0.004   # blur before edges, fraction of the view's longer side (the bristles are not contours)
CONTOUR_NEAR = 0.012    # the paint's edge may be this far (fraction of the longer side) and still count
CONTOUR_HIGH = 1.0      # a drawing edge at least this x its 90th percentile is a contour...
CONTOUR_LOW = 0.45      # ...that the paint lacks when its nearby edge is under this x the paint's 90th percentile


def edge_strength(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blurred-lightness gradient magnitude inside the mask (its own outline left out), over its 90th percentile."""
    h, w = mask.shape
    s = max(1.0, CONTOUR_SIGMA * max(h, w))
    lum = ndimage.gaussian_filter(rgb.mean(axis=2), s)
    mag = np.hypot(ndimage.sobel(lum, 0), ndimage.sobel(lum, 1))
    inner = ndimage.binary_erosion(mask, iterations=max(2, int(3 * s)))
    mag = np.where(inner, mag, 0.0)
    p90 = np.percentile(mag[inner], 90) if inner.any() else 1.0
    return mag / max(p90, 1e-6)


def stray_contours(draw_rgb, draw_mask, paint_rgb, paint_mask, px, py, qx, qy) -> np.ndarray:
    """Per sample: the drawing (sampled at its aligned pixel qx, qy) has a contour there and the paint (at the
    sample's own pixel px, py) has none nearby: an ear or eye drawn where the mesh has plain skin."""
    h, w = draw_mask.shape
    ed = edge_strength(draw_rgb, draw_mask)
    ep = edge_strength(paint_rgb, paint_mask)
    ep = ndimage.maximum_filter(ep, size=max(3, int(2 * CONTOUR_NEAR * max(h, w)) | 1))
    return (ed[qy, qx] > CONTOUR_HIGH) & (ep[py, px] < CONTOUR_LOW) & paint_mask[py, px]

ALIGN_PX = 256          # registration runs at this size (the longer side)
ALIGN_ROUNDS = (60, 40, 30)  # demons iterations per level, coarse to fine
ALIGN_SIGMA = 0.012     # displacement smoothing, as a fraction of ALIGN_PX
ALIGN_MAX = 0.1         # displacement cap, as a fraction of the view's longer side
ALIGN_TONE = 0.5        # weight of the tone features against the silhouette (2)
ALIGN_ALPHA = 0.3       # Thirion's diff term weight (smaller: bolder steps)
ALIGN_STEP = 2.0        # demons step
ALIGN_DIFFUSE = 8.0     # smoothing of the field itself, pixels at ALIGN_PX: whole regions move together


def render_base(px, py, seen, base_col, shape):
    """The base texture (Hunyuan's paint, which sits on the mesh exactly) as the fitted camera sees it: the seen
    samples splatted at their pixels, holes closed by normalised blur inside their footprint. Returns (rgb, mask)."""
    h, w = shape
    img = np.zeros((h, w, 3))
    wt = np.zeros((h, w))
    np.add.at(wt, (py[seen], px[seen]), 1.0)
    for ch in range(3):
        np.add.at(img[:, :, ch], (py[seen], px[seen]), base_col[seen, ch])
    mask = ndimage.binary_closing(wt > 0, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    s = max(1.0, 0.004 * max(h, w))
    wb = ndimage.gaussian_filter(wt, s)
    out = np.stack([ndimage.gaussian_filter(img[:, :, ch], s) for ch in range(3)], axis=2) / np.maximum(wb, 1e-6)[:, :, None]
    return out, mask


def _features(rgb, mask):
    """Silhouette plus normalised lightness and redness inside it: what a drawing and the paint share (ears, eyes,
    snout, hooves) whatever their overall tone."""
    lum = rgb.mean(axis=2)
    red = rgb[:, :, 0] - rgb[:, :, 1]
    feats = [mask.astype(float) * 2.0]
    for f in (lum, red):
        m = mask & np.isfinite(f)
        mu, sd = (f[m].mean(), f[m].std() + 1e-6) if m.any() else (0.0, 1.0)
        feats.append(np.where(mask, (f - mu) / sd, 0.0) * ALIGN_TONE)
    return np.stack(feats, axis=0)


def _warp(stack, dy, dx):
    h, w = stack.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    return np.stack([ndimage.map_coordinates(c, [yy + dy, xx + dx], order=1, mode="nearest") for c in stack])


def align(fixed_rgb, fixed_mask, moving_rgb, moving_mask):
    """A smooth displacement (dy, dx), at the moving image's size, such that moving[y + dy, x + dx] lines up with
    fixed[y, x]: Thirion's demons on silhouette + tone features, coarse to fine, Gaussian-regularised and capped."""
    h, w = moving_mask.shape
    scale = ALIGN_PX / max(h, w)
    sh, sw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))

    def small(a, order=1):
        return ndimage.zoom(a, (sh / a.shape[0], sw / a.shape[1]) + ((1,) if a.ndim == 3 else ()), order=order)
    f = _features(small(fixed_rgb), small(fixed_mask.astype(float)) > 0.5)
    m = _features(small(moving_rgb), small(moving_mask.astype(float)) > 0.5)
    dy = np.zeros((sh, sw))
    dx = np.zeros((sh, sw))
    sigma = ALIGN_SIGMA * ALIGN_PX
    cap = ALIGN_MAX * ALIGN_PX
    for level, rounds in zip((4, 2, 1), ALIGN_ROUNDS):
        fs = np.stack([ndimage.gaussian_filter(c, level) for c in f])
        ms = np.stack([ndimage.gaussian_filter(c, level) for c in m])
        gy = np.stack([ndimage.sobel(c, 0) / 8 for c in fs])
        gx = np.stack([ndimage.sobel(c, 1) / 8 for c in fs])
        for _ in range(rounds):
            diff = _warp(ms, dy, dx) - fs
            num_y = (diff * gy).sum(0)
            num_x = (diff * gx).sum(0)
            den = (gy ** 2 + gx ** 2).sum(0) + ALIGN_ALPHA ** 2 * (diff ** 2).sum(0) + 1e-3
            # at most a pixel per step where the gradient is weak: the update is a direction, not a jump
            uy, ux = num_y / den, num_x / den
            mag = np.hypot(uy, ux)
            big = mag > 1.0
            uy[big] /= mag[big]
            ux[big] /= mag[big]
            num_y, num_x, den = uy, ux, 1.0
            # fluid-like: the update is smoothed wide, the field itself only lightly, so it can travel far
            # (an ear 30 px off) while staying smooth
            dy = ndimage.gaussian_filter(dy - ALIGN_STEP * ndimage.gaussian_filter(num_y / den, sigma), ALIGN_DIFFUSE)
            dx = ndimage.gaussian_filter(dx - ALIGN_STEP * ndimage.gaussian_filter(num_x / den, sigma), ALIGN_DIFFUSE)
            mag = np.hypot(dy, dx)
            over = mag > cap
            dy[over] *= cap / mag[over]
            dx[over] *= cap / mag[over]
    before = float(np.abs(f - m).mean())
    after = float(np.abs(f - _warp(m, dy, dx)).mean())
    align.report = {"error_before": round(before, 4), "error_after": round(after, 4),
                    "shift_max_px": round(float(np.hypot(dy, dx).max() / scale), 1)}
    up = (h / sh, w / sw)
    return ndimage.zoom(dy, up, order=1)[:h, :w] / scale, ndimage.zoom(dx, up, order=1)[:h, :w] / scale
