"""seam_bake (retexture --select): labels stay in patches, gains even out views, seams meet, the base follows the
views' colour, and align pulls a shifted picture back onto its target."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import seam_bake as sb  # noqa: E402


def grid(n: int = 20):
    """A flat n x n quad grid split into triangles (z = 0)."""
    plane = trimesh.creation.box(extents=(1, 1, 1)).subdivide().subdivide().subdivide()
    return np.asarray(plane.vertices), np.asarray(plane.faces)


class SeamBakeTests(unittest.TestCase):
    def test_labels_form_patches_where_scores_are_noisy(self) -> None:
        V, F = grid()
        adj = sb.face_adjacency(F)
        rng = np.random.default_rng(1)
        left = V[F].mean(1)[:, 0] < 0
        scores = np.stack([np.where(left, 0.6, 0.4), np.where(left, 0.4, 0.6)], 1) + rng.normal(0, 0.15, (len(F), 2))
        scores = np.clip(scores, 0.01, None)
        scores = np.c_[scores, np.full(len(F), sb.BASE_SCORE)]
        raw = np.argmax(scores, 1)
        lab = sb.label_faces(scores.astype(np.float32), adj)
        cut = lambda l: int(sum(l[a] != l[b] for a, b in zip(*adj.nonzero())))  # noqa: E731
        self.assertLess(cut(lab), cut(raw) * 0.6)

    def test_gains_even_out_a_darker_view(self) -> None:
        nf = 200
        true = np.full((nf, 3), 150.0)
        means = np.stack([true, true * 0.9, true * 0.6], 1)  # view 1 a little darker, the base much darker
        scores = np.c_[np.full(nf, 0.8), np.full(nf, 0.8), np.full(nf, sb.BASE_SCORE)]
        g = sb.solve_gains(means, scores, np.array([True, True, False]))
        self.assertAlmostEqual(float(g[0, 0] * 1.0 / (g[1, 0] * 0.9)), 1.0, delta=0.03)
        self.assertGreater(g[2, 0], 1.4)  # the paint is brought up to the views

    def test_seams_meet_after_levelling(self) -> None:
        V, F = grid()
        mf, nv = sb.merged_faces(V, F)
        label = (V[F].mean(1)[:, 0] > 0).astype(int)
        fid = np.repeat(np.arange(len(F)), 6)
        bary = np.tile(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5]]), (len(F), 1))
        colour = np.where(label[fid][:, None] == 1, 180.0, 120.0) * np.ones((len(fid), 3))
        off = sb.level_seams(mf, nv, label, fid, bary.astype(float), colour)
        out = colour + off
        # vertices on the seam: both labels' colours there now agree
        seam_v = np.intersect1d(mf[label == 0].ravel(), mf[label == 1].ravel())
        on = np.isin(mf[fid], seam_v).any(1) & (bary.max(1) == 1)
        v_of = mf[fid][np.arange(len(fid)), bary.argmax(1)]
        for v in seam_v[:10]:
            vals = out[on & (v_of == v), 0]
            self.assertLess(float(np.ptp(vals)), 8.0)

    def test_align_recovers_a_shift(self) -> None:
        h = w = 128
        yy, xx = np.mgrid[0:h, 0:w]
        disc = lambda cy, cx: ((yy - cy) ** 2 + (xx - cx) ** 2 < 30 ** 2)  # noqa: E731
        fixed = disc(64, 64)
        moving = disc(64, 72)  # the picture's subject 8 px to the right
        rgb_f = np.dstack([fixed * 200.0] * 3)
        rgb_m = np.dstack([moving * 200.0] * 3)
        dy, dx = sb.align(rgb_f, fixed, rgb_m, moving)
        self.assertGreater(float(np.median(dx[fixed])), 5.0)  # the inside moves with the edge, not just the edge
        self.assertLess(sb.align.report["error_after"], sb.align.report["error_before"])


if __name__ == "__main__":
    unittest.main()
