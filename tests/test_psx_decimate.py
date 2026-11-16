"""psx_lowpoly's decimation: crumbs go, thin parts the decimator pinches off stay, and --head-weight spends more
of the budget on the head without moving it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import psx_lowpoly as P  # noqa: E402


def creature() -> trimesh.Trimesh:
    """A long body (front +Z) with four thin legs and a head, unioned into one detailed mesh."""
    body = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    body.apply_scale([0.6, 0.5, 1.0])
    body.apply_translation([0, 1.2, 0])
    head = trimesh.creation.icosphere(subdivisions=4, radius=0.3)
    head.apply_translation([0, 1.5, 1.05])
    legs = []
    for x in (-0.3, 0.3):
        for z in (-0.6, 0.6):
            leg = trimesh.creation.cylinder(radius=0.05, height=1.0, sections=24)
            leg.apply_translation([x, 0.5, z])
            legs.append(leg)
    return trimesh.boolean.union([body, head, *legs]) if trimesh.boolean.engines_available else \
        trimesh.util.concatenate([body, head, *legs])


def test_floating_crumb_is_dropped() -> None:
    body = trimesh.creation.icosphere(subdivisions=4)
    crumb = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    crumb.apply_translation([3, 0, 0])
    low = P.decimate(trimesh.util.concatenate([body, crumb]), 400)
    assert low.bounds[1][0] < 1.5 and len(low.faces) <= 420


def test_small_pieces_after_decimation_are_kept_slivers_are_not() -> None:
    body = trimesh.creation.icosphere(subdivisions=3)
    leg = trimesh.creation.cylinder(radius=0.05, height=1.0, sections=8)  # about 1.5% of the area, along Z
    leg.apply_translation([0, -1.5, 0])
    sliver = trimesh.Trimesh(vertices=[[2, 0, 0], [2.01, 0, 0], [2, 0.01, 0]], faces=[[0, 1, 2]])
    joined = trimesh.util.concatenate([body, leg, sliver])
    kept = P._keep(joined, lambda p: len(p.faces) > 1 and p.area >= P.SLIVER * joined.area)
    assert kept.bounds[0][1] < -1.4  # the leg stayed
    assert kept.bounds[1][0] < 1.5  # the sliver went


def test_head_weight_spends_more_on_the_head_and_keeps_its_place() -> None:
    mesh = creature()
    plain = P.decimate_head(mesh, 600, 1.0)
    heavy = P.decimate_head(mesh, 600, 2.0)
    front = mesh.bounds[0][2] + 0.8 * (mesh.bounds[1][2] - mesh.bounds[0][2])

    def head_faces(m: trimesh.Trimesh) -> int:
        return int((m.triangles_center[:, 2] > front).sum())

    assert head_faces(heavy) > head_faces(plain)
    assert np.allclose(heavy.bounds, mesh.bounds, atol=0.05)
    assert len(heavy.faces) <= 630


def test_near_duplicate_vertices_are_welded_so_the_cut_reaches_its_target() -> None:
    # the CRT: vertices a hair apart along its creases stalled the decimator at 650 whatever it was asked
    box = trimesh.creation.box(extents=(1, 1, 1)).subdivide().subdivide().subdivide().subdivide()
    split = trimesh.Trimesh(box.triangles.reshape(-1, 3), np.arange(len(box.faces) * 3).reshape(-1, 3), process=False)
    split.vertices = split.vertices + np.random.default_rng(1).normal(scale=1e-7, size=split.vertices.shape)
    low = P.simplify(split, 300)
    assert 280 <= len(low.faces) <= 330


def test_slivers_dropped_after_the_cut_are_made_up() -> None:
    # the scrap pile: thin sheets break into one-face slivers the cut drops; the budget is raised to make them up
    sheets = []
    for k in range(40):
        s = trimesh.creation.box(extents=(1.0, 0.002, 0.6)).subdivide().subdivide()
        s.apply_translation([0, k * 0.05, 0])
        sheets.append(s)
    body = trimesh.creation.icosphere(subdivisions=5, radius=0.8)
    body.apply_translation([0, 0.9, 0])
    low = P.decimate(trimesh.util.concatenate([body, *sheets]), 900)
    assert len(low.faces) >= 0.9 * 900
