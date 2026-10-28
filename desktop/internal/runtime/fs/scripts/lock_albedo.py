#!/usr/bin/env python3
"""Front-lock: project the original photo onto a painted GLB's albedo.

Hunyuan paint hallucinates the occluded back. This pass overwrites texels that
the original camera actually saw, so the hero view matches the concept. Unseen
texels keep the Hunyuan albedo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def look_at(eye, target, up):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / np.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def sample_face_bary(mesh, n):
    points, face_idx = trimesh.sample.sample_surface(mesh, n)
    faces = mesh.faces[face_idx]
    tri = mesh.vertices[faces]
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    v2 = points - tri[:, 0]
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1)
    return points, face_idx, bary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("photo")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--strength", type=float, default=0.85)
    args = ap.parse_args()

    import trimesh

    mesh = trimesh.load(args.glb, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print("lock: expected a triangle mesh", file=sys.stderr)
        return 1
    vis = mesh.visual
    if not hasattr(vis, "uv") or vis.uv is None:
        print("lock: mesh has no UVs — run paint first", file=sys.stderr)
        return 1
    material = getattr(vis, "material", None)
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is None:
        print("lock: no albedo texture on GLB", file=sys.stderr)
        return 1
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    albedo = np.array(image.convert("RGB"), dtype=np.float32)
    th, tw = albedo.shape[:2]
    photo = np.array(Image.open(args.photo).convert("RGB"), dtype=np.float32)
    ph, pw = photo.shape[:2]

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    uvs = np.asarray(vis.uv, dtype=np.float32)
    center = verts.mean(axis=0)
    scale = float(np.linalg.norm(verts.max(0) - verts.min(0)) + 1e-6)
    eye = center + np.array([0.72, 0.48, 0.92], dtype=np.float32) * scale * 1.35
    target = center + np.array([0.0, 0.02, 0.0], dtype=np.float32) * scale
    view = look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    proj = perspective(np.deg2rad(35.0), 1.0, 0.05, 40.0)
    vp = proj @ view

    points, face_idx, bary = sample_face_bary(mesh, 350_000)
    faces = mesh.faces[face_idx]
    uv_pts = (
        bary[:, 0:1] * uvs[faces[:, 0]]
        + bary[:, 1:2] * uvs[faces[:, 1]]
        + bary[:, 2:3] * uvs[faces[:, 2]]
    )
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nlen = np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    n = n / nlen
    view_dir = eye - points
    view_dir = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8)
    ndotv = np.einsum("ij,ij->i", n, view_dir)

    homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    clip = (vp @ homo.T).T
    w = clip[:, 3]
    valid = (w > 1e-5) & (ndotv > 0.12)
    ndc = clip[:, :3] / np.clip(w[:, None], 1e-5, None)
    px = ((ndc[:, 0] * 0.5 + 0.5) * (pw - 1)).astype(np.int32)
    py = ((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (ph - 1)).astype(np.int32)
    in_frame = valid & (px >= 0) & (px < pw) & (py >= 0) & (py < ph) & (ndc[:, 2] > -1) & (ndc[:, 2] < 1)
    depth = ndc[:, 2]

    sw, sh = 768, 768
    sx = np.clip(((ndc[:, 0] * 0.5 + 0.5) * (sw - 1)).astype(np.int32), 0, sw - 1)
    sy = np.clip(((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (sh - 1)).astype(np.int32), 0, sh - 1)
    pix = sy * sw + sx
    order = np.argsort(depth)
    keep = np.zeros(sw * sh, dtype=np.int32)
    keep[:] = -1
    sel = order[in_frame[order]]
    keep[pix[sel]] = sel
    winners = keep[keep >= 0]
    print(f"lock: {len(winners)} visible samples")

    out = albedo.copy()
    tx = np.clip(uv_pts[winners, 0], 0, 1) * (tw - 1)
    ty = np.clip(1.0 - uv_pts[winners, 1], 0, 1) * (th - 1)
    tx = tx.astype(np.int32)
    ty = ty.astype(np.int32)
    src = photo[py[winners], px[winners]]
    blend = np.clip((ndotv[winners] - 0.12) / 0.55, 0.0, 1.0) * args.strength
    dest = out[ty, tx]
    out[ty, tx] = dest * (1.0 - blend[:, None]) + src * blend[:, None]

    locked = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    vis.material.baseColorTexture = locked
    vis.material.image = locked
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out)
    sidecar = Path(args.out).with_suffix(".albedo.png")
    locked.save(sidecar)
    print(f"lock: wrote {args.out} and {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
