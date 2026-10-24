#!/usr/bin/env python3
"""Headless render of a rigged GLB: every clip at a few frames, from fixed ortho cameras, plus a contact sheet.

    .venv/bin/python scripts/rig_render.py --stem psx-cow --clips auto
    .venv/bin/python scripts/rig_render.py --stem S --views side,front,q --modes texture,sticks --clips 'Walk@0.25,0.5;Peck@0.15'
    .venv/bin/python scripts/rig_render.py --asset /path/to/copy/of/out/S --clips auto --blender

CPU path (the default), numpy and PIL only:
- linear-blend skinning with rig_motion.world_mats and rig_metrics.skin_frames;
- a numpy z-buffer rasteriser. Each face takes one texture colour at its UV centroid (nearest sampling, the PS1
  look) and a flat Lambert shade;
- one ortho camera centre and scale, fixed from the rest-pose bbox (floor included) and shared by every frame and view,
  so anything that grows, sinks or flies away shows as such;
- a red floor line at y = 0 (edge-on in the side, front and back views; an outline in top, bottom and q).

Modes: `texture`; `groups` (each corner coloured by its top bone's role); `sticks` (faded texture with x-ray joint
balls and bone lines at 50% alpha, coloured by role); `heat:<bone>` (that bone's weight on a ramp); `texture_px`
(the texture sampled per pixel: each pixel's UV from its barycentrics, nearest texel, the same flat Lambert shade)
and `texture_px:unlit` (the same, unshaded). One texel a face is the rig sheets' PS1 feel, but on a low-poly mesh
it paints a long sliver triangle as one stripe whatever its UVs cover, so a model check (workbench eyes.look)
wants texture_px: it shows the texture as an engine would.

Views (glTF frame: +Y up, +Z = the creature's forward, +X = its left). These match rig_fit.vlm_cue and brender:
- side: camera at +X; image right is -Z (a correctly faced head at +Z shows on the LEFT);
- xside: camera at -X; image right is +Z (head on the RIGHT);
- front: camera at +Z (the face); image right is +X;  back: camera at -Z; image right is -X;
- top: camera at +Y; image right is +X, image up is -Z;
- bottom: camera at -Y; image right is +X, image up is +Z (top's image flipped upside down);
- q: three-quarter from (+X, +Y/2, +Z); image right is roughly (+X, -Z).
VIEW_RIGHT gives the world vector of image right per view.

Outputs (under <asset>/rig/preview/): <clip>_<frac>_<view>_<mode>.png, contact_sheet.png (rows Rest then every clip,
columns side, front, q) and frames.json (per frame: height_rel, tilt, min_y_rel, max_disp_rel, stance-foot y, beak y).

--blender renders the same job through Blender Workbench (scripts/blender/rig_render_bl.py) into rig/preview/blender/,
for a parity cross-check only.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig  # noqa: E402
import rig_metrics  # noqa: E402
import rig_motion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG = (127, 182, 232)
FLOOR = (255, 0, 0)
CAP = 18  # caption strip height in the contact sheet
FILL = 0.85  # the rest bbox diagonal spans this share of the image
FPS = 30
VIEWS = ("side", "xside", "front", "back", "top", "bottom", "q")
PX_MODES = ("texture_px", "texture_px:unlit")
SHEET_VIEWS = ("side", "front", "q")
ROLE_RGB = {"leg": (26, 102, 255), "wing": (255, 128, 0), "spine": (255, 255, 26), "head": (255, 26, 255),
            "tail": (26, 255, 77), "other": (255, 255, 255), "root": (110, 110, 110)}

# view -> (direction from the target to the camera, up hint); glTF frame
_CAM = {"side": ((1, 0, 0), (0, 1, 0)), "xside": ((-1, 0, 0), (0, 1, 0)), "front": ((0, 0, 1), (0, 1, 0)),
        "back": ((0, 0, -1), (0, 1, 0)), "top": ((0, 1, 0), (0, 0, -1)), "bottom": ((0, -1, 0), (0, 0, 1)),
        "q": ((1, 0.5, 1), (0, 1, 0))}
EDGE_ON = ("side", "xside", "front", "back")


def basis(view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward) unit vectors of a view's camera; forward points from the camera into the scene."""
    if view not in _CAM:
        raise ValueError(f"unknown view {view!r}; one of {VIEWS}")
    d, u = (np.asarray(v, float) for v in _CAM[view])
    f = -d / np.linalg.norm(d)
    r = np.cross(f, u)
    r /= np.linalg.norm(r)
    return r, np.cross(r, f), f


VIEW_RIGHT = {v: [round(float(x), 4) for x in basis(v)[0]] for v in VIEWS}


# ---------------------------------------------------------------- camera

def camera(points: np.ndarray, size: int = 384, fill: float = FILL) -> dict:
    """The fixed ortho camera: centre = the bbox centre of `points` plus the floor point under it, scale (pixels per
    unit) so that the bbox diagonal spans `fill` of the image."""
    P = np.asarray(points, float)
    lo, hi = P.min(0), P.max(0)
    lo[1], hi[1] = min(lo[1], 0.0), max(hi[1], 0.0)
    diag = max(float(np.linalg.norm(hi - lo)), 1e-6)
    return {"centre": ((lo + hi) / 2).tolist(), "scale": fill * size / diag, "size": int(size),
            "bbox": [lo.tolist(), hi.tolist()]}


def project(points: np.ndarray, view: str, cam: dict) -> np.ndarray:
    """(N,3) world points -> (N,3) [pixel x, pixel y, depth]. Pixel centres sit at i + 0.5; depth grows away from
    the camera."""
    r, u, f = basis(view)
    Q = np.asarray(points, float) - np.asarray(cam["centre"], float)
    s, n = cam["scale"], cam["size"]
    return np.stack([n / 2 + Q @ r * s, n / 2 - Q @ u * s, Q @ f], axis=1)


def floor_row(cam: dict, view: str) -> float:
    """Pixel row of the floor (y = 0) in an edge-on view."""
    if view not in EDGE_ON:
        raise ValueError(f"the floor is edge-on only in {EDGE_ON}")
    c = np.asarray(cam["centre"], float)
    return float(project(np.array([[c[0], 0.0, c[2]]]), view, cam)[0, 1])


# ---------------------------------------------------------------- rasteriser

def raster(V: np.ndarray, F: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer: V (N,3) projected [x, y, depth], F (T,3). Returns face id per pixel (-1 = empty) and the
    barycentric weights (size, size, 3). No back-face culling (PS1 meshes have mixed winding)."""
    face = np.full(size * size, -1, np.int64)
    bary = np.zeros((size * size, 3))
    if len(F) == 0:
        return face.reshape(size, size), bary.reshape(size, size, 3)
    T = V[F]  # (T,3,3)
    x0 = np.clip(np.ceil(T[:, :, 0].min(1) - 0.5), 0, size).astype(int)
    x1 = np.clip(np.floor(T[:, :, 0].max(1) - 0.5), -1, size - 1).astype(int)
    y0 = np.clip(np.ceil(T[:, :, 1].min(1) - 0.5), 0, size).astype(int)
    y1 = np.clip(np.floor(T[:, :, 1].max(1) - 0.5), -1, size - 1).astype(int)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    a = T[:, 0, :2]
    e1, e2 = T[:, 1, :2] - a, T[:, 2, :2] - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok = (w > 0) & (h > 0) & (np.abs(det) > 1e-12)
    ext = np.maximum(w, h)
    frag_idx, frag_tri, frag_z, frag_b = [], [], [], []
    K = 1
    while True:
        sel = np.nonzero(ok & (ext <= K) & (ext > K // 2 if K > 1 else ext <= 1))[0]
        if len(sel):
            gy, gx = np.mgrid[0:K, 0:K]
            gx, gy = gx.ravel(), gy.ravel()
            step = max(1, 4_000_000 // (K * K))
            for s0 in range(0, len(sel), step):
                t = sel[s0:s0 + step]
                px = x0[t, None] + gx[None]
                py = y0[t, None] + gy[None]
                inb = (px <= x1[t, None]) & (py <= y1[t, None])
                dx = px + 0.5 - a[t, 0, None]
                dy = py + 0.5 - a[t, 1, None]
                b1 = (dx * e2[t, 1, None] - dy * e2[t, 0, None]) / det[t, None]
                b2 = (e1[t, 0, None] * dy - e1[t, 1, None] * dx) / det[t, None]
                b0 = 1 - b1 - b2
                eps = -1e-7
                m = inb & (b0 >= eps) & (b1 >= eps) & (b2 >= eps)
                if not m.any():
                    continue
                ti = np.broadcast_to(t[:, None], m.shape)[m]
                B = np.stack([b0[m], b1[m], b2[m]], 1)
                frag_idx.append(py[m] * size + px[m])
                frag_tri.append(ti)
                frag_b.append(B)
                frag_z.append((B * T[ti, :, 2]).sum(1))
        if K >= ext[ok].max(initial=0):
            break
        K *= 2
    if not frag_idx:
        return face.reshape(size, size), bary.reshape(size, size, 3)
    idx = np.concatenate(frag_idx)
    tri = np.concatenate(frag_tri)
    z = np.concatenate(frag_z)
    B = np.concatenate(frag_b)
    order = np.lexsort((z, idx))
    first = np.r_[True, idx[order][1:] != idx[order][:-1]]
    win = order[first]
    face[idx[win]] = tri[win]
    bary[idx[win]] = B[win]
    return face.reshape(size, size), bary.reshape(size, size, 3)


def ramp(x: np.ndarray) -> np.ndarray:
    """0..1 -> RGB 0..255 (dark blue, cyan, green, yellow, red)."""
    stops = np.array([[20, 20, 90], [0, 200, 255], [40, 220, 40], [255, 230, 0], [255, 30, 0]], float)
    x = np.clip(np.asarray(x, float), 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


# ---------------------------------------------------------------- roles

_HEAD = ("head", "mouth", "beak", "jaw", "eye", "skull", "ear", "nose", "horn", "snout")
_LEG = ("leg", "foot", "toe", "thigh", "calf", "shin", "knee", "ankle", "hoof", "heel", "ball", "metatars", "tars")
_WING = ("wing", "arm", "hand", "finger", "feather", "shoulder", "clavicle", "humerus", "elbow", "wrist", "thumb",
         "scapula", "radius", "ulna")
_SPINE = ("spine", "neck", "hips", "pelvis", "chest", "torso", "body", "waist", "belly")


def role(name: str, partmap: dict | None = None) -> str:
    """leg, wing, spine (spine and neck), head, tail, root or other. `partmap` overrides by exact name. Names on a
    rig_fit.CHAINS chain take that chain's role; the rest go by name."""
    if partmap and name in partmap:
        return partmap[name]
    if name == "root":
        return "root"
    hit = _chain_roles().get(name)
    if hit:
        return hit
    n = name.lower()
    if n.startswith(("front_", "back_", "fore_", "hind_")) and not any(k in n for k in ("tail", "spine", "neck")):
        return "leg"
    for words, r in ((_HEAD, "head"), (_LEG, "leg"), (_WING, "wing"), (("tail",), "tail"), (_SPINE, "spine")):
        if any(k in n for k in words):
            return r
    return "other"


_CHAIN_ROLES: dict | None = None


def _chain_roles() -> dict:
    global _CHAIN_ROLES
    if _CHAIN_ROLES is None:
        _CHAIN_ROLES = {}
        try:
            import rig_fit
            for spec in rig_fit.CHAINS.values():
                for key, r in (("legs", "leg"), ("wings", "wing"), ("tail", "tail")):
                    v = spec.get(key) or []
                    for ch in (v if v and isinstance(v[0], list) else [v]):
                        for nm in ch:
                            _CHAIN_ROLES.setdefault(nm, r)
                for nm in spec.get("neck") or []:
                    _CHAIN_ROLES.setdefault(nm, "head" if any(k in nm.lower() for k in _HEAD) else "spine")
        except Exception:  # noqa: BLE001 - name rules alone still work
            pass
    return _CHAIN_ROLES


# ---------------------------------------------------------------- scene loading

def _material(doc: dict, binary: bytes, prim: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """A primitive's base colour factor (RGB 0..1) as texture mode has always read it (0.72 grey when unset), the
    factor as glTF defines it (1 when unset: what multiplies the texture in an engine, and in texture_px), and its
    decoded base-colour texture (uint8 H x W x 3, or None)."""
    mat = doc.get("materials", [{}])[prim["material"]] if "material" in prim and doc.get("materials") else {}
    pbr = mat.get("pbrMetallicRoughness", {})
    factor = np.asarray(pbr.get("baseColorFactor", [0.72, 0.72, 0.72, 1]), float)[:3]
    gltf_factor = np.asarray(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1]), float)[:3]
    try:
        tex_i = pbr["baseColorTexture"]["index"]
        img = doc["images"][doc["textures"][tex_i]["source"]]
        v = doc["bufferViews"][img["bufferView"]]
        o = v.get("byteOffset", 0)
        tex = np.asarray(Image.open(io.BytesIO(binary[o:o + v["byteLength"]])).convert("RGB"))
    except (KeyError, IndexError, TypeError, OSError):
        return factor, gltf_factor, None
    return factor, gltf_factor, tex


def _texels(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest texels (float RGB 0..255) at (N,2) UVs, wrapped (glTF's default REPEAT); v counts down the image."""
    th, tw = tex.shape[:2]
    return tex[np.clip(((uv[:, 1] % 1) * th).astype(int), 0, th - 1),
               np.clip(((uv[:, 0] % 1) * tw).astype(int), 0, tw - 1)].astype(float)


def _texture_colours(factor: np.ndarray, tex: np.ndarray | None, UV: np.ndarray | None, F: np.ndarray) -> np.ndarray:
    """Per-face RGB 0..255: the base-colour texture at the UV centroid, nearest texel; else the base colour factor."""
    if tex is None or UV is None:
        return np.tile(factor * 255, (len(F), 1))
    return _texels(tex, UV[F].mean(1)) * factor


class Item:
    """One GLB: its skinned or static meshes, its skin (joint names, parents, IBMs) and its clips."""

    def __init__(self, path: Path, partmap: dict | None = None) -> None:
        self.path = Path(path)
        self.doc, self.bin = rig.read_glb(self.path)
        d, b = self.doc, self.bin
        self.nodes = d["nodes"]
        self.parent = {c: i for i, n in enumerate(self.nodes) for c in n.get("children", [])}
        skin = d["skins"][0] if d.get("skins") else None
        self.joints = list(skin["joints"]) if skin else []
        self.names = [self.nodes[j].get("name", f"j{j}") for j in self.joints]
        if skin and "inverseBindMatrices" in skin:
            self.ibm = rig.accessor(d, b, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(float)
        else:
            self.ibm = np.array([np.linalg.inv(m) for m in self._world({})[self.joints]]) if skin else np.zeros((0, 4, 4))
        jset = {j: k for k, j in enumerate(self.joints)}
        self.jparent = [jset.get(self.parent.get(j), -1) for j in self.joints]
        self.roles = [role(n, partmap) for n in self.names]
        rest_world = self._world({})
        self.meshes = []  # dicts: P, F, col, J, W, skinned, M (static node matrix)
        for i, n in enumerate(self.nodes):
            if "mesh" not in n or n.get("name", "").startswith("Icosphere"):
                continue
            for pr in d["meshes"][n["mesh"]]["primitives"]:
                if pr.get("mode", 4) != 4:
                    continue
                a = pr["attributes"]
                P = rig.accessor(d, b, a["POSITION"]).astype(float)
                F = (rig.accessor(d, b, pr["indices"]).reshape(-1, 3) if "indices" in pr
                     else np.arange(len(P) - len(P) % 3).reshape(-1, 3)).astype(np.int64)
                UV = rig.accessor(d, b, a["TEXCOORD_0"]).astype(float) if "TEXCOORD_0" in a else None
                factor, gltf_factor, tex = _material(d, b, pr)
                # col is the per-face colour (texture mode); UV, tex and glTF's factor stay for texture_px's
                # per-pixel sampling, so each texture is decoded once whichever mode draws it.
                m = {"P": P, "F": F, "col": _texture_colours(factor, tex, UV, F), "UV": UV, "tex": tex,
                     "factor": gltf_factor, "skinned": "skin" in n and skin is not None and "JOINTS_0" in a,
                     "M": rest_world[i]}
                if m["skinned"]:
                    m["J"] = rig.accessor(d, b, a["JOINTS_0"]).astype(int)
                    m["W"] = rig.accessor(d, b, a["WEIGHTS_0"]).astype(float)
                    m["W"] = m["W"] / np.maximum(m["W"].sum(1, keepdims=True), 1e-12)
                self.meshes.append(m)
        self.clips = {}
        for an in d.get("animations", []):
            dur, ch = rig_motion.channels(d, b, an)
            self.clips[an.get("name", f"anim{len(self.clips)}")] = (dur, ch)

    def _world(self, local: dict) -> np.ndarray:
        return np.array(rig_motion.world_mats(self.nodes, self.parent, local))

    def pose(self, clip: str | None, t: float) -> tuple[list[np.ndarray], np.ndarray]:
        """Posed mesh vertices (one array per mesh) and joint world positions (B,3). An unknown clip gives rest."""
        local = rig_motion.sample(self.nodes, self.clips[clip][1], t) if clip in self.clips else {}
        wm = self._world(local)
        jw = wm[self.joints] if self.joints else np.zeros((0, 4, 4))
        mats = jw @ self.ibm if len(self.joints) else None
        out = []
        for m in self.meshes:
            if m["skinned"]:
                out.append(rig_metrics.skin_frames(m["P"], m["J"], m["W"], mats)[0])
            else:
                out.append(m["P"] @ m["M"][:3, :3].T + m["M"][:3, 3])
        return out, jw[:, :3, 3] if len(self.joints) else np.zeros((0, 3))

    def top_roles(self, m: dict) -> np.ndarray:
        if not m["skinned"]:
            return np.array(["other"] * len(m["P"]))
        top = m["J"][np.arange(len(m["P"])), m["W"].argmax(1)]
        return np.array(self.roles, dtype=object)[top]

    def bone_weight(self, m: dict, bone: str) -> np.ndarray:
        if bone not in self.names:
            raise ValueError(f"heat: no bone {bone!r} in {self.path.name}")
        if not m["skinned"]:
            return np.zeros(len(m["P"]))
        k = self.names.index(bone)
        return (m["W"] * (m["J"] == k)).sum(1)


# ---------------------------------------------------------------- one image

def _shade(Vf: np.ndarray, F: np.ndarray, view: str) -> np.ndarray:
    r, u, f = basis(view)
    L = -0.35 * r + 0.55 * u - 1.0 * f
    L /= np.linalg.norm(L)
    T = Vf[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return 0.4 + 0.6 * np.abs(n @ L)


def draw(items: list[Item], posed: list[tuple[list[np.ndarray], np.ndarray]], view: str, mode: str, cam: dict,
         floor: bool = True, caption: str | None = None) -> Image.Image:
    """One image of an already posed scene."""
    n = cam["size"]
    img = np.zeros((n, n, 3))
    img[:] = BG
    pil = Image.fromarray(img.astype(np.uint8))
    if floor:
        dr = ImageDraw.Draw(pil)
        if view in EDGE_ON:
            r0 = int(round(floor_row(cam, view)))
            dr.rectangle([0, r0, n - 1, r0 + 1], fill=FLOOR)
        else:
            lo, hi = np.asarray(cam["bbox"][0]), np.asarray(cam["bbox"][1])
            c, half = (lo + hi) / 2, 0.625 * (hi - lo) + 1e-6
            q = np.array([[c[0] - half[0], 0, c[2] - half[2]], [c[0] + half[0], 0, c[2] - half[2]],
                          [c[0] + half[0], 0, c[2] + half[2]], [c[0] - half[0], 0, c[2] + half[2]]])
            pq = project(q, view, cam)
            dr.polygon([(float(x), float(y)) for x, y, _ in pq], outline=FLOOR, width=2)
    base = np.asarray(pil).astype(float).reshape(-1, 3)

    # all meshes of all items into one z-buffer
    allV, allF, cols = [], [], []
    off = 0
    heat_bone = mode.split(":", 1)[1] if mode.startswith("heat:") else None
    per_pixel = mode in PX_MODES
    spans = []  # texture_px: (first face, end face, mesh, per-face shade) of each mesh in the shared z-buffer
    nf = 0
    for it, (Vs, _) in zip(items, posed):
        for m, V in zip(it.meshes, Vs):
            F = m["F"]
            shade = np.ones(len(F)) if mode == "texture_px:unlit" else _shade(V, F, view)
            if per_pixel:
                spans.append((nf, nf + len(F), m, shade))
            nf += len(F)
            if mode == "groups":
                rgb = np.array([ROLE_RGB[x] for x in it.top_roles(m)], float)  # per corner
                c = rgb[F] * shade[:, None, None]
            elif heat_bone is not None:
                w = it.bone_weight(m, heat_bone) if heat_bone in it.names else np.zeros(len(m["P"]))
                c = ramp(w)[F] * shade[:, None, None]
            else:
                fc = m["col"]
                if mode == "sticks":
                    fc = 0.5 * fc + 0.5 * 200.0
                c = np.repeat((fc * shade[:, None])[:, None, :], 3, axis=1)
            allV.append(project(V, view, cam))
            allF.append(F + off)
            cols.append(c)
            off += len(V)
    if allV:
        face, bary = raster(np.concatenate(allV), np.concatenate(allF), n)
        C = np.concatenate(cols)  # (T,3 corners,3 rgb)
        face = face.ravel()
        hit = face >= 0
        if hit.any():
            B = bary.reshape(-1, 3)[hit]
            if mode == "groups" or heat_bone is None:
                pick = B.argmax(1)  # nearest corner: flat per-corner colour, no smoothing
                base[hit] = C[face[hit], pick]
            else:
                base[hit] = (C[face[hit]] * B[:, :, None]).sum(1)
            # texture_px: each pixel's own UV (the corners' UVs by its barycentrics), nearest texel. A low-poly
            # mesh's long sliver triangles cross many texels; one texel per face paints each as a stripe.
            fh = face[hit]
            for f0, f1, m, sh in spans:
                if m["tex"] is None or m["UV"] is None:
                    continue  # no texture: the per-face colour above is already the factor
                sel = (fh >= f0) & (fh < f1)
                if not sel.any():
                    continue
                fl = fh[sel] - f0
                uv = (m["UV"][m["F"][fl]] * B[sel][:, :, None]).sum(1)
                rows = np.flatnonzero(hit)[sel]
                base[rows] = _texels(m["tex"], uv) * m["factor"] * sh[fl, None]
    out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8).reshape(n, n, 3)).convert("RGBA")

    if mode == "sticks":
        ov = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        lw = max(2, n // 160)
        br = max(2, n // 110)
        for it, (_, jp) in zip(items, posed):
            if not len(jp):
                continue
            pj = project(jp, view, cam)
            order = np.argsort(-pj[:, 2])  # far first
            for k in order:
                p = it.jparent[k]
                if p >= 0:
                    rgb = ROLE_RGB[it.roles[p]]
                    od.line([(pj[p, 0], pj[p, 1]), (pj[k, 0], pj[k, 1])], fill=(*rgb, 128), width=lw)
            for k in order:
                x, y = pj[k, :2]
                rr = br * (0.7 if it.roles[k] == "root" else 1.0)
                od.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(*ROLE_RGB[it.roles[k]], 128), outline=(0, 0, 0, 128))
        out = Image.alpha_composite(out, ov)
    if caption:
        ImageDraw.Draw(out).text((4, 3), caption, fill=(0, 0, 0, 255))
    return out.convert("RGB")


# ---------------------------------------------------------------- frames

def _loops(it: Item, clip: str, H: float) -> bool:
    dur = it.clips[clip][0]
    a = it.pose(clip, 0.0)[1]
    b = it.pose(clip, dur)[1]
    return bool(np.abs(a - b).max(initial=0) <= 1e-4 * max(H, 1e-9))


def feet_and_beak(it: Item) -> tuple[list[int], list[int]]:
    """Leaf joints with the leg role (feet) and with the head role (beak/mouth tips)."""
    has_child = {p for p in it.jparent if p >= 0}
    leaves = [k for k in range(len(it.names)) if k not in has_child]
    feet = [k for k in leaves if it.roles[k] == "leg"]
    beak = [k for k in leaves if it.roles[k] == "head"]
    return feet, beak


def auto_fracs(it: Item, clip: str, H: float, samples: int = 60) -> list[float]:
    """0, .25, .5, .75 (plus 1.0 for a clip that does not loop), plus events: the time each foot that actually moves
    is lowest, the beak's lowest time if it comes within 0.05H of the floor, and the time the pose (joints about their
    mean, so travel does not count) is furthest from rest (Flap's wings up: 0.25 can fall between two beats)."""
    fr = [0.0, 0.25, 0.5, 0.75]
    if not _loops(it, clip, H):
        fr.append(1.0)
    dur = it.clips[clip][0]
    if dur <= 0:
        return [0.0]
    feet, beak = feet_and_beak(it)
    ts = np.linspace(0, 1, samples, endpoint=False)
    Jt = np.array([it.pose(clip, t * dur)[1] for t in ts])
    Y = Jt[:, :, 1] if len(Jt) and Jt.shape[1] else np.zeros((samples, 0))
    ev = []
    if Jt.ndim == 3 and Jt.shape[1]:
        J0 = it.pose(None, 0.0)[1]
        rel = (Jt - Jt.mean(1, keepdims=True)) - (J0 - J0.mean(0))[None]
        d = np.linalg.norm(rel, axis=2).max(1)
        if d.max() > 0.02 * H:
            ev.append(float(ts[int(d.argmax())]))
    for k in feet:
        if np.ptp(Y[:, k]) > 0.01 * H:
            ev.append(float(ts[int(Y[:, k].argmin())]))
    if beak:
        yb = Y[:, beak].min(1)
        if yb.min() < 0.05 * H:
            ev.append(float(ts[int(yb.argmin())]))
    for e in sorted(ev):
        if all(abs(e - f) > 0.03 for f in fr):
            fr.append(round(e, 4))
    return sorted(fr)


def parse_clips(spec: str | dict | None) -> dict | str:
    """'auto' | "Walk@0.25,0.5;Peck@0.15" | "Walk;Peck" (auto fractions) -> dict clip -> [fractions] or 'auto'."""
    if spec is None or spec == "auto" or isinstance(spec, dict):
        return spec or "auto"
    out: dict = {}
    for part in filter(None, (s.strip() for s in spec.split(";"))):
        name, _, fr = part.partition("@")
        out[name.strip()] = [float(x) for x in fr.split(",") if x.strip()] if fr else "auto"
    return out


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-") or "x"


def mode_tag(mode: str) -> str:
    return _safe(mode.replace(":", "-"))


# ---------------------------------------------------------------- contact sheet

def contact_sheet(pngs: list, rows: int, cols: int, captions: list | None = None, out: Path | str | None = None,
                  tile: int | None = None) -> Image.Image:
    """Grid of images (row-major; None or '-' = blank) with a CAP-pixel caption strip above each tile. Size is
    exactly (cols * tile, rows * (tile + CAP)); tile defaults to the first image's width. Saved to `out` if given."""
    def load(p):
        if p in (None, "-"):
            return None
        if isinstance(p, Image.Image):
            return p
        with Image.open(p) as im:
            return im.convert("RGB")
    ims = [load(p) for p in pngs]
    if tile is None:
        tile = next((im.size[0] for im in ims if im is not None), 256)
    S = Image.new("RGB", (cols * tile, rows * (tile + CAP)), "white")
    d = ImageDraw.Draw(S)
    font = ImageFont.load_default()
    for k in range(min(len(ims), rows * cols)):
        x, y = (k % cols) * tile, (k // cols) * (tile + CAP)
        im = ims[k]
        if im is not None:
            S.paste(im.convert("RGB").resize((tile, tile), Image.BOX if im.size[0] > tile else Image.NEAREST), (x, y + CAP))
        cap = captions[k] if captions and k < len(captions) and captions[k] else (
            Path(pngs[k]).stem if isinstance(pngs[k], (str, Path)) and pngs[k] != "-" else "")
        if cap:
            d.text((x + 3, y + 3), str(cap)[: max(4, tile // 6)], fill="black", font=font)
    if out is not None:
        S.save(out)
    return S


THUMB = "thumb.png"


def sheet_thumb(sheet: Path | str, cols: list, out: Path | str) -> Path:
    """The version thumbnail: the sheet's Rest cell at three-quarters (or its last column), caption strip cut off."""
    with Image.open(sheet) as S:
        tile = S.size[0] // len(cols)
        c = cols.index("q") if "q" in cols else len(cols) - 1
        S.crop((c * tile, CAP, (c + 1) * tile, CAP + tile)).save(out)
    return Path(out)


# ---------------------------------------------------------------- the job

DEFAULT_JOB = {"views": ["side", "xside", "front", "top", "q"], "modes": ["texture", "sticks"], "frames": "auto",
               "floor": True, "size": 384, "sheet_tile": 256}


def asset_dir(asset) -> Path:
    p = Path(asset)
    if p.is_dir():
        return p
    q = ROOT / "out" / str(asset)
    if q.is_dir():
        return q
    raise FileNotFoundError(f"no asset {asset!r} (looked for {p} and {q})")


def render(asset, job: dict | None = None) -> dict:
    """Render a job for an asset (an out/<stem> directory or a stem). job keys (all optional): glbs (paths relative
    to the asset; default [rigged.glb]), views, modes, frames ({clip: [fractions] | 'auto'} or 'auto'), floor, size,
    sheet_tile, partmap, out (output directory; default <asset>/rig/preview), blender (bool), sheet_pick ('peak' =
    the frame with the largest displacement from rest, or a fraction). Returns a summary dict (also frames.json)."""
    A = asset_dir(asset)
    job = {**DEFAULT_JOB, **(job or {})}
    glbs = [Path(g) if Path(g).is_absolute() else A / g for g in job.get("glbs") or ["rigged.glb"]]
    glbs = [g for g in glbs if g.is_file()]
    if not glbs:
        raise FileNotFoundError(f"nothing to render in {A} (glbs {job.get('glbs')})")
    out = Path(job["out"]) if job.get("out") else A / "rig" / "preview"
    out.mkdir(parents=True, exist_ok=True)
    items = [Item(g, job.get("partmap")) for g in glbs]
    prim = next((it for it in items if any(m["skinned"] for m in it.meshes)), next((it for it in items if it.meshes), items[0]))
    rest = [it.pose(None, 0.0) for it in items]
    V0 = np.concatenate(rest[items.index(prim)][0]) if prim.meshes else rest[items.index(prim)][1]
    allpts = np.concatenate([np.concatenate(Vs + [jp]) for Vs, jp in rest])
    H = float(np.ptp(V0[:, 1])) if len(V0) else 1.0
    cam = camera(allpts, int(job["size"]))
    views = list(job["views"])
    modes = list(job["modes"])
    for v in views:
        basis(v)

    frames_spec = parse_clips(job["frames"])
    if frames_spec == "auto":  # every clip except the reference Rest (the Rest row is the bind pose)
        frames_spec = {c: "auto" for c in prim.clips if c != "Rest"}
    plan: list[tuple[str, str | None, float]] = [("Rest", None, 0.0)]
    for clip, fr in frames_spec.items():
        if clip not in prim.clips:
            print(f"rig_render: no clip {clip!r} in {prim.path.name}; skipped", file=sys.stderr)
            continue
        fr = auto_fracs(prim, clip, H) if fr == "auto" else fr
        plan += [(clip, clip, float(f)) for f in fr]

    feet, beak = feet_and_beak(prim)
    leg_roles = None
    frames = []
    Vstack = []
    for label, clip, frac in plan:
        t = frac * prim.clips[clip][0] if clip else 0.0
        posed = [it.pose(clip, t) for it in items]
        Vp = np.concatenate(posed[items.index(prim)][0]) if prim.meshes else posed[items.index(prim)][1]
        Vstack.append(Vp)
        jp = posed[items.index(prim)][1]
        rec = {"clip": label, "frac": round(frac, 4), "t": round(t, 4), "pngs": {}}
        if feet:
            rec["stance_foot_y_rel"] = round(float(jp[feet, 1].min() / H), 4)
            rec["feet_y_rel"] = {prim.names[k]: round(float(jp[k, 1] / H), 4) for k in feet}
        if prim.meshes and prim.meshes[0]["skinned"]:
            if leg_roles is None:
                leg_roles = np.concatenate([prim.top_roles(m) == "leg" for m in prim.meshes])
            if leg_roles.any():
                rec["foot_mesh_y_rel"] = round(float(Vp[leg_roles, 1].min() / H), 4)
        rec["beak_y_rel"] = round(float(jp[beak, 1].min() / H), 4) if beak else None
        for view in views:
            for mode in modes:
                im = draw(items, posed, view, mode, cam, bool(job["floor"]), caption=f"{label} {frac:.2f} {view} {mode}")
                p = out / f"{_safe(label)}_{frac:.2f}_{view}_{mode_tag(mode)}.png"
                im.save(p)
                rec["pngs"][f"{view}:{mode}"] = str(p)
        frames.append(rec)
        print(f"rig_render: {label} {frac:.2f}", file=sys.stderr)
    st = rig_metrics.frame_stats(np.array(Vstack), V0, H)
    for k, rec in enumerate(frames):
        rec["height_rel"] = st["height_rel"][k]
        rec["tilt"] = st["long_axis_tilt_deg"][k]
        rec["min_y_rel"] = st["min_y_rel"][k]
        rec["max_disp_rel"] = st["max_disp_rel"][k]

    # contact sheet: Rest then one frame per clip; columns side, front, q (whatever of them was rendered)
    sv = [v for v in SHEET_VIEWS if v in views] or views[:3]
    smode = "sticks" if "sticks" in modes else modes[0]
    rows = []
    for label in ["Rest"] + [c for c in dict.fromkeys(f["clip"] for f in frames) if c != "Rest"]:
        cand = [f for f in frames if f["clip"] == label]
        pick = job.get("sheet_pick", "peak")
        if pick == "peak":
            best = max(cand, key=lambda f: f["max_disp_rel"])
        else:
            best = min(cand, key=lambda f: abs(f["frac"] - float(pick)))
        rows.append(best)
    cells, caps = [], []
    for f in rows:
        for v in sv:
            cells.append(f["pngs"][f"{v}:{smode}"])
            caps.append(f"{f['clip']} {f['frac']:.2f} {v}")
    sheet = out / "contact_sheet.png"
    contact_sheet(cells, len(rows), len(sv), caps, sheet, tile=int(job["sheet_tile"]))
    sheet_thumb(sheet, sv, out / THUMB)

    result = {"asset": str(A), "glbs": [str(g) for g in glbs], "out": str(out), "H": round(H, 6), "camera": cam,
              "floor_rows": {v: round(floor_row(cam, v), 3) for v in views if v in EDGE_ON}, "view_right": VIEW_RIGHT,
              "views": views, "modes": modes, "clips": list(dict.fromkeys(f["clip"] for f in frames if f["clip"] != "Rest")),
              "sheet": str(sheet), "sheet_rows": [f"{f['clip']}@{f['frac']}" for f in rows], "sheet_cols": sv,
              "sheet_mode": smode, "frames": frames,
              "min_y_rel_min": round(min(f["min_y_rel"] for f in frames), 4)}
    if job.get("blender"):
        result["blender"] = render_blender(items, plan, views, modes, cam, out / "blender", bool(job["floor"]),
                                           job.get("partmap"), job.get("pose"))
    (out / "frames.json").write_text(json.dumps(result, indent=1))
    return result


# ---------------------------------------------------------------- textured views (facing)

def textured_views(glb, views=("side", "xside", "front", "back"), size: int = 384, out_dir=None,
                   floor: bool = False, mode: str = "texture") -> dict:
    """Rest-pose textured renders of any GLB (skinned or static) for the facing VLM. Returns {view: png path};
    see VIEW_RIGHT / the module docstring for which world axis is image right in each view. mode is texture (one
    texel a face, what the facing VLM has always been shown) or one of PX_MODES (per pixel)."""
    if mode != "texture" and mode not in PX_MODES:
        raise ValueError(f"textured_views: mode {mode!r}; texture or one of {PX_MODES}")
    it = Item(Path(glb))
    Vs, jp = it.pose(None, 0.0)
    pts = np.concatenate(Vs) if Vs else jp
    cam = camera(pts, size)
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="rig_render_"))
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for v in views:
        p = out / f"{Path(glb).stem}_{v}.png"
        draw([it], [(Vs, jp)], v, mode, cam, floor).save(p)
        res[v] = str(p)
    return res


# ---------------------------------------------------------------- Blender parity

BL_SCRIPT = Path(__file__).resolve().parent / "blender" / "rig_render_bl.py"


def render_blender(items: list[Item], plan: list, views: list, modes: list, cam: dict, out: Path, floor: bool = True,
                   partmap: dict | None = None, pose: dict | None = None, timeout: int = 1800) -> dict:
    """The same frames through Blender Workbench (parity only). Same centre, ortho scale and resolution as the CPU
    camera; transparent film composited over BG here. Returns {ok, pngs, log_tail}."""
    exe = rig.blender_path()
    if not exe:
        return {"ok": False, "why": "no Blender"}
    out.mkdir(parents=True, exist_ok=True)
    c = cam["centre"]
    roles = {n: r for it in items for n, r in zip(it.names, it.roles)}
    roles.update(partmap or {})
    lo, hi = np.asarray(cam["bbox"][0]), np.asarray(cam["bbox"][1])
    mid, half = (lo + hi) / 2, 0.625 * (hi - lo) + 1e-6
    fbox = [[mid[0] - half[0], -(mid[2] + half[2])], [mid[0] + half[0], -(mid[2] - half[2])]]
    cams = {v: dict(zip(("right", "up", "fwd"), (b.tolist() for b in basis(v)))) for v in views}
    jobs = []
    for label, clip, frac in plan:
        t = frac * items[0].clips[clip][0] if clip and clip in items[0].clips else 0.0
        for mode in modes:
            jobs.append({"glbs": [str(it.path) for it in items], "clip": clip, "t": t, "rest": clip is None,
                         "mode": mode, "views": views, "size": cam["size"], "fps": FPS,
                         "centre": [c[0], -c[2], c[1]], "ortho": cam["size"] / cam["scale"],
                         "floor": floor, "floor_thick": 2.0 / cam["scale"], "partmap": roles, "pose": pose or {}, "cams": cams,
                         "floor_box": fbox,
                         "role_rgb": {k: [x / 255 for x in v] for k, v in ROLE_RGB.items()},
                         "out_prefix": str(out / f"{_safe(label)}_{frac:.2f}_{{view}}_{mode_tag(mode)}")})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(jobs, fh)
    proc = subprocess.run([exe, "-b", "--factory-startup", "-P", str(BL_SCRIPT), "--", fh.name],
                          capture_output=True, text=True, timeout=timeout)
    Path(fh.name).unlink(missing_ok=True)
    pngs = []
    for j in jobs:
        for v in views:
            p = Path(j["out_prefix"].replace("{view}", v) + ".png")
            if p.is_file():
                im = Image.open(p).convert("RGBA")
                bg = Image.new("RGBA", im.size, (*BG, 255))
                Image.alpha_composite(bg, im).convert("RGB").save(p)
                pngs.append(str(p))
    tail = (proc.stdout + proc.stderr)[-2000:]
    return {"ok": proc.returncode == 0 and "RIG_RENDER_BL_OK" in proc.stdout and len(pngs) == len(jobs) * len(views),
            "pngs": pngs, "log_tail": tail}


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stem", help="out/<stem>")
    g.add_argument("--asset", help="an asset directory (e.g. a scratch copy of out/<stem>)")
    ap.add_argument("--views", default=",".join(DEFAULT_JOB["views"]))
    ap.add_argument("--modes", default=",".join(DEFAULT_JOB["modes"]))
    ap.add_argument("--clips", default="auto", help="auto | 'Walk@0.25,0.5;Peck@0.15'")
    ap.add_argument("--glbs", default="rigged.glb", help="comma list, relative to the asset")
    ap.add_argument("--size", type=int, default=DEFAULT_JOB["size"])
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--out", help="output directory (default <asset>/rig/preview)")
    ap.add_argument("--blender", action="store_true", help="also render through Blender Workbench (parity)")
    a = ap.parse_args()
    job = {"views": [v for v in a.views.split(",") if v], "modes": [m for m in a.modes.split(",") if m],
           "frames": parse_clips(a.clips), "glbs": [x for x in a.glbs.split(",") if x], "size": a.size,
           "floor": not a.no_floor, "blender": a.blender}
    if a.out:
        job["out"] = a.out
    res = render(a.asset or a.stem, job)
    slim = {k: res[k] for k in ("out", "sheet", "sheet_rows", "sheet_cols", "clips", "min_y_rel_min", "H")}
    slim["images"] = sum(len(f["pngs"]) for f in res["frames"])
    if "blender" in res:
        slim["blender"] = {k: v for k, v in res["blender"].items() if k != "pngs"}
        slim["blender"]["images"] = len(res["blender"].get("pngs", []))
    print(json.dumps(slim, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
