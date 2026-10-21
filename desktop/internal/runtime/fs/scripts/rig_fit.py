#!/usr/bin/env python3
"""Fit a body-plan template to a mesh before skinning (used by rig.py's posed_template).

1. Facing: the template's own body (its mesh, skinned at the reference frame) is registered to the mesh at yaw
   0/90/180/270 with a per-axis scale and offset (bounding box, then ICP); symmetric Chamfer / height scores each.
   When the runner-up is within 1.3x of the best, independent cues pick among the close yaws: where the toes
   point, which end of the body is narrower, a small concentrated patch of red (comb, wattle), and the local
   vision model asked which side the head is on in two side renders.
2. Scale: the registration's per-axis scale (template axes), instead of one height-only scale.
3. Chains: legs onto the mesh's floor clusters, neck/head onto the head end, tail onto the far end, folded wings
   onto the body sides; bone lengths clamped to 0.7-1.6x and bends capped at 50 degrees (wings excepted: they are
   laid flat). Joints in thin parts are centred between the mesh's front/back (or left/right) ray hits, as
   Mesh2Motion's mesh-volume drag does (MeshDragBonePlacement.ts).
4. Pose gate: share of joints outside the mesh's box, and share of bone samples outside the mesh and farther
   than 0.08x height from its surface (generalised winding number).

Everything here works in the TEMPLATE frame (+Z forward, +Y up, template units): the mesh is mapped into it by
the inverse of T(p) = Ry(yaw) (p * s) + offset, so the chain fits see the mesh the way the template does.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

YAWS = (0, 90, 180, 270)
TIE = 1.3            # runner-up within this factor of the best Chamfer: ask the cues
POOR_MATCH = 1.6     # max/min per-axis scale above this: the template is a poor body-plan match
LEN_CLAMP = (0.7, 1.6)
BEND_CAP = np.radians(50)

# Per class: chains of joint names, root first. "key" is the joint aimed at the target.
CHAINS = {
    "avian": {
        "legs": [["UpperLeg_L", "LowerLeg_L", "AnkleLeg_L", "Foot_L", "Toes_L", "Toes_tip_L"],
                 ["UpperLeg_R", "LowerLeg_R", "AnkleLeg_R", "Foot_R", "Toes_R", "Toes_tip_R"]],
        "neck": ["spine_2", "spine_3", "head", "mouth_upper", "mouth_upper_tip"],
        "tail": ["tail_1", "tail_2", "tail_3", "tail_tip"],
        "wings": [["wing_1_L", "wing_2_L", "wing_3_L", "wing_4_L", "wing_5_L", "wing_tip_L"],
                  ["wing_1_R", "wing_2_R", "wing_3_R", "wing_4_R", "wing_5_R", "wing_tip_R"]],
    },
    "quadruped": {
        "legs": [["front_scapula_l", "front_humerus_l", "front_leg_upper_l", "front_leg_lower_l", "front_leg_ankle_l",
                  "front_leg_foot_l", "front_leg_leaf_l"],
                 ["front_scapula_r", "front_humerus_r", "front_leg_upper_r", "front_leg_lower_r", "front_leg_ankle_r",
                  "front_leg_foot_r", "front_leg_leaf_r"],
                 ["back_leg_pelvis_l", "back_leg_upper_l", "back_leg_lower_l", "back_leg_ankle_l", "back_leg_foot_l",
                  "back_leg_toe_l", "back_leg_leaf_l"],
                 ["back_leg_pelvis_r", "back_leg_upper_r", "back_leg_lower_r", "back_leg_ankle_r", "back_leg_foot_r",
                  "back_leg_toe_r", "back_leg_leaf_r"]],
        "neck": None,  # rig.fit_head_chain (horse HEAD_CHAIN) already does this one
        "tail": ["tail_1", "tail_2", "tail_3", "tail_4", "tail_leaf"],
        "wings": [],
    },
    "humanoid": {
        "legs": [["thigh_l", "calf_l", "foot_l", "ball_l", "ball_leaf_l"], ["thigh_r", "calf_r", "foot_r", "ball_r", "ball_leaf_r"]],
        "neck": None, "tail": None, "wings": [],
    },
}


# ---------------------------------------------------------------- template body and registration

def template_body(m2m: Path, template: dict) -> tuple[np.ndarray, np.ndarray]:
    """The template's own mesh skinned at frame 0 of its reference clip (vertices, faces), template frame."""
    import rig
    import rig_motion
    doc, b = rig.read_glb(m2m / template["clips"][0])
    ref = template["ref"] if isinstance(template["ref"], str) else None
    nodes = doc["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    mi = next(i for i, n in enumerate(nodes) if "mesh" in n and "skin" in n)
    pr = doc["meshes"][nodes[mi]["mesh"]]["primitives"][0]
    a = pr["attributes"]
    P = rig.accessor(doc, b, a["POSITION"]).astype(float)
    J = rig.accessor(doc, b, a["JOINTS_0"]).astype(int)
    W = rig.accessor(doc, b, a["WEIGHTS_0"]).astype(float)
    F = rig.accessor(doc, b, pr["indices"]).reshape(-1, 3)
    skn = doc["skins"][nodes[mi]["skin"]]
    IBM = rig.accessor(doc, b, skn["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    local = {}
    if ref:
        anim = next(x for x in doc["animations"] if x["name"] == ref)
        _, ch = rig_motion.channels(doc, b, anim)
        local = rig_motion.sample(nodes, ch, 0.0)
    wm = rig_motion.world_mats(nodes, parent, local)
    M = np.stack([wm[j] @ IBM[k] for k, j in enumerate(skn["joints"])])
    V = rig_motion.skin(np.c_[P, np.ones(len(P))], J, W, M)
    # the mesh node's own transform (if any) is applied by world_mats only for joints; the vertices are in skin space
    return V, F


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _sample(V: np.ndarray, F: np.ndarray, n: int = 6000) -> np.ndarray:
    import trimesh
    return np.asarray(trimesh.Trimesh(V, F, process=False).sample(n, seed=0), float)


def register(TV: np.ndarray, TF: np.ndarray, ms: np.ndarray, rounds: int = 8) -> dict:
    """Per yaw: Chamfer/height, per-axis scale (WORLD axes, i.e. after the yaw), offset. ms: mesh surface samples."""
    ms = ms[np.random.default_rng(0).choice(len(ms), min(len(ms), 6000), replace=False)]
    H = float(np.ptp(ms[:, 1]))
    mt = cKDTree(ms)
    res = {}
    for yaw in YAWS:
        Ry = yaw_matrix(yaw)
        ts0 = _sample(TV @ Ry.T, TF)
        sc = np.ptp(ms, 0) / np.ptp(ts0, 0)
        off = ms.min(0) + np.ptp(ms, 0) / 2 - (ts0.min(0) + np.ptp(ts0, 0) / 2) * sc
        off[1] = ms[:, 1].min() - ts0[:, 1].min() * sc[1]
        for _ in range(rounds):  # ICP on per-axis scale + offset (no rotation), both directions
            ts = ts0 * sc + off
            _, i1 = mt.query(ts)
            _, i2 = cKDTree(ts).query(ms)
            A = np.r_[ts0, ts0[i2]]
            B = np.r_[ms[i1], ms]
            for ax in range(3):
                X = np.c_[A[:, ax], np.ones(len(A))]
                sol = np.linalg.lstsq(X, B[:, ax], rcond=None)[0]
                sc[ax] = np.clip(sol[0], 0.3 * sc[ax], 3 * sc[ax])
                off[ax] = sol[1]
        ts = ts0 * sc + off
        d1, _ = mt.query(ts)
        d2, _ = cKDTree(ts).query(ms)
        res[yaw] = {"chamfer": round(float((d1.mean() + d2.mean()) / 2 / H), 4), "scale_world": sc.round(4).tolist(),
                    "offset": off.round(4).tolist()}
    return res


# ---------------------------------------------------------------- facing cues

def _texel_colours(doc: dict, binary: bytes) -> np.ndarray | None:
    import io
    import rig
    from PIL import Image
    try:
        pr = doc["meshes"][0]["primitives"][0]
        uv = rig.accessor(doc, binary, pr["attributes"]["TEXCOORD_0"])
        tex = doc["materials"][pr["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        img = doc["images"][doc["textures"][tex]["source"]]
        v = doc["bufferViews"][img["bufferView"]]
        data = binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
        im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:  # noqa: BLE001 - no texture: the colour cue just doesn't vote
        return None
    h, w, _ = im.shape
    return im[(uv[:, 1] % 1 * (h - 1)).astype(int), (uv[:, 0] % 1 * (w - 1)).astype(int)].astype(int)


def geometric_cues(P: np.ndarray, cls: str, colours: np.ndarray | None) -> dict:
    """Signed votes for 'the head is at +Z' along Z (and +X along X), each with the number it came from.
    toes: lowest 3% of points vs the shins above them (feet point forward); only if the shift is > 3% of height.
    narrow: the narrower upper end of the body is the head (not for people: their arms make the ends meaningless).
    red: a small, concentrated patch of red texels (comb, wattle) at one end."""
    H = float(np.ptp(P[:, 1]))
    lo = P[:, 1].min()
    c = (P.min(0) + P.max(0)) / 2
    out = {}
    for ax, name in ((2, "z"), (0, "x")):
        L = float(np.ptp(P[:, ax]))
        votes = {}
        feet = P[P[:, 1] < lo + 0.03 * H]
        shin = P[(P[:, 1] > lo + 0.08 * H) & (P[:, 1] < lo + 0.2 * H)]
        if len(feet) > 5 and len(shin) > 5:
            d = float(feet[:, ax].mean() - shin[:, ax].mean())
            votes["toes"] = {"value": round(d / H, 3), "vote": int(np.sign(d)) if abs(d) > 0.03 * H else 0}
        if cls != "humanoid":
            up = P[:, 1] > lo + 0.5 * H
            other = 2 if ax == 0 else 0
            ends = []
            for sgn in (1, -1):
                S = P[up & (sgn * (P[:, ax] - c[ax]) > 0.3 * L)]
                ends.append(float(np.ptp(S[:, other])) if len(S) > 5 else np.nan)
            if np.all(np.isfinite(ends)) and max(ends) > 0:
                r = min(ends) / max(ends)
                votes["narrow"] = {"value": [round(e, 3) for e in ends], "vote": (1 if ends[0] < ends[1] else -1) if r < 0.8 else 0}
        if colours is not None:
            red = (colours[:, 0] > 150) & (colours[:, 1] < 90) & (colours[:, 2] < 90)
            share = float(red.mean())
            if red.sum() >= 20 and share < 0.05:
                d = float(P[red, ax].mean() - c[ax])
                votes["red"] = {"value": round(d / max(L, 1e-9), 3), "n": int(red.sum()),
                                "vote": int(np.sign(d)) if abs(d) > 0.15 * L else 0}
        out[name] = votes
    return out


VLM_Q = ("This is a rendered side view of a 3D model. If it is a creature or character with a head, is the head (face) on the "
         "LEFT or the RIGHT half of the image? If it has no head or face (an object, prop or box), answer NONE. "
         "Answer with exactly one word: LEFT, RIGHT or NONE.")

RENDER = r"""
import bpy, sys, mathutils
from mathutils import Vector
a = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a[0])
sc = bpy.context.scene
ms = [o for o in bpy.data.objects if o.type == 'MESH']
pts = [o.matrix_world @ v.co for o in ms for v in o.data.vertices]
lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
c = (lo + hi) / 2; size = max((hi - lo).length, 0.1)
sc.render.engine = 'BLENDER_WORKBENCH'; sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'TEXTURE'
sc.render.resolution_x = sc.render.resolution_y = 448
w = bpy.data.worlds.new('w'); sc.world = w; w.color = (0.55, 0.7, 0.9)
cam = bpy.data.cameras.new('c'); cam.type = 'ORTHO'; cam.ortho_scale = size * 0.9
co = bpy.data.objects.new('cam', cam); sc.collection.objects.link(co); sc.camera = co
for name, d in (('side', Vector((1, 0, 0))), ('xside', Vector((-1, 0, 0)))):
    co.location = c + d * 10; f = -d; up = Vector((0, 0, 1)); rt = f.cross(up).normalized(); cu = rt.cross(f).normalized()
    co.rotation_mode = 'QUATERNION'; co.rotation_quaternion = mathutils.Matrix((rt, cu, -f)).transposed().to_quaternion()
    sc.render.filepath = a[1] + '_' + name + '.png'; bpy.ops.render.render(write_still=True)
print('RENDER_OK')
"""


def vlm_cue(source: Path, blender: str | None, out_prefix: Path) -> dict:
    """Ask the local vision model which side the head is on in a +X and a -X side render. Camera at +X: image right
    is -Z; camera at -X: image right is +Z. Votes only when both views agree. keep_alive 0 unloads the model."""
    if not blender:
        return {"vote": 0, "why": "no Blender"}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(RENDER)
    try:
        proc = subprocess.run([blender, "-b", "--factory-startup", "-P", fh.name, "--", str(source), str(out_prefix)],
                              capture_output=True, text=True, timeout=300)
    finally:
        os.unlink(fh.name)
    if "RENDER_OK" not in proc.stdout:
        return {"vote": 0, "why": "render failed"}
    model = os.environ.get("RIG_VLM_MODEL", "qwen3.8:27b")
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
    answers = {}
    for view in ("side", "xside"):
        body = {"model": model, "prompt": VLM_Q, "stream": False, "think": False, "keep_alive": 0, "options": {"temperature": 0},
                "images": [base64.b64encode(Path(f"{out_prefix}_{view}.png").read_bytes()).decode()]}
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
            answers[view] = json.loads(r.read())["response"].strip().upper().split()[0].strip(".")
        except Exception as e:  # noqa: BLE001 - no model: the cue doesn't vote
            return {"vote": 0, "why": f"vision model unavailable ({e})"}
    s1 = {"RIGHT": -1, "LEFT": 1}.get(answers["side"])
    s2 = {"RIGHT": 1, "LEFT": -1}.get(answers["xside"])
    return {"answers": answers, "vote": s1 if s1 is not None and s1 == s2 else 0, "model": model}


PRIOR_CONFLICT = 1.3  # a declared facing is questioned when the registration's best is opposite it by this margin


def choose_yaw(reg: dict, cues: dict, vlm: dict | None, prior: dict | None = None) -> dict:
    """Best Chamfer, unless the runner-up is within TIE: then the cues pick among the close yaws.

    prior ({"yaw": 0, "source": ...}): the facing is already known (a canonical draw, or a psx facing check that
    agreed), so the template is turned by the prior and the registration only verifies it: cue_conflict when its
    best yaw is the opposite one by more than PRIOR_CONFLICT. No VLM call."""
    order = sorted(reg, key=lambda y: reg[y]["chamfer"])
    best = order[0]
    margin = reg[order[1]]["chamfer"] / max(reg[best]["chamfer"], 1e-9)
    close = [y for y in order if reg[y]["chamfer"] <= TIE * reg[best]["chamfer"]]
    out = {"chamfer": {str(y): reg[y]["chamfer"] for y in YAWS}, "margin": round(margin, 3), "close": close,
           "registration_best": best, "cues": cues, "vlm": None, "yaw": best, "decided_by": "registration"}
    if prior is not None:
        y = int(prior.get("yaw", 0)) % 360
        out.update(yaw=y, decided_by="declared", prior=prior)
        if best == (y + 180) % 360 and margin > PRIOR_CONFLICT:
            out["cue_conflict"] = True
            out["prior_conflict"] = f"the registration's best is {best}° (margin {margin:.2f}), opposite the declared facing"
        return out
    if len(close) == 1:
        return out
    if callable(vlm):  # only render and ask when the registration can't decide
        vlm = vlm()
    out["vlm"] = vlm

    def fwd(y: int) -> np.ndarray:
        return yaw_matrix(y) @ np.array([0, 0, 1.0])

    geo, vis = {}, {}
    for y in close:
        f = fwd(y)
        g = 0.0
        for axname, k in (("z", 2), ("x", 0)):
            if abs(f[k]) > 0.5:
                g += float(np.sign(f[k])) * sum(v["vote"] for v in cues.get(axname, {}).values())
        geo[y] = g
        vis[y] = float(np.sign(f[2])) * (vlm or {}).get("vote", 0) if abs(f[2]) > 0.5 else 0.0
    out["cue_scores"] = {str(y): {"geometric": geo[y], "vlm": vis[y]} for y in close}
    total = {y: geo[y] + 2 * vis[y] for y in close}  # the vision model's two-view agreement counts double
    top = max(total.values())
    if top <= 0:
        out["decided_by"] = "registration (no cue voted)"
        return out
    winners = [y for y in close if total[y] == top]
    pick = min(winners, key=lambda y: reg[y]["chamfer"])
    out["yaw"] = pick
    who = [n for n, s in (("geometric", geo[pick]), ("vlm", vis[pick])) if s > 0]
    out["decided_by"] = "+".join(who) if who else "registration"
    if geo[pick] < 0 or vis[pick] < 0:
        out["cue_conflict"] = True
    return out


# ---------------------------------------------------------------- chain fitting (template frame)

def _rot_between(u: np.ndarray, v: np.ndarray, cap: float | None = BEND_CAP) -> tuple[np.ndarray, float]:
    u = u / max(np.linalg.norm(u), 1e-12)
    v = v / max(np.linalg.norm(v), 1e-12)
    ax = np.cross(u, v)
    s = np.linalg.norm(ax)
    ang = float(np.arctan2(s, u @ v))
    if s < 1e-9:
        return np.eye(3), 0.0
    ax /= s
    a = min(ang, cap) if cap is not None else ang
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K, ang


class Rays:
    """Axis-aligned rays through the mesh (template frame), for centring joints between front/back hits."""

    def __init__(self, V: np.ndarray, F: np.ndarray) -> None:
        import trimesh
        self.mesh = trimesh.Trimesh(V, F, process=False)
        self.ray = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        self.size = float(np.ptp(V, 0).max())

    def interval(self, p: np.ndarray, axis: int) -> tuple[float, float] | None:
        """The (entry, exit) pair of hits along `axis` through p that contains p, or the nearest one."""
        d = np.zeros(3)
        d[axis] = 1.0
        o = p.copy()
        o[axis] -= 2 * self.size
        locs, _, _ = self.ray.intersects_location([o], [d], multiple_hits=True)
        if len(locs) < 2:
            return None
        h = np.sort(np.unique(np.round(locs[:, axis], 6)))
        if len(h) < 2:
            return None
        pairs = [(h[i], h[i + 1]) for i in range(0, len(h) - 1, 2)]
        x = p[axis]
        inside = [q for q in pairs if q[0] <= x <= q[1]]
        if inside:
            return inside[0]
        return min(pairs, key=lambda q: min(abs(q[0] - x), abs(q[1] - x)))


def _centre(P: dict, name: str, rays: Rays, axes: tuple[int, ...], thin: float, cap: float) -> float:
    """Move joint `name` to the midpoint of its hit interval along each axis, when that interval is thinner than
    `thin` (a leg or neck, not the body) and the move is under `cap`. Returns how far it moved."""
    p = P[name].copy()
    for ax in axes:
        iv = rays.interval(p, ax)
        if iv is None or iv[1] - iv[0] > thin:
            continue
        mid = (iv[0] + iv[1]) / 2
        if abs(mid - p[ax]) <= cap:
            p[ax] = mid
    moved = float(np.linalg.norm(p - P[name]))
    P[name] = p
    return moved


def _descendants(children: dict, name: str) -> list[str]:
    out = []
    for c in children.get(name, []):
        out += [c] + _descendants(children, c)
    return out


def _move_chain(P: dict, children: dict, chain: list[str], new: dict[str, np.ndarray]) -> None:
    """Set chain joints to `new` positions; every other descendant follows its nearest chain ancestor rigidly
    (the rotation and length change of that ancestor's bone)."""
    old = {n: P[n].copy() for n in P}
    inchain = set(chain)
    for k, n in enumerate(chain):
        if n not in new:
            continue
        P[n] = new[n]
    for k, n in enumerate(chain):
        nxt = chain[k + 1] if k + 1 < len(chain) else None
        prv = chain[k - 1] if k > 0 else None
        if nxt is not None:
            u, v = old[nxt] - old[n], P[nxt] - P[n]
        elif prv is not None:
            u, v = old[n] - old[prv], P[n] - P[prv]
        else:
            continue
        Rm, _ = _rot_between(u, v, None)
        s = np.linalg.norm(v) / max(np.linalg.norm(u), 1e-12)
        for d in _descendants(children, n):
            if d in inchain:
                continue
            # follow the nearest chain ancestor only
            P[d] = P[n] + s * (Rm @ (old[d] - old[n]))
        # (descendants that hang off a later chain joint are overwritten when that joint is processed)


def fit_to_target(P: dict, children: dict, chain: list[str], key: str, target: np.ndarray) -> dict:
    """Turn (capped at 50 degrees) and scale (0.7-1.6x) the chain about its root so `key` reaches `target`."""
    root = chain[0]
    u, v = P[key] - P[root], target - P[root]
    Rm, ang = _rot_between(u, v)
    ratio = float(np.linalg.norm(v) / max(np.linalg.norm(u), 1e-12))
    s = float(np.clip(ratio, *LEN_CLAMP))
    k = chain.index(key)
    before = chain[1:k + 1]
    foot = [key] + _descendants(children, key)  # the key's own subtree (foot, toes) only moves: it stays flat
    new = {n: P[root] + s * (Rm @ (P[n] - P[root])) for n in before}
    shift = new[key] - P[key]
    for n in foot[1:]:
        new[n] = P[n] + shift
    # descendants of the turned joints that aren't in the chain move with the same similarity
    others = {d for n in before for d in _descendants(children, n)} - set(new) - set(chain)
    for d in others:
        P[d] = P[root] + s * (Rm @ (P[d] - P[root]))
    for n, p in new.items():
        P[n] = p
    return {"turn_deg": round(float(np.degrees(ang)), 1), "turn_capped": bool(ang > BEND_CAP), "length_ratio": round(ratio, 3),
            "length_clamped": bool(ratio != s), "residual": round(float(np.linalg.norm(P[key] - target)), 4)}


def fit_to_line(P: dict, children: dict, chain: list[str], line: np.ndarray) -> dict:
    """Lay the chain along `line` (from its root): the chain's total length is stretched to the line's
    (clamped 0.7-1.6x) and each joint lands at its arc length, each bone turned at most 50 degrees."""
    seg = [np.linalg.norm(P[chain[k + 1]] - P[chain[k]]) for k in range(len(chain) - 1)]
    tl = float(sum(seg))
    ml = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    ratio = ml / max(tl, 1e-12)
    s = float(np.clip(ratio, *LEN_CLAMP))
    import rig
    new = {chain[0]: P[chain[0]].copy()}
    capped = 0
    acc = 0.0
    for k in range(len(chain) - 1):
        acc += seg[k] * s
        want = rig._at_arclen(line, acc)
        a = new[chain[k]]
        oldd = P[chain[k + 1]] - P[chain[k]]
        Rm, ang = _rot_between(oldd, want - a)
        capped += ang > BEND_CAP
        new[chain[k + 1]] = a + Rm @ (oldd / max(np.linalg.norm(oldd), 1e-12)) * seg[k] * s
    tip_res = float(np.linalg.norm(new[chain[-1]] - line[-1]))
    _move_chain(P, children, chain, new)
    return {"length_ratio": round(ratio, 3), "length_clamped": bool(ratio != s), "bends_capped": int(capped),
            "tip_residual": round(tip_res, 4)}


def end_line(mesh_t: np.ndarray, start: np.ndarray, sign: float, y_min: float) -> np.ndarray:
    """The mesh's centreline from `start` to its far end along sign*Z, through the upper half of each slice
    (rig._centreline, mirrored for the back end)."""
    import rig
    M = np.diag([sign, 1.0, sign])
    line, _ = rig._centreline(mesh_t @ M, start @ M, y_min)
    return line @ M


def floor_clusters(mesh_t: np.ndarray, seeds: np.ndarray, share: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Lowest `share` of the points, one cluster per template foot: a point goes to the nearest foot (in Z) on its
    own side (left/right of the middle). Returns cluster centres (x, z) and sizes; an empty cluster keeps its seed."""
    y = mesh_t[:, 1]
    low = mesh_t[y <= np.quantile(y, share)]
    cx = float(np.median(mesh_t[:, 0]))
    side_low = np.sign(low[:, 0] - cx)
    side_seed = np.sign(seeds[:, 0] - cx)
    cost = np.abs(low[:, None, 2] - seeds[None, :, 2]) + 1e3 * (side_low[:, None] != side_seed[None])
    lab = cost.argmin(1)
    C = seeds[:, [0, 2]].astype(float).copy()
    for _ in range(3):  # re-centre in z per side, so a staggered template stance doesn't split one hoof
        for k in range(len(C)):
            if (lab == k).any():
                C[k] = low[lab == k][:, [0, 2]].mean(0)
        cost = np.abs(low[:, None, 2] - C[None, :, 1]) + 1e3 * (side_low[:, None] != side_seed[None])
        lab = cost.argmin(1)
    for k in range(len(C)):
        if (lab == k).any():
            C[k] = low[lab == k][:, [0, 2]].mean(0)
    return C, np.bincount(lab, minlength=len(C))


def outside_score(P: dict, chain: list[str], tV: np.ndarray, F: np.ndarray, tree: cKDTree, H_t: float) -> float:
    """How far the chain's bones stray outside the mesh: mean over samples along its bones of the distance to the
    surface for samples outside (winding number < 0.5), 0 inside; / height."""
    t = np.linspace(0, 1, 7)
    pts = np.concatenate([P[a][None] * (1 - t[:, None]) + P[b][None] * t[:, None] for a, b in zip(chain[:-1], chain[1:])])
    wn = winding_number(tV, F, pts, chunk=4000)
    d = tree.query(pts)[0]
    return float(np.where(np.abs(wn) > 0.5, 0.0, d).mean() / H_t)


def fit_chains(cls: str, P: dict, children: dict, mesh_t: np.ndarray, tV: np.ndarray, tF: np.ndarray, H_t: float) -> list[dict]:
    """All chain fits for the class, in the template frame. P: joint name -> position (changed in place)."""
    spec = CHAINS.get(cls)
    if not spec:
        return []
    rays = Rays(tV, tF)
    tree = cKDTree(mesh_t)
    report = []
    floor_t = float(mesh_t[:, 1].min())
    # legs: the floor joint (first chain joint below 8% of the template's height) goes onto its floor cluster
    legs = [c for c in spec["legs"] if all(n in P for n in c)]
    if legs:
        lowj = min(P[n][1] for n in P)
        keys = []
        for c in legs:
            k = next((n for n in c if P[n][1] - lowj < 0.08 * H_t), c[-1])
            keys.append(k)
        foot_groups = [[k] + [d for d in _descendants(children, k)] for k in keys]
        seeds = np.array([np.mean([P[n] for n in g], 0) for g in foot_groups])
        C, counts = floor_clusters(mesh_t, seeds)
        for c, k, g, cc, cnt in zip(legs, keys, foot_groups, C, counts):
            gc = np.mean([P[n] for n in g], 0)
            target = np.array([cc[0] + P[k][0] - gc[0], P[k][1] - lowj + floor_t, cc[1] + P[k][2] - gc[2]])
            r = fit_to_target(P, children, c, k, target)
            mid = c[1:c.index(k)]
            moves = [_centre(P, n, rays, (0, 2), thin=0.3 * H_t, cap=0.15 * H_t) for n in mid]
            report.append({"chain": c[0], "kind": "leg", "key": k, "cluster_points": int(cnt), **r,
                           "centred_max": round(max(moves, default=0.0), 4)})
    # neck/head onto the head end, tail onto the far end
    for kind, sign in (("neck", 1.0), ("tail", -1.0)):
        c = spec.get(kind)
        if not c or not all(n in P for n in c):
            continue
        y_min = floor_t + 0.6 * (P[c[0]][1] - floor_t)
        try:
            line = end_line(mesh_t, P[c[0]], sign, y_min)
        except Exception as e:  # noqa: BLE001 - too few points that way: keep the template's chain
            report.append({"chain": c[0], "kind": kind, "skipped": str(e)})
            continue
        before = {n: P[n].copy() for n in P}
        s0 = outside_score(P, c, tV, tF, tree, H_t)
        r = fit_to_line(P, children, c, line)
        moves = [_centre(P, n, rays, (0,), thin=0.5 * H_t, cap=0.15 * H_t) for n in c[1:]]
        s1 = outside_score(P, c, tV, tF, tree, H_t)
        kept = s1 <= s0 + 0.01  # a fit that pulls the chain out of the body (a cow's hanging tail) is undone
        if not kept:
            P.update(before)
        report.append({"chain": c[0], "kind": kind, **r, "centred_max": round(max(moves, default=0.0), 4),
                       "outside_before": round(s0, 4), "outside_after": round(s1, 4), "kept": bool(kept)})
    # wings: folded, laid flat just inside the body's sides
    surf = mesh_t
    cx = float(np.median(surf[:, 0]))
    for c in spec.get("wings", []):
        if not all(n in P for n in c):
            continue
        side = 1.0 if P[c[-1]][0] > cx else -1.0
        S = surf[side * (surf[:, 0] - cx) > 0]
        tree = cKDTree(S[:, 1:])
        new = {}
        moved = []
        for n in c:
            p = P[n].copy()
            d, i = tree.query(p[1:])
            if d > 0.03 * H_t:  # outside the side silhouette: onto it
                p[1:] = S[i, 1:]
            near = S[np.linalg.norm(S[:, 1:] - p[1:], axis=1) < 0.06 * H_t]
            if len(near):
                edge = float((side * (near[:, 0] - cx)).max())
                p[0] = cx + side * 0.85 * edge
            new[n] = p
            moved.append(float(np.linalg.norm(p - P[n])))
        _move_chain(P, children, c, new)
        report.append({"chain": c[0], "kind": "wing", "max_move": round(max(moved), 4)})
    return report


# ---------------------------------------------------------------- pose gate

def winding_number(V: np.ndarray, F: np.ndarray, Q: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Generalised winding number of each query point (van Oosterom-Strackee solid angles)."""
    A0, B0, C0 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.zeros(len(Q))
    for s in range(0, len(F), chunk):
        A = A0[None, s:s + chunk] - Q[:, None]
        B = B0[None, s:s + chunk] - Q[:, None]
        C = C0[None, s:s + chunk] - Q[:, None]
        a, b, c = (np.linalg.norm(X, axis=2) for X in (A, B, C))
        det = np.einsum("qfi,qfi->qf", A, np.cross(B, C))
        den = a * b * c + np.einsum("qfi,qfi->qf", A, B) * c + np.einsum("qfi,qfi->qf", B, C) * a + np.einsum("qfi,qfi->qf", C, A) * b
        out += 2 * np.arctan2(det, den).sum(1)
    return out / (4 * np.pi)


def pose_gate(joint_pos: np.ndarray, segs: list[tuple[int, int]], V: np.ndarray, F: np.ndarray, surface: np.ndarray) -> dict:
    """Joints outside the mesh's box (2% margin) and bone samples outside the mesh AND > 0.08H from its surface."""
    lo, hi = V.min(0), V.max(0)
    H = float(hi[1] - lo[1])
    m = 0.02 * (hi - lo)
    outside = ~np.all((joint_pos >= lo - m) & (joint_pos <= hi + m), axis=1)
    t = np.linspace(0, 1, 9)
    pts = np.concatenate([joint_pos[a][None] * (1 - t[:, None]) + joint_pos[b][None] * t[:, None] for a, b in segs]) if segs else joint_pos
    step = max(1, int(np.ceil(len(F) * len(pts) / 4e7)))  # bound the work on dense meshes (every step-th sample)
    q = pts[::step]
    wn = winding_number(V, F, q)
    d = cKDTree(surface).query(q)[0] / H
    far_out = (np.abs(wn) < 0.5) & (d > 0.08)
    ext = np.ptp(joint_pos, 0) / np.maximum(hi - lo, 1e-9)
    res = {"joints_outside_box": round(float(outside.mean()), 3), "bone_far_outside_0p08H": round(float(far_out.mean()), 3),
           "skeleton_over_mesh_extent": ext.round(2).tolist(), "samples": int(len(q))}
    red = res["joints_outside_box"] > 0.20 or res["bone_far_outside_0p08H"] > 0.30
    res["level"] = "fail" if red else "pass"
    return res
