#!/usr/bin/env python3
"""Rig an asset: pick a body-plan template, skin the textured mesh to it, bake library clips, export for s&box.

    .venv/bin/python scripts/rig.py --stem psx-cow [--class quadruped] [--source psx.glb] [--clips Walk,Idle] [--no-fbx]

1. Class: --class, else the asset's intent / the local planner (Ollama clef-flash) on lifted.png.
   humanoid -> Mesh2Motion human, quadruped -> horse, multileg -> spider, avian -> bird.
2. Rest pose: an animal whose head is turned from its body (Hunyuan3D turns side-on heads toward the camera) is
   rigged from a straightened copy, rig/rest_straight.glb (scripts/rest_check.py); the source mesh is not changed.
   The template is posed like the mesh was generated (people arms down at Idle_A, animals at Idle), then fitted
   (scripts/rig_fit.py): turned to face the way the mesh does (its own body registered at 0/90/180/270, ties
   broken by toes / narrower end / red patch / the local vision model), proportioned per axis when its body
   plan is far off (a chicken on the flying-bird template), legs put on the floor clusters, neck/head and tail
   run through the mesh, wings laid on the body sides. When the template is a poor body-plan match (a chicken on the
   flying bird), or that fit misses the body, the model's own skeleton predicted by skin-tokens' `rig` mode is the
   target instead (scripts/rig_pred.py): hips on its root, trunk+neck+head, tail and legs laid along its chains by
   arc length, chains it doesn't have (a chicken's wings) folded away and their clips (Flap, Glide) not shipped.
   A skeleton that still misses the body is rigged with the predicted skeleton itself (clips mapped chain by
   chain); with no prediction it stops the rig.
   Built path (--template auto|build; design section 4): a body plan (scripts/rig_plans/, rig_plan.resolve) that
   has a builder and no fixed template (a ground bird) gets a skeleton built for this model instead
   (rig_build on the sampled prediction, rig_template: rig/template/rig.glb, yaw 0, scale 1, nothing folded away),
   skinned by skin-tokens like any other, then its own clips made and gated on those weights (rig_clips:
   every clip of the plan, rig/template/clips.glb + clips_report.json + template.json, verified) and baked as a
   near-identity. A failed build falls back to the class's template (auto) or stops (build); the fallback fits
   on the build's own sampled prediction (pred.reused_from "build"), so a rig never predicts twice.
   Every fallback is loud: a `DEGRADED rig <code>: <message>` line and a rig.json `degraded` entry (build_failed,
   poor_match, clips_dropped, pred_failed, pred_skeleton).
3. skin-tokens.cpp (vendor/skin-tokens.cpp, Metal) learns the weights onto that named skeleton. The mesh is not
   moved; the skeleton is fitted to it. No --postprocess: its voxel cleanup never links corners at exactly the same
   point, so on flat-shaded (unshared-corner) meshes each triangle got its own bones and the mesh tore when it moved.
4. Influences from bones far from the vertex (compared with its nearest bone) are dropped (raw weights sometimes
   tie a hand to a foot, or a chicken's foot to a wing tip). Weight on
   the root and on tip (leaf) bones goes to the nearest real parent, as Mesh2Motion does. Corners at the same point
   then get the same weights, so no seam can open.
5. Clips (a ground bird gets Idle and Walk; flight clips only if they keep the body together): each template clip is played on the template, and every bone of ours takes the same world-space rotation
   change from the reference pose (skin-tokens' bones have identity rest rotations, so the clip's local rotations
   can't be used as they are). The root also moves by the template root's offset, scaled to our size.
   Floor pass (rig_pred.floor_pass): per frame the body is grounded the way the template is (touching its floor
   when the template touches), each leg is bent / stretched (two-bone IK) so its foot keeps the template foot's
   clearance, and a head through the floor is lifted at the neck. People are only grounded. rig/rigged_nofloor.glb
   keeps the clips before it.
6. Check: rig/inspect.json and rig/summary.json (scripts/rig_inspect.py), with every clip played and measured for
   tears and stretch (scripts/rig_motion.py).
7. out/<stem>/rigged.glb (textured, clips inside), out/<stem>/rig/<name>.fbx (+ .png) through Blender and checked
   by re-importing it (rig_fbx: rig/fbx_check.json), rig/rig.json, and the contact sheet (rig_render:
   rig/preview/contact_sheet.png, --render 0 skips it). --reframe 1 gives the joints designed frames (bone rolls,
   rig_frames) as the very last step; the skinning is unchanged. The default (auto) does so for a built skeleton
   only; the fixed path keeps identity rests (byte-for-byte as before).

Prints STAGES/STAGE lines for the app's progress, like generate.sh.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

import rest_check

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("FACTORY_ROOT") or ROOT) / "out"  # the factory's asset root; vendor/ stays under ROOT
M2M = ROOT / "vendor" / "mesh2motion-app" / "static"
SKIN = ROOT / "vendor" / "skin-tokens.cpp"
FPS = 30

# Body-plan class -> template rig, clip file(s), the clip whose first frame matches a generated mesh, default clips.
TEMPLATES = {
    "humanoid": {
        "rig": "rigs/rig-human.glb",
        "clips": ["animations/human-base-animations.glb", "animations/human-addon-animations.glb"],
        # The clips' own neutral (arms down, elbows soft). A straight-arm reference made every clip bend the arms.
        "ref": "Idle_A",
        "default": ["Idle_A", "Walk", "Jog", "Interact", "Farm_Watering", "Farm_Harvest", "Chop_Tree", "Death_D"],
    },
    "quadruped": {
        "rig": "rigs/rig-horse.glb",
        "clips": ["animations/horse-animations.glb"],
        "ref": "Idle",
        "default": ["Idle", "Walk", "Trot", "Run", "Eating", "Kick", "Sleep", "Death"],
    },
    "multileg": {
        "rig": "rigs/rig-spider.glb",
        "clips": ["animations/spider-animations.glb"],
        "ref": "Idle",
        "default": ["Idle", "Walk", "Attack", "Bite", "Hit", "Death"],
    },
    "avian": {
        "rig": "rigs/rig-bird.glb",
        "clips": ["animations/bird-animations.glb"],
        "ref": "Idle",
        "default": ["Idle", "Walk", "Flap", "Glide"],
    },
}
# The body-plan classes; rig.TEMPLATES also holds templates built for one asset ("built:<stem>", rig_template),
# so anything that lists the classes reads this, never TEMPLATES' keys.
CLASSES = tuple(TEMPLATES)
PRED_ENV = {"0": "never", "never": "never", "auto": "auto", "always": "always"}  # RIG_PRED's old values
UNSUPPORTED = {
    "bust": "Busts (head and shoulders) don't get a body rig; they need head-turn and talk motion, which comes later.",
    "rigid": "Vehicles and props are rigged by parts with pivots, which comes in a later step.",
    "static": "This looks like a static object; there is nothing to rig.",
}


def stage(key: str, label: str) -> None:
    print(f"STAGE {key} {label}", flush=True)


def deg(x: float) -> str:
    """Signed degrees for notes: −22°, +0°."""
    return f"{x:+.0f}°".replace("-", "−")


# ---------------------------------------------------------------- glTF helpers

COMP = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path: Path) -> tuple[dict, bytes]:
    b = path.read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    doc = json.loads(b[20:20 + n])
    rest = b[20 + n:]
    binary = b""
    if len(rest) >= 8:
        bl = struct.unpack("<I", rest[:4])[0]
        binary = rest[8:8 + bl]
    return doc, binary


def accessor(doc: dict, binary: bytes, i: int) -> np.ndarray:
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    k = WIDTH[a["type"]]
    dt = np.dtype(COMP[a["componentType"]])
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride")
    if stride and stride != dt.itemsize * k:
        raw = np.frombuffer(binary, "u1", count=stride * a["count"], offset=off).reshape(a["count"], stride)
        return raw[:, :dt.itemsize * k].copy().view(dt).reshape(a["count"], k)
    return np.frombuffer(binary, dt, count=a["count"] * k, offset=off).reshape(a["count"], k).copy()


class GlbWriter:
    """Builds one GLB: arrays become accessors on a single buffer; raw blobs (PNG) become plain buffer views."""

    def __init__(self) -> None:
        self.doc: dict = {"asset": {"version": "2.0", "generator": "ai-asset-pipeline rig.py"}, "bufferViews": [], "accessors": []}
        self.chunks: list[bytes] = []
        self.offset = 0

    def _view(self, data: bytes, target: int | None = None) -> int:
        view = {"buffer": 0, "byteOffset": self.offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        self.doc["bufferViews"].append(view)
        pad = (4 - len(data) % 4) % 4
        self.chunks.append(data + b"\0" * pad)
        self.offset += len(data) + pad
        return len(self.doc["bufferViews"]) - 1

    def blob(self, data: bytes) -> int:
        return self._view(data)

    def array(self, arr: np.ndarray, kind: str, target: int | None = None, minmax: bool = False) -> int:
        comp = {np.dtype("<f4"): 5126, np.dtype("<u4"): 5125, np.dtype("<u2"): 5123, np.dtype("u1"): 5121}[arr.dtype]
        acc = {"bufferView": self._view(np.ascontiguousarray(arr).tobytes(), target), "componentType": comp,
               "count": int(arr.shape[0]), "type": kind}
        if minmax:
            acc["min"] = arr.min(0).astype(float).tolist()
            acc["max"] = arr.max(0).astype(float).tolist()
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def write(self, path: Path) -> None:
        body = b"".join(self.chunks)
        self.doc["buffers"] = [{"byteLength": len(body)}]
        js = json.dumps(self.doc, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(body))
        out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(body), 0x004E4942) + body
        path.write_bytes(out)


# ---------------------------------------------------------------- skeleton maths

def node_local(n: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.array(n.get("translation", [0, 0, 0]), float), np.array(n.get("rotation", [0, 0, 0, 1]), float),
            np.array(n.get("scale", [1, 1, 1]), float))


class Skeleton:
    """Named joints of a glTF scene, with forward kinematics from per-joint local TRS."""

    def __init__(self, doc: dict) -> None:
        self.nodes = doc["nodes"]
        self.parent = {c: i for i, n in enumerate(self.nodes) for c in n.get("children", [])}
        self.index = {n.get("name"): i for i, n in enumerate(self.nodes) if n.get("name")}
        self.order = []  # parents before children
        seen = set()

        def visit(i: int) -> None:
            if i in seen:
                return
            if i in self.parent:
                visit(self.parent[i])
            seen.add(i)
            self.order.append(i)

        for i in range(len(self.nodes)):
            visit(i)
        self.rest = {i: node_local(n) for i, n in enumerate(self.nodes)}

    def world(self, local: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[int, tuple[np.ndarray, R]]:
        """World position and rotation per node (scale folded into positions)."""
        out: dict[int, tuple[np.ndarray, R, np.ndarray]] = {}
        for i in self.order:
            t, q, s = local.get(i, self.rest[i])
            rot = R.from_quat(q)
            if i in self.parent:
                pp, pr, ps = out[self.parent[i]]
                out[i] = (pp + pr.apply(ps * t), pr * rot, ps * s)
            else:
                out[i] = (t, rot, s)
        return {i: (p, r) for i, (p, r, _) in out.items()}


def prepare_clip(doc: dict, binary: bytes, anim: dict, skel: Skeleton) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """The clip's channels that land on `skel`'s joints: (joint index, path, key times, key values)."""
    out = []
    for ch in anim["channels"]:
        name = doc["nodes"][ch["target"]["node"]].get("name")
        path = ch["target"]["path"]
        if name in skel.index and path in ("translation", "rotation", "scale"):
            smp = anim["samplers"][ch["sampler"]]
            out.append((skel.index[name], path, accessor(doc, binary, smp["input"])[:, 0], accessor(doc, binary, smp["output"])))
    return out


def sample_clip(channels: list, skel: Skeleton, t: float) -> dict[int, tuple]:
    """Template local TRS at time t (linear / slerp), on top of the rest pose."""
    local = {i: list(trs) for i, trs in skel.rest.items()}
    slot = {"translation": 0, "rotation": 1, "scale": 2}
    for i, path, times, vals in channels:
        j = int(np.searchsorted(times, t))
        if j <= 0:
            v = vals[0]
        elif j >= len(times):
            v = vals[-1]
        else:
            a = (t - times[j - 1]) / max(times[j] - times[j - 1], 1e-9)
            if path == "rotation":
                q0 = R.from_quat(vals[j - 1])
                v = (q0 * R.from_rotvec((q0.inv() * R.from_quat(vals[j])).as_rotvec() * a)).as_quat()
            else:
                v = vals[j - 1] * (1 - a) + vals[j] * a
        local[i][slot[path]] = np.array(v, float)
    return {i: tuple(v) for i, v in local.items()}


def clip_duration(doc: dict, binary: bytes, anim: dict) -> float:
    return max(float(accessor(doc, binary, s["input"])[-1, 0]) for s in anim["samplers"])


# ---------------------------------------------------------------- steps

def classify(stem_dir: Path) -> dict:
    """Body plan from the local planner: {"choice", "confidence"}. Raises if it can't say."""
    image = next((stem_dir / n for n in ("lifted.png", "input.png", "input.jpg") if (stem_dir / n).is_file()), None)
    if image is None:
        raise SystemExit("rig: no picture to classify; pass --class")
    questions = {
        "plan": {"type": "choice", "instructions": "What kind of subject is the main object in the image, for rigging",
                 "criteria": {"humanoid": "A whole person or humanoid character, legs visible",
                              "bust": "Only the head and shoulders or upper body of a person",
                              "quadruped": "A four-legged animal such as a cow, dog or horse",
                              "multileg": "Six or more legs, e.g. a spider, insect or spider mech",
                              "avian": "A bird",
                              "rigid": "A vehicle or machine with rigid moving parts, e.g. a tank",
                              "static": "An object that doesn't move, e.g. a crate, lantern or staff"}}}
    body = {"model": os.environ.get("RIG_PLANNER_MODEL", "clef-flash"), "state": "Picture of a game asset to rig.",
            "images": [base64.b64encode(image.read_bytes()).decode()], "questions": questions,
            "keep_alive": 0}  # one question per rig: unload the planner (12 GB) now, not after ollama's 5 min
    req = urllib.request.Request(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/v1/systemone",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        answer = json.load(urllib.request.urlopen(req, timeout=180))["answers"]["plan"]
    except Exception as e:  # no Ollama, no model, or another reply shape
        raise SystemExit(f"rig: the planner couldn't classify this ({e}); pass --class")
    print(f"rig: planner says {answer['choice']} ({answer.get('confidence', 0):.2f})", flush=True)
    return {"choice": answer["choice"], "confidence": answer.get("confidence")}


def fit_transform(skel: Skeleton, rdoc: dict, local: dict[int, tuple], mesh_points: np.ndarray) -> tuple[float, np.ndarray]:
    """Uniform scale and offset that put the posed template on the mesh: same height, lowest joints on the mesh's
    floor, root bone over the centre of the mesh's slice at the root's height."""
    world = skel.world(local)
    joints = rdoc["skins"][0]["joints"]
    jp = np.array([world[i][0] for i in joints])
    jlo, jhi = jp[:, 1].min(), jp[:, 1].max()
    mlo, mhi = mesh_points[:, 1].min(), mesh_points[:, 1].max()
    scale = (mhi - mlo) / max(jhi - jlo, 1e-9)
    root = next(i for i in joints if skel.parent.get(i) not in joints)
    rp = world[root][0]
    y = mlo + (rp[1] - jlo) / max(jhi - jlo, 1e-9) * (mhi - mlo)
    band = mesh_points[np.abs(mesh_points[:, 1] - y) <= 0.05 * (mhi - mlo)]
    if len(band) < 10:
        band = mesh_points
    centre = (band.min(0) + band.max(0)) / 2
    return scale, np.array([centre[0], mlo, centre[2]]) - np.array([rp[0], jlo, rp[2]]) * scale


def ref_local(template: dict, skel: Skeleton) -> dict[int, tuple]:
    """Local TRS of the template's reference pose: frame 0 of a named clip, or the rest pose with the upper arms
    turned down by {"arms_down": degrees} about the forward (+Z) axis."""
    ref = template["ref"]
    if isinstance(ref, str):
        adoc, abin = read_glb(M2M / template["clips"][0])
        anim = next(a for a in adoc["animations"] if a["name"] == ref)
        return sample_clip(prepare_clip(adoc, abin, anim, skel), skel, 0.0)
    local = {i: tuple(v.copy() for v in trs) for i, trs in skel.rest.items()}
    world = skel.world(local)
    for side, sign in (("l", -1.0), ("r", 1.0)):
        i = skel.index.get(f"upperarm_{side}")
        if i is None:
            continue
        p = skel.parent[i]
        turn = R.from_euler("z", sign * ref["arms_down"], degrees=True)
        t, _, sc = local[i]
        local[i] = (t, (world[p][1].inv() * turn * world[i][1]).as_quat(), sc)
    return local


def aim_arms(skel: Skeleton, local: dict[int, tuple], mesh_t: np.ndarray, side_sign: dict[str, float]) -> list[str]:
    """Turn each upper arm so its hand joint points at the mesh's hand (template frame). Returns notes.

    The mesh's hand is the point about one arm's length from the shoulder, below it, on that side, that is
    farthest from the body's centre line. The turn is capped at 60 degrees."""
    notes = []
    world = skel.world(local)
    centre_x = np.median(mesh_t[:, 0])
    for side, sign in side_sign.items():
        ua, hand = skel.index.get(f"upperarm_{side}"), skel.index.get(f"hand_{side}")
        if ua is None or hand is None:
            continue
        sh, hd = world[ua][0], world[hand][0]
        arm = np.linalg.norm(hd - sh)
        d = np.linalg.norm(mesh_t - sh, axis=1)
        cand = mesh_t[(np.abs(d - arm) < 0.2 * arm) & (mesh_t[:, 1] < sh[1]) & (np.sign(mesh_t[:, 0] - centre_x) == sign)]
        if len(cand) < 5:
            notes.append(f"Couldn't find the {SIDE[side]} hand")
            continue
        target = cand[np.argmax(np.abs(cand[:, 0] - centre_x))]
        a, b = (hd - sh) / arm, (target - sh) / np.linalg.norm(target - sh)
        axis = np.cross(a, b)
        ang = float(np.arccos(np.clip(a @ b, -1, 1)))
        if np.linalg.norm(axis) < 1e-6 or ang < np.radians(2):
            continue
        ang = min(ang, np.radians(60))
        turn = R.from_rotvec(axis / np.linalg.norm(axis) * ang)
        p = skel.parent[ua]
        t, _, sc = local[ua]
        local[ua] = (t, (world[p][1].inv() * turn * world[ua][1]).as_quat(), sc)
        world = skel.world(local)
        notes.append(f"Turned the {SIDE[side]} arm {np.degrees(ang):.0f}° to reach the hand")
    return notes


SIDE = {"l": "left", "r": "right"}
HEAD_CHAIN = ["spine_4", "spine_5", "spine_6", "head", "head_leaf"]  # four-legged templates (horse)


def _descendants(skel: Skeleton, i: int) -> list[int]:
    out = [i]
    for c in skel.nodes[i].get("children", []):
        out += _descendants(skel, c)
    return out


def _centreline(mesh_t: np.ndarray, start: np.ndarray, y_min: float, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """The mesh's neck/head centreline in the template frame (+Z forward, +Y up), from `start` to the muzzle:
    per slice along Z the middle of the upper half, smoothed with a quadratic so sparse low-poly slices don't zigzag."""
    sel = mesh_t[(mesh_t[:, 2] >= start[2] - 0.05) & (mesh_t[:, 1] > y_min)]
    z = sel[:, 2]
    edges = np.linspace(z.min(), z.max(), n + 1)
    pts = []
    for a, b in zip(edges[:-1], edges[1:]):
        S = sel[(z >= a) & (z < b)]
        if len(S) < 8:
            continue
        S = S[S[:, 1] > np.quantile(S[:, 1], 0.5)]
        pts.append([(S[:, 0].min() + S[:, 0].max()) / 2, S[:, 1].mean(), S[:, 2].mean()])
    pts = np.array(pts)
    tip = sel[z > np.quantile(z, 0.985)].mean(0)
    zz = np.r_[pts[:, 2], tip[2]]
    w = np.r_[np.ones(len(pts)), 3.0]
    cx = np.polyfit(zz, np.r_[pts[:, 0], tip[0]], 2, w=w)
    cy = np.polyfit(zz, np.r_[pts[:, 1], tip[1]], 2, w=w)
    zs = np.linspace(start[2], tip[2], 40)
    line = np.c_[np.polyval(cx, zs), np.polyval(cy, zs), zs]
    line[0] = start
    return line, tip


def _at_arclen(poly: np.ndarray, s: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    if s >= cum[-1]:
        return poly[-1] + (poly[-1] - poly[-2]) / seg[-1] * (s - cum[-1])
    k = max(int(np.searchsorted(cum, s)) - 1, 0)
    return poly[k] + (poly[k + 1] - poly[k]) * (s - cum[k]) / seg[k]


def fit_head_chain(skel: Skeleton, local: dict[int, tuple], mesh_t: np.ndarray, scale: float = 1.0) -> list[str]:
    """Run the template's neck and head through the mesh's own neck and head: stretch the skull so the muzzle
    joint reaches the mesh's muzzle (ears keep their size), then turn spine_5, spine_6 and head so each child
    lands on the mesh's centreline. Without this the neck stopped 0.3 m short of the cow's muzzle and the head's
    vertices were bound to bones beside them. mesh_t: mesh surface samples in the posed template's frame;
    scale: template units to metres, for the notes."""
    idx = skel.index
    W = skel.world(local)
    p = {n: W[idx[n]][0] for n in HEAD_CHAIN}
    line, tip = _centreline(mesh_t, p["spine_5"], 0.6 * p["spine_4"][1])
    tl = sum(np.linalg.norm(p[HEAD_CHAIN[k + 1]] - p[HEAD_CHAIN[k]]) for k in range(1, 4))
    ml = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    s = float(np.clip(ml / tl, 0.7, 1.6))
    ears = {j for n in ("ear_1_l", "ear_1_r") if n in idx for j in _descendants(skel, idx[n])[1:]}
    for i in _descendants(skel, idx["spine_5"])[1:]:
        if i not in ears:
            t, q, sc = local[i]
            local[i] = (t * s, q, sc)
    notes = [f"Stretched the neck and head {s:.2f}× to reach the muzzle"]
    turns = []
    for a, b in (("spine_5", "spine_6"), ("spine_6", "head"), ("head", "head_leaf")):
        W = skel.world(local)
        pa, pb = W[idx[a]][0], W[idx[b]][0]
        chain = [W[idx[n]][0] for n in HEAD_CHAIN[1:HEAD_CHAIN.index(b) + 1]]
        target = _at_arclen(line, sum(np.linalg.norm(chain[k + 1] - chain[k]) for k in range(len(chain) - 1)))
        u, v = (pb - pa) / np.linalg.norm(pb - pa), (target - pa) / np.linalg.norm(target - pa)
        ax = np.cross(u, v)
        ang = float(np.arccos(np.clip(u @ v, -1, 1)))
        if np.linalg.norm(ax) < 1e-9:
            continue
        turn = R.from_rotvec(ax / np.linalg.norm(ax) * min(ang, np.radians(50)))
        i = idx[a]
        t, _, sc = local[i]
        local[i] = (t, (W[skel.parent[i]][1].inv() * turn * W[i][1]).as_quat(), sc)
        turns.append(f"{np.degrees(ang):.0f}°")
    W = skel.world(local)
    notes.append("Bent the neck and head to follow the model (" + ", ".join(turns) + f"); the muzzle bone ends "
                 f"{np.linalg.norm(W[idx['head_leaf']][0] - tip) * scale:.2f} m from the model's muzzle")
    return notes


# Fit steps, all on (E2 experiment ablated them one at a time): a facing, b per-axis proportions, c chains,
# d nearest-bone weight filter, e ground-bird clip set, f pose gate
FLAGS = frozenset("abcdef")


def _template_class(template: dict) -> str | None:
    """The body-plan class a template is for: a built template says so; a vendor one is found by identity."""
    return template.get("class") or next((k for k, v in TEMPLATES.items() if v is template), None)


def decide(plan: dict, want: str) -> str:
    """'fixed' or 'build' for a resolved body plan and --template auto|fixed|build (design section 4)."""
    builder, fixed = bool(plan.get("builder")), bool(plan.get("fixed_template"))
    if want == "fixed":
        return "fixed"
    if want == "build":
        if not builder:
            raise SystemExit(f"rig: the {plan.get('label') or plan.get('id')} body plan has no skeleton builder; "
                             "use --template fixed or auto")
        return "build"
    return "build" if builder and not fixed else "fixed"  # no builder, or both (v1 has none): the fixed template


def fixed_plan(plan: dict, cls: str) -> dict:
    """The body plan the class's fixed template goes with. A plan that only builds (ground_bird) has none of its
    own, so the fixed path reads the class's wrapper plan (avian: flying_bird), which matches today's tables."""
    if plan.get("fixed_template") or not plan.get("builder"):
        return plan
    import rig_plan
    return rig_plan.load(cls)


def ground_bird_trial(plan: dict, cls: str, P: np.ndarray, names: list[str], explicit: bool, built: bool) -> list[str]:
    """The fixed path's ground-bird trial (FLAGS e): on an upright mesh whose plan asks for it (flying_bird's
    trial_when_upright), every clip but Idle and Walk ships only if it keeps the body together. Never on a built
    template (its clips were made and gated for this model) or when the clips were asked for by name."""
    import rig_plan
    if built or explicit or "e" not in FLAGS or not rig_plan.spec(plan, cls, "trial_when_upright", False):
        return []
    upright = float(np.ptp(P[:, 1])) > 0.9 * float(max(np.ptp(P[:, 0]), np.ptp(P[:, 2])))
    return [n for n in names if n not in ("Idle", "Walk")] if upright else []


def facing_prior_yaw(facing_prior: dict, source_name: str) -> dict | None:
    """The fixed path's facing prior: yaw 0 when psx.glb is known to have its head at +Z (a canonical draw, or a
    psx facing check that agreed or flipped it there). None otherwise, or for a mesh that isn't psx.glb."""
    if source_name != "psx.glb":
        return None
    chk = facing_prior.get("facing_check") or {}
    if facing_prior.get("canonical") or chk.get("agrees") or chk.get("flipped"):
        why = "canonical draw" if facing_prior.get("canonical") else ("psx flipped it" if chk.get("flipped") else "psx check agreed")
        return {"yaw": 0, "source": facing_prior.get("facing_source"), "why": why}
    return None


def posed_template(template: dict, out: Path, mesh_points: np.ndarray, surface: np.ndarray | None = None,
                   facing: dict | None = None, mesh_faces: np.ndarray | None = None, pred_fn=None,
                   force_pred: bool = False, plan: dict | None = None) -> tuple[dict, dict, list[str]]:
    """The template rig at frame 0 of the reference clip, turned, scaled and fitted onto the mesh; written to `out`
    for skin-tokens (which then runs with --fit none).

    T(p) = (Ry(yaw) p) * s + offset maps the template frame onto the mesh. yaw comes from registering the
    template's own body at 0/90/180/270 (rig_fit.register) plus the facing cues in `facing`; s is the
    registration's per-axis scale (or the height ratio); x/z offset from the registration, y so the lowest joint
    sits on the floor. Then, in the template frame (mesh mapped by T^-1): people's arms are aimed, the horse's
    neck/head is run through the mesh (fit_head_chain), and legs / neck / tail / wings are fitted (rig_fit).
    Joint positions are written as plain translations (unit scale), so the skeleton GLB holds no non-uniform
    scale; ref_store keeps the template-frame reference pose plus "__fit__" (yaw, scale) for bake_clips."""
    import rig_fit
    rdoc, rbin = read_glb(M2M / template["rig"])
    skel = Skeleton(rdoc)
    local = ref_local(template, skel)
    cls = _template_class(template)
    joints = rdoc["skins"][0]["joints"]
    samples = surface if surface is not None else mesh_points
    notes: list[str] = []
    fitinfo: dict = {"flags": "".join(sorted(FLAGS))}
    yaw, sw, reg = 0, None, None
    if FLAGS & {"a", "b"}:
        TV, TF = rig_fit.template_body(M2M, template)
        reg = rig_fit.register(TV, TF, samples)
        fitinfo["registration"] = {str(k): v for k, v in reg.items()}
        if "a" in FLAGS:
            choice = rig_fit.choose_yaw(reg, (facing or {}).get("geometric") or {}, (facing or {}).get("vlm"),
                                        prior=(facing or {}).get("prior"))
            yaw = int(choice["yaw"])
            fitinfo["facing"] = choice
            notes.append(f"Faced the template {yaw}° (Chamfer " + ", ".join(f"{y}°: {choice['chamfer'][str(y)]:.3f}" for y in rig_fit.YAWS)
                         + f"; margin {choice['margin']:.2f}, decided by {choice['decided_by']})")
        s_reg = np.array(reg[yaw]["scale_world"], float)
        ratio = float(s_reg.max() / s_reg.min())
        # Per-axis proportions only when the template's are far off (> 1.6x between axes: a chicken on a flying
        # bird); a close template (a cow on the horse, 1.2x) keeps its own, which bound better. People: arms are
        # aimed instead; their T/A pose breaks a body-box scale.
        if "b" in FLAGS and cls != "humanoid" and ratio > rig_fit.POOR_MATCH:
            sw = s_reg
            fitinfo["scale_world"] = sw.round(4).tolist()
            fitinfo["scale_ratio"] = round(ratio, 3)
            fitinfo["poor_match"] = ratio > rig_fit.POOR_MATCH
            notes.append(f"Proportioned the template {sw[0] / sw[1]:.2f} : 1 : {sw[2] / sw[1]:.2f} (x : y : z) to the model's body"
                         + (f"; a poor body-plan match ({ratio:.1f}× between axes)" if ratio > rig_fit.POOR_MATCH else ""))
        else:
            fitinfo["scale_ratio_unused"] = round(ratio, 3)
    Ry = R.from_euler("y", yaw, degrees=True)
    RyM = Ry.as_matrix()
    mlo, mhi = mesh_points[:, 1].min(), mesh_points[:, 1].max()

    def placement() -> tuple[np.ndarray, np.ndarray]:
        W = skel.world(local)
        Q = np.array([RyM @ W[i][0] for i in joints])
        if sw is not None:
            # height as before (the joints span the mesh's height); the registration only sets the proportions,
            # and keeps the template body's centre where the registration put it
            k = (mhi - mlo) / max(np.ptp(Q[:, 1]), 1e-9)
            s = sw * (k / sw[1])
            cb = RyM @ ((TV.min(0) + TV.max(0)) / 2)
            off = (sw * cb + np.array(reg[yaw]["offset"], float)) - s * cb
            off[1] = mlo - (Q[:, 1] * s[1]).min()
            return s, off
        k = (mhi - mlo) / max(np.ptp(Q[:, 1]), 1e-9)
        root = next(i for i in joints if skel.parent.get(i) not in joints)
        rp = RyM @ W[root][0]
        jlo = Q[:, 1].min()
        y = mlo + (rp[1] - jlo) * k
        band = mesh_points[np.abs(mesh_points[:, 1] - y) <= 0.05 * (mhi - mlo)]
        if len(band) < 10:
            band = mesh_points
        centre = (band.min(0) + band.max(0)) / 2
        return np.full(3, k), np.array([centre[0], mlo, centre[2]]) - np.array([rp[0], jlo, rp[2]]) * k

    def to_t(X: np.ndarray, s: np.ndarray, off: np.ndarray) -> np.ndarray:  # mesh -> template frame (row vectors)
        return ((X - off) / s) @ RyM

    for _ in range(2):  # place, aim the arms at the placed mesh, place again
        s, off = placement()
        if "upperarm_l" not in skel.index:
            break
        notes += aim_arms(skel, local, to_t(mesh_points, s, off), {"l": 1.0, "r": -1.0})
    s, off = placement()
    # the model's predicted skeleton as the target (rig_pred) when the template is a poor body-plan match, or asked
    pred_lines = pred_fn(yaw) if pred_fn is not None and (force_pred or fitinfo.get("poor_match")) else None
    if surface is not None and all(n in skel.index for n in HEAD_CHAIN) and not pred_lines:
        notes += fit_head_chain(skel, local, to_t(surface, s, off), float(s[1]))
    W = skel.world(local)
    names = {i: skel.nodes[i].get("name") for i in joints}
    Pt = {names[i]: W[i][0].copy() for i in joints}
    if "c" in FLAGS:
        children = {}
        for i in joints:
            p = skel.parent.get(i)
            if p in joints:
                children.setdefault(names[p], []).append(names[i])
        mesh_t = to_t(samples, s, off)
        if mesh_faces is not None:
            tV = to_t(mesh_points, s, off)
            H_t = float(np.ptp(tV[:, 1]))
            if pred_lines:
                import rig_pred
                lt = {k: ({**v, "line": to_t(v["line"], s, off)} if k != "legs" else [{**q, "line": to_t(q["line"], s, off)} for q in v])
                      for k, v in pred_lines.items()}
                reg_p = rig_pred.register(cls, Pt, children, lt, mesh_t, tV, mesh_faces, H_t, plan=plan)
                fitinfo["pred_register"] = reg_p
                fitinfo["unsupported"] = reg_p["unsupported"]
                for c in reg_p["paths"]:
                    notes.append("Laid the template's " + c["kind"] + (" " + c["chain"] if "chain" in c else "") + " on the predicted "
                                 + ", ".join(c.get("pred", [])[:1] + c.get("pred", [])[-1:]) + ": " + ", ".join(
                                     f"{k} {v}" for k, v in c.items() if k not in ("chain", "kind", "pred")))
                if reg_p["unsupported"]:
                    notes.append("No predicted chain for the template's " + ", ".join(reg_p["unsupported"]) + ": folded onto the body, unsupported")
            else:
                chains = rig_fit.fit_chains(cls, Pt, children, mesh_t, tV, mesh_faces, H_t)
                fitinfo["chains"] = chains
                for c in chains:
                    notes.append("Fitted the " + c["kind"] + " from " + c["chain"] + ": " + ", ".join(
                        f"{k} {v}" for k, v in c.items() if k not in ("chain", "kind")))
    Pm = {n: RyM @ p * s + off for n, p in Pt.items()}
    # Write: every joint's world rotation is Ry * the template's; translations put it exactly at Pm; unit scale.
    rot_w = {i: Ry * W[i][1] for i in joints}
    for i in range(len(rdoc["nodes"])):
        if i not in joints and i in skel.parent.values():  # Armature and other holders: identity
            for k in ("translation", "rotation", "scale", "matrix"):
                rdoc["nodes"][i].pop(k, None)
    for i in joints:
        p = skel.parent.get(i)
        if p in joints:
            t = rot_w[p].inv().apply(Pm[names[i]] - Pm[names[p]])
            q = (rot_w[p].inv() * rot_w[i]).as_quat()
        else:
            t, q = Pm[names[i]], rot_w[i].as_quat()
        rdoc["nodes"][i]["translation"] = np.asarray(t, float).tolist()
        rdoc["nodes"][i]["rotation"] = np.asarray(q, float).tolist()
        rdoc["nodes"][i]["scale"] = [1.0, 1.0, 1.0]
    tops = [i for i in range(len(rdoc["nodes"])) if i not in skel.parent]
    rdoc["nodes"].append({"name": "fit", "children": tops, "extras": {"yaw_deg": yaw, "scale": s.round(5).tolist(),
                                                                       "offset": off.round(5).tolist()}})
    for sc in rdoc.get("scenes", []):
        sc["nodes"] = [len(rdoc["nodes"]) - 1] + [n for n in sc["nodes"] if n not in tops]
    rdoc.pop("animations", None)
    # inverse binds that match the new pose (skin-tokens reads joints; a viewer reads these)
    ibm = []
    for i in joints:
        M = np.eye(4)
        M[:3, :3] = rot_w[i].as_matrix()
        M[:3, 3] = Pm[names[i]]
        ibm.append(np.linalg.inv(M).T.reshape(-1))
    ibm = np.array(ibm, "<f4")
    skn = rdoc["skins"][0]
    acc = rdoc["accessors"][skn["inverseBindMatrices"]]
    view = rdoc["bufferViews"][acc["bufferView"]]
    o = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    rbin = rbin[:o] + ibm.tobytes() + rbin[o + ibm.nbytes:]
    ref_store = {skel.nodes[i].get("name"): [t.tolist(), q.tolist(), sc.tolist()] for i, (t, q, sc) in local.items() if skel.nodes[i].get("name")}
    fitinfo.update(yaw=yaw, scale=s.round(5).tolist(), offset=off.round(5).tolist())
    ref_store["__fit__"] = fitinfo
    js = json.dumps(rdoc, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    body = rbin + b"\0" * ((4 - len(rbin) % 4) % 4)
    rdoc_bytes = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(body))
    rdoc_bytes += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(body), 0x004E4942) + body
    out.write_bytes(rdoc_bytes)
    return rdoc, ref_store, notes


def weld(source: Path, out: Path, tol: float = 1e-5) -> np.ndarray:
    """Write a shared-vertex copy of the source mesh (positions and triangles only) for skinning.
    Returns each source vertex's index in it."""
    d, b = read_glb(source)
    prim = d["meshes"][0]["primitives"][0]
    P = accessor(d, b, prim["attributes"]["POSITION"]).astype(np.float32)
    F = accessor(d, b, prim["indices"]).reshape(-1) if "indices" in prim else np.arange(len(P))
    _, first, inv = np.unique(np.round(P.astype(float) / tol).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    g = GlbWriter()
    pos = g.array(P[first].astype("<f4"), "VEC3", 34962, True)
    idx = g.array(inv[F].astype("<u4"), "SCALAR", 34963)
    g.doc.update(scene=0, scenes=[{"nodes": [0]}], nodes=[{"mesh": 0, "name": "welded"}],
                 meshes=[{"primitives": [{"attributes": {"POSITION": pos}, "indices": idx}]}])
    g.write(out)
    return inv


def run_skin(mesh: Path, skeleton: Path, out: Path, fit: str = "none") -> float:
    cli = SKIN / "build" / "release" / "bin" / "skintokens-cli"
    model = SKIN / "models" / "SkinTokens-GGUF" / "F16"
    if not cli.is_file() or not model.is_dir():
        raise SystemExit("rig: skin-tokens.cpp is not built or its weights are missing (see docs/spikes-rigging.md)")
    t = time.time()
    proc = subprocess.run([str(cli), "skin", str(model), str(mesh), str(skeleton), str(out), "--device", "auto",
                           "--fit", fit], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("rig: skin-tokens failed: " + (proc.stderr or proc.stdout)[-800:])
    return time.time() - t


def local_weights(P: np.ndarray, J: np.ndarray, W: np.ndarray, joint_pos: np.ndarray, child: dict[int, int],
                  height: float, skip: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop influences whose bone is far from the vertex (relative to its nearest influence); renormalise."""
    def seg_dist(p: np.ndarray, j: int) -> np.ndarray:
        a = joint_pos[j]
        b = joint_pos[child[j]] if j in child else a
        ab = b - a
        tt = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1) if ab @ ab > 1e-12 else np.zeros(len(p))
        return np.linalg.norm(p - (a + tt[:, None] * ab), axis=1)

    D = np.zeros(W.shape)
    for k in range(J.shape[1]):
        for j in np.unique(J[:, k]):
            sel = J[:, k] == j
            D[sel, k] = seg_dist(P[sel], int(j))
    D[W <= 0] = np.inf
    if "d" not in FLAGS:
        near = D.min(1, keepdims=True)
        keep = (D <= np.maximum(near * 3.0, near + 0.08 * height)) & (W > 0)
        keep[np.arange(len(W)), W.argmax(1)] = True
        dropped = int(((W > 0) & ~keep).sum())
        W2 = np.where(keep, W, 0).astype(np.float32)
        W2 /= W2.sum(1, keepdims=True)
        return J, W2, dropped
    # Relative to the vertex's nearest BONE (any bone, point to segment), not its nearest influence; the strongest
    # influence is kept only while it is within 0.15x height of that nearest bone. A vertex left with nothing goes
    # wholly to its nearest bone.
    allD = np.stack([seg_dist(P, j) if j not in skip else np.full(len(P), np.inf) for j in range(len(joint_pos))], 1)
    nb = allD.min(1, keepdims=True)
    keep = (D <= np.maximum(nb * 3.0, nb + 0.08 * height)) & (W > 0)
    top = W.argmax(1)
    rows = np.arange(len(W))
    keep[rows, top] |= (D[rows, top] - nb[:, 0]) <= 0.15 * height
    dropped = int(((W > 0) & ~keep).sum())
    W2 = np.where(keep, W, 0).astype(np.float64)
    empty = W2.sum(1) <= 0
    J = J.copy()
    if empty.any():
        J[empty] = 0
        J[empty, 0] = allD[empty].argmin(1)
        W2[empty] = 0
        W2[empty, 0] = 1.0
    W2 = (W2 / W2.sum(1, keepdims=True)).astype(np.float32)
    local_weights.reassigned = int(empty.sum())
    return J, W2, dropped


def to_dense(J: np.ndarray, W: np.ndarray, n: int) -> np.ndarray:
    M = np.zeros((len(J), n))
    np.add.at(M, (np.repeat(np.arange(len(J)), J.shape[1]), J.reshape(-1)), W.reshape(-1))
    return M


def to_top4(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    J = np.argsort(-M, axis=1)[:, :4]
    W = np.take_along_axis(M, J, 1)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    J[W <= 0] = 0
    return J.astype(np.int64), W.astype(np.float32)


def fold_tips(J: np.ndarray, W: np.ndarray, parent: dict[int, int], root: int, hips: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Move weight on the root and on leaf bones to the nearest parent that is neither. Returns the share moved."""
    n = max(int(J.max()) + 1, max(parent, default=0) + 1, hips + 1)
    M = to_dense(J, W, n)
    has_child = set(parent.values())
    skip = {k for k in range(n) if k not in has_child} | {root}
    moved = float(M[:, sorted(skip)].sum() / max(M.sum(), 1e-12))
    for k in sorted(skip):
        p = parent.get(k)
        while p is not None and p in skip:
            p = parent.get(p)
        target = hips if p is None else p
        if target != k:
            M[:, target] += M[:, k]
            M[:, k] = 0
    J2, W2 = to_top4(M)
    return J2, W2, moved


def share_corners(P: np.ndarray, J: np.ndarray, W: np.ndarray, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray, int]:
    """Give every vertex at the same position the same weights (their mean). Returns how many points disagreed."""
    _, inv = np.unique(np.round(P / tol).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(J.max()) + 1
    M = to_dense(J, W, n)
    mean = np.zeros((int(inv.max()) + 1, n))
    np.add.at(mean, inv, M)
    mean /= np.bincount(inv)[:, None]
    split = len(np.unique(inv[np.abs(M - mean[inv]).sum(1) > 1e-3]))
    J2, W2 = to_top4(mean[inv])
    return J2, W2, split


def bake_clips(template: dict, names: list[str], ours: dict, our_skel: Skeleton, ref_store: dict) -> tuple[list[dict], list[str]]:
    """Each clip as per-frame local rotations for our joints (+ root translation)."""
    tdoc, _ = read_glb(M2M / template["rig"])
    tskel = Skeleton(tdoc)
    sources = [read_glb(M2M / f) for f in template["clips"]]
    anims = {a["name"]: (d, b, a) for d, b in sources for a in d.get("animations", [])}
    ref = {tskel.index[n]: tuple(np.array(v) for v in trs) for n, trs in ref_store.items() if n in tskel.index}
    ref_w = tskel.world(ref)
    joints = ours["skins"][0]["joints"]
    names_of = {i: our_skel.nodes[i].get("name") for i in joints}
    oroot = next(i for i in joints if our_skel.parent.get(i) not in joints)
    troot = tskel.index[names_of[oroot]]
    th = np.ptp([ref_w[i][0][1] for i in tskel.index.values()])
    our_w = our_skel.world({})
    oh = np.ptp([our_w[i][0][1] for i in joints])
    scale = oh / max(th, 1e-9)
    our_rest_rot = {i: our_w[i][1] for i in joints}
    fit = ref_store.get("__fit__") or {}
    yaw_r = R.from_euler("y", float(fit.get("yaw", 0)), degrees=True)  # the template frame turned onto the mesh
    root_scale = np.asarray(fit["scale"], float) if "scale" in fit else np.full(3, scale)
    baked, missing = [], []
    for name in names:
        if name not in anims:
            missing.append(name)
            continue
        d, b, a = anims[name]
        dur = clip_duration(d, b, a)
        channels = prepare_clip(d, b, a, tskel)
        times = np.arange(0, dur + 1e-6, 1.0 / FPS)
        rot = {i: [] for i in joints}
        root_t = []
        for t in times:
            tw = tskel.world(sample_clip(channels, tskel, float(t)))
            want = {}
            for i in joints:
                ti = tskel.index.get(names_of[i])
                delta = yaw_r * tw[ti][1] * ref_w[ti][1].inv() * yaw_r.inv() if ti is not None else R.identity()
                want[i] = delta * our_rest_rot[i]
            for i in joints:
                p = our_skel.parent.get(i)
                pw = want[p] if p in want else (our_w[p][1] if p is not None else R.identity())
                rot[i].append((pw.inv() * want[i]).as_quat())
            off = tw[troot][0] - ref_w[troot][0]
            root_t.append(np.array(our_skel.nodes[oroot].get("translation", [0, 0, 0])) + yaw_r.apply(off) * root_scale)
        baked.append({"name": name, "times": times.astype(np.float32), "rot": {i: np.array(v, np.float32) for i, v in rot.items()},
                      "root": oroot, "root_t": np.array(root_t, np.float32)})
    return baked, missing


def write_rigged(out: Path, ours: dict, obin: bytes, src: dict, sbin: bytes, J: np.ndarray, W: np.ndarray,
                 clips: list[dict]) -> None:
    """Our skeleton + the source mesh (positions, normals, UVs, material, texture) + filtered weights + clips."""
    g = GlbWriter()
    sp = src["meshes"][0]["primitives"][0]
    attrs = {}
    for key in ("POSITION", "NORMAL", "TEXCOORD_0"):
        if key in sp["attributes"]:
            arr = accessor(src, sbin, sp["attributes"][key]).astype("<f4")
            attrs[key] = g.array(arr, {"POSITION": "VEC3", "NORMAL": "VEC3", "TEXCOORD_0": "VEC2"}[key], 34962, key == "POSITION")
    attrs["JOINTS_0"] = g.array(J.astype("<u2"), "VEC4", 34962)
    attrs["WEIGHTS_0"] = g.array(W.astype("<f4"), "VEC4", 34962)
    prim = {"attributes": attrs}
    if "indices" in sp:
        idx = accessor(src, sbin, sp["indices"]).reshape(-1).astype("<u4")
        prim["indices"] = g.array(idx, "SCALAR", 34963)
    # Material, texture and sampler from the source (psx.glb keeps its nearest sampler).
    if "material" in sp and src.get("materials"):
        mat = json.loads(json.dumps(src["materials"][sp["material"]]))
        g.doc["materials"] = [mat]
        prim["material"] = 0
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        if "metallicRoughnessTexture" not in pbr:
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = 1.0
        tex_ref = pbr.get("baseColorTexture")
        if tex_ref is not None:
            tex = src["textures"][tex_ref["index"]]
            img = src["images"][tex["source"]]
            v = src["bufferViews"][img["bufferView"]]
            data = sbin[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
            g.doc["images"] = [{"bufferView": g.blob(data), "mimeType": img.get("mimeType", "image/png")}]
            g.doc["samplers"] = [src["samplers"][tex["sampler"]]] if "sampler" in tex and src.get("samplers") else [{}]
            g.doc["textures"] = [{"source": 0, "sampler": 0}]
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    g.doc["meshes"] = [{"name": "model", "primitives": [prim]}]
    # Skeleton: skin-tokens' nodes as they are, with the mesh node pointing at our mesh.
    nodes = json.loads(json.dumps(ours["nodes"]))
    for n in nodes:
        if "mesh" in n:
            n["mesh"] = 0
            n["skin"] = 0
    g.doc["nodes"] = nodes
    g.doc["scenes"] = ours["scenes"]
    g.doc["scene"] = ours.get("scene", 0)
    skin = json.loads(json.dumps(ours["skins"][0]))
    skin["inverseBindMatrices"] = g.array(accessor(ours, obin, skin["inverseBindMatrices"]).astype("<f4"), "MAT4")
    g.doc["skins"] = [skin]
    anims = []
    for c in clips:
        tacc = g.array(c["times"].reshape(-1, 1), "SCALAR")
        g.doc["accessors"][tacc]["min"] = [float(c["times"][0])]
        g.doc["accessors"][tacc]["max"] = [float(c["times"][-1])]
        samplers, channels = [], []
        for node, q in c["rot"].items():
            samplers.append({"input": tacc, "output": g.array(q, "VEC4"), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": "rotation"}})
        samplers.append({"input": tacc, "output": g.array(c["root_t"], "VEC3"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": c["root"], "path": "translation"}})
        anims.append({"name": c["name"], "samplers": samplers, "channels": channels})
    if anims:
        g.doc["animations"] = anims
    g.write(out)


def far_motion(glb: Path, samples: int = 16) -> dict:
    """Per clip, motion that comes from far bones (B2): per vertex sum_k w_k |M_k v - M_n v| over influences whose bone
    is > 0.15x height farther than the vertex's nearest bone n, / height. p99 over vertices (worst frame) and the
    largest share of vertices above 0.1x height at any frame. fail: share > 10% or p99 > 0.75x height."""
    import rig_motion
    doc, b = read_glb(glb)
    pr = doc["meshes"][0]["primitives"][0]
    a = pr["attributes"]
    P = accessor(doc, b, a["POSITION"]).astype(float)
    J = accessor(doc, b, a["JOINTS_0"]).astype(int)
    W = accessor(doc, b, a["WEIGHTS_0"]).astype(float)
    skn = doc["skins"][0]
    joints = skn["joints"]
    IBM = accessor(doc, b, skn["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    nodes = doc["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    H = float(np.ptp(P[:, 1]))
    Ph = np.c_[P, np.ones(len(P))]
    wm0 = rig_motion.world_mats(nodes, parent, {})
    jp = np.array([wm0[j][:3, 3] for j in joints])
    kid = {k: [joints.index(c) for c in nodes[j].get("children", []) if c in joints] for k, j in enumerate(joints)}

    def segd(p: np.ndarray, k: int) -> np.ndarray:
        out = np.linalg.norm(p - jp[k], axis=1)
        for c in kid[k]:
            ab = jp[c] - jp[k]
            t = np.clip(((p - jp[k]) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            out = np.minimum(out, np.linalg.norm(p - (jp[k] + t[:, None] * ab), axis=1))
        return out
    D = np.stack([segd(P, k) for k in range(len(joints))], 1)
    nearest = D.argmin(1)
    far = (np.take_along_axis(D, J, 1) - D.min(1)[:, None]) / H > 0.15
    out = {}
    for anim in doc.get("animations", []):
        dur, ch = rig_motion.channels(doc, b, anim)
        p99, share = 0.0, 0.0
        for t in np.linspace(0, dur, samples):
            wm = rig_motion.world_mats(nodes, parent, rig_motion.sample(nodes, ch, float(t)))
            M = np.array([wm[j] @ IBM[k] for k, j in enumerate(joints)])
            Y = np.einsum("nij,nj->ni", M[nearest, :3], Ph)
            mv = np.zeros(len(P))
            for k in range(J.shape[1]):
                sel = far[:, k] & (W[:, k] > 0)
                if sel.any():
                    Xk = np.einsum("nij,nj->ni", M[J[sel, k], :3], Ph[sel])
                    mv[sel] += W[sel, k] * np.linalg.norm(Xk - Y[sel], axis=1)
            mv /= H
            p99 = max(p99, float(np.percentile(mv, 99)))
            share = max(share, float((mv > 0.1).mean()))
        out[anim["name"]] = {"p99": round(p99, 3), "share": round(share, 4),
                             "level": "fail" if share > 0.10 or p99 > 0.75 else "pass"}
    return out


def blender_path() -> str | None:
    for p in (os.environ.get("BLENDER"), shutil.which("blender"),
              "/Applications/Blender.app/Contents/MacOS/Blender",
              str(Path.home() / "Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/MacOS/Blender")):
        if p and Path(p).is_file() and os.access(p, os.X_OK):
            return p
    return None


def export_fbx(glb: Path, fbx: Path, clips: list[str] | None = None) -> tuple[str, dict | None]:
    """The FBX for s&box through Blender (scripts/rig_fbx.py: today's export at 30 fps), re-imported and checked:
    clip count and names, bones, influences, fps. Writes rig/fbx_check.json next to the FBX. (note, check)."""
    import rig_fbx
    return rig_fbx.export_and_check(glb, fbx, clips)


def _plain(x):
    """numpy scalars and arrays in rig.json, as JSON."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


def template_info(template: dict, plan: dict) -> dict:
    """rig.json `template`: which template the rig used and how (mode built | fixed)."""
    out = {"mode": "built" if template.get("built") else "fixed", "rig": template["rig"] if not template.get("built") else "rig/template/rig.glb",
           "plan": plan.get("id"), "built": bool(template.get("built"))}
    if template.get("built"):
        out["dir"] = "rig/template/"
    return out


def slim_fbx_check(check: dict | None) -> dict | None:
    if not check:
        return None
    keep = ("ok", "problems", "warnings", "scene_fps", "max_influences", "up_axis", "unit_scale", "clips")
    out = {k: check.get(k) for k in keep if k in check}
    out["actions"] = len(check.get("actions") or [])
    out["bones"] = check.get("bones") if isinstance(check.get("bones"), int) else len(check.get("bone_names") or [])
    return out


# ---------------------------------------------------------------- the built path (a template made for this model)

TEMPLATE_FILES = ("skeleton.json", "build.json", "rig.glb", "clips.glb", "template.json", "validate.json",
                  "clips_report.json", "frames.json")


def _slim_build(info: dict) -> dict:
    """rig.json fit.build: the build report without the arrays skeleton.json already holds."""
    drop = {"skeleton", "pos", "parents", "parent_index", "names"}
    out = {k: v for k, v in info.items() if k not in drop}
    chk = dict(out.get("checks") or {})
    chk.pop("flagged_joints", None)
    out["checks"] = chk
    return json.loads(json.dumps(out, default=str))


def build_skeleton(asset: Path, source: Path, work: Path, plan: dict, pred_opt: str, facing_prior: dict, note,
                   keep: dict | None = None) -> dict:
    """Design section 9.2: predict (sampled, as the plan says), build the plan's skeleton on the mesh
    (rig_build), write rig/template/rig.glb and rig/template_posed.glb, and the fit record (yaw 0, scale 1, nothing
    unsupported). Returns {info, ref_store, pred_ctx}. Raises rig_build.BuildError, or whatever the prediction
    raised; the caller falls back to the class's template in auto mode.

    keep (filled before the build can raise): {"pred": Pred, "pred_ctx": {...}} once the prediction is made, or
    {"pred_error": str} when it failed, so the fallback binds this same prediction instead of making a second one
    (one prediction per rig; the t0.7 sample binds the vendor template as well as greedy does, scratchpad A/B)."""
    import rig_build
    import rig_pred
    import rig_template
    tdir = work / "template"
    pp = plan.get("pred") or {}
    beams = int(pp.get("beams") or 1)
    temp = pp.get("temperature")
    pred, pred_ctx, pmode = None, {}, "none"
    if pred_opt != "never":
        kw = {"beams": beams if beams > 1 else None, "temperature": float(temp) if temp else None}
        try:
            pglb, psecs, cached = rig_pred.predict(source, work, **kw)
            pred = rig_pred.Pred(pglb)
        except Exception as e:
            if keep is not None:
                keep["pred_error"] = str(e)[:300]
            raise
        pmode = "sampled" if kw["temperature"] else ("beam" if kw["beams"] else "default")
        pred_ctx = {"seconds": round(psecs, 1), "cached": cached, "joints": len(pred.names), "mode": pmode,
                    "file": f"rig/{pglb.name}", **{k: v for k, v in kw.items() if v}}
        if keep is not None:
            keep.update(pred=pred, pred_ctx=dict(pred_ctx))
        note(f"Predicted a skeleton of its own for the model ({len(pred.names)} joints, {pmode}, {psecs:.0f} s"
             + (", from the cache" if cached else "") + ")")
    info = rig_build.build(source, plan, pred, tdir, pred_mode=pmode)
    rig_template.write_rig(tdir / "skeleton.json", tdir / "rig.glb")
    rig_template.write_posed(tdir / "rig.glb", work / "template_posed.glb")
    chk = info.get("checks") or {}
    fc = facing_prior.get("facing_check") or {}
    declared = bool(facing_prior.get("canonical") or fc.get("agrees") or fc.get("flipped"))
    fitinfo = {"flags": "".join(sorted(FLAGS)),
               "registration": {"built": True, "scale_ratio": 1.0, "poor_match": False},
               "facing": {"yaw": 0, "decided_by": "declared" if declared else "built",
                          "check": info.get("facing_check")},
               "yaw": 0, "scale": [1.0, 1.0, 1.0], "offset": [0.0, 0.0, 0.0], "unsupported": [],
               "gate": chk.get("gate"), "build": _slim_build(info)}
    store = rig_template.ref_store(tdir / "rig.glb", fitinfo)
    lens = chk.get("lengths_H") or {}
    note(f"Built a {plan.get('label') or plan['id']} skeleton for the model: {info.get('joints')} joints, legs from "
         f"{info.get('legs_source')}, mirror error {chk.get('mirror_err_H', 0):.3f}× height, leg {lens.get('leg', 0):.2f} / "
         f"neck {lens.get('neck', 0):.2f} / tail {lens.get('tail', 0):.2f}× height")
    return {"info": info, "ref_store": store, "pred_ctx": pred_ctx}


def make_clips(asset: Path, plan: dict, wanted: list[str], ours: dict, obin: bytes, skel: Skeleton, ref_store: dict,
               src: dict, sbin: bytes, P: np.ndarray, faces: np.ndarray, J: np.ndarray, W: np.ndarray, height: float,
               names_j: list[str], parent_j: dict[int, int], jpos: np.ndarray, info: dict, note):
    """Design section 9.2.8-10: synthesise the plan's clips on the real (cleaned) weights, write
    rig/template/clips.glb + clips_report.json + template.json, load and verify the template, and bake the clips
    that passed onto our skeleton (a near-identity bake). Returns (clips, missing, J, W, clips_report, clips_failed,
    template); J/W are the weights the clips were gated on (after any wing-weight repair)."""
    import rig_clips
    import rig_template
    tdir = asset / "rig" / "template"
    known = [c["name"] for c in plan.get("clips") or []]
    plan_run = plan if not wanted else {**plan, "clips": [c for c in plan["clips"] if c["name"] in wanted]}
    missing = [n for n in wanted if n not in known]
    skel_s = {"names": names_j, "parent": [parent_j.get(k, -1) for k in range(len(names_j))], "pos": jpos,
              "H": info.get("H", height), "cx": info.get("cx")}
    pre = None
    import rig_build
    # weight on the other side's bones (a staggered stance's inner toes bound to the far foot) drags a planted foot
    J, W, side_fix = rig_build.side_weight_repair(P, J, W, skel_s, plan)
    if side_fix.get("applied"):
        note(f"Moved {side_fix.get('moved_mass', 0):.1f} weight on {side_fix.get('verts')} points off the other "
             f"side's bones ({', '.join(list(side_fix.get('bones') or {})[:4])}) before making the clips")
    # floor points (a hind toe) bound to the tarsus swing through the floor in a stride: they belong to the foot
    J, W, sole_fix = rig_build.sole_weight_repair(P, J, W, skel_s, plan)
    if sole_fix.get("applied"):
        note(f"Moved {sole_fix.get('moved_mass', 0):.1f} weight on {sole_fix.get('verts')} floor points onto the feet "
             f"(from {', '.join(list(sole_fix.get('bones') or {})[:4])}) before making the clips")
    if (plan.get("caps") or {}).get("wings"):  # folded wings the skin barely gave weight to can't move in Flap
        J, W, pre = rig_build.wing_weight_repair(P, J, W, skel_s, plan)
        if pre.get("applied"):
            note(f"The wings owned {pre.get('share_before', 0):.1%} of the body sides; gave {pre.get('reassigned')} points "
                 f"to the wing bones ({pre.get('share_after', 0):.1%}) before making the clips")
    region = None
    if (plan.get("caps") or {}).get("wings"):  # the wing region (points nearest a wing bone) on the wings, not the body
        J, W, region = rig_build.wing_region_repair(P, J, W, skel_s, plan)
        if region.get("applied"):
            note(f"The wing region put {region.get('share_before', 0):.0%} of its weight on the wings; moved "
                 f"{region.get('reassigned')} points onto them ({region.get('share_after', 0):.0%}) and took the wings off "
                 f"{region.get('stripped')} points far from them before making the clips")
    sclips, crep = rig_clips.synthesise(plan_run, skel_s, P, faces, J, W, height, names=names_j)
    smooth = crep.get("wing_weight_smooth")
    if smooth and smooth.get("applied"):
        note(f"Softened the wings' weight edge over {smooth.get('verts')} points before making the clips")
    J = np.asarray(crep["J"]).astype(np.int64)
    W = np.asarray(crep["W"], np.float32)
    wr = crep.get("wing_weight_repair") or {}
    if wr.get("applied"):
        note(f"The wings owned {wr.get('share_before', 0):.1%} of the body sides; gave {wr.get('reassigned')} points "
             f"to the wing bones ({wr.get('share_after', 0):.1%}) so Flap can move them")
    ok = [c for c in sclips if c["passed"]]
    failed = crep.get("clips_failed") or []
    for f in failed:
        note(f"warn: couldn't make {f['name']} for this model: " + "; ".join(f.get("reasons") or ["no reason given"]))
    rig_clips.write_report(tdir / "clips_report.json", crep)
    if not ok:
        raise SystemExit("rig: none of the body plan's clips kept the body together on this model: "
                         + "; ".join(f"{f['name']}: {', '.join(f.get('reasons') or [])}" for f in failed))
    rig_template.write_clips(tdir / "rig.glb", src, sbin, J, W, ok, tdir / "clips.glb", names=names_j)
    rig_template.write_template_json(tdir, plan, [c["name"] for c in ok])
    template = rig_template.load_template(asset)
    # a clip that failed its gates is a `clips` fail row (clips_failed), not a broken template: verify the defaults made
    made = {c["name"] for c in ok}
    probs = rig_template.verify(tdir / "rig.glb", tdir / "clips.glb",
                                {**plan, "default": [d for d in plan.get("default") or [] if d in made]})
    errs = [p for p in probs if p["level"] == "error"]
    if errs:
        raise SystemExit("rig: the template built for this model breaks the template contract: "
                         + "; ".join(f"{p['key']}: {p['msg']}" for p in errs))
    for p in probs:
        note(f"Template check: {p['msg']}")
    names = [c["name"] for c in ok]
    clips, gone = bake_clips(template, names, ours, skel, ref_store)
    by = {c["name"]: c for c in ok}
    for c in clips:
        s_ = by[c["name"]]
        c["track"] = s_.get("track")
        c["loop"] = s_.get("loop")
    note(f"Made {len(ok)} clips for this model in {crep.get('seconds', 0):.0f} s: " + ", ".join(
        f"{c['name']} (energy {c.get('energy', 0):.2f})" for c in ok))
    report = {"file": "rig/template/clips_report.json", "ok": names, "failed": [f["name"] for f in failed],
              "seconds": crep.get("seconds"), "energy": {c["name"]: round(float(c.get("energy", 0)), 3) for c in ok},
              "wing_weight_repair": {k: v for k, v in wr.items() if k != "bones"} or None,
              "wing_weight_repair_before": {k: v for k, v in (pre or {}).items() if k != "bones"} or None,
              "side_weight_repair": side_fix,
              "sole_weight_repair": sole_fix,
              "wing_region_repair": {k: v for k, v in (region or {}).items() if k != "bones"} or None,
              "wing_weight_smooth": smooth,
              "contacts": {c["name"]: {k: np.asarray(v, bool).reshape(-1).astype(int).tolist()
                                       for k, v in ((c.get("track") or {}).get("contact") or {}).items()} for c in ok}}
    return clips, missing + gone, J, W, report, failed, template


def apply_reframe(work: Path, plan: dict, built: bool, posed: Path, names_j: list[str], ours: dict, obin: bytes,
                  src: dict, sbin: bytes, P: np.ndarray, J: np.ndarray, W: np.ndarray, clips: list[dict],
                  out_glb: Path, note) -> dict:
    """Design section 5: designed local frames, applied last (after the floor pass and the checks) and only when
    the skinning comes out identical. Rewrites out_glb; returns what was done."""
    import rig_frames
    try:
        if built:
            Q = rig_frames.frames(json.loads((work / "template" / "skeleton.json").read_text()), plan)
            qsrc = "rig/template/skeleton.json"
        else:
            Q = rig_frames.frames_from_template(posed, names_j)
            qsrc = "rig/template_posed.glb"
        o2, b2, c2 = rig_frames.reframe(ours, obin, clips, Q)
        H = max(float(np.ptp(P[:, 1])), 1e-9)
        delta = rig_frames.check_identical((ours, obin, clips), (o2, b2, c2), P, J, W) / H
    except Exception as e:  # noqa: BLE001 - the rig stands without bone rolls
        note(f"warn: couldn't give the joints their frames ({str(e)[:160]}); kept identity rests")
        return {"applied": False, "error": str(e)[:300]}
    if delta > 1e-4:
        note(f"warn: the joint frames would move the skin ({delta:.1e}× height); kept identity rests")
        return {"applied": False, "max_delta_H": delta, "source": qsrc}
    write_rigged(out_glb, o2, b2, src, sbin, J, W, c2)
    fdir = work / "template" if built else work
    rig_frames.write_frames(fdir / "frames.json", Q, {"source": qsrc})
    note(f"Gave the joints designed frames (from {qsrc}); the skinning is unchanged ({delta:.1e}× height)")
    return {"applied": True, "max_delta_H": delta, "source": qsrc, "frames": str((fdir / "frames.json").relative_to(work.parent))}


def render_sheet(asset: Path, note) -> dict:
    """rig/preview/contact_sheet.png and frames.json (scripts/rig_render.py, CPU)."""
    try:
        import rig_render
        t0 = time.time()
        res = rig_render.render(asset, {"frames": "auto"})
    except Exception as e:  # noqa: BLE001 - a missing picture never fails the rig
        note(f"warn: couldn't render the contact sheet ({str(e)[:160]})")
        return {"error": str(e)[:300]}
    rel = lambda p: str(Path(p).relative_to(asset)) if p and Path(p).is_absolute() and asset in Path(p).parents else p  # noqa: E731
    out = {"sheet": rel(res.get("sheet")), "frames": rel(str(Path(res.get("out") or asset / "rig" / "preview") / "frames.json")),
           "min_y_rel_min": res.get("min_y_rel_min"), "seconds": round(time.time() - t0, 1),
           "rows": res.get("sheet_rows"), "cols": res.get("sheet_cols")}
    fr = res.get("frames") or []
    feet = [f.get("foot_mesh_y_rel") for f in fr if f.get("foot_mesh_y_rel") is not None]
    if feet:
        out["foot_mesh_y_rel"] = [round(float(min(feet)), 4), round(float(max(feet)), 4)]  # lowest leg point, min / max
    note(f"Rendered the contact sheet ({len(fr)} frames, {out['seconds']:.0f} s): {out['sheet']}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--class", dest="cls", default="", help="humanoid | quadruped | multileg | avian (default: ask the planner)")
    ap.add_argument("--source", default="", help="mesh in out/<stem>/ to rig (default psx.glb, then textured_photo.glb, textured.glb)")
    ap.add_argument("--clips", default="", help="comma-separated clip names (default: a set per class)")
    ap.add_argument("--bind", choices=("corners", "weld"), default="corners",
                    help="skin every corner as it is (default), or weld corners at the same point first")
    ap.add_argument("--straighten", choices=("auto", "off"), default="auto",
                    help="auto: rig a straightened copy when an animal's head is turned from its body (default)")
    ap.add_argument("--no-fbx", action="store_true")
    env = os.environ.get
    ap.add_argument("--template", choices=("auto", "fixed", "build"), default=env("RIG_TEMPLATE") or "auto",
                    help="auto (default): build a skeleton for the model when its body plan has a builder and no fixed "
                         "template (a ground bird), else the class's template; fixed: the class's template (the old "
                         "path); build: always build (an error when the body plan has no builder)")
    ap.add_argument("--plan", default=env("RIG_PLAN") or "", help="body plan id (scripts/rig_plans/); default: intent.json "
                    "rig.plan, then the class rule")
    ap.add_argument("--binder", choices=("skin", "pred", "heat"), default=env("RIG_BINDER") or "skin",
                    help="skin: skin-tokens (default); pred: the predicted rig's weights; heat: Blender's heat bind")
    ap.add_argument("--pred", choices=("auto", "always", "never"), default=PRED_ENV.get(env("RIG_PRED") or "auto", "auto"),
                    help="the model's own predicted skeleton as the fitting target: auto (poor match or a missed body), "
                         "always, never")
    ap.add_argument("--floor", choices=("0", "1"), default=env("RIG_FLOOR") or "1", help="the floor pass (default 1)")
    ap.add_argument("--render", choices=("0", "1"), default=env("RIG_RENDER") or "1",
                    help="render the contact sheet (rig/preview/) after the rig (default 1)")
    ap.add_argument("--reframe", choices=("0", "1", "auto"), default=env("RIG_REFRAME") or "auto",
                    help="give the joints designed local frames (bone rolls) as the last step; auto (the default): on "
                         "for a skeleton built for the model (no vendor rolls to keep), off on the fixed path")
    args = ap.parse_args(argv)

    asset = OUT / args.stem
    if not asset.is_dir():
        raise SystemExit(f"rig: no asset {args.stem}")
    print("STAGES classify rest pose skin clean clips check" + ("" if args.no_fbx else " fbx"), flush=True)
    stage("classify", "Working out the body plan")
    # no mesh is the real problem for a prop with only a picture (or none): say so before asking the planner
    source = asset / args.source if args.source else next(
        (asset / n for n in ("psx.glb", "textured_photo.glb", "textured.glb") if (asset / n).is_file()), None)
    if source is None or not source.is_file():
        raise SystemExit("rig: no textured mesh to rig yet; make the model first")
    import rig_plan
    plan_cls = None
    if args.plan:
        try:
            plan_cls = rig_plan.load(args.plan).get("class")
        except KeyError:
            raise SystemExit(f"rig: unknown body plan {args.plan!r} ({', '.join(p['id'] for p in rig_plan.all_plans())})")
        if args.cls and plan_cls and args.cls != plan_cls:  # as POST /rig answers 400
            raise SystemExit(f"rig: plan {args.plan} is for class {plan_cls}, not {args.cls}")
    want_cls = args.cls or plan_cls  # the plan says the class: no need to ask the planner
    planner = classify(asset) if not want_cls else {"choice": want_cls, "confidence": None}
    cls = planner["choice"]
    if cls in UNSUPPORTED:
        raise SystemExit(f"rig: {UNSUPPORTED[cls]}")
    if cls not in CLASSES:
        raise SystemExit(f"rig: unknown class {cls!r} ({', '.join(CLASSES)})")
    template = TEMPLATES[cls]
    work = asset / "rig"
    work.mkdir(exist_ok=True)
    for old in ("inspect.json", "summary.json", "fbx_check.json"):  # the last version's checks must never stand in
        (work / old).unlink(missing_ok=True)
    for old in TEMPLATE_FILES:  # nor a template built by an earlier run
        (work / "template" / old).unlink(missing_ok=True)
    original = source
    notes: list[str] = []

    def note(text: str) -> None:
        notes.append(text)
        print("rig: " + text, flush=True)

    degraded: list[dict] = []  # rig.json `degraded`: every fallback, said out loud (DEGRADED <stage> <code>: <message>)

    def degrade(code: str, message: str) -> None:
        degraded.append({"stage": "rig", "code": code, "message": message})
        print(f"DEGRADED rig {code}: {message}", flush=True)

    stage("rest", "Checking the rest pose")
    sdoc, sbin0 = read_glb(source)
    mesh_points = accessor(sdoc, sbin0, sdoc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).astype(float)
    rest = {"measured": rest_check.measure(mesh_points), "straightened": None}
    straight = work / "rest_straight.glb"
    straight.unlink(missing_ok=True)
    if args.straighten == "auto" and rest["measured"] and rest["measured"]["turned"]:
        rest["straightened"] = rest_check.straighten(source, straight)["after"]
        note(f"Head was turned {deg(rest['measured']['head_yaw_deg'])}; rigged a straightened copy "
             f"({deg(rest['straightened']['head_yaw_deg'])}). Your model file is unchanged.")
        source = straight
        sdoc, sbin0 = read_glb(source)
        mesh_points = accessor(sdoc, sbin0, sdoc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).astype(float)
    elif rest["measured"]:
        note(f"Head is turned {deg(rest['measured']['head_yaw_deg'])} from the body; rigged as made" if rest["measured"]["turned"]
             else f"Head lines up with the body ({deg(rest['measured']['head_yaw_deg'])})")

    try:
        plan = rig_plan.resolve(asset, cls, args.plan or None, mesh_points, write=False)
    except KeyError:
        raise SystemExit(f"rig: unknown body plan {args.plan!r} ({', '.join(p['id'] for p in rig_plan.all_plans())})")
    if (plan.get("resolved") or {}).get("class_mismatch"):
        raise SystemExit(f"rig: {plan['resolved']['class_mismatch']}")
    (work / "plan.json").unlink(missing_ok=True)  # written below with the plan the run actually uses
    mode = decide(plan, args.template)
    facing_prior = {k: plan[k] for k in ("facing", "facing_source", "canonical", "facing_check") if k in plan}
    note(f"Body plan: {plan.get('label') or plan['id']} ({plan['resolved']['why']}); "
         + ("building a skeleton for this model" if mode == "build" else "the class's template"))

    stage("pose", "Building a skeleton for the model" if mode == "build" else "Posing the template like the model")
    posed = work / "template_posed.glb"
    surface = np.asarray(trimesh.load(source, force="mesh").sample(40000, seed=0), float)
    import rig_fit
    sprim = sdoc["meshes"][0]["primitives"][0]
    faces = accessor(sdoc, sbin0, sprim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in sprim else np.arange(len(mesh_points)).reshape(-1, 3)
    built, build_error = None, None
    build_keep: dict = {}  # the build's prediction, reused by the fallback (one prediction per rig)
    if mode == "build":
        try:
            built = build_skeleton(asset, source, work, plan, args.pred, facing_prior, note, keep=build_keep)
        except Exception as e:  # noqa: BLE001 - BuildError, a failed prediction or a template the verifier rejects
            import rig_build
            code = e.code if isinstance(e, rig_build.BuildError) else type(e).__name__
            msg = e.msg if isinstance(e, rig_build.BuildError) else str(e)[:300]
            if args.template == "build":
                raise SystemExit(f"rig: couldn't build a skeleton ({code}: {msg})")
            build_error = {"code": code, "msg": msg}
            note(f"warn: couldn't build a skeleton ({code}: {msg}); using the class's vendor template")
            degrade("build_failed", f"{code}: {msg}; rigged on the class's vendor template instead")
            mode = "fixed"
    plan_used = plan if built is not None else fixed_plan(plan, cls)
    if built is not None:  # rig/plan.json: the plan the built template was made with (rig_clips/rig_frames read it)
        rig_plan.write_plan(work, plan_used)
    if built is None:  # the fixed path: today's template, posed and fitted onto the mesh
        vlm_cache = work / "facing_vlm.json"

        def vlm() -> dict:
            key = f"{source.stat().st_size}:{int(source.stat().st_mtime)}"
            if vlm_cache.is_file():
                got = json.loads(vlm_cache.read_text())
                if got.get("key") == key:
                    return got["vlm"]
            got = rig_fit.vlm_cue(source, blender_path(), work / "facing")
            vlm_cache.write_text(json.dumps({"key": key, "vlm": got}))
            return got
        facing = {"geometric": rig_fit.geometric_cues(mesh_points, cls, rig_fit._texel_colours(sdoc, sbin0)), "vlm": vlm,
                  "prior": facing_prior_yaw(facing_prior, original.name)}
        import rig_pred
        # The model's own predicted skeleton (skin-tokens `rig`) is the fitting target when the template is a poor
        # body-plan match (rig_fit.POOR_MATCH: a chicken on the flying bird) or the geometric fit misses the body.
        # RIG_PRED: auto (default) | always | 0 (never).
        use_pred = {"never": "0"}.get(args.pred, args.pred)
        pred_ctx: dict = {}
        holder: dict = {}
        mesh_tm = trimesh.Trimesh(mesh_points, faces, process=False)
        red = rig_pred._red(sdoc, sbin0)

        if "pred" in build_keep:  # the build predicted, then failed: bind that same prediction, don't make another
            pr_ = build_keep["pred"]
            holder["pred"] = pr_
            same = len(pr_.P) == len(mesh_points) and float(np.abs(pr_.P - mesh_points).max()) < 1e-6
            pred_ctx.update(build_keep["pred_ctx"], same_vertices=same, independent=rig_pred.label(pr_, None, red),
                            reused_from="build")
            note(f"Reusing the skeleton predicted for the build ({len(pr_.names)} joints, {pred_ctx.get('mode')}) "
                 "to fit the template")
        elif "pred_error" in build_keep:  # the build's prediction failed: don't try a second one
            holder["pred"] = None
            pred_ctx.update(error=build_keep["pred_error"], reused_from="build")
            degrade("pred_failed", f"no predicted skeleton (the build's prediction failed: {build_keep['pred_error'][:120]}); "
                    "the template is fitted by its own shape")

        def get_pred():
            if "pred" not in holder:
                holder["pred"] = None
                try:
                    pglb, psecs, cached = rig_pred.predict(source, work)
                    pr_ = rig_pred.Pred(pglb)
                    same = len(pr_.P) == len(mesh_points) and float(np.abs(pr_.P - mesh_points).max()) < 1e-6
                    pred_ctx.update(seconds=round(psecs, 1), cached=cached, joints=len(pr_.names), same_vertices=same,
                                    independent=rig_pred.label(pr_, None, red))
                    note(f"Predicted a skeleton of its own for the model ({len(pr_.names)} joints, {psecs:.0f} s"
                         + (", from the cache" if cached else "") + ")")
                    holder["pred"] = pr_
                except Exception as e:  # noqa: BLE001 - no prediction: the geometric fit (rig_fit) does it alone
                    pred_ctx["error"] = str(e)[:300]
                    note(f"Couldn't predict a skeleton ({str(e)[:120]}); fitting the template by its own shape")
                    degrade("pred_failed", f"no predicted skeleton ({str(e)[:120]}); the template is fitted by its own shape")
            return holder["pred"]

        def pred_fn(yaw: int):
            pr_ = get_pred()
            if pr_ is None:
                return None
            f = rig_fit.yaw_matrix(yaw) @ np.array([0, 0, 1.0])
            lab = rig_pred.label(pr_, f, red)
            pred_ctx.update(labels=lab, template_yaw=int(yaw), head_chain_yaw=rig_pred.head_chain_yaw(pr_, lab),
                            independent_head_chain_yaw=rig_pred.head_chain_yaw(pr_, pred_ctx["independent"]))
            pred_ctx["yaw_agrees"] = pred_ctx["independent_head_chain_yaw"] == int(yaw)
            if lab["legs"] < 2:
                pred_ctx["skipped"] = f"only {lab['legs']} predicted legs"
                return None
            lines = rig_pred.target_lines(pr_, lab, mesh_tm, float(mesh_points[:, 1].min()))
            pred_ctx["lines"] = {k: ({kk: vv for kk, vv in v.items() if kk != "line"} if k != "legs" else
                                     [{kk: vv for kk, vv in q.items() if kk != "line"} for q in v]) for k, v in lines.items()}
            pred_ctx["used"] = True
            return lines

        can_pred = use_pred != "0" and cls in rig_pred.PATHS

        def pose(force: bool) -> tuple[dict, dict, np.ndarray, list, dict, dict]:
            _, rs, pn_ = posed_template(template, posed, mesh_points, surface, facing, faces,
                                        pred_fn if can_pred else None, force or use_pred == "always", plan=plan_used)
            if pred_ctx.get("used") and pred_ctx.get("labels"):
                lab = pred_ctx["labels"]
                pn_ = pn_ + ["Predicted chains: " + ", ".join(f"{d['label']} {d['names'][0]}..{d['names'][-1]}" for d in lab["chains"])
                             + f"; its head chain points {pred_ctx['independent_head_chain_yaw']}° on its own cues "
                             f"({pred_ctx['independent']['how']}), the template was turned {pred_ctx['template_yaw']}°"
                             + ("" if pred_ctx["yaw_agrees"] else " (they disagree)")]
            pd_, _ = read_glb(posed)
            sk_ = Skeleton(pd_)
            w_ = sk_.world({})
            pj_ = pd_["skins"][0]["joints"]
            jp_ = np.array([w_[i][0] for i in pj_])
            sg_ = [(pj_.index(sk_.parent[i]), k) for k, i in enumerate(pj_) if sk_.parent.get(i) in pj_]
            return rs, pd_, jp_, sg_, rig_fit.pose_gate(jp_, sg_, mesh_points, faces, surface), {"notes": pn_, "pj": pj_}

        ref_store, pdoc, jp_pose, segs, gate, extra = pose(False)
        for n in extra["notes"]:
            note(n)
        if gate["level"] == "fail" and can_pred and not pred_ctx.get("used"):
            note(f"Pose check: {gate['joints_outside_box']:.0%} of joints outside the model's box, "
                 f"{gate['bone_far_outside_0p08H']:.0%} of the bones well outside it; fitting again on the model's predicted skeleton")
            first_gate = gate
            ref_store, pdoc, jp_pose, segs, gate, extra = pose(True)
            for n in extra["notes"]:
                note(n)
            ref_store["__fit__"]["first_gate"] = first_gate
        pj = extra["pj"]
        pred = holder.get("pred") if pred_ctx.get("used") else None
        fitinfo = ref_store["__fit__"]
        fitinfo["gate"] = gate
        if pred is not None and pred_ctx.get("labels"):
            Hm = float(np.ptp(mesh_points[:, 1]))
            unsup = set(fitinfo.get("unsupported") or [])
            pn = [pdoc["nodes"][i].get("name", "") for i in pj]
            skip = {k for k, n in enumerate(pn) if ("wings" in unsup and n.startswith("wing")) or n == "root"}
            t_segs = [(jp_pose[a], jp_pose[b]) for a, b in segs if a not in skip and b not in skip]
            p_segs = [(pred.jp[pred.par[k]], pred.jp[k]) for k in range(len(pred.names)) if pred.par[k] >= 0]
            keep = [k for k in range(len(pn)) if k not in skip]
            d_pt = rig_pred.cd_j2b(pred.jp, t_segs) / Hm
            d_tp = rig_pred.cd_j2b(jp_pose[keep], p_segs) / Hm
            pred_ctx["cd_j2b"] = {"pred_to_template": round(float(d_pt.mean()), 4), "template_to_pred": round(float(d_tp.mean()), 4),
                                  "symmetric": round(float((d_pt.mean() + d_tp.mean()) / 2), 4), "pred_to_template_max": round(float(d_pt.max()), 4)}
            note(f"Predicted joints sit {pred_ctx['cd_j2b']['pred_to_template']:.3f}× height from the fitted template's bones "
                 f"(template joints {pred_ctx['cd_j2b']['template_to_pred']:.3f}× from the predicted bones)")
        note(f"Pose check: {gate['joints_outside_box']:.0%} of joints outside the model's box, "
             f"{gate['bone_far_outside_0p08H']:.0%} of the bones well outside the model; skeleton/model size "
             + " / ".join(f"{x:.2f}" for x in gate["skeleton_over_mesh_extent"]))
        if fitinfo.get("poor_match"):
            degrade("poor_match", f"scale {float(fitinfo.get('scale_ratio') or 0):.1f}x between axes; the {cls} "
                    "template is a poor body-plan match for this model")
        if (gate["level"] == "fail" or os.environ.get("RIG_FORCE_FALLBACK")) and pred is not None and pred_ctx.get("labels") \
                and pred_ctx.get("same_vertices"):
            # E3 fallback: the template can't be fitted; ship the predicted rig itself with the template's clips mapped
            # chain by chain onto it
            note(f"The {cls} template doesn't fit ({gate['joints_outside_box']:.0%} of joints outside the box, "
                 f"{gate['bone_far_outside_0p08H']:.0%} of bones well outside); rigging with the model's own predicted skeleton")
            stage("skin", "Using the predicted skin weights")
            names = [c.strip() for c in args.clips.split(",") if c.strip()] or template["default"]
            needs = rig_plan.spec(plan_used, cls, "needs", {}) or {}
            has = {d["label"] for d in pred_ctx["labels"]["chains"]}
            unsupported_clips = [n for g, cl in needs.items() if g not in has for n in cl if n in names]
            names = [n for n in names if n not in unsupported_clips]
            degrade("pred_skeleton", "the template didn't fit; rigged on the model's own predicted skeleton, "
                    "clips mapped chain by chain, no floor pass")
            if unsupported_clips:
                degrade("clips_dropped", ",".join(unsupported_clips))
            stage("clips", "Baking animation clips")
            fclips, finfo = rig_pred.fallback_clips(pred, pred_ctx["labels"], template, cls, names, float(fitinfo.get("yaw", 0)),
                                                    plan=plan_used)
            pdoc2 = json.loads(json.dumps(pred.doc))
            pdoc2.pop("animations", None)
            src, sbin = read_glb(source)
            out_glb = asset / "rigged.glb"
            write_rigged(out_glb, pdoc2, pred.b, src, sbin, pred.J, pred.W.astype(np.float32), fclips)
            far = far_motion(out_glb)
            report = {"class": cls, "template": template_info(template, plan_used), "plan": plan_used["id"],
        "plan_resolved": plan["id"],
                      "facing_prior": facing_prior, "build_error": build_error, "source": original.name, "rigged": out_glb.name, "rest": rest,
                      "joints": len(pred.names), "clips": [c["name"] for c in fclips], "missing_clips": [], "skin_seconds": 0.0,
                      "binder": "pred-rig", "fallback": {"why": "pose gate", "gate": gate, **finfo}, "fit": fitinfo,
                      "far_motion": far, "unsupported_clips": unsupported_clips, "made": int(time.time()), "fbx": None,
                      "fbx_note": "skipped", "pred": {k: v for k, v in pred_ctx.items() if k != "independent"},
                      "degraded": degraded}
            (work / "rig.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
            bad = [c for c, v in far.items() if v["level"] == "fail"]
            summary = {"version": 1, "made": report["made"], "class": cls, "level": "fail" if bad else "warn",
                       "line": "Rigged on the model's own predicted skeleton (the template didn't fit)", "tears": 0,
                       "motion": None, "head": None, "worst_stretch": None,
                       "problems": [{"level": "warn", "key": "fallback", "label": "Template didn't fit: predicted skeleton, clips mapped chain by chain, no floor pass"}]
                       + [{"level": "fail", "key": "far", "label": f"{c}: points pulled by far bones"} for c in bad]}
            (work / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
            note("Fallback rig written: " + ", ".join(report["clips"]) + (" (left out " + ", ".join(unsupported_clips) + ")" if unsupported_clips else ""))
            if not args.no_fbx:
                stage("fbx", "Writing the FBX for s&box")
                fbx = work / f"{args.stem.replace('-', '_')}.fbx"
                report["fbx_note"], fchk = export_fbx(out_glb, fbx, [c["name"] for c in fclips])
                report["fbx_check"] = slim_fbx_check(fchk)
                report["fbx"] = f"rig/{fbx.name}" if fbx.is_file() else None
                (work / "rig.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
            print("RIG_RESULT " + json.dumps(report, default=str), flush=True)
            return 0
        if gate["level"] == "fail" and "f" in FLAGS:
            import rig_plan
            others = [p["id"] for p in rig_plan.all_plans() if p.get("class") == cls and p.get("id") != plan_used.get("id")]
            advice = (f"try another body plan ({', '.join(others)})" if others else
                      f"the {cls} class has no other body plan, so this model can't be rigged with it yet")
            raise SystemExit(f"rig: the {cls} template doesn't fit this model: {gate['joints_outside_box']:.0%} of its joints "
                             f"are outside the model and {gate['bone_far_outside_0p08H']:.0%} of its bones lie well outside it "
                             f"(red above 20% / 30%). Not skinning a skeleton that misses the body; {advice}.")

    else:  # the built path: the skeleton built for this model is the template (yaw 0, scale 1, no fallback)
        import rig_pred
        ref_store = built["ref_store"]
        fitinfo = ref_store["__fit__"]
        pred_ctx = built["pred_ctx"]
        pred = None
        gate = fitinfo.get("gate") or {}
        if gate:
            note(f"Pose check: {gate.get('joints_outside_box', 0):.0%} of joints outside the model's box, "
                 f"{gate.get('bone_far_outside_0p08H', 0):.0%} of the bones well outside the model")

    stage("skin", "Learning the skin weights")
    raw = work / "skinned_raw.glb"
    inv = None
    if args.bind == "weld":
        inv = weld(source, work / "welded.glb")
        note(f"Joined {len(inv)} corners into {int(inv.max()) + 1} points before skinning")
    binder = args.binder
    binder_report = None
    if binder == "pred" and pred is not None and pred_ctx.get("same_vertices") and pred_ctx.get("labels"):
        t0 = time.time()
        bdoc, bbin = rig_pred.ours_from_posed(posed)
        srcd, srcb = read_glb(source)
        Jb, Wb, tinfo = rig_pred.transfer(pred, pred_ctx["labels"], mesh_points, bdoc, cls, fitinfo.get("unsupported") or [])
        write_rigged(raw, bdoc, bbin, srcd, srcb, Jb, Wb, [])
        inv = None
        secs = time.time() - t0
        pred_ctx["transfer"] = tinfo
        note(f"Moved the predicted skeleton's weights onto the template's bones in {secs:.1f} s ({tinfo['bones_used']} bones used)")
    else:
        if binder == "heat":
            import rig_heat
            try:
                hrep = rig_heat.run_heat(work / "welded.glb" if inv is not None else source, posed, raw)
                secs = float(hrep["seconds"])
                for n in hrep.get("notes") or []:
                    note(n)
                binder_report = {k: v for k, v in hrep.items() if k != "notes"}
                note(f"Heat-bound the skin weights in Blender in {secs:.0f} s")
            except rig_heat.HeatError as e:
                binder_report = {"error": str(e)[:400]}
                note(f"warn: the heat bind failed ({str(e)[:160]}); learning the weights with skin-tokens instead")
                binder = "skin"
        if binder != "heat":
            binder = "skin"
            secs = run_skin(work / "welded.glb" if inv is not None else source, posed, raw)
            note(f"Learned the skin weights in {secs:.0f} s")

    stage("clean", "Cleaning up the weights")
    ours, obin = read_glb(raw)
    src, sbin = read_glb(source)
    oprim = ours["meshes"][0]["primitives"][0]["attributes"]
    P = accessor(src, sbin, src["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).astype(float)
    J = accessor(ours, obin, oprim["JOINTS_0"]).astype(np.int64)
    W = accessor(ours, obin, oprim["WEIGHTS_0"]).astype(float)
    if inv is not None:
        J, W = J[inv], W[inv]
    if len(J) != len(P):
        raise SystemExit("rig: skin-tokens changed the vertex count; can't carry the texture across")
    skel = Skeleton(ours)
    joints = ours["skins"][0]["joints"]
    world = skel.world({})
    jpos = np.array([world[i][0] for i in joints])
    child = {}
    for k, i in enumerate(joints):
        kids = [joints.index(c) for c in skel.nodes[i].get("children", []) if c in joints]
        if kids:
            child[k] = kids[0]
    height = float(np.ptp(P[:, 1]))
    raw_J, raw_W = J.copy(), W.copy()
    local_weights.reassigned = 0
    root_k = next(k for k, i in enumerate(joints) if skel.parent.get(i) not in joints)
    # the root's bone runs from the floor to the hips through the belly; its weight is folded away anyway
    skip_k = [root_k]
    if fitinfo.get("unsupported"):  # folded, unsupported chains (a ground bird's wings) never stand in as the nearest bone
        groups = (rig_pred.paths_for(cls, plan_used) or {}).get("groups", {})
        roots_u = [r_ for g in fitinfo["unsupported"] for r_ in groups.get(g, [])]
        sub = [joints.index(skel.index[r_]) for r_ in roots_u if r_ in skel.index]
        while sub:
            k = sub.pop()
            skip_k.append(k)
            sub += [joints.index(c) for c in skel.nodes[joints[k]].get("children", []) if c in joints]
    J, W, dropped = local_weights(P, J, W, jpos, child, height, skip=tuple(skip_k))
    if local_weights.reassigned:
        note(f"{local_weights.reassigned} points had only far bones and went wholly to their nearest bone")
    kept = W > 0
    note(f"Ignored {dropped} pulls from bones too far away")
    names_j = [skel.nodes[i].get("name", "") for i in joints]
    parent_j = {k: joints.index(skel.parent[i]) for k, i in enumerate(joints) if skel.parent.get(i) in joints}
    root_j = next(k for k in range(len(joints)) if k not in parent_j)
    hips_j = names_j.index("hips") if "hips" in names_j else root_j
    J, W, folded = fold_tips(J, W, parent_j, root_j, hips_j)
    note(f"Moved {folded:.1%} of the weight from root and tip bones to their parents")
    J, W, split = share_corners(P, J, W)
    note(f"Gave every copy of a point the same bones ({split} points had copies on different bones)" if split
         else "Checked that every copy of a point has the same bones, so corners move together")

    out_glb = asset / "rigged.glb"
    wanted = [c.strip() for c in args.clips.split(",") if c.strip()]
    unsupported_clips, trial, trial_stretch, dropped_clips = [], [], {}, []
    clips_report, clips_failed = None, []
    if built is not None:
        stage("clips", "Making animation clips for this model")
        clips, missing, J, W, clips_report, clips_failed, template = make_clips(
            asset, plan, wanted, ours, obin, skel, ref_store, src, sbin, P, faces, J, W, height, names_j, parent_j,
            jpos, built["info"], note)
        names = [c["name"] for c in clips]
    else:
        stage("clips", "Baking animation clips")
        names = wanted or template["default"]
        if fitinfo.get("unsupported"):
            needs = rig_plan.spec(plan_used, cls, "needs", {}) or {}
            unsupported_clips = [n for g in fitinfo["unsupported"] for n in needs.get(g, []) if n in names]
            if unsupported_clips:
                names = [n for n in names if n not in unsupported_clips]
                note("Left out " + ", ".join(unsupported_clips) + " (the model has no " + ", ".join(
                    g for g in fitinfo["unsupported"] if needs.get(g)) + " for them to move)")
        trial = ground_bird_trial(plan_used, cls, P, names, bool(args.clips), False)
        if trial:
            note("A ground bird (as tall as it is long): Idle and Walk; flight clips only if they keep the body together")
        clips, missing = bake_clips(template, names, ours, skel, ref_store)
    write_rigged(out_glb, ours, obin, src, sbin, J, W, clips)
    far = far_motion(out_glb)
    if any(c in far for c in names):
        shutil.copy2(out_glb, work / "rigged_allclips.glb")
    if trial:  # B2 alone passes a correctly bound chicken whose Flap still spreads its body sides into slabs
        import rig_motion
        Jt = np.asarray(J)
        mt = rig_motion.measure(out_glb, Jt[np.arange(len(Jt)), np.asarray(W).argmax(1)],
                                [skel.nodes[i].get("name", "") for i in joints])
        trial_stretch = {n: mt["clips"][n]["stretch_p99"] for n in trial if n in mt["clips"]}
    dropped_clips = [n for n in trial if far.get(n, {}).get("level") == "fail" or trial_stretch.get(n, 0) > 3.0]
    if dropped_clips:
        clips = [c for c in clips if c["name"] not in dropped_clips]
        write_rigged(out_glb, ours, obin, src, sbin, J, W, clips)
        note("Left out " + ", ".join(dropped_clips) + " (" + "; ".join(
            f"{n}: far-bone pull {far[n]['share']:.0%} of points, p99 {far[n]['p99']:.2f}× height; 1% of edges stretch "
            f"{trial_stretch.get(n, 0):.1f}×" for n in dropped_clips) + "; kept when under 10% / 0.75× / 3×)")

    floor = None
    if args.floor != "0" and clips:
        fixing = True  # creatures: grounding + leg IK + neck lift; people: grounding only
        if fixing:
            shutil.copy2(out_glb, work / "rigged_nofloor.glb")
        t0 = time.time()
        lateral = R.from_euler("y", float(fitinfo.get("yaw", 0)), degrees=True).apply([1.0, 0, 0])
        Hs = float(np.ptp(P[:, 1]))
        floor = rig_pred.floor_pass(clips, ours, P, np.asarray(J), np.asarray(W), template, cls, Hs, lateral,
                                    plan=plan_used, prefer_track=built is not None)
        if fixing:
            write_rigged(out_glb, ours, obin, src, sbin, J, W, clips)
            far = far_motion(out_glb)
        note(("Floor pass (" if rig_pred.paths_for(cls, plan_used) else "Floor pass, grounding only (") + f"{time.time() - t0:.0f} s): " + "; ".join(
            f"{c} lowest {v['low_min_before']:+.2f}→{v['low_min_after']:+.2f}× height" for c, v in floor.items()))
    gone_clips = list(dict.fromkeys(list(unsupported_clips) + list(dropped_clips) + [f["name"] for f in clips_failed]))
    if gone_clips:
        degrade("clips_dropped", ",".join(gone_clips))
    # last: designed joint frames (bone rolls for the FBX); the skinning is unchanged, so nothing above is re-run
    reframed = None
    if (args.reframe == "1" or (args.reframe == "auto" and built is not None)) and clips:
        reframed = apply_reframe(work, plan, built is not None, posed, names_j, ours, obin, src, sbin, P, J, W, clips,
                                 out_glb, note)
    report = {
        "class": cls, "template": template_info(template, plan_used), "plan": plan_used["id"],
        "plan_resolved": plan["id"],
        "source": original.name, "rigged": out_glb.name, "rest": rest,
        "joints": len(joints), "clips": [c["name"] for c in clips], "missing_clips": missing,
        "skin_seconds": round(secs, 1), "dropped_influences": dropped, "folded_weight": round(folded, 4), "split_points": split,
        "bind": args.bind, "fbx": None, "fbx_note": "skipped", "made": int(time.time()),
        "fit": fitinfo, "far_motion": far, "dropped_clips": dropped_clips, "trial_stretch_p99": trial_stretch,
        "reassigned_points": int(local_weights.reassigned),
        "binder": binder, "pred": {k: v for k, v in pred_ctx.items() if k not in ("independent",)} | (
            {"independent": {k: v for k, v in pred_ctx["independent"].items() if k != "chains"}} if pred_ctx.get("independent") else {}),
        "unsupported_clips": unsupported_clips, "floor": floor,
        "clips_report": clips_report, "clips_failed": clips_failed, "facing_prior": facing_prior,
        "build_error": build_error, "binder_report": binder_report, "reframe": reframed, "fbx_check": None,
        "degraded": degraded,
    }
    stage("check", "Checking the rig")
    bundle = None
    try:
        import rig_inspect
        bundle = rig_inspect.build(asset, report, planner, notes, raw_J, raw_W, kept)
        for c in bundle["checks"]:
            if c["level"] != "pass":
                print(f"rig: {c['level']}: {c['label']}", flush=True)
        print("rig: Checked: " + bundle["summary"]["line"], flush=True)
    except Exception as e:  # the rig itself is done; a broken report shouldn't fail it
        print(f"rig: couldn't check the rig ({e})", flush=True)

    fbx_note = "skipped"
    name = original.stem if original.stem not in ("psx", "textured_photo", "textured") else args.stem.replace("-", "_")
    psx_meta = asset / "psx" / "psx.json"
    if psx_meta.is_file():
        name = json.loads(psx_meta.read_text()).get("name") or name
    if not args.no_fbx:
        stage("fbx", "Writing the FBX for s&box")
        fbx = work / f"{name}.fbx"
        fbx_note, fchk = export_fbx(out_glb, fbx, [c["name"] for c in clips])
        report["fbx_check"] = slim_fbx_check(fchk)
        if fchk is not None:
            print(f"rig: {'pass' if fchk.get('ok') else 'fail'}: {fbx_note}", flush=True)
        tex = asset / "psx" / f"{name}.png"
        if tex.is_file():
            shutil.copy2(tex, work / tex.name)
    report.update({"fbx": f"rig/{name}.fbx" if (work / f"{name}.fbx").is_file() else None, "fbx_note": fbx_note})
    if args.render == "1" and clips:
        report["render"] = render_sheet(asset, note)
    (work / "rig.json").write_text(json.dumps(report, indent=2, default=_plain) + "\n", encoding="utf-8")
    if bundle is not None:  # bundle["rig"] is report: now with the FBX check and the contact sheet
        try:
            rig_inspect.add_late_rows(bundle, asset, report)
        except Exception as e:  # noqa: BLE001
            print(f"rig: couldn't add the FBX check to the report ({e})", flush=True)
        rig_inspect.write(asset, bundle)
    print("RIG_RESULT " + json.dumps(report, default=_plain), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
