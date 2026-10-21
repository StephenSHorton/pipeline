#!/usr/bin/env python3
"""Write out/<stem>/rig/inspect.json: every decision rig.py made, so a rig can be checked instead of trusted.

    .venv/bin/python scripts/rig_inspect.py --stem psx-cow      # rebuild it for a take rigged before this existed

rig.py calls build() at the end of each rig. The file holds:
  planner     the body plan, how sure the planner was, and the picture it looked at
  template    which Mesh2Motion rig, the reference clip, the fit scale/offset, how far the skinner moved joints
  bones       our skeleton (name, parent, rest position) and, per bone, which template bone drives it
  ref_pose    how far the reference pose turns each bone from the template's rest
  clips       per clip and bone, the largest world-space turn from the reference pose
  weights     raw and final per-vertex top bone, weights dropped as too far away, and the corner-split check
  fit         per bone, how far the vertices it owns sit from it; the head's turn against the head bone
  motion      every clip played on the rig (scripts/rig_motion.py): tears and stretch, and when
  checks      pass / warn / fail rows for the app, each with the rig step it belongs to and a short label
  summary     one line and its numbers, also written to rig/summary.json for the versions strip
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig  # noqa: E402
import rig_motion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLANS = {"quadruped": "Four legs", "humanoid": "Person", "multileg": "Many legs", "avian": "Bird"}
STEPS = ("plan", "rest", "skeleton", "skin", "motion")
LEVELS = {"fail": 0, "warn": 1, "pass": 2}
MINOR = re.compile(r"leaf|tip|_end$", re.I)  # end-of-chain bones: nothing hangs off them worth checking
NO_FIT = re.compile(r"leaf|tip|_end$|^ear", re.I)  # ears move few points and nothing hangs off them: not worth a fit row


def plain_bone(name: str) -> str:
    """A bone name in words: back_leg_foot_r -> "right back foot", front_leg_lower_l -> "left front lower leg"."""
    words = [w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name.replace("_", " "))]
    side = [{"l": "left", "r": "right"}[w] for w in words if w in ("l", "r")]
    where = [w for w in words if w in ("front", "back")]
    rest = [{"upperarm": "upper arm", "lowerarm": "lower arm", "ball": "ball of foot"}.get(w, w) for w in words
            if w not in ("l", "r", "front", "back") and not w.isdigit()]
    if "leg" in rest:
        rest.remove("leg")
        if not [w for w in rest if w not in ("upper", "lower") and len(w) > 1]:
            rest = [w for w in rest if len(w) > 1] + ["leg"] + [w for w in rest if len(w) == 1]  # "lower leg", "leg a"
    return " ".join(side + where + rest) or name.replace("_", " ")


def summarize(info: dict, checks: list[dict], motion: dict | None, tears: int) -> dict:
    """The one line (and its numbers) the app shows for this rig wherever it shows one."""
    level = min((c["level"] for c in checks), key=LEVELS.get, default="pass")
    by = {c["key"]: c for c in checks}
    rest = info.get("rest") or {}
    measured, straight = rest.get("measured"), rest.get("straightened")
    hc = by.get("head")
    mc = (motion or {}).get("clips") or {}
    parts = []
    if by.get("seam", {}).get("level") == "fail":
        parts.append(f"tears at {tears} points")
    elif by.get("motion", {}).get("level") == "fail":
        parts.append(f"{by['motion']['clip'].replace('_', ' ')} tears at {by['motion']['points']} points")
    else:
        parts.append("no tears")
    if straight and measured:
        parts.append(f"head straightened {abs(measured['head_yaw_deg']):.0f}°")
    elif measured and measured.get("turned") and rest.get("before_straightening"):
        parts.append(f"head turned {abs(measured['head_yaw_deg']):.0f}°")
    elif measured and measured.get("turned"):
        parts.append(f"head kept turned {abs(measured['head_yaw_deg']):.0f}°")
    elif hc and hc["level"] != "pass":
        parts.append(f"head {hc['off_deg']:.0f}° off")
    worst = max(mc, key=lambda c: mc[c]["stretch_max"], default=None)
    p99 = bool(worst) and all("stretch_p99" in v for v in mc.values())
    w99 = max(mc, key=lambda c: mc[c]["stretch_p99"]) if p99 else None
    if worst and p99:
        # the single worst edge saturates on dense meshes (a fine-looking Idle read 56×); 1% of edges is the number
        parts.append(f"1% of edges stretch {mc[w99]['stretch_p99']:.1f}× ({w99.replace('_', ' ')})")
    elif worst:
        parts.append(f"worst stretch {mc[worst]['stretch_max']:.1f}× ({worst.replace('_', ' ')})")
    line = " · ".join(parts)
    problems = sorted((c for c in checks if c["level"] != "pass"), key=lambda c: (LEVELS[c["level"]], STEPS.index(c["step"])))
    return {"version": 1, "made": int(info.get("made") or time.time()), "class": info["class"], "level": level,
            "line": line[:1].upper() + line[1:], "tears": int(tears),
            "motion": {c: {"tears": v["tears"], "stretch_max": v["stretch_max"],
                           **({"stretch_p99": v["stretch_p99"]} if "stretch_p99" in v else {})} for c, v in mc.items()}
            if motion else None,
            "head": None if not measured and not hc else {
                "before": measured["head_yaw_deg"] if measured else None,
                "after": straight["head_yaw_deg"] if straight else None, "straightened": bool(straight),
                "off_deg": hc["off_deg"] if hc else None, "made_before": bool(rest.get("before_straightening"))},
            # the same figure as the line: 1% of edges (p99) when every clip has it; the single worst edge (which
            # saturates on a buried vertex: chicken-canon Run 22x while every view looks clean) is edge_max
            "worst_stretch": ({"clip": w99, "x": round(mc[w99]["stretch_p99"], 1), "metric": "p99",
                               "edge_max": {"clip": worst, "x": round(mc[worst]["stretch_max"], 1)}} if p99 else
                              {"clip": worst, "x": round(mc[worst]["stretch_max"], 1), "metric": "max"}) if worst else None,
            "problems": [{"level": c["level"], "key": c["key"], "label": c["label"]} for c in problems]}


def dense(J: np.ndarray, W: np.ndarray, n: int) -> np.ndarray:
    return rig.to_dense(J, W, n)


def corner_split(P: np.ndarray, M: np.ndarray) -> tuple[dict, np.ndarray]:
    """How much copies of one point disagree: 0 = all the same, 1 = no bone in common."""
    _, grp, cnt = np.unique(np.round(P / 1e-5).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    grp = grp.reshape(-1)
    mean = np.zeros((int(grp.max()) + 1, M.shape[1]))
    np.add.at(mean, grp, M)
    mean /= cnt[:, None]
    split = np.abs(M - mean[grp]).sum(1) / 2
    top = M.argmax(1)
    conflicting = len(np.unique(grp[top != mean[grp].argmax(1)]))
    return {"positions": int(grp.max()) + 1, "vertices": len(P), "corners_split_gt_0_1": int((split > 0.1).sum()),
            "positions_with_conflicting_top_bone": conflicting, "max_split": round(float(split.max()), 3)}, split


def head_turn(names: list[str], parent: list[int], jp: np.ndarray, P: np.ndarray, top: np.ndarray) -> dict | None:
    """Sideways turn of the head mass (as seen from above) against the head bone. Needs head + a forward tip bone."""
    if "head" not in names:
        return None
    tip = next((n for n in ("nose", "head_leaf") if n in names), None)
    hk = [k for k, n in enumerate(names) if n.startswith(("head", "nose", "mouth", "ear", "jaw"))]
    hs = np.isin(top, hk)
    if tip is None or hs.sum() < 10:
        return None
    h = names.index("head")
    neck = jp[parent[h]] if parent[h] >= 0 else jp[h]
    hc = P[hs].mean(0)
    bone = jp[names.index(tip)] - neck
    mesh = hc - neck
    if np.hypot(bone[0], bone[2]) < 0.2 * np.linalg.norm(bone):  # an upright head (people): yaw isn't defined
        return None
    yaw = lambda v: float(np.degrees(np.arctan2(v[0], v[2])))  # noqa: E731
    off = (yaw(mesh) - yaw(bone) + 180) % 360 - 180
    H = float(np.ptp(P[:, 1]))
    return {"head_verts": int(hs.sum()), "yaw_bone_deg": round(yaw(bone), 1), "yaw_mesh_deg": round(yaw(mesh), 1),
            "off_deg": round(off, 1), "lateral_offset_m": round(float(hc[0] - jp[h][0]), 3),
            "lateral_offset_rel_h": round(float(hc[0] - jp[h][0]) / H, 3)}


# ---------------------------------------------------------------- the template a rig used, and the new rows

LYING = re.compile(r"death|dead|die|sleep|lie|lay|sit|fall", re.I)  # clips that end off their feet
LOOPING = re.compile(r"idle|walk|run|trot|jog|gallop|flap|glide|fly|swim|sleep|peck|eat", re.I)  # vendor clips that loop
PART_MIN = {"neck": 0.95, "legs": 0.8, "wings": 0.5}  # acceptance 5: the part's own bones hold this share of its weight


def is_built(info: dict) -> bool:
    t = info.get("template")
    return bool(isinstance(t, dict) and t.get("built"))


def drawn_plan(asset: Path) -> str | None:
    """The plan hint a canonical pipeline draw left in intent.json, if any."""
    try:
        rig = json.loads((asset / "intent.json").read_text()).get("rig") or {}
    except (OSError, ValueError):
        return None
    return rig.get("plan_hint") if rig.get("canonical") else None


def template_of(info: dict, asset: Path) -> tuple[dict, str | None]:
    """(template, error): the built template from rig/template/template.json when rig.json says so (blocking:
    a missing file is an error, not the class's template), else rig.TEMPLATES[class]."""
    if not is_built(info):
        return rig.TEMPLATES[info["class"]], None
    import rig_template
    try:
        return rig_template.template_for(info, asset), None
    except FileNotFoundError as e:
        return rig.TEMPLATES[info["class"]], str(e)


def _plan_of(info: dict, template: dict) -> dict | None:
    pid = (info.get("template") or {}).get("plan") if isinstance(info.get("template"), dict) else None
    pid = pid or template.get("plan") or info.get("plan")
    if not pid or not isinstance(pid, str):
        return None
    try:
        import rig_plan
        return rig_plan.load(pid)
    except Exception:  # noqa: BLE001
        return None


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _roles(fin: dict, joints: list[int], names: list[str]) -> list[str]:
    try:
        import rig_render
    except Exception:  # noqa: BLE001
        rig_render = None
    out = []
    for j, nm in zip(joints, names):
        r = ((fin["nodes"][j].get("extras") or {}).get("role"))
        if not r and rig_render is not None:
            r = rig_render.role(nm)
        out.append(r or "other")
    return out


def _clip_frames(doc: dict, binary: bytes, joints: list[int], IBM: np.ndarray, fps: float = 30.0, max_frames: int = 90,
                 frames: dict | None = None):
    """Per animation: (name, times, joint world positions (F,B,3), skin mats (F,B,4,4), local rotation per joint
    (F,B,4)), sampled at the clip's own rate (capped at max_frames), or at exactly frames[name] keys."""
    nodes = doc["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    out = []
    for anim in doc.get("animations", []):
        dur, ch = rig_motion.channels(doc, binary, anim)
        n = int((frames or {}).get(anim.get("name")) or np.clip(round(dur * fps) + 1, 2, max_frames))
        times = np.linspace(0, dur, n)
        jp, M, Q = [], [], []
        for t in times:
            local = rig_motion.sample(nodes, ch, float(t))
            wm = rig_motion.world_mats(nodes, parent, local)
            jp.append([wm[j][:3, 3] for j in joints])
            M.append([wm[j] @ IBM[k] for k, j in enumerate(joints)])
            Q.append([np.asarray(local.get(j, rig_motion._rest(nodes[j]))[1], float) for j in joints])
        out.append((anim.get("name") or f"clip {len(out) + 1}", times, np.array(jp), np.array(M), np.array(Q)))
    return out


def seam_excess_deg(Q: np.ndarray) -> float:
    """A loop's seam for a clip authored with the wrap implied (vendor clips: the last key is one step before the
    first, not a copy of it): per joint, the last->first turn minus the larger of its neighbouring frame steps
    (first->second, second-last->last), so one ordinary frame step is not read as a pop. Q (F, J, 4) xyzw."""
    import rig_metrics
    Q = np.asarray(Q, float)
    if Q.ndim != 3 or len(Q) < 3:
        return 0.0
    worst = 0.0
    for j in range(Q.shape[1]):
        q = Q[:, j]
        seam = float(rig_metrics._qangle_deg(q[-1], q[0]))
        step = max(float(rig_metrics._qangle_deg(q[0], q[1])), float(rig_metrics._qangle_deg(q[-2], q[-1])))
        worst = max(worst, seam - step)
    return round(max(worst, 0.0), 4)


def motion_metrics(fin: dict, fbin: bytes, P: np.ndarray, Jf: np.ndarray, Wf: np.ndarray, names: list[str],
                   cls: str, plan: dict | None, H: float, contacts: dict | None = None, built: bool = True) -> dict:
    """Per clip on the final rig: C-shape (bounding-box growth vs rest), stance foot slide, loop seam. contacts
    ({clip: {Foot_L: [0/1 per frame], ...}}, a built rig's own, rig.json clips_report) replace the guessed stance
    frames (each foot within 0.015H of its lowest)."""
    import rig_metrics
    import rig_pred
    joints = fin["skins"][0]["joints"]
    IBM = rig.accessor(fin, fbin, fin["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    paths = rig_pred.paths_for(cls, plan) if cls in rig_pred.PATHS or plan else None
    if paths and paths.get("legs"):
        feet = [c[-1] for c in paths["legs"]]
    else:
        import rig_fit
        feet = [c[2] for c in (rig_fit.CHAINS.get(cls) or {}).get("legs") or [] if len(c) > 2]
    fk = [names.index(f) for f in feet if f in names]
    loops = {c["name"]: bool(c.get("loop", True)) for c in (plan or {}).get("clips") or []}
    sub = np.arange(len(P)) if len(P) <= 12000 else np.unique(np.r_[np.arange(0, len(P), len(P) // 8000), np.argsort(P[:, 1])[:500]])
    Ph = np.c_[P[sub], np.ones(len(sub))]
    V0 = P[sub]
    out = {}
    contacts = contacts or {}
    nf = {n: len(next(iter(c.values()))) for n, c in contacts.items() if c}
    for name, times, jp, M, Q in _clip_frames(fin, fbin, joints, IBM, frames=nf):
        if name == "Rest":
            continue
        step = max(1, len(times) // 24)
        Vs = np.array([rig_motion.skin(Ph, Jf[sub], Wf[sub], M[f]) for f in range(0, len(times), step)])
        cs = rig_metrics.c_shape(Vs, V0)
        lying = bool(LYING.search(name))
        loop = loops.get(name, bool(LOOPING.search(name)) and not lying)
        rec = {"cshape": cs["ext_ratio"], "cshape_max": cs["ext_ratio_max"], "lying": lying, "loop": loop}
        if loop:
            raw = round(float(rig_metrics.loop_seam({"rot": {k: Q[:, k] for k in range(Q.shape[1])}})), 3)
            # built clips repeat frame 0 as the last frame (the contract): any turn there is a seam. Vendor clips
            # imply the wrap, so only a turn bigger than a frame step is one.
            rec["seam_deg"] = raw if built else round(seam_excess_deg(Q), 3)
            if not built:
                rec["seam_raw_deg"] = raw
        if fk and not lying and len(times) > 2:
            fps = (len(times) - 1) / max(float(times[-1]), 1e-9)
            fp = jp[:, fk]  # (F, legs, 3)
            lo = fp[:, :, 1].min(0)
            contact = fp[:, :, 1] <= lo[None] + 0.015 * H
            own = contacts.get(name) or {}
            if own and all(f in own and len(own[f]) == len(times) for f in feet if f in names):
                contact = np.stack([np.asarray(own[f], bool) for f in feet if f in names], 1)
                rec["contacts"] = "clip"
            vel = np.gradient(fp, axis=0) * fps
            st = vel[contact]
            v_exp = np.median(st, 0) * np.array([1.0, 0.0, 1.0]) if len(st) else np.zeros(3)
            slides = [rig_metrics.foot_slide(fp[:, i], contact[:, i], v_exp, fps=fps, H=H)["slide"] for i in range(len(fk))]
            rec["slide"] = round(float(max(slides)), 4)
            rec["v_expected"] = np.round(v_exp, 4).tolist()
        out[name] = rec
    return out


def _part_chains(cls: str, plan: dict | None, names: list[str]) -> dict:
    import rig_fit
    chains = dict(rig_fit.CHAINS.get(cls) or {})
    return {k: v for k, v in chains.items() if v}


def new_rows(check, asset: Path, info: dict, template: dict, terr: str | None, built: bool, plan: dict | None,
             fin: dict, fbin: bytes, P: np.ndarray, F: np.ndarray, Jf: np.ndarray, Wf: np.ndarray, jp: np.ndarray,
             parent: list[int], names: list[str], H: float) -> dict:
    """The template, built, a2, foot, parts, cshape, slide and loop_seam rows (design section 9). Returns the
    measurements for inspect.json."""
    import rig_metrics
    cls = info["class"]
    extra: dict = {}
    G = (plan or {}).get("gates") or {}
    strict = "fail" if built else "warn"  # vendor clips are reported, never failed, on these new rows
    # C-shape and slide on vendor clips are measured only: a vendor gait's stride widens a thin body's box (operator
    # Walk 2.6x in z) and its stance frames are guessed, so they pass with their numbers instead of warning
    measured = "fail" if built else "pass"
    # template
    if terr:
        check("fail", "template", "skeleton", "Built template missing", terr + ".")
    elif built:
        v = _json(asset / "rig" / "template" / "validate.json") or {}
        errs = [p for p in v.get("problems") or [] if p.get("level") == "error"]
        warns = [p for p in v.get("problems") or [] if p.get("level") == "warn"]
        if not v:
            check("warn", "template", "skeleton", "Template not verified", "rig/template/validate.json is missing.")
        elif errs:
            check("fail", "template", "skeleton", f"Template breaks the contract ({len(errs)})",
                  "; ".join(p.get("msg", "") for p in errs) + ".", problems=errs)
        else:
            check("pass", "template", "skeleton", f"Skeleton built for this model ({template.get('plan')})",
                  f"A {template.get('plan')} skeleton and its clips were made for this model and pass the template "
                  "checks" + (" (" + "; ".join(p.get("msg", "") for p in warns) + ")" if warns else "") + ".",
                  plan=template.get("plan"), warnings=warns)
    elif info.get("build_error"):
        be = info["build_error"]
        check("warn", "template", "skeleton", "Couldn't build a skeleton",
              f"Building a skeleton for this model failed ({be.get('code')}: {be.get('msg')}), so the class's template "
              "was used.", **be)
    # built: the skeleton's own self-check (build.json)
    if built:
        b = _json(asset / "rig" / "template" / "build.json") or (info.get("fit") or {}).get("build") or {}
        ck = b.get("checks") or {}
        cd = ((b.get("cd_j2b") or {}).get("pred_to_built_H") or {}).get("mean")
        # the gates read the labelled chains (cd) and the floor points (feet); the all-joints and centroid figures
        # of design 12.5 as first written are shown beside them (their redefinition awaits the design owner)
        cd_all = ((b.get("cd_j2b") or {}).get("pred_all_to_built_H") or {}).get("mean")
        feet = {s: (f or {}).get("to_points_H") for s, f in (ck.get("feet") or {}).items()}
        feet_c = {s: (f or {}).get("to_centroid_H") for s, f in (ck.get("feet") or {}).items()}
        bad = []
        if cd is not None and cd > 0.03:
            bad.append(f"predicted joints {cd:.3f}× height from its bones")
        if ck.get("outside"):
            bad.append(f"{len(ck['outside'])} joints outside the body ({', '.join(map(str, ck['outside'][:4]))})")
        if (ck.get("mirror_err_H") or 0) > 0.01:
            bad.append(f"left/right differ by {ck['mirror_err_H']:.3f}× height")
        if (ck.get("gate") or {}).get("level") == "fail":
            bad.append("the pose check failed")
        far_feet = [s for s, d in feet.items() if d is not None and d > 0.02]
        if far_feet:
            bad.append("feet off the floor clusters (" + ", ".join(f"{s} {feet[s]:.3f}" for s in far_feet) + ")")
        if ck.get("problems"):
            bad += [str(p) for p in ck["problems"]]
        check("fail" if bad else "pass", "built", "skeleton", "Built skeleton off the model" if bad else "Built skeleton fits",
              ("; ".join(bad) + ".") if bad else
              f"Every joint is inside the body (tips and soles excepted), the predicted joints sit "
              f"{(cd or 0):.3f}× height from its bones"
              + (f" (the labelled chains; all {cd_all:.3f}×)" if cd_all is not None else "")
              + f", left and right match to {ck.get('mirror_err_H', 0):.3f}× height"
              + ("; feet to their floor points " + ", ".join(f"{k} {v:.3f}" for k, v in feet.items() if v is not None)
                 + " (to the cluster centres " + ", ".join(f"{k} {v:.3f}" for k, v in feet_c.items() if v is not None) + ")"
                 if any(v is not None for v in feet.values()) else "") + ".",
              cd_j2b=cd, cd_j2b_all=cd_all, mirror_err_H=ck.get("mirror_err_H"), feet_to_points_H=feet,
              feet_to_centroid_H=feet_c, outside=ck.get("outside") or [],
              contact_outside=ck.get("contact_outside") or [])
    segs = rig_metrics.segs_from_parent(parent)
    D = rig_metrics.seg_dist(P, jp, segs) if segs else None
    if D is not None:
        a2 = rig_metrics.a2_far_weight(P, Jf, Wf, jp, segs, H, names, D=D)
        extra["a2"] = a2
        lvl = "pass" if a2["share"] < 0.02 else "warn"
        check(lvl, "a2", "skin", f"{a2['share']:.1%} of the weight on far bones",
              f"{a2['share']:.1%} of all skin weight sits on bones more than 0.15× height farther from their point than "
              "its nearest bone" + (" (most: " + ", ".join(f"{plain_bone(str(n))} {s:.1%}" for n, s in a2["worst"][:3]) + ")"
                                    if a2["worst"] and lvl != "pass" else "") + " (warn above 2%).",
              share=a2["share"], worst=a2["worst"])
        chains = _part_chains(cls, plan, names)
        if chains.get("legs"):
            ft = rig_metrics.foot_off_chain(P, Jf, Wf, names, chains["legs"])
            extra["foot"] = ft
            lvl = "pass" if ft["share"] <= 0.05 else "warn"
            check(lvl, "foot", "skin", "Feet on leg bones" if lvl == "pass" else f"{ft['share']:.0%} of the soles off the legs",
                  f"{ft['share']:.0%} of the points near the floor follow a bone that isn't on a leg"
                  + (" (" + ", ".join(f"{plain_bone(str(n))} {c}" for n, c in ft["owners"][:3]) + ")" if ft["owners"] else "")
                  + " (warn above 5%).", share=ft["share"], owners=ft["owners"])
        if chains:
            try:
                masks = None
                if built and chains.get("wings") and ((plan or {}).get("caps") or {}).get("wings") == "folded":
                    # the built skin takes the wing weight off the rump/vent points under the feathers
                    # (rig_build.wing_region_repair's box), so the wings' part is the points on the body sides
                    import rig_build
                    cx = float(info["cx"]) if info.get("cx") is not None else float(np.median(P[:, 0]))
                    masks = {"wings": rig_build._wing_side_margin(P, names, np.asarray(jp, float), cx, H) > 0}
                ps = rig_metrics.part_shares(P, Jf, Wf, chains, jp, segs, names, D=D, masks=masks)
            except Exception as e:  # noqa: BLE001
                ps = {}
                print(f"rig_inspect: couldn't measure the part shares ({e})", file=sys.stderr)
            extra["parts"] = ps
            low = {k: v["share"] for k, v in ps.items() if k in PART_MIN and v.get("n") and v["share"] < PART_MIN[k]}
            if ps:
                check("warn" if low else "pass", "parts", "skin",
                      ("Parts leak weight: " + ", ".join(f"{k} {v:.0%}" for k, v in low.items())) if low else "Parts follow their own bones",
                      "Per part, the share of its points' weight on its own bones: " + ", ".join(
                          f"{k} {v['share']:.0%}" for k, v in ps.items() if v.get("n")) + " (wanted: neck 95%, legs 80%, wings 50%).",
                      shares={k: v["share"] for k, v in ps.items()})
    try:
        mm = motion_metrics(fin, fbin, P, Jf, Wf, names, cls, plan, H,
                            contacts=(info.get("clips_report") or {}).get("contacts") if built else None, built=built)
    except Exception as e:  # noqa: BLE001
        print(f"rig_inspect: couldn't measure the clip shapes ({e})", file=sys.stderr)
        mm = {}
    extra["clips"] = mm
    if mm:
        lim, flap_lat = float(G.get("cshape", 1.6)), float(G.get("cshape_flap_lateral", 2.2))

        def over(n: str, r: dict) -> float:
            e = r["cshape"]
            lims = [flap_lat if re.search(r"flap|glide|fly", n, re.I) else lim, lim, lim]
            return max(e[k] / lims[k] for k in range(3))
        upright = [n for n, r in mm.items() if not r["lying"]]
        bad = [n for n in upright if over(n, mm[n]) > 1.0]
        if upright:
            w = max(upright, key=lambda n: over(n, mm[n]))
            check(measured if bad else "pass", "cshape", "motion",
                  (f"{w.replace('_', ' ')} spreads the body {mm[w]['cshape_max']:.1f}×" if built else
                   f"Library clips: {w.replace('_', ' ')} spreads the box {mm[w]['cshape_max']:.1f}× (measured only)")
                  if bad else "Body keeps its shape",
                  "Largest growth of the body's box against rest, per clip (x, y, z): " + "; ".join(
                      f"{n.replace('_', ' ')} " + " / ".join(f"{x:.2f}" for x in mm[n]["cshape"]) for n in upright)
                  + f" (limit {lim}×, {flap_lat}× sideways for wing clips; lying clips are left out).",
                  clips={n: mm[n]["cshape"] for n in upright}, over=bad)
        sl = {n: r["slide"] for n, r in mm.items() if r.get("slide") is not None}
        if sl:
            lim_s = float(G.get("foot_slide", 0.03))
            w = max(sl, key=sl.get)
            badl = [n for n, v in sl.items() if v > lim_s]
            check(measured if badl else "pass", "slide", "motion",
                  (f"{w.replace('_', ' ')}: feet slide {sl[w]:.3f}× height" if built else
                   f"Library clips: {w.replace('_', ' ')} feet move {sl[w]:.3f}× height on guessed contacts (measured only)")
                  if badl else "Planted feet stay put",
                  "Largest slide of a planted foot, per clip: " + ", ".join(f"{n.replace('_', ' ')} {v:.3f}" for n, v in sl.items())
                  + f" (× height; limit {lim_s}).", clips=sl, over=badl)
        sm = {n: r["seam_deg"] for n, r in mm.items() if r.get("seam_deg") is not None}
        if sm:
            lim_d = float(G.get("loop_seam_deg", 0.5))
            w = max(sm, key=sm.get)
            bads = [n for n, v in sm.items() if v > lim_d]
            check("warn" if bads else "pass", "loop_seam", "motion",
                  f"{w.replace('_', ' ')} jumps {sm[w]:.1f}° where it loops" if bads else "Loops join up",
                  ("Largest joint turn between a looping clip's last and first frame: " if built else
                   "Largest joint turn where a library clip wraps, beyond one frame step: ") + ", ".join(
                      f"{n.replace('_', ' ')} {v:.2f}°" for n, v in sm.items()) + f" (warn above {lim_d}°).", clips=sm, over=bads)
    return extra


def fbx_row(check, info: dict, asset: Path) -> None:
    fc = info.get("fbx_check")
    if fc is None and info.get("fbx"):
        fc = _json(asset / "rig" / "fbx_check.json")
    if not fc:
        return
    try:
        import rig_fbx
        line = rig_fbx.note_for(fc) if "actions" in fc and isinstance(fc["actions"], list) else None
    except Exception:  # noqa: BLE001
        line = None
    acts = fc["actions"] if isinstance(fc.get("actions"), int) else len(fc.get("actions") or [])
    bones = fc["bones"] if isinstance(fc.get("bones"), int) else len(fc.get("bone_names") or [])
    if not fc.get("ok"):
        check("fail", "fbx", "motion", "FBX check failed", "; ".join(fc.get("problems") or ["no report"]) + ".",
              problems=fc.get("problems") or [], warnings=fc.get("warnings") or [])
        return
    lvl = "warn" if fc.get("warnings") else "pass"
    check(lvl, "fbx", "motion", f"FBX checked: {acts} clips, {fc.get('scene_fps', 0):g} fps",
          (line or f"FBX checked: {acts} clips, {bones} bones, {fc.get('max_influences', 0)} weights max, "
           f"{fc.get('scene_fps', 0):g} fps") + "." + (" " + "; ".join(fc["warnings"]) + "." if fc.get("warnings") else ""),
          actions=acts, bones=bones, max_influences=fc.get("max_influences"), scene_fps=fc.get("scene_fps"),
          warnings=fc.get("warnings") or [])


def add_late_rows(bundle: dict, asset: Path, info: dict) -> None:
    """Rows from steps that run after the check (the FBX export): add them and re-make the summary."""
    checks = [c for c in bundle["checks"] if c["key"] != "fbx"]

    def check(level: str, key: str, step: str, label: str, text: str, **extra) -> None:
        checks.append({"level": level, "key": key, "step": step, "label": label, "text": text, **extra})
    fbx_row(check, info, asset)
    bundle["checks"] = checks
    bundle["rig"] = info
    tears = (bundle.get("weights") or {}).get("seam_final", {}).get("positions_with_conflicting_top_bone", 0)
    bundle["summary"] = summarize(info, checks, bundle.get("motion"), tears)


def build(asset: Path, info: dict, planner: dict, notes: list[str], raw_J: np.ndarray, raw_W: np.ndarray,
          kept: np.ndarray | None = None) -> dict:
    """raw_J/raw_W: the skinner's weights per source vertex; kept: the far-influence mask on them (None: unknown)."""
    template, terr = template_of(info, asset)
    built = is_built(info) and terr is None
    bplan = _plan_of(info, template) if built else None
    fin, fbin = rig.read_glb(asset / info["rigged"])
    pos, _ = rig.read_glb(asset / "rig" / "template_posed.glb")
    sk = rig.Skeleton(fin)
    joints = fin["skins"][0]["joints"]
    names = [fin["nodes"][j].get("name", "") for j in joints]
    w = sk.world({})
    jp = np.array([w[j][0] for j in joints])
    parent = [joints.index(sk.parent[j]) if sk.parent.get(j) in joints else -1 for j in joints]
    psk = rig.Skeleton(pos)
    pw = psk.world({})
    tpos = {pos["nodes"][i].get("name"): pw[i][0] for i in pw if pos["nodes"][i].get("name")}
    moved = max((float(np.linalg.norm(jp[k] - tpos[n])) for k, n in enumerate(names) if n in tpos), default=0.0)
    fitnode = next((n for n in pos["nodes"] if n.get("name") == "fit"), {})

    pf = fin["meshes"][0]["primitives"][0]
    P = rig.accessor(fin, fbin, pf["attributes"]["POSITION"]).astype(float)
    F = rig.accessor(fin, fbin, pf["indices"]).reshape(-1, 3).astype(int) if "indices" in pf else np.arange(len(P)).reshape(-1, 3)
    Jf = rig.accessor(fin, fbin, pf["attributes"]["JOINTS_0"]).astype(int)
    Wf = rig.accessor(fin, fbin, pf["attributes"]["WEIGHTS_0"]).astype(float)
    n = len(names)
    Mf, Mr = dense(Jf, Wf, n), dense(raw_J, raw_W, n)
    top_f, top_r = Mf.argmax(1), Mr.argmax(1)
    seam_final, _ = corner_split(P, Mf)
    seam_raw, split_raw = corner_split(P, Mr)
    H = float(np.ptp(P[:, 1]))

    dropped_by_bone: dict[str, int] = {}
    dropped_list = []
    if kept is not None:
        for v, k in zip(*np.nonzero((raw_W > 0) & ~kept)):
            b = names[raw_J[v, k]]
            dropped_by_bone[b] = dropped_by_bone.get(b, 0) + 1
            dropped_list.append([int(v), int(raw_J[v, k]), round(float(raw_W[v, k]), 3)])

    tdoc, _ = rig.read_glb(rig.M2M / template["rig"])
    tsk = rig.Skeleton(tdoc)
    ref_w = tsk.world(rig.ref_local(template, tsk))
    rest_w = tsk.world({})
    mapping, ref_pose = [], []
    for k, nm in enumerate(names):
        ti = tsk.index.get(nm)
        mapping.append({"ours": nm, "template": nm if ti is not None else None,
                        "rule": "same name" if ti is not None else "no template bone: holds its rest pose"})
        ref_pose.append(round(float(np.degrees((ref_w[ti][1] * rest_w[ti][1].inv()).magnitude())), 1) if ti is not None else None)
    sources = [rig.read_glb(rig.M2M / f) for f in template["clips"]]
    anims = {a["name"]: (d, b, a) for d, b in sources for a in d.get("animations", [])}
    clips = {}
    for cn in info["clips"]:
        if cn not in anims:
            continue
        d, b, a = anims[cn]
        ch = rig.prepare_clip(d, b, a, tsk)
        dur = rig.clip_duration(d, b, a)
        peak = np.zeros(n)
        for t in np.linspace(0, dur, 24):
            tw = tsk.world(rig.sample_clip(ch, tsk, float(t)))
            for k, nm in enumerate(names):
                ti = tsk.index.get(nm)
                if ti is not None:
                    peak[k] = max(peak[k], np.degrees((tw[ti][1] * ref_w[ti][1].inv()).magnitude()))
        clips[cn] = {"duration": round(dur, 2), "peak_deg": [round(float(x), 1) for x in peak]}

    child = {}
    for c in range(n):
        if parent[c] >= 0 and parent[c] not in child:
            child[parent[c]] = c

    def segd(p: np.ndarray, k: int) -> np.ndarray:
        a = jp[k]
        ab = (jp[child[k]] if k in child else a) - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1) if ab @ ab > 1e-12 else np.zeros(len(p))
        return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)

    fit = []
    for k, nm in enumerate(names):
        sel = top_f == k
        if sel.sum() < 3:
            continue
        c = P[sel].mean(0)
        fit.append({"bone": nm, "verts": int(sel.sum()), "median_dist_rel_h": round(float(np.median(segd(P[sel], k)) / H), 3),
                    "centroid_off_bone_rel_h": round(float(segd(c[None], k)[0] / H), 3)})
    head = head_turn(names, parent, jp, P, top_f)

    checks: list[dict] = []
    def check(level: str, key: str, step: str, label: str, text: str, **extra) -> None:
        checks.append({"level": level, "key": key, "step": step, "label": label, "text": text, **extra})
    plan = PLANS.get(info["class"], info["class"])
    conf = planner.get("confidence")
    if conf is None and planner.get("unrecorded"):
        check("pass", "planner", "plan", plan, f"Rigged as {plan.lower()}. This version didn't record how that was picked.",
              by_hand=None)
    elif conf is None and drawn_plan(asset):
        check("pass", "planner", "plan", f"{plan}, from the drawing",
              f"The pipeline drew this {plan.lower()} in the standard pose, so it knew the body plan: "
              f"{drawn_plan(asset).replace('_', ' ')}.", by_hand=False)
    elif conf is None:
        check("pass", "planner", "plan", f"{plan}, set by hand", f"You set the body plan by hand: {plan.lower()}.", by_hand=True)
    elif conf < 0.6:
        check("warn", "planner", "plan", f"Unsure of body plan ({conf:.0%})",
              f"The picture looked most like {PLANS.get(planner['choice'], planner['choice']).lower()}, but only {conf:.0%} sure.")
    else:
        check("pass", "planner", "plan", f"{plan} ({conf:.0%} sure)", f"The picture reads as {plan.lower()}, {conf:.0%} sure.",
              by_hand=False)
    rest = info.get("rest") or {}
    measured, straight = rest.get("measured"), rest.get("straightened")
    if measured:
        before = measured["head_yaw_deg"]
        deg = f"{abs(before):.0f}°"
        if straight:
            check("pass", "rest", "rest", f"Head straightened {deg}",
                  f"The head was turned {deg} from the body, so the rig uses a straightened copy.",
                  before=before, after=straight["head_yaw_deg"], straightened=True)
        elif measured.get("turned") and rest.get("before_straightening"):
            check("warn", "rest", "rest", f"Head turned {deg} when made",
                  f"The head is turned {deg} from the body. This version was rigged before turned heads were straightened.",
                  before=before, after=None, straightened=False, before_straightening=True)
        elif measured.get("turned"):
            check("warn", "rest", "rest", f"Head kept turned {deg}", f"The head is turned {deg} from the body and was rigged as made.",
                  before=before, after=None, straightened=False)
        else:
            check("pass", "rest", "rest", f"Head straight ({deg})", "The head lines up with the body.",
                  before=before, after=None, straightened=False)
    if head is not None:
        off = abs(head["off_deg"])  # a visually straight cow still reads about 6°: ears and horns aren't symmetric
        level = "fail" if off > 15 else "warn" if off > 8 else "pass"
        check(level, "head", "skeleton", f"Head {off:.0f}° off its bone" if level != "pass" else "Head on its bone",
              f"The head points {off:.0f}° away from its bone." if level != "pass"
              else f"The head lines up with its bone ({off:.0f}°).", bone="head", off_deg=round(off, 1))
    torso = ("root", "hips", "spine", "chest", "ribcage", "pelvis", "neck")  # these run along the back, away from the belly
    worst = max((f for f in fit if not f["bone"].startswith(torso) and not NO_FIT.search(f["bone"])),
                key=lambda f: f["centroid_off_bone_rel_h"], default=None)
    outliers = sorted((f for f in fit if not f["bone"].startswith(torso) and not NO_FIT.search(f["bone"])
                       and f["centroid_off_bone_rel_h"] > 0.12), key=lambda f: -f["centroid_off_bone_rel_h"])
    if worst and worst["centroid_off_bone_rel_h"] > 0.12:
        nm = plain_bone(worst["bone"])
        off_m = round(worst["centroid_off_bone_rel_h"] * H, 2)
        more = f" (+{len(outliers) - 1} more)" if len(outliers) > 1 else ""
        check("warn", "fit", "skeleton", f"{nm[:1].upper() + nm[1:]} {off_m:.2f} m off{more}",
              f"The {nm} bone sits {off_m:.2f} m from the part it moves." + (
                  " Also: " + ", ".join(f"{plain_bone(f['bone'])} {f['centroid_off_bone_rel_h'] * H:.2f} m ({f['verts']} points)"
                                        for f in outliers[1:]) + "." if len(outliers) > 1 else ""),
              bone=worst["bone"], off_m=off_m, off_rel_h=worst["centroid_off_bone_rel_h"],
              all=[{"bone": f["bone"], "off_rel_h": f["centroid_off_bone_rel_h"], "verts": f["verts"]} for f in outliers])
    fi = info.get("fit") or {}
    fc = fi.get("facing")
    if fc and built:
        check("pass", "facing", "skeleton", "Built facing the model's head",
              "The skeleton was built in the model's own frame, head forward (+Z), so nothing was turned "
              f"(facing {fc.get('decided_by')}).", yaw=fc.get("yaw", 0), decided_by=fc.get("decided_by"))
    elif fc:
        lvl = "warn" if fc.get("cue_conflict") else "pass"
        check(lvl, "facing", "skeleton", f"Template turned {fc['yaw']}°",
              f"The template was turned {fc['yaw']}° to face the way the model does (decided by {fc['decided_by']}, "
              f"registration margin {fc.get('margin', 0):.2f})." + (" The cues disagreed." if fc.get("cue_conflict") else "")
              + (f" {fc['prior_conflict']}; re-make the low-poly model if the head is at the back." if fc.get("prior_conflict") else ""),
              yaw=fc["yaw"], decided_by=fc["decided_by"])
    if fi.get("poor_match") and not built:
        check("warn", "match", "skeleton", f"Poor body-plan match ({fi['scale_ratio']:.1f}×)",
              f"The template had to be stretched {fi['scale_ratio']:.1f}× more along one axis than another to fit this model.",
              ratio=fi["scale_ratio"])
    gate = fi.get("gate")
    flags = None
    try:
        import rig_metrics
        import trimesh
        surf = np.asarray(trimesh.Trimesh(P, F, process=False).sample(20000, seed=0), float)
        flags = rig_metrics.pose_flags(jp, rig_metrics.segs_from_parent(parent), P, F, surf, H, names)
    except Exception as e:  # noqa: BLE001 - a missing measurement never fails the check
        print(f"rig_inspect: couldn't flag the joints ({e})", file=sys.stderr)
    if gate:
        bad = gate["level"] == "fail"
        flagged = [{"joint": r["joint"], "flags": r["flags"], "d_rel": r["d_rel"]} for r in (flags or {}).get("flagged", [])]
        check("fail" if bad else "pass", "pose", "skeleton",
              f"Skeleton misses the body ({gate['joints_outside_box']:.0%} of joints outside)" if bad else "Skeleton inside the body",
              f"{gate['joints_outside_box']:.0%} of the joints are outside the model's box and {gate['bone_far_outside_0p08H']:.0%} "
              "of the bones lie well outside the model." + (
                  " Outside: " + ", ".join(f"{plain_bone(str(r['joint']))} ({'/'.join(r['flags'])})" for r in flagged[:6]) + "."
                  if flagged else ""), **{k: v for k, v in gate.items() if k != "level"}, flagged=flagged)
    far = {c: v for c, v in (info.get("far_motion") or {}).items() if c in info.get("clips", [])}
    if far:
        badc = [c for c, v in far.items() if v["level"] == "fail"]
        if badc:
            w = max(badc, key=lambda c: far[c]["share"])
            check("fail", "far", "motion", f"{w.replace('_', ' ')}: {far[w]['share']:.0%} of points pulled by far bones",
                  f"In {', '.join(c.replace('_', ' ') for c in badc)}, points move with bones far from them "
                  f"(worst {w}: {far[w]['share']:.0%} of points, p99 {far[w]['p99']:.2f}× height).", clips=badc)
        else:
            check("pass", "far", "motion", "Points follow nearby bones", "In every clip, points move with the bones near them.")
    if moved > 0.01 * H:
        check("warn", "moved", "skeleton", f"Joints moved {moved:.2f} m",
              f"The skinning step moved some joints up to {moved:.2f} m from where they were fitted.", m=round(moved, 3))
    pts = seam_final["positions_with_conflicting_top_bone"]
    if pts:
        check("fail", "seam", "skin", f"Tears at {pts} points",
              f"{pts} points have corners on different bones, so the model tears there when it moves.", points=pts)
    else:
        check("pass", "seam", "skin", "Corners move together", "Every copy of a point follows the same bones.", points=0)
    if info.get("folded_weight", 0) > 0.2:
        check("warn", "folded", "skin", f"Weight on tip bones ({info['folded_weight']:.0%})",
              f"{info['folded_weight']:.0%} of the weight was on root or tip bones and was moved to their parents.",
              share=info["folded_weight"])
    failed = info.get("clips_failed") or []
    if failed:
        nm_ = [f["name"] for f in failed]
        check("fail", "clips", "motion", f"Couldn't make {', '.join(nm_)}",
              "These clips of the body plan didn't keep the body together on this model and are not shipped: " + "; ".join(
                  f"{f['name']}: {', '.join(f.get('reasons') or [])}" for f in failed) + ".", failed=failed,
              missing=info.get("missing_clips") or [])
    elif info.get("missing_clips"):
        miss = info["missing_clips"]
        check("warn", "clips", "motion", f"{len(miss)} clip{'s' if len(miss) != 1 else ''} missing",
              "No clip for this body plan: " + ", ".join(miss) + ".", missing=miss)
    elif built:
        check("pass", "clips", "motion", f"{len(info.get('clips') or [])} clips made for this model",
              "Every clip of the body plan was made for this model and passed its checks: " + ", ".join(info.get("clips") or []) + ".")
    try:
        motion = rig_motion.measure(asset / info["rigged"], top_f, names)
    except Exception as e:  # a failed measurement never fails the rig
        print(f"rig_inspect: couldn't measure the clips ({e})", file=sys.stderr)
        motion = None
    if motion is not None and motion["clips"]:
        mc = motion["clips"]
        tc = max(mc, key=lambda c: mc[c]["tears"])
        clip = tc.replace("_", " ")
        if mc[tc]["tears"]:
            check("fail", "motion", "motion", f"{clip}: {mc[tc]['tears']} points tear",
                  f"At {mc[tc]['tear_t']:.1f} s in {clip}, {mc[tc]['tears']} points come apart, the most of any clip.",
                  clip=tc, t=mc[tc]["tear_t"], points=mc[tc]["tears"])
        else:
            check("pass", "motion", "motion", f"No tears in {len(mc)} clips",
                  f"All {len(mc)} clips were played on this rig and no corners came apart.", clip=None, t=None, points=0)
    if motion is not None and motion["clips"]:
        mc = motion["clips"]
        sc_ = max(mc, key=lambda c: mc[c]["stretch_p99"])
        if mc[sc_]["stretch_p99"] > 3:  # a destroyed rig (the chicken's Flap on the backwards template) read 15×
            own = mc[sc_].get("stretch_owners") or []
            check("fail" if mc[sc_]["stretch_p99"] > 6 else "warn", "stretch", "motion",
                  f"{sc_.replace('_', ' ')} stretches {mc[sc_]['stretch_p99']:.1f}×",
                  f"In {sc_.replace('_', ' ')}, 1% of the edges stretch (or squash) more than {mc[sc_]['stretch_p99']:.1f}× "
                  "(warn above 3×, fail above 6×)." + (" Most of them are on " + ", ".join(
                      plain_bone(str(n)) for n, _ in own[:3]) + "." if own else ""), clip=sc_, owners=own)
    floor = {c: v for c, v in (info.get("floor") or {}).items() if c in info.get("clips", [])}
    if floor:
        pen = {c: max(0.0, -v["low_min_after"]) for c, v in floor.items()}
        flt = {c: max(0.0, v["float_max_after"] or 0.0) for c, v in floor.items()}
        cp, cf = max(pen, key=pen.get), max(flt, key=flt.get)
        worst_v = max(pen[cp], flt[cf])
        lvl = "fail" if worst_v > 0.10 else "warn" if worst_v > 0.03 else "pass"
        if lvl == "pass":
            lab = f"On the floor in {len(floor)} clips"
        elif pen[cp] >= flt[cf]:
            lab = f"{cp.replace('_', ' ')} goes {pen[cp]:.2f}× height through the floor"
        else:
            lab = f"{cf.replace('_', ' ')} floats {flt[cf]:.2f}× height"
        check(lvl, "floor", "motion", lab,
              "Per clip, the lowest point against the floor and how high the body floats while the template stands or "
              "lies on its floor: " + "; ".join(f"{c.replace('_', ' ')} {-pen[c]:+.2f} / {flt[c]:+.2f}" for c in floor)
              + " (× height; warn above 0.03, fail above 0.10).", clips={c: [round(-pen[c], 3), round(flt[c], 3)] for c in floor})
    pr = info.get("pred") or {}
    if pr.get("cd_j2b"):
        agree = pr.get("yaw_agrees", True)
        check("pass" if agree else "warn", "pred", "skeleton",
              "Laid on the model's own skeleton" if agree else "Predicted skeleton faces the other way",
              f"A skeleton predicted for this model ({pr.get('joints')} joints) was the fitting target: its joints sit "
              f"{pr['cd_j2b']['pred_to_template']:.3f}× height from the template's bones. Its head chain points "
              f"{pr.get('independent_head_chain_yaw')}° on its own cues ({(pr.get('independent') or {}).get('how')}); "
              f"the template was turned {pr.get('template_yaw')}°.", cd_j2b=pr["cd_j2b"])
    if info.get("unsupported_clips"):
        u = info["unsupported_clips"]
        check("warn", "unsupported", "motion", f"{', '.join(u)} left out",
              f"This model has no {', '.join((info.get('fit') or {}).get('unsupported') or [])} for "
              f"{', '.join(u)} to move, so those clips are not shipped.", clips=u)
    measures = new_rows(check, asset, info, template, terr, built, bplan, fin, fbin, P, F, Jf, Wf, jp, parent, names, H)
    fbx_row(check, info, asset)
    summary = summarize(info, checks, motion, pts)
    roles = _roles(fin, joints, names)

    image = next((nm for nm in ("lifted.png", "input.png", "input.jpg") if (asset / nm).is_file()), None)
    bundle = {
        "version": 2, "rig": info,
        "planner": {**planner, "image": image},
        "notes": notes,
        "template": {"rig": template["rig"], "ref_clip": template["ref"], "built": built,
                     "plan": template.get("plan") if built else None, "dir": template.get("dir") if built else None,
                     "fit_scale": ((info.get("fit") or {}).get("scale") or fitnode.get("scale") or [1])[0],
                     "fit_scale_xyz": (info.get("fit") or {}).get("scale"), "fit_yaw": (info.get("fit") or {}).get("yaw"),
                     "fit_offset": (info.get("fit") or {}).get("offset") or fitnode.get("translation", [0, 0, 0]),
                     "max_joint_move_by_skinner_m": round(moved, 4)},
        "roles": roles, "measures": measures, "pose_flags": flags,
        "bones": [{"name": nm, "label": plain_bone(nm), "role": roles[k], "parent": parent[k], "pos": jp[k].round(4).tolist(), "template": mapping[k]["template"],
                   "rule": mapping[k]["rule"], "ref_vs_rest_deg": ref_pose[k]} for k, nm in enumerate(names)],
        "clips": clips,
        "weights": {"top_final": top_f.tolist(), "top_raw": top_r.tolist(),
                    "split_raw": split_raw.round(3).tolist(), "seam_raw": seam_raw, "seam_final": seam_final,
                    "dropped": {"total": len(dropped_list) if kept is not None else info.get("dropped_influences"),
                                "by_bone": dict(sorted(dropped_by_bone.items(), key=lambda x: -x[1])),
                                "list": dropped_list}},
        "fit": fit, "head": head, "checks": checks, "motion": motion, "summary": summary,
    }
    write(asset, bundle)
    return bundle


def write(asset: Path, bundle: dict) -> None:
    """rig/inspect.json, and the small rig/summary.json the versions strip reads."""
    (asset / "rig" / "inspect.json").write_text(json.dumps(bundle, separators=(",", ":"), default=_plain), encoding="utf-8")
    (asset / "rig" / "summary.json").write_text(json.dumps(bundle["summary"], indent=1, default=_plain) + "\n", encoding="utf-8")


def _plain(x):
    """numpy scalars and arrays in the report, as JSON."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


def measure_rest(asset: Path, info: dict) -> dict | None:
    """A rig from before the rest-pose check: measure the head turn on the model it was made from. Only when that
    model is still the one on disk (same point count as the rig), and only in inspect.json: rig.json stays as made."""
    import rest_check

    src = asset / str(info.get("source") or "")
    if not info.get("source") or not src.is_file():
        return None
    try:
        sdoc, sbin = rig.read_glb(src)
        fin, fbin = rig.read_glb(asset / info["rigged"])
        P = rig.accessor(sdoc, sbin, sdoc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).astype(float)
        n = fin["accessors"][fin["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"]
        measured = rest_check.measure(P) if len(P) == n else None
    except Exception as e:  # noqa: BLE001 - a missing measurement never fails the check
        print(f"rig_inspect: couldn't measure the head ({e})", file=sys.stderr)
        return None
    return {"measured": measured, "straightened": None, "before_straightening": True} if measured else None


def rebuild(asset: Path) -> dict:
    """For a take rigged before inspect.json existed: read back what it can from rig.json, rig.log and the GLBs."""
    info = json.loads((asset / "rig" / "rig.json").read_text())
    if not info.get("rest"):
        info["rest"] = measure_rest(asset, info)
    log = (asset / "rig" / "rig.log").read_text() if (asset / "rig" / "rig.log").is_file() else ""
    m = re.search(r"planner says (\w+) \(([\d.]+)\)", log)
    planner = ({"choice": m.group(1), "confidence": float(m.group(2))} if m
               else {"choice": info["class"], "confidence": None, "unrecorded": not log})
    notes = [ln[5:] for ln in log.splitlines() if ln.startswith("rig: ") and "planner says" not in ln
             and not ln[5:].startswith(("pass: ", "warn: ", "fail: ", "Checked: ", "couldn't"))]
    raw, rbin = rig.read_glb(asset / "rig" / "skinned_raw.glb")
    fin, fbin = rig.read_glb(asset / info["rigged"])
    ra = raw["meshes"][0]["primitives"][0]["attributes"]
    J = rig.accessor(raw, rbin, ra["JOINTS_0"]).astype(np.int64)
    W = rig.accessor(raw, rbin, ra["WEIGHTS_0"]).astype(float)
    P = rig.accessor(fin, fbin, fin["meshes"][0]["primitives"][0]["attributes"]["POSITION"]).astype(float)
    if len(J) != len(P):  # skinned on a welded copy: map each corner to its welded point
        Pr = rig.accessor(raw, rbin, ra["POSITION"]).astype(float)
        keys = {tuple(k): i for i, k in enumerate(np.round(Pr / 1e-5).astype(np.int64))}
        inv = np.array([keys[tuple(k)] for k in np.round(P / 1e-5).astype(np.int64)])
        J, W = J[inv], W[inv]
    return build(asset, info, planner, notes, J, W)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True)
    args = ap.parse_args()
    asset = Path(os.environ.get("FACTORY_ROOT") or ROOT) / "out" / args.stem
    if not (asset / "rig" / "rig.json").is_file():
        raise SystemExit(f"rig_inspect: {args.stem} has no rig yet")
    b = rebuild(asset)
    for c in b["checks"]:
        print(f"{c['level']:5} {c['step']:9} {c['label']:30} {c['text']}")
    print(b["summary"]["line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
