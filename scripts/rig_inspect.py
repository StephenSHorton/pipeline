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
    if worst and all("stretch_p99" in v for v in mc.values()):
        # the single worst edge saturates on dense meshes (a fine-looking Idle read 56×); 1% of edges is the number
        w99 = max(mc, key=lambda c: mc[c]["stretch_p99"])
        parts.append(f"1% of edges stretch {mc[w99]['stretch_p99']:.1f}× ({w99.replace('_', ' ')})")
    elif worst:
        parts.append(f"worst stretch {mc[worst]['stretch_max']:.1f}× ({worst.replace('_', ' ')})")
    line = " · ".join(parts)
    problems = sorted((c for c in checks if c["level"] != "pass"), key=lambda c: (LEVELS[c["level"]], STEPS.index(c["step"])))
    return {"version": 1, "made": int(info.get("made") or time.time()), "class": info["class"], "level": level,
            "line": line[:1].upper() + line[1:], "tears": int(tears),
            "motion": {c: {"tears": v["tears"], "stretch_max": v["stretch_max"]} for c, v in mc.items()} if motion else None,
            "head": None if not measured and not hc else {
                "before": measured["head_yaw_deg"] if measured else None,
                "after": straight["head_yaw_deg"] if straight else None, "straightened": bool(straight),
                "off_deg": hc["off_deg"] if hc else None, "made_before": bool(rest.get("before_straightening"))},
            "worst_stretch": {"clip": worst, "x": round(mc[worst]["stretch_max"], 1)} if worst else None,
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


def build(asset: Path, info: dict, planner: dict, notes: list[str], raw_J: np.ndarray, raw_W: np.ndarray,
          kept: np.ndarray | None = None) -> dict:
    """raw_J/raw_W: the skinner's weights per source vertex; kept: the far-influence mask on them (None: unknown)."""
    template = rig.TEMPLATES[info["class"]]
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
    if fc:
        lvl = "warn" if fc.get("cue_conflict") else "pass"
        check(lvl, "facing", "skeleton", f"Template turned {fc['yaw']}°",
              f"The template was turned {fc['yaw']}° to face the way the model does (decided by {fc['decided_by']}, "
              f"registration margin {fc['margin']:.2f})." + (" The cues disagreed." if fc.get("cue_conflict") else ""),
              yaw=fc["yaw"], decided_by=fc["decided_by"])
    if fi.get("poor_match"):
        check("warn", "match", "skeleton", f"Poor body-plan match ({fi['scale_ratio']:.1f}×)",
              f"The template had to be stretched {fi['scale_ratio']:.1f}× more along one axis than another to fit this model.",
              ratio=fi["scale_ratio"])
    gate = fi.get("gate")
    if gate:
        bad = gate["level"] == "fail"
        check("fail" if bad else "pass", "pose", "skeleton",
              f"Skeleton misses the body ({gate['joints_outside_box']:.0%} of joints outside)" if bad else "Skeleton inside the body",
              f"{gate['joints_outside_box']:.0%} of the joints are outside the model's box and {gate['bone_far_outside_0p08H']:.0%} "
              "of the bones lie well outside the model.", **{k: v for k, v in gate.items() if k != "level"})
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
    if info.get("missing_clips"):
        miss = info["missing_clips"]
        check("warn", "clips", "motion", f"{len(miss)} clip{'s' if len(miss) != 1 else ''} missing",
              "No clip for this body plan: " + ", ".join(miss) + ".", missing=miss)
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
            check("fail" if mc[sc_]["stretch_p99"] > 6 else "warn", "stretch", "motion",
                  f"{sc_.replace('_', ' ')} stretches {mc[sc_]['stretch_p99']:.1f}×",
                  f"In {sc_.replace('_', ' ')}, 1% of the edges stretch (or squash) more than {mc[sc_]['stretch_p99']:.1f}× "
                  "(warn above 3×, fail above 6×).", clip=sc_)
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
    summary = summarize(info, checks, motion, pts)

    image = next((nm for nm in ("lifted.png", "input.png", "input.jpg") if (asset / nm).is_file()), None)
    bundle = {
        "version": 2, "rig": info,
        "planner": {**planner, "image": image},
        "notes": notes,
        "template": {"rig": template["rig"], "ref_clip": template["ref"],
                     "fit_scale": ((info.get("fit") or {}).get("scale") or fitnode.get("scale") or [1])[0],
                     "fit_scale_xyz": (info.get("fit") or {}).get("scale"), "fit_yaw": (info.get("fit") or {}).get("yaw"),
                     "fit_offset": (info.get("fit") or {}).get("offset") or fitnode.get("translation", [0, 0, 0]),
                     "max_joint_move_by_skinner_m": round(moved, 4)},
        "bones": [{"name": nm, "label": plain_bone(nm), "parent": parent[k], "pos": jp[k].round(4).tolist(), "template": mapping[k]["template"],
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
    (asset / "rig" / "inspect.json").write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    (asset / "rig" / "summary.json").write_text(json.dumps(bundle["summary"], indent=1) + "\n", encoding="utf-8")


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
