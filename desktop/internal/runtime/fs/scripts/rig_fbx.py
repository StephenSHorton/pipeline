#!/usr/bin/env python3
"""Write the s&box FBX from rigged.glb and check it by importing it back: one Blender call, one rig/fbx_check.json.

    .venv/bin/python scripts/rig_fbx.py --stem psx-cow                # export rig/<name>.fbx and check it
    .venv/bin/python scripts/rig_fbx.py --stem psx-cow --check-only   # check the FBX that is already there

export_and_check(glb, fbx, clips) runs scripts/blender/fbx_check.py, which reproduces the old rig.export_fbx (drop the
importer's bone-shape mesh, name the armature 'rig', strip action prefixes), sets the scene to 30 fps, exports with
add_leaf_bones off, every action, no NLA, FACE smoothing and path_mode STRIP, then resets the scene, re-imports the
FBX and prints FBX_CHECK <json>. This module turns that report into rig/fbx_check.json:

  ok        no problems
  problems  action count or names differ from the shipped clips, bone names differ from the skin joints,
            more than 4 weights on a vertex, scene fps is not 30, no armature or no skinned mesh, an action whose
            length (frames / fps) is more than one frame off its GLB clip's duration
  warnings  a clip (not a lying one: Death, Sleep, ...) whose mid-clip height is outside 0.5-1.5x its first frame,
            or a clip with zero height
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "blender" / "fbx_check.py"
FPS = 30
MAX_INFLUENCES = 4
HEIGHT_RANGE = (0.5, 1.5)
LYING = re.compile(r"death|die|dead|sleep|lie|lay|laying|lying|rest|sit|crouch|roll|fall", re.I)
NO_BLENDER = "Blender not found; skipped the FBX (set BLENDER to its binary)."


def clip_name(action: str) -> str:
    """'rig|rig|Walk' (what the FBX importer names a take) -> 'Walk'."""
    return action.split("|")[-1]


def glb_facts(glb: Path) -> tuple[list[str], list[str]]:
    """(skin joint names, animation names) of a rigged GLB."""
    doc, _ = rig.read_glb(Path(glb))
    nodes = doc.get("nodes", [])
    joints = []
    if doc.get("skins"):
        joints = [nodes[j].get("name", f"node_{j}") for j in doc["skins"][0]["joints"]]
    return joints, [a.get("name", f"animation_{i}") for i, a in enumerate(doc.get("animations", []))]


def glb_durations(glb: Path) -> dict[str, float]:
    """{animation name: seconds} of a rigged GLB (the largest sampler input time of each animation)."""
    doc, _ = rig.read_glb(Path(glb))
    accs = doc.get("accessors", [])
    out: dict[str, float] = {}
    for i, a in enumerate(doc.get("animations", [])):
        t = 0.0
        for smp in a.get("samplers", []):
            acc = accs[smp["input"]] if smp.get("input") is not None and smp["input"] < len(accs) else {}
            mx = acc.get("max") or [0.0]
            t = max(t, float(mx[0]))
        out[a.get("name", f"animation_{i}")] = t
    return out


def evaluate(report: dict, clips: list[str], joints: list[str],
             durations: dict[str, float] | None = None) -> tuple[list[str], list[str]]:
    """(problems, warnings) for one FBX_CHECK report against the clips and skin joints that were shipped.

    durations ({clip: seconds} from the GLB): each action's (frame_end - frame_start) / scene fps must match its clip
    within one frame, or the FBX plays the clip at the wrong speed."""
    problems: list[str] = []
    warnings: list[str] = []
    if not report.get("armatures"):
        problems.append("no armature in the FBX")
    if not report.get("mesh_vertices"):
        problems.append("no skinned mesh in the FBX")
    actions = report.get("actions") or []
    names = [clip_name(a["name"]) for a in actions]
    if len(actions) != len(clips):
        problems.append(f"{len(actions)} actions in the FBX, {len(clips)} clips shipped")
    missing = [c for c in clips if c not in names]
    extra = [n for n in names if n not in clips]
    if missing or extra:
        problems.append("action names differ from the clips" + (": missing " + ", ".join(missing) if missing else "")
                        + (("; " if missing else ": ") + "extra " + ", ".join(extra) if extra else ""))
    bones = list(report.get("bone_names") or [])
    if joints and set(bones) != set(joints):
        lost = [j for j in joints if j not in set(bones)]
        added = [b for b in bones if b not in set(joints)]
        problems.append("bone names differ from the skin joints"
                        + (": missing " + ", ".join(lost[:8]) if lost else "")
                        + (("; " if lost else ": ") + "extra " + ", ".join(added[:8]) if added else ""))
    mi = int(report.get("max_influences") or 0)
    if mi > MAX_INFLUENCES:
        problems.append(f"{mi} weights on a vertex (s&box takes {MAX_INFLUENCES})")
    fps = report.get("scene_fps")
    if fps is None or abs(float(fps) - FPS) > 1e-3:
        problems.append(f"scene fps {fps}, expected {FPS}")
    if durations and fps:
        for a in actions:
            want = durations.get(clip_name(a["name"]))
            if want is None or a.get("frame_end") is None or a.get("frame_start") is None:
                continue
            got = (float(a["frame_end"]) - float(a["frame_start"])) / float(fps)
            if abs(got - want) > 1.0 / float(fps) + 1e-6:
                problems.append(f"{clip_name(a['name'])} lasts {got:.3f} s in the FBX, {want:.3f} s in the GLB "
                                "(wrong frame rate)")
    for a in actions:
        h0, hm = a.get("height_first"), a.get("height_mid")
        if h0 is None or hm is None:
            continue
        name = clip_name(a["name"])
        if h0 <= 1e-6 or hm <= 1e-6:
            warnings.append(f"{name}: zero height in the FBX")
        elif not LYING.search(name) and not (HEIGHT_RANGE[0] <= hm / h0 <= HEIGHT_RANGE[1]):
            warnings.append(f"{name}: mid-clip height is {hm / h0:.2f}x the first frame")
    return problems, warnings


def note_for(check: dict) -> str:
    if check.get("ok"):
        return (f"FBX checked: {len(check.get('actions') or [])} clips, {check.get('bones', 0)} bones, "
                f"{check.get('max_influences', 0)} weights max, {check.get('scene_fps', 0):g} fps")
    return "FBX check failed: " + "; ".join(check.get("problems") or ["no report"])


def export_and_check(glb: Path, fbx: Path, clips: list[str] | None = None, check_only: bool = False,
                     out_json: Path | None = None, blender: str | None = None,
                     timeout: int = 600) -> tuple[str, dict | None]:
    """Export glb -> fbx (unless check_only) and verify the FBX; write fbx_check.json beside the FBX.

    Returns (note for rig.json fbx_note / the app, check dict or None when Blender is missing)."""
    glb, fbx = Path(glb), Path(fbx)
    blender = blender or rig.blender_path()
    if not blender:
        return NO_BLENDER, None
    joints, anims = glb_facts(glb) if glb.is_file() else ([], [])
    durations = glb_durations(glb) if glb.is_file() else {}
    clips = list(clips) if clips is not None else anims
    out_json = Path(out_json) if out_json else fbx.parent / "fbx_check.json"
    fbx.parent.mkdir(parents=True, exist_ok=True)
    cmd = [blender, "-b", "--factory-startup", "-P", str(SCRIPT), "--", str(glb), str(fbx)]
    if check_only:
        cmd.append("--check-only")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        stdout, stderr = (out.decode(errors="replace") if isinstance(out, bytes) else out), f"timed out after {timeout}s"
    lines = stdout.splitlines()
    exported = [ln for ln in lines if ln.startswith("FBX_OK")]
    reports = [ln for ln in lines if ln.startswith("FBX_CHECK ")]
    check: dict = {"glb": str(glb), "fbx": str(fbx), "clips": clips, "skin_joints": len(joints),
                   "check_only": bool(check_only), "made": int(time.time())}
    if not fbx.is_file() or (not check_only and not exported):
        tail = (stderr or stdout)[-400:]
        check.update(ok=False, problems=["Blender could not write the FBX"], warnings=[], log=tail)
        _write(out_json, check)
        return "Blender could not write the FBX: " + tail, check
    if not reports:
        tail = (stderr or stdout)[-400:]
        check.update(ok=False, problems=["Blender could not read the FBX back"], warnings=[], log=tail)
        _write(out_json, check)
        return "FBX written but Blender could not read it back: " + tail, check
    report = json.loads(reports[-1][len("FBX_CHECK "):])
    problems, warnings = evaluate(report, clips, joints, durations)
    check.update(report)
    check.update(ok=not problems, problems=problems, warnings=warnings)
    _write(out_json, check)
    return note_for(check), check


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def paths_for(stem: str) -> tuple[Path, Path, list[str] | None]:
    """(rigged.glb, rig/<name>.fbx, shipped clips) for an asset, from its rig.json when there is one."""
    asset = rig.OUT / stem
    work = asset / "rig"
    report = {}
    if (work / "rig.json").is_file():
        try:
            report = json.loads((work / "rig.json").read_text())
        except (OSError, ValueError):
            report = {}
    fbx = None
    if report.get("fbx"):
        fbx = asset / report["fbx"]
    else:
        found = sorted(work.glob("*.fbx")) if work.is_dir() else []
        fbx = found[0] if len(found) == 1 else work / f"{stem.replace('-', '_')}.fbx"
    clips = report.get("clips") if isinstance(report.get("clips"), list) else None
    if clips and isinstance(clips[0], dict):
        clips = [c.get("name") for c in clips]
    return asset / "rigged.glb", fbx, clips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stem", required=True)
    ap.add_argument("--check-only", action="store_true", help="check the FBX already written; do not export")
    args = ap.parse_args()
    glb, fbx, clips = paths_for(args.stem)
    if not glb.is_file():
        print(f"no rigged.glb for {args.stem}", file=sys.stderr)
        return 2
    if args.check_only and not fbx.is_file():
        print(f"no FBX at {fbx}", file=sys.stderr)
        return 2
    note, check = export_and_check(glb, fbx, clips, check_only=args.check_only)
    print(note)
    if check is None:
        return 3
    for w in check.get("warnings") or []:
        print("warning:", w)
    return 0 if check.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
