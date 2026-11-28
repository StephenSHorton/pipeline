"""Skeleton clips for the Animate stage (scripts/animate.py): read, retime, draw and compare.

A skeleton clip is {"frames": [{"joints2d": {joint: [x, y, confidence]}}]}, x and y in 0..1 of the picture
(origin top-left), with Apple Vision's joint names: what bin/bodypose (scripts/bodypose.swift) reads from
frames. scripts/skeletons/ holds the ones Pipeline ships (walk_side.json: a side-on walk, 73 frames at 24 fps).

- draw_video: the clip as an OpenPose-18 skeleton video on black, the control video LTX-2.3's Union-Control
  IC-LoRA takes (the same limb colours OpenPose uses).
- retime: the clip at another frame rate or length, joints interpolated.
- cycles: the code-made walk, run and attack (pipeline#14), each facing right or left: shipped as the walker's
  (walker_<cycle>_<facing>.json), or made for an asset's own sprite-stack character by
  desktop/frontend/scripts/skeleton-cycle.ts (cycle_for). Each starts facing the camera, turns to the side over
  turn_frames, then loops a cycle of cycle_frames; its "figure" says where the standing figure is (top, ground line,
  middle, 0..1), which the start picture is fitted to (info).
- pose_error: how closely a clip's own skeleton (read back from its frames) follows the control, as a percent
  of the picture's width.

Ported from the pipeline#3 proof of concept (video-trials runners/skeleton_video.py, retime_pose.py,
pose_error.py).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SKELETONS = HERE / "skeletons"

# OpenPose 18: the Vision joint drawn at each OpenPose index.
OPENPOSE = ["head_joint", "neck_1_joint", "right_shoulder_1_joint", "right_forearm_joint", "right_hand_joint",
            "left_shoulder_1_joint", "left_forearm_joint", "left_hand_joint", "right_upLeg_joint", "right_leg_joint",
            "right_foot_joint", "left_upLeg_joint", "left_leg_joint", "left_foot_joint", "right_eye_joint",
            "left_eye_joint", "right_ear_joint", "left_ear_joint"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
# The joints a walk or a punch moves, reported apart by pose_error.
MOVING = ("left_forearm_joint", "right_forearm_joint", "left_hand_joint", "right_hand_joint",
          "left_leg_joint", "right_leg_joint", "left_foot_joint", "right_foot_joint")
CONFIDENT = 0.3
CYCLES = ("walk", "run", "attack")
FACINGS = ("right", "left")
SHIPPED_CYCLE = "walker_{cycle}_{facing}"
# what each cycle does, for the prompt (the flat skeleton does not say which way the body faces: the prompt must)
CYCLE_WORDS = {
    "walk": "walks on the spot with a steady walking stride, legs alternating and arms swinging",
    "run": "runs on the spot with a fast running stride, knees lifting and arms pumping",
    "attack": "raises the fists and throws a straight right punch, then pulls back to the guard",
}


class SkeletonError(ValueError):
    pass


def named(name: str) -> Path:
    """A shipped skeleton by name (walk_side), or a path to a skeleton .json."""
    p = Path(name)
    if p.suffix.lower() == ".json" and p.is_file():
        return p
    if not name.replace("_", "").replace("-", "").isalnum():
        raise SkeletonError(f"no skeleton {name!r}")
    p = SKELETONS / f"{name}.json"
    if not p.is_file():
        have = ", ".join(sorted(q.stem for q in SKELETONS.glob("*.json")))
        raise SkeletonError(f"no skeleton {name!r}; Pipeline has: {have}")
    return p


def shipped() -> list[str]:
    return sorted(q.stem for q in SKELETONS.glob("*.json"))


def shipped_cycle(cycle: str, facing: str) -> str:
    """The shipped skeleton for a code-made cycle and facing (the walker's)."""
    if cycle not in CYCLES:
        raise SkeletonError(f"cycle must be one of {', '.join(CYCLES)}")
    if facing not in FACINGS:
        raise SkeletonError(f"facing must be one of {', '.join(FACINGS)}")
    return SHIPPED_CYCLE.format(cycle=cycle, facing=facing)


def cycle_prompt(cycle: str, facing: str, who: str = "A game character") -> str:
    """A prompt for a code-made cycle: it turns from the camera to face its side, as the skeleton does."""
    return (f"Side view. {who} turns to face {facing} and {CYCLE_WORDS[cycle]}, always seen from the side in profile, "
            "staying in the centre of the frame. Static locked-off camera, full body visible from head to feet, plain "
            "background, even flat lighting.")


def info(path: str | Path) -> dict[str, Any]:
    """What a skeleton file says besides its frames: the frame count, turn_frames and cycle_frames (code-made cycles),
    and figure {top, ground, x} when it says where the standing figure is (else None)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SkeletonError(f"cannot read skeleton {path}: {e}") from e
    if not isinstance(doc, dict):
        raise SkeletonError(f"{path} is not a skeleton clip")
    fig = doc.get("figure")
    figure = None
    if isinstance(fig, dict):
        try:
            top, ground, x = float(fig["top"]), float(fig["ground"]), float(fig.get("x", 0.5))
        except (KeyError, TypeError, ValueError):
            top = ground = x = -1.0
        if 0 <= top < ground <= 1 and 0 <= x <= 1:
            figure = {"top": top, "ground": ground, "x": x}
    whole = lambda k: doc.get(k) if isinstance(doc.get(k), int) and doc.get(k) >= 0 else None  # noqa: E731
    return {"frames": len(doc.get("frames") or []), "turn_frames": whole("turn_frames"),
            "cycle_frames": whole("cycle_frames"), "figure": figure}


def scale_about(frames: list[dict], cx: float, gy: float, f: float) -> list[dict]:
    """Every joint moved towards (cx, gy) by f (0..1 of the picture): the same clip, smaller, standing on the same
    spot."""
    return [{n: [cx + (p[0] - cx) * f, gy + (p[1] - gy) * f] + list(p[2:]) for n, p in j.items()} for j in frames]


def action_of(name: str) -> str:
    """The strip a shipped skeleton makes by default (walker_run_left -> run-left, walk_side -> walk): a strip
    name the 2D sheet takes (letters, digits and hyphens), one per skeleton, so a run never overwrites the walk."""
    words = name.lower().removeprefix("walker_").split("_")
    if words[-1:] == ["side"]:
        words = words[:-1]
    return "-".join(w for w in words if w) or "walk"


def label_of(name: str) -> str:
    """A shipped skeleton in plain words (walker_run_left -> "Run, facing left", walk_side -> "Walk, side on")."""
    words = name.lower().removeprefix("walker_").split("_")
    tail = ""
    if words[-1:] in (["left"], ["right"]):
        tail = f", facing {words.pop()}"
    elif words[-1:] == ["side"]:
        words.pop()
        tail = ", side on"
    head = " ".join(w for w in words if w) or name
    return head[:1].upper() + head[1:] + tail


def load(path: str | Path) -> list[dict[str, list[float]]]:
    """The clip's joints per frame. A frame where no skeleton was found keeps the last frame's joints."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SkeletonError(f"cannot read skeleton {path}: {e}") from e
    frames = doc.get("frames") if isinstance(doc, dict) else None
    if not isinstance(frames, list) or not frames:
        raise SkeletonError(f"{path} holds no frames")
    out, last = [], {}
    for fr in frames:
        j = dict(last)
        j.update((fr or {}).get("joints2d") or {})
        last = j
        out.append(j)
    if not any(out):
        raise SkeletonError(f"{path} holds no joints")
    return out


def retime(frames: list[dict], src_fps: float, dst_fps: float, count: int) -> list[dict]:
    """count frames at dst_fps from a clip at src_fps, each joint interpolated between its two nearest frames."""
    out = []
    for k in range(count):
        t = k * src_fps / dst_fps
        i = min(int(t), len(frames) - 1)
        i2 = min(i + 1, len(frames) - 1)
        a = t - i if i2 != i else 0.0
        A, B = frames[i], frames[i2]
        j = {}
        for name in set(A) | set(B):
            if name in A and name in B:
                j[name] = [A[name][c] * (1 - a) + B[name][c] * a for c in range(2)] + [min(A[name][2], B[name][2])]
            else:
                j[name] = list((A if name in A else B)[name])
        out.append(j)
    return out


def fit_frames(frames: list[dict], count: int) -> list[dict]:
    """Exactly count frames: cut a longer clip, hold the last pose of a shorter one."""
    if len(frames) >= count:
        return frames[:count]
    return frames + [frames[-1]] * (count - len(frames))


def draw_frame(joints: dict, w: int, h: int):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), "black")
    d = ImageDraw.Draw(im)
    pts = {k: (joints[n][0] * w, joints[n][1] * h) for k, n in enumerate(OPENPOSE) if n in joints}
    for li, (a, b) in enumerate(LIMBS):
        if a in pts and b in pts:
            d.line([pts[a], pts[b]], fill=COLOURS[li % 18], width=max(4, w // 120))
    r = max(4, w // 160)
    for k, p in pts.items():
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=COLOURS[k])
    return im


def draw_video(frames: list[dict], w: int, h: int, fps: float, out: Path) -> Path:
    """The skeleton as an H.264 video on black (what --video-conditioning takes)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SkeletonError("ffmpeg is not installed")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, j in enumerate(frames):
            draw_frame(j, w, h).save(Path(tmp) / f"{i:04d}.png")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-framerate", f"{fps:g}", "-i", str(Path(tmp) / "%04d.png"),
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "12", str(out)], check=True)
    return out


def pose_error(control: list[dict], clip: list[dict], skip: int = 0) -> dict[str, Any]:
    """Mean distance between the same joints of control frame i and clip frame i (both confident), in percent of
    the width; frames before skip are left out (frame 0 is the character picture itself, not a pose). Moving
    joints (elbows, hands, knees, feet) are also reported apart."""
    alld, limbd, per = [], [], []
    for i in range(skip, min(len(control), len(clip))):
        A, B = control[i] or {}, clip[i] or {}
        ds, ls = [], []
        for n in set(A) & set(B):
            if A[n][2] > CONFIDENT and B[n][2] > CONFIDENT:
                d = ((A[n][0] - B[n][0]) ** 2 + (A[n][1] - B[n][1]) ** 2) ** 0.5 * 100
                ds.append(d)
                if n in MOVING:
                    ls.append(d)
        if ds:
            alld.append(sum(ds) / len(ds))
            per.append(round(alld[-1], 2))
        if ls:
            limbd.append(sum(ls) / len(ls))
    return {
        "frames_compared": len(alld),
        "mean_joint_error_pct": round(sum(alld) / max(1, len(alld)), 2),
        "mean_limb_error_pct": round(sum(limbd) / max(1, len(limbd)), 2),
        "per_frame_pct": per,
    }


# The skeleton as a tree from the hips (Vision's root): (parent, joint, the part whose length it is). Each part's
# length is fitted on its own; left and right share one.
TREE = [("root", "neck_1_joint", "torso"),
        ("neck_1_joint", "left_shoulder_1_joint", "shoulders"), ("neck_1_joint", "right_shoulder_1_joint", "shoulders"),
        ("left_shoulder_1_joint", "left_forearm_joint", "upper_arm"),
        ("right_shoulder_1_joint", "right_forearm_joint", "upper_arm"),
        ("left_forearm_joint", "left_hand_joint", "forearm"), ("right_forearm_joint", "right_hand_joint", "forearm"),
        ("root", "left_upLeg_joint", "hips"), ("root", "right_upLeg_joint", "hips"),
        ("left_upLeg_joint", "left_leg_joint", "thigh"), ("right_upLeg_joint", "right_leg_joint", "thigh"),
        ("left_leg_joint", "left_foot_joint", "shin"), ("right_leg_joint", "right_foot_joint", "shin"),
        ("neck_1_joint", "head_joint", "head"),
        ("head_joint", "left_eye_joint", "face"), ("head_joint", "right_eye_joint", "face"),
        ("head_joint", "left_ear_joint", "face"), ("head_joint", "right_ear_joint", "face")]
PARTS = ("torso", "shoulders", "upper_arm", "forearm", "hips", "thigh", "shin", "head", "face")
FEET = ("left_foot_joint", "right_foot_joint")
RATIO_BAND = 2.0  # a part is fitted to at most this many times (or 1/this of) the whole body's scale
# What a cycle made for a character's own sprite-stack rig takes from its picture (pipeline#34): where the hips are.
# The rig pivots its thighs at the crotch, where its thigh parts start; Vision, and the pose skeletons LTX learnt
# from, put the hips at the top of the thigh, about where the belt is (the walker: 0.52 of the frame against the
# rig's 0.64). The torso, hip width and leg bones take the picture's lengths so the hips move there with the neck
# and feet where they were; the arms, shoulders and head keep the rig's, which Vision reads less surely. "stance":
# the feet as far apart as the picture's (the walker is drawn standing feet apart, the rig's legs hang straight).
HIP_PARTS = ("torso", "hips", "thigh", "shin", "stance")
LEGS = tuple((f"{s}_upLeg_joint", f"{s}_leg_joint", f"{s}_foot_joint") for s in ("left", "right"))


def _sure(j: dict, name: str) -> bool:
    return name in j and len(j[name]) >= 3 and j[name][2] > CONFIDENT


def _height(j: dict) -> float | None:
    """Neck to the lower foot, in picture heights: the body's size the whole fit is scaled by."""
    feet = [j[f][1] for f in FEET if _sure(j, f)]
    if not _sure(j, "neck_1_joint") or not feet:
        return None
    h = max(feet) - j["neck_1_joint"][1]
    return h if h > 0.05 else None


def _part_lengths(j: dict) -> dict[str, float]:
    """Each part's length in a pose (left and right averaged where both are read)."""
    got: dict[str, list[float]] = {}
    for parent, child, part in TREE:
        if _sure(j, parent) and _sure(j, child):
            got.setdefault(part, []).append(((j[child][0] - j[parent][0]) ** 2 +
                                             (j[child][1] - j[parent][1]) ** 2) ** 0.5)
    return {p: sum(v) / len(v) for p, v in got.items() if sum(v) / len(v) > 1e-3}


def refit(frames: list[dict], target: dict, parts: tuple[str, ...] | None = None) -> tuple[list[dict], dict[str, Any]]:
    """The clip refitted to a picture's own body (target: the joints Vision read from it, the character standing
    as the clip's first frame stands): each part (torso, shoulder and hip widths, upper arm, forearm, thigh, shin,
    head; or only those named in parts) takes the picture's length, the motion stays the clip's, the hips start
    where the picture's are and the lower foot stays on the picture's ground line (rising and falling with the
    clip's, as a run's does).

    A part's bone is scaled, not set to a length, so a limb the clip foreshortens (a turn, a stride towards the
    camera) stays foreshortened. A part the picture has no sure read of, one the clip lacks, or one left out of
    parts takes the body's scale (neck to feet) instead; each is kept within RATIO_BAND of that scale. Raises
    SkeletonError when the picture has no neck, hips or feet to fit to.

    With "stance" in parts, each leg is also spread sideways so the first frame's feet stand as far apart as the
    picture's: the foot by a share of its hip's distance from the middle of the hips, the knee by half that. The
    hips come together as the body turns side-on, so the spread goes with them."""
    first = frames[0] if frames else {}
    h_pic, h_ctl = _height(target), _height(first)
    if not h_pic or not h_ctl or not _sure(target, "root") or not _sure(first, "root"):
        raise SkeletonError("the picture's skeleton has no neck, hips and feet to fit to")
    scale = h_pic / h_ctl
    pic, ctl = _part_lengths(target), _part_lengths(first)
    ratios, fitted = {}, []
    for part in PARTS:
        r = pic[part] / ctl[part] if part in pic and part in ctl and (parts is None or part in parts) else None
        if r is not None:
            r = min(max(r, scale / RATIO_BAND), scale * RATIO_BAND)
            fitted.append(part)
        ratios[part] = r if r is not None else scale
    ground = max(target[f][1] for f in FEET if _sure(target, f))
    c_ground0 = max(first[f][1] for f in FEET if f in first)
    c_root0 = first["root"]
    rels = []
    for j in frames:
        rel = {"root": (0.0, 0.0)}
        for parent, child, part in TREE:
            if child in j and parent in j and parent in rel:
                dx, dy = j[child][0] - j[parent][0], j[child][1] - j[parent][1]
                px, py = rel[parent]
                rel[child] = (px + dx * ratios[part], py + dy * ratios[part])
        for name in j:  # a joint off the tree (not one Vision names): the body's scale about the hips
            if name not in rel:
                rel[name] = ((j[name][0] - j["root"][0]) * scale, (j[name][1] - j["root"][1]) * scale)
        rels.append(rel)
    spread = _stance(rels[0], target) if rels and parts is not None and "stance" in parts else 0.0
    out = []
    for j, rel in zip(frames, rels):
        if spread and all(n in rel for leg in LEGS for n in leg):
            mid = (rel[LEGS[0][0]][0] + rel[LEGS[1][0]][0]) / 2
            for hip, knee, foot in LEGS:
                off = spread * (rel[hip][0] - mid)
                rel[knee] = (rel[knee][0] + off / 2, rel[knee][1])
                rel[foot] = (rel[foot][0] + off, rel[foot][1])
        x0 = target["root"][0] + (j["root"][0] - c_root0[0]) * scale
        feet = [rel[f][1] for f in FEET if f in rel]
        if feet:  # the lower foot on the ground line, lifted as far (in this body's size) as the clip lifts it
            c_low = max(j[f][1] for f in FEET if f in j)
            y0 = ground + (c_low - c_ground0) * scale - max(feet)
        else:
            y0 = target["root"][1] + (j["root"][1] - c_root0[1]) * scale
        out.append({n: [round(x0 + x, 4), round(y0 + y, 4)] + list(j[n][2:3] or [1.0]) for n, (x, y) in rel.items()})
    xs = [v[0] for fr in out for v in fr.values()]
    ys = [v[1] for fr in out for v in fr.values()]
    info = {"scale": round(scale, 3), "ratios": {p: round(r, 3) for p, r in ratios.items()},
            "fitted_parts": fitted + (["stance"] if spread else []), "stance": round(spread, 3), "box": [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3),
                                            round(max(ys), 3)]}
    return out, info


def _stance(rel: dict, target: dict) -> float:
    """How far to spread the legs (a share of each hip's distance from the middle of the hips) for the first frame's
    feet to stand as far apart as the picture's; 0 when either has no sure read of both hips and feet."""
    (lh, _, lf), (rh, _, rf) = LEGS
    if not all(n in rel for n in (lh, rh, lf, rf)) or not all(_sure(target, n) for n in (lf, rf)):
        return 0.0
    half_hips = (rel[lh][0] - rel[rh][0]) / 2
    if abs(half_hips) < 1e-3:
        return 0.0
    want = (target[lf][0] - target[rf][0]) / 2 - (rel[lf][0] - rel[rf][0]) / 2
    return max(-RATIO_BAND, min(RATIO_BAND, want / half_hips))


def read_picture(bodypose: Path, picture: Path, scratch: Path) -> dict:
    """One picture's skeleton by Apple Vision (bin/bodypose), {} when none is read or the tool fails."""
    folder = Path(scratch) / "start"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copyfile(picture, folder / "0000.png")
    read = Path(scratch) / "start.json"
    try:
        got = read_frames(bodypose, folder, read)
    except (OSError, ValueError, subprocess.CalledProcessError):
        got = []
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        read.unlink(missing_ok=True)
    return got[0] if got else {}


def read_frames(bodypose: Path, frames_dir: Path, out: Path) -> list[dict]:
    """Each frame's skeleton by Apple Vision (bin/bodypose), as joints per frame ({} where none was found)."""
    subprocess.run([str(bodypose), str(frames_dir), str(out)], check=True, stdout=subprocess.DEVNULL)
    doc = json.loads(Path(out).read_text(encoding="utf-8"))
    return [(fr or {}).get("joints2d") or {} for fr in doc.get("frames") or []]
