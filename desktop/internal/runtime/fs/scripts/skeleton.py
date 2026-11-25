"""Skeleton clips for the Animate stage (scripts/animate.py): read, retime, draw and compare.

A skeleton clip is {"frames": [{"joints2d": {joint: [x, y, confidence]}}]}, x and y in 0..1 of the picture
(origin top-left), with Apple Vision's joint names: what bin/bodypose (scripts/bodypose.swift) reads from
frames. scripts/skeletons/ holds the ones Pipeline ships (walk_side.json: a side-on walk, 73 frames at 24 fps).

- draw_video: the clip as an OpenPose-18 skeleton video on black, the control video LTX-2.3's Union-Control
  IC-LoRA takes (the same limb colours OpenPose uses).
- retime: the clip at another frame rate or length, joints interpolated.
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


def read_frames(bodypose: Path, frames_dir: Path, out: Path) -> list[dict]:
    """Each frame's skeleton by Apple Vision (bin/bodypose), as joints per frame ({} where none was found)."""
    subprocess.run([str(bodypose), str(frames_dir), str(out)], check=True, stdout=subprocess.DEVNULL)
    doc = json.loads(Path(out).read_text(encoding="utf-8"))
    return [(fr or {}).get("joints2d") or {} for fr in doc.get("frames") or []]
