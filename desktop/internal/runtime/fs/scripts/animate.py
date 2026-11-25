#!/usr/bin/env python3
"""Animate stage: one of an asset's character pictures plus a skeleton clip becomes a short animation, made on the
Mac with LTX-2.3 and its Union-Control IC-LoRA (Lightricks, LTX-2 Community License: see NOTICE), then cut into
frames, transparent cut-outs and a sprite strip, saved as a new 2D Version of the asset.

    python scripts/animate.py --root R --stem S --prompt "The soldier faces right and walks to the right ..." \\
        [--skeleton walk_side] [--action walk] [--picture flat/source.png] [--seed 42] [--frames 73] [--size 704]
        [--side flat/views/side.png [--side-frame 24] [--side-strength 0.8]]
        [--remove "backpack" [--keep "chest rig, radio and pouches"] [--remove-frames 24,48,72]]

What it writes, under out/<stem>/:
  flat/anim/<action>/guide.png      the picture LTX starts from: the character, square, on its ground
  flat/anim/<action>/side.png       with --side: the side view, fitted the same way, LTX's second guide
  flat/anim/<action>/first/         with --remove: the first pass's frames at --remove-frames, as LTX drew them
  flat/anim/<action>/keys/          with --remove: the same frames with the named kit erased, LTX's later guides
  flat/anim/<action>/control.json   the skeleton per frame (24 fps), refitted to the guide picture's own body
                                    (its "fit"), and skeleton.mp4 drawn from it
  flat/anim/<action>/clip.mp4       what LTX-2.3 made (frame 0 is the guide picture)
  flat/anim/<action>/frames/        every frame, NNNN.png
  flat/anim/<action>/cutouts/       every frame cut out (Apple Vision's foreground mask, bin/lift-subject; else
                                    the ground keyed out). A cut-out far smaller than the frames around it
                                    (Vision drops motion-blurred legs) is redone against the ground those
                                    frames show behind it, or, if that fails too, left out of the strip
  flat/anim/<action>/anim.json      what made it, the AI-disclosure note, and the skeleton check when it ran
                                    (all built in out/<stem>/.animate/ first and moved in only when the run worked)
  flat/sheets/<action>.png          the sprite strip: every --every'th cut-out after frame 0, cropped to one box,
                                    in the 2D stage's motion-strip slot (sheets/manifest.json), shown in the app
The 2D Version is saved before (what was there) and after (the result).

The prompt must say which way the character faces and moves: a flat skeleton does not.

The skeleton is refitted to the picture's own body before it is drawn (skeleton.refit): Apple Vision reads the
guide picture's skeleton and each part of the shipped one (torso, shoulder and hip widths, upper arm, forearm,
thigh, shin, head) takes the picture's length, the hips start where the picture's are and the feet stay on its
ground line. Moving and scaling alone left a realistic operator ~6% off the cartoon-built walker while standing
still (docs/animate-operator-cycles.md). Without bin/bodypose, or with no skeleton read, it is used as shipped.

A front picture alone leaves the model to guess what the character looks like side-on, and it invents kit (a
backpack, a pack where the chest rig was). --side gives it a side view too: a second --image guide (a keyframe,
not the first frame) at --side-frame, by default one second in (frame 44 tested worse on the operator). It must
face the way the prompt says the character moves. Its character is cut out, scaled to the first picture's
character's height and stood on the same spot, on the same ground (flat/anim/<action>/side.png). It costs some
skeleton accuracy: the character holds the side view's pose around that frame.

--remove "backpack" stops invented kit without that cost. Animate runs LTX twice: a first pass from the front
picture alone (which follows the skeleton, and invents the kit), then Qwen-Image-Edit erases the named kit from that
pass's own frames at --remove-frames (default every second from frame 24, and the last frame), keeping the pose LTX
drew under the skeleton; those frames are the second pass's keyframe guides (strength 0.8). --keep names the kit
that must stay (asked only to remove the backpack, the model took the operator's chest rig too, and the guide carried
that through the whole clip). An erased frame whose skeleton (Apple Vision) is more than 2.5% off the control's (or 1% further than
the first pass's own frame, when that is already off) is erased again with the next seed, up to three tries, keeping
the closest. About three times as long as one pass. On the
operator's walk: 1.87% (front picture alone 1.68%, side view 5.1%), no backpack, chest rig kept. The edit model is freed before
the second pass.

LTX runs as its own process, the LTX venv's python under -I, from the ltx-2-mlx clone's source folders
(scripts/ltx_run.py), with local paths only and HF_HUB_OFFLINE=1, so it never downloads. Where it looks (each
overridable by environment):
  PIPELINE_LTX_DIR     <root>/vendor/ltx-2-mlx                 the dgrauet/ltx-2-mlx clone (MIT)
  PIPELINE_LTX_PYTHON  <PIPELINE_LTX_DIR>/.venv/bin/python     its libraries (mlx, mlx-lm, ...), nothing installed
  PIPELINE_LTX_MODEL   <root>/weights/ltx-2.3-mlx-q8           distilled-1.1 transformer, connector, VAEs, upscaler
  PIPELINE_LTX_LORA    <root>/weights/ic-lora-union/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
  PIPELINE_LTX_GEMMA   <root>/weights/gemma-3-12b-it-4bit, else the Hugging Face cache's
                       mlx-community/gemma-3-12b-it-4bit snapshot
  PIPELINE_BODYPOSE    <root>/bin/bodypose (scripts/bodypose.swift): reads the guide picture's skeleton, which
                       the shipped skeleton is refitted to, and the result's skeleton for the check
scripts/bootstrap.sh animate sets these up. About 40 GB at its peak and a minute or so a clip: one big model at a
time on the Mac (the factory runs it in its one job slot, after freeing the draw model).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import skeleton as SK  # noqa: E402
from factory_env import check_stem, stem_dir  # noqa: E402

LORA_NAME = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"
FPS = 24
MODEL_NOTE = "LTX-2.3 distilled 1.1 (q8, dgrauet/ltx-2-mlx) with the IC-LoRA Union-Control (ref0.5), Lightricks"
LICENCE = "LTX-2 Community License Agreement (Lightricks), see NOTICE"
DISCLOSURE = ("Machine-generated: made with the LTX-2.3 video model. Under the LTX-2 Community License, content "
              "published from it must say plainly that it is machine-generated.")
STAGES = ("guide", "skeleton", "ltx", "frames", "cutouts", "strip", "pose_check")
REMOVE_STAGES = ("guide", "skeleton", "first_pass", "erase", "ltx", "frames", "cutouts", "strip", "pose_check")
KEY_STRENGTH = 0.8
KEY_TRIES = 3
KEY_POSE_PCT = 2.5  # an erased frame further than this from the control's skeleton is erased again
# What the edit model is asked, with the kit named: erase only that, keep the pose and everything else. Naming the
# kit that stays matters: asked only to remove a backpack, it took the operator's chest rig too.
ERASE_PROMPT = ("Remove only the {what} from the person, together with anything that holds it on, such as straps. "
                "{keep}Keep everything else exactly the same: the pose, position, size, face, hair, clothes, "
                "everything else they wear or carry, the lighting and the background.")


def erase_prompt(what: str, keep: str | None = None) -> str:
    return ERASE_PROMPT.format(what=what, keep=f"Keep the {keep} exactly as they are. " if keep else "")
PICTURES = ("lifted.png", "flat/source.png", "input.png", "input.jpg", "input.jpeg", "input.webp")


class AnimateError(RuntimeError):
    pass


def stage(name: str, label: str) -> None:
    print(f"STAGE {name} {label}", flush=True)


# ------------------------------------------------------------------ where LTX is

def ltx_paths(root: Path) -> dict[str, Path]:
    env = os.environ.get
    clone = Path(env("PIPELINE_LTX_DIR") or root / "vendor" / "ltx-2-mlx")
    paths = {
        "clone": clone,
        "python": Path(env("PIPELINE_LTX_PYTHON") or clone / ".venv" / "bin" / "python"),
        "model": Path(env("PIPELINE_LTX_MODEL") or root / "weights" / "ltx-2.3-mlx-q8"),
        "lora": Path(env("PIPELINE_LTX_LORA") or root / "weights" / "ic-lora-union" / LORA_NAME),
        "gemma": Path(env("PIPELINE_LTX_GEMMA") or root / "weights" / "gemma-3-12b-it-4bit"),
    }
    if not env("PIPELINE_LTX_GEMMA") and not paths["gemma"].is_dir():
        cached = hf_snapshot(GEMMA_REPO)
        if cached:
            paths["gemma"] = cached
    return paths


def hf_snapshot(repo: str) -> Path | None:
    """The newest local snapshot of a Hugging Face repo in the cache, without asking the network."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" /
                                                       "huggingface") / "hub")
    snaps = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
    found = sorted((p for p in snaps.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def check_ltx(paths: dict[str, Path]) -> None:
    missing = []
    if not (paths["clone"] / "packages" / "ltx-pipelines-mlx" / "src").is_dir():
        missing.append(f"the ltx-2-mlx clone at {paths['clone']}")
    if not paths["python"].is_file():
        missing.append(f"its python at {paths['python']}")
    if not (paths["model"] / "config.json").is_file():
        missing.append(f"the LTX-2.3 weights at {paths['model']}")
    if not paths["lora"].is_file():
        missing.append(f"the Union-Control IC-LoRA at {paths['lora']}")
    if not paths["gemma"].is_dir():
        missing.append(f"the Gemma text encoder ({GEMMA_REPO}) at {paths['gemma']}")
    if missing:
        raise AnimateError("LTX-2.3 is not set up (scripts/bootstrap.sh animate): no " + "; no ".join(missing))


def ltx_argv(paths: dict[str, Path], *, prompt: str, out: Path, control: Path, guide: Path, size: int,
             frames: int, seed: int, side: Path | None = None, side_frame: int = FPS,
             side_strength: float = 0.8, keys: list[tuple[Path, int]] = ()) -> list[str]:
    images = ["--image", str(guide), "0", "1.0"]
    if side is not None:  # a later frame: ltx-2-mlx binds it as a keyframe guide, not the first latent
        images += ["--image", str(side), str(side_frame), f"{side_strength:g}"]
    for path, frame in keys:  # --remove's erased frames, keyframe guides the same way
        images += ["--image", str(path), str(frame), f"{KEY_STRENGTH:g}"]
    return [str(paths["python"]), "-I", str(HERE / "ltx_run.py"), str(paths["clone"]), "ic-lora",
            "--model", str(paths["model"]), "--gemma", str(paths["gemma"]),
            "--lora", str(paths["lora"]), "1.0",
            "--video-conditioning", str(control), "1.0",
            *images,
            "-H", str(size), "-W", str(size), "-f", str(frames), "--frame-rate", str(FPS),
            "--seed", str(seed), "--prompt", prompt, "--output", str(out)]


def ltx_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    return env


# ------------------------------------------------------------------ the picture

def pick_picture(folder: Path, rel: str | None) -> Path:
    if rel:
        p = (folder / rel).resolve()
        if folder.resolve() not in p.parents or not p.is_file():
            raise AnimateError(f"no picture {rel!r} in this asset")
        return p
    for name in PICTURES:
        if (folder / name).is_file():
            return folder / name
    raise AnimateError("this asset has no character picture yet (draw or add one first)")


def make_guide(picture: Path, size: int, out: Path, onto: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    """The picture as LTX's first frame: square, the whole picture fitted in, padded with its own ground. A cut-out
    (with transparency) and a picture on a strong key colour (magenta) go onto white instead, which the prompt
    asks for. With onto (the side view: it must sit on the first picture's ground), the picture is cut from its
    own ground and put onto that one. Returns the ground colour."""
    from PIL import Image

    from flat2d.parts import ground_key

    im = Image.open(picture)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        ground = onto or (255, 255, 255)
        base = Image.new("RGB", rgba.size, ground)
        base.paste(rgba, (0, 0), rgba)
        rgb = base
    else:
        rgb = im.convert("RGB")
        key = tuple(ground_key(rgb)["rgb"])
        light = min(key) >= 190 and max(key) - min(key) <= 40
        if (onto is None and light) or onto == key:
            ground = key
        else:
            from flat2d.atlas import unmix

            ground = onto or (255, 255, 255)
            cut = unmix(rgb, key)
            base = Image.new("RGB", rgb.size, ground)
            base.paste(cut, (0, 0), cut)
            rgb = base
    w, h = rgb.size
    s = size / max(w, h)
    small = rgb.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    sq = Image.new("RGB", (size, size), ground)
    sq.paste(small, ((size - small.size[0]) // 2, (size - small.size[1]) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    sq.save(out)
    return ground


def _figure_box(im, ground: tuple[int, int, int], lift: Path | None, scratch: Path):
    """The character's box in a picture: Apple Vision's foreground (bin/lift-subject) when it is there, else what
    stands off the ground colour (rows and columns with enough of it; a box that runs into the frame's edge, as a
    studio floor's shading does, counts as not found).
    Returns (cut-out RGBA or None, box or None)."""
    import numpy as np
    from PIL import Image

    if lift:
        src, dst = scratch / "lift-in.png", scratch / "lift-out.png"
        im.convert("RGB").save(src)
        r = subprocess.run([str(lift), str(src), str(dst)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cut = Image.open(dst).convert("RGBA") if r.returncode == 0 and dst.is_file() else None
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        if cut is not None:
            box = cut.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
            if box:
                return cut, box
    a = np.asarray(im.convert("RGB")).astype(int)
    m = np.abs(a - np.array(ground)).sum(2) > 60
    rows = np.where(m.sum(1) > m.shape[1] * 0.01)[0]
    cols = np.where(m.sum(0) > m.shape[0] * 0.01)[0]
    if not len(rows) or not len(cols):
        return None, None
    box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    h, w = m.shape
    if box[0] <= 1 or box[1] <= 1 or box[2] >= w - 1 or box[3] >= h - 1:  # runs into the frame: a floor, not a body
        return None, None
    return None, box


def fit_side(picture: Path, guide: Path, size: int, out: Path, ground: tuple[int, int, int],
             lift: Path | None) -> bool:
    """The side view as LTX's second guide, its character the same height as the first picture's and standing on
    the same spot (feet on the same line, centred on the same column), on the first picture's ground: a figure
    smaller or off to one side pulls the body off the skeleton. Falls back to make_guide's whole-picture fit (and
    returns False) when either character cannot be found."""
    from PIL import Image

    scratch = out.parent
    _, front = _figure_box(Image.open(guide), ground, lift, scratch)
    make_guide(picture, size, out, onto=ground)  # the side view on the first picture's ground, as the fallback
    fitted = Image.open(out).convert("RGB")
    cut, box = _figure_box(fitted, ground, lift, scratch)
    if not front or not box:
        return False
    if cut is None:
        from flat2d.atlas import unmix

        cut = unmix(fitted, ground)
    fig = cut.crop(box)
    h = front[3] - front[1]
    w = max(1, round(fig.size[0] * h / fig.size[1]))
    fig = fig.resize((w, h), Image.LANCZOS)
    sq = Image.new("RGB", (size, size), ground)
    cx = (front[0] + front[2]) // 2
    sq.paste(fig, (cx - w // 2, front[3] - h), fig)
    sq.save(out)
    return True


# ------------------------------------------------------------------ erasing invented kit (--remove)

def remove_frames(frames: int, spec: str | None = None) -> list[int]:
    """The frames whose kit is erased and that guide the second pass: by default every second from frame 24 and the
    last frame (24, 48, 72 for 73 frames), or the given ones ("24,48,72"), each 1 to frames-1, at most 8."""
    if spec is None or not str(spec).strip():
        picks = list(range(FPS, frames - 1, FPS)) + [frames - 1]
    else:
        try:
            picks = [int(x) for x in str(spec).replace(" ", "").split(",") if x]
        except ValueError:
            raise AnimateError("remove-frames must be frame numbers, like 24,48,72") from None
        if not picks or any(not 1 <= f <= frames - 1 for f in picks):
            raise AnimateError(f"remove-frames must be from 1 to {frames - 1} (frame 0 is the first picture)")
    picks = sorted(set(picks))
    if len(picks) > 8:
        raise AnimateError("at most 8 remove-frames")
    return picks


def erase_kit(sources: list[Path], picks: list[int], prompt: str, dest: Path, seed: int,
              control: list[dict], tool: Path | None) -> tuple[list[Path], list[dict]]:
    """Each frame with the named kit erased by the photo-edit model (Qwen-Image-Edit-2511, flat2d), the same size,
    saved under dest with the same name. With bin/bodypose, a result whose skeleton is more than KEY_POSE_PCT off
    the control's at that frame is erased again with the next seed (KEY_TRIES in all) and the closest kept. The
    edit model is freed after, before LTX needs the memory. Returns the files and, per frame, its seed and error."""
    from PIL import Image

    from flat2d import optional as O

    dest.mkdir(parents=True, exist_ok=True)
    out, notes = [], []
    try:
        for src, frame in zip(sources, picks):
            with Image.open(src) as im:
                size, before = im.size, im.convert("RGB")
            best = None
            # The first pass's own frame may already be off the skeleton: an edit is only asked to stay near it.
            own = key_pose(tool, before, control[frame], dest.parent) if tool else None
            limit = max(KEY_POSE_PCT, (own or 0) + 1.0)
            for k in range(KEY_TRIES if tool else 1):
                got = O.edit_image(prompt, src, seed=seed + k, width=size[0], height=size[1])
                got = getattr(got, "image", got).convert("RGB")
                if got.size != size:
                    got = got.resize(size, Image.LANCZOS)
                pct = key_pose(tool, got, control[frame], dest.parent) if tool else None
                if best is None or (pct is not None and (best[1] is None or pct < best[1])):
                    best = (got, pct, seed + k)
                if pct is not None and pct <= limit:
                    break
            target = dest / src.name
            best[0].save(target)
            out.append(target)
            notes.append({"frame": frame, "seed": best[2], "pose_pct": best[1], "first_pass_pose_pct": own})
    finally:
        O.unload()
    return out, notes


def key_pose(tool: Path, im, joints: dict, scratch: Path) -> float | None:
    """How far a picture's skeleton (Apple Vision) is from the given joints, % of the width; None if none found."""
    one = scratch / ".pose"  # beside the run's control.json, as bin/bodypose's stand-in in the tests reads it
    if one.exists():
        shutil.rmtree(one)
    one.mkdir()
    try:
        im.save(one / "0000.png")
        got = SK.read_frames(tool, one, one / "pose.json")
    finally:
        shutil.rmtree(one, ignore_errors=True)
    if not got or not got[0]:
        return None
    res = SK.pose_error([joints], got[:1])
    return res["mean_joint_error_pct"] if res["frames_compared"] else None


# ------------------------------------------------------------------ after LTX

def split_frames(clip: Path, dest: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AnimateError("ffmpeg is not installed")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(clip), "-start_number", "0", str(dest / "%04d.png")],
                   check=True)
    files = sorted(dest.glob("*.png"))
    if not files:
        raise AnimateError("the clip has no frames")
    return files


def lift_tool(root: Path) -> Path | None:
    for p in (root / "bin" / "lift-subject", HERE.parent / "bin" / "lift-subject"):
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def cut_out(frames: list[Path], dest: Path, lift: Path | None) -> tuple[list[Path], str]:
    """Each frame cut from its ground: Apple Vision's foreground mask when bin/lift-subject is there, else the
    frame's own ground colour (its border's) keyed out: a soft edge, then specks dropped."""
    from PIL import Image

    from flat2d.atlas import unmix
    from flat2d.parts import ground_key

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    how = "vision" if lift else "key"
    out = []
    for f in frames:
        target = dest / f.name
        done = False
        if lift:
            r = subprocess.run([str(lift), str(f), str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            done = r.returncode == 0 and target.is_file()
            if not done:
                how = "vision+key"
        if not done:  # each frame's own ground: the model drifts it a little from the guide's
            with Image.open(f) as im:
                rgb = im.convert("RGB")
                unmix(rgb, tuple(ground_key(rgb)["rgb"])).save(target)
        out.append(target)
    return out, how


# A cut-out has lost part of the character when, against the median of the frames around it (LOST_SPAN each side),
# its area is below LOST_AREA of theirs, its height below LOST_HEIGHT of theirs, or its lowest point sits more than
# LOST_FEET of the frame's height above theirs (the feet gone). Apple Vision drops motion-blurred legs this way.
LOST_SPAN = 4
LOST_AREA = 0.8
LOST_HEIGHT = 0.9
LOST_FEET = 0.05
# The repair: the ground behind the character, as the median of the frames around it (PLATE_SPAN each side) where
# their own cut-outs are not (grown by PLATE_GROW px), and whatever differs from it by more than PLATE_HOLD
# (summed over RGB) is the character, fully so from PLATE_HOLD + PLATE_FEATHER.
PLATE_SPAN = 8
PLATE_GROW = 15
PLATE_HOLD = 30
PLATE_FEATHER = 45
PLATE_LIMB = 0.03
PLATE_EDGE = 3


def _shape(path: Path):
    """(area, top, bottom) of a cut-out's character, in pixels; area 0 when it is empty."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        a = np.asarray(im.convert("RGBA").getchannel("A")) > 24
    rows = np.where(a.any(1))[0]
    if not len(rows):
        return 0, 0, 0
    return int(a.sum()), int(rows.min()), int(rows.max()) + 1


def lost_frames(shapes: list[tuple[int, int, int]], frame_h: int) -> dict[int, str]:
    """The frames whose cut-out lost much of the character, by index, each with why, judged against the median of
    the frames around it (see LOST_*). Needs at least three frames each side to judge."""
    import statistics

    lost = {}
    for i, (area, top, bottom) in enumerate(shapes):
        near = [shapes[j] for j in range(max(0, i - LOST_SPAN), min(len(shapes), i + LOST_SPAN + 1))
                if j != i and shapes[j][0] > 0]
        if len(near) < 3:
            continue
        m_area = statistics.median(s[0] for s in near)
        m_h = statistics.median(s[2] - s[1] for s in near)
        m_bottom = statistics.median(s[2] for s in near)
        if area < LOST_AREA * m_area:
            lost[i] = f"area {area / m_area:.0%} of its neighbours'"
        elif bottom - top < LOST_HEIGHT * m_h:
            lost[i] = f"height {(bottom - top) / m_h:.0%} of its neighbours'"
        elif m_bottom - bottom > LOST_FEET * frame_h:
            lost[i] = f"feet {(m_bottom - bottom) / frame_h:.0%} of the frame above its neighbours'"
    return lost


def repair_cutout(frames: list[Path], cutouts: list[Path], i: int, out: Path) -> None:
    """Redo frame i's cut-out against the ground behind it: the camera is still, so the frames around it show the
    ground where the character is not; the median of those is a picture of the ground alone, and what differs from
    it is the character. The result keeps the first cut-out and adds what it lost, within the box the character
    fills in the frames around it."""
    import numpy as np
    from PIL import Image, ImageFilter

    near = [j for j in range(max(0, i - PLATE_SPAN), min(len(frames), i + PLATE_SPAN + 1)) if j != i]
    with Image.open(cutouts[i]) as im:
        first = np.asarray(im.convert("RGBA").getchannel("A"))
    h, w = first.shape
    box = [w, h, 0, 0]
    masks = []
    for j in near:
        with Image.open(cutouts[j]) as im:
            a = im.convert("RGBA").getchannel("A").point(lambda v: 255 if v > 24 else 0)
        b = a.getbbox()
        if b:
            box = [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
        masks.append(np.asarray(a.filter(ImageFilter.MaxFilter(PLATE_GROW))) > 0)
    if box[2] <= box[0]:
        return
    pad = PLATE_GROW
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad)
    stack = np.empty((len(near), y1 - y0, x1 - x0, 3), np.float32)
    for k, j in enumerate(near):
        with Image.open(frames[j]) as im:
            stack[k] = np.asarray(im.convert("RGB"), np.float32)[y0:y1, x0:x1]
        stack[k][masks[k][y0:y1, x0:x1]] = np.nan
    with warnings.catch_warnings():  # a pixel the character covers in every frame around has no ground: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        plate = np.nanmedian(stack, 0)
    with Image.open(frames[i]) as im:
        rgb = np.asarray(im.convert("RGB"))
    d = np.nan_to_num(np.abs(rgb[y0:y1, x0:x1].astype(np.float32) - plate).sum(2), nan=0.0)
    add = np.zeros_like(first)
    add[y0:y1, x0:x1] = (np.clip((d - PLATE_HOLD) / PLATE_FEATHER, 0, 1) * 255).astype(np.uint8)
    alpha = np.maximum(first, add)
    # What the ground plate adds and does not touch the first cut-out is floor (shadow, flecks), unless it is big
    # enough to be a limb of its own (PLATE_LIMB of the first cut-out's area).
    from scipy import ndimage

    # Judged on the solid part (alpha over half), so a faint shadow does not join a fleck to a boot; the faint
    # edge is kept only within PLATE_EDGE px of what is kept.
    body = first > 24
    lab, _ = ndimage.label(alpha > 127)
    sizes = np.bincount(lab.ravel())
    keep = (lab > 0) & (np.isin(lab, np.unique(lab[body])) | (sizes[lab] >= PLATE_LIMB * max(1, int(body.sum()))))
    keep = ndimage.binary_dilation(keep, iterations=PLATE_EDGE) | body
    alpha = np.where(keep, alpha, 0).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(out)


def check_cutouts(frames: list[Path], cutouts: list[Path]) -> list[dict]:
    """Find the cut-outs that lost much of the character (lost_frames), redo each against the ground behind it
    (repair_cutout) and keep the redo when it no longer looks lost; a frame still lost after that is marked dropped,
    and the strip leaves it out. Returns one row per lost frame: its number, why, and repaired or dropped."""
    from PIL import Image

    if len(cutouts) < 7:
        return []
    with Image.open(cutouts[0]) as im:
        frame_h = im.size[1]
    shapes = [_shape(p) for p in cutouts]
    lost = lost_frames(shapes, frame_h)
    rows = []
    for i, why in sorted(lost.items()):
        redo = cutouts[i].with_suffix(".redo.png")
        repair_cutout(frames, cutouts, i, redo)
        fixed = False
        if redo.is_file():
            trial = list(shapes)
            trial[i] = _shape(redo)
            fixed = i not in lost_frames(trial, frame_h)
            if fixed:
                redo.replace(cutouts[i])
                shapes[i] = trial[i]
            else:
                redo.unlink()
        rows.append({"frame": int(cutouts[i].stem), "why": why, "fix": "repaired" if fixed else "dropped"})
    return rows


def strip_cells(cutouts: list[Path], every: int, cell_h: int, margin: int = 4, dropped: set[int] = frozenset()):
    """Every every'th cut-out after frame 0 (frame 0 is the guide picture, not a pose), all cropped to the one box
    that holds the character in every one of them, so the strip's frames line up, then scaled to cell_h tall.
    A frame in dropped (its cut-out lost the character, check_cutouts) is left out of the strip."""
    from PIL import Image

    picks = [p for p in cutouts[1::every] if int(p.stem) not in dropped] or cutouts[:1]
    ims = [Image.open(p).convert("RGBA") for p in picks]
    box = None
    for im in ims:
        b = im.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
        if b:
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    w, h = ims[0].size
    if box is None:
        box = (0, 0, w, h)
    box = (max(0, box[0] - margin), max(0, box[1] - margin), min(w, box[2] + margin), min(h, box[3] + margin))
    bw, bh = box[2] - box[0], box[3] - box[1]
    s = min(1.0, cell_h / bh)
    size = (max(1, round(bw * s)), max(1, round(bh * s)))
    cells = [im.crop(box).resize(size, Image.LANCZOS) for im in ims]
    return cells, [int(p.stem) for p in picks], list(box)


def bodypose_tool(root: Path) -> Path | None:
    env = os.environ.get("PIPELINE_BODYPOSE")
    for p in ([Path(env)] if env else []) + [root / "bin" / "bodypose", HERE.parent / "bin" / "bodypose"]:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def check_pose(tool: Path, frames_dir: Path, control: list[dict], out: Path) -> dict:
    """The result's skeleton (Apple Vision) against the control, frame by frame from frame 1."""
    got = SK.read_frames(tool, frames_dir, out)
    res = SK.pose_error(control, got, skip=1)
    res["frames_with_skeleton"] = sum(1 for j in got if j)
    return res


def fit_control(root: Path, control: list[dict], work: Path) -> tuple[list[dict], dict]:
    """The skeleton refitted to the guide picture's own body (skeleton.refit: its hips, shoulders, limb lengths
    and feet, read by Apple Vision), so a character built unlike the skeleton's does not start off it. Without
    bin/bodypose, or when Vision reads no neck, hips and feet in the picture, the skeleton is used as shipped and
    the returned note says why."""
    tool = bodypose_tool(root)
    if not tool:
        print("animate: no bin/bodypose; the skeleton is used as shipped, not fitted to the picture", flush=True)
        return control, {"fitted": False, "why": "no bin/bodypose"}
    target = SK.read_picture(tool, work / "guide.png", work)
    try:
        fitted, info = SK.refit(control, target)
    except SK.SkeletonError as e:
        print(f"animate: {e}; the skeleton is used as shipped", flush=True)
        return control, {"fitted": False, "why": str(e)}
    print(f"animate: skeleton fitted to the picture (x{info['scale']}; " +
          ", ".join(f"{p} x{r}" for p, r in info["ratios"].items()) + ")", flush=True)
    return fitted, {"fitted": True, **info}


# ------------------------------------------------------------------ the run

def animate(root: Path, stem: str, *, prompt: str, skeleton: str = "walk_side", action: str = "walk",
            picture: str | None = None, seed: int = 42, frames: int = 73, size: int = 704, every: int = 3,
            cell: int = 256, skeleton_fps: float = FPS, check: bool = True, save: bool = True,
            side: str | None = None, side_frame: int | None = None, side_strength: float = 0.8,
            remove: str | None = None, remove_at: str | None = None, keep: str | None = None) -> dict:
    from flat2d.atlas import action_name

    import flow

    root = Path(root).resolve()
    stem = check_stem(stem)
    action = action_name(action)
    folder = stem_dir(root, stem)
    if not folder.is_dir():
        raise AnimateError(f"no asset {stem!r}")
    prompt = " ".join(str(prompt or "").split())
    if not prompt:
        raise AnimateError("a prompt is needed, and it must say which way the character faces and moves")
    if frames < 9 or (frames - 1) % 8:
        raise AnimateError("frames must be 8k+1 (9, 17, ... 73, ... 121)")
    if size % 32 or not 256 <= size <= 1024:
        raise AnimateError("size must be a multiple of 32 from 256 to 1024")
    if not 1 <= every <= 12:
        raise AnimateError("every must be 1 to 12")
    if side_frame is None:
        side_frame = min(FPS, frames - 1)
    if side and not 1 <= side_frame <= frames - 1:
        raise AnimateError(f"side-frame must be 1 to {frames - 1} (frame 0 is the first picture)")
    if side and not 0 < side_strength <= 1:
        raise AnimateError("side-strength must be above 0 and at most 1")
    remove = " ".join(str(remove or "").split()) or None
    keep = " ".join(str(keep or "").split()) or None
    if remove and len(remove) > 120:
        raise AnimateError("remove is too long: name the kit to erase in a few words, like backpack")
    if keep and not remove:
        raise AnimateError("keep goes with remove: it names the kit that stays when the removed kit is erased")
    if keep and len(keep) > 200:
        raise AnimateError("keep is too long: name the kit that stays in a few words")
    picks = remove_frames(frames, remove_at) if remove else []
    if remove:
        from flat2d.sheet import edit_backend

        if not edit_backend():
            raise AnimateError("--remove needs the photo-edit model (Qwen-Image-Edit, mflux), which is not set up")
    paths = ltx_paths(root)
    check_ltx(paths)
    pic = pick_picture(folder, picture)
    side_pic = pick_picture(folder, side) if side else None
    control_src = SK.named(skeleton)

    print("STAGES " + " ".join(REMOVE_STAGES if remove else STAGES), flush=True)
    t0 = time.time()
    if save:  # what was there is kept as a Version before this run changes the 2D files
        flow.autosave(root, stem, ("flat",))
    # Built in a scratch folder outside the 2D files and moved into flat/anim/<action>/ only when it all worked, so
    # a failed or cancelled run leaves the 2D files as the picked Version has them.
    work = folder / ".animate" / action
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        doc = _run(root, stem, folder, work, paths, pic, control_src, side_pic=side_pic, side_frame=side_frame,
                   side_strength=side_strength, remove=remove, keep=keep, picks=picks, prompt=prompt, action=action,
                   seed=seed,
                   frames=frames, size=size, every=every, cell=cell, skeleton_fps=skeleton_fps, check=check, t0=t0)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    if save:
        flow.snapshot(root, stem, "flat", note=f"Animate: {action} ({doc['skeleton']})", made_from=("concept",))
    print(f"animate: {stem} {action}: {doc['frames']} frames, strip of {len(doc['strip']['from_frames'])} in "
          f"{doc['seconds']} s", flush=True)
    return doc


def _run(root: Path, stem: str, folder: Path, work: Path, paths: dict[str, Path], pic: Path, control_src: Path, *,
         side_pic: Path | None, side_frame: int, side_strength: float, remove: str | None, keep: str | None,
         picks: list[int],
         prompt: str, action: str, seed: int,
         frames: int, size: int, every: int, cell: int, skeleton_fps: float, check: bool, t0: float) -> dict:
    from flat2d.atlas import save_cells
    from flat2d.store import write_center

    stage("guide", "Fit the picture")
    ground = make_guide(pic, size, work / "guide.png")
    side_fitted = None
    if side_pic is not None:
        side_fitted = fit_side(side_pic, work / "guide.png", size, work / "side.png", tuple(ground), lift_tool(root))

    stage("skeleton", "Fit and draw the skeleton video")
    control = SK.fit_frames(SK.retime(SK.load(control_src), skeleton_fps, FPS, frames)
                            if skeleton_fps != FPS else SK.load(control_src), frames)
    control, fit = fit_control(root, control, work)
    (work / "control.json").write_text(json.dumps({"fps": FPS, "fit": fit,
                                                   "frames": [{"joints2d": j} for j in control]}) + "\n")
    SK.draw_video(control, size, size, FPS, work / "skeleton.mp4")

    keys: list[tuple[Path, int]] = []
    t_ltx = time.time()
    if remove:
        stage("first_pass", "First pass with LTX-2.3")
        first = work / "first_clip.mp4"
        argv = ltx_argv(paths, prompt=prompt, out=first, control=work / "skeleton.mp4", guide=work / "guide.png",
                        size=size, frames=frames, seed=seed)
        rc = subprocess.run(argv, env=ltx_env(), cwd=str(root), stdin=subprocess.DEVNULL).returncode
        if rc != 0 or not first.is_file():
            raise AnimateError(f"LTX-2.3 stopped (exit {rc}) without a first-pass clip")
        every_frame = split_frames(first, work / "first_frames")
        (work / "first").mkdir()
        sources = []
        for f in picks:
            if f >= len(every_frame):
                raise AnimateError(f"the first pass has no frame {f}")
            sources.append(work / "first" / f"{f:04d}.png")
            shutil.copy(every_frame[f], sources[-1])
        shutil.rmtree(work / "first_frames")
        first.unlink()
        ltx_s = time.time() - t_ltx

        stage("erase", f"Erase the {remove}")
        t_erase = time.time()
        erased, key_notes = erase_kit(sources, picks, erase_prompt(remove, keep), work / "keys", seed, control,
                                      bodypose_tool(root) if check else None)
        for n in key_notes:
            print(f"animate: frame {n['frame']} erased (seed {n['seed']}), skeleton within {n['pose_pct']}% "
                  f"(first pass {n['first_pass_pose_pct']}%)", flush=True)
        erase_s = round(time.time() - t_erase, 1)
        keys = list(zip(erased, picks))
        t_ltx = time.time()
    else:
        ltx_s = 0.0

    stage("ltx", "Animate with LTX-2.3")
    clip = work / "clip.mp4"
    clip.unlink(missing_ok=True)
    argv = ltx_argv(paths, prompt=prompt, out=clip, control=work / "skeleton.mp4", guide=work / "guide.png",
                    size=size, frames=frames, seed=seed, side=work / "side.png" if side_pic else None,
                    side_frame=side_frame, side_strength=side_strength, keys=keys)
    rc = subprocess.run(argv, env=ltx_env(), cwd=str(root), stdin=subprocess.DEVNULL).returncode
    if rc != 0 or not clip.is_file():
        raise AnimateError(f"LTX-2.3 stopped (exit {rc}) without a clip")
    ltx_s = round(ltx_s + time.time() - t_ltx, 1)

    stage("frames", "Cut the frames")
    frame_files = split_frames(clip, work / "frames")

    stage("cutouts", "Cut out the character")
    cut_files, how = cut_out(frame_files, work / "cutouts", lift_tool(root))
    lost = check_cutouts(frame_files, cut_files)
    for row in lost:
        print(f"animate: frame {row['frame']} cut-out lost the character ({row['why']}): {row['fix']}", flush=True)

    stage("strip", "Pack the sprite strip")
    cells, picked, box = strip_cells(cut_files, every, cell,
                                     dropped={r["frame"] for r in lost if r["fix"] == "dropped"})
    strip_fps = max(1, round(FPS / every))

    stage("pose_check", "Check the skeleton")
    pose = None
    tool = bodypose_tool(root) if check else None
    if tool:
        pose = check_pose(tool, work / "frames", control, work / "pose.json")
        print(f"animate: skeleton within {pose['mean_joint_error_pct']}% of the control "
              f"(moving joints {pose['mean_limb_error_pct']}%)", flush=True)
    else:
        print("animate: no bin/bodypose; the skeleton check was skipped", flush=True)

    doc = {
        "maker": "animate",
        "model": MODEL_NOTE,
        "licence": LICENCE,
        "machine_generated": True,
        "disclosure": DISCLOSURE,
        "picture": pic.relative_to(folder).as_posix(),
        "side": {"picture": side_pic.relative_to(folder).as_posix(), "frame": side_frame,
                 "strength": side_strength, "fitted_to_figure": side_fitted} if side_pic else None,
        "remove": {"what": remove, "keep": keep, "frames": picks, "strength": KEY_STRENGTH,
                   "edit_model": "Qwen-Image-Edit-2511", "prompt": erase_prompt(remove, keep), "seconds": erase_s,
                   "keys": key_notes} if remove else None,
        "skeleton": control_src.stem if control_src.parent == SK.SKELETONS else str(control_src),
        "skeleton_fit": fit,
        "prompt": prompt,
        "seed": seed,
        "frames": len(frame_files),
        "fps": FPS,
        "size": size,
        "cutouts": how,
        "lost_cutouts": lost,
        "strip": {"file": f"sheets/{action}.png", "every": every, "from_frames": picked, "box": box,
                  "cell": {"w": cells[0].size[0], "h": cells[0].size[1]}, "fps": strip_fps},
        "pose_check": {k: v for k, v in pose.items() if k != "per_frame_pct"} if pose else None,
        "ltx_seconds": ltx_s,
        "seconds": round(time.time() - t0, 1),
        "made": int(time.time()),
    }
    (work / "anim.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    final = folder / "flat" / "anim" / action
    if final.exists():
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work), str(final))
    row = save_cells(root, stem, action, cells, strip_fps, made_by="animate", anim=f"anim/{action}/anim.json",
                     from_frames=picked)
    write_center(root, stem, row["sheet"])
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--stem", required=True)
    ap.add_argument("--prompt", required=True, help="what happens, and which way the character faces")
    ap.add_argument("--skeleton", default="walk_side", help=f"one of {', '.join(SK.shipped())}, or a .json path")
    ap.add_argument("--skeleton-fps", type=float, default=FPS)
    ap.add_argument("--action", default="walk")
    ap.add_argument("--picture", default=None, help="a picture under out/<stem>/ (default: the concept still)")
    ap.add_argument("--side", default=None,
                    help="a side view under out/<stem>/, facing the way the character moves: a second guide")
    ap.add_argument("--side-frame", type=int, default=None, help=f"the frame the side view guides (default {FPS})")
    ap.add_argument("--side-strength", type=float, default=0.8, help="how hard the side view pulls, 0 to 1")
    ap.add_argument("--remove", default=None,
                    help="kit the model invents, to erase (like backpack): a first pass, then its frames erased and "
                         "used as guides for a second")
    ap.add_argument("--keep", default=None,
                    help="with --remove: kit that must stay (like chest rig, radio and pouches)")
    ap.add_argument("--remove-frames", default=None,
                    help=f"the frames erased and used as guides (default every {FPS} from {FPS}, and the last)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=73)
    ap.add_argument("--size", type=int, default=704)
    ap.add_argument("--every", type=int, default=3, help="every n'th frame goes in the sprite strip")
    ap.add_argument("--cell", type=int, default=256, help="the strip's frame height at most, in pixels")
    ap.add_argument("--no-check", action="store_true", help="skip reading the result's skeleton back")
    a = ap.parse_args(argv)
    try:
        animate(Path(a.root), a.stem, prompt=a.prompt, skeleton=a.skeleton, action=a.action, picture=a.picture,
                seed=a.seed, frames=a.frames, size=a.size, every=a.every, cell=a.cell, skeleton_fps=a.skeleton_fps,
                check=not a.no_check, side=a.side, side_frame=a.side_frame, side_strength=a.side_strength,
                remove=a.remove, remove_at=a.remove_frames, keep=a.keep)
    except (AnimateError, SK.SkeletonError, ValueError) as e:
        print(f"animate: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
