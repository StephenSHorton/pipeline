"""One order for Pieces, Motion, Say, and Sound. The page does not keep its own sentences."""
from __future__ import annotations

import re
from pathlib import Path

from flat2d.atlas import pack_request
from flat2d.optional import render_sound, speak
from flat2d.sheet import sfx_mode, voice_ready
from flat2d.store import split_source

_LABEL = {"pieces": "Pieces", "motion": "Motion", "say": "Say", "sound": "Sound"}


def action_slug(text: str) -> str:
    raw = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")[:40].strip("-")
    return slug or "sheet"


def _text(value) -> str:
    return " ".join(str(value or "").split())


def _has_clip(payload: dict) -> bool:
    video = payload.get("video")
    if isinstance(video, str) and video.strip():
        return True
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return False
    return any(isinstance(item, str) and item.strip() for item in frames)


def _pieces_note(result: dict) -> str:
    parts = result.get("parts") or []
    if len(parts) >= 2:
        return f"Pieces: {len(parts)}."
    reason = result.get("reason") or ""
    if (reason == "whole" or result.get("whole")) and len(parts) == 1:
        return "Pieces: one shape, saved as its cutout."
    if reason == "whole" or result.get("whole"):
        return "Pieces: one shape, left whole."
    if reason == "keyed" or result.get("keyed"):
        return "Pieces: the costume is the key color, so it was not saved."
    if reason == "specks":
        return "Pieces: nothing saved. Marks off the key were smaller than 24 pixels."
    return "Pieces: nothing separate was saved."


def _motion_plan(when: str, payload: dict) -> tuple[str | None, str]:
    """Action to pack, or None when Motion is skipped. Empty home name stays idle."""
    line = _text(payload.get("motion"))
    if when == "home":
        if not _has_clip(payload):
            return None, "Motion: attach a clip. The frames are not drawn from the sentence."
        return ("idle" if not line else action_slug(line)), ""
    if not line:
        return None, "Name the motion first."
    if not _has_clip(payload):
        return None, "Attach a clip of that motion. The prompt is its name."
    return action_slug(line), ""


def apply_steps(root: Path, stem: str, payload: dict) -> dict:
    when = payload.get("when") or "revise"
    steps = payload.get("steps")
    if not isinstance(steps, list):
        steps = []
    raw_part = payload.get("part")
    if when == "home" or raw_part is None or not str(raw_part).strip():
        part_id = "asset"
    else:
        part_id = str(raw_part).strip()
    notes: list[str] = []
    ran: list[str] = []
    skipped: list[str] = []
    pieces: dict | None = None
    center = ""
    hear = ""
    stub = False

    for step in steps:
        if step not in _LABEL:
            continue
        try:
            if step == "pieces":
                result = split_source(root, stem, payload.get("pieces") or "")
                ran.append(step)
                pieces = result
                notes.append(_pieces_note(result))
                continue
            if step == "motion":
                action, skip = _motion_plan(when, payload)
                if action is None:
                    skipped.append(step)
                    notes.append(skip)
                    continue
                pack = {
                    "action": action,
                    "fps": payload.get("fps") or 10,
                    "count": payload.get("count") or 8,
                }
                video = payload.get("video")
                if isinstance(video, str) and video.strip():
                    pack["video"] = video
                else:
                    pack["frames"] = payload.get("frames") or []
                result = pack_request(root, stem, pack)
                ran.append(step)
                center = str(result.get("sheet") or "")
                notes.append(f"Motion: {result.get('frames')} frames in {result.get('action')}.")
                continue
            if step == "say":
                line = _text(payload.get("say"))
                if not line:
                    skipped.append(step)
                    notes.append(
                        "Say: write the line. A spoken line is not invented."
                        if when == "home"
                        else "Write the line first."
                    )
                    continue
                if not voice_ready():
                    skipped.append(step)
                    notes.append("mlx-audio is not installed")
                    continue
                result = speak(
                    root,
                    stem,
                    {"id": part_id, "line": line, "speaker": str(payload.get("speaker") or "")},
                )
                ran.append(step)
                hear = str(result.get("voice") or "")
                stub = False
                notes.append("Say: spoken.")
                continue
            sentence = _text(payload.get("sound"))
            if when == "home":
                sentence = sentence or _text(payload.get("prompt"))
            if not sentence:
                skipped.append(step)
                notes.append(
                    "Sound: describe the effect, or describe the thing."
                    if when == "home"
                    else "Describe the sound first."
                )
                continue
            if sfx_mode() == "no-runner":
                skipped.append(step)
                notes.append("SFX weights are installed. Set FLAT2D_SFX_CMD.")
                continue
            result = render_sound(root, stem, {"id": part_id, "projectile": sentence})
            ran.append(step)
            hear = str(result.get("sound") or "")
            stub = bool(hear) and bool(result.get("stub"))
            notes.append(
                "Sound: stand-in. The effect runner is not set." if stub else "Sound: written."
            )
        except Exception as err:
            ran.append(step)
            notes.append(f"{_LABEL[step]}: {err}")

    out = {"ok": True, "stem": stem, "notes": notes, "ran": ran, "skipped": skipped}
    if pieces is not None:
        parts = pieces.get("parts") or []
        out["parts"] = parts
        out["pieces"] = len(parts) >= 2
        out["whole"] = bool(pieces.get("whole"))
        out["keyed"] = bool(pieces.get("keyed"))
        out["reason"] = pieces.get("reason") if pieces.get("reason") is not None else ""
    if center:
        out["center"] = center
    out["hear"] = hear
    out["stub"] = stub
    return out
