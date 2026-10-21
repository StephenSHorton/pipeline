"""Names, mechanics sheet, and the sound-prompt line. No model calls."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PART_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
ROLES = ("hull", "leg", "turret", "weapon", "other")


def flat_dir(root: Path, stem: str) -> Path:
    return Path(root) / "out" / stem / "flat"


def slug_id(name: str, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    if PART_ID.match(text):
        return text
    if PART_ID.match(fallback):
        return fallback
    raise ValueError(f"invalid part id {fallback!r}")


def sound_file(part_id: str) -> str:
    if not PART_ID.match(part_id):
        raise ValueError(f"invalid part id {part_id!r}")
    return part_id + ".wav"


SPEAKER = re.compile(r"^[a-z]{2}_[a-z0-9_]{1,24}$")
DEFAULT_SPEAKER = "af_heart"


def voice_file(part_id: str) -> str:
    if not PART_ID.match(part_id):
        raise ValueError(f"invalid part id {part_id!r}")
    return part_id + ".wav"


def speaker_name(value: str) -> str:
    text = (value or "").strip() or DEFAULT_SPEAKER
    if not SPEAKER.match(text):
        raise ValueError(f"invalid speaker {value!r}")
    return text


def sound_prompt(sentence: str) -> str:
    text = " ".join((sentence or "").split())
    if not text:
        raise ValueError("empty sound sentence")
    return "short game sound effect, one shot, no music: " + text


def unique_ids(rows: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    out: list[dict] = []
    for i, row in enumerate(rows):
        fallback = f"part-{i + 1:02d}"
        base = slug_id(str(row.get("name") or ""), str(row.get("id") or fallback))
        n = seen.get(base, 0) + 1
        seen[base] = n
        pid = base if n == 1 else slug_id(f"{base}-{n}", fallback)
        role = str(row.get("role") or "other")
        if role not in ROLES:
            role = "other"
        try:
            speaker = speaker_name(str(row.get("speaker") or ""))
        except ValueError:
            speaker = DEFAULT_SPEAKER
        out.append(
            {
                "id": pid,
                "file": str(row.get("file") or f"parts/part-{i + 1:02d}.png"),
                "name": str(row.get("name") or pid),
                "role": role,
                "motion": str(row.get("motion") or ""),
                "projectile": str(row.get("projectile") or ""),
                "sound": str(row.get("sound") or ""),
                "line": str(row.get("line") or ""),
                "speaker": speaker,
                "voice": str(row.get("voice") or ""),
            }
        )
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def weights_root() -> Path:
    env = os.environ.get("FLAT2D_WEIGHTS")
    return Path(env) if env else repo_root() / "weights" / "flat2d"


def lift_binary() -> Path | None:
    """bin/lift-subject: Apple Vision subject masks (Mac). The 3D lane builds it too."""
    env = os.environ.get("FLAT2D_LIFT")
    path = Path(env) if env else repo_root() / "bin" / "lift-subject"
    return path if path.is_file() and os.access(path, os.X_OK) else None


def sam_dir() -> Path:
    env = os.environ.get("FLAT2D_SAM_DIR")
    return Path(env) if env else weights_root() / "sam2"


def sfx_dir() -> Path:
    env = os.environ.get("FLAT2D_SFX_DIR")
    return Path(env) if env else weights_root() / "stable-audio-3-small-sfx"


def sfx_cmd() -> str:
    return (os.environ.get("FLAT2D_SFX_CMD") or "").strip()


def sfx_mode() -> str:
    """stub writes a stand-in wav. no-runner has weights and no command. ready can render."""
    if not weights_ready(sfx_dir()):
        return "stub"
    if not sfx_cmd():
        return "no-runner"
    return "ready"


def sam_checkpoint() -> Path | None:
    folder = sam_dir()
    if not folder.is_dir():
        return None
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        found = sorted(folder.glob(pattern))
        if found:
            return found[0]
    return None


def weights_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model_index.json").is_file():
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("*.pt")) or any(path.glob("*.ckpt"))


def llm_url() -> str:
    return (os.environ.get("FLAT2D_LLM_URL") or "").strip().rstrip("/")


def mflux_ready() -> bool:
    import importlib.util

    return importlib.util.find_spec("mflux") is not None


def qwen_ready() -> bool:
    """mflux 0.20 and later ship Qwen-Image 2.1. Checked on disk so mlx is not imported."""
    import importlib.util

    spec = importlib.util.find_spec("mflux")
    if spec is None or not spec.submodule_search_locations:
        return False
    return any((Path(p) / "models" / "qwen21").is_dir() for p in spec.submodule_search_locations)


def still_choice() -> str:
    return (os.environ.get("FLAT2D_STILL") or "qwen-image").strip().lower()


def qwen_repo() -> str:
    return (os.environ.get("FLAT2D_QWEN") or "Qwen/Qwen-Image-2.1-Turbo").strip()


def still_backend() -> str:
    """Qwen-Image 2.1 Turbo on Apple MLX, or Z-Image Turbo when FLAT2D_STILL=z-image
    or mflux predates Qwen-Image 2.1. Empty when mflux is not installed."""
    if not mflux_ready():
        return ""
    if still_choice() != "z-image" and qwen_ready():
        return "qwen-image"
    return "z-image"


def voice_ready() -> bool:
    import importlib.util

    return importlib.util.find_spec("mlx_audio") is not None


def kokoro_repo() -> str:
    return (os.environ.get("FLAT2D_KOKORO") or "mlx-community/Kokoro-82M-bf16").strip()


def backends() -> dict:
    return {
        "still": still_backend(),
        "sfx": weights_ready(sfx_dir()),
        "sfx_mode": sfx_mode(),
        "llm": bool(llm_url()),
        "voice": voice_ready(),
        "kokoro": kokoro_repo(),
        "speaker": DEFAULT_SPEAKER,
    }
