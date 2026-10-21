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
    return _mflux_has("qwen21")


def q2512_ready() -> bool:
    """mflux's Qwen-Image (2512) txt2img. Checked on disk so mlx is not imported."""
    return _mflux_has("qwen")


def _mflux_has(model_dir: str) -> bool:
    import importlib.util

    spec = importlib.util.find_spec("mflux")
    if spec is None or not spec.submodule_search_locations:
        return False
    return any((Path(p) / "models" / model_dir).is_dir() for p in spec.submodule_search_locations)


# The draw models. Only Apache-2.0 weights are on the default and fallback path; Qwen-Image 2.1 Turbo is under the
# Qwen RESEARCH licence (non-commercial only) and is reachable only with FLAT2D_ALLOW_NC=1, marked degraded.
STILLS = {
    "qwen-2512": {"model": "qwen-image-2512+lightning8", "licence": "apache-2.0"},
    "z-image": {"model": "z-image-turbo", "licence": "apache-2.0"},
    "qwen-image": {"model": "qwen-image-2.1-turbo", "licence": "qwen-research (non-commercial)"},
    "stub": {"model": "stub", "licence": "none"},
}
DEFAULT_STILL = "qwen-2512"
NC_STILLS = {"qwen-image"}
Q2512_STEPS = 8


def still_choice() -> str:
    raw = (os.environ.get("FLAT2D_STILL") or DEFAULT_STILL).strip().lower()
    return {"qwen-2.1": "qwen-image", "qwen-image-2.1": "qwen-image", "qwen2512": "qwen-2512",
            "qwen-image-2512": "qwen-2512", "zimage": "z-image"}.get(raw, raw)


def allow_nc() -> bool:
    return (os.environ.get("FLAT2D_ALLOW_NC") or "").strip().lower() in {"1", "yes", "true", "on"}


def qwen_repo() -> str:
    return (os.environ.get("FLAT2D_QWEN") or "Qwen/Qwen-Image-2.1-Turbo").strip()


def q2512_repo() -> str:
    return (os.environ.get("FLAT2D_QWEN2512") or "Qwen/Qwen-Image-2512").strip()


def q2512_steps() -> int:
    try:
        n = int(os.environ.get("FLAT2D_Q2512_STEPS") or Q2512_STEPS)
    except ValueError:
        n = Q2512_STEPS
    return 4 if n <= 4 else 8


def q2512_lora() -> str:
    """The lightx2v Lightning LoRA for the step count (4 or 8), from the Hugging Face cache (downloaded once)."""
    env = (os.environ.get("FLAT2D_Q2512_LORA") or "").strip()
    if env:
        return env
    from huggingface_hub import hf_hub_download

    return hf_hub_download("lightx2v/Qwen-Image-2512-Lightning",
                           f"Qwen-Image-2512-Lightning-{q2512_steps()}steps-V1.0-bf16.safetensors")


def q2512_saved_dir() -> Path:
    return weights_root() / "qwen-image-2512-q8"


def q2512_saved() -> Path | None:
    """The pre-saved 8-bit base model (flat2d/optional.save_qwen2512), when it is complete."""
    path = q2512_saved_dir()
    return path if (path / "transformer").is_dir() and (path / "text_encoder").is_dir() and (path / "vae").is_dir() \
        else None


def still_backend() -> str:
    """The draw model: Qwen-Image-2512 + Lightning (Apache-2.0) on Apple MLX by default, Z-Image Turbo (Apache-2.0)
    when FLAT2D_STILL=z-image or mflux has no Qwen-Image. Qwen-Image 2.1 Turbo (non-commercial) only when
    FLAT2D_STILL=qwen-image AND FLAT2D_ALLOW_NC=1; otherwise asking for it gets the default.
    Empty when mflux is not installed. FLAT2D_STILL=stub is a fake model for tests (optional.stub_still)."""
    choice = still_choice()
    if choice == "stub":
        return "stub"
    if not mflux_ready():
        return ""
    if choice in NC_STILLS:
        if allow_nc() and qwen_ready():
            return "qwen-image"
        choice = DEFAULT_STILL
    if choice == "z-image":
        return "z-image"
    return "qwen-2512" if q2512_ready() else "z-image"


# The four-view sheet's size (plan 1.1, picked by the A/B in scratchpad qmv/p2/ab.md): the largest that keeps a warm
# sheet within 75 s. A square size is a 2x2 grid, a wide one a strip. FLAT2D_SHEET=WxH overrides it.
SHEET_SIZE = (1152, 1152)
SHEET_REDRAWS = 1


def sheet_size() -> tuple[int, int, str]:
    """(width, height, layout) of the four-view sheet; sizes are rounded down to a multiple of 16."""
    raw = (os.environ.get("FLAT2D_SHEET") or "").lower().replace(" ", "")
    w, h = SHEET_SIZE
    if "x" in raw:
        try:
            w, h = (max(256, int(v)) for v in raw.split("x", 1))
        except ValueError:
            w, h = SHEET_SIZE
    w, h = w // 16 * 16, h // 16 * 16
    return w, h, "grid" if w == h else "strip"


def still_info(backend: str | None = None) -> dict:
    """{backend, model, licence} for views.json and /2d/status."""
    b = still_backend() if backend is None else backend
    return {"backend": b, **STILLS.get(b, {"model": b or "none", "licence": "unknown"})}


def still_degraded(backend: str | None = None) -> list[dict[str, str]]:
    """Degraded entries for a draw on this backend: any backend but the default is one (a stub is a test, not a
    fallback), and a non-commercial model is 'licence nc'."""
    from factory_env import degraded_entry

    b = still_backend() if backend is None else backend
    out: list[dict[str, str]] = []
    if b in ("", "stub") or b == DEFAULT_STILL:
        return out
    info = STILLS.get(b, {"model": b, "licence": "unknown"})
    out.append(degraded_entry("draw", "backend", f"drawn with {info['model']}, not {STILLS[DEFAULT_STILL]['model']}"))
    if b in NC_STILLS:
        out.append(degraded_entry("licence", "nc", f"{info['model']} is under the {info['licence']} licence "
                                  "(FLAT2D_ALLOW_NC=1); not for commercial use"))
    return out


def voice_ready() -> bool:
    import importlib.util

    return importlib.util.find_spec("mlx_audio") is not None


def kokoro_repo() -> str:
    return (os.environ.get("FLAT2D_KOKORO") or "mlx-community/Kokoro-82M-bf16").strip()


def backends() -> dict:
    return {
        "still": still_backend(),
        "still_licence": still_info()["licence"],
        "sfx": weights_ready(sfx_dir()),
        "sfx_mode": sfx_mode(),
        "llm": bool(llm_url()),
        "voice": voice_ready(),
        "kokoro": kokoro_repo(),
        "speaker": DEFAULT_SPEAKER,
    }
