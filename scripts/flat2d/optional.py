"""Optional local models. Missing weights disable the button. No paid API."""
from __future__ import annotations

import json
import os
import random
import urllib.request
from pathlib import Path

from flat2d.parts import ground_key
from flat2d.sheet import (
    backends,
    kokoro_repo,
    llm_url,
    qwen_repo,
    sfx_cmd,
    sfx_mode,
    sound_file,
    sound_prompt,
    speaker_name,
    still_backend,
    voice_file,
    voice_ready,
    write_json,
)
from flat2d.store import flat_dir


class FlatUnavailable(RuntimeError):
    pass


def parse_llm_json(text: str) -> dict:
    """Accept raw JSON or a fenced block. The local model often wraps the object."""
    body = (text or "").strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        start = body.find("{")
        end = body.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(body[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected object", body, 0)
    return parsed


def draft_mechanics(root: Path, stem: str, note: str, parts: list[dict]) -> dict:
    url = llm_url()
    if not url:
        raise FlatUnavailable("FLAT2D_LLM_URL is not set")
    endpoint = url if url.endswith("/chat/completions") else url + "/v1/chat/completions"
    brief = {
        "note": note,
        "parts": [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "role": p.get("role"),
                "motion": p.get("motion"),
                "projectile": p.get("projectile"),
                "line": p.get("line"),
            }
            for p in parts
        ],
    }
    ask = (
        "Return only JSON. Shape: {\"parts\": [{\"id\", \"role\", \"motion\", \"projectile\", \"line\"}]}. "
        "role is hull, leg, turret, weapon, or other. motion is one sentence. "
        "projectile is one sentence or empty. line is one short spoken bark, or empty. "
        "Use the ids you were given.\n"
        + json.dumps(brief)
    )
    body = json.dumps(
        {
            "model": os.environ.get("FLAT2D_LLM_MODEL") or "local",
            "messages": [{"role": "user", "content": ask}],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    text = raw["choices"][0]["message"]["content"]
    try:
        parsed = parse_llm_json(text)
    except json.JSONDecodeError:
        side = flat_dir(root, stem) / "mechanics.raw.txt"
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(text, encoding="utf-8")
        raise FlatUnavailable("local model did not return JSON")
    by_id = {str(p.get("id")): p for p in parsed.get("parts") or []}
    merged = []
    for p in parts:
        extra = by_id.get(str(p.get("id"))) or {}
        row = dict(p)
        if extra.get("role"):
            row["role"] = extra["role"]
        if extra.get("motion"):
            row["motion"] = extra["motion"]
        if extra.get("projectile"):
            row["projectile"] = extra["projectile"]
        if extra.get("line"):
            row["line"] = extra["line"]
        merged.append(row)
    payload = {"ok": True, "stem": stem, "note": note, "parts": merged, "drafted": True}
    write_json(flat_dir(root, stem) / "mechanics.json", payload)
    return payload


def sound_spec(part: dict) -> tuple[str, str]:
    sentence = (part.get("projectile") or part.get("motion") or "").strip()
    try:
        prompt = sound_prompt(sentence)
    except ValueError as e:
        raise FlatUnavailable("empty sound sentence") from e
    return sound_file(str(part.get("id") or "")), prompt


def still_prompt(text: str) -> str:
    """One object on a flat magenta ground, nothing else. The subject goes first (with it last,
    Qwen-Image dressed "a knight's sword" as a knight), and shadows, floors and effects are
    ruled out one by one: they are what the magenta pull cannot take back out."""
    return (
        text + ". A single object only, centered, floating, isolated on a solid flat magenta #FF00FF background. "
        "Flat even lighting from the front. No shadow, no drop shadow, no cast shadow, no contact shadow, "
        "no reflection, no floor, no ground plane, no glow, no smoke, no particles, no effects, nothing else in the image."
    )


CLEAN_TOL = 45  # a ground noisier than this (see ground_key) is cloudy
CLEAN_SHADOW = 0.005  # share of the picture that may be shadow on the ground
REDRAWS = 2  # new seeds to try when a still comes back with a cloudy ground or a shadow


def ground_flaws(image) -> list[str]:
    """What a still's ground has that the magenta pull cannot take out: "cloudy", "shadow"."""
    import numpy as np

    key = ground_key(image)
    if key["tol"] > CLEAN_TOL:
        return ["cloudy"]
    from flat2d.store import vision_mask

    rgb = np.asarray(image.convert("RGB"), dtype=float)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "still.png"
        image.save(path, format="PNG")
        mask = vision_mask(path)
    if mask is None:
        return []
    subject = np.frombuffer(bytes(mask), np.uint8).reshape(rgb.shape[:2]).astype(bool)
    k = np.array(key["rgb"], dtype=float)
    ground = np.abs(rgb - k).sum(-1) <= key["tol"]
    t = np.clip((rgb @ k) / max(float(k @ k), 1.0), 0.0, 1.0)
    darker_ground = np.abs(rgb - t[..., None] * k).sum(-1) <= 36
    shadow = ~subject & ~ground & darker_ground & (t < 0.85)
    return ["shadow"] if shadow.mean() > CLEAN_SHADOW else []


def clean_still(prompt: str, seed: int):
    """Draw, and redraw on a new seed while the ground is cloudy or holds a shadow.
    When every try has a flaw, the one with the flattest ground is kept.
    Returns the image, the seed it was drawn with, and the flaws the redraws were for."""
    replaced: list[str] = []
    best = None
    for attempt in range(REDRAWS + 1):
        image = still_image(prompt, seed=seed)
        if hasattr(image, "image"):
            image = image.image
        flaws = ground_flaws(image)
        if not flaws:
            return image, seed, replaced
        noise = ground_key(image)["tol"]
        if best is None or noise < best[0]:
            best = (noise, image, seed)
        if attempt < REDRAWS:
            replaced.extend(flaws)
            seed = random.randint(1, 2**31 - 1)
    return best[1], best[2], replaced


_ZIMAGE = None


def zimage_still(prompt: str, seed: int = 1):
    """Z-Image Turbo through mflux. Weights download on first use."""
    global _ZIMAGE
    if _ZIMAGE is None:
        from mflux.models.z_image import ZImageTurbo

        _ZIMAGE = ZImageTurbo(quantize=8)
    return _ZIMAGE.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=8,
        height=1024,
        width=1024,
    )


_QWEN = None


def qwen_still(prompt: str, seed: int = 1):
    """Qwen-Image 2.1 Turbo through mflux, sampled on the checkpoint's own 8-node grid.

    The transformer is quantized to 8 bits; the Qwen3-VL text encoder stays bf16
    (mflux never quantizes it), so plan on about 30 GB of unified memory.
    """
    global _QWEN
    from flat2d.turbo import TurboGrid, load_grid

    if _QWEN is None:
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        _QWEN = QwenImage21(quantize=8, model_path=qwen_repo())
        TurboGrid.NODES = load_grid(qwen_repo())
    return _QWEN.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=len(TurboGrid.NODES),
        height=1024,
        width=1024,
        scheduler="flat2d.turbo.TurboGrid",
    )


def still_image(prompt: str, seed: int = 1):
    """The still from whichever local model still_backend() picked."""
    backend = still_backend()
    if backend == "qwen-image":
        return qwen_still(prompt, seed=seed)
    if backend == "z-image":
        return zimage_still(prompt, seed=seed)
    raise FlatUnavailable("No local image model. Install mflux.")


def generate_still(root: Path, stem: str, prompt: str) -> dict:
    text = " ".join((prompt or "").split())
    if not text:
        raise FlatUnavailable("empty prompt")
    backend = still_backend()
    if not backend:
        raise FlatUnavailable("No local image model. Install mflux.")
    from flat2d.store import save_source

    # A new seed each press, like Ideas; seed 1 every time drew the same picture.
    image, seed, redrawn = clean_still(still_prompt(text), random.randint(1, 2**31 - 1))
    buf = __import__("io").BytesIO()
    image.save(buf, format="PNG")
    saved = save_source(root, stem, buf.getvalue())
    key = ground_key(image)
    write_json(flat_dir(root, stem) / "key.json", key)
    note = flat_dir(root, stem) / "prompt.txt"
    note.write_text(text + "\n", encoding="utf-8")
    saved["prompt"] = text
    saved["key"] = key["rgb"]
    saved["still"] = backend
    saved["seed"] = seed
    saved["redrawn"] = redrawn
    return saved


def write_stub_wav(dest: Path, sentence: str) -> str:
    """A short PCM wav whose samples depend on the sentence. No model."""
    import struct

    prompt = sound_prompt(sentence)
    raw = prompt.encode("utf-8")
    count = 480 + (sum(raw) % 320)
    samples = bytes((80 + (raw[i % len(raw)] + i * 13) % 120) for i in range(count))
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + count,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        8000,
        8000,
        1,
        8,
        b"data",
        count,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(header + samples)
    return prompt


def _record_fields(root: Path, stem: str, part: dict, fields: dict) -> None:
    from flat2d.sheet import read_json

    path = flat_dir(root, stem) / "mechanics.json"
    mech = read_json(path) or {"ok": True, "stem": stem, "note": "", "parts": []}
    rows = list(mech.get("parts") or [])
    pid = str(part.get("id") or "")
    pfile = str(part.get("file") or "")
    found = False
    for row in rows:
        if str(row.get("id") or "") == pid or (pfile and str(row.get("file") or "") == pfile):
            row.update(fields)
            found = True
            break
    if not found:
        row = dict(part)
        row.update(fields)
        rows.append(row)
    mech["parts"] = rows
    mech["ok"] = True
    mech["stem"] = stem
    write_json(path, mech)


def _record_sound(root: Path, stem: str, part: dict, rel: str) -> None:
    _record_fields(root, stem, part, {"sound": rel})


def render_sound(root: Path, stem: str, part: dict) -> dict:
    filename, prompt = sound_spec(part)
    dest = flat_dir(root, stem) / "sounds" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = sfx_mode()
    if mode == "stub":
        write_stub_wav(dest, (part.get("projectile") or part.get("motion") or ""))
    elif mode == "no-runner":
        raise FlatUnavailable(
            "SFX weights are present. Set FLAT2D_SFX_CMD to the local runner "
            "(prompt and output path are the two arguments)."
        )
    else:
        import subprocess

        proc = subprocess.run([sfx_cmd(), prompt, str(dest)], capture_output=True, text=True)
        if proc.returncode != 0 or not dest.is_file():
            err = (proc.stderr or proc.stdout or "sfx runner failed").strip()
            raise FlatUnavailable(err)
    rel = f"sounds/{filename}"
    _record_sound(root, stem, part, rel)
    return {"ok": True, "stem": stem, "sound": rel, "prompt": prompt, "stub": mode == "stub", "sfx_mode": mode}


_KOKORO = None
_KOKORO_REPO: str | None = None


def kokoro_model():
    """Load Kokoro once. The repo id is mlx-community/Kokoro-82M-bf16 unless FLAT2D_KOKORO is set."""
    global _KOKORO, _KOKORO_REPO
    repo = kokoro_repo()
    if _KOKORO is not None and _KOKORO_REPO == repo:
        return _KOKORO
    from mlx_audio.tts.utils import load

    _KOKORO = load(repo)
    _KOKORO_REPO = repo
    return _KOKORO


def write_pcm16(dest: Path, samples, rate: int) -> None:
    import struct

    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise FlatUnavailable("kokoro produced no audio")
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        rate,
        rate * 2,
        2,
        16,
        b"data",
        len(pcm),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(header + pcm)


def speak(root: Path, stem: str, part: dict) -> dict:
    line = " ".join(str(part.get("line") or "").split())
    if not line:
        raise FlatUnavailable("empty line")
    if not voice_ready():
        raise FlatUnavailable("mlx-audio is not installed")
    try:
        speaker = speaker_name(str(part.get("speaker") or ""))
        filename = voice_file(str(part.get("id") or ""))
    except ValueError as e:
        raise FlatUnavailable(str(e)) from e
    dest = flat_dir(root, stem) / "voices" / filename
    try:
        model = kokoro_model()
        chunks = []
        rate = 24000
        for result in model.generate(line, voice=speaker, speed=1.0, lang_code="a"):
            chunks.append(result.audio)
            rate = int(result.sample_rate)
    except FlatUnavailable:
        raise
    except Exception as e:
        raise FlatUnavailable(f"kokoro failed: {e}") from e
    if not chunks:
        raise FlatUnavailable("kokoro produced no audio")
    import numpy as np

    audio = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks])
    write_pcm16(dest, audio, rate)
    rel = f"voices/{filename}"
    _record_fields(root, stem, part, {"voice": rel, "line": line, "speaker": speaker})
    return {"ok": True, "stem": stem, "voice": rel, "line": line, "speaker": speaker, "kokoro": kokoro_repo()}
