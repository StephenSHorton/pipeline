"""Optional local models. Missing weights disable the button. No paid API."""
from __future__ import annotations

import contextlib
import json
from concurrent.futures import ThreadPoolExecutor
import os
import random
import time
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


class DrawCancelled(Exception):
    """Cancel was pressed while a still was drawing. The runner catches it."""


# The running draw's hook (flat2d/drawjob.py), or None for a plain call. It has step(k, n),
# preview(pil), attempt(i, total, flaws), ground(), loading(bool) and cancelled().
DRAW = None


@contextlib.contextmanager
def drawing(hook):
    global DRAW
    before = DRAW
    DRAW = hook
    try:
        yield hook
    finally:
        DRAW = before


# MLX streams belong to the thread that made them, so a model built on one job's thread fails on the
# next ("There is no Stream(cpu, 1) in current thread"). Every MLX model is built and run on this one thread.
_MODEL_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flat2d-model")


def on_model_thread(fn, *args, **kwargs):
    return _MODEL_THREAD.submit(fn, *args, **kwargs).result()


def previews_on() -> bool:
    return (os.environ.get("FLAT2D_PREVIEW") or "").strip().lower() not in ("off", "0", "no", "false")


def _check_cancel() -> None:
    if DRAW is not None and DRAW.cancelled():
        raise DrawCancelled("Cancelled")


class _Relay:
    """One mflux callback per cached model, forwarding to DRAW when a draw has a hook.
    mflux's registry only appends, so this is registered once, when the model is built.
    in_loop(t) fires before step t is evaluated (MLX is lazy), so t means t steps are done."""

    def __init__(self, model, preview: bool):
        self.model = model
        self.preview = preview
        self.prev = None
        self.loading = False
        self.warm = False  # the weights are in memory: step 0 has been evaluated once

    def call_before_loop(self, seed, prompt, latents, config, canny_image=None, depth_image=None, control_images=None):
        self.prev = latents

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        if t >= 1:
            self.warm = True
        hook = DRAW
        if hook is None:
            self.prev = latents
            return
        if t >= 1 and self.loading:
            self.loading = False
            hook.loading(False)
        # The weights load in the eval after in_loop(0): a cancel during the load stops at step 1,
        # once they are in memory ("Stops after loading"), so a cancelled load is never half a model.
        if hook.cancelled() and not (t == 0 and self.loading):
            raise DrawCancelled("Cancelled")
        if t >= 1:
            hook.step(t, config.num_inference_steps)
            if self.preview and previews_on() and getattr(hook, "wants_preview", True) and t in (1, 3, 5) and self.prev is not None:
                try:
                    self._preview(hook, t, latents, config)
                except Exception:
                    pass  # a preview never breaks the draw
        self.prev = latents

    def _preview(self, hook, t, latents, config) -> None:
        from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
        from mflux.utils.image_util import ImageUtil

        s = config.scheduler.sigmas
        v = (latents - self.prev) / (s[t + 1] - s[t])
        x0 = latents - s[t + 1] * v
        lat = Qwen21LatentCreator.unpack_latents(latents=x0, height=config.height, width=config.width)
        hook.preview(ImageUtil.to_pil(self.model.vae.decode(lat)))

    def call_after_loop(self, seed, prompt, latents, config):
        self.prev = None
        self.warm = True
        if DRAW is not None:
            n = config.num_inference_steps
            DRAW.step(n, n)


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


def still_prompt(text: str, pose: str | None = None) -> str:
    """One object on a flat magenta ground, nothing else. The subject goes first (with it last,
    Qwen-Image dressed "a knight's sword" as a knight), and shadows, floors and effects are
    ruled out one by one: they are what the magenta pull cannot take back out.
    pose (a flat2d.pose.CREATURE_POSE kind, or the clause itself) replaces the floating-object clause with the
    canonical creature pose: standing on the ground in strict profile, head toward the RIGHT edge."""
    if pose:
        from flat2d.pose import CREATURE_POSE

        return text + ". " + CREATURE_POSE.get(pose, pose)
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


def clean_still(prompt: str, seed: int, check=None):
    """Draw, and redraw on a new seed while the ground is cloudy or holds a shadow.
    When every try has a flaw, the one with the flattest ground is kept.
    Returns the image, the seed it was drawn with, and the flaws the redraws were for.
    check(image) -> {flaws, rank} adds its own flaws (the facing check: "facing", "profile") inside the same
    REDRAWS loop; among flawed tries a clean ground comes first, then the lowest rank, then the flattest ground."""
    global _STUB_FLAWS_LEFT
    _STUB_FLAWS_LEFT = _stub_flaws()
    replaced: list[str] = []
    best = None
    flaws: list[str] = []
    for attempt in range(REDRAWS + 1):
        _check_cancel()
        if attempt and DRAW is not None:
            DRAW.attempt(attempt + 1, REDRAWS + 1, list(flaws))
        image = still_image(prompt, seed=seed)
        if hasattr(image, "image"):
            image = image.image
        if DRAW is not None:
            DRAW.preview(image)
            DRAW.ground()
        flaws = ground_flaws(image)
        rank = 0
        if check is not None:
            got = check(image) or {}
            rank = int(got.get("rank") or 0)
            ground_bad = bool(flaws)
            flaws = flaws + [f for f in got.get("flaws") or [] if f not in flaws]
        else:
            ground_bad = bool(flaws)
        if not flaws:
            return image, seed, replaced
        score = (ground_bad, rank, ground_key(image)["tol"])
        if best is None or score < best[0]:
            best = (score, image, seed)
        if attempt < REDRAWS:
            replaced.extend(flaws)
            seed = random.randint(1, 2**31 - 1)
    if DRAW is not None:
        DRAW.preview(best[1])  # the one kept, not the last one drawn
    return best[1], best[2], replaced


_ZIMAGE = None
_ZIMAGE_RELAY = None


def _cold(relay) -> None:
    """Weights not in memory yet (first draw, or the last one stopped before step 1): this draw loads them."""
    relay.loading = False
    if not relay.warm and DRAW is not None:
        DRAW.loading(True)
        relay.loading = True


def zimage_still(prompt: str, seed: int = 1):
    """Z-Image Turbo through mflux. Weights download on first use."""
    global _ZIMAGE, _ZIMAGE_RELAY
    if _ZIMAGE is None:
        from mflux.models.z_image import ZImageTurbo

        model = ZImageTurbo(quantize=8)
        relay = _Relay(model, preview=False)  # the x0 preview is only checked on Qwen-Image
        model.callbacks.register(relay)
        _ZIMAGE, _ZIMAGE_RELAY = model, relay
    _cold(_ZIMAGE_RELAY)
    return _ZIMAGE.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=8,
        height=1024,
        width=1024,
    )


_QWEN = None
_QWEN_RELAY = None


def qwen_still(prompt: str, seed: int = 1):
    """Qwen-Image 2.1 Turbo through mflux, sampled on the checkpoint's own 8-node grid.

    The transformer is quantized to 8 bits; the Qwen3-VL text encoder stays bf16
    (mflux never quantizes it), so plan on about 30 GB of unified memory.
    """
    global _QWEN, _QWEN_RELAY
    from flat2d.turbo import TurboGrid, load_grid

    if _QWEN is None:
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        if DRAW is not None:
            DRAW.loading(True)  # building the model takes a while too
        model = QwenImage21(quantize=8, model_path=qwen_repo())
        TurboGrid.NODES = load_grid(qwen_repo())
        relay = _Relay(model, preview=True)
        model.callbacks.register(relay)
        _QWEN, _QWEN_RELAY = model, relay
    _cold(_QWEN_RELAY)  # weights load lazily inside step 1
    return _QWEN.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=len(TurboGrid.NODES),
        height=1024,
        width=1024,
        scheduler="flat2d.turbo.TurboGrid",
    )


_STUB_LOADED = False
_STUB_FLAWS_LEFT = 0
STUB_SIZE = 1024


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name) or default))
    except ValueError:
        return default


def _stub_flaws() -> int:
    try:
        return max(0, int(os.environ.get("FLAT2D_STUB_FLAWS") or 0))
    except ValueError:
        return 0


def _stub_color(seed: int) -> tuple[int, int, int]:
    """A disc colour from the seed, kept well away from magenta and its shadows (green stays high)."""
    rng = random.Random(seed)
    return rng.randint(0, 200), rng.randint(140, 255), rng.randint(0, 200)


def _stub_picture(seed: int, fill: float, cloudy: bool = False):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (STUB_SIZE, STUB_SIZE), (255, 0, 255))
    r = int(STUB_SIZE * 0.3 * max(0.05, min(fill, 1.0)))
    c = STUB_SIZE // 2
    ImageDraw.Draw(im).ellipse((c - r, c - r, c + r, c + r), fill=_stub_color(seed))
    if cloudy:
        import numpy as np

        rgb = np.asarray(im, dtype=np.int16)
        noise = np.random.default_rng(seed).integers(-60, 61, size=rgb.shape)
        im = Image.fromarray(np.clip(rgb + noise, 0, 255).astype(np.uint8), "RGB")
    return im


def stub_still(prompt: str, seed: int = 1):
    """FLAT2D_STILL=stub: a fake model for tests and the app without weights. It loads once,
    takes FLAT2D_STUB_STEP seconds per step, fires the same hook calls as the real models,
    and returns a magenta still with a disc coloured by the seed."""
    global _STUB_LOADED, _STUB_FLAWS_LEFT
    if not _STUB_LOADED:
        if DRAW is not None:
            DRAW.loading(True)
        time.sleep(_env_float("FLAT2D_STUB_LOAD", 1.0))
        _STUB_LOADED = True
        loading = DRAW is not None
    else:
        loading = False
    step_s = _env_float("FLAT2D_STUB_STEP", 0.4)
    n = 8
    for k in range(1, n + 1):
        _check_cancel()
        time.sleep(step_s)
        if DRAW is not None:
            if loading:
                loading = False
                DRAW.loading(False)
            DRAW.step(k, n)
            if k in (2, 4, 6) and previews_on() and getattr(DRAW, "wants_preview", True):
                DRAW.preview(_stub_picture(seed, k / n))
    cloudy = _STUB_FLAWS_LEFT > 0
    if cloudy:
        _STUB_FLAWS_LEFT -= 1
    return _stub_picture(seed, 1.0, cloudy=cloudy)


def model_loaded() -> bool:
    """The picked still model is already in memory, so a draw skips the load."""
    backend = still_backend()
    if backend == "stub":
        return _STUB_LOADED
    if backend == "qwen-image":
        return _QWEN_RELAY is not None and _QWEN_RELAY.warm
    if backend == "z-image":
        return _ZIMAGE_RELAY is not None and _ZIMAGE_RELAY.warm
    return False


def still_image(prompt: str, seed: int = 1):
    """The still from whichever local model still_backend() picked."""
    backend = still_backend()
    if backend == "stub":
        return stub_still(prompt, seed=seed)
    if backend == "qwen-image":
        return on_model_thread(qwen_still, prompt, seed=seed)
    if backend == "z-image":
        return on_model_thread(zimage_still, prompt, seed=seed)
    raise FlatUnavailable("No local image model. Install mflux.")


def facing_checker(expected: str):
    """A clean_still check for a posed creature: facing.detect_image on each try, judged against the expected
    facing ('right' for a profile, 'camera' for a humanoid). Results are kept per image for the caller."""
    import tempfile

    import facing

    seen: dict[int, dict] = {}

    def check(image) -> dict:
        _check_cancel()
        if DRAW is not None and hasattr(DRAW, "line"):
            DRAW.line("Checking which way the head points.")
        with tempfile.TemporaryDirectory(prefix="still_facing_") as tmp:
            path = Path(tmp) / "still.png"
            image.save(path, format="PNG")
            det = facing.detect_image(path)
        verdict = facing.judge(det, expected)
        seen[id(image)] = {"detect": det, **verdict}
        return verdict

    check.seen = seen
    return check


def canonical_result(image, kind: str, got: dict | None):
    """After the draws: mirror a profile whose head points left (lossless for a profile), or give up canonical
    for a picture that is not a profile. Returns (image, key fields for key.json)."""
    from PIL import ImageOps

    from flat2d.pose import FACING, POSE_NAME

    expected = FACING[kind]
    det = (got or {}).get("detect") or {}
    check = {k: det.get(k) for k in ("head", "profile", "confidence", "how") if k in det}
    canonical = True
    if not got or det.get("how") == "unavailable":
        check["how"] = "unchecked"
        if det.get("why"):
            check["why"] = det["why"]
    elif got.get("ok"):
        check["how"] = "vlm"
    elif got.get("mirror"):
        image = ImageOps.mirror(image)
        check["how"] = "mirrored"
        check["head_before"] = det.get("head")
    else:
        canonical = False
        check["how"] = "not_canonical"
        check["why"] = ("not a strict side profile; three-quarter views break retexture --mirror"
                        if expected == "right" and det.get("profile") is not True
                        else f"the head points {det.get('head') or 'nowhere clear'}, not {expected}")
    fields = {"pose": POSE_NAME[kind], "facing": expected, "canonical": canonical, "facing_check": check}
    if not canonical:
        fields["facing"] = det.get("head") if det.get("head") in ("left", "right", "camera", "away") else None
    return image, fields


def generate_still(root: Path, stem: str, prompt: str, intent: dict | None = None) -> dict:
    text = " ".join((prompt or "").split())
    if not text:
        raise FlatUnavailable("empty prompt")
    backend = still_backend()
    if not backend:
        raise FlatUnavailable("No local image model. Install mflux.")
    from flat2d.pose import FACING, pose_for
    from flat2d.store import save_source

    kind = pose_for(intent, text)
    check = facing_checker(FACING[kind]) if kind else None
    # A new seed each press, like Ideas; seed 1 every time drew the same picture.
    image, seed, redrawn = clean_still(still_prompt(text, kind), random.randint(1, 2**31 - 1), check=check)
    posed = None
    if kind:
        image, posed = canonical_result(image, kind, check.seen.get(id(image)))
    buf = __import__("io").BytesIO()
    image.save(buf, format="PNG")
    saved = save_source(root, stem, buf.getvalue())
    key = ground_key(image)
    if posed:
        key = {**key, **posed}
    write_json(flat_dir(root, stem) / "key.json", key)
    note = flat_dir(root, stem) / "prompt.txt"
    note.write_text(text + "\n", encoding="utf-8")
    saved["prompt"] = text
    saved["key"] = key["rgb"]
    saved["still"] = backend
    saved["seed"] = seed
    saved["redrawn"] = redrawn
    if posed:
        saved.update(posed)
        saved["kind"] = kind
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
    def run():
        chunks, rate = [], 24000
        for result in kokoro_model().generate(line, voice=speaker, speed=1.0, lang_code="a"):
            chunks.append(result.audio)
            rate = int(result.sample_rate)
        return chunks, rate

    try:
        chunks, rate = on_model_thread(run)
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
