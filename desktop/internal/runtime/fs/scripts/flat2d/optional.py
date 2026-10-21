"""Optional local models. Missing weights disable the button. No paid API."""
from __future__ import annotations

import contextlib
import json
from concurrent.futures import ThreadPoolExecutor
import os
import random
import time
import traceback
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
    still_degraded,
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
NO_REDRAW = {"facing_unavailable"}  # flaws a new seed cannot fix


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
        if all(f in NO_REDRAW for f in flaws):
            return image, seed, replaced  # a new seed cannot bring the vision model back
        score = (ground_bad, rank, ground_key(image)["tol"])
        if best is None or score < best[0]:
            best = (score, image, seed)
        if attempt < REDRAWS:
            replaced.extend(flaws)
            seed = random.randint(1, 2**31 - 1)
    if DRAW is not None:
        DRAW.preview(best[1])  # the one kept, not the last one drawn
    return best[1], best[2], replaced


# MLX keeps freed buffers for reuse, up to the memory limit by default: beside Qwen-Image's ~22 GiB of weights
# that cache settled at ~44 GiB. FLAT2D_MLX_CACHE_GB caps it (0 turns the cache off).
MLX_CACHE_GB = 4.0


def _cap_mlx_cache() -> None:
    """Before an MLX model is built. The limit is process-wide, so the last call wins."""
    import mlx.core as mx

    mx.set_cache_limit(int(_env_float("FLAT2D_MLX_CACHE_GB", MLX_CACHE_GB) * 2**30))


def _release(model) -> None:
    """After a draw, finished, failed or cancelled, on the model thread: drop the prompt's text features
    (Qwen-Image keeps one entry per prompt ever drawn; a redraw of the same prompt encodes it again, a second
    or two) and hand the buffers MLX kept for reuse back to the system."""
    cache = getattr(model, "prompt_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    try:
        import mlx.core as mx
    except ImportError:  # a stand-in model in tests
        return
    mx.clear_cache()


def _draw(model, relay, **kwargs):
    try:
        return model.generate_image(**kwargs)
    except BaseException as e:
        # The traceback holds the sampler's frames, latents and all; empty them so the release frees them now.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        relay.prev = None  # a cancelled draw never reaches call_after_loop
        _release(model)


_ZIMAGE = None
_ZIMAGE_RELAY = None


def _cold(relay) -> None:
    """Weights not in memory yet (first draw, or the last one stopped before step 1): this draw loads them."""
    relay.loading = False
    if not relay.warm and DRAW is not None:
        DRAW.loading(True)
        relay.loading = True


def zimage_still(prompt: str, seed: int = 1, width: int = 1024, height: int = 1024):
    """Z-Image Turbo (Apache-2.0) through mflux, 8-bit, 8 steps. Weights download on first use."""
    global _ZIMAGE, _ZIMAGE_RELAY
    if _ZIMAGE is None:
        from mflux.models.z_image import ZImageTurbo

        _cap_mlx_cache()
        model = ZImageTurbo(quantize=8)
        relay = _Relay(model, preview=False)  # the x0 preview is only checked on Qwen-Image
        model.callbacks.register(relay)
        _ZIMAGE, _ZIMAGE_RELAY = model, relay
    _cold(_ZIMAGE_RELAY)
    return _draw(
        _ZIMAGE,
        _ZIMAGE_RELAY,
        seed=seed,
        prompt=prompt,
        num_inference_steps=8,
        height=height,
        width=width,
    )


_QWEN = None
_QWEN_RELAY = None


def qwen_still(prompt: str, seed: int = 1, width: int = 1024, height: int = 1024):
    """Qwen-Image 2.1 Turbo through mflux, sampled on the checkpoint's own 8-node grid.
    Qwen RESEARCH licence (non-commercial): reachable only with FLAT2D_ALLOW_NC=1 (sheet.still_backend).

    The transformer is quantized to 8 bits; the Qwen3-VL text encoder stays bf16
    (mflux never quantizes it): ~22 GiB of weights stay resident, ~36 GiB peak while
    drawing, and the MLX buffer cache is capped (MLX_CACHE_GB) and emptied after each draw.
    """
    global _QWEN, _QWEN_RELAY
    from flat2d.turbo import TurboGrid, load_grid

    if _QWEN is None:
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        if DRAW is not None:
            DRAW.loading(True)  # building the model takes a while too
        _cap_mlx_cache()
        model = QwenImage21(quantize=8, model_path=qwen_repo())
        TurboGrid.NODES = load_grid(qwen_repo())
        relay = _Relay(model, preview=True)
        model.callbacks.register(relay)
        _QWEN, _QWEN_RELAY = model, relay
    _cold(_QWEN_RELAY)  # weights load lazily inside step 1
    return _draw(
        _QWEN,
        _QWEN_RELAY,
        seed=seed,
        prompt=prompt,
        num_inference_steps=len(TurboGrid.NODES),
        height=height,
        width=width,
        scheduler="flat2d.turbo.TurboGrid",
    )


_Q2512 = None
_Q2512_RELAY = None


def _qwen_cfg1(model, seed: int, prompt: str, num_inference_steps: int, width: int, height: int,
               scheduler: str = "flat2d.turbo.Lightning"):
    """mflux QwenImage.generate_image at CFG 1, with ONE transformer pass a step. At guidance 1 mflux's guided
    noise is the positive noise exactly (neg + 1 * (pos - neg), then rescaled to its own norm), so its second,
    negative pass is wasted work: this loop leaves it out and halves the step time. Same latents, callbacks
    and decode as mflux."""
    import mlx.core as mx
    from mflux.models.common.config.config import Config
    from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
    from mflux.utils.image_util import ImageUtil

    config = Config(width=width, height=height, guidance=1.0, scheduler=scheduler, model_config=model.model_config,
                    num_inference_steps=num_inference_steps)
    latents = LatentCreator.create_for_txt2img_or_img2img(
        seed=seed, width=config.width, height=config.height,
        img2img=Img2Img(vae=model.vae, latent_creator=QwenLatentCreator, sigmas=config.scheduler.sigmas,
                        init_time_step=config.init_time_step, image_path=None, tiling_config=model.tiling_config))
    embeds, mask, _neg, _neg_mask = QwenPromptEncoder.encode_prompt(
        prompt=prompt, negative_prompt=None, prompt_cache=model.prompt_cache,
        qwen_tokenizer=model.tokenizers["qwen"], qwen_text_encoder=model.text_encoder)
    ctx = model.callbacks.start(seed=seed, prompt=prompt, config=config)
    ctx.before_loop(latents)
    for t in config.time_steps:
        latents = config.scheduler.scale_model_input(latents, t)
        noise = model.transformer(t=t, config=config, hidden_states=latents, encoder_hidden_states=embeds,
                                  encoder_hidden_states_mask=mask)
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
        ctx.in_loop(t, latents)
        mx.eval(latents)
    ctx.after_loop(latents)
    latents = QwenLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
    decoded = VAEUtil.decode(vae=model.vae, latent=latents, tiling_config=model.tiling_config)
    return ImageUtil.to_image(decoded_latents=decoded, config=config, seed=seed, prompt=prompt,
                              quantization=model.bits, lora_paths=model.lora_paths, lora_scales=model.lora_scales,
                              image_path=None, image_strength=None,
                              generation_time=config.time_steps.format_dict["elapsed"])


def build_qwen2512():
    """Qwen-Image-2512 (Apache-2.0) with lightx2v's Qwen-Image-2512-Lightning LoRA (Apache-2.0), 8-bit.
    A pre-saved 8-bit copy (sheet.q2512_saved(), written by save_qwen2512) loads instead of the bf16 cache when
    it is there; it holds the base weights only, and the LoRA is applied on top either way."""
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    from flat2d.sheet import q2512_lora, q2512_repo, q2512_saved

    saved = q2512_saved()
    if saved is not None:
        return QwenImage(quantize=None, model_path=str(saved), lora_paths=[q2512_lora()], lora_scales=[1.0])
    return QwenImage(quantize=8, model_path=q2512_repo(), lora_paths=[q2512_lora()], lora_scales=[1.0])


def qwen2512_still(prompt: str, seed: int = 1, width: int = 1024, height: int = 1024):
    """The default draw: Qwen-Image-2512 + Lightning, sheet.Q2512_STEPS steps on the Lightning schedule, CFG 1."""
    global _Q2512, _Q2512_RELAY
    from flat2d.sheet import q2512_steps

    if _Q2512 is None:
        if DRAW is not None:
            DRAW.loading(True)
        _cap_mlx_cache()
        model = build_qwen2512()
        relay = _Relay(model, preview=False)  # the x0 preview is a 2.1-only relay
        model.callbacks.register(relay)
        _Q2512, _Q2512_RELAY = model, relay
    _cold(_Q2512_RELAY)
    try:
        return _qwen_cfg1(_Q2512, seed=seed, prompt=prompt, num_inference_steps=q2512_steps(), width=width,
                          height=height)
    except BaseException as e:
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        _Q2512_RELAY.prev = None
        _release(_Q2512)


def unload() -> dict:
    """Free every draw model (the factory calls this before shape and paint: plan 1.9) and hand MLX's buffer
    cache back to the system. Safe to call when nothing is loaded. Returns what was freed."""
    global _ZIMAGE, _ZIMAGE_RELAY, _QWEN, _QWEN_RELAY, _Q2512, _Q2512_RELAY, _STUB_LOADED

    def run() -> list[str]:
        freed = [name for name, m in (("qwen-2512", _Q2512), ("qwen-2.1", _QWEN), ("z-image", _ZIMAGE)) if m is not None]
        return freed

    freed = on_model_thread(run)
    _Q2512 = _Q2512_RELAY = _QWEN = _QWEN_RELAY = _ZIMAGE = _ZIMAGE_RELAY = None
    if _STUB_LOADED:
        freed.append("stub")
    _STUB_LOADED = False

    def clear() -> None:
        import gc

        gc.collect()
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.clear_cache()

    on_model_thread(clear)
    return {"ok": True, "freed": freed}


_STUB_LOADED = False
_STUB_FLAWS_LEFT = 0
_STUB_SHEET_FLAWS_LEFT = 0
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


def _stub_sheet(seed: int, width: int, height: int, flaw: str = ""):
    """The stub's four-view sheet: front | right | back | left (row by row on a square grid), one creature-ish
    shape per panel, its head (a smaller disc) toward the RIGHT edge in the right panel and the LEFT in the left.
    flaw ('height', 'colour', 'side', 'empty' or 'swap': the side panels swapped, so the head in the right panel
    points left) breaks one check on purpose."""
    from PIL import Image, ImageDraw

    from flat2d.pose import PANELS

    im = Image.new("RGB", (width, height), (255, 0, 255))
    d = ImageDraw.Draw(im)
    grid = width == height
    cw, ch = (width // 2, height // 2) if grid else (width // 4, height)
    colour = _stub_color(seed)
    for i, pid in enumerate(PANELS):
        ox, oy = ((i % 2) * cw, (i // 2) * ch) if grid else (i * cw, 0)
        cx, cy = ox + cw // 2, oy + ch // 2
        bh = int(ch * (0.25 if flaw == "height" and pid == "back" else 0.35))
        fill = (40, 40, 40) if flaw == "colour" and pid == "back" else colour
        if flaw == "empty" and pid == "front":
            continue
        side = pid in ("right", "left") and not (flaw == "side" and pid == "left")  # 'side': left drawn head-on
        if flaw == "swap" and side:
            pid = "left" if pid == "right" else "right"
        if side:
            bw = int(cw * 0.3)
            d.ellipse((cx - bw, cy - bh // 2, cx + bw, cy + bh // 2), fill=fill)
            r = int(bh * 0.35)
            hx = cx + bw if pid == "right" else cx - bw
            d.ellipse((hx - r, cy - bh // 2 - r, hx + r, cy - bh // 2 + r), fill=fill)
            d.rectangle((cx - 4, cy + bh // 2 - 2, cx + 4, cy + bh // 2 + int(bh * 0.15)), fill=fill)
        else:
            bw = int(cw * 0.18)
            r = int(bh * 0.35)
            d.ellipse((cx - bw, cy - bh // 2, cx + bw, cy + bh // 2), fill=fill)
            d.ellipse((cx - r, cy - bh // 2 - r, cx + r, cy - bh // 2 + r), fill=fill)
            d.rectangle((cx - 4, cy + bh // 2 - 2, cx + 4, cy + bh // 2 + int(bh * 0.15)), fill=fill)
    return im


def stub_still(prompt: str, seed: int = 1, width: int = STUB_SIZE, height: int = STUB_SIZE):
    """FLAT2D_STILL=stub: a fake model for tests and the app without weights. It loads once,
    takes FLAT2D_STUB_STEP seconds per step, fires the same hook calls as the real models,
    and returns a magenta still with a disc coloured by the seed; a sheet-sized ask (sheet_size) gets a stub
    four-view sheet instead: FLAT2D_STUB_SHEET_FLAW (height, colour, side, empty, swap) breaks one check on the
    first FLAT2D_STUB_SHEET_FLAWS sheets of a draw (default 1), and FLAT2D_STUB_FLAWS grades their ground."""
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
    if (width, height) != (STUB_SIZE, STUB_SIZE):
        global _STUB_SHEET_FLAWS_LEFT
        flaw = ""
        if _STUB_SHEET_FLAWS_LEFT > 0:
            _STUB_SHEET_FLAWS_LEFT -= 1
            flaw = (os.environ.get("FLAT2D_STUB_SHEET_FLAW") or "").strip()
        sheet = _stub_sheet(seed, width, height, flaw)
        if cloudy:  # a graded ground, darker toward the bottom, like the real model paints
            import numpy as np

            from PIL import Image

            rgb = np.asarray(sheet, dtype=np.int16)
            ground = (rgb == (255, 0, 255)).all(-1)
            ramp = np.linspace(0, 70, height, dtype=np.int16)[:, None]
            rgb[..., 0] = np.where(ground, rgb[..., 0] - ramp, rgb[..., 0])
            rgb[..., 2] = np.where(ground, rgb[..., 2] - ramp, rgb[..., 2])
            sheet = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
        return sheet
    return _stub_picture(seed, 1.0, cloudy=cloudy)


def model_loaded() -> bool:
    """The picked still model is already in memory, so a draw skips the load."""
    backend = still_backend()
    if backend == "stub":
        return _STUB_LOADED
    if backend == "qwen-2512":
        return _Q2512_RELAY is not None and _Q2512_RELAY.warm
    if backend == "qwen-image":
        return _QWEN_RELAY is not None and _QWEN_RELAY.warm
    if backend == "z-image":
        return _ZIMAGE_RELAY is not None and _ZIMAGE_RELAY.warm
    return False


def still_image(prompt: str, seed: int = 1, width: int = 1024, height: int = 1024):
    """The still from whichever local model still_backend() picked."""
    backend = still_backend()
    if backend == "stub":
        return stub_still(prompt, seed=seed, width=width, height=height)
    if backend == "qwen-2512":
        return on_model_thread(qwen2512_still, prompt, seed=seed, width=width, height=height)
    if backend == "qwen-image":
        return on_model_thread(qwen_still, prompt, seed=seed, width=width, height=height)
    if backend == "z-image":
        return on_model_thread(zimage_still, prompt, seed=seed, width=width, height=height)
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
    if not got:
        check["how"] = "unchecked"
    elif det.get("how") == "unavailable":
        # Never a pass: the prompt asked for the canonical pose, but nothing saw it.
        canonical = False
        check["how"] = "unavailable"
        check["why"] = det.get("why") or "vision model unavailable"
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
    clear_views(root, stem)  # a single still: views from an older sheet must not sit beside it
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
    saved["degraded"] = still_degraded(backend)
    if posed:
        saved.update(posed)
        saved["kind"] = kind
        if posed["facing_check"].get("how") == "unavailable":
            from factory_env import degraded_entry

            saved["degraded"].append(degraded_entry("draw", "facing_unchecked", posed["facing_check"]["why"]))
    return saved


VIEW_FILES = ("sheet.png", "views.json")


def clear_views(root: Path, stem: str) -> None:
    """Remove flat/sheet.png, flat/views/ and flat/views.json (a concept take keeps its own copy)."""
    import shutil

    folder = flat_dir(root, stem)
    for name in VIEW_FILES:
        (folder / name).unlink(missing_ok=True)
    shutil.rmtree(folder / "views", ignore_errors=True)


def wants_views(intent: dict | None, views=None) -> bool:
    """A 3D-bound draw (intent output ps1, model or stack) is a four-view sheet unless views is 1."""
    from flat2d.pose import CREATURE_OUTPUTS

    if views is not None and str(views).strip() in ("1", "one", "single"):
        return False
    if isinstance(intent, dict) and str(intent.get("views") or "").strip() == "1":
        return False
    return isinstance(intent, dict) and intent.get("output") in CREATURE_OUTPUTS


def draw(root: Path, stem: str, prompt: str, intent: dict | None = None, views=None) -> dict:
    """The draw a request gets: generate_views for a 3D-bound intent, else (or with views: 1) generate_still."""
    if wants_views(intent, views):
        return generate_views(root, stem, prompt, intent)
    return generate_still(root, stem, prompt, intent=intent)


def _phase(name: str) -> None:
    if DRAW is not None and hasattr(DRAW, "phase"):
        DRAW.phase(name)


def generate_views(root: Path, stem: str, prompt: str, intent: dict | None = None) -> dict:
    """The four-view draw (plan 1.1-1.3): one sheet, front | right | back | left, split at its magenta gaps,
    cut out and checked (flat2d/views.py). At most SHEET_REDRAWS new sheets, for hard flaws only (a cloudy ground
    or a shadow is a note); when every sheet is flawed the best one is kept with consistent: false.
    Writes flat/sheet.png, flat/views/*.png, flat/views.json, flat/source.png (the canonical panel), key.json
    (+ "views": "flat/views.json") and prompt.txt."""
    text = " ".join((prompt or "").split())
    if not text:
        raise FlatUnavailable("empty prompt")
    backend = still_backend()
    if not backend:
        raise FlatUnavailable("No local image model. Install mflux.")
    from flat2d import views as V
    from flat2d.pose import FACING, POSE_NAME, canonical_view, pose_for, sheet_prompt
    from flat2d.sheet import SHEET_REDRAWS, sheet_size, still_info
    from flat2d.store import save_source
    from factory_env import degraded_entry

    global _STUB_FLAWS_LEFT, _STUB_SHEET_FLAWS_LEFT
    _STUB_FLAWS_LEFT = _stub_flaws()
    try:
        _STUB_SHEET_FLAWS_LEFT = max(0, int(os.environ.get("FLAT2D_STUB_SHEET_FLAWS") or 1))
    except ValueError:
        _STUB_SHEET_FLAWS_LEFT = 1
    kind = pose_for(intent, text)
    w, h, layout = sheet_size()
    ask = sheet_prompt(text, kind, layout)
    seed = random.randint(1, 2**31 - 1)
    tries: list[dict] = []
    redrawn: list[str] = []
    for attempt in range(SHEET_REDRAWS + 1):
        _check_cancel()
        if attempt and DRAW is not None:
            DRAW.attempt(attempt + 1, SHEET_REDRAWS + 1, V.redrawable(tries[-1]["result"]["flaws"]))
        image = still_image(ask, seed=seed, width=w, height=h)
        image = getattr(image, "image", image).convert("RGB")
        if DRAW is not None:
            DRAW.preview(image)
        key = ground_key(image)
        _check_cancel()
        result = V.check_panels(image, key, kind, layout, phase=_phase)
        tries.append({"image": image, "seed": seed, "key": key, "result": result})
        if DRAW is not None and hasattr(DRAW, "line"):
            DRAW.line("Views: " + (", ".join(result["flaws"]) if result["flaws"] else "consistent") + ".")
        again = V.redrawable(result["flaws"])
        if not again or attempt == SHEET_REDRAWS:
            break
        redrawn.extend(again)
        seed = random.randint(1, 2**31 - 1)
    best = min(tries, key=lambda t: V.score(t["result"]))
    result, image = best["result"], best["image"]
    if DRAW is not None and best is not tries[-1]:
        DRAW.preview(image)  # the one kept, not the last one drawn
    canonical = canonical_view(kind)
    degraded = list(still_degraded(backend))
    fc = result["checks"].get("facing") or {}
    if fc.get("how") == "unavailable":
        degraded.append(degraded_entry("draw", "facing_unchecked", fc.get("why") or "vision model unavailable"))
    if "mask_key" in result["notes"]:
        degraded.append(degraded_entry("views", "mask_key", "Apple Vision unavailable; views cut out on the magenta key"))
    # A flawed sheet is consistent: false in views.json; the factory's views handoff says DEGRADED views inconsistent.
    folder = flat_dir(root, stem)
    clear_views(root, stem)
    buf = __import__("io").BytesIO()
    result["panels"][canonical].convert("RGB").save(buf, format="PNG")
    saved = save_source(root, stem, buf.getvalue())
    info = still_info(backend)
    meta = {"w": w, "h": h, "layout": layout, "seed": best["seed"], "redraws": len(tries) - 1,
            "model": info["model"], "licence": info["licence"], "backend": backend, "prompt": ask}
    doc = V.write_views(folder, image, result, kind=kind, canonical=canonical, meta=meta, degraded=degraded)
    key = ground_key(result["panels"][canonical])
    key["views"] = "flat/views.json"
    posed = None
    if kind:
        expected = FACING[kind]
        check = {k: fc.get(k) for k in ("head", "profile", "confidence", "how", "why", "head_before") if k in fc}
        canonical_ok = not result["flaws"] and fc.get("how") != "unavailable"
        if fc.get("how") == "not_needed":
            check = {"how": "front", "why": "a front-canonical figure"}
        posed = {"pose": POSE_NAME[kind], "facing": expected, "canonical": canonical_ok, "facing_check": check}
        key.update(posed)
    write_json(folder / "key.json", key)
    (folder / "prompt.txt").write_text(text + "\n", encoding="utf-8")
    saved.update({"prompt": text, "key": key["rgb"], "still": backend, "seed": best["seed"], "redrawn": redrawn,
                  "views": "flat/views.json", "consistent": doc["consistent"], "flaws": doc["flaws"],
                  "notes": [V.NOTE_TEXT.get(n, n) for n in doc["notes"]], "degraded": degraded,
                  "sheet": "flat/sheet.png", "canonical_view": canonical})
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

    _cap_mlx_cache()
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
        model = kokoro_model()
        try:
            for result in model.generate(line, voice=speaker, speed=1.0, lang_code="a"):
                chunks.append(result.audio)
                rate = int(result.sample_rate)
        finally:
            _release(model)
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
