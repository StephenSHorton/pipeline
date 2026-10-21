"""Sampling schedules as mflux schedulers.

TurboGrid: the fixed grid Qwen-Image 2.1 Turbo was distilled on. It applies to the 2.1 path only (an NC model,
reachable only with FLAT2D_ALLOW_NC=1). Lightning: the schedule lightx2v's Qwen-Image-2512-Lightning LoRAs were
distilled with (the default draw).

The Turbo checkpoint stores its 8 sigma nodes in model_index.json (`sample_sigmas`).
They are used as is: no resolution shift, no terminal rescale, then a final 0.
mflux's stock Qwen-Image 2.1 schedule shifts a linspace instead, which the model
was not trained on, so Generate passes this class by its dotted path.
"""
from __future__ import annotations

import json

# sample_sigmas from Qwen/Qwen-Image-2.1-Turbo model_index.json, used when that file
# cannot be read (offline with a partial cache, or an override repo without it).
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


def read_grid(model_index: dict) -> tuple[float, ...]:
    nodes = model_index.get("sample_sigmas") or ()
    try:
        nodes = tuple(float(v) for v in nodes)
    except (TypeError, ValueError):
        return TURBO_SIGMAS
    ok = nodes and all(0 < v <= 1 for v in nodes) and all(a > b for a, b in zip(nodes, nodes[1:]))
    return nodes if ok else TURBO_SIGMAS


def load_grid(repo: str) -> tuple[float, ...]:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, "model_index.json")
        with open(path, encoding="utf-8") as fh:
            return read_grid(json.load(fh))
    except Exception:
        return TURBO_SIGMAS


# lightx2v Qwen-Image-Lightning's scheduler config: FlowMatchEuler, use_dynamic_shifting with base_shift = max_shift
# = log(3) and time_shift_type exponential, so every resolution gets the same shift mu = log(3) on a linspace
# (1 .. 1/N), with no terminal stretch, then a final 0.
LIGHTNING_SHIFT = 3.0


def lightning_sigmas(steps: int, shift: float = LIGHTNING_SHIFT) -> list[float]:
    nodes = [1.0 - i * (1.0 - 1.0 / steps) / max(steps - 1, 1) for i in range(steps)]
    return [shift * s / (1.0 + (shift - 1.0) * s) for s in nodes] + [0.0]


try:
    import mlx.core as mx
    from mflux.models.common.schedulers.base_scheduler import BaseScheduler
except ImportError:  # no mflux: the classes are never asked for
    TurboGrid = None
    Lightning = None
else:

    class TurboGrid(BaseScheduler):
        NODES = TURBO_SIGMAS

        def __init__(self, config):
            steps = config.num_inference_steps
            if steps != len(self.NODES):
                raise ValueError(f"Qwen-Image 2.1 Turbo samples {len(self.NODES)} fixed steps, got {steps}")
            self.config = config
            self._sigmas = mx.array(list(self.NODES) + [0.0], dtype=mx.float32)
            self._timesteps = mx.arange(steps, dtype=mx.float32)

        @property
        def sigmas(self) -> mx.array:
            return self._sigmas

        @property
        def timesteps(self) -> mx.array:
            return self._timesteps

        def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
            dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
            return latents + noise.astype(latents.dtype) * dt

    class Lightning(BaseScheduler):
        """mflux scheduler with the Lightning LoRA's own schedule (any step count; the LoRA is made for 4 or 8)."""

        def __init__(self, config):
            self.config = config
            self._sigmas = mx.array(lightning_sigmas(config.num_inference_steps), dtype=mx.float32)
            self._timesteps = mx.arange(config.num_inference_steps, dtype=mx.float32)

        @property
        def sigmas(self) -> mx.array:
            return self._sigmas

        @property
        def timesteps(self) -> mx.array:
            return self._timesteps

        def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
            dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
            return latents + noise.astype(latents.dtype) * dt

