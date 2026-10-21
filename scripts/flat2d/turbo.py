"""The fixed sampling grid Qwen-Image 2.1 Turbo was distilled on, as an mflux scheduler.

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


try:
    import mlx.core as mx
    from mflux.models.common.schedulers.base_scheduler import BaseScheduler
except ImportError:  # no mflux: the class is never asked for
    TurboGrid = None
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
