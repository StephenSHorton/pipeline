#!/usr/bin/env python3
"""Edit a still with the factory's local image-edit model (Qwen-Image-Edit), one result per seed.

    .venv/bin/python scripts/edit_part.py <ref.png> <out_prefix> "<prompt>" <seed> [<seed> ...]
writes <out_prefix>_<seed>.png. Used to take the wheels off a car picture and to pull one wheel out of it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flat2d import optional

ref, prefix, prompt, seeds = sys.argv[1], sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4:]]
for s in seeds:
    img = optional.on_model_thread(optional.qwen_edit_still, prompt, ref, seed=s)
    if hasattr(img, "image"):
        img = img.image
    p = f"{prefix}_{s}.png"
    img.save(p)
    print("edited", p, flush=True)
