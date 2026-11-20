#!/usr/bin/env python3
"""Draw candidate stills for a part-by-part model (body alone, one wheel alone) with the factory's still model.

    .venv/bin/python scripts/draw_parts.py out/racecar/parts body_side "<prompt>" 2
writes <dir>/<name>_<seed>.png, one per seed."""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flat2d import optional

out, name, text, n = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
for i in range(n):
    img, seed, redrawn = optional.clean_still(optional.still_prompt(text), random.randint(1, 2**31 - 1))
    p = out / f"{name}_{seed}.png"
    img.save(p)
    print("drew", p, "redrawn", redrawn, flush=True)
