#!/usr/bin/env python3
"""Four hand-made pixel art views of a small car, for scripts/stack_views.py. No image model.

  python3 tests/fixtures/views_car.py out/views-car/flat/pixel
  python3 scripts/stack_views.py --stem views-car

The car is 12 wide, 22 long and 9 high, drawn with a 1 px dark outline on a transparent ground, in the
orientation pipeline#9 draws: top with the nose at the bottom, side with the nose at the left, front and
back as seen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = {
    "r": (196, 54, 48),  # paint
    "d": (148, 38, 40),  # paint in shade
    "g": (126, 196, 226),  # glass
    "y": (44, 42, 52),  # tyre
    "h": (246, 226, 128),  # headlight
    "t": (232, 84, 64),  # tail light
    "m": (170, 170, 180),  # bumper
}
INK = (24, 20, 30)

TOP = [  # 12 wide, 22 long, nose at the bottom
    "rmmmmmmmmmmr",
    "rtrrrrrrrrtr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrggggggggrr",
    "rrggggggggrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrggggggggrr",
    "rrggggggggrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rrrrrrrrrrrr",
    "rhhrrrrrrhhr",
    "rmmmmmmmmmmr",
]
SIDE = [  # 22 long, 9 high, nose at the left
    "......rrrrrrrrrr......",
    ".....rggggrggggrr.....",
    "....rrggggrggggrrr....",
    "...rrrggggrggggrrrr...",
    "hrrrrrrrrrrrrrrrrrrrrt",
    "rddddddddddddddddddddr",
    "mddddddddddddddddddddm",
    "..yyyy..........yyyy..",
    "..yyyy..........yyyy..",
]
FRONT = [  # 12 wide, 9 high
    "...rrrrrr...",
    "..rggggggr..",
    "..rggggggr..",
    ".rrggggggrr.",
    "rhhrrrrrrhhr",
    "rddddddddddr",
    "mmmmmmmmmmmm",
    "yyy......yyy",
    "yyy......yyy",
]
BACK = [  # 12 wide, 9 high
    "...rrrrrr...",
    "..rggggggr..",
    "..rggggggr..",
    ".rrggggggrr.",
    "rttrrrrrrttr",
    "rddddddddddr",
    "mmmmmmmmmmmm",
    "yyy......yyy",
    "yyy......yyy",
]
VIEWS = {"top": TOP, "side": SIDE, "front": FRONT, "back": BACK}


def draw(rows: list[str], margin: int = 2) -> np.ndarray:
    """The view as RGBA with a 1 px ink outline on the edge of its silhouette and a transparent margin."""
    h, w = len(rows), len(rows[0])
    mask = np.array([[ch != "." for ch in row] for row in rows])
    out = np.zeros((h, w, 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                out[y, x] = (*PALETTE[ch], 255)
    pad = np.pad(mask, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    out[mask & ~inner, :3] = INK
    return np.pad(out, ((margin, margin), (margin, margin), (0, 0)))


def write(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in VIEWS.items():
        Image.fromarray(draw(rows), "RGBA").save(folder / f"{name}.png")


if __name__ == "__main__":
    write(Path(sys.argv[1] if len(sys.argv) > 1 else "out/views-car/flat/pixel"))
