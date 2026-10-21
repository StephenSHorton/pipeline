#!/usr/bin/env python3
"""Turnaround contract: one canonical photo → named views for shape + bake.

Expected files (drop into out/<stem>/views/, then lift-subject each):

  front.jpg          dead-front, even light — best geometry for people
  threequarter.jpg   3/4 — best geometry for props
  left.jpg / right.jpg
  back.jpg
  top.jpg            (props)

Shape uses front (people) or threequarter (props). Retexture uses every lifted view.
This script only checks/lists; view synthesis is Imagine image_edit from the canonical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PEOPLE = ["front", "left", "back", "threequarter"]
PROPS = ["front", "threequarter", "left", "back", "top"]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: surround.py <out/<stem>>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    views = root / "views"
    lifted = views / "lifted"
    present = sorted(p.stem for p in views.glob("*.jpg")) + sorted(p.stem for p in views.glob("*.png") if p.parent == views)
    lifted_ok = sorted(p.stem for p in lifted.glob("*.png")) if lifted.is_dir() else []
    report = {"dir": str(root), "views": present, "lifted": lifted_ok}
    shape = None
    for cand in ("front", "threequarter", "original"):
        p = lifted / f"{cand}.png"
        if p.is_file():
            shape = str(p)
            break
    report["shape_from"] = shape
    print(json.dumps(report, indent=2))
    return 0 if shape else 1


if __name__ == "__main__":
    raise SystemExit(main())
