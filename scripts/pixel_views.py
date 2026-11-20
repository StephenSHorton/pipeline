#!/usr/bin/env python3
"""Clean a drawn pixel art sheet's four views into true pixel art again, without the image model.

The pixel art draw (POST /2d/draw with views "pixel") already cleans its views; this re-runs only the cleaning, for
a new palette size or a fixed cleaner:

    python scripts/pixel_views.py --stem car [--colours 24] [--target 32]

Reads out/<stem>/flat/views/{top,side,front,back}.png and flat/sheet.png; writes out/<stem>/flat/pixel/ and saves it
as a new take of the asset's flat stage (flow.autosave). See flat2d/pixelart.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factory_env import stem_dir  # noqa: E402
from flat2d import pixelart  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stem", required=True)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--colours", type=int, default=pixelart.PALETTE)
    ap.add_argument("--target", type=int, default=pixelart.TARGET)
    ap.add_argument("--no-save", action="store_true", help="do not record a flat take")
    args = ap.parse_args(argv)
    root = Path(args.root)
    folder = stem_dir(root, args.stem) / "flat"
    if not (folder / "views" / "top.png").is_file():
        print(f"pixel_views: no pixel art views in {folder}/views (draw with views 'pixel' first)", file=sys.stderr)
        return 2
    doc = pixelart.clean_folder(folder, colours=args.colours, target=args.target)
    if not args.no_save:
        from flow import autosave

        autosave(root, args.stem, ("flat",))
    print(json.dumps({k: doc[k] for k in ("grid", "sizes", "flaws", "notes")} | {"colours": len(doc["palette"])},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
