"""A photo is a reference, never a view: the photo draw's helpers (no image model here).

reference_path(root, rel)  the picture a /2d/draw names as its reference (in/ or out/ under the root, a picture).
caption(photo)             a short subject phrase for the sheet prompt when the user gave no words (the vision
                           model, facing.ask); None when it is unavailable or says nothing usable.
likeness(photo, sheet)     does the drawn sheet show the photo's subject? One vision ask on the photo and the
                           sheet side by side, told to ignore style, colour grading, background and angle (an old
                           sepia photo against a clean drawing must not fail on colour). {same: bool|None, how,
                           answer}; how 'unavailable' is never a pass.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

CAPTION_Q = ("Name the main subject of this picture in a short phrase of at most six words, for example "
             "'a black and white dairy cow' or 'a brass oil lantern'. Answer with the phrase only.")
LIKENESS_Q = ("The left side is a photo. The right side is a drawing of four views of a subject turning in place. "
              "Is the drawn subject the same subject as in the photo: the same kind of thing, with the same shape, "
              "parts and markings? Ignore the drawing style, colour grading, background and viewing angle. "
              "Answer YES or NO.")
CAPTION_MAX = 60
PANE = 512


def reference_path(root: Path, rel: str) -> Path:
    """The reference picture, root-relative under in/ or out/. ValueError for anything else."""
    from factory_env import NO_PICTURE, require_picture

    text = str(rel or "").strip()
    if not text:
        raise ValueError("empty reference")
    root = Path(root).resolve()
    path = (root / text).resolve()
    if not any(path.is_relative_to(root / d) for d in ("in", "out")):
        raise ValueError("reference must be a picture under in/ or out/")
    if not path.is_file():
        raise FileNotFoundError(f"no picture at {text}")
    try:
        require_picture(path)
    except ValueError:
        raise ValueError(NO_PICTURE) from None
    return path


def clean_caption(text: str) -> str | None:
    line = " ".join(str(text or "").split("\n")[0].split()).strip(" \"'`*.")
    line = re.sub(r"^(the subject is|subject:|this is|it is|it's)\s+", "", line, flags=re.I)
    if not line or len(line) > CAPTION_MAX or not re.search(r"[a-z]", line, re.I):
        return None
    return line


def caption(photo: Path) -> str | None:
    import facing

    try:
        return clean_caption(facing.ask(photo, CAPTION_Q))
    except facing.VLMUnavailable:
        return None


def side_by_side(photo: Path, sheet: Image.Image) -> Image.Image:
    left = ImageOps.contain(ImageOps.exif_transpose(Image.open(photo)).convert("RGB"), (PANE, PANE))
    right = ImageOps.contain(sheet.convert("RGB"), (PANE, PANE))
    out = Image.new("RGB", (PANE * 2 + 16, PANE), (255, 255, 255))
    out.paste(left, ((PANE - left.width) // 2, (PANE - left.height) // 2))
    out.paste(right, (PANE + 16 + (PANE - right.width) // 2, (PANE - right.height) // 2))
    return out


def likeness(photo: Path, sheet: Image.Image) -> dict:
    import facing

    with tempfile.TemporaryDirectory(prefix="likeness_") as tmp:
        png = Path(tmp) / "pair.png"
        side_by_side(photo, sheet).save(png)
        try:
            answer = facing.ask(png, LIKENESS_Q)
        except facing.VLMUnavailable as e:
            return {"same": None, "how": "unavailable", "why": str(e)}
    same = facing.parse_yes(answer)
    return {"same": same, "how": "vlm" if same is not None else "unclear", "answer": answer[:80]}
