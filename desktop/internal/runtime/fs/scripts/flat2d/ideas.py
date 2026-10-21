"""Ideas: several stills from one prompt. Keep the ones you like, then use one as the concept.

out/<stem>/ideas/idea-NN.png and ideas/ideas.json. Using an idea copies it to flat/source.png,
the same place a dropped or generated still goes, so 2D, 3D and the stack all start from it.
"""
from __future__ import annotations

import io
import random
import re
from pathlib import Path

from PIL import Image

from flat2d.optional import FlatUnavailable, clean_still, still_prompt
from flat2d.parts import ground_key
from flat2d.sheet import read_json, still_backend, write_json
from flat2d.store import FlatError, flat_dir, save_source

IDEA_FILE = re.compile(r"^idea-\d{2,3}\.png$")
MAX_IDEAS = 120


def ideas_dir(root: Path, stem: str) -> Path:
    return Path(root) / "out" / stem / "ideas"


def read_ideas(root: Path, stem: str) -> dict:
    data = read_json(ideas_dir(root, stem) / "ideas.json") or {}
    items = [i for i in data.get("items") or [] if isinstance(i, dict) and IDEA_FILE.match(str(i.get("file") or ""))]
    return {"ok": True, "stem": stem, "prompt": str(data.get("prompt") or ""), "items": items}


def _write(root: Path, stem: str, data: dict) -> dict:
    write_json(ideas_dir(root, stem) / "ideas.json", {"prompt": data["prompt"], "items": data["items"]})
    return data


def _find(data: dict, file: str) -> dict:
    for item in data["items"]:
        if item["file"] == file:
            return item
    raise FlatError(f"no idea {file!r}")


def draw_idea(root: Path, stem: str, prompt: str, seed: int | None = None) -> dict:
    """One more still for the grid. A new seed each time, so the same prompt gives different pictures."""
    text = " ".join((prompt or "").split())
    if not text:
        raise FlatError("describe the thing first")
    if not still_backend():
        raise FlatUnavailable("No local image model. Install mflux.")
    data = read_ideas(root, stem)
    if len(data["items"]) >= MAX_IDEAS:
        raise FlatError("too many ideas; clear the ones you did not keep")
    seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
    image, seed, redrawn = clean_still(still_prompt(text), seed)
    used = [int(i["file"][5:-4]) for i in data["items"]]
    name = f"idea-{1 + max(used, default=0):02d}.png"
    dest = ideas_dir(root, stem) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="PNG")
    item = {"file": name, "seed": seed, "prompt": text, "keep": False}
    if redrawn:
        item["redrawn"] = redrawn
    data["items"].append(item)
    data["prompt"] = text
    _write(root, stem, data)
    return {**data, "item": item}


def keep_idea(root: Path, stem: str, file: str, keep: bool) -> dict:
    data = read_ideas(root, stem)
    _find(data, file)["keep"] = bool(keep)
    return _write(root, stem, data)


def clear_ideas(root: Path, stem: str) -> dict:
    """Delete every idea not kept and not in use."""
    data = read_ideas(root, stem)
    stay = []
    for item in data["items"]:
        if item.get("keep") or item.get("used"):
            stay.append(item)
        else:
            (ideas_dir(root, stem) / item["file"]).unlink(missing_ok=True)
    data["items"] = stay
    return _write(root, stem, data)


def use_idea(root: Path, stem: str, file: str) -> dict:
    """Make this idea the concept still. It is keyed on the ground it was drawn on, like any generated still."""
    data = read_ideas(root, stem)
    item = _find(data, file)
    src = ideas_dir(root, stem) / file
    if not src.is_file():
        raise FlatError(f"no idea {file!r}")
    buf = io.BytesIO()
    still = Image.open(src)
    still.save(buf, format="PNG")
    saved = save_source(root, stem, buf.getvalue())
    write_json(flat_dir(root, stem) / "key.json", ground_key(still))
    (flat_dir(root, stem) / "prompt.txt").write_text(item.get("prompt", "") + "\n", encoding="utf-8")
    for other in data["items"]:
        other.pop("used", None)
    item["used"] = True
    item["keep"] = True
    _write(root, stem, data)
    return {**data, "source": saved["source"], "used": file}
