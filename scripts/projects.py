"""Projects: assets that belong to one game world, plus the parts they share.

A project lives at out/.projects/<slug>/ (the dot keeps it out of the stem list):

    project.json            name, style, member stems, part index
    parts/<part-id>/        one shared part
        part.png            2D cut (RGBA), when the part came from a flat sheet
        part.glb            3D piece, when the part came from a mesh (later)
        part.json           name, tags, where it came from

Membership and the part library sit beside the per-asset out/<stem>/ contract;
nothing in an asset folder moves. A part remembers the asset, stage and take it
was cut from, so flow.py's stale tracking can say when the source has moved on.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from factory_env import check_stem, stem_dir

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
HEX = re.compile(r"^#[0-9a-f]{6}$")
DIR = ".projects"
MANIFEST = "project.json"
KINDS = ("2d", "3d")


class ProjectError(ValueError):
    pass


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")[:41].strip("-")
    if not SLUG.match(slug):
        raise ProjectError(f"cannot make a project id from {text!r}")
    return slug


def check_slug(slug: str) -> str:
    if not isinstance(slug, str) or not SLUG.match(slug):
        raise ProjectError(f"invalid project id {slug!r}")
    return slug


def projects_dir(root: Path) -> Path:
    return Path(root) / "out" / DIR


def project_dir(root: Path, slug: str) -> Path:
    return projects_dir(root) / check_slug(slug)


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _palette(raw) -> list[str]:
    out: list[str] = []
    for item in raw or []:
        text = str(item).strip().lower()
        if not text.startswith("#"):
            text = "#" + text
        if not HEX.match(text):
            raise ProjectError(f"palette color {item!r} is not #rrggbb")
        if text not in out:
            out.append(text)
    return out


def _style(prompt: str = "", palette=None) -> dict[str, Any]:
    return {"prompt": " ".join(str(prompt or "").split()), "palette": _palette(palette)}


def load(root: Path, slug: str) -> dict[str, Any]:
    path = project_dir(root, slug) / MANIFEST
    if not path.is_file():
        raise ProjectError(f"no project {slug!r}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ProjectError(f"unreadable {MANIFEST}: {e}") from e
    data.setdefault("style", _style())
    data.setdefault("assets", [])
    data.setdefault("parts", [])
    return data


def save(root: Path, project: dict[str, Any]) -> dict[str, Any]:
    project["updated"] = time.time()
    _write(project_dir(root, project["id"]) / MANIFEST, project)
    return project


def create(root: Path, name: str, style_prompt: str = "", palette=None) -> dict[str, Any]:
    name = " ".join(str(name or "").split())[:80]
    if not name:
        raise ProjectError("empty project name")
    slug = slugify(name)
    if (project_dir(root, slug) / MANIFEST).exists():
        raise ProjectError(f"project {slug!r} already exists")
    now = time.time()
    project = {
        "version": 1,
        "id": slug,
        "name": name,
        "style": _style(style_prompt, palette),
        "assets": [],
        "parts": [],
        "created": now,
    }
    return save(root, project)


def list_projects(root: Path) -> list[dict[str, Any]]:
    base = projects_dir(root)
    if not base.is_dir():
        return []
    rows = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and SLUG.match(p.name) and (p / MANIFEST).is_file():
            data = load(root, p.name)
            rows.append(
                {
                    "id": data["id"],
                    "name": data.get("name") or data["id"],
                    "assets": len(data["assets"]),
                    "parts": len(data["parts"]),
                }
            )
    return rows


def set_style(root: Path, slug: str, prompt: str | None = None, palette=None) -> dict[str, Any]:
    project = load(root, slug)
    style = project["style"]
    if prompt is not None:
        style["prompt"] = _style(prompt)["prompt"]
    if palette is not None:
        style["palette"] = _palette(palette)
    return save(root, project)


def add_asset(root: Path, slug: str, stem: str) -> dict[str, Any]:
    try:
        check_stem(stem)
    except ValueError as e:
        raise ProjectError(str(e)) from e
    if not stem_dir(root, stem).is_dir():
        raise ProjectError(f"no asset {stem!r}")
    project = load(root, slug)
    if stem not in project["assets"]:
        project["assets"].append(stem)
    return save(root, project)


def remove_asset(root: Path, slug: str, stem: str) -> dict[str, Any]:
    """Drop the asset from the project. Parts cut from it stay in the library."""
    project = load(root, slug)
    project["assets"] = [s for s in project["assets"] if s != stem]
    return save(root, project)


def projects_for(root: Path, stem: str) -> list[str]:
    return [row["id"] for row in list_projects(root) if stem in load(root, row["id"])["assets"]]


def _picked_take(folder: Path, stage: str) -> str | None:
    """The flow.py pick for a stage, when the asset has a flow.json. Read-only."""
    path = folder / "flow.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = (data.get("stages") or {}).get(stage) or {}
    pick = entry.get("pick")
    return pick if isinstance(pick, str) else None


def _tags(raw) -> list[str]:
    items = raw if isinstance(raw, list) else str(raw or "").split(",")
    out: list[str] = []
    for item in items:
        text = " ".join(str(item).lower().split())[:32]
        if text and text not in out:
            out.append(text)
    return out


def _unique_part_id(project: dict[str, Any], base: str) -> str:
    taken = {p["id"] for p in project["parts"]}
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}"[:41] in taken:
        n += 1
    return f"{base}-{n}"[:41]


def add_flat_part(
    root: Path,
    slug: str,
    stem: str,
    part_id: str,
    name: str | None = None,
    tags=None,
) -> dict[str, Any]:
    """Copy one Split crop from out/<stem>/flat/ into the project's part library."""
    project = load(root, slug)
    if stem not in project["assets"]:
        raise ProjectError(f"{stem!r} is not in project {slug!r}")
    folder = stem_dir(root, stem)
    sheet = folder / "flat" / "parts.json"
    try:
        rows = json.loads(sheet.read_text(encoding="utf-8")).get("parts") or []
    except (OSError, json.JSONDecodeError) as e:
        raise ProjectError(f"{stem!r} has no split parts") from e
    row = next((r for r in rows if r.get("id") == part_id), None)
    if row is None:
        raise ProjectError(f"{stem!r} has no part {part_id!r}")
    src = (folder / "flat" / str(row.get("file") or "")).resolve()
    if not src.is_file() or (folder / "flat").resolve() not in src.parents:
        raise ProjectError(f"part file for {part_id!r} is missing")
    label = " ".join(str(name or row.get("name") or part_id).split())[:48]
    pid = _unique_part_id(project, slugify(label) if re.search(r"[a-z0-9]", label.lower()) else part_id)
    dest = project_dir(root, slug) / "parts" / pid
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / "part.png")
    meta = {
        "id": pid,
        "name": label,
        "kind": "2d",
        "tags": _tags(tags),
        "role": row.get("role") or "other",
        "files": {"png": "part.png"},
        "from": {"stem": stem, "stage": "flat", "part": part_id, "take": _picked_take(folder, "flat")},
        "w": row.get("w"),
        "h": row.get("h"),
        "added": time.time(),
    }
    _write(dest / "part.json", meta)
    project["parts"].append({k: meta[k] for k in ("id", "name", "kind", "tags", "from")})
    save(root, project)
    return meta


def remove_part(root: Path, slug: str, part_id: str) -> dict[str, Any]:
    project = load(root, slug)
    if not any(p["id"] == part_id for p in project["parts"]):
        raise ProjectError(f"no part {part_id!r} in {slug!r}")
    project["parts"] = [p for p in project["parts"] if p["id"] != part_id]
    shutil.rmtree(project_dir(root, slug) / "parts" / check_slug(part_id), ignore_errors=True)
    return save(root, project)


def styled_prompt(project: dict[str, Any], text: str) -> str:
    """The project's style line ahead of an asset prompt, so stills share one look."""
    text = " ".join(str(text or "").split())
    style = project.get("style") or {}
    bits = []
    if style.get("prompt"):
        bits.append(style["prompt"].rstrip(".") + ".")
    if style.get("palette"):
        bits.append("Palette: " + ", ".join(style["palette"]) + ".")
    return " ".join(bits + [text]).strip()
