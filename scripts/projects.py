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

An asset is in one project or none ("Unsorted"): adding it to a project takes it
out of every other. Membership may name a stem whose folder is not made yet (a
job about to create it), so writes never prune; reads for display filter.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

from factory_env import check_stem, stem_dir

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
HEX = re.compile(r"^#[0-9a-f]{6}$")
DIR = ".projects"
MANIFEST = "project.json"
KINDS = ("2d", "3d")


# Every load -> change -> save holds this: the factory serves requests on threads, and a move touches two manifests.
_LOCK = threading.Lock()


class ProjectError(ValueError):
    pass


class MissingProject(ProjectError):
    """No such project (the routes answer 404, not 400)."""


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
        raise MissingProject(f"no project {slug!r}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ProjectError(f"unreadable {MANIFEST}: {e}") from e
    # the right shape or a ProjectError (skipped and logged on listing), never a TypeError that breaks GET /assets
    if not isinstance(data, dict):
        raise ProjectError(f"{MANIFEST} of {slug!r} is not an object")
    data["id"] = slug  # the folder is the id: save() writes where id points
    data.setdefault("style", _style())
    data.setdefault("assets", [])
    data.setdefault("parts", [])
    if not isinstance(data["style"], dict):
        data["style"] = _style()
    for key in ("assets", "parts"):
        if not isinstance(data[key], list):
            raise ProjectError(f"{MANIFEST} of {slug!r}: {key} is not a list")
    data["assets"] = [s for s in data["assets"] if isinstance(s, str)]
    return data


def save(root: Path, project: dict[str, Any]) -> dict[str, Any]:
    project["updated"] = time.time()
    _write(project_dir(root, project["id"]) / MANIFEST, project)
    return project


def _name(name) -> str:
    name = " ".join(str(name or "").split())[:80]
    if not name:
        raise ProjectError("empty project name")
    return name


def _unique_slug(root: Path, base: str) -> str:
    """base, else base-2, base-3: the base is trimmed so the suffix still fits the 41-char id."""
    if not project_dir(root, base).exists():
        return base
    n = 2
    while True:
        suffix = f"-{n}"
        slug = base[: 41 - len(suffix)].rstrip("-") + suffix
        if not project_dir(root, slug).exists():
            return slug
        n += 1


def create(root: Path, name: str, style_prompt: str = "", palette=None, unique: bool = False) -> dict[str, Any]:
    """unique: a taken id gets -2, -3 instead of failing (the app's New project never fails on a repeat name)."""
    name = _name(name)
    with _LOCK:
        return _create(root, name, style_prompt, palette, unique)


def _create(root: Path, name: str, style_prompt: str, palette, unique: bool) -> dict[str, Any]:
    if unique:
        # The app's New project never fails on the name: accents fold ("Café" -> cafe) and a name with no letters
        # or digits to make an id from ("牧場", "!!!") gets "project", then -2, -3; the name itself stays as typed
        try:
            slug = slugify(unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode())
        except ProjectError:
            slug = "project"
    else:
        slug = slugify(name)
    if unique:
        slug = _unique_slug(root, slug)
    elif (project_dir(root, slug) / MANIFEST).exists():
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


def _manifests(root: Path) -> list[dict[str, Any]]:
    """Every readable project, in slug order. One bad project.json is logged and skipped, never fatal."""
    base = projects_dir(root)
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and SLUG.match(p.name) and (p / MANIFEST).is_file():
            try:
                out.append(load(root, p.name))
            except ProjectError as e:
                print(f"projects: skipping {p.name}: {e}", file=sys.stderr)
    return out


def row(root: Path, project: dict[str, Any]) -> dict[str, Any]:
    """What the app shows for a project: member stems whose out/<stem>/ exists (a pending job's stem is hidden, not pruned)."""
    return {
        "id": project["id"],
        "name": project.get("name") or project["id"],
        "assets": [s for s in project["assets"] if isinstance(s, str) and _exists(root, s)],
        "created": project.get("created"),
        "updated": project.get("updated"),
    }


def _exists(root: Path, stem: str) -> bool:
    try:
        return stem_dir(root, stem).is_dir()
    except ValueError:
        return False


def index(root: Path) -> list[dict[str, Any]]:
    """GET /projects: every project's row, by name (case-insensitive)."""
    rows = [row(root, p) for p in _manifests(root)]
    return sorted(rows, key=lambda r: (str(r["name"]).casefold(), r["id"]))


def membership(root: Path) -> dict[str, str]:
    """{stem: project id} in one pass. Old multi-membership data: the first project by id wins."""
    owner: dict[str, str] = {}
    for project in _manifests(root):
        for stem in project["assets"]:
            if isinstance(stem, str):
                owner.setdefault(stem, project["id"])
    return owner


def set_style(root: Path, slug: str, prompt: str | None = None, palette=None) -> dict[str, Any]:
    with _LOCK:
        project = load(root, slug)
        style = project["style"]
        if prompt is not None:
            style["prompt"] = _style(prompt)["prompt"]
        if palette is not None:
            style["palette"] = _palette(palette)
        return save(root, project)


def rename_project(root: Path, slug: str, name: str) -> dict[str, Any]:
    """The display name only: the id (and folder) never changes, so links and memberships hold."""
    name = _name(name)
    with _LOCK:
        project = load(root, slug)
        project["name"] = name
        return save(root, project)


def delete_project(root: Path, slug: str) -> str:
    """Removes out/.projects/<slug>/ only. Its assets become Unsorted; nothing under out/<stem>/ is touched."""
    with _LOCK:
        load(root, slug)
        shutil.rmtree(project_dir(root, slug))
    return slug


def _check_stem(stem: str) -> str:
    try:
        return check_stem(stem)
    except ValueError as e:
        raise ProjectError(str(e)) from e


def _drop(root: Path, stem: str, keep: str | None = None) -> None:
    for project in _manifests(root):
        if project["id"] != keep and stem in project["assets"]:
            project["assets"] = [s for s in project["assets"] if s != stem]
            save(root, project)


def add_asset(root: Path, slug: str, stem: str, allow_missing: bool = False) -> dict[str, Any]:
    """Move the asset into this project, out of any other. allow_missing: the stem a job is about to create."""
    _check_stem(stem)
    if not allow_missing and not stem_dir(root, stem).is_dir():
        raise ProjectError(f"no asset {stem!r}")
    with _LOCK:
        return _add(root, slug, stem)


def _add(root: Path, slug: str, stem: str) -> dict[str, Any]:
    project = load(root, slug)
    _drop(root, stem, keep=slug)
    if stem not in project["assets"]:
        project["assets"].append(stem)
    return save(root, project)


def remove_asset(root: Path, slug: str, stem: str) -> dict[str, Any]:
    """Drop the asset from the project. Parts cut from it stay in the library."""
    with _LOCK:
        project = load(root, slug)
        project["assets"] = [s for s in project["assets"] if s != stem]
        return save(root, project)


def rename_stem(root: Path, old: str, new: str) -> None:
    """After out/<old>/ became out/<new>/: the membership follows. Any stale claim on new goes first (one project)."""
    with _LOCK:
        _drop(root, new)
        for project in _manifests(root):
            if old in project["assets"]:
                project["assets"] = [new if s == old else s for s in project["assets"]]
                save(root, project)


def drop_stem(root: Path, stem: str) -> None:
    """After out/<stem>/ was deleted: out of every project."""
    with _LOCK:
        _drop(root, stem)


def file_made(root: Path, stem: str, body: dict) -> None:
    """A make's first request (POST /2d/draw, /2d/drop), once accepted, files the new asset: body["project"] a
    project id, or null for none. Every make says where its asset goes, so a claim left by a make that never ran
    (filing before the job was accepted did that) can't put a later asset of that name in an old project. No
    "project" key (the 3D build after a drawing, a redraw) leaves the membership alone. Failures only log."""
    if not isinstance(body, dict) or "project" not in body:
        return
    to = body.get("project")
    try:
        if to:
            add_asset(root, str(to), stem, allow_missing=True)
        else:
            drop_stem(root, stem)
    except (ProjectError, OSError) as e:
        print(f"projects: could not file {stem} into {to!r}: {e}", file=sys.stderr)


def projects_for(root: Path, stem: str) -> list[str]:
    return [p["id"] for p in _manifests(root) if stem in p["assets"]]


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
    with _LOCK:
        return _add_flat_part(root, slug, stem, part_id, name, tags)


def _add_flat_part(root: Path, slug: str, stem: str, part_id: str, name, tags) -> dict[str, Any]:
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
    with _LOCK:
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


def _json(payload: dict, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, indent=2).encode("utf-8") + b"\n"
    return status, body, "application/json; charset=utf-8"


def _create_with(root: Path, name, assets) -> dict[str, Any]:
    """POST /projects: a new project (a repeat name gets -2), with its starting assets moved in, under one lock."""
    name = _name(name)
    if assets is None:
        assets = []
    if not isinstance(assets, list):
        raise ProjectError("assets must be a list of stems")
    stems = [_check_stem(str(s)) for s in assets]
    with _LOCK:
        project = _create(root, name, "", None, unique=True)
        for stem in stems:
            project = _add(root, project["id"], stem)
    return project


def handle(root: Path, method: str, parts: list[str], body: bytes) -> tuple[int, bytes, str] | None:
    """GET/POST /projects, GET/PATCH/DELETE /projects/<id>, POST /projects/<id>/assets,
    DELETE /projects/<id>/assets/<stem>. parts is the unquoted path split on /. None when the path is not ours."""
    if not parts or parts[0] != "projects":
        return None
    missing = _json({"ok": False, "error": f"not found: {method} /{'/'.join(parts)}"}, 404)
    try:
        payload = json.loads(body.decode("utf-8") or "{}") if body else {}
        if not isinstance(payload, dict):
            raise ProjectError("send one object")
        if len(parts) == 1:
            if method == "GET":
                return _json({"ok": True, "projects": index(root)})
            if method == "POST":
                project = _create_with(root, payload.get("name"), payload.get("assets"))
                return _json({"ok": True, "project": row(root, project)})
            return missing
        slug = check_slug(parts[1])
        if len(parts) == 2:
            if method == "GET":
                return _json({"ok": True, "project": row(root, load(root, slug))})
            if method == "PATCH":
                return _json({"ok": True, "project": row(root, rename_project(root, slug, payload.get("name")))})
            if method == "DELETE":
                return _json({"ok": True, "deleted": delete_project(root, slug)})
            return missing
        if parts[2] != "assets" or len(parts) > 4:
            return missing
        if method == "POST" and len(parts) == 3:
            stem = _check_stem(str(payload.get("stem") or ""))
            project = add_asset(root, slug, stem, allow_missing=True)
            return _json({"ok": True, "project": row(root, project), "stem": stem})
        if method == "DELETE" and len(parts) == 4:
            # Unsorting a stem that is already out of this project is a no-op, not an error: the app may race itself.
            stem = _check_stem(parts[3])
            return _json({"ok": True, "project": row(root, remove_asset(root, slug, stem)), "stem": stem})
        return missing
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return _json({"ok": False, "error": f"invalid JSON: {e}"}, 400)
    except MissingProject as e:
        return _json({"ok": False, "error": str(e)}, 404)
    except ProjectError as e:
        return _json({"ok": False, "error": str(e)}, 400)
    except OSError as e:
        return _json({"ok": False, "error": str(e)}, 500)
