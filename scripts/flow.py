"""Stage history for one asset: takes you can step back to.

The generators keep writing the out/<stem>/ contract. A take is a copy of one
stage's contract files under out/<stem>/takes/<stage>/<id>/. Picking a take
copies it back into place, so the viewer and generate.sh never know.

Stages run ideas -> concept -> flat | mesh -> stack -> export. Each take records which
upstream takes were picked when it was saved; when an upstream pick changes,
the downstream take reads as stale until it is rebuilt or re-picked.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

from factory_env import check_stem, stem_dir

STAGES = ("ideas", "concept", "flat", "mesh", "rig", "stack", "export")

UPSTREAM: dict[str, tuple[str, ...]] = {
    "ideas": (),
    "concept": ("ideas",),
    "flat": ("concept",),
    "mesh": ("concept",),
    # A rig is the mesh's skeleton, weights and baked clips (scripts/rig.py).
    "rig": ("mesh",),
    # A stack is sliced from the mesh, or built straight from the concept still.
    "stack": ("mesh", "concept"),
    # The game folder: whatever 2D, stack and 3D files were picked when it was written.
    "export": ("concept", "flat", "mesh", "rig", "stack"),
}

# Contract files per stage, relative to out/<stem>/. Globs, matched with Path.glob.
FILES: dict[str, tuple[str, ...]] = {
    "ideas": ("ideas/*",),
    "concept": (
        "input.png",
        "input.jpg",
        "input.jpeg",
        "input.webp",
        "lifted.png",
        "views/**/*",
        "flat/source.png",
        "flat/key.json",
        "flat/prompt.txt",
        # The four-view sheet (flat2d/views.py): a concept take carries its views, so picking an older take never
        # leaves a newer views.json beside another drawing's source.png (views.json also holds its sha256).
        "flat/sheet.png",
        "flat/views/*",
        "flat/views.json",
    ),
    # flat/view.json (which picture is centered) is a view setting, not part of any take.
    "flat": (
        "flat/parts.json",
        "flat/mechanics.json",
        "flat/parts/*",
        "flat/sheets/*",
        "flat/sounds/*",
        "flat/voices/*",
    ),
    "mesh": (
        "mesh*.glb",
        "textured*.glb",
        "pbr.glb",
        "pixel.glb",
        "psx.glb",
        "psx/*",
        "anigen*.glb",
        "unirig.glb",
        "m2m.glb",
        "lods.json",
        "inspect.json",
        "preview.png",
        # What generate.sh made the mesh from and how: the orientation fit, its fallbacks, the isolated views.
        "shape.json",
        "degraded.json",
        "views/lifted/*",
    ),
    # rig/template/* is a template built for this model (rig.glb, clips.glb, its reports); of rig/preview/ only the
    # contact sheet and frames.json (rig_render.py): the per-frame pictures (hundreds a rig) can be rendered again.
    "rig": ("rigged.glb", "rig/*", "rig/template/*", "rig/preview/contact_sheet.png", "rig/preview/frames.json"),
    "stack": ("stack/*",),
    "export": ("export/**/*",),
}

# Paths a stage's globs reach that belong to another stage: views/lifted/ is generate.sh's (the mesh take), not
# the concept's views/**.
EXCLUDE: dict[str, tuple[str, ...]] = {"concept": ("views/lifted/",)}

MANIFEST = "flow.json"
TAKES = "takes"


class FlowError(ValueError):
    pass


def check_stage(stage: str) -> str:
    if stage not in STAGES:
        raise FlowError(f"unknown stage {stage!r}; one of {', '.join(STAGES)}")
    return stage


def stage_files(folder: Path, stage: str) -> list[str]:
    seen: set[str] = set()
    skip = EXCLUDE.get(check_stage(stage), ())
    for pattern in FILES[stage]:
        for p in folder.glob(pattern):
            if p.is_file() and not p.name.startswith("."):
                rel = p.relative_to(folder).as_posix()
                if not rel.startswith(skip):
                    seen.add(rel)
    return sorted(seen)


INPUTS = ("input.png", "input.jpg", "input.jpeg", "input.webp")
# Written about a rig, not part of it: checking a version again must not make it a new one.
RIG_DERIVED = {"rig/inspect.json", "rig/summary.json", "rig/rig.log", "rig/render.log", "rig/frames_check.json"}
# A contact sheet is a picture of the rig: rendering it again doesn't make a new version either.
RIG_DERIVED_DIRS = ("rig/preview/",)


def rig_derived(rel: str) -> bool:
    return rel in RIG_DERIVED or rel.startswith(RIG_DERIVED_DIRS)


def defining(stage: str, rels: list[str]) -> list[str]:
    """The files that say what a take is. A 3D run writes the cutout, and copies a
    2D still to input.*; those ride along in the take but do not make a new one. A rig's
    checks and log are written about it, so they don't either."""
    if stage == "rig":
        return [r for r in rels if not rig_derived(r)] or rels
    if stage != "concept":
        return rels
    derived = {"lifted.png"}
    if "flat/source.png" in rels:
        derived.update(INPUTS)
    return [r for r in rels if r not in derived] or rels


def marks(folder: Path, stage: str, rels: list[str]) -> set[str]:
    """Fingerprints a take on disk may carry: by its defining files, or by all of them (takes saved before)."""
    return {m for m in (fingerprint(folder, defining(stage, rels)), fingerprint(folder, rels)) if m}


def fingerprint(folder: Path, rels: list[str]) -> str | None:
    """Size and mtime of each file. Takes are copied with copy2, so a picked take matches itself."""
    if not rels:
        return None
    h = hashlib.sha1()
    for rel in rels:
        st = (folder / rel).stat()
        h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def _empty() -> dict[str, Any]:
    return {"version": 1, "stages": {s: {"pick": None, "takes": []} for s in STAGES}}


def read_manifest(folder: Path) -> dict[str, Any]:
    path = folder / MANIFEST
    data = _empty()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise FlowError(f"unreadable {MANIFEST}: {e}") from e
        for stage, entry in (raw.get("stages") or {}).items():
            if stage in STAGES and isinstance(entry, dict):
                data["stages"][stage] = {
                    "pick": entry.get("pick"),
                    "takes": [t for t in entry.get("takes") or [] if isinstance(t, dict)],
                }
    return data


def write_manifest(folder: Path, data: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(folder / MANIFEST)


def _find(entry: dict[str, Any], take_id: str | None) -> dict[str, Any] | None:
    for take in entry["takes"]:
        if take.get("id") == take_id:
            return take
    return None


def _picks(data: dict[str, Any], stages: tuple[str, ...]) -> dict[str, str]:
    return {s: data["stages"][s]["pick"] for s in stages if data["stages"][s]["pick"]}


def _stale(data: dict[str, Any], take: dict[str, Any] | None) -> list[str]:
    """Upstream stages whose pick moved since this take was saved."""
    if not take:
        return []
    made_from = take.get("from") or {}
    return sorted(s for s, tid in made_from.items() if data["stages"][s]["pick"] != tid)


def _summary(path: Path) -> dict[str, Any] | None:
    """A rig's rig/summary.json (the line the versions strip shows), or None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def state(root: Path, stem: str) -> dict[str, Any]:
    folder = stem_dir(root, stem)
    if not folder.is_dir():
        raise FileNotFoundError(f"no asset {stem!r}")
    data = read_manifest(folder)
    stages = []
    for stage in STAGES:
        entry = data["stages"][stage]
        live = stage_files(folder, stage)
        mark = marks(folder, stage, live)
        picked = _find(entry, entry["pick"])
        takes = entry["takes"]
        extra: dict[str, Any] = {}
        if stage == "rig":
            takes = [{**t, "summary": _summary(folder / TAKES / "rig" / str(t.get("id")) / "rig" / "summary.json")} for t in takes]
            extra["summary"] = _summary(folder / "rig" / "summary.json")
        stages.append(
            {
                "stage": stage,
                "pick": entry["pick"],
                "takes": takes,
                "live": bool(live),
                "files": live,
                # Files on disk that no take holds yet: a fresh generate, or an asset from before takes.
                "unsaved": bool(live) and not any(t.get("print") in mark for t in entry["takes"]),
                "stale": _stale(data, picked),
                "upstream": list(UPSTREAM[stage]),
                **extra,
            }
        )
    return {"ok": True, "stem": stem, "stages": stages}


def snapshot(
    root: Path, stem: str, stage: str, note: str = "", made_from: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Save the stage's files on disk as a take and pick it. Same files as a take already held: pick that one.

    made_from narrows which upstream picks the take records, e.g. a stack built from the
    concept picture alone should not go stale when the mesh changes."""
    folder = stem_dir(root, stem)
    check_stage(stage)
    live = stage_files(folder, stage)
    if not live:
        raise FlowError(f"nothing on disk for {stage}")
    data = read_manifest(folder)
    entry = data["stages"][stage]
    found = marks(folder, stage, live)
    mark = fingerprint(folder, defining(stage, live))
    take = next((t for t in entry["takes"] if t.get("print") in found), None)
    created = take is None
    if take is not None and (take.get("files") != live or take.get("print") != mark):
        # Same take, new derived files (a fresh cutout): refresh its copy.
        _copy_into(folder, stage, take, live)
    if take is None:
        nums = [str(t.get("id", ""))[1:] for t in entry["takes"]]
        n = 1 + max((int(x) for x in nums if x.isdigit()), default=0)
        take = {
            "id": f"t{n}",
            "made": int(time.time()),
            "note": " ".join(str(note or "").split())[:200],
            "from": _picks(data, made_from if made_from is not None else UPSTREAM[stage]),
            "files": live,
            "print": mark,
        }
        dest = folder / TAKES / stage / take["id"]
        for rel in live:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / rel, dest / rel)
        entry["takes"].append(take)
    entry["pick"] = take["id"]
    write_manifest(folder, data)
    return {"ok": True, "stem": stem, "stage": stage, "take": take, "created": created}


def _copy_into(folder: Path, stage: str, take: dict[str, Any], live: list[str]) -> None:
    """Make a take's copy match the files on disk, and its file list and fingerprint with it."""
    dest = folder / TAKES / stage / take["id"]
    for rel in set(take.get("files") or []) - set(live):
        (dest / rel).unlink(missing_ok=True)
    for rel in live:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / rel, dest / rel)
    take["files"] = live
    take["print"] = fingerprint(folder, defining(stage, live))


def resave(root: Path, stem: str, stage: str, take_id: str) -> dict[str, Any]:
    """Refresh take_id's copy from the files on disk, in place (e.g. a rig checked again). Keeps id, made, note, from."""
    folder = stem_dir(root, stem)
    check_stage(stage)
    live = stage_files(folder, stage)
    if not live:
        raise FlowError(f"nothing on disk for {stage}")
    data = read_manifest(folder)
    take = _find(data["stages"][stage], take_id)
    if take is None:
        raise FlowError(f"no take {take_id!r} in {stage}")
    _copy_into(folder, stage, take, live)
    write_manifest(folder, data)
    return take


def pick(root: Path, stem: str, stage: str, take_id: str) -> dict[str, Any]:
    """Put a take's files back in the contract paths. Unsaved files on disk are saved as a take first."""
    folder = stem_dir(root, stem)
    check_stage(stage)
    data = read_manifest(folder)
    take = _find(data["stages"][stage], take_id)
    if take is None:
        raise FlowError(f"no take {take_id!r} in {stage}")
    src = folder / TAKES / stage / take_id
    missing = [rel for rel in take.get("files") or [] if not (src / rel).is_file()]
    if missing:
        raise FlowError(f"take {take_id} is missing {missing[0]}")
    live = stage_files(folder, stage)
    if live and not any(t.get("print") in marks(folder, stage, live) for t in data["stages"][stage]["takes"]):
        snapshot(root, stem, stage, note="kept before going back")
        data = read_manifest(folder)
    for rel in stage_files(folder, stage):
        (folder / rel).unlink()
    for rel in take["files"]:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, folder / rel)
    data["stages"][stage]["pick"] = take_id
    write_manifest(folder, data)
    return state(root, stem)


# 2D actions that write files, and the stage each one changes.
FLAT_WRITES = {
    "drop": ("concept",),
    "generate": ("concept",),
    "use": ("concept",),
    "split": ("flat",),
    "save": ("flat",),
    "steps": ("flat",),
    "pack": ("flat",),
    "sound": ("flat",),
    "speak": ("flat",),
}


def autosave(root: Path, stem: str, stages: tuple[str, ...], made_from: tuple[str, ...] | None = None) -> list[str]:
    """Save a take for each stage that has files, after a run wrote them. A failed save never fails the run."""
    saved = []
    for stage in stages:
        try:
            if stage_files(stem_dir(root, stem), stage):
                snapshot(root, stem, stage, made_from=made_from)
                saved.append(stage)
        except (FlowError, OSError, ValueError) as e:
            print(f"flow: could not save {stage} take for {stem}: {e}", file=sys.stderr)
    return saved


def _json(payload: dict, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, indent=2).encode("utf-8") + b"\n"
    return status, body, "application/json; charset=utf-8"


def handle(
    root: Path,
    method: str,
    path: str,
    body: bytes,
    busy: Callable[[str], bool] = lambda stem: False,
) -> tuple[int, bytes, str] | None:
    """GET /flow/<stem>, POST /flow/<stem>/snapshot, POST /flow/<stem>/pick. None when the path is not ours."""
    parts = [p for p in unquote(path).split("/") if p]
    if not parts or parts[0] != "flow":
        return None
    try:
        stem = check_stem(parts[1]) if len(parts) >= 2 else ""
    except ValueError as e:
        return _json({"ok": False, "error": str(e)}, 400)
    if not stem or len(parts) > 3:
        return _json({"ok": False, "error": f"not found: {method} {path}"}, 404)
    try:
        if method == "GET" and len(parts) == 2:
            return _json(state(root, stem))
        if method != "POST" or len(parts) != 3 or parts[2] not in {"snapshot", "pick"}:
            return _json({"ok": False, "error": f"not found: {method} {path}"}, 404)
        if busy(stem):
            return _json({"ok": False, "error": "a generate is writing this asset", "busy": True}, 409)
        payload = json.loads(body.decode("utf-8") or "{}") if body else {}
        stage = str(payload.get("stage") or "")
        if parts[2] == "snapshot":
            return _json(snapshot(root, stem, stage, str(payload.get("note") or "")))
        return _json(pick(root, stem, stage, str(payload.get("take") or "")))
    except json.JSONDecodeError as e:
        return _json({"ok": False, "error": f"invalid JSON: {e}"}, 400)
    except FileNotFoundError as e:
        return _json({"ok": False, "error": str(e)}, 404)
    except FlowError as e:
        return _json({"ok": False, "error": str(e)}, 400)
