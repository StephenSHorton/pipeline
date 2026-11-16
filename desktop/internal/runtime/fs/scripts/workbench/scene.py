#!/usr/bin/env python3
"""The asset's scene file: out/<stem>/workbench/scene.json.

Milestone 1 has one object, the asset's clay, so the file mostly records where the clay and its mesh are. It is
kept now, in a subset of Clide3D's scene layout (objects with a name, a kind and a TRS transform), so the scenes and
worlds of milestone 2 extend this file rather than replace it, and the two projects can swap scenes later.

The clay itself stays in clay.json beside it (the file mesher.py reads); the scene points at it.
"""
from __future__ import annotations

import json
from pathlib import Path

FORMAT = "clide3d-subset"
VERSION = 1


def path(folder: Path) -> Path:
    return Path(folder) / "workbench" / "scene.json"


def empty(stem: str) -> dict:
    return {"format": FORMAT, "version": VERSION, "name": stem, "units": "m", "up": "Y", "objects": []}


def load(folder: Path, stem: str) -> dict:
    try:
        data = json.loads(path(folder).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty(stem)
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        return empty(stem)
    return data


def clay_object(stem: str, glb: str | None = None) -> dict:
    obj = {
        "name": stem,
        "kind": "clay",
        "clay": "clay.json",
        "transform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
    }
    if glb:
        obj["mesh"] = glb
    return obj


def record_clay(folder: Path, stem: str, glb: str | None = None) -> dict:
    """Make sure the scene holds the stem's clay object (one per asset in milestone 1); glb records its mesh,
    relative to workbench/. The object's transform is kept when it is already there."""
    data = load(folder, stem)
    objs = data["objects"]
    found = next((o for o in objs if isinstance(o, dict) and o.get("kind") == "clay"), None)
    if found is None:
        found = clay_object(stem)
        objs.append(found)
    if glb:
        found["mesh"] = glb
    p = path(folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data
