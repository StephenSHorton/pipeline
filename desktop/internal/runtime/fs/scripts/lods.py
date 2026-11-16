#!/usr/bin/env python3
"""out/<stem>/lods.json: the finished models the app's Model tabs list, in order.

    .venv/bin/python scripts/lods.py out/S            # rebuild it from the files there (reduce.sh, generate's lods stage)

reduce.sh and generate.sh's lods stage rebuild the whole list from what the folder holds; a tool that adds one model
later (the workbench's psx) upserts just that entry with put(), so the entries the app added on its own (paint_save's
Painted, clean_mesh's Clean) survive. Both build the entry with entry(), so a PSX the workbench publishes reads
exactly like one generate.sh made.

Only finished models: the photo bake (else the paint it would have gone on) and the PS1 low-poly. The hero and
lookdev meshes are steps on the way there, not models to pick, and game / proxy are no longer made (Stephen: no
arbitrary copies at other triangle counts). An older lods.json that lists them still reads; the app hides them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# (id, candidate files, label, metadata files) in the list's order. The first candidate file that exists is the
# entry; its faces / vertices / has_uvs come from the first metadata file that exists.
SPECS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    # The photo bake is what the asset ships; with no bake (a make without the photo step), the paint it goes on.
    ("photo", ("textured_photo.glb", "textured.glb", "pbr.glb"), "Photo bake", ()),
    # PS1 low-poly: psx_lowpoly.py (generate --quality psx, or the workbench's psx tool).
    ("psx", ("psx.glb",), "PSX", ("psx/psx.json",)),
)
SPEC = {s[0]: s for s in SPECS}


def entry(out: Path, lid: str, files: tuple[str, ...] | list[str] | None = None) -> dict | None:
    """One lods.json entry for SPEC[lid] (files overrides its candidates), or None when no candidate exists."""
    _, cands, label, metas = SPEC[lid]
    for f in files if files is not None else cands:
        if not (out / f).is_file():
            continue
        item = {"id": lid, "file": f, "label": label}
        for m in metas:
            mp = out / m
            if mp.is_file():
                d = json.loads(mp.read_text())
                item["faces"] = d.get("faces", d.get("triangles"))
                item["vertices"] = d.get("vertices")
                item["has_uvs"] = d.get("has_uvs", True if "atlas" in d else None)
                break
        return item
    return None


def build(out: Path, stem: str) -> dict:
    """The whole document from the folder's files, as generate.sh's lods stage writes it."""
    lods = []
    for lid, _, _, _ in SPECS:
        item = entry(out, lid)
        if item:
            lods.append(item)
    if (out / "stack" / "stack.json").is_file():
        lods.append({"id": "stack", "file": "stack/stack.json", "label": "Sprite stack", "kind": "stack"})
    return {"stem": stem, "lods": lods}


def put(out: Path, stem: str, item: dict) -> dict:
    """Upsert one entry into out/lods.json and write it. An entry with the same id is replaced where it stands;
    a new one goes after the last entry SPECS orders before it (after photo, for psx), else first. Entries this
    module does not know (paint, clean, ...) keep their places. No lods.json yet (a make that stopped before its lods
stage): the whole list from the folder, as build() would write it, so the models already there stay in the tabs."""
    path = out / "lods.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("not an object")
    except (OSError, ValueError):
        # A list of only this entry would hide the folder's other models: the app's tabs list nothing else once
        # lods.json is non-empty. A metadata file build() cannot read leaves just this entry, as before.
        try:
            doc = build(out, stem)
        except (OSError, ValueError):
            doc = {"stem": stem, "lods": []}
    lods = list(doc.get("lods") or [])
    at = next((i for i, x in enumerate(lods) if x.get("id") == item["id"]), None)
    if at is not None:
        lods[at] = item
        lods = [x for i, x in enumerate(lods) if i == at or x.get("id") != item["id"]]
    else:
        ids = [s[0] for s in SPECS]
        before = set(ids[:ids.index(item["id"])]) if item["id"] in ids else set()
        at = max((i + 1 for i, x in enumerate(lods) if x.get("id") in before), default=0)
        lods.insert(at, item)
    doc["lods"] = lods
    doc.setdefault("stem", stem)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)  # the app reads lods.json while we write; never let it see half a file
    return doc


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out is None or not out.is_dir():
        print("usage: lods.py out/<stem>", file=sys.stderr)
        return 2
    doc = build(out, out.name)
    (out / "lods.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("reduce: wrote", out / "lods.json")
    for x in doc["lods"]:
        print(" ", x["id"], x["file"], x.get("faces"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
