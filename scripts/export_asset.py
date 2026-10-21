#!/usr/bin/env python3
"""Export: one folder per asset that a game can load.

Copies the picked files of every stage that has any into out/<stem>/export/ and
writes asset.json, which says what each file is. Nothing is converted; the files
are the ones the stages already wrote.

    export/
      asset.json        what is here, and which takes it was made from
      concept.png       the concept, cut out on transparency
      2d/parts/         one PNG per piece, named by the sheet
      2d/sheets/        one strip per motion, frames left to right
      2d/sounds/        sound per piece
      2d/voices/        spoken line per piece
      stack/            sheet.png (layers bottom to top), stack.json, preview.png
      3d/               model.glb, game.glb, proxy.glb, pixel.glb, rigged.glb, psx.glb
      3d/psx/           <name>.obj + .mtl + .png: the PS1 low-poly (Y up, +Z front unless psx.axes says)
      3d/rig/           <name>.fbx (+ .png): the rigged model with every clip, for s&box ModelDoc

Engine writers (a Godot scene, a Unity prefab) can read asset.json later.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from factory_env import check_stem, repo_root, stem_dir

DIR = "export"
MANIFEST = "asset.json"
FORMAT = "pipeline-asset"
VERSION = 1

CONCEPT = ("lifted.png", "flat/source.png", "input.png", "input.jpg", "input.jpeg", "input.webp")
# Each 3D role takes the first file that exists.
MESHES: dict[str, tuple[str, ...]] = {
    # The photo bake beats Hunyuan's paint: it is the real picture where the camera saw it, the paint elsewhere.
    "model": ("textured_paint.glb", "textured_photo.glb", "pbr.glb", "textured.glb", "mesh.glb", "mesh_hero.glb", "mesh_mv.glb"),
    "game": ("textured_game.glb", "mesh_game.glb"),
    "proxy": ("mesh_proxy.glb",),
    "pixel": ("pixel.glb",),
    "rigged": ("rigged.glb", "m2m.glb", "anigen_keep.glb", "anigen.glb", "unirig.glb"),
}
SAFE = re.compile(r"[^a-z0-9._-]+")


class ExportError(ValueError):
    pass


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _flat_file(asset: Path, rel: Any, subdir: str) -> Path | None:
    """A file a sheet row names (relative to flat/), only when it sits in that subfolder."""
    if not isinstance(rel, str) or not rel.startswith(subdir + "/") or ".." in Path(rel).parts:
        return None
    p = asset / "flat" / rel
    return p if p.is_file() else None


def _name(text: Any, fallback: str) -> str:
    slug = SAFE.sub("-", str(text or "").strip().lower()).strip("-.")
    return slug or fallback


def flat_section(asset: Path, out: Path) -> dict | None:
    flat = asset / "flat"
    parts_doc = _read(flat / "parts.json") or {}
    mech = _read(flat / "mechanics.json") or {}
    rows = mech.get("parts") if isinstance(mech.get("parts"), list) else parts_doc.get("parts")
    rows = [r for r in rows or [] if isinstance(r, dict)]
    parts = []
    used: set[str] = set()
    for i, row in enumerate(rows):
        src = _flat_file(asset, row.get("file"), "parts")
        if src is None:
            continue
        base = _name(row.get("name"), str(row.get("id") or f"part-{i + 1:02d}"))
        name, n = base, 2
        while name in used:
            name, n = f"{base}-{n}", n + 1
        used.add(name)
        entry: dict[str, Any] = {
            "id": str(row.get("id") or name),
            "name": str(row.get("name") or ""),
            "image": f"2d/parts/{name}.png",
        }
        _copy(src, out / entry["image"])
        for key in ("role", "motion", "projectile", "line", "speaker"):
            if row.get(key):
                entry[key] = row[key]
        for key, sub in (("sound", "sounds"), ("voice", "voices")):
            audio = _flat_file(asset, row.get(key), sub)
            if audio is not None:
                entry[key] = f"2d/{sub}/{name}{audio.suffix}"
                _copy(audio, out / entry[key])
        parts.append(entry)
    motions = {}
    manifest = _read(flat / "sheets" / "manifest.json") or {}
    for action, row in sorted((manifest.get("actions") or {}).items()):
        if not isinstance(row, dict):
            continue
        src = _flat_file(asset, row.get("file"), "sheets")
        if src is None:
            continue
        rel = f"2d/sheets/{src.name}"
        _copy(src, out / rel)
        motions[action] = {
            "sheet": rel,
            "frames": row.get("frames"),
            "fps": row.get("fps"),
            "cell": row.get("cell"),
            "layout": "horizontal",
            "frame_layout": row.get("frame_layout"),
        }
    if not parts and not motions:
        return None
    section: dict[str, Any] = {"parts": parts, "motions": motions}
    if mech.get("note"):
        section["note"] = mech["note"]
    return section


def stack_section(asset: Path, out: Path) -> dict | None:
    src = asset / "stack"
    meta = _read(src / "stack.json")
    if not isinstance(meta, dict) or not (src / "sheet.png").is_file():
        return None
    for name in ("sheet.png", "stack.json", "preview.png"):
        if (src / name).is_file():
            _copy(src / name, out / "stack" / name)
    keep = ("resolution", "layers", "pitch", "origin", "bounds", "extents", "axis", "sheet_layout", "maker", "volume")
    section = {"sheet": "stack/sheet.png", "meta": "stack/stack.json"}
    if (src / "preview.png").is_file():
        section["preview"] = "stack/preview.png"
    section.update({k: meta[k] for k in keep if k in meta})
    return section


def mesh_section(asset: Path, out: Path) -> dict | None:
    lods = {}
    doc = _read(asset / "lods.json") or {}
    for row in doc.get("lods") or []:
        if isinstance(row, dict) and isinstance(row.get("file"), str):
            lods[row["file"]] = row
    section = {}
    for role, names in MESHES.items():
        src = next((asset / n for n in names if (asset / n).is_file() and (asset / n).stat().st_size >= 64), None)
        if src is None:
            continue
        rel = f"3d/{role}.glb"
        _copy(src, out / rel)
        entry: dict[str, Any] = {"file": rel, "from": src.name}
        info = lods.get(src.name) or {}
        for key in ("faces", "vertices", "has_uvs"):
            if info.get(key) is not None:
                entry[key] = info[key]
        section[role] = entry
    psx = psx_entry(asset, out)
    if psx:
        section["psx"] = psx
    rig = (_read(asset / "rig" / "rig.json") or {}) if "rigged" in section and section["rigged"]["from"] == "rigged.glb" else {}
    if rig:
        entry = section["rigged"]
        entry.update({k: rig[k] for k in ("class", "clips", "joints") if k in rig})
        fbx = asset / str(rig.get("fbx") or "")
        if rig.get("fbx") and fbx.is_file():
            entry["fbx"] = f"3d/rig/{fbx.name}"
            _copy(fbx, out / entry["fbx"])
            tex = fbx.with_suffix(".png")
            if tex.is_file():
                entry["fbx_texture"] = f"3d/rig/{tex.name}"
                _copy(tex, out / entry["fbx_texture"])
    return section or None


def psx_entry(asset: Path, out: Path) -> dict | None:
    """The PS1 low-poly (psx_lowpoly.py): psx.glb plus the .obj/.png pair an engine like s&box imports."""
    meta = _read(asset / "psx" / "psx.json")
    glb = asset / "psx.glb"
    if not isinstance(meta, dict) or not glb.is_file():
        return None
    entry: dict[str, Any] = {"file": "3d/psx.glb", "from": glb.name}
    _copy(glb, out / entry["file"])
    for key, suffix in (("obj", ".obj"), ("mtl", ".mtl"), ("texture", ".png")):
        src = asset / "psx" / f"{meta.get('name')}{suffix}"
        if src.is_file():
            entry[key] = f"3d/psx/{src.name}"
            _copy(src, out / entry[key])
    entry["axes"] = {"obj": meta.get("obj_axes"), "glb": meta.get("glb_axes")}
    for key in ("triangles", "atlas", "height", "size", "facing"):
        if meta.get(key) is not None:
            entry[key] = meta[key]
    entry["sampling"] = "nearest, no mips"
    return entry


def concept_image(asset: Path, out: Path) -> str:
    """The concept as a cutout on transparency: lifted.png, else the 2D still cut the way
    Stack from picture cuts it (Vision, then the key), else the picture as it is."""
    if (asset / "lifted.png").is_file():
        _copy(asset / "lifted.png", out / "concept.png")
        return "concept.png"
    if (asset / "flat" / "source.png").is_file():
        try:
            import numpy as np
            from PIL import Image

            from stack_image import clean_colors, ground, subject

            im, mask, _ = subject(asset)
            if mask.any():
                from scipy.ndimage import distance_transform_edt

                rgba = np.asarray(clean_colors(im, mask, ground(asset)).convert("RGBA")).copy()
                # Transparent pixels carry the nearest subject color, so filtering in an
                # engine does not pull the key color into the edge.
                _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
                rgba = rgba[iy, ix]
                rgba[..., 3] = np.where(mask, 255, 0)
                Image.fromarray(rgba, "RGBA").save(out / "concept.png")
                return "concept.png"
        except Exception as e:  # a still the cutters can't read still exports as it is
            print(f"export: could not cut the concept still: {e}", file=sys.stderr)
    src = next((asset / n for n in CONCEPT if (asset / n).is_file()), None)
    if src is None:
        return ""
    rel = "concept" + src.suffix.lower()
    _copy(src, out / rel)
    return rel


def build(root: Path, stem: str) -> dict:
    """Write out/<stem>/export/ from what is on disk now. The old export is replaced."""
    asset = stem_dir(root, stem)
    if not asset.is_dir():
        raise FileNotFoundError(f"no asset {stem!r}")
    out = asset / DIR
    tmp = asset / (DIR + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        doc: dict[str, Any] = {"format": FORMAT, "version": VERSION, "stem": stem, "made": int(time.time())}
        concept = concept_image(asset, tmp)
        if concept:
            doc["concept"] = concept
        prompt = asset / "flat" / "prompt.txt"
        if prompt.is_file():
            doc["prompt"] = prompt.read_text(encoding="utf-8").strip()
        for key, fn in (("2d", flat_section), ("stack", stack_section), ("3d", mesh_section)):
            section = fn(asset, tmp)
            if section:
                doc[key] = section
        if not any(k in doc for k in ("2d", "stack", "3d")):
            raise ExportError("nothing to export yet: make a 2D sheet, a sprite stack or a 3D mesh first")
        flow = _read(asset / "flow.json") or {}
        picks = {s: e.get("pick") for s, e in (flow.get("stages") or {}).items() if isinstance(e, dict) and e.get("pick") and s != DIR}
        if picks:
            doc["takes"] = picks
        (tmp / MANIFEST).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)
    files = sorted(p.relative_to(asset).as_posix() for p in out.rglob("*") if p.is_file())
    return {"ok": True, "stem": stem, "folder": str(out), "manifest": f"{DIR}/{MANIFEST}", "files": files, "asset": doc}


def _json(payload: dict, status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload, indent=2).encode("utf-8") + b"\n", "application/json; charset=utf-8"


def handle(root: Path, method: str, path: str) -> tuple[int, bytes, str] | None:
    """GET /export/<stem> reads asset.json; POST /export/<stem> rebuilds the folder. None when not ours."""
    parts = [p for p in unquote(path).split("/") if p]
    if not parts or parts[0] != "export":
        return None
    if len(parts) != 2 or method not in {"GET", "POST"}:
        return _json({"ok": False, "error": f"not found: {method} {path}"}, 404)
    try:
        stem = check_stem(parts[1])
        if method == "POST":
            return _json(build(root, stem))
        doc = _read(stem_dir(root, stem) / DIR / MANIFEST)
        if doc is None:
            return _json({"ok": False, "error": "not exported yet"}, 404)
        return _json({"ok": True, "stem": stem, "asset": doc})
    except FileNotFoundError as e:
        return _json({"ok": False, "error": str(e)}, 404)
    except ValueError as e:
        return _json({"ok": False, "error": str(e)}, 400)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", required=True, help="out/<stem>/ folder")
    p.add_argument("--root", type=Path, default=None, help="factory root that holds out/")
    args = p.parse_args()
    try:
        result = build(args.root or repo_root(), args.stem)
    except (ValueError, FileNotFoundError) as e:
        print(f"export: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ("ok", "folder", "files")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
