#!/usr/bin/env python3
"""The workbench's MCP tools: name -> JSON schema, handler, and whether it is CPU work here or a factory job.

Sync tools run in this process and return their answer (clay, eyes, mesh edits, bake, psx). Job tools hand model
work to the factory on :8766 and return its job_id to poll with job_status; the factory keeps its one job slot.

Every path a caller gives is resolved under the repo root (factory_env.resolve_under_root) and every stem is
checked (check_stem), so a tool can read only the repo and write only out/<stem>/workbench/, plus the PSX that psx
publishes as generate.sh does (out/<stem>/psx.glb, psx/ and its lods.json entry; all three are mesh-stage files, so
its take holds them). Before writing into an asset it asks the factory whether a running job owns that stem and refuses if one does; after the write it asks the
factory to snapshot the mesh stage, so each change is a take that can be picked back (pick_take). If the factory is
not running at all, nothing can be racing the write: it goes ahead, and the result says no take was kept.

The renders look and compare_reference make go to a temp folder, not out/<stem>/workbench/: everything under
workbench/ belongs to the mesh stage's takes, and a look is not a change to the asset. Each call's folder is
removed by a later call once it is RENDER_KEEP_S old, so a long session does not fill the temp disk.

A take copies the whole mesh stage (tens of MB of mesh*.glb and textured*.glb on a real asset), so the JSON-only
clay_set and clay_edit keep none: clay.json rides along in the next clay_mesh's take, and a pick saves it first.
Products are written to a hidden temp name and renamed into place, so a factory job or a take never copies a
half-written GLB.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flow  # noqa: E402
import lods  # noqa: E402
from factory_env import (  # noqa: E402
    PSX_FACING,
    PSX_KEYS,
    check_stem,
    clean_psx,
    pick_glb,
    repo_root,
    require_picture,
    resolve_under_root,
    stem_dir,
)
from workbench import bake, clay, edit, scene  # noqa: E402
from workbench.narrate import Narrator  # noqa: E402
from workbench.factory_client import FactoryClient, FactoryDown  # noqa: E402

FACINGS = sorted(PSX_FACING)
# No game / proxy: those LODs are no longer made (reduce.sh cuts only lookdev).
LODS = ("psx", "mesh", "hero", "lookdev", "mv", "clay")
VIEWS = ("side", "xside", "front", "back", "top", "bottom", "q")
STAGES = list(flow.STAGES)
RENDER_KEEP_S = 600  # a look's folder outlives its answer (the path is in it) by this long
NOT_YET = ("not yet: {op} is not built. Use remove_floaters, delete_sphere, stretch, thicken_legs, shorten_legs, "
           "widen_head, trim_belly or tidy_legs, or rebuild the shape as clay.")


PLAYBOOK = Path(__file__).resolve().with_name("playbook.md")
# The playbook's rules digest sits between these markers; the MCP server sends it as its instructions (Claude Code
# cuts server instructions at 4,096 characters, so the whole playbook goes through the playbook tool).
DIGEST = ("<!-- digest -->", "<!-- /digest -->")


def playbook_text() -> str:
    try:
        return PLAYBOOK.read_text(encoding="utf-8")
    except OSError as e:
        raise FileNotFoundError(f"no playbook at {PLAYBOOK} ({e.strerror})") from None


def playbook_digest(text: str | None = None) -> str:
    """The playbook's digest section, markers dropped; "" when the file or its markers are missing."""
    try:
        text = playbook_text() if text is None else text
    except FileNotFoundError:
        return ""
    start, end = text.find(DIGEST[0]), text.find(DIGEST[1])
    if start < 0 or end < start:
        return ""
    return text[start + len(DIGEST[0]):end].strip()


class ToolError(ValueError):
    """A request the tool refuses (bad path, busy asset, nothing to work on): reported as isError."""


class ArgError(ValueError):
    """Arguments that do not match the tool's schema."""


def export_glb(mesh, dest: Path) -> None:
    """Write a GLB under a hidden name (takes skip dot files) and rename it into place: a reader sees the old file
    or the whole new one."""
    tmp = dest.with_name(f".{dest.name}.tmp")
    mesh.export(tmp, file_type="glb")
    os.replace(tmp, dest)


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[[dict], dict]
    kind: str = "sync"  # sync: answered here; job: a factory job_id to poll


# ---------------------------------------------------------------- schema checking (the subset the tools use)

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def _is(kind: str, value: Any) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "null":
        return value is None
    return isinstance(value, _TYPES[kind])


def check(schema: dict, value: Any, where: str = "arguments") -> list[str]:
    """Problems with value against schema: type, required, properties, additionalProperties, enum, items,
    minItems/maxItems, minimum/maximum. Enough for the tools here; no jsonschema dependency."""
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(_is(k, value) for k in kinds):
            return [f"{where} must be {' or '.join(kinds)}"]
    errs: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{where} must be one of {', '.join(map(str, schema['enum']))}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errs.append(f"{where}.{key} is required")
        for key, v in value.items():
            if key in props:
                errs += check(props[key], v, f"{where}.{key}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{where}.{key} is not an argument (one of {', '.join(props) or 'none'})")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{where} needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{where} takes at most {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += check(schema["items"], v, f"{where}[{i}]")
    if _is("number", value):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{where} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{where} must be at most {schema['maximum']}")
    return errs


def plain(obj: Any) -> Any:
    """JSON-ready: numpy scalars and arrays, Paths and tuples turned into their plain forms."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------- schema pieces

def _obj(props: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


STEM = {"type": "string", "description": "the asset's folder name under out/"}
GLB = {"type": "string", "description": "a .glb path relative to the repo root, e.g. out/mlg-sheep/psx.glb"}
LOD = {"type": "string", "enum": list(LODS),
       "description": "which of the asset's models a stem means (default psx.glb, then mesh.glb ...; clay = "
                      "workbench/clay.glb)"}
VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
FACING = {"type": "string", "enum": FACINGS,
          "description": "where the subject looks in the picture: camera (a front view), right / left (side-on, "
                         "head toward image right / left), away"}
FILL = {"type": "string", "enum": ["photo", "none"],
        "description": "photo (default): texels no photo reached get a triplanar swatch of the photo's subject; "
                       "none: they keep the base texture"}
BLOB = {
    "type": "object",
    "description": "a clay blob (Clide3D field names). form: {type: Sphere|Ellipsoid|Capsule|Box|Torus|Cylinder, "
                   "and its dims: Sphere radius; Ellipsoid radii [x,y,z]; Capsule radius, height; Box size [x,y,z], "
                   "rounding; Torus major, minor; Cylinder radius, height (along local Y), rounding (rim radius, m), "
                   "bulge (barrel: radius grows by bulge*radius at mid-height, -0.5..1)}. op add|carve, blend "
                   "(smooth-min radius, m), color [r,g,b] 0..1 or 0..255 or '#rrggbb', mirror (also across X=0), "
                   "translation [x,y,z] m, rotation [x,y,z] Euler XYZ degrees, noise {amp, scale, seed, octaves} "
                   "(surface wobble in the blob's frame; amp is clamped to keep it a distance). Y is up, +Z is the "
                   "front. For ground contact carve a wide Box under the floor (its blend rounds the edge).",
    "properties": {
        "name": {"type": "string"},
        "form": {"type": "object"},
        "op": {"type": "string", "enum": list(clay.OPS)},
        "blend": {"type": "number", "minimum": 0},
        "color": {"type": ["array", "string"]},
        "mirror": {"type": "boolean"},
        "translation": VEC3,
        "rotation": VEC3,
        "noise": {"type": ["object", "null"],
                  "description": "amp: most the surface moves (m); scale: feature size (m), keep it at least 4 grid "
                                 "cells; seed: integer; octaves 1-4 (default 2). amp 0 or null turns it off.",
                  "properties": {"amp": {"type": "number", "minimum": 0},
                                 "scale": {"type": "number", "minimum": 0},   # clay refuses 0 itself
                                 "seed": {"type": "integer"},
                                 "octaves": {"type": "integer", "minimum": 1, "maximum": 4}},
                  "additionalProperties": False},
    },
    "required": ["name", "form"],
    "additionalProperties": False,
}


class Workbench:
    def __init__(self, root: Path | None = None, factory: FactoryClient | None = None):
        self.root = Path(root).resolve() if root else repo_root()
        self.factory = factory or FactoryClient()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._renders = Path(tempfile.gettempdir()) / f"workbench_renders_{os.getpid()}"
        self.tools: dict[str, Tool] = {t.name: t for t in self._registry()}
        # Every call is a line in the app's activity feed, so Stephen can watch the work as it happens.
        self.narrator = Narrator(self.root, getattr(self.factory, "activity", None) or (lambda entry: None))

    # ------------------------------------------------------------ calling

    def call(self, name: str, args: dict | None) -> dict:
        """Run a tool. KeyError: no such tool. ArgError: arguments break the schema. ToolError (and ValueError,
        FileNotFoundError, BakeError): the tool refused or failed; the message says why."""
        tool = self.tools[name]
        args = {} if args is None else args
        errs = check(tool.schema, args)
        if errs:
            raise ArgError("; ".join(errs))
        eid = self.narrator.start(name, args)
        try:
            out = plain(tool.fn(args))
        except Exception as e:
            self.narrator.end(eid, name, args, None, error_text(e))
            raise
        self.narrator.end(eid, name, args, out)
        return out

    def listing(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "inputSchema": t.schema} for t in self.tools.values()]

    # ------------------------------------------------------------ paths

    def rel(self, p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    def folder(self, stem: str, must_exist: bool = True) -> Path:
        d = stem_dir(self.root, check_stem(stem))
        if must_exist and not d.is_dir():
            raise ToolError(f"no asset {stem!r} (no out/{stem}/)")
        return d

    def bench(self, stem: str) -> Path:
        return self.folder(stem, must_exist=False) / "workbench"

    def path(self, raw: str, suffixes: tuple[str, ...] | None = None) -> Path:
        p = resolve_under_root(self.root, raw)
        if suffixes and p.suffix.lower() not in suffixes:
            raise ToolError(f"{raw}: must be a {' or '.join(suffixes)} file")
        if not p.is_file():
            raise ToolError(f"no file at {raw}")
        return p

    def picture(self, raw: str) -> Path:
        p = resolve_under_root(self.root, raw)
        try:
            return require_picture(p)
        except ValueError:
            raise ToolError(f"{raw}: not a picture (jpg, png or webp) under the repo") from None

    def cutout(self, raw: str) -> Path:
        """A picture with the subject cut out. An opaque one would make its whole frame the silhouette that the bake
        fits a camera to, and paint the background onto the model."""
        from PIL import Image

        p = self.picture(raw)
        with Image.open(p) as im:
            alpha = im.getchannel("A") if "A" in im.getbands() else None
            if alpha is None and im.mode == "P" and "transparency" in im.info:
                alpha = im.convert("RGBA").getchannel("A")
            opaque = alpha is None or alpha.getextrema()[0] >= 250
        if opaque:
            raise ToolError(f"{raw}: no transparent background to cut the subject from; bake a cutout such as "
                            "out/<stem>/lifted.png or out/<stem>/views/lifted/*.png")
        return p

    def target(self, args: dict) -> Path:
        """The GLB a look/measure/compare is about: glb when given, else the stem's model by lod."""
        if args.get("glb"):
            return self.path(args["glb"], (".glb",))
        if not args.get("stem"):
            raise ToolError("give stem or glb")
        stem = args["stem"]
        folder = self.folder(stem)
        lod = args.get("lod")
        if lod == "clay":
            p = folder / "workbench" / "clay.glb"
            if not p.is_file():
                raise ToolError(f"{stem} has no workbench/clay.glb yet (clay_mesh makes it)")
            return p
        p = pick_glb(self.root, stem, lod) if lod else (pick_glb(self.root, stem, "psx") or pick_glb(self.root, stem))
        if p is None:
            raise ToolError(f"{stem} has no model{' for lod ' + lod if lod else ''}")
        return p

    def numbered(self, folder: Path, prefix: str) -> Path:
        nums = [int(m.group(1)) for p in folder.glob(f"{prefix}_*.glb")
                if (m := re.fullmatch(rf"{prefix}_(\d+)\.glb", p.name))]
        return folder / f"{prefix}_{1 + max(nums, default=0)}.glb"

    # ------------------------------------------------------------ the write guard

    def _lock(self, stem: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(stem, threading.Lock())

    def write(self, stem: str, note: str, fn: Callable[[], dict], take: bool = True) -> dict:
        """Run fn (a write into out/<stem>/workbench/) only when no factory job owns the stem, then keep the result
        as a mesh take (unless take is False). One workbench write per stem at a time: fn reads what it changes
        itself, under the lock, so two callers editing one stem cannot drop each other's change."""
        check_stem(stem)
        with self._lock(stem):
            busy = self.factory.writing(stem)
            if busy:
                raise ToolError(f"a factory job is writing {stem}; wait for it (job_status) or cancel it first")
            out = fn()
            if not take:
                out["take"] = {"ok": False, "skipped": "a JSON-only change keeps no take of its own; the next "
                               "clay_mesh's take carries it"}
                return out
            if busy is None:
                out["take"] = {"ok": False, "error": "factory not running: wrote without its job check, kept no take"}
                return out
            try:
                snap = self.factory.snapshot(stem, "mesh", note)
            except FactoryDown as e:
                snap = {"ok": False, "error": str(e)}
            if snap.get("ok"):
                take = snap.get("take") or {}
                out["take"] = {"ok": True, "stage": "mesh", "id": take.get("id"), "created": snap.get("created")}
            else:
                out["take"] = {"ok": False, "error": snap.get("error") or "snapshot failed"}
            return out

    def job(self, fn: Callable[[], dict]) -> dict:
        try:
            out = fn()
        except FactoryDown as e:
            raise ToolError(f"{e}. Start the app (or .venv/bin/python scripts/factory_http.py) first.") from None
        if not out.get("ok", True):
            raise ToolError(out.get("error") or f"the factory refused ({out.get('http_status')})")
        return out

    def render_dir(self) -> Path:
        """A fresh folder for one look or compare. Older ones past RENDER_KEEP_S go: their PNG went out as base64."""
        self._renders.mkdir(parents=True, exist_ok=True)
        for old in self._renders.iterdir():
            try:
                if time.time() - old.stat().st_mtime > RENDER_KEEP_S:
                    shutil.rmtree(old, ignore_errors=True)
            except OSError:
                pass
        return Path(tempfile.mkdtemp(dir=self._renders))

    # ------------------------------------------------------------ eyes

    def asset_info(self, a: dict) -> dict:
        stem = a["stem"]
        folder = self.folder(stem)
        files = []
        for p in sorted(folder.rglob("*")):
            rel = p.relative_to(folder).as_posix()
            if p.is_file() and not rel.startswith((flow.TAKES + "/", ".")) and "/." not in rel:
                if "/" in rel and not rel.startswith(("workbench/", "psx/", "flat/", "views/")):
                    continue
                files.append({"path": rel, "bytes": p.stat().st_size})
        docs = {}
        for key, rel in (("shape", "shape.json"), ("degraded", "degraded.json"), ("fit", "textured_photo.fit.json"),
                         ("psx", "psx/psx.json"), ("intent", "intent.json"), ("scene", "workbench/scene.json")):
            try:
                docs[key] = json.loads((folder / rel).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                docs[key] = None
        try:
            takes = {s["stage"]: {"pick": s.get("pick"), "takes": [t.get("id") for t in s.get("takes") or []]}
                     for s in flow.state(self.root, stem)["stages"]}
        except Exception as e:  # noqa: BLE001 - a broken manifest is a fact to report
            takes = {"error": str(e)}
        model = pick_glb(self.root, stem, "psx") or pick_glb(self.root, stem)
        return {"stem": stem, "folder": self.rel(folder), "model": self.rel(model) if model else None,
                "files": files, **docs, "takes": takes}

    def look(self, a: dict) -> dict:
        from workbench import eyes

        glb = self.target(a)
        views = a.get("views") or list(eyes.LOOK_VIEWS)
        out = self.render_dir() / f"{glb.stem}_look.png"
        sampling = a.get("sampling") or "pixel"
        png = eyes.look(glb, views, int(a.get("size") or 384), out, sampling)
        return {"glb": self.rel(glb), "views": views, "sampling": sampling, "png": png, "_image": png}

    def measure(self, a: dict) -> dict:
        from workbench import eyes

        glb = self.target(a)
        out = eyes.measure(glb, toward=None if a.get("auto_head") else eyes.FORWARD)
        return {"glb": self.rel(glb), **out}

    def compare_reference(self, a: dict) -> dict:
        from workbench import eyes

        glb = self.target(a)
        image = self.picture(a["image"])
        out = self.render_dir() / f"{glb.stem}_compare.png"
        res = eyes.compare_reference(glb, image, a.get("facing"), out)
        return {"glb": self.rel(glb), "image": self.rel(image), **res, "_image": res["png"]}

    def review(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        folder = self.folder(stem)
        args = {k: self.path(a[k], (".glb",)) for k in ("glb", "bake", "paint") if a.get(k)}
        return self.write(stem, "review", lambda: self._review(folder, **args), take=False)

    def _review(self, folder: Path, **kw) -> dict:
        """workbench/review.py into <folder>/workbench/review/; the answer drops the critique list (the playbook
        has it) and attaches the sheet."""
        from workbench import review

        res = review.review(folder, **kw)
        out = {k: v for k, v in res.items() if k != "critiques"}
        out["model"], out["sheet"] = self.rel(Path(res["model"])), self.rel(Path(res["sheet"]))
        out["_image"] = res["sheet"]
        return out

    def rig_review(self, a: dict) -> dict:
        from workbench import rigeyes

        stem = check_stem(a["stem"])
        folder = self.folder(stem)
        heat = a.get("heat")
        if isinstance(heat, str) and heat != "flagged":
            raise ArgError('heat must be a list of bone names or "flagged"')
        # The heat sheet goes where look's do (a temp folder swept later): the MCP answer attaches it after this
        # returns. The frames it is tiled from are removed before (unless keep_images).
        try:
            res = rigeyes.review(folder, float(a.get("threshold") or rigeyes.THRESHOLD), a.get("clips"), heat or None,
                                 bool(a.get("keep_images")), self.render_dir() if heat else None)
        except rigeyes.RigError as e:
            raise ToolError(str(e)) from None
        res["glb"] = self.rel(res["glb"])
        if res.get("sheet"):
            res["_image"] = str(res["sheet"])
        return {"stem": stem, **res}

    def rig_preview(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        self.folder(stem)
        argv = [sys.executable, str(Path(__file__).with_name("rig_preview.py")), stem, "--root", str(self.root)]
        for k in ("class", "plan"):
            if a.get(k):
                argv += [f"--{k}", str(a[k])]
        # its own process: it swaps rig.py's skin step for a stop, which must never touch this server's modules
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=300, cwd=self.root)
        try:
            got = json.loads(proc.stdout)
        except ValueError:
            raise ToolError("rig_preview failed: " + (proc.stderr or proc.stdout)[-400:]) from None
        if not got.get("joints"):
            raise ToolError("rig_preview stopped before the skeleton: " + proc.stderr[-400:])
        return {"stem": stem, **got}

    def playbook(self, a: dict) -> dict:
        return {"playbook": playbook_text()}

    # ------------------------------------------------------------ clay

    def _clay(self, stem: str) -> list:
        p = self.bench(stem) / "clay.json"
        if not p.is_file():
            return []
        return clay.load(p)

    def _save_clay(self, stem: str, blobs: list) -> dict:
        bench = self.bench(stem)
        clay.save(blobs, bench / "clay.json")
        scene.record_clay(bench.parent, stem)
        return {"stem": stem, "clay": self.rel(bench / "clay.json"), "blobs": [b.to_dict() for b in blobs]}

    def clay_get(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        p = self.bench(stem) / "clay.json"
        blobs = self._clay(stem)
        return {"stem": stem, "exists": p.is_file(), "clay": self.rel(p), "blobs": [b.to_dict() for b in blobs],
                "forms": {k: list(v) for k, v in clay.FORMS.items()}}

    def clay_set(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        new = clay.from_json(a["blobs"])  # ValueError: a bad blob, before anything is written

        def run() -> dict:
            if a.get("replace"):
                blobs = new
            else:
                by_name = {b.name: b for b in new}
                blobs = [by_name.pop(b.name, b) for b in self._clay(stem)] + list(by_name.values())
            return self._save_clay(stem, blobs)

        return self.write(stem, "workbench clay_set", run, take=False)

    def clay_edit(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        patch = dict(a["patch"])
        remove = bool(patch.pop("remove", False))
        form = patch.pop("form", None)
        if remove and (patch or form is not None):
            raise ToolError("remove takes no other fields")
        if form is not None and not isinstance(form, dict):
            raise ToolError("patch.form must be an object")

        def run() -> dict:
            blobs = self._clay(stem)
            names = [b.name for b in blobs]
            if a["name"] not in names:
                raise ToolError(f"no blob {a['name']!r} in {stem}'s clay (has: {', '.join(names) or 'none'})")
            i = names.index(a["name"])
            if remove:
                del blobs[i]
            else:
                d = blobs[i].to_dict()
                if form is not None:
                    same = form.get("type", d["form"]["type"]) == d["form"]["type"]
                    d["form"] = {**d["form"], **form} if same else dict(form)
                    d["form"].pop("note", None)
                if isinstance(patch.get("noise"), dict) and d.get("noise"):
                    # Like form: {"amp": 0.02} changes the amp and keeps the seed the bale was tuned with.
                    patch["noise"] = {**d["noise"], **patch["noise"]}
                d.update(patch)
                blobs[i] = clay.Blob.from_dict(d)
                clay.from_json([b.to_dict() for b in blobs])  # a rename onto another blob's name
            return self._save_clay(stem, blobs)

        return self.write(stem, f"workbench clay_edit {a['name']}", run, take=False)

    def clay_mesh(self, a: dict) -> dict:
        from workbench import mesher

        stem = check_stem(a["stem"])
        bench = self.bench(stem)

        def run() -> dict:
            blobs = self._clay(stem)
            if not blobs:
                raise ToolError(f"{stem} has no clay yet (clay_set first)")
            info = mesher.clay_glb(blobs, bench / "clay.glb", grid=int(a.get("grid") or 96), height=a.get("height"),
                                   texture=int(a.get("texture") or 1024))
            scene.record_clay(bench.parent, stem, "clay.glb")
            info["path"] = self.rel(bench / "clay.glb")
            return {"stem": stem, **info}

        return self.write(stem, "workbench clay_mesh", run)

    # ------------------------------------------------------------ mesh edits, bake, psx

    def mesh_edit(self, a: dict) -> dict:
        import trimesh
        import clean_mesh

        stem = check_stem(a["stem"])
        op = a["op"]
        params = a.get("params") or {}
        if op in ("straighten_head", "symmetrize", "nudge"):
            raise ToolError(NOT_YET.format(op=op))
        src = self.path(a["src"], (".glb",))
        mesh = clean_mesh._as_mesh(trimesh.load(src, process=False))
        if op == "remove_floaters":
            errs = check(_obj({"keep": {"type": "integer", "minimum": 1}, "min_faces": {"type": "integer", "minimum": 0}}),
                         params, "params")
            if errs:
                raise ArgError("; ".join(errs))
            out_mesh, info = clean_mesh.remove_floaters(mesh, int(params.get("keep", 1)), int(params.get("min_faces", 0)))
        elif op == "stretch":
            # Proportions: the drawn views can squash a long animal into a square, so its length needs giving back.
            errs = check(_obj({"axis": {"type": "string", "enum": ["x", "y", "z"]},
                               "factor": {"type": "number", "minimum": 0.25, "maximum": 4}}, ("axis", "factor")),
                         params, "params")
            if errs:
                raise ArgError("; ".join(errs))
            i = "xyz".index(params["axis"])
            v = np.asarray(mesh.vertices, float).copy()
            lo, hi = v[:, i].min(), v[:, i].max()
            # Height grows up from the floor; length and width about the middle, so it stays centred.
            pivot = lo if i == 1 else (lo + hi) / 2
            v[:, i] = pivot + (v[:, i] - pivot) * float(params["factor"])
            out_mesh = mesh.copy()
            out_mesh.vertices = v
            info = {"axis": params["axis"], "factor": float(params["factor"]),
                    "extents_before": [round(float(x), 4) for x in mesh.extents],
                    "extents_after": [round(float(x), 4) for x in out_mesh.extents]}
        elif op in ("close", "gaps"):
            # Holes and parts the drawings put in however they were asked (round 3: the jerry can's handle).
            from workbench import solid
            sphere = _obj({"center": VEC3, "radius": {"type": "number", "minimum": 0}}, ("center", "radius"))
            errs = check(_obj({"center": VEC3, "radius": {"type": "number", "minimum": 0},
                               "size": {"type": "number", "minimum": 0},
                               "carve": {"type": "array", "items": sphere}}), params, "params")
            if not errs and ("center" in params) != ("radius" in params):
                errs = ["params: center and radius go together"]
            if errs:
                raise ArgError("; ".join(errs))
            if op == "gaps":
                return {"stem": stem, "op": op, "src": self.rel(src),
                        "gaps": solid.gaps(mesh, float(params.get("size", 0)))}
            try:
                out_mesh, info = solid.solidify(mesh, True, float(params.get("size", 0)), params.get("center"),
                                                params.get("radius"), params.get("carve"))
            except solid.SolidError as e:
                raise ToolError(f"close: {e}") from None
        elif op in edit.OPS:
            # Proportions from a critique (the sheep: legs too thin, too tall, head too narrow). Only vertices move.
            frac = {"type": "number", "minimum": 0.02, "maximum": 0.98}
            spec = {"thicken_legs": ({"factor": {"type": "number", "minimum": 1, "maximum": 3}, "top": frac},
                                     ("factor",)),
                    "shorten_legs": ({"factor": {"type": "number", "minimum": 0.4, "maximum": 1}, "top": frac},
                                     ("factor",)),
                    "widen_head": ({"factor": {"type": "number", "minimum": 1, "maximum": 2}, "start": frac,
                                    "ramp": {"type": "number", "minimum": 0.01, "maximum": 0.5},
                                    "head": {"type": "string", "enum": sorted(edit.AXES)}}, ("factor",)),
                    "trim_belly": ({"line": frac, "blend": {"type": "number", "minimum": 0.005, "maximum": 0.2},
                                    "keep_legs": {"type": "boolean"}}, ()),
                    "tidy_legs": ({"thickness": {"type": "number", "minimum": 0.5, "maximum": 3},
                                   "straighten": {"type": "number", "minimum": 0, "maximum": 1},
                                   "smooth": {"type": "number", "minimum": 0, "maximum": 1}, "top": frac}, ())}[op]
            errs = check(_obj(*spec), params, "params")
            if errs:
                raise ArgError("; ".join(errs))
            try:
                out_mesh, info = edit.apply(mesh, op, **{k: params[k] for k in spec[0] if k in params})
            except edit.EditError as e:
                raise ToolError(f"{op}: {e}") from None
            info.pop("op", None)
        else:
            errs = check(_obj({"center": VEC3, "radius": {"type": "number", "minimum": 0}}, ("center", "radius")),
                         params, "params")
            if errs:
                raise ArgError("; ".join(errs))
            out_mesh, info = clean_mesh.delete_sphere(mesh, np.asarray(params["center"], float), float(params["radius"]))
        bench = self.bench(stem)

        def run() -> dict:
            bench.mkdir(parents=True, exist_ok=True)
            dest = self.numbered(bench, "edit")
            export_glb(out_mesh, dest)
            return {"stem": stem, "op": op, "src": self.rel(src), "glb": self.rel(dest), **info}

        return self.write(stem, f"workbench mesh_edit {op}", run)

    def bake_photo(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        glb = self.path(a["glb"], (".glb",))
        views = a["views"]
        if any(v.get("facing") for v in views[1:]):
            raise ToolError("only the first view's facing can be stated; name the others' files front/back/left/"
                            "right/side, which is how retexture reads them")
        pics = [self.cutout(v["path"]) for v in views]
        bench = self.bench(stem)
        facing = views[0].get("facing")
        # Every texel from the photo unless asked otherwise: a front-on photo also mirrors onto the far end, and
        # fill paints the photo's own swatch on whatever still has only the base (Stephen's call on the hay bale).
        ends = a.get("mirror_ends", True) is not False and bake.end_on(pics[0], facing)
        mirror = bool(a.get("mirror")) or ends
        filling = (a.get("fill") or "photo") == "photo"
        # delight true: the photo's own lighting out before it bakes (and before fill takes its swatch), so a side in
        # shadow in the picture does not bake in as a dark patch (the hay bale). Off unless asked, as in generate.sh:
        # it cannot tell an evenly lit dark material (a cow's patches, a tyre) from shade, and would lift it to grey.
        delit = bool(a.get("delight"))

        def run() -> dict:
            bench.mkdir(parents=True, exist_ok=True)
            dest = self.numbered(bench, "bake")
            info = bake.retexture(self.root, glb, dest, pics, facing, mirror,
                                  a.get("erode"), a.get("keep"), bool(a.get("first_wins", True)), ends, delit)
            info["mirror"] = mirror
            info["mirror_ends"] = ends
            info["delight"] = delit
            if filling:
                info["fill"] = bake.fill(self.root, dest, pics[0], dest, a.get("keep"), delit)
            info["glb"] = self.rel(dest)
            return {"stem": stem, "src": self.rel(glb), "views": [self.rel(p) for p in pics], **info}

        return self.write(stem, "workbench bake_photo", run)

    def bake_wrap(self, a: dict) -> dict:
        from workbench import wrap

        stem = check_stem(a["stem"])
        glb = self.path(a["glb"], (".glb",))
        pic = self.cutout(a["picture"])
        bench = self.bench(stem)

        def run() -> dict:
            bench.mkdir(parents=True, exist_ok=True)
            dest = self.numbered(bench, "wrap")
            info = wrap.wrap(glb, pic, dest, bool(a.get("flip")))
            return {"stem": stem, "src": self.rel(glb), **info, "glb": self.rel(dest), "picture": self.rel(pic)}

        return self.write(stem, "workbench bake_wrap", run)

    def psx(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        glb = self.path(a["glb"], (".glb",))
        name = a.get("name") or re.sub(r"[^A-Za-z0-9_-]+", "_", stem)[:64]
        if a.get("legs") == "clean" and (a.get("leg_weight") or 1) != 1:
            raise ArgError("leg_weight weights the legs the cut keeps and legs clean replaces them: give one")
        clean_psx({"name": name})
        folder = self.folder(stem, must_exist=False)

        def run() -> dict:
            # Published where generate.sh's PS1 step puts it (psx.glb, psx/, the lods.json entry), so the app's
            # Model tabs and pick_glb(stem, "psx") show this model, not an older one beside a workbench copy.
            # psx_lowpoly writes into a hidden folder first and the files move in when it is done: a take or the
            # app never reads half a GLB, and a failed cut leaves the published PSX as it was.
            folder.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=".psx_", dir=folder))
            try:
                report = bake.psx(self.root, glb, tmp / "psx", tmp / "psx.glb", name, float(a["height"]),
                                  a["facing"], a.get("faces"), a.get("atlas"),
                                  **({"head_weight": a["head_weight"]} if a.get("head_weight") else {}),
                                  **({"leg_weight": a["leg_weight"]} if a.get("leg_weight") else {}),
                                  **{k: a[k] for k in ("legs", "leg_sides", "leg_scale", "leg_taper") if k in a})
                # The cut takes tens of seconds and write() checked the stem only before it. A generate started
                # meanwhile writes these same files (psx.glb, psx/, lods.json); moving ours in would mix the two.
                # Asked again here the window shrinks to a few renames; the finally drops the cut, the PSX stays.
                if self.factory.writing(stem):
                    raise ToolError(f"a factory job started writing {stem} during the PSX cut; the cut was dropped "
                                    "and the published PSX is as it was. Wait for the job (job_status), then run "
                                    "psx again")
                out_dir = folder / "psx"
                if out_dir.is_dir():  # generate.sh clears psx/ before its cut too: no other name's files linger
                    for p in out_dir.iterdir():
                        if p.is_file():
                            p.unlink()
                out_dir.mkdir(exist_ok=True)
                for p in (tmp / "psx").iterdir():
                    if p.is_file():
                        os.replace(p, out_dir / p.name)
                os.replace(tmp / "psx.glb", folder / "psx.glb")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            # Earlier workbench runs published under workbench/; leaving them would keep two PSX models to mix up.
            bench = folder / "workbench"
            (bench / "psx.glb").unlink(missing_ok=True)
            shutil.rmtree(bench / "psx", ignore_errors=True)
            item = lods.entry(folder, "psx")
            lods.put(folder, stem, item)
            # Every cut is reviewed before anyone sees it: the numbers and the sheet come back with the answer.
            try:
                rev = self._review(folder)
            except Exception as e:  # noqa: BLE001 - a failed review must not lose the published cut
                rev = {"verdict": "review failed", "error": error_text(e)}
            # psx.json names its files relative to the folder; the answer gives repo paths.
            out = {"stem": stem, **report, "src": self.rel(glb), "glb": self.rel(folder / "psx.glb"),
                   "folder": self.rel(out_dir), "obj": self.rel(out_dir / report["obj"]),
                   "texture": self.rel(out_dir / report["texture"]), "lod": item,
                   "review": {k: v for k, v in rev.items() if k != "_image"}}
            if rev.get("_image"):
                out["_image"] = rev["_image"]
            return out

        return self.write(stem, "workbench psx", run)

    # ------------------------------------------------------------ factory jobs

    def draw_views(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        image = self.picture(a["image"])
        body = {"kind": "flat2d", "stem": stem, "prompt": a["prompt"], "reference": self.rel(image),
                "intent": a.get("intent") or {"output": "ps1"}}
        out = self.job(lambda: self.factory.draw(body))
        return {**out, "next": "poll job_status with this job_id; the views land in out/<stem>/flat/views.json"}

    def generate(self, a: dict) -> dict:
        image = self.picture(a["image"])
        body: dict[str, Any] = {"image": self.rel(image), "quality": a.get("quality") or "textured"}
        if a.get("stem"):
            body["stem"] = check_stem(a["stem"])
        if a.get("mesh"):
            mesh = self.path(a["mesh"], (".glb",))
            body["mesh"] = self.rel(mesh)
        if "paint" in a:
            body["paint"] = bool(a["paint"])
        # The workbench fills by default (every texel from the photo); a draft has no bake to fill.
        fill = a.get("fill") or "photo"
        if body["quality"] != "draft":
            body["fill"] = fill
            # Not de-lit unless asked (generate.sh --delight on), as bake_photo; a draft has no bake to de-light.
            body["delight"] = bool(a.get("delight"))
        if "mirror_ends" in a:
            body["mirror_ends"] = bool(a["mirror_ends"])
        for key in ("prompt", "poly", "seed", "intent"):
            if a.get(key) not in (None, ""):
                body[key] = a[key]
        if a.get("images"):
            body["images"] = [self.rel(self.picture(p)) for p in a["images"]]
        for key, value in clean_psx(a.get("psx")).items():
            body["facing" if key == "facing" else f"psx_{key}"] = value
        out = self.job(lambda: self.factory.generate(body))
        return {**out, "next": "poll job_status with this job_id"}

    def rig(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        self.folder(stem)
        body: dict[str, Any] = {k: a[k] for k in ("class", "plan") if a.get(k)}
        if "reuse_skin" in a:
            body["reuse_skin"] = bool(a["reuse_skin"])
        out = self.job(lambda: self.factory.rig(stem, body))
        return {**out, "next": "poll job_status with this job_id; when it is done, rig_review the stem"}

    def job_status(self, a: dict) -> dict:
        try:
            out = self.factory.status(a.get("job_id"))
        except FactoryDown as e:
            raise ToolError(str(e)) from None
        if out.get("log_tail"):
            out["log_tail"] = "\n".join(out["log_tail"].splitlines()[-20:])
        return out

    def cancel(self, a: dict) -> dict:
        return self.job(lambda: self.factory.cancel(a["job_id"]))

    def pick_take(self, a: dict) -> dict:
        stem = check_stem(a["stem"])
        # A pick deletes the stage's live files, workbench/ included: not while a workbench write is mid-way.
        with self._lock(stem):
            return self.job(lambda: self.factory.pick(stem, a["stage"], a["take"]))

    # ------------------------------------------------------------ the registry

    def _registry(self) -> list[Tool]:
        target = {"stem": STEM, "glb": GLB, "lod": LOD}
        return [
            Tool("asset_info", "An asset's files, shape.json, degraded.json, the photo fit, psx.json, intent, scene "
                 "and takes per stage.", _obj({"stem": STEM}, ("stem",)), self.asset_info),
            Tool("look", "Contact sheet of textured renders from fixed cameras (default side, xside, front, back, "
                 "top, q; bottom on request). Returns the PNG (path and image). Give stem (its model by lod) or glb. "
                 "sampling pixel (default) samples the texture per pixel, as an engine shows it; face colours each "
                 "triangle by one texel (the rig sheets' look), which turns a low-poly mesh's thin triangles into "
                 "stripes.",
                 _obj({**target, "views": {"type": "array", "items": {"type": "string", "enum": list(VIEWS)},
                                           "minItems": 1},
                       "size": {"type": "integer", "minimum": 64, "maximum": 1024},
                       "sampling": {"type": "string", "enum": ["pixel", "face"]}}), self.look),
            Tool("measure", "Bounds, triangles, pieces, head_offset_pct (sideways head offset, % of body width), "
                 "head_yaw_deg, mirror_iou (left/right symmetry), circularity (top or end view). The creature is "
                 "taken to face +Z unless auto_head.",
                 _obj({**target, "auto_head": {"type": "boolean"}}), self.measure),
            Tool("compare_reference", "IoU of the model's silhouette against a photo's from a fitted camera, the "
                 "regions missing or extra, and a reference/model/overlap PNG.",
                 _obj({**target, "image": {"type": "string", "description": "picture path relative to the repo"},
                       "facing": FACING}, ("image",)), self.compare_reference),
            Tool("rig_review", "Where a rig stretches, bone by bone, and why (workbench/rigeyes.py; reads the rig the "
                 "viewer shows, writes nothing into out/). Gives the rig check's summary (rig/inspect.json: line, "
                 "per-clip stretch p99/max, tears, foot slide, problems); per clip each edge's worst stretch (p99, "
                 "max) and how many edges pass threshold (default 2x), with the bones owning them (top bone of both "
                 "ends, UV-seam copies welded); each bone's weight share; a pivot check per leg (its root joint's xz "
                 "distance from the leg at the leg line, over the body length; over 0.25 is flagged: the sheep's "
                 "mid-body hind pivot read 0.39, fixed 0.13); and a diagnosis in plain words. heat (bone names, or "
                 "\"flagged\") adds heat maps of those bones from side, front, q and back at rest and at each bone's "
                 "worst frame, as one image; its frames are deleted unless keep_images. Run it after every rig and "
                 "every weights or joint change; ask for heat only for the bones the numbers flag.",
                 _obj({"stem": STEM, "clips": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                       "threshold": {"type": "number", "minimum": 1.1, "maximum": 10},
                       "heat": {"type": ["array", "string"], "items": {"type": "string"}, "minItems": 1, "maxItems": 6,
                                "description": "bone names, or \"flagged\" (the bones the diagnosis names, at most 4)"},
                       "keep_images": {"type": "boolean"}}, ("stem",)), self.rig_review),
            Tool("review", "Run every check Stephen's critiques taught on an asset (workbench/review.py): head offset, "
                 "mirror, pieces, tris, magenta spill, the --select bake against Hunyuan's paint (a feature in the "
                 "wrong place, patchwork), degraded.json. Writes workbench/review/{review.json,sheet.png} and "
                 "returns the sheet: the head painted above grey (count the ears and eyes), the whole model, where "
                 "the bake disagrees with the paint. The psx tool runs it after every cut. verdict is never a pass: "
                 "read the sheet before showing anyone.",
                 _obj({"stem": STEM, "glb": GLB, "bake": {"type": "string"}, "paint": {"type": "string"}},
                      ("stem",)), self.review),
            Tool("rig_preview", "Where a re-rig would put the joints, in seconds and without one (workbench/"
                 "rig_preview.py: rig.py on a temp copy of the asset, stopped before the skin step; the predicted "
                 "skeleton only from its cache, so no model loads; out/ is not touched). Gives the posed trunk and leg "
                 "joints (mesh frame) and register's body report (length ratios, trunk anchors). Use it before a re-rig "
                 "that moves a body joint: those relearn the skin weights (about 3 minutes, one model at a time).",
                 _obj({"stem": STEM, "class": {"type": "string"}, "plan": {"type": "string"}}, ("stem",)),
                 self.rig_preview),
            Tool("playbook", "The pipeline's playbook in full (workbench/playbook.md): the working rules, the mesh "
                 "steps and the rig review that have worked, each with why. Read it before driving the pipeline.",
                 _obj({}), self.playbook),
            Tool("clay_get", "The asset's clay blobs (out/<stem>/workbench/clay.json) and the forms with their dims.",
                 _obj({"stem": STEM}, ("stem",)), self.clay_get),
            Tool("clay_set", "Write clay blobs. Blobs replace the ones with the same name and the rest are added; "
                 "replace: true makes the list the whole clay. Applied in order: add unions, carve cuts.",
                 _obj({"stem": STEM, "blobs": {"type": "array", "items": BLOB, "minItems": 1},
                       "replace": {"type": "boolean"}}, ("stem", "blobs")), self.clay_set),
            Tool("clay_edit", "Change one blob: patch holds the fields to change (form merges unless its type "
                 "changes; noise merges too, and noise null or amp 0 turns it off; name renames). patch "
                 "{remove: true} deletes it.",
                 _obj({"stem": STEM, "name": {"type": "string"}, "patch": {"type": "object"}},
                      ("stem", "name", "patch")), self.clay_edit),
            Tool("clay_mesh", "Mesh the clay to out/<stem>/workbench/clay.glb: surface nets, UVs, blob colours "
                 "baked into the texture. height scales it (m, feet on y = 0); grid is cells along the longest side.",
                 _obj({"stem": STEM, "grid": {"type": "integer", "minimum": 16, "maximum": 160},
                       "height": {"type": "number", "minimum": 0.001}, "texture": {"type": "integer",
                                                                              "minimum": 64, "maximum": 4096}},
                      ("stem",)), self.clay_mesh),
            Tool("mesh_edit", "Edit a GLB into out/<stem>/workbench/edit_N.glb. op remove_floaters (params keep, "
                 "min_faces), delete_sphere (params center [x,y,z], radius) or stretch (params axis x|y|z, factor: "
                 "scales one axis, height up from the floor, the others about the middle; UVs and texture kept). "
                 "Proportion ops (workbench/edit.py; only vertices move, so UVs and texture are kept): thicken_legs "
                 "(params factor 1..3, top?) scales each leg's x,z about its own centre line, fading to 1 over the "
                 "top 15% of the leg band, feet stay on the floor; shorten_legs (factor 0.4..1, top?) squashes the "
                 "leg band toward the floor and lowers the body by the height removed, smoothly at the join; "
                 "widen_head (factor 1..2, start?, ramp? default 0.1, head? +x|-x|+z|-z) scales the head's lateral "
                 "axis about its centre line, ramping in from start to start + ramp along the length (0 rump, 1 "
                 "nose); trim_belly (line?, blend? default 0.03, keep_legs? default true) lifts what hangs under the "
                 "belly (a keel of wool, spikes) up to line (share of height; default the belly's underside between "
                 "the legs, or the leg band's top if higher) with a smooth knee of +-blend, leaving legs (and feet), "
                 "head and tail, then smooths the lifted underside; it refuses on a shape without legs unless line is "
                 "given; tidy_legs (thickness? 0.5..3 default 1, straighten? 0..1 default 1, smooth? 0..1 default 1, "
                 "top?) keeps the legs part of the mesh and tidies them: each leg's centre line becomes a straight "
                 "line through its centres (its slant kept, the zig-zag gone), its radius a smooth taper (fuller at "
                 "the top, a slight hoof flare) times thickness, fading out over the top 15% of the band so it joins "
                 "the belly, then Taubin smoothing of the leg takes the lumps off; feet stay on the floor; the answer "
                 "reports per leg slant, zig-zag and lumps before and after, the radius profile and turned faces "
                 "(use it before psx with leg_weight instead of legs clean; the sheep: thickness 1.5). "
                 "top (share of height where legs meet belly), start and head are read from the shape when "
                 "omitted; the answer says what was found (top, legs and their centres, start, head end). When the "
                 "shape doesn't show them (fewer than two legs, legs it can't tell apart, no head end that rises "
                 "above the body or ends that look alike, no neck) the op refuses and says why: give the "
                 "parameter it names (top, head, start, or line for trim_belly). "
                 "close (params center? [x,y,z], radius?, size? default 0, carve? [{center, radius}]; workbench/"
                 "solid.py) is for shapes the drawings won't make: the mesh is filled solid on a voxel grid, holes "
                 "straight through it (a jerry can's handle, a propane collar's slots, see-through windows) are "
                 "filled, size > 0 also closes dents under about 2 x size, carve spheres are emptied first (wing "
                 "mirrors), and it is surfaced again as one closed skin. Give center and radius to fill only there: "
                 "filling everywhere also fills recesses that are meant (the can's embossed panels). The answer has "
                 "no UVs: rebuild it with generate (mesh = the edit, paint off, image the drawn flat/source.png) so "
                 "the drawings bake onto the new shape. gaps (params size? default 0) writes nothing and lists what "
                 "close would fill, largest first (center, voxels, extent): find the hole there, then close it. "
                 "straighten_head, symmetrize and nudge are not built yet.",
                 _obj({"stem": STEM, "src": GLB, "op": {"type": "string", "enum": ["remove_floaters", "delete_sphere",
                                                                                   "stretch", "thicken_legs",
                                                                                   "shorten_legs", "widen_head",
                                                                                   "trim_belly", "tidy_legs",
                                                                                   "close", "gaps",
                                                                                   "straighten_head", "symmetrize",
                                                                                   "nudge"]},
                       "params": {"type": "object"}}, ("stem", "src", "op")), self.mesh_edit),
            Tool("bake_photo", "Bake photos onto a GLB with UVs (retexture.py) into out/<stem>/workbench/bake_N.glb. "
                 "The first view's facing can be stated; later views are read from their file names. The first "
                 "view wins where it lands unless first_wins is false. mirror_ends (default true) also bakes a "
                 "front-on or back-on first view, flipped, onto the opposite end; mirror does the same for any view "
                 "(a side-on one onto the far side). fill photo (default) then paints a triplanar swatch of the "
                 "photo's subject onto every texel no photo reached (workbench/fill.py); none leaves them base. "
                 "delight (default false) takes the photo's own lighting out first (delight.py: shade lifted to the "
                 "lit level), so a shadowed side does not bake in or reach the swatch; ask for it on a photo with a "
                 "side in shadow (the hay bale), not on one with large dark material (it is lifted to grey too).",
                 _obj({"stem": STEM, "glb": GLB,
                       "views": {"type": "array", "minItems": 1, "items": _obj(
                           {"path": {"type": "string"}, "facing": FACING}, ("path",))},
                       "mirror": {"type": "boolean"}, "erode": {"type": "number", "minimum": 0, "maximum": 0.2},
                       "keep": {"type": "number", "minimum": 0, "maximum": 1}, "first_wins": {"type": "boolean"},
                       "fill": FILL, "mirror_ends": {"type": "boolean"}, "delight": {"type": "boolean"}},
                      ("stem", "glb", "views")), self.bake_photo),
            Tool("bake_wrap", "Wrap one lengthwise cutout round a round prop (workbench/wrap.py) into "
                 "workbench/wrap_N.glb: every point takes the picture's colour for where it sits along the long axis "
                 "and how far round it, so stripes that run end to end (a watermelon) have no seam or fill swatch. "
                 "Give a picture of the prop lying lengthwise; flip turns it end for end.",
                 _obj({"stem": STEM, "glb": GLB, "picture": {"type": "string"}, "flip": {"type": "boolean"}},
                      ("stem", "glb", "picture")), self.bake_wrap),
            Tool("psx", "Cut a baked GLB down to a PS1 model (psx_lowpoly.py, facing check off) and publish it as "
                 "generate.sh's PS1 step does: out/<stem>/psx.glb, out/<stem>/psx/<name>.obj + .mtl + .png + "
                 "psx.json, and the PSX entry in lods.json (after Photo bake), so the app's Model tabs show it. It "
                 "replaces the asset's PSX (pick_take brings back an earlier mesh take's). height is in .obj units "
                 "(s&box inches); facing is where the subject looked in the photo. head_weight (creatures; 2 is a good start) "
                 "gives the head that many times its share of the triangles, so ears and face survive the cut. "
                 "legs clean (creatures) swaps each decimated leg, lumpy and splayed at the hoof, for a straight "
                 "tapered prism of leg_sides sides (default 6), leg_scale times the measured leg radius (default 1; "
                 "above 1 for thicker legs, instead of thicken_legs) and leg_taper bottom over top (default 0.8); a "
                 "shape with no legs keeps its own and psx.json's legs says why. leg_weight (creatures; default 1 = "
                 "off; the sheep: 2.5) keeps the legs part of the mesh and gives them that many times their share of "
                 "the triangles so the cut does not pinch them (tidy them first with mesh_edit tidy_legs); it "
                 "composes with head_weight, not with legs clean.",
                 _obj({"stem": STEM, "glb": GLB, "height": {"type": "number", "minimum": 0.001, "maximum": 100000},
                       "facing": FACING, "faces": {"type": "integer", "minimum": 1, "maximum": 65536},
                       "atlas": {"type": "integer", "minimum": 1, "maximum": 65536},
                       "head_weight": {"type": "number", "minimum": 1, "maximum": 4},
                       "leg_weight": {"type": "number", "minimum": 1, "maximum": 4},
                       "legs": {"type": "string", "enum": ["keep", "clean"]},
                       "leg_sides": {"type": "integer", "minimum": 3, "maximum": 16},
                       "leg_scale": {"type": "number", "minimum": 0.25, "maximum": 4},
                       "leg_taper": {"type": "number", "minimum": 0.2, "maximum": 2},
                       "name": {"type": "string"}}, ("stem", "glb", "height", "facing")), self.psx),
            Tool("draw_views", "Factory job (loads the draw model): draw the four turnaround views from a photo as "
                 "a reference (POST /2d/draw). Returns a job_id for job_status.",
                 _obj({"stem": STEM, "image": {"type": "string"}, "prompt": {"type": "string"},
                       "intent": {"type": "object"}}, ("stem", "image", "prompt")), self.draw_views, "job"),
            Tool("generate", "Factory job: generate.sh through POST /generate. mesh (a .glb under out/<stem>/) is "
                 "used as the shape instead of hy3d's; paint false skips the paint model (unwrap, then the photo "
                 "bake), so a clay asset loads no shape or paint model (quality psx may still ask the vision model for the facing, "
                 "inside the job). psx: {facing, height, faces, atlas, up, name}. fill photo (default; not for "
                 "draft) gives every texel the photo: a front-on photo mirrored onto the far end (mirror_ends, "
                 "default true) and a swatch of it on the rest; delight (default false; not for draft) takes the "
                 "photo's lighting out before the bake and the fill (for a side in shadow, not dark material). "
                 "Returns a job_id.",
                 _obj({"stem": STEM, "image": {"type": "string"}, "mesh": GLB, "paint": {"type": "boolean"},
                       "fill": FILL, "mirror_ends": {"type": "boolean"}, "delight": {"type": "boolean"},
                       "quality": {"type": "string", "enum": ["draft", "textured", "pbr", "stack", "psx"]},
                       "psx": _obj({k: {"type": ["string", "number"]} for k in PSX_KEYS}),
                       "images": {"type": "array", "items": {"type": "string"}},
                       "prompt": {"type": "string"}, "poly": {"type": "string"}, "seed": {"type": ["string", "integer"]},
                       "intent": {"type": "object"}}, ("image",)), self.generate, "job"),
            Tool("rig", "Factory job: rig the asset (POST /rig/<stem>, scripts/rig.py). Give class (humanoid, "
                 "quadruped, multileg, avian) or plan when known: without either, rig.py asks the planner model what "
                 "the subject is. reuse_skin true keeps the skin weights learned before when the mesh and joint names "
                 "match (limb joints may have moved: a pivot fix; a moved body joint or one moved over 0.6x height "
                 "learns them again), so no skin model loads and the job is CPU only. "
                 "Returns a job_id; when it is done, rig_review the stem.",
                 _obj({"stem": STEM, "class": {"type": "string", "enum": ["humanoid", "quadruped", "multileg", "avian"]},
                       "plan": {"type": "string", "description": "a body plan id (GET /rig/plans), e.g. horse"},
                       "reuse_skin": {"type": "boolean"}}, ("stem",)), self.rig, "job"),
            Tool("job_status", "The factory's job: state, stage progress, log tail, result. No job_id: the current "
                 "job.", _obj({"job_id": {"type": "string"}}), self.job_status, "job"),
            Tool("cancel", "Cancel the factory's running job.", _obj({"job_id": {"type": "string"}}, ("job_id",)),
                 self.cancel, "job"),
            Tool("pick_take", "Go back (or forward) to a take of a stage (POST /flow/<stem>/pick); asset_info lists "
                 "them.", _obj({"stem": STEM, "stage": {"type": "string", "enum": STAGES},
                                "take": {"type": "string"}}, ("stem", "stage", "take")), self.pick_take, "job"),
        ]


def error_text(exc: BaseException) -> str:
    """The message a failed tool reports. Unexpected errors carry their type, and their traceback goes to stderr."""
    if isinstance(exc, (ToolError, ArgError, FileNotFoundError, bake.BakeError)) or type(exc) is ValueError:
        return str(exc)
    traceback.print_exc(file=sys.stderr)
    return f"{exc.__class__.__name__}: {exc}"
