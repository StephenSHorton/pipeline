#!/usr/bin/env python3
"""Loopback HTTP adapter over the local MLX factory, for curl and the app. The agent workbench's MCP server
(scripts/workbench/) is a separate process and a client of this one.

GET  /health
GET  /inspect/<stem>
POST /generate
GET  /status
GET  /status/<job_id>
POST /cancel/<job_id>
POST /intent/<stem>
POST /rig/<stem>
POST /rig/<stem>/inspect
POST /rig/<stem>/render  the rig's contact sheet (rig/preview/) as a job
POST /rig/<stem>/weights hand-edited skin weights into rigged.glb (scripts/rig_weights.py), FBX again, a rig version
GET  /rig/plans          the body plans (scripts/rig_plans/)
POST /facing/detect      which way the head points in a picture {image|stem}
GET  /glb/<stem>
GET  /stack/<stem>
POST /export/<stem>
POST /upload
POST /generate
POST /2d/draw            a 2D still (or Ideas) as a job: /status, /cancel
GET  /2d/<stem>/drawing  the picture filling in while it draws
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from activity import Activity, JobWatch
from flat2d import drawjob
from flat2d.routes import handle as handle_flat2d
from flat2d.sheet import still_backend
import flow
from flow import FLAT_WRITES, autosave
from flow import handle as handle_flow
from export_asset import handle as handle_export
import projects
from factory_env import (
    NO_PICTURE,
    PSX_KEYS,
    UPLOAD_EXTS,
    add_degraded,
    auto_stem,
    check_stem,
    clean_psx,
    degraded_entry,
    degraded_line,
    factory_python,
    file_sha256,
    generate_argv,
    json_degraded,
    normalize_poly,
    parse_degraded,
    pick_glb,
    poly_label,
    repo_root,
    require_picture,
    resolve_under_root,
    slug_stem,
    stem_dir,
)
from inspect_glb import inspect as inspect_glb
from clean_mesh import clean_name

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
QUALITIES = ("draft", "textured", "pbr", "stack", "psx")
NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
ALLOWED_HOSTS = LOOPBACK
DEFAULT_PORT = 8766


def local_origin(origin: str | None) -> bool:
    """No Origin (curl, scripts, the workbench) or a page on this machine: a loopback host or the app's own
    webview (wails://). Any other web page is refused, so a site open in a browser can't read or delete assets."""
    if not origin:
        return True
    u = urlparse(origin)
    host = (u.hostname or "").strip("[]")
    return u.scheme == "wails" or host in LOOPBACK or host == "wails.localhost"


def local_host(host_header: str | None) -> bool:
    """The Host header names this machine, so a DNS-rebound name can't reach the factory."""
    if not host_header:
        return True
    host = host_header.rsplit(":", 1)[0] if not host_header.startswith("[") else host_header[1:].split("]")[0]
    return host in LOOPBACK


def _dir_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return False


def _executable(path: Path) -> bool:
    """The file a path (or symlink chain) resolves to exists and can be run; a dangling link is False."""
    try:
        target = Path(os.path.realpath(path))
    except OSError:
        return False
    return target.is_file() and os.access(target, os.X_OK)


def _has_weights(folder: Path) -> bool:
    try:
        return folder.is_dir() and any(folder.glob("*.safetensors")) and (folder / "config.yaml").is_file()
    except OSError:
        return False


_HY2_PROBE: dict[tuple[str, int], bool] = {}


def _hy2_imports(root: Path) -> bool:
    """vendor/hunyuan3d-2's venv imports torch and hy3dgen (what scripts/shape_mv.sh needs). Run once per venv
    (keyed on its python's mtime), since /health is polled and the import takes about a second."""
    hy2 = root / "vendor" / "hunyuan3d-2"
    py = hy2 / ".venv" / "bin" / "python"
    if not _executable(py):
        return False
    try:
        key = (str(py.resolve()), py.stat().st_mtime_ns)
    except OSError:
        return False
    if key not in _HY2_PROBE:
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import torch, hy3dgen, importlib.util as u; "
                "sys.exit(0 if u.find_spec('hy3dgen.shapegen') else 1)")
        try:
            ok = subprocess.run([str(py), "-c", code, str(hy2)], cwd=str(root), capture_output=True,
                                timeout=60).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        _HY2_PROBE[key] = ok
    return _HY2_PROBE[key]


# What each draw backend's weights are licensed under (flat2d.sheet.still_backend names), used only when the sheet
# module can't say itself (flat2d.sheet.still_info()); a backend neither knows reads "unknown", never a guess.
DRAW_LICENCES = {
    "qwen-2512": "apache-2.0",  # Qwen-Image-2512 + Lightning LoRA
    "qwen-image": "qwen-research (non-commercial)",  # Qwen-Image-2.1-Turbo: Qwen RESEARCH LICENSE, NC only
    "z-image": "apache-2.0",
    "qwen-edit": "apache-2.0",  # Qwen-Image-Edit-2511 + Lightning LoRA: the photo draw
    "stub": "none (test stub)",
}


def draw_feature() -> dict[str, Any]:
    from flat2d import sheet

    try:
        backend = still_backend()
    except Exception as e:  # noqa: BLE001 - a broken draw stack is a fact to report, not a crash of /health
        return {"backend": "", "licence": "unknown", "error": str(e)[:200]}
    out: dict[str, Any] = {"backend": backend or ""}
    try:
        info = sheet.still_info(backend) or {}
    except Exception:  # noqa: BLE001
        info = {}
    licence = info.get("licence")
    if licence and licence != "unknown" and info.get("model"):
        out["model"] = str(info["model"])
    else:
        licence = DRAW_LICENCES.get(backend, "unknown") if backend else "none"
    out["licence"] = str(licence)
    # The photo draw (a picture the user added is drawn from, never used as a view): '' when there is none.
    try:
        photo = sheet.edit_info()
    except Exception:  # noqa: BLE001
        photo = {"backend": ""}
    out["photo"] = {"backend": photo.get("backend") or "", "model": photo.get("model") or "none",
                    "licence": DRAW_LICENCES.get(photo.get("backend") or "", "none")}
    return out


def factory_features(root: Path) -> dict[str, Any]:
    """What this machine can really do, probed (not claimed): isolate, shape, the 2mv path, remesh and the draw."""
    root = Path(root)
    lift = _executable(root / "bin" / "lift-subject")
    hy3d = _executable(root / "bin" / "hy3d")
    shape_weights = _has_weights(root / "weights" / "shape-small")
    mv_dir = root / "weights" / "tencent" / "Hunyuan3D-2mv"
    mv_weights = _has_weights(mv_dir / "hunyuan3d-dit-v2-mv-turbo") or _has_weights(mv_dir / "hunyuan3d-dit-v2-mv")
    hy2 = _hy2_imports(root)
    remesh = _executable(root / "bin" / "autoremesher")
    return {
        "isolate": "vision" if lift else "none",
        "shape": "hy3d-mlx" if hy3d and shape_weights else "none",
        "shape_weights": shape_weights,
        "paint": "mlx-rgb",
        "paint_pbr": True,
        "shape_mv": bool(hy2 and mv_weights),
        "shape_mv_venv": hy2,
        "shape_mv_weights": mv_weights,
        "unirig": False,
        "remesh": remesh,
        "draw": draw_feature(),
    }


# What each generate.sh stage is called in the app (generate.sh prints "STAGES a b c" then "STAGE a <label>").
STAGE_LABELS = {
    "isolate": "Cut out the subject",
    "shape": "Build the shape",
    "reduce": "Reduce the mesh",
    "paint": "Paint the texture",
    "unwrap": "Unwrap the mesh",
    "retexture": "Bake the photo on",
    "fill": "Fill what the photo missed",
    "psx": "PS1 low-poly",
    "lods": "List the finished models",  # lods.py only: no game / proxy LODs are cut any more
    "inspect": "Check the model",
    "stack": "Sprite stack",
    "preview": "Thumbnail",
    # scripts/rig.py
    "classify": "Work out the body plan",
    "rest": "Check the rest pose",
    "pose": "Pose the skeleton",
    "skin": "Learn the skin weights",
    "clean": "Clean up the weights",
    "clips": "Bake the clips",
    "check": "Check the rig",
    "fbx": "FBX for s&box",
    # flat2d/drawjob.py: load draw ground redraw pieces motion say sound idea1..idea8
    **drawjob.LABELS,
}
SUB_PROGRESS = re.compile(r"\[\s*(\d{1,3})%\]")
def rig_env(root: Path) -> dict[str, str]:
    """rig.py and rig_inspect.py write into FACTORY_ROOT/out, not next to their own (maybe symlinked) scripts/."""
    return {**os.environ, "FACTORY_ROOT": str(root)}


# Body plans scripts/rig.py has a template for.
RIG_CLASSES = ("humanoid", "quadruped", "multileg", "avian")
# POST /rig/<stem> options that pass straight through as rig.py flags of the same name.
RIG_CHOICES = {
    "template": ("auto", "fixed", "build"),
    "binder": ("skin", "pred", "heat"),
    "pred": ("auto", "always", "never"),
}
RIG_SWITCHES = ("floor", "render", "reframe")
CLIP_NAME = re.compile(r"[A-Za-z0-9 _-]{1,48}")
# POST /rig/<stem>/render {job}: the rig_render.py options a job may set.
RENDER_VIEWS = ("side", "xside", "front", "back", "top", "q")
RENDER_MODE = re.compile(r"(texture|groups|sticks|heat:[A-Za-z0-9_.-]{1,64})")
RENDER_CLIPS = re.compile(r"[A-Za-z0-9 _.,;@-]{1,400}")
# What the asset is meant to be; the app picks its stages and Export card from this.
OUTPUTS = ("ps1", "model", "stack", "sprite")


def rig_plans() -> list[dict[str, Any]]:
    """Every body plan (scripts/rig_plans/*.json), slimmed for the app's body-plan select."""
    import rig_plan

    out = []
    for p in rig_plan.all_plans():
        out.append({
            "id": p.get("id"),
            "label": p.get("label") or p.get("id"),
            "blurb": p.get("blurb") or "",
            "class": p.get("class"),
            "aliases": list(p.get("aliases") or []),
            "built": bool(p.get("builder")),
            "fixed": bool(p.get("fixed_template")),
            "clips": [str(c.get("name")) for c in p.get("clips") or [] if isinstance(c, dict) and c.get("name")],
        })
    return out


def _plan_class(plan_id: str) -> str:
    """The class of a body plan id (or alias); ValueError when there is no such plan."""
    for p in rig_plans():
        if plan_id == p["id"] or plan_id in p["aliases"]:
            return str(p["class"] or "")
    raise ValueError(f"plan must be one of {', '.join(str(p['id']) for p in rig_plans())}")


def _switch(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value).strip().lower()
    if text in ("0", "1"):
        return text
    if text in ("true", "on", "yes"):
        return "1"
    if text in ("false", "off", "no"):
        return "0"
    raise ValueError(f"{key} must be 0 or 1")


def rig_argv(python: Path | str, root: Path, stem: str, body: dict[str, Any]) -> list[str]:
    """rig.py's argv for a POST /rig/<stem> body, every value checked (ValueError on a bad one):
    {class, plan, template, binder, pred, floor, render, reframe, clips, straighten, reuse_skin}."""
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    argv = [str(python), str(Path(root) / "scripts" / "rig.py"), "--stem", stem]
    cls = str(body.get("class") or "")
    if cls and cls not in RIG_CLASSES:
        raise ValueError(f"class must be one of {', '.join(RIG_CLASSES)}")
    plan = str(body.get("plan") or "")
    if plan:
        plan_cls = _plan_class(plan)
        if cls and plan_cls and cls != plan_cls:
            raise ValueError(f"plan {plan} is for class {plan_cls}, not {cls}")
        cls = cls or plan_cls  # rig.py needn't ask the planner when the plan says the class
    if cls:
        argv += ["--class", cls]
    if plan:
        argv += ["--plan", plan]
    for key, allowed in RIG_CHOICES.items():
        value = body.get(key)
        if value in (None, ""):
            continue
        if str(value) not in allowed:
            raise ValueError(f"{key} must be one of {', '.join(allowed)}")
        argv += [f"--{key}", str(value)]
    for key in RIG_SWITCHES:
        value = body.get(key)
        if value in (None, ""):
            continue
        argv += [f"--{key}", _switch(key, value)]
    clips = body.get("clips")
    if isinstance(clips, str):
        clips = [c for c in clips.split(",") if c.strip()]
    if clips not in (None, "", []):
        if not isinstance(clips, list):
            raise ValueError("clips must be a list of clip names")
        names = [str(c).strip() for c in clips]
        bad = [n for n in names if not CLIP_NAME.fullmatch(n)]
        if bad:
            raise ValueError(f"bad clip name {bad[0]!r}")
        argv += ["--clips", ",".join(names)]
    straighten = str(body.get("straighten") or "")
    if straighten not in ("", "auto", "off"):
        raise ValueError("straighten must be auto or off")
    if straighten:
        argv += ["--straighten", straighten]
    # keep the skin weights learned before (same mesh; same joints, even moved): a re-rig without skin-tokens
    if body.get("reuse_skin") not in (None, "") and _switch("reuse_skin", body["reuse_skin"]) == "1":
        argv += ["--reuse-skin"]
    return argv


def render_argv(python: Path | str, root: Path, stem: str, job: Any) -> list[str]:
    """rig_render.py's argv for a POST /rig/<stem>/render {job} body. job is "default" (or missing) or a dict of
    {views, modes, clips (or frames), size, floor}; ValueError on a bad value."""
    argv = [str(python), str(Path(root) / "scripts" / "rig_render.py"), "--stem", stem]
    if job in (None, "", "default", "auto"):
        return argv
    if not isinstance(job, dict):
        raise ValueError('job must be "default" or an object')
    for key, ok in (("views", lambda v: v in RENDER_VIEWS), ("modes", lambda v: bool(RENDER_MODE.fullmatch(v)))):
        value = job.get(key)
        if value in (None, "", []):
            continue
        items = value.split(",") if isinstance(value, str) else value
        if not isinstance(items, list) or not items or not all(isinstance(v, str) and ok(v) for v in items):
            raise ValueError(f"bad render {key}")
        argv += [f"--{key}", ",".join(items)]
    clips = job.get("clips", job.get("frames"))
    if isinstance(clips, dict):
        parts = []
        for name, fr in clips.items():
            if not CLIP_NAME.fullmatch(str(name)):
                raise ValueError(f"bad clip name {name!r}")
            if fr == "auto":
                parts.append(str(name))
            elif isinstance(fr, list) and fr and all(isinstance(f, (int, float)) and 0 <= f <= 1 for f in fr):
                parts.append(f"{name}@" + ",".join(f"{float(f):g}" for f in fr))
            else:
                raise ValueError(f"bad frames for {name}")
        clips = ";".join(parts) or "auto"
    if clips not in (None, ""):
        if not isinstance(clips, str) or not RENDER_CLIPS.fullmatch(clips):
            raise ValueError("bad render clips")
        argv += ["--clips", clips]
    if job.get("size") not in (None, ""):
        try:
            size = int(job["size"])
        except (TypeError, ValueError):
            raise ValueError("size must be 64 to 1024") from None
        if not 64 <= size <= 1024:
            raise ValueError("size must be 64 to 1024")
        argv += ["--size", str(size)]
    if job.get("floor") is not None and _switch("floor", job["floor"]) == "0":
        argv.append("--no-floor")
    return argv


def parse_progress(text: str) -> dict[str, Any] | None:
    """Stage plan and position from a generate.log: STAGES/STAGE markers plus the last '[ 42%]' inside the stage."""
    plan: list[str] = []
    current = ""
    sub = 0.0
    for line in text.splitlines():
        if line.startswith("STAGES "):
            plan = line.split()[1:]
            current, sub = "", 0.0
        elif line.startswith("STAGE "):
            bits = line.split(maxsplit=2)
            if len(bits) >= 2:
                current, sub = bits[1], 0.0
        elif current:
            m = SUB_PROGRESS.search(line)
            if m:
                sub = min(int(m.group(1)), 100) / 100.0
    if not plan:
        return None
    index = plan.index(current) if current in plan else -1
    return {
        "stages": [{"key": k, "label": STAGE_LABELS.get(k, k)} for k in plan],
        "stage": current or None,
        "stage_index": index,
        "stage_count": len(plan),
        "sub": round(sub, 2),
        "fraction": round(max(index, 0) / len(plan) + (sub / len(plan) if index >= 0 else 0), 3),
    }


def _given(value: Any) -> bool:
    return value not in (None, "", [])


def clean_intent(raw: Any) -> dict[str, Any] | None:
    """The app's chosen output for an asset, checked: {"output": ps1|model|stack|sprite, ...settings}. None when
    no intent was sent; a bad value for a known key raises ValueError (a 400), never a quiet drop. Keys this
    doesn't know are ignored."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("intent must be a JSON object")
    if raw.get("output") not in OUTPUTS:
        raise ValueError("intent needs output: " + " | ".join(OUTPUTS))
    out: dict[str, Any] = {"output": raw["output"]}
    psx = clean_psx({k: raw.get(k) for k in PSX_KEYS})
    if psx:
        out["psx"] = psx
    if _given(raw.get("poly")):
        if raw["poly"] not in ("high", "medium", "low"):
            raise ValueError("intent poly must be high, medium or low")
        out["poly"] = raw["poly"]
    steps = raw.get("steps")
    if steps is not None:
        allowed = ("pieces", "motion", "say", "sound")
        if not isinstance(steps, list) or any(x not in allowed for x in steps):
            raise ValueError("intent steps must be a list of " + ", ".join(allowed))
        out["steps"] = [str(x) for x in steps]
    # What the subject is (the canonical draw and the rig read these): {subject, kind, name}.
    for key, limit in (("subject", 48), ("kind", 48), ("name", 64)):
        value = raw.get(key)
        if not _given(value):
            continue
        if not (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9 _-]{1,%d}" % limit, value.strip())):
            raise ValueError(f"intent {key} must be 1 to {limit} letters, digits, spaces, _ or -")
        out[key] = value.strip()
    if _given(raw.get("facing_source")):
        if raw["facing_source"] not in FACING_SOURCES:
            raise ValueError("intent facing_source must be one of " + ", ".join(FACING_SOURCES))
        out["facing_source"] = raw["facing_source"]
    rig = raw.get("rig")
    if rig is not None:
        if not isinstance(rig, dict):
            raise ValueError("intent rig must be an object")
        keep: dict[str, Any] = {}
        for key in ("plan", "plan_hint"):
            if not _given(rig.get(key)):
                continue
            if not (isinstance(rig[key], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,48}", rig[key])):
                raise ValueError(f"intent rig {key} must be 1 to 48 letters, digits, _ or -")
            keep[key] = rig[key]
        if rig.get("canonical") is not None:
            if not isinstance(rig["canonical"], bool):
                raise ValueError("intent rig canonical must be true or false")
            keep["canonical"] = rig["canonical"]
        if keep:
            out["rig"] = keep
    return out


# Where a stored facing came from (section 8 of the rig design): the canonical draw, the VLM, a tap, the mesh cues.
FACING_SOURCES = ("drawn", "detected", "user", "cues")
# Written by the draw (canonical, plan hint) or a tap; a later intent that doesn't say keeps them.
KEPT_INTENT = ("rig", "facing_source", "subject", "kind")


def write_intent(root: Path, stem: str, intent: dict[str, Any]) -> None:
    path = stem_dir(root, stem) / "intent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_intent(path.parent) or {}
    kept = {k: old[k] for k in KEPT_INTENT if k in old and k not in intent}
    path.write_text(json.dumps({**kept, **intent, "made": int(time.time())}, indent=2) + "\n", encoding="utf-8")


def read_intent(folder: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads((folder / "intent.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("output") in OUTPUTS else None


def _asset_shot(folder: Path, candidates: tuple[str, ...]) -> str | None:
    for rel in candidates:
        if (folder / rel).is_file():
            return rel
    return None


class Factory:
    def __init__(
        self,
        root: Path,
        generate_runner: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.python = factory_python(self.root)
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None
        self._checking: set[str] = set()  # stems whose rig is being checked again (POST /rig/<stem>/inspect)
        self._runner = generate_runner or self._run_generate_sh
        self._sync_draws = 0  # plain POST /2d/generate or /2d/idea drawing right now (they share the model's hook)
        # What Claude (the workbench posts its tool calls) and the jobs are doing, for the app (GET /activity).
        self.activity = Activity(self.root / "out" / "_logs" / "activity.jsonl")
        self.job_watch = JobWatch(self.activity, self.status, self.root)

    def watch_jobs(self, stop: threading.Event, every: float = 1.0) -> None:
        """The activity entry of the running job, kept current about once a second until stop is set."""
        while not stop.wait(every):
            try:
                self.job_watch.tick()
            except Exception:  # noqa: BLE001 - the watch never takes the factory down
                traceback.print_exc()

    def health(self) -> dict[str, Any]:
        job = self._snapshot()
        return {
            "ok": True,
            "service": "ai-asset-pipeline",
            "backend": "mlx",
            "python": str(factory_python(self.root)),
            "root": str(self.root),
            "gpu": "Apple Silicon",
            "busy": bool(job and job.get("state") == "running"),
            "job_id": (job or {}).get("job_id"),
            "features": factory_features(self.root),
        }

    def inspect_stem(self, stem: str, lod: str | None = None) -> dict[str, Any]:
        check_stem(stem)
        glb = pick_glb(self.root, stem, lod)
        if glb is None:
            return {
                "ok": False,
                "stem": stem,
                "error": "no GLB for stem",
                "dir": str(stem_dir(self.root, stem)),
            }
        report = inspect_glb(str(glb))
        report["stem"] = stem
        report["glb"] = str(glb)
        stack_meta = stem_dir(self.root, stem) / "stack" / "stack.json"
        if stack_meta.is_file():
            try:
                report["stack"] = json.loads(stack_meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report["stack"] = {"ok": False}
        sidecar = stem_dir(self.root, stem) / "inspect.json"
        try:
            sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
        return report

    def glb_file(self, stem: str, lod: str | None = None) -> Path | None:
        check_stem(stem)
        return pick_glb(self.root, stem, lod)

    def save_upload(self, name: str, data: bytes) -> dict[str, Any]:
        if not data:
            raise ValueError("empty upload")
        raw = Path(name or "upload.jpg").name
        ext = Path(raw).suffix.lower()
        if ext not in UPLOAD_EXTS:
            raise ValueError(NO_PICTURE)
        stem = NAME_RE.sub("-", Path(raw).stem).strip("-._") or "upload"
        stem = stem[:48]
        check_stem(stem)
        dest = self.root / "in" / f"{stem}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Decode it first (in a temp file beside the target), so an upload that isn't a picture never lands in in/.
        tmp = dest.with_name(f".upload-{uuid.uuid4().hex[:8]}{ext}")
        try:
            tmp.write_bytes(data)
            require_picture(tmp)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        return {
            "ok": True,
            "path": f"in/{dest.name}",
            "bytes": len(data),
            "stem": stem,
        }

    def list_assets(self) -> dict[str, Any]:
        out = self.root / "out"
        assets: list[dict[str, Any]] = []
        # One membership pass for the whole list, not a manifest read per asset. A bad manifest never hides the Library.
        try:
            owner = projects.membership(self.root)
        except (projects.ProjectError, OSError) as e:
            print(f"assets: could not read projects: {e}", file=sys.stderr)
            owner = {}
        if out.is_dir():
            for d in sorted(out.iterdir()):
                if not d.is_dir() or d.name.startswith("."):
                    continue
                try:
                    check_stem(d.name)
                except ValueError:
                    continue
                has_mesh = pick_glb(self.root, d.name) is not None
                has_stack = (d / "stack" / "stack.json").is_file()
                has_lods = (d / "lods.json").is_file()
                has_flat = (
                    (d / "flat" / "source.png").is_file()
                    or (d / "flat" / "parts.json").is_file()
                    or (d / "flat" / "sheets" / "manifest.json").is_file()
                )
                has_ideas = (d / "ideas" / "ideas.json").is_file()
                if not (has_mesh or has_stack or has_lods or has_flat or has_ideas):
                    continue
                preview = _asset_shot(
                    d,
                    (
                        "preview.png",
                        "input.png",
                        "input.jpg",
                        "flat/source.png",
                        "stack/preview.png",
                        "ideas/idea-01.png",
                    ),
                )
                assets.append(
                    {
                        "id": d.name,
                        "title": d.name.replace("-", " "),
                        "has_mesh": has_mesh,
                        "has_stack": has_stack,
                        "has_flat": has_flat,
                        "has_ideas": has_ideas,
                        "has_psx": (d / "psx.glb").is_file(),
                        "output": (read_intent(d) or {}).get("output"),
                        "input": _asset_shot(d, ("input.png", "input.jpg")),
                        "preview": preview,
                        "project": owner.get(d.name),
                        "mtime": int(d.stat().st_mtime),
                    }
                )
        assets.sort(key=lambda a: a.get("mtime") or 0, reverse=True)
        job = self._snapshot()
        return {
            "ok": True,
            "root": str(out),
            "assets": assets,
            "busy": bool(job and job.get("state") == "running"),
            "job": {"job_id": job.get("job_id"), "stem": job.get("stem"), "state": job.get("state")} if job else None,
        }

    def clean_mesh(self, body: dict[str, Any]) -> dict[str, Any]:
        stem = check_stem(str(body.get("stem") or ""))
        op = str(body.get("op") or "floaters")
        if op not in {"floaters", "sphere"}:
            raise ValueError("op must be floaters or sphere")
        argv = [
            str(self.python),
            str(self.root / "scripts" / "clean_mesh.py"),
            "--stem",
            stem,
            "--op",
            op,
        ]
        # Clean the file on screen (a name inside the asset folder), into its *_clean.glb.
        source = str(body.get("from") or "")
        out_name = "mesh_clean.glb"
        if source:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.glb", source) or not (stem_dir(self.root, stem) / source).is_file():
                raise ValueError(f"no such mesh: {source}")
            out_name = clean_name(source)
            argv += ["--glb", str(stem_dir(self.root, stem) / source), "--out-name", out_name]
        if op == "floaters":
            argv += ["--keep", str(int(body.get("keep") or 1))]
            if body.get("min_faces"):
                argv += ["--min-faces", str(int(body["min_faces"]))]
        else:
            center = body.get("center")
            if not (isinstance(center, (list, tuple)) and len(center) == 3):
                raise ValueError("sphere needs center [x,y,z]")
            argv += ["--center", str(float(center[0])), str(float(center[1])), str(float(center[2]))]
            argv += ["--radius", str(float(body.get("radius") or 0.05))]
        proc = subprocess.run(argv, cwd=str(self.root), capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            return {"ok": False, "error": err, "stem": stem}
        try:
            info = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            info = {"ok": True}
        info["ok"] = True
        info["stem"] = stem
        info["file"] = out_name
        return info

    def rename_asset(self, stem: str, dest: str) -> dict[str, Any]:
        stem = check_stem(stem)
        dest_slug = slug_stem(dest) or str(dest or "").strip()
        dest_slug = check_stem(dest_slug)
        if dest_slug == stem:
            return {"ok": True, "stem": dest_slug, "from": stem, "renamed": False, "path": f"out/{dest_slug}"}
        job = self._snapshot()
        if job and job.get("state") == "running" and job.get("stem") == stem:
            raise RuntimeError("cannot rename while generate is running")
        src = stem_dir(self.root, stem)
        dst = self.root / "out" / dest_slug
        if not src.is_dir():
            return {"ok": False, "error": "not found", "stem": stem}
        if dst.exists():
            raise ValueError(f"asset {dest_slug!r} already exists")
        src.rename(dst)
        with self._lock:
            if self._job and self._job.get("stem") == stem:
                self._job["stem"] = dest_slug
        # The folder already moved: a projects failure is logged, never turned into a failed rename.
        try:
            projects.rename_stem(self.root, stem, dest_slug)
        except (projects.ProjectError, OSError) as e:
            print(f"rename: could not move {stem}'s project membership: {e}", file=sys.stderr)
        return {"ok": True, "stem": dest_slug, "from": stem, "renamed": True, "path": f"out/{dest_slug}"}

    def delete_asset(self, stem: str) -> dict[str, Any]:
        stem = check_stem(stem)
        job = self._snapshot()
        if job and job.get("state") == "running" and job.get("stem") == stem:
            raise RuntimeError("cannot delete while generate is running")
        d = stem_dir(self.root, stem)
        if not d.is_dir():
            return {"ok": False, "error": "not found", "stem": stem}
        # Move, don't erase: out/.trash/ is on the same disk (an instant rename), the library skips dot folders,
        # and a mis-click or a stray request can be undone by moving the folder back.
        trash = d.parent / ".trash"
        trash.mkdir(exist_ok=True)
        dest = trash / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}"
        n = 2
        while dest.exists():
            dest = trash / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}-{n}"
            n += 1
        d.rename(dest)
        try:
            projects.drop_stem(self.root, stem)
        except (projects.ProjectError, OSError) as e:
            print(f"delete: could not drop {stem} from its project: {e}", file=sys.stderr)
        return {"ok": True, "deleted": True, "stem": stem, "path": f"out/{stem}",
                "trashed": str(dest.relative_to(d.parent.parent))}

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Stop a running generate (the whole process group) and mark it cancelled."""
        with self._lock:
            job = self._job
            if not job or job.get("job_id") != job_id:
                return {"ok": False, "error": "unknown job", "job_id": job_id}
            if job.get("state") != "running":
                return {"ok": True, "job_id": job_id, "state": job.get("state")}
            job["cancel"] = True
            pid = job.get("pid")
        if pid:
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        return {"ok": True, "job_id": job_id, "state": "cancelling"}

    def reap(self) -> None:
        """The sidecar is going away: stop the running job's process group. Jobs start in their own session,
        so the app's SIGTERM to the sidecar's group never reaches them. A 2D draw has no pid; the cancel flag
        stops it at its next step."""
        with self._lock:
            job = self._job
            if not job or job.get("state") != "running":
                return
            job["cancel"] = True
            pid = job.get("pid")
        if pid:
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def set_intent(self, stem: str, raw: Any) -> dict[str, Any]:
        stem = check_stem(stem)
        intent = clean_intent(raw)
        if intent is None:
            raise ValueError("intent needs output: " + " | ".join(OUTPUTS))
        if not stem_dir(self.root, stem).is_dir():
            raise FileNotFoundError(f"no asset {stem!r}")
        write_intent(self.root, stem, intent)
        return {"ok": True, "stem": stem, "intent": intent}

    def _snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._job) if self._job else None

    def writing(self, stem: str) -> bool:
        """A job is writing this asset, or its rig is being checked again (picking a version then would mix them)."""
        job = self._snapshot()
        with self._lock:
            checking = stem in self._checking
        return checking or bool(job and job.get("state") == "running" and job.get("stem") == stem)

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        job = self._snapshot()
        if job is None:
            return {"ok": True, "state": "idle", "job_id": None}
        if job_id and job.get("job_id") != job_id:
            return {"ok": False, "error": "unknown job", "job_id": job_id, "state": "unknown"}
        out = dict(job)
        out["ok"] = True
        out["now"] = time.time()
        log_path = out.get("log")
        if log_path and Path(log_path).is_file():
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                out["log_tail"] = "\n".join(text.splitlines()[-40:])
                out["progress"] = parse_progress(text)
            except OSError:
                out["log_tail"] = ""
        out["degraded"] = self._degraded(job)
        if isinstance(out.get("result"), dict):
            out["result"] = {**out["result"], "degraded": out["degraded"]}
        return out

    def _degraded(self, job: dict[str, Any]) -> list[dict[str, str]]:
        """Every fallback this job took, from one grammar ("DEGRADED <stage> <code>: <message>" lines in its log),
        what the factory itself noted (spec "degraded"), and the "degraded" lists in rig/rig.json (a rig) or
        flat/views.json (a draw), when this job wrote them."""
        out: list[dict[str, str]] = []
        for entry in job.get("degraded") or []:
            add_degraded(out, entry)
        log_path = job.get("log")
        if log_path:
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            for entry in parse_degraded(text):
                add_degraded(out, entry)
        kind = job.get("kind")
        try:
            folder = stem_dir(self.root, str(job.get("stem") or ""))
        except ValueError:
            return out
        sidecar = None
        if kind == "rig":
            sidecar = folder / "rig" / "rig.json"
        elif kind in ("flat2d", "ideas"):
            sidecar = folder / "flat" / "views.json"
        elif job.get("views_json"):
            sidecar = Path(job["views_json"])
        if sidecar is not None and sidecar.is_file():
            # An older run's sidecar says nothing about this job (a generate's views.json is hash-checked already).
            fresh = kind is None or sidecar.stat().st_mtime >= float(job.get("started") or 0) - 1
            if fresh:
                for entry in json_degraded(sidecar):
                    add_degraded(out, entry)
        return out

    def mesh_of(self, mesh: str) -> tuple[Path, str]:
        """The supplied mesh and the asset it belongs to: a .glb file under out/<stem>/ (ValueError otherwise)."""
        path = resolve_under_root(self.root, mesh)
        out_dir = (self.root / "out").resolve()
        if path.suffix.lower() != ".glb":
            raise ValueError("mesh must be a .glb file")
        if not path.is_relative_to(out_dir) or len(path.relative_to(out_dir).parts) < 2:
            raise ValueError("mesh must be a .glb under out/<stem>/")
        stem = check_stem(path.relative_to(out_dir).parts[0])
        if not path.is_file():
            raise FileNotFoundError(f"no mesh at {mesh}")
        return path, stem

    def views_handoff(self, img: Path) -> tuple[Path | None, list[dict[str, str]]]:
        """The draw's views.json to pass to generate.sh (--views-json), or None, plus the degraded notes.
        Views pass only when the picture is out/<stem>/flat/source.png and its sha256 is the one views.json was
        made from (a concept take swapped in later must never bake another drawing's views). Every view file must
        be a picture (ValueError otherwise)."""
        out_dir = (self.root / "out").resolve()
        img = img.resolve()
        is_flat = img.name == "source.png" and img.parent.name == "flat" and img.parent.parent.parent == out_dir
        views_path = img.parent / "views.json" if is_flat else None
        if views_path is None or not views_path.is_file():
            # A drawing with no views: drawn with views 1, before views existed, or an older concept take picked
            # (a take carries its own views, so an older one without them leaves none rather than stale ones).
            why = "this drawing has no turnaround views; one picture" if is_flat else "one picture"
            return None, [degraded_entry("shape", "single_view", why)]
        try:
            doc = json.loads(views_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
        want = doc.get("source_sha256") if isinstance(doc, dict) else None
        if not want or str(want).lower() != file_sha256(img):
            return None, [degraded_entry("views", "stale",
                                         "flat/views.json was drawn for another picture; using one picture")]
        views = doc.get("views")
        if not isinstance(views, list) or not views:
            return None, [degraded_entry("shape", "single_view", "views.json lists no views")]
        folder = img.parent.parent
        for view in views:
            rel = view.get("file") if isinstance(view, dict) else None
            if not isinstance(rel, str) or not rel:
                raise ValueError(NO_PICTURE)
            require_picture(resolve_under_root(folder, rel))
        notes: list[dict[str, str]] = []
        if doc.get("consistent") is False:
            flaws = [str(f) for f in doc.get("flaws") or []]
            ids = {v.get("id") for v in views if isinstance(v, dict)}
            if flaws == ["side"] and {"front", "back"} <= ids and doc.get("canonical") in ("front", "back"):
                # Only the side views are off (an A-pose person's side came back turned to the back, twice, on
                # the farmer): front and back still agree, and two views beat one picture.
                kept = dict(doc, views=[v for v in views if v.get("id") in ("front", "back")], consistent=True,
                            flaws=[], dropped=["side"])
                fb = views_path.with_name("views.front_back.json")
                fb.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
                return fb, [degraded_entry("views", "sides_dropped",
                                           "the side views disagree with the front; built from front and back")]
            why = ", ".join(flaws) or "the views disagree"
            return None, [degraded_entry("views", "inconsistent", f"{why}; using one picture")]
        return views_path, notes

    def start_generate(
        self,
        *,
        image: str,
        quality: str = "stack",
        stem: str | None = None,
        poly: str = "medium",
        seed: str = "0",
        images: list[str] | None = None,
        prompt: str | None = None,
        psx: dict[str, Any] | None = None,
        intent: dict[str, Any] | None = None,
        mesh: str | None = None,
        paint: bool = True,
        fill: str | None = None,
        mirror_ends: bool | None = None,
        delight: bool | None = None,
    ) -> dict[str, Any]:
        """mesh: a .glb under out/<stem>/ that generate.sh takes as the shape instead of hy3d's (the stem is that
        folder's when none is given). paint False: no paint model, the mesh is unwrapped and only the photo bakes.
        The picture is required either way: isolate, the orientation check and the bake read it.
        fill "photo": every texel from the photo (generate.sh --fill photo); "none" or absent: as before.
        mirror_ends: mirror a front-on photo onto the far end too (generate.sh's default: on with fill photo).
        delight: take the photo's lighting out before the bake and the fill (generate.sh --delight; default off)."""
        # Everything is checked before the job slot is taken: a refused request leaves /status idle.
        quality = str(quality or "stack")
        if quality not in QUALITIES:
            raise ValueError(f"quality must be one of {', '.join(QUALITIES)}")
        if not isinstance(paint, bool):
            raise ValueError("paint must be true or false")
        if not paint and quality == "pbr":
            raise ValueError("quality pbr needs the paint; paint false has none")
        fill = None if fill in (None, "") else fill
        if fill not in (None, "photo", "none"):
            raise ValueError("fill must be photo or none")
        if fill == "photo" and quality == "draft":
            raise ValueError("fill photo needs a photo bake; quality draft has none")
        if mirror_ends is not None and not isinstance(mirror_ends, bool):
            raise ValueError("mirror_ends must be true or false")
        if delight is not None and not isinstance(delight, bool):
            raise ValueError("delight must be true or false")
        poly_in = str(poly or "medium")
        poly = normalize_poly(poly_in)
        # No 3D without a picture: the main image, every ref and every view must be one.
        if not image:
            raise ValueError(NO_PICTURE)
        paths: list[str] = [str(image)]
        for extra in images or []:
            if extra and str(extra) not in paths:
                paths.append(str(extra))
        resolved = [resolve_under_root(self.root, p) for p in paths]
        for p in resolved:
            require_picture(p)
        img, refs = resolved[0], resolved[1:]
        views_json, degraded = self.views_handoff(img)
        psx_opts = clean_psx(psx)
        intent_doc = clean_intent(intent)
        prompt = str(prompt or "").strip()
        mesh_path = None
        if mesh:
            mesh_path, mesh_stem = self.mesh_of(str(mesh))
            if stem and check_stem(str(stem)) != mesh_stem:
                raise ValueError(f"mesh must be under out/{stem}/")
            stem = mesh_stem
            # The shape never ran from the one picture, so "one picture" is no fallback to report.
            degraded = [d for d in degraded if not (d.get("stage") == "shape" and d.get("code") == "single_view")]
        if not stem:
            stem = auto_stem(self.root, prompt, [img, *refs])
        else:
            stem = check_stem(str(stem))

        job_id = uuid.uuid4().hex[:8]
        spec = {
            "job_id": job_id,
            "state": "running",
            "stem": stem,
            "image": str(img),
            "refs": [str(p) for p in refs],
            "prompt": prompt,
            "quality": quality,
            "poly": poly,
            "poly_in": poly_in,
            "poly_label": poly_label(poly),
            "seed": str(seed),
            "psx": psx_opts if quality == "psx" else {},
            "intent": intent_doc,
            "views_json": str(views_json) if views_json else None,
            "degraded": degraded,
            "mesh": str(mesh_path) if mesh_path else None,
            "paint": paint,
            "fill": fill or "none",
            "mirror_ends": mirror_ends,
            "delight": delight,
            "log": str(stem_dir(self.root, stem) / "generate.log"),
            "started": time.time(),
            "glb_path": None,
            "returncode": None,
            "error": None,
        }
        with self._lock:
            if self._job and self._job.get("state") == "running":
                raise RuntimeError("generate already running")
            if self._sync_draws:
                raise RuntimeError("the factory is busy with another job")
            # The log joins the job once _start_log has begun this run's, so /status never shows the last run's.
            self._job = {k: v for k, v in spec.items() if k != "log"}

        thread = threading.Thread(target=self._run_job, args=(spec,), daemon=False)
        thread.start()
        return {
            "ok": True,
            "accepted": True,
            "job_id": job_id,
            "state": "running",
            "stem": stem,
            "status_url": f"/status/{job_id}",
        }

    def start_flat(self, body: dict[str, Any]) -> dict[str, Any]:
        """Draw a 2D still (kind flat2d, then the picked steps) or Ideas (kind ideas) in the job slot.
        The prompt, step texts and clip go to the runner only: the job dict is what /status returns."""
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        kind = str(body.get("kind") or "flat2d")
        if kind not in ("flat2d", "ideas"):
            raise ValueError("kind must be flat2d or ideas")
        stem = check_stem(str(body.get("stem") or ""))
        prompt = " ".join(str(body.get("prompt") or "").split())
        drop = kind == "flat2d" and body.get("source") == "drop"
        if body.get("source") not in (None, "", "drop"):
            raise ValueError("source must be drop")
        if drop:
            if not (stem_dir(self.root, stem) / "flat" / "source.png").is_file():
                raise FileNotFoundError("no picture to work from; drop one first")
        reference = None
        if body.get("reference") not in (None, ""):
            # A photo the user added: a reference the four views are drawn from, never a view itself.
            if kind != "flat2d" or drop:
                raise ValueError("reference is for a flat2d draw")
            from flat2d.optional import wants_views
            from flat2d.photo import reference_path

            if not wants_views(clean_intent(body.get("intent")), body.get("views")):
                raise ValueError("reference needs a 3D-bound draw (intent output ps1, model or stack)")
            reference = reference_path(self.root, str(body["reference"]))
        if not drop and not prompt and reference is None:
            raise ValueError("empty prompt")
        backend = still_backend()
        if not drop and not backend:
            raise ValueError("No local image model. Install mflux.")
        if reference is not None:
            # The photo draw's own model: its times (draw-times.json) and estimates never mix with the words draw's.
            from flat2d.sheet import edit_backend

            backend = edit_backend() or backend
        count = 1
        if kind == "ideas":
            try:
                count = int(body.get("count") or 1)
            except (TypeError, ValueError):
                raise ValueError("count must be 1 to 8") from None
            if not 1 <= count <= drawjob.MAX_IDEAS:
                raise ValueError("count must be 1 to 8")
        raw_steps = body.get("steps") if kind == "flat2d" else None
        if raw_steps is not None and not isinstance(raw_steps, list):
            raise ValueError("steps must be a list")
        steps: list[str] = []
        for step in raw_steps or []:
            if step not in drawjob.STEPS:
                raise ValueError(f"unknown step {step!r}")
            if step not in steps:
                steps.append(step)
        payload: dict[str, Any] = {"prompt": prompt, "intent": clean_intent(body.get("intent"))}
        if reference is not None:
            payload["reference"] = str(reference)
        views = body.get("views")
        if views not in (None, ""):  # API-only opt-in (plan 1.1): views 1 is the old single draw, 4 the sheet
            if str(views).strip() not in ("1", "4"):
                raise ValueError("views must be 1 or 4")
            payload["views"] = int(str(views).strip())
        for key in ("pieces", "motion", "say", "sound"):
            if isinstance(body.get(key), str):
                payload[key] = body[key]
        if isinstance(body.get("video"), str) and body["video"]:
            payload["video"] = body["video"]
        elif isinstance(body.get("frames"), list):
            payload["frames"] = [f for f in body["frames"] if isinstance(f, str)]
        from flat2d.optional import model_loaded

        plan = drawjob.plan_for(kind, loaded=drop or model_loaded(photo=reference is not None), steps=steps,
                                count=count, drop=drop, views=reference is not None)
        job_id = uuid.uuid4().hex[:8]
        spec = {
            "job_id": job_id,
            "kind": kind,
            "state": "running",
            "stem": stem,
            "prompt": prompt,
            "steps": steps,
            "revise": bool(body.get("revise")) and kind == "flat2d",
            "source": "drop" if drop else None,
            "reference": str(reference.relative_to(self.root.resolve())) if reference is not None else None,
            "backend": backend,
            "count": count,
            "log": str(stem_dir(self.root, stem) / "draw.log"),
            "started": time.time(),
            "finished": None,
            "returncode": None,
            "error": None,
            "stage_times": {},
            "preview_url": None,
            "estimate_s": drawjob.estimate(self.root, backend, plan),
            "stage_est": drawjob.stage_estimates(drawjob.read_times(self.root).get(backend), plan),
            "result": {},
        }
        with self._lock:
            if (self._job and self._job.get("state") == "running") or stem in self._checking or self._sync_draws:
                raise RuntimeError("the factory is busy with another job")
            # A fresh log before the job is visible, so the first /status never shows the last run's.
            drawjob.fresh_log(Path(spec["log"]))
            self._job = dict(spec)
        threading.Thread(target=drawjob.run, args=(self, spec, payload), kwargs={"write_intent": write_intent},
                         daemon=False).start()
        projects.file_made(self.root, stem, body)  # accepted: the new asset joins its project (or none)
        return {"ok": True, "accepted": True, "job_id": job_id, "state": "running", "stem": stem, "kind": kind,
                "status_url": f"/status/{job_id}"}

    def sync_draw(self, on: bool) -> None:
        """A plain POST /2d/generate or /2d/idea began (True) or ended (False). Raises when a job runs."""
        with self._lock:
            if on:
                if self._job and self._job.get("state") == "running":
                    raise RuntimeError("the factory is busy with another job")
                self._sync_draws += 1
            else:
                self._sync_draws = max(0, self._sync_draws - 1)

    def start_rig(self, stem: str, body: dict[str, Any]) -> dict[str, Any]:
        """Rig an asset in the background (scripts/rig.py): the same job slot, /status and /cancel as generate."""
        stem = check_stem(stem)
        if not stem_dir(self.root, stem).is_dir():
            raise FileNotFoundError(f"no asset {stem!r}")
        argv = rig_argv(self.python, self.root, stem, body)
        return self._start_rig_job(stem, "rig", argv, stem_dir(self.root, stem) / "rig" / "rig.log")

    def start_render(self, stem: str, body: dict[str, Any]) -> dict[str, Any]:
        """Render the rig's contact sheet (scripts/rig_render.py, CPU) into rig/preview/, in the job slot."""
        stem = check_stem(stem)
        folder = stem_dir(self.root, stem)
        if not folder.is_dir():
            raise FileNotFoundError(f"no asset {stem!r}")
        if not (folder / "rigged.glb").is_file():
            raise ValueError("this asset has no rig yet")
        argv = render_argv(self.python, self.root, stem, (body or {}).get("job"))
        return self._start_rig_job(stem, "render", argv, folder / "rig" / "render.log")

    def _start_rig_job(self, stem: str, kind: str, argv: list[str], log: Path) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:8]
        spec = {"job_id": job_id, "kind": kind, "state": "running", "stem": stem, "argv": argv, "log": str(log),
                "started": time.time(), "returncode": None, "error": None}
        with self._lock:
            if (self._job and self._job.get("state") == "running") or stem in self._checking or self._sync_draws:
                raise RuntimeError("the factory is busy with another job")
            self._job = dict(spec)
        threading.Thread(target=self._run_rig_job, args=(spec,), daemon=False).start()
        return {"ok": True, "accepted": True, "job_id": job_id, "state": "running", "stem": stem, "kind": kind,
                "status_url": f"/status/{job_id}"}

    def detect_facing(self, body: dict[str, Any]) -> dict[str, Any]:
        """Which way the head points in a picture (scripts/facing.py, 3 VLM calls, ~11 s cold). Synchronous; holds
        the slot while it asks. body: {image: path under the root} or {stem} (its flat/source.png, else input.*)."""
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        if body.get("image"):
            png = resolve_under_root(self.root, str(body["image"]))
            if png.suffix.lower() not in UPLOAD_EXTS:
                raise ValueError("image must be a .png, .jpg or .webp")
            if not png.is_file():
                raise FileNotFoundError(f"image not found: {body['image']}")
        elif body.get("stem"):
            folder = stem_dir(self.root, check_stem(str(body["stem"])))
            png = next((folder / rel for rel in ("flat/source.png", "input.png", "input.jpg", "input.jpeg", "input.webp")
                        if (folder / rel).is_file()), None)
            if png is None:
                raise FileNotFoundError(f"no picture for {body['stem']!r}")
        else:
            raise ValueError("facing/detect needs image or stem")
        import facing  # numpy and the VLM client: only when asked

        self.sync_draw(True)
        try:
            det = facing.detect_image(png)
        except OSError as e:  # PIL can't read it (UnidentifiedImageError is an OSError)
            raise ValueError(f"not a picture: {png.name} ({e})") from None
        finally:
            self.sync_draw(False)
        conf = float(det.get("confidence") or 0.0)
        low = det.get("how") == "vlm" and (det.get("head") is None or conf < facing.ACCEPT)
        return {"ok": True, **det, "low": low, "accept": facing.ACCEPT,
                "image": png.relative_to(self.root).as_posix() if self.root in png.parents else str(png)}

    def _keep_rig_take(self, folder: Path, work: Callable[[], Any]) -> tuple[Any, dict[str, Any] | None]:
        """Run work() on a rig, then refresh the picked rig version in place if the rig on disk is still that version
        (its checks or contact sheet ride along; they never make a new version)."""
        entry = flow.read_manifest(folder)["stages"]["rig"]
        live = flow.stage_files(folder, "rig")
        found = flow.marks(folder, "rig", live)
        take = next((t for t in entry["takes"] if t.get("print") in found), None)
        before = flow.fingerprint(folder, flow.defining("rig", live))
        result = work()
        if take is not None and flow.fingerprint(folder, flow.defining("rig", flow.stage_files(folder, "rig"))) == before:
            flow.resave(self.root, folder.name, "rig", str(take["id"]))
        else:
            take = None
        return result, take

    def inspect_rig(self, stem: str) -> dict[str, Any]:
        """Check the rig on disk again (rig/inspect.json, rig/summary.json), on the CPU, in seconds. A version saved
        before the check is refreshed in place, so checking never makes a new version."""
        stem = check_stem(stem)
        folder = stem_dir(self.root, stem)
        if not folder.is_dir():
            raise FileNotFoundError(f"no asset {stem!r}")
        info_path = folder / "rig" / "rig.json"
        if not info_path.is_file():
            raise ValueError("this asset has no rig yet")
        try:
            rigged = str(json.loads(info_path.read_text(encoding="utf-8")).get("rigged") or "rigged.glb")
        except (OSError, ValueError):
            raise ValueError("this version's rig/rig.json is unreadable; Re-rig instead") from None
        for rel in ("rig/skinned_raw.glb", rigged, "rig/template_posed.glb"):
            if not (folder / rel).is_file():
                raise ValueError(f"this version is missing {rel}; Re-rig instead")
        with self._lock:
            if (self._job and self._job.get("state") == "running") or stem in self._checking:
                raise RuntimeError("the factory is busy with another job")
            self._checking.add(stem)
        try:
            import rig_inspect

            bundle, take = self._keep_rig_take(folder, lambda: rig_inspect.rebuild(folder))
        finally:
            with self._lock:
                self._checking.discard(stem)
        return {"ok": True, "stem": stem, "take": take["id"] if take else None, "checks": bundle["checks"],
                "summary": bundle["summary"]}

    def save_rig_weights(self, stem: str, raw: Any, exporter: Any = "default") -> dict[str, Any]:
        """The Weights panel's Save (scripts/rig_weights.py): the rig on disk is kept as a version first, the edited
        JOINTS_0/WEIGHTS_0 go into rigged.glb (and its same-mesh siblings), the FBX is exported again, and the result
        is saved as a new rig version. Holds the stem like a recheck, so no job or other write starts meanwhile."""
        import rig_weights

        stem = check_stem(stem)
        folder = stem_dir(self.root, stem)
        if not folder.is_dir():
            raise FileNotFoundError(f"no asset {stem!r}")
        if not (folder / "rig" / "rig.json").is_file():
            raise ValueError("this asset has no rig yet")
        with self._lock:
            if (self._job and self._job.get("state") == "running" and self._job.get("stem") == stem) or stem in self._checking:
                raise RuntimeError("a job is writing this asset")
            self._checking.add(stem)
        try:
            autosave(self.root, stem, ("rig",))
            result = rig_weights.save_weights(folder, raw, rig_weights.default_exporter if exporter == "default" else exporter)
            take = None
            try:
                take = flow.snapshot(self.root, stem, "rig", note="Edited weights")["take"]
            except (flow.FlowError, OSError, ValueError) as e:
                print(f"flow: could not save the edited weights as a rig version for {stem}: {e}", file=sys.stderr)
        finally:
            with self._lock:
                self._checking.discard(stem)
        return {"ok": True, "stem": stem, "take": take["id"] if take else None, **result}

    def _run_rig_job(self, spec: dict[str, Any]) -> None:
        """A rig (kind rig: saves a rig version before and after) or its contact sheet (kind render: refreshes the
        picked version in place, since a render is about the rig, not part of it)."""
        log = Path(spec["log"])
        folder = stem_dir(self.root, spec["stem"])
        render = spec.get("kind") == "render"
        try:
            if folder.is_dir() and not render:
                autosave(self.root, spec["stem"], ("rig",))
            log.parent.mkdir(parents=True, exist_ok=True)
            sheet = folder / "rig" / "preview" / "contact_sheet.png"
            sheet_before = sheet.stat().st_mtime_ns if sheet.is_file() else None

            def run() -> int:
                rc = -signal.SIGTERM
                if not (self._snapshot() or {}).get("cancel"):
                    with log.open("w", encoding="utf-8") as fh:
                        proc = subprocess.Popen(spec["argv"], cwd=str(self.root), stdout=fh, stderr=subprocess.STDOUT,
                                                text=True, start_new_session=True, env=rig_env(self.root))
                        with self._lock:
                            cancelled = False
                            if self._job and self._job.get("job_id") == spec["job_id"]:
                                self._job["pid"] = proc.pid
                                cancelled = bool(self._job.get("cancel"))
                        if cancelled:  # Cancel or reap arrived between the check above and the pid being known
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                            except OSError:
                                pass
                        rc = proc.wait()
                return rc

            rc = self._keep_rig_take(folder, run)[0] if render else run()
            cancelled = bool((self._snapshot() or {}).get("cancel"))
            if render:
                ok = rc == 0 and sheet.is_file() and sheet.stat().st_mtime_ns != sheet_before
            else:
                ok = rc == 0 and (folder / "rigged.glb").is_file()
            err = None
            word = "rig_render:" if render else "rig:"
            if cancelled:
                state, err = "cancelled", "Cancelled"
            elif ok:
                state = "ok"
                if not render:
                    autosave(self.root, spec["stem"], ("rig",))
            else:
                state = "error"
                text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
                lines = [ln.strip() for ln in text.splitlines() if (ln.strip().startswith(word) and not render) or "Error" in ln]
                if render and not lines:  # its own lines are progress; the last line says what stopped it
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][-1:]
                err = (lines[-1][len(word):].strip() if lines and lines[-1].startswith(word)
                       else (lines[-1] if lines else f"{word[:-1]} exited {rc}"))[:400]
            with self._lock:
                if self._job and self._job.get("job_id") == spec["job_id"]:
                    self._job.update({"state": state, "returncode": rc, "finished": time.time(), "error": err})
        except Exception as e:
            traceback.print_exc()
            with self._lock:
                if self._job and self._job.get("job_id") == spec["job_id"]:
                    self._job.update({"state": "error", "error": str(e), "finished": time.time()})

    def _run_generate_sh(self, spec: dict[str, Any]) -> int:
        log_path = Path(spec["log"])
        argv = generate_argv(
            self.root,
            image=Path(spec["image"]),
            quality=spec["quality"],
            stem=spec["stem"],
            poly=str(spec.get("poly_in") or spec["poly"]),
            seed=spec["seed"],
            refs=[Path(p) for p in spec.get("refs") or []],
            prompt=spec.get("prompt") or None,
            psx=spec.get("psx") or None,
            views_json=Path(spec["views_json"]) if spec.get("views_json") else None,
            degraded=list(spec.get("degraded") or []),
            mesh=Path(spec["mesh"]) if spec.get("mesh") else None,
            paint=spec.get("paint", True) is not False,
            fill=spec.get("fill"),
            mirror_ends=spec.get("mirror_ends"),
            delight=spec.get("delight"),
        )
        spec["argv"] = argv
        with self._lock:
            if self._job and self._job.get("job_id") == spec["job_id"]:
                self._job["argv"] = argv
        env = os.environ.copy()
        env["FACTORY_ROOT"] = str(self.root)
        if (self._snapshot() or {}).get("cancel"):
            return -signal.SIGTERM  # cancelled before it started
        # The log was started by _run_job (the unload note and the factory's DEGRADED lines); generate.sh appends.
        with log_path.open("a", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.root),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with self._lock:
                cancelled = False
                if self._job and self._job.get("job_id") == spec["job_id"]:
                    self._job["pid"] = proc.pid
                    cancelled = bool(self._job.get("cancel"))
            if cancelled:  # Cancel arrived between the check above and the pid being known
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except OSError:
                    pass
            return int(proc.wait())

    def _unload_draw(self) -> str:
        """Free the draw model before any 3D: paint alone peaks at 61-74 GB. Returns the log line saying what
        happened; a missing or failing unload is a warning in the job log, never a silent skip or a failed job."""
        try:
            from flat2d import optional
        except Exception as e:  # noqa: BLE001
            return f"generate: WARNING could not import flat2d.optional to unload the draw model ({e})"
        unload = getattr(optional, "unload", None)
        if not callable(unload):
            return ("generate: WARNING flat2d.optional.unload() is missing; the draw model may still be in memory "
                    "while the 3D stages run")
        try:
            unload()
        except Exception as e:  # noqa: BLE001
            return f"generate: WARNING unloading the draw model failed ({type(e).__name__}: {e})"
        return "generate: unloaded the draw model"

    def _start_log(self, spec: dict[str, Any]) -> None:
        """A fresh generate.log for this run (the last one is kept as generate.prev.log), opened with the
        unload note and a DEGRADED line for every fallback the factory chose before generate.sh runs."""
        log_path = Path(spec["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.is_file():
            log_path.replace(log_path.with_name("generate.prev.log"))
        lines = [self._unload_draw()]
        if spec.get("views_json"):
            lines.append(f"generate: views from {spec['views_json']}")
        lines += [degraded_line(e) for e in spec.get("degraded") or []]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self._lock:
            if self._job and self._job.get("job_id") == spec["job_id"]:
                self._job["log"] = spec["log"]

    def _run_job(self, spec: dict[str, Any]) -> None:
        try:
            # Keep what this run is about to overwrite.
            if stem_dir(self.root, spec["stem"]).is_dir():
                autosave(self.root, spec["stem"], ("concept", "mesh", "stack"))
            if spec.get("intent"):
                write_intent(self.root, spec["stem"], spec["intent"])
            self._start_log(spec)
            rc = -signal.SIGTERM if (self._snapshot() or {}).get("cancel") else self._runner(spec)
            if (self._snapshot() or {}).get("cancel"):
                with self._lock:
                    if self._job and self._job.get("job_id") == spec["job_id"]:
                        self._job.update({"state": "cancelled", "returncode": rc, "finished": time.time(), "error": "Cancelled"})
                return
            glb = pick_glb(self.root, spec["stem"])
            report = inspect_glb(str(glb)) if glb else {"ok": False, "error": "no GLB"}
            state = "ok" if rc == 0 and report.get("ok") else "error"
            err = None if state == "ok" else _generate_error(rc, report, spec.get("log"))
            if state == "ok":
                autosave(self.root, spec["stem"], ("concept", "mesh", "stack"))
            degraded = self._degraded(spec)
            with self._lock:
                if self._job and self._job.get("job_id") == spec["job_id"]:
                    self._job.update(
                        {
                            "state": state,
                            "returncode": rc,
                            "glb_path": str(glb) if glb else None,
                            "inspect": report,
                            "finished": time.time(),
                            "error": err,
                            "degraded": degraded,
                            "result": {"glb_path": str(glb) if glb else None, "degraded": degraded},
                        }
                    )
        except Exception as e:
            traceback.print_exc()
            with self._lock:
                if self._job and self._job.get("job_id") == spec["job_id"]:
                    self._job.update(
                        {
                            "state": "error",
                            "error": str(e),
                            "finished": time.time(),
                        }
                    )


def _generate_error(rc: int, report: dict[str, Any] | None, log_path: str | None) -> str:
    text = ""
    if log_path:
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
    miss = re.search(r"ModuleNotFoundError: No module named '([^']+)'", text)
    if miss:
        return f"missing python module {miss.group(1)}"
    for ln in reversed(text.splitlines()):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("generate:") or "Error" in s:
            if s == "no GLB":
                continue
            return s[:400]
    err = (report or {}).get("error")
    if err and err != "no GLB":
        return str(err)
    if rc:
        return f"generate exited {rc}"
    return "generate produced no GLB"


def _json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, indent=2).encode("utf-8") + b"\n"
    return status, body, "application/json; charset=utf-8"


def dispatch(factory: Factory, method: str, path: str, query: dict[str, list[str]], body: bytes) -> tuple[int, bytes, str]:
    path = unquote(path)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    parts = [p for p in path.split("/") if p]

    # A 2D draw as a job (flat2d/drawjob.py). Not a FLAT_WRITES action: the runner saves its own takes.
    if method == "POST" and path == "/2d/draw":
        try:
            raw = json.loads(body.decode("utf-8") or "{}") if body else {}
            return _json_bytes(factory.start_flat(raw if isinstance(raw, dict) else {}), 202)
        except json.JSONDecodeError as e:
            return _json_bytes({"ok": False, "error": f"invalid JSON: {e}"}, 400)
        except (ValueError, FileNotFoundError) as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)
    if method == "GET" and len(parts) == 3 and parts[0] == "2d" and parts[2] == "drawing":
        try:
            drawing = stem_dir(factory.root, check_stem(parts[1])) / "flat" / "drawing.jpg"
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        try:
            return 200, drawing.read_bytes(), "image/jpeg"
        except OSError:
            return _json_bytes({"ok": False, "error": "no drawing"}, 404)
    # The plain draws wait while any job runs, and a job waits for them: they share the model's hook.
    sync_draw = method == "POST" and path in ("/2d/generate", "/2d/idea")
    if sync_draw:
        try:
            factory.sync_draw(True)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "busy": True, "error": str(e)}, 409)
    try:
        return _dispatch(factory, method, path, parts, query, body)
    finally:
        if sync_draw:
            factory.sync_draw(False)


def _dispatch(
    factory: Factory, method: str, path: str, parts: list[str], query: dict[str, list[str]], body: bytes
) -> tuple[int, bytes, str]:
    # 2D writes save a take before (so nothing on disk is lost) and after (so the result is one).
    flat_stem = None
    if method == "POST" and len(parts) == 2 and parts[0] == "2d" and parts[1] in FLAT_WRITES:
        try:
            flat_stem = check_stem(str(json.loads(body.decode("utf-8") or "{}").get("stem") or ""))
        except (ValueError, AttributeError):
            flat_stem = None
        # A running job (a draw, its steps, a rig) owns the asset: a write now would race it.
        if flat_stem and factory.writing(flat_stem):
            return _json_bytes({"ok": False, "busy": True, "error": "a job is writing this asset"}, 409)
        if flat_stem and stem_dir(factory.root, flat_stem).is_dir():
            autosave(factory.root, flat_stem, FLAT_WRITES[parts[1]])
    flat = handle_flat2d(factory.root, method, path, body)
    if flat is not None:
        if flat_stem and flat[0] < 300:
            autosave(factory.root, flat_stem, FLAT_WRITES[parts[1]])
        return flat

    flow = handle_flow(factory.root, method, path, body, busy=factory.writing)
    if flow is not None:
        return flow

    grouped = projects.handle(factory.root, method, parts, body)
    if grouped is not None:
        return grouped

    if method == "POST" and len(parts) == 2 and parts[0] == "export":
        try:
            export_stem = check_stem(parts[1])
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        if factory.writing(export_stem):
            return _json_bytes({"ok": False, "error": "a generate is writing this asset", "busy": True}, 409)
        if stem_dir(factory.root, export_stem).is_dir():
            autosave(factory.root, export_stem, ("export",))
        result = handle_export(factory.root, method, path)
        if result[0] < 300:
            autosave(factory.root, export_stem, ("export",))
        return result
    exported = handle_export(factory.root, method, path)
    if exported is not None:
        return exported

    if method == "GET" and path in {"/", ""}:
        return _json_bytes(
            {
                "ok": True,
                "service": "ai-asset-pipeline",
                "routes": [
                    "GET /health",
                    "GET /inspect/<stem>",
                    "POST /generate",
                    "GET /status",
                    "GET /status/<job_id>",
                    "POST /cancel/<job_id>",
                    "POST /intent/<stem>",
                    "POST /rig/<stem>",
                    "POST /rig/<stem>/inspect",
                    "POST /rig/<stem>/render",
                    "POST /rig/<stem>/weights",
                    "GET /rig/plans",
                    "POST /facing/detect",
                    "GET /glb/<stem>",
                    "GET /stack/<stem>",
                    "POST /stack/<stem>",
                    "POST /upload",
                    "POST /generate",
                    "GET /assets",
                    "PATCH /assets/<stem>",
                    "DELETE /assets/<stem>",
                    "POST /clean",
                    "POST /paint/stack",
                    "POST /paint/mesh",
                    "GET /2d/status",
                    "POST /2d/draw",
                    "GET /2d/<stem>/drawing",
                    "POST /2d/drop",
                    "POST /2d/split",
                    "POST /2d/save",
                    "GET /2d/<stem>/ideas",
                    "POST /2d/idea",
                    "POST /2d/keep",
                    "POST /2d/clear",
                    "POST /2d/use",
                    "GET /flow/<stem>",
                    "POST /flow/<stem>/snapshot",
                    "POST /flow/<stem>/pick",
                    "GET /projects",
                    "POST /projects",
                    "GET /projects/<id>",
                    "PATCH /projects/<id>",
                    "DELETE /projects/<id>",
                    "POST /projects/<id>/assets",
                    "DELETE /projects/<id>/assets/<stem>",
                    "GET /export/<stem>",
                    "POST /export/<stem>",
                ],
            }
        )

    if method == "GET" and path == "/health":
        return _json_bytes(factory.health())

    if path == "/activity":
        if method == "GET":
            try:
                since = int((query.get("since") or ["0"])[0])
            except ValueError:
                return _json_bytes({"ok": False, "error": "since must be a number"}, 400)
            return _json_bytes(factory.activity.since(since))
        if method == "POST":
            try:
                raw = json.loads(body.decode("utf-8") or "{}") if body else {}
                if not isinstance(raw, dict):
                    raise ValueError("send one entry as an object")
                entry_id = raw.pop("id", None)
                return _json_bytes({"ok": True, "entry": factory.activity.put(entry_id, **raw)})
            except (ValueError, TypeError) as e:
                return _json_bytes({"ok": False, "error": str(e)}, 400)

    if method == "GET" and path == "/assets":
        return _json_bytes(factory.list_assets())

    if method == "PATCH" and len(parts) == 2 and parts[0] == "assets":
        try:
            payload = json.loads(body.decode("utf-8") or "{}") if body else {}
            dest = payload.get("stem") or payload.get("name") or payload.get("to") or ""
            result = factory.rename_asset(parts[1], str(dest))
        except json.JSONDecodeError as e:
            return _json_bytes({"ok": False, "error": f"invalid JSON: {e}"}, 400)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)
        return _json_bytes(result, 200 if result.get("ok") else 404)

    if method == "DELETE" and len(parts) == 2 and parts[0] == "assets":
        try:
            payload = factory.delete_asset(parts[1])
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)
        return _json_bytes(payload, 200 if payload.get("ok") else 404)

    if method == "GET" and len(parts) == 2 and parts[0] == "inspect":
        try:
            lod = (query.get("lod") or [None])[0]
            report = factory.inspect_stem(parts[1], lod)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        status = 200 if report.get("ok") else 404
        return _json_bytes(report, status)

    if method == "GET" and (path == "/status" or (len(parts) == 2 and parts[0] == "status")):
        job_id = parts[1] if len(parts) == 2 else (query.get("job") or [None])[0]
        payload = factory.status(job_id)
        status = 200 if payload.get("ok") else 404
        return _json_bytes(payload, status)

    if method == "POST" and len(parts) == 2 and parts[0] == "stack":
        # A sprite stack straight from the concept picture (scripts/stack_image.py). No mesh.
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        if not stem_dir(factory.root, stem).is_dir():
            return _json_bytes({"ok": False, "error": "no such asset", "stem": stem}, 404)
        if factory.writing(stem):
            return _json_bytes({"ok": False, "error": "a generate is writing this asset", "busy": True}, 409)
        autosave(factory.root, stem, ("stack",))
        argv = [
            str(factory.python),
            str(factory.root / "scripts" / "stack_image.py"),
            "--stem",
            stem,
            "--root",
            str(factory.root),
        ]
        proc = subprocess.run(argv, cwd=str(factory.root), capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip().splitlines()[-1]
            return _json_bytes({"ok": False, "error": err.removeprefix("stack_image: "), "stem": stem}, 400 if proc.returncode == 2 else 500)
        try:
            info = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            info = {}
        autosave(factory.root, stem, ("stack",), made_from=("concept",))
        return _json_bytes({**info, "ok": True, "stem": stem})

    if method == "GET" and len(parts) == 2 and parts[0] == "stack":
        try:
            check_stem(parts[1])
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        meta_path = stem_dir(factory.root, parts[1]) / "stack" / "stack.json"
        if not meta_path.is_file():
            return _json_bytes({"ok": False, "error": "no sprite stack", "stem": parts[1]}, 404)
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return _json_bytes({"ok": False, "error": str(e)}, 500)
        payload["ok"] = True
        payload["stem"] = parts[1]
        return _json_bytes(payload)

    if method == "GET" and len(parts) >= 2 and parts[0] == "glb":
        stem = parts[1]
        lod = parts[2] if len(parts) > 2 else (query.get("lod") or [None])[0]
        try:
            glb = factory.glb_file(stem, lod)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        if glb is None or not glb.is_file():
            return _json_bytes({"ok": False, "error": "no GLB for stem", "stem": stem}, 404)
        data = glb.read_bytes()
        return 200, data, "model/gltf-binary"

    if method == "POST" and path in {"/paint/stack", "/paint/mesh"}:
        kind = "stack" if path.endswith("stack") else "mesh"
        stem = (query.get("stem") or [None])[0]
        if not stem:
            return _json_bytes({"ok": False, "error": "stem query required"}, 400)
        try:
            check_stem(stem)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        if not body:
            return _json_bytes({"ok": False, "error": "empty body"}, 400)
        autosave(factory.root, stem, (kind,))
        tmp = stem_dir(factory.root, stem) / f".paint_{kind}.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(body)
        argv = [
            str(factory.python),
            str(factory.root / "scripts" / "paint_save.py"),
            "--stem",
            stem,
            "--kind",
            kind,
            "--png",
            str(tmp),
        ]
        proc = subprocess.run(argv, cwd=str(factory.root), capture_output=True, text=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            return _json_bytes({"ok": False, "error": err, "stem": stem}, 500)
        try:
            info = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            info = {"ok": True}
        info["ok"] = True
        info["stem"] = stem
        autosave(factory.root, stem, (kind,))
        return _json_bytes(info)

    if method == "POST" and path == "/clean":
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
            result = factory.clean_mesh(payload)
        except (json.JSONDecodeError, ValueError) as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except FileNotFoundError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 404)
        return _json_bytes(result, 200 if result.get("ok") else 500)

    if method == "POST" and path == "/upload":
        name = (query.get("name") or ["upload.jpg"])[0]
        data = body
        trim = body.lstrip()
        if trim.startswith(b"{"):
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError as e:
                return _json_bytes({"ok": False, "error": f"invalid upload json: {e}"}, 400)
            name = str(payload.get("name") or name)
            raw = payload.get("data") or payload.get("bytes") or ""
            if not raw:
                return _json_bytes({"ok": False, "error": "empty upload"}, 400)
            if isinstance(raw, str):
                import base64

                try:
                    data = base64.b64decode(raw, validate=False)
                except Exception as e:
                    return _json_bytes({"ok": False, "error": f"invalid upload base64: {e}"}, 400)
            elif isinstance(raw, list):
                data = bytes(raw)
            else:
                return _json_bytes({"ok": False, "error": "empty upload"}, 400)
        try:
            return _json_bytes(factory.save_upload(name, data), 201)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            return _json_bytes({"ok": False, "error": str(e)}, 500)

    if method == "GET" and path == "/rig/plans":
        try:
            return _json_bytes({"ok": True, "plans": rig_plans()})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return _json_bytes({"ok": False, "error": f"couldn't read the body plans: {e}"}, 500)

    if method == "POST" and path == "/facing/detect":
        try:
            raw = json.loads(body.decode("utf-8") or "{}") if body else {}
            return _json_bytes(factory.detect_facing(raw if isinstance(raw, dict) else {}))
        except json.JSONDecodeError as e:
            return _json_bytes({"ok": False, "error": f"invalid JSON: {e}"}, 400)
        except FileNotFoundError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 404)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return _json_bytes({"ok": False, "error": f"couldn't check the facing: {e}"}, 500)

    if method == "POST" and len(parts) == 3 and parts[0] == "rig" and parts[2] == "render":
        try:
            raw = json.loads(body.decode("utf-8") or "{}") if body else {}
            return _json_bytes(factory.start_render(parts[1], raw if isinstance(raw, dict) else {}), 202)
        except json.JSONDecodeError as e:
            return _json_bytes({"ok": False, "error": f"invalid JSON: {e}"}, 400)
        except FileNotFoundError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 404)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)

    if method == "POST" and len(parts) == 3 and parts[0] == "rig" and parts[2] == "weights":
        try:
            raw = json.loads(body.decode("utf-8") or "{}") if body else {}
            return _json_bytes(factory.save_rig_weights(parts[1], raw))
        except json.JSONDecodeError as e:
            return _json_bytes({"ok": False, "error": f"invalid JSON: {e}"}, 400)
        except FileNotFoundError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 404)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)
        except Exception as e:
            traceback.print_exc()
            return _json_bytes({"ok": False, "error": f"couldn't save the weights: {e}"}, 500)

    if method == "POST" and len(parts) == 3 and parts[0] == "rig" and parts[2] == "inspect":
        try:
            return _json_bytes(factory.inspect_rig(parts[1]))
        except FileNotFoundError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 404)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)
        except Exception as e:
            traceback.print_exc()
            return _json_bytes({"ok": False, "error": f"couldn't check this rig: {e}"}, 500)

    if method == "POST" and len(parts) == 2 and parts[0] == "rig":
        try:
            raw = json.loads(body.decode("utf-8") or "{}") if body else {}
            return _json_bytes(factory.start_rig(parts[1], raw if isinstance(raw, dict) else {}), 202)
        except json.JSONDecodeError as e:
            return _json_bytes({"ok": False, "error": f"invalid JSON: {e}"}, 400)
        except FileNotFoundError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 404)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)

    if method == "POST" and len(parts) == 2 and parts[0] == "cancel":
        result = factory.cancel(parts[1])
        return _json_bytes(result, 200 if result.get("ok") else 404)

    if method == "POST" and len(parts) == 2 and parts[0] == "intent":
        try:
            raw = json.loads(body.decode("utf-8") or "{}") if body else {}
            return _json_bytes(factory.set_intent(parts[1], raw))
        except json.JSONDecodeError as e:
            return _json_bytes({"ok": False, "error": f"invalid JSON: {e}"}, 400)
        except FileNotFoundError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 404)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)

    if method == "POST" and path == "/generate":
        payload: dict[str, Any] = {}
        for k, v in query.items():
            if v:
                payload[k] = v[0]
        if body:
            try:
                payload.update(json.loads(body.decode("utf-8") or "{}"))
            except json.JSONDecodeError as e:
                return _json_bytes({"ok": False, "error": f"invalid JSON: {e}"}, 400)
        try:
            raw_images = payload.get("images") or payload.get("refs") or []
            if isinstance(raw_images, str):
                raw_images = [p for p in raw_images.split(",") if p]
            if not isinstance(raw_images, list):
                raw_images = []
            paint = payload.get("paint")
            if paint is None:
                paint = True
            elif isinstance(paint, str):  # from the query string
                paint = {"true": True, "on": True, "1": True, "false": False, "off": False, "0": False}.get(
                    paint.strip().lower(), paint)
            mirror_ends = payload.get("mirror_ends")
            if isinstance(mirror_ends, str):  # from the query string
                mirror_ends = {"true": True, "on": True, "1": True, "false": False, "off": False, "0": False}.get(
                    mirror_ends.strip().lower(), mirror_ends)
            delight = payload.get("delight")
            if isinstance(delight, str):  # from the query string
                delight = {"true": True, "on": True, "1": True, "false": False, "off": False, "0": False}.get(
                    delight.strip().lower(), delight)
            mesh = payload.get("mesh")
            if mesh not in (None, "") and not isinstance(mesh, str):
                raise ValueError("mesh must be a path to a .glb under out/<stem>/")
            result = factory.start_generate(
                image=str(payload.get("image") or ""),
                quality=str(payload.get("quality") or "stack"),
                stem=payload.get("stem") or None,
                poly=str(payload.get("poly") or "medium"),
                seed=str(payload.get("seed") or "0"),
                images=[str(p) for p in raw_images],
                prompt=str(payload.get("prompt") or "") or None,
                psx={k: payload.get(k if k == "facing" else f"psx_{k}") for k in PSX_KEYS},
                intent=payload.get("intent") if isinstance(payload.get("intent"), dict) else None,
                mesh=mesh or None,
                paint=paint,
                fill=payload.get("fill"),
                mirror_ends=mirror_ends,
                delight=delight,
            )
            return _json_bytes(result, 202)
        except FileNotFoundError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except ValueError as e:
            return _json_bytes({"ok": False, "error": str(e)}, 400)
        except RuntimeError as e:
            return _json_bytes({"ok": False, "error": str(e), "busy": True}, 409)
        except Exception as e:
            traceback.print_exc()
            return _json_bytes({"ok": False, "error": str(e)}, 500)

    return _json_bytes({"ok": False, "error": f"not found: {method} {path}"}, 404)


def make_handler(factory: Factory):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return b""
            return self.rfile.read(n)

        def _cors(self) -> None:
            origin = self.headers.get("Origin")
            if origin and local_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _refuse(self) -> bool:
            """403 for a web page that isn't the app or this machine, or a Host that isn't loopback."""
            if local_origin(self.headers.get("Origin")) and local_host(self.headers.get("Host")):
                return False
            body = json.dumps({"error": "origin not allowed"}).encode()
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            if content_type == "model/gltf-binary":
                stem = "asset.glb"
                path = urlparse(self.path).path
                bits = [p for p in unquote(path).split("/") if p]
                if len(bits) >= 2:
                    stem = bits[1] + ".glb"
                self.send_header("Content-Disposition", f'attachment; filename="{stem}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self._refuse():
                return
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "GET", parsed.path, parse_qs(parsed.query), b"")
            self._send(status, body, ctype)

        def do_POST(self) -> None:  # noqa: N802
            if self._refuse():
                return
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "POST", parsed.path, parse_qs(parsed.query), self._read_body())
            self._send(status, body, ctype)

        def do_PATCH(self) -> None:  # noqa: N802
            if self._refuse():
                return
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "PATCH", parsed.path, parse_qs(parsed.query), self._read_body())
            self._send(status, body, ctype)

        def do_DELETE(self) -> None:  # noqa: N802
            if self._refuse():
                return
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "DELETE", parsed.path, parse_qs(parsed.query), b"")
            self._send(status, body, ctype)

        def do_OPTIONS(self) -> None:  # noqa: N802
            if self._refuse():
                return
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802
            if self._refuse():
                return
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "GET", parsed.path, parse_qs(parsed.query), b"")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()

    return Handler


def stop_handler(factory: Factory, httpd: ThreadingHTTPServer):
    """SIGTERM/SIGINT: reap the job first, then stop serving. The app (desktop/internal/sidecar/proc.go) sends
    SIGTERM and SIGKILLs the group 100 ms later, so the kill cannot wait. shutdown() blocks until serve_forever
    returns, and serve_forever is on this (the main) thread, so it runs on its own."""
    stopping = threading.Event()

    def handle(signum, frame) -> None:
        # Set before reaping: a second signal landing while reap() holds the factory lock would re-enter it and
        # deadlock. serve()'s finally reaps again, so a signal that returns here loses nothing.
        if stopping.is_set():
            return
        stopping.set()
        factory.reap()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    return handle


def serve(host: str, port: int, root: Path, log_path: Path | None = None) -> None:
    if host not in ALLOWED_HOSTS:
        raise SystemExit(f"loopback only (got {host})")
    factory = Factory(root)
    httpd = ThreadingHTTPServer((host, port), make_handler(factory))
    health = factory.health()
    banner = (
        f"factory_http listening http://{host}:{port}\n"
        f"  python={health['python']}\n"
        f"  gpu={health['gpu']}\n"
        f"  GET /health  GET /inspect/<stem>  POST /generate  GET /status  GET /glb/<stem>  POST /2d/draw\n"
    )
    sys.stdout.write(banner)
    sys.stdout.flush()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(banner, encoding="utf-8")
    pid_path = root / "out" / "factory_http.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    watch_stop = threading.Event()
    threading.Thread(target=factory.watch_jobs, args=(watch_stop,), name="job-watch", daemon=True).start()
    handle = stop_handler(factory, httpd)
    before = {sig: signal.signal(sig, handle) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        httpd.serve_forever()
    finally:
        for sig, old in before.items():
            signal.signal(sig, old)
        watch_stop.set()
        factory.reap()  # also when serve_forever stopped some other way; after the handlers, so it is not re-entered
        httpd.server_close()
        try:
            pid_path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("FACTORY_HTTP_PORT") or DEFAULT_PORT))
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root()
    log_path = root / "out" / "factory_http.log"
    serve(args.host, args.port, root, log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
