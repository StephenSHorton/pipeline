#!/usr/bin/env python3
"""Loopback HTTP adapter over the local MLX factory. Curl only; no MCP.

GET  /health
GET  /inspect/<stem>
POST /generate
GET  /status
GET  /status/<job_id>
POST /cancel/<job_id>
POST /intent/<stem>
GET  /glb/<stem>
GET  /stack/<stem>
POST /export/<stem>
POST /upload
POST /generate
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

from flat2d.routes import handle as handle_flat2d
from flow import FLAT_WRITES, autosave
from flow import handle as handle_flow
from export_asset import handle as handle_export
from factory_env import (
    PSX_KEYS,
    auto_stem,
    check_stem,
    clean_psx,
    factory_python,
    generate_argv,
    normalize_poly,
    pick_glb,
    poly_label,
    repo_root,
    resolve_under_root,
    slug_stem,
    stem_dir,
)
from inspect_glb import inspect as inspect_glb
from clean_mesh import clean_name

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
UPLOAD_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
ALLOWED_HOSTS = LOOPBACK
DEFAULT_PORT = 8766


def _dir_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return False


def factory_features(root: Path) -> dict[str, Any]:
    remesh = (root / "bin" / "autoremesher").is_file() or (root / "bin" / "AutoRemesher.app").is_dir()
    lift = (root / "bin" / "lift-subject").is_file()
    return {
        "isolate": "vision" if lift else "none",
        "shape": "mini-mlx",
        "paint": "mlx-rgb",
        "paint_pbr": True,
        "shape_mv": False,
        "unirig": False,
        "remesh": remesh,
    }


# What each generate.sh stage is called in the app (generate.sh prints "STAGES a b c" then "STAGE a <label>").
STAGE_LABELS = {
    "isolate": "Cut out the subject",
    "shape": "Build the shape",
    "reduce": "Reduce the mesh",
    "paint": "Paint the texture",
    "retexture": "Bake the photo on",
    "psx": "PS1 low-poly",
    "lods": "Detail levels",
    "inspect": "Check the model",
    "stack": "Sprite stack",
    "preview": "Thumbnail",
}
SUB_PROGRESS = re.compile(r"\[\s*(\d{1,3})%\]")
# What the asset is meant to be; the app picks its stages and Export card from this.
OUTPUTS = ("ps1", "model", "stack", "sprite")


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


def clean_intent(raw: Any) -> dict[str, Any] | None:
    """The app's chosen output for an asset, checked: {"output": ps1|model|stack|sprite, ...settings}."""
    if not isinstance(raw, dict) or raw.get("output") not in OUTPUTS:
        return None
    out: dict[str, Any] = {"output": raw["output"]}
    psx = clean_psx({k: raw.get(k) for k in PSX_KEYS})
    if psx:
        out["psx"] = psx
    if raw.get("poly") in ("high", "medium", "low"):
        out["poly"] = raw["poly"]
    steps = raw.get("steps")
    if isinstance(steps, list):
        out["steps"] = [str(x) for x in steps if x in ("pieces", "motion", "say", "sound")]
    return out


def write_intent(root: Path, stem: str, intent: dict[str, Any]) -> None:
    path = stem_dir(root, stem) / "intent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**intent, "made": int(time.time())}, indent=2) + "\n", encoding="utf-8")


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
        self._runner = generate_runner or self._run_generate_sh

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
            ext = ".jpg"
        stem = NAME_RE.sub("-", Path(raw).stem).strip("-._") or "upload"
        stem = stem[:48]
        check_stem(stem)
        dest = self.root / "in" / f"{stem}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {
            "ok": True,
            "path": f"in/{dest.name}",
            "bytes": len(data),
            "stem": stem,
        }

    def list_assets(self) -> dict[str, Any]:
        out = self.root / "out"
        assets: list[dict[str, Any]] = []
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
        return {"ok": True, "stem": dest_slug, "from": stem, "renamed": True, "path": f"out/{dest_slug}"}

    def delete_asset(self, stem: str) -> dict[str, Any]:
        stem = check_stem(stem)
        job = self._snapshot()
        if job and job.get("state") == "running" and job.get("stem") == stem:
            raise RuntimeError("cannot delete while generate is running")
        d = stem_dir(self.root, stem)
        if not d.is_dir():
            return {"ok": False, "error": "not found", "stem": stem}
        shutil.rmtree(d)
        return {"ok": True, "deleted": True, "stem": stem, "path": f"out/{stem}"}

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
        job = self._snapshot()
        return bool(job and job.get("state") == "running" and job.get("stem") == stem)

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        job = self._snapshot()
        if job is None:
            return {"ok": True, "state": "idle", "job_id": None}
        if job_id and job.get("job_id") != job_id:
            return {"ok": False, "error": "unknown job", "job_id": job_id, "state": "unknown"}
        out = dict(job)
        out["ok"] = True
        log_path = out.get("log")
        if log_path and Path(log_path).is_file():
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                out["log_tail"] = "\n".join(text.splitlines()[-40:])
                out["progress"] = parse_progress(text)
            except OSError:
                out["log_tail"] = ""
        return out

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
    ) -> dict[str, Any]:
        if quality not in {"draft", "textured", "pbr", "stack", "psx"}:
            quality = "stack"
        poly_in = str(poly or "medium")
        poly = normalize_poly(poly_in)
        paths: list[str] = []
        if image:
            paths.append(str(image))
        for extra in images or []:
            if extra and str(extra) not in paths:
                paths.append(str(extra))
        if not paths:
            raise FileNotFoundError("image not found")
        resolved = [resolve_under_root(self.root, p) for p in paths]
        img = resolved[0]
        if not img.is_file():
            raise FileNotFoundError(f"image not found: {paths[0]}")
        refs: list[Path] = []
        for extra_path, extra_src in zip(resolved[1:], paths[1:]):
            if not extra_path.is_file():
                raise FileNotFoundError(f"image not found: {extra_src}")
            refs.append(extra_path)
        prompt = str(prompt or "").strip()
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
            "psx": clean_psx(psx) if quality == "psx" else {},
            "intent": clean_intent(intent),
            "started": time.time(),
            "glb_path": None,
            "returncode": None,
            "error": None,
        }
        with self._lock:
            if self._job and self._job.get("state") == "running":
                raise RuntimeError("generate already running")
            self._job = dict(spec)

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

    def _run_generate_sh(self, spec: dict[str, Any]) -> int:
        log_path = stem_dir(self.root, spec["stem"]) / "generate.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        spec["log"] = str(log_path)
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
        )
        spec["argv"] = argv
        with self._lock:
            if self._job and self._job.get("job_id") == spec["job_id"]:
                self._job["log"] = spec["log"]
                self._job["argv"] = argv
        env = os.environ.copy()
        env["FACTORY_ROOT"] = str(self.root)
        # One log per run: the previous one is kept as generate.prev.log so /status never shows an old run.
        if log_path.is_file():
            log_path.replace(log_path.with_name("generate.prev.log"))
        if (self._snapshot() or {}).get("cancel"):
            return -signal.SIGTERM  # cancelled before it started
        with log_path.open("w", encoding="utf-8") as fh:
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

    def _run_job(self, spec: dict[str, Any]) -> None:
        try:
            # Keep what this run is about to overwrite.
            if stem_dir(self.root, spec["stem"]).is_dir():
                autosave(self.root, spec["stem"], ("concept", "mesh", "stack"))
            if spec.get("intent"):
                write_intent(self.root, spec["stem"], spec["intent"])
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

    # 2D writes save a take before (so nothing on disk is lost) and after (so the result is one).
    flat_stem = None
    if method == "POST" and len(parts) == 2 and parts[0] == "2d" and parts[1] in FLAT_WRITES:
        try:
            flat_stem = check_stem(str(json.loads(body.decode("utf-8") or "{}").get("stem") or ""))
        except (ValueError, AttributeError):
            flat_stem = None
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
                    "GET /export/<stem>",
                    "POST /export/<stem>",
                ],
            }
        )

    if method == "GET" and path == "/health":
        return _json_bytes(factory.health())

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
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

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
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "GET", parsed.path, parse_qs(parsed.query), b"")
            self._send(status, body, ctype)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "POST", parsed.path, parse_qs(parsed.query), self._read_body())
            self._send(status, body, ctype)

        def do_PATCH(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "PATCH", parsed.path, parse_qs(parsed.query), self._read_body())
            self._send(status, body, ctype)

        def do_DELETE(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "DELETE", parsed.path, parse_qs(parsed.query), b"")
            self._send(status, body, ctype)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            status, body, ctype = dispatch(factory, "GET", parsed.path, parse_qs(parsed.query), b"")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()

    return Handler


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
        f"  GET /health  GET /inspect/<stem>  POST /generate  GET /status  GET /glb/<stem>\n"
    )
    sys.stdout.write(banner)
    sys.stdout.flush()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(banner, encoding="utf-8")
    pid_path = root / "out" / "factory_http.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    try:
        httpd.serve_forever()
    finally:
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
