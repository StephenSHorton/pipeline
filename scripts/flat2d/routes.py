"""Loopback routes for the 2D lane. Returns None when the path is not ours."""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from factory_env import check_stem, stem_dir

from flat2d.atlas import pack_request
from flat2d.ideas import clear_ideas, draw_idea, keep_idea, read_ideas, use_idea
from flat2d.optional import FlatUnavailable, draft_mechanics, draw, render_sound, speak
from flat2d.sheet import backends
from flat2d.steps import apply_steps
from flat2d.store import FlatError, load_sheet, save_sheet, save_source, source_path, split_source

PART_FILE = re.compile(r"^part-\d\d\.png$")
SHEET_FILE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}\.png$")
SOUND_FILE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}\.wav$")


def _json(payload: dict, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, indent=2).encode("utf-8") + b"\n"
    return status, body, "application/json; charset=utf-8"


def _file(path: Path, ctype: str) -> tuple[int, bytes, str] | None:
    if not path.is_file():
        return None
    return 200, path.read_bytes(), ctype


def handle(root: Path, method: str, path: str, body: bytes) -> tuple[int, bytes, str] | None:
    path = unquote(path)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    if not parts or parts[0] != "2d":
        return None

    if method == "GET" and path == "/2d/status":
        from flat2d.drawjob import estimate, read_times
        from flat2d.optional import model_loaded

        info = backends()
        still = info["still"]
        row = read_times(root).get(still) if still else None
        return _json(
            {
                "ok": True,
                **info,
                "loaded": model_loaded(),
                "times": {k: v for k, v in row.items() if isinstance(v, (int, float))} if isinstance(row, dict) else {},
                "estimate_s": {
                    "draw": estimate(root, still, ["draw", "ground"]),
                    "draw_first": estimate(root, still, ["load", "draw", "ground"]),
                    "idea": estimate(root, still, ["idea1"]),
                },
            }
        )

    if len(parts) == 2 and method == "GET":
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        sheet = load_sheet(root, stem)
        sheet.update(backends())
        return _json(sheet)

    if len(parts) == 3 and method == "GET" and parts[2] == "source":
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        found = _file(source_path(root, stem), "image/png")
        if found is None:
            return _json({"ok": False, "error": "no source image"}, 404)
        return found

    if len(parts) == 3 and method == "GET" and parts[2] == "ideas":
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        return _json({**read_ideas(root, stem), "still": backends()["still"]})

    if len(parts) == 4 and method == "GET" and parts[2] == "parts" and PART_FILE.match(parts[3]):
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        found = _file(stem_dir(root, stem) / "flat" / "parts" / parts[3], "image/png")
        if found is None:
            return _json({"ok": False, "error": "no part"}, 404)
        return found

    if len(parts) == 4 and method == "GET" and parts[2] == "sounds" and SOUND_FILE.match(parts[3]):
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        found = _file(stem_dir(root, stem) / "flat" / "sounds" / parts[3], "audio/wav")
        if found is None:
            return _json({"ok": False, "error": "no sound"}, 404)
        return found

    if len(parts) == 4 and method == "GET" and parts[2] == "sheets" and parts[3] == "manifest.json":
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        found = _file(stem_dir(root, stem) / "flat" / "sheets" / "manifest.json", "application/json; charset=utf-8")
        if found is None:
            return _json({"ok": False, "error": "no sheet"}, 404)
        return found

    if len(parts) == 4 and method == "GET" and parts[2] == "sheets" and SHEET_FILE.match(parts[3]):
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        found = _file(stem_dir(root, stem) / "flat" / "sheets" / parts[3], "image/png")
        if found is None:
            return _json({"ok": False, "error": "no sheet"}, 404)
        return found

    if len(parts) == 4 and method == "GET" and parts[2] == "voices" and SOUND_FILE.match(parts[3]):
        try:
            stem = check_stem(parts[1])
        except ValueError as e:
            return _json({"ok": False, "error": str(e)}, 400)
        found = _file(stem_dir(root, stem) / "flat" / "voices" / parts[3], "audio/wav")
        if found is None:
            return _json({"ok": False, "error": "no voice"}, 404)
        return found

    if method != "POST" or len(parts) != 2:
        return _json({"ok": False, "error": f"not found: {method} {path}"}, 404)

    action = parts[1]
    try:
        payload: dict[str, Any] = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as e:
        return _json({"ok": False, "error": f"invalid JSON: {e}"}, 400)
    try:
        stem = check_stem(str(payload.get("stem") or ""))
    except ValueError as e:
        return _json({"ok": False, "error": str(e)}, 400)

    if action == "steps":
        when = payload.get("when")
        if when is None or (isinstance(when, str) and not str(when).strip()):
            when = "revise"
        elif when not in ("home", "revise"):
            return _json({"ok": False, "error": "when must be home or revise"}, 400)
        payload["when"] = when
        try:
            return _json(apply_steps(root, stem, payload))
        except Exception as e:
            return _json({"ok": False, "error": str(e)}, 500)

    try:
        if action == "drop":
            raw = payload.get("data") or ""
            if not isinstance(raw, str) or not raw:
                return _json({"ok": False, "error": "empty image"}, 400)
            data = base64.b64decode(raw, validate=False)
            return _json(save_source(root, stem, data), 201)
        if action == "split":
            return _json(split_source(root, stem, payload.get("prompt") or payload.get("names") or ""))
        if action == "save":
            return _json(save_sheet(root, stem, payload.get("parts") or [], str(payload.get("note") or "")))
        if action == "draft":
            return _json(draft_mechanics(root, stem, str(payload.get("note") or ""), payload.get("parts") or []))
        if action == "generate":
            intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else None
            return _json(draw(root, stem, str(payload.get("prompt") or ""), intent=intent, views=payload.get("views")))
        if action == "sound":
            part = payload.get("part") or {}
            if not isinstance(part, dict):
                return _json({"ok": False, "error": "part required"}, 400)
            return _json(render_sound(root, stem, part))
        if action == "speak":
            part = payload.get("part") or {}
            if not isinstance(part, dict):
                return _json({"ok": False, "error": "part required"}, 400)
            return _json(speak(root, stem, part))
        if action == "pack":
            return _json(pack_request(root, stem, payload))
        if action == "idea":
            seed = payload.get("seed")
            return _json(draw_idea(root, stem, str(payload.get("prompt") or ""), int(seed) if seed else None), 201)
        if action == "keep":
            return _json(keep_idea(root, stem, str(payload.get("file") or ""), bool(payload.get("keep", True))))
        if action == "clear":
            return _json(clear_ideas(root, stem))
        if action == "use":
            return _json(use_idea(root, stem, str(payload.get("file") or "")))
    except FlatError as e:
        return _json({"ok": False, "error": str(e)}, 400)
    except FlatUnavailable as e:
        return _json({"ok": False, "error": str(e), "available": False}, 400)
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)

    return _json({"ok": False, "error": f"not found: {method} {path}"}, 404)
