#!/usr/bin/env python3
"""A urllib client for the factory on :8766, the only process allowed to load a model.

The workbench never draws, shapes or paints itself: those go to the factory as jobs, so its single job slot and the
"one big model at a time" rule stay where they are. The workbench also asks the factory before it writes into an
asset (GET /status: is a job writing that stem?) and after (POST /flow/<stem>/snapshot: keep the change as a take).

Every call returns the factory's JSON as a dict, an HTTP error's body included ({"ok": false, "error": ...} with
"http_status" added), so a refused request is a result to report, not an exception. Only an unreachable factory
raises FactoryDown.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

DEFAULT_URL = "http://127.0.0.1:8766"


class FactoryDown(ConnectionError):
    pass


def default_url() -> str:
    url = os.environ.get("WORKBENCH_FACTORY_URL")
    if url:
        return url.rstrip("/")
    port = os.environ.get("FACTORY_HTTP_PORT")
    return f"http://127.0.0.1:{port}" if port else DEFAULT_URL


class FactoryClient:
    def __init__(self, url: str | None = None, timeout: float = 30.0):
        self.url = (url or default_url()).rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None,
              timeout: float | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read()
            status = e.code
        except (urllib.error.URLError, OSError) as e:
            raise FactoryDown(f"factory not reachable at {self.url}: {getattr(e, 'reason', e)}") from None
        try:
            out = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            out = {"ok": False, "error": raw.decode("utf-8", "replace")[:400]}
        if not isinstance(out, dict):
            out = {"ok": True, "value": out}
        if status >= 400:
            out.setdefault("ok", False)
            out["http_status"] = status
        return out

    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health")

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        return self._call("GET", "/status" + (f"/{quote(job_id, safe='')}" if job_id else ""))

    def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/generate", body)

    def draw(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/2d/draw", body)

    def rig(self, stem: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", f"/rig/{quote(stem, safe='')}", body)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._call("POST", f"/cancel/{quote(job_id, safe='')}")

    def snapshot(self, stem: str, stage: str = "mesh", note: str = "") -> dict[str, Any]:
        return self._call("POST", f"/flow/{quote(stem, safe='')}/snapshot", {"stage": stage, "note": note})

    def pick(self, stem: str, stage: str, take: str) -> dict[str, Any]:
        return self._call("POST", f"/flow/{quote(stem, safe='')}/pick", {"stage": stage, "take": take})

    def activity(self, entry: dict[str, Any]) -> dict[str, Any]:
        """One line of the app's activity feed (POST /activity). A short timeout: the feed never holds up a tool."""
        return self._call("POST", "/activity", entry, timeout=2.0)

    def writing(self, stem: str) -> bool | None:
        """A running job owns this stem (True), none does (False), or the factory is down (None: no factory, no
        job, so nothing can be racing the write)."""
        try:
            st = self.status()
        except FactoryDown:
            return None
        return st.get("state") == "running" and st.get("stem") == stem
