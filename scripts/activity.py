"""What Claude and the factory are doing, for the app to show while it happens (GET /activity on the factory).

An entry is one action: a workbench tool call (source "claude", posted by scripts/workbench/ through POST /activity)
or a factory job (source "job", written by JobWatch as the job runs). Each change to an entry is a new copy of the
whole entry with the next seq, so the app asks for `since=<last seq>` and replaces entries by id.

    {"id", "seq", "ts", "source", "tool", "stem", "title", "state": running|ok|error|cancelled, "started",
     "ms", "detail", "glb", "image", "take", "job_id", "error"}

`glb` is a model the action wrote (out/-relative) for the viewer to load as it lands; `image` a picture to show
beside the line (a look sheet, a drawn view). Kept in memory (the last KEEP) and appended to
out/_logs/activity.jsonl, whose tail is read back at start so a restart keeps the recent history.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

KEEP = 400
STATES = frozenset({"running", "ok", "error", "cancelled"})
FIELDS = ("source", "tool", "stem", "title", "state", "started", "ms", "detail", "glb", "image", "take", "job_id",
          "error", "kind")
TEXT_MAX = 400


def _clean(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in FIELDS:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value[:TEXT_MAX]
        elif not isinstance(value, (int, float, bool)):
            continue
        out[key] = value
    if "state" in out and out["state"] not in STATES:
        raise ValueError(f"state must be one of {sorted(STATES)}")
    return out


class Activity:
    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()
        self._seq = 0
        self._entries: dict[str, dict[str, Any]] = {}  # id -> latest copy, oldest first (dicts keep order)
        if path is not None:
            self._read_back(path)

    def _read_back(self, path: Path) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-KEEP * 4:]
        except OSError:
            return
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict) and isinstance(e.get("id"), str) and isinstance(e.get("seq"), int):
                # A restart ended whatever was running: nothing will finish it now.
                self._entries.pop(e["id"], None)
                self._entries[e["id"]] = e
                self._seq = max(self._seq, e["seq"])
        for e in self._entries.values():
            if e.get("state") == "running":
                e["state"], e["detail"] = "cancelled", "the factory restarted"
        self._trim()

    def _trim(self) -> None:
        while len(self._entries) > KEEP:
            self._entries.pop(next(iter(self._entries)))

    def put(self, entry_id: str, **fields: Any) -> dict[str, Any]:
        """Add or change an entry (fields merge into the last copy). Returns the new copy."""
        if not isinstance(entry_id, str) or not entry_id or len(entry_id) > 80:
            raise ValueError("id must be a short string")
        clean = _clean(fields)
        with self._lock:
            old = self._entries.pop(entry_id, None) or {"id": entry_id, "started": self.clock()}
            self._seq += 1
            entry = {**old, **clean, "id": entry_id, "seq": self._seq, "ts": round(self.clock(), 3)}
            if entry.get("state") in ("ok", "error", "cancelled") and "ms" not in clean and entry.get("started"):
                entry["ms"] = int((entry["ts"] - float(entry["started"])) * 1000)
            self._entries[entry_id] = entry
            self._trim()
            line = json.dumps(entry)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except OSError:
                    pass
            return dict(entry)

    def get(self, entry_id: str) -> dict[str, Any] | None:
        with self._lock:
            e = self._entries.get(entry_id)
            return dict(e) if e else None

    def since(self, seq: int = 0, limit: int = 200) -> dict[str, Any]:
        with self._lock:
            fresh = [dict(e) for e in self._entries.values() if e["seq"] > seq]
            last = self._seq
        fresh.sort(key=lambda e: e["seq"])
        return {"ok": True, "seq": last, "now": round(self.clock(), 3), "entries": fresh[-limit:]}


# Model files a generate writes as it goes, in the order they show the asset taking shape.
JOB_MODELS = ("mesh_hero.glb", "mesh.glb", "textured.glb", "textured_photo.glb", "psx.glb", "rigged.glb")
JOB_TITLES = {"generate": "Building", "flat2d": "Drawing", "rig": "Rigging", "render": "Rendering the rig sheet of"}


class JobWatch:
    """Turns the factory's one job into an activity entry: started, each stage, each model it writes, the end.
    `tick` is called about once a second (the factory's watch thread; tests call it by hand)."""

    def __init__(self, activity: Activity, status: Callable[[], dict[str, Any]], root: Path) -> None:
        self.activity = activity
        self.status = status
        self.root = Path(root)
        self._job: str | None = None
        self._stage: str | None = None
        self._models: dict[str, float] = {}
        self._pics: dict[str, float] = {}

    def _id(self, job_id: str) -> str:
        return f"job-{job_id}"

    def tick(self) -> None:
        st = self.status()
        job_id = st.get("job_id")
        if not job_id:
            return
        stem = st.get("stem") or ""
        kind = st.get("kind") or "generate"
        if job_id != self._job and st.get("state") != "running" and self.activity.get(self._id(job_id)) is None:
            # A job that ended before this watch saw it run (the factory just started): not news.
            self._job = job_id
            return
        if job_id != self._job:
            self._job, self._stage = job_id, None
            self._models = self._model_times(stem)
            self._pics = self._pic_times(stem)
            verb = JOB_TITLES.get(kind, "Working on")
            self.activity.put(self._id(job_id), source="job", kind=kind, stem=stem, job_id=job_id, state="running",
                              title=f"{verb} {stem}", started=st.get("started") or time.time())
        eid = self._id(job_id)
        progress = st.get("progress") or {}
        stage = progress.get("stage")
        label = next((s.get("label") for s in progress.get("stages") or [] if s.get("key") == stage), stage)
        if st.get("state") == "running":
            if stage and stage != self._stage:
                self._stage = stage
                n, count = progress.get("stage_index", -1) + 1, progress.get("stage_count") or 0
                self.activity.put(eid, detail=f"{label} ({n} of {count})" if count else str(label))
            self._new_models(eid, stem)
            self._new_pics(eid, stem)
            return
        old = self.activity.get(eid)
        if old is None or old.get("state") != "running":
            return
        self._new_models(eid, stem)
        self._new_pics(eid, stem)
        state = st.get("state") if st.get("state") in ("ok", "error", "cancelled") else "error"
        fields: dict[str, Any] = {"state": state}
        if state == "error":
            fields["error"] = str(st.get("error") or "failed")
        elif state == "ok":
            fields["detail"] = "Done"
        finished, started = st.get("finished"), st.get("started")
        if isinstance(finished, (int, float)) and isinstance(started, (int, float)):
            fields["ms"] = int((finished - started) * 1000)
        self.activity.put(eid, **fields)

    def _model_times(self, stem: str) -> dict[str, float]:
        out: dict[str, float] = {}
        if not stem:
            return out
        folder = self.root / "out" / stem
        for name in JOB_MODELS:
            try:
                out[name] = (folder / name).stat().st_mtime
            except OSError:
                pass
        return out

    def _new_models(self, eid: str, stem: str) -> None:
        now = self._model_times(stem)
        fresh = [n for n in JOB_MODELS if n in now and now[n] != self._models.get(n)]
        self._models = now
        if fresh:
            # The furthest along: a textured model beats the bare shape written in the same second.
            self.activity.put(eid, glb=f"{stem}/{fresh[-1]}")

    # A draw shows its picture while it forms (flat/drawing.jpg, rewritten each few steps), then the four views.
    PICTURES = ("flat/drawing.jpg", "flat/sheet.png")

    def _pic_times(self, stem: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for rel in self.PICTURES if stem else ():
            try:
                out[rel] = (self.root / "out" / stem / rel).stat().st_mtime
            except OSError:
                pass
        return out

    def _new_pics(self, eid: str, stem: str) -> None:
        now = self._pic_times(stem)
        fresh = [r for r in self.PICTURES if r in now and now[r] != self._pics.get(r)]
        self._pics = now
        if fresh:
            self.activity.put(eid, image=f"{stem}/{fresh[-1]}")
