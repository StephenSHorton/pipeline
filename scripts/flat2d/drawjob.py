"""A 2D draw (or a run of Ideas) as a factory job: one slot, /status, /cancel, like 3D and rig.

The still is drawn in the factory process, where the model stays loaded. The hook below gets
the model's steps (flat2d/optional.py DRAW) and writes out/<stem>/draw.log in the grammar
factory_http.parse_progress reads: "STAGES a b c", "STAGE a <label>", "Step 4 of 8 [ 50%]".
How long each stage took is kept in out/draw-times.json, per still model, for the next estimate.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from factory_env import stem_dir

from flat2d import optional
from flat2d.optional import DrawCancelled, FlatUnavailable
from flat2d.store import FlatError, flat_dir

LABELS = {
    "load": "Load the image model",
    "draw": "Draw the picture",
    "ground": "Check the ground",
    "redraw": "Draw it again",
    # The four-view sheet. Not 'check' or 'split': factory_http merges these after the rig labels ("check" is
    # "Check the rig"), so a draw stage must never reuse a rig or generate stage name.
    "views_split": "Split the four views",
    "views_check": "Check the views agree",
    "pieces": "Cut the pieces",
    "motion": "Pack the motion",
    "say": "Speak the line",
    "sound": "Make the sound",
    **{f"idea{i}": f"Picture {i}" for i in range(1, 9)},
}
STEPS = ("pieces", "motion", "say", "sound")
TEXTS = ("prompt", "pieces", "motion", "say", "sound")
MAX_IDEAS = 8
PREVIEW_PX = 768
FLAW_TEXT = {"cloudy": "Ground is cloudy.", "shadow": "There is a shadow on the ground.",
             "facing": "The head points the wrong way.", "profile": "It is not a side view.",
             "height": "The views are not the same height.", "colour": "The views are not the same colours.",
             "side": "The two side views do not match.", "empty": "A view is missing.",
             "pieces": "A view holds more than one subject.", "drift": "The views drift from your photo."}
FLAW_PAST = {"cloudy": "ground was cloudy", "shadow": "ground had a shadow", "facing": "head pointed the wrong way",
             "profile": "not a side view", "height": "views were different heights", "colour": "views were "
             "different colours", "side": "side views did not match", "empty": "a view was missing",
             "pieces": "a view held more than one subject", "drift": "views drifted from your photo"}
VIEW_STAGES = ("views_split", "views_check")


def times_path(root: Path) -> Path:
    return Path(root) / "out" / "draw-times.json"


def read_times(root: Path) -> dict[str, Any]:
    try:
        doc = json.loads(times_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _time_key(stage: str) -> str:
    if stage == "draw":
        return "attempt"
    if stage.startswith("idea"):
        return "idea"
    return stage


def stage_estimates(known: Any, plan: list[str], redraws: int = 1, extra: dict[str, float] | None = None) -> dict[str, float]:
    """Seconds each planned stage took last time (only the ones known). A redraw stage is one more
    draw and ground check for each try it holds; extra adds seconds to a stage (an Idea drawn again)."""
    if not isinstance(known, dict):
        return {}
    num = lambda k: float(known[k]) if isinstance(known.get(k), (int, float)) else None  # noqa: E731
    out: dict[str, float] = {}
    for stage in plan:
        if stage == "redraw":
            a, g = num("attempt"), num("ground")
            # "ground" is a still's ground check, or a sheet's split and check (Hook.phase starts that clock too).
            value = None if a is None else (a + (g or 0.0)) * max(1, redraws)
        else:
            value = num(_time_key(stage))
        if value is not None:
            out[stage] = round(value + (extra or {}).get(stage, 0.0), 2)
    return out


def total_estimate(per_stage: dict[str, float], plan: list[str]) -> float | None:
    if not plan or any(stage not in per_stage for stage in plan):
        return None
    return round(sum(per_stage[stage] for stage in plan), 1)


def estimate(root: Path, backend: str, plan: list[str]) -> float | None:
    """Seconds the plan took last time on this model: None until every stage in it has run once."""
    return total_estimate(stage_estimates(read_times(root).get(backend), plan), plan)


def fresh_log(log: Path) -> None:
    """A new draw.log for a job, the last one kept as draw.prev.log. Done before /status can read it."""
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.is_file():
        log.replace(log.with_name("draw.prev.log"))
    log.write_text("", encoding="utf-8")


def write_times(root: Path, backend: str, seconds: dict[str, float]) -> None:
    if not backend or not seconds:
        return
    doc = read_times(root)
    row = doc.get(backend) if isinstance(doc.get(backend), dict) else {}
    row.update({k: round(v, 2) for k, v in seconds.items()})
    doc[backend] = row
    doc["updated"] = int(time.time())
    path = times_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def plan_for(kind: str, *, loaded: bool, steps: list[str] | None = None, count: int = 1, drop: bool = False,
             views: bool = False) -> list[str]:
    """The stages a draw job shows. views: the four-view sheet (draw, split, check) instead of a still's ground."""
    head = [] if loaded else ["load"]
    if kind == "ideas":
        return head + [f"idea{i}" for i in range(1, count + 1)]
    if drop:
        return list(steps or [])
    return head + ["draw", *(VIEW_STAGES if views else ("ground",))] + list(steps or [])


class Hook:
    """What optional.DRAW calls while a still draws. Writes the log and updates the job, under the factory lock."""

    wants_preview = True

    def __init__(self, factory, job_id: str, stem: str, log: Path, plan: list[str], kind: str, backend: str = ""):
        self.factory = factory
        self.known = read_times(factory.root).get(backend) if backend else None
        self.redraws = 0
        self.extra: dict[str, float] = {}
        self.why = ""  # the reason the picture is being drawn again, shown beside its steps
        self.job_id = job_id
        self.stem = stem
        self.kind = kind
        self.plan = list(plan)
        self.current = ""
        self.target = "draw"  # the stage a step belongs in
        self.times: dict[str, dict[str, float | None]] = {}
        self.previews = 0
        self.attempt_t0: float | None = None
        self.attempts: list[float] = []
        self.ground_t0: float | None = None
        self.grounds: list[float] = []
        log.parent.mkdir(parents=True, exist_ok=True)
        self.fh = log.open("a", encoding="utf-8")  # factory.start_flat made it fresh (fresh_log)
        self.lock = threading.Lock()

    # job and log
    def update(self, **fields: Any) -> None:
        with self.factory._lock:
            job = self.factory._job
            if job and job.get("job_id") == self.job_id:
                job.update(fields)

    def line(self, text: str) -> None:
        with self.lock:
            if not self.fh.closed:
                self.fh.write(text + "\n")
                self.fh.flush()

    def set_plan(self, plan: list[str]) -> None:
        self.plan = list(plan)
        self.line("STAGES " + " ".join(self.plan))
        self.reestimate()

    def reestimate(self) -> None:
        """The plan grew (a load, a redraw) or a stage takes another try: what is left changes with it."""
        per = stage_estimates(self.known, self.plan, self.redraws, self.extra)
        self.update(stage_est=per, estimate_s=total_estimate(per, self.plan))

    def stage(self, key: str) -> None:
        now = time.time()
        if self.current and self.current in self.times and self.times[self.current]["ended"] is None:
            self.times[self.current]["ended"] = now
        self.current = key
        self.times[key] = {"started": now, "ended": None}
        self.line(f"STAGE {key} {LABELS.get(key, key)}")
        self.update(stage_times={k: dict(v) for k, v in self.times.items()})

    def end_stages(self) -> None:
        now = time.time()
        for row in self.times.values():
            if row["ended"] is None:
                row["ended"] = now
        self.update(stage_times={k: dict(v) for k, v in self.times.items()})

    def close(self, text: str) -> None:
        self.end_stages()
        self.line(text)
        with self.lock:
            self.fh.close()

    def seconds(self, key: str) -> float | None:
        row = self.times.get(key)
        if not row or row.get("ended") is None:
            return None
        return max(0.0, float(row["ended"]) - float(row["started"]))

    # optional.DRAW
    def cancelled(self) -> bool:
        with self.factory._lock:
            job = self.factory._job
            return bool(job and job.get("job_id") == self.job_id and job.get("cancel"))

    def check(self) -> None:
        if self.cancelled():
            raise DrawCancelled("Cancelled")

    def loading(self, on: bool) -> None:
        if on:
            if self.current != "load":
                if "load" not in self.plan:
                    at = self.plan.index(self.target) if self.target in self.plan else 0
                    self.set_plan(self.plan[:at] + ["load"] + self.plan[at:])
                self.stage("load")
                usual = (self.known or {}).get("load") if isinstance(self.known, dict) else None
                if isinstance(usual, (int, float)) and usual >= 1:
                    self.line(f"Loading the model. Last time this took about {round(usual)} s.")
                else:
                    self.line("First draw: loading the model. This one takes longer.")
        else:
            self.attempt_t0 = time.time()

    def step(self, k: int, n: int) -> None:
        if self.current != self.target:
            self.stage(self.target)
        if self.attempt_t0 is None:
            self.attempt_t0 = time.time()
        n = max(int(n), 1)
        head = f"Again, {self.why} · " if self.why else ""
        self.line(f"{head}Step {k} of {n} [{round(100 * k / n):3d}%]")

    def ground(self) -> None:
        now = time.time()
        if self.attempt_t0 is not None:
            self.attempts.append(now - self.attempt_t0)
            self.attempt_t0 = None
        self.ground_t0 = now
        if self.kind == "flat2d" and self.current == "draw":
            self.stage("ground")
        self.line("Checking the ground.")

    def phase(self, key: str) -> None:
        """A four-view sheet moved on: 'views_split' (the drawing is done; split and cut out) or 'views_check'.
        During a redraw both stay inside the redraw stage, as the ground check does."""
        now = time.time()
        if key == "views_split":
            if self.attempt_t0 is not None:
                self.attempts.append(now - self.attempt_t0)
                self.attempt_t0 = None
            self.ground_t0 = now
            text = "Splitting the four views."
        else:
            text = "Checking the views agree."
        if self.kind == "flat2d" and self.target == "draw" and key in self.plan:
            self.stage(key)
        self.line(text)

    def ground_done(self) -> None:
        if self.ground_t0 is not None:
            self.grounds.append(time.time() - self.ground_t0)
            self.ground_t0 = None

    def attempt(self, i: int, total: int, flaws: list[str]) -> None:
        self.ground_done()
        kinds = list(dict.fromkeys(f.split(":", 1)[0] for f in flaws))  # "empty:front" reads as "empty"
        why = " ".join(FLAW_TEXT.get(f, f) for f in kinds) or "The ground needs another try."
        self.why = " and ".join(FLAW_PAST.get(f, f) for f in kinds) or "new seed"
        if self.kind == "flat2d":
            self.target = "redraw"
            self.redraws += 1
            if "redraw" not in self.plan:
                at = self.plan.index("ground") + 1 if "ground" in self.plan else len(self.plan)
                self.set_plan(self.plan[:at] + ["redraw"] + self.plan[at:])
                self.stage("redraw")
            else:
                self.reestimate()
        else:
            # An Idea drawn again stays in its stage: its step count starts over, and so does its share of time.
            per_idea = (self.known or {}).get("idea") if isinstance(self.known, dict) else None
            if isinstance(per_idea, (int, float)) and self.current:
                self.extra[self.current] = self.extra.get(self.current, 0.0) + float(per_idea)
                self.reestimate()
        # "[  0%]" starts the stage's step bar over (parse_progress keeps the last one it saw).
        self.line(f"{why} Drawing again on a new seed ({i} of {total}). [  0%]")
        self.attempt_t0 = time.time()

    def preview(self, image) -> None:
        dest = flat_dir(self.factory.root, self.stem) / "drawing.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name("drawing.jpg.tmp")
        im = image.convert("RGB")
        im.thumbnail((PREVIEW_PX, PREVIEW_PX))
        im.save(tmp, format="JPEG", quality=82)
        os.replace(tmp, dest)
        self.previews += 1
        self.update(preview_url=f"/2d/{self.stem}/drawing?j={self.job_id}&v={self.previews}")


def _finish(hook: Hook | None, factory, job_id: str, state: str, error: str | None, result: dict | None = None) -> None:
    if hook is not None:
        hook.close({"ok": "Done.", "cancelled": "Cancelled."}.get(state, f"Error: {error}"))
    fields: dict[str, Any] = {"state": state, "error": error, "finished": time.time()}
    if result is not None:
        fields["result"] = result
    with factory._lock:
        job = factory._job
        if job and job.get("job_id") == job_id:
            job.update(fields)


def run(factory, spec: dict[str, Any], payload: dict[str, Any], write_intent=None) -> None:
    """The job thread. The payload (texts, clip, intent) stays here, out of the job dict /status returns.
    write_intent(root, stem, intent) is factory_http's, passed in so this module never imports it."""
    job_id, kind = spec["job_id"], spec["kind"]
    hooks: list[Hook] = []
    try:
        if payload.get("intent") and write_intent is not None:
            write_intent(factory.root, spec["stem"], payload["intent"])
        draw = _run_ideas if kind == "ideas" else _run_flat
        result = draw(factory, spec, payload, hooks)
        if hooks:
            hooks[0].end_stages()
        _remember(factory.root, spec.get("backend") or "", hooks[0] if hooks else None)
        _finish(hooks[0] if hooks else None, factory, job_id, "ok", None, result)
    except DrawCancelled:
        _finish(hooks[0] if hooks else None, factory, job_id, "cancelled", "Cancelled")
    except (FlatError, FlatUnavailable) as e:
        _finish(hooks[0] if hooks else None, factory, job_id, "error", str(e))
    except Exception as e:
        traceback.print_exc()
        _finish(hooks[0] if hooks else None, factory, job_id, "error", str(e) or e.__class__.__name__)


def _remember(root: Path, backend: str, hook: Hook | None) -> None:
    if hook is None:
        return
    out: dict[str, float] = {}
    load = hook.seconds("load")
    if load is not None:
        out["load"] = load
    if hook.attempts:
        out["attempt"] = hook.attempts[-1]
    if hook.grounds:
        out["ground"] = hook.grounds[-1]
    for step in (*STEPS, *VIEW_STAGES):
        s = hook.seconds(step)
        if s is not None:
            out[step] = s
    ideas = [s for s in (hook.seconds(f"idea{i}") for i in range(1, MAX_IDEAS + 1)) if s is not None]
    if ideas:
        out["idea"] = ideas[-1] if len(ideas) == 1 else sum(ideas[1:]) / len(ideas[1:])
    try:
        write_times(root, backend, out)
    except OSError as e:
        print(f"drawjob: could not write draw-times.json: {e}")


def canonical_intent(root: Path, stem: str, intent: dict[str, Any] | None, saved: dict[str, Any]) -> str:
    """After a posed creature draw: intent.json gets psx.facing (the drawn facing) and rig {plan_hint, canonical}.
    Returns a note for the log when the picture was mirrored or is not canonical ("" otherwise)."""
    from flat2d.pose import PLAN_HINT

    check = saved.get("facing_check") or {}
    canonical = bool(saved.get("canonical"))
    path = stem_dir(root, stem) / "intent.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = dict(intent or {})
    if isinstance(doc, dict) and doc.get("output"):
        psx = dict(doc.get("psx") or {}) if isinstance(doc.get("psx"), dict) else {}
        if saved.get("facing"):
            psx["facing"] = saved["facing"]
        doc["psx"] = psx
        rig = dict(doc.get("rig") or {}) if isinstance(doc.get("rig"), dict) else {}
        rig["canonical"] = canonical
        hint = PLAN_HINT.get(str(saved.get("kind") or ""))
        if hint:
            rig["plan_hint"] = hint
        doc["rig"] = rig
        doc["facing_source"] = "drawn" if canonical else "detected"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("intent.json.tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    if check.get("how") == "mirrored":
        return "The head pointed left; the picture was mirrored so it points right."
    if not canonical:
        return "Warning: the drawing is not a strict side view; it will be treated like a photo."
    return ""


def _run_flat(factory, spec: dict[str, Any], payload: dict[str, Any], hooks: list[Hook]) -> dict[str, Any]:
    from flow import autosave
    from flat2d.steps import apply_steps

    root, stem, job_id = factory.root, spec["stem"], spec["job_id"]
    drop = spec.get("source") == "drop"
    revise = bool(spec.get("revise"))
    steps = list(spec.get("steps") or [])
    if stem_dir(root, stem).is_dir() and not drop:
        autosave(root, stem, ("concept",))
    had_parts = (flat_dir(root, stem) / "parts.json").is_file() and "pieces" not in steps
    views = not drop and optional.wants_views(payload.get("intent"), payload.get("views"))
    reference = payload.get("reference") or None
    plan = plan_for("flat2d", loaded=optional.model_loaded(photo=bool(reference)), steps=steps, drop=drop,
                    views=views)
    hook = Hook(factory, job_id, stem, stem_dir(root, stem) / "draw.log", plan, "flat2d", spec.get("backend") or "")
    hooks.append(hook)
    hook.set_plan(plan)
    result: dict[str, Any] = {"source": "flat/source.png", "seed": None, "redrawn": [], "notes": [], "ran": [],
                              "pieces": False, "hear": "", "center": "", "revise": revise}
    if not drop:
        if "load" in plan:
            hook.loading(True)
            hook.attempt_t0 = None
        else:
            hook.stage("draw")
            hook.attempt_t0 = time.time()
        with optional.drawing(hook):
            saved = optional.draw(root, stem, str(payload.get("prompt") or ""), intent=payload.get("intent"),
                                  views=payload.get("views"), reference=reference)
        hook.ground_done()
        result["seed"] = saved.get("seed")
        result["redrawn"] = list(saved.get("redrawn") or [])
        result["degraded"] = list(saved.get("degraded") or [])
        from factory_env import degraded_line

        for entry in result["degraded"]:
            hook.line(degraded_line(entry))
        if saved.get("views"):
            for k in ("views", "sheet", "consistent", "flaws", "canonical_view"):
                result[k] = saved.get(k)
            for note in saved.get("notes") or []:
                result["notes"].append(note)
                hook.line(note)
        if saved.get("pose"):
            for k in ("pose", "facing", "canonical", "facing_check"):
                result[k] = saved.get(k)
            note = canonical_intent(root, stem, payload.get("intent"), saved)
            if note:
                result["notes"].append(note)
                hook.line(note)
        hook.line("Picture saved.")
        autosave(root, stem, ("concept",))
    texts = {k: payload[k] for k in TEXTS if isinstance(payload.get(k), str)}
    for clip in ("video", "frames"):
        if payload.get(clip):
            texts[clip] = payload[clip]
    ran_any = False
    for step in steps:
        hook.check()
        hook.stage(step)
        out = apply_steps(root, stem, {**texts, "steps": [step], "when": "revise" if revise else "home",
                                       "part": "asset", "speaker": "af_heart"})
        for note in out.get("notes") or []:
            result["notes"].append(note)
            hook.line(note)
        result["ran"].extend(out.get("ran") or [])
        ran_any = ran_any or bool(out.get("ran"))
        if out.get("pieces"):
            result["pieces"] = True
        if out.get("hear"):
            result["hear"] = out["hear"]
        if isinstance(out.get("center"), str) and out["center"]:
            result["center"] = out["center"]
    if ran_any:
        autosave(root, stem, ("flat",))
    if had_parts:
        result["notes"].append("Pieces still show the previous still.")
    return result


def _run_ideas(factory, spec: dict[str, Any], payload: dict[str, Any], hooks: list[Hook]) -> dict[str, Any]:
    from flat2d.ideas import draw_idea

    root, stem, job_id = factory.root, spec["stem"], spec["job_id"]
    count = int(spec.get("count") or 1)
    plan = plan_for("ideas", loaded=optional.model_loaded(), count=count)
    hook = Hook(factory, job_id, stem, stem_dir(root, stem) / "draw.log", plan, "ideas", spec.get("backend") or "")
    hooks.append(hook)
    hook.set_plan(plan)
    items: list[dict[str, Any]] = []
    result: dict[str, Any] = {"items": [], "ideas": None}
    hook.target = "idea1"
    if "load" in plan:
        hook.loading(True)
    for i in range(1, count + 1):
        hook.check()
        hook.target = f"idea{i}"
        hook.why = ""
        if hook.current != "load":
            hook.stage(hook.target)
        hook.attempt_t0 = None if hook.current == "load" else time.time()
        with optional.drawing(hook):
            drawn = draw_idea(root, stem, str(payload.get("prompt") or ""), intent=payload.get("intent"))
        hook.ground_done()
        items.append(drawn["item"])
        result = {"items": list(items), "ideas": {"prompt": drawn.get("prompt"), "items": drawn.get("items")}}
        hook.line(f"Picture {i} of {count} saved.")
        hook.update(result=result)
    return result
