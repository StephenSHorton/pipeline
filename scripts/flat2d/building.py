"""POST /2d/building: a building as a sprite stack from three drawn walls, as a factory job (the draw job's slot,
/status, /cancel and draw-times.json). The pure part is scripts/stack_building.py; this module draws the walls.

One model at a time: the text-to-image model draws the plain wall (optional.still_image), then the edit model adds
the front feature to a copy, and the back feature to another (optional.edit_image). The two models swap themselves
out (qwen_edit_still drops the text-to-image models before it loads, and the other way round), as the photo draw
does. Like every 2D draw, the last model stays loaded when the job ends; generate unloads it before shape.
"""
from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from factory_env import check_stem, stem_dir

from flat2d import drawjob, optional
from flat2d.optional import DrawCancelled, FlatUnavailable
from flat2d.pose import CLEAN_STYLE
from flat2d.store import FlatError, flat_dir

# The prompts proven by hand (2026-11-21 walls_add2.py), with the building's words taken from the request.
TAIL = ("Strictly orthographic, no perspective. One object, centred, with a wide empty magenta margin on every side. "
        "No shadows, no ground, no floor, no text, no labels, no borders. Plain flat uniform magenta #FF00FF "
        "background.")
WALL_LEAD = "Flat elevation of one plain wall of a building, seen exactly straight on, level: "
# The ground floor sentence: from the app's first real runs (2026-11-24) the model drew a full-width shop front on a
# three-storey town building's plain wall three times out of three despite "no shop front".
PLAIN = (" A plain wall: no shop front, no door, no awning, no signs. The ground floor is the same plain wall as the "
         "floors above, with the same windows in the same places. Nothing stands on the roof. The side walls are not "
         "visible and the top surface of the roof is not visible. ")
KEEP = (" Keep everything else exactly as it is: the upper floors and their windows, the cornice, the corner trim, "
        "the wall's material, the width and height, the outline and drawing style, and the plain flat magenta "
        "background.")
SEEDS = {"base": 12, "front": 23, "back": 33}  # the hand run's seeds; a request's seed moves all three together
WALL_TRIES = 3  # plain-wall draws at most, while it has a door (each about 45 s with the model loaded)
WALL_SEED_STEP = 1000  # the next try's seed: far from the front and back edits' seeds
WORDS_MAX = 1200
# Stages (drawjob.LABELS holds their labels): a load before each model, the three pictures, then the code.
LOAD_EDIT = "load_edit"
WALL, FRONT, BACK, FIT = "wall", "front_add", "back_add", "walls_fit"
TIME_KEY = "walls"  # draw-times.json row: walls:<still>+<edit>, never the plain draw's row


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text if not text or text[-1] in ".!?" else text + "."


def prompts(wall: str, front: str, back: str = "") -> dict[str, str]:
    """The three prompts: the plain wall (text-to-image), the front edit and the back edit (edit model).
    front and back are the edit instructions as written, e.g. "Add a shop front to the ground floor of this
    building: ... and add a dark green fabric awning across the top of the ground floor"."""
    out = {"base": WALL_LEAD + _sentence(wall) + PLAIN + CLEAN_STYLE + " " + TAIL,
           "front": _sentence(front) + KEEP}
    if back:
        out["back"] = _sentence(back) + KEEP
    return out


def wants_awning(front: str) -> str:
    """'yes' when the front words ask for an awning (the build then fails without one), else 'auto'."""
    return "yes" if "awning" in str(front or "").lower() else "auto"


def time_key(still: str, edit: str) -> str:
    return f"{TIME_KEY}:{still}+{edit}"


def parse(body: dict[str, Any]) -> dict[str, Any]:
    """The request, checked: {stem, wall, front, back?, look?, size?, seed?, wide?, deep?, roof?, ridge?,
    roof_colour?}. wide, deep: window bays across the front and along the side (default as drawn). roof: "flat"
    (default) or "gable"; ridge: "across" (default) or "along"; roof_colour: a name or #rrggbb. ValueError on a
    bad value."""
    import stack_building as SB
    from flat2d.pose import pixel_size

    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    stem = check_stem(str(body.get("stem") or ""))
    words = {}
    for key in ("wall", "front", "back"):
        raw = body.get(key)
        if raw is not None and not isinstance(raw, str):
            raise ValueError(f"{key} must be words")
        words[key] = " ".join((raw or "").split())
        if len(words[key]) > WORDS_MAX:
            raise ValueError(f"{key} is longer than {WORDS_MAX} characters")
    if not words["wall"]:
        raise ValueError("wall: describe the plain wall to draw")
    if not words["front"]:
        raise ValueError("front: say what to add to the front")
    look = str(body.get("look") or SB.LOOK).strip().lower()
    if look not in SB.LOOKS:
        raise ValueError(f"look must be {' or '.join(SB.LOOKS)}")
    size = SB.SIZE if body.get("size") in (None, "") else pixel_size(body.get("size"))
    seed = body.get("seed")
    if seed in (None, ""):
        seed = None
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError("seed must be a whole number") from None
        if not 0 <= seed < 2**31 - 64:
            raise ValueError("seed must be between 0 and 2147483583")
    bays = {}
    for key in ("wide", "deep"):
        raw = body.get(key)
        if raw in (None, ""):
            bays[key] = None
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, str)) or not str(raw).strip().isdigit():
            raise ValueError(f"{key} must be a whole number of window bays")
        bays[key] = int(raw)
        if not 1 <= bays[key] <= SB.BAY_MAX:
            raise ValueError(f"{key} must be 1 to {SB.BAY_MAX} window bays")
    roof = str(body.get("roof") or "flat").strip().lower()
    if roof not in SB.ROOFS:
        raise ValueError(f"roof must be {' or '.join(SB.ROOFS)}")
    ridge = str(body.get("ridge") or "across").strip().lower()
    if ridge not in SB.RIDGES:
        raise ValueError(f"ridge must be {' or '.join(SB.RIDGES)}")
    roof_colour = body.get("roof_colour")
    if roof_colour in (None, ""):
        roof_colour = None
    else:
        try:
            SB.roof_rgb(roof_colour if isinstance(roof_colour, str) else "?")
        except SB.WallsError as e:
            raise ValueError(str(e)) from None
        roof_colour = roof_colour.strip().lower()
    seeds = dict(SEEDS) if seed is None else {"base": seed, "front": seed + 11, "back": seed + 21}
    return {"stem": stem, **words, "look": look, "size": size, "seeds": seeds, "awning": wants_awning(words["front"]),
            **bays, "roof": roof, "ridge": ridge, "roof_colour": roof_colour}


def plan_for(back: bool, loaded: bool, edit_load: bool) -> list[str]:
    return ((["load"] if not loaded else []) + [WALL] + ([LOAD_EDIT] if edit_load else []) + [FRONT]
            + ([BACK] if back else []) + [FIT, drawjob.CARVE])


class BuildHook(drawjob.Hook):
    """The draw hook with one load stage per model: 'load' before the plain wall, 'load_edit' before the edits."""

    load_key = "load"

    def loading(self, on: bool) -> None:
        key = self.load_key
        if on:
            if self.current != key:
                if key not in self.plan:
                    at = self.plan.index(self.target) if self.target in self.plan else 0
                    self.set_plan(self.plan[:at] + [key] + self.plan[at:])
                self.stage(key)
                usual = (self.known or {}).get(key) if isinstance(self.known, dict) else None
                if isinstance(usual, (int, float)) and usual >= 1:
                    self.line(f"Loading the model. Last time this took about {round(usual)} s.")
                else:
                    self.line("Loading the model. The first load takes longer.")
        else:
            self.attempt_t0 = time.time()

    def drop_unused(self) -> None:
        """A load stage the model never needed (it was warm) leaves the plan, so the bar does not wait on it."""
        unused = [k for k in ("load", LOAD_EDIT) if k in self.plan and k not in self.times]
        if unused:
            self.set_plan([k for k in self.plan if k not in unused])


def _draw(hook: BuildHook, stage: str, load: str, fn, dest: Path):
    hook.check()
    hook.target, hook.load_key = stage, load
    if hook.current != stage:
        hook.stage(stage)
    with optional.drawing(hook):
        image = fn()
    image = getattr(image, "image", image).convert("RGB")
    hook.preview(image)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="PNG")
    hook.line(f"{dest.stem.capitalize()} wall drawn.")
    return dest


def _plain_wall(hook: BuildHook, SB, ask: str, seeds: dict[str, int], work: Path) -> Path:
    """The plain wall, drawn again with another seed while it has a door (it is every side, so a door on it shows on
    all four; the model sometimes draws one despite the prompt). The text-to-image model stays loaded between tries.
    After WALL_TRIES the try with the fewest door rows is kept, and the job says so."""
    tried: list[tuple[int, int, Path]] = []
    for n in range(WALL_TRIES):
        seed = seeds["base"] + n * WALL_SEED_STEP
        dest = work / ("base.png" if n == 0 else f"base-{n + 1}.png")
        _draw(hook, WALL, "load", lambda: optional.still_image(ask, seed=seed), dest)
        check = SB.plain_wall_door(dest)
        # The least door: no shop front first, then the fewest door rows.
        tried.append(((int(check["shop_front"]), check["hits"]), seed, dest))
        # Said on every try, so the log shows the check ran (and what it saw) even when the first wall passes.
        hook.line(f"Door check, plain wall {n + 1}: {check['hits']} of {len(check['runs'])} bottom rows have a "
                  f"ground-level opening ({SB.DOOR_HITS} or more is a door); the ground floor matches the floors "
                  f"above {check['ground']:.2f} (under {SB.GROUND_MATCH} is a shop front): "
                  f"{'a shop front' if check['shop_front'] else 'a door' if check['door'] else 'no door'}.")
        if not check["door"]:
            break
        if n + 1 < WALL_TRIES:
            hook.line(f"The plain wall has a door on it (or a shop front), which would show on every side. Drawing it again "
                      f"({n + 2} of {WALL_TRIES}).")
    (shop, hits), seed, best = min(tried, key=lambda t: t[0])
    if shop or hits >= SB.DOOR_HITS:
        hook.line("Every plain wall had a door on it; keeping the one with the least. It will show on every side.")
    seeds["base"] = seed
    if best != work / "base.png":
        shutil.copy2(best, work / "base.png")
    return work / "base.png"


def run(factory, spec: dict[str, Any], payload: dict[str, Any]) -> None:
    """The job thread (factory_http.Factory.start_building starts it)."""
    import stack_building as SB
    from flat2d.sheet import edit_info, still_info

    root, stem, job_id = factory.root, spec["stem"], spec["job_id"]
    hook = BuildHook(factory, job_id, stem, stem_dir(root, stem) / "draw.log", spec["plan"], "building",
                     spec.get("backend") or "")
    hook.set_plan(spec["plan"])
    work = flat_dir(root, stem) / ".walls-draw"
    state, error, result = "ok", None, None
    try:
        asks = prompts(payload["wall"], payload["front"], payload.get("back") or "")
        seeds = payload["seeds"]
        shutil.rmtree(work, ignore_errors=True)
        walls: dict[str, Path] = {}
        walls["base"] = _plain_wall(hook, SB, asks["base"], seeds, work)
        hook.line("Adding the front.")
        walls["front"] = _draw(hook, FRONT, LOAD_EDIT,
                               lambda: optional.edit_image(asks["front"], walls["base"], seed=seeds["front"]),
                               work / "front.png")
        if "back" in asks:
            hook.line("Adding the back.")
            walls["back"] = _draw(hook, BACK, LOAD_EDIT,
                                  lambda: optional.edit_image(asks["back"], walls["base"], seed=seeds["back"]),
                                  work / "back.png")
        hook.check()
        hook.drop_unused()
        hook.stage(FIT)
        still, edit = still_info(), edit_info()
        meta = {"sheet": {"layout": "walls", "seeds": seeds, "prompts": asks, "look": payload["look"],
                          "size": payload["size"], "still": {k: still[k] for k in ("backend", "model", "licence")},
                          "edit": {k: edit[k] for k in ("backend", "model", "licence")}}}
        try:
            out = SB.build(root, stem, walls, look=payload["look"], size=payload["size"], awning=payload["awning"],
                           meta=meta, line=hook.line, check=hook.check,
                           carving=lambda: hook.stage(drawjob.CARVE), wide=payload.get("wide"),
                           deep=payload.get("deep"), roof=payload.get("roof", "flat"),
                           ridge=payload.get("ridge", "across"), roof_colour=payload.get("roof_colour"))
        except SB.WallsError as e:
            raise FlatError(str(e)) from None
        result = {"stack": {"file": "stack/stack.json", "size": out["size"], "colors": out["colors"],
                            "layers": out["layers"]}, "look": payload["look"], "awning": out["awning"], "roof": out["roof"],
                  "bays": {k: v for k, v in out["stack"].get("bays", {}).items() if k in ("drawn", "wide", "deep")},
                  "flaws": out["flaws"], "seeds": seeds, "views": "flat/views.json"}
        hook.end_stages()
        seconds = {k: s for k in hook.plan if (s := hook.seconds(k)) is not None}
        try:
            drawjob.write_times(root, spec.get("backend") or "", seconds)
        except OSError as e:
            print(f"building: could not write draw-times.json: {e}")
    except DrawCancelled:
        state, error = "cancelled", "Cancelled"
    except (FlatError, FlatUnavailable) as e:
        state, error = "error", str(e)
    except Exception as e:
        traceback.print_exc()
        state, error = "error", str(e) or e.__class__.__name__
    finally:
        shutil.rmtree(work, ignore_errors=True)
    drawjob._finish(hook, factory, job_id, state, error, result)
