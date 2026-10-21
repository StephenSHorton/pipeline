#!/usr/bin/env python3
"""Body plans as data: scripts/rig_plans/<id>.json.

A plan says what a creature's skeleton is (joints, chains), which clips it gets and how they are gated, and
whether rig.py builds a template for it (`builder`) or wraps one of the vendor templates (`fixed_template`).

    load(id) -> dict                         a plan by id or alias
    validate(plan) -> list[str]              problems (empty when the plan is good)
    all_plans() -> list[dict]                every plan, sorted by id
    resolve(asset, cls, override=None, P=None) -> dict
                                             the plan for an asset; also writes <asset>/rig/plan.json
    spec(plan_or_none, cls, key, default)    a plan value, else today's class tables (rig_pred.PATHS, rig_fit.CHAINS)
    joints(plan) -> list[(name, parent)]     the joint tree with `{s}` expanded to each side suffix

Picking the plan (resolve): the explicit override (--plan), then intent.json `rig.plan`, then a rule over the
class: an upright avian (mesh height > 0.9 x max of the x/z extent, the same test rig.py uses) is `ground_bird`,
any other avian is `flying_bird`; quadruped, humanoid and multileg are `horse`, `biped` and `spider`.

CLI: rig_plan.py list | show <id> | validate [--all | <id>...] | resolve --stem S [--class C] [--plan ID]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PLANS = Path(__file__).resolve().parent / "rig_plans"
ROOT = Path(__file__).resolve().parents[1]

# The generators rig_clips implements (design section 2.6). rig_clips.GENERATORS, when it exists, is the source of
# truth; this set stands in until it lands. "vendor" means "sampled from the plan's fixed template".
KNOWN_GENS = {"idle_bird", "walk_biped", "run_biped", "peck", "flap_folded", "sleep_lie", "death_fall"}
VENDOR_GEN = "vendor"

# Class -> plan for classes without a choice to make.
CLASS_PLAN = {"quadruped": "horse", "humanoid": "biped", "multileg": "spider"}
UPRIGHT_MIN = 0.9

# Keys spec() reads from plan["paths"] (then from the plan's top level), falling back to rig_pred.PATHS[cls].
PATH_KEYS = ("needs", "groups", "neck_lift", "neck_from", "body", "tail", "legs")

_MISSING = object()


# ---------------------------------------------------------------- loading

def _read(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError(f"{path.name}: a plan is a JSON object")
    return plan


def _files() -> list[Path]:
    return sorted(PLANS.glob("*.json"))


def all_plans() -> list[dict]:
    """Every plan in scripts/rig_plans, sorted by id (fresh copies)."""
    out = [_read(p) for p in _files()]
    return sorted(out, key=lambda p: str(p.get("id", "")))


def load(plan_id: str) -> dict:
    """The plan with this id (or alias). A fresh copy each call, so callers may change it. KeyError if unknown."""
    key = str(plan_id or "").strip()
    if not key:
        raise KeyError("no plan id given")
    f = PLANS / f"{key}.json"
    if f.is_file() and "/" not in key and "\\" not in key:
        plan = _read(f)
        if plan.get("id") == key:
            return plan
    for plan in all_plans():
        if plan.get("id") == key or key in (plan.get("aliases") or []):
            return plan
    raise KeyError(f"unknown body plan {key!r} ({', '.join(p.get('id', '?') for p in all_plans())})")


# ---------------------------------------------------------------- the joint tree

def suffixes(plan: dict) -> list[str]:
    s = plan.get("side_suffix") or ["_L", "_R"]
    return [str(x) for x in s]


def sided(name: str, suffix: str) -> str:
    """`{s}` -> the side: 'UpperLeg_{s}' with '_L' is 'UpperLeg_L'; a bare '{s}' takes the suffix as given."""
    if "_{s}" in name:
        return name.replace("_{s}", suffix)
    return name.replace("{s}", suffix.lstrip("_"))


def joints(plan: dict) -> list[tuple[str, str | None]]:
    """[(name, parent)] in plan order; a `{s}` entry expands to every side suffix (L first, then R)."""
    out: list[tuple[str, str | None]] = []
    for entry in plan.get("joints") or []:
        name, parent = entry[0], entry[1]
        if "{s}" in name or (parent and "{s}" in parent):
            for suf in suffixes(plan):
                out.append((sided(name, suf), sided(parent, suf) if parent else None))
        else:
            out.append((name, parent))
    return out


def joint_names(plan: dict) -> list[str]:
    return [n for n, _ in joints(plan)]


def chain_names(plan: dict, chain: str) -> list[list[str]]:
    """A chain's joint names, one list per side for a sided chain (`names` are bare: 'UpperLeg' -> 'UpperLeg_L')."""
    c = (plan.get("chains") or {}).get(chain)
    if c is None:
        return []
    if isinstance(c, list):
        return [list(c)]
    names = list(c.get("names") or [])
    if c.get("sided"):
        return [[n + suf for n in names] for suf in suffixes(plan)]
    return [names]


# ---------------------------------------------------------------- validation

def _gens() -> set[str]:
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import rig_clips  # noqa: F401  (lands with WP6)
        gens = getattr(rig_clips, "GENERATORS", None)
        if gens:
            return set(gens)
    except Exception:
        pass
    return set(KNOWN_GENS)


def _flatten(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [n for v in x.values() for n in _flatten(v)]
    if isinstance(x, (list, tuple)):
        return [n for v in x for n in _flatten(v)]
    return []


def validate(plan: dict) -> list[str]:
    """Problems with a plan, as plain sentences (empty when it is good)."""
    errs: list[str] = []
    pid = plan.get("id")
    if not isinstance(pid, str) or not pid:
        errs.append("plan has no id")
    if not plan.get("class"):
        errs.append("plan has no class")
    builder, fixed = plan.get("builder"), plan.get("fixed_template")
    if not builder and not fixed:
        errs.append("plan has neither a builder nor a fixed template")
    if builder and builder != "rig_build":
        errs.append(f"unknown builder {builder!r}")
    if fixed is not None:
        if not isinstance(fixed, dict):
            errs.append("fixed_template is not an object")
        else:
            for k in ("rig", "clips", "ref", "default"):
                if k not in fixed:
                    errs.append(f"fixed_template has no {k!r}")
            m2m = ROOT / "vendor" / "mesh2motion-app" / "static"
            if m2m.is_dir():
                for f in [fixed.get("rig")] + list(fixed.get("clips") or []):
                    if f and not (m2m / f).is_file():
                        errs.append(f"fixed_template file {f} is missing")
    sfx = suffixes(plan)
    if len(sfx) != 2 or sfx[0] == sfx[1]:
        errs.append(f"side_suffix must be two different suffixes, not {sfx}")

    # joints (a builder plan must declare its tree; a wrapper plan takes it from the template file)
    tree = joints(plan)
    names = [n for n, _ in tree]
    if builder or tree:
        if not tree:
            errs.append("plan has no joints")
        seen: set[str] = set()
        for n in names:
            if n in seen:
                errs.append(f"joint {n} is declared twice")
            seen.add(n)
        for n in ("root", "hips"):
            if n not in seen:
                errs.append(f"joint {n!r} is missing")
        roots = [n for n, p in tree if p is None]
        if roots != ["root"]:
            errs.append(f"the tree must have exactly one root named 'root', not {roots}")
        for n, p in tree:
            if p is not None and p not in seen:
                errs.append(f"joint {n} has an undeclared parent {p}")
            if n == "fit":
                errs.append("no joint may be named 'fit'")
        if "hips" in seen and dict(tree).get("hips") != "root":
            errs.append("hips must be the child of root")
        # side-suffix rule: every sided name has its mirror, and nothing else ends in a side suffix
        for n in names:
            for i, s in enumerate(sfx):
                if n.endswith(s):
                    twin = n[: -len(s)] + sfx[1 - i]
                    if twin not in seen:
                        errs.append(f"joint {n} has no mirror {twin}")
        for entry in plan.get("joints") or []:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                errs.append(f"joint entry {entry!r} is not [name, parent]")
                continue
            n, p = entry
            if "{s}" not in n and p and "{s}" in p:
                errs.append(f"joint {n} is unsided but its parent {p} is sided")
        # chains and paths name declared joints
        for cname in (plan.get("chains") or {}):
            for side in chain_names(plan, cname):
                for n in side:
                    if n not in seen:
                        errs.append(f"chain {cname} names an undeclared joint {n}")
            c = plan["chains"][cname]
            if isinstance(c, dict) and c.get("attach") and c["attach"] not in seen:
                errs.append(f"chain {cname} attaches to an undeclared joint {c['attach']}")
        paths = plan.get("paths") or {}
        for k in ("body", "tail", "legs", "groups", "neck_lift", "neck_from"):
            for n in _flatten(paths.get(k)):
                if n not in seen:
                    errs.append(f"paths.{k} names an undeclared joint {n}")
        for k in ("leg_ik",):
            li = plan.get(k) or {}
            for role in ("femur", "A", "B", "K"):
                if role in li:
                    for s in sfx:
                        if li[role] + s not in seen:
                            errs.append(f"leg_ik.{role} {li[role]} has no joint {li[role] + s}")
        for n in plan.get("neck_ik") or []:
            if n not in seen:
                errs.append(f"neck_ik names an undeclared joint {n}")

    # clips
    clips = plan.get("clips")
    if not isinstance(clips, list) or not clips:
        errs.append("plan has no clips")
        clips = []
    gens = _gens()
    cnames: list[str] = []
    for c in clips:
        if not isinstance(c, dict) or not c.get("name"):
            errs.append(f"clip {c!r} has no name")
            continue
        n = c["name"]
        if n in cnames:
            errs.append(f"clip {n} is listed twice")
        cnames.append(n)
        g = c.get("gen")
        if g == VENDOR_GEN:
            if not fixed:
                errs.append(f"clip {n} is a vendor clip but the plan has no fixed template")
        elif g not in gens:
            errs.append(f"clip {n} has an unknown generator {g!r}")
        elif not builder:
            errs.append(f"clip {n} is synthesised ({g}) but the plan has no builder")
        if g != VENDOR_GEN:
            if not isinstance(c.get("T"), (int, float)) or c["T"] <= 0:
                errs.append(f"clip {n} has no positive duration T")
            if not isinstance(c.get("loop"), bool):
                errs.append(f"clip {n} does not say whether it loops")
            me = c.get("min_energy")
            if me is not None and not (0 < float(me) <= 1):
                errs.append(f"clip {n} min_energy {me} is outside (0, 1]")
    ref = plan.get("ref")
    if not ref:
        errs.append("plan has no ref clip")
    elif builder and ref in cnames:
        errs.append(f"the ref clip {ref} is written automatically and must not be in clips")
    default = plan.get("default")
    if not isinstance(default, list) or not default:
        errs.append("plan has no default clips")
    else:
        for n in default:
            if n not in cnames:
                errs.append(f"default clip {n} is not in clips")
        if len(set(default)) != len(default):
            errs.append("default lists a clip twice")
    if fixed and isinstance(fixed, dict) and isinstance(default, list) and fixed.get("default") not in (None, default):
        errs.append("default differs from fixed_template.default")
    return errs


# ---------------------------------------------------------------- resolve

def _upright(P) -> tuple[bool, float]:
    import numpy as np
    P = np.asarray(P, float).reshape(-1, 3)
    h = float(np.ptp(P[:, 1]))
    w = float(max(np.ptp(P[:, 0]), np.ptp(P[:, 2])))
    ratio = h / w if w > 1e-12 else float("inf")
    return h > UPRIGHT_MIN * w, ratio


def _source_points(asset: Path):
    """The mesh rig.py would rig (psx.glb, then textured_photo.glb, then textured.glb), as raw positions."""
    import numpy as np
    import rig
    for n in ("psx.glb", "textured_photo.glb", "textured.glb"):
        f = asset / n
        if f.is_file():
            doc, b = rig.read_glb(f)
            pts = [rig.accessor(doc, b, pr["attributes"]["POSITION"]) for m in doc.get("meshes", [])
                   for pr in m.get("primitives", []) if "POSITION" in pr.get("attributes", {})]
            if pts:
                return np.concatenate(pts).astype(float)
    return None


def _json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def facing_info(asset: Path) -> dict:
    """{facing, facing_source, canonical} from psx/psx.json (facing_check), flat/key.json and intent.json."""
    psx = _json(asset / "psx" / "psx.json")
    key = _json(asset / "flat" / "key.json")
    intent = _json(asset / "intent.json")
    ipsx = intent.get("psx") if isinstance(intent.get("psx"), dict) else {}
    facing = psx.get("facing") or key.get("facing") or ipsx.get("facing") or intent.get("facing")
    canonical = bool(key.get("canonical") or psx.get("canonical") or intent.get("canonical"))
    source = psx.get("facing_source") or key.get("facing_source") or intent.get("facing_source")
    if not source:
        source = "drawn" if canonical else ("user" if facing else None)
    check = psx.get("facing_check") if isinstance(psx.get("facing_check"), dict) else None
    out = {"facing": facing, "facing_source": source, "canonical": canonical}
    if check is not None:
        out["facing_check"] = check
    return out


def rule(cls: str, P=None) -> tuple[str, dict]:
    """The plan id the class rule gives, and why."""
    if cls == "avian":
        if P is None:
            return "flying_bird", {"upright": None, "why": "avian (no mesh to measure)"}
        up, ratio = _upright(P)
        why = (f"avian, upright (height {ratio:.2f}x its length)" if up
               else f"avian, not upright (height {ratio:.2f}x its length)")
        return ("ground_bird" if up else "flying_bird"), {"upright": up, "height_ratio": round(ratio, 3), "why": why}
    if cls in CLASS_PLAN:
        return CLASS_PLAN[cls], {"why": cls}
    raise ValueError(f"no body plan for class {cls!r}")


def write_plan(work, plan: dict) -> None:
    """<work>/plan.json (rig/plan.json): the plan a run used."""
    work = Path(work)
    try:
        work.mkdir(parents=True, exist_ok=True)
        (work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def resolve(asset, cls: str, override: str | None = None, P=None, write: bool = True) -> dict:
    """The plan for an asset: --plan, then intent.json rig.plan, then the class rule. Writes <asset>/rig/plan.json
    unless write is False (rig.py writes the plan it actually used, after build-vs-fixed).

    An intent.json plan for another class is ignored (the class rule decides; resolved.ignored_intent says so); an
    explicit override for another class is kept and marked resolved.class_mismatch (rig.py stops on it).
    The returned dict is the plan plus `facing`, `facing_source`, `canonical` and `resolved` ({by, why, ...})."""
    asset = Path(asset)
    intent = _json(asset / "intent.json")
    irig = intent.get("rig") if isinstance(intent.get("rig"), dict) else {}
    resolved: dict
    if override:
        plan = load(override)
        resolved = {"by": "override", "why": f"--plan {override}"}
    elif irig.get("plan") and _intent_plan_ok(str(irig["plan"]), cls):
        plan = load(str(irig["plan"]))
        resolved = {"by": "intent", "why": f"intent.json rig.plan {irig['plan']}"}
    else:
        if P is None and cls == "avian":
            P = _source_points(asset)
        pid, why = rule(cls, P)
        plan = load(pid)
        resolved = {"by": "rule", **why}
        if irig.get("plan"):
            resolved["ignored_intent"] = f"intent.json rig.plan {irig['plan']} is not for class {cls}"
    if plan.get("class") != cls:
        resolved["class_mismatch"] = f"plan {plan.get('id')} is for {plan.get('class')}, the asset is {cls}"
    plan.update(facing_info(asset))
    plan["resolved"] = resolved
    if write:
        write_plan(asset / "rig", plan)
    return plan


def _intent_plan_ok(pid: str, cls: str) -> bool:
    try:
        return load(pid).get("class") == cls
    except KeyError:
        return False


# ---------------------------------------------------------------- spec

def _class_table(cls: str, key: str):
    """Today's tables: rig_pred.PATHS[cls] and rig_fit.CHAINS[cls] (read-only)."""
    if key == "chains":
        import rig_fit
        return rig_fit.CHAINS.get(cls, _MISSING)
    import rig_pred
    paths = rig_pred.PATHS.get(cls)
    if paths is None:
        return _MISSING
    if key == "paths":
        return paths
    return paths.get(key, _MISSING)


def spec(plan, cls: str, key: str, default=None):
    """A plan value with today's class tables as the fallback.

    'paths' and 'chains' return plan[key] if present, else rig_pred.PATHS[cls] / rig_fit.CHAINS[cls]. Path keys
    (needs, groups, neck_lift, neck_from, body, tail, legs) come from plan['paths'][key] or plan[key], else
    rig_pred.PATHS[cls][key]. Any other key is plan[key]. The default when none of these has it."""
    if plan:
        if key in plan and plan[key] is not None:
            return plan[key]
        if key in PATH_KEYS and key in (plan.get("paths") or {}):
            return plan["paths"][key]
    if key in ("paths", "chains") or key in PATH_KEYS:
        try:
            got = _class_table(cls, key)
        except ImportError:
            return default
        if got is not _MISSING:
            return got
    return default


# ---------------------------------------------------------------- CLI

def _summary(p: dict) -> str:
    kind = "builds" if p.get("builder") else "fixed template"
    n = len(joint_names(p))
    return f"{p.get('id', '?'):<12} {p.get('class', '?'):<10} {kind:<15} {str(n) + ' joints' if n else '':<10} " \
           f"{', '.join(p.get('default') or [])}"


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sh = sub.add_parser("show")
    sh.add_argument("id")
    sh.add_argument("--joints", action="store_true", help="the expanded joint tree only")
    va = sub.add_parser("validate")
    va.add_argument("ids", nargs="*")
    va.add_argument("--all", action="store_true")
    rs = sub.add_parser("resolve")
    rs.add_argument("--stem", required=True)
    rs.add_argument("--class", dest="cls", default="")
    rs.add_argument("--plan", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for p in all_plans():
            print(_summary(p))
        return 0
    if a.cmd == "show":
        p = load(a.id)
        print(json.dumps(joints(p) if a.joints else p, indent=2))
        return 0
    if a.cmd == "validate":
        if a.all or not a.ids:
            plans = [(f.stem, _read(f)) for f in _files()]
        else:
            plans = [(i, load(i)) for i in a.ids]
        bad = 0
        for name, p in plans:
            errs = validate(p)
            if p.get("id") != name and (a.all or not a.ids):
                errs.append(f"file {name}.json holds plan id {p.get('id')!r}")
            print(f"{name}: " + ("ok" if not errs else f"{len(errs)} problem(s)"))
            for e in errs:
                print("  - " + e)
            bad += bool(errs)
        return 1 if bad else 0
    if a.cmd == "resolve":
        import rig
        asset = rig.OUT / a.stem
        if not asset.is_dir():
            print(f"rig_plan: no asset {a.stem}", file=sys.stderr)
            return 2
        cls = a.cls
        if not cls:
            intent = _json(asset / "intent.json")
            cls = (intent.get("rig") or {}).get("class") if isinstance(intent.get("rig"), dict) else None
            cls = cls or _json(asset / "rig" / "rig.json").get("class")
        if not cls:
            print("rig_plan: give --class (the asset has no rig class yet)", file=sys.stderr)
            return 2
        p = resolve(asset, cls, a.plan)
        print(json.dumps({k: p.get(k) for k in ("id", "class", "builder", "facing", "facing_source", "canonical",
                                                 "resolved", "default")}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
