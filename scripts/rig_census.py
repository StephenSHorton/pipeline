"""Rig census: re-rig a set of assets under one or more flag sets on a scratch copy and tabulate the results.

  .venv/bin/python scripts/rig_census.py --stems cow,operator,chicken-canon --flags 'template=auto' \\
      [--flags 'template=fixed,binder=heat'] --root /scratch/census [--label L] [--baseline B] [--min-free-gb 40]

For each stem and flag set it:
  1. refuses to start when free + inactive memory is below --min-free-gb (40 GB: skin-tokens needs it);
  2. copies out/<stem> into <root>/fr/out/<stem> (a scratch FACTORY_ROOT; the real out/ is never touched) and
     removes the copy's old rig.json / inspect.json / summary.json / clips_report.json / fbx_check.json, so a failed
     run can never be read back as a result;
  3. runs rig.py --stem S <flags>, then rig_inspect.py --stem S, then rig_render.py --stem S (each with
     FACTORY_ROOT set). A flag k=v becomes `--k v` when rig.py's --help lists `--k`, and is ALWAYS also set as the env
     RIG_<K>=v, so flags rig.py doesn't have yet still reach it the old way; unsupported flags and every step's
     errors are recorded in the row instead of stopping the census;
  4. gathers rig.json, summary.json, inspect.json, clips_report.json and fbx_check.json into one row (row_from).
It writes <root>/census/<label>/table.json and table.md (plus runs/<stem>__<flags>/ with the run's rig/ outputs).
--baseline FILE diffs against a saved table.json: clip lists must match, check levels must be the same or better,
a2 and stretch_p99 within ±10%, fit/gate/floor values within 1%. Exit 1 when the diff finds a regression.
`class` defaults to the class in the asset's existing rig.json (so the planner isn't asked again); pass class=... to
override, or --ask-planner to let rig.py ask.
Ports scratch e2/e3 run.sh, report.py, tools_summ.py, e1 table.py and baseline.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
GATHER = ("rig.json", "summary.json", "inspect.json", "clips_report.json", "fbx_check.json")
LEVEL = {"pass": 0, "info": 0, "warn": 1, "fail": 2}
# relative tolerances for the baseline diff
TOL_LOOSE = 0.10  # a2, stretch_p99 (and other motion numbers)
TOL_TIGHT = 0.01  # fit, gate and floor values
LOOSE_KEYS = ("a2", "stretch_p99", "foot_slide", "foot")
TIGHT_KEYS = ("scale_ratio", "gate", "floor", "cd_j2b")
ROW_KEYS = ("stem", "flags", "label", "rc", "errors", "flags_unsupported", "plan", "template_mode", "class", "yaw",
            "decided_by", "scale_ratio", "poor_match", "clips", "dropped_clips", "unsupported_clips", "clips_failed",
            "checks", "level", "a2", "foot", "stretch_p99", "foot_slide", "floor", "gate", "cd_j2b", "fbx_ok",
            "seconds", "steps")


# ---------------------------------------------------------------- memory guard

def free_gb() -> float:
    """Free + inactive memory in GB (vm_stat, as run.sh did); inf where vm_stat is missing (not macOS)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return float("inf")
    page = int((re.search(r"page size of (\d+)", out) or [0, 16384])[1])
    pages = 0
    for key in ("Pages free", "Pages inactive"):
        m = re.search(key + r":\s+(\d+)", out)
        pages += int(m.group(1)) if m else 0
    return pages * page / 1e9 if pages else float("inf")


class NotEnoughMemory(SystemExit):
    pass


def guard(min_gb: float) -> float:
    got = free_gb()
    if got < min_gb:
        raise NotEnoughMemory(f"rig_census: only {got:.0f} GB free+inactive, need {min_gb:.0f} GB for the skinner")
    return got


# ---------------------------------------------------------------- flags

def parse_flags(s: str) -> dict[str, str]:
    """'template=fixed,binder=heat,no-fbx' -> {'template': 'fixed', 'binder': 'heat', 'no-fbx': ''}"""
    out: dict[str, str] = {}
    for part in (p.strip() for p in (s or "").split(",")):
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip().lstrip("-")] = v.strip()
    return out


def flags_label(flags: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" if v else k for k, v in flags.items()) or "default"


_HELP: dict[str, set[str]] = {}


def supported_options(script: Path) -> set[str]:
    """Long options that a script's --help lists (cached)."""
    key = str(script)
    if key not in _HELP:
        try:
            out = subprocess.run([PY, str(script), "--help"], capture_output=True, text=True, timeout=120).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        _HELP[key] = set(re.findall(r"--([a-zA-Z][\w-]*)", out))
    return _HELP[key]


def command_for(flags: dict[str, str], supported: set[str]) -> tuple[list[str], dict[str, str], list[str]]:
    """CLI args for the flags rig.py has, env RIG_<K> for every flag, and the flags it does not have."""
    args, env, missing = [], {}, []
    for k, v in flags.items():
        env["RIG_" + k.upper().replace("-", "_")] = v if v else "1"
        if k in supported:
            args += [f"--{k}"] + ([v] if v else [])
        else:
            missing.append(k)
    return args, env, missing


def drop_args(args: list[str], bad: set[str]) -> list[str]:
    """args without the options in `bad` (and the value after each)."""
    out, i = [], 0
    while i < len(args):
        if args[i] in bad:
            i += 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
            continue
        out.append(args[i])
        i += 1
    return out


# ---------------------------------------------------------------- reading a run

def _load(d: Path, name: str):
    for p in (d / "rig" / name, d / "rig" / "template" / name):  # never out/<stem>/<name>: older takes left some there
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except ValueError:
                return None
    return None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _check_values(c: dict) -> dict:
    """The numeric values of an inspect check row (top level and one level down)."""
    vals = {}
    for k, v in c.items():
        if k in ("level", "key", "step", "label", "text", "by_hand"):
            continue
        if _num(v) is not None:
            vals[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if _num(v2) is not None:
                    vals[f"{k}.{k2}"] = v2
    return vals


def _first(*xs):
    return next((x for x in xs if x is not None), None)


def row_from(rigj: dict | None, summary: dict | None = None, inspect: dict | None = None,
             clips_report: dict | None = None, fbx_check: dict | None = None) -> dict:
    """One census row from a run's JSON files. Every key in ROW_KEYS[6:] is present; unknown values are None."""
    r = rigj or {}
    fit = r.get("fit") or {}
    t = r.get("template")
    tmode = (t.get("mode") or ("built" if t.get("built") else "fixed")) if isinstance(t, dict) else ("fixed" if t else None)
    plan = r.get("plan") if not isinstance(r.get("plan"), dict) else r["plan"].get("id")
    if plan is None and isinstance(t, dict):
        plan = t.get("plan")
    facing = fit.get("facing") or {}
    reg = fit.get("registration") or {}
    checks = {}
    for c in (inspect or {}).get("checks") or []:
        if isinstance(c, dict) and c.get("key"):
            checks[c["key"]] = {"level": c.get("level"), **_check_values(c)}

    def cval(key, *fields):
        c = checks.get(key) or {}
        return _first(*(c.get(f) for f in fields))

    mc = ((inspect or {}).get("motion") or {}).get("clips") or {}
    p99s = [v.get("stretch_p99") for v in mc.values() if isinstance(v, dict) and _num(v.get("stretch_p99")) is not None]
    floor_d = r.get("floor") or {}
    floor_worst = None
    if isinstance(floor_d, dict) and floor_d:
        vals = []
        for v in floor_d.values():
            if isinstance(v, dict):
                vals += [max(0.0, -(v.get("low_min_after") or 0.0)), max(0.0, v.get("float_max_after") or 0.0)]
        floor_worst = round(max(vals), 4) if vals else None
    gate = fit.get("gate") or {}
    gate_vals = {k: v for k, v in gate.items() if _num(v) is not None}
    cd = ((r.get("pred") or {}).get("cd_j2b") or {})
    cd_j2b = _first(cval("built", "cd_j2b"), cd.get("symmetric") if isinstance(cd, dict) else None)
    crep = clips_report or r.get("clips_report") or {}
    failed = r.get("clips_failed")
    if failed is None and isinstance(crep, dict) and crep:
        cl = crep.get("clips", crep)
        if isinstance(cl, dict):
            failed = sorted(n for n, v in cl.items() if isinstance(v, dict) and v.get("passed") is False) or []
    fbx = fbx_check or r.get("fbx_check")
    slide = cval("slide", "value", "worst", "slide", "max")
    if slide is None and isinstance(crep, dict):
        sl = [v.get("foot_slide") for v in (crep.get("clips", crep) or {}).values()
              if isinstance(v, dict) and _num(v.get("foot_slide")) is not None]
        slide = max(sl) if sl else None
    return {
        "plan": plan,
        "template_mode": tmode,
        "class": r.get("class"),
        "yaw": _first(facing.get("yaw"), fit.get("yaw")),
        "decided_by": facing.get("decided_by"),
        "scale_ratio": _first(fit.get("scale_ratio"), reg.get("scale_ratio") if isinstance(reg, dict) else None),
        "poor_match": _first(fit.get("poor_match"), reg.get("poor_match") if isinstance(reg, dict) else None),
        "clips": r.get("clips"),
        "dropped_clips": r.get("dropped_clips"),
        "unsupported_clips": r.get("unsupported_clips"),
        "clips_failed": failed,
        "checks": checks,
        "level": (summary or {}).get("level"),
        "a2": cval("a2", "share", "value", "a2"),
        "foot": cval("foot", "share", "value", "foot"),
        "stretch_p99": round(max(p99s), 3) if p99s else None,
        "foot_slide": slide,
        "floor": floor_worst,
        "gate": gate_vals or None,
        "cd_j2b": cd_j2b,
        "fbx_ok": fbx.get("ok") if isinstance(fbx, dict) else None,
        "seconds": None,
    }


# ---------------------------------------------------------------- running

def _step(cmd: list[str], env: dict, timeout: float, log: Path) -> dict:
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = -9, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), f"timeout after {timeout:.0f} s"
    except OSError as e:
        rc, out, err = -1, "", str(e)
    log.write_text(f"$ {' '.join(cmd)}\n{out}\n--- stderr ---\n{err}\n", encoding="utf-8")
    return {"cmd": cmd[1:], "rc": rc, "seconds": round(time.time() - t0, 1),
            "error": None if rc == 0 else (err or out).strip()[-600:]}


def run_one(stem: str, flags: dict[str, str], root: Path, label: str, ask_planner: bool = False,
            timeout: float = 3600, render: bool = True) -> dict:
    src = Path(os.environ.get("FACTORY_ROOT") or ROOT) / "out" / stem
    fr = root / "fr"
    dst = fr / "out" / stem
    tag = re.sub(r"[^\w=,.-]+", "_", f"{stem}__{flags_label(flags)}")
    rundir = root / "census" / label / "runs" / tag
    row = {"stem": stem, "flags": flags_label(flags), "label": label, "rc": None, "errors": [],
           "flags_unsupported": [], "steps": []}
    if not src.is_dir():
        row["errors"].append(f"no asset out/{stem}")
        return row | {k: None for k in ROW_KEYS if k not in row}
    old = _load(src, "rig.json") or {}
    flags = dict(flags)
    if "class" not in flags and not ask_planner and old.get("class"):
        flags["class"] = old["class"]
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)
    for n in GATHER:
        for p in (dst / "rig" / n, dst / "rig" / "template" / n):
            p.unlink(missing_ok=True)
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    rigpy = HERE / "rig.py"
    args, fenv, missing = command_for(flags, supported_options(rigpy))
    row["flags_unsupported"] = missing
    if missing:
        row["errors"].append(f"rig.py has no --{', --'.join(missing)}: passed as env "
                             + ", ".join("RIG_" + m.upper().replace("-", "_") for m in missing))
    env = dict(os.environ, FACTORY_ROOT=str(fr), PIPELINE_ROOT=str(ROOT), **fenv)
    t0 = time.time()
    s = _step([PY, str(rigpy), "--stem", stem, *args], env, timeout, rundir / "rig.log")
    if s["rc"] == 2 and "unrecognized arguments" in (s["error"] or ""):  # --help lied: retry with env only
        line = next(ln for ln in s["error"].splitlines() if "unrecognized arguments" in ln)
        row["errors"].append("rig.py: " + line)
        bad = set(line.split("unrecognized arguments:", 1)[1].split())
        args = drop_args(args, bad)
        row["flags_unsupported"] += sorted(a.lstrip("-") for a in bad if a.startswith("--"))
        s = _step([PY, str(rigpy), "--stem", stem, *args], env, timeout, rundir / "rig.log")
    row["steps"].append(s)
    row["rc"] = s["rc"]
    if s["rc"] != 0:
        row["errors"].append("rig.py: " + (s["error"] or f"exit {s['rc']}"))
    else:
        for script, name in (("rig_inspect.py", "inspect"), ("rig_render.py", "render")):
            if name == "render" and not render:
                continue
            path = HERE / script
            if not path.is_file():
                row["errors"].append(f"{script} not in scripts/ yet; skipped")
                row["steps"].append({"cmd": [script], "rc": None, "seconds": 0, "error": "missing"})
                continue
            st = _step([PY, str(path), "--stem", stem], env, timeout, rundir / f"{name}.log")
            row["steps"].append(st)
            if st["rc"] != 0:
                row["errors"].append(f"{script}: " + (st["error"] or f"exit {st['rc']}"))
    secs = round(time.time() - t0, 1)
    files = {n: _load(dst, n) for n in GATHER}
    if s["rc"] != 0:  # never tabulate a half-written run
        files = {n: None for n in GATHER}
    row.update(row_from(files["rig.json"], files["summary.json"], files["inspect.json"], files["clips_report.json"],
                        files["fbx_check.json"]))
    row["seconds"] = secs
    if (dst / "rig").is_dir():
        shutil.copytree(dst / "rig", rundir / "rig", dirs_exist_ok=True)
    return row


def run(stems: list[str], flagsets: list[dict[str, str]], root: Path, label: str, min_free_gb: float = 40.0,
        ask_planner: bool = False, timeout: float = 3600, render: bool = True) -> dict:
    """Every stem under every flag set -> the table (also written to <root>/census/<label>/table.json, table.md)."""
    root = Path(root)
    rows = []
    for flags in flagsets:
        for stem in stems:
            got = guard(min_free_gb)
            print(f"rig_census: {stem} [{flags_label(flags)}] ({got:.0f} GB free)", flush=True)
            row = run_one(stem, flags, root, label, ask_planner, timeout, render)
            print(f"rig_census:   rc={row['rc']} {row.get('template_mode')} clips={row.get('clips')} "
                  f"{row['seconds']} s" + (f" errors: {row['errors']}" if row["errors"] else ""), flush=True)
            rows.append(row)
    table = {"version": 1, "label": label, "made": int(time.time()), "root": str(root),
             "flagsets": [flags_label(f) for f in flagsets], "stems": stems, "rows": rows}
    write_table(table, root / "census" / label)
    return table


# ---------------------------------------------------------------- baseline diff

def _rel(a, b) -> float:
    if a == b:
        return 0.0
    den = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / den


def _close(a, b, tol: float, floor_abs: float = 1e-3) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= floor_abs or _rel(a, b) <= tol


def diff_rows(new: dict, base: dict) -> list[dict]:
    """Regressions and changes of one row against its baseline row. Each item: {key, base, new, kind} where kind is
    'regression' (fails the census), 'improved', 'new' or 'removed'."""
    out = []

    def add(key, b, n, kind):
        out.append({"key": key, "base": b, "new": n, "kind": kind})

    for k in ("template_mode", "class", "plan", "clips", "dropped_clips", "unsupported_clips", "clips_failed",
              "poor_match", "fbx_ok"):
        b, n = base.get(k), new.get(k)
        if b is None and k in ("plan", "clips_failed", "fbx_ok"):  # outputs the baseline's rig.py didn't write yet
            if n is not None:
                add(k, None, n, "new")
            continue
        if n != b:
            kind = "regression"
            if k == "poor_match" and base.get(k) and not new.get(k):
                kind = "improved"
            if k == "fbx_ok" and new.get(k) and not base.get(k):
                kind = "improved"
            add(k, base.get(k), new.get(k), kind)
    if (base.get("rc") == 0) and new.get("rc") != 0:
        add("rc", base.get("rc"), new.get("rc"), "regression")
    bc, nc = base.get("checks") or {}, new.get("checks") or {}
    for key, b in bc.items():
        n = nc.get(key)
        if n is None:
            add(f"checks.{key}", b.get("level"), None, "removed")  # e.g. no match row on a built template
            continue
        lb, ln = LEVEL.get(b.get("level"), 0), LEVEL.get(n.get("level"), 0)
        if ln > lb:
            add(f"checks.{key}.level", b.get("level"), n.get("level"), "regression")
        elif ln < lb:
            add(f"checks.{key}.level", b.get("level"), n.get("level"), "improved")
    for key in nc:
        if key not in bc:
            add(f"checks.{key}", None, nc[key].get("level"), "new")
    for k in LOOSE_KEYS:
        b, n = _num(base.get(k)), _num(new.get(k))
        if b is not None and n is not None and not _close(n, b, TOL_LOOSE):
            add(k, b, n, "regression" if n > b else "improved")
    for k in TIGHT_KEYS:
        b, n = base.get(k), new.get(k)
        if isinstance(b, dict) and isinstance(n, dict):
            for kk in b:
                if _num(b[kk]) is not None and _num(n.get(kk)) is not None and not _close(n[kk], b[kk], TOL_TIGHT):
                    add(f"{k}.{kk}", b[kk], n[kk], "regression")
        elif _num(b) is not None and _num(n) is not None and not _close(n, b, TOL_TIGHT):
            add(k, b, n, "regression" if n > b else "improved")
    return out


def diff_tables(new: dict, base: dict) -> dict:
    by = {(r["stem"], r["flags"]): r for r in base.get("rows", [])}
    by_stem = {}
    for r in base.get("rows", []):
        by_stem.setdefault(r["stem"], r)
    rows, bad = [], 0
    for r in new.get("rows", []):
        b = by.get((r["stem"], r["flags"])) or by_stem.get(r["stem"])
        if b is None:
            rows.append({"stem": r["stem"], "flags": r["flags"], "base_flags": None, "items": [], "regressions": 0,
                         "note": "not in the baseline"})
            continue
        items = diff_rows(r, b)
        nreg = sum(1 for i in items if i["kind"] == "regression")
        bad += nreg
        rows.append({"stem": r["stem"], "flags": r["flags"], "base_flags": b["flags"], "items": items,
                     "regressions": nreg})
    return {"ok": bad == 0, "regressions": bad, "rows": rows,
            "tolerance": {"loose": TOL_LOOSE, "loose_keys": list(LOOSE_KEYS), "tight": TOL_TIGHT,
                          "tight_keys": list(TIGHT_KEYS)}}


# ---------------------------------------------------------------- output

def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, list):
        return ", ".join(map(str, v)) or "[]"
    return str(v)


def to_markdown(table: dict) -> str:
    cols = ("stem", "flags", "rc", "template_mode", "plan", "class", "yaw", "decided_by", "scale_ratio", "poor_match",
            "clips", "dropped_clips", "unsupported_clips", "clips_failed", "a2", "foot", "stretch_p99", "foot_slide",
            "floor", "cd_j2b", "fbx_ok", "seconds")
    lines = [f"# Rig census: {table.get('label')}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table["rows"]:
        lines.append("| " + " | ".join(_fmt(r.get(c)).replace("|", "/") for c in cols) + " |")
    lines += ["", "## Checks", ""]
    keys = sorted({k for r in table["rows"] for k in (r.get("checks") or {})})
    if keys:
        lines += ["| stem | flags | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
        for r in table["rows"]:
            ch = r.get("checks") or {}
            lines.append(f"| {r['stem']} | {r['flags']} | " + " | ".join((ch.get(k) or {}).get("level") or "–" for k in keys) + " |")
    errs = [(r["stem"], r["flags"], e) for r in table["rows"] for e in r.get("errors") or []]
    if errs:
        lines += ["", "## Errors", ""] + [f"- {s} [{f}]: {e.splitlines()[-1] if e else e}" for s, f, e in errs]
    d = table.get("diff")
    if d:
        lines += ["", f"## Against the baseline: {'no regressions' if d['ok'] else str(d['regressions']) + ' regressions'}", ""]
        for r in d["rows"]:
            for i in r["items"]:
                lines.append(f"- {r['stem']} [{r['flags']}] {i['kind']}: {i['key']} {_fmt(i['base'])} → {_fmt(i['new'])}")
            if r.get("note"):
                lines.append(f"- {r['stem']} [{r['flags']}]: {r['note']}")
    return "\n".join(lines) + "\n"


def write_table(table: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table.json").write_text(json.dumps(table, indent=2, default=str) + "\n", encoding="utf-8")
    (outdir / "table.md").write_text(to_markdown(table), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", required=True, help="comma-separated asset stems under out/")
    ap.add_argument("--flags", action="append", default=None,
                    help="one flag set, k=v[,k=v...] (repeat for more sets); e.g. template=fixed")
    ap.add_argument("--root", default=str(Path(tempfile.gettempdir()) / "rig_census"),
                    help="scratch root: <root>/fr is the FACTORY_ROOT copy, <root>/census/<label>/ the table")
    ap.add_argument("--label", default="", help="table name (default: the flag sets)")
    ap.add_argument("--baseline", default="", help="a saved table.json to diff against")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--ask-planner", action="store_true", help="don't pass the class from the asset's rig.json")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per step")
    args = ap.parse_args(argv)
    stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    flagsets = [parse_flags(f) for f in (args.flags or [""])]
    label = args.label or re.sub(r"[^\w=,.-]+", "_", ";".join(flags_label(f) for f in flagsets))
    base = None
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
    table = run(stems, flagsets, Path(args.root), label, args.min_free_gb, args.ask_planner, args.timeout,
                not args.no_render)
    outdir = Path(args.root) / "census" / label
    if base is not None:
        table["diff"] = diff_tables(table, base)
        table["baseline"] = str(args.baseline)
        write_table(table, outdir)
    print(f"rig_census: wrote {outdir / 'table.json'}")
    print(to_markdown(table))
    return 0 if base is None or table["diff"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
