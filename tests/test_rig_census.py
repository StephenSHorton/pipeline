#!/usr/bin/env python3
"""scripts/rig_census.py (table schema, baseline diff, memory guard, flag tolerance) and rig_heat's transplant."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig_census  # noqa: E402
import rig_heat  # noqa: E402


def fake_fixed() -> dict:
    """Shaped like today's chicken-2 rig.json (template is a string, no plan / clips_report / fbx_check)."""
    return {
        "class": "avian", "template": "rigs/rig-bird.glb", "clips": ["Idle", "Walk"], "dropped_clips": [],
        "unsupported_clips": ["Flap", "Glide"],
        "fit": {"facing": {"yaw": 180, "decided_by": "geometric+vlm"}, "scale_ratio": 2.297, "poor_match": True,
                "yaw": 180, "gate": {"joints_outside_box": 0.0, "bone_far_outside_0p08H": 0.002, "samples": 486,
                                     "level": "pass", "skeleton_over_mesh_extent": [0.86, 0.99, 0.84]}},
        "pred": {"cd_j2b": {"symmetric": 0.0184}},
        "floor": {"Idle": {"low_min_after": -0.0, "float_max_after": 0.0},
                  "Walk": {"low_min_after": -0.004, "float_max_after": 0.012}},
    }


def fake_inspect(a2: float = 0.02, match: str = "warn", stretch: float = 1.44) -> dict:
    checks = [{"level": "pass", "key": "planner", "step": "plan", "label": "Bird", "text": "", "by_hand": False},
              {"level": "pass", "key": "facing", "step": "skeleton", "label": "", "text": "", "yaw": 180},
              {"level": "pass", "key": "pred", "step": "skeleton", "label": "", "text": "",
               "cd_j2b": {"pred_to_template": 0.0101, "symmetric": 0.0184}},
              {"level": "warn", "key": "a2", "step": "skin", "label": "", "text": "", "share": a2}]
    if match:
        checks.insert(2, {"level": match, "key": "match", "step": "skeleton", "label": "", "text": "", "ratio": 2.297})
    return {"checks": checks, "motion": {"clips": {"Idle": {"stretch_p99": 1.06, "stretch_max": 1.2},
                                                    "Walk": {"stretch_p99": stretch, "stretch_max": 5.72}}}}


def fake_built() -> dict:
    r = fake_fixed()
    r.update(template={"mode": "built", "plan": "ground_bird", "dir": "rig/template/", "built": True},
             plan="ground_bird", clips=["Idle", "Walk", "Run", "Peck", "Flap", "Sleep", "Death"],
             unsupported_clips=[], clips_failed=[], fbx_check={"ok": True, "problems": []})
    r["fit"].update(facing={"yaw": 0, "decided_by": "declared"}, scale_ratio=1.0, poor_match=False, yaw=0)
    return r


class raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"{self.exc.__name__} not raised")
        return issubclass(et, self.exc)


class MonkeyPatch:
    """The two pytest monkeypatch calls these cases use, undone by undo()."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        old = getattr(obj, name)
        self._undo.append(lambda: setattr(obj, name, old))
        setattr(obj, name, value)

    def setenv(self, key, value):
        import os
        old = os.environ.get(key)
        self._undo.append(lambda: os.environ.__setitem__(key, old) if old is not None else os.environ.pop(key, None))
        os.environ[key] = value

    def undo(self):
        for f in reversed(self._undo):
            f()
        self._undo.clear()


class approx:
    def __init__(self, v, rel=1e-6):
        self.v, self.rel = v, rel

    def __eq__(self, other):
        return other is not None and abs(other - self.v) <= self.rel * max(abs(self.v), 1e-12)

    def __repr__(self):
        return f"approx({self.v})"


def row(rigj, inspect=None, **kw) -> dict:
    r = {"stem": "chicken", "flags": "template=fixed", "label": "t", "rc": 0, "errors": [], "flags_unsupported": [],
         "steps": []}
    r.update(rig_census.row_from(rigj, {"level": "warn"}, inspect if inspect is not None else fake_inspect(), **kw))
    return r


# ---------------------------------------------------------------- schema

def case_row_schema_fixed_and_built():
    for rigj in (fake_fixed(), fake_built(), None):
        r = row(rigj)
        for k in rig_census.ROW_KEYS:
            assert k in r, k
    f = row(fake_fixed())
    assert f["template_mode"] == "fixed" and f["plan"] is None and f["class"] == "avian"
    assert f["yaw"] == 180 and f["decided_by"] == "geometric+vlm" and f["poor_match"] is True
    assert f["scale_ratio"] == approx(2.297)
    assert f["unsupported_clips"] == ["Flap", "Glide"] and f["clips_failed"] is None and f["fbx_ok"] is None
    assert f["stretch_p99"] == approx(1.44) and f["a2"] == approx(0.02)
    assert f["floor"] == approx(0.012) and f["cd_j2b"] == approx(0.0184)
    assert f["checks"]["match"] == {"level": "warn", "ratio": 2.297}
    assert f["checks"]["pred"]["cd_j2b.symmetric"] == 0.0184
    assert f["gate"] == {"joints_outside_box": 0.0, "bone_far_outside_0p08H": 0.002, "samples": 486}
    b = row(fake_built(), fake_inspect(match=""))
    assert b["template_mode"] == "built" and b["plan"] == "ground_bird" and b["fbx_ok"] is True
    assert b["clips"] == ["Idle", "Walk", "Run", "Peck", "Flap", "Sleep", "Death"] and b["clips_failed"] == []
    assert b["poor_match"] is False and b["decided_by"] == "declared"
    json.dumps(b)  # the row is plain JSON


def case_clips_failed_from_clips_report():
    r = fake_built()
    del r["clips_failed"]
    got = rig_census.row_from(r, None, None, {"clips": {"Walk": {"passed": True, "foot_slide": 0.01},
                                                         "Flap": {"passed": False, "foot_slide": 0.0}}})
    assert got["clips_failed"] == ["Flap"] and got["foot_slide"] == 0.01


def case_table_files(tmp_path):
    table = {"label": "t", "rows": [row(fake_fixed()), row(fake_built(), fake_inspect(match=""))]}
    rig_census.write_table(table, tmp_path)
    back = json.loads((tmp_path / "table.json").read_text())
    assert len(back["rows"]) == 2
    md = (tmp_path / "table.md").read_text()
    assert "| stem |" in md and "ground_bird" in md and "## Checks" in md


# ---------------------------------------------------------------- baseline diff

def _diff(new_row, base_row):
    return rig_census.diff_tables({"rows": [new_row]}, {"rows": [base_row]})


def case_baseline_same_is_ok():
    a, b = row(fake_fixed()), row(fake_fixed())
    d = _diff(a, b)
    assert d["ok"] and d["regressions"] == 0 and d["rows"][0]["items"] == []


def case_baseline_tolerances():
    base = row(fake_fixed(), fake_inspect(a2=0.020, stretch=1.44))
    assert _diff(row(fake_fixed(), fake_inspect(a2=0.0205, stretch=1.50)), base)["ok"]  # within ±10%
    d = _diff(row(fake_fixed(), fake_inspect(a2=0.040)), base)
    assert not d["ok"] and [i["key"] for i in d["rows"][0]["items"]] == ["a2"]
    d = _diff(row(fake_fixed(), fake_inspect(stretch=1.8)), base)
    assert not d["ok"] and d["rows"][0]["items"][0]["key"] == "stretch_p99"
    # fit/gate/floor values: 1%
    r = fake_fixed()
    r["fit"]["scale_ratio"] = 2.297 * 1.005
    assert _diff(row(r), base)["ok"]
    r["fit"]["scale_ratio"] = 2.297 * 1.05
    assert not _diff(row(r), base)["ok"]
    r = fake_fixed()
    r["fit"]["gate"]["samples"] = 500
    d = _diff(row(r), base)
    assert not d["ok"] and d["rows"][0]["items"][0]["key"] == "gate.samples"


def case_baseline_levels_and_clips():
    base = row(fake_fixed())
    d = _diff(row(fake_fixed(), fake_inspect(match="fail")), base)
    assert not d["ok"] and d["rows"][0]["items"][0] == {"key": "checks.match.level", "base": "warn", "new": "fail",
                                                         "kind": "regression"}
    d = _diff(row(fake_fixed(), fake_inspect(match="pass")), base)
    assert d["ok"] and d["rows"][0]["items"][0]["kind"] == "improved"
    d = _diff(row(fake_fixed(), fake_inspect(match="")), base)  # a built template has no match row
    assert d["ok"] and d["rows"][0]["items"][0]["kind"] == "removed"
    r = fake_fixed()
    r["clips"] = ["Idle"]
    d = _diff(row(r), base)
    assert not d["ok"] and d["rows"][0]["items"][0]["key"] == "clips"
    # a newer rig.py adds plan / clips_failed / fbx_check: new, never a regression
    r = fake_fixed()
    r.update(plan="ground_bird", clips_failed=[], fbx_check={"ok": True})
    d = _diff(row(r), base)
    assert d["ok"] and {i["kind"] for i in d["rows"][0]["items"]} == {"new"}
    # a run that used to work and now fails
    bad = copy.deepcopy(base)
    bad["rc"] = 1
    assert not _diff(bad, base)["ok"]


# ---------------------------------------------------------------- guard and runner

def case_memory_guard_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(rig_census, "free_gb", lambda: 10.0)
    called = []
    monkeypatch.setattr(rig_census, "run_one", lambda *a, **k: called.append(a))
    with raises(rig_census.NotEnoughMemory):
        rig_census.run(["cow"], [{"template": "fixed"}], tmp_path, "t", min_free_gb=40)
    assert called == []


def case_memory_guard_proceeds(monkeypatch, tmp_path):
    monkeypatch.setattr(rig_census, "free_gb", lambda: 100.0)
    monkeypatch.setattr(rig_census, "run_one", lambda stem, flags, root, label, *a: row(fake_fixed()) | {
        "stem": stem, "flags": rig_census.flags_label(flags), "seconds": 1.0})
    t = rig_census.run(["cow", "chicken"], [{"template": "fixed"}], tmp_path, "t", min_free_gb=40)
    assert [r["stem"] for r in t["rows"]] == ["cow", "chicken"]
    assert (tmp_path / "census" / "t" / "table.json").is_file() and (tmp_path / "census" / "t" / "table.md").is_file()


def case_free_gb_parses_vm_stat(monkeypatch):
    out = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      1000000.\n"
           "Pages active:   5.\nPages inactive:      500000.\n")

    class P:
        stdout = out
    monkeypatch.setattr(rig_census.subprocess, "run", lambda *a, **k: P())
    assert rig_census.free_gb() == approx(1_500_000 * 16384 / 1e9)


def case_flags():
    f = rig_census.parse_flags("template=fixed, binder=heat,no-fbx")
    assert f == {"template": "fixed", "binder": "heat", "no-fbx": ""}
    args, env, missing = rig_census.command_for(f, {"stem", "class", "no-fbx"})
    assert args == ["--no-fbx"] and missing == ["template", "binder"]
    assert env == {"RIG_TEMPLATE": "fixed", "RIG_BINDER": "heat", "RIG_NO_FBX": "1"}
    assert rig_census.drop_args(["--class", "avian", "--template", "fixed", "--no-fbx"], {"--template"}) == \
        ["--class", "avian", "--no-fbx"]


FAKE_RIG = '''
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--stem", required=True)
ap.add_argument("--class", dest="cls", default="")
a = ap.parse_args()
d = Path(os.environ["FACTORY_ROOT"]) / "out" / a.stem / "rig"
d.mkdir(parents=True, exist_ok=True)
(d / "rig.json").write_text(json.dumps({"class": a.cls, "template": "rigs/rig-bird.glb", "clips": ["Idle"],
    "dropped_clips": [], "unsupported_clips": [], "fit": {"yaw": 0},
    "env_template": os.environ.get("RIG_TEMPLATE")}))
(d / "summary.json").write_text(json.dumps({"level": "pass"}))
'''


def case_run_one_tolerates_missing_flags(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "rig.py").write_text(FAKE_RIG)
    (scripts / "rig_inspect.py").write_text("import sys; sys.exit(0)\n")
    src = tmp_path / "src"
    (src / "out" / "bird" / "rig").mkdir(parents=True)
    (src / "out" / "bird" / "rig" / "rig.json").write_text(json.dumps({"class": "avian", "clips": ["stale"]}))
    (src / "out" / "bird" / "psx.glb").write_bytes(b"x")
    monkeypatch.setattr(rig_census, "HERE", scripts)
    monkeypatch.setenv("FACTORY_ROOT", str(src))
    rig_census._HELP.clear()
    r = rig_census.run_one("bird", {"template": "fixed"}, tmp_path / "root", "t")
    assert r["rc"] == 0 and r["flags_unsupported"] == ["template"]
    assert any("--template" in e for e in r["errors"]) and any("rig_render.py" in e for e in r["errors"])
    assert r["class"] == "avian" and r["clips"] == ["Idle"]  # the fresh run, not the copied stale rig.json
    got = json.loads((tmp_path / "root" / "fr" / "out" / "bird" / "rig" / "rig.json").read_text())
    assert got["env_template"] == "fixed"
    assert json.loads((src / "out" / "bird" / "rig" / "rig.json").read_text())["clips"] == ["stale"]  # source untouched
    assert (tmp_path / "root" / "census" / "t" / "runs").is_dir()


def case_run_one_failed_run_is_not_tabulated(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "rig.py").write_text("import sys\nif '--help' in sys.argv: print('--stem'); sys.exit(0)\n"
                                    "sys.exit('rig: boom')\n")
    src = tmp_path / "src"
    (src / "out" / "bird" / "rig").mkdir(parents=True)
    (src / "out" / "bird" / "rig" / "rig.json").write_text(json.dumps({"class": "avian", "clips": ["stale"]}))
    monkeypatch.setattr(rig_census, "HERE", scripts)
    monkeypatch.setenv("FACTORY_ROOT", str(src))
    rig_census._HELP.clear()
    r = rig_census.run_one("bird", {}, tmp_path / "root", "t")
    assert r["rc"] == 1 and r["clips"] is None and any("boom" in e for e in r["errors"])


# ---------------------------------------------------------------- rig_heat transplant (pure numpy)

def case_heat_transplant():
    # welded points: 3 points; source: 5 corners, two of them duplicates of welded points, one near an empty point
    Q = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [5, 5, 5]], float)
    groups = ["hips", "spine", "head", "ghost"]
    weights = [[[0, 1.0]], [[0, 0.5], [1, 0.5]], [[2, 0.9], [1, 0.1], [3, 0.2]], []]
    P = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 2, 0], [5, 5, 5.0001]], float)
    names = ["root", "hips", "spine", "head"]
    J, W, rep = rig_heat.transplant(P, Q, groups, weights, names, height=2.0)
    assert J.shape == (5, 4) and np.allclose(W.sum(1), 1)
    assert J[0, 0] == 1 and W[0, 0] == approx(1)
    assert set(J[2, :2]) == {1, 2} and W[2, 0] == approx(0.5)
    assert J[3, 0] == 3 and W[3, 0] == approx(0.9) and W[3, 1] == approx(0.1)  # 'ghost' dropped
    assert rep["unknown_groups"] == ["ghost"] and rep["unweighted_welded"] == 1 and rep["filled_points"] == 1
    assert rep["bones_unused"] == ["root"]
    assert rep["dist_max_H"] < 1e-3


class RigCensusTests(unittest.TestCase):
    """Runs every case_* above with fresh monkeypatch / tmp_path stand-ins (works under unittest and pytest)."""


def _bind(fn):
    import inspect
    params = inspect.signature(fn).parameters

    def method(self):
        mp = MonkeyPatch()
        with tempfile.TemporaryDirectory() as td:
            kw = {}
            if "monkeypatch" in params:
                kw["monkeypatch"] = mp
            if "tmp_path" in params:
                kw["tmp_path"] = Path(td)
            try:
                fn(**kw)
            finally:
                mp.undo()
    method.__name__ = "test_" + fn.__name__[5:]
    return method


for _n, _f in list(globals().items()):
    if _n.startswith("case_") and callable(_f):
        setattr(RigCensusTests, "test_" + _n[5:], _bind(_f))

if __name__ == "__main__":
    unittest.main()
