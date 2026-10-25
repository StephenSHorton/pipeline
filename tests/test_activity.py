"""The activity feed: scripts/activity.py (entries, the job watch), the factory's /activity routes and the
workbench's narrator (scripts/workbench/narrate.py)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from activity import Activity, JobWatch  # noqa: E402
from workbench.narrate import Narrator, detail_of, glb_of  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_put_merges_and_since_returns_new_copies(tmp_path: Path) -> None:
    clock = Clock()
    a = Activity(tmp_path / "a.jsonl", clock=clock)
    a.put("x", source="claude", title="Measuring sheep", state="running", stem="sheep")
    seq = a.since()["seq"]
    clock.t += 2.5
    done = a.put("x", state="ok", detail="head 3% off centre")
    assert done["title"] == "Measuring sheep" and done["ms"] == 2500
    fresh = a.since(seq)["entries"]
    assert [e["id"] for e in fresh] == ["x"] and fresh[0]["state"] == "ok"
    assert a.since(done["seq"])["entries"] == []


def test_bad_state_and_id_refused(tmp_path: Path) -> None:
    a = Activity(None)
    with pytest.raises(ValueError):
        a.put("x", state="weird")
    with pytest.raises(ValueError):
        a.put("", state="ok")


def test_restart_reads_back_and_ends_running(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    a = Activity(path)
    a.put("done", state="ok", title="Measured")
    a.put("busy", state="running", title="Baking")
    b = Activity(path)
    got = {e["id"]: e for e in b.since()["entries"]}
    assert got["done"]["state"] == "ok"
    assert got["busy"]["state"] == "cancelled"
    assert b.put("new", state="ok")["seq"] > max(e["seq"] for e in got.values())


def test_job_watch_stages_models_and_end(tmp_path: Path) -> None:
    folder = tmp_path / "out" / "sheep"
    folder.mkdir(parents=True)
    st: dict = {"job_id": "j1", "stem": "sheep", "kind": "generate", "state": "running", "started": 1.0,
                "progress": {"stage": "shape", "stage_index": 0, "stage_count": 3,
                             "stages": [{"key": "shape", "label": "Shape"}]}}
    a = Activity(None)
    w = JobWatch(a, lambda: st, tmp_path)
    w.tick()
    e = a.get("job-j1")
    assert e["state"] == "running" and e["title"] == "Building sheep" and e["detail"] == "Shape (1 of 3)"
    (folder / "mesh.glb").write_bytes(b"x")
    w.tick()
    assert a.get("job-j1")["glb"] == "sheep/mesh.glb"
    seq = a.since()["seq"]
    w.tick()  # nothing new: no new copy
    assert a.since(seq)["entries"] == []
    st.update(state="ok", finished=31.0)
    w.tick()
    e = a.get("job-j1")
    assert e["state"] == "ok" and e["ms"] == 30000
    seq = a.since()["seq"]
    w.tick()  # an ended job is written once
    assert a.since(seq)["entries"] == []


def test_job_watch_shows_the_drawing_as_it_forms(tmp_path: Path) -> None:
    flat = tmp_path / "out" / "sheep" / "flat"
    flat.mkdir(parents=True)
    st = {"job_id": "d1", "stem": "sheep", "kind": "flat2d", "state": "running"}
    a = Activity(None)
    w = JobWatch(a, lambda: st, tmp_path)
    w.tick()
    assert "image" not in a.get("job-d1") and a.get("job-d1")["title"] == "Drawing sheep"
    (flat / "drawing.jpg").write_bytes(b"x")
    w.tick()
    assert a.get("job-d1")["image"] == "sheep/flat/drawing.jpg"
    (flat / "sheet.png").write_bytes(b"x")
    os.utime(flat / "sheet.png", (5e9, 5e9))
    w.tick()
    assert a.get("job-d1")["image"] == "sheep/flat/sheet.png"


def test_factory_routes(tmp_path: Path) -> None:
    import factory_http

    f = factory_http.Factory(tmp_path, generate_runner=lambda spec: 0)
    status, body, _ = factory_http.dispatch(f, "POST", "/activity", {}, json.dumps(
        {"id": "wb-1", "source": "claude", "state": "running", "title": "Looking at sheep"}).encode())
    assert status == 200 and json.loads(body)["entry"]["seq"] == 1
    status, body, _ = factory_http.dispatch(f, "GET", "/activity", {"since": ["0"]}, b"")
    assert status == 200 and json.loads(body)["entries"][0]["title"] == "Looking at sheep"
    status, _, _ = factory_http.dispatch(f, "POST", "/activity", {}, b'{"id": "x", "state": "nope"}')
    assert status == 400
    status, _, _ = factory_http.dispatch(f, "GET", "/activity", {"since": ["x"]}, b"")
    assert status == 400


def test_narrator_lines(tmp_path: Path) -> None:
    sent: list[dict] = []
    n = Narrator(tmp_path, sent.append)
    png = tmp_path / "sheet.png"
    png.write_bytes(b"png")
    eid = n.start("look", {"stem": "sheep"})
    n.end(eid, "look", {"stem": "sheep"}, {"glb": "out/sheep/psx.glb", "png": str(png)})
    assert sent[0]["state"] == "running" and sent[0]["title"] == "Looking at sheep"
    assert sent[1]["title"] == "Looked at sheep" and "glb" not in sent[1]  # a look only reads the model
    assert (tmp_path / "out" / sent[1]["image"]).read_bytes() == b"png"
    eid = n.start("psx", {"stem": "sheep"})
    n.end(eid, "psx", {"stem": "sheep"}, {"glb": "out/sheep/psx.glb", "take": {"ok": True, "id": "t3"}})
    assert sent[-1]["glb"] == "sheep/psx.glb" and sent[-1]["detail"] == "kept as version t3"
    eid = n.start("bake_photo", {"stem": "sheep"})
    n.end(eid, "bake_photo", {"stem": "sheep"}, None, "no photo")
    assert sent[-1]["state"] == "error" and sent[-1]["error"] == "no photo"
    assert n.start("job_status", {}) is None


def test_narrator_show_carries_the_stage_and_the_compare(tmp_path: Path) -> None:
    sent: list[dict] = []
    n = Narrator(tmp_path, sent.append)
    eid = n.start("show", {"stem": "can"})
    n.end(eid, "show", {"stem": "can"}, {"stem": "can", "view": "mesh", "compare": "t4", "glb": "out/can/psx.glb"})
    assert sent[-1]["title"] == "Showed Stephen can"
    assert (sent[-1]["view"], sent[-1]["compare"], sent[-1]["glb"]) == ("mesh", "t4", "can/psx.glb")
    a = Activity(tmp_path / "a.jsonl")
    a.put("x", source="claude", tool="show", stem="can", state="ok", view="rig", compare="previous")
    got = a.since(0)["entries"][-1]
    assert (got["view"], got["compare"]) == ("rig", "previous")


def test_narrator_survives_a_dead_feed(tmp_path: Path) -> None:
    def boom(entry: dict) -> None:
        raise OSError("down")

    n = Narrator(tmp_path, boom)
    n.end(n.start("measure", {"stem": "s"}), "measure", {"stem": "s"}, {})


def test_details() -> None:
    line = detail_of("measure", {"head_offset_pct": 26.9, "head_yaw_deg": -46.2, "mirror_iou": 0.7849, "tris": 1169})
    assert line == "head 26.9% off centre · turned -46° · symmetry 0.78 · 1,169 triangles"
    assert detail_of("measure", {"circularity": 0.94}) == "roundness 0.94"
    assert glb_of({"path": "out/bale/workbench/clay.glb"}) == "bale/workbench/clay.glb"
    assert glb_of({"glb": "/abs/x.glb"}) is None


def test_job_watch_skips_a_job_that_ended_before_it_looked(tmp_path: Path) -> None:
    a = Activity(None)
    w = JobWatch(a, lambda: {"job_id": "old", "stem": "s", "state": "ok"}, tmp_path)
    w.tick()
    w.tick()
    assert a.since()["entries"] == []
