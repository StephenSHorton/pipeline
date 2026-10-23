"""Each workbench tool call as a line in the app's activity feed (the factory's POST /activity, scripts/activity.py).

The call is posted when it starts ("Measuring mlg-sheep") and again when it ends, with what it found or made: the
numbers a measure gave, the model a write produced (`glb`, which the app's viewer loads as it lands), the look sheet
(copied to out/_logs/activity/ so the app can show it) and the version it was kept as. Posting is best effort: a
factory that is down or slow never holds up or fails the tool.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

# tool -> (while it runs, once it is done), each followed by the asset's name.
TITLES: dict[str, tuple[str, str]] = {
    "asset_info": ("Reading the files of", "Read the files of"),
    "look": ("Looking at", "Looked at"),
    "measure": ("Measuring", "Measured"),
    "compare_reference": ("Comparing with the photo:", "Compared with the photo:"),
    "clay_get": ("Reading the clay of", "Read the clay of"),
    "clay_set": ("Shaping the clay of", "Shaped the clay of"),
    "clay_edit": ("Changing the clay of", "Changed the clay of"),
    "clay_mesh": ("Turning the clay into a mesh:", "Turned the clay into a mesh:"),
    "mesh_edit": ("Editing the mesh of", "Edited the mesh of"),
    "bake_photo": ("Baking the photo onto", "Baked the photo onto"),
    "psx": ("Making the PS1 model of", "Made the PS1 model of"),
    "draw_views": ("Starting the four-view drawing of", "Started the four-view drawing of"),
    "rig": ("Starting the rig of", "Started the rig of"),
    "rig_review": ("Reviewing the rig of", "Reviewed the rig of"),
    "playbook": ("Reading the playbook", "Read the playbook"),
    "generate": ("Starting the 3D build of", "Started the 3D build of"),
    "cancel": ("Stopping the job", "Stopped the job"),
    "pick_take": ("Going back to a version of", "Went back to a version of"),
}
# Polling is not an action: job_status runs every few seconds while a job goes and would bury the feed.
QUIET = frozenset({"job_status"})


def _num(v: Any, fmt: str) -> str | None:
    return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def detail_of(tool: str, out: dict[str, Any]) -> str:
    """One plain line about the result."""
    bits: list[str] = []
    if tool == "measure":
        off = _num(out.get("head_offset_pct"), ".1f")
        if off is not None:
            bits.append(f"head {off}% off centre")
        yaw = _num(out.get("head_yaw_deg"), ".0f")
        if yaw is not None:
            bits.append(f"turned {yaw}°")
        sym = _num(out.get("mirror_iou"), ".2f")
        if sym is not None:
            bits.append(f"symmetry {sym}")
        circ = _num(out.get("circularity"), ".2f")
        if circ is not None and out.get("head_offset_pct") is None:
            bits.append(f"roundness {circ}")
        tris = out.get("tris")
        if isinstance(tris, int):
            bits.append(f"{tris:,} triangles")
    elif tool == "compare_reference":
        iou = _num(out.get("iou"), ".2f")
        if iou is not None:
            bits.append(f"silhouette match {iou}")
    elif tool in ("clay_set", "clay_edit", "clay_get"):
        blobs = out.get("blobs")
        if isinstance(blobs, list):
            bits.append(f"{len(blobs)} blob{'s' if len(blobs) != 1 else ''}")
    elif tool == "rig_review":
        clips = out.get("clips") if isinstance(out.get("clips"), dict) else {}
        worst = max(clips, key=lambda c: clips[c].get("over", 0), default=None)
        if worst and clips[worst].get("over"):
            bits.append(f"{worst}: {clips[worst]['over']} edges past {out.get('threshold', 2):g}x")
        elif clips:
            bits.append("no edge past the threshold")
        bad = [lg["root"] for lg in (out.get("pivots") or {}).get("legs") or [] if not lg.get("ok", True)]
        if bad:
            bits.append(f"pivot off its leg: {', '.join(bad)}")
        flagged = out.get("flagged")
        if isinstance(flagged, list) and flagged:
            bits.append(f"look at {', '.join(flagged[:3])}")
    elif tool in ("draw_views", "generate", "rig") and out.get("job_id"):
        bits.append(f"job {out['job_id']}")
    elif tool == "pick_take" and isinstance(out.get("take"), str):
        bits.append(f"version {out['take']}")
    take = out.get("take")
    if isinstance(take, dict) and take.get("ok") and take.get("id"):
        bits.append(f"kept as version {take['id']}")
    return " · ".join(bits)


def glb_of(out: dict[str, Any]) -> str | None:
    """The model a write made, out/-relative, for the viewer (a look's or a measure's glb is only what it read)."""
    for key in ("glb", "path"):
        v = out.get(key)
        if isinstance(v, str) and v.startswith("out/") and v.endswith(".glb"):
            return v[len("out/"):]
    return None


WRITES = frozenset({"clay_mesh", "mesh_edit", "bake_photo", "psx"})


class Narrator:
    def __init__(self, root: Path, post: Callable[[dict[str, Any]], Any]) -> None:
        self.root = Path(root)
        self.post = post

    def _send(self, entry: dict[str, Any]) -> None:
        try:
            self.post(entry)
        except Exception:  # noqa: BLE001 - the feed is never worth failing a tool over
            pass

    def start(self, tool: str, args: dict[str, Any]) -> str | None:
        if tool in QUIET:
            return None
        eid = f"wb-{uuid.uuid4().hex[:10]}"
        stem = args.get("stem") if isinstance(args.get("stem"), str) else ""
        if not stem and isinstance(args.get("glb"), str):
            parts = Path(args["glb"]).parts
            stem = parts[1] if len(parts) > 2 and parts[0] == "out" else ""
        running = TITLES.get(tool, (tool, tool))[0]
        self._send({"id": eid, "source": "claude", "tool": tool, "stem": stem, "state": "running",
                    "title": f"{running} {stem}".strip()})
        return eid

    def end(self, eid: str | None, tool: str, args: dict[str, Any], out: dict[str, Any] | None,
            error: str | None = None) -> None:
        if eid is None:
            return
        stem = args.get("stem") if isinstance(args.get("stem"), str) else ""
        done = TITLES.get(tool, (tool, tool))[1]
        entry: dict[str, Any] = {"id": eid, "title": f"{done} {stem}".strip()}
        if error is not None:
            entry.update(state="error", error=error, title=f"{TITLES.get(tool, (tool, tool))[0]} {stem}".strip())
            self._send(entry)
            return
        out = out or {}
        entry["state"] = "ok"
        detail = detail_of(tool, out)
        if detail:
            entry["detail"] = detail
        glb = glb_of(out) if tool in WRITES else None
        if glb:
            entry["glb"] = glb
        if out.get("job_id"):
            entry["job_id"] = out["job_id"]
        png = out.get("png")
        if isinstance(png, str) and png.endswith(".png"):
            shown = self._keep_picture(Path(png), eid)
            if shown:
                entry["image"] = shown
        self._send(entry)

    def _keep_picture(self, png: Path, eid: str) -> str | None:
        """A look sheet lives in a temp folder; the app can only show what is under out/."""
        src = png if png.is_absolute() else self.root / png
        dest = self.root / "out" / "_logs" / "activity" / f"{eid}.png"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        except OSError:
            return None
        self._prune(dest.parent)
        return f"_logs/activity/{dest.name}"

    @staticmethod
    def _prune(folder: Path, keep: int = 200) -> None:
        pics = sorted(folder.glob("wb-*.png"), key=lambda p: p.stat().st_mtime)
        for old in pics[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass
