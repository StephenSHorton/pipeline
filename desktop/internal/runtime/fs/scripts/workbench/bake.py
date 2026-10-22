#!/usr/bin/env python3
"""Bake the photo onto a mesh (retexture.py), fill what it missed (workbench/fill.py) and cut it down to a PS1 model
(psx_lowpoly.py), as subprocesses.

Both are the pipeline's own stages, run with the factory's python exactly as generate.sh runs them, so a workbench
bake is the bake an asset would get. They run in a child process because each holds a few hundred MB of samples
and textures that the long-lived MCP server should hand back when it is done.

psx always runs with its facing check off. The check asks a vision model (ollama) where the head is, and the
workbench never loads a model itself; the facing is the one the caller states.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from factory_env import factory_python

TIMEOUT = 900
SCRIPTS = Path(__file__).resolve().parents[1]  # the stages run from this checkout, whatever root the data is under


class BakeError(RuntimeError):
    pass


def end_on(view: Path, facing: str | None) -> bool:
    """Whether a view looks at one end of the subject (a front or back view): stated, else read from its name the
    way retexture reads it. Those are the views whose far end --mirror covers with the flipped photo."""
    if facing:
        return facing in ("camera", "away")
    stem = Path(view).stem.lower()
    if any(w in stem for w in ("handle", "side", "left", "right")):
        return False
    return "front" in stem or "back" in stem


def _run(root: Path, argv: list[str], env: dict[str, str] | None = None) -> str:
    try:
        proc = subprocess.run([str(factory_python(SCRIPTS.parent)), *argv], cwd=root, capture_output=True, text=True,
                              timeout=TIMEOUT, env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired:
        raise BakeError(f"{Path(argv[0]).name} took over {TIMEOUT} s") from None
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-12:])
        raise BakeError(f"{Path(argv[0]).name} exited {proc.returncode}: {tail}")
    return proc.stdout


def retexture(root: Path, glb: Path, out: Path, views: list[Path], first_facing: str | None = None,
              mirror: bool = False, erode: float | None = None, keep: float | None = None,
              first_wins: bool = False, mirror_ends: bool = False) -> dict:
    """retexture.py glb -o out --view ... Only the first view's facing can be stated; the others are read from
    their file names (front, back, left, right, side), as retexture does. mirror_ends (with mirror) passes
    --mirror-ends: a front-on or back-on view mirrors onto the opposite end, not through the z mid-plane."""
    argv = [str(SCRIPTS / "retexture.py"), str(glb), "-o", str(out)]
    for v in views:
        argv += ["--view", str(v)]
    if first_facing:
        argv += ["--first-facing", first_facing]
    if mirror:
        argv.append("--mirror")
        if mirror_ends:
            argv.append("--mirror-ends")
    if erode:
        argv += ["--erode", str(erode)]
    if keep is not None:
        argv += ["--keep", str(keep)]
    if first_wins and len(views) > 1:
        argv.append("--first-wins")
    log = _run(root, argv)
    if not out.is_file():
        raise BakeError("retexture wrote no GLB")
    info: dict = {"glb": str(out)}
    fit = out.with_suffix(".fit.json")
    if fit.is_file():
        try:
            info["fit"] = json.loads(fit.read_text(encoding="utf-8"))
        except ValueError:
            pass
    info["log_tail"] = "\n".join(log.strip().splitlines()[-6:])
    return info


def fill(root: Path, glb: Path, photo: Path, out: Path, keep: float | None = None) -> dict:
    """fill.py glb --photo photo -o out (glb's .cover.png from retexture). Returns its .fill.json report."""
    argv = [str(SCRIPTS / "workbench" / "fill.py"), str(glb), "-o", str(out), "--photo", str(photo),
            "--cover", str(glb.with_suffix(".cover.png"))]
    if keep is not None:
        argv += ["--keep", str(keep)]
    _run(root, argv)
    report = out.with_suffix(".fill.json")
    try:
        return json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BakeError("fill wrote no report") from None


def psx(root: Path, glb: Path, out_dir: Path, glb_out: Path, name: str, height: float, facing: str,
        faces: int | None = None, atlas: int | None = None) -> dict:
    """psx_lowpoly.py with its facing check off. Returns psx.json's report (the PSX_RESULT line)."""
    argv = [str(SCRIPTS / "psx_lowpoly.py"), str(glb), "--out", str(out_dir), "--name", name,
            "--height", str(height), "--facing", facing, "--glb", str(glb_out), "--facing-check", "off"]
    if faces:
        argv += ["--faces", str(faces)]
    if atlas:
        argv += ["--atlas", str(atlas)]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _run(root, argv, env={"PSX_FACING_CHECK": "off"})
    for line in reversed(log.splitlines()):
        if line.startswith("PSX_RESULT "):
            return json.loads(line[len("PSX_RESULT "):])
    raise BakeError("psx_lowpoly printed no PSX_RESULT")
