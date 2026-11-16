#!/usr/bin/env python3
"""Where a re-rig would put the joints, without the re-rig: rig.py on a scratch copy of the asset, stopped before
the skin step.

    .venv/bin/python scripts/workbench/rig_preview.py mlg-sheep [--class quadruped] [--plan horse] [--joints hips,spine_5]

A re-rig that moves a body joint relearns the skin weights (skin-tokens, about 3 minutes on the GPU, one model at a
time). Its skeleton is decided before that, from the cached predicted skeleton, so it can be read in seconds: the
asset (its rig/ folder and source files, not takes/ or workbench/) is copied into a temp FACTORY_ROOT, rig.py runs
there with run_skin replaced by a stop (and the skeleton prediction allowed only from its cache: no model loads),
and the posed joints (mesh frame) and register's body report are printed as
JSON. Nothing in out/ changes. On the sheep this showed the shoulder anchor taken from the horse's offsets moved
spine_4 by 0.008 before a 3-minute re-rig was spent on it.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOINTS = ("hips", "spine_1", "spine_2", "spine_3", "spine_4", "spine_5", "spine_6", "head", "front_scapula_l",
          "front_humerus_l", "back_leg_pelvis_l", "back_leg_upper_l", "tail_1")


class _Stop(Exception):
    pass


def preview(stem: str, cls: str = "quadruped", plan: str = "horse", joints: tuple[str, ...] = JOINTS,
            root: Path = ROOT) -> dict:
    src = root / "out" / stem
    if not src.is_dir():
        raise SystemExit(f"no asset {stem!r} (no out/{stem}/)")
    with tempfile.TemporaryDirectory(prefix="rig_preview_") as tmp:
        shutil.copytree(src, Path(tmp) / "out" / stem, ignore=shutil.ignore_patterns("takes", "workbench"))
        os.environ["FACTORY_ROOT"] = tmp
        sys.path.insert(0, str(ROOT / "scripts"))
        import numpy as np
        import rig
        import rig_pred
        rig.OUT = Path(tmp) / "out"
        got: dict = {}

        def stop(mesh, skeleton, out, fit="none"):
            doc, _ = rig.read_glb(Path(skeleton))
            sk = rig.Skeleton(doc)
            w = sk.world({})
            pos = {sk.nodes[i].get("name"): w[i][0] for i in w}
            got["joints"] = {n: np.round(pos[n], 4).tolist() for n in joints if n in pos}
            raise _Stop

        register = rig_pred.register

        def reg(*a, **k):
            r = register(*a, **k)
            got["body"] = next((c for c in r["paths"] if c.get("kind") == "body"), None)
            got["trunk_anchors"] = r.get("trunk_anchors")
            return r

        predict = rig_pred.predict

        def cached_only(*a, **k):  # the preview never loads a model: a stale or missing prediction stops it
            class _NoRun:
                @staticmethod
                def run(*_a, **_k):
                    raise RuntimeError("no cached predicted skeleton for this mesh (rig it once first)")
            real, rig_pred.subprocess = rig_pred.subprocess, _NoRun
            try:
                return predict(*a, **k)
            finally:
                rig_pred.subprocess = real

        rig.run_skin, rig_pred.register, rig_pred.predict = stop, reg, cached_only
        try:
            with contextlib.redirect_stdout(sys.stderr):  # rig.py's notes; the answer is the return value
                rig.main(["--stem", stem, "--class", cls, "--plan", plan, "--render", "0", "--no-fbx"])
        except _Stop:
            pass
        finally:
            rig_pred.register, rig_pred.predict = register, predict
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stem")
    ap.add_argument("--class", dest="cls", default="quadruped")
    ap.add_argument("--plan", default="horse")
    ap.add_argument("--joints", default=",".join(JOINTS))
    ap.add_argument("--root", default=str(ROOT), help="the folder holding out/ (default: this repo)")
    a = ap.parse_args(argv)
    out = preview(a.stem, a.cls, a.plan, tuple(j for j in a.joints.split(",") if j), Path(a.root))
    print(json.dumps(out, indent=1, default=str))
    return 0 if out.get("joints") else 1


if __name__ == "__main__":
    raise SystemExit(main())
