#!/usr/bin/env python3
"""Re-run an asset's review and rig check from scratch and print them in a few lines.

    .venv/bin/python scripts/pig_check.py [stem] [review|rig]      # default: mlg-pig, both

The review is workbench/review.py (pieces, tris, paint against the paint it was baked toward, head, mirror); the rig
check is workbench/rigeyes.review (the same numbers as the workbench's rig_review). Nothing is read from an earlier
run's JSON: both are computed now from out/<stem>/. Exit 1 when a part that ran has a failed check (the rig check
fails at level "fail"; "warn" is exit 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factory_env import repo_root, stem_dir  # noqa: E402
from workbench import review, rigeyes  # noqa: E402


def main() -> int:
    stem = sys.argv[1] if len(sys.argv) > 1 else "mlg-pig"
    part = sys.argv[2] if len(sys.argv) > 2 else "both"
    if part not in ("both", "review", "rig"):
        print("usage: pig_check.py [stem] [review|rig]", file=sys.stderr)
        return 2
    folder = stem_dir(repo_root(), stem)
    bad = False
    if part in ("both", "review"):
        rev = review.review(folder)
        print(f"REVIEW {stem}: {rev['verdict']}  ({Path(rev['model']).name})")
        for c in rev["checks"]:
            mark = {True: "pass", False: "FAIL", None: "info"}[c["ok"]]
            lim = "" if c["limit"] is None else f" (limit {c['limit']})"
            print(f"  {mark:4} {c['name']:12} {c['value']}{lim}")
        bad = bad or bool(rev["failed"])
    if part in ("both", "rig"):
        res = rigeyes.review(folder, rigeyes.THRESHOLD, None, None, False, None)
        chk = res["check"]
        print(f"RIG CHECK {stem}: {chk['level']}  {chk['line']}  (stale: {chk.get('stale')})")
        for p in chk["problems"]:
            print(f"  - {p}")
        print("  stretch p99 per clip: " + ", ".join(f"{k} {v['stretch_p99']}" for k, v in chk["clips"].items()))
        bad = bad or chk["level"] == "fail"
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
