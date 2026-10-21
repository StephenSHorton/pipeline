"""The embedded runtime copy of scripts/ must match scripts/ exactly.

The desktop app ships desktop/internal/runtime/fs/scripts (filled by
desktop/frontend/scripts/copy-factory.mjs).  Anything that differs from
scripts/ is code the installed app runs but the repo does not test, so the
only allowed differences are the two Swift helpers that are built into the
app binary instead of being shipped as scripts.
"""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts"
DST = ROOT / "desktop" / "internal" / "runtime" / "fs" / "scripts"

KNOWN = {
    "Only in scripts: lift-subject.swift",
    "Only in scripts: preview-glb.swift",
}


def mirror_diff():
    out = subprocess.run(
        ["diff", "-rq", "-x", "__pycache__", "-x", ".*", "scripts", str(DST.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


@unittest.skipUnless(shutil.which("diff"), "needs diff")
class RuntimeMirror(unittest.TestCase):
    def test_mirror_matches_scripts(self):
        self.assertTrue(DST.is_dir(), f"missing {DST}")
        extra = [ln for ln in mirror_diff() if ln not in KNOWN]
        self.assertEqual(extra, [], "runtime mirror differs from scripts/; run "
                         "node desktop/frontend/scripts/copy-factory.mjs:\n" + "\n".join(extra))

    def test_plans_and_blender_scripts_ship(self):
        plans = sorted(p.name for p in (DST / "rig_plans").glob("*.json"))
        self.assertEqual(plans, sorted(p.name for p in (SRC / "rig_plans").glob("*.json")))
        self.assertIn("ground_bird.json", plans)
        for name in ("fbx_check.py", "heat_bind.py", "rig_render_bl.py"):
            self.assertTrue((DST / "blender" / name).is_file(), name)

    def test_copy_factory_keeps_json(self):
        js = (ROOT / "desktop" / "frontend" / "scripts" / "copy-factory.mjs").read_text()
        self.assertIn('".json"', js)


if __name__ == "__main__":
    unittest.main()
