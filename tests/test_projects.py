#!/usr/bin/env python3
"""Projects: membership, style, and the shared part library. No model weights."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import projects  # noqa: E402
from factory_env import taken_stems  # noqa: E402
from projects import ProjectError  # noqa: E402


def make_asset(root: Path, stem: str, pick: str | None = None) -> None:
    flat = root / "out" / stem / "flat"
    (flat / "parts").mkdir(parents=True)
    Image.new("RGBA", (12, 8), (200, 40, 40, 255)).save(flat / "parts" / "part-01.png")
    rows = [{"id": "part-01", "file": "parts/part-01.png", "name": "Left Arm", "role": "leg", "w": 12, "h": 8}]
    (flat / "parts.json").write_text(json.dumps({"parts": rows}), encoding="utf-8")
    if pick:
        flow = {"version": 1, "stages": {"flat": {"pick": pick, "takes": []}}}
        (root / "out" / stem / "flow.json").write_text(json.dumps(flow), encoding="utf-8")


class ProjectsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_create_list_and_style(self) -> None:
        p = projects.create(self.root, "Rust Belt", "worn steel, chunky silhouettes", ["FF8800", "#223344"])
        self.assertEqual(p["id"], "rust-belt")
        self.assertEqual(p["style"]["palette"], ["#ff8800", "#223344"])
        with self.assertRaises(ProjectError):
            projects.create(self.root, "rust belt")
        with self.assertRaises(ProjectError):
            projects.set_style(self.root, "rust-belt", palette=["orange"])
        projects.set_style(self.root, "rust-belt", prompt="  flat  shaded ")
        row = projects.list_projects(self.root)[0]
        self.assertEqual((row["id"], row["assets"], row["parts"]), ("rust-belt", 0, 0))
        line = projects.styled_prompt(projects.load(self.root, "rust-belt"), "a radio")
        self.assertEqual(line, "flat shaded. Palette: #ff8800, #223344. a radio")

    def test_project_folder_is_not_a_stem(self) -> None:
        make_asset(self.root, "radio")
        projects.create(self.root, "World")
        self.assertEqual(taken_stems(self.root), {"radio"})

    def test_membership(self) -> None:
        make_asset(self.root, "radio")
        projects.create(self.root, "World")
        with self.assertRaises(ProjectError):
            projects.add_asset(self.root, "world", "missing")
        projects.add_asset(self.root, "world", "radio")
        projects.add_asset(self.root, "world", "radio")
        self.assertEqual(projects.load(self.root, "world")["assets"], ["radio"])
        self.assertEqual(projects.projects_for(self.root, "radio"), ["world"])
        projects.remove_asset(self.root, "world", "radio")
        self.assertEqual(projects.projects_for(self.root, "radio"), [])

    def test_flat_part_into_library(self) -> None:
        make_asset(self.root, "mech", pick="t2")
        projects.create(self.root, "World")
        with self.assertRaises(ProjectError):
            projects.add_flat_part(self.root, "world", "mech", "part-01")
        projects.add_asset(self.root, "world", "mech")
        with self.assertRaises(ProjectError):
            projects.add_flat_part(self.root, "world", "mech", "part-09")
        meta = projects.add_flat_part(self.root, "world", "mech", "part-01", tags="arm, Arm, metal")
        self.assertEqual(meta["id"], "left-arm")
        self.assertEqual(meta["tags"], ["arm", "metal"])
        self.assertEqual(meta["from"], {"stem": "mech", "stage": "flat", "part": "part-01", "take": "t2"})
        part = self.root / "out" / ".projects" / "world" / "parts" / "left-arm"
        self.assertEqual(Image.open(part / "part.png").size, (12, 8))
        again = projects.add_flat_part(self.root, "world", "mech", "part-01")
        self.assertEqual(again["id"], "left-arm-2")
        projects.remove_part(self.root, "world", "left-arm")
        self.assertFalse(part.exists())
        self.assertEqual([p["id"] for p in projects.load(self.root, "world")["parts"]], ["left-arm-2"])

    def test_rejects_bad_ids(self) -> None:
        for bad in ("../x", "", "X Y"):
            with self.assertRaises(ProjectError):
                projects.load(self.root, bad)


if __name__ == "__main__":
    unittest.main()
