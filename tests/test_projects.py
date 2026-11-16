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

    def test_one_project_per_asset(self) -> None:
        make_asset(self.root, "radio")
        projects.create(self.root, "World")
        projects.create(self.root, "Other")
        projects.add_asset(self.root, "world", "radio")
        projects.add_asset(self.root, "other", "radio")
        self.assertEqual(projects.load(self.root, "world")["assets"], [])
        self.assertEqual(projects.projects_for(self.root, "radio"), ["other"])
        self.assertEqual(projects.membership(self.root), {"radio": "other"})

    def test_missing_folder_is_kept_but_not_listed(self) -> None:
        make_asset(self.root, "radio")
        projects.create(self.root, "World")
        with self.assertRaises(ProjectError):
            projects.add_asset(self.root, "world", "../x", allow_missing=True)
        # A job about to make out/lamp/: the membership is written now and shows once the folder exists.
        projects.add_asset(self.root, "world", "lamp", allow_missing=True)
        projects.add_asset(self.root, "world", "radio")
        self.assertEqual(projects.load(self.root, "world")["assets"], ["lamp", "radio"])
        self.assertEqual(projects.index(self.root)[0]["assets"], ["radio"])
        self.assertEqual(projects.membership(self.root), {"lamp": "world", "radio": "world"})
        (self.root / "out" / "lamp").mkdir()
        self.assertEqual(projects.index(self.root)[0]["assets"], ["lamp", "radio"])

    def test_index_sorts_by_name_and_skips_bad_manifests(self) -> None:
        projects.create(self.root, "zeta")
        projects.create(self.root, "Alpha")
        (self.root / "out" / ".projects" / "broken").mkdir()
        (self.root / "out" / ".projects" / "broken" / "project.json").write_text("{nope", encoding="utf-8")
        rows = projects.index(self.root)
        self.assertEqual([r["id"] for r in rows], ["alpha", "zeta"])
        self.assertEqual(set(rows[0]), {"id", "name", "assets", "created", "updated"})
        self.assertEqual(projects.membership(self.root), {})

    def test_unique_slug_on_create(self) -> None:
        self.assertEqual(projects.create(self.root, "World", unique=True)["id"], "world")
        self.assertEqual(projects.create(self.root, "world", unique=True)["id"], "world-2")
        self.assertEqual(projects.create(self.root, "WORLD!", unique=True)["id"], "world-3")
        long = "a" * 41
        self.assertEqual(projects.create(self.root, long, unique=True)["id"], long)
        self.assertEqual(projects.create(self.root, long, unique=True)["id"], "a" * 39 + "-2")

    def test_rename_and_delete_project(self) -> None:
        make_asset(self.root, "radio")
        projects.create(self.root, "World")
        projects.add_asset(self.root, "world", "radio")
        renamed = projects.rename_project(self.root, "world", "  Rust   Belt ")
        self.assertEqual((renamed["id"], renamed["name"]), ("world", "Rust Belt"))
        with self.assertRaises(ProjectError):
            projects.rename_project(self.root, "world", " ")
        with self.assertRaises(projects.MissingProject):
            projects.rename_project(self.root, "nope", "x")
        self.assertEqual(projects.delete_project(self.root, "world"), "world")
        self.assertFalse((self.root / "out" / ".projects" / "world").exists())
        self.assertTrue((self.root / "out" / "radio" / "flat" / "parts.json").is_file())
        self.assertEqual(projects.membership(self.root), {})
        with self.assertRaises(projects.MissingProject):
            projects.delete_project(self.root, "world")

    def test_rename_and_drop_stem(self) -> None:
        projects.create(self.root, "World")
        projects.create(self.root, "Other")
        projects.add_asset(self.root, "world", "radio", allow_missing=True)
        projects.add_asset(self.root, "other", "lamp", allow_missing=True)
        # lamp's stale claim in Other gives way: the renamed asset keeps radio's project.
        projects.rename_stem(self.root, "radio", "lamp")
        self.assertEqual(projects.membership(self.root), {"lamp": "world"})
        self.assertEqual(projects.load(self.root, "other")["assets"], [])
        projects.drop_stem(self.root, "lamp")
        self.assertEqual(projects.membership(self.root), {})
        projects.drop_stem(self.root, "never-there")

    def test_concurrent_adds_keep_every_stem(self) -> None:
        import threading

        projects.create(self.root, "World")
        projects.create(self.root, "Other")
        stems = [f"asset-{i}" for i in range(40)]
        threads = [
            threading.Thread(target=projects.add_asset, args=(self.root, ("world", "other")[i % 2], s), kwargs={"allow_missing": True})
            for i, s in enumerate(stems)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        owner = projects.membership(self.root)
        self.assertEqual(sorted(owner), sorted(stems))
        self.assertTrue(all(owner[s] == ("world", "other")[i % 2] for i, s in enumerate(stems)))

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

    def test_any_name_makes_a_project(self) -> None:
        # the app's New project never fails on a name: accents fold, a name with nothing to make an id from gets
        # "project" (Stephen may name one in any language); the name stays as typed
        self.assertEqual(projects.create(self.root, "Café", unique=True)["id"], "cafe")
        p = projects.create(self.root, "牧場", unique=True)
        self.assertEqual((p["id"], p["name"]), ("project", "牧場"))
        self.assertEqual(projects.create(self.root, "!!!", unique=True)["id"], "project-2")

    def test_a_misshapen_manifest_is_skipped_not_fatal(self) -> None:
        make_asset(self.root, "hen")
        projects.create(self.root, "Farm")
        projects.add_asset(self.root, "farm", "hen")
        for i, bad in enumerate(("[]", '{"assets": null}', '{"assets": "hen"}')):
            d = self.root / "out" / ".projects" / f"bad-{i}"
            d.mkdir(parents=True)
            (d / "project.json").write_text(bad, encoding="utf-8")
            with self.assertRaises(ProjectError):
                projects.load(self.root, f"bad-{i}")
        self.assertEqual(projects.membership(self.root), {"hen": "farm"})
        self.assertEqual([r["id"] for r in projects.index(self.root)], ["farm"])

    def test_a_make_says_where_its_asset_goes(self) -> None:
        projects.create(self.root, "Farm")
        # a make that never ran left a claim on "cow" (no folder): the next make of "cow" says none, and it's gone
        projects.add_asset(self.root, "farm", "cow", allow_missing=True)
        projects.file_made(self.root, "cow", {"project": None})
        self.assertNotIn("cow", projects.load(self.root, "farm")["assets"])
        projects.file_made(self.root, "cow", {"project": "farm"})
        self.assertEqual(projects.load(self.root, "farm")["assets"], ["cow"])
        projects.file_made(self.root, "cow", {"prompt": "a cow"})  # the 3D build after the drawing: left alone
        self.assertEqual(projects.load(self.root, "farm")["assets"], ["cow"])
        projects.file_made(self.root, "cow", {"project": "gone"})  # a deleted project: logged, nothing raised
        self.assertEqual(projects.load(self.root, "farm")["assets"], ["cow"])

    def test_rejects_bad_ids(self) -> None:
        for bad in ("../x", "", "X Y"):
            with self.assertRaises(ProjectError):
                projects.load(self.root, bad)


if __name__ == "__main__":
    unittest.main()
