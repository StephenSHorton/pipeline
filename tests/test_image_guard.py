#!/usr/bin/env python3
"""No 3D without a picture: factory_env.require_picture, and the same message in every entry point."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import factory_env  # noqa: E402
import factory_http  # noqa: E402

MESSAGE = "3D needs a picture. Draw or add one first."


class RequirePictureTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def picture(self, name: str) -> Path:
        from PIL import Image

        path = self.dir / name
        fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[path.suffix.lower()]
        Image.new("RGB", (20, 10), (1, 2, 3)).save(path, fmt)
        return path

    def test_the_message_is_one_string(self) -> None:
        self.assertEqual(factory_env.NO_PICTURE, MESSAGE)
        go = (ROOT / "desktop" / "internal" / "factory" / "generate.go").read_text(encoding="utf-8")
        self.assertIn(f'const NoPicture = "{MESSAGE}"', go)
        self.assertIn("errors.New(NoPicture)", go)

    def test_the_suffixes_are_the_upload_suffixes(self) -> None:
        self.assertEqual(set(factory_env.UPLOAD_EXTS), {".jpg", ".jpeg", ".png", ".webp"})
        self.assertIs(factory_http.UPLOAD_EXTS, factory_env.UPLOAD_EXTS)

    def test_real_pictures_pass(self) -> None:
        for name in ("a.jpg", "b.jpeg", "c.png", "d.webp", "E.PNG"):
            with self.subTest(name=name):
                self.assertEqual(factory_env.require_picture(self.picture(name)), self.dir / name)

    def test_everything_else_is_refused(self) -> None:
        text = self.dir / "notes.jpg"
        text.write_text("not a picture\n")
        truncated = self.dir / "cut.png"
        truncated.write_bytes(self.picture("whole.png").read_bytes()[:30])
        gif = self.dir / "anim.gif"
        from PIL import Image

        Image.new("RGB", (4, 4)).save(gif, "GIF")
        mesh = self.dir / "mesh.glb"
        mesh.write_bytes(b"glTF\x02\x00\x00\x00")
        for bad in (None, "", self.dir / "missing.png", text, truncated, gif, mesh, self.dir):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as caught:
                    factory_env.require_picture(bad)
                self.assertEqual(str(caught.exception), MESSAGE)


if __name__ == "__main__":
    unittest.main()
