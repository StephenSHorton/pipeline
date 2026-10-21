#!/usr/bin/env python3
"""Color-key split, saved sheet, and stub wav. No model weights."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import factory_http  # noqa: E402
from flat2d.atlas import pack_request, save_action, unmix  # noqa: E402
from flat2d.optional import (  # noqa: E402
    FlatUnavailable,
    parse_llm_json,
    render_sound,
    speak,
    still_prompt,
)
from flat2d.parts import MIN_AREA, TOL, ground_key, reading_order, see_through, split_image  # noqa: E402
from flat2d.sheet import (  # noqa: E402
    flat_dir,
    read_json,
    sfx_mode,
    sound_file,
    sound_prompt,
    still_backend,
    unique_ids,
    voice_file,
    write_json,
)
from flat2d.steps import apply_steps  # noqa: E402
from flat2d.store import FlatError, load_sheet, save_sheet, save_source, split_source  # noqa: E402
from flat2d.turbo import TURBO_SIGMAS, read_grid  # noqa: E402


def paint(w: int, h: int, blobs: list[tuple[int, int, int, int, tuple[int, int, int]]]) -> Image.Image:
    im = Image.new("RGB", (w, h), (255, 255, 255))
    px = im.load()
    for x0, y0, x1, y1, color in blobs:
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = color
    return im


def opaque_of(crop: Image.Image, color: tuple[int, int, int]) -> int:
    px = crop.load()
    w, h = crop.size
    count = 0
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a and (r, g, b) == color:
                count += 1
    return count


def png_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def http_json(url: str, method: str = "GET", data: dict | None = None) -> tuple[int, dict]:
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


class Flat2DTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = {
            "FLAT2D_SFX_DIR": os.environ.get("FLAT2D_SFX_DIR"),
            "FLAT2D_SAM_DIR": os.environ.get("FLAT2D_SAM_DIR"),
            "FLAT2D_SFX_CMD": os.environ.get("FLAT2D_SFX_CMD"),
            "FLAT2D_LIFT": os.environ.get("FLAT2D_LIFT"),
        }
        os.environ.pop("FLAT2D_SFX_CMD", None)
        empty = Path(tempfile.mkdtemp())
        os.environ["FLAT2D_SFX_DIR"] = str(empty / "sfx")
        os.environ["FLAT2D_SAM_DIR"] = str(empty / "sam")
        os.environ["FLAT2D_LIFT"] = str(empty / "lift-subject")  # the key, not Apple Vision

    def tearDown(self) -> None:
        for key, val in self._env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_split_separated_and_touching(self) -> None:
        separated = paint(
            48,
            32,
            [
                (4, 4, 14, 14, (180, 40, 40)),
                (28, 16, 42, 28, (40, 40, 40)),
            ],
        )
        crops = split_image(separated, min_area=20)
        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0].getpixel((0, 0))[3], 0)

        touching = paint(
            40,
            20,
            [
                (4, 4, 12, 16, (180, 40, 40)),
                (12, 4, 22, 16, (40, 40, 160)),
            ],
        )
        self.assertEqual(len(split_image(touching, min_area=20)), 1)

        red = (180, 40, 40)
        blue = (40, 40, 160)
        close = paint(
            40,
            20,
            [
                (2, 4, 10, 16, red),
                (12, 4, 22, 16, blue),
            ],
        )
        close_crops = split_image(close, min_area=20)
        self.assertEqual(len(close_crops), 2)
        self.assertEqual(sorted(opaque_of(c, red) for c in close_crops), [0, 8 * 12])
        self.assertEqual(sorted(opaque_of(c, blue) for c in close_crops), [0, 10 * 12])

        hull = paint(
            40,
            32,
            [
                (4, 4, 9, 25, red),
                (4, 4, 21, 9, red),
                (4, 20, 21, 25, red),
                (12, 12, 18, 18, blue),
            ],
        )
        hull_crops = split_image(hull, min_area=20)
        self.assertEqual(len(hull_crops), 2)
        body, gun = hull_crops
        self.assertEqual(opaque_of(body, blue), 0)
        self.assertEqual(opaque_of(gun, red), 0)
        self.assertEqual(opaque_of(gun, blue), 36)

    def test_one_shape_is_not_saved_as_a_piece(self) -> None:
        im = paint(32, 24, [(4, 4, 20, 18, (180, 40, 40))])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            save_source(root, "unit", buf.getvalue())
            split = split_source(root, "unit", "legs, body")
            self.assertTrue(split["whole"])
            self.assertFalse(split["keyed"])
            self.assertEqual(split["reason"], "whole")
            self.assertEqual(split["parts"], [])
            self.assertFalse((root / "out" / "unit" / "flat" / "parts" / "part-01.png").exists())

    def test_split_names_follow_the_prompt(self) -> None:
        im = paint(
            48,
            32,
            [
                (4, 4, 14, 14, (180, 40, 40)),
                (28, 16, 42, 28, (40, 40, 40)),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            save_source(root, "unit", buf.getvalue())
            split = split_source(root, "unit", "Hull, Gun")
            self.assertFalse(split["whole"])
            self.assertFalse(split["keyed"])
            self.assertEqual(split["reason"], "")
            self.assertEqual([row["name"] for row in split["parts"]], ["Hull", "Gun"])

    def _save_and_split(self, im: Image.Image, key: list[int] | None = None):
        root_cm = tempfile.TemporaryDirectory()
        tmp = root_cm.__enter__()
        self.addCleanup(root_cm.cleanup)
        root = Path(tmp)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        save_source(root, "unit", buf.getvalue())
        if key is not None:
            write_json(flat_dir(root, "unit") / "key.json", {"rgb": key})
        split = split_source(root, "unit")
        part = root / "out" / "unit" / "flat" / "parts" / "part-01.png"
        return root, split, part

    def test_key_color_with_no_off_key_pixel_is_keyed(self) -> None:
        key = (255, 0, 255)
        flat = Image.new("RGB", (32, 24), key)
        blob = flat.copy()
        px = blob.load()
        near = (240, 0, 240)
        self.assertLessEqual(sum(abs(a - b) for a, b in zip(near, key)), TOL)
        for y in range(6, 18):
            for x in range(6, 20):
                px[x, y] = near
        for im in (flat, blob):
            _root, split, part = self._save_and_split(im, [255, 0, 255])
            self.assertFalse(split["whole"])
            self.assertTrue(split["keyed"])
            self.assertEqual(split["reason"], "keyed")
            self.assertEqual(split["parts"], [])
            self.assertEqual(split["method"], "key")
            self.assertFalse(part.exists())

    def test_specks_under_min_area_are_not_keyed(self) -> None:
        im = Image.new("RGB", (32, 24), (255, 0, 255))
        px = im.load()
        speck_n = 8
        self.assertLess(speck_n, MIN_AREA)
        for i in range(speck_n):
            px[4 + i, 8] = (0, 0, 0)
        _root, split, part = self._save_and_split(im, [255, 0, 255])
        self.assertFalse(split["whole"])
        self.assertFalse(split["keyed"])
        self.assertEqual(split["reason"], "specks")
        self.assertEqual(split["parts"], [])
        self.assertEqual(split["method"], "key")
        self.assertFalse(part.exists())

    def test_transparent_without_key_is_none(self) -> None:
        im = Image.new("RGBA", (32, 24), (0, 0, 0, 0))
        root, split, part = self._save_and_split(im, None)
        self.assertFalse((flat_dir(root, "unit") / "key.json").is_file())
        self.assertFalse(split["whole"])
        self.assertFalse(split["keyed"])
        self.assertEqual(split["reason"], "none")
        self.assertEqual(split["parts"], [])
        self.assertEqual(split["method"], "key")
        self.assertFalse(part.exists())

    def test_blob_on_another_color_is_at_least_one_crop(self) -> None:
        im = Image.new("RGB", (40, 30), (255, 0, 255))
        px = im.load()
        for y in range(8, 22):
            for x in range(8, 22):
                px[x, y] = (40, 40, 160)
        _root, split, _part = self._save_and_split(im, [255, 0, 255])
        self.assertTrue(split["whole"] or len(split["parts"]) >= 1)
        self.assertNotIn(split["reason"], ("keyed", "specks", "none"))

    def test_sound_prompt_names_the_wav(self) -> None:
        self.assertEqual(
            sound_prompt("metal foot plants"),
            "short game sound effect, one shot, no music: metal foot plants",
        )
        self.assertEqual(sound_file("front-leg"), "front-leg.wav")
        with self.assertRaises(ValueError):
            sound_prompt("   ")
        with self.assertRaises(ValueError):
            sound_file("../nope")

    def test_sheet_and_stub_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "out" / "unit" / "flat"
            flat.mkdir(parents=True)
            paint(
                48,
                32,
                [
                    (4, 4, 14, 14, (180, 40, 40)),
                    (28, 16, 42, 28, (40, 40, 40)),
                ],
            ).save(flat / "source.png")
            split = split_source(root, "unit")
            self.assertEqual(split["method"], "key")
            self.assertEqual(len(split["parts"]), 2)
            saved = save_sheet(
                root,
                "unit",
                [
                    {
                        "id": "part-01",
                        "file": "parts/part-01.png",
                        "name": "Hull",
                        "role": "hull",
                        "motion": "sits on the pad",
                    },
                    {
                        "id": "part-02",
                        "file": "parts/part-02.png",
                        "name": "Gun",
                        "role": "weapon",
                        "projectile": "small energy bolt",
                    },
                ],
                "",
            )
            gun = next(p for p in saved["parts"] if p["role"] == "weapon")
            heard = render_sound(root, "unit", gun)
            wav = root / "out" / "unit" / "flat" / heard["sound"]
            self.assertTrue(wav.is_file())
            self.assertGreater(wav.stat().st_size, 44)
            self.assertTrue(heard["sound"].endswith(".wav"))
            mech = read_json(flat / "mechanics.json")
            recorded = next(p for p in mech["parts"] if p["id"] == gun["id"])
            self.assertEqual(recorded["sound"], heard["sound"])
            self.assertEqual(recorded["projectile"], "small energy bolt")
            self.assertEqual(recorded["role"], "weapon")

    def test_unique_ids_keep_fallback(self) -> None:
        rows = unique_ids([{"name": "!!!", "file": "parts/part-01.png"}])
        self.assertEqual(rows[0]["id"], "part-01")

    def test_explicit_key_ignores_red_corners(self) -> None:
        im = Image.new("RGB", (40, 30), (255, 255, 255))
        px = im.load()
        for y in range(30):
            for x in range(40):
                if x < 2 or y < 2 or x >= 38 or y >= 28:
                    px[x, y] = (180, 40, 40)
        for y in range(12, 20):
            for x in range(16, 24):
                px[x, y] = (40, 40, 160)
        corner = split_image(im, min_area=20)
        keyed = split_image(im, min_area=20, key=(255, 255, 255))
        self.assertTrue(any(opaque_of(c, (255, 255, 255)) > 0 for c in corner))
        self.assertTrue(all(opaque_of(c, (255, 255, 255)) == 0 for c in keyed))
        self.assertEqual(len(keyed), 2)

    def test_drop_clears_key_and_split_reads_it(self) -> None:
        im = Image.new("RGB", (40, 30), (255, 255, 255))
        px = im.load()
        for y in range(30):
            for x in range(40):
                if x < 2 or y < 2 or x >= 38 or y >= 28:
                    px[x, y] = (180, 40, 40)
        for y in range(12, 20):
            for x in range(16, 24):
                px[x, y] = (40, 40, 160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            save_source(root, "unit", buf.getvalue())
            write_json(flat_dir(root, "unit") / "key.json", {"rgb": [255, 0, 255]})
            save_source(root, "unit", buf.getvalue())
            self.assertFalse((flat_dir(root, "unit") / "key.json").is_file())
            write_json(flat_dir(root, "unit") / "key.json", {"rgb": [255, 255, 255]})
            split = split_source(root, "unit")
            self.assertEqual(split["method"], "key")
            self.assertEqual(len(split["parts"]), 2)
            for row in split["parts"]:
                crop = Image.open(flat_dir(root, "unit") / row["file"])
                self.assertEqual(opaque_of(crop, (255, 255, 255)), 0)

    def test_sam_failure_is_reported(self) -> None:
        from unittest.mock import patch

        sam = Path(os.environ["FLAT2D_SAM_DIR"])
        sam.mkdir()
        (sam / "model.pt").write_bytes(b"x")
        im = paint(32, 24, [(4, 4, 14, 14, (180, 40, 40))])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            save_source(root, "unit", buf.getvalue())
            with patch("flat2d.store.split_sam", side_effect=RuntimeError("bad config")):
                with self.assertRaises(FlatError) as caught:
                    split_source(root, "unit")
            self.assertIn("sam2 failed", str(caught.exception))
            self.assertIn("bad config", str(caught.exception))

    def test_sfx_mode_and_device(self) -> None:
        self.assertEqual(sfx_mode(), "stub")
        self.assertIn("#FF00FF", still_prompt("radio"))
        self.assertTrue(still_prompt("radio").startswith("radio. A single object only"))
        self.assertIn("No shadow", still_prompt("radio"))
        folder = Path(os.environ["FLAT2D_SFX_DIR"])
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"x")
        self.assertEqual(sfx_mode(), "no-runner")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FlatUnavailable):
                render_sound(Path(tmp), "unit", {"id": "gun", "projectile": "bolt"})
        os.environ["FLAT2D_SFX_CMD"] = "echo"
        self.assertEqual(sfx_mode(), "ready")

    def test_pack_unmixes_and_writes_a_strip(self) -> None:
        im = Image.new("RGB", (20, 16), (255, 0, 255))
        px = im.load()
        for y in range(4, 12):
            for x in range(4, 12):
                px[x, y] = (180, 40, 40)
        px[3, 8] = (240, 10, 240)
        cut = unmix(im)
        self.assertEqual(cut.getpixel((0, 0))[3], 0)
        self.assertEqual(cut.getpixel((6, 6))[3], 255)
        self.assertGreater(cut.getpixel((3, 8))[3], 0)
        self.assertLess(cut.getpixel((3, 8))[3], 255)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        raw = base64.b64encode(buf.getvalue()).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packed = pack_request(root, "unit", {"action": "walk", "fps": 8, "frames": [raw, raw]})
            self.assertEqual(packed["frames"], 2)
            with Image.open(root / "out" / "unit" / "flat" / packed["sheet"]) as strip:
                self.assertEqual(strip.size[0], 40)
            manifest = read_json(root / "out" / "unit" / "flat" / "sheets" / "manifest.json")
            self.assertEqual(manifest["actions"]["walk"]["fps"], 8)

    def test_pack_ignores_a_corner_and_refuses_a_gradient(self) -> None:
        im = Image.new("RGB", (24, 20), (255, 0, 255))
        px = im.load()
        for y in range(6, 14):
            for x in range(6, 14):
                px[x, y] = (180, 40, 40)
        px[0, 0] = (180, 40, 40)
        px[2, 2] = (40, 200, 40)
        cut = unmix(im)
        self.assertEqual(cut.getpixel((0, 0))[3], 0)
        self.assertEqual(cut.getpixel((8, 8))[3], 255)
        self.assertEqual(cut.getpixel((2, 2))[3], 0)
        grad = Image.new("RGB", (32, 32), (255, 0, 255))
        gp = grad.load()
        for y in range(32):
            for x in range(32):
                gp[x, y] = (255 - y * 4, y * 2, 255 - x * 4)
        with self.assertRaises(FlatError) as caught:
            unmix(grad)
        self.assertIn("not flat", str(caught.exception))

    def test_line_and_fenced_draft(self) -> None:
        self.assertEqual(voice_file("front-leg"), "front-leg.wav")
        with self.assertRaises(ValueError):
            voice_file("../nope")
        rows = unique_ids(
            [
                {
                    "id": "part-01",
                    "file": "parts/part-01.png",
                    "name": "Gun",
                    "line": "Hold fire.",
                    "speaker": "am_adam",
                    "voice": "voices/gun.wav",
                }
            ]
        )
        self.assertEqual(rows[0]["line"], "Hold fire.")
        self.assertEqual(rows[0]["speaker"], "am_adam")
        self.assertEqual(rows[0]["voice"], "voices/gun.wav")
        self.assertEqual(unique_ids([{"name": "Gun", "speaker": "nope"}])[0]["speaker"], "af_heart")
        parsed = parse_llm_json('```json\n{"parts": [{"id": "gun", "line": "Now."}]}\n```')
        self.assertEqual(parsed["parts"][0]["line"], "Now.")

    def test_speak_writes_a_wav(self) -> None:
        from unittest.mock import patch

        import numpy as np

        class _Result:
            audio = np.array([0.0, 0.25, -0.25, 0.1], dtype=np.float32)
            sample_rate = 24000

        class _Model:
            def generate(self, text, voice="af_heart", speed=1.0, lang_code="a"):
                self.text = text
                self.voice = voice
                yield _Result()

        model = _Model()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("flat2d.optional.voice_ready", return_value=True), patch(
                "flat2d.optional.kokoro_model", return_value=model
            ):
                with self.assertRaises(FlatUnavailable):
                    speak(root, "unit", {"id": "gun", "line": "   "})
                heard = speak(
                    root,
                    "unit",
                    {"id": "gun", "line": "Hold fire.", "speaker": "am_adam"},
                )
                self.assertEqual(model.text, "Hold fire.")
                self.assertEqual(model.voice, "am_adam")
                mech = read_json(flat_dir(root, "unit") / "mechanics.json")
                self.assertEqual(mech["parts"][0]["line"], "Hold fire.")
                from flat2d.routes import handle

                code, raw, _ctype = handle(
                    root,
                    "POST",
                    "/2d/speak",
                    json.dumps(
                        {"stem": "unit", "part": {"id": "gun", "line": "Again.", "speaker": "af_heart"}}
                    ).encode(),
                )
                posted = json.loads(raw)
                self.assertEqual(code, 200, posted)
                self.assertEqual(posted["line"], "Again.")
            wav = root / "out" / "unit" / "flat" / heard["voice"]
            self.assertTrue(wav.is_file())
            self.assertGreater(wav.stat().st_size, 44)
            mech = read_json(flat_dir(root, "unit") / "mechanics.json")
            self.assertEqual(mech["parts"][0]["voice"], heard["voice"])
            self.assertEqual(mech["parts"][0]["line"], "Again.")
            from flat2d.routes import handle

            code, raw, ctype = handle(
                root,
                "GET",
                "/2d/unit/voices/" + heard["voice"].split("/")[-1],
                b"",
            )
            self.assertEqual(code, 200)
            self.assertEqual(ctype, "audio/wav")
            self.assertGreater(len(raw), 44)

    def test_factory_http_drop_split_save_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factory = factory_http.Factory(root)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), factory_http.make_handler(factory))
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{port}"
            try:
                runs = []
                for _ in range(2):
                    separated = paint(
                        48,
                        32,
                        [
                            (4, 4, 14, 14, (180, 40, 40)),
                            (28, 16, 42, 28, (40, 40, 40)),
                        ],
                    )
                    touching = paint(
                        40,
                        20,
                        [
                            (4, 4, 12, 16, (180, 40, 40)),
                            (12, 4, 22, 16, (40, 40, 160)),
                        ],
                    )
                    status, dropped = http_json(
                        base + "/2d/drop",
                        "POST",
                        {"stem": "family", "data": png_b64(separated)},
                    )
                    self.assertEqual(status, 201, dropped)
                    status, split = http_json(base + "/2d/split", "POST", {"stem": "family"})
                    self.assertEqual(status, 200, split)
                    self.assertEqual(len(split["parts"]), 2)
                    self.assertEqual(split["method"], "key")
                    status, flags = http_json(base + "/2d/status")
                    self.assertFalse(flags["sfx"])
                    self.assertEqual(flags["sfx_mode"], "stub")
                    self.assertEqual(flags["still"], still_backend())
                    status, saved = http_json(
                        base + "/2d/save",
                        "POST",
                        {
                            "stem": "family",
                            "parts": [
                                {
                                    "id": split["parts"][0]["id"],
                                    "file": split["parts"][0]["file"],
                                    "name": "Hull",
                                    "role": "hull",
                                    "motion": "holds still",
                                },
                                {
                                    "id": split["parts"][1]["id"],
                                    "file": split["parts"][1]["file"],
                                    "name": "Gun",
                                    "role": "weapon",
                                    "projectile": "small energy bolt",
                                },
                            ],
                        },
                    )
                    self.assertEqual(status, 200, saved)
                    gun = next(p for p in saved["parts"] if p["role"] == "weapon")
                    self.assertEqual(gun["projectile"], "small energy bolt")
                    status, heard = http_json(base + "/2d/sound", "POST", {"stem": "family", "part": gun})
                    self.assertEqual(status, 200, heard)
                    self.assertTrue(heard["stub"])
                    wav = root / "out" / "family" / "flat" / heard["sound"]
                    self.assertTrue(wav.is_file())
                    self.assertGreater(wav.stat().st_size, 44)
                    status, sheet = http_json(base + "/2d/family")
                    self.assertEqual(status, 200, sheet)
                    recorded = next(p for p in sheet["mechanics"] if p["id"] == gun["id"])
                    self.assertEqual(recorded["sound"], heard["sound"])
                    self.assertEqual(recorded["role"], "weapon")
                    status, touch_drop = http_json(
                        base + "/2d/drop",
                        "POST",
                        {"stem": "touch", "data": png_b64(touching)},
                    )
                    self.assertEqual(status, 201, touch_drop)
                    status, touch = http_json(base + "/2d/split", "POST", {"stem": "touch"})
                    self.assertTrue(touch["whole"])
                    self.assertEqual(touch["parts"], [])
                    runs.append(
                        {
                            "split_count": len(split["parts"]),
                            "touch_count": len(touch["parts"]),
                            "sound": heard["sound"],
                            "role": recorded["role"],
                            "projectile": recorded["projectile"],
                            "status": status,
                        }
                    )
                self.assertEqual(runs[0]["split_count"], runs[1]["split_count"])
                self.assertEqual(runs[0]["touch_count"], runs[1]["touch_count"])
                self.assertEqual(runs[0]["status"], runs[1]["status"])
                scratch = os.environ.get("FLAT2D_SCRATCH")
                if scratch:
                    Path(scratch).mkdir(parents=True, exist_ok=True)
                    (Path(scratch) / "factory-2d.json").write_text(
                        json.dumps(runs, indent=2) + "\n",
                        encoding="utf-8",
                    )
            finally:
                httpd.shutdown()
                thread.join(timeout=5)
                httpd.server_close()

    def test_view_center(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(load_sheet(root, "gone")["center"], "")
            im = Image.new("RGB", (12, 12), (255, 0, 255))
            px = im.load()
            for y in range(3, 9):
                for x in range(3, 9):
                    px[x, y] = (180, 40, 40)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            save_source(root, "unit", buf.getvalue())
            view = flat_dir(root, "unit") / "view.json"
            self.assertEqual(read_json(view), {"center": "source"})
            view.unlink()
            self.assertTrue((flat_dir(root, "unit") / "source.png").is_file())
            self.assertEqual(load_sheet(root, "unit")["center"], "source")
            walk = save_action(root, "unit", "walk", [im, im], 8)
            self.assertEqual(walk["sheet"], "sheets/walk.png")
            self.assertEqual(read_json(view), {"center": walk["sheet"]})
            self.assertEqual(load_sheet(root, "unit")["center"], walk["sheet"])
            idle = save_action(root, "unit", "idle", [im], 8)
            self.assertEqual(idle["sheet"], "sheets/idle.png")
            manifest = read_json(flat_dir(root, "unit") / "sheets" / "manifest.json")
            self.assertEqual(list(manifest["actions"]), ["walk", "idle"])
            view.unlink()
            self.assertEqual(load_sheet(root, "unit")["center"], "source")
            write_json(view, {"center": "sheets/walk.png"})
            self.assertEqual(load_sheet(root, "unit")["center"], "sheets/walk.png")
            write_json(view, {"center": "sheets/../../../open-data-dir"})
            self.assertEqual(load_sheet(root, "unit")["center"], "")
            write_json(view, {"center": "sheets/not a sheet.png"})
            self.assertEqual(load_sheet(root, "unit")["center"], "")


    def test_motion_without_a_clip_is_skipped(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("flat2d.steps.pack_request") as pack:
                home = apply_steps(
                    root,
                    "unit",
                    {"when": "home", "steps": ["motion"], "motion": "Walk cycle", "frames": []},
                )
                named = apply_steps(
                    root,
                    "unit",
                    {"when": "revise", "steps": ["motion"], "motion": "Walk cycle"},
                )
                blank = apply_steps(
                    root,
                    "unit",
                    {"when": "revise", "steps": ["motion"], "motion": "", "frames": [png_b64(Image.new("RGB", (4, 4)))]},
                )
            self.assertFalse(pack.called)
        self.assertEqual(home["ran"], [])
        self.assertEqual(home["skipped"], ["motion"])
        self.assertNotIn("center", home)
        self.assertEqual(home["notes"], ["Motion: attach a clip. The frames are not drawn from the sentence."])
        self.assertEqual(named["notes"], ["Attach a clip of that motion. The prompt is its name."])
        self.assertEqual(blank["notes"], ["Name the motion first."])
        self.assertNotIn("center", blank)
        for body in (home, named, blank):
            self.assertNotIn("parts", body)
            self.assertNotIn("whole", body)
            self.assertNotIn("keyed", body)
            self.assertNotIn("reason", body)

    def test_motion_slugs_the_line_before_pack(self) -> None:
        from unittest.mock import patch

        from flat2d.atlas import pack_request as real_pack
        from flat2d.steps import action_slug

        frame = png_b64(Image.new("RGB", (8, 8), (255, 0, 255)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("flat2d.steps.action_slug", wraps=action_slug) as slug:
                idle = apply_steps(
                    root,
                    "unit",
                    {"when": "home", "steps": ["motion"], "motion": "", "frames": [frame]},
                )
            self.assertFalse(slug.called)
            self.assertEqual(idle["center"], "sheets/idle.png")
            self.assertEqual(idle["ran"], ["motion"])
            with patch("flat2d.steps.pack_request", wraps=real_pack) as pack:
                out = apply_steps(
                    root,
                    "unit",
                    {
                        "when": "revise",
                        "steps": ["motion", "say"],
                        "motion": "Walk cycle",
                        "say": "",
                        "frames": [frame],
                    },
                )
            self.assertEqual(pack.call_args[0][2]["action"], "walk-cycle")
        self.assertEqual(out["ran"], ["motion"])
        self.assertEqual(out["skipped"], ["say"])
        self.assertEqual(out["center"], "sheets/walk-cycle.png")
        self.assertEqual(out["notes"][0], "Motion: 1 frames in walk-cycle.")
        self.assertEqual(out["notes"][1], "Write the line first.")
        self.assertNotIn("parts", out)
        self.assertNotIn("whole", out)
        self.assertNotIn("keyed", out)
        self.assertNotIn("reason", out)

    def test_say_without_a_line_is_skipped(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("flat2d.steps.speak") as speak:
                home = apply_steps(root, "unit", {"when": "home", "steps": ["say"], "say": "", "prompt": "a thing"})
                revise = apply_steps(root, "unit", {"steps": ["say"], "say": "   ", "prompt": "a thing"})
            self.assertFalse(speak.called)
        self.assertEqual(home["notes"], ["Say: write the line. A spoken line is not invented."])
        self.assertEqual(home["skipped"], ["say"])
        self.assertEqual(home["ran"], [])
        self.assertEqual(revise["notes"], ["Write the line first."])
        self.assertEqual(revise["skipped"], ["say"])

    def test_sound_home_falls_back_to_the_prompt(self) -> None:
        from flat2d.optional import write_stub_wav

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = apply_steps(
                root,
                "unit",
                {"when": "home", "steps": ["sound"], "sound": "", "prompt": "top down spider mech"},
            )
            wav = flat_dir(root, "unit") / "sounds" / "asset.wav"
            expect = Path(tmp) / "expect.wav"
            write_stub_wav(expect, "top down spider mech")
            self.assertEqual(wav.read_bytes(), expect.read_bytes())
        self.assertEqual(out["ran"], ["sound"])
        self.assertEqual(out["skipped"], [])
        self.assertEqual(out["hear"], "sounds/asset.wav")
        self.assertTrue(out["stub"])
        self.assertEqual(out["notes"], ["Sound: stand-in. The effect runner is not set."])

    def test_sound_revise_blank_part_writes_asset_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = apply_steps(root, "unit", {"when": "revise", "steps": ["sound"], "sound": "a latch"})
            blank = apply_steps(root, "unit", {"steps": ["sound"], "sound": "a latch", "part": "  "})
            bad = apply_steps(
                root, "unit", {"when": "revise", "steps": ["sound"], "sound": "a latch", "part": "Not A Slug"}
            )
            self.assertTrue((flat_dir(root, "unit") / "sounds" / "asset.wav").is_file())
            self.assertFalse((flat_dir(root, "unit") / "sounds" / "Not A Slug.wav").exists())
        self.assertEqual(missing["hear"], "sounds/asset.wav")
        self.assertEqual(blank["hear"], "sounds/asset.wav")
        self.assertEqual(bad["ran"], ["sound"])
        self.assertFalse(bad["hear"])
        self.assertIn("Not A Slug", bad["notes"][0])

    def test_sound_revise_empty_does_not_render(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("flat2d.steps.render_sound") as render:
                out = apply_steps(
                    root,
                    "unit",
                    {"when": "revise", "steps": ["sound"], "sound": "", "prompt": "top down spider mech", "part": ""},
                )
                omitted = apply_steps(
                    root,
                    "unit",
                    {"steps": ["sound"], "sound": "   ", "prompt": "top down spider mech"},
                )
            self.assertFalse(render.called)
        self.assertEqual(out["notes"], ["Describe the sound first."])
        self.assertEqual(out["skipped"], ["sound"])
        self.assertEqual(out["ran"], [])
        self.assertEqual(omitted["notes"], ["Describe the sound first."])
        self.assertFalse(out["stub"])

    def test_sound_no_runner_skips_without_calling_render(self) -> None:
        from unittest.mock import patch

        folder = Path(os.environ["FLAT2D_SFX_DIR"])
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"x")
        self.assertEqual(sfx_mode(), "no-runner")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("flat2d.steps.render_sound") as render:
                out = apply_steps(root, "unit", {"when": "home", "steps": ["sound"], "sound": "a latch"})
            self.assertFalse(render.called)
        self.assertEqual(out["notes"], ["SFX weights are installed. Set FLAT2D_SFX_CMD."])
        self.assertEqual(out["skipped"], ["sound"])
        self.assertEqual(out["ran"], [])
        self.assertNotEqual(
            out["notes"][0],
            "SFX weights are present. Set FLAT2D_SFX_CMD to the local runner "
            "(prompt and output path are the two arguments).",
        )

    def test_pieces_notes_follow_the_reason(self) -> None:
        from flat2d.routes import handle

        cases = {
            "whole": (
                paint(32, 24, [(4, 4, 20, 18, (180, 40, 40))]),
                None,
                "Pieces: one shape, left whole.",
                "whole",
            ),
            "keyed": (Image.new("RGB", (32, 24), (255, 0, 255)), [255, 0, 255], "Pieces: the costume is the key color, so it was not saved.", "keyed"),
            "specks": (Image.new("RGB", (32, 24), (255, 0, 255)), [255, 0, 255], "Pieces: nothing saved. Marks off the key were smaller than 24 pixels.", "specks"),
            "none": (Image.new("RGBA", (32, 24), (0, 0, 0, 0)), None, "Pieces: nothing separate was saved.", "none"),
        }
        specks = cases["specks"][0]
        px = specks.load()
        for i in range(8):
            px[4 + i, 8] = (0, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, (im, key, note, reason) in cases.items():
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                save_source(root, name, buf.getvalue())
                if key is not None:
                    write_json(flat_dir(root, name) / "key.json", {"rgb": key})
                out = apply_steps(root, name, {"when": "home", "steps": ["pieces"]})
                self.assertEqual(out["notes"], [note], name)
                self.assertEqual(out["reason"], reason, name)
                self.assertEqual(out["ran"], ["pieces"], name)
                self.assertEqual(out["parts"], [], name)
                self.assertFalse(out["pieces"], name)
            separated = paint(
                48,
                32,
                [
                    (4, 4, 14, 14, (180, 40, 40)),
                    (28, 16, 42, 28, (40, 40, 40)),
                ],
            )
            buf = io.BytesIO()
            separated.save(buf, format="PNG")
            save_source(root, "two", buf.getvalue())
            many = apply_steps(root, "two", {"when": "revise", "steps": ["pieces"], "pieces": "Hull, Gun"})
            self.assertEqual(many["notes"], ["Pieces: 2."])
            self.assertTrue(many["pieces"])
            self.assertEqual([row["name"] for row in many["parts"]], ["Hull", "Gun"])
            code, raw, _ctype = handle(
                root,
                "POST",
                "/2d/steps",
                json.dumps({"stem": "missing", "steps": ["pieces", "nope"]}).encode(),
            )
            posted = json.loads(raw)
        self.assertEqual(code, 200, posted)
        self.assertEqual(posted["notes"], ["Pieces: no source image"])
        self.assertEqual(posted["ran"], ["pieces"])
        self.assertEqual(posted["skipped"], [])
        self.assertNotIn("reason", posted)
        self.assertNotIn("parts", posted)

    def test_steps_batch_keeps_pieces_reason_and_stub_hear(self) -> None:
        im = Image.new("RGB", (32, 24), (255, 0, 255))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            save_source(root, "unit", buf.getvalue())
            write_json(flat_dir(root, "unit") / "key.json", {"rgb": [255, 0, 255]})
            out = apply_steps(
                root,
                "unit",
                {"when": "home", "steps": ["pieces", "sound"], "pieces": "hull, legs", "sound": "a latch", "prompt": "unused"},
            )
        self.assertEqual(out["reason"], "keyed")
        self.assertTrue(out["keyed"])
        self.assertFalse(out["whole"])
        self.assertFalse(out["pieces"])
        self.assertEqual(out["parts"], [])
        self.assertEqual(out["ran"], ["pieces", "sound"])
        self.assertEqual(out["hear"], "sounds/asset.wav")
        self.assertTrue(out["stub"])
        self.assertEqual(
            out["notes"],
            [
                "Pieces: the costume is the key color, so it was not saved.",
                "Sound: stand-in. The effect runner is not set.",
            ],
        )

    def test_motion_only_omits_piece_fields(self) -> None:
        frame = png_b64(Image.new("RGB", (8, 8), (20, 20, 20)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = apply_steps(
                root,
                "unit",
                {"when": "revise", "steps": ["motion"], "motion": "idle", "frames": [frame]},
            )
        self.assertEqual(out["ran"], ["motion"])
        self.assertEqual(out["center"], "sheets/idle.png")
        self.assertNotIn("parts", out)
        self.assertNotIn("pieces", out)
        self.assertNotIn("whole", out)
        self.assertNotIn("keyed", out)
        self.assertNotIn("reason", out)

    def test_steps_route_rejects_a_bad_when_and_empty_sound_is_unavailable(self) -> None:
        from flat2d.routes import handle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, raw, _ctype = handle(
                root,
                "POST",
                "/2d/steps",
                json.dumps({"stem": "unit", "when": "later", "steps": ["sound"]}).encode(),
            )
            self.assertEqual(code, 400)
            self.assertIn("when", json.loads(raw)["error"])
            with self.assertRaises(FlatUnavailable) as caught:
                render_sound(root, "unit", {"id": "asset", "projectile": "   "})
            self.assertEqual(str(caught.exception), "empty sound sentence")
            code, raw, _ctype = handle(
                root,
                "POST",
                "/2d/sound",
                json.dumps({"stem": "unit", "part": {"id": "asset", "projectile": ""}}).encode(),
            )
        self.assertEqual(code, 400)
        body = json.loads(raw)
        self.assertEqual(body["error"], "empty sound sentence")
        self.assertFalse(body["available"])


    def test_still_backend_is_licence_clean_and_nc_needs_an_opt_in(self) -> None:
        from unittest.mock import patch

        from flat2d.sheet import STILLS, still_degraded, still_info

        keep = {k: os.environ.pop(k, None) for k in ("FLAT2D_STILL", "FLAT2D_ALLOW_NC")}
        try:
            with patch("flat2d.sheet.mflux_ready", return_value=False):
                self.assertEqual(still_backend(), "")
            with patch("flat2d.sheet.mflux_ready", return_value=True), patch(
                "flat2d.sheet.qwen_ready", return_value=True
            ):
                with patch("flat2d.sheet.q2512_ready", return_value=True):
                    self.assertEqual(still_backend(), "qwen-2512")  # the default: Apache-2.0
                    self.assertEqual(still_degraded(), [])
                    os.environ["FLAT2D_STILL"] = "z-image"
                    self.assertEqual(still_backend(), "z-image")
                    self.assertEqual([(d["stage"], d["code"]) for d in still_degraded()], [("draw", "backend")])
                    # Qwen-Image 2.1 Turbo is non-commercial: asking for it without the opt-in gets the default.
                    os.environ["FLAT2D_STILL"] = "qwen-image"
                    self.assertEqual(still_backend(), "qwen-2512")
                    os.environ["FLAT2D_ALLOW_NC"] = "1"
                    self.assertEqual(still_backend(), "qwen-image")
                    self.assertIn(("licence", "nc"), [(d["stage"], d["code"]) for d in still_degraded()])
                    os.environ.pop("FLAT2D_ALLOW_NC")
                    os.environ.pop("FLAT2D_STILL")
                with patch("flat2d.sheet.q2512_ready", return_value=False):
                    self.assertEqual(still_backend(), "z-image")  # the fallback never reaches the NC model
            for name in ("qwen-2512", "z-image"):
                self.assertEqual(STILLS[name]["licence"], "apache-2.0")
            self.assertEqual(still_info("qwen-2512")["model"], "qwen-image-2512+lightning8")
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_still_image_routes_to_the_picked_model(self) -> None:
        from unittest.mock import patch

        with patch("flat2d.optional.still_backend", return_value="qwen-2512"), patch(
            "flat2d.optional.qwen2512_still", return_value="q5"
        ) as q5, patch("flat2d.optional.qwen_still", return_value="q") as qwen:
            from flat2d.optional import still_image

            self.assertEqual(still_image("a tank", seed=3, width=1152, height=1152), "q5")
            q5.assert_called_once_with("a tank", seed=3, width=1152, height=1152)
            qwen.assert_not_called()
        with patch("flat2d.optional.still_backend", return_value="qwen-image"), patch(
            "flat2d.optional.qwen_still", return_value="q"
        ) as qwen, patch("flat2d.optional.zimage_still", return_value="z") as zimage:
            self.assertEqual(still_image("a tank", seed=3), "q")
            qwen.assert_called_once_with("a tank", seed=3, width=1024, height=1024)
            zimage.assert_not_called()
        with patch("flat2d.optional.still_backend", return_value="z-image"), patch(
            "flat2d.optional.zimage_still", return_value="z"
        ):
            self.assertEqual(still_image("a tank"), "z")
        with patch("flat2d.optional.still_backend", return_value=""):
            with self.assertRaises(FlatUnavailable):
                still_image("a tank")

    def test_models_run_on_one_thread_whichever_job_calls(self) -> None:
        # MLX streams are per thread: a model built in one job's thread failed in the next job's.
        import threading
        from unittest.mock import patch

        from flat2d.optional import DrawCancelled, still_image

        seen = []

        def qwen(prompt, seed=1, width=1024, height=1024):
            seen.append(threading.get_ident())
            if prompt == "stop":
                raise DrawCancelled("Cancelled")
            return "q"

        with patch("flat2d.optional.still_backend", return_value="qwen-image"), patch(
            "flat2d.optional.qwen_still", side_effect=qwen
        ):
            for _ in range(2):
                t = threading.Thread(target=still_image, args=("a tank",))
                t.start()
                t.join()
            with self.assertRaises(DrawCancelled):
                still_image("stop")
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 1)
        self.assertNotEqual(seen[0], threading.get_ident())

    def test_mlx_cache_is_capped_before_the_build_and_emptied_after_each_draw(self) -> None:
        # Uncapped, MLX's buffer cache settled at ~44 GiB beside Qwen-Image's ~22 GiB of weights.
        import types
        from unittest.mock import patch

        import flat2d.turbo  # noqa: F401 - imported for real before mlx is faked below
        from flat2d import optional
        from flat2d.optional import DrawCancelled

        order = []
        mx = types.SimpleNamespace(
            set_cache_limit=lambda n: order.append(("limit", n)),
            clear_cache=lambda: order.append(("clear",)),
        )

        class QwenImage21:
            def __init__(self, quantize, model_path):
                order.append(("build",))
                self.prompt_cache = {}
                self.callbacks = types.SimpleNamespace(register=lambda relay: None)

            def generate_image(self, seed, prompt, **kwargs):
                self.prompt_cache[prompt] = "features"
                order.append(("draw", prompt, len(self.prompt_cache)))
                if prompt == "stop":
                    latents = object()  # noqa: F841 - held by the traceback until the release empties it
                    raise DrawCancelled("Cancelled")
                return "image"

        qwen_mod = types.ModuleType("mflux.models.qwen21.variants.txt2img.qwen_image_21")
        qwen_mod.QwenImage21 = QwenImage21
        mlx_pkg = types.ModuleType("mlx")
        mlx_pkg.core = mx
        with patch.dict(sys.modules, {"mlx": mlx_pkg, "mlx.core": mx, qwen_mod.__name__: qwen_mod}), patch.object(
            optional, "_QWEN", None
        ), patch.object(optional, "_QWEN_RELAY", None), patch("flat2d.turbo.load_grid", return_value=(1.0, 0.5)), patch(
            "flat2d.turbo.TurboGrid", types.SimpleNamespace(NODES=())
        ), patch.dict(
            os.environ, {"FLAT2D_MLX_CACHE_GB": "2"}
        ):
            self.assertEqual(optional.qwen_still("a tank", seed=2), "image")
            self.assertEqual(optional._QWEN.prompt_cache, {})
            try:  # not assertRaises: it strips the traceback this checks
                optional.qwen_still("stop")
            except DrawCancelled as e:
                tb = e.__traceback__
            else:
                self.fail("the cancel did not propagate")
            self.assertEqual(optional._QWEN.prompt_cache, {})
            self.assertIsNone(optional._QWEN_RELAY.prev)
            while tb.tb_next is not None:
                tb = tb.tb_next
            self.assertEqual(tb.tb_frame.f_code.co_name, "generate_image")
            self.assertNotIn("latents", tb.tb_frame.f_locals)
            optional.qwen_still("a cow")
        self.assertEqual(
            order,
            [("limit", 2 * 2**30), ("build",), ("draw", "a tank", 1), ("clear",),
             ("draw", "stop", 1), ("clear",), ("draw", "a cow", 1), ("clear",)],
        )

    def test_zimage_cache_is_capped_and_emptied(self) -> None:
        import types
        from unittest.mock import patch

        from flat2d import optional

        order = []
        mx = types.SimpleNamespace(
            set_cache_limit=lambda n: order.append(("limit", n)),
            clear_cache=lambda: order.append(("clear",)),
        )

        class ZImageTurbo:  # Z-Image keeps no prompt_cache
            def __init__(self, quantize):
                order.append(("build",))
                self.callbacks = types.SimpleNamespace(register=lambda relay: None)

            def generate_image(self, **kwargs):
                order.append(("draw",))
                raise RuntimeError("out of memory")

        z_mod = types.ModuleType("mflux.models.z_image")
        z_mod.ZImageTurbo = ZImageTurbo
        mlx_pkg = types.ModuleType("mlx")
        mlx_pkg.core = mx
        env = {k: v for k, v in os.environ.items() if k != "FLAT2D_MLX_CACHE_GB"}
        with patch.dict(sys.modules, {"mlx": mlx_pkg, "mlx.core": mx, "mflux.models.z_image": z_mod}), patch.object(
            optional, "_ZIMAGE", None
        ), patch.object(optional, "_ZIMAGE_RELAY", None), patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                optional.zimage_still("a tank")
        self.assertEqual(order, [("limit", int(optional.MLX_CACHE_GB * 2**30)), ("build",), ("draw",), ("clear",)])

    def test_turbo_grid_reads_model_index_and_rejects_bad_values(self) -> None:
        self.assertEqual(read_grid({"sample_sigmas": [1.0, 0.8, 0.3]}), (1.0, 0.8, 0.3))
        self.assertEqual(read_grid({}), TURBO_SIGMAS)
        self.assertEqual(read_grid({"sample_sigmas": [0.3, 0.8]}), TURBO_SIGMAS)
        self.assertEqual(read_grid({"sample_sigmas": [1.0, 0.0]}), TURBO_SIGMAS)
        self.assertEqual(read_grid({"sample_sigmas": ["x"]}), TURBO_SIGMAS)
        self.assertEqual(len(TURBO_SIGMAS), 8)


class GroundKeyTests(unittest.TestCase):
    """Generated stills: the model paints a near-magenta, uneven ground with shadows."""

    def setUp(self) -> None:
        # The key alone: no Apple Vision, no SAM checkpoint from this machine.
        empty = Path(tempfile.mkdtemp())
        for name, value in (("FLAT2D_LIFT", empty / "lift-subject"), ("FLAT2D_SAM_DIR", empty / "sam")):
            old = os.environ.get(name)
            self.addCleanup(lambda n=name, o=old: os.environ.__setitem__(n, o) if o is not None else os.environ.pop(n, None))
            os.environ[name] = str(value)

    GROUND = (215, 20, 170)

    def still(self) -> Image.Image:
        import random

        rnd = random.Random(7)
        im = Image.new("RGB", (192, 128))
        px = im.load()
        for y in range(128):
            for x in range(192):
                px[x, y] = tuple(max(0, min(255, c + rnd.randint(-12, 12))) for c in self.GROUND)

        def fill(x0, y0, x1, y1, color):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    px[x, y] = color

        fill(16, 80, 88, 100, tuple(int(c * 0.55) for c in self.GROUND))  # soft shadow: the ground, darker
        fill(20, 24, 80, 80, (60, 70, 50))  # dark hull
        fill(120, 40, 168, 68, (200, 180, 60))  # gun
        fill(140, 8, 148, 16, (190, 50, 160))  # a patch of uneven ground the key misses
        return im

    def test_border_sets_the_key(self) -> None:
        key = ground_key(self.still())
        self.assertLessEqual(sum(abs(a - b) for a, b in zip(key["rgb"], self.GROUND)), 12)
        self.assertGreaterEqual(key["tol"], TOL)
        self.assertEqual(ground_key(Image.new("RGB", (32, 32), (255, 0, 255))), {"rgb": [255, 0, 255], "tol": TOL})

    def test_split_drops_ground_shadow_and_patch(self) -> None:
        im = self.still()
        self.assertEqual([c.size for c in split_image(im, key=(255, 0, 255))], [(192, 128)])
        key = ground_key(im)
        crops = split_image(im, key=tuple(key["rgb"]), tol=key["tol"])
        self.assertEqual([c.size for c in crops], [(68, 64), (56, 36)])
        self.assertEqual(crops[0].getpixel((30, 30)), (60, 70, 50, 255))

    def test_segmenter_mask_wins_and_key_adds_what_it_missed(self) -> None:
        im = self.still()
        w, h = im.size
        hull_only = bytearray(w * h)
        for y in range(24, 80):
            for x in range(20, 80):
                hull_only[y * w + x] = 1
        key = ground_key(im)
        crops = split_image(im, key=tuple(key["rgb"]), tol=key["tol"], subject=hull_only)
        self.assertEqual([c.size for c in crops], [(68, 64), (56, 36)])

    def test_vision_mask_reads_the_lift_binary(self) -> None:
        from flat2d.store import vision_mask

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake = tmp / "lift-subject"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "from PIL import Image\n"
                "im = Image.new('RGBA', (4, 2), (0, 0, 0, 0))\n"
                "im.putpixel((1, 0), (9, 9, 9, 255))\n"
                "im.save(sys.argv[2])\n"
                "print('lift-subject: vision -> ' + sys.argv[2])\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            src = tmp / "still.png"
            Image.new("RGB", (4, 2), (255, 0, 255)).save(src)
            old = os.environ.get("FLAT2D_LIFT")
            os.environ["FLAT2D_LIFT"] = str(fake)
            try:
                self.assertEqual(vision_mask(src), bytearray([0, 1, 0, 0, 0, 0, 0, 0]))
                os.environ["FLAT2D_LIFT"] = str(tmp / "missing")
                self.assertIsNone(vision_mask(src))
            finally:
                if old is None:
                    os.environ.pop("FLAT2D_LIFT", None)
                else:
                    os.environ["FLAT2D_LIFT"] = old

    def test_clear_glass_becomes_see_through(self) -> None:
        key = (220, 40, 180)
        crop = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
        px = crop.load()
        for y in range(20):
            for x in range(20):
                if (x + y) % 4 == 0:
                    px[x, y] = tuple(int(0.4 * 250 + 0.6 * k) for k in key) + (255,)  # highlight over ground
                else:
                    px[x, y] = tuple(int(0.1 * 200 + 0.9 * k) for k in key) + (255,)  # thin glass over ground
        out = see_through(crop, key)
        a = out.getpixel((1, 0))[3]
        self.assertLess(a, 60)
        r, g, b, _ = out.getpixel((0, 0))
        self.assertLess(max(r, g, b) - min(r, g, b), 30)  # the highlight left is colorless

    def test_pink_subject_and_red_liquid_stay_solid(self) -> None:
        key = (220, 40, 180)
        # A solid pink subject is shaded: darker copies of the ground, from 55% to 85%.
        pink = Image.new("RGBA", (20, 20))
        for y in range(20):
            for x in range(20):
                pink.putpixel((x, y), tuple(int(k * (0.55 + 0.3 * x / 19)) for k in key) + (255,))
        self.assertEqual(see_through(pink, key).tobytes(), pink.tobytes())
        red = Image.new("RGBA", (20, 20), (200, 30, 40, 255))
        self.assertEqual(see_through(red, key).getpixel((5, 5)), (200, 30, 40, 255))
        gray_key = see_through(Image.new("RGBA", (4, 4), (90, 90, 90, 255)), (255, 255, 255))
        self.assertEqual(gray_key.getpixel((1, 1)), (90, 90, 90, 255))

    def test_names_follow_reading_order(self) -> None:
        im = Image.new("RGB", (60, 60), (255, 0, 255))
        px = im.load()
        for y in range(10, 20):
            for x in range(5, 12):
                px[x, y] = (40, 40, 40)  # small, left, a little lower
        for y in range(4, 30):
            for x in range(30, 55):
                px[x, y] = (200, 200, 60)  # big, right, higher
        for y in range(44, 54):
            for x in range(4, 20):
                px[x, y] = (40, 120, 40)  # next row
        crops = split_image(im, key=(255, 0, 255))
        self.assertEqual([c.info["box"][0] for c in crops], [26, 0, 1])  # biggest first
        self.assertEqual([c.info["box"][0] for c in reading_order(crops)], [1, 26, 0])

    def test_one_object_on_a_key_is_saved_as_its_cutout(self) -> None:
        from flat2d.steps import _pieces_note
        from flat2d.store import save_source, split_source

        im = Image.new("RGB", (40, 30), (255, 0, 255))
        px = im.load()
        for y in range(8, 22):
            for x in range(10, 30):
                px[x, y] = (90, 60, 30)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            save_source(root, "unit", buf.getvalue())
            write_json(flat_dir(root, "unit") / "key.json", {"rgb": [255, 0, 255]})
            split = split_source(root, "unit", "Crate")
            self.assertTrue(split["whole"])
            self.assertEqual([(p["name"], p["w"], p["h"]) for p in split["parts"]], [("Crate", 28, 22)])
            self.assertTrue((root / "out" / "unit" / "flat" / "parts" / "part-01.png").is_file())
            self.assertEqual(_pieces_note(split), "Pieces: one shape, saved as its cutout.")

    def test_segmenter_specks_are_not_parts(self) -> None:
        im = Image.new("RGB", (200, 200), (255, 0, 255))
        subject = bytearray(200 * 200)
        for y in range(20, 180):
            for x in range(20, 180):
                subject[y * 200 + x] = 1
        for y in range(190, 197):
            for x in range(190, 197):
                subject[y * 200 + x] = 1  # 49 px next to a 25,600 px subject
        crops = split_image(im, key=(255, 0, 255), subject=subject)
        self.assertEqual(len(crops), 1)

    def test_cut_edge_loses_the_key_tint(self) -> None:
        from flat2d.parts import PAD

        im = Image.new("RGB", (40, 40), (255, 0, 255))
        px = im.load()
        for y in range(10, 30):
            for x in range(10, 30):
                px[x, y] = (210, 120, 200) if x == 10 else (150, 150, 150)
        crop = split_image(im, key=(255, 0, 255))[0]
        self.assertEqual(crop.getpixel((PAD, PAD + 5)), (130, 120, 120, 255))
        self.assertEqual(crop.getpixel((PAD + 10, PAD + 10)), (150, 150, 150, 255))


if __name__ == "__main__":
    unittest.main()
