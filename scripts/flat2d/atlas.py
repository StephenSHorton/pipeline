"""Pack one action into a transparent strip plus a manifest. No model calls."""
from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from flat2d.sheet import PART_ID, flat_dir, read_json, write_json
from flat2d.store import FlatError, write_center

FEATHER = 48
HOLD = 12
# A border whose 90th percentile is farther than this from its median is not a flat field.
FLAT_P90 = 48
# Keep a solid component at least this big, plus a two-pixel fringe. Smaller specks go.
SPECK_MIN = 24
SPECK_RADIUS = 2


def action_name(name: str) -> str:
    text = (name or "").strip().lower()
    if not PART_ID.match(text):
        raise FlatError(f"invalid action {name!r}")
    return text


def unmix(im: Image.Image, key: tuple[int, int, int] | None = None) -> Image.Image:
    """Pull a flat background out of the pixel so the fringe is transparency, not a tinted edge."""
    rgb = im.convert("RGB")
    w, h = rgb.size
    raw = rgb.tobytes()
    bg = key if key is not None else _border_key(raw, w, h)
    out = bytearray(w * h * 4)
    for i in range(w * h):
        o = i * 3
        r, g, b = raw[o], raw[o + 1], raw[o + 2]
        dist = abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2])
        if dist <= HOLD:
            alpha = 0
        elif dist >= HOLD + FEATHER:
            alpha = 255
        else:
            alpha = int(round((dist - HOLD) * 255 / FEATHER))
        if 0 < alpha < 255:
            scale = 255 / alpha
            r = _clamp((r - bg[0]) * scale + bg[0])
            g = _clamp((g - bg[1]) * scale + bg[1])
            b = _clamp((b - bg[2]) * scale + bg[2])
        p = i * 4
        out[p : p + 4] = bytes((r, g, b, alpha))
    return _drop_specks(Image.frombytes("RGBA", (w, h), bytes(out)))


def pack_frames(frames: list[Image.Image], action: str) -> tuple[Image.Image, dict]:
    if not frames:
        raise FlatError("no frames")
    cells = [unmix(frame) for frame in frames]
    cell_w = max(im.size[0] for im in cells)
    cell_h = max(im.size[1] for im in cells)
    sheet = Image.new("RGBA", (cell_w * len(cells), cell_h), (0, 0, 0, 0))
    layout = []
    for i, im in enumerate(cells):
        x = i * cell_w
        sheet.paste(im, (x, 0), im)
        layout.append({"i": i, "x": x, "y": 0, "w": im.size[0], "h": im.size[1]})
    return sheet, {
        "file": f"sheets/{action}.png",
        "frames": len(cells),
        "cell": {"w": cell_w, "h": cell_h},
        "frame_layout": layout,
    }


def frames_from_video(data: bytes, count: int) -> list[Image.Image]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FlatError("ffmpeg is not installed")
    count = max(1, min(int(count or 8), 24))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        src = folder / "in.mp4"
        src.write_bytes(data)
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
            capture_output=True,
            text=True,
        )
        try:
            duration = float((probe.stdout or "").strip() or "0")
        except ValueError:
            duration = 0
        if duration <= 0:
            duration = 1
        rate = count / duration
        pattern = str(folder / "f%02d.png")
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-vf", f"fps={rate}", "-frames:v", str(count), pattern],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "ffmpeg failed").strip().splitlines()
            raise FlatError(err[-1] if err else "ffmpeg failed")
        paths = sorted(folder.glob("f*.png"))
        if not paths:
            raise FlatError("video produced no frames")
        frames = []
        for path in paths:
            with Image.open(path) as im:
                frames.append(im.convert("RGB"))
        return frames


def pack_request(root: Path, stem: str, payload: dict) -> dict:
    action = action_name(str(payload.get("action") or "idle"))
    try:
        fps = int(payload.get("fps") or 10)
        count = int(payload.get("count") or 8)
    except (TypeError, ValueError) as e:
        raise FlatError("fps and count must be numbers") from e
    video = payload.get("video")
    if isinstance(video, str) and video.strip():
        frames = frames_from_video(base64.b64decode(video), count)
    else:
        raws = payload.get("frames") or []
        if not isinstance(raws, list) or not raws:
            raise FlatError("frames or video required")
        frames = []
        for raw in raws:
            if not isinstance(raw, str) or not raw:
                raise FlatError("empty frame")
            with Image.open(io.BytesIO(base64.b64decode(raw))) as im:
                frames.append(im.convert("RGB"))
    return save_action(root, stem, action, frames, fps)


def save_action(root: Path, stem: str, action: str, frames: list[Image.Image], fps: int) -> dict:
    action = action_name(action)
    image, row = pack_frames(frames, action)
    row["fps"] = max(1, min(int(fps or 10), 30))
    dest = flat_dir(root, stem) / "sheets" / f"{action}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    manifest_path = flat_dir(root, stem) / "sheets" / "manifest.json"
    manifest = read_json(manifest_path) or {"ok": True, "stem": stem, "actions": {}}
    actions = dict(manifest.get("actions") or {})
    actions[action] = row
    manifest = {"ok": True, "stem": stem, "actions": actions}
    write_json(manifest_path, manifest)
    write_center(root, stem, row["file"])
    return {"ok": True, "stem": stem, "action": action, "sheet": row["file"], "manifest": "sheets/manifest.json", **row}


def _border_key(raw: bytes, w: int, h: int) -> tuple[int, int, int]:
    """Median border color. One odd corner cannot move it. A gradient border is refused."""
    samples = [_at(raw, w, x, y) for x, y in _border_points(w, h)]
    if not samples:
        raise FlatError("empty frame")
    key = tuple(sorted(channel)[len(samples) // 2] for channel in zip(*samples))
    dists = sorted(abs(r - key[0]) + abs(g - key[1]) + abs(b - key[2]) for r, g, b in samples)
    if dists[min(len(dists) - 1, int(len(dists) * 0.9))] > FLAT_P90:
        raise FlatError("field is not flat")
    return key[0], key[1], key[2]


def _border_points(w: int, h: int) -> list[tuple[int, int]]:
    pts = [(x, 0) for x in range(w)]
    if h > 1:
        pts.extend((x, h - 1) for x in range(w))
    pts.extend((0, y) for y in range(1, h - 1))
    if w > 1:
        pts.extend((w - 1, y) for y in range(1, h - 1))
    return pts


def _at(raw: bytes, w: int, x: int, y: int) -> tuple[int, int, int]:
    o = (y * w + x) * 3
    return raw[o], raw[o + 1], raw[o + 2]


def _drop_specks(im: Image.Image) -> Image.Image:
    """Keep solid components large enough to be a body, plus a two-pixel fringe."""
    w, h = im.size
    raw = im.tobytes()
    solid = [i for i in range(w * h) if raw[i * 4 + 3] == 255]
    seen = bytearray(w * h)
    keep = bytearray(w * h)
    for start in solid:
        if seen[start]:
            continue
        stack = [start]
        seen[start] = 1
        comp = [start]
        while stack:
            i = stack.pop()
            x, y = i % w, i // w
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                j = ny * w + nx
                if seen[j] or raw[j * 4 + 3] != 255:
                    continue
                seen[j] = 1
                stack.append(j)
                comp.append(j)
        if len(comp) < SPECK_MIN:
            continue
        fringe = list(comp)
        for _ in range(SPECK_RADIUS):
            nxt = []
            for i in fringe:
                x, y = i % w, i // w
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    j = ny * w + nx
                    if keep[j]:
                        continue
                    keep[j] = 1
                    nxt.append(j)
            fringe = nxt
        for i in comp:
            keep[i] = 1
    out = bytearray(raw)
    for i in range(w * h):
        if not keep[i]:
            out[i * 4 + 3] = 0
    return Image.frombytes("RGBA", (w, h), bytes(out))


def _clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


