#!/usr/bin/env python3
"""Factory paths and generate argv."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# The pictures 3D can start from (bin/lift-subject reads every one of them, .webp included).
UPLOAD_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
NO_PICTURE = "3D needs a picture. Draw or add one first."


def require_picture(path: str | Path | None) -> Path:
    """The path, when it is a picture 3D can start from: a known image suffix that PIL can open and verify.
    Anything else (no path, a mesh, a text file renamed .jpg, a missing file) raises ValueError(NO_PICTURE)."""
    if not path:
        raise ValueError(NO_PICTURE)
    p = Path(path)
    if p.suffix.lower() not in UPLOAD_EXTS or not p.is_file():
        raise ValueError(NO_PICTURE)
    try:
        from PIL import Image

        with Image.open(p) as im:
            im.verify()
    except Exception:  # noqa: BLE001 - UnidentifiedImageError, truncated data, a decompression bomb: all "not a picture"
        raise ValueError(NO_PICTURE) from None
    return p


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# One grammar for every fallback, in the generate, rig and draw logs: "DEGRADED <stage> <code>: <message>".
DEGRADED_RE = re.compile(r"^DEGRADED (\S+) (\S+): ?(.*)$")


def degraded_entry(stage: str, code: str, message: str) -> dict[str, str]:
    return {"stage": str(stage), "code": str(code), "message": str(message)}


def degraded_line(entry: dict[str, str]) -> str:
    return f"DEGRADED {entry['stage']} {entry['code']}: {entry['message']}"


def parse_degraded(text: str) -> list[dict[str, str]]:
    """Every DEGRADED line in a log, in order, without repeats."""
    out: list[dict[str, str]] = []
    for line in (text or "").splitlines():
        m = DEGRADED_RE.match(line.strip())
        if m:
            add_degraded(out, degraded_entry(*m.groups()))
    return out


def coerce_degraded(item: Any) -> dict[str, str] | None:
    """A degraded entry from a rig.json / views.json list: a dict {stage, code, message} or a DEGRADED-grammar string."""
    if isinstance(item, dict):
        stage, code = item.get("stage"), item.get("code")
        if not stage and not code and not item.get("message"):
            return None
        return degraded_entry(stage or "unknown", code or "unknown", item.get("message") or "")
    if isinstance(item, str) and item.strip():
        text = item.strip()
        m = DEGRADED_RE.match(text if text.startswith("DEGRADED ") else "DEGRADED " + text)
        if m:
            return degraded_entry(*m.groups())
        return degraded_entry("unknown", "unknown", text)
    return None


def add_degraded(into: list[dict[str, str]], entry: dict[str, str] | None) -> None:
    if entry and entry not in into:
        into.append(entry)


def json_degraded(path: str | Path) -> list[dict[str, str]]:
    """The "degraded" list of a JSON file (rig/rig.json, flat/views.json); [] when it is missing or unreadable."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[dict[str, str]] = []
    if isinstance(doc, dict) and isinstance(doc.get("degraded"), list):
        for item in doc["degraded"]:
            add_degraded(out, coerce_degraded(item))
    return out

# GET /glb?lod=: the file each name serves. game / proxy LODs are no longer made (reduce.sh cuts only lookdev), so
# those names fall through to DEFAULT_GLB like any unknown one, and nothing here reaches for their files.
LOD_FILES = {
    "mesh": ("mesh.glb",),
    "hero": ("mesh_hero.glb", "mesh.glb"),
    "lookdev": ("mesh_lookdev.glb", "mesh.glb"),
    "mv": ("mesh_mv.glb", "mesh_hero.glb", "mesh.glb"),
    "psx": ("psx.glb",),
}
DEFAULT_GLB = ("mesh.glb", "mesh_hero.glb", "mesh_mv.glb")

# UI labels High / Medium / Low. Canonical CLI names stay hero / lookdev / proxy.
# Medium is lookdev: same fidelity class, lower octree so shape is faster.
POLY_ALIASES = {
    "high": "hero",
    "medium": "lookdev",
    "low": "proxy",
    "hero": "hero",
    "lookdev": "lookdev",
    "game": "game",
    "proxy": "proxy",
}
POLY_LABELS = {
    "hero": "High",
    "lookdev": "Medium",
    "proxy": "Low",
    "game": "Game",
}


def normalize_poly(poly: str | None) -> str:
    key = str(poly or "medium").strip().lower()
    canon = POLY_ALIASES.get(key)
    if canon is None:
        raise ValueError(f"unknown poly {poly}")
    return canon


def poly_label(poly: str | None) -> str:
    return POLY_LABELS.get(normalize_poly(poly), "Medium")


def repo_root(start: Path | None = None) -> Path:
    env = os.environ.get("FACTORY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for p in [here, *here.parents]:
        if (p / "scripts" / "inspect_glb.py").is_file():
            return p
    return Path(__file__).resolve().parents[1]


def factory_python(root: Path | None = None) -> Path:
    root = repo_root() if root is None else Path(root)
    p = root / ".venv" / "bin" / "python"
    if p.is_file():
        return p
    return Path(sys.executable)


def generate_script(root: Path) -> Path:
    return root / "scripts" / "generate.sh"


PSX_KEYS = ("facing", "height", "faces", "atlas", "up", "name")
PSX_FACING = {"camera", "right", "left", "away"}


def clean_psx(raw: dict[str, Any] | None) -> dict[str, str]:
    """generate.sh --quality psx options from a request, checked. A missing or empty value is left out (generate.sh
    has a default: facing camera, and so on); an unknown key or a bad value raises ValueError instead of being
    dropped, so a typo never builds a model quietly different from the one asked for."""
    out: dict[str, str] = {}
    for key, value in (raw or {}).items():
        if value in (None, ""):
            continue
        if key not in PSX_KEYS:
            raise ValueError(f"unknown psx option {key!r} (one of {', '.join(PSX_KEYS)})")
        text = str(value).strip()
        if key == "facing":
            if text not in PSX_FACING:
                raise ValueError(f"facing must be one of {', '.join(sorted(PSX_FACING))} (the side of the picture "
                                 f"the head points to), not {text!r}")
        elif key == "up":
            if text not in {"z", "y"}:
                raise ValueError(f"up must be z or y, not {text!r}")
        elif key in {"faces", "atlas"}:
            if not (text.isdigit() and 0 < int(text) <= 65536):
                raise ValueError(f"{key} must be a whole number from 1 to 65536, not {text!r}")
        elif key == "height":
            try:
                ok = 0 < float(text) <= 100000
            except ValueError:
                ok = False
            if not ok:
                raise ValueError(f"height must be a number above 0 and up to 100000, not {text!r}")
        elif key == "name":
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", text):
                raise ValueError(f"name must be 1 to 64 letters, digits, _ or -, not {text!r}")
        out[key] = text
    return out


def generate_argv(
    root: Path,
    *,
    image: Path,
    quality: str,
    stem: str,
    poly: str,
    seed: str = "0",
    refs: list[Path] | None = None,
    prompt: str | None = None,
    psx: dict[str, Any] | None = None,
    views_json: Path | None = None,
    degraded: list[dict[str, str]] | None = None,
    mesh: Path | None = None,
    paint: bool = True,
    fill: str | None = None,
    mirror_ends: bool | None = None,
    delight: bool | None = None,
) -> list[str]:
    cmd = [
        str(generate_script(root)),
        "--image",
        str(image),
        "--quality",
        quality,
        "--stem",
        stem,
        "--poly",
        poly,
        "--seed",
        str(seed),
    ]
    if prompt:
        cmd += ["--prompt", prompt]
    for ref in refs or []:
        cmd += ["--ref", str(ref)]
    # The draw's four views (flat/views.json), only when the factory checked they belong to this picture.
    if views_json:
        cmd += ["--views-json", str(views_json)]
    # Fallbacks the factory chose before the run (views stale or inconsistent): generate.sh records them in
    # degraded.json and inspect.json beside its own.
    for entry in degraded or []:
        cmd += ["--degraded", f"{entry['stage']} {entry['code']}: {entry['message']}"]
    # A shape made elsewhere (the workbench's clay) instead of hy3d's; no paint: unwrap, then the photo bake.
    if mesh:
        cmd += ["--mesh", str(mesh)]
    if not paint:
        cmd += ["--paint", "off"]
    # Every texel from the photo (mirrored onto the far end, the rest filled from its swatch). Only when asked, so a
    # plain make's argv is as it was; mirror_ends only when stated (generate.sh's default follows fill).
    if fill == "photo":
        cmd += ["--fill", "photo"]
    if mirror_ends is not None:
        cmd += ["--mirror-ends", "on" if mirror_ends else "off"]
    # The photo's lighting out before the bake; only when stated (generate.sh's default is off, a make as before).
    if delight is not None:
        cmd += ["--delight", "on" if delight else "off"]
    # --quality psx options: facing, height, faces, atlas, up, name.
    for key, value in (psx or {}).items():
        if value not in (None, ""):
            cmd += [f"--{'facing' if key == 'facing' else 'psx-' + key}", str(value)]
    return cmd


def check_stem(stem: str) -> str:
    if not STEM_RE.match(stem):
        raise ValueError(f"invalid stem {stem!r}")
    return stem


_SLUG_LEAD = frozenset({"a", "an", "the"})
_GENERIC_EXACT = frozenset(
    {
        "image",
        "img",
        "photo",
        "pic",
        "picture",
        "screenshot",
        "screen-shot",
        "untitled",
        "download",
        "upload",
        "file",
        "asset",
        "ref",
        "reference",
        "capture",
        "scan",
        "dcim",
        "source",
        "input",
        "preview",
    }
)
_GENERIC_CAMERA = re.compile(
    r"^(?:img|dscn?|dscf|pict|pxl|screenshot|screen-?shot|image|photo|pic|"
    r"untitled|download|upload|file|asset|ref|reference|capture|scan|dcim)"
    r"[-_.]?\d[\d._-]*$",
    re.I,
)


def slug_stem(text: str | None, *, limit: int = 40) -> str:
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    while words and words[0] in _SLUG_LEAD:
        words.pop(0)
    slug = "-".join(words).strip("-")
    if not slug:
        return ""
    if len(slug) <= limit:
        return slug
    cut = slug[:limit]
    if "-" in cut:
        cut = cut.rsplit("-", 1)[0]
    return cut.rstrip("-")


def is_generic_stem(stem: str | None) -> bool:
    raw = str(stem or "").strip()
    if not raw:
        return True
    slug = slug_stem(raw) or raw.lower()
    if slug in _GENERIC_EXACT or re.fullmatch(r"asset-\d+", slug):
        return True
    compact = raw.replace(" ", "").replace("-", "_")
    return bool(
        _GENERIC_CAMERA.match(raw)
        or _GENERIC_CAMERA.match(slug)
        or _GENERIC_CAMERA.match(compact)
    )


def _read_u16(data: bytes, off: int, be: bool) -> int:
    if off + 2 > len(data):
        return 0
    return int.from_bytes(data[off : off + 2], "big" if be else "little")


def _read_u32(data: bytes, off: int, be: bool) -> int:
    if off + 4 > len(data):
        return 0
    return int.from_bytes(data[off : off + 4], "big" if be else "little")


def _exif_ascii(tiff: bytes, tag: int) -> str | None:
    if len(tiff) < 8:
        return None
    if tiff[:2] == b"MM":
        be = True
    elif tiff[:2] == b"II":
        be = False
    else:
        return None
    if _read_u16(tiff, 2, be) != 42:
        return None
    ifd = _read_u32(tiff, 4, be)
    if ifd + 2 > len(tiff):
        return None
    n = _read_u16(tiff, ifd, be)
    for i in range(n):
        e = ifd + 2 + i * 12
        if e + 12 > len(tiff):
            break
        if _read_u16(tiff, e, be) != tag:
            continue
        typ = _read_u16(tiff, e + 2, be)
        cnt = _read_u32(tiff, e + 4, be)
        if typ != 2 or cnt < 1:
            continue
        if cnt <= 4:
            raw = tiff[e + 8 : e + 8 + cnt]
        else:
            off = _read_u32(tiff, e + 8, be)
            raw = tiff[off : off + cnt]
        text = raw.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
        return text or None
    return None


def _caption_jpeg(data: bytes) -> str | None:
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:
            break
        seglen = int.from_bytes(data[i + 2 : i + 4], "big")
        if seglen < 2 or i + 2 + seglen > len(data):
            break
        payload = data[i + 4 : i + 2 + seglen]
        if marker == 0xFE:
            text = payload.decode("utf-8", "replace").strip("\x00").strip()
            if text:
                return text
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            found = _exif_ascii(payload[6:], 0x010E)
            if found:
                return found
        i += 2 + seglen
    return None


def _caption_png(data: bytes) -> str | None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    i = 8
    wanted = {b"Title", b"Description", b"Comment"}
    found: dict[bytes, str] = {}
    while i + 12 <= len(data):
        ln = int.from_bytes(data[i : i + 4], "big")
        typ = data[i + 4 : i + 8]
        if i + 8 + ln > len(data):
            break
        chunk = data[i + 8 : i + 8 + ln]
        i += 12 + ln
        if typ == b"IEND":
            break
        if typ != b"tEXt":
            continue
        key, _, val = chunk.partition(b"\x00")
        if key in wanted:
            found[key] = val.decode("latin-1", "replace").strip()
    for key in (b"Description", b"Title", b"Comment"):
        if found.get(key):
            return found[key]
    return None


def caption_from_image(path: Path) -> str | None:
    """Cheap label from a ref: JPEG/PNG text chunk or a non-generic filename.

    No chat LLM, no text-gen model, and no look at a finished mesh.
    """
    path = Path(path)
    try:
        data = path.read_bytes()[: 1 << 20]
    except OSError:
        data = b""
    text = None
    if data[:3] == b"\xff\xd8\xff":
        text = _caption_jpeg(data)
    elif data.startswith(b"\x89PNG"):
        text = _caption_png(data)
    slug = slug_stem(text or "")
    if slug and not is_generic_stem(slug):
        return slug
    slug = slug_stem(path.stem)
    if slug and not is_generic_stem(slug):
        return slug
    return None


def taken_stems(root: Path) -> set[str]:
    out = Path(root) / "out"
    if not out.is_dir():
        return set()
    return {p.name for p in out.iterdir() if p.is_dir() and not p.name.startswith(".")}


def unique_stem(root: Path, base: str, taken: set[str] | None = None, *, keep: bool = False) -> str:
    if keep:
        slug = check_stem(str(base or "").strip())
    else:
        slug = slug_stem(base) or str(base or "").strip()
        slug = check_stem(slug)
    used = taken if taken is not None else taken_stems(root)
    if slug not in used:
        return slug
    n = 2
    while True:
        cand = f"{slug}-{n}"
        if cand not in used:
            return cand
        n += 1


def next_generic_stem(root: Path, taken: set[str] | None = None) -> str:
    used = taken if taken is not None else taken_stems(root)
    n = 1
    while True:
        cand = f"asset-{n}"
        if cand not in used:
            return cand
        n += 1


def owner_stem(root: Path, path: Path) -> str | None:
    """Stem of an image that already lives under out/<stem>/, when that name is real."""
    try:
        rel = Path(path).resolve().relative_to((Path(root) / "out").resolve())
    except (OSError, ValueError):
        return None
    if len(rel.parts) < 2:
        return None
    stem = rel.parts[0]
    try:
        check_stem(stem)
    except ValueError:
        return None
    if is_generic_stem(stem):
        return None
    return stem


def auto_stem(
    root: Path,
    prompt: str | None = None,
    images: list[Path] | None = None,
) -> str:
    used = taken_stems(root)
    slug = slug_stem(prompt or "")
    if slug and not is_generic_stem(slug):
        return unique_stem(root, slug, used)
    for img in images or []:
        owner = owner_stem(root, Path(img))
        if owner:
            return unique_stem(root, owner, used, keep=True)
    for img in images or []:
        cap = caption_from_image(Path(img))
        if cap and not is_generic_stem(cap):
            return unique_stem(root, cap, used)
    return next_generic_stem(root, used)


def stem_dir(root: Path, stem: str) -> Path:
    return root / "out" / check_stem(stem)


def resolve_under_root(root: Path, path: str | Path) -> Path:
    raw = Path(path)
    p = raw if raw.is_absolute() else (root / raw)
    p = p.resolve()
    root_r = root.resolve()
    if p != root_r and root_r not in p.parents:
        raise ValueError(f"path escapes factory root: {path}")
    return p


def pick_glb(root: Path, stem: str, lod: str | None = None) -> Path | None:
    d = stem_dir(root, stem)
    names = LOD_FILES.get(lod or "", DEFAULT_GLB)
    if lod and lod not in LOD_FILES:
        names = DEFAULT_GLB
    for name in names:
        p = d / name
        if p.is_file() and p.stat().st_size >= 64:
            return p
    return None
