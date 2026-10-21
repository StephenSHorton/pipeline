#!/usr/bin/env python3
"""Factory paths and generate argv."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

LOD_FILES = {
    "mesh": ("mesh.glb",),
    "hero": ("mesh_hero.glb", "mesh.glb"),
    "lookdev": ("mesh_lookdev.glb", "mesh.glb"),
    "game": ("mesh_game.glb", "mesh.glb"),
    "proxy": ("mesh_proxy.glb", "mesh.glb"),
    "mv": ("mesh_mv.glb", "mesh_hero.glb", "mesh.glb"),
    "psx": ("psx.glb",),
}
DEFAULT_GLB = ("mesh.glb", "mesh_hero.glb", "mesh_mv.glb", "mesh_proxy.glb")

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
    """generate.sh --quality psx options from a request, checked; unknown or bad values are dropped, except an
    unknown facing, which raises ValueError (a typo there would build the model turned the wrong way). A missing
    facing is fine: generate.sh defaults to camera."""
    out: dict[str, str] = {}
    for key, value in (raw or {}).items():
        if key not in PSX_KEYS or value in (None, ""):
            continue
        text = str(value).strip()
        if key == "facing":
            if text not in PSX_FACING:
                raise ValueError(f"facing must be one of {', '.join(sorted(PSX_FACING))} (the side of the picture "
                                 f"the head points to), not {text!r}")
            out[key] = text
        elif key == "up" and text in {"z", "y"}:
            out[key] = text
        elif key in {"faces", "atlas"} and text.isdigit() and 0 < int(text) <= 65536:
            out[key] = text
        elif key == "height":
            try:
                if 0 < float(text) <= 100000:
                    out[key] = text
            except ValueError:
                pass
        elif key == "name" and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", text):
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
