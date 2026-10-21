"""Read and write out/<stem>/flat/."""
from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image

from flat2d.parts import TOL, reading_order, split_image, split_sam
from flat2d.sheet import flat_dir, lift_binary, read_json, sam_checkpoint, unique_ids, write_json

SOURCE = "source.png"
CENTER_SHEET = re.compile(r"^sheets/[a-z0-9][a-z0-9-]{0,40}\.png$")


class FlatError(ValueError):
    pass


def source_path(root: Path, stem: str) -> Path:
    return flat_dir(root, stem) / SOURCE


def key_path(root: Path, stem: str) -> Path:
    return flat_dir(root, stem) / "key.json"


def read_key(root: Path, stem: str) -> tuple[int, int, int] | None:
    data = read_json(key_path(root, stem))
    rgb = (data or {}).get("rgb")
    if not isinstance(rgb, list) or len(rgb) != 3:
        return None
    try:
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    except (TypeError, ValueError):
        return None


def read_tol(root: Path, stem: str) -> int:
    """How far from the key still counts as ground. Older key files have none."""
    tol = (read_json(key_path(root, stem)) or {}).get("tol")
    return int(tol) if isinstance(tol, (int, float)) and tol > 0 else TOL


def save_source(root: Path, stem: str, data: bytes) -> dict:
    if not data:
        raise FlatError("empty image")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise FlatError(f"not an image: {e}") from e
    dest = source_path(root, stem)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGBA").save(dest)
    key = key_path(root, stem)
    if key.is_file():
        key.unlink()
    write_center(root, stem, "source")
    return {"ok": True, "stem": stem, "source": f"flat/{SOURCE}", "bytes": dest.stat().st_size}


def write_center(root: Path, stem: str, center: str) -> None:
    write_json(flat_dir(root, stem) / "view.json", {"center": center})


def piece_names(raw) -> list[str]:
    """Comma-separated labels from the pieces prompt. Order follows the crops."""
    if isinstance(raw, list):
        items = [str(item) for item in raw]
    else:
        items = str(raw or "").replace(";", ",").replace("\n", ",").split(",")
    names = []
    for item in items:
        text = " ".join(item.split())
        if text:
            names.append(text[:48])
    return names


def vision_mask(src: Path) -> bytearray | None:
    """Apple Vision's subject mask for a still, 1 = subject. None off the Mac, or when
    Vision found no subject (lift-subject then falls back to a corner fill; the key does that better)."""
    import subprocess
    import tempfile

    binary = lift_binary()
    if binary is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "lifted.png"
        try:
            run = subprocess.run([str(binary), str(src), str(out)], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if run.returncode != 0 or ": vision ->" not in run.stdout or not out.is_file():
            return None
        with Image.open(out) as lifted:
            alpha = lifted.convert("RGBA").getchannel("A").tobytes()
        return bytearray(1 if a >= 128 else 0 for a in alpha)


def split_source(root: Path, stem: str, names=None) -> dict:
    src = source_path(root, stem)
    if not src.is_file():
        raise FlatError("no source image")
    im = Image.open(src)
    key = None
    ckpt = sam_checkpoint()
    labels = piece_names(names)
    if ckpt is not None:
        try:
            crops = split_sam(im, ckpt)
        except Exception as e:
            raise FlatError(f"sam2 failed: {e}") from e
        method = "sam2"
    else:
        key = read_key(root, stem)
        tol = read_tol(root, stem)
        subject = vision_mask(src) if key is not None else None
        crops = split_image(im, key=key, tol=tol, subject=subject)
        method = "vision" if subject is not None else "key"
    part_dir = flat_dir(root, stem) / "parts"
    if part_dir.is_dir():
        for old in part_dir.glob("part-*.png"):
            old.unlink()
    if labels:
        # Names are typed the way the picture reads, so hand them out in that order.
        crops = reading_order(crops)
    # A keyed still with one shape on it is one object: its cutout is the part.
    single = len(crops) == 1 and key is not None
    if len(crops) < 2 and not single:
        if len(crops) == 1:
            # One connected shape is the whole picture. Saving it as part-01 is not a cut.
            whole, keyed, reason = True, False, "whole"
        elif method == "sam2" or key is None:
            whole, keyed, reason = False, False, "none"
        else:
            raw = im.convert("RGB").tobytes()
            kr, kg, kb = key
            off = False
            for i in range(0, len(raw), 3):
                if abs(raw[i] - kr) + abs(raw[i + 1] - kg) + abs(raw[i + 2] - kb) > tol:
                    off = True
                    break
            whole = False
            keyed, reason = (False, "specks") if off else (True, "keyed")
        payload = {
            "ok": True,
            "stem": stem,
            "method": method,
            "parts": [],
            "whole": whole,
            "keyed": keyed,
            "reason": reason,
        }
        write_json(flat_dir(root, stem) / "parts.json", payload)
        return payload
    part_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, crop in enumerate(crops):
        name = f"part-{i + 1:02d}.png"
        crop.save(part_dir / name)
        rows.append(
            {
                "id": f"part-{i + 1:02d}",
                "file": f"parts/{name}",
                "name": labels[i] if i < len(labels) else "",
                "role": "other",
                "motion": "",
                "projectile": "",
                "sound": "",
                "line": "",
                "speaker": "af_heart",
                "voice": "",
                "w": crop.size[0],
                "h": crop.size[1],
            }
        )
    payload = {
        "ok": True,
        "stem": stem,
        "method": method,
        "parts": rows,
        "whole": single,
        "keyed": False,
        "reason": "whole" if single else "",
    }
    write_json(flat_dir(root, stem) / "parts.json", payload)
    return payload


def save_sheet(root: Path, stem: str, parts: list, note: str) -> dict:
    rows = unique_ids(parts if isinstance(parts, list) else [])
    payload = {"ok": True, "stem": stem, "note": note or "", "parts": rows}
    write_json(flat_dir(root, stem) / "mechanics.json", payload)
    prev = read_json(flat_dir(root, stem) / "parts.json") or {"parts": []}
    by_file = {str(p.get("file")): p for p in prev.get("parts") or []}
    for row in rows:
        old = by_file.get(row["file"])
        if old:
            old["name"] = row["name"]
            old["role"] = row["role"]
            old["id"] = row["id"]
    if prev.get("parts") is not None:
        write_json(flat_dir(root, stem) / "parts.json", prev)
    return payload


def load_sheet(root: Path, stem: str) -> dict:
    base = flat_dir(root, stem)
    parts = read_json(base / "parts.json")
    mech = read_json(base / "mechanics.json")
    view = read_json(base / "view.json")
    center = ""
    if view is None:
        if source_path(root, stem).is_file():
            center = "source"
    elif isinstance(view, dict):
        raw = view.get("center")
        if raw == "source" or (isinstance(raw, str) and CENTER_SHEET.fullmatch(raw)):
            center = raw
    return {
        "ok": True,
        "stem": stem,
        "source": source_path(root, stem).is_file(),
        "parts": (parts or {}).get("parts") or [],
        "method": (parts or {}).get("method") or "",
        "note": (mech or {}).get("note") or "",
        "mechanics": (mech or {}).get("parts") or [],
        "center": center,
    }
