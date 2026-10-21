#!/usr/bin/env python3
"""Facing: which way a picture's subject points, and whether a placed mesh agrees.

    .venv/bin/python scripts/facing.py image out/<stem>/flat/source.png
    .venv/bin/python scripts/facing.py mesh out/<stem>/psx.glb --declared right [--cls avian]

Contract (design section 8): `facing` is the side of the PICTURE the head points to, as you look at it, never the
animal's own left or right. Values are camera | left | right | away. psx_lowpoly turns every declared facing so
the head ends at glTF +Z.

detect_image(png) asks the local vision model (Ollama, the rig_fit.VLM_Q model and one-word pattern) two
questions: which edge the head is nearest, and whether the picture is a strict side profile. A third ask on the
mirrored picture checks the first answer (a left/right answer must swap), which gives the confidence.

check_mesh(glb, cls, declared) votes on where the head of a placed mesh is: rig_fit.geometric_cues (toes, narrow
end, red comb) plus the vision model on rig_render.textured_views from cameras at +X, -X, +Z and -Z (falls back to
rig_fit.vlm_cue, +X/-X only, if the CPU render is unavailable). flipped_needed is set only when the toes AND the
vision model both put the head at -Z.

Every vision call goes through ask(png, question); tests patch it. FACING_VLM=off turns the model off.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FACINGS = ("camera", "left", "right", "away")
OPPOSITE = {"right": "left", "left": "right", "camera": "away", "away": "camera"}
ACCEPT = 0.6  # the confidence a draw needs before it is accepted without a redraw

HEAD_Q = ("Which edge of the picture is the animal's head nearest: LEFT, RIGHT, or is it FACING the camera, "
          "or turned AWAY? Answer one word.")
# The design's literal "Is this a strict side profile? YES/NO." said NO to clean side views drawn from the
# canonical prompt (qwen3.8:27b, 2026-10-20); this wording says YES to those and NO to front and 3/4 views.
PROFILE_Q = ("Is the body seen side-on, so that its length runs across the picture from left to right? "
             "Answer YES or NO.")
MESH_Q = ("This is a rendered view of a 3D model. If it is a creature or character with a head, which edge of the "
          "picture is the head nearest: LEFT, RIGHT, or is it FACING the camera, or turned AWAY? If it has no head "
          "or face (an object, prop or box), answer NONE. Answer one word.")

# One-word answers to facing values. CAMERA / FRONT are what models say for FACING.
WORDS = {"LEFT": "left", "RIGHT": "right", "FACING": "camera", "CAMERA": "camera", "FRONT": "camera",
         "TOWARD": "camera", "TOWARDS": "camera", "AWAY": "away", "BACK": "away", "NONE": "none"}

# Mesh views (rig_render convention): the +Z sign each answer gives the head.
# side: camera +X, image right is -Z. xside: camera -X, image right is +Z. front: camera +Z. back: camera -Z.
VIEW_SIGN = {
    "side": {"left": 1, "right": -1},
    "xside": {"left": -1, "right": 1},
    "front": {"camera": 1, "away": -1},
    "back": {"camera": -1, "away": 1},
}
MESH_VIEWS = ("side", "xside", "front", "back")


class VLMUnavailable(RuntimeError):
    pass


def _model() -> str:
    return os.environ.get("RIG_VLM_MODEL", "qwen3.8:27b")


def num_ctx() -> int:
    """The context the vision model is loaded with. ollama's VRAM-based default on this machine is 262144, which
    wires about 41 GB for qwen3.8:27b; one picture and a one-line question need under 1k tokens, and 8192 wires
    about 22 GB (measured 2026-10-20). VLM_NUM_CTX overrides it."""
    try:
        return max(2048, int(os.environ.get("VLM_NUM_CTX") or 8192))
    except ValueError:
        return 8192


def ask(png, question: str, keep: bool = False) -> str:
    """One vision question about one PNG; the raw answer text. keep=True leaves the model loaded for the next ask
    (keep_alive 60s); the last ask of a batch passes keep=False, which unloads it (keep_alive 0, like rig_fit)."""
    if (os.environ.get("FACING_VLM") or "").strip().lower() in {"off", "0", "no", "false"}:
        raise VLMUnavailable("FACING_VLM=off")
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
    body = {"model": _model(), "prompt": question, "stream": False, "think": False,
            "keep_alive": "60s" if keep else 0, "options": {"temperature": 0, "num_ctx": num_ctx()},
            "images": [base64.b64encode(Path(png).read_bytes()).decode()]}
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(),
                                                          {"Content-Type": "application/json"}), timeout=600)
        return str(json.loads(r.read()).get("response") or "")
    except Exception as e:  # noqa: BLE001 - no model: the caller records it and does not vote
        raise VLMUnavailable(f"vision model unavailable ({e})") from e


def release() -> None:
    """Unload the vision model now (ollama keep_alive 0, no prompt). Never raises."""
    if (os.environ.get("FACING_VLM") or "").strip().lower() in {"off", "0", "no", "false"}:
        return
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
    try:
        urllib.request.urlopen(urllib.request.Request(url, json.dumps({"model": _model(), "keep_alive": 0}).encode(),
                                                      {"Content-Type": "application/json"}), timeout=10).read()
    except Exception:  # noqa: BLE001 - nothing to unload, or no server
        pass


def first_word(text: str) -> str:
    for w in str(text or "").upper().replace("*", " ").replace('"', " ").split():
        w = w.strip(".,:;!?'()[]")
        if w:
            return w
    return ""


def parse_head(text: str) -> str | None:
    """LEFT / RIGHT / FACING / AWAY (and NONE) to a facing value; None when the answer is not one of them."""
    w = first_word(text)
    if w in WORDS:
        return WORDS[w]
    # "The head is on the RIGHT" and similar: take the only facing word in the answer.
    found = {WORDS[t.strip(".,:;!?'()[]")] for t in str(text or "").upper().split() if t.strip(".,:;!?'()[]") in WORDS}
    return found.pop() if len(found) == 1 else None


def parse_yes(text: str) -> bool | None:
    w = first_word(text)
    return True if w in {"YES", "Y", "TRUE"} else False if w in {"NO", "N", "FALSE"} else None


# ---------------------------------------------------------------- pictures

def detect_image(png) -> dict:
    """{head: left|right|camera|away|None, profile: bool|None, confidence, how:'vlm'|'unavailable', answers}.
    The head question is asked again on the mirrored picture: a left/right answer must swap (camera/away must stay)
    for full confidence."""
    from PIL import Image, ImageOps

    png = Path(png)
    out: dict = {"head": None, "profile": None, "confidence": 0.0, "how": "vlm", "model": _model(), "answers": {}}
    with tempfile.TemporaryDirectory(prefix="facing_") as tmp:
        mirrored = Path(tmp) / f"{png.stem}_mirrored.png"
        ImageOps.mirror(Image.open(png).convert("RGB")).save(mirrored)
        try:
            a_head = ask(png, HEAD_Q, keep=True)
            a_prof = ask(png, PROFILE_Q, keep=True)
            a_mirr = ask(mirrored, HEAD_Q, keep=False)
        except VLMUnavailable as e:
            release()  # an ask that kept the model loaded may have gone through before the failure
            return {**out, "how": "unavailable", "why": str(e)}
    out["answers"] = {"head": a_head.strip()[:80], "profile": a_prof.strip()[:80], "mirrored": a_mirr.strip()[:80]}
    head, prof, mirr = parse_head(a_head), parse_yes(a_prof), parse_head(a_mirr)
    if head == "none":
        head = None
    out["head"], out["profile"] = head, prof
    if head is None:
        return out
    expect = {"left": "right", "right": "left"}.get(head, head)
    conf = 0.9 if mirr == expect else 0.3 if mirr is not None else 0.6
    if prof is None:
        conf -= 0.2
    # A side profile with the head on camera/away (or a non-profile with a side head) is self-contradictory.
    if prof is True and head in ("camera", "away") or prof is False and head in ("left", "right"):
        conf -= 0.1
    out["confidence"] = round(max(conf, 0.0), 2)
    return out


def judge(det: dict, expected: str = "right") -> dict:
    """What a draw does with a detection: {ok, flaws, rank, mirror}. rank orders fallbacks (0 best):
    0 accepted, 1 the head on the wrong side of a clean profile (mirrorable), 2 anything else.
    An unavailable model is never a pass: the flaw 'facing_unavailable' (a new seed cannot fix it)."""
    if det.get("how") == "unavailable":
        return {"ok": False, "flaws": ["facing_unavailable"], "rank": 2, "mirror": False, "checked": False}
    head, prof, conf = det.get("head"), det.get("profile"), float(det.get("confidence") or 0)
    side = expected in ("left", "right")
    if head == expected and (prof is True or not side) and conf >= ACCEPT:
        return {"ok": True, "flaws": [], "rank": 0, "mirror": False, "checked": True}
    flaws = []
    if head != expected:
        flaws.append("facing")
    if side and prof is not True:
        flaws.append("profile")
    if not flaws:
        flaws.append("facing")  # right answer, low confidence
    mirror = side and head == OPPOSITE[expected] and prof is True
    rank = 1 if mirror else (1 if head == expected and prof is True else 2)
    return {"ok": False, "flaws": flaws, "rank": rank, "mirror": mirror, "checked": True}


# ---------------------------------------------------------------- meshes

def _mesh_points(glb: Path) -> tuple[np.ndarray, np.ndarray | None]:
    import rig
    import rig_fit

    doc, binary = rig.read_glb(glb)
    pr = doc["meshes"][0]["primitives"][0]
    P = rig.accessor(doc, binary, pr["attributes"]["POSITION"]).astype(np.float64)
    return P, rig_fit._texel_colours(doc, binary)


def vlm_mesh_vote(glb: Path, out_dir: Path | None = None) -> dict:
    """The vision model on four textured CPU renders (rig_render.textured_views): a +Z head vote of +1/-1/0."""
    try:
        import rig_render
        tmp = None if out_dir else tempfile.mkdtemp(prefix="facing_mesh_")
        pngs = rig_render.textured_views(glb, views=MESH_VIEWS, out_dir=out_dir or tmp)
    except Exception as e:  # noqa: BLE001 - CPU render unavailable: the old Blender +X/-X cue
        return _vlm_cue_fallback(glb, out_dir, f"textured_views failed ({e})")
    answers, signs = {}, {}
    try:
        for i, view in enumerate(MESH_VIEWS):
            text = ask(pngs[view], MESH_Q, keep=i < len(MESH_VIEWS) - 1)
            got = parse_head(text)
            answers[view] = got or first_word(text)
            signs[view] = VIEW_SIGN[view].get(got or "", 0)
    except VLMUnavailable as e:
        return {"vote": 0, "why": str(e), "how": "unavailable", "pngs": pngs if out_dir else None}
    finally:
        if not out_dir and tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    pos = sum(1 for s in signs.values() if s > 0)
    neg = sum(1 for s in signs.values() if s < 0)
    vote = 1 if pos >= 2 and pos > 2 * neg else -1 if neg >= 2 and neg > 2 * pos else 0
    return {"vote": vote, "answers": answers, "signs": signs, "how": "textured_views", "model": _model(),
            "pngs": pngs if out_dir else None}


def _vlm_cue_fallback(glb: Path, out_dir: Path | None, why: str) -> dict:
    try:
        import rig
        import rig_fit
        prefix = Path(out_dir or tempfile.mkdtemp(prefix="facing_cue_")) / "facing"
        got = rig_fit.vlm_cue(Path(glb), rig.blender_path(), prefix)
        got["how"] = "vlm_cue"
        got["why_fallback"] = why
        return got
    except Exception as e:  # noqa: BLE001
        return {"vote": 0, "how": "unavailable", "why": f"{why}; vlm_cue failed ({e})"}


def check_mesh(glb, cls: str = "animal", declared: str | None = None, out_dir=None, vlm: bool = True) -> dict:
    """Where the head of a placed mesh is. psx_lowpoly turns every declared facing to +Z, so +Z is expected.
    {head_axis: '+Z'|'-Z'|None, votes:{toes, narrow, red, vlm}, agrees, flipped_needed, declared, cues, vlm}."""
    glb = Path(glb)
    P, colours = _mesh_points(glb)
    cues = __import__("rig_fit").geometric_cues(P, cls, colours)
    z = cues.get("z", {})
    votes = {k: int(z[k]["vote"]) if k in z else 0 for k in ("toes", "narrow", "red")}
    v = vlm_mesh_vote(glb, Path(out_dir) if out_dir else None) if vlm else {"vote": 0, "how": "off"}
    votes["vlm"] = int(v.get("vote") or 0)
    total = sum(votes.values())
    head_axis = "+Z" if total > 0 else "-Z" if total < 0 else None
    flipped_needed = votes["toes"] < 0 and votes["vlm"] < 0
    agrees = not flipped_needed and votes["toes"] >= 0 and votes["vlm"] >= 0 and total >= 0
    return {"head_axis": head_axis, "votes": votes, "agrees": bool(agrees), "flipped_needed": bool(flipped_needed),
            "declared": declared, "cls": cls, "cues": cues, "vlm": {k: v[k] for k in v if k != "pngs"}}


def flip_why(declared: str | None) -> str:
    said = declared or "camera"
    return (f"Facing corrected: the picture said {said}, the model's toes and the VLM say "
            f"{OPPOSITE.get(said, 'the other way')}.")


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("image", help="which way the head points in a picture")
    a.add_argument("png", type=Path)
    b = sub.add_parser("mesh", help="where the head of a placed GLB is")
    b.add_argument("glb", type=Path)
    b.add_argument("--declared", choices=FACINGS, default=None)
    b.add_argument("--cls", default="animal")
    b.add_argument("--out", type=Path, default=None, help="keep the renders here")
    b.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "image":
        res = detect_image(args.png)
    else:
        res = check_mesh(args.glb, args.cls, args.declared, out_dir=args.out, vlm=not args.no_vlm)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
