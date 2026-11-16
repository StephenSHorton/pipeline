"""The canonical draw pose for creatures (design section 7).

A pipeline-drawn creature is drawn the way the rig wants it: a strict side profile, head toward the RIGHT edge of
the picture (psx facing 'right', so the head ends at +Z), standing on both (or all four) legs. Humanoids are drawn
from the front in an A-pose, facing the camera.

pose_for(intent, prompt) is the kind ('avian', 'quadruped', 'creature', 'humanoid') or None for anything that is not
a creature headed for a 3D model. still_prompt(text, CREATURE_POSE[kind]) is the prompt.
"""
from __future__ import annotations

import re
from typing import Any

_TAIL = ("A single creature only, centered, isolated on a solid flat magenta #FF00FF background. "
         "Flat even lighting from the front. No shadow, no drop shadow, no cast shadow, no reflection, "
         "no ground plane, no text.")

CREATURE_POSE = {
    "avian": (
        "Full body, seen exactly from the side in strict profile view, orthographic, the whole creature visible "
        "from beak to tail and feet. Standing upright on both legs on flat level ground, both legs straight down "
        "and visible, feet flat and slightly apart. Head and beak pointing toward the RIGHT edge of the image, eye "
        "level, tail toward the LEFT edge. Wings folded against the body, mouth closed, neutral relaxed rest pose, "
        "no motion. Not three-quarter, not turned toward the camera. " + _TAIL
    ),
    "quadruped": (
        "Full body, seen exactly from the side in strict profile view, orthographic, the whole creature visible "
        "from nose to tail and hooves or paws. Standing square on flat level ground, all four legs visible and "
        "straight, feet flat. Head pointing toward the RIGHT edge of the image, eye level, tail toward the LEFT "
        "edge. Mouth closed, neutral relaxed rest pose, no motion. Not three-quarter, not turned toward the "
        "camera. " + _TAIL
    ),
    # A creature the word lists do not place (a mech, a spider, a monster): profile right, every leg visible.
    "creature": (
        "Full body, seen exactly from the side in strict profile view, orthographic, the whole creature visible "
        "from head to tail and feet. Standing on flat level ground, every leg straight down and visible, feet "
        "flat. Head pointing toward the RIGHT edge of the image, eye level, tail toward the LEFT edge. Mouth "
        "closed, neutral relaxed rest pose, no motion. Not three-quarter, not turned toward the camera. " + _TAIL
    ),
    "humanoid": (
        "Full body, front view facing the camera, orthographic, the whole figure visible from head to feet. "
        "Standing upright in an A-pose, arms straight and held 30° from the body, legs straight and slightly "
        "apart, feet flat on flat level ground. Eye level, mouth closed, neutral relaxed expression, no motion. "
        "Not three-quarter, not turned to the side. A single character only, centered, isolated on a solid flat "
        "magenta #FF00FF background. Flat even lighting from the front. No shadow, no drop shadow, no cast "
        "shadow, no reflection, no ground plane, no text."
    ),
}

# What each kind is drawn as, for key.json, intent.json and the facing check.
POSE_NAME = {"avian": "profile_right", "quadruped": "profile_right", "creature": "profile_right",
             "humanoid": "front_apose"}
FACING = {"avian": "right", "quadruped": "right", "creature": "right", "humanoid": "camera"}
PLAN_HINT = {"avian": "ground_bird", "quadruped": "horse", "humanoid": "biped", "creature": None}

CREATURE_OUTPUTS = {"ps1", "model", "stack"}

AVIAN = ("bird", "chicken", "hen", "rooster", "cockerel", "chick", "duck", "duckling", "goose", "swan", "turkey",
         "owl", "parrot", "pigeon", "dove", "crow", "raven", "seagull", "gull", "penguin", "ostrich", "emu",
         "peacock", "eagle", "hawk", "falcon", "sparrow", "robin", "flamingo", "heron", "stork", "pheasant",
         "quail", "dodo", "kiwi", "vulture")
QUADRUPED = ("horse", "pony", "foal", "cow", "bull", "calf", "ox", "cattle", "dog", "puppy", "hound", "wolf", "fox",
             "cat", "kitten", "lion", "tiger", "leopard", "panther", "pig", "piglet", "boar", "hog", "sheep", "lamb",
             "ram", "goat", "deer", "stag", "elk", "moose", "reindeer", "donkey", "mule", "camel", "llama", "alpaca",
             "rabbit", "bunny", "hare", "mouse", "rat", "bear", "elephant", "giraffe", "zebra", "rhino", "hippo",
             "buffalo", "bison", "yak", "raccoon", "badger", "otter", "squirrel", "hedgehog", "lizard", "crocodile",
             "alligator", "dinosaur", "triceratops")
# A creature word next to one of these is a thing, not the creature ("a chicken drumstick", "a farmer's hat").
PROP = ("hat", "cap", "helmet", "mask", "costume", "saddle", "drumstick", "nugget", "egg", "feather", "skull",
        "bone", "statue", "figurine", "plush", "sign", "logo", "icon", "collar", "leash", "bowl", "kennel",
        "coop", "barn", "cage", "trophy", "toy", "badge", "emblem", "weathervane")
HUMANOID_SUBJECTS = {"humanoid", "character", "person", "human", "biped"}

_WORD = re.compile(r"[a-z]+")


def _words(prompt: str | None) -> list[str]:
    out = []
    for w in _WORD.findall((prompt or "").lower()):
        out.append(w)
        if len(w) > 3 and w.endswith("s"):  # plurals: "chickens", "horses"
            out.append(w[:-1])
            if w.endswith("es"):
                out.append(w[:-2])
    return out


def word_kind(prompt: str | None) -> str | None:
    words = set(_words(prompt))
    if words & set(PROP):
        return None
    if words & set(AVIAN):
        return "avian"
    if words & set(QUADRUPED):
        return "quadruped"
    return None


def pose_for(intent: Any, prompt: str | None = None) -> str | None:
    """The CREATURE_POSE kind for a draw, or None (draw it the old way: a floating object).
    Only a 3D-bound intent (output ps1, model or stack) gets a pose. intent.kind / intent.subject name it when
    given (subject 'creature' with no word match is the generic 'creature'); otherwise the word lists decide.
    Humanoids come only from intent.subject/kind: a person word in a prompt is too often a hat or a prop."""
    if not isinstance(intent, dict) or intent.get("output") not in CREATURE_OUTPUTS:
        return None
    kind = str(intent.get("kind") or "").strip().lower()
    if kind in CREATURE_POSE:
        return kind
    subject = str(intent.get("subject") or "").strip().lower()
    if subject in HUMANOID_SUBJECTS:
        return "humanoid"
    if subject and subject not in {"creature", "animal"}:
        return None  # 'prop', 'vehicle', ...: never posed
    found = word_kind(prompt) or word_kind(intent.get("name"))
    if found:
        return found
    return "creature" if subject in {"creature", "animal"} else None


def is_creature(intent: Any, prompt: str | None = None) -> bool:
    return pose_for(intent, prompt) is not None


# ---------------------------------------------------------------- the four-view sheet (plan 1.1)

# Panel order on every sheet, left to right (a 2x2 grid reads the same order, row by row).
PANELS = ("front", "right", "back", "left")

# The pose for a sheet: the same standard pose as CREATURE_POSE, without the one-view framing (centered, single
# creature) and without any ground or baseline phrase: a drawn baseline is a full-width line the split must drop.
SHEET_POSE = {
    "avian": ("Standing upright on both legs, both legs straight down and visible, feet flat and slightly apart, "
              "wings folded against the body, mouth closed, eye level, neutral relaxed rest pose, no motion."),
    "quadruped": ("Standing square on all four legs, every leg straight and visible, feet flat, mouth closed, "
                  "eye level, neutral relaxed rest pose, no motion."),
    "creature": ("Standing with every leg straight down and visible, feet flat, mouth closed, eye level, neutral "
                 "relaxed rest pose, no motion."),
    "humanoid": ("Standing upright in an A-pose, arms straight and held 30° from the body, legs straight and "
                 "slightly apart, feet flat, mouth closed, neutral relaxed expression, no motion."),
}
# A turned head on a side view cannot be baked: no one camera sees it, so its face and far ear landed on the pig's
# shoulders (retexture fit IoU 0.79 for those panels, 0.93 for front and back). Asked for on every creature sheet.
SHEET_HEAD = {"humanoid": "The head faces the same way as the body in every view, never turned toward the viewer.",
              "creature": ("The head points straight along the body in every view, never turned toward the viewer: "
                           "in the two side views it is a strict profile, only the near eye and the near ear "
                           "visible.")}
SHEET_FLOAT = "The object floats in the air in every view, upright, not standing on anything."
SHEET_TAIL = ("Exactly four views, no more, {where}, each whole and separate, the same size, with wide empty gaps between "
              "them. Plain flat uniform magenta #FF00FF background with no gradient, no vignette and no floor. Flat even "
              "lighting, every view lit the same, no highlights and no shading. No shadow, no text, no labels, no "
              "borders.")
# Where each panel sits, by layout. The wording "View n (where)" plus "turned to face RIGHT, its head at the right
# and its tail at the left" is what made Qwen-Image-2512 draw the two profiles facing opposite ways (scratchpad
# qmv/p2/ab.md); "head pointing toward the RIGHT edge" drew both profiles facing left.
SHEET_WHERE = {"strip": ("in one row", "from left to right", ("", "", "", ""), "evenly spaced in one row"),
               "grid": ("in a 2 by 2 grid, read row by row", "", (" (top left)", " (top right)", " (bottom left)",
                                                               " (bottom right)"), "one in each quarter of the picture")}


def sheet_views(kind: str | None) -> dict[str, str]:
    """What each panel shows. A side-on creature's profiles say which way it faces (right panel: head toward the
    RIGHT edge, the canonical view); a prop's say which side of the object is seen."""
    if kind == "humanoid":
        return {"front": "front, facing the viewer",
                "right": "the character turned to face RIGHT, its face toward the right edge",
                "back": "back, facing away from the viewer",
                "left": "the character turned to face LEFT, its face toward the left edge"}
    if kind:
        return {"front": "front, facing the viewer",
                "right": "the creature turned to face RIGHT, its head at the right and its tail at the left",
                "back": "back, facing away from the viewer",
                "left": "the creature turned to face LEFT, its head at the left and its tail at the right"}
    return {"front": "front view", "right": "right side view", "back": "back view", "left": "left side view"}


def sheet_prompt(text: str, kind: str | None = None, layout: str = "grid") -> str:
    """One picture holding four orthographic views of the subject in PANELS order: front | right | back | left,
    row by row in a 2x2 grid (layout 'grid', the default: a tall prop in a wide strip came back five times) or in
    one row (layout 'strip'). kind is a CREATURE_POSE kind (the pose clause) or None for a prop (the
    floating-object clause). No baseline phrase: the views are split at their magenta gaps."""
    rows, order, cells, spread = SHEET_WHERE["grid" if layout == "grid" else "strip"]
    v = sheet_views(kind)
    listing = " ".join(f"View {i + 1}{cells[i]}: {v[pid]}." for i, pid in enumerate(PANELS))
    if kind:
        who = "character" if kind == "humanoid" else "creature"
        head = (f"Character turnaround sheet {rows}, the same {who} turning in place, four orthographic full-body "
                f"views{' ' + order if order else ''}.")
        same = ("The two side views are mirror images of each other and face in opposite directions. "
                + SHEET_POSE.get(kind, SHEET_POSE["creature"]) + " "
                + SHEET_HEAD["humanoid" if kind == "humanoid" else "creature"])
    else:
        head = (f"Product turnaround sheet {rows}, the same single object turning in place, four orthographic "
                f"views{' ' + order if order else ''}.")
        same = "One object in each view, the same size, proportions and colours. " + SHEET_FLOAT
    return " ".join([" ".join((text or "").split()).rstrip(".") + ".", head, listing, same,
                     SHEET_TAIL.format(where=spread)])


def canonical_view(kind: str | None) -> str:
    """The panel that becomes flat/source.png: the head-right profile for a side-on creature, the front otherwise."""
    return "right" if kind and FACING.get(kind) == "right" else "front"
