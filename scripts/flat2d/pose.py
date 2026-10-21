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
