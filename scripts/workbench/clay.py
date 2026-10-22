#!/usr/bin/env python3
"""Clay: an asset blocked out as signed-distance blobs.

A shape model guesses depth from one photo, so a hay bale comes back a rounded box. Some things are
quicker to state than to guess: "a cylinder 1.2 m across, 0.9 m long, hay-tan". A blob is one such
statement. Blobs are applied in list order: `add` unions a blob in, `carve` cuts it out, and `blend` is the
smooth-min radius that rounds the joint (0 = a hard edge). `mirror` also places the blob reflected across
X = 0, the plane the pipeline treats as left/right symmetry.

Field names follow Clide3D (github.com/addamb/clide3d) so the two projects can share scenes later. Clide3D
has no Cylinder; its nearest form is a Lathe (a revolved profile), so a Cylinder is saved with a note
saying so, and a Lathe with cylinder dims loads as a Cylinder.

Units are metres, Y is up, rotation is Euler XYZ in degrees, colour is sRGB 0..1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Form type -> its dims and their defaults (None = required). Sizes are full extents, not half.
FORMS = {
    "Sphere": {"radius": None},
    "Ellipsoid": {"radii": None},          # [x, y, z] semi-axes
    "Capsule": {"radius": None, "height": None},   # height: total, caps included, along local Y
    "Box": {"size": None, "rounding": 0.0},        # size: [x, y, z]
    "Torus": {"major": None, "minor": None},       # ring in the local XZ plane
    "Cylinder": {"radius": None, "height": None},  # along local Y
}
LATHE_NOTE = "Clide3D has no Cylinder; it calls this a Lathe (a revolved rectangle profile)."
OPS = ("add", "carve")


@dataclass
class Form:
    type: str
    dims: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Form":
        kind = d.get("type")
        if kind == "Lathe":
            kind = "Cylinder"
        if kind not in FORMS:
            raise ValueError(f"form type {d.get('type')!r} is not one of {', '.join(FORMS)}")
        dims = {}
        for key, default in FORMS[kind].items():
            value = d.get(key, default)
            if value is None:
                raise ValueError(f"{kind} needs {key!r}")
            if isinstance(value, (list, tuple)):
                value = [float(v) for v in value]
                if len(value) != 3:
                    raise ValueError(f"{kind}.{key} needs 3 numbers")
            else:
                value = float(value)
            dims[key] = value
        return cls(kind, dims)

    def to_dict(self) -> dict:
        out = {"type": self.type, **self.dims}
        if self.type == "Cylinder":
            out["note"] = LATHE_NOTE
        return out

    def half_extent(self) -> np.ndarray:
        d = self.dims
        if self.type == "Sphere":
            return np.full(3, d["radius"])
        if self.type == "Ellipsoid":
            return np.abs(np.asarray(d["radii"]))
        if self.type in ("Capsule", "Cylinder"):
            return np.array([d["radius"], max(d["height"] / 2, d["radius"]), d["radius"]])
        if self.type == "Box":
            return np.abs(np.asarray(d["size"])) / 2
        r = d["major"] + d["minor"]
        return np.array([r, d["minor"], r])

    def sdf(self, p: np.ndarray) -> np.ndarray:
        """Signed distance at local points (N, 3): negative inside."""
        d = self.dims
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        if self.type == "Sphere":
            return np.linalg.norm(p, axis=1) - d["radius"]
        if self.type == "Ellipsoid":
            # Inigo Quilez's bound: exact on the surface, close enough near it for meshing.
            r = np.maximum(np.abs(np.asarray(d["radii"])), 1e-6)
            k0 = np.linalg.norm(p / r, axis=1)
            k1 = np.linalg.norm(p / (r * r), axis=1)
            return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)
        if self.type == "Capsule":
            half = max(d["height"] / 2 - d["radius"], 0.0)
            q = p.copy()
            q[:, 1] = y - np.clip(y, -half, half)
            return np.linalg.norm(q, axis=1) - d["radius"]
        if self.type == "Box":
            rr = min(d["rounding"], float(np.min(np.abs(d["size"]))) / 2)
            q = np.abs(p) - (np.abs(np.asarray(d["size"])) / 2 - rr)
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(q.max(axis=1), 0.0)
            return outside + inside - rr
        if self.type == "Torus":
            ring = np.hypot(x, z) - d["major"]
            return np.hypot(ring, y) - d["minor"]
        # Cylinder
        q = np.stack([np.hypot(x, z) - d["radius"], np.abs(y) - d["height"] / 2], axis=1)
        return np.minimum(q.max(axis=1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1)


def _vec3(value, name: str) -> list[float]:
    v = [float(x) for x in value]
    if len(v) != 3:
        raise ValueError(f"{name} needs 3 numbers")
    return v


def parse_color(value) -> list[float]:
    """sRGB as [r, g, b] in 0..1, [r, g, b] in 0..255, or '#rrggbb'."""
    if isinstance(value, str):
        h = value.lstrip("#")
        if len(h) != 6:
            raise ValueError(f"colour {value!r} is not #rrggbb")
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    rgb = [float(c) for c in value][:3]
    if len(rgb) != 3:
        raise ValueError("colour needs 3 channels")
    if max(rgb) > 1.0:
        rgb = [c / 255 for c in rgb]
    return [min(max(c, 0.0), 1.0) for c in rgb]


@dataclass
class Blob:
    name: str
    form: Form
    op: str = "add"
    blend: float = 0.0
    color: list = field(default_factory=lambda: [0.8, 0.8, 0.8])
    mirror: bool = False
    translation: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @classmethod
    def from_dict(cls, d: dict) -> "Blob":
        if not d.get("name"):
            raise ValueError("blob needs a name")
        if "form" not in d:
            raise ValueError(f"blob {d['name']!r} needs a form")
        op = d.get("op", "add")
        if op not in OPS:
            raise ValueError(f"blob {d['name']!r}: op {op!r} is not add or carve")
        return cls(
            name=str(d["name"]),
            form=Form.from_dict(d["form"]),
            op=op,
            blend=max(float(d.get("blend", 0.0)), 0.0),
            color=parse_color(d.get("color", [0.8, 0.8, 0.8])),
            mirror=bool(d.get("mirror", False)),
            translation=_vec3(d.get("translation", [0, 0, 0]), "translation"),
            rotation=_vec3(d.get("rotation", [0, 0, 0]), "rotation"),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "form": self.form.to_dict(),
            "op": self.op,
            "blend": self.blend,
            "color": [round(c, 4) for c in self.color],
            "mirror": self.mirror,
            "translation": list(self.translation),
            "rotation": list(self.rotation),
        }

    def _rot(self) -> Rotation:
        return Rotation.from_euler("xyz", self.rotation, degrees=True)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        """This blob's own distance at world points, mirror copy included."""
        inv = self._rot().inv()
        t = np.asarray(self.translation, dtype=np.float64)
        d = self.form.sdf(inv.apply(p - t))
        if self.mirror:
            # The reflected copy seen from p is the original seen from p reflected; min() is right even when
            # the blob straddles X = 0 (|x| folding is not).
            q = p * np.array([-1.0, 1.0, 1.0])
            d = np.minimum(d, self.form.sdf(inv.apply(q - t)))
        return d

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = np.abs(self._rot().as_matrix()) @ self.form.half_extent()
        t = np.asarray(self.translation, dtype=np.float64)
        lo, hi = t - half, t + half
        if self.mirror:
            lo = np.minimum(lo, (t * [-1, 1, 1]) - half)
            hi = np.maximum(hi, (t * [-1, 1, 1]) + half)
        return lo - self.blend, hi + self.blend


def smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum; k is the blend radius (0 = plain min)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


FAR = 1e9


def sdf(blobs: list[Blob], p: np.ndarray) -> np.ndarray:
    """Distance to the whole clay at world points (N, 3)."""
    p = np.asarray(p, dtype=np.float64)
    d = np.full(len(p), FAR)
    for b in blobs:
        di = b.sdf(p)
        if b.op == "add":
            d = smin(d, di, b.blend)
        else:
            d = -smin(-d, di, b.blend)
    return d


def colour_at(blobs: list[Blob], p: np.ndarray) -> np.ndarray:
    """sRGB 0..1 of the nearest added blob at each point (a carved face shows what it cut into)."""
    adds = [b for b in blobs if b.op == "add"]
    if not adds:
        return np.full((len(p), 3), 0.8)
    dist = np.stack([b.sdf(np.asarray(p, dtype=np.float64)) for b in adds], axis=1)
    palette = np.asarray([b.color for b in adds])
    return palette[dist.argmin(axis=1)]


def bounds(blobs: list[Blob]) -> tuple[np.ndarray, np.ndarray]:
    adds = [b.bounds() for b in blobs if b.op == "add"]
    if not adds:
        raise ValueError("clay has no added blobs")
    return np.min([lo for lo, _ in adds], axis=0), np.max([hi for _, hi in adds], axis=0)


def from_json(data) -> list[Blob]:
    items = data.get("blobs", []) if isinstance(data, dict) else data
    blobs = [Blob.from_dict(d) for d in items]
    names = [b.name for b in blobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"blob names repeat: {', '.join(dup)}")
    return blobs


def to_json(blobs: list[Blob]) -> dict:
    return {"units": "m", "up": "Y", "blobs": [b.to_dict() for b in blobs]}


def load(path: str | Path) -> list[Blob]:
    return from_json(json.loads(Path(path).read_text()))


def save(blobs: list[Blob], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_json(blobs), indent=2) + "\n")
