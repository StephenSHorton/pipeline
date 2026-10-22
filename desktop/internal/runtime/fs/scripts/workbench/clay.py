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

Three things keep a blocked-out shape from reading as a CAD primitive:
- A Cylinder's `rounding` (rim radius, like Box's) and `bulge` (barrel: the radius grows by bulge * radius at
  mid-height, back to the plain radius at the caps). A rolled bale has soft rims and a fat middle.
- A blob's `noise`: {amp, scale, seed, octaves}, seeded gradient noise added to that blob's distance, so the
  surface wanders by up to `amp` metres in features about `scale` across. It lives on the blob, in the blob's
  own frame, rather than on the whole clay: Clide3D's noise is a sculpt brush applied to a finished mesh, but
  here a blob's noise moves and mirrors with the blob, a smooth part can sit beside a rough one, and a carve
  (a cut face, a ground plane) stays as clean as it was authored. Clide3D blobs have no noise field and
  ignore it.
- Ground contact is a carve, not a field: a wide Box carved away below the floor gives a flat contact patch,
  and its `blend` rounds the sag edge. A flatten field on the blob would be a plane in the blob's own frame,
  which tilts with its rotation; a carve is in world space and already composes.

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
    "Cylinder": {"radius": None, "height": None, "rounding": 0.0, "bulge": 0.0},  # along local Y
}
# Dims added after clay.json files existed: written only when they differ from the default, so an old file
# loads and saves back byte for byte. (Box's rounding predates this and is always written.)
OPTIONAL_DIMS = {("Cylinder", "rounding"), ("Cylinder", "bulge")}
BULGE_RANGE = (-0.5, 1.0)   # -0.5 pinches the waist to half; past 1 the barrel is nearly a ball
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
        dims = {k: v for k, v in self.dims.items()
                if (self.type, k) not in OPTIONAL_DIMS or v != FORMS[self.type][k]}
        out = {"type": self.type, **dims}
        if self.type == "Cylinder":
            out["note"] = LATHE_NOTE
        return out

    def half_extent(self) -> np.ndarray:
        d = self.dims
        if self.type == "Sphere":
            return np.full(3, d["radius"])
        if self.type == "Ellipsoid":
            return np.abs(np.asarray(d["radii"]))
        if self.type == "Capsule":
            return np.array([d["radius"], max(d["height"] / 2, d["radius"]), d["radius"]])
        if self.type == "Cylinder":
            r = d["radius"] * (1 + max(_bulge(d), 0.0))
            return np.array([r, max(d["height"] / 2, d["radius"]), r])
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
        # Cylinder. Rounding is the Box trick in 2D (shrink by the rim radius, take the distance, grow back), so a
        # rounded straight cylinder is still exact.
        r, half = d["radius"], d["height"] / 2
        rr = min(max(d["rounding"], 0.0), r, half)
        radial = np.hypot(x, z) - r
        bulge = _bulge(d)
        k = 1.0
        if bulge:
            # Barrel: a parabola in y, bulge * r at the middle and 0 at the caps (staves meet the lid at an angle,
            # as on a real barrel). Measured straight out, a sloped wall reads too far by sqrt(1 + slope^2), so the
            # whole distance is divided by it. Only the result: dividing radial before the rim rounding would round
            # in squashed units where the slope peaks (the caps), shrinking the flat cap face to r - k * rr, nearly
            # a point at large rounding. A positive scale leaves the zero set where it is, so the surface is
            # exactly the authored one and the field stays a lower-bound distance estimate.
            t = np.clip(y / half, -1.0, 1.0)
            radial = radial - bulge * r * (1.0 - t * t)
            slope = 2.0 * bulge * r * t / half
            k = np.sqrt(1.0 + slope * slope)
        q = np.stack([radial + rr, np.abs(y) - half + rr], axis=1)
        return (np.minimum(q.max(axis=1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1) - rr) / k


def _bulge(d: dict) -> float:
    return float(np.clip(d.get("bulge", 0.0), *BULGE_RANGE))


# ------------------------------------------------------------ noise
# Gradient (Perlin) noise from an integer hash rather than a permutation table drawn from an RNG: a seed gives
# the same surface on every machine and numpy version, and nothing is kept between calls.
_GRADIENTS = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1],
                       [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float64)
# Measured over 400k points: one octave stays within about +-1 and its slope under 3.1 per lattice cell.
NOISE_PEAK_SLOPE = 3.1
# The most a blob's noise may tilt its distance field. At 0.5 the field's gradient stays between 0.5 and 1.5,
# so it is still a usable distance estimate: surface nets, the projection step and blends all behave.
NOISE_MAX_SLOPE = 0.5
OCTAVES = (1, 4)


def _hash(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    # Negative lattice indices wrap to uint32 the same way everywhere, and uint32 array products wrap silently.
    h = ((ix.astype(np.uint32) * np.uint32(0x8DA6B343)) ^ (iy.astype(np.uint32) * np.uint32(0xD8163841))
         ^ (iz.astype(np.uint32) * np.uint32(0xCB1AB31F)) ^ np.uint32((seed * 0x9E3779B9) & 0xFFFFFFFF))
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return h


def perlin(p: np.ndarray, seed: int) -> np.ndarray:
    """One octave of gradient noise at points (N, 3), lattice spacing 1, values within about +-1."""
    i = np.floor(p).astype(np.int64)
    f = p - i
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)   # quintic fade: no creases along the lattice planes
    corner = []
    for c in range(8):
        o = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1])
        g = _GRADIENTS[_hash(i[:, 0] + o[0], i[:, 1] + o[1], i[:, 2] + o[2], seed) % np.uint32(12)]
        corner.append((g * (f - o)).sum(axis=1))
    x = [corner[k] + (corner[k + 1] - corner[k]) * u[:, 0] for k in (0, 2, 4, 6)]
    y = [x[k] + (x[k + 1] - x[k]) * u[:, 1] for k in (0, 2)]
    return y[0] + (y[1] - y[0]) * u[:, 2]


@dataclass
class Noise:
    """Displacement noise on a blob: up to about `amp` metres, features about `scale` metres across."""
    amp: float
    scale: float
    seed: int = 0
    octaves: int = 2

    @staticmethod
    def max_amp(scale: float, octaves: int) -> float:
        """Largest amp whose slope stays under NOISE_MAX_SLOPE. Octave k has weight 0.5^k at frequency 2^k and
        the sum is divided by its weights, so the worst-case slope is peak * octaves / weights / scale."""
        weights = sum(0.5 ** k for k in range(octaves))
        return NOISE_MAX_SLOPE * scale * weights / (NOISE_PEAK_SLOPE * octaves)

    @classmethod
    def from_dict(cls, d, blob: str) -> "Noise | None":
        if d is None:
            return None
        if not isinstance(d, dict):
            raise ValueError(f"blob {blob!r}: noise must be {{amp, scale, seed?, octaves?}}")
        # amp first: amp 0 is how clay_edit switches noise off, and on a blob without noise there is no scale to
        # merge it with, so {"amp": 0} alone must not be refused for the scale it does not need.
        amp = float(d.get("amp", 0.0))
        if amp <= 0:
            return None
        if "scale" not in d:
            raise ValueError(f"blob {blob!r}: noise needs 'scale' (feature size, m)")
        scale = float(d["scale"])
        if scale <= 0:
            raise ValueError(f"blob {blob!r}: noise scale must be > 0")
        octaves = int(d.get("octaves", 2))
        if not OCTAVES[0] <= octaves <= OCTAVES[1]:
            raise ValueError(f"blob {blob!r}: noise octaves must be {OCTAVES[0]}..{OCTAVES[1]}")
        # Clamped here, so clay.json records the noise that was actually meshed.
        amp = round(min(amp, cls.max_amp(scale, octaves)), 6)
        return cls(amp=amp, scale=scale, seed=int(d.get("seed", 0)), octaves=octaves)

    def to_dict(self) -> dict:
        return {"amp": self.amp, "scale": self.scale, "seed": self.seed, "octaves": self.octaves}

    def __call__(self, p: np.ndarray) -> np.ndarray:
        """Displacement at local points (N, 3), within about +-amp."""
        total = np.zeros(len(p))
        weights = 0.0
        for k in range(self.octaves):
            # Each octave gets its own seed and a fractional offset, so lattice points never line up across them.
            total += 0.5 ** k * perlin(p * (2.0 ** k / self.scale) + 0.31 * (k + 1), self.seed * 4 + k)
            weights += 0.5 ** k
        return self.amp * total / weights


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
    noise: Noise | None = None

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
            noise=Noise.from_dict(d.get("noise"), str(d["name"])),
        )

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "form": self.form.to_dict(),
            "op": self.op,
            "blend": self.blend,
            "color": [round(c, 4) for c in self.color],
            "mirror": self.mirror,
            "translation": list(self.translation),
            "rotation": list(self.rotation),
        }
        if self.noise is not None:  # only when set, so a clay.json from before noise saves back unchanged
            out["noise"] = self.noise.to_dict()
        return out

    def _rot(self) -> Rotation:
        return Rotation.from_euler("xyz", self.rotation, degrees=True)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        """This blob's own distance at world points, mirror copy included."""
        inv = self._rot().inv()
        t = np.asarray(self.translation, dtype=np.float64)
        d = self._local_sdf(inv.apply(p - t))
        if self.mirror:
            # The reflected copy seen from p is the original seen from p reflected; min() is right even when
            # the blob straddles X = 0 (|x| folding is not).
            q = p * np.array([-1.0, 1.0, 1.0])
            d = np.minimum(d, self._local_sdf(inv.apply(q - t)))
        return d

    def _local_sdf(self, local: np.ndarray) -> np.ndarray:
        # Noise is read in the blob's own frame: it travels with the blob, and the mirror copy is its reflection.
        d = self.form.sdf(local)
        return d + self.noise(local) if self.noise is not None else d

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = np.abs(self._rot().as_matrix()) @ self.form.half_extent()
        t = np.asarray(self.translation, dtype=np.float64)
        lo, hi = t - half, t + half
        if self.mirror:
            lo = np.minimum(lo, (t * [-1, 1, 1]) - half)
            hi = np.maximum(hi, (t * [-1, 1, 1]) + half)
        pad = self.blend + (self.noise.amp if self.noise is not None else 0.0)
        return lo - pad, hi + pad


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
