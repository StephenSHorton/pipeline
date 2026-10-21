#!/usr/bin/env python3
"""PS1 budget pass: a photo-baked GLB cut down to a low-poly, flat-shaded model with a tiny crunched atlas.

    .venv/bin/python scripts/psx_lowpoly.py out/<stem>/textured_photo.glb --out out/<stem>/psx --name cow \
        --height 58 [--facing right] [--faces 1200] [--atlas 256] [--yaw 0] [--up z] [--glb out/<stem>/psx.glb]

What a PS1 artist did by hand: decimate the detailed mesh to --faces, unwrap again (xatlas) and re-bake the
texture from the detailed mesh onto the low one (each texel takes the colour of the closest point on the
original), crunch the atlas to --atlas px and 15-bit colour, and write unshared corners with one normal per face
so it shades flat.

Writes <out>/<name>.obj + <name>.png + psx.json, and with --glb the same model as a GLB (nearest sampler, flat
normals, glTF axes, metres) for the viewer and engines that take glTF.

Axes. Hunyuan's +Z is the side the photo saw. --facing says where the subject looks in the photo: camera (a
portrait, the default), right or left (side-on, like the cow), away. The model is turned so the subject looks
down glTF +Z. The .obj then follows --up: z (default) is Z up with that front turned to +X, which is what s&box
wants (Source units, 1 unit = 1 inch); y keeps glTF axes. --height is in .obj units, feet on 0, centred.

Facing gate. After the model is placed (head down +Z by --facing), facing.check_mesh votes on where the head
really is: the toes, the narrow end and a red comb (rig_fit.geometric_cues) plus the vision model on four textured
renders. If the toes AND the vision model both put the head at -Z, the model is turned a further 180 degrees (a
re-place at +180: x and z negated) and psx.json records facing_check {flipped: true, why} and facing_source
'cues'. --facing-check off (or PSX_FACING_CHECK=off) skips it; a failing check never fails the pass.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))  # facing.py, rig*.py next to this file

# Degrees about +Y (glTF up) that turn the subject to look down +Z, by where it looks in the photo.
FACING = {"camera": 0.0, "right": -90.0, "left": 90.0, "away": 180.0}
INCH = 0.0254  # the GLB is in metres; --height is in .obj units, inches for s&box


def load_textured(path: Path) -> tuple[trimesh.Trimesh, np.ndarray]:
    scene = trimesh.load(str(path), force="scene")
    meshes = [g for g in scene.dump(concatenate=False) if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise SystemExit(f"{path}: no triangle mesh")
    mesh = max(meshes, key=lambda m: len(m.faces))
    visual = mesh.visual
    if not hasattr(visual, "uv") or visual.uv is None:
        raise SystemExit(f"{path}: the mesh has no UVs (use the textured or textured_photo GLB)")
    material = visual.material
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is None:
        raise SystemExit(f"{path}: no base colour texture")
    return mesh, np.asarray(image.convert("RGB"), dtype=np.float32)


def place(mesh: trimesh.Trimesh, height: float, yaw: float) -> trimesh.Trimesh:
    """glTF axes: turned yaw degrees about +Y, scaled to height, feet on y = 0, centred on x and z."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if yaw:
        a = np.deg2rad(yaw)
        c, s = np.cos(a), np.sin(a)
        v = v @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])  # x' = c x + s z, z' = -s x + c z: +X -> -Z at +90
    lo, hi = v.min(0), v.max(0)
    v *= height / max(hi[1] - lo[1], 1e-6)
    lo, hi = v.min(0), v.max(0)
    centre = (lo + hi) / 2
    centre[1] = lo[1]
    out = mesh.copy()
    out.vertices = v - centre
    return out


def to_obj_axes(v: np.ndarray, up: str) -> np.ndarray:
    """glTF (Y up, front +Z) to Z up with the front on +X: (x, y, z) = (z, x, y), a rotation, keeps handedness."""
    if up == "y":
        return v
    return np.stack([v[:, 2], v[:, 0], v[:, 1]], axis=1)


def decimate(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    plain = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if len(plain.faces) <= faces:
        return plain
    low = plain.simplify_quadric_decimation(face_count=faces)
    low.remove_unreferenced_vertices()
    # Hunyuan leaves floating crumbs; at a PS1 budget every triangle on one counts. Keep the big pieces.
    parts = low.split(only_watertight=False)
    if len(parts) > 1:
        keep = [p for p in parts if p.area >= 0.02 * low.area] or [max(parts, key=lambda p: p.area)]
        low = trimesh.util.concatenate(keep)
    return low


def bake(high: trimesh.Trimesh, texture: np.ndarray, low: trimesh.Trimesh, atlas: int):
    """Unwrap the low mesh and fill its atlas from the closest point on the textured high mesh."""
    pack = xatlas.PackOptions()
    pack.resolution = atlas
    pack.padding = 2
    pack.bilinear = False
    pack.bruteForce = True
    chart = xatlas.ChartOptions()
    chart.max_iterations = 4
    atl = xatlas.Atlas()
    atl.add_mesh(np.asarray(low.vertices, np.float32), np.asarray(low.faces, np.uint32))
    atl.generate(chart, pack)
    vmap, indices, uvs = atl[0]
    verts = np.asarray(low.vertices)[vmap]
    faces = np.asarray(indices, dtype=np.int64)

    # Every texel centre that lands inside a triangle, as a point on the low mesh.
    px = uvs * atlas
    points, texels = [], []
    for f in faces:
        t = px[f]
        x0, y0 = np.floor(t.min(0)).astype(int)
        x1, y1 = np.ceil(t.max(0)).astype(int)
        xs, ys = np.meshgrid(np.arange(max(x0, 0), min(x1, atlas)), np.arange(max(y0, 0), min(y1, atlas)))
        if xs.size == 0:
            continue
        p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
        a, b, c = t
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        lam = np.linalg.solve(m, (p - a).T).T
        u, w = lam[:, 0], lam[:, 1]
        inside = (u >= -0.02) & (w >= -0.02) & (u + w <= 1.02)
        if not inside.any():
            continue
        bary = np.stack([1 - u - w, u, w], axis=1)[inside]
        points.append(bary @ verts[f])
        texels.append(p[inside].astype(int))
    points = np.concatenate(points)
    texels = np.concatenate(texels)

    closest, _, tri = trimesh.proximity.closest_point(high, points)
    bary = trimesh.triangles.points_to_barycentric(high.triangles[tri], closest)
    huv = np.einsum("ij,ijk->ik", bary, np.asarray(high.visual.uv)[high.faces[tri]])
    th, tw = texture.shape[:2]
    sx = np.clip((huv[:, 0] % 1.0) * tw, 0, tw - 1).astype(int)
    sy = np.clip((1.0 - huv[:, 1] % 1.0) * th, 0, th - 1).astype(int)

    image = np.zeros((atlas, atlas, 3), np.float32)
    filled = np.zeros((atlas, atlas), bool)
    ty = atlas - 1 - texels[:, 1]
    image[ty, texels[:, 0]] = texture[sy, sx]
    filled[ty, texels[:, 0]] = True

    # Bleed colour into empty texels so seams never sample black.
    _, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
    image = image[iy, ix]
    return verts, faces, uvs, image, float(filled.mean())


def crunch(image: np.ndarray) -> Image.Image:
    rgb = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray((rgb >> 3) << 3, "RGB")  # 15-bit colour, like PS1 VRAM


def flat_corners(verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray):
    """Unshared corners, one normal per face: (positions, uvs, normals) per corner, 3 per triangle."""
    pos = verts[faces].reshape(-1, 3)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return pos, uvs[faces].reshape(-1, 2), np.repeat(n, 3, axis=0)


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray, name: str, up: str) -> None:
    pos, uv, nrm = flat_corners(to_obj_axes(verts, up), faces, uvs)
    lines = [f"# {name}: ai-asset-pipeline PSX low-poly", f"mtllib {name}.mtl", f"o {name}", f"usemtl {name}"]
    lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in pos]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in nrm[::3]]
    lines += [f"f {i + 1}/{i + 1}/{i // 3 + 1} {i + 2}/{i + 2}/{i // 3 + 1} {i + 3}/{i + 3}/{i // 3 + 1}"
              for i in range(0, len(pos), 3)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".mtl").write_text(f"newmtl {name}\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {name}.png\n",
                                        encoding="utf-8")


def write_glb(path: Path, verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray, png: Image.Image,
              name: str) -> None:
    """glTF axes, metres, unlit-friendly matte material, nearest filtering with no mips."""
    pos, uv, nrm = flat_corners(verts * INCH, faces, uvs)
    uv = uv.copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # glTF UV origin is top-left
    buf = BytesIO()
    png.save(buf, "PNG")
    count = len(pos)
    # Corners are unshared, so the index buffer is 0..n-1; importers and inspect_glb expect one anyway.
    wide = count > 65535
    index = np.arange(count, dtype="<u4" if wide else "<u2")
    blobs = [pos.astype("<f4").tobytes(), nrm.astype("<f4").tobytes(), uv.astype("<f4").tobytes(), buf.getvalue(),
             index.tobytes()]
    views, offset = [], 0
    for i, b in enumerate(blobs):
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(b)}
        if i < 3:
            view["target"] = 34962
        elif i == 4:
            view["target"] = 34963
        views.append(view)
        offset += (len(b) + 3) & ~3
    doc = {
        "asset": {"version": "2.0", "generator": "ai-asset-pipeline psx_lowpoly"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                                  "indices": 3, "material": 0}]}],
        "materials": [{"name": f"{name}_psx", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],  # NEAREST, clamp
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": count, "type": "VEC2"},
            {"bufferView": 4, "componentType": 5125 if wide else 5123, "count": count, "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    body = b"".join(b + b"\0" * (((len(b) + 3) & ~3) - len(b)) for b in blobs)
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (((len(js) + 3) & ~3) - len(js))
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(body))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(body), 0x004E4942) + body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


def turn_180(verts: np.ndarray) -> np.ndarray:
    """The placed model turned 180 degrees about +Y: the same as place() at yaw + 180 (place centres x and z)."""
    out = np.array(verts, dtype=np.float64, copy=True)
    out[:, 0] *= -1
    out[:, 2] *= -1
    return out


def drawn_facing(src: Path) -> dict:
    """{facing, canonical, facing_source} from the asset's flat/key.json (the source GLB sits in out/<stem>/), for
    when generate.sh does not pass --canonical / --facing-source. {} when there is none."""
    try:
        key = json.loads((Path(src).resolve().parent / "flat" / "key.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(key, dict) or not key.get("facing"):
        return {}
    canonical = bool(key.get("canonical"))
    return {"facing": key["facing"], "canonical": canonical,
            "facing_source": key.get("facing_source") or ("drawn" if canonical else None)}


def subject_kind(src: Path) -> tuple[bool, str | None]:
    """(is a creature, flat2d pose kind) from the asset's intent.json (the source GLB sits in out/<stem>/). With no
    intent.json the subject is unknown and the gate runs (True, None); an intent that names no creature (a crate, a
    tank, subject 'prop') is (False, None): its 'toes' and 'head' mean nothing, so the facing gate is skipped."""
    try:
        intent = json.loads((Path(src).resolve().parent / "intent.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True, None
    if not isinstance(intent, dict):
        return True, None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from flat2d.pose import pose_for
    except Exception:  # noqa: BLE001 - no pose table: keep the gate
        return True, None
    kind = pose_for({**intent, "output": intent.get("output") or "ps1"}, intent.get("prompt"))
    return kind is not None, kind


def facing_gate(verts, faces, uvs, png, name: str, cls: str, declared: str, work: Path, check=None) -> dict:
    """Write a temporary GLB of the placed low model, run facing.check_mesh on it. Returns facing_check."""
    if check is None:
        import facing
        check = facing.check_mesh
    tmp = work / f".{name}_facing.glb"
    try:
        write_glb(tmp, verts, faces, uvs, png, name)
        got = check(tmp, cls, declared)
    finally:
        tmp.unlink(missing_ok=True)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="textured GLB (textured_photo.glb, textured.glb)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--name", required=True)
    ap.add_argument("--height", type=float, required=True, help="height in .obj units (s&box: inches)")
    ap.add_argument("--facing", choices=sorted(FACING), default="camera",
                    help="where the subject looks in the photo; it is turned to look down the model's front")
    ap.add_argument("--faces", type=int, default=1200, help="triangle budget (PS1 props ran a few hundred to ~1.5k)")
    ap.add_argument("--atlas", type=int, default=256, help="texture size; the PS1 topped out at 256")
    ap.add_argument("--yaw", type=float, default=0, help="extra turn about the up axis, degrees")
    ap.add_argument("--up", choices=("z", "y"), default="z", help=".obj axes: z = Z up, +X front (s&box); y = glTF")
    ap.add_argument("--glb", dest="glb_out", type=Path, default=None, help="also write the model as a GLB here")
    ap.add_argument("--facing-source", choices=("drawn", "detected", "user", "cues"), default=None,
                    help="where --facing came from (recorded in psx.json)")
    ap.add_argument("--canonical", action="store_true", help="the picture is a pipeline-drawn canonical profile")
    ap.add_argument("--cls", default="animal", help="class for the facing cues (humanoid skips the narrow-end cue)")
    ap.add_argument("--facing-check", choices=("auto", "off"), default=None,
                    help="check the placed model's head is at +Z and turn it if the toes and the VLM disagree "
                         "(default auto; PSX_FACING_CHECK=off turns it off)")
    args = ap.parse_args()

    high, texture = load_textured(args.src)
    high = place(high, args.height, FACING[args.facing] + args.yaw)
    low = decimate(high, args.faces)
    verts, faces, uvs, image, used = bake(high, texture, low, args.atlas)
    png = crunch(image)

    gate = args.facing_check or ("off" if (os.environ.get("PSX_FACING_CHECK") or "").lower() in {"off", "0", "no"}
                                 else "auto")
    facing_check: dict = {"skipped": True} if gate == "off" else {}
    drawn = drawn_facing(args.src)
    canonical = bool(args.canonical or (drawn.get("canonical") and drawn.get("facing") == args.facing))
    source = args.facing_source or (drawn.get("facing_source") if drawn.get("facing") == args.facing else None)
    creature, kind = subject_kind(args.src)
    cls = "humanoid" if kind == "humanoid" else args.cls
    if gate == "auto" and not creature and not canonical:
        gate = "off"
        facing_check = {"skipped": True, "why": "not a creature (intent.json): no head or toes to check"}
    if gate == "auto":
        try:
            import facing
            got = facing_gate(verts, faces, uvs, png, args.name, cls, args.facing, args.out)
            facing_check = {"flipped": False, "agrees": got["agrees"], "head_axis": got["head_axis"],
                            "votes": got["votes"], "vlm": got.get("vlm"), "declared": args.facing}
            if got["flipped_needed"]:
                verts = turn_180(verts)
                facing_check["flipped"] = True
                facing_check["why"] = facing.flip_why(args.facing)
                facing_check["decided_by"] = "toes+vlm"
                source = "cues"
                print("psx_lowpoly: warning: " + facing_check["why"], file=sys.stderr)
            else:
                facing_check["decided_by"] = "declared" if (canonical and got["agrees"]) else (
                    "agrees" if got["agrees"] else "declared (cues disagree, not both toes and VLM)")
        except Exception as e:  # noqa: BLE001 - the check is advisory; the pass still writes the model
            facing_check = {"error": f"{e.__class__.__name__}: {e}"}
            print(f"psx_lowpoly: facing check failed: {e}", file=sys.stderr)

    obj = args.out / f"{args.name}.obj"
    tex = args.out / f"{args.name}.png"
    write_obj(obj, verts, faces, uvs, args.name, args.up)
    png.save(tex)
    if args.glb_out:
        write_glb(args.glb_out, verts, faces, uvs, png, args.name)
    size = to_obj_axes(verts, args.up)
    report = {
        "name": args.name,
        "obj": obj.name,
        "texture": tex.name,
        "glb": args.glb_out.name if args.glb_out else None,
        "triangles": int(len(faces)),
        "source": args.src.name,
        "source_triangles": int(len(high.faces)),
        "atlas": args.atlas,
        "atlas_used": round(used, 3),
        "obj_axes": "z-up, +x front" if args.up == "z" else "y-up, +z front (glTF)",
        "glb_axes": "y-up, +z front (glTF), metres",
        "height": args.height,
        "facing": args.facing,
        "facing_source": source,
        "canonical": canonical,
        "facing_check": facing_check,
        "size": [round(float(x), 1) for x in size.max(0) - size.min(0)],
    }
    (args.out / "psx.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PSX_RESULT " + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
