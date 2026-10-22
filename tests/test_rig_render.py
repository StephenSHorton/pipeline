#!/usr/bin/env python3
"""scripts/rig_render.py on its own 6-joint stick rig: CPU render, floor row, frames.json, contact sheet, views,
roles, auto frames, textured_views; plus a Blender Workbench parity check (skipped without Blender)."""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig  # noqa: E402
import rig_render as rr  # noqa: E402

NAMES = ["root", "hips", "spine", "head", "leg_L", "foot_L"]
PARENT = [-1, 0, 1, 2, 1, 4]
JPOS = np.array([[0, 0.0, 0], [0, 0.5, 0], [0, 0.9, 0.2], [0, 1.2, 0.4], [0.2, 0.5, 0], [0.2, 0.0, 0]])
# tubes: (from joint, to point, owning bone); the leg tube ends exactly on the floor
TUBES = [(1, JPOS[2], 1), (2, JPOS[3], 2), (3, JPOS[3] + [0, 0.1, 0.25], 3), (4, JPOS[5], 4),
         (5, JPOS[5] + [0, 0, 0.2], 5)]
H_EXPECT = 1.3 + 0.05  # head tube tip plus the tube radius


def tube(a, b, ring=8, rings=4, r=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    u = np.cross(ax, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(ax, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ax / np.linalg.norm(ax), u)
    P = []
    for i in range(rings):
        c = a + ax * i / (rings - 1)
        for k in range(ring):
            th = 2 * np.pi * k / ring
            P.append(c + r * (np.cos(th) * u + np.sin(th) * v))
    F = []
    for i in range(rings - 1):
        for k in range(ring):
            p0, p1 = i * ring + k, i * ring + (k + 1) % ring
            F += [[p0, p1, p0 + ring], [p1, p1 + ring, p0 + ring]]
    return np.array(P), np.array(F)


def write_rig(path: Path) -> None:
    """A 6-joint GLB: textured tube mesh, rigid weights, one looping 'Swing' clip on leg_L (identity at t=0)."""
    Ps, Fs, Js = [], [], []
    off = 0
    for j, end, bone in TUBES:
        P, F = tube(JPOS[j], end)
        Ps.append(P)
        Fs.append(F + off)
        Js.append(np.full(len(P), bone))
        off += len(P)
    P = np.concatenate(Ps).astype("<f4")
    P[:, 1] = np.maximum(P[:, 1], 0)  # the foot tube's lowest ring sits at y = 0
    F = np.concatenate(Fs).astype("<u4")
    Jb = np.concatenate(Js)
    J = np.zeros((len(P), 4), "u1")
    J[:, 0] = Jb
    W = np.zeros((len(P), 4), "<f4")
    W[:, 0] = 1
    N = np.tile(np.array([0, 1, 0], "<f4"), (len(P), 1))
    UV = np.tile(np.array([0.25, 0.25], "<f4"), (len(P), 1))  # all in the green texel
    g = rig.GlbWriter()
    d = g.doc
    tex = np.zeros((2, 2, 3), np.uint8)
    tex[:] = (40, 170, 60)
    buf = io.BytesIO()
    Image.fromarray(tex).save(buf, "PNG")
    iv = g.blob(buf.getvalue())
    d["images"] = [{"bufferView": iv, "mimeType": "image/png"}]
    d["samplers"] = [{"magFilter": 9728, "minFilter": 9728}]
    d["textures"] = [{"source": 0, "sampler": 0}]
    d["materials"] = [{"name": "m", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0}}]
    attrs = {"POSITION": g.array(P, "VEC3", 34962, minmax=True), "NORMAL": g.array(N, "VEC3", 34962),
             "TEXCOORD_0": g.array(UV, "VEC2", 34962), "JOINTS_0": g.array(J, "VEC4", 34962),
             "WEIGHTS_0": g.array(W, "VEC4", 34962)}
    ind = g.array(F.reshape(-1), "SCALAR", 34963)
    d["meshes"] = [{"name": "Body", "primitives": [{"attributes": attrs, "indices": ind, "material": 0}]}]
    nodes = []
    for k, n in enumerate(NAMES):
        t = JPOS[k] - (JPOS[PARENT[k]] if PARENT[k] >= 0 else 0)
        nodes.append({"name": n, "translation": [float(x) for x in t],
                      "children": [c for c in range(len(NAMES)) if PARENT[c] == k]})
    for n in nodes:
        if not n["children"]:
            del n["children"]
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})
    d["nodes"] = nodes
    ibm = np.array([np.linalg.inv(np.block([[np.eye(3), JPOS[k][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])).T.reshape(16)
                    for k in range(len(NAMES))], "<f4")
    d["skins"] = [{"joints": list(range(len(NAMES))), "inverseBindMatrices": g.array(ibm, "MAT4"), "skeleton": 0}]
    times = np.array([0, 0.25, 0.5, 0.75, 1.0], "<f4")[:, None]
    ang = [0, 35, 0, -35, 0]
    q = np.array([R.from_euler("x", a, degrees=True).as_quat() for a in ang], "<f4")
    ti = g.array(times, "SCALAR")
    d["accessors"][ti]["min"], d["accessors"][ti]["max"] = [0.0], [1.0]
    d["animations"] = [{"name": "Swing", "samplers": [{"input": ti, "output": g.array(q, "VEC4"), "interpolation": "LINEAR"}],
                        "channels": [{"sampler": 0, "target": {"node": 4, "path": "rotation"}}]}]
    d["scenes"] = [{"nodes": [0, len(NAMES)]}]
    d["scene"] = 0
    g.write(path)


CHECKER = 8  # cells a side in the quad's texture


def write_quad(path: Path) -> None:
    """A static 2 x 2 quad in the XY plane facing +Z (two triangles), UVs over the whole of a black-and-white
    checker texture: per-face sampling gives each triangle one colour, per-pixel sampling the checker."""
    P = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], "<f4")
    UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], "<f4")  # glTF v runs down the image: the top edge is v = 0
    F = np.array([[0, 1, 2], [0, 2, 3]], "<u4")
    cells = (np.indices((CHECKER, CHECKER)).sum(0) % 2).astype(np.uint8) * 255
    tex = np.repeat(np.repeat(cells, 16, 0), 16, 1)
    buf = io.BytesIO()
    Image.fromarray(np.stack([tex] * 3, -1)).save(buf, "PNG")
    g = rig.GlbWriter()
    d = g.doc
    d["images"] = [{"bufferView": g.blob(buf.getvalue()), "mimeType": "image/png"}]
    d["samplers"] = [{"magFilter": 9728, "minFilter": 9728}]
    d["textures"] = [{"source": 0, "sampler": 0}]
    d["materials"] = [{"name": "m", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0}}]
    attrs = {"POSITION": g.array(P, "VEC3", 34962, minmax=True), "TEXCOORD_0": g.array(UV, "VEC2", 34962)}
    d["meshes"] = [{"name": "Quad", "primitives": [{"attributes": attrs, "indices": g.array(F.reshape(-1), "SCALAR",
                                                                                            34963), "material": 0}]}]
    d["nodes"] = [{"name": "Quad", "mesh": 0}]
    d["scenes"] = [{"nodes": [0]}]
    d["scene"] = 0
    g.write(path)


def non_bg(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB")).astype(int)
    return np.abs(a - np.array(rr.BG)).sum(2) > 30


class CPURender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rig_render_test_"))
        cls.asset = cls.tmp / "stick"
        cls.asset.mkdir()
        write_rig(cls.asset / "rigged.glb")
        cls.res = rr.render(cls.asset, {"views": ["side", "xside", "front", "top", "q"],
                                        "modes": ["texture", "sticks", "groups", "heat:leg_L"],
                                        "frames": "auto", "size": 160, "sheet_tile": 120})
        cls.out = cls.asset / "rig" / "preview"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pngs_have_content(self):
        pngs = [p for f in self.res["frames"] for p in f["pngs"].values()]
        self.assertGreater(len(pngs), 20)
        for p in pngs:
            self.assertTrue(Path(p).is_file() and Path(p).stat().st_size > 0, p)
            m = non_bg(Image.open(p))
            m[:14] = False  # the caption
            self.assertGreater(int(m.sum()), 100, p)
        self.assertTrue((self.out / "Rest_0.00_side_texture.png").is_file())
        self.assertTrue((self.out / "Swing_0.25_q_heat-leg_L.png").is_file())

    def test_texture_colour_per_face(self):
        a = np.asarray(Image.open(self.out / "Rest_0.00_side_texture.png").convert("RGB")).astype(int)
        body = non_bg(Image.fromarray(a.astype(np.uint8))) & ~np.all(a == rr.FLOOR, axis=2)
        body[:14] = False
        g = a[body]
        # shaded green texel: green dominates
        self.assertGreater(float(np.mean((g[:, 1] > g[:, 0]) & (g[:, 1] > g[:, 2]))), 0.9)

    def test_floor_line_at_expected_row(self):
        cam = self.res["camera"]
        for view in ("side", "front"):
            row = int(round(rr.floor_row(cam, view)))
            self.assertAlmostEqual(self.res["floor_rows"][view], rr.floor_row(cam, view), places=2)
            # the projection of y = 0 is that row; the floor sits under the mesh
            c = cam["centre"]
            self.assertAlmostEqual(float(rr.project(np.array([[c[0], 0, c[2]]]), view, cam)[0, 1]), rr.floor_row(cam, view))
            a = np.asarray(Image.open(self.out / f"Rest_0.00_{view}_texture.png").convert("RGB"))
            red = np.all(a == rr.FLOOR, axis=2)
            self.assertGreater(red[row].mean(), 0.8, view)
            self.assertEqual(int(red[row - 4].sum()), 0)
            self.assertEqual(int(red[row + 5].sum()), 0)
            # the foot stands on the line: mesh pixels just above it
            body = non_bg(Image.fromarray(a)) & ~red
            self.assertGreater(int(body[row - 3:row].sum()), 0)

    def test_frames_json(self):
        fj = json.loads((self.out / "frames.json").read_text())
        rest = [f for f in fj["frames"] if f["clip"] == "Rest"][0]
        self.assertAlmostEqual(rest["min_y_rel"], 0.0, places=3)
        self.assertAlmostEqual(rest["height_rel"], 1.0, places=3)
        self.assertAlmostEqual(rest["max_disp_rel"], 0.0, places=3)
        s0 = [f for f in fj["frames"] if f["clip"] == "Swing" and f["frac"] == 0][0]
        self.assertAlmostEqual(s0["min_y_rel"], 0.0, places=3)
        self.assertGreaterEqual(min(f["min_y_rel"] for f in fj["frames"]), -0.03)
        for f in fj["frames"]:
            for k in ("height_rel", "tilt", "min_y_rel", "max_disp_rel", "stance_foot_y_rel", "beak_y_rel"):
                self.assertIn(k, f)
        self.assertAlmostEqual(fj["H"], H_EXPECT, delta=0.06)
        # the swing moves the foot: some frame displaces it
        self.assertGreater(max(f["max_disp_rel"] for f in fj["frames"]), 0.1)

    def test_auto_frames(self):
        fr = sorted(f["frac"] for f in self.res["frames"] if f["clip"] == "Swing")
        self.assertEqual(fr[:1], [0.0])
        for x in (0.25, 0.5, 0.75):
            self.assertIn(x, fr)
        self.assertNotIn(1.0, fr)  # it loops
        self.assertEqual(self.res["clips"], ["Swing"])

    def test_contact_sheet_size(self):
        im = Image.open(self.res["sheet"])
        self.assertEqual(im.size, (3 * 120, 2 * (120 + rr.CAP)))  # rows Rest + Swing, columns side, front, q
        self.assertEqual(self.res["sheet_cols"], ["side", "front", "q"])
        self.assertEqual(len(self.res["sheet_rows"]), 2)

    def test_contact_sheet_function(self):
        a = self.out / "Rest_0.00_side_texture.png"
        im = rr.contact_sheet([a, None, a, "-", a], rows=2, cols=3, captions=["x", "", "y", "", "z"], tile=50)
        self.assertEqual(im.size, (150, 2 * (50 + rr.CAP)))


class Conventions(unittest.TestCase):
    def test_views(self):
        cam = rr.camera(np.array([[-1, 0, -1], [1, 1, 1.0]]), 100)
        head = np.array([[0, 0.5, 1.0]])  # +Z = forward
        c = rr.project(np.array([cam["centre"]]), "side", cam)[0, 0]
        self.assertLess(rr.project(head, "side", cam)[0, 0], c)  # camera at +X: head on the left (as rig_fit.vlm_cue)
        self.assertGreater(rr.project(head, "xside", cam)[0, 0], c)
        left = np.array([[1.0, 0.5, 0]])
        self.assertGreater(rr.project(left, "front", cam)[0, 0], c)
        self.assertLess(rr.project(left, "back", cam)[0, 0], c)
        self.assertGreater(rr.project(left, "top", cam)[0, 0], c)
        up = np.array([[0, 1.0, 0]])
        self.assertLess(rr.project(up, "side", cam)[0, 1], rr.project(np.zeros((1, 3)), "side", cam)[0, 1])
        self.assertEqual(rr.VIEW_RIGHT["xside"], [0.0, 0.0, 1.0])
        for v in rr.VIEWS:
            r, u, f = rr.basis(v)
            self.assertAlmostEqual(float(np.linalg.det(np.array([r, u, -f]))), 1.0, places=6)

    def test_roles(self):
        self.assertEqual(rr.role("UpperLeg_L"), "leg")
        self.assertEqual(rr.role("Toes_tip_R"), "leg")
        self.assertEqual(rr.role("wing_feather_1_1_L"), "wing")
        self.assertEqual(rr.role("mouth_upper_tip"), "head")
        self.assertEqual(rr.role("spine_2"), "spine")
        self.assertEqual(rr.role("tail_tip"), "tail")
        self.assertEqual(rr.role("root"), "root")
        self.assertEqual(rr.role("front_humerus_l"), "leg")
        self.assertEqual(rr.role("upperarm_l"), "wing")
        self.assertEqual(rr.role("thing", {"thing": "tail"}), "tail")

    def test_parse_clips(self):
        self.assertEqual(rr.parse_clips("auto"), "auto")
        self.assertEqual(rr.parse_clips("Walk@0.25,0.5;Peck@0.15"), {"Walk": [0.25, 0.5], "Peck": [0.15]})
        self.assertEqual(rr.parse_clips("Walk"), {"Walk": "auto"})

    def test_raster_depth(self):
        V = np.array([[10, 10, 5], [90, 10, 5], [10, 90, 5], [10, 10, 1], [90, 10, 1], [10, 90, 1.0]])
        face, bary = rr.raster(V, np.array([[0, 1, 2], [3, 4, 5]]), 100)
        self.assertEqual(int(face[20, 20]), 1)  # nearer face wins
        self.assertEqual(int(face[95, 95]), -1)
        self.assertAlmostEqual(float(bary[20, 20].sum()), 1.0)

    def test_textured_views_static(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            write_rig(tmp / "r.glb")
            res = rr.textured_views(tmp / "r.glb", out_dir=tmp / "v", size=96)
            self.assertEqual(set(res), {"side", "xside", "front", "back"})
            for p in res.values():
                self.assertGreater(int(non_bg(Image.open(p)).sum()), 50)
            # side and xside are mirror images of the same silhouette
            a, b = non_bg(Image.open(res["side"])), non_bg(Image.open(res["xside"]))
            self.assertGreater((a & b[:, ::-1]).sum() / max((a | b[:, ::-1]).sum(), 1), 0.85)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TextureSampling(unittest.TestCase):
    """texture (one texel a face, unchanged) against texture_px (per pixel) on the checker quad."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rig_render_px_"))
        write_quad(cls.tmp / "quad.glb")
        write_rig(cls.tmp / "rig.glb")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def front(self, mode: str, size: int = 128) -> np.ndarray:
        it = rr.Item(self.tmp / "quad.glb")
        Vs, jp = it.pose(None, 0.0)
        return np.asarray(rr.draw([it], [(Vs, jp)], "front", mode, rr.camera(np.concatenate(Vs), size), False))

    def cell_values(self, img: np.ndarray) -> np.ndarray:
        """Grey level at each checker cell's centre, rows top to bottom, read off the quad's pixel box."""
        ys, xs = np.nonzero(np.abs(img.astype(int) - np.array(rr.BG)).sum(2) > 30)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        c = (np.arange(CHECKER) + 0.5) / CHECKER
        return img[(y0 + c * (y1 - y0)).astype(int)[:, None], (x0 + c * (x1 - x0)).astype(int)[None], 0].astype(int)

    def test_per_pixel_shows_the_checker(self):
        g = self.cell_values(self.front("texture_px:unlit"))
        want = (np.indices((CHECKER, CHECKER)).sum(0) % 2) * 255
        np.testing.assert_array_equal(g, want)
        shaded = self.cell_values(self.front("texture_px"))
        self.assertTrue(((shaded > 100) == (want > 0)).all())  # Lambert darkens the white cells, keeps the pattern
        self.assertTrue((shaded[want > 0] < 255).all() and (shaded[want == 0] == 0).all())

    def test_per_face_is_one_colour_a_triangle(self):
        img = self.front("texture")
        hit = np.abs(img.astype(int) - np.array(rr.BG)).sum(2) > 30
        self.assertLessEqual(len(np.unique(img[hit].reshape(-1, 3), axis=0)), 2)

    def test_texture_mode_unchanged(self):
        # Pixels of the stick rig's texture renders, pinned when texture_px was added: the rig sheets and the facing
        # VLM's views must not move. (Checked against the renderer before the change: 133 rig and bale PNGs equal.)
        import hashlib
        res = rr.textured_views(self.tmp / "rig.glb", ("side", "front", "top", "q"), 96, self.tmp / "v")
        got = {v: hashlib.md5(np.asarray(Image.open(p).convert("RGB")).tobytes()).hexdigest()[:12]
               for v, p in res.items()}
        self.assertEqual(got, PINNED_TEXTURE)

    def test_bottom_view(self):
        cam = rr.camera(np.array([[-1, 0, -1], [1, 1, 1.0]]), 100)
        c = 50
        self.assertGreater(rr.project(np.array([[1.0, 0.5, 0]]), "bottom", cam)[0, 0], c)  # +X right, like top
        self.assertLess(rr.project(np.array([[0, 0.5, 1.0]]), "bottom", cam)[0, 1], c)  # +Z up the image
        res = rr.textured_views(self.tmp / "rig.glb", ("top", "bottom"), 96, self.tmp / "tb", mode="texture_px")
        a, b = non_bg(Image.open(res["top"])), non_bg(Image.open(res["bottom"]))
        self.assertGreater(int(b.sum()), 50)
        self.assertGreater((a & b[::-1]).sum() / max((a | b[::-1]).sum(), 1), 0.85)  # top flipped upside down

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            rr.textured_views(self.tmp / "quad.glb", ("front",), 32, self.tmp / "bad", mode="groups")


PINNED_TEXTURE = {"side": "a29543e48c10", "front": "aca337ef43fe", "top": "b8aaf87f59ea", "q": "5ee0700fe150"}


@unittest.skipUnless(rig.blender_path(), "no Blender")
class BlenderParity(unittest.TestCase):
    def test_parity(self):
        tmp = Path(tempfile.mkdtemp(prefix="rig_render_bl_"))
        try:
            write_rig(tmp / "rigged.glb")
            res = rr.render(tmp, {"views": ["side", "front"], "modes": ["texture"], "frames": {"Swing": [0.25]},
                                  "size": 128, "floor": False, "blender": True})
            bl = res["blender"]
            self.assertTrue(bl["ok"], bl.get("log_tail"))
            for name in ("Rest_0.00_side_texture", "Swing_0.25_side_texture", "Swing_0.25_front_texture"):
                a = non_bg(Image.open(tmp / "rig/preview" / f"{name}.png"))
                b = non_bg(Image.open(tmp / "rig/preview/blender" / f"{name}.png"))
                a[:14] = False
                b[:14] = False
                iou = (a & b).sum() / max((a | b).sum(), 1)
                self.assertGreater(iou, 0.75, f"{name} IoU {iou:.2f}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
