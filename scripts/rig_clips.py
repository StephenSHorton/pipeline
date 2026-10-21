#!/usr/bin/env python3
"""Clip synthesiser for built templates (design section 2.6): every clip in the body plan, made for this model's own
skeleton and gated on its own skin.

    synthesise(plan, skel, P, F, J, W, H, names=None) -> (clips, report)
    gate(clip, P, J, W, H, skel=..., faces=..., plan=...) -> {gate: {value, limit, ok}, passed, reasons}

`skel` is the built skeleton with identity rests: {names, parent, rest_t} (or skeleton.json's {joints: [...]}).
P (N,3) / F (faces) / J, W (N,K) are the bound mesh and its skin weights (the real skin-tokens weights). `names`
lists the joint names in J's index space (e.g. skinned_raw.glb's skin order); when given, J is read in that order and
every clip's `rot` keys and `root` are indices into `names` (so rig_template.write_clips(..., names=names) and
rig.bake_clips line up). Without it, indices are skel order.

Each clip is a bake_clips-shaped dict: {name, times (F,) float32 at 30 fps, rot {joint: (F,4) local xyzw float32},
root, root_t (F,3) (the root joint's translation), loop, track {contact {Foot_L, Foot_R, beak?: bool (F,)}}, params,
energy, gates, passed, reasons}. ALL plan clips are returned, failed ones too (passed False, with reasons); the
report lists them in clips_failed. A clip is never dropped silently.

Energy search: on a gate failure the amplitudes (not the timing) scale by gates.energy_step (0.8), up to gates.tries
attempts (Flap: 6); a clip whose next energy would fall below its min_energy fails. Flap only: when it fails at
e = 0.8, rig_build.wing_weight_repair runs once and the attempt is retried (before any clip, a plan with
chains.wings.smooth_iters softens the wing weight edge: rig_build.wing_weight_smooth); the repaired weights are returned in
report['J'] / report['W'] (always the weights the clips were gated on) and report['wing_weight_repair'].

CLI: rig_clips.py --stem S [--clips Walk,Peck] [--report]
  Re-synthesises clips on an existing rig/template/ (skeleton.json + rig.glb) with the weights of the existing
  rig/template/clips.glb Body (else rig/skinned_raw.glb) and rewrites rig/template/clips.glb, clips_report.json and
  template.json, so clip parameters can be tuned without a re-skin.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rig_clips_gaits as gaits  # noqa: E402
import rig_metrics  # noqa: E402

GENERATORS = dict(gaits.GENERATORS)
FPS = gaits.FPS
LYING = {"Sleep", "Death", "lay_to_idle"}
DEFAULT_GATES = {"stretch_p99": 2.5, "stretch_ship": 3.0, "far_share": 0.10, "far_p99_H": 0.75, "cshape": 1.6,
                 "cshape_flap_lateral": 2.2, "floor_min_H": -0.01, "stance_clear_H": 0.02, "foot_slide": 0.03,
                 "loop_seam_deg": 0.5, "head_hold_drift_H": 0.012, "beak_gap_H": [-0.005, 0.02], "belly_gap_H": 0.03,
                 "energy_step": 0.8, "tries": 4}
FLAP_TRIES = 6
STRETCH_MAX_VERTS = 12000


# ---------------------------------------------------------------- helpers

def _gates(plan: dict) -> dict:
    g = dict(DEFAULT_GATES)
    g.update(plan.get("gates") or {})
    return g


def _remap(J: np.ndarray, names: list[str] | None, sk: gaits.Skel) -> tuple[np.ndarray, list[int]]:
    """J in skel order, and the map skel index -> output index (names order, or identity)."""
    J = np.asarray(J).astype(int)
    if names is None:
        return J, list(range(len(sk.names)))
    missing = [n for n in sk.names if n not in names]
    if missing:
        raise ValueError(f"synthesise: skeleton joints not in names: {missing[:5]}")
    extra = [n for n in names if n not in sk.idx]
    if extra:
        raise ValueError(f"synthesise: names has joints the skeleton lacks: {extra[:5]}")
    to_sk = np.array([sk.i(n) for n in names])
    out_of = [names.index(n) for n in sk.names]
    return to_sk[J], out_of


def to_clip(tr: gaits.Track, name: str, spec: dict, e: float, out_of: list[int]) -> dict:
    q = tr.quats()
    sk = tr.sk
    root_t = (sk.rest[sk.root] + tr.root_off).astype(np.float32)
    rot = {int(out_of[j]): v for j, v in q.items()}
    return {"name": name, "times": tr.times.astype(np.float32), "rot": rot, "root": int(out_of[sk.root]),
            "root_t": root_t, "loop": bool(spec.get("loop", True)),
            "track": {"contact": {k: np.asarray(v, bool) for k, v in tr.contact.items()}},
            "params": _params_record(spec, tr, e), "energy": round(float(e), 4)}


def _params_record(spec: dict, tr: gaits.Track, e: float) -> dict:
    rec = dict(spec.get("params") or {})
    rec["T"] = round(tr.T, 4)
    rec["frames"] = tr.F
    for k, v in tr.info.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            rec[k] = round(v, 5) if isinstance(v, float) else v
        elif isinstance(v, dict) and all(isinstance(x, (int, float, str, bool)) for x in v.values()):
            rec[k] = v
    return rec


# ---------------------------------------------------------------- gates

def _far_motion(Vk: list[np.ndarray], Vn: np.ndarray, Wf: np.ndarray, H: float, n_all: int) -> tuple[float, float]:
    """rig.far_motion's B2 measure on in-memory frames: per vertex sum_k w_k |M_k v - M_n v| over far influences, / H;
    p99 over all n_all vertices at the worst frame and the largest share of vertices above 0.1H (vertices without a
    far influence count as 0)."""
    mv = np.zeros((Vn.shape[0], n_all))
    part = np.zeros(Vn.shape[:2])
    for k, Xk in enumerate(Vk):
        part += Wf[None, :, k] * np.linalg.norm(Xk - Vn, axis=2)
    mv[:, :part.shape[1]] = part / max(H, 1e-9)
    p99 = float(np.percentile(mv, 99, axis=1).max()) if mv.size else 0.0
    share = float((mv > 0.1).mean(1).max()) if mv.size else 0.0
    return p99, share


class _Gater:
    """Precomputed mesh facts for gating clips of one creature."""

    def __init__(self, ctx: gaits.Ctx) -> None:
        self.ctx = ctx
        P = ctx.P
        sk = ctx.sk
        self.edges = rig_metrics.edges_from_faces(P, ctx.Fc) if len(ctx.Fc) else np.zeros((0, 2), int)
        if len(self.edges) and len(P) > STRETCH_MAX_VERTS:
            keep = np.random.default_rng(0).choice(len(self.edges), size=min(len(self.edges), 40000), replace=False)
            self.edges = self.edges[keep]
        segs = rig_metrics.segs_from_parent(sk.parent)
        D = rig_metrics.seg_dist(P, sk.rest, segs)
        self.nearest = D.argmin(1)
        Db = np.take_along_axis(D, ctx.J, 1)
        self.far = (Db - D.min(1)[:, None]) / max(ctx.H, 1e-9) > rig_metrics.FAR_REL
        self.far_any = self.far.any(1) & (ctx.W > 0).any(1)

    def lbs(self, pos, rot):
        return self.ctx.skin(pos, rot)

    def far_motion(self, pos, rot) -> tuple[float, float]:
        ctx = self.ctx
        sel = self.far_any
        if not sel.any():
            return 0.0, 0.0
        P, J, W = ctx.P[sel], ctx.J[sel], ctx.W[sel] * self.far[sel]
        t = pos - np.einsum("fbij,bj->fbi", rot, ctx.sk.rest)
        n = self.nearest[sel]

        def at(idx):
            return np.einsum("fnij,nj->fni", rot[:, idx], P) + t[:, idx]
        Vn = at(n)
        Vk = [at(J[:, k]) for k in range(J.shape[1])]
        return _far_motion(Vk, Vn, W, ctx.H, len(ctx.P))


def wing_skin(ctx: gaits.Ctx, Vs: np.ndarray, track: gaits.Track, bin_H: float = 0.04, q: float = 90.0) -> dict | None:
    """How far the folded wings' SKIN comes up in a clip (Vs (F,N,3) skinned points). The wing points are those
    whose top bone is a wing bone (wing_*, feathers included), per side. Returns
    {rise_H: max over frames of the side-minimum q-th percentile of the points' own rise over rest (the hop
    taken out), clear_H: max over frames of the side-minimum side-view clearance (the highest wing point above
    the highest non-wing point in the same 0.04H slice of z, both as skinned in that frame: what shows above the
    back in a side view), per_frame_rise, per_frame_clear} in H, or None when no point is on a wing bone."""
    sk, H = ctx.sk, ctx.H
    sides = [[i for i, n in enumerate(sk.names) if n.startswith("wing_") and n.endswith(s)] for s in ctx.sides]
    masks = [np.isin(ctx.top, b) for b in sides if b]
    masks = [m for m in masks if m.any()]
    if not masks:
        return None
    wv = np.logical_or.reduce(masks)
    hop = np.asarray(track.root_off[:, 1], float)[:, None]
    F = Vs.shape[0]
    rise = np.full((F, len(masks)), -np.inf)
    clear = np.full((F, len(masks)), -np.inf)
    zb = np.floor(Vs[:, :, 2] / (bin_H * H)).astype(np.int64)
    z0 = int(zb.min())
    nb = int(zb.max()) - z0 + 1
    body = ~wv
    for k, m in enumerate(masks):
        rise[:, k] = np.percentile((Vs[:, m, 1] - ctx.P[m, 1][None] - hop) / H, q, axis=1)
    for f in range(F):
        top = np.full(nb, -np.inf)
        np.maximum.at(top, zb[f, body] - z0, Vs[f, body, 1])
        # a low-poly back has slices with no vertex on top (one long face spans them): each slice takes the highest
        # body point within one slice either side
        top = np.maximum.reduce([top, np.r_[-np.inf, top[:-1]], np.r_[top[1:], -np.inf]])
        for k, m in enumerate(masks):
            t = top[zb[f, m] - z0]
            ok = np.isfinite(t)
            if ok.any():
                clear[f, k] = float(((Vs[f, m, 1][ok] - t[ok]) / H).max())
    r, c = rise.min(1), clear.min(1)
    return {"rise_H": float(r.max()), "clear_H": float(c.max()), "per_frame_rise": r, "per_frame_clear": c}


def gate(clip: dict, P, J, W, H, skel=None, faces=None, plan=None, *, ctx: gaits.Ctx | None = None,
         track: gaits.Track | None = None, gater: _Gater | None = None) -> dict:
    """Gate one clip on LBS over every frame with the real weights. Returns {name: {value, limit, ok}, passed,
    reasons}. Pass `skel`/`faces`/`plan` (or a prepared ctx) and either the clip (rot in skel order) or its track."""
    if ctx is None:
        sk = skel if isinstance(skel, gaits.Skel) else gaits.Skel(skel)
        ctx = gaits.Ctx(plan or {}, sk, P, faces, J, W, H)
    sk = ctx.sk
    plan = plan or ctx.plan
    G = _gates(plan)
    gater = gater or _Gater(ctx)
    if track is None:
        track = _track_from_clip(clip, sk)
    name = clip["name"]
    H = ctx.H
    pos, rot = track.fk()
    Vs = gater.lbs(pos, rot)
    V0 = ctx.P
    out: dict = {}

    def put(key, value, limit, ok, **extra):
        out[key] = {"value": None if value is None else round(float(value), 5), "limit": limit, "ok": bool(ok), **extra}

    # stretch
    if len(gater.edges):
        so = rig_metrics.stretch_owners(V0, Vs, gater.edges, ctx.top, sk.names)
        put("stretch_p99", so["p99"], G["stretch_p99"], so["p99"] <= G["stretch_p99"], owners=so["owners"][:3])
    # far-bone motion
    p99, share = gater.far_motion(pos, rot)
    put("far_share", share, G["far_share"], share <= G["far_share"])
    put("far_p99_H", p99, G["far_p99_H"], p99 <= G["far_p99_H"])
    # C-shape per axis (lying clips: in the hips' frame, height exempt)
    lying = name in LYING or bool(track.info.get("lying"))
    if lying:
        hi = sk.i(ctx.hips)
        Rh = rot[:, hi]
        ph = pos[:, hi]
        Vb = np.einsum("fji,fnj->fni", Rh, Vs - ph[:, None])
        V0b = V0 - sk.rest[hi]
        cs = rig_metrics.c_shape(Vb, V0b)
    else:
        cs = rig_metrics.c_shape(Vs, V0)
    lim = [G["cshape"]] * 3
    if name == "Flap":
        lim[0] = G["cshape_flap_lateral"]
    axes_ok = [cs["ext_ratio"][k] <= lim[k] or (lying and k == 1) for k in range(3)]
    put("cshape", max(cs["ext_ratio"][k] for k in range(3) if not (lying and k == 1)), lim, all(axes_ok),
        ext_ratio=cs["ext_ratio"], frame="hips" if lying else "world")
    # floor
    lo = (Vs[:, :, 1].min(1) - ctx.floor) / H
    put("floor_min_H", lo.min(), G["floor_min_H"], lo.min() >= G["floor_min_H"])
    # stance feet: clearance and slide
    v = float(track.info.get("v", 0.0) or 0.0)
    clear, slide = [], []
    for lg in ctx.legs:
        k = lg["names"]["K"]
        c = track.contact.get(k)
        if c is None or not np.any(c):
            continue
        fv = ctx.foot_v.get(k)
        if fv is not None and fv.any():
            fy = (Vs[:, fv, 1].min(1) - ctx.floor) / H
            clear.append(float(fy[c].max()))
        fs = rig_metrics.foot_slide(pos[:, lg["ix"]["K"]], c, v_expected=np.array([0, 0, -v]), fps=FPS, H=H)
        slide.append(fs["slide"])
    if clear:
        put("stance_clear_H", max(clear), G["stance_clear_H"], max(clear) <= G["stance_clear_H"])
    if slide:
        put("foot_slide", max(slide), G["foot_slide"], max(slide) <= G["foot_slide"])
    # loop seam
    if clip.get("loop", track.loop):
        seam = rig_metrics.loop_seam({"rot": track.quats()})
        rt = np.asarray(clip.get("root_t", sk.rest[sk.root] + track.root_off), float)
        seam_t = float(np.linalg.norm(rt[-1] - rt[0])) / H
        put("loop_seam_deg", seam, G["loop_seam_deg"], seam <= G["loop_seam_deg"] and seam_t <= 1e-3, root_H=round(seam_t, 6))
    # clip-specific
    if "head_hold" in track.info and ctx.tip:
        hm = np.asarray(track.info["head_hold"], bool)
        tip = pos[:, sk.i(ctx.tip)]
        drift = 0.0
        f = 0
        while f < len(hm):
            if not hm[f]:
                f += 1
                continue
            g = f
            while g + 1 < len(hm) and hm[g + 1]:
                g += 1
            res = tip[f:g + 1] - tip[f]
            res[:, 2] += v * (track.times[f:g + 1] - track.times[f])
            drift = max(drift, float(np.linalg.norm(res[:, [0, 2]], axis=1).max()))
            f = g + 1
        put("head_hold_drift_H", drift / H, G["head_hold_drift_H"], drift / H <= G["head_hold_drift_H"])
    if "beak_hold" in track.info and ctx.tip:
        hm = np.asarray(track.info["beak_hold"], bool)
        gy = (pos[:, sk.i(ctx.tip), 1] - ctx.floor) / H
        lo_, hi_ = G["beak_gap_H"]
        per = []
        f = 0
        while f < len(hm):
            if not hm[f]:
                f += 1
                continue
            g = f
            while g + 1 < len(hm) and hm[g + 1]:
                g += 1
            per.append(float(gy[f:g + 1].min()))
            f = g + 1
        ok = bool(per) and all(lo_ <= x <= hi_ for x in per)
        worst = max(per, key=lambda x: abs(x - 0.5 * (lo_ + hi_))) if per else None
        put("beak_gap_H", worst, [lo_, hi_], ok, per_peck=[round(x, 5) for x in per])
    if name == "Sleep" or track.info.get("lying") and track.loop:
        bv = ctx.belly_v if ctx.belly_v.any() else np.ones(len(ctx.P), bool)
        bg = float(((Vs[:, bv, 1].min(1) - ctx.floor) / H).max())
        put("belly_gap_H", bg, G["belly_gap_H"], bg <= G["belly_gap_H"])
    if not clip.get("loop", track.loop):
        hold = float(track.info.get("hold_s", 0.3))
        k = max(2, int(round(hold * FPS)))
        tail = track.local[-k:]
        held = float(np.abs(tail - tail[-1:]).max()) if len(tail) else 0.0
        rt = track.root_off[-k:]
        held = max(held, float(np.abs(rt - rt[-1:]).max()) / H)
        put("holds", held, 1e-6, held <= 1e-6, frames=k)
    if "abduct_peak" in track.info:  # the peak stroke: the larger of abduction and elevation
        pk = max(float(track.info["abduct_peak"]), float(track.info.get("elevate_peak") or 0.0))
        amin = float((clip.get("params") or {}).get("abduct_min", 20))
        put("abduct_deg", pk, amin, pk >= amin - 1e-9)
    fp = clip.get("params") or {}
    if "tip_rise_H" in track.info and (fp.get("rise_min_H") is not None or fp.get("clear_min_H") is not None):
        # Flap: the wing SKIN must visibly come up (a flank flare with no rising wing is not a flap). Measured on the
        # skinned points whose top bone is a wing bone, not on the wing-tip joints (the joints rose 0.155H on the drawn
        # chicken while its wing skin rose 0.066H at the top and stayed flush with the back)
        sk_ = wing_skin(ctx, Vs, track)
        if sk_ is not None:
            rv, cv = sk_["rise_H"], sk_["clear_H"]
            extra = {"tip_rise_H": round(float(track.info["tip_rise_H"]), 5)}
            if fp.get("rise_min_H") is not None:
                rmin = float(fp["rise_min_H"])
                put("wing_rise_H", rv, rmin, rv >= rmin - 1e-9, **extra)
            if fp.get("clear_min_H") is not None:
                cmin = float(fp["clear_min_H"])
                put("wing_clear_H", cv, cmin, cv >= cmin - 1e-9)
        else:
            put("wing_rise_H", 0.0, fp.get("rise_min_H"), False, why="no skin points on the wing bones")
    reasons = [_reason(k, d) for k, d in out.items() if not d["ok"]]
    return {**out, "passed": not reasons, "reasons": reasons}


def _reason(key: str, d: dict) -> str:
    v, lim = d["value"], d["limit"]
    if key == "stretch_p99":
        return f"stretch {v:.2f}x > {lim}"
    if key == "cshape":
        return f"C-shape extent {d.get('ext_ratio')} over {lim}"
    if key == "floor_min_H":
        return f"goes {-v:.3f}H through the floor"
    if key == "beak_gap_H":
        return f"beak gap {d.get('per_peck')}H outside {lim}"
    if key == "abduct_deg":
        return f"wing stroke {v:.0f} deg < {lim}"
    if key == "wing_rise_H":
        return f"wing skin rises {v:.3f}H < {lim}H"
    if key == "wing_clear_H":
        return f"wing skin clears the back by {v:.3f}H < {lim}H"
    return f"{key} {v} vs {lim}"


def _track_from_clip(clip: dict, sk: gaits.Skel, out_of: list[int] | None = None) -> gaits.Track:
    """A Track rebuilt from a clip dict (rot keyed by skel index or, with out_of, by output index)."""
    from scipy.spatial.transform import Rotation as R
    times = np.asarray(clip["times"], float)
    tr = gaits.Track(sk, len(times) - 1, loop=bool(clip.get("loop", True)))
    inv = {o: s for s, o in enumerate(out_of)} if out_of is not None else None
    for k, q in clip["rot"].items():
        j = sk.i(k) if isinstance(k, str) else (inv[int(k)] if inv is not None else int(k))
        tr.local[:, j] = R.from_quat(np.asarray(q, float)).as_matrix()
    tr.root_off = np.asarray(clip["root_t"], float) - sk.rest[sk.root]
    tr.contact = {k: np.asarray(v, bool) for k, v in ((clip.get("track") or {}).get("contact") or {}).items()}
    return tr


# ---------------------------------------------------------------- the synthesiser

def _spec_ok(spec: dict, ctx: gaits.Ctx) -> list[str]:
    need = spec.get("needs") or []
    miss = []
    for n in need:
        if n == "legs" and len(ctx.legs) < 2:
            miss.append("legs")
        elif n == "neck" and len(ctx.neck) < 3:
            miss.append("neck")
        elif n == "beak" and not ctx.tip:
            miss.append("beak")
        elif n == "wings" and not ctx.wing:
            miss.append("wings")
    return miss


def synthesise(plan: dict, skel, P, F, J, W, H, names: list[str] | None = None) -> tuple[list[dict], dict]:
    """Every clip of plan['clips'], gated with the energy search. Returns (clips, report)."""
    t_start = time.time()
    sk = skel if isinstance(skel, gaits.Skel) else gaits.Skel(skel, H=H)
    smooth_info = None
    if (plan.get("caps") or {}).get("wings") and F is not None:
        # a soft wing weight edge (plan chains.wings.smooth_iters) before any clip, so every clip is gated on it
        J, W, smooth_info = _wing_smooth(P, F, J, W, names, sk, plan, H)
    Jsk, out_of = _remap(J, names, sk)
    Wf = np.asarray(W, float)
    G = _gates(plan)
    ctx = gaits.Ctx(plan, sk, P, F, Jsk, Wf, H)
    gater = _Gater(ctx)
    clips, rep_clips, failed = [], {}, []
    repair_info = None
    J_cur, W_cur = np.asarray(J).astype(int), np.asarray(W)
    for spec in plan.get("clips") or []:
        name = spec["name"]
        gen = GENERATORS.get(spec.get("gen"))
        min_e = float(spec.get("min_energy", 0.5))
        step = float(G.get("energy_step", 0.8))
        tries = FLAP_TRIES if spec.get("gen") == "flap_folded" else int(spec.get("tries", G.get("tries", 4)))
        attempts, best = [], None
        reasons: list[str] = []
        if gen is None:
            reasons = [f"no generator '{spec.get('gen')}'"]
        miss = _spec_ok(spec, ctx) if gen else []
        if miss:
            reasons = [f"the skeleton lacks {', '.join(miss)}"]
            gen = None
        e = 1.0
        repaired = False
        sp_ = spec.get("params") or {}
        # the wing stroke's peak angle from the fold (elevation is Flap's main stroke, abduction rides on it)
        abd = max(float(sp_.get("abduct") or 0), float(sp_.get("elevate") or 0)) if sp_.get("abduct") is not None else None
        abd_min = sp_.get("abduct_min")
        while gen is not None:
            if e < min_e - 1e-9:
                reasons.append(f"energy {e:.2f} would be below min_energy {min_e}")
                break
            if abd is not None and abd_min is not None and abd * e < abd_min - 1e-9:
                reasons.append(f"peak wing stroke {abd * e:.0f} deg would be below abduct_min {abd_min}")
                break
            if len(attempts) >= tries:
                reasons.append(f"no pass in {tries} tries")
                break
            tr = gen(ctx, spec, e)
            clip = to_clip(tr, name, spec, e, out_of)
            g = gate(clip, ctx.P, ctx.J, ctx.W, H, ctx=ctx, track=tr, gater=gater, plan=plan)
            att = {"energy": round(e, 4), "passed": g["passed"], "reasons": g["reasons"],
                   "gates": {k: v["value"] for k, v in g.items() if isinstance(v, dict)}}
            attempts.append(att)
            clip["gates"] = {k: v for k, v in g.items() if isinstance(v, dict)}
            best = clip
            if g["passed"]:
                reasons = []
                break
            reasons = [f"{r} at energy {e:.2f}" for r in g["reasons"]]
            if (spec.get("gen") == "flap_folded" and not repaired and e <= 0.8 + 1e-9
                    and (plan.get("caps") or {}).get("wings")):
                repaired = True
                J2, W2, info = _wing_repair(ctx, J_cur, W_cur, names, sk, plan)
                repair_info = info
                att["wing_weight_repair"] = {k: v for k, v in info.items() if k != "bones"}
                if info.get("applied"):
                    J_cur, W_cur = J2, W2
                    Jsk, _ = _remap(J_cur, names, sk)
                    ctx = gaits.Ctx(plan, sk, P, F, Jsk, np.asarray(W_cur, float), H)
                    gater = _Gater(ctx)
                    continue  # retry at the same energy on the repaired weights
            e *= step
        passed = bool(attempts) and attempts[-1]["passed"]
        if best is None:
            best = {"name": name, "times": np.zeros(1, np.float32), "rot": {}, "root": int(out_of[sk.root]),
                    "root_t": sk.rest[sk.root][None].astype(np.float32), "loop": bool(spec.get("loop", True)),
                    "track": {"contact": {}}, "params": dict(spec.get("params") or {}), "energy": 0.0, "gates": {}}
        best["passed"] = passed
        best["reasons"] = [] if passed else reasons
        clips.append(best)
        rep_clips[name] = {"gen": spec.get("gen"), "params": best.get("params"), "energy": best.get("energy"),
                           "min_energy": min_e, "attempts": attempts, "gates": best.get("gates"), "passed": passed,
                           "reasons": best["reasons"]}
        if not passed:
            failed.append({"name": name, "reasons": best["reasons"]})
    report = {"plan": plan.get("id"), "clips": rep_clips, "clips_ok": [c["name"] for c in clips if c["passed"]],
              "clips_failed": failed, "names": list(names) if names is not None else list(sk.names),
              "wing_weight_repair": repair_info, "wing_weight_smooth": smooth_info,
              "seconds": round(time.time() - t_start, 2),
              "J": J_cur, "W": W_cur}
    return clips, report


def _wing_smooth(P, F, J, W, names, sk: gaits.Skel, plan: dict, H: float):
    """rig_build.wing_weight_smooth on the caller's index space (J/W in `names` order, else skel order)."""
    import rig_build
    nm = list(names) if names is not None else list(sk.names)
    order = [sk.i(n) for n in nm]
    skd = {"names": nm, "H": H, "cx": sk.cx, "pos": sk.rest[order],
           "parent": [(-1 if sk.parent[i] < 0 else nm.index(sk.names[sk.parent[i]])) for i in order]}
    return rig_build.wing_weight_smooth(P, F, J, W, skd, plan)


def _wing_repair(ctx: gaits.Ctx, J, W, names, sk: gaits.Skel, plan: dict):
    """rig_build.wing_weight_repair on the caller's index space (J/W in `names` order)."""
    import rig_build
    skd = {"names": list(names) if names is not None else list(sk.names), "H": ctx.H, "cx": sk.cx}
    order = [sk.i(n) for n in skd["names"]]
    skd["pos"] = sk.rest[order]
    skd["parent"] = [(-1 if sk.parent[i] < 0 else skd["names"].index(sk.names[sk.parent[i]])) for i in order]
    return rig_build.wing_weight_repair(ctx.P, J, W, skd, plan)


# ---------------------------------------------------------------- report IO

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else None
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


def report_json(report: dict) -> dict:
    """The report without arrays (J/W), ready for clips_report.json."""
    return _jsonable({k: v for k, v in report.items() if k not in ("J", "W")})


def write_report(path: Path, report: dict) -> dict:
    rj = report_json(report)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rj, indent=1))
    return rj


def summary_lines(report: dict) -> list[str]:
    out = []
    for name, r in report["clips"].items():
        g = r.get("gates") or {}
        bits = []
        for k in ("stretch_p99", "cshape", "floor_min_H", "foot_slide", "loop_seam_deg", "head_hold_drift_H",
                  "beak_gap_H", "belly_gap_H", "abduct_deg", "wing_rise_H", "wing_clear_H"):
            if k in g and g[k].get("value") is not None:
                bits.append(f"{k} {g[k]['value']:.3f}")
        state = "ok  " if r["passed"] else "FAIL"
        out.append(f"{state} {name:<6} e={r.get('energy')} tries={len(r['attempts'])}  " + ", ".join(bits)
                   + ("" if r["passed"] else "  <- " + "; ".join(r["reasons"])))
    return out


# ---------------------------------------------------------------- CLI

def _clips_from_glb(doc: dict, binary: bytes, keep: list[str]) -> list[dict]:
    """Existing clips (bake_clips shape, rot keyed by joint name) from a clips.glb, for the ones not re-made."""
    import rig
    skin = doc["skins"][0]
    jn = [doc["nodes"][j]["name"] for j in skin["joints"]]
    node_name = {i: n.get("name") for i, n in enumerate(doc["nodes"])}
    out = []
    for an in doc.get("animations", []):
        if an["name"] not in keep:
            continue
        rot, root_t, times = {}, None, None
        for ch in an["channels"]:
            smp = an["samplers"][ch["sampler"]]
            tt = rig.accessor(doc, binary, smp["input"])[:, 0]
            val = rig.accessor(doc, binary, smp["output"])
            nm = node_name[ch["target"]["node"]]
            times = tt if times is None else times
            if ch["target"]["path"] == "rotation" and nm in jn:
                rot[nm] = np.asarray(val, np.float32)
            elif ch["target"]["path"] == "translation" and nm == "root":
                root_t = np.asarray(val, np.float32)
        ex = an.get("extras") or {}
        out.append({"name": an["name"], "times": np.asarray(times, np.float32), "rot": rot, "root": "root",
                    "root_t": root_t, "loop": ex.get("loop", True), "params": ex.get("params"),
                    "energy": ex.get("energy")})
    return out


def _body(doc: dict, binary: bytes):
    import rig
    mi = next(i for i, n in enumerate(doc["nodes"]) if "mesh" in n and "skin" in n)
    pr = doc["meshes"][doc["nodes"][mi]["mesh"]]["primitives"][0]
    a = pr["attributes"]
    P = rig.accessor(doc, binary, a["POSITION"]).astype(float)
    J = rig.accessor(doc, binary, a["JOINTS_0"]).astype(int)
    W = rig.accessor(doc, binary, a["WEIGHTS_0"]).astype(float)
    Fc = rig.accessor(doc, binary, pr["indices"]).reshape(-1, 3).astype(int) if "indices" in pr else np.arange(len(P)).reshape(-1, 3)
    skin = doc["skins"][doc["nodes"][mi]["skin"]]
    names = [doc["nodes"][j]["name"] for j in skin["joints"]]
    return P, Fc, J, W, names


def main(argv: list[str] | None = None) -> int:
    import rig
    import rig_plan
    import rig_template
    ap = argparse.ArgumentParser(description="Re-synthesise a built template's clips (no re-skin).")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--clips", default=None, help="comma-separated clip names to re-make (default: all)")
    ap.add_argument("--report", action="store_true", help="print the per-clip gate table")
    ap.add_argument("--plan", default=None, help="plan id (default: rig/plan.json, template.json, ground_bird)")
    a = ap.parse_args(argv)
    asset = rig.OUT / a.stem
    tdir = asset / "rig" / "template"
    skj, rig_glb = tdir / "skeleton.json", tdir / "rig.glb"
    if not skj.is_file() or not rig_glb.is_file():
        print(f"rig_clips: no built template in {tdir} (skeleton.json + rig.glb)", file=sys.stderr)
        return 1
    if a.plan:
        plan = rig_plan.load(a.plan)
    elif (asset / "rig" / "plan.json").is_file():
        plan = json.loads((asset / "rig" / "plan.json").read_text())
    elif (tdir / "template.json").is_file() and json.loads((tdir / "template.json").read_text()).get("plan"):
        plan = rig_plan.load(json.loads((tdir / "template.json").read_text())["plan"])
    else:
        plan = rig_plan.load("ground_bird")
    skel = json.loads(skj.read_text())
    clips_glb = tdir / "clips.glb"
    src_path = asset / "psx.glb"
    if clips_glb.is_file():
        cdoc, cbin = rig.read_glb(clips_glb)
        P, Fc, J, W, names = _body(cdoc, cbin)
        src_doc, src_bin = (rig.read_glb(src_path) if src_path.is_file() else (cdoc, cbin))
        old = (cdoc, cbin)
        print(f"weights: {clips_glb.name} Body ({len(P)} verts)")
    else:
        raw = asset / "rig" / "skinned_raw.glb"
        if not raw.is_file() or not src_path.is_file():
            print("rig_clips: need rig/template/clips.glb or rig/skinned_raw.glb + psx.glb", file=sys.stderr)
            return 1
        rdoc, rbin = rig.read_glb(raw)
        Pr, _, Jr, Wr, names = _body(rdoc, rbin)
        src_doc, src_bin = rig.read_glb(src_path)
        pr = src_doc["meshes"][0]["primitives"][0]
        P = rig.accessor(src_doc, src_bin, pr["attributes"]["POSITION"]).astype(float)
        Fc = rig.accessor(src_doc, src_bin, pr["indices"]).reshape(-1, 3).astype(int)
        from scipy.spatial import cKDTree
        near = cKDTree(Pr).query(P)[1]
        J, W = Jr[near], Wr[near]
        old = None
        print(f"weights: skinned_raw.glb mapped onto psx.glb ({len(P)} verts)")
    want = [s.strip() for s in a.clips.split(",")] if a.clips else None
    p2 = copy.deepcopy(plan)
    if want:
        unknown = [w for w in want if w not in [c["name"] for c in plan.get("clips") or []]]
        if unknown:
            print(f"rig_clips: not in plan {plan.get('id')}: {unknown}", file=sys.stderr)
            return 1
        p2["clips"] = [c for c in plan["clips"] if c["name"] in want]
    if old is not None and ((p2.get("chains") or {}).get("wings") or {}).get("smooth_iters"):
        p2["chains"]["wings"]["smooth_iters"] = 0  # clips.glb's weights were softened when it was made
    H = float(skel.get("H") or np.ptp(P[:, 1]))
    clips, rep = synthesise(p2, skel, P, Fc, J, W, H, names=names)
    J, W = rep["J"], rep["W"]
    rp = tdir / "clips_report.json"
    if want and rp.is_file():
        prev = json.loads(rp.read_text())
        prev_clips = prev.get("clips") or {}
        prev_clips.update(report_json(rep)["clips"])
        rep_all = {**prev, **report_json(rep), "clips": prev_clips}
        rep_all["clips_ok"] = [n for n, r in prev_clips.items() if r.get("passed")]
        rep_all["clips_failed"] = [{"name": n, "reasons": r.get("reasons")} for n, r in prev_clips.items() if not r.get("passed")]
    else:
        rep_all = report_json(rep)
    # keep the clips that were not re-made (from the existing clips.glb), in plan order
    keep = []
    if want and old is not None:
        keep = [c for c in _clips_from_glb(old[0], old[1], [c["name"] for c in plan["clips"] if c["name"] not in want])]
    by = {c["name"]: c for c in keep + clips}
    ordered = [by[c["name"]] for c in plan["clips"] if c["name"] in by]
    ok = set(rep_all.get("clips_ok") or [])
    out_clips = [c for c in ordered if c["name"] in ok]  # failed clips stay out of the GLB (listed in the report)
    rig_template.write_clips(rig_glb, src_doc, src_bin, J, W, out_clips, clips_glb, names=names)
    rp.write_text(json.dumps(rep_all, indent=1))
    if (tdir / "template.json").is_file():
        rig_template.write_template_json(tdir, plan, rep_all.get("clips_ok") or [])
    if a.report:
        for line in summary_lines(rep):
            print(line)
    print(f"wrote {clips_glb} ({len(out_clips)} clips) and {rp}; failed: "
          f"{[f['name'] for f in rep_all['clips_failed']] or 'none'} ({rep['seconds']}s)")
    return 0 if not rep["clips_failed"] else 2


if __name__ == "__main__":
    sys.exit(main())
