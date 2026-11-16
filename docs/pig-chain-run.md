# The pig, fixed, and one timed run of the whole chain (pipeline#4, 2026-11-16)

Machine: the Mac (M5 Max, 128 GB), one job at a time through the factory (`:8766`) and the workbench (`:8768`).
Asset: `out/mlg-pig`, from its photo `out/mlg-pig/input.png`.

## 1. The three failing checks, and what fixed each

| check (before) | what it really was | fix | after |
|---|---|---|---|
| review **pieces 3** (limit 1) | not a leg: the curly **tail**. The shape comes with it as a dozen thin fragments; the cut kept two of them as specks beside the rump | `psx_lowpoly.bridge_pieces`: every loose piece takes its vertex nearest the body onto that body vertex (within 10% of the model's longest side), **creatures only** (a prop's wheels and post are real parts). `psx.json` records it as `bridged`. `tests/test_psx_bridge.py` | **pieces 1**, 1,205 triangles |
| review **paint 0.16** (limit 0.08) | the photo bake carried the photo's own shade (dark belly) against TRELLIS's flat pink paint | `--delight` takes a **strength** (`generate.sh`, factory, workbench `generate`): the pig's bake at 1.0 overshoots (0.173), at 0.65 it is the best of a sweep | **paint 0.084**, still over the limit (see below) |
| rig check **head 20° off its bone** (fail) | the rig (made 2026-10-28) was built on an older mesh whose head sat 4 cm off the body's middle; `psx.glb` had been re-cut since | re-rig on the new cut (`rig`, quadruped, horse): 506 s | **head 14° off: a warn** (a fail is over 15°); **no fail**, 4 other warns (below) |

### Paint: not passing, and why

Delight sweep on the same inputs (the paint-vs-bake patch, worst view, limit 0.08):

| delight | 0 (before) | 0.5 | 0.6 | 0.65 | 0.7 | 0.75 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| paint | 0.160 | 0.099 | 0.085 | 0.080 | 0.085 | 0.104 | 0.152 | 0.163 | 0.173 |

(on the first TRELLIS paint; on the paint made by the later make, 0.62 to 0.68 all scored 0.083 to 0.089, and the pig that
is kept scores **0.084**.) The belly and side patches are gone (side views at most 0.047). What is left is the
**rump** (`xside` view): in the patch the paint is 221/168/157 and the bake 213/149/143, about 15 lightness apart.
It reads as a patch only after the check's colour gain: delight leaves the whole bake about 15 to 18% lighter than
TRELLIS's flat paint (gain 0.85), the check scales it down by that, and the rump, which was already light, falls
below the paint. With gain 1 the xside patch is 0.054, but then every other view fails (back 0.37): the bake is
globally lighter than the paint. The drawn back view really has a pale-to-red rump gradient and an anus ring that the
flat paint lacks, so part of the difference is drawn detail. More strength sweeps will not move it (about 30 bakes
have plateaued); the choices are to accept 0.084, to loosen the limit a little, or to darken the bake to the paint's
level (which makes the pig less like its photo).

### Rig check on the fixed pig

`level: warn`, no fails: head 14° off its bone (fail line 15°), poor body-plan match 2.1×, "predicted skeleton faces
the other way", Eating stretches 3.1× (it was 2.12× on the October rig: new with the re-rig), Trot jumps 28° where it
loops. Per-clip stretch p99: Idle 1.12, Walk 1.77, Trot 1.73, Run 1.73, Eating 3.1, Kick 1.81, Sleep 3.03, Death 2.25.
Whether "warn" counts as "passes" is Stephen's call.

### Head: the check is touchy on a short head

`head_turn` measures the head-owned vertices' centre against the neck→nose line. On the pig the neck joint is only
0.05 m behind that centre, so a 1.3 cm sideways offset reads as 14°. Rest-pose head yaw measures -1.7° and the head
is 0.6% off centre. Right ear 150 vertices, left ear 105 (the cut's, not wrong ownership). Not changed.

## 2. One fresh end-to-end run (photo to export), timed

Run on the same stem, so each stage is a take (`pick_take` brought the fixed pig back afterwards: concept t18,
mesh t31, rig t5; the run is concept t20, mesh t33, rig t6, export t1).

| stage | tool | time |
|---|---|---|
| Draw the four views from the photo | `draw_views` (Qwen-Image-Edit) | **580 s** (one redraw: side views did not match; load 41 s, attempt 236 s, views check 41 s) |
| Make: shape, paint, photo bake, stack | `generate` quality textured, paint trellis, fill photo, delight 0.65 | **446 s** = isolate 4, shape 42 (Hunyuan3D-2mv turbo: load 14, gen 12), reduce 1, TRELLIS paint 354 (voxels 258), bake 14, fill 4, inspect 5, stack 13, preview 5 |
| PS1 cut | `psx` height 34, facing right, head_weight 2, faces 1100 | **8.7 s** |
| Rig | `rig` quadruped, horse | **307 s** (pred 124, skin 104, render 10) |
| Export | `POST /export/mlg-pig` | **under 1 s** (model, psx obj/glb, rigged.glb, fbx, stack) |
| **Total** | | **1,343 s, 22.4 min** (job times from `out/_logs/activity.jsonl`; the stage splits inside a job are from its log) |

For comparison, the fixed pig's own make (supplied TRELLIS mesh, so no draw and no shape) took 206 s, its rig 506 s.

### What broke or was weak

1. **The fresh pig is worse than the fixed one.** The drawn views were consistent (height spread 0.05, side IoU
   0.65) but the shape from them is a stubby body, silhouette IoU to the photo **0.55**, and the paint is blotchy red.
   The review: **paint 0.133 (fail)**; `head_offset` **could not find a head** ("body longer than wide only") and the
   sheet's head rows show the rump; magenta spill 3% (the fixed pig: 0.01%). pieces 1, mirror 0.98, 1,098 triangles
   pass. This is the chain's weak spot: the supplied TRELLIS mesh (`trellis/v24.glb`) made the good pig, the
   from-the-photo shape does not.
2. **Rig on the fresh pig: no fail, 4 warns**: poor body-plan match 2.5×, "predicted skeleton faces the other way" (the
   rig's own red-patch cue; the declared facing won), Sleep stretches 3.1×, Trot jumps 31° where it loops. Head 0.9° off.
3. **"Predicted skeleton faces the other way" on the fixed pig too**, not in the October rig's check; probably the redder rump of the new bake reading as a face to the red-patch cue (not tested). Harmless here (the drawn facing wins) but it is a false alarm.
4. **Draw is the slowest stage (10 min) and its first attempt failed** the side-view check.
5. **Models are shared**: another card's TRELLIS/shape jobs were running on the same Mac; this run waited for them. Nothing in the tools shows that: `/status` is idle while another worktree's script holds 20 GB.
6. **The workbench refuses a symlinked `out/`** ("path escapes factory root"), so a card working in a git worktree
   cannot link the main checkout's pig in; this run used an APFS clone of `out/mlg-pig` in the worktree. The fixed
   pig therefore lives only in that clone (`out/` is never committed); the main checkout's `out/mlg-pig` is still the
   old failing pig until the clone is copied back.

## 3. Re-run

```sh
.venv/bin/python scripts/pig_check.py [stem] [review|rig]   # computed now; exit 1 on a failed check (rig: a fail-level check)
```
