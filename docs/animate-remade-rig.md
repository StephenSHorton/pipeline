# The walker's walk on LTX-2.3 with the remade rig (pipeline#37)

pipeline#36 moved the walker rig's knee from layer 13 to 18 and its hip from the crotch (22) to 30, so Animate's
fit to the drawing (pipeline#34) stretches the legs far less: thigh 1.528 → 1.155, shin 1.423 → 0.909. This run
asks whether LTX follows the walk any closer for it.

## How it was run

`scripts/animate.py --stem walker --cycle walk --facing right --seed 42` (and `--seed 7`), the way pipeline#34 ran,
on scratch copies of `out/walker` with Apple Vision's `bin/bodypose`, the walker's own side view and the default
prompt. Two copies, same code (main at pipeline#36):

- **remade rig:** today's `out/walker` (Stack Version t28, `knee 18`, `hip 30`).
- **old rig:** pipeline#34's own copy (`knee 13`, hip at the crotch). A control: pipeline#33's placement fix
  landed after pipeline#34 measured, so this shows whether the code alone moved the number.

## Result

| run | seed | joint % | limb % | thigh fit | shin fit |
|---|---:|---:|---:|---:|---:|
| old rig, pipeline#34 as measured | 42 | 4.03 | 5.63 | 1.528 | 1.423 |
| old rig, pipeline#34 as measured | 7 | 4.11 | 5.66 | 1.528 | 1.423 |
| old rig, today's code | 42 | 4.03 | 5.63 | 1.528 | 1.423 |
| old rig, today's code | 7 | 4.11 | 5.66 | 1.528 | 1.423 |
| remade rig, today's code | 42 | **3.91** | 5.90 | 1.155 | 0.909 |
| remade rig, today's code | 7 | **4.36** | 6.87 | 1.155 | 0.909 |

- The control repeats pipeline#34 to the hundredth, so the rig is the only change.
- Joint error (`pose_check` mean): level at about 4%. Seed 42 is 0.12 better, seed 7 0.25 worse; the two-seed mean
  goes 4.07 → 4.14. Two seeds cannot show a direction at this spread.
- Limb error is worse on both seeds (5.63 → 5.90, 5.66 → 6.87).
- Drift after the turn grows a little (mean size 0.41 / 0.36% → 0.74 / 0.65%), still under 1.5% in every frame.

So the remade rig makes the skeleton honest about the drawing (the fit hardly stretches it) without making LTX
follow it closer: the 4% left is not the leg stretch.

The clips, `anim.json`, `control.json`, the run scripts, `table.py` and a side-by-side GIF are in
`out/_experiments/pipeline-37-remade-rig/` on MACBOOK-STEPHEN (not in git).
