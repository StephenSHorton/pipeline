# Playbook: driving the pipeline

How Claude makes, checks and fixes assets through the workbench. Each rule is something that has worked on a real
asset, with the reason. The part between the digest markers is what the MCP server sends at connect (Claude Code cuts
server instructions at 4,096 characters); the `playbook` tool returns this whole file. Add to it when a fix proves
itself; take a rule out when the code stops matching it.

<!-- digest -->
Workbench rules (the playbook tool has the full playbook, with the why for each):
1. Make ONE model, stop, and wait for Stephen's critique. Improve the pipeline from it, then make the next. Never run a whole asset list.
2. Iterate inside the asset: every write is a take of the same stem, and pick_take goes back. Never make foo_v2 or test stems.
3. One big model at a time: only factory jobs load models (draw_views, generate, rig), one slot. Check job_status before starting one.
4. The photo is a reference, never a view: draw_views first, then generate with image out/<stem>/flat/source.png. Any other picture builds from one picture.
5. Numbers before pictures: measure / rig_review before and after every change. look or heat only where the numbers point.
6. Report honestly. Give the numbers before and after. Say when a change barely helped, and separate what a tool can fix from what needs another stage.
7. Review before showing: psx returns the review (else call review). Fix what failed, then read the sheet: count the ears and eyes on the painted head row against the grey row. Each new critique becomes a check or a sheet row in review.py CRITIQUES.
8. Show Stephen in the app, never a PNG: call show (stem, stage, compare = the earlier take for a before and after); the app opens it in the viewer and Compare.
Mesh: a long animal drawn tall gets its length back from proportion_check. If shape.json's proportion skipped, use mesh_edit stretch on z. For legs or head, use the proportion ops (thicken_legs, shorten_legs, widen_head, trim_belly, tidy_legs). When one refuses, give it the parameter the refusal names (top, head, start, line). For the PS1 cut, use psx with head_weight 2; for legs, tidy_legs then leg_weight. A leg left as its own piece after the cut is pinched, not a crumb: weight it, don't delete it.
Rig, after every rig or weights/joint change:
a. rig_review the stem. Read the check line, each clip's stretch p99/max, tears, foot slide and problems.
b. Read the per-bone numbers (no pictures): edges past 2x per clip and the bones that own them, weight shares, and the pivot check (a leg's root joint over its own leg; over 0.25 of a body length is wrong).
c. Only for the flagged bones, rig_review heat "flagged" (side/front/q/back, rest and worst frame; temp images).
d. Diagnose before editing. A weights problem is a bone owning the wrong region or bleed across legs: fix it with POST /rig/<stem>/weights. A thigh above the leg line owns nothing by design; a stretched seam there needs a softer band, not thigh weight. A joint problem is a joint in the wrong place: fix where it goes, check it with rig_preview (seconds, no model), then rig. Leg pivots keep the skin (reuse_skin true); a moved body joint relearns it (about 3 minutes). The sheep's mid-body hind pivot and hips were joint problems that weight edits barely moved.
e. Show before and after in the app's Compare view (Rig tab), not as image files. The activity feed already shows Stephen each call (Follow Claude), so take no status pictures.
<!-- /digest -->

## Why the rules

- **One model, then critique.** On the MLG farm list seven makes were chained after the first, every one needed the
  same fixes, and the batch was wasted.
- **Iterate inside one asset.** asset_info lists a stem's takes; a rig and a weights save make rig takes. A scratch
  copy goes in `out/_wbtest_*` and is deleted afterwards. Sibling stems filled the library with copies of one model,
  and the Versions panel and Compare only work within a stem.
- **One big model at a time.** Never load a model in a script (no Qwen, Hunyuan, skin-tokens or Ollama). Two big
  models co-loaded ran the 128 GB Mac out of memory, and neither /status nor `ps` shows MLX memory.
- **The photo is a reference.** The factory passes the drawn views only for `out/<stem>/flat/source.png`
  (views_handoff); any other image is built as one picture and marked degraded `single_view`.
- **Numbers first.** Numbers compare across takes and cost nothing to keep; pictures fill the context and the disk.
- **Report honestly.** A poor body-plan match is a template problem that no weight edit fixes. Stephen improves the
  pipeline from what you report.
- **Tell Stephen through the app.** The activity feed shows each call (Follow Claude opens each model as it lands);
  Compare's left side is an earlier version. He asked for no pictures that aren't needed.

## Review

- **Every model is reviewed before Stephen sees it** (review.py, on every psx cut and at the end of generate.sh, into
  `workbench/review/`). `paint` compares a `--select` bake with Hunyuan's paint: the pig's ghost-ear bakes scored
  0.12-0.22, the fixed bake 0.054. The pig had gone to Stephen with two sets of ears and eyes.
- **The verdict is never a pass.** Count ears and eyes on the painted head row against the grey row (shading only, so
  a painted ear with no ear under it shows). The ghost ear was skin-coloured and passed every colour check.
- **A missed critique joins `CRITIQUES`**: a check when a number separates bad from good (give the margin), else a
  sheet row.

## Mesh

- **Proportions after the shape.** Each drawn view sits in a square quarter of the sheet, which squashes a long
  animal (the sheep: photo 0.74 height/length, drawn side 1.04). `proportion_check.py` stretches it back when the
  photo is side-on; see `shape.json` → `proportion`. If it skipped, use mesh_edit `stretch` (z for a creature facing
  +Z), then measure.
- **Proportion ops** move only vertices, so UVs and texture stay: `thicken_legs` (1..3), `shorten_legs` (0.4..1),
  `widen_head` (1..2), `trim_belly` (a keel of wool under the belly), `tidy_legs` (the sheep used thickness 1.5).
  When one refuses, give the parameter it names rather than forcing another op.
- **The PS1 cut.** `psx` with `head_weight` 2 for creatures. For legs, either `tidy_legs` then `leg_weight` (sheep
  2.5) or `legs: "clean"` with `leg_scale` (sheep 1.6); they don't combine. Use `quality: "textured"` then psx, not
  `quality: "psx"`, whose facing check asks Ollama.
- **Crumbs and slivers.** Before decimating the cut drops crumbs under 2% of the area; after, only slivers under 0.1%,
  because decimation pinches thin legs and ears into their own pieces. Weight such a leg; don't remove_floaters it.
- **Bakes** take cutouts only (`lifted.png`, `views/lifted/*.png`). `fill: "photo"` (default) gives every texel the
  photo. `delight: true` only for a photo with a side in shadow; it lifts dark material (a cow's patches) to grey.
- **Clay** (clay_set, clay_mesh) is for shapes a photo can't give depth to (the hay bale); then generate with `mesh`
  and `paint: false`, which loads no model.
- **From words (the MLG list).** Draw, generate `quality: "textured"` (fill photo), then `psx`. Hunyuan's paint shows
  where no view lands and comes out maroon on this key (the pig's back stripe, until fill photo covered it). Grey
  metal drawn on magenta takes a cast the despill misses: for a grey or white prop, generate with the drawn `mesh`
  and `paint: false`, then bake_photo views with pixels leaning both red and blue over green pulled to grey (orange
  wood is untouched). A round subject (watermelon) keeps no orientation: model it as clay and `bake_wrap` one
  lengthwise view round it, pulling red under green first to remove a magenta bounce. `delight` washed the melon to
  lime.
- **Read the drawn sheet before building.** The image model ignores some asks (a "lowered" mailbox flag came back
  raised) and draws one-sided details on both sides. Fix that on the mesh (paint the flag's back red), not by
  redrawing.
- **A complex vehicle needs more than 1,200 triangles:** the tractor lost its exhaust and got jagged wheels at 1,200;
  `faces: 1800` kept both. Say so when a model goes over budget.
- **Fix a shape the drawings won't make on the mesh.** The image model draws a hole through a jerry can's handle
  however it is told. `mesh_edit gaps` lists what would fill; `close` with a center and radius fills only that hole
  (closing everywhere filled the can's embossed panels); `carve` empties a part first. Rebuild with generate (mesh =
  the edit, image = flat/source.png) with **paint on**: with paint off the filled handle baked a grey slab.

## Rig

1. **Rig through the factory** (`rig`, POST /rig/<stem>). Give `class` or `plan` when you know it, else rig.py asks
   the planner model. `reuse_skin: true` keeps learned weights when mesh and joints match; limb joints may move, but
   a moved body joint or any joint moved over 0.6x height relearns them. The log and rig.json say which happened.
2. **Read the rig check:** `rig_review {stem}` → `check`: summary, per-clip stretch p99/max (over every edge at every
   frame), tears, foot slide, problems. `check.stale` means the rig changed after the check.
3. **Per-bone review, no pictures:** per clip, the edges past 2x (`over`) and the bones that own them (UV-seam copies
   welded first); `weights` (each bone's share); `pivots`, each leg's root joint (horse template back_leg_pelvis,
   front_scapula) as xz distance to its leg at the leg line over body length, flagged over 0.25. `diagnosis` puts it
   in words.
4. **Heat only where flagged:** `rig_review {stem, heat: "flagged"}` (or bone names) renders each bone's heat from
   side, front, three-quarter and back, at rest and worst frame, as one image, in a temp folder (`keep_images` keeps
   it). Nothing goes into out/.
5. **Diagnose before editing: weights or joints?**
   - Weights: a bone owns the wrong region, or one leg bleeds onto another. Fix with a weights edit (Weights panel or
     POST /rig/<stem>/weights), which saves a rig take.
   - A thigh above the leg line (`pivots` → `upper_above_line`) owns nothing on purpose: it keeps the wool off the
     leg. On the sheep, moving leg weight onto it made every clip worse (Run 27 → 43 edges past 2x); softening the
     leg band 0.12 → 0.18 helped (Walk 13 → 7, Kick 14 → 10).
   - rig_review counts stretch; the app's Rig check also counts squash. Judge by both: smoothing 12 instead of 6
     halved the sheep's stretched edges but pushed Sleep p99 from 2.8x to 3.3x (take t5; t6 went back to 6).
   - Test weight ideas offline: rig.clean_weights on rig/skinned_raw.glb with the rig's skeleton reproduces the rig's
     weights exactly, so a setting can be measured in seconds.
   - Joints: a joint in the wrong place. Fix its placement (rig_pred.register, trunk_anchors, place_pivots), check
     with `rig_preview` (seconds, no model, out/ untouched), then `rig` with `reuse_skin: true`. A moved body joint
     relearns the skin (about 3 minutes) and the weights differ, so say so when comparing takes.
   - Prefer the predicted skeleton's own joints over the template's proportions. On the sheep, the hips where its
     hind legs hang, the shoulder where its front legs hang and the neck root where its head starts (t9) took hips
     from 45% to 25% of the weight and Run 25 → 16, Sleep 18 → 9 edges past 2x. Placing them by horse offsets put
     the neck root mid-chest and sank Eating's head (t7, t8 were wasted re-rigs: preview first).
   - A bad pivot defeats weight edits: the sheep's hind pivot sat 0.39 of a body length from the leg; moving it back
     (0.13) took Walk from 53 edges past 2x to 13.
   - The pig: the predicted skeleton found three legs, hung all four mid-body and called an ear its head. rig_pred
     now mirrors a missing fourth leg, places the trunk by the legs' tops and runs an ear-shaped head chain on to the
     snout (Run 55 → 16, Eating 46 → 1). When one leg's weights are near zero, read the joint positions (rig_preview
     from the command line: the MCP server keeps rig_pred imported).
   - Clips: a clip that looks wrong is usually the floor pass (rig_pred.floor_pass) or the bake, not the weights.
     The viewer and rig_review open rig/rigged_allclips.glb. Judge "grounded" by the trunk's gap to the floor, never
     the lowest point (a lying sheep's folded hoof held its belly up). A horse clip's turn can move a very different
     share of our mesh (the horse's Eating folded the sheep's chest). Replay the floor pass offline on
     rig/rigged_nofloor.glb (tests/test_rig_floor.py shows how) before a re-rig.
6. **Show the change in Compare:** the Rig tab's Compare shows an earlier version left and the current one right,
   with one camera and one clip.
