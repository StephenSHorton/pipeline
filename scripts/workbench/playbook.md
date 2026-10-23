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
Mesh: a long animal drawn tall gets its length back from proportion_check. If shape.json's proportion skipped, use mesh_edit stretch on z. For legs or head, use the proportion ops (thicken_legs, shorten_legs, widen_head, trim_belly, tidy_legs). When one refuses, give it the parameter the refusal names (top, head, start, line). For the PS1 cut, use psx with head_weight 2; for legs, tidy_legs then leg_weight. A leg left as its own piece after the cut is pinched, not a crumb: weight it, don't delete it.
Rig, after every rig or weights/joint change:
a. rig_review the stem. Read the check line, each clip's stretch p99/max, tears, foot slide and problems.
b. Read the per-bone numbers (no pictures): edges past 2x per clip and the bones that own them, weight shares, and the pivot check (a leg's root joint over its own leg; over 0.25 of a body length is wrong).
c. Only for the flagged bones, rig_review heat "flagged" (side/front/q/back, rest and worst frame; temp images).
d. Diagnose before editing. A weights problem is a bone owning the wrong region, bleed across legs, or a thigh that owns nothing: fix it with POST /rig/<stem>/weights. A joint problem is a pivot in the wrong place: fix the joint, then rig with reuse_skin true. The sheep's mid-body hind pivot was a joint problem that weight edits barely moved.
e. Show before and after in the app's Compare view (Rig tab), not as image files. The activity feed already shows Stephen each call (Follow Claude), so take no status pictures.
<!-- /digest -->

## Working rules

- **One model, then critique.** Make a single asset, show it, stop. Fix the pipeline from Stephen's critique, then
  make the next. Why: on the MLG farm list seven makes were chained after the first, every one needed the same fixes,
  and the batch was wasted.
- **Iterate inside one asset.** Every workbench write snapshots a take of the stem's stage (asset_info lists them,
  pick_take goes back). A rig and a weights save make rig takes. Never create `foo_v2` or test stems. If you need a
  scratch copy, use `out/_wbtest_*` and delete it afterwards. Why: sibling stems filled the library with copies of
  one model, and the app's Versions panel and Compare view only work within a stem.
- **One big model at a time.** Never load a model yourself (no Qwen, Hunyuan, skin-tokens or Ollama in a script).
  draw_views, generate and rig are factory jobs, and the factory has one job slot. Poll job_status, and start the
  next job only when the slot is free. Why: two big models co-loaded ran the 128 GB Mac out of memory, and neither
  /status nor `ps` shows MLX memory.
- **The photo is a reference.** Every 3D make draws all four turnaround views from the photo and/or the words:
  draw_views first, then generate with `image: out/<stem>/flat/source.png`. Why: the factory passes the drawn views
  only for that picture (views_handoff); any other image is built as one picture and marked degraded `single_view`.
- **Numbers first, pictures when the numbers point somewhere.** measure (mesh) or rig_review (rig) before and after
  every change. Use look, compare_reference or heat images to see what a number flagged, not as a status shot. Why:
  numbers compare across takes and cost nothing to keep; pictures fill the context and the disk.
- **Report honestly.** Give the before and after numbers. When a change barely helps, say so. Keep "what this tool
  can fix" apart from "what needs a different stage": a poor body-plan match is a template problem, and no weight
  edit fixes it. Why: Stephen improves the pipeline from what you report.
- **Tell Stephen through the app.** The activity feed shows each workbench call as it runs (Follow Claude opens
  the asset and each model as it lands). For a before and after, point him to Compare in the viewer, where the left
  side is an earlier version. Why: he asked not to clutter space with pictures that aren't needed.

## Mesh

- **Proportions after the shape.** generate.sh runs `proportion_check.py` right after `mv_orient.py`, because each
  drawn view sits in a square quarter of the sheet and that squashes a long animal: the sheep's photo was 0.74
  height/length, its drawn side 1.04. It stretches the length back only when the photo is side-on; the outcome is in
  `shape.json` → `proportion` (applied, factor, why). If it skipped and the model is still short, use mesh_edit
  `stretch` (axis z for a creature facing +Z), then measure again.
- **Proportion ops from a critique** (mesh_edit, only vertices move, so the UVs and texture stay): `thicken_legs`
  (factor 1..3), `shorten_legs` (0.4..1), `widen_head` (1..2), `trim_belly` (a keel of wool or spikes under the
  belly) and `tidy_legs` (straight, tapered, smooth legs; the sheep used thickness 1.5). Each op reads top, start and
  head from the shape. When it can't, it refuses and says why: give the parameter the refusal names (`top`, `head`,
  `start`, or `line` for trim_belly) rather than forcing another op.
- **The PS1 cut.** `psx` with `head_weight` 2 for creatures, so the ears and face survive. For legs, choose one way:
  `tidy_legs` and then `leg_weight` (the sheep: 2.5), or `legs: "clean"` with `leg_scale` (the sheep: 1.6). The two
  don't combine. Use `quality: "textured"` and then the psx tool, not `quality: "psx"`, whose facing check asks Ollama.
- **Crumbs and slivers.** The cut drops floating crumbs (under 2% of the area) on the detailed mesh, before
  decimating. After decimating it drops only slivers (under 0.1%), because decimation pinches thin legs and ears off
  into pieces of their own. So a leg that comes out as its own piece is pinched, not a crumb: give the legs weight,
  and don't remove_floaters it.
- **Bakes** take cutouts only (`lifted.png`, `views/lifted/*.png`). `fill: "photo"` (the default) gives every texel
  the photo. Use `delight: true` only for a photo with a side in shadow; it lifts dark material (a cow's patches) to
  grey too.
- **Clay** (clay_set, clay_mesh) is for shapes a photo can't give depth to (the hay bale). Then generate with
  `mesh` and `paint: false`, which loads no model.

## Rig

1. **Rig through the factory:** the `rig` tool (POST /rig/<stem>). Give `class` (or `plan`) when you know it;
   without one, rig.py asks the planner model what the subject is. `reuse_skin: true` keeps the learned skin
   weights when the mesh and joints match; limb joints may have moved (a pivot fix), but if a body joint moved or a
   joint moved over 0.6x height it learns them again. Either way the log and rig.json say which happened.
2. **Read the rig check:** `rig_review {stem}` → `check`: the summary line, per-clip stretch p99/max and tears,
   foot slide and problems. The check's p99 is over every edge at every frame. `check.stale` means the rig changed
   after the check was made.
3. **Per-bone review, no pictures:** in the same answer, for each clip, read the edges whose worst stretch passes
   2x (`over`) and the bones that own them (the top bone of both ends, UV-seam copies welded first). Then read
   `weights` (each bone's share) and `pivots`: each leg's root joint (the horse template's back_leg_pelvis,
   front_scapula) should sit over its own leg. The pivot check is the xz distance to the leg at the leg line,
   divided by the body length, and over 0.25 is flagged. `diagnosis` puts the numbers into words.
4. **Heat only where flagged:** `rig_review {stem, heat: "flagged"}` (or a list of bone names) renders each bone's
   heat map from side, front, three-quarter and back, at rest and at the frame where that bone stretches most, as
   one image. The frames go to a temp folder that is deleted afterwards (`keep_images` keeps them). Nothing is
   written into out/.
5. **Diagnose before editing.** Decide whether it is weights or joints:
   - Weights: a bone owns the wrong region, weight bleeds from one leg onto another, or a thigh owns nothing.
     Fix it with a weights edit (the app's Weights panel, or POST /rig/<stem>/weights), which saves a rig take.
   - Joints: a pivot is in the wrong place. Fix where the joint goes (on the predicted-skeleton path that is
     rig_pred.place_pivots), then `rig` with `reuse_skin: true`.
   - The sheep: its hind pivot sat 0.39 of a body length from the leg, so Walk swung the leg about a point mid-body.
     Weight edits barely moved the stretch. Putting the pivot back over the leg (0.13) took Walk from 53 edges past
     2x to 13 (rig_review; 106 to 26 before seam copies were welded), and the check's Walk p99 from 2.12 to 1.62.
6. **Show the change in Compare:** the Rig tab's Compare toggle shows an earlier version on the left and the
   current one on the right, with one camera and one clip. Point Stephen there instead of making before/after
   images.
