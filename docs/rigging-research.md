# Rigging research: why the chicken broke, and what to rig with next

October 20, 2026. Branch `rigging` (PR #27). It covers out/a-ps1-style-graphics-of-a-chicken-2, with out/psx-cow and
out/operator as controls, plus a spider and two props in one experiment.

How to read the citations:

- `rr/` means the scratch folder
  `/private/tmp/claude-501/-Users-stephenhorton/57bc55f1-4326-4edc-94ae-25e4359ff405/scratchpad/rr/`. The raw numbers,
  logs, tools and every render are there. It is a temp folder. The renders of
  Hunyuan3D meshes were removed from this doc because the Hunyuan licence restricts showing its output outside its
  Territory; their captions remain.
- `rig.py:NNN` line numbers refer to the code **before** the fixes (commit dd2fa05), because that is the code that
  broke. The fixes are commits 41db319 (E2) and 7cfd243 (E3).
- H means the model's height. Distances are given as a fraction of it, for example 0.12H.
- The old chicken take is still on disk: `out/a-ps1-style-graphics-of-a-chicken-2/takes/rig/t1` is the broken one,
  `t2` is after E2 and `t3` is after E3, which is the one shipped now.

---

## 1. What went wrong with the chicken

1. **The skeleton was backwards.** The bird template's head points along +Z, but the chicken's head is at −Z: its red
   comb vertices average z = −0.32. So the template's head bones sat behind the chicken's tail, and its tail bones ran
   out past the beak.
2. **The skeleton was also about 2× too long and 2× too wide.** The fit only matched height (scale 1.498). The
   template is a horizontal flying seagull, while the chicken is compact and upright. Skeleton extent over mesh extent
   was 2.05 / 1.00 / 2.06 on x/y/z, and 56% of the joints were outside the mesh's box.
3. **The skinner then tied body parts to the wrong bones.** The feet went to wing-tip bones (wing_5_L had 105 foot
   vertices), the comb went to tail and wing-feather bones, and the head got 0% head-bone weight.
4. **Flap and Glide swing those wing bones 106–128°.** That turned the legs into flat blades and the comb into the red
   spike in your screenshot. Idle looked fine only because no Idle bone turns more than 5.2°.
5. **The checks said "warn: No tears · worst stretch 29.5×".** The only warning was "Right wing 0.97 m off"
   (`takes/rig/t1/rig/summary.json`). Stretch had no fail level, and the tear check could not fail.

*(Render removed: Chicken: top row is the rig you saw, middle is E2, bottom is E3 (shipped).)*

*Top row: the old rig (t1). Idle and Walk are fine; Flap 0.48 and Glide 0.32 are the flat slab with blade legs. Middle
row: E2, which fits the template before skinning. Walk is good, and the flight clips are still a flight pose, so they
are not shipped. Bottom rows: E3, shipped now. Only Idle and Walk are shipped, with feet on the floor line; the `nf`
frames are before the floor pass. Source: `rr/e3/img/sheet_chicken_3way.png`.*

All of this reproduces exactly. A scratch re-rig with the old code gave the same 1209 dropped pulls, 5.7% folded weight,
"Right wing 0.97 m off" and 29.5× (`rr/chicken/rerig.log`).

---

## 2. Why our checks said "warn", and the check that would have caught it

### Why each check passed

| Check | What it said | Why it could not catch this |
|---|---|---|
| Tears | 0 | `share_corners` (rig.py:575-586) forces every copy of a point onto the same weights *before* the tear check runs. On current-code rigs the check cannot fail, and it says nothing about *which* bone owns a point. |
| Stretch | "worst 29.5× (Flap)" in the text line | No rule ever turned stretch into a level. `level` is the minimum over check rows (rig_inspect.py:58), and only tears made a row (rig_inspect.py:288-303). `stretch_max` is also a single edge: the operator's Idle_A reads 56.6× and looks fine. `stretch_p99` was computed (rig_motion.py:135) but never used. |
| Fit | "Right wing 0.97 m off", warn | Only the single worst bone was shown, and only as a warn (rig_inspect.py:263-268). Six bones were over the 0.12H line (wing_5_R 0.80H, wing_5_L 0.72H, wing_4_R 0.61H, wing_3_R 0.47H, tail_3 0.44H, wing_3_L 0.39H). The "wing" was actually the **feet** bound to a wing bone (census). |
| Head | null | It needs a `head` bone with a tip that points forward (rig_inspect.py:113-127). |
| Rest check | null | `rest_check.measure` returns None for any mesh taller than 0.9× its length (rest_check.py:60-62). The chicken is 1.21 tall and 1.233 long, so no facing or head measurement was ever made. |

### The checks that separate broken from fine

All three were computed on the chicken, the cow and the operator, with `rr/chicken/tools/checks.py`,
`rr/skinres/tools/bindcheck.py` and `rr/chicken/checks/*.json`.

| Check | When it runs | Chicken | Chicken turned 180° | Cow (current) | Operator | Rule |
|---|---|---|---|---|---|---|
| **B2: far-bone motion.** Per vertex, sum of w·displacement from bones more than 0.15H farther than its nearest bone. Reported as worst-clip p99 / max share of vertices over 0.1H. | after clips | **1.81H / 25.2%** (Flap) | 1.46H / 29.7% | 0.40H / 3.0% | 0 / 0% | red if share > 10% or p99 > 0.75H |
| **A2: far-bone weight.** Share of vertices with more than half their weight on far bones. | after skinning | **15.9%** | 23.7% | 1.7% (old on-disk rig: 5.6%, its known bad back leg) | 0% | amber > 3%, red > 8% |
| **Joints outside the mesh box** (2% margin) | before skinning | **56%** | 53% | 7% | 5% | red > 20% |
| **bone_far_outside_0p08H.** Share of bone samples that are outside the mesh *and* more than 0.08H from its surface. | before skinning | **0.634** | 0.619 | 0.141 | 0.022 | red > 0.30 |
| **FOOT_off_chain.** Share of floor vertices owned by a bone that isn't in a floor chain. | after skinning | **0.564** | 0.539 | 0 | 0 | red > 0.2 |
| **stretch_p99.** Worst 1% of welded edges. | after clips | **15.1 (Flap), 16.7 (Glide)**; Idle 1.17 | — | 2.96 (Sleep) | 3.64 (Death_D) | now: warn > 3, fail > 6 |

Sources: `rr/chicken/` diagnosis, `rr/skinres/bind_*.json`, `rr/census/`.

Two things to note:

- **The pose-stage check alone would have stopped the chicken before the 68 s skinner ran.** A check before skinning
  is cheap, and the chicken failed it by a wide margin (56% against a red line of 20%).
- **B2 is not enough by itself.** With a correct binding (E2, fit applied), the chicken's Flap *passes* B2 (1.5% of
  points, p99 0.14H), yet it still renders as a flat slab (image 7). That frame has stretch_p99 16.4 and 4.7× its rest
  extent. So flight clips need stretch_p99 as a second gate. This is now in the code (rig.py at the
  `if trial:` B2/stretch gate, around line 1276 on 7cfd243).

*(Render removed: After a correct bind, Flap still lies in a flight pose.)*

*E2 run a+b+c+d (`rr/e2/img/v_chABCD.png`). The bind is right: comb on head bones, legs on leg bones, Walk steps
cleanly. But Flap and Glide still put the body into the seagull's horizontal pose, and the yfront frames show the
body sides spread into a wide slab. This is why Flap and Glide are not shipped on a ground bird.*

**Landed on `rigging`:**

- the pose gate (`rig_fit.py`, "pose gate" section);
- B2 as `far_motion` (rig.py:868 on 7cfd243), which writes a fail row in rig_inspect.py:298-309;
- stretch_p99 rows (warn > 3, fail > 6), with the summary line now quoting the p99;
- a floor row (warn > 0.03H, fail > 0.10H);
- the poor-body-plan-match and unsupported-clips rows;
- the fit row now says "+N more" instead of hiding them.

**Not landed:** A2 and FOOT_off_chain as summary rows. They are measured in scratch only, by `rr/skinres/tools/bindcheck.py`.

---

## 3. Root causes in our pipeline, ranked

1. **Nothing knows which way the mesh faces.**
   - intent.json and psx/psx.json say `"facing": "right"` for the chicken, but the picture (image 2) shows it facing
     **left**. psx_lowpoly.py:15-17,36 turns the mesh so that the "facing" side becomes +Z, so the chicken's tail ended
     up at +Z.
   - rig.py assumes +Z forward (comment at rig.py:365), and `fit_transform` (rig.py:278-294) has no rotation term.
   - The cow has the same setting and really does face right, which is why it worked (head at z = +0.86).
   - Mesh facing is not a constant the code can assume.
2. **The fit matches height only.**
   - It is one uniform scale (mesh height ÷ template joint height) plus an offset (rig.py:278-294).
   - Nothing fits legs, wings, tail or spine for any class. Only humanoid arms (rig.py:318-350) and the horse's
     neck/head chain (rig.py:354, 398-436) get fitted.
   - The bird template at Idle frame 0 is 2.1× longer than tall; the chicken is about 1.0×.
   - Mesh2Motion, where our templates come from, expects a person to rotate the model, scale the skeleton and drag each
     bone into the mesh (`vendor/mesh2motion-app/src/create.html` ~106-130, `MeshDragBonePlacement.ts:188-220`). We
     dropped that step and replaced it with nothing.
3. **One template per class, and "avian" means a flying seagull.**
   - Four classes map to four templates (rig.py:55-81); avian gets Flap/Glide by default (rig.py:79).
   - The planner has 7 fixed answers and no "other" (rig.py:258-265).
   - Five vendored CC0 templates (fox, dragon, snake, shark, kaiju) are never used.
   - A chicken and an eagle get the same rig and clips. The farmer hat came back as "bust" at 0.60 and was refused with
     the wrong message (`out/psx-farmer_hat/rig/rig.log`).
4. **Checks had no teeth** (section 2):
   - stretch had no level;
   - tears could not fail;
   - only one fit outlier was shown;
   - the rest check skips upright meshes;
   - nothing checked facing, length/height ratio, ground contact, or which body part a bone owns.
5. **The skinner trusts whatever skeleton it gets.**
   - skin-tokens is called with `--fit none` (rig.py:510) and checks nothing about where the skeleton sits.
   - The SkinTokens paper trains with rewards for bones lying inside the mesh (arXiv 2602.04805 §4.4.1). A skeleton
     with 56% of its joints outside is out-of-distribution input.
   - This is not a skinner bug. Rebinding the same bad skeleton with Blender bone heat also left Flap broken (images 5
     and 6).
6. **No floor contact anywhere.**
   - Retargeting is world-delta rotations plus a scaled root offset, with no IK (rig.py:601-629).
   - The old cow's hooves went 0.16–0.23H under the floor in Walk/Run/Eating, and it floated +0.13H in Sleep and
     Death (census).
7. **Low: the far-influence filter.**
   - `local_weights` (rig.py:534-535) compares an influence with the vertex's nearest *influence*, not its nearest
     *bone*, and always keeps the top bone. So a foot whose 4 influences are all wing bones keeps them.
   - It matters only when the fit is wrong. In E2, fixing it changed A2 by less than 0.0033 once the fit was right.

*(Render removed: The source picture: the chicken faces left, but intent.json says "right".)*

*`out/a-ps1-style-graphics-of-a-chicken-2/lifted.png` (copy: `rr/census/chicken_lifted_small.png`).*

*(Render removed: The posed template over the chicken: head bones (pink, far left) behind the tail; tail bones (green) out past the beak; wing bones (orange) through the body.)*

*`rr/chicken/img/02_posed_over_psx_side.png`.*

*(Render removed: Weights by group: tail colours on the comb, orange wing weight on the body and legs.)*

*`rr/chicken/img/05b_weights_group_skel_side.png`. Measured shares (`rr/chicken/tools/parts.py`):*

- *comb: tail 49%, wing 35%, head 4%;*
- *head region: wing 58%, tail 42%, head 0%;*
- *legs (y < 0.3): 53% wing.*

*The raw skinner output has the same shares, so cleanup did not cause this.*

*(Render removed: side by side, the learned skinner (shipped then) and Blender bone heat on the same bad skeleton.)*

*Flap t=0.48, from `rr/skinres/img/st_asis_*` and `heat_asis_*`. Bone heat removes the blade legs (A2 0.159 → 0.020,
FOOT 0.564 → 0), but the bird is still thrown into a flight pose (B2 share 16.7%). Changing the skinner does not fix a
wrong skeleton.*

---

## 4. The landscape

These are the options the decision depends on, picked from about 60 that were reviewed. Every repo listed was opened
and grepped. The full tables are in the research notes (`rr/research-academic/`, `rr/prodtools/`, `rr/skinres/`,
`rr/anim-src/`).

Column meanings: "Any skel." = handles an arbitrary skeleton. "Clips after" = where the animation comes from once the
rig exists.

| Option | Kind | Code / weights | License | Runs on this Mac | Any skel. | Clips after | URL |
|---|---|---|---|---|---|---|---|
| **skin-tokens.cpp `rig` mode** (SkinTokens/TokenRig) | academic, **already vendored** | yes / yes (F16 GGUF) | MIT weights, Apache-2.0 port | **yes, Metal**. Rig mode 24 s chicken, 57–88 s cow | yes, predicted (bones are unnamed `bone_N`) | none; needs our chain labels | https://github.com/VAST-AI-Research/SkinTokens · https://huggingface.co/LocalAI-io/SkinTokens-GGUF |
| UniRig | academic | yes / partial | MIT | no (flash-attn, spconv, CUDA ≥ 8 GB) | yes | none | https://github.com/VAST-AI-Research/UniRig |
| Puppeteer | academic | yes / yes | Apache-2.0 | no (xformers, pytorch3d cu118) | yes | video-guided optimisation | https://github.com/Seed3D/Puppeteer |
| RigAnything | academic | yes / yes | **Adobe noncommercial** | patchable, untested | yes | none | https://github.com/Isabella98Liu/RigAnything |
| MagicArticulate | academic | skeleton only | Apache-2.0 | no (flash-attn) | yes | none | https://github.com/Seed3D/MagicArticulate |
| Make-It-Animatable | academic | yes / yes | MIT | no (cu121 pins) | **humanoid only** | Mixamo | https://github.com/jasongzy/Make-It-Animatable |
| AniGen / AniGen-mac | academic (image → rigged asset) | yes / yes | MIT (code and HF weights), but `extensions/CUBVH` is instant-ngp code, non-commercial (THIRD_PARTY_LICENSES.md) | fork claims MPS (vendor/AniGen-mac) | yes | `anigen/4drecon`: fit the rig to a monocular video (added July 2026, separate install) | https://github.com/VAST-AI-Research/AniGen |
| **Mesh2Motion** (source of our templates and clips) | production, open | yes / CC0 rigs and clips | MIT + CC0 | **yes** (web app; we use its data) | 9 fixed templates | template clips | https://github.com/Mesh2Motion/mesh2motion-app |
| Tripo Rig Check + Auto Rig | production, cloud | no | commercial (25 credits per rig) | cloud | 7 body plans + "others" | about 11 biped presets (90+ on v1.0), one walk or march each for quadruped, hexapod, octopod, serpentine and aquatic, none for avian (https://developers.tripo3d.ai/en/docs/animations-retarget) | https://developers.tripo3d.ai/en/docs/animations-rig |
| Meshy Rigging API | production, cloud | no | commercial | cloud | API: humanoid only. Web app: humanoid / quadruped / smart rig (https://www.meshy.ai/features/ai-auto-rigging) | humanoid library | https://docs.meshy.ai/en/api/rigging |
| Anything World | production, cloud | no | commercial | cloud | category templates, adjusted by hand | per category | https://anything-world.gitbook.io/anything-world/api/preparing-your-3d-model.md |
| Armature (FabricatorStudio) | production, desktop | no | free (terms vary by source) | **no** (Windows/NVIDIA) | modular chains + prop | none | https://fabricator.studio/armature/ |
| Rigify metarigs (bird, cat, horse, wolf, shark) | production, ships with Blender | yes, in the owner's Blender 5.2 (`addons_core/rigify/metarigs/Animals/`) | GPL-2.0-or-later source; licence of the bone layout as data unclear | **yes** (not run in the pipeline; bones dumped in `rr/critic/bird.json`) | fixed metarigs, placed by hand | none | Blender.app/Contents/Resources/5.2/scripts/addons_core/rigify/metarigs/Animals/bird.py |
| Auto-Rig Pro | production, Blender add-on | paid source | $25–50 | runs in Blender (not installed) | Smart = humanoid; animals by hand | remap | https://www.lucky3d.fr/auto-rig-pro/doc/auto_rig.html |
| Spore (Hecker et al. 2008) | production technique | paper | n/a | n/a | yes, via capability tags | ground-relative IK; clips gated on `UprightSpine`, `HasFeet` | https://chrishecker.com/Real-time_Motion_Retargeting_to_Highly_Varied_User-Created_Morphologies |
| **Blender bone heat** (Pinocchio heat) | method | Blender | GPL tool, unrestricted output | **yes**, ~1 s. Needs a weld at 2 mm first. | any supplied | template clips | https://people.csail.mit.edu/ibaran/papers/2007-SIGGRAPH-Pinocchio.pdf |
| robust-laplacian + potpourri3d | method (own heat or geodesic skinner) | yes | MIT | **yes** (cp314 arm64 wheels verified) | any | template clips | https://pypi.org/project/robust-laplacian/ |
| libigl (bbw, fast_winding_number) | method | yes | MPL-2 (tetgen AGPL) | **yes** (2.6.3 wheel verified) | any | template clips | https://pypi.org/project/libigl/ |
| Robust weight transfer (Abdrashitov 2023) | method (template weights → our mesh) | yes | MIT reference code | pure libigl, untested | the template's | template clips | https://github.com/rin-23/RobustSkinWeightsTransferCode |
| Orient Anything V2 | method (facing from an image) | yes / yes (5 GB) | CC-BY-4.0 | needs a CUDA patch | n/a | n/a | https://github.com/SpatialVision/Orient-Anything-V2 |
| AnyTop | animation (any skeleton, diffusion) | yes / yes | MIT code; Truebones data must be bought | CPU fallback, one `device='cuda'` to patch | yes, BVH with joint names | generates BVH | https://github.com/Anytop2025/Anytop |
| Motion2Motion | animation (cross-topology transfer) | yes | **no licence** | CPU (`-d cpu`) | yes | BVH | https://github.com/LinghaoChan/Motion2Motion_codes |
| SinMDM | animation (variations of one clip) | yes / yes | MIT | CPU fallback | yes | variations | https://github.com/SinMDM/SinMDM |
| SATA | animation (cross-species latent) | yes / yes | Apache-2.0 code | unknown, leans CUDA | yes | BVH | https://github.com/zzysteve/SATA |
| UniMate | animation (rigged asset + text → motion, 7 body plans) | **no code** | n/a | n/a | yes | the ideal shape | https://arxiv.org/abs/2609.05415 |
| AnimaX | animation | empty repo | Apache-2.0 | n/a | yes | IK joint angles | https://github.com/anima-x/anima-x |
| Truebones Zoo | clip library (75+ animals, Chicken included) | data | $99, royalty-free for games, no redistribution | data | one skeleton per animal | retarget | https://truebones.gumroad.com/l/skZMC |
| Wan2.2 I2V-A14B + Lightning (mlx-video) | video source for M6 | yes / yes | Apache-2.0 | **yes**: 10 min per 49 frames, a real cow walk (docs/spikes-rigging.md) | n/a | needs a tracker and fit step | https://huggingface.co/lightx2v/Wan2.2-Lightning |
| s&box `SetIk` / `SetBoneTransform` | runtime procedural | engine API | ours | n/a | any, once chains are labelled | foot planting, springs, look-at | https://sbox.game/api/allclasses/Sandbox.SkinnedModelRenderer/SetIk |

**Also looked at and rejected:**

| Option | Why |
|---|---|
| RigNet | GPL, CUDA, 2020 |
| Anymate | no licence file |
| RigMo | CC BY-NC, no weights |
| ViP-Rig, SPRig, Rigel3D, HumanRig | no code |
| PartField | CUDA, licence unread |
| Hunyuan3D-Part | licence not stated |
| SAMPart3D | trains per shape, GPU |
| AnimateAnyMesh | outputs morph targets, not bones; its own port's README calls the motion weak |
| TapMo, R2ET, MoMa | humanoid or CUDA-pinned |
| MANN dog mocap, BITE, Animal Avatar | non-commercial |
| MagicPony / 3D-Fauna | tiny-cuda-nn, so CUDA only |
| Rodin | does not rig |
| Ponymation | empty repo, no licence |
| AI4AnimationPy | CC BY-NC 4.0 |
| OmniMotionGPT | MIT, but SMAL quadruped skeleton only; its data comes from DeformingThings4D, terms not read |
| WalkTheDog | MIT, but it aligns motion between existing datasets, so it needs chicken motion first |
| Unity, Unreal, Houdini, Rokoko | not mesh-to-skeleton tools |

I found no paper or code named "DreamRig" or "Walk the Random Skeleton".

**What production tools teach us:**

- **No product works out facing for itself.**
  - Meshy needs the face toward +Z, or it returns HTTP 422 and refunds the credits.
  - Anything World needs +Z in Maya / −Y in Blender, and its `auto_rotate` is off by default.
  - Auto-Rig Pro wants −Y; Houdini wants +Z.
- **No product skins blind.**
  - Tripo runs a free Rig Check (0 credits) first.
  - Mixamo needs markers placed first.
  - Unity shows bones in red or green.
  - Anything World asks you to adjust the skeleton by hand.
- **No product ships a ground bird.** Tripo documents no avian clips, and the Mesh2Motion and Bake3D birds are flyers.
  Spore is the shipped design for "one animation, many body plans": it tags parts and gates clips on predicates such as
  an upright spine (`rr/prodtools/spore.txt` lines 129, 352-356, 508-510).

---

## 5. What the experiments showed

The experiments ran one at a time on this Mac (128 GB; free+inactive memory was 66–70 GB before each model load).

### E1: skin-tokens `rig` mode as the whole binder

It predicts its own skeleton and weights. Source: `rr/e1/`, `res/table.json`.

- **Chicken.** It predicts 15 joints (tail 5, neck/head 3, two 3-joint legs, no wings) in 24 s.
  - 0% of joints outside the box; extent 0.58 / 0.94 / 0.79.
  - A2 0, FOOT 0.
  - Head 100% on head bones, legs 77% on leg bones (92% with `--postprocess`), comb 72%.
  - A hand pose and the mapped Walk give B2 0.006H / 0% and stretch_p99 1.26–1.33.
  - **It passes every gate.**
- **Cow.** Plausible in motion, with Walk B2 0.218H / 3.7%.
  - It **fails A2** (6.4% beam): belly and thigh vertices are owned by spine bones.
  - The leg chains stop 0.10–0.17H above the floor.
  - Its Sleep stretch_p99 is 3.1–4.1.
- **Spider.** Passes A2 and B2 with 8 legs found, but the "head" is a palp.
- **Operator.** B2 0. The radio gets no floor chain and is refused; the bust (farmer hat) is refused. Both correct.
- **Topology is not stable.** The cow gave 29 joints (beam) vs 35 (samples); the spider 46 vs 42.
  - The CLI fixes seed = 0 (skintokens.hpp:215), so the "samples" were one fixed draw per temperature. True stability
    was never tested.
- **`--postprocess` tears seams** on every asset (operator 192 points). It needs `share_corners` afterwards.
- **Verdict:** a great *fitting target*, not ready as the shipped rig. Bones are unnamed, its clips would come only
  from our own chain mapping, and the cow results are worse.

*(Render removed: skin-tokens' own predicted chicken skeleton: tail (green), neck/head (pink), legs (blue), all inside the body.)*

*`rr/research-academic/img/01_chicken_pred_skel_side.png`.*

*(Render removed: E1 spider, operator, props.)*

*`rr/e1/img/cs_spider_ops.png`. Top row: the spider's 8 predicted legs inside the mesh, and the mapped Walk. Bottom
row: the operator's weights, the operator's hand pose, and the small radio and hat skeletons (both refused).*

### E2: fit the template properly before skinning

Landed as 41db319: `scripts/rig_fit.py`. Each step was ablated on the chicken, cumulatively. Source:
`rr/e2/res/all_runs.json`.

The steps:

- **(a) facing:** register the template's own body at 4 yaws and choose by Chamfer distance; break ties with cues.
- **(b) per-axis proportions:** used when the scale ratio is over 1.6.
- **(c) per-chain fit:** legs, neck/head, tail, and wings folded onto the body sides.
- **(d) nearest-bone weight filter.**
- **(e) ground-bird clip set.**
- **(f) pose gate.**

| Chicken | Old rig | a | a+b | a+b+c | Final (a–f) |
|---|---|---|---|---|---|
| Joints outside box | 56% | 53% | 9% | 2% | **0%** |
| Extent x/y/z | 2.05/1.00/2.06 | same | 0.98/1.10/1.03 | 0.86/0.90/1.02 | 0.89/0.88/0.97 |
| bone_far_outside | 0.634 | 0.619 | 0.074 | 0.008 | **0.004** |
| A2 | 15.9% | 8.9% | 0.5% | 0.3% | **0** |
| FOOT_off_chain | 0.564 | 0 | 0 | 0 | **0** |
| Head region on head + neck | 0% | 60% | 87% | 85% | **96%** |
| Legs on leg bones | 47% | 94% | 91% | 96% | **94%** |
| Walk B2 share / p99 | 12.1% / 0.35H | 0.4% / 0.07H | | | **0% / 0.07H** |
| Flap stretch_p99 | 15.1 | | | | 13.8 (**not shipped**) |

Notes:

- **Facing.** 180° was chosen every time. The Chamfer distances at 0/90/180/270 were 0.058 / 0.078 / 0.049 / 0.077, a
  thin margin of 1.19. So the decision went to the cues: toes, the narrower end and the red patch all said −Z, and the
  VLM said LEFT/RIGHT on the two side renders, which also means −Z.
- **Facing alone is not enough.** Step "a" only gets the comb to 72%, and Flap is still 12.5% B2. The proportion fit
  (b) is what pulls the skeleton inside the body.
- **Skinner swap.** With the fit right, skin-tokens beat bone heat on A2 for all three assets: chicken 0 vs 3.2%, cow
  1.8% vs 2.2%, operator 0 vs 3.6%. The skinner is not the bottleneck.
- **Cow** (current → E2): back-toe outlier 0.218H → 0.01H; worst B2 share 3.0% → 2.4%; back legs on leg bones
  72% → 79%. But A2 went 1.67% → 1.84%. 70 of those 76 vertices are belly vertices next to the floor-to-hips root bone.
- **Operator:** unchanged. B2 0 in all 8 clips, A2 0.
- **Real app re-rig** of the chicken through POST /rig (`takes/rig/t2`): "warn · worst stretch 15.3× (Walk)". The
  warnings were "Left upper leg 0.39 m off (+4 more)", which is a bone owning 5 vertices, and "Poor body-plan match
  (2.3×)". Flap and Glide were left out.

### E3: the predicted skeleton as the fitting target, plus a floor pass

Landed as 7cfd243: `scripts/rig_pred.py`. Source: `rr/e3/res/*.json`.

E3 uses the skin-tokens predicted skeleton as the target for fitting the template, but only when the template is a poor
body-plan match (ratio > 1.6, the chicken) or when the geometric fit fails the pose gate. It skins with skin-tokens,
bakes the template's own clips, then runs a floor pass in numpy:

- it grounds the root in lying clips;
- it lifts leg tips to the floor.

**Chicken** (format: old → E2 → E3/shipped)

| Measure | Old | E2 | E3 (shipped) |
|---|---|---|---|
| Joints outside | 56% | 0% | 0% |
| CD-J2B (÷H) | 0.194 | 0.078 | **0.018** |
| A2 / FOOT | 0.159 / 0.564 | 0 / 0 | 0 / 0 |
| Walk B2 p99 / share | 0.349 / 12.1% | 0.074 / 0 | **0.009 / 0** |
| Walk stretch_p99 | 2.56 | 2.41 | **1.44** |
| Walk lowest point / float | −0.015 / +0.004 | −0.028 / +0.038 | **0.000 / 0.000** |
| Head on head bones | 0% | 63% | **90%** |
| Legs on leg bones | 47% | **94%** | 78% |

E3's legs share is lower than E2's: 15% of the leg weight sits on the folded wings. The other chicken findings:

- The FBX contains only Idle and Walk, so the screenshot frames cannot occur. The Blender re-import check is in
  `rr/e3/res/fbxcheck.txt`.
- The shipped summary is "warn". The warnings are "Poor body-plan match 2.3×", which is true of the template, and
  "Flap, Glide left out", which is intended (`takes/rig/t3`).
- The predicted head chain agrees with E2's Chamfer facing. On its own, the "top of the mesh" cue would have named the
  tail as the head.

**Cow**

- E3's predicted-target fit was *worse* than E2 on A2 (2.31% vs 1.84%) and on some B2 clips (Death share 4.5% vs
  2.4%). By the experiment's own rule, the cow keeps E2's fit plus the floor pass.
- Floor pass, lowest point before → after:

  | Clip | Before | After |
  |---|---|---|
  | Walk | −0.07 | 0.00 |
  | Kick | −0.11 | 0.00 |
  | Eating | −0.23 | −0.005 |

- Float before → after: Sleep +0.26 → 0, Death +0.42 → +0.005.
- Summary "warn": "Left back lower leg 0.21 m off (+4 more)", "Sleep stretches 3.6×".

**Operator:** same binding as E2. Death_D floated 0.456H; it now floats 0.007H.

**Wall time** (app start to finish): chicken 102 s (prediction 23.6 s, skin 72 s), cow 94 s, operator 157 s. The
learned skinner is about 70–120 s of each run, and its time barely depends on vertex count. Profiled, 91% of it is
`qwen.generate_skin_codes` (census).

**Unmet criteria:**

- stretch_p99 < 3 on cow Sleep (3.6) and operator Death_D (3.54). It is squash at folded knees and elbows, a limit of
  linear-blend skinning (`rr/e3/tools_stretchwho.py`).
- Foot sliding was not measured.

*(Render removed: Template fitted to the predicted chicken skeleton (E3).)*
*(Render removed: E3 weights by group: head (pink), tail (green), wings folded on the sides (orange), legs (blue).)*

*`rr/e3/img/chE3skin_posed_side.png`, `chE3skin_groups_side.png`.*

*(Render removed: Cow: old rig, E2, E3 (shipped), E3 before the floor pass.)*

*`rr/e3/img/sheet_cow_3way.png`. Rows: old rig, E2, shipped, shipped without the floor pass. On the old rig and E2,
Eating puts the head through the floor line, and Sleep and Death hang in the air. In the shipped row (third), the head,
hooves and lying body sit on the line.*

*(Render removed: Operator: old, E2, E3, E3 before floor.)*

*`rr/e3/img/sheet_operator_3way.png`. Walk is identical across all four. Death_D sits lower after the floor pass. The
grey line here is at the origin, not the operator's floor, so read it only as a reference.*

---

## 6. Options for us, ranked

| # | Option | Fixes | Cost | Risk | Evidence |
|---|---|---|---|---|---|
| 1 | **Keep E2 + E3 as landed** (fit before skinning, predicted skeleton as the target when the match is poor, floor pass, honest rows) | facing, proportions, wrong-bone binding, flight clips on a ground bird, floor contact, false "warn" | done; ~25 s more per poor-match rig | thresholds tuned on 3 assets; E3's leg share is lower than E2's | §5; 202 tests pass; real app runs `takes/rig/t3` |
| 2 | **Make facing an input contract** at the psx stage: check intent `facing` against an image or VLM cue, and turn the mesh or fail loudly | the root cause (§3.1) for every class, not just at rig time | ~1 day | the VLM was right on the chicken, cow and operator (`rr/e1/res/vlm.json`), but it was tried on only three creatures, so it needs the geometric cues too | every vendor in §4 does this |
| 3 | **Split avian into ground and flying birds; gate clips on posture** (Spore-style predicates) | Flap and Glide on chickens, permanently | small | needs the planner (M1) to emit it | E2 (e); Spore lines 508-510 |
| 4 | **Planner menu from the vendors** (biped/quad/avian-ground/avian-flying/serpentine/aquatic/multileg/**prop/other**), wire up the 5 unused CC0 templates, and refuse at low confidence | snakes, fish and props forced into the wrong class; the hat "bust" | 1–2 days | each new template needs its own fit-chain map | rig.py:55-81, 258-265; Tripo and Anything World menus |
| 5 | **Promote A2 and FOOT_off_chain to summary rows**, ignore bones with fewer than 20 vertices in the fit row, and treat the root bone specially in A2/B2 | the remaining "trust me bro" gaps | < 1 day | — | E2 follow-ups; the 5-vertex UpperLeg_L warning |
| 6 | **Runtime foot IK** in s&box (`SetIk`), plus spring chains for tail, comb and ears | foot sliding, uneven ground, secondary motion | medium | s&box side work | anim research; the cow's floor numbers |
| 7 | **M6 video fit** (the Puppeteer pattern): Wan2.2 video of the rest render, then a tracker and depth, then joint fit | clips for body plans that have no template (ground bird peck, spider) | large | CoTracker3 and VDA-Large are CC BY-NC; we need an Apache tracker and VDA-Small | Puppeteer `optimization.py:519`; docs/spikes-rigging.md |
| 8 | Ship skin-tokens `rig` mode as the rig itself, with chain-mapped clips | classes with no template | medium | unstable topology, unnamed bones, cow A2 fail, postprocess tears | E1 |
| 9 | Second binder (bone heat, own robust-laplacian heat, or Abdrashitov template-weight transfer) | a second opinion when skin-tokens is out of distribution | ~1 day each | lost to skin-tokens once the fit was right | E2 binder table |
| 10 | Cloud riggers (Tripo, Meshy) or CUDA research riggers | — | paid, or no Mac support | licence and dependency blockers | §4 |

### Recommendation

**Keep the template + clip library path, and make the fit trustworthy rather than replacing the rigger.** That is
options 1 → 2 → 5 → 3/4. The learned skeleton is used as a *fitting target*, not as the shipped rig. The evidence:

- once the skeleton sat inside the body, the same skinner bound the chicken correctly (A2 15.9% → 0);
- the models and products that could replace it are CUDA-only, cloud, non-commercial, or ship few or no creature clips (§4). The exceptions are AniGen-mac (MIT apart from CUBVH, claims MPS, has a video fit), which replaces the image → mesh stage rather than just the rigger and was never run here, and Tripo, which is paid cloud and has one walk or march per non-bird body plan.

### Phased plan

**This week**

1. Merge what is on `rigging` (41db319, 7cfd243) once you have looked at the three re-rigged assets in the app.
2. Option 5: A2 and FOOT rows, the < 20-vertex filter on fit outliers, root handling. Re-inspect old takes so they get
   the new rows; today rig_inspect only shows rows that rig.json already carries (E2 follow-up).
3. Option 2, the first part: at psx time, compare intent `facing` with the head-end cues used in E2 (toes, narrower
   end, red patch) and write a warning into psx.json. Fail only when the cues agree strongly against the intent.
4. Run the "next experiment" from the research: rig and score 5–10 more outputs (a snake or fish, a second biped, two
   props). That tells us whether the 1.6 ratio, the 1.3 tie margin and the 20% / 30% pose gate thresholds hold beyond
   three assets.

**With M1 (planner)**

- Option 4's class menu, including ground vs flying bird and an explicit prop/other outcome.
- Low confidence becomes a refusal with the right message.
- Clip sets per body plan (option 3).
- The planner can also answer "which end is the head" as a third facing cue.

**With M2 (rest-edit to "standing upright, facing forward")**

- This is the cleanest fix for facing. If the rest-edit image always shows the creature in profile with its head to
  image-right, legs straight down and wings folded (Anything World's published input rules), then psx_lowpoly's turn
  is right by construction. The rig-time yaw search becomes a check instead of a guess.
- Keep the E2 facing check as the gate that confirms it.

**With M6 (video fit)**

- Use Puppeteer's animation stage as the blueprint: Wan2.2 I2V on our rest render, then tracking and depth, then a
  joint fit on the fitted template skeleton.
- Gate every generated clip with B2, stretch_p99 and floor, exactly like template clips.
- This is the route to ground-bird pecks and walks without buying Truebones. Buying Truebones Zoo ($100) is the
  shortcut if you want chicken, flamingo or snake motion sooner; check its terms for in-house retarget sources.

**Later**

- Runtime foot IK (option 6).
- A hand-made CC0 ground-bird template: 3-bone digitigrade legs with curled toes, a short neck and folded wings, as in
  Auto-Rig Pro's bird notes. Blender's own Rigify bird metarig is close to this already: 75 bones, thigh/shin/foot with
  four toes per foot, a raised neck and beak, body about 1.4× longer than tall, but wings spread at rest
  (`rr/critic/rigify_bird_metarig.png`, copy: [`rigging-research/14_rigify_bird_metarig.png`](rigging-research/14_rigify_bird_metarig.png)). It has no clips, and its GPL source makes its licence as a template unclear.
- Watch UniMate and AnimaX for a code release; either would supply clips for any rigged asset.

---

## 7. What we did NOT try or verify

- **Sample size.** Only three creatures were studied end to end (chicken, cow, operator), plus one spider and two props
  in E1. Every threshold is tuned on those: the 1.6 ratio, the 1.3 tie margin, the 20% / 30% pose gate, the B2 10% /
  0.75H line, A2 3% / 8%, and stretch 3 / 6.
- **Topology stability.** skin-tokens `rig` mode was never really tested for it: the seed is fixed at 0
  (skintokens.hpp:215) and there is no `--seed` flag.
- **Not run or not built:**
  - skin-tokens `--geometric` binding;
  - the Abdrashitov weight transfer from the template's own skinned mesh;
  - our own robust-laplacian heat skinner.
- **Not measured:** foot sliding during stance (clips play in place).
- **Not tested:** s&box import. Only the FBX was re-imported, in Blender (`rr/e3/res/fbxcheck.txt`).
- **The E3 fallback** (ship the predicted rig with chain-mapped clips) was force-tested only. It has never triggered on
  a real run, and in that branch Walk goes −0.084H below the floor because no floor pass runs there.
- **VLM facing** was right on the chicken, cow and operator (`rr/e1/res/vlm.json`; the operator renders are in
  `rr/critic/vlm_operator_side_xside.png`). It has not been tried on other upright creatures.
- **Not fixed:** the farmer-hat refusal message ("Busts (head and shoulders)…"), and the intent.json `facing` value for
  the chicken.
- **No GPU contention check.** The skinner's time varied from 68 s to 162 s; I suspect the GPU is shared with the
  factory, but never confirmed it.
- **Desk research only, not run:** AnyTop, SinMDM, SATA, Motion2Motion, Puppeteer, RigAnything, Orient Anything V2,
  PartField, Hunyuan3D-Part. Licence and Mac claims for them come from their READMEs and source greps.
- **Pages I could not read:** Adobe's Mixamo help page returned 403; Mixamo's marker list comes from CG Channel (2012)
  and an Adobe forum post. Blender's Rigify manual also returned 403, but Rigify's metarigs are readable in the local
  Blender install.
- **Missing plan.** The Rigging Plan artifact (claude.ai/artifact/LB2iJnmxmFPZEPyi6Phhgj) could not be opened from this
  session, so the M1/M2/M6 names come from the task brief, not from the plan itself.
