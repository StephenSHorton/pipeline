# Rigging spikes (M0)

Measured on this machine (Mac17,6, 128 GB) on Oct 17, 2026, before building the rig stages. The plan these test is in
the Rigging Plan page (claude.ai/artifact/LB2iJnmxmFPZEPyi6Phhgj). Spike files are in `out/_spike/` (gitignored).

## Skinning: skin-tokens.cpp on Metal

`vendor/skin-tokens.cpp`, built with `-DSKINTOKENS_ENABLE_VULKAN=OFF -DGGML_METAL=ON` (needs `brew install
nlohmann-json`), weights `LocalAI-io/SkinTokens-GGUF` F16 (MIT). `skin` mode against a Mesh2Motion template
skeleton, `--device auto` (Metal), `--fit global`, on a 12k-face game LOD:

| mesh | template | time |
|---|---|---|
| psx-cow `mesh_game.glb`, turned so the head points +Z | `rig-horse.glb` (56 joints, 47 used) | 75 s |
| operator `mesh_game.glb` | `rig-human.glb` (66 joints) | 86-91 s |

Weights segment cleanly: limbs per segment, spine bands, head, tail, hooves.

Findings that change the plan:

- **skin-tokens resets bone frames.** The output skeleton keeps joint positions but every bone's rest rotation is
  identity. Mesh2Motion clips store rotations relative to the template's bone frames, so they can't be played as is.
  What works: play the clip on the template, take each bone's world rotation change from a reference pose, and give our
  bone the same world change (`local = parentWorld⁻¹ · delta · restWorld`), with the root moved by the scaled offset.
  Cow Walk/Eating and human Walk/Farm_Watering/Chop_Tree play correctly this way.
- **Bind against a template posed like the mesh.** The human template is a T-pose; generated people stand arms down.
  Binding to the T-pose and retargeting from Idle_A swings the hands through the body. Posing the template at
  `Idle_A` frame 0 before `skin` (a rewritten copy of the rig GLB) and using Idle_A as the retarget reference fixes it.
  `--fit articulated` did not move the arms here (it needs an animated skeleton).
- **Raw weights include far bones** (a hand weighted to a foot tip). `--postprocess` removes most of it; a few
  slivers remain on wide swings. M3 needs a distance filter that drops influences from bones far from the vertex.

## Export to s&box

s&box ModelDoc imports FBX (several animations per file), not glTF, and keeps at most 4 weights per vertex
(wiki.facepunch.com/sbox/ModelDoc_FAQ). Blender 5.2.2 (Steam install:
`~/Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/MacOS/Blender`) converts the
skinned GLB to FBX headless; re-importing shows the 56-bone armature, 56 vertex groups and at most 4 influences.
Clips are baked into the FBX once the retarget writes them into the GLB (M3).

## Rest pose: Qwen-Image-Edit-2511 through mflux

`mflux-generate-qwen-edit --model qwen-image-edit-2511 --quantize 8` (Apache-2.0, ~58 GB download).

| input | edit | steps | time | result |
|---|---|---|---|---|
| psx-cow photo (lantern slide) | stand square, side view, white background | default | 23 min | both seeds keep the coat pattern, halter and face; seed 1 square stance, seed 2 turns its head |
| same | same | 8, Lightning LoRA | 8.5 min | clean, identity kept, looks more modern (adds an ear tag) |
| farmer drawn mid-stride with a scythe | neutral A-pose, no tool | 8, Lightning LoRA | 4 min each | both seeds: standing neutral, tool removed, face/hat/overalls kept; arms hang straight down rather than at 45° |

Lightning LoRA: `lightx2v/Qwen-Image-Edit-2511-Lightning` 8-step bf16 (Apache-2.0), `--lora <file> 1.0 --steps 8
--guidance 1.0`. Use it by default; offer the full-step edit as "best quality".

## Video: Wan2.2 through mlx-video

`vendor/mlx-video` (Blaizzy), own venv; the module path is `mlx_video.models.wan_2.{convert,generate}` (the README's
`mlx_video.wan2` is out of date). Weights: TI2V-5B 32 GB (23 GB MLX), I2V-A14B 118 GB (64 GB MLX). Conversion: 42 s for
5B.

| model | input | frames | time | result |
|---|---|---|---|---|
| TI2V-5B | rest cow, "walks slowly forward to the right, locked-off side view" | 49 @ 1280×704 | 11 min | identity and camera stable, but almost no walking: legs shift, one hoof lifts |
| I2V-A14B, 40 steps | same | 49 @ 1280×704 | 3 h | a real walk: legs alternate, front leg lifts and steps; identity and side camera hold |
| I2V-A14B + Lightning, 4 steps | same | 49 @ 1280×704 | 10 min | the same quality of walk |

Use I2V-A14B with the Apache `lightx2v/Wan2.2-Lightning` I2V 4-step LoRA (`Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1`,
`--steps 4 --guide-scale 1 --lora-high … 1 --lora-low … 1`). The 5B model is not worth keeping for motion.

## Planner

`clef-flash` through Ollama `/v1/systemone`: psx-cow → quadruped 0.98, psx-farmer_hat → humanoid 0.95, about 0.9 s
warm (5 s cold).

## What this means for the plan

- Rigging, clip playback, rest-pose edits, video and the planner all run locally at usable speed: about 1.5 min to
  skin, 4-8 min per rest edit, 10 min per 2 s clip, under a second per decision.
- M3 needs three pieces the plan underestimated: the clip retarget (world-delta, above), binding against a template
  posed like the mesh, and a weight-locality filter.
- Video jobs and image edits each need most of the GPU; the job queue must run one heavy model at a time.
