# Plan

Repo: `ai-asset-pipeline` (was prototyped as `mac-3d-factory` on an M5 Max 128 GB). macOS on Apple Silicon only (MLX).

## What already works

| Stage | Tool | Notes |
|---|---|---|
| Isolate | `bin/lift-subject` (Vision) | |
| Shape | `bin/hy3d shape` = Hunyuan3D-DiT-v2-**mini** (0.6B, MLX-Swift) | **Single image.** Extra views do not change geometry |
| Reduce | `scripts/reduce.sh` quadric LODs | lookdev 80k / game 12k / proxy 2k |
| Remesh | AutoRemesher CLI `bin/autoremesher` | quads before paint; Hunyuan soup → ~5–8k quads in ~2 s |
| Paint RGB | `hy3d paint --model rgb` | Hunyuan Paint 2.0, 2048 atlas |
| Paint PBR | `hy3d paint --model pbr` | Paint 2.1 albedo + metallic-roughness, 4096 atlas. Radio-turn 116 s, operator-turn 145 s |
| Photo bake | `scripts/retexture.py` | silhouette-IoU camera fit, halo-free; extra views bake albedo only |
| People motion | AniGen-mac (MPS) + `scripts/m2m_keep.py` | keeps AniGen weights; Mesh2Motion clips retargeted from Idle_A rest |
| People motion (worse skins) | `scripts/m2m_bind.py` | closest-bone rebind onto 66-bone M2M human |
| Props | `scripts/articulate.py` | Recoil-style piece pivots |
| Viewer | `viewer/index.html` | LOD strip, PBR maps, clips, turnaround strip |

Facade: `scripts/generate.sh --image … --quality draft|textured|pbr|stack --poly hero|lookdev|game|proxy --stem NAME`

`--quality stack` runs textured paint then slices. Every quality also emits `out/<stem>/stack/` (160² sheet; viewer defaults to 120 texels). `--no-stack` skips it. The stack is the shippable product; the GLB is the sculpt.

`generate.sh` calls `scripts/shape_mv.sh` when `out/<stem>/views/lifted/{front,left,back}.png` exist. Without `vendor/hunyuan3d-mlx` or `vendor/hunyuan3d-2` that script **exits 2** and generate falls back to mini.

## Output contract (do not rename)

Keep these paths so the viewer and later game import keep working:

```
out/<stem>/
  input.*                 original
  lifted.png              subject cutout used for single-image shape/paint
  views/lifted/
    front.png left.png back.png   required for 2mv
    threequarter.png top.png      optional; bake only
  mesh_hero.glb           full-res shape
  mesh.glb                copy of hero (or reduced if --poly)
  mesh_lookdev.glb        ~80k
  mesh_quad.glb           AutoRemesher (or InstantMeshes / PartUV result)
  mesh_game.glb / mesh_proxy.glb
  textured.glb            RGB paint
  pbr.glb                 albedo + metallicRoughnessTexture
  textured_photo.glb      photo bake
  anigen.glb              skinned people
  anigen_keep.glb         keep-weights + M2M clips
  lods.json               viewer LOD list
  inspect.json
  stack/sheet.png         sprite-stack atlas (bottom → top)
  stack/preview.png       stacked composite
  stack/stack.json        layers, pitch, factory display defaults
```

CLI flags on `generate.sh` stay stable.

## Next

### 1. Character A-pose before shape (optional, people only)

If the input person is not already a front A-pose, run a canonicalizer **before** multi-view shape.

- CharacterGen 2D stage: https://github.com/zjp-shadow/CharacterGen (SIGGRAPH 2024). Hugging Face `zjpshadow/CharacterGen`.
- Goal: emit `views/lifted/{front,left,back}.png` that are actually the same identity in A-pose, not Imagine-orbit guesses.

We already generate A-pose sheets on the Mac with Imagine + Vision lift (`out/operator-turn/views/`). Use CharacterGen only if those sheets drift.

Skip if the user already dropped a proper turnaround.

### 2. Mesh2Motion official BYO-rig (only if keep-weights looks cheap)

`m2m_keep.py` is a CLI stand-in for Mesh2Motion’s experimental “upload your rigged mesh” web UI (Jan 2026). It maps bones by hierarchy and copies rest-relative rotations using Idle_A as the source rest (M2M bind is T-pose, AniGen is A-pose).

If the bones don’t map cleanly, drive Mesh2Motion’s `AnimationRetargetService` from a headless three.js / node script using `vendor/mesh2motion-app` (clone: https://github.com/Mesh2Motion/mesh2motion-app, CC0 clips).

Do not rewrite JOINTS_0 / WEIGHTS_0 / IBMs. That is the whole point of keep-weights.

## Viewer / pipeline wiring

After a mesh exists:

1. `scripts/reduce.sh <stem>` — LODs
2. `scripts/quad_remesh.sh` — quads
3. paint → `pbr.glb`
4. refresh `lods.json` (reduce.sh already lists `pbr`, `quad`, `keep`, `anigen`, `m2m`)
5. `python3 -m http.server 8765 --bind 127.0.0.1` from repo root
6. `viewer/?asset=<stem>&lod=pbr`

Viewer already shows PBR maps. Do not clamp `metalness` when `metalnessMap` / `roughnessMap` exist (`viewer/index.html`).

## What not to do

- Do not check in Hugging Face weights or `out/*.glb`.
- Do not rebind AniGen skins onto Mesh2Motion’s 66 bones unless keep-weights is impossible (that is `m2m_bind.py`, already known-worse).
- Hunyuan Community license: no distribution in EU/UK/South Korea.

## Pixel style

Leaning into the accidental low-poly instead of chasing photoreal. `scripts/stylize_pixel.py` → `out/<stem>/pixel.glb`: ~1400 faceted tris, 128px atlas, 16-color concept palette, glTF NEAREST sampler. Viewer lod `pixel` uses nearest + no ACES. It projects the concept still onto the faceted mesh.
