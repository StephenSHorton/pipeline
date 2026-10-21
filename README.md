# ai-asset-pipeline

Local image-to-3D on Apple Silicon (MLX). No Meshy. See **[PLAN.md](PLAN.md)**.

Turnaround (better pictures → better geometry): drop named views in `out/<stem>/views/` (`front`, `threequarter`, `left`, `back`, `top`). Shape from front (people) or threequarter (props). Extra views only change **geometry** if a multi-view shape DiT is present (`scripts/shape_mv.sh`); otherwise they still photo-bake.

```bash
scripts/generate.sh --image out/operator-turn/views/front.jpg --quality pbr --stem operator-turn
```

## Passes

1. **isolate** — Vision foreground mask.
2. **blockout** — Hunyuan3D-2 mini shape (MLX-Swift). If `views/lifted/{front,left,back}.png` exist, `shape_mv.sh` tries Hunyuan3D-DiT-v2-mv; otherwise mini stays single-image.
3. **reduce** — named LODs: lookdev 80k, game 12k, proxy 2k (or `--poly` at generate time)
4. **remesh** — AutoRemesher quads (`mesh_quad.glb`) before paint so xatlas isn't packing Hunyuan triangle soup
5. **surround** — extra views of the same object
6. **paint** — Hunyuan RGB (`--quality textured`) or Paint 2.1 PBR albedo+MR (`--quality pbr`) on the remeshed mesh
7. **retexture** — fit a camera to each reference photo (silhouette IoU), bake into UVs; occluded texels keep Hunyuan paint
8. **move** — people: `m2m_keep.py` retargets Mesh2Motion clips onto AniGen skins **without** rewriting weights. `m2m_bind.py` is the closest-bone rebind. objects: `articulate.py` piece pivots
9. **stack** — horizontal slices of the painted mesh → `out/<stem>/stack/{sheet.png,preview.png,stack.json}`
10. **show** — `viewer/` (`?asset=radio-turn&lod=pbr`): generated mesh left, sprite stack right.

```bash
scripts/bootstrap.sh draft
scripts/generate.sh --image in/radio-concept.jpg --quality draft --stem radio
python3 -m http.server 8765 --bind 127.0.0.1
# http://127.0.0.1:8765/viewer/?asset=radio
```

Qualities: `draft` (mesh only), `textured` (RGB atlas), `pbr` (albedo + metallic-roughness), `stack` (textured then sprite-stack — the game-ready product), `psx` (PS1 low-poly: ~1,200 flat-shaded triangles wearing the photo itself in a 256 px atlas, `out/<stem>/psx/` + `psx.glb`; see **[examples/mlg-psx.md](examples/mlg-psx.md)**). Every generate also writes `out/<stem>/stack/` unless you pass `--no-stack`.

Sprite stack is how we ship. Hunyuan meshes stay in the folder as the sculpt; the stack is the lowered-fidelity volume you actually put in a game. Inspector: `viewer/?asset=<stem>` (mesh left, stack right). Default display is 120 texels / 1.20 contrast+saturation; the sheet is 160². Paint in the inspector is a local brush (not Hunyuan): it writes `stack/sheet.png` and `textured_paint.glb`.

Poly: `--poly hero` (octree 256), `lookdev` (octree 128 → 80k), `game` (octree 64 → 12k), `proxy` (octree 64 → 2k). Or `scripts/reduce.sh <stem>` after the fact.

```bash
scripts/m2m_keep.py out/operator/anigen.glb -o out/operator/anigen_keep.glb
scripts/m2m_bind.py out/operator/textured.glb -o out/operator/m2m.glb
```

Grok workflow: `/ai-asset-pipeline` with `args.image` and `args.quality`, run from the repository root (`args.factory_root` points it elsewhere).

HTTP (loopback, curl, no MCP): `.venv/bin/python scripts/factory_http.py` then the copy-paste commands in **[examples/curl.md](examples/curl.md)** (`/health`, `/inspect/radio`, `POST /generate`, `/glb/radio`).

## Layout

| path | what |
|---|---|
| `scripts/generate.sh` | image → GLB facade |
| `scripts/shape_mv.sh` | multi-view Hunyuan-2mv when a vendor backend is present |
| `bin/hy3d` | Hunyuan3D CLI (MLX-Swift, gitignored build) |
| `weights/` | checkpoints, gitignored, `bootstrap.sh` downloads |
| `in/` | source images |
| `out/<stem>/` | per-asset products, gitignored |
| `viewer/` | local Three.js inspector |
| `viewer/` | inspector: mesh left, sprite stack right |
| `scripts/sprite_stack.py` | mesh → `out/<stem>/stack/` |
| `PLAN.md` | what works and what is next |

## One flow

Home asks **What are you making?**: a PS1 model, a 3D model, a sprite stack or a 2D sprite. Each shows only its own settings and a line saying what Make will do; the choice is saved with the asset (`out/<stem>/intent.json`), and the stages, canvas and Export screen follow it. The loading screen lists the factory's real stages (`STAGES`/`STAGE` lines from `generate.sh`, parsed by `/status`), with Cancel and Run in background. The desktop app runs every asset through **Ideas → Concept → 2D / 3D / Stack → Export**. A prompt with no picture draws a grid of ideas; **Use** one and it becomes the concept still that 2D, 3D and the stack start from. Each stage keeps takes in `out/<stem>/takes/` (listed in `flow.json`), so you can step back to an earlier still, sheet, mesh or stack from the bar above the tools. **Export** copies the picked 2D pieces and motion strips, sprite stack and 3D files into `out/<stem>/export/` with an `asset.json` that says what each file is. See **[docs/unified-pipeline.md](docs/unified-pipeline.md)**.

## Rigging

The **Rig** stage (after 3D) gives a model a skeleton and animation clips: pick a body plan (or let the local planner
pick from the picture), press **Rig it**, and play the clips on the model. `scripts/rig.py` does the work:

- Mesh2Motion's CC0 template for the body plan (person, four legs, many legs, bird), posed like generated models
  stand and placed on the mesh (`vendor/mesh2motion-app`).
- skin-tokens.cpp on Metal learns the skin weights onto that skeleton (`vendor/skin-tokens.cpp`, MIT weights).
- The template's library clips are retargeted onto the rig and baked into `out/<stem>/rigged.glb`.
- Blender writes `out/<stem>/rig/<name>.fbx` with every clip for s&box (FBX, at most 4 weights per vertex).

`POST /rig/<stem>` runs it as a job. Busts, vehicles and props aren't rigged yet. Setup and measurements:
**[docs/spikes-rigging.md](docs/spikes-rigging.md)**.

## 2D lane

Separate from the mesh factory. Open `viewer/flat/index.html` (link **2D** in the viewer bar). The factory on port 8766 has to be running.

Drop a picture on a flat background, then Split. That keys the corner color (or magenta, after Generate) and writes `out/<stem>/flat/parts/`. Name each crop and write how it moves or what it fires. Save writes `mechanics.json`. Reopening the page reloads that sheet. Touching limbs stay one part.

Generate uses Qwen-Image 2.1 Turbo (`Qwen/Qwen-Image-2.1-Turbo`) through mflux 0.20 or later, sampled on the 8 fixed steps the checkpoint ships in `model_index.json`. The weights (about 33 GB) download on the first run. The transformer is quantized to 8 bits but the text encoder stays bf16, so plan on about 30 GB of unified memory. `FLAT2D_STILL=z-image` switches back to Z-Image Turbo, which is lighter; an older mflux without Qwen-Image 2.1 also falls back to it. `FLAT2D_QWEN` overrides the repo id. Qwen-Image 2.1 weights are under the Qwen Research License: research and evaluation only, commercial use needs a separate license from Qwen. A generated still is prompted onto magenta `#FF00FF`, subject first, and `key.json` records the magenta it was actually drawn on (sampled from the border). Split cuts a generated still with Apple Vision (`bin/lift-subject`), keeps the key only for big pieces Vision missed, and turns ground seen through clear glass back into transparency. Make one object per picture: the prompt asks for a single object with no shadow, floor, glow or other effects, because those are what break the magenta pull. A still whose ground comes back cloudy or holds a shadow is redrawn on a new seed, up to twice, keeping the flattest; Qwen-Image does this on roughly 4 seeds in 10. Pieces that touch stay one piece. A still with one object is saved as that object's cutout. Names typed for Split go to pieces left to right, then row by row. Draft stays off until `FLAT2D_LLM_URL` is set. Sound writes a stand-in wav until Stable Audio 3 Small SFX weights are present and `FLAT2D_SFX_CMD` is the runner (`prompt`, then output path). If the weights are there and the command is not, Sound stays off. Speak reads the part's line with Kokoro (`mlx-community/Kokoro-82M-bf16`, American English, voice `af_heart` unless the sheet picks another) through `mlx-audio` and writes `out/<stem>/flat/voices/<id>.wav`. Install it in the factory venv with `uv pip install --python .venv/bin/python -r requirements-voice.txt` (`mlx-audio` and `misaki[en]`, which Kokoro uses to turn text into phonemes). `FLAT2D_KOKORO` overrides the repo id. A SAM 2 miss is reported; it does not silently fall back to the color key. No API key is read.

Weights live in `weights/flat2d/` (`sam2`, `stable-audio-3-small-sfx`), gitignored. Override a folder with `FLAT2D_SAM_DIR`, `FLAT2D_SFX_DIR`, or the root with `FLAT2D_WEIGHTS`.

## What this is not

Apple has no generative 3D SDK. Object Capture (`PhotogrammetrySession`) is a separate reconstruction lane for real objects with many photos. This pipeline is the hallucinated-from-one-image (or a short turnaround) lane.

## License

MIT for this repository's own code and docs; see [LICENSE](LICENSE). The models it downloads keep their own licenses, listed in [NOTICE](NOTICE). In particular, the Hunyuan3D weights are under the Tencent Hunyuan Community License, which is not open source and does not cover the EU, the UK or South Korea.
