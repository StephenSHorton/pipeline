# One pipeline

Today Pipeline has two lanes that meet only at the gallery. Home asks **3D or 2D** first. 2D makes one still (Z-Image on magenta, or a dropped picture) and then Pieces, Motion, Say, Sound. 3D needs reference images up front, runs `generate.sh`, and the sprite stack falls out of the mesh. A still can cross into 3D through the gallery's **Use** button, but nothing remembers that it did. Every run overwrites `out/<stem>/`, so there is no going back.

The goal is one flow per asset: throw out ideas as pictures, keep the ones you like, push a favorite into 2D, a sprite stack, or 3D, edit at any step, and step back or forward without losing work.

## Stages

```
ideas ──► concept ──┬──► flat   (2D sheet: pieces, motion, say, sound)
                    ├──► mesh   (shape, LODs, paint, rig)
                    └──► stack  (sprite stack, from the mesh or straight from the concept)
                                         └──► export (game-ready bundle)
```

| stage | what you do there | made by (today) | files (the existing contract) |
|---|---|---|---|
| **ideas** | Prompt once, get a grid of stills. Keep favorites, reroll the rest. | `flat2d.optional.generate_still`, run N times with different seeds | new: `ideas/*.png`, `ideas/ideas.json` |
| **concept** | One picked still, edited: revise prompt, cut out, add turnaround views. | `/2d/drop`, `/2d/generate`, `isolate`, view generation | `input.*`, `lifted.png`, `views/`, `flat/source.png`, `flat/key.json` |
| **flat** | Split into pieces, name motion, voice and sound, pack sheets. | `flat2d` (`/2d/steps`, `/2d/split`, `/2d/save`, `/2d/pack`) | `flat/parts*`, `flat/sheets/`, `flat/sounds/`, `flat/voices/`, `flat/view.json` |
| **mesh** | Shape, LODs, cleanup, paint, rig. | `generate.sh` via `POST /generate`, `/clean`, `/paint/mesh` | `mesh*.glb`, `textured*.glb`, `pbr.glb`, `lods.json`, ... |
| **stack** | Slice, palette, paint, look. | `sprite_stack.py`, `/paint/stack` | `stack/` |
| **export** | Pick formats and write one folder a game can load. | new | new: `export/` |

Nothing in the out contract moves. Stages are a way of grouping the files that already exist.

## Takes: going back and forward

A **take** is a saved copy of one stage's files at `out/<stem>/takes/<stage>/<id>/`, listed in `out/<stem>/flow.json`. Each stage has a list of takes and one **pick**.

- **Save** (`POST /flow/<stem>/snapshot`) copies the stage's files on disk into a new take and picks it. Saving files a take already holds just picks that take.
- **Go back** (`POST /flow/<stem>/pick`) copies a take's files back into the contract paths. If what's on disk was never saved, it is saved as its own take first, so going back never loses anything.
- **Stale**: every take records which upstream takes were picked when it was saved. If you go back to an earlier concept, the mesh made from the later one shows as stale until you rebuild it or pick the mesh take that matches. Nothing downstream is deleted.
- **Forward** is just running the next stage on the current pick, or picking a later take again.

Because takes copy files in and out of the paths the generators already use, `generate.sh`, `sprite_stack.py`, `flat2d`, and both viewers keep working unchanged. Old assets with no `flow.json` show their files as **unsaved** and get a take the first time you save or step back.

Copies cost disk (a GLB set can be tens of MB per take). Hard links would be cheaper but the generators rewrite files in place, which would corrupt the saved take. Copy is the safe start; a "trim old takes" button can come later.

The first piece of this is in `scripts/flow.py` with tests in `tests/test_flow.py`.

## The app

- **Home** keeps the prompt box but drops the 3D/2D toggle. Generate makes ideas (a grid of 4 stills by default). Attaching a picture skips straight to concept.
- **Workspace** gets a stage rail across the top: `Ideas · Concept · 2D · 3D · Stack · Export`. Each stage shows its takes as a strip of thumbnails with the pick highlighted, stale stages get a badge, and a stage with no take yet is a button that runs it from the current pick.
- The existing step bar (`Picture / Motion / Pieces / Say / Sound`, `Look / Sprite stack / Paint / Audio`) and the cubbies stay as the tools inside each stage.
- The revise box ("Describe a change…") becomes stage-aware: in Concept it edits the still, in 2D it drives the picked step, in 3D it can re-run paint.
- After any run finishes, the app saves a take for that stage, so the history fills itself without a Save button. Save stays for manual edits like brush paint.

## Sprite stacks from pictures

A stack used to exist only after a mesh, which meant the full MLX shape pass just to get one. `scripts/stack_image.py` starts from the concept picture instead, so a game can be made from pictures alone:

1. **Outline puff** (done): the subject's outline (`lifted.png`, Apple Vision, or the key color) is filled with balls along its ridge, so a disc becomes a sphere and a blade stays as thin as it is wide. Each voxel column takes its pixel's color, mirrored front to back, on the stack's 160² grid. It runs in about a second with no model. When `torch` and `transformers` are in the factory venv, Depth Anything V2 Small's nearness thickens the parts nearer the camera (`STACK_DEPTH_MODEL` picks the repo, `STACK_DEPTH=off` skips it). `POST /stack/<stem>` runs it; the Stack stage's **Stack from picture** button calls that.
2. **Turnaround carve** (later): front/left/back/top views from `views/lifted/` carved as a visual hull, colored from the nearest view. Better for things you rotate.

Both write the existing stack contract, so the stack viewer, palette, and paint already work on them. The stack take records `from: {concept: ...}` instead of `{mesh: ...}`.

## Export

One folder per asset a game can load, `out/<stem>/export/`, written by `scripts/export_asset.py` (`POST /export/<stem>`, or the **Export** stage's **Export** button):

```
export/
  asset.json      what each file is, and which takes it was made from
  concept.png     the concept cut out on transparency
  2d/parts/       one PNG per piece, named by the sheet (sword.png)
  2d/sheets/      one strip per motion, frames left to right
  2d/sounds/, 2d/voices/
  stack/          sheet.png (layers bottom to top), stack.json, preview.png
  3d/             model.glb, game.glb, proxy.glb, pixel.glb, rigged.glb (whichever exist)
```

Only the picked files go in, and nothing is converted. Export is a stage like the others: each export is a take, and it shows as stale when the concept, 2D, 3D or stack pick moves. **Show in Finder** opens the folder. Engine writers (a Godot scene, a Unity prefab) can read `asset.json` later.

## Order of work

1. **Takes**: `flow.py`, the `/flow` routes, tests.
2. **Stage rail**: the viewer's 3D/2D toggle is now the rail (`Concept · 2D · 3D · Stack`) with a strip of takes for the open stage. The factory saves a take before and after each 2D write, paint save and generate. 3D and Stack offer **Build 3D** from the picked concept still, and **Rebuild 3D** when the mesh is stale. Home keeps its 3D/2D toggle until Ideas lands, since a prompt alone can't start 3D yet.
3. **Ideas**: a prompt on Home with no picture attached starts a new asset on the Ideas grid and draws four stills, each with its own seed. **Keep** marks the good ones, **Clear unkept** deletes the rest, **More ideas** draws four more, and **Use** copies one to `flat/source.png` and opens Concept. Routes: `GET /2d/<stem>/ideas`, `POST /2d/idea`, `/2d/keep`, `/2d/clear`, `/2d/use`. Ideas use the same local image model as 2D Generate (Qwen-Image 2.1 Turbo, or Z-Image with `FLAT2D_STILL=z-image`).
4. **Stack from a picture**: the outline-puff stack maker, the optional depth model, and the **Stack from picture** button on the Stack stage.
5. **Export** (done): the plain folder and the Export stage. An engine writer can follow.

Each step ships on its own and leaves the app working.
