# Plan

Repo: `pipeline` (GitHub `StephenSHorton/pipeline`; prototyped as `mac-3d-factory`, then `ai-asset-pipeline`). A local, no-cloud
asset factory for macOS on Apple Silicon only (MLX). Rewritten 2026-11-16 from the code as of 2026-11-09 (`mokei-ui`).

## What it is for

You give it a picture and/or words; it makes game assets, aimed at the MLG Farming PS1-look test game (s&box):

| You pick (Home: *What are you making?*) | You get |
|---|---|
| **PS1 model** | about 1,200 flat-shaded triangles wearing the photo itself in a 256 px atlas: `psx/` (`.obj` + `.png`) and `psx.glb` |
| **3D model** | a painted mesh with detail levels (Hunyuan3D shape; Hunyuan or TRELLIS.2 paint) |
| **Sprite stack** | horizontal slices of the model as a sheet; "how we ship" |
| **2D sprite** | a cut-out piece sheet with names, sound and voice lines |

## How one make runs

1. **Isolate** the subject (`bin/lift-subject`, Apple Vision).
2. **Draw four views** (front, left, back, three-quarter) of the subject. A photo is a reference, never a view: the
   views are drawn from it (Qwen-Image-Edit), or from words when there is no photo.
3. **Shape** from those views with Hunyuan3D-2mv (`scripts/shape_mv.sh`, MPS). When only the side views come out
   wrong, shape from front and back. With one picture only, the shape falls back to Hunyuan3D-2 mini and the
   result is marked degraded.
4. **Reduce and remesh** (`reduce_mesh.py`, AutoRemesher quads before paint).
5. **Paint**: Hunyuan Paint (RGB 2048, or PBR 4096), or **TRELLIS.2** (`--paint trellis`, MIT, in
   `vendor/trellis-mac`). TRELLIS.2's paint is plain colour, so `--flat-paint` fits the drawn views' eyes and snout on
   top.
6. **Photo bake** (`retexture.py`): fit a camera per view, bake per triangle; occluded texels keep the paint.
7. **PS1 cut** (`psx_lowpoly.py`): weld near-duplicates, cap holes, keep legs, ears and small pieces that touch the
   body, bake the photo into a 256 px atlas, publish `psx.glb`.
8. **Review** (`scripts/workbench/review.py`) before the model is shown: pieces, paint patchwork, mirror, triangle
   budget, ears and eyes on a close-up sheet. Each of Stephen's critiques became a check there.
9. **Rig** (animals and people): body plan (person, four legs, many legs, bird), skeleton from Mesh2Motion's CC0
   templates, skin weights learned by skin-tokens.cpp on Metal, library clips baked with a floor pass, FBX for s&box.
   `rig_review` and `rig_scrub` check it.
10. **Stack / export**: sprite stack from the mesh; Export copies the chosen files into `out/<stem>/export/` with an
    `asset.json`.

Every stage keeps numbered **takes** in `out/<stem>/takes/` (`flow.json`; see `docs/unified-pipeline.md`), and assets
group into **projects** (`docs/projects.md`).

## Two ways to drive it, one factory

- **The desktop app** (`desktop/`: Wails host, React + shadcn on Mokei UI in `desktop/frontend/`). The app flow:
  **Home** picks what you are making, **Make** runs it with the factory's real stages, **Library** lists assets by
  project, and each asset has **Ideas → Concept → 2D / 3D / Rig / Stack → Export**, with Versions and Compare
  (shift-click a version). The factory is `scripts/factory_http.py` on `:8766`.
- **The workbench** (`scripts/workbench/`, an MCP server on `:8768`, a client of the factory): Claude makes, measures,
  fixes and shows assets: look, measure, compare, mesh edits (close holes, carve parts, thicken legs, trim belly),
  review before showing, and `show`, which opens an asset in the app with a before and after. CPU work runs in the
  workbench; model work goes to the factory as a job (one big model at a time). **Read
  `scripts/workbench/playbook.md` first** (rule 1: make ONE model, wait for the critique, improve the pipeline, then
  make the next). Walkthrough: `examples/workbench.md`.

## Where it stands (2026-11-09)

- **Works:** the Make flow for all four kinds, photo-as-reference, TRELLIS.2 paint, the PS1 cut, the review step,
  rigging with clips and FBX, the workbench, projects, takes and Compare. 793 tests passed (12 skipped) before the
  last two commits.
- **Pig:** the one model that took most of the last two weeks of work (2026-10-24 to 11-03). It still fails its own
  review (a leg is a pinched separate piece; a belly patch of paint disagrees with the bake) and its rig check (head
  20° off its bone; its rig is from an older mesh).
- **Hard-surface props (mailbox, wheelbarrow, car) have no working route.** The code-built route (Claude writes a
  Three.js page, a headless browser exports it) was added 2026-11-03 and removed 2026-11-09 (PR #8) because it was
  useless in practice. The image-to-3D models lose to it on these props too (playbook, #351). "Car from parts"
  (`racecar-1127`: draw a body and a wheel, assemble one body + four named wheels) is an unmerged, never-run-end-to-end
  experiment. Vehicles and props cannot be rigged yet.
- **Not checked since the break:** whether every model-heavy path (draw, shape, paint, rig) still runs end to end.

## Next

Pick one; they are listed in the order that costs least.

1. **Hard-surface props:** run car from parts end to end on one car and judge it, or decide props are out of scope
   and say so in the README.
2. **Finish the pig** (leg piece, belly paint, head bone), then run its photo fresh through Make → PS1 → Rig → Export
   with timings, to find what broke while nobody was looking. Then the next MLG animal, under the playbook's rule 1.
3. **Tidy:** the branches and worktrees whose commits are already on `mokei-ui` or `main`; keep the old private
   `main` (it holds `in/mlg/cow.jpg`) as an archive and do not push it.

## Output contract (do not rename)

Keep these paths so the app, the viewer and the game import keep working:

```
out/<stem>/
  input.*                 original
  intent.json             what was being made (ps1 | 3d | stack | 2d)
  lifted.png              subject cutout
  views/lifted/{front,left,back}.png   drawn views (threequarter, top: bake only)
  flat/source.png         the concept still the views are drawn from
  mesh_hero.glb           full-res shape; mesh.glb is the working copy
  mesh_lookdev.glb        ~80k, the mesh paint works on
  mesh_quad.glb           AutoRemesher result
  textured.glb / pbr.glb  Hunyuan or TRELLIS.2 paint (RGB / albedo + metallic-roughness)
  textured_photo.glb      photo bake
  psx/ and psx.glb        PS1 cut: .obj + .mtl + .png, and the GLB
  rigged.glb, rig/*.fbx   rig and clips
  stack/{sheet,preview}.png, stack.json
  takes/<stage>/<id>/, flow.json        takes and the pick per stage
  workbench/              the workbench's writes (review/, clay.json, ...)
  export/                 Export's folder, with asset.json
  lods.json, inspect.json
```

`scripts/generate.sh` flags stay stable: `--image`, `--quality draft|textured|pbr|stack|psx`,
`--poly hero|lookdev|game|proxy`, `--paint on|off|trellis`, `--stem`, `--mesh <glb>`, `--fill`, `--mirror-ends`.

## What not to do

- Do not check in weights, `vendor/` clones or `out/`.
- Do not load a model in a script outside the factory's job slot: two big models co-loaded ran the 128 GB Mac out of
  memory (playbook).
- Do not make sibling stems (`foo_v2`) to try a fix: iterate inside the asset as takes.
- Do not push the old private `main` branch: it holds `in/mlg/cow.jpg`, which the public line leaves out on purpose.
- Licences: Hunyuan Community licence (no distribution in the EU, UK or South Korea); Qwen-Image 2.1 weights are
  research-only. See [NOTICE](NOTICE).

## Older routes still in the repo

From the September pipeline, kept but not the main path: `articulate.py` (prop piece pivots), AniGen-mac +
`m2m_keep.py` / `m2m_bind.py` (people skins and Mesh2Motion clips), and the `viewer/` Three.js inspector (LOD strip,
PBR maps, clips, turnaround strip; `python3 -m http.server 8765` then `viewer/?asset=<stem>`). The rig stage above is
the current route for animals and people.
