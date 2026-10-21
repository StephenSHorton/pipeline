# Projects: one game world, shared parts

Today every asset is alone in `out/<stem>/`. A **project** groups the assets that belong to the same game world and keeps a **part library** they draw from, so a turret, a hand or a bolt pattern made once shows up on every mech in that world. The pieces that make assets look related are the shared style (prompt, palette, stack look) and the shared parts.

This builds on [one pipeline](unified-pipeline.md): stages and takes stay per asset. A project sits beside them and only points into them.

## Data

```
out/.projects/<slug>/
  project.json            name, style, member stems, part index
  parts/<part-id>/
    part.png              2D cut (RGBA), from a flat sheet
    part.glb              3D piece, from a mesh
    part.json             name, tags, role, from {stem, stage, take, part}
```

```json
{
  "version": 1, "id": "rust-belt", "name": "Rust Belt",
  "style": { "prompt": "worn steel, chunky silhouettes", "palette": ["#ff8800", "#223344"] },
  "assets": ["mech-a", "radio"],
  "parts": [{ "id": "left-arm", "name": "Left Arm", "kind": "2d", "tags": ["arm"],
              "from": { "stem": "mech-a", "stage": "flat", "part": "part-01", "take": "t2" } }]
}
```

- The dot keeps the folder out of the stem list, and `out/` is already gitignored.
- An asset can be in more than one project. Leaving a project does not delete the parts cut from it.
- `from.take` is the flow pick the part was cut from, so the library can show a part as stale when its source asset moves on (the same rule `flow.py` uses between stages).
- When an asset **uses** a library part, its own take records `uses: [{project, part}]`, so changing the library part marks that asset stale too.

`scripts/projects.py` has this much now, with tests: create, style, membership, copying a Split crop into the library, and `styled_prompt`.

## Identifying parts

This is the open problem. The plan is to get usable parts from tools we already run, then add a learned 3D segmenter once there's something to compare against.

### 2D

1. **Split** already cuts parts: SAM 2 automatic masks, or the magenta key. Swap to SAM 2.1 (one config line, per the model audit). Naming stays manual, with Draft's LLM suggesting names once `FLAT2D_LLM_URL` is set.
2. **Matching across assets**: embed each crop with DINOv2 (already downloaded for Paint 2.1) and store the vector in `part.json`. When an asset joins a project, or after Split, offer "this looks like *Left Arm* from mech-a" for crops above a similarity cut. That's how the library grows without hand-filing every piece.

### 3D

1. **Lift 2D masks onto the mesh** (first, no new weights). Render the mesh from the views `retexture.py` already fits cameras for, run SAM on each render, and let each face vote for a label across views. Faces that agree become a submesh. `articulate.py`'s heuristic splits get replaced by this. This is the SAMPart3D / SAMesh idea and works on any mesh we make.
2. **Learned part segmentation** (try next): PartField (NVIDIA) and P3-SAM from Tencent's Hunyuan3D-Part segment meshes directly into parts. Both need checking on licence and on whether they run on MPS or only CUDA; if CUDA only, it's a 5080 job like 2mv.
3. **Part-aware generation** (watch): X-Part (also Hunyuan3D-Part) generates a mesh as separate parts from boxes. If it holds up, parts come out of generation already separated and the segmentation step goes away.

A 3D part saved to the library is the submesh as `part.glb`, with its pivot from the segment's bounds, plus a render of it as `part.png` so 2D matching can find it too.

## Using the project when generating

Cheapest first:

1. **Style line**: every still made inside a project gets the project's style prompt and palette ahead of the user's prompt (`projects.styled_prompt`). Ideas and 2D Generate both go through `still_prompt`, so this is one call site.
2. **Palette lock**: the stack's palette step and Paint's albedo get pulled to the project palette.
3. **2D kit-bash**: a flat sheet can take a library part as one of its sprites directly. The part *is* the sprite, so assets share it exactly. For a new still, the library part is placed on the magenta canvas first and the image model fills in around it (Qwen-Image edit takes reference images).
4. **3D kit-bash**: drop the segmented region from the new mesh, place the library `part.glb` at that region's bounds, align (ICP on the overlap), merge, then repaint the seam with the existing retexture bake.
5. **Conditioned generation** (later): feed library part renders as extra reference views or IP-adapter style references so new assets come out related without manual placement.

## App

- Home gets a project picker. Assets made while a project is open join it.
- The gallery can filter by project.
- On the 2D Pieces step and (later) the 3D parts view, each part gets **Add to project**. The project page shows the library as a grid with the asset each part came from and uses.
- Routes, once the `/flow` routes in PR #14 land so they don't fight over `factory_http.py`: `GET /projects`, `POST /projects`, `GET /projects/<id>`, `POST /projects/<id>/assets`, `POST /projects/<id>/parts`.

## Order of work

1. **Data and library** (this PR): `scripts/projects.py` and tests. No routes, no UI.
2. **Routes and the project picker**, after PR #14 merges. Style line wired into `still_prompt`.
3. **Add to project** on 2D Pieces, and the library page.
4. **2D matching** with DINOv2 embeddings.
5. **3D parts by lifting SAM masks**, then a side-by-side with PartField / P3-SAM.
6. **Kit-bash** in 2D, then 3D.
