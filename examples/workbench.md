# The agent workbench (MCP)

Claude makes, measures and fixes assets through an MCP server, `scripts/workbench/mcp_server.py`, on
`http://127.0.0.1:8768/mcp`. It is its own process and a client of the factory on `:8766`:

- **CPU work runs in the workbench:** clay, measuring, mesh edits, the photo bake and the PS1 cut.
- **Model work goes to the factory as a job:** drawing views, and the shape and paint in `generate`. The
  factory keeps its one job slot, so the "one big model at a time" rule holds.

## Start it

The factory has to be running for the job tools: the desktop app starts it, or run
`.venv/bin/python scripts/factory_http.py`. Then start the workbench (leave the window open):

```sh
.venv/bin/python scripts/workbench/mcp_server.py            # --port 8768, --factory http://127.0.0.1:8766
```

Register it with Claude Code once:

```sh
claude mcp add --transport http workbench http://127.0.0.1:8768/mcp
```

Check it from a shell:

```sh
curl -sS http://127.0.0.1:8768/mcp -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
curl -sS http://127.0.0.1:8768/mcp -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"measure","arguments":{"glb":"out/mlg-sheep/psx.glb"}}}'
```

Protocol and limits:

- It only listens on loopback, and it refuses requests whose `Origin` is not a local page.
- It answers POSTs with JSON. There is no SSE stream, so GET returns 405.

## How writes work

The workbench only ever writes under `out/<stem>/workbench/`. Every write follows the same steps:

1. **Busy check:** it asks the factory `GET /status` and refuses if a running job owns that stem.
2. **Take:** afterwards it calls `POST /flow/<stem>/snapshot` (stage `mesh`). Each change becomes a take, and
   `pick_take` goes back to an earlier one. The exceptions are `clay_set` and `clay_edit`: they change only
   `clay.json`, and a take copies the whole mesh stage (tens of MB on a real asset). Their change rides along in
   the next `clay_mesh` take, and a pick saves unsaved files as a take before going back.
   The busy check, the read of whatever is being changed, and the write all happen under one lock per stem, and
   `pick_take` waits for that lock too. GLBs are written under a hidden temp name and then renamed into place, so a
   factory job never copies a half-written file.
3. **Factory down:** if the factory is not running, the write still goes ahead, because no job can be racing it.
   No take is kept, and the result says so.

`look` and `compare_reference` write their PNGs to a temp folder, because a render is not a change to the asset.
Each result gives the PNG's path and also attaches the image. A render's folder is removed by a later call once it is
ten minutes old.

Paths are relative to the repo root, and nothing outside it is read.

## Tools

| tool | args | what |
|---|---|---|
| `asset_info` | `stem` | The asset's files plus `shape.json`, `degraded.json`, the photo fit (`textured_photo.fit.json`), `psx/psx.json`, intent and scene. Also lists the takes in each stage. |
| `look` | `stem` or `glb`, `lod?`, `views?`, `size?` | Contact sheet of textured renders: side, xside, front, back, top, q. |
| `measure` | `stem` or `glb`, `lod?`, `auto_head?` | Bounds, tris, pieces, `head_offset_pct`, `head_yaw_deg`, `mirror_iou`, `circularity`. |
| `compare_reference` | `stem` or `glb`, `image`, `facing?` | Silhouette IoU against the photo, the regions that are missing or extra, and a reference/model/overlap PNG. |
| `clay_get` | `stem` | `workbench/clay.json` and the forms with their dims. |
| `clay_set` | `stem`, `blobs`, `replace?` | Writes blobs: same-named blobs are replaced and the rest are added. With `replace: true` the list becomes the whole clay. |
| `clay_edit` | `stem`, `name`, `patch` | Changes one blob. `form` merges unless its type changes. `name` renames the blob, and `{remove: true}` deletes it. |
| `clay_mesh` | `stem`, `grid?`, `height?`, `texture?` | Writes `workbench/clay.glb` with UVs and the blob colours. |
| `mesh_edit` | `stem`, `src`, `op`, `params?` | Writes `workbench/edit_N.glb`. Ops: `remove_floaters` (`keep`, `min_faces`) and `delete_sphere` (`center`, `radius`). |
| `bake_photo` | `stem`, `glb`, `views`, `mirror?`, `erode?`, `keep?`, `first_wins?`, `fill?`, `mirror_ends?` | Runs `retexture.py` and writes `workbench/bake_N.glb`. By default every texel gets the photo: with `mirror_ends` (default true) a front-on or back-on first view is also baked, flipped, onto the opposite end, and with `fill: "photo"` (default) `workbench/fill.py` paints a triplanar swatch of the photo's subject on whatever the bake still missed. `fill: "none"` leaves those texels as the base texture. You can state the facing of the first view only; later views are read from their file names. Every view must be a cutout with a transparent background, such as `lifted.png` or `views/lifted/*.png`: an opaque photo is refused, because its background would be baked onto the model. |
| `psx` | `stem`, `glb`, `height`, `facing`, `faces?`, `atlas?`, `name?` | Runs `psx_lowpoly.py` with its facing check off. Writes `workbench/psx/` and `workbench/psx.glb`. `height` is in .obj units (s&box inches). |
| `draw_views` | `stem`, `image`, `prompt`, `intent?` | Factory job (draw model): four turnaround views drawn with the photo as a reference. |
| `generate` | `image`, `stem?`, `mesh?`, `paint?`, `fill?`, `mirror_ends?`, `quality?`, `psx?`, ... | Factory job: runs `generate.sh`. With `mesh` it uses that GLB as the shape instead of hy3d's. With `paint: false` it unwraps and bakes the photo, so a clay asset loads no model. `fill` defaults to `"photo"` here (`generate.sh --fill photo`): the photo is mirrored onto the far end and a swatch of it fills the rest. |
| `job_status`, `cancel`, `pick_take` | `job_id?`, `job_id`, `stem` + `stage` + `take` | Poll or stop the factory's job, or pick a take. |

`lod` picks which model a `stem` means:
- Default: `psx.glb`, then `mesh.glb` and the rest.
- `clay`: `workbench/clay.glb`.
- Otherwise one of the factory's LODs: `mesh`, `hero`, `lookdev`, `game`, `proxy`, `mv`.

Not built yet:
- `mesh_edit`'s `straighten_head`, `symmetrize` and `nudge`. They come with `edit.py`, after the first asset has
  been critiqued.

## Walkthrough: the hay bale

The current bale (`out/mlg-hay_bale/psx.glb`) came from one photo, so Hunyuan guessed its depth and made a
rounded box. Through the workbench only:

1. **Measure the current bale:** `measure {"stem": "mlg-hay_bale"}` gives `circularity 0.71` from the side.
   A round bale should be above 0.9.
2. **Block it out as clay:** a hay-tan cylinder lying on its side, its round end facing the camera like the photo.
   ```json
   {"stem": "mlg-hay_bale_v2", "blobs": [{"name": "bale",
     "form": {"type": "Cylinder", "radius": 0.75, "height": 1.2},
     "rotation": [90, 0, 0], "color": "#c8a24f", "blend": 0}]}
   ```
3. **Mesh it:** `clay_mesh {"stem": "mlg-hay_bale_v2", "height": 1.5}`, then `measure` it with `lod: "clay"`.
   Circularity should now be about 0.99.
4. **Bake the photo on with the pipeline:** `generate` with
   `{"stem": "mlg-hay_bale_v2", "image": "in/mlg/hay_bale.jpg", "mesh": "out/mlg-hay_bale_v2/workbench/clay.glb",
   "paint": false, "quality": "textured"}`. No model loads. Poll `job_status` until it is done.
   - The workbench sends `fill: "photo"`: the photo of the round end is mirrored onto the other end and a swatch
     of its straw covers the rolled side. Without it (the first remake) everything the photo did not see stayed
     flat clay tan.
   - Use `quality: "textured"` plus the `psx` tool rather than `quality: "psx"`, because generate.sh's own PS1 step
     runs the facing check, and that check asks ollama.
5. **Make the PS1 model:** `psx` with
   `{"stem": "mlg-hay_bale_v2", "glb": "out/mlg-hay_bale_v2/textured_photo.glb", "height": 60, "facing": "camera"}`.
6. **Check it:** `look {"glb": "out/mlg-hay_bale_v2/workbench/psx.glb"}` and
   `compare_reference` against the photo.

Stop there for review.
