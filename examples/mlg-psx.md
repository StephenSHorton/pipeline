# MLG Farming: PSX photo assets

Test run for MLG Farming's PSX look experiment (`scenes/psx_test.scene` in the mlg-farming repo): low-poly models
wearing over-real photos. One command takes a photo to a PS1-budget model (about 1,200 flat-shaded triangles, a
256 px 15-bit atlas) with the real photo baked on. The game's PSX shader adds the vertex snap, affine warp, no-mip
point sampling and dither, so the model itself stays plain.

## Inputs (`in/mlg/`, public domain)

| file | source |
|---|---|
| `cow.jpg` | not in the repository: add your own side-on photo of a cow (the test used a 1900s lantern slide of a Holstein, cropped to the cow) |
| `farmer_hat.jpg` | FSA/OWI Kodachrome, Library of Congress 1a34434, film border trimmed |
| `farmer_cap.jpg` | FSA/OWI Kodachrome "Guide at Little Norway, Blue Mounds, Wis.", Library of Congress 1a55005 |

The cow is side-on, so the shape is a guess from one side; the farmers are head-and-shoulders portraits, so they
come out as busts (the game puts them on simple bodies).

## In the app

1. On Home, pick **PS1 model** under *What are you making?*
2. **Add a photo**: `in/mlg/cow.jpg`. A wide photo sets Facing to *Side, facing right*; check it.
3. Set **Height**: the *Animal 58* chip for the cow, *Bust 30* for the farmers. File name defaults to the photo's name.
4. The line above the button says what will happen. Press **Make PS1 model**.
5. The loading screen lists each stage as it runs; *Run in background* and *Cancel* are there. When it's done the
   cow opens full width. Press **PS1 240p** to see it the way the game draws it.
6. **Export** lists `cow.obj` + `cow.png` + `cow.mtl` with the s&box import steps. To tweak Facing or Height,
   use **Change settings…** on the 3D stage and press **Remake**.

## Run from the command line


```bash
scripts/generate.sh --image in/mlg/cow.jpg        --quality psx --stem mlg-cow        --facing right  --psx-height 58 --psx-name cow
scripts/generate.sh --image in/mlg/farmer_hat.jpg --quality psx --stem mlg-farmer_hat --facing camera --psx-height 30 --psx-name farmer_hat
scripts/generate.sh --image in/mlg/farmer_cap.jpg --quality psx --stem mlg-farmer_cap --facing camera --psx-height 30 --psx-name farmer_cap
for s in cow farmer_hat farmer_cap; do .venv/bin/python scripts/export_asset.py --stem mlg-$s; done
```

About 3 minutes each on an M-series Mac. What `--quality psx` does:

1. Hunyuan shape and RGB paint, as `--quality textured`.
2. `retexture.py` bakes the photo itself over the paint through a camera fitted to its silhouette
   (`textured_photo.glb`). Texels the photo didn't see keep Hunyuan's paint. `--facing left|right` means a side-on
   photo, so it is also baked, flipped, onto the far side: a cow is close enough to symmetric, and the real photo
   reads from both sides. The top and the two ends are still Hunyuan's.
3. `psx_lowpoly.py` decimates to `--psx-faces` (1200), unwraps again, re-bakes the photo bake onto the low mesh,
   crunches it to `--psx-atlas` (256) px at 15-bit colour, and writes:
   - `out/<stem>/psx/<name>.obj` + `.mtl` + `.png`: flat-shaded, Z up, the subject looking down +X, `--psx-height`
     units tall (s&box: inches), feet on 0. `--psx-up y` keeps glTF axes instead.
   - `out/<stem>/psx.glb`: the same model in glTF axes (Y up, +Z front, metres) with a nearest sampler.
   - `out/<stem>/psx/psx.json`: budget, axes, size.

`--facing` is where the subject looks in the photo: `camera` (portraits), `right`, `left`, `away`. Over HTTP,
`POST /generate` takes `"quality": "psx"` with `facing`, `psx_height`, `psx_faces`, `psx_atlas`, `psx_up` and
`psx_name`.

## Check it

`python3 -m http.server 8765 --bind 127.0.0.1`, then `http://127.0.0.1:8765/viewer/?asset=mlg-cow&lod=psx&game=1`.
The PSX LOD shows with nearest sampling and no tone mapping; Pixel at Px 3 to 4 is about the game's 240p.

## Into the game

`export_asset.py` writes `out/<stem>/export/`, and `asset.json` says what each file is. The PS1 model is its own
3D role:

```json
"psx": {
  "file": "3d/psx.glb", "obj": "3d/psx/cow.obj", "mtl": "3d/psx/cow.mtl", "texture": "3d/psx/cow.png",
  "axes": {"obj": "z-up, +x front", "glb": "y-up, +z front (glTF), metres"},
  "triangles": 1198, "atlas": 256, "height": 58.0, "size": [87.4, 31.2, 57.9], "sampling": "nearest, no mips"
}
```

`3d.model` is the full-detail photo bake (`textured_photo.glb`), not Hunyuan's paint.

s&box imports the `.obj` (right-click > Create Model). Give it a material on `shaders/psx.shader` with the png as
`TextureColor` (see mlg-farming's `materials/psx/*.vmat`).
