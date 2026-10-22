# Game look: low-res pixel rendering for 3D assets

The target is the look of the reference RTS clips: real 3D meshes drawn at a low internal resolution, blown up with hard pixel edges, an isometric camera, soft cast shadows and a muted, matte palette. At that resolution a generated mesh doesn't need to be clean. Lumpy topology, noisy normals and soft textures all collapse into a few pixels, so the pipeline can stay fast and loose.

## Preview it

Mesh view → **Pixel** in the inspector (or `viewer/?asset=<stem>&game=1`). It turns on for both panes, mesh and sprite stack, so the two can be judged side by side in the same look. The choice is remembered per browser.

| control | what it does |
|---|---|
| Pixel | Render both panes into a small target (screen size ÷ Px) and upscale it. While it's on, it replaces the stack pane's own Pixel and Outline. |
| Px | Screen pixels per game pixel. 3 to 5 matches the reference. |
| Iso | Orthographic camera locked to the isometric pitch (35.26° above the ground). Yaw still orbits. The camera snaps to whole texels so edges don't crawl. |
| Outline | Dark line where depth jumps (silhouettes), light line on creases (normal changes). Computed per game pixel. |
| Shadow | Soft shadow map from the key light onto a lighter ground. Replaces the blob. |
| Colors | Posterize to N steps per channel, after tone mapping. Off by default. |
| Dither | 4×4 ordered dither between those steps, on the game-pixel grid. |

Code: `viewer/pixel-look.js` (the desktop app's port is `desktop/frontend/src/screens/asset/stack/gameLook.ts`; its test reads this file's shaders). `viewer/stack-lab.js` makes its own instance for the stack pane. It only changes how a frame is drawn. OrbitControls and the perspective camera still drive the view, and raycasts (carve, paint, stack paint) use the camera the screen shows.

## Where sprite stacks fit

They're two different kinds of quantization, and they stack:

- **The sprite stack quantizes the geometry.** Slicing the mesh into a voxel volume (120 texels by default) throws away the mesh's own surface: bad topology, thin bits and noisy normals become even cubes. That's the strongest fidelity filter the pipeline has, and the README already treats the stack as the shipped asset.
- **The pixel pass quantizes the screen.** It works on anything: the raw Hunyuan mesh, the 12k game LOD, or the stack's cubes. The game look runs on both panes, so Voxel, Round and Plane can be compared in it directly.

What the reference clips show is closer to the second one: smooth-shaded vehicles and buildings, not voxel cubes. So the likely route is to ship the **game or proxy LOD mesh** and let the pixel pass do the hiding, and use the **stack** for small units or props where chunkier volume reads better, or when the engine is 2D. In the game look, the steps between a stack's slices are smaller than one game pixel, so even Plane (flat sprite slices, the way a 2D sprite-stacking engine draws them) reads as a solid object. That may make Round's smoothing unnecessary at game size.

## Porting it to the game

The pieces are the same in any engine. Order matters:

1. **Render the 3D world to a low-res target.** Size = window ÷ pixel size, integer. No MSAA, nearest sampling. Keep the UI at full resolution, on top.
2. **Orthographic camera at a fixed pitch.** Isometric is `atan(1/√2)` = 35.26° down. Yaw can rotate.
3. **Snap the camera to the texel grid.** One texel in world units is `ortho_height / target_height`. Project the camera position onto its own right and up axes, round each to a whole texel, and render from the rounded position. Keep the remainder `(actual − snapped) / texel` (in texels, between −0.5 and 0.5).
4. **Render one texel bigger on every side** and, when upscaling, sample with `uv = (screen_uv * low_res + 1 + remainder) / (low_res + 2)`. That puts the sub-texel motion back, so pans are smooth while the pixels stay locked to the world (no shimmer).
5. **Outlines** (optional), in the upscale or as a post pass on the low-res target, sampling the 4 neighbors of each texel:
   - depth edge: neighbor is farther by more than about 3 texels' worth of world distance, divided by how much the surface faces the camera (`clamp(abs(normal.z), 0.04, 1)`) so ground seen at an angle doesn't fire. Darken ×0.55.
   - normal edge: `(1 − dot(n, n_neighbor))` above 0.25, only on the side where `dot(n − n_neighbor, (1,1,1)) > 0` and the neighbor isn't behind. Brighten ×1.3.
6. **Upscale filter.** Plain nearest works. The pass here uses the "pixel-art AA" trick (smoothstep across the texel edge using `fwidth`), which keeps edges crisp at non-integer scales and while the remainder shifts.
7. **Lighting.** One directional key light with soft shadows (PCF), a hemisphere or ambient fill, low specular and low environment reflection. Matte reads better at this size.
8. **Palette** (optional). Posterize after tone mapping in display space, or look up a palette LUT. Dither on the low-res grid, never the screen grid.

### Godot 4

- Put the 3D world in a `SubViewport` sized to window ÷ pixel size, with MSAA off and `canvas_item_default_texture_filter` nearest. Show it through a `SubViewportContainer` with `stretch` off, or a `TextureRect` with a `ShaderMaterial` that does steps 4 to 6 (`hint_depth_texture` and `hint_normal_roughness_texture` are available in a spatial post-process quad inside the SubViewport, so do the outlines there and keep the upscale shader simple).
- `Camera3D` projection orthogonal, rotation `(-35.26°, yaw, 0)`, snap in `_process` before rendering, and pass the remainder to the upscale shader as a uniform.
- `DirectionalLight3D` with shadows on and `shadow_blur` around 1 to 2.

### Unity (URP)

- Camera renders into a `RenderTexture` at the low size, point filter, no MSAA. A full-screen Raw Image or a final blit pass does the upscale with the remainder offset.
- Orthographic camera, snapping in `LateUpdate`. Outlines as a Renderer Feature reading `_CameraDepthTexture` and `_CameraNormalsTexture` (turn on the Depth Normals prepass).

### Anything else

Two passes (color + depth, and a normal pass or G-buffer normals) into a small target, then one full-screen shader. `viewer/pixel-look.js` is under 400 lines of three.js and GLSL and can be read as the reference.
