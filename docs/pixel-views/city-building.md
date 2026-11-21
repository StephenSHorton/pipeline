# City building: the first city test asset (pipeline#15)

`out/city_building` in the main checkout: a three storey red brick shop building, drawn by the Mac's model in the
clean game-art style (`POST /2d/draw`, `views: "pixel"`, `style: "clean"`, `size: 160`, `stack: true`) and carved
into a sprite stack. Three drawings, kept as Versions 1 to 3; **Version 2 is picked**: 160 long x 149 wide x 179
layers, 24 colours.

| Picture | What it shows |
| --- | --- |
| `city-building-drawn-sheet-160.png` | Version 2's drawn sheet (top, side / front, back) |
| `city-building-cleaned-views-160.png` | the four views cut at 160 cells |
| `city-building-stack-in-app.png` | the stack in the app's Stack stage |
| `city-building-drawn-sheet-v1.png`, `-v3.png` | the other two drawings |

## What a building does to the method

- **Upright:** the layer count comes from the side view's height, the length (160) from the side view's width, so a
  tall building gets more layers (179) than cells along its length. It stands upright.
- **The top view:** in all three drawings the model drew the "top" as the front facade seen from a little above,
  with the roof on top, not a roof plan. The carve paints the roof from it, so the roof carries facade colours
  (awning green, window rows). Asking for "only its upper surfaces visible ... none of its sides" in the shared
  prompt, and then spelling out a roof plan in the description, did not change it.
- **Pavement:** every drawing put a grey pavement strip under the side, front and back views, despite "no pavement,
  no base strip". It carves into a thin plate along the bottom layer and widens the measured footprint.
- **Overhangs:** the awning and recessed door carve solid; the awning reads only as colour on the wall.
- **Windows:** read well on the front, side and back walls.

## Method fixes made on this card

- `pixelart.clean(..., grid=False)` for the clean style: a picture with no pixel grid by design is always cut
  straight to the asked-for size. Version 1 came out 163 long because the brickwork looked like a 3 px grid.
- `pixelart.check_sheet`: for a tall object (side and front taller than wide) the long-object checks (top view
  tallest, side widest, side not upright, top not sideways) are skipped; they flagged a good building sheet and
  forced a redraw. A tall object is checked instead for `top_length`: the top view's length must match the side
  view's width. That catches a facade drawn as the top, unless a pavement strip widens the side view.
- Prompt: the top view asks for only upper surfaces; the clean style rules out pavement and base strips by name.

## Four wall views and a roof picture (pipeline#16)

**Version 9 is picked**: 160 long x 162 wide x 204 layers, 22 colours. Drawn on the Mac's model (`POST /2d/draw`,
`views: "pixel"`, `style: "clean"`, `size: 160`, `stack: true`, the asset's own description).

A building's sheet no longer asks for a view from above at all. Asked for one, the model always drew the front again
with the roof tilted into view (Versions 1 to 7 all have that picture top left). Now:

- **Wall sheet** (`pose.pixel_walls_prompt`, used when `pose.is_building`: the intent's subject or the description's
  first clause names a building): four straight-on elevations in the 2x2 grid, front, left side (front toward the
  left edge), back, right side (front toward the right edge), the camera level so the top of the roof is never
  visible. `pixelart.check_walls` splits it, paints out a strip drawn under each wall (`pixelart.base_rows`), and
  checks one subject per view and that the back agrees with the front and the two sides with each other. The second
  side is kept as `flat/views/side2.png`; the stack uses the front, left side and back.
- **Roof picture** (`pose.pixel_roof_prompt`): only the description's roof clauses, straight down, the parapet's
  outer edge the outline with no wall strip along any edge. It becomes the top view, fitted to the side's length
  and the front's width (`pixelart.roof_size`), less any front wall still peeking out under it
  (`pixelart.roof_edge_rows`). Kept as `flat/views/roof.png`.
- A tall object that is not a building keeps the old sheet, and gets the roof picture only when its top view is as
  long as its walls are tall (`top_length`).

| Version | What happened |
| --- | --- |
| 4 | first roof wording: the roof picture came back as the whole building at an angle |
| 5, 6, 7 | old sheet (top-left still the front seen from above) with a roof picture as the top view |
| 8 | wall sheet: four straight-on walls, no roof top; but the back drawn as a second shop front, and the right side facing the same way as the left |
| 9 | wall sheet with clearer back and right-side wording: the back is plain brick; the right side still faces the same way as the left |

| Picture | What it shows |
| --- | --- |
| `city-building-v9-sheet.jpg` | Version 9's sheet, as drawn |
| `city-building-v9-roof.jpg` | Version 9's roof picture, as drawn |
| `city-building-roof-before-after.png` | Version 2 and Version 9 in the app's Stack stage, from the side and from above |
| `city-building-stack-roof-top-down.png` | the stack's top colour per column, Version 2 left, Version 9 right |
| `city-building-drawings.jpg` | the drawings of Versions 2 and 4 to 7 (also on their own: `city-building-drawn-sheet-v5.png`, `city-building-roof-drawing.png` for Version 5, `-first-try.png` for Version 4, `-v6.png` for Version 6) |

**Still not what was asked, in Version 9's drawings:**
- The right side (bottom right) faces the same way as the left side; it is not used by the stack.
- The roof picture still has a thin band of front wall along its bottom edge, a small hatch on its back edge and a
  vent on its left edge. The band is cut off before fitting, not fixed in the drawing; the hatch and vent stay.
- The roof was drawn wider than long and is squeezed to the walls' footprint.

**Pavement strip:** every wall view still came with a grey pavement strip (12-17 px); it is painted out, and the
stack's bottom layer is 131 x 138 cells over walls of 129 x 135 (Version 2's was 149 x 160 over 102 x 110).
**Rooftop units** stand up only where the wall views show them, so on the roof they read partly as paint.
