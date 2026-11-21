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

## The roof as its own picture (pipeline#16)

**Version 7 is picked**: Version 5's drawings re-cut with the finished code (the front wall under the roof cut off),
160 long x 156 wide x 212 layers, 24 colours. Versions 4 to 6 were drawn the same way (`POST /2d/draw`,
`views: "pixel"`, `style: "clean"`, `size: 160`, `stack: true`, the asset's own description):

| Version | What happened |
| --- | --- |
| 4 | first roof wording: the roof picture came back as the whole building in three-quarter view |
| 5 | roof clauses only: a roof plan (grey roof, parapet, AC unit, stair hut with water tank) |
| 6 | roof clauses only: a roof plan again, but a band of facade across its whole front edge, and the sheet's back view 15% too tall (the views disagree on width) |
| 7 | Version 5 re-cut (no new drawing) |

| Picture | What it shows |
| --- | --- |
| `city-building-roof-before-after.png` | Version 2 and Version 7 in the app's Stack stage, from the side and from above |
| `city-building-stack-roof-top-down.png` | the stack's top colour per column, Version 2 left, Version 7 right |
| `city-building-roof-drawing.png` | Version 5's roof picture, as drawn |
| `city-building-roof-drawing-first-try.png` | Version 4's roof picture: the first wording |

- **Roof picture:** a tall object (`pixelart.tall`: side and front taller than wide) whose sheet top view is not as
  long as the side view is wide (`top_length`, now a note, not a redraw flaw: a facade seen from above is as tall as
  the walls) gets a second picture, the roof only, straight down (`pose.pixel_roof_prompt`). It replaces the sheet's
  top view: cropped to its subject, less the front wall peeking out under it (`pixelart.roof_edge_rows`: bottom rows
  set in from the roof's sides), and fitted to the side view's length and the front view's width at the side's scale
  (`pixelart.roof_size`), then cleaned with the other three. Kept as `flat/views/roof.png`; `views.json` says so
  under `roof`. The front and back are then checked against the roof's size.
- **Its wording:** the first try handed the model the whole description; the shop front, awning and window rows won.
  The roof picture now hears only the description's clauses about the roof (`pose.roof_text`), framed as a tile of
  a top-down city map. 2 of 2 draws with it came back as plans.
- **Pavement strip:** `pixelart.base_rows` finds bottom rows of the side, front and back views that reach past the
  walls above them (the median wall edges a little higher up, so a cornice or awning does not count), at most 6% of
  the height, and paints them out in the ground colour before anything is measured. On Version 5 it cut 13-14 px
  under each view; the bottom layers are now 1-2 cells wider than the walls (the plinth), where Version 2's bottom 4
  layers filled the whole 160 x 149 footprint. On the race car sheets it cuts only the dark ground line under one
  side view (6 and 5 rows on one sheet, none on the other three).
- **Still not right:** the AC unit and stair hut stand up only where the side and front silhouettes put them, so
  they read mostly as paint on the roof, with a few thin streaks; the roof was drawn about 1.5 times as wide as long
  and squeezed to near square, so the units look narrow; the parapet is grey where the walls' cornice is cream; a
  facade band right across the roof's front (Version 6) is not cut.
