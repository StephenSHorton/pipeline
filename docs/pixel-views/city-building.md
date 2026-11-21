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
