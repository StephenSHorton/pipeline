# A hand-made pixel art sprite stack

`pixel-mushroom/` is a 15-layer, 43×43 stack in the factory's format (`stack.json` + `sheet.png`, layers left to
right from the bottom), with `"maker": "views"`, so the app's Stack stage shows it in its pixel art look. It
was made by hand, with `make_pixel_mushroom.py`, to test that look before the four-view carver writes real ones. It has the
things the old look spoiled: one-pixel spots and a one-pixel stalk (Round melted them), a firefly that touches
the cap only at an edge (Main piece dropped it) and an 8-colour palette (the grade shifted it).

To open it in the app, copy the folder into the data folder's `out/`, under any name, as its `stack/`:

```bash
mkdir -p out/pixel-mushroom && cp -r examples/pixel-art-stack/pixel-mushroom out/pixel-mushroom/stack
```
