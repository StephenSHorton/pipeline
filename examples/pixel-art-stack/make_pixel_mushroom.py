# A hand-made pixel art sprite stack in the stack.json + sheet.png format: 15 layers of 43x43, 8 colours.
import json, math, sys
from pathlib import Path
from PIL import Image
R, L = 43, 15
dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "pixel-mushroom"
C = R // 2
GRASS, GRASS2 = (72, 160, 64), (40, 100, 48)
STEM, STEM2 = (232, 216, 172), (190, 160, 110)
CAP, CAP2 = (204, 40, 52), (130, 22, 40)
SPOT, GLOW = (250, 248, 240), (255, 214, 64)
sheet = Image.new("RGBA", (R * L, R), (0, 0, 0, 0))
px = sheet.load()
def put(layer, x, z, c):
    px[layer * R + x, z] = c + (255,)
d = lambda x, z: math.hypot(x - C, z - C)
for z in range(R):
    for x in range(R):
        r = d(x, z)
        if r <= 18:
            put(0, x, z, GRASS2 if (x + z) % 2 == 0 and r > 7 else GRASS)
        for y in range(1, 7):
            if r <= 4.6:
                put(y, x, z, STEM2 if y == 3 or r > 3.8 else STEM)
        for y, rad in ((7, 14.5), (8, 15.5), (9, 15.5), (10, 13.5), (11, 9.5)):
            if r <= rad:
                c = CAP2 if (y == 7 and r > rad - 1.2) else CAP
                if y >= 9 and (x * 7 + z * 3) % 11 == 0 and r < rad - 1:
                    c = SPOT
                put(y, x, z, c)
# one-pixel stalk and its tip: what Round melts
for y in (12, 13):
    put(y, C, C, STEM2)
put(14, C, C, GLOW)
# a firefly that touches the cap only at an edge: what Main piece drops
put(12, C + 10, C, GLOW)
put(13, C + 11, C, GLOW)
# grass tufts, one pixel each
for x, z in ((C - 12, C - 3), (C + 8, C + 11), (C - 4, C + 14), (C + 13, C - 8)):
    put(1, x, z, GRASS)
dest.mkdir(parents=True, exist_ok=True)
sheet.save(dest / "sheet.png")
pitch = 0.02
meta = {
    "name": "pixel-mushroom", "stem": "pixel-mushroom", "sheet": "sheet.png", "resolution": R, "layers": L,
    "pitch": pitch, "origin": [-R * pitch / 2, 0, -R * pitch / 2], "extents": [R * pitch, L * pitch, R * pitch],
    "axis": {"right": "x", "up": "y", "forward": "z", "slice": "y"}, "sheet_layout": "horizontal-bottom-to-top",
    "defaults": {"texels": 120, "spacing": 1, "height": 1, "brightness": 1.0, "contrast": 1.2, "saturation": 1.2},
    "kind": "stack", "maker": "views",
}
(dest / "stack.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", dest)
