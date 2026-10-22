// The Stack stage's copy and dock state, word for word from the legacy #stack-pane (index.html 2167-2236) and
// stack-lab.js. Mirrors tests/test_stack_looks.py (voxel default, "Parked", Main's title, round 0.45, full orbit)
// and the stack half of tests/test_pixel_look.py (the step-bar Pixel's title).
import { describe, expect, it } from "vitest";
import {
  buildingStack,
  CUBBY_DEFAULT,
  CUBBY_LABELS,
  defaultsFor,
  defaultsFromSearch,
  GAME_TOGGLE,
  hexToRgb,
  LOOK_BUTTONS,
  lookLabel,
  openCubby,
  ORBIT,
  PAINT_COPY,
  PAINT_TOGGLE,
  PALETTE_COPY,
  PBR_SLIDERS,
  pbrReadout,
  phaseText,
  radiusReadout,
  readout,
  resetState,
  rgbToHex,
  sliderMax,
  SLIDERS,
  sliderValue,
  STACK_COPY,
  stateForMeta,
  statsParts,
  statsText,
  STEP_LINKS,
  TOGGLE_BUTTONS,
  TOGGLE_RESET,
  worldPaintRadius,
  type StackMeta,
} from "./stack";

const meta: StackMeta = { layers: 87, resolution: 160, pitch: 0.01, extents: [1.6, 0.87, 1.6] };

describe("copy", () => {
  it("empty states, stats failure and the blurbs match the legacy page", () => {
    expect(STACK_COPY).toEqual({
      pane: "Sprite stack",
      empty: "No sprite stack yet. Generate writes out/<stem>/stack/.",
      slicing: "Slicing…",
      generating: "Generating…",
      noStack: "No sprite stack",
      lookBlurb: "Voxel cubes are the asset. Round welds neighbors into one surface. Pixel is the screen-space look. Plane and Contour stay parked.",
      paletteHint: "Reduce to N distinct hues. Rare lights are kept. Then pick a swatch for metal / roughness / glow.",
      paintHint: "Color the mesh and the stack with the same 3D brush. Save writes files under out/.",
    });
    expect(phaseText("empty")).toBe("No sprite stack yet. Generate writes out/<stem>/stack/.");
    expect(phaseText("slicing")).toBe("Slicing…");
    expect(phaseText("generating")).toBe("Generating…");
    expect(phaseText("ready")).toBe("");
  });

  it("looks: Voxel is the asset, Plane and Contour are parked", () => {
    expect(LOOK_BUTTONS).toEqual([
      { look: "voxel", label: "Voxel", title: "Instanced cubes per texel. The factory asset." },
      { look: "plane", label: "Plane", title: "Parked. One textured quad per slice." },
      { look: "contour", label: "Contour", title: "Parked. Marching-squares slabs. Not the product path." },
    ]);
  });

  it("the toggles, Reset and Paint", () => {
    expect(TOGGLE_BUTTONS.map((t) => [t.label, t.title])).toEqual([
      ["Pixel", "Render the stack at a low internal resolution, then upscale with t3ssel8r's pixel-art AA. Does not change the cubes."],
      ["Main", "Keep only the largest connected voxel piece. Off keeps Hunyuan scraps."],
      ["Outline", "1-pixel screen-space outlines from the depth buffer, at the internal pixel grid."],
      ["Sync", "Keep this camera at the same angle and slow turn as the mesh."],
      ["Shade", "Darken lower slabs so height is easier to read."],
    ]);
    expect(TOGGLE_RESET).toEqual({ label: "Reset", title: "Restore factory stack look: 120 texels, spacing/height 1, contrast and saturation 1.20." });
    expect(PAINT_TOGGLE).toEqual({ label: "Paint", title: "Same paint mode as the left pane." });
    expect(GAME_TOGGLE).toEqual({
      label: "Pixel",
      title: "Game look: render both panes small with crisp pixels, iso camera and shadows. Settings are under View.",
    });
  });

  it("the cubbies: Sprite stack open by default, Spacing and Palette links, one open at a time", () => {
    expect(CUBBY_LABELS).toEqual({ "stack-look": "Sprite stack", "stack-sliders": "Stack", "stack-palette": "Palette", paint: "Paint" });
    expect(CUBBY_DEFAULT).toBe("stack-look");
    expect(STEP_LINKS).toEqual([
      { cubby: "stack-sliders", label: "Spacing" },
      { cubby: "stack-palette", label: "Palette" },
    ]);
    expect(openCubby("stack-look", "stack-palette")).toBe("stack-palette");
    expect(openCubby("stack-look", "stack-look")).toBeNull();
    expect(openCubby(null, "stack-sliders")).toBe("stack-sliders");
  });

  it("the sliders: labels, ranges, steps and titles", () => {
    expect(SLIDERS.map((s) => [s.label, s.min, s.max, s.step, s.title])).toEqual([
      ["Px", 1, 8, 1, "Internal pixel size. 1 is native. 3–4 is the t3ssel8r-style chunk."],
      [
        "Round",
        0,
        1,
        0.01,
        "Merge neighboring voxels into one surface. 0 is cubes. Low values are a tiny blend (squared curve). 1 is a smooth iso. Factory is 0.45.",
      ],
      ["Spacing", 0.5, 18, 0.05, "Gap between slabs. 1 is packed. Raise it to pull the loaf apart."],
      ["Height", 0.1, 4, 0.05, "Thickness of each slab. Raise to fill gaps between layers."],
      ["Layers", 4, 256, 1, "How many slabs from the bottom. Lower crops the top of the object."],
      ["Texels", 16, 256, 8, "XY resolution of each slab. Lower is chunkier pixels; higher is finer."],
      ["Bright", 0.3, 2.2, 0.05, "Overall brightness of the stack."],
      ["Contrast", 0.3, 2.2, 0.05, "Contrast. 1.20 is the factory look."],
      ["Saturate", 0, 2.2, 0.05, "Color intensity. 0 is grey, 1.20 is the factory look."],
    ]);
  });

  it("the initial readouts are 3, 0.45, 1.0, 1.00, 87, 120, 1.00, 1.20, 1.20", () => {
    const d = defaultsFromSearch("");
    expect(SLIDERS.map((s) => readout(s, d[s.key]))).toEqual(["3", "0.45", "1.0", "1.00", "87", "120", "1.00", "1.20", "1.20"]);
  });

  it("the palette and paint copy", () => {
    expect(PALETTE_COPY.label).toBe("Palette");
    expect(PALETTE_COPY.count).toEqual({ label: "Colors", min: 2, max: 32, step: 1, value: 8, title: "How many distinct colors to keep." });
    expect(PALETTE_COPY.reduce).toEqual({ label: "Reduce", title: "Keep unique hues (chroma-weighted), not random k-means." });
    expect(PALETTE_COPY.swatchTitle(0)).toBe("Swatch 1. Paint uses this color.");
    expect(PBR_SLIDERS.map((s) => [s.label, s.min, s.max, s.step, pbrReadout(s.value), s.title])).toEqual([
      ["Metal", 0, 1, 0.05, "0.00", "Metallic for the selected color."],
      ["Rough", 0, 1, 0.05, "0.70", "Roughness for the selected color. Low is glossy."],
      ["Glow", 0, 1, 0.05, "0.00", "Emissive glow for the selected color."],
    ]);
    expect(PAINT_COPY).toEqual({
      paint: { label: "Paint", title: "Click-drag to paint. Right-drag still orbits." },
      drop: { label: "Drop", title: "Next click samples color from the surface." },
      erase: { label: "Erase", title: "Brush restores the original colors." },
      undo: { label: "Undo", title: "Undo the last stroke." },
      save: { label: "Save", title: "Write painted stack sheet and textured_paint.glb." },
      color: { label: "Color", title: "Brush color", value: "#2ecc71" },
      radius: { label: "Brush", min: 0.01, max: 0.2, step: 0.005, value: 0.04, title: "Paint sphere size, as a fraction of the mesh." },
    });
    expect(radiusReadout(0.04, false)).toBe("0.04");
    expect(radiusReadout(0.04)).toBe("0.040");
  });
});

describe("defaults", () => {
  it("legacy DEFAULTS: voxel, pixel, outline, main, sync and shade on, round 0.45, Px 3", () => {
    expect(defaultsFromSearch("")).toEqual({
      spacing: 1,
      height: 1,
      texels: 120,
      layers: 87,
      brightness: 1,
      contrast: 1.2,
      saturation: 1.2,
      shade: true,
      auto: true,
      look: "voxel",
      pixel: true,
      pixelSize: 3,
      outline: true,
      round: 0.45,
      main: true,
    });
  });

  it("?look= ?pixel=0 ?px= ?outline=0 ?round= ?main=0 override, clamped", () => {
    const d = defaultsFromSearch("?look=contour&pixel=0&px=12&outline=0&round=2&main=0");
    expect([d.look, d.pixel, d.pixelSize, d.outline, d.round, d.main]).toEqual(["contour", false, 8, false, 1, false]);
    expect(defaultsFromSearch("?look=nope&px=0&round=").look).toBe("voxel");
    expect(defaultsFromSearch("?px=0").pixelSize).toBe(3);
    expect(defaultsFromSearch("?round=x").round).toBe(0);
    expect(defaultsFromSearch("?round=").round).toBe(0.45);
  });

  it("stack.json defaults apply per stack, never to look, round or main", () => {
    const base = defaultsFromSearch("");
    const d = defaultsFor(base, { ...meta, defaults: { texels: 96, contrast: 1.4, round: 0, main: false, look: "plane" } as StackMeta["defaults"] });
    expect([d.texels, d.contrast, d.round, d.main, d.look]).toEqual([96, 1.4, 0.45, true, "voxel"]);
    expect(defaultsFor(base, meta)).toEqual(base);
    expect(base.texels).toBe(120);
  });

  it("a load resets layers, texels (capped at the resolution), spacing, height and the grade; keeps the rest", () => {
    const base = defaultsFromSearch("");
    const was = { ...base, spacing: 4, height: 2, brightness: 2, round: 0.1, pixel: false, look: "plane" as const, shade: false };
    const s = stateForMeta(was, base, { ...meta, resolution: 64, layers: 40 });
    expect([s.layers, s.texels, s.spacing, s.height, s.brightness]).toEqual([40, 64, 1, 1, 1]);
    expect([s.round, s.pixel, s.look, s.shade]).toEqual([0.1, false, "plane", false]);
  });

  it("Reset restores every default and all layers, but keeps the look", () => {
    const base = defaultsFromSearch("");
    const was = { ...base, look: "contour" as const, auto: false, shade: false, round: 0, main: false, pixel: false, pixelSize: 6, layers: 10, texels: 32 };
    const s = resetState(was, base, meta);
    expect(s).toEqual({ ...base, look: "contour", layers: 87, texels: 120 });
  });
});

describe("the stats line", () => {
  it("bold look label, px, slabs and resolution, as legacy's innerHTML", () => {
    const s = defaultsFromSearch("");
    const parts = statsParts(s, meta);
    expect(statsText(parts)).toBe("Voxel · Pixel · Round 0.45 · Main · 3px · 87 slabs · up to 160²");
    expect(parts.filter((p) => p.b).map((p) => p.text)).toEqual(["Voxel · Pixel · Round 0.45 · Main", "87", "160²"]);
  });

  it("Round shows on voxel only and above 0.001; no px without Pixel", () => {
    const s = defaultsFromSearch("");
    expect(lookLabel({ ...s, look: "plane" })).toBe("Plane · Pixel · Main");
    expect(lookLabel({ ...s, look: "contour", pixel: false, main: false })).toBe("Contour");
    expect(lookLabel({ ...s, round: 0.001 })).toBe("Voxel · Pixel · Main");
    expect(statsText(statsParts({ ...s, pixel: false, round: 0, main: false }, meta))).toBe("Voxel · 87 slabs · up to 160²");
  });
});

describe("sliders", () => {
  it("Layers and Texels top out at the stack's own counts", () => {
    const layers = SLIDERS.find((s) => s.key === "layers")!;
    const texels = SLIDERS.find((s) => s.key === "texels")!;
    expect(sliderMax(layers, meta)).toBe(87);
    expect(sliderMax(texels, meta)).toBe(160);
    expect(sliderMax(layers, null)).toBe(256);
  });

  it("Round clamps to 0..1 and Px to 1..8", () => {
    expect(sliderValue("round", 1.5)).toBe(1);
    expect(sliderValue("pixelSize", 0)).toBe(3);
    expect(sliderValue("pixelSize", 9)).toBe(8);
    expect(sliderValue("spacing", 2.5)).toBe(2.5);
  });
});

describe("paint helpers", () => {
  it("hex round-trips, and the radius is a fraction of the diagonal (1 when there is none)", () => {
    expect(hexToRgb("#2ecc71")).toEqual([46, 204, 113]);
    expect(rgbToHex(46, 204, 113)).toBe("#2ecc71");
    expect(worldPaintRadius(2, 0.04)).toBe(0.08);
    expect(worldPaintRadius(0, 0.04)).toBe(0.04);
  });
});

describe("the camera and the build state", () => {
  it("full orbit: the camera can go under the stack (maxPolarAngle = Math.PI)", () => {
    expect(ORBIT.maxPolarAngle).toBe(Math.PI);
    expect(ORBIT.minPolarAngle).toBe(0);
    expect(ORBIT.autoRotateSpeed).toBe(0.7);
  });

  it("Generating… shows while a build runs on this asset, not a 2D draw, ideas, a rig or a render", () => {
    const j = (kind: string, state = "running", stem = "hen") => ({ kind, state, stem });
    expect(buildingStack(j(""), "hen")).toBe(true);
    expect(buildingStack(j("", "ok"), "hen")).toBe(false);
    expect(buildingStack(j("", "running", "cow"), "hen")).toBe(false);
    for (const k of ["flat2d", "ideas", "rig", "render"]) expect(buildingStack(j(k), "hen")).toBe(false);
    expect(buildingStack(null, "hen")).toBe(false);
  });
});
