// The Stack stage's copy and pure state (legacy #stack-pane, index.html 2167-2236; stack-lab.js DEFAULTS,
// applyMetaToSliders, the Reset handler and the stats line, 5-27, 1165-1190, 1619-1702; the Paint cubby and
// the palette host code, index.html 2113-2124, 5384-5642). StackStage.tsx lays these out; view.ts draws.
// No React, no fetch here.

/** "slices": the sprite-stack look (pipeline#6), each layer a flat picture, pixel-sharp, outlined, lower layers darker. */
export type StackLook = "voxel" | "plane" | "contour" | "slices";
export const LOOKS: StackLook[] = ["voxel", "plane", "contour", "slices"];

/** Everything the dock drives (legacy stack-lab.js `state`). */
export interface StackState {
  spacing: number;
  height: number;
  texels: number;
  layers: number;
  brightness: number;
  contrast: number;
  saturation: number;
  shade: boolean;
  /** Sync: the slow turn. */
  auto: boolean;
  look: StackLook;
  pixel: boolean;
  pixelSize: number;
  outline: boolean;
  round: number;
  main: boolean;
  /** Pixel art: unlit, the sheet's exact colours, sharp textures (a stack drawn as pixel art, maker "views" or a building's "walls"). */
  flat: boolean;
  /** A pixel art stack lit by the scene: sun, sky and (in Game look) shadows on the sheet's colours. */
  lit: boolean;
}

/** stack.json, as the page reads it (layers, resolution, pitch, extents, defaults). */
export interface StackMeta {
  layers: number;
  resolution: number;
  pitch: number;
  extents: number[];
  defaults?: Partial<Record<keyof StackState, unknown>> | null;
  maker?: string;
  /** A walking character (pipeline#2): its character.json, next to stack.json. */
  character?: string;
  /** How this Version is first shown (stack_character.py walk --look): lit 3D cubes, or flat sprite-stack slices. */
  look?: "cubes" | "slices";
  /** A character's body: carved from the drawn views, or sliced from a 3D model of them. */
  body?: "carved" | "mesh";
}

/** The look a Version asks for in stack.json, if any. */
export const metaLook = (meta: Pick<StackMeta, "look">): StackLook | null =>
  meta.look === "slices" ? "slices" : meta.look === "cubes" ? "voxel" : null;

/** Legacy DEFAULTS with the URL overrides (?look= ?pixel=0 ?px= ?outline=0 ?round= ?main=0). */
export function defaultsFromSearch(search = ""): StackState {
  const urlParams = new URLSearchParams(search);
  const urlLook = urlParams.get("look") as StackLook | null;
  return {
    spacing: 1,
    height: 1,
    texels: 120,
    // The DOM default (index.html 2209); the dock is hidden until stack.json sets the real count.
    layers: 87,
    brightness: 1,
    contrast: 1.2,
    saturation: 1.2,
    shade: true,
    // Stephen: spin starts off (the 3D viewer too); the Sync toggle turns it on.
    auto: false,
    look: urlLook && LOOKS.includes(urlLook) ? urlLook : "voxel",
    pixel: urlParams.get("pixel") !== "0",
    pixelSize: Math.max(1, Math.min(8, Number(urlParams.get("px")) || 3)),
    outline: urlParams.get("outline") !== "0",
    round: (() => {
      const raw = urlParams.get("round");
      if (raw == null || raw === "") return 0.45;
      return Math.max(0, Math.min(1, Number(raw) || 0));
    })(),
    main: urlParams.get("main") !== "0",
    flat: false,
    lit: urlParams.get("lit") === "1",
  };
}

/** A stack drawn as pixel art (carved from four pixel art views): its sheet is the art, pixel for pixel. */
export const isPixelArt = (meta: Pick<StackMeta, "maker"> | null | undefined) =>
  meta?.maker === "views" || meta?.maker === "walls";

/**
 * How a pixel art stack is shown: every texel as one square cube in the sheet's exact colour. No Round (it melts
 * one-pixel features), no Main piece (it drops parts that touch only at a corner), no Shade, a neutral grade, no
 * screen-space pixel grid or outline, and every texel (Texels = the resolution).
 */
export function pixelArtLook(meta: Pick<StackMeta, "resolution">): Partial<StackState> {
  return {
    flat: true,
    round: 0,
    main: false,
    shade: false,
    pixel: false,
    outline: false,
    brightness: 1,
    contrast: 1,
    saturation: 1,
    texels: meta.resolution,
  };
}

/** The keys a pixel art stack sets on load; a model stack loaded after one gets the page's back. */
const PIXEL_ART_KEYS = ["flat", "round", "main", "shade", "pixel", "outline"] as const;

/** The keys stack.json's `defaults` may set (sprite_stack.py FACTORY_DEFAULTS). Look, round and main stay the page's. */
const META_KEYS = ["spacing", "height", "texels", "brightness", "contrast", "saturation"] as const;

/**
 * The factory defaults for one stack: the page's, then stack.json's `defaults`. Per load: legacy assigned them into
 * its module DEFAULTS, so one asset's defaults stuck to the next asset that had none.
 */
export function defaultsFor(base: StackState, meta: StackMeta): StackState {
  const out = { ...base };
  const d = meta.defaults && typeof meta.defaults === "object" ? meta.defaults : null;
  if (d) {
    for (const k of META_KEYS) {
      const v = Number(d[k]);
      if (d[k] != null && Number.isFinite(v)) out[k] = v;
    }
  }
  // The pixel art look wins over stack.json's grade and texels: the colours and the grid are the art.
  return isPixelArt(meta) ? { ...out, ...pixelArtLook(meta) } : out;
}

/** A stack loaded (legacy applyMetaToSliders): layers, texels, spacing, height and the grade go back to its defaults;
 * look, pixel, round, main, shade and sync stay as they were. */
export function stateForMeta(state: StackState, defaults: StackState, meta: StackMeta): StackState {
  // A pixel art stack sets its own look; the next model stack goes back to the page's (defaults is the page's then).
  const look: Partial<StackState> = {};
  if (isPixelArt(meta) || state.flat) for (const k of PIXEL_ART_KEYS) (look as Record<string, unknown>)[k] = defaults[k];
  // A Version that asks for a look gets it; one that does not, after a Version that asked for Sprite, goes back to
  // the page's look (as the pixel art keys do), so the Sprite look never sticks to every stack opened after it.
  const asked = metaLook(meta);
  if (asked) look.look = asked;
  else if (state.look === "slices") look.look = defaults.look;
  return {
    ...state,
    ...look,
    layers: meta.layers,
    texels: Math.min(defaults.texels, meta.resolution),
    spacing: defaults.spacing,
    height: defaults.height,
    brightness: defaults.brightness,
    contrast: defaults.contrast,
    saturation: defaults.saturation,
  };
}

/** Reset (legacy #stack-reset): every default, all layers, texels capped at the resolution. The look stays. */
export function resetState(state: StackState, defaults: StackState, meta: StackMeta): StackState {
  return {
    ...state,
    spacing: defaults.spacing,
    height: defaults.height,
    layers: meta.layers,
    texels: Math.min(defaults.texels, meta.resolution),
    brightness: defaults.brightness,
    contrast: defaults.contrast,
    saturation: defaults.saturation,
    shade: defaults.shade,
    auto: defaults.auto,
    pixel: defaults.pixel,
    pixelSize: defaults.pixelSize,
    outline: defaults.outline,
    round: defaults.round,
    main: defaults.main,
    flat: defaults.flat,
  };
}

/* ---------------------------------------------------------------- the stats line (stack-lab.js 1165-1190) */

/** "Voxel · Pixel · Round 0.45 · Main" (legacy lookLabel). */
export function lookLabel(state: Pick<StackState, "look" | "pixel" | "round" | "main"> & { flat?: boolean }): string {
  let name = "Voxel";
  if (state.look === "plane") name = "Plane";
  if (state.look === "contour") name = "Contour";
  if (state.look === "slices") return "Sprite stack";
  if (state.flat) name += " · Pixel art";
  if (state.pixel) name += " · Pixel";
  if (state.look === "voxel" && state.round > 0.001) name += ` · Round ${state.round.toFixed(2)}`;
  if (state.main) name += " · Main";
  return name;
}

/** One run of the stats line; `b` runs are bold (legacy <b>). */
export interface StatsPart {
  text: string;
  b?: boolean;
}

/** `<b>{lookLabel}</b>{px} · <b>{layers}</b> slabs · up to <b>{resolution}²</b>`, as parts (no innerHTML). */
export function statsParts(state: StackState, meta: Pick<StackMeta, "layers" | "resolution">): StatsPart[] {
  const px = state.pixel ? ` · ${state.pixelSize}px` : "";
  return [
    { text: lookLabel(state), b: true },
    { text: `${px} · ` },
    { text: String(meta.layers), b: true },
    { text: " slabs · up to " },
    { text: `${meta.resolution}²`, b: true },
  ];
}

export const statsText = (parts: StatsPart[]) => parts.map((p) => p.text).join("");

/* ---------------------------------------------------------------- copy, word for word */

export const STACK_COPY = {
  pane: "Sprite stack",
  /** #stack-empty: initial, on any load failure, and after unload. It names the Versions panel's two actions. */
  empty: "No sprite stack yet. Use Build sprite stack, Stack from picture or Draw views & stack (bottom left).",
  slicing: "Slicing…",
  generating: "Generating…",
  /** #stack-stats when the stack would not load. */
  noStack: "No sprite stack",
  lookBlurb: "Voxel cubes are what the game uses. Sprite draws each layer as a flat pixel picture, outlined. Plane and Contour are older experiments.",
  /** The walking character's motion choice (MOTION_BUTTONS). */
  motion: "Motion",
  paletteHint: "Reduce to N distinct hues. Rare lights are kept. Then pick a swatch for metal / roughness / glow.",
  paintHint: "Paint the stack and the model with one 3D brush. Save keeps it.",
} as const;

/** The three cubbies (legacy CUBBY_LABELS): the dock's bar labels. */
export type StackCubby = "stack-look" | "stack-sliders" | "stack-palette" | "paint";
export const CUBBY_LABELS: Record<StackCubby, string> = {
  "stack-look": "Look",
  "stack-sliders": "Shape",
  "stack-palette": "Colors",
  paint: "Paint",
};
/** Sprite stack is the one open by default (index.html 2172). */
export const CUBBY_DEFAULT: StackCubby = "stack-look";
/** Legacy openStep: open the named cubby, or close it when it is the open one. */
export const openCubby = (open: StackCubby | null, name: StackCubby): StackCubby | null => (open === name ? null : name);

export const LOOK_BUTTONS: { look: StackLook; label: string; title: string }[] = [
  { look: "voxel", label: "Voxel", title: "Instanced cubes per texel. The factory asset." },
  { look: "plane", label: "Plane", title: "An older experiment: one flat picture per slice." },
  { look: "contour", label: "Contour", title: "An older experiment: smooth outlined slabs. Not what the game uses." },
  {
    look: "slices",
    label: "Sprite",
    title: "The sprite-stack look: each layer a flat picture, one cell to a screen pixel, stacked bottom-up, with a dark outline and lower layers darker.",
  },
];

/** A walking character's motion (pipeline#15): the rig's own walk (pipeline#2), or a skeleton cycle from code
 * (pipeline#14) played round the same lap. */
export type WalkMotion = "rig" | "walk" | "run" | "attack";
export const MOTION_DEFAULT: WalkMotion = "rig";
/** ?motion=walk|run|attack opens on that motion, as ?look= opens on a look. */
export function motionFromSearch(search = ""): WalkMotion {
  const m = new URLSearchParams(search).get("motion");
  return MOTION_BUTTONS.some((b) => b.motion === m) ? (m as WalkMotion) : MOTION_DEFAULT;
}
export const MOTION_BUTTONS: { motion: WalkMotion; label: string; title: string }[] = [
  { motion: "rig", label: "Rig walk", title: "The rig's own walk: planted feet that stay on their square." },
  { motion: "walk", label: "Walk", title: "The skeleton's walk, made in code (the same frames as the walk video)." },
  { motion: "run", label: "Run", title: "The skeleton's run, made in code." },
  { motion: "attack", label: "Attack", title: "The skeleton's punch, made in code. It stays on the spot." },
];

export type StackToggle ="pixel" | "main" | "outline" | "auto" | "shade" | "lit";
/** #stack-sliders, in legacy order; Reset sits after them, on its own row (TOGGLE_RESET). */
export const TOGGLE_BUTTONS: { key: StackToggle; label: string; title: string }[] = [
  {
    key: "pixel",
    label: "Pixel edges",
    title: "Draw the stack at a low resolution, then scale it up with crisp pixel edges. Does not change the cubes.",
  },
  { key: "main", label: "Main piece", title: "Keep only the largest connected piece. Off keeps the stray bits too." },
  { key: "outline", label: "Outline", title: "1-pixel screen-space outlines from the depth buffer, at the internal pixel grid." },
  { key: "auto", label: "Follow 3D", title: "Keep this camera at the same angle and slow turn as the 3D model." },
  { key: "shade", label: "Shade", title: "Darken lower slabs so height is easier to read." },
];
/** Pixel art only: the cubes take the scene's light. Off is the sheet's exact colours. */
export const LIGHT_TOGGLE: { key: StackToggle; label: string; title: string } = {
  key: "lit",
  label: "Light",
  title: "Light the stack: sun and sky on its faces, and cast shadows in Game look. Off shows the sheet's exact colours.",
};
/** The switches a stack shows: a pixel art stack adds Light (a model stack is always lit). */
export const togglesFor = (state: Pick<StackState, "flat">) => (state.flat ? [...TOGGLE_BUTTONS, LIGHT_TOGGLE] : TOGGLE_BUTTONS);
export const TOGGLE_RESET = {
  label: "Reset",
  title: "Restore factory stack look: 120 texels, spacing/height 1, contrast and saturation 1.20.",
};

/** The step-bar game look (#game-look-bar), which the Stack stage showed: the shared pipeline.pixelLook. */
export const GAME_TOGGLE = {
  label: "Game look",
  title: "Game look: render both panes small with crisp pixels, iso camera and shadows. Settings are under View.",
};

export type SliderKey = "pixelSize" | "round" | "spacing" | "height" | "layers" | "texels" | "brightness" | "contrast" | "saturation";
export interface SliderSpec {
  key: SliderKey;
  label: string;
  min: number;
  /** The DOM max; Layers and Texels follow stack.json (sliderMax). */
  max: number;
  step: number;
  /** toFixed digits for the readout; null = String(value). */
  digits: number | null;
  title: string;
}

/** The Stack cubby's sliders (index.html 2204-2214), in order. */
export const SLIDERS: SliderSpec[] = [
  { key: "pixelSize", label: "Pixel size", min: 1, max: 8, step: 1, digits: null, title: "How big each screen pixel is. 1 is full resolution; 3–4 is chunky pixel art." },
  {
    key: "round",
    label: "Round",
    min: 0,
    max: 1,
    step: 0.01,
    digits: 2,
    title: "Merge neighboring voxels into one surface. 0 is cubes. Low values are a tiny blend (squared curve). 1 is a smooth iso. Factory is 0.45.",
  },
  { key: "spacing", label: "Spacing", min: 0.5, max: 18, step: 0.05, digits: 1, title: "Gap between slabs. 1 is packed. Raise it to pull the loaf apart." },
  { key: "height", label: "Height", min: 0.1, max: 4, step: 0.05, digits: 2, title: "Thickness of each slab. Raise to fill gaps between layers." },
  { key: "layers", label: "Layers", min: 4, max: 256, step: 1, digits: null, title: "How many slabs from the bottom. Lower crops the top of the object." },
  { key: "texels", label: "Texels", min: 16, max: 256, step: 8, digits: null, title: "XY resolution of each slab. Lower is chunkier pixels; higher is finer." },
  { key: "brightness", label: "Bright", min: 0.3, max: 2.2, step: 0.05, digits: 2, title: "Overall brightness of the stack." },
  { key: "contrast", label: "Contrast", min: 0.3, max: 2.2, step: 0.05, digits: 2, title: "Contrast. 1.20 is the factory look." },
  { key: "saturation", label: "Saturate", min: 0, max: 2.2, step: 0.05, digits: 2, title: "Color intensity. 0 is grey, 1.20 is the factory look." },
];

/** The sliders a stack shows: a pixel art stack has no Round (cubes only) and no Texels (every texel is the art),
 * and Layers goes down to one slab. */
export const slidersFor = (state: Pick<StackState, "flat">): SliderSpec[] =>
  state.flat
    ? SLIDERS.filter((s) => s.key !== "round" && s.key !== "texels").map((s) => (s.key === "layers" ? { ...s, min: 1 } : s))
    : SLIDERS;

/** A slider's readout (legacy textContent: toFixed(digits), or the integer as is). */
export const readout = (spec: Pick<SliderSpec, "digits">, value: number): string =>
  spec.digits == null ? String(value) : Number(value).toFixed(spec.digits);

/** Layers' max is the stack's layer count and Texels' its resolution (applyMetaToSliders). */
export function sliderMax(spec: SliderSpec, meta: Pick<StackMeta, "layers" | "resolution"> | null): number {
  if (meta && spec.key === "layers") return meta.layers;
  if (meta && spec.key === "texels") return meta.resolution;
  return spec.max;
}

/** A slider moved: the value as the state keeps it (legacy clamps Round to 0..1 and Px to 1..8, default 3). */
export function sliderValue(key: SliderKey, raw: number): number {
  if (key === "round") return Math.max(0, Math.min(1, Number(raw) || 0));
  if (key === "pixelSize") return Math.max(1, Math.min(8, Number(raw) || 3));
  return Number(raw);
}

/* ---------------------------------------------------------------- palette (index.html 2224-2232, 5592-5642) */

export interface PaletteEntry {
  r: number;
  g: number;
  b: number;
  metal: number;
  rough: number;
  emit: number;
}

export const PALETTE_COPY = {
  label: "Palette",
  count: { label: "Colors", min: 2, max: 32, step: 1, value: 8, title: "How many distinct colors to keep." },
  reduce: { label: "Reduce", title: "Keep unique hues (chroma-weighted), not random k-means." },
  swatchTitle: (i: number) => `Swatch ${i + 1}. Paint uses this color.`,
} as const;

export type PbrKey = "metal" | "rough" | "emit";
export const PBR_SLIDERS: { key: PbrKey; label: string; min: number; max: number; step: number; value: number; title: string }[] = [
  { key: "metal", label: "Metal", min: 0, max: 1, step: 0.05, value: 0, title: "Metallic for the selected color." },
  { key: "rough", label: "Rough", min: 0, max: 1, step: 0.05, value: 0.7, title: "Roughness for the selected color. Low is glossy." },
  { key: "emit", label: "Glow", min: 0, max: 1, step: 0.05, value: 0, title: "Emissive glow for the selected color." },
];
export const pbrReadout = (v: number) => Number(v).toFixed(2);

/* ---------------------------------------------------------------- paint (index.html 2113-2124, 5372-5515) */

export const PAINT_COPY = {
  paint: { label: "Brush", title: "Click-drag to paint. Right-drag still orbits." },
  drop: { label: "Drop", title: "Next click samples color from the surface." },
  erase: { label: "Erase", title: "Brush restores the original colors." },
  undo: { label: "Undo", title: "Undo the last stroke." },
  save: { label: "Save", title: "Write painted stack sheet and textured_paint.glb." },
  color: { label: "Color", title: "Brush color", value: "#2ecc71" },
  radius: { label: "Size", min: 0.01, max: 0.2, step: 0.005, value: 0.04, title: "Paint sphere size, as a fraction of the mesh." },
} as const;
/** Brush's readout: the page's "0.04" until the slider moves, then toFixed(3) (legacy paintRadiusEl input). */
export const radiusReadout = (v: number, moved = true) => (moved ? Number(v).toFixed(3) : String(v));

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(String(hex).replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
export function rgbToHex(r: number, g: number, b: number): string {
  return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("");
}

/** Brush radius in world units: (diagonal || 1) * the slider (legacy worldPaintRadius, on the stack's own box here). */
export const worldPaintRadius = (diagonal: number, value: number) => (diagonal || 1) * value;

/* ---------------------------------------------------------------- the camera (stack-lab.js 139-151) */

/** OrbitControls as legacy set them: full polar range (tests/test_stack_looks.py pins maxPolarAngle = Math.PI). */
export const ORBIT = {
  enableDamping: true,
  dampingFactor: 0.05,
  enablePan: true,
  minDistance: 0.3,
  maxDistance: 48,
  minPolarAngle: 0,
  maxPolarAngle: Math.PI,
  autoRotateSpeed: 0.7,
} as const;

/** What the stage shows in the middle: the stack, or one of the three empty texts. */
export type StackPhase = "empty" | "slicing" | "generating" | "ready";
export const phaseText = (phase: StackPhase): string =>
  phase === "slicing" ? STACK_COPY.slicing : phase === "generating" ? STACK_COPY.generating : phase === "empty" ? STACK_COPY.empty : "";

/** A job that writes out/<stem>/stack/ is running on this asset (legacy clearStage on a build): not a 2D draw,
 * ideas, a rig or a contact-sheet render. */
export function buildingStack(job: { state: string; stem: string; kind: string } | null, stem: string): boolean {
  return !!job && job.state === "running" && job.stem === stem && !["flat2d", "ideas", "rig", "render"].includes(job.kind);
}
