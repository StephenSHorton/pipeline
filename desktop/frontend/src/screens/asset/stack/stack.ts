// The Stack stage's copy and pure state (legacy #stack-pane, index.html 2167-2236; stack-lab.js DEFAULTS,
// applyMetaToSliders, the Reset handler and the stats line, 5-27, 1165-1190, 1619-1702; the Paint cubby and
// the palette host code, index.html 2113-2124, 5384-5642). StackStage.tsx lays these out; view.ts draws.
// No React, no fetch here.

export type StackLook = "voxel" | "plane" | "contour";
export const LOOKS: StackLook[] = ["voxel", "plane", "contour"];

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
}

/** stack.json, as the page reads it (layers, resolution, pitch, extents, defaults). */
export interface StackMeta {
  layers: number;
  resolution: number;
  pitch: number;
  extents: number[];
  defaults?: Partial<Record<keyof StackState, unknown>> | null;
  maker?: string;
}

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
  };
}

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
  return out;
}

/** A stack loaded (legacy applyMetaToSliders): layers, texels, spacing, height and the grade go back to its defaults;
 * look, pixel, round, main, shade and sync stay as they were. */
export function stateForMeta(state: StackState, defaults: StackState, meta: StackMeta): StackState {
  return {
    ...state,
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
  };
}

/* ---------------------------------------------------------------- the stats line (stack-lab.js 1165-1190) */

/** "Voxel · Pixel · Round 0.45 · Main" (legacy lookLabel). */
export function lookLabel(state: Pick<StackState, "look" | "pixel" | "round" | "main">): string {
  let name = "Voxel";
  if (state.look === "plane") name = "Plane";
  if (state.look === "contour") name = "Contour";
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
  empty: "No sprite stack yet. Use Build sprite stack or Stack from picture (bottom left).",
  slicing: "Slicing…",
  generating: "Generating…",
  /** #stack-stats when the stack would not load. */
  noStack: "No sprite stack",
  lookBlurb: "Voxel cubes are what the game uses. Plane and Contour are older experiments.",
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
];

export type StackToggle = "pixel" | "main" | "outline" | "auto" | "shade";
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
