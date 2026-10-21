// Pure Make-panel logic: the kinds, the automatic guesses (size, name), and the exact request bodies the legacy
// page sends (readIntent / lookSpec / intentBody / startFlatJob / buildFromConcept, index.html). Kept free of
// React so make.test.ts can pin the bodies.
import { slugStem } from "@/lib/names";
import { listAssets, type Asset, type AssetOutput, type Facing, type FacingSource, type Intent } from "@/lib/api";

/* ---------------------------------------------------------------- kinds */

export interface KindDef {
  id: AssetOutput;
  label: string;
  make: string;
  hint: string;
  /** The legacy OUTPUTS name (finish toast: "<name> ready: …"). */
  name: string;
}

export const KINDS: readonly KindDef[] = [
  { id: "ps1", label: "PS1 model", make: "Make PS1 model", hint: "Low-poly, wearing your photo", name: "PS1 model" },
  { id: "model", label: "3D model", make: "Make 3D model", hint: "Painted mesh, detail levels", name: "3D model" },
  { id: "stack", label: "Sprite stack", make: "Make sprite stack", hint: "Voxel slices for 2D engines", name: "Sprite stack" },
  { id: "sprite", label: "2D sprite", make: "Make 2D sprite", hint: "Picture, pieces, motion, sound", name: "2D sprite" },
];

export const kindDef = (id: string | null | undefined): KindDef => KINDS.find((k) => k.id === id) || KINDS[0];
export const isThreeD = (k: AssetOutput) => k !== "sprite";

const KIND_KEY = "make-output";
export function loadKind(): AssetOutput {
  try {
    const v = localStorage.getItem(KIND_KEY);
    if (v && KINDS.some((k) => k.id === v)) return v as AssetOutput;
  } catch {
    /* private window */
  }
  return "ps1";
}
export function saveKind(k: AssetOutput): void {
  try {
    localStorage.setItem(KIND_KEY, k);
  } catch {
    /* ignore */
  }
}

/* ---------------------------------------------------------------- size (PS1 height) */

/** The legacy height chips. */
export const SIZES = [
  { label: "Person", inches: 72, subject: "person" },
  { label: "Animal", inches: 58, subject: "animal" },
  { label: "Bust", inches: 30, subject: "" },
  { label: "Prop", inches: 24, subject: "prop" },
] as const;

/** Height guessed from the kind of thing (the approved design's word lists), from the description or, with no
 * words, the photo's file name. People first ("a man holding a lantern" is a person), then busts, then props;
 * anything else is an animal. Index into SIZES. */
export function guessSize(text: string, fileName?: string): number {
  const t = (text.trim() || (fileName || "").replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ")).toLowerCase();
  if (/\b(man|woman|person|knight|guard|farmer|girl|boy)\b/.test(t)) return 0;
  if (/(bust|statue)/.test(t)) return 2;
  if (/(lantern|radio|lamp|chair|crate|sword|cup|box|barrel)/.test(t)) return 3;
  return 1;
}

/* ---------------------------------------------------------------- facing */

export type FacingPick = "auto" | Facing;
export const FACING_CYCLE: readonly FacingPick[] = ["auto", "left", "right", "camera", "away"];
export const FACING_SHORT: Record<Facing, string> = { left: "left", right: "right", camera: "toward me", away: "away" };

/* ---------------------------------------------------------------- detail (3D model / sprite stack) */

export type Poly = "high" | "medium" | "low";
export const POLYS: readonly { id: Poly; label: string; hint: string }[] = [
  { id: "high", label: "High", hint: "Maximum fidelity · ~full sculpt" },
  { id: "medium", label: "Medium", hint: "Keep fidelity, faster · up to 80k faces" },
  { id: "low", label: "Low", hint: "Low-poly attempt · ~2k faces" },
];

/* ---------------------------------------------------------------- 2D sprite steps */

export type Step = "pieces" | "say" | "sound";
/** The legacy step fields' placeholders (ix:1929-1942). Motion needs a clip, which only the classic view takes. */
export const STEPS: readonly { id: Step; label: string; placeholder: string }[] = [
  { id: "pieces", label: "Pieces", placeholder: "Name the pieces… (optional)" },
  { id: "say", label: "Say", placeholder: "What it says… (optional)" },
  { id: "sound", label: "Sound", placeholder: "The sound it makes… (optional)" },
];

/* ---------------------------------------------------------------- names (legacy slugStem / flatName / freshStem) */

export { slugStem };

export function flatName(prompt: string, fileName?: string): string {
  const raw = (prompt || (fileName && fileName.replace(/\.[^.]+$/, "")) || "sheet").toLowerCase();
  const slug = raw.replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
  return slug || "sheet";
}

/** A stem nobody uses yet: base, base-2, base-3… (GET /assets). */
export async function freshStem(base: string): Promise<string> {
  const list = await listAssets().catch(() => ({ assets: [] as Asset[] }));
  const taken = new Set((list.assets || []).map((a) => a.id));
  let stem = base;
  for (let n = 2; taken.has(stem); n++) stem = `${base}-${n}`;
  return stem;
}

/** The automatic name: from the photo's file name, else from the description. "" when neither. */
export function autoName(text: string, fileName?: string): string {
  if (fileName) return slugStem(fileName);
  return slugStem(text);
}

/* ---------------------------------------------------------------- request bodies */

export interface MakeChoice {
  kind: AssetOutput;
  prompt: string;
  /** PS1: the facing sent (camera when nothing better is known), and where it came from. */
  facing: Facing;
  facingSource?: FacingSource;
  height: number;
  /** Slug with dashes ("" = automatic). */
  name: string;
  poly: Poly;
  steps: Step[];
  /** Only when the user picked a size in Options: the draw's pose subject (person / animal / prop). */
  subject?: string;
  /** The words for each picked 2D step (legacy #gen-pieces / #gen-say / #gen-sound). */
  stepText?: Partial<Record<Step, string>>;
}

/** Legacy readIntent: {output, psx:{facing,height,name?}, facing_source?} | {output, poly} | {output, steps}. */
export interface NestedIntent {
  output: AssetOutput;
  psx?: { facing?: Facing; height?: number | string; name?: string };
  poly?: Poly;
  steps?: string[];
  facing_source?: FacingSource;
  subject?: string;
  kind?: string;
  rig?: Record<string, unknown>;
  name?: string;
}

export function readIntent(c: MakeChoice, opts: { drawn?: boolean } = {}): NestedIntent {
  const intent: NestedIntent = { output: c.kind };
  if (c.kind === "ps1") {
    // A drawn picture's facing comes from the drawing (canonical: side-on, head right); only a user pick is sent.
    intent.psx = opts.drawn && c.facingSource !== "user" ? { height: c.height } : { facing: c.facing, height: c.height };
    if (c.name) intent.psx.name = c.name.replace(/-/g, "_");
    if (c.facingSource && !(opts.drawn && c.facingSource !== "user")) intent.facing_source = c.facingSource;
  } else if (c.kind === "model" || c.kind === "stack") {
    intent.poly = c.poly;
  } else {
    intent.steps = [...c.steps];
  }
  if (c.subject && isThreeD(c.kind)) intent.subject = c.subject;
  return intent;
}

/** Legacy intentBody: the flat form /intent and /generate take. */
export function intentBody(intent: NestedIntent): Intent {
  const body: Record<string, unknown> = { output: intent.output };
  if (intent.psx) Object.assign(body, intent.psx);
  if (intent.poly) body.poly = intent.poly;
  if (intent.steps) body.steps = intent.steps;
  for (const k of ["facing_source", "subject", "kind", "rig"] as const) if (intent[k]) body[k] = intent[k];
  return body as unknown as Intent;
}

/** Legacy lookSpec. */
export function lookSpec(intent: NestedIntent): Record<string, unknown> {
  if (intent.output === "ps1") {
    const p = intent.psx || {};
    const spec: Record<string, unknown> = { quality: "psx", facing: p.facing || "camera", psx_height: Number(p.height) || 48 };
    if (p.name) spec.psx_name = p.name;
    return spec;
  }
  return { quality: intent.output === "model" ? "textured" : "stack" };
}

/** POST /2d/draw for a photo, 3D kinds: the photo is a reference the four-view sheet is drawn FROM, never a view, then
 * /generate on flat/source.png like a drawing. image: the upload's root-relative path (in/…). */
export function photoDrawBody(c: MakeChoice, stem: string, image: string): Record<string, unknown> {
  return { ...drawFirstBody(c, stem), reference: image };
}

/** POST /2d/draw for a description with no picture, 3D kinds: the four-view sheet (wants_views keys off intent.output). */
export function drawFirstBody(c: MakeChoice, stem: string): Record<string, unknown> {
  return { kind: "flat2d", stem, prompt: c.prompt, intent: intentBody(readIntent(c, { drawn: true })), steps: [], views: "4" };
}

/** POST /2d/draw for a 2D sprite (legacy startFlatJob). source "drop" when the picture was dropped first. */
export function spriteDrawBody(c: MakeChoice, stem: string, dropped: boolean): Record<string, unknown> {
  const body: Record<string, unknown> = { kind: "flat2d", stem, prompt: c.prompt, intent: intentBody(readIntent(c)), steps: [...c.steps] };
  // Legacy pickedStepFields: each picked step's text goes along ("" when left empty).
  for (const st of c.steps) body[st] = (c.stepText?.[st] || "").trim();
  if (dropped) body.source = "drop";
  return body;
}

/** The drawn sheet → 3D (legacy buildFromConcept). saved: out/<stem>/intent.json; key: out/<stem>/flat/key.json. */
export function buildFromDrawingBody(
  stem: string,
  want: NestedIntent,
  saved: NestedIntent | null,
  key: { facing?: Facing; canonical?: boolean; facing_source?: FacingSource } | null,
): Record<string, unknown> {
  let intent: NestedIntent = saved && saved.output === want.output ? { ...saved } : { ...want };
  if (intent.output === "ps1") {
    const psx = { ...(want.psx || {}), ...(intent.psx || {}) };
    psx.height = Number(psx.height) || Number(want.psx?.height) || 48;
    intent = { ...intent, psx };
    const userSet = want.facing_source === "user" && want.psx?.facing;
    if (userSet) intent = { ...intent, psx: { ...psx, facing: want.psx!.facing }, facing_source: "user" };
    else if (key && key.facing) intent = { ...intent, psx: { ...psx, facing: key.facing }, facing_source: key.canonical ? "drawn" : key.facing_source || "detected" };
  }
  const still = `out/${stem}/flat/source.png`;
  return { image: still, images: [still], stem, ...lookSpec(intent), poly: intent.poly || want.poly || "medium", intent: intentBody(intent) };
}

/** factory_env NO_PICTURE: the factory's own words when 3D is asked for with no picture (make.test pins them). */
export const NO_PICTURE = "3D needs a picture. Draw or add one first.";

/**
 * The 3D intent a finished draw is for, from the out/<stem>/intent.json it saved (nested, like readIntent) when this
 * page's pick is gone (a reload mid-draw). A PS1 height the file lacks is guessed from its name, as the panel does.
 */
export function recoverDrawIntent(saved: NestedIntent, stem: string): NestedIntent {
  const want: NestedIntent = { ...saved };
  if (want.output === "ps1") {
    const psx = { ...(want.psx || {}) };
    if (!(Number(psx.height) > 0)) psx.height = SIZES[guessSize(String(want.name || psx.name || stem).replace(/[_-]+/g, " "))].inches;
    want.psx = psx;
  }
  return want;
}

/* ---------------------------------------------------------------- library */

export type LibKind = AssetOutput | "ideas" | "";

/** Legacy assetKind: the saved intent, else a guess from the files. */
export function assetKind(a: Asset): LibKind {
  if (a.output && KINDS.some((k) => k.id === a.output)) return a.output;
  if (a.has_psx) return "ps1";
  if (a.has_flat && !a.has_mesh) return "sprite";
  if (a.has_stack && !a.has_mesh) return "stack";
  if (a.has_mesh) return "model";
  return a.has_ideas ? "ideas" : "";
}

export const KIND_LABEL: Record<string, string> = { ps1: "PS1 model", model: "3D model", stack: "Sprite stack", sprite: "2D sprite", ideas: "Ideas" };

/** The library filter chips (legacy #lib-filter). */
export const LIB_FILTERS: readonly { id: LibKind; label: string }[] = [
  { id: "", label: "All" },
  { id: "ps1", label: "PS1" },
  { id: "model", label: "3D" },
  { id: "stack", label: "Stack" },
  { id: "sprite", label: "2D" },
  { id: "ideas", label: "Ideas" },
];

/** Legacy relativeTime (seconds since epoch). */
export function relativeTime(sec: number, nowMs = Date.now()): string {
  const d = nowMs / 1000 - sec;
  if (d < 3600) return `${Math.max(1, Math.round(d / 60))} min ago`;
  if (d < 86400) return `${Math.round(d / 3600)} h ago`;
  if (d < 86400 * 14) return `${Math.round(d / 86400)} days ago`;
  return new Date(sec * 1000).toLocaleDateString();
}

export function matchesFilter(a: Asset, kind: LibKind, query: string): boolean {
  const k = assetKind(a);
  if (kind && k !== kind) return false;
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return `${a.id} ${a.title || ""} ${KIND_LABEL[k] || ""}`.toLowerCase().includes(q);
}

/** "About N s|N min" (legacy roughSeconds: minutes from 90 s, else rounded to 5 s, at least 5). */
export function roughSeconds(s: number): string {
  if (s >= 90) return `${Math.round(s / 60)} min`;
  return `${Math.max(5, Math.round(s / 5) * 5)} s`;
}

export const DELETE_MESSAGE = (id: string) => `Delete this generation from disk?\n\nout/${id}/\n\nLocal files only. This cannot be undone.`;

export interface FlatFlags {
  loaded?: boolean;
  times?: Record<string, number | null | undefined>;
  estimate_s?: { draw?: number | null };
}

/** Legacy drawTimeNote: how long the picture (and picked steps) took last time, from GET /2d/status. */
export function drawTimeNote(f: FlatFlags | null | undefined, steps: readonly string[] = []): string {
  const t = (f && f.times) || {};
  const num = (v: unknown): v is number => typeof v === "number" && v >= 0;
  const cold = !!f && f.loaded === false;
  const loadHint = cold ? " The first draw also loads the model." : "";
  const est = (f && f.estimate_s) || {};
  const picture = num(t.attempt) ? t.attempt + (num(t.ground) ? t.ground : 0) : est.draw;
  if (!num(picture) || !(picture > 0)) return loadHint;
  const extra = steps.map((k) => t[k]);
  const load = cold && num(t.load) ? t.load : 0;
  if (extra.every(num)) {
    const total = picture + (extra as number[]).reduce((a, b) => a + b, 0) + load;
    return ` About ${roughSeconds(total)} last time${load ? ", with loading the model" : ""}.${load ? "" : loadHint}`;
  }
  return ` The picture took about ${roughSeconds(picture)} last time.${loadHint}`;
}
