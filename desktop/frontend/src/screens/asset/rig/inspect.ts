// The Rig check panel's pure logic and copy (legacy public/rig-inspect.js: the constants 9-160, the checks 145-172,
// legend / boneCard / headBlock 555-669, checkRow / motionBlock / compareBlock 770-874, renderPanel 878-946 and the
// sheet 948-966). Words are verbatim; RigPanel.tsx only lays these out, and the 3D side (colour modes, skeleton,
// hover card) lives in ./overlay.ts. No React, no fetch, no three here.

/** out/<stem>/rig/inspect.json (scripts/rig_inspect.py). Bone index = skin joint order = this order. */
export interface RigBone {
  name: string;
  parent: number;
  label?: string;
  role?: string;
  template?: string;
  rule?: string;
  ref_vs_rest_deg?: number;
}

export interface RigCheck {
  key: string;
  level: "pass" | "warn" | "fail" | string;
  label?: string;
  text?: string;
  step?: string;
  bone?: string;
  clip?: string;
  t?: number;
  yaw?: number;
  points?: number;
  by_hand?: boolean;
  straightened?: boolean;
  before_straightening?: boolean;
  clips?: Record<string, number | number[]>;
  over?: string[];
  worst?: [string, number][];
  owners?: [string, number][];
  [k: string]: unknown;
}

export interface ClipMotion {
  tears: number;
  stretch_p99?: number;
  stretch_max?: number;
  tear_t?: number;
  worst_t?: number;
  worst_bone?: string;
}

export interface RestInfo {
  measured?: { turned?: boolean; head_yaw_deg: number } | null;
  straightened?: { head_yaw_deg: number } | null;
  before_straightening?: unknown;
}

export interface HeadSummary {
  straightened?: boolean;
  made_before?: boolean;
  off_deg?: number;
  before?: number;
  after?: number;
}

export interface RigCheckSummary {
  tears?: number;
  head?: HeadSummary | null;
  motion?: Record<string, { stretch_p99?: number; stretch_max?: number }>;
  class?: string;
  level?: string;
  line?: string;
}

export interface RigInspectDoc {
  bones: RigBone[];
  checks?: RigCheck[];
  roles?: string[];
  planner?: { image?: string; choice?: string; confidence?: number; unrecorded?: boolean };
  template?: {
    built?: boolean;
    plan?: string;
    rig?: string;
    ref_clip?: string;
    fit_scale?: number;
    fit_offset?: number[];
    max_joint_move_by_skinner_m?: number;
  };
  rig?: { plan?: string; class?: string; rest?: RestInfo | null; source?: string; [k: string]: unknown };
  weights?: {
    top_final?: number[];
    dropped?: { total?: number; list?: [number, number, number][] };
    seam_raw?: { positions_with_conflicting_top_bone?: number };
  };
  fit?: { bone: string; median_dist_rel_h: number; centroid_off_bone_rel_h?: number }[];
  clips?: Record<string, { peak_deg: number[] }>;
  motion?: { clips?: Record<string, ClipMotion> } | null;
  head?: { off_deg: number } | null;
  summary?: RigCheckSummary;
  notes?: string[];
}

/** The colour modes: id, button, title (rig-inspect.js 9-15). */
export const MODES = [
  ["texture", "Texture", "The painted model."],
  ["bones", "Bones", "Each part in the colour of the bone that moves it most."],
  ["heat", "One bone", "How strongly the picked bone pulls each point: red is fully, blue is not at all."],
  ["dropped", "Dropped", "Weight thrown away because the bone was too far from the point."],
  ["split", "Split", "Points whose corners follow different bones. They pull apart when the model moves."],
] as const;
export type RigMode = (typeof MODES)[number][0];

/** Top looks down with the head up the screen, away from the stage bar. The vector is in the model's own space. */
export const VIEWS = [
  ["side", "Side", [1, 0, 0]],
  ["front", "Front", [0, 0, 1]],
  ["top", "Top", [0, 1, -0.001]],
  ["free", "3/4", [0.72, 0.38, 1]],
] as const;
export type RigViewId = (typeof VIEWS)[number][0];

export const ORDER: Record<string, number> = { fail: 0, warn: 1, pass: 2 };
export const MINOR = /leaf|tip|_end$/;
// Case-insensitive: the bird and spider rigs name legs UpperLeg_L, Foot_R, Toes_tip_L.
export const NO_FIT = /leaf|tip|_end$|^ear/i; // ears move few points: never a fit problem worth leading with
export const LEG = /leg|foot|toe|thigh|shin|calf|knee|hoof|ball|humerus|femur|tibia/i;
export const ARM = /arm|hand|shoulder|clavicle|elbow|wrist|finger/i;
// Stick colours by bone role (inspect.json bones[].role), the same as rig_render.py's ROLE_RGB.
export const ROLE_RGB: Record<string, [number, number, number]> = { leg: [26, 102, 255], wing: [255, 128, 0], spine: [255, 255, 26], head: [255, 26, 255], tail: [26, 255, 77],
  other: [240, 234, 220], root: [110, 110, 110] };
export const ROLE_WORDS: Record<string, string> = { leg: "legs", wing: "wings", spine: "spine and neck", head: "head", tail: "tail", other: "other", root: "root" };
// The rig's own steps, in the order rig.py runs them; every check belongs to one.
export const STEPS = [["plan", "Body plan"], ["rest", "Rest pose"], ["skeleton", "Skeleton"], ["skin", "Skin"], ["motion", "In motion"]] as const;
export const KEY_STEP: Record<string, string> = { planner: "plan", rest: "rest", head: "skeleton", fit: "skeleton", moved: "skeleton", facing: "skeleton", match: "skeleton",
  pose: "skeleton", pred: "skeleton", template: "skeleton", built: "skeleton", seam: "skin", folded: "skin", a2: "skin", foot: "skin",
  parts: "skin", clips: "motion", motion: "motion", far: "motion", stretch: "motion", floor: "motion", unsupported: "motion",
  cshape: "motion", slide: "motion", loop_seam: "motion", fbx: "motion" };
// Classes (the planner's choice) and body plans (scripts/rig_plans/: rig.json plan, inspect template.plan). Not RIG_NAMES.
export const PLANS: Record<string, string> = { quadruped: "Four legs", humanoid: "Person", multileg: "Many legs", avian: "Bird",
  ground_bird: "Ground bird", flying_bird: "Flying bird", horse: "Four legs", biped: "Two legs, two arms", spider: "Many legs" };

type Help = string | Partial<Record<string, string>> | ((c: RigCheck, doc: RigInspectDoc | null) => string);
interface CheckHelp {
  why?: Help;
  did?: Help;
  fix?: Help;
  note?: Help;
}

const turnedAtBirth = (doc: RigInspectDoc | null) => !!doc?.rig?.rest?.measured?.turned;
const straightenedDoc = (doc: RigInspectDoc | null) => !!doc?.rig?.rest?.straightened;

/** Why a check matters, what the rigger did about it, and what you can do (To fix), word for word (rig-inspect.js 41-137). */
export const CHECK_HELP: Record<string, CheckHelp> = {
  planner: {
    why: (c) => (c.by_hand || /set by hand$/.test(c.label || "") ? "" : "The skeleton is picked from the model's picture."),
    fix: { warn: "Pick the body plan next to Re-rig if this is wrong." },
  },
  rest: {
    why: "3D generation often turns a side-on animal's head toward the camera. The skeleton's neck runs straight, so every clip would keep the turn.",
    did: (c) => (c.straightened ? "Fitted the neck bones to the straightened head." : c.level === "warn" && !c.before_straightening ? "Rigged the head as made, because you asked." : ""),
  },
  head: {
    why: (_c, doc) => (turnedAtBirth(doc) ? "3D generation turned the head toward the camera, but the neck bones run straight, so head moves swing it sideways."
      : "When the head points away from its bone, head moves swing it sideways."),
    fix: (c, doc) => (c.level === "pass" || straightenedDoc(doc) ? "" : turnedAtBirth(doc) ? "Straighten and rig, under Rest pose." : ""),
  },
  fit: {
    why: "This bone sits away from the part it moves, so that part swings around the wrong point.",
    did: (c) => (ARM.test(c.bone || "") && !LEG.test(c.bone || "") ? "Arms are aimed at the hands when rigging." : ""),
    note: (c) => (LEG.test(c.bone || "") ? "Legs aren't aimed at the feet when rigging yet, so re-rigging won't move it. It may not show in the clips." : ""),
  },
  moved: { why: "The skinning step moved some joints from where they were fitted." },
  seam: {
    why: "Many models store a point once per triangle corner, as the low-poly look does. When copies of one point follow different bones, they pull apart as it moves.",
    did: { pass: "Gave every copy of a point the same bones." },
    fix: { fail: "Re-rig. Rigs made now give every copy of a point the same bones." },
  },
  folded: { why: "A lot of weight was on root or tip bones, which don't move parts well.", did: "Moved it to their parent bones." },
  clips: { why: "These clips don't exist for this body plan." },
  facing: {
    why: "3D models don't all face the same way. A skeleton laid on backwards ties the head to tail bones and the feet to wing bones.",
    did: (c) => (c.yaw ? `Turned the skeleton ${c.yaw}° to face the way the model does.` : ""),
  },
  match: {
    why: "The skeleton for this body plan is built for a different shape, so it had to be stretched more one way than another to fit. Its clips may not suit this model.",
    did: "Fitted each leg, neck, tail and wing chain to the model's own parts.",
    fix: "If the body plan is wrong, pick another next to Re-rig.",
  },
  pose: {
    why: "Bones have to sit inside the body. Bones outside it pull the wrong parts, which the skinning step can't fix.",
    fix: { fail: "Pick another body plan next to Re-rig. This model may not fit any of our skeletons yet." },
  },
  pred: {
    why: "A skeleton predicted for this model alone shows where its legs, neck and tail really are. Our skeleton was laid onto it, so its clips still work.",
  },
  far: {
    why: "Points should move with the bones near them. Points pulled by far-away bones fling across the model when it moves, even with no tears.",
    fix: { fail: "Re-rig, or pick another body plan next to Re-rig." },
  },
  stretch: {
    why: "Edges that grow or shrink a lot smear the texture. Folded knees and elbows always squash some; above 6× it shows clearly.",
  },
  floor: {
    why: "Feet should touch the floor, and lying clips should rest on it rather than sink or float.",
    did: "Moved the body and feet to the floor in each clip.",
  },
  unsupported: {
    why: "These clips need parts this model doesn't use that way, like a flap for a bird that stands. They would throw the model into the wrong shape.",
    did: "Left them out of the rig and the FBX.",
  },
  motion: {
    why: "A tear only shows when the model moves, so every clip is played and measured.",
    fix: { fail: "Re-rig to fix it." },
  },
  template: {
    why: "A skeleton built for this model sits on its own legs, neck, tail and wings, so its clips are made for its shape instead of borrowed from another animal's.",
    did: { pass: "Built the skeleton from the model's predicted joints and made every clip for it.", warn: "Used the body plan's template instead." },
    fix: { fail: "Re-rig. If it keeps failing, pick the body plan's template under Advanced.", warn: "Re-rig with Skeleton: build for the model under Advanced to see why." },
  },
  built: {
    why: "Every joint of a built skeleton has to sit inside the body, both sides have to match, and the feet have to stand on the model's feet.",
    fix: { fail: "Re-rig. A model whose head doesn't point to +Z (the picture's right) can't be built; redraw it side-on, head right." },
  },
  a2: {
    why: "Weight on a bone far from the point it moves flings that point across the model in motion.",
    fix: { warn: "Re-rig, or try another skin-weight binder under Advanced." },
  },
  foot: {
    why: "The soles should follow the leg bones. A sole pulled by the body slides and sinks when the legs step.",
  },
  parts: {
    why: "Each part (neck, legs, wings) should mostly follow its own bones; a part that leaks weight to the body bends with it instead.",
    fix: { warn: "Re-rig, or try another skin-weight binder under Advanced." },
  },
  cshape: {
    why: "A clip shouldn't grow the body's box much beyond its rest size; when it does, the body is being stretched rather than moved.",
  },
  slide: {
    why: "A foot that is planted should stay put on the floor. A sliding foot looks like ice-skating.",
    did: (_c, doc) => (doc?.template?.built ? "Measured on the clips' own planted frames." : "Planted frames are guessed for these library clips, so this only warns."),
  },
  loop_seam: {
    why: "A looping clip's last frame should match its first, or the loop jumps once per cycle. (Tears at corners are under Skin.)",
  },
  fbx: {
    why: "The FBX is read back into Blender after export, to prove s&box will see every clip, every bone and at most 4 weights per point.",
    fix: { fail: "Re-rig. If it keeps failing, export the rigged GLB instead." },
  },
};

export function helpText(h: Help | undefined, c: RigCheck, doc: RigInspectDoc | null): string {
  return (typeof h === "function" ? h(c, doc) : h && typeof h === "object" ? h[c.level] : h) || "";
}

export const MOTION_HELP = "Every clip was played on this rig and measured. Tears: points whose corners come apart. Stretch: how far the worst 1% of edges grow or shrink; low-poly knees always change some.";
// The worst 1% of edges, which the checks use; older measurements only have the single worst edge.
export const stretchOf = (c: { stretch_p99?: number; stretch_max?: number }) => Number(c.stretch_p99 ?? c.stretch_max);
export const tidy = (n: unknown) => String(n).replace(/_/g, " ");
export const cap = (x: string) => (x ? x[0].toUpperCase() + x.slice(1) : x);
export const pct = (x: number) => `${Math.round(x * 100)}%`;
export const deg = (x: number) => `${x < -0.5 ? "−" : "+"}${Math.abs(Math.round(x))}°`;
const num = (re: RegExp, s: unknown) => (String(s).match(re) || [])[1] || "";

/** A check with its step and short label filled in (every row the panel shows). */
export type ShownCheck = RigCheck & { step: string; label: string };

/**
 * inspect.json's checks with a step and a short label each. Files from before v2 have neither: the step comes from
 * the key, the label from the sentence. Tip bones (ears, tails' last bone) never count as a fit problem.
 */
export function rigChecks(doc: RigInspectDoc | null | undefined): ShownCheck[] {
  const v1 = (c: RigCheck) => (c.key === "seam" && c.level === "fail" ? `Tears at ${num(/(\d+)/, c.text)} points`
    : c.key === "head" && c.level !== "pass" ? `Head ${num(/(\d+)\s*°/, c.text)}° off its bone` : String(c.text));
  return (doc?.checks || [])
    .filter((c) => !(c.key === "fit" && NO_FIT.test(c.bone || String(c.text).split(" ")[0])))
    .map((c) => ({ ...c, step: c.step || KEY_STEP[c.key] || "skeleton", label: c.label || v1(c) }));
}

// Worst first: fail before warn, tears before anything else at that level (they are what you see), then the
// rig's step order.
const TEARS = ["seam", "motion"];
export function worstFirst(a: Pick<RigCheck, "level" | "key" | "step">, b: Pick<RigCheck, "level" | "key" | "step">): number {
  const st: string[] = STEPS.map((s) => s[0]);
  const t = (c: Pick<RigCheck, "key">) => (TEARS.includes(c.key) ? 0 : 1);
  return ORDER[a.level] - ORDER[b.level] || t(a) - t(b) || st.indexOf(a.step || KEY_STEP[a.key]) - st.indexOf(b.step || KEY_STEP[b.key]);
}

/** The finding the Rig stage leads with. */
export function worstCheck(doc: RigInspectDoc | null | undefined): ShownCheck | null {
  return rigChecks(doc).sort(worstFirst)[0] || null;
}

/* ------------------------------------------------------------------ the panel's static copy */

export const INSPECT_COPY = {
  label: "Rig check",
  checks: "Checks",
  checksTitle: "What the rigger decided and how it measured.",
  sheet: "Sheet",
  sheetTitle: "Every clip rendered on the CPU from fixed cameras: the contact sheet.",
  howRigged: "How it was rigged",
  // The clip table's heading: "Clips", so it does not repeat the "In motion" check group above it.
  inMotion: "Clips",
  notMeasured: "Not measured for this version.",
  checking: "Checking…",
  measure: "Measure",
  measureTitle: "Play every clip on this rig and measure it. Takes a few seconds.",
  clip: "Clip",
  tears: "Tears",
  stretch: "Stretch",
  play: "Play",
  colour: "Colour",
  show: "Show",
  skeleton: "Skeleton",
  skeletonTitle: "Draw the bones and their names on top of the model.",
  original: "Original",
  originalTitle: "Your model as it was made, as a wireframe over the rig at rest.",
  findBone: "Find a bone",
  pickBone: "Pick a bone to see what it does.",
  // The planner's own section; the check group above is "Body plan" too.
  bodyPlan: "Body plan picture",
  chosen: "Chosen",
  howSure: "How sure",
  notRecorded: "Not recorded",
  byHand: "Set by hand",
  skeletonFit: "Skeleton fit",
  riggerDid: "What the rigger did",
  showMe: "Show me",
  why: "Why:",
  did: "What we did:",
  fix: "To fix:",
  stretchP99: "Stretch: how much the worst 1% of edges grow or shrink. Lower is better; above 3× it starts to show, above 6× it fails.",
  stretchMax: "Stretch: how much the edge that changes most grows or shrinks. Lower is better; knees and elbows always stretch some.",
  fewerClips: "Fewer clips",
  sameModel: "same model",
  otherModel: "made from a different model",
  same: "same",
} as const;

/** Look from a view: "Look from the side." (rig-inspect.js 917). */
export const viewTitle = (label: string) => `Look from the ${label.toLowerCase()}.`;

/** The paused line: "Paused at 1.2 s in Walk · " then a Play link. */
export const pausedText = (clip: string, t: number) => `Paused at ${t.toFixed(1)} s in ${tidy(clip)} · `;

/** The In motion table's clip button title. */
export const momentTitle = (clip: string, tears: number) => (tears ? `Pause ${tidy(clip)} where it tears most` : `Pause ${tidy(clip)} at its worst stretch`);

/** "Bones · 41". */
export const bonesTitle = (n: number) => `Bones · ${n}`;

/* ------------------------------------------------------------------ badge (built template) */

/** The plan a built template was made from, in words ("" when none). */
export function planName(doc: RigInspectDoc): string {
  const t = doc.template || {};
  return PLANS[t.plan || ""] || PLANS[doc.rig?.plan || ""] || t.plan || "";
}

/** The built-template badge, or null (rig-inspect.js 897). */
export function builtBadge(doc: RigInspectDoc): { text: string; title: string } | null {
  if (!doc.template?.built) return null;
  const name = planName(doc);
  return {
    text: `Skeleton built for this model${name ? ` · ${name}` : ""}`,
    title: `rig/template/: a skeleton and clips made for this model${name ? ` from the ${name.toLowerCase()} body plan` : ""}.`,
  };
}

/* ------------------------------------------------------------------ check rows */

export interface CheckBody {
  text: string;
  why: string;
  did: string;
  fix: string;
  note: string;
}

const same = (a: string, b: string) => a.replace(/\W+/g, " ").trim().toLowerCase() === b.replace(/\W+/g, " ").trim().toLowerCase();

/** An open row's lines (rig-inspect.js 779-796): the sentence when it differs from the label, Why unless it repeats it. */
export function checkBody(c: ShownCheck, doc: RigInspectDoc | null): CheckBody {
  const h = CHECK_HELP[c.key] || {};
  const [why, did, fix, note] = (["why", "did", "fix", "note"] as const).map((k) => helpText(h[k], c, doc));
  return {
    text: c.text !== c.label ? String(c.text ?? "") : "",
    why: why && !same(why, c.text || "") ? why : "",
    did,
    fix,
    note,
  };
}

/** The checks grouped under their steps, in STEPS order; empty steps left out. */
export function stepGroups(doc: RigInspectDoc): { id: string; title: string; checks: ShownCheck[] }[] {
  const checks = rigChecks(doc);
  return STEPS.map(([id, title]) => ({ id, title, checks: checks.filter((c) => c.step === id) as ShownCheck[] })).filter((g) => g.checks.length);
}

/* ------------------------------------------------------------------ In motion */

/** The notes under the In motion table (rig-inspect.js 809-826). `splitPoints` is the mesh's own count, when bound. */
export function motionNotes(doc: RigInspectDoc, splitPoints = 0): string[] {
  const clips = Object.entries(doc.motion?.clips || {});
  if (!clips.length) return [];
  const lines: string[] = [];
  // Tears at rest and tears in a clip are counted differently; say how they relate.
  const seam = rigChecks(doc).find((c) => c.key === "seam" && c.level === "fail");
  const most = clips.reduce((a, b) => (b[1].tears > a[1].tears ? b : a));
  if (seam && most[1].tears) lines.push(`${seam.points ?? splitPoints} points can tear. How many come apart depends on the move: up to ${most[1].tears} in ${tidy(most[0])}.`);
  // Stretch that sits on a bone the Skeleton step flagged.
  const fit = rigChecks(doc).find((c) => c.key === "fit" && c.level !== "pass" && c.bone);
  const on = fit ? clips.filter(([, c]) => !c.tears && c.worst_bone === fit.bone).map(([n]) => tidy(n)) : [];
  if (fit && on.length) {
    const list = on.length > 1 ? `${on.slice(0, -1).join(", ")} and ${on[on.length - 1]}` : on[0];
    lines.push(`The worst stretch in ${list} is at the ${doc.bones.find((b) => b.name === fit.bone)?.label || tidy(fit.bone)}, the bone flagged under Skeleton.`);
  }
  lines.push(clips.some(([, c]) => c.stretch_p99 != null) ? INSPECT_COPY.stretchP99 : INSPECT_COPY.stretchMax);
  return lines;
}

/* ------------------------------------------------------------------ Since version N */

export interface RigCompare {
  n: number;
  id: string;
  summary: RigCheckSummary | null;
  current: RigCheckSummary | null;
  sameModel: boolean;
}

/** Lower is better: which side reads better, and the arrow on the new one. */
export interface PairCell {
  a: string;
  b: string;
  aTone: "better" | "worse" | "";
  bTone: "better" | "worse" | "";
  mark: "" | " ▼" | " ▲";
}

export function pair(a: number, b: number, as: string = String(a), bs: string = String(b), arrow = true): PairCell {
  const cls = (x: number, y: number) => (x < y ? "better" : x > y ? "worse" : "") as PairCell["aTone"];
  const mark = !arrow ? "" : b < a ? " ▼" : b > a ? " ▲" : "";
  return { a: as, b: bs, aTone: cls(a, b), bTone: cls(b, a), mark };
}

/** The head in words, and the number compared (rig-inspect.js 830-838). */
export function headWords(h: HeadSummary | null | undefined): [string, number] {
  if (!h) return ["not measured", 0];
  const n = (x: number | undefined) => Math.abs(Math.round(x || 0));
  if (h.straightened) return [`straightened ${n(h.before)}°`, n(h.off_deg ?? h.after)];
  if (h.made_before && h.off_deg != null && n(h.off_deg) > 8) return [`${n(h.off_deg)}° off`, n(h.off_deg)];
  if (h.before != null && n(h.before) > 8) return [`kept turned ${n(h.before)}°`, n(h.off_deg ?? h.before)];
  if (h.off_deg != null && n(h.off_deg) > 8) return [`${n(h.off_deg)}° off`, n(h.off_deg)];
  return ["straight", n(h.off_deg)];
}

export interface CompareView {
  heading: string;
  back: string;
  takeId: string;
  /** "Version N was made before rig checks." when the previous one has no summary. */
  before: string;
  rows: { label: string; value: string | PairCell }[];
  /** The "All N clips" / "Fewer clips" toggle, or "" when two or fewer clips changed. */
  more: string;
}

/** The compare block (rig-inspect.js 840-874), or null when the picked version is the first. */
export function compareView(c: RigCompare | null, doc: RigInspectDoc, allClips: boolean): CompareView | null {
  if (!c) return null;
  const base = { heading: `Since version ${c.n}`, back: `Go back to version ${c.n}`, takeId: c.id };
  if (!c.summary) return { ...base, before: `Version ${c.n} was made before rig checks.`, rows: [], more: "" };
  const a = c.summary;
  const b = c.current || doc.summary || {};
  const rows: CompareView["rows"] = [];
  if (typeof a.tears === "number" && typeof b.tears === "number") rows.push({ label: "Tears", value: pair(a.tears, b.tears) });
  if (a.head || b.head) {
    const [aw, ar] = headWords(a.head);
    const [bw, br] = headWords(b.head);
    rows.push({ label: "Head", value: aw === bw ? INSPECT_COPY.same : pair(ar, br, aw, bw, false) });
  }
  // Stretch per clip: the two that changed most, the rest behind a toggle. The same figure as the In motion table
  // (1% of edges) when both versions have it; older summaries only kept the single worst edge.
  const p99 = Object.entries(b.motion || {}).every(([n, m]) => m.stretch_p99 != null && (!a.motion || !a.motion[n] || a.motion[n].stretch_p99 != null));
  const sOf = (m: { stretch_p99?: number; stretch_max?: number }) => Number(p99 ? m.stretch_p99 : m.stretch_max);
  const both = Object.entries(b.motion || {})
    .filter(([n]) => a.motion && a.motion[n])
    .map(([n, m]) => [n, sOf(a.motion![n]), sOf(m)] as [string, number, number])
    .sort((x, y) => Math.abs(Math.log(y[2] / y[1])) - Math.abs(Math.log(x[2] / x[1])));
  const shown = allClips ? both : both.slice(0, 2);
  for (const [n, o, m] of shown) rows.push({ label: `${tidy(n)} stretch${p99 ? " (1% of edges)" : " (worst edge)"}`, value: pair(o, m, `${o.toFixed(1)}×`, `${m.toFixed(1)}×`) });
  const more = both.length > 2 ? (allClips ? INSPECT_COPY.fewerClips : `All ${both.length} clips`) : "";
  if (a.class && b.class) rows.push({ label: "Body plan", value: a.class === b.class ? INSPECT_COPY.same : `${PLANS[a.class] || a.class} → ${PLANS[b.class] || b.class}` });
  rows.push({ label: "Model", value: c.sameModel ? INSPECT_COPY.sameModel : INSPECT_COPY.otherModel });
  return { ...base, before: "", rows, more };
}

/* ------------------------------------------------------------------ legend, bone card, plan, fit */

export type LegendTone = "" | "warn" | "fail";

/**
 * The colour legend (rig-inspect.js 555-571) as parts: `lead` is bold or toned, `rest` plain. `bone` is the picked
 * bone's nice name ("" for none); `dropped` says whether this version recorded where weight was dropped.
 */
export function legendParts(mode: RigMode, o: { bone: string; doc: RigInspectDoc; dropped: boolean; splitPoints: number }): { lead: string; leadTone: LegendTone | "bold"; rest: string; ramp: boolean } {
  const w = o.doc.weights || {};
  const plain = (rest: string) => ({ lead: "", leadTone: "" as const, rest, ramp: false });
  if (mode === "heat") {
    return o.bone ? { lead: o.bone, leadTone: "bold", rest: ": red pulls fully, blue a little, dark blue not at all.", ramp: true }
      : { lead: "Pick a bone in the list or click the model.", leadTone: "warn", rest: "", ramp: false };
  }
  if (mode === "bones") return plain(`Each part takes the colour of the bone that moves it most.${o.bone ? ` ${o.bone} is lit, the rest dimmed.` : ""}`);
  if (mode === "dropped") {
    const total = w.dropped?.total;
    if (!o.dropped) return plain(total ? `${total} weights were dropped, but this version did not record where. Re-rig to see them.` : "Nothing was dropped.");
    return plain(`Yellow to red: weight thrown away because the bone was too far. ${total} in all.`);
  }
  if (mode === "split") {
    const raw = w.seam_raw?.positions_with_conflicting_top_bone;
    const before = raw != null && raw !== o.splitPoints ? ` Straight from the skinner it was ${raw}.` : "";
    return o.splitPoints ? { lead: `${o.splitPoints} points have corners on different bones.`, leadTone: "fail", rest: ` They tear when the model moves. Re-rig to fix.${before}`, ramp: false }
      : plain(`No split points. Corners at the same point move together.${before}`);
  }
  return plain("The painted model.");
}

export interface CardRow {
  label: string;
  value: string;
  tone?: LegendTone;
}

/** The picked bone's card rows (rig-inspect.js 573-593). `nice` maps a raw bone name to its words. */
export function boneCardRows(o: { name: string; doc: RigInspectDoc; owned: number; clip: string | null; nice: (raw: string) => string }): CardRow[] {
  const { doc, name } = o;
  const b = doc.bones.find((x) => x.name === name) || ({} as Partial<RigBone>);
  const k = doc.bones.indexOf(b as RigBone);
  const f = (doc.fit || []).find((x) => x.bone === name);
  const c = o.clip && doc.clips ? doc.clips[o.clip] : undefined;
  const peak = c && k >= 0 ? c.peak_deg[k] : null;
  const pb = b.parent != null && b.parent >= 0 ? doc.bones[b.parent] : undefined;
  const parent = pb ? o.nice(pb.name) || pb.name : "";
  const off = f ? f.centroid_off_bone_rel_h : null;
  const rows: CardRow[] = [];
  if (parent) rows.push({ label: "Hangs from", value: parent });
  rows.push(b.template ? { label: "Driven by", value: `template bone ${b.template}` } : { label: "Driven by", value: b.rule || "nothing", tone: "warn" });
  if (b.ref_vs_rest_deg != null) rows.push({ label: "Turned in the start pose", value: `${b.ref_vs_rest_deg}°` });
  rows.push({ label: "Points it moves most", value: String(o.owned) });
  if (f) rows.push({ label: "Usual distance to its points", value: `${pct(f.median_dist_rel_h)} of height` });
  if (off != null) rows.push({ label: "Its part's centre is off by", value: `${pct(off)} of height`, tone: off > 0.12 ? "warn" : "" });
  if (peak != null && o.clip) rows.push({ label: `Biggest turn in ${tidy(o.clip)}`, value: `${peak}°` });
  return rows;
}

/** The Body plan table: Chosen and How sure (rig-inspect.js 925-931). */
export function planRows(doc: RigInspectDoc): CardRow[] {
  const pl = doc.planner || {};
  const sure = pl.confidence != null ? pct(pl.confidence) : pl.unrecorded ? INSPECT_COPY.notRecorded : INSPECT_COPY.byHand;
  return [
    { label: INSPECT_COPY.chosen, value: PLANS[pl.choice || ""] || pl.choice || PLANS[doc.rig?.class || ""] || "?" },
    { label: INSPECT_COPY.howSure, value: sure, tone: pl.confidence != null && pl.confidence < 0.6 ? "warn" : "" },
  ];
}

/** The Skeleton fit rows (rig-inspect.js 933-941). Stick colours are drawn by the panel from `roles`. */
export function fitRows(doc: RigInspectDoc): CardRow[] {
  const t = doc.template || {};
  const fmt = (x: unknown, n = 3) => (typeof x === "number" ? x.toFixed(n) : "?");
  const name = planName(doc);
  const rows: CardRow[] = [
    { label: "Template", value: t.built ? `Built for this model${name ? ` (${name})` : ""}` : String(t.rig || "?").split("/").pop()!.replace(/\.glb$/, "") },
    { label: "Start pose", value: `${t.ref_clip || "?"}, first frame` },
    { label: "Scale", value: fmt(t.fit_scale) },
    { label: "Offset (m)", value: (t.fit_offset || []).map((x) => fmt(x)).join(", ") },
    { label: "Joints moved by skinning", value: `${fmt(t.max_joint_move_by_skinner_m, 2)} m` },
  ];
  if (doc.head) {
    const off = Math.abs(doc.head.off_deg);
    rows.push({ label: "Head vs head bone", value: `${off.toFixed(0)}° apart`, tone: off > 15 ? "fail" : off > 6 ? "warn" : "" });
  }
  return rows;
}

/** The roles shown as stick colours (root left out), in first-seen order; [] hides the row. */
export function stickRoles(roles: string[]): string[] {
  return roles.some(Boolean) ? [...new Set(roles.filter((r) => r && r !== "root"))] : [];
}

/** A role's stick colour as #rrggbb (unknown roles get "other"'s cream). */
export function roleHex(role: string): string {
  const rgb = ROLE_RGB[role];
  if (!rgb) return "#f0eadc";
  return `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

/* ------------------------------------------------------------------ head (Rest pose row) */

export const HEAD_SEGMENTS = [
  ["rigged", "Rigged", "The rig as it moves."],
  ["original", "Original", "Your model as it was made."],
  ["both", "Both", "Your model as a wireframe over the rig."],
] as const;
export type HeadMode = (typeof HEAD_SEGMENTS)[number][0];

export const RIG_AS_MADE = { label: "Rig it as made", title: "Rigs again with the head as the model has it. This version stays in Versions.", straighten: "off" } as const;
export const STRAIGHTEN_AND_RIG = { label: "Straighten and rig", title: "Rigs again with the head straightened. This version stays in Versions.", straighten: "auto" } as const;
export const NEVER_CHANGED = "Your model file is never changed.";

/** The Rest pose row's head block (rig-inspect.js 654-669), or null when the rest pose was not measured. */
export function headBlock(rest: RestInfo | null | undefined): { read: string; segments: boolean; button: typeof RIG_AS_MADE | typeof STRAIGHTEN_AND_RIG | null; kept: string } | null {
  if (!rest || !rest.measured) return null;
  const m = rest.measured;
  const st = rest.straightened;
  const read = st ? `Head turned ${deg(m.head_yaw_deg)} → ${deg(st.head_yaw_deg)}`
    : m.turned && rest.before_straightening ? `Head turned ${deg(m.head_yaw_deg)} when it was made`
      : m.turned ? `Head turned ${deg(m.head_yaw_deg)}, kept as made` : `Head straight (${Math.abs(Math.round(m.head_yaw_deg))}°)`;
  return { read, segments: !!st, button: st ? RIG_AS_MADE : m.turned ? STRAIGHTEN_AND_RIG : null, kept: st ? NEVER_CHANGED : "" };
}

/* ------------------------------------------------------------------ Show me */

/**
 * Whether a check has a Show me (rig-inspect.js 762-768). `names` are the skin's bone names; `headUI` is true when
 * the original model can be shown; `dropped` when this version recorded where weight was dropped.
 */
export function canShow(c: RigCheck, o: { doc: RigInspectDoc; names: string[]; headUI: boolean; dropped: boolean }): boolean {
  const { doc, names } = o;
  const rest = doc.rig?.rest;
  const keys = ["planner", "rest", "head", "fit", "moved", "seam", "folded", "motion", "stretch", "far", "floor", "pose", "facing", "pred", "template",
    "built", "a2", "foot", "parts", "cshape", "slide", "loop_seam", "fbx"];
  return keys.includes(c.key)
    && (!["cshape", "slide", "loop_seam"].includes(c.key) || Object.keys(c.clips || {}).length > 0)
    && (c.key !== "fbx" || !!doc.motion)
    && (c.key !== "rest" || (rest?.straightened ? o.headUI : !!(rest?.measured?.turned && names.includes("head"))))
    && (!["head", "fit"].includes(c.key) || names.includes(c.bone || "head"))
    && (c.key !== "motion" || (c.level === "fail" ? !!c.clip : !!doc.motion))
    && (c.key !== "folded" || o.dropped);
}

/** A per-clip check's clip to pause (its `over` list first, else its worst value), or "" (rig-inspect.js 744-753). */
export function worstClip(c: RigCheck): string {
  const vals = Object.entries(c.clips || {}).map(([n, v]) => [n, Array.isArray(v) ? Math.max(...v) : Number(v)] as [string, number]);
  return (c.over && c.over[0]) || (vals.sort((a, b) => b[1] - a[1])[0] || [])[0] || "";
}

/** The side a bone is seen from: a right-hand bone from the right. */
export const sideOf = (n: string | undefined): RigViewId | [number, number, number] => (/(^|_)r$|right/i.test(n || "") ? [-1, 0, 0] : "side");

/* ------------------------------------------------------------------ bones list */

/** The bones in words for the panel, labels and hover card: inspect.json's label or the tidy name, numbered when two share one. */
export function niceNames(names: string[], bones: RigBone[]): string[] {
  const lab = names.map((n) => cap(bones.find((b) => b.name === n)?.label || tidy(n)));
  const seen: Record<string, number> = {};
  for (const l of lab) seen[l] = (seen[l] || 0) + 1;
  return lab.map((l, i) => (seen[l] > 1 ? `${l} ${(names[i].match(/(\d+)/) || [])[1] || ""}`.trim() : l));
}

/** How deep a bone hangs (for the list's indent), capped like legacy. */
export function boneDepth(bones: RigBone[], i: number): number {
  let d = 0;
  let p = bones[i]?.parent;
  while (p != null && p >= 0 && d < 12) {
    d++;
    p = bones[p]?.parent;
  }
  return d;
}

/** The bone filter: the raw name (case-sensitive) or the words (not). */
export function boneFound(filter: string, raw: string, nice: string): boolean {
  return !filter || raw.includes(filter) || nice.toLowerCase().includes(filter.toLowerCase());
}

/* ------------------------------------------------------------------ Sheet tab */

/** rig/preview/contact_sheet.png (rig_render.py): Rest then every clip; columns side, front and three-quarter. */
export const SHEET_FILE = "rig/preview/contact_sheet.png";

export const SHEET_COPY = {
  looking: "Looking for the contact sheet…",
  none: "No contact sheet for this rig yet. Rigs made now render one; this takes under a minute on the CPU.",
  openTitle: "Open the sheet full size",
  alt: "Contact sheet: every clip of this rig",
  caption: "Rows: the rest pose, then each clip at its biggest move. Columns: side, front, three-quarter. The red line is the floor.",
  rendering: "Rendering…",
  again: "Render again",
  render: "Render the sheet",
} as const;

export interface SheetState {
  stem: string;
  /** null: not looked for yet. */
  ok: boolean | null;
  v: number;
  busy: boolean;
  error: string;
}

export const SHEET0: SheetState = { stem: "", ok: null, v: 0, busy: false, error: "" };

/** What the Sheet tab shows (rig-inspect.js 959-966). The "Looking…" branch has no button, as legacy. */
export function sheetView(s: SheetState, stem: string): { kind: "looking" | "none" | "found"; button: string } {
  const button = s.busy ? SHEET_COPY.rendering : s.ok ? SHEET_COPY.again : SHEET_COPY.render;
  if (s.ok === null || s.stem !== stem) return { kind: "looking", button: "" };
  return { kind: s.ok ? "found" : "none", button };
}

/* ------------------------------------------------------------------ hover card */

/** The hover card's words (rig-inspect.js 436-449). */
export const tipTitle = (v: number) => `Point ${v}`;
export const tipDropped = (w: number) => `${w.toFixed(2)} weight dropped`;
export const tipSplit = (x: number) => `Corners disagree by ${pct(x)}`;
export const tipPick = (bone: string) => `Click to pick ${bone}`;
