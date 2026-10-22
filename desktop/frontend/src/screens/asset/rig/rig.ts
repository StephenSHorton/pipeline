// The Rig stage's own controls, pure (legacy renderFlow's rig branch, index.html 2914-2999, and its helpers
// 3034-3210): the status button, Check this version, the built badge, the body plan select (GET /rig/plans), the
// Advanced fold, Rig it / Re-rig and the body POST /rig/<stem> gets; plus the finish toast's words (finishJob,
// 6560-6606). Fixes two legacy bugs: the body plan and Advanced choices are per asset (legacy rigClassChoice and
// rigAdvanced carried over to the next asset, 3037, 3060), and every re-rig sends the plan the select shows (legacy
// panel re-rigs sent rigClassChoice, "" until you picked, while the bar sent pick.value, 2997 vs 4785). No React here.
import { enc, outUrl, type RigDoc, type RigPlan } from "@/lib/api";
import type { FlowEntry, FlowTake } from "@/lib/flow";
import { RIG_NAMES } from "@/screens/asset/export/export";
import { rigChecks, worstCheck, worstFirst, type RestInfo, type RigCompare, type RigInspectDoc } from "./inspect";

export { RIG_NAMES };

export const RIG_COPY = {
  noMesh: "Make the model first, then rig it here.",
  clips: "Clips",
  pass: "Rig looks right",
  passTitle: "Every check passed. Click to see how it was rigged.",
  problemsTail: "Click to see it on the model.",
  checking: "Checking…",
  checkVersion: "Check this version",
  checkAgainTitle: "This version was checked before its clips were measured. Check it again and play every clip. Takes a few seconds.",
  checkFirstTitle: "Work out the rig checks for this version and measure its clips. Takes a few seconds.",
  madeBadge: "Clips made for this model",
  planTitle: "The body plan decides which skeleton the model gets, and which clips.",
  advancedTitle: "Skeleton, skin weights and predicted skeleton for the next Re-rig.",
  rigIt: "Rig it",
  reRig: "Re-rig",
} as const;

/** GET /rig/plans (legacy loadRigPlans: factoryJson("/rig/plans")). */
export const PLANS_PATH = "/rig/plans";
/** POST /rig/<stem>: the rig job. */
export const rigPath = (stem: string) => `/rig/${enc(stem)}`;
/** POST /rig/<stem>/inspect: the checks again, in seconds on the CPU. */
export const inspectPath = (stem: string) => `/rig/${enc(stem)}/inspect`;
/** POST /rig/<stem>/render: the contact sheet, a job in the factory slot. */
export const renderPath = (stem: string) => `/rig/${enc(stem)}/render`;
/** Built templates: rig/template/clips_report.json (legacy loadRigCheck). */
export const CLIPS_REPORT = "rig/template/clips_report.json";

/* ------------------------------------------------------------------ built template ("made for this model") */

export interface ClipReport {
  gates?: Record<string, { value?: unknown; limit?: unknown; ok?: boolean } | unknown>;
  attempts?: unknown[];
  energy?: number | string | null;
}

/** A built template's clips (clips_report.json): each made for this model and passing its own gates. */
export interface RigMade {
  stem: string;
  plan: string;
  clips: Record<string, ClipReport>;
  failed: string[];
}

type RigInfo = RigDoc & { rest?: RestInfo | null; template?: RigDoc["template"] | string };

/** rig.json's template is {mode, rig, plan, built, dir} since built templates (a plain path before). */
export function rigMadeFrom(stem: string, info: RigInfo | null, report: { clips?: Record<string, ClipReport> } | null): RigMade | null {
  const tpl = info && info.template && typeof info.template === "object" ? info.template : null;
  if (!tpl || !tpl.built) return null;
  return { stem, plan: tpl.plan || info?.plan || "", clips: report?.clips || {}, failed: info?.clips_failed || [] };
}

/** Whether rig.json says the template was built (then clips_report.json is worth reading). */
export const templateBuilt = (info: RigInfo | null) => !!(info && info.template && typeof info.template === "object" && info.template.built);

/** rig.json is the source of truth for how the rig was made; inspect.json files from before v2 lack rest. */
export function mergeRigInfo(doc: RigInspectDoc, info: RigInfo | null): RigInspectDoc {
  const r = doc.rig || {};
  return { ...doc, rig: { ...(info || {}), ...r, rest: r.rest || info?.rest || null } as RigInspectDoc["rig"] };
}

/** inspect.json is valid when it has a bones list. */
export const validInspect = (doc: unknown): doc is RigInspectDoc => !!doc && Array.isArray((doc as RigInspectDoc).bones);

export const GATE_WORDS: Record<string, string> = { stretch_p99: "stretch", cshape: "body shape", floor_min_H: "floor", foot_slide: "foot slide",
  loop_seam_deg: "loop seam", head_hold_drift_H: "head hold", beak_gap_H: "beak to ground", belly_gap_H: "belly to ground",
  abduct_deg: "wing stroke", wing_rise_H: "wing skin rises", wing_clear_H: "wing above the back", holds: "holds the end" };

/** One clip's gates, one per line, then its energy (legacy clipGateText, index.html 3081-3091). */
export function clipGateText(r: ClipReport): string {
  const g = r.gates || {};
  const fmt = (x: unknown): string => (Array.isArray(x) ? x.map(fmt).join("–") : typeof x === "number" ? (Math.abs(x) >= 10 ? x.toFixed(0) : x.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")) : String(x));
  const bits = Object.entries(g)
    .filter(([, d]) => d && typeof d === "object" && (d as { value?: unknown }).value != null)
    .map(([k, d]) => {
      const x = d as { value: unknown; limit?: unknown; ok?: boolean };
      return `${x.ok === false ? "✗" : "✓"} ${GATE_WORDS[k] || k.replace(/_/g, " ")} ${fmt(x.value)}${x.limit != null ? ` (limit ${fmt(x.limit)})` : ""}`;
    });
  const tries = Array.isArray(r.attempts) && r.attempts.length > 1 ? ` · ${r.attempts.length} tries` : "";
  return (bits.join("\n") + (r.energy != null ? `\nenergy ${fmt(Number(r.energy))}${tries}` : "")).trim();
}

/**
 * A clip button's words on the Rig stage (index.html 2929-2948): its name, its tooltip with the tears and the
 * stretch from the checks, and " · made for this model" plus its gates on a built template.
 */
export function clipButton(name: string, doc: RigInspectDoc | null, made: RigMade | null): { text: string; title: string; tears: boolean; made: boolean } {
  const text = name.replace(/_/g, " ");
  const m = doc?.motion?.clips?.[name];
  const stretch = m ? (m.stretch_p99 != null ? `1% of edges stretch ${Number(m.stretch_p99).toFixed(1)}×` : `worst stretch ${Number(m.stretch_max).toFixed(1)}×`) : "";
  const gate = made && made.clips[name] ? clipGateText(made.clips[name]) : "";
  const madeHere = made ? ` · made for this model${gate ? `\n${gate}` : ""}` : "";
  if (m && m.tears) return { text, title: `${text} · tears at ${m.tears} points · ${stretch}${madeHere}`, tears: true, made: !!made };
  return { text, title: (m ? `Play ${text} · no tears · ${stretch}` : `Play ${text}`) + madeHere, tears: false, made: !!made };
}

/** "A skeleton was built for this model (…) and every clip passed its checks on it." */
export const madeBadgeTitle = (plan: string) => `A skeleton was built for this model (${plan || "its body plan"}) and every clip passed its checks on it.`;

/* ------------------------------------------------------------------ status button + recheck */

export interface RigStatus {
  level: string;
  key: string;
  label: string;
  title: string;
}

/** The Rig stage's one-line status: the worst finding, fail before warn, then step order (legacy rigStatus). */
export function rigStatus(doc: RigInspectDoc | null): RigStatus | null {
  const worst = worstCheck(doc);
  if (!worst) return null;
  const bad = worst.level !== "pass";
  const problems = rigChecks(doc).filter((c) => c.level !== "pass").sort(worstFirst);
  return {
    level: bad ? worst.level : "pass",
    key: bad ? worst.key : "",
    label: bad ? worst.label : RIG_COPY.pass,
    title: bad ? [...problems.map((c) => c.label), RIG_COPY.problemsTail].join("\n") : RIG_COPY.passTitle,
  };
}

/**
 * Check this version: shown on a rigged version with no checks, or checks from before clips were measured
 * ("Checking…" while it runs). `loaded` is false until inspect.json has been read for this version.
 */
export function recheckView(o: { rigged: boolean; loaded: boolean; doc: RigInspectDoc | null; checking: boolean }): { kind: "none" | "checking" | "button"; title: string } {
  if (!o.rigged || !o.loaded || !(o.doc === null || !o.doc.motion || o.checking)) return { kind: "none", title: "" };
  if (o.checking) return { kind: "checking", title: "" };
  return { kind: "button", title: o.doc ? RIG_COPY.checkAgainTitle : RIG_COPY.checkFirstTitle };
}

/** A seam or motion check at fail: Re-rig is the primary button. */
export const hasTears = (doc: RigInspectDoc | null) => rigChecks(doc).some((c) => (c.key === "seam" || c.key === "motion") && c.level === "fail");

/* ------------------------------------------------------------------ body plan + Advanced */

export type PlanOption = [value: string, label: string];

/** The select's options: auto then GET /rig/plans, or the fallback RIG_NAMES before (or without) plans. */
export function planOptions(plans: RigPlan[] | null): PlanOption[] {
  return plans && plans.length
    ? [["", RIG_NAMES[""]], ...plans.map((p) => [`plan:${p.id}`, `${p.label}${p.built ? " · built for the model" : ""}`] as PlanOption)]
    : (Object.entries(RIG_NAMES) as PlanOption[]);
}

/** Until you pick one, the plan this rig was built with (a built template), not "auto"; a value not offered shows auto. */
export function shownChoice(choice: string, made: RigMade | null, options: PlanOption[]): string {
  const shown = choice || (made && made.plan ? `plan:${made.plan}` : "");
  return options.some(([v]) => v === shown) ? shown : "";
}

/** Advanced rig options (rig.py flags); "" leaves rig.py's default. [key, label, options, help]. */
export const RIG_ADVANCED = [
  ["template", "Skeleton", [["", "auto"], ["build", "build for the model"], ["fixed", "the body plan's template"]],
    "auto builds a skeleton for the model when its body plan can (a ground bird), else uses the body plan's template."],
  ["binder", "Skin weights", [["", "skin-tokens"], ["pred", "predicted rig's"], ["heat", "Blender heat"]],
    "How the skin weights are learned."],
  ["pred", "Predicted skeleton", [["", "auto"], ["always", "always"], ["never", "never"]],
    "Use the model's own predicted skeleton as the fitting target: auto (when the match is poor), always, never."],
] as const;
export type AdvancedKey = (typeof RIG_ADVANCED)[number][0];
export type Advanced = Record<AdvancedKey, string>;
export const ADVANCED0: Advanced = { template: "", binder: "", pred: "" };

/** "Advanced" or "Advanced · 2" (how many are set). */
export function advancedSummary(a: Advanced): string {
  const n = RIG_ADVANCED.filter(([k]) => a[k]).length;
  return `Advanced${n ? ` · ${n}` : ""}`;
}

/** The choices one asset's Rig stage holds: they reset on another asset (legacy kept them for the session). */
export interface RigChoices {
  stem: string;
  choice: string;
  advanced: Advanced;
  advancedOpen: boolean;
}

export const choicesFor = (stem: string): RigChoices => ({ stem, choice: "", advanced: { ...ADVANCED0 }, advancedOpen: false });

/** The choices for `stem`: the held ones when they are this asset's, else a fresh set. */
export function choicesOf(held: RigChoices, stem: string): RigChoices {
  return held.stem === stem ? held : choicesFor(stem);
}

/* ------------------------------------------------------------------ Rig it / Re-rig */

/** Re-rig repeats how this version was made: a head kept as made stays as made unless asked otherwise. */
export function rigKeptHead(rest: RestInfo | null | undefined): boolean {
  return !!(rest && rest.measured && rest.measured.turned && !rest.straightened && !rest.before_straightening);
}

/** The Rig it / Re-rig button (index.html 2994-2998): primary when not rigged yet or the rig tears. */
export function rigButton(rigged: boolean, tears: boolean, kept: boolean): { label: string; title: string; primary: boolean } {
  return {
    label: rigged ? RIG_COPY.reRig : RIG_COPY.rigIt,
    title: `Fit a skeleton to the model, learn its skin weights and bake library clips. About 1-2 minutes.${kept ? " Head kept as made." : ""}`,
    primary: !rigged || tears,
  };
}

/**
 * The body POST /rig/<stem> gets (legacy rigAsset): {} for auto, {plan} for "plan:<id>", {class} for a class; the
 * set Advanced flags; straighten when asked, or "off" when this version kept the head as made.
 */
export function rigBody(cls: string, advanced: Advanced, o: { straighten?: string; keptHead?: boolean } = {}): Record<string, string> {
  const body: Record<string, string> = !cls ? {} : cls.startsWith("plan:") ? { plan: cls.slice(5) } : { class: cls };
  for (const [k] of RIG_ADVANCED) if (advanced[k]) body[k] = advanced[k];
  const straighten = o.straighten ? o.straighten : o.keptHead ? "off" : "";
  if (straighten) body.straighten = straighten;
  return body;
}

/** The rig job's title (chip and card). */
export const rigTitle = (name: string) => `Rigging ${name}`;

/* ------------------------------------------------------------------ versions: the one before, the original model */

/** The picked rig version, when what is on screen is that version (not unsaved). */
export function rigTakePicked(entry: FlowEntry | null): FlowTake | null {
  return entry && !entry.unsaved ? entry.takes.find((t) => t.id === entry.pick) || null : null;
}

/** The rig version before the one on screen, to say what changed since (legacy rigCompare). */
export function rigCompare(entry: FlowEntry | null, doc: RigInspectDoc | null): RigCompare | null {
  const take = rigTakePicked(entry);
  const i = take && entry ? entry.takes.indexOf(take) : -1;
  if (!entry || !take || i < 1) return null;
  const prev = entry.takes[i - 1];
  return {
    n: i,
    id: prev.id,
    summary: (prev.summary as RigCompare["summary"]) || null,
    current: ((take.summary || entry.summary || doc?.summary) as RigCompare["current"]) || null,
    sameModel: (prev.from?.mesh || "") === (take.from?.mesh || ""),
  };
}

/** The model the rig was made from: the picked mesh, or the mesh version this rig version came from (legacy originalModel). */
export function originalUrl(stem: string, info: { source?: string; rest?: RestInfo | null } | null | undefined, rigTake: FlowTake | null, meshPick: string | null | undefined): string {
  if (!info || !info.source || !info.rest || !info.rest.measured) return "";
  const from = rigTake?.from?.mesh;
  return from && from !== meshPick ? outUrl(stem, `takes/mesh/${from}/${info.source}`, false) : outUrl(stem, info.source, false);
}

/* ------------------------------------------------------------------ finish (legacy finishJob's rig branch) */

/** " · Four legs, 12 clips, FBX for s&box" from rig.json; "" when it could not be read. */
export function rigFacts(r: RigDoc | null): string {
  if (!r) return "";
  return ` · ${RIG_NAMES[r.class || ""] || r.class}, ${(r.clips || []).length} clips${r.fbx ? ", FBX for s&box" : ""}`;
}

/** The ps1 facts legacy put first: " · 1,234 triangles, 128 px texture". They stay when rig.json can't be read. */
export function psxFacts(psx: { triangles?: number; atlas?: number } | null): string {
  return psx ? ` · ${Number(psx.triangles).toLocaleString()} triangles, ${psx.atlas} px texture` : "";
}

/** "Rig ready: <name> · Four legs, 12 clips · <summary line> · Degraded: …". */
export function rigReadyText(name: string, facts: string, summaryLine: string, degraded: string): string {
  return `Rig ready: ${name}${facts}${summaryLine ? ` · ${summaryLine}` : ""}${degraded ? ` · ${degraded}` : ""}`;
}

/** The finish toast's action: Inspect a failing rig, else Export; Open when you were not on it. */
export function finishAction(o: { here: boolean; level: string | undefined }): "Inspect" | "Export →" | "Open" {
  if (!o.here) return "Open";
  return o.level === "fail" ? "Inspect" : "Export →";
}

/** The job card's error when the model would not load after a finished rig (index.html 6597). */
export const didNotLoad = (err: string) => `It finished, but the model did not load: ${err}`;
/** The toast when the job was already gone (index.html 6593). */
export const didNotLoadNamed = (name: string, err: string) => `${name} finished, but the model did not load: ${err}`;

/** The check the finish toast's Inspect opens on: the summary's worst problem. */
export function inspectKey(problems: { level?: string; key?: string; step?: string }[] | undefined): string {
  const sorted = [...(problems || [])].map((p) => ({ level: p.level || "pass", key: p.key || "", step: p.step })).sort(worstFirst);
  return sorted[0]?.key || "";
}
