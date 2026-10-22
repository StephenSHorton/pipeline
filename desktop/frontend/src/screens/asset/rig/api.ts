// The Rig stage's factory calls and its one store (legacy loadRigPlans, loadRigCheck, recheckRig's POST, the
// sheet HEAD; index.html 3041-3050, 3100-3125, 3214-3234; rig-inspect.js 951-958). Module-level, so the finish
// seam (./finish.ts) can leave a check to open on ("Inspect") and the bar and the panel read the same checks.
import { create } from "zustand";
import { factoryJson, headOut, readOutJson, type RigDoc, type RigPlan } from "@/lib/api";
import { SHEET0, SHEET_FILE, type RestInfo, type RigInspectDoc, type SheetState } from "./inspect";
import {
  choicesOf,
  CLIPS_REPORT,
  inspectPath,
  mergeRigInfo,
  PLANS_PATH,
  rigMadeFrom,
  templateBuilt,
  validInspect,
  type ClipReport,
  type RigChoices,
  type RigMade,
  choicesFor,
} from "./rig";

type RigInfo = RigDoc & { rest?: RestInfo | null };

/** One asset's checks: inspect.json (null when this version has none), rig.json, and a built template's clips. */
export interface RigCheckState {
  stem: string;
  doc: RigInspectDoc | null;
  info: RigInfo | null;
  made: RigMade | null;
}

interface RigStore {
  /** Body plan and Advanced, for `choices.stem` only (choicesOf resets them for another asset). */
  choices: RigChoices;
  /** GET /rig/plans: null until asked, [] when the factory has none (or failed). */
  plans: RigPlan[] | null;
  check: RigCheckState | null;
  /** Check this version / Measure is running for this stem. */
  checking: string;
  /** A check to open the Rig check on once it loads (the finish toast's Inspect, a recheck with the panel open). */
  focus: { stem: string; key: string } | null;
  sheet: SheetState;
}

export const useRig = create<RigStore>(() => ({
  choices: choicesFor(""),
  plans: null,
  check: null,
  checking: "",
  focus: null,
  sheet: { ...SHEET0 },
}));

/** Change this asset's body plan / Advanced choices (a fresh set when the held ones were another asset's). */
export function setChoices(stem: string, patch: Partial<Omit<RigChoices, "stem">>): void {
  const held = choicesOf(useRig.getState().choices, stem);
  useRig.setState({ choices: { ...held, ...patch, stem } });
}

let plansAsked = false;
/** GET /rig/plans, once per session (legacy loadRigPlans); an error leaves [] and the RIG_NAMES fallback. */
export async function loadRigPlans(get: () => Promise<{ plans?: RigPlan[] }> = () => factoryJson(PLANS_PATH)): Promise<void> {
  if (plansAsked) return;
  plansAsked = true;
  try {
    useRig.setState({ plans: (await get()).plans || [] });
  } catch {
    useRig.setState({ plans: [] });
  }
}

/** For tests: forget that the plans were asked. */
export function resetRigPlans(): void {
  plansAsked = false;
  useRig.setState({ plans: null });
}

let checkSeq = 0;
/**
 * Read the checks for the rig version on disk (legacy loadRigCheck): inspect.json and rig.json, then a built
 * template's clips_report.json. Only the newest read lands, and only while `stem` is still the asset asked for.
 */
export async function loadRigCheck(stem: string, still: () => boolean = () => true): Promise<RigCheckState | null> {
  const seq = ++checkSeq;
  const t = Date.now();
  const [doc, info] = await Promise.all([readOutJson<unknown>(stem, "rig/inspect.json", t), readOutJson<RigInfo>(stem, "rig/rig.json", t)]);
  const report = templateBuilt(info) ? await readOutJson<{ clips?: Record<string, ClipReport> }>(stem, CLIPS_REPORT, t) : null;
  if (seq !== checkSeq || !still()) return null;
  const ok = validInspect(doc);
  const state: RigCheckState = { stem, doc: ok ? mergeRigInfo(doc, info) : null, info, made: rigMadeFrom(stem, info, report) };
  useRig.setState({ check: state });
  return state;
}

/** POST /rig/<stem>/inspect "{}": the checks again for the picked version, in place (the answer is not used, as legacy). */
export async function postRecheck(stem: string): Promise<void> {
  await factoryJson(inspectPath(stem), { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
}

/** HEAD out/<stem>/rig/preview/contact_sheet.png (legacy checkSheet); the Sheet tab shows what it finds. */
export async function checkSheet(stem: string, head: (stem: string, rel: string) => Promise<boolean> = headOut): Promise<void> {
  useRig.setState((s) => ({ sheet: { ...s.sheet, stem, ok: false } }));
  const ok = await head(stem, SHEET_FILE);
  if (useRig.getState().sheet.stem !== stem) return;
  useRig.setState((s) => ({ sheet: { ...s.sheet, ok, v: Date.now() } }));
}

/** A new asset's Sheet tab starts over (legacy open(): sheet reset when the stem changed). */
export function sheetFor(stem: string): void {
  if (useRig.getState().sheet.stem !== stem) useRig.setState({ sheet: { ...SHEET0 } });
}
