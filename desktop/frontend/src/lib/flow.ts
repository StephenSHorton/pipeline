// Stages and versions of the open asset (GET /flow/<stem>, scripts/flow.py). The rail steps between stages;
// the takes strip steps back to an earlier version of the open stage. Ported from the legacy page
// (index.html 2716-2830): STAGE_VIEW, TAKE_SHOTS, OUTPUT_STAGES, refreshFlow's default stage, renderFlow's rail.
import { create } from "zustand";
import { enc, factoryJson, onFlowDirty, type AssetOutput } from "./api";

export const STAGES = ["ideas", "concept", "flat", "mesh", "rig", "stack", "export"] as const;
export type Stage = (typeof STAGES)[number];
export type Workflow = "2d" | "3d";
export type FlatMode = "picture" | "motion" | "pieces" | "say" | "sound";

export const STAGE_LABEL: Record<Stage, string> = {
  ideas: "Ideas",
  concept: "Concept",
  flat: "2D",
  mesh: "3D",
  rig: "Rig",
  stack: "Stack",
  export: "Export",
};

/** Which workflow and 2D mode a stage opens in, and its name in stale notes. */
export const STAGE_VIEW: Record<Stage, { workflow?: Workflow; mode?: FlatMode; name: string }> = {
  ideas: { workflow: "2d", mode: "picture", name: "idea" },
  concept: { workflow: "2d", mode: "picture", name: "concept" },
  flat: { workflow: "2d", mode: "pieces", name: "2D sheet" },
  mesh: { workflow: "3d", name: "3D mesh" },
  rig: { workflow: "3d", name: "rig" },
  stack: { workflow: "3d", name: "sprite stack" },
  export: { name: "export" },
};

/** A version's thumbnail: the first of its files matching these, in order. */
export const TAKE_SHOTS: Partial<Record<Stage, RegExp[]>> = {
  concept: [/^flat\/source\.png$/, /^lifted\.png$/, /^input\.(png|jpe?g|webp)$/],
  flat: [/^flat\/sheets\/.+\.png$/, /^flat\/parts\/.+\.png$/, /^flat\/source\.png$/],
  mesh: [/^preview\.png$/],
  stack: [/^stack\/preview\.png$/],
  export: [/^export\/stack\/preview\.png$/, /^export\/concept\.(png|jpe?g|webp)$/],
};

/** The stages an output needs; the others show on the rail only when they already hold something. */
export const OUTPUT_STAGES: Record<AssetOutput, Stage[]> = {
  ps1: ["concept", "mesh", "rig", "export"],
  model: ["concept", "mesh", "rig", "export"],
  stack: ["concept", "stack", "export"],
  sprite: ["concept", "flat", "export"],
};

export interface FlowSummary {
  line?: string;
  level?: "ok" | "warn" | "fail" | string;
  problems?: { key: string; [k: string]: unknown }[];
  [k: string]: unknown;
}

export interface FlowTake {
  id: string;
  made: number | string;
  note?: string;
  from?: Record<string, string>;
  files: string[];
  print?: string;
  summary?: FlowSummary;
}

export interface FlowEntry {
  stage: Stage;
  pick: string | null;
  takes: FlowTake[];
  live: boolean;
  files: string[];
  unsaved: boolean;
  stale: string[];
  upstream: string[];
  summary?: FlowSummary;
}

export interface FlowState {
  ok: boolean;
  stem: string;
  stages: FlowEntry[];
}

export const flowEntry = (flow: FlowState | null | undefined, stage: Stage): FlowEntry | null =>
  flow?.stages.find((s) => s.stage === stage) ?? null;

export const isLive = (flow: FlowState | null | undefined, stage: Stage) => !!flowEntry(flow, stage)?.live;

export function takeShot(stage: Stage, files: string[] | undefined): string {
  for (const re of TAKE_SHOTS[stage] ?? []) {
    const hit = (files ?? []).find((f) => re.test(f));
    if (hit) return hit;
  }
  return "";
}

/** The concept picture, root-relative (out/<stem>/...), or "". */
export function conceptStill(flow: FlowState | null | undefined, stem: string): string {
  const files = flowEntry(flow, "concept")?.files ?? [];
  const still =
    files.find((f) => f === "flat/source.png") || files.find((f) => f === "lifted.png") || files.find((f) => /^input\.(png|jpe?g|webp)$/.test(f));
  return still ? `out/${stem}/${still}` : "";
}

/** The stage to open when none is chosen yet (refreshFlow). `workflow` comes from /control/workflow. */
export function defaultStage(flow: FlowState | null | undefined, workflow: Workflow | ""): Stage {
  const live = (s: Stage) => isLive(flow, s);
  if (live("ideas") && !live("concept")) return "ideas";
  if (workflow === "2d") return live("flat") ? "flat" : "concept";
  if (live("mesh") || live("stack") || !live("concept")) return live("mesh") || !live("stack") ? "mesh" : "stack";
  return "concept";
}

/** A rail button is hidden when the output does not need the stage, it holds nothing, and it is not open. */
export function railHidden(flow: FlowState | null | undefined, output: AssetOutput | undefined, stage: Stage, current: Stage): boolean {
  const want = output ? OUTPUT_STAGES[output] : undefined;
  return !!want && !want.includes(stage) && !isLive(flow, stage) && stage !== current;
}

export function railTitle(entry: FlowEntry | null): string {
  if (!entry || !entry.live) return "Nothing yet";
  const n = entry.takes.length;
  const versions = n ? ` · ${n} saved version${n === 1 ? "" : "s"}` : "";
  const sum = entry.summary;
  const problem = sum?.line && (sum.level === "fail" || sum.level === "warn") ? ` · ${sum.line}` : "";
  return entry.stale.length ? `Ready · built from an older ${entry.stale.join(" and ")}${versions}${problem}` : `Ready${versions}${problem}`;
}

/** "Made from an older <names joined with ' and '>." for the takes strip; "" when not stale. */
export function staleNote(entry: FlowEntry | null): string {
  if (!entry?.stale.length) return "";
  const names = entry.stale.map((s) => STAGE_VIEW[s as Stage]?.name ?? s);
  return `Made from an older ${names.join(" and ")}.`;
}

/** Solo pane in 3D: stack → stack; rig → mesh; mesh → mesh unless the intent is a stack and a stack is live. */
export function soloPane(flow: FlowState | null | undefined, stage: Stage, output: AssetOutput | undefined): "" | "mesh" | "stack" {
  if (stage === "stack") return "stack";
  if (stage === "rig") return "mesh";
  if (stage === "mesh" && ((output && output !== "stack") || !isLive(flow, "stack"))) return "mesh";
  return "";
}

/* ---------------------------------------------------------------- store */

interface FlowStore {
  stem: string;
  flow: FlowState | null;
  /** Bumped on every flow load, so a stage can re-read its own files after a pick. */
  version: number;
  load: (stem: string) => Promise<FlowState | null>;
  set: (flow: FlowState | null) => void;
}

export const useFlow = create<FlowStore>((set, get) => ({
  stem: "",
  flow: null,
  version: 0,
  async load(stem) {
    if (stem !== get().stem) set({ stem, flow: null });
    const data = await factoryJson<FlowState>(`/flow/${enc(stem)}`).catch(() => null);
    if (get().stem !== stem) return null;
    const flow = data?.ok ? data : null;
    set((s) => ({ flow, version: s.version + 1 }));
    return flow;
  },
  set(flow) {
    set((s) => ({ flow, version: s.version + 1 }));
  },
}));

let flowTimer: ReturnType<typeof setTimeout> | undefined;
/** 300 ms after the last write, re-read the open asset's flow (legacy queueFlowRefresh). */
export function queueFlowRefresh(): void {
  clearTimeout(flowTimer);
  flowTimer = setTimeout(() => {
    const { stem, load } = useFlow.getState();
    if (stem) void load(stem);
  }, 300);
}

if (typeof window !== "undefined") onFlowDirty(() => queueFlowRefresh());

/** POST /flow/<s>/pick: the response is the whole flow state. */
export async function pickTake(stem: string, stage: Stage, take: string): Promise<FlowState> {
  const flow = await factoryJson<FlowState>(`/flow/${enc(stem)}/pick`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stage, take }),
  });
  if (useFlow.getState().stem === stem) useFlow.getState().set(flow);
  return flow;
}

/** POST /flow/<s>/snapshot: "Keep this version". Re-reads the flow after. */
export async function snapshotTake(stem: string, stage: Stage): Promise<void> {
  await factoryJson(`/flow/${enc(stem)}/snapshot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stage }),
  });
  if (useFlow.getState().stem === stem) await useFlow.getState().load(stem);
}
