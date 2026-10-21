// The factory job lifecycle as one small store. One job slot on the factory, so one job here.
// Ported from the legacy page (runJob, followJob, cancelJob, reattachJob, jobFraction, secondsLeft,
// overlayPctText, degradedLine, shapeMvNote). Screen work on finish (open the asset, the "ready" toast with
// its facts) belongs to the screens: register it with onJobFinished / onJobEnded.
import { create } from "zustand";
import {
  ApiError,
  cancelJob as apiCancel,
  factoryJson,
  getHealth,
  jobStatus,
  type DegradedEntry,
  type Health,
  type HealthFeatures,
  type JobKind,
  type JobProgress,
  type JobResult,
  type JobSnapshot,
  type JobStarted,
  type StageTime,
} from "./api";
import { toast } from "./ui";

/* ---------------------------------------------------------------- pure helpers (exact legacy text) */

/** The factory's fallbacks ({stage, code, message}, the DEGRADED grammar) as one plain line; "" when none. */
export function degradedLine(list: unknown): string {
  const items = (Array.isArray(list) ? (list as DegradedEntry[]) : []).filter((e) => e && (e.stage || e.code || e.message));
  if (!items.length) return "";
  return "Degraded: " + items.map((e) => `${e.stage || "?"} ${e.code || "?"}${e.message ? `: ${e.message}` : ""}`).join(" · ");
}

/** The multi-view shape, as /health.features probed it (factory_features): only what the probe shows. */
export function shapeMvNote(f: HealthFeatures | null | undefined): string {
  if (!f || typeof f.shape_mv !== "boolean") return "";
  if (f.shape_mv) return ". Multi-view shape (Hunyuan3D-2mv) is installed; it is used when a drawing's four views match it.";
  const why = !f.shape_mv_venv && !f.shape_mv_weights ? "its Python env and weights are missing"
    : !f.shape_mv_venv ? "its Python env is not installed" : "its weights are missing";
  return `. Multi-view shape is off (${why}): the shape comes from the first picture; extra views only photo-bake.`;
}

/** The factory pill: "idle" / "busy" / "Apple Silicon / MLX" / gpu. Tooltip: `Factory: ${healthLabel(h)}${shapeMvNote(h.features)}`. */
export function healthLabel(h: Health | null | undefined): string {
  if (!h) return "idle";
  if (h.busy) return "busy";
  if (h.backend === "mlx") return "Apple Silicon / MLX";
  return h.gpu || "idle";
}

export function displayStem(stem: string | null | undefined): string {
  return String(stem || "").replace(/-/g, " ");
}

export interface JobTiming {
  stage_times?: Record<string, StageTime>;
  stage_est?: Record<string, number>;
  estimate_s?: number;
  started?: number;
  /** factory clock minus ours, seconds (status.now - Date.now()/1000). */
  skew?: number;
}

export function factoryNow(timing?: JobTiming | null): number {
  return Date.now() / 1000 + ((timing && timing.skew) || 0);
}

export function jobStageText(p: JobProgress | null | undefined): string {
  if (!p || p.stage_index < 0) return "Starting";
  const st = p.stages[p.stage_index];
  if (st && /^idea\d+$/.test(st.key)) return `${st.label} of ${p.stages.filter((x) => /^idea\d+$/.test(x.key)).length}`;
  return `${st ? st.label : p.stage} ${p.stage_index + 1}/${p.stage_count}`;
}

/** Seconds on a stage row: live for the running one, final for a done one. Times come from the factory clock. */
export function stageSeconds(times: Record<string, StageTime> | undefined, key: string, nowS: number): string {
  const t = times && times[key];
  if (!t || !t.started) return "";
  const end = t.ended || nowS;
  return `${Math.max(0, Math.round(end - t.started))} s`;
}

/** The small line under a stage row: its step % while running, and its seconds. */
export function stageSubText(progress: JobProgress, i: number, state: JobState | undefined, timing?: JobTiming | null): string {
  const st = progress.stages[i];
  const now = i === progress.stage_index && state !== "ok";
  const parts: string[] = [];
  if (now && progress.sub) parts.push(Math.round(progress.sub * 100) + "%");
  if (timing && timing.stage_times && st) {
    const done = i < progress.stage_index || state === "ok";
    const t = timing.stage_times[st.key];
    if (t && (done || now || t.ended)) {
      const sec = stageSeconds(timing.stage_times, st.key, factoryNow(timing));
      if (sec) parts.push(sec);
    }
  }
  return parts.join(" · ");
}

/** Seconds each planned stage took last time (stage_est), when the factory knows every one of them. */
export function stageWeights(progress: JobProgress | null | undefined, timing?: JobTiming | null): number[] | null {
  const est = timing && timing.stage_est;
  if (!est || !progress || !progress.stages || !progress.stages.length) return null;
  if (progress.stages.some((st) => typeof est[st.key] !== "number")) return null;
  return progress.stages.map((st) => est[st.key]);
}

/** How far into the running stage: its step bar when it has one, else its time against last time's. */
function stagePart(progress: JobProgress, timing: JobTiming | null | undefined, weights: number[]): number {
  const i = progress.stage_index;
  if (progress.sub) return Math.min(1, progress.sub);
  const st = progress.stages[i];
  const t = st && timing && timing.stage_times && timing.stage_times[st.key];
  if (!t || !t.started || !(weights[i] > 0)) return 0;
  return Math.min(0.95, (factoryNow(timing) - t.started) / weights[i]);
}

/** The bar, weighted by how long each stage took last time; else every stage counts the same. */
export function jobFraction(progress: JobProgress | null | undefined, timing?: JobTiming | null): number {
  const w = stageWeights(progress, timing);
  const total = w ? w.reduce((a, b) => a + b, 0) : 0;
  if (!w || !(total > 0) || !progress || progress.stage_index < 0) return (progress && progress.fraction) || 0;
  const done = w.slice(0, progress.stage_index).reduce((a, b) => a + b, 0);
  return (done + w[progress.stage_index] * stagePart(progress, timing, w)) / total;
}

export function secondsLeft(progress: JobProgress | null | undefined, timing?: JobTiming | null): number | null {
  const w = stageWeights(progress, timing);
  if (w && progress && progress.stage_index >= 0) {
    const i = progress.stage_index;
    return w.slice(i + 1).reduce((a, b) => a + b, 0) + w[i] * (1 - stagePart(progress, timing, w));
  }
  if (timing && typeof timing.estimate_s === "number" && timing.started) return timing.estimate_s - (factoryNow(timing) - timing.started);
  return null;
}

/** "Done" / "Cancelled" / "Stopped" / "Starting…" / "<stage> i/n · NN%" plus " · about N s left" (rounded to 5, only when >= 5). */
export function overlayPctText(progress: JobProgress | null | undefined, state: JobState | undefined, pct: number, timing?: JobTiming | null): string {
  if (!(state === "running" || !state)) return state === "ok" ? "Done" : state === "cancelled" ? "Cancelled" : "Stopped";
  if (!progress) return "Starting…";
  let text = `${jobStageText(progress)} · ${pct}%`;
  // No claim when it runs past last time (a slow load, an extra try): the stages say where it is.
  const left = secondsLeft(progress, timing);
  if (typeof left === "number" && left >= 5) text += ` · about ${Math.round(left / 5) * 5} s left`;
  return text;
}

/** 100 when ok, at least 2 otherwise. */
export function overlayPct(progress: JobProgress | null | undefined, state: JobState | undefined, timing?: JobTiming | null): number {
  if (state === "ok") return 100;
  if (!progress || !progress.stages) return 2;
  return Math.max(2, Math.round(jobFraction(progress, timing) * 100));
}

/**
 * The overlay's step line and Details log (legacy drawJob). Running: the last log line (minus "[ NN%]") is
 * the step and the 30 before it the log. Error/cancelled: the error (or the stopped text), and the last 40.
 */
export function jobLines(j: Pick<Job, "log" | "state" | "stopping" | "error">): { step: string; log: string } {
  const lines = (j.log || "").trim().split("\n").filter((l) => l && !/^STAGES? /.test(l));
  if (j.state === "running") {
    const step = j.stopping ? "Stopping…" : (lines.pop() || "Starting…").replace(/\s*\[\s*\d+%\]/, "");
    return { step, log: lines.slice(-30).join("\n") };
  }
  if (j.state === "error") return { step: j.error || "The factory stopped.", log: lines.slice(-40).join("\n") };
  if (j.state === "cancelled") return { step: "Nothing after the last finished step was written.", log: lines.slice(-40).join("\n") };
  return { step: "", log: lines.slice(-30).join("\n") };
}

/* ---------------------------------------------------------------- the store */

export type JobState = "running" | "ok" | "error" | "cancelled";

export interface JobOpts {
  /** POST path; default /generate. e.g. "/2d/draw", "/rig/<stem>". */
  path?: string;
  title?: string;
  kind?: JobKind;
  rig?: boolean;
  /** false: follow on the chip only, no overlay. Default true. */
  watch?: boolean;
}

export interface Job extends JobTiming {
  id: string;
  stem: string;
  kind: JobKind;
  state: JobState;
  progress: JobProgress | null;
  log: string;
  error?: string;
  title?: string;
  spec?: Record<string, unknown>;
  opts?: JobOpts;
  stopping?: boolean;
  preview_url?: string;
  result?: JobResult | null;
  prompt?: string;
  count?: number;
  revise?: boolean;
  source?: string;
  degraded?: DegradedEntry[];
  /** Set while a screen chains the next job from this finished one (draw → build): the card says it, not "Opening". */
  next?: string;
}

/** What the job is doing, for the chip and the overlay title: the same words in both. */
export function jobDoing(j: Pick<Job, "stem" | "kind" | "spec" | "opts" | "revise" | "source">): string {
  const name = displayStem(j.stem);
  const spec = (j.spec || {}) as { revise?: boolean; source?: string };
  if (j.kind === "rig" || j.opts?.rig) return `Rigging ${name}`;
  if (j.kind === "ideas") return `Drawing ideas for ${name}`;
  if (j.kind === "flat2d") {
    if (j.revise || spec.revise) return `Redrawing ${name}`;
    if ((j.source || spec.source) === "drop") return `Finishing ${name}`;
    return `Drawing ${name}`;
  }
  return `Making ${name}`;
}

/** The job chip (legacy renderJobChip): what it is doing, or how it ended and what a click does. */
export function jobChipText(j: Job): string {
  const name = displayStem(j.stem);
  if (j.next) return `${j.next} · Show`;
  if (j.state === "running") return `${j.title || jobDoing(j)} · ${jobStageText(j.progress)} · Show`;
  if (j.state === "ok") return `${name} is ready · Open`;
  if (j.state === "cancelled") return `${name} cancelled · Dismiss`;
  return `${name} failed · Details`;
}

/** "Couldn't <verb> <name>". */
export function jobVerb(j: Pick<Job, "kind" | "opts">): string {
  if (j.kind === "rig" || j.opts?.rig) return "rig";
  if (j.kind === "ideas") return "draw ideas for";
  if (j.kind === "flat2d") return "draw";
  return "make";
}

/** A 2D draw (not a dropped picture): the overlay keeps a square for the picture as it draws. */
export function jobDraws(j: Pick<Job, "kind" | "source" | "spec"> | null): boolean {
  if (!j) return false;
  if (j.kind === "ideas") return true;
  return j.kind === "flat2d" && (j.source || (j.spec as { source?: string } | undefined)?.source) !== "drop";
}

interface JobsStore {
  job: Job | null;
  /** The overlay is open on this job (else the chip tracks it). */
  watching: boolean;
  /** A start is in flight or a job runs: Make/Go buttons hold. */
  busy: boolean;
  /** Last /health line for the factory pill ("" until asked). */
  statusText: string;
  statusState: "" | "running" | "ok" | "error";
  setWatching: (w: boolean) => void;
  /** Drop a finished/failed job (close the overlay and the chip). A running job keeps running. */
  dismiss: () => void;
}

export const useJobs = create<JobsStore>((set, get) => ({
  job: null,
  watching: false,
  busy: false,
  // Shown until the first /health answers (a cold factory takes a while), like the legacy page.
  statusText: "Starting the factory…",
  statusState: "running",
  setWatching: (w) => set({ watching: w }),
  dismiss: () => {
    const j = get().job;
    if (j && j.state === "running") set({ watching: false });
    else set({ job: null, watching: false });
  },
}));

type JobListener = (job: Job) => void;
const finished = new Set<JobListener>();
const ended = new Set<JobListener>();
/** state "ok". The screen opens the asset / shows the ready toast (degradedLine(job.degraded) in its text). */
export function onJobFinished(fn: JobListener): () => void {
  finished.add(fn);
  return () => finished.delete(fn);
}
/** Any end: ok, error, cancelled, lost. Legacy jobEnded (refresh /2d/status, route line, ideas). */
export function onJobEnded(fn: JobListener): () => void {
  ended.add(fn);
  return () => ended.delete(fn);
}

const patch = (p: Partial<Job>) => {
  const j = useJobs.getState().job;
  if (j) useJobs.setState({ job: { ...j, ...p } });
};
const current = (id: string) => {
  const j = useJobs.getState().job;
  return j && j.id === id ? j : null;
};
function emit(set: Set<JobListener>, j: Job) {
  for (const fn of set) {
    try {
      fn(j);
    } catch (e) {
      console.error(e);
    }
  }
}

const POLL_MS = 1200;
/** Newest follow wins: an older poll loop (StrictMode double reattach, 409 follow racing reattach) stops. */
let followGen = 0;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const COPIED = ["preview_url", "stage_times", "stage_est", "estimate_s", "result", "started", "prompt", "count", "revise", "source", "degraded"] as const;

/** Start a job. On 409 (the slot is taken) ask /health and follow the running job instead. */
export async function startJob(spec: Record<string, unknown>, opts: JobOpts = {}): Promise<void> {
  const watch = opts.watch !== false;
  useJobs.setState({ busy: true, ...(watch ? { watching: true } : {}) });
  let started: JobStarted;
  try {
    const path = opts.path || "/generate";
    started = await factoryJson<JobStarted>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec) });
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      const h = await getHealth().catch(() => null);
      if (h && h.busy && h.job_id) {
        toast("The factory is busy with another job. Showing that one.");
        return followJob(h.job_id, "", { watch });
      }
    }
    const msg = err instanceof Error ? err.message : String(err);
    const j: Job = { id: "", stem: String(spec.stem || "asset"), kind: opts.kind || "", state: "error", error: msg, progress: null, log: "", spec, opts, title: opts.title };
    useJobs.setState({ job: j, busy: false, statusText: msg, statusState: "error" });
    emit(ended, j);
    return;
  }
  return followJob(started.job_id, started.stem || String(spec.stem || "asset"), { watch, spec, opts, kind: started.kind || opts.kind });
}

/** Follow a job: /status/<id> at once, then every 1.2 s, until it is not running. */
export async function followJob(
  jobId: string,
  stem: string,
  o: { watch?: boolean; spec?: Record<string, unknown>; opts?: JobOpts; kind?: JobKind } = {},
): Promise<void> {
  const gen = ++followGen;
  const mine = () => gen === followGen && current(jobId);
  const j: Job = { id: jobId, stem, kind: o.kind || "", state: "running", progress: null, log: "", spec: o.spec, opts: o.opts, title: o.opts?.title };
  useJobs.setState({ job: j, busy: true, ...(o.watch ? { watching: true } : {}) });
  for (let first = true; ; first = false) {
    if (!first) await sleep(POLL_MS);
    if (!mine()) return;
    let st: JobSnapshot;
    try {
      st = await jobStatus(jobId);
    } catch (err) {
      if (!mine()) return;
      patch({ state: "error", error: `Lost the factory: ${err instanceof Error ? err.message : err}` });
      useJobs.setState({ busy: false });
      emit(ended, useJobs.getState().job!);
      return;
    }
    const cur = mine();
    if (!cur) return;
    const next: Partial<Job> = {
      stem: st.stem || cur.stem,
      kind: st.kind || cur.kind,
      progress: st.progress || cur.progress,
      log: st.log_tail || "",
    };
    for (const k of COPIED) if (st[k] !== undefined) (next as Record<string, unknown>)[k] = st[k];
    if (typeof st.now === "number") next.skew = st.now - Date.now() / 1000;
    patch(next);
    if (st.state === "running") {
      if (useJobs.getState().statusState === "running") useJobs.setState({ statusText: "", statusState: "" });
      continue;
    }
    useJobs.setState({ busy: false });
    if (st.state === "ok") {
      patch({ state: "ok" });
      useJobs.setState({ statusText: "", statusState: "ok" });
      const done = useJobs.getState().job!;
      emit(finished, done);
      emit(ended, done);
      return;
    }
    const cancelled = st.state === "cancelled";
    patch({ state: cancelled ? "cancelled" : "error", error: st.error || "" });
    useJobs.setState({ statusText: cancelled ? "Cancelled." : st.error || "Failed.", statusState: cancelled ? "" : "error" });
    const after = useJobs.getState().job!;
    emit(ended, after);
    // You pressed Stop and were not watching: nothing to come back to, so no chip.
    if (cancelled && after.stopping && !useJobs.getState().watching) useJobs.setState({ job: null });
    return;
  }
}

/** POST /cancel/<id>. The poll sees "cancelled" and ends the job. Returns an error message or "". */
export async function cancelJob(): Promise<string> {
  const j = useJobs.getState().job;
  if (!j || j.state !== "running" || j.stopping) return "";
  patch({ stopping: true });
  try {
    await apiCancel(j.id);
    return "";
  } catch (err) {
    if (current(j.id)) patch({ stopping: false });
    return err instanceof Error ? err.message : String(err);
  }
}

/** Back from a reload or a crash: pick up the job the factory is running (chip only, no overlay). */
export async function reattachJob(h?: Health | null): Promise<void> {
  if (useJobs.getState().job) return;
  const health = h ?? (await getHealth().catch(() => null));
  if (!health || !health.busy || !health.job_id) return;
  const st = await jobStatus(health.job_id).catch(() => null);
  void followJob(health.job_id, (st && st.stem) || "", { watch: false });
}

/** At load: /health sets the factory line ("The factory is busy with another job." / "Ready."), then reattach. */
export async function checkFactory(): Promise<Health | null> {
  try {
    const h = await getHealth();
    useJobs.setState(h.busy ? { statusText: "The factory is busy with another job.", statusState: "running" } : { statusText: "Ready.", statusState: "" });
    await reattachJob(h);
    return h;
  } catch (err) {
    useJobs.setState({ statusText: err instanceof Error ? err.message : String(err), statusState: "error" });
    return null;
  }
}
