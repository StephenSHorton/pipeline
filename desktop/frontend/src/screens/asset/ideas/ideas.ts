// The Ideas grid's pure parts (legacy renderIdeas / ideasJobHere / ideasPending / paintIdeaLive and the copy around
// them, index.html 3593-3735). The component and its tests both read from here, so the words stay in one place.
import { enc } from "@/lib/api";
import type { Job } from "@/lib/jobs";
import type { IdeaItem, IdeasDoc } from "./api";

export const IDEAS_COPY = {
  noModel: "No local image model. Install mflux to draw ideas.",
  empty: "No ideas yet. Describe the thing below.",
  busy: "The factory is busy with another job.",
  allKept: "Every idea here is kept or in use.",
  stopped: "Stopped.",
  useTitle: "Make this the concept. 2D, 3D and the stack start from it.",
  useWait: "Wait for the ideas to finish.",
  waiting: "Waiting…",
  stopsAfterLoading: "Stops after loading",
  placeholder: "Describe one object…",
  promptTitle: "One object per picture. Make each piece of a set as its own asset.",
  clearTitle: "Delete the ideas you did not keep.",
  moreTitle: "Draw four more from this prompt.",
} as const;

/** More ideas draws this many from the prompt. */
export const MORE_COUNT = 4;

/** /out/<stem>/ideas/<file>?s=<seed>: the seed busts the cache when a slot is redrawn under the same name. */
export function ideaUrl(stem: string, item: Pick<IdeaItem, "file" | "seed">): string {
  return `/out/${enc(stem)}/ideas/${item.file.split("/").map(encodeURIComponent).join("/")}?s=${item.seed ?? ""}`;
}

type JobLike = Pick<Job, "state" | "kind" | "stem"> & Partial<Pick<Job, "count" | "progress" | "result" | "log" | "stopping" | "preview_url">>;

/** The running Ideas job, when it draws into this asset. */
export function ideasJobHere(job: JobLike | null | undefined, stem: string): job is JobLike {
  return !!job && job.state === "running" && job.kind === "ideas" && job.stem === stem;
}

/** Pictures the running Ideas job has still to draw here (0 when none runs here). */
export function ideasPending(job: JobLike | null | undefined, stem: string): number {
  if (!ideasJobHere(job, stem)) return 0;
  const total = job.count || (job.progress?.stages ?? []).filter((st) => /^idea\d+$/.test(st.key)).length || 1;
  return Math.max(0, total - (job.result?.items ?? []).length);
}

/** How many ideas the job has already written (a change reloads the grid). */
export const jobItems = (job: JobLike | null | undefined) => (job?.result?.items ?? []).length;

/** The model is loading: Stop waits for it ("Stops after loading"). */
export const ideasLoading = (job: JobLike | null | undefined) => job?.progress?.stage === "load";

/**
 * The live slot's line: "Stopping…", "Loading the model…", or `Picture k of N[ · again (why)][ · step a/b]`, the
 * step from the last log line that has one, unless a redraw started after it ("Again, ground was cloudy · Step 3 of 8").
 */
export function liveSlotText(job: JobLike, pending: number): string {
  const total = job.count || pending;
  const k = jobItems(job) + 1;
  const last = (job.log || "").split("\n").reverse().find((l) => /Step \d+ of \d+|Drawing again/.test(l)) || "";
  const m = last.match(/Step (\d+) of (\d+)/);
  const why = (last.match(/^Again, (.+?) · Step/) || last.match(/^(.+?)\. Drawing again/) || [])[1] || "";
  const again = why
    ? ` · again (${why.replace(/^ground (was|is|had) (a )?|^There is a /i, "").replace(/ on the ground$/i, "").toLowerCase()})`
    : "";
  if (job.stopping) return "Stopping…";
  if (ideasLoading(job)) return "Loading the model…";
  return `Picture ${Math.min(k, total)} of ${total}${again}${m ? ` · step ${m[1]}/${m[2]}` : ""}`;
}

/** Ideas that Clear unkept would delete. */
export const unkeptCount = (items: IdeaItem[] | undefined) => (items ?? []).filter((i) => !i.keep && !i.used).length;

/** "Delete the N idea(s) you did not keep?" */
export const clearConfirmText = (n: number) => `Delete the ${n} idea${n === 1 ? "" : "s"} you did not keep?`;

/** The status line after a load: the no-model text when there is no still model and no job drawing here. */
export function loadStatus(doc: IdeasDoc | null, jobHere: boolean): string | null {
  return doc && !doc.still && !jobHere ? IDEAS_COPY.noModel : null;
}

/** The status line after an Ideas job here ends without finishing: "Stopped." or its error. */
export function endedStatus(job: Pick<Job, "state" | "error">): string {
  return job.state === "cancelled" ? IDEAS_COPY.stopped : job.error || "";
}

/** The prompt, kept while you step to another stage and back (legacy #ideas-prompt); another asset starts empty. A
 * Home ideas job that hands over to the grid fills it from the job's prompt. */
let draft = { stem: "", text: "" };
export const draftFor = (stem: string) => (draft.stem === stem ? draft.text : "");
export function setIdeasDraft(stem: string, text: string): void {
  draft = { stem, text };
}

type Watched = Pick<Job, "id" | "state" | "kind" | "stem"> & Partial<Pick<Job, "result">>;

/**
 * Legacy followIdeas: a watched Ideas job hands over from the overlay to the grid once its first picture lands,
 * unless that asset is already open or this job has already handed over.
 */
export function shouldFollow(job: Watched | null | undefined, watching: boolean, routeStem: string | null, opened: ReadonlySet<string>): job is Watched {
  if (!job || job.kind !== "ideas" || job.state !== "running" || !watching || opened.has(job.id)) return false;
  if (jobItems(job) < 1) return false;
  return routeStem !== job.stem;
}

/**
 * Legacy finishIdeasJob, for every Ideas job wherever it started: watching opens the asset on Ideas (no toast);
 * that asset already open drops the job quietly (the grid reloads itself); anywhere else a toast offers to open it.
 */
export function finishAction(watching: boolean, routeStem: string | null, stem: string): "open" | "drop" | "toast" {
  if (routeStem === stem) return "drop";
  return watching ? "open" : "toast";
}
