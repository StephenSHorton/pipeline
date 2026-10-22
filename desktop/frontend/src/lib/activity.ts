// What Claude and the factory are doing, from GET /activity (scripts/activity.py): each workbench tool call Claude
// makes and each job, as it happens. The store polls every 1.2 s (followJob's pace) with since=<last seq> and
// replaces entries by id. Listeners hear the changes after the first answer; of the history that first answer
// carries they hear only what is still running, so nothing jumps to an asset Claude touched an hour ago.
import { create } from "zustand";
import { factoryJson } from "@/lib/api";

export type ActivityState = "running" | "ok" | "error" | "cancelled";

export interface ActivityEntry {
  id: string;
  seq: number;
  ts: number;
  source?: "claude" | "job";
  tool?: string;
  kind?: string;
  stem?: string;
  title?: string;
  state?: ActivityState;
  started?: number;
  ms?: number;
  detail?: string;
  /** A model the action wrote, out/-relative ("<stem>/workbench/clay.glb"): the viewer shows it as it lands. */
  glb?: string;
  /** A picture to show with the line, out/-relative (a look sheet, the drawing as it forms). */
  image?: string;
  job_id?: string;
  error?: string;
}

interface ActivityAnswer {
  ok: boolean;
  seq: number;
  now: number;
  entries: ActivityEntry[];
}

export const KEEP = 150;
const FOLLOW_KEY = "pipeline.followClaude";

export function readFollow(): boolean {
  try {
    return localStorage.getItem(FOLLOW_KEY) !== "0";
  } catch {
    return true;
  }
}

function writeFollow(on: boolean): void {
  try {
    localStorage.setItem(FOLLOW_KEY, on ? "1" : "0");
  } catch {
    /* a private window: the switch still works for this visit */
  }
}

/** Newest first (by when the action started), one copy per id, at most KEEP. */
export function mergeEntries(list: ActivityEntry[], fresh: ActivityEntry[]): ActivityEntry[] {
  const byId = new Map(list.map((e) => [e.id, e]));
  for (const e of fresh) {
    const old = byId.get(e.id);
    if (!old || e.seq > old.seq) byId.set(e.id, e);
  }
  return [...byId.values()].sort((a, b) => (b.started ?? b.ts) - (a.started ?? a.ts) || b.seq - a.seq).slice(0, KEEP);
}

/** The action going on now: the newest running entry (Claude's own before a job's, when both run). */
export function currentEntry(list: ActivityEntry[]): ActivityEntry | null {
  const running = list.filter((e) => e.state === "running");
  return running.find((e) => e.source === "claude") ?? running[0] ?? null;
}

/** "0.4 s", "12 s", "3 min 05 s". */
export function durationText(ms: number | undefined): string {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return "";
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)} s`;
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

/** How long it took, or how long it has been going (nowS: the factory's clock, from the last answer). */
export function entryTime(e: ActivityEntry, nowS: number): string {
  if (e.state === "running") return e.started ? durationText((nowS - e.started) * 1000) : "";
  return durationText(e.ms);
}

/** "<stem>/rest/of/file.glb" → the stem and the file under out/<stem>/. */
export function splitOut(rel: string): { stem: string; file: string } | null {
  const i = rel.indexOf("/");
  if (i <= 0 || i === rel.length - 1) return null;
  return { stem: rel.slice(0, i), file: rel.slice(i + 1) };
}

/** The picture's URL; the seq busts the cache, since a forming drawing keeps one file name. */
export function imageUrl(e: ActivityEntry): string | null {
  if (!e.image) return null;
  return `/out/${e.image.split("/").map(encodeURIComponent).join("/")}?v=${e.seq}`;
}

type Listener = (entry: ActivityEntry, prev: ActivityEntry | undefined) => void;
const listeners = new Set<Listener>();

/** Every new copy of an entry after the first answer, with the copy it replaces. */
export function onActivity(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

interface ActivityStore {
  entries: ActivityEntry[];
  seq: number;
  /** The factory's clock at the last answer, plus the local seconds since (for running times). */
  now: number;
  open: boolean;
  follow: boolean;
  primed: boolean;
  setOpen: (open: boolean) => void;
  setFollow: (on: boolean) => void;
}

export const useActivity = create<ActivityStore>((set) => ({
  entries: [],
  seq: 0,
  now: 0,
  open: false,
  follow: readFollow(),
  primed: false,
  setOpen: (open) => set({ open }),
  setFollow: (on) => {
    writeFollow(on);
    set({ follow: on });
  },
}));

/** Take in one answer: merge, then tell the listeners (not for the first answer's history). */
export function absorb(answer: ActivityAnswer): void {
  const st = useActivity.getState();
  const before = new Map(st.entries.map((e) => [e.id, e]));
  const fresh = (answer.entries || []).filter((e) => e && typeof e.id === "string");
  useActivity.setState({
    entries: mergeEntries(st.entries, fresh),
    seq: Math.max(st.seq, answer.seq || 0),
    now: answer.now || st.now,
    primed: true,
  });
  // The first answer is history, except what is still running: a window opened mid-job follows that job.
  const told = st.primed ? fresh : fresh.filter((e) => e.state === "running");
  for (const e of told.sort((a, b) => a.seq - b.seq)) {
    const prev = before.get(e.id);
    before.set(e.id, e);
    for (const fn of listeners) {
      try {
        fn(e, prev);
      } catch (err) {
        console.error("activity listener", err);
      }
    }
  }
}

let timer: ReturnType<typeof setInterval> | null = null;
let inFlight = false;

export async function pollActivity(): Promise<void> {
  if (inFlight) return;
  inFlight = true;
  try {
    const answer = await factoryJson<ActivityAnswer>(`/activity?since=${useActivity.getState().seq}`);
    absorb(answer);
  } catch {
    // A factory that is starting or gone: the next tick asks again.
  } finally {
    inFlight = false;
  }
}

/** Start the 1.2 s poll (once; a second call does nothing). The running times tick between answers. */
export function startActivityWatch(): () => void {
  if (timer) return () => {};
  void pollActivity();
  timer = setInterval(() => {
    void pollActivity();
    const st = useActivity.getState();
    if (st.now) useActivity.setState({ now: st.now + 1.2 });
  }, 1200);
  return () => {
    if (timer) clearInterval(timer);
    timer = null;
  };
}
