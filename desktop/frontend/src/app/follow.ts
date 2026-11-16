// Follow Claude (the activity chip's switch, on by default): when Claude starts work on an asset the app opens it,
// and each model the work writes (a clay mesh, a bake, the PS1 model, a build's shape and then its textured model)
// is loaded into the viewer as it lands, so the asset takes shape on screen. A job started outside the app (by
// the workbench) is picked up as the app's job, so its card and chip show its stages.
//
// Never pulls you away mid-task: not while a viewer tool (Carve, Paint) is armed or you are typing in a field.
//
// A workbench `show` is Claude asking you to look (the asset on a stage, an earlier version beside it in Compare):
// it opens even with Follow off, since it was asked for, but still waits on the same two things.
import { navigate, parseRoute } from "@/app/router";
import { onActivity, splitOut, useActivity, type ActivityEntry } from "@/lib/activity";
import { useFlow } from "@/lib/flow";
import { reattachJob, useJobs } from "@/lib/jobs";
import { useStage } from "@/screens/asset/stage";
import { useCompare } from "@/screens/viewer/compare";
import type { ViewerStage } from "@/screens/viewer/panels/panels";
import { useViewerTools, viewerTools } from "@/viewer/tools";

/** Something the person is doing that a jump would interrupt. */
export function busyHere(): boolean {
  if (useViewerTools.getState().tool) return true;
  const el = typeof document !== "undefined" ? document.activeElement : null;
  if (!el) return false;
  // Typing: a focused field with something in it (Home's Make box takes focus on its own, empty).
  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) return el.value.trim() !== "";
  return (el as HTMLElement).isContentEditable;
}

function currentStem(): string | null {
  const r = parseRoute(location.search, location.hash);
  return r.name === "asset" ? r.stem : null;
}

export interface FollowStep {
  open?: string;
  model?: { stem: string; file: string };
  refresh?: boolean;
  adopt?: boolean;
  /** A workbench show: open stem on stage, with compare (a take id or "previous") on the left. */
  show?: { stem: string; stage: ViewerStage; compare?: string };
}

/** What a new copy of an entry asks of the screen. Pure, for the tests. */
export function followStep(
  e: ActivityEntry,
  prev: ActivityEntry | undefined,
): FollowStep {
  const out: FollowStep = {};
  if (e.tool === "show") {
    // Asked for: once, when it lands, whatever Follow says.
    if (prev?.state === "ok" || e.state !== "ok" || !e.stem) return out;
    const stage: ViewerStage = e.view === "rig" ? "rig" : "mesh";
    out.show = { stem: e.stem, stage, ...(e.compare ? { compare: e.compare } : {}) };
    const m = e.glb ? splitOut(e.glb) : null;
    if (m) out.model = m;
    return out;
  }
  // The feed sends an entry's latest copy only, so a quick action can arrive already done: new is enough.
  const started = !prev || (e.state === "running" && prev.state !== "running");
  if (started && e.stem) out.open = e.stem;
  if (started && e.state === "running" && e.source === "job") out.adopt = true;
  if (e.glb && e.glb !== prev?.glb) {
    const m = splitOut(e.glb);
    if (m) out.model = m;
  }
  // A workbench write ended: it kept a version, so the versions panel has a new row.
  if (e.source === "claude" && e.state === "ok" && prev?.state !== "ok" && e.glb) out.refresh = true;
  return out;
}

const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));

/** Wait (up to ~4 s) for the viewer of this stem to be up, after a jump or a stage change. */
async function viewerReady(stem: string): Promise<boolean> {
  for (let i = 0; i < 40; i++) {
    if (currentStem() === stem && useViewerTools.getState().loader) return true;
    await new Promise((r) => setTimeout(r, 100));
  }
  return false;
}

async function showModel(stem: string, file: string, refresh: boolean): Promise<void> {
  if (currentStem() !== stem || busyHere()) return;
  const stage = useStage.getState();
  if (stage.stem === stem && stage.stage !== "mesh" && stage.stage !== "rig") stage.enter("mesh");
  if (!(await viewerReady(stem))) return;
  if (refresh) {
    // The flow re-read reloads the usual model; the file goes on screen after it, not under it.
    await useFlow.getState().load(stem);
    await nextFrame();
  }
  if (currentStem() !== stem) return;
  await viewerTools.loadFile(file).catch(() => {});
}

/** Claude asked: open the asset on the stage and, for a before and after, Compare with the earlier version. */
async function showAsked(show: NonNullable<FollowStep["show"]>, model?: { stem: string; file: string }): Promise<void> {
  if (busyHere()) return;
  if (currentStem() !== show.stem) navigate({ name: "asset", stem: show.stem });
  const stage = useStage.getState();
  for (let i = 0; i < 40 && useStage.getState().stem !== show.stem; i++) await new Promise((r) => setTimeout(r, 100));
  if (useStage.getState().stage !== show.stage) stage.enter(show.stage);
  if (!(await viewerReady(show.stem))) return;
  if (model) await viewerTools.loadFile(model.file).catch(() => {});
  const c = useCompare.getState();
  if (show.compare) {
    c.start(show.stem, show.stage);
    if (show.compare !== "previous") useCompare.getState().setLeft(show.compare);
  } else if (c.stem) {
    c.stop();
  }
}

export function startFollow(): () => void {
  return onActivity((e, prev) => {
    const step = followStep(e, prev);
    if (step.show) {
      void showAsked(step.show, step.model);
      return;
    }
    if (step.adopt && !useJobs.getState().job) void reattachJob();
    if (!useActivity.getState().follow) return;
    if (step.open && currentStem() !== step.open && !busyHere()) navigate({ name: "asset", stem: step.open });
    if (step.model) void showModel(step.model.stem, step.model.file, !!step.refresh);
  });
}
