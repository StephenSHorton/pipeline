// Follow Claude (the activity chip's switch, on by default): when Claude starts work on an asset the app opens it,
// and each model the work writes (a clay mesh, a bake, the PS1 model, a build's shape and then its textured model)
// is loaded into the viewer as it lands, so the asset takes shape on screen. A job started outside the app (by
// the workbench) is picked up as the app's job, so its card and chip show its stages.
//
// Never pulls you away mid-task: not while a viewer tool (Carve, Paint) is armed or you are typing in a field.
import { navigate, parseRoute } from "@/app/router";
import { onActivity, splitOut, useActivity, type ActivityEntry } from "@/lib/activity";
import { useFlow } from "@/lib/flow";
import { reattachJob, useJobs } from "@/lib/jobs";
import { useStage } from "@/screens/asset/stage";
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

/** What a new copy of an entry asks of the screen. Pure, for the tests. */
export function followStep(
  e: ActivityEntry,
  prev: ActivityEntry | undefined,
): { open?: string; model?: { stem: string; file: string }; refresh?: boolean; adopt?: boolean } {
  const out: { open?: string; model?: { stem: string; file: string }; refresh?: boolean; adopt?: boolean } = {};
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

export function startFollow(): () => void {
  return onActivity((e, prev) => {
    const step = followStep(e, prev);
    if (step.adopt && !useJobs.getState().job) void reattachJob();
    if (!useActivity.getState().follow) return;
    if (step.open && currentStem() !== step.open && !busyHere()) navigate({ name: "asset", stem: step.open });
    if (step.model) void showModel(step.model.stem, step.model.file, !!step.refresh);
  });
}
