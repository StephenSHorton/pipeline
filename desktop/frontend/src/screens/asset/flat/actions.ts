// What the revise form and the Audio panel's buttons do (legacy flat-revise submit, audio-speak, audio-sound,
// index.html 4093-4211). Side effects only through startJob, askConfirm and /2d/steps, so the tests mock those.
import { displayStem, startJob, useJobs } from "@/lib/jobs";
import { askConfirm } from "@/lib/ui";
import type { FlatMode } from "@/lib/flow";
import { clipFields, postSteps } from "./api";
import {
  applyAudio,
  applySteps,
  audioBody,
  pictureBlock,
  REDRAW_ASK,
  reviseBody,
  revisePart,
  STEP_RUNNING,
  type AudioRow,
  type SheetState,
} from "./flat";

/** Redraw specs started here: their finish toasts here ("Redrawn: <name>"), not on Home. */
export const flatSpecs = new WeakSet<object>();

/** A job on this asset is running (legacy jobOwns). */
export function jobOwns(stem: string): boolean {
  const j = useJobs.getState().job;
  return !!(j && j.state === "running" && j.stem === stem);
}

const anyRunning = () => useJobs.getState().job?.state === "running";

/**
 * Picture mode: ask, then a redraw job (POST /2d/draw). The chip follows it (watch false: the job card lives on
 * Home). Returns the status line when it cannot start, "" when it was asked and declined, or "started".
 */
export async function redraw(stem: string, prompt: string): Promise<{ status: string; started: boolean }> {
  const block = pictureBlock(prompt, anyRunning());
  if (block) return { status: block, started: false };
  if (!(await askConfirm(REDRAW_ASK, "Redraw"))) return { status: "", started: false };
  const spec = { kind: "flat2d", stem, prompt, revise: true };
  flatSpecs.add(spec);
  // Not awaited: startJob follows the job until it ends, and the button says "Drawing…" meanwhile.
  void startJob(spec, { path: "/2d/draw", kind: "flat2d", title: `Redrawing ${displayStem(stem)}`, watch: false });
  return { status: "", started: true };
}

export type StepResult = (SheetState & { hear: string; status: string; ok: true }) | { ok: false; status: string };

/** Motion / Pieces / Say / Sound: one step at once on the selected part (or the asset row). `onRunning` gets the
 * running text before the POST. */
export async function reviseStep(
  stem: string,
  mode: Exclude<FlatMode, "picture">,
  prompt: string,
  sheet: SheetState,
  audioAsset: AudioRow | null,
  files: (Blob & { name: string })[],
  onRunning?: (text: string) => void,
): Promise<StepResult> {
  if (jobOwns(stem)) return { ok: false, status: "Wait for the job on this asset to finish." };
  onRunning?.(STEP_RUNNING[mode] || "");
  try {
    const part = revisePart(sheet.parts, sheet.index, audioAsset);
    const clip = mode === "motion" && files.length ? await clipFields(files) : null;
    const data = await postSteps(reviseBody(stem, mode, prompt, part, clip));
    return { ok: true, ...applySteps(sheet, mode, prompt, part.id, data) };
  } catch (err) {
    return { ok: false, status: err instanceof Error ? err.message : String(err) };
  }
}

/** The panel's Speak ("say") and Sound buttons: the asset row. Sound waits for a job on this asset; Speak does not
 * (legacy). `onRunning` gets "Speaking." / "Writing the sound." before the POST. */
export async function panelAudio(
  stem: string,
  step: "say" | "sound",
  text: string,
  speaker: string,
  onRunning?: (text: string) => void,
): Promise<{ rel: string; status: string }> {
  if (!stem) return { rel: "", status: "Open an asset." };
  if (step === "sound" && jobOwns(stem)) return { rel: "", status: "Wait for the job on this asset to finish." };
  onRunning?.(step === "say" ? "Speaking." : "Writing the sound.");
  try {
    return applyAudio(step, await postSteps(audioBody(stem, step, text, speaker)));
  } catch (err) {
    return { rel: "", status: err instanceof Error ? err.message : String(err) };
  }
}
