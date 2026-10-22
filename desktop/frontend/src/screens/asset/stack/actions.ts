// The Stack stage's one write: Paint's Save (legacy paintSaveBtn, index.html 5488-5515). Side effects only through
// the injectable deps, so the tests pass fakes.
import { useJobs } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { postStackSheet } from "./api";

/** /paint/* has no busy check on the server (unlike POST /stack's 409), so a running job on the asset blocks Save. */
export const WAIT_FOR_JOB = "Wait for the job on this asset to finish.";

/** A job on this asset is running (legacy jobOwns; flat/actions.ts has the same check for 2D). */
export function jobOwns(stem: string): boolean {
  const j = useJobs.getState().job;
  return !!(j && j.state === "running" && j.stem === stem);
}

export interface SaveDeps {
  owns: (stem: string) => boolean;
  post: (stem: string, png: Blob) => Promise<unknown>;
  toast: (text: string) => void;
}

const saveDeps: SaveDeps = { owns: jobOwns, post: postStackSheet, toast: (t) => toast(t) };

/**
 * Save: the sheet as it is painted (and reduced), at native resolution, to POST /paint/stack. Errors are a toast,
 * as legacy. There is no mesh on this stage, so no /paint/mesh. Resolves true when the sheet was written.
 */
export async function saveStackPaint(stem: string, sheet: () => Promise<Blob | null>, deps: SaveDeps = saveDeps): Promise<boolean> {
  if (deps.owns(stem)) {
    deps.toast(WAIT_FOR_JOB);
    return false;
  }
  try {
    const png = await sheet();
    if (!png) return false;
    await deps.post(stem, png);
    return true;
  } catch (err) {
    deps.toast((err as Error)?.message || String(err));
    return false;
  }
}
