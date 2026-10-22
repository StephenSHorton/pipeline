// Where a rig job's finish goes (legacy finishJob's rig branch, index.html 6560-6606: the "Rig ready: …" toast,
// its "Inspect" / "Export →" / "Open" actions, the "It finished, but the model did not load" line). Home's
// chain.ts skips rig and render jobs ("the viewer owns rig jobs"); this is that owner. Registered once at module
// scope, imported by App.tsx, so it hears a rig job end on any screen. Stub: finishRig does nothing yet.
import { onJobEnded, type Job } from "@/lib/jobs";

/** The jobs this file owns: exactly the ones chain.ts skips, so none is finished twice or not at all. */
export function isRigJob(job: Pick<Job, "kind">): boolean {
  return job.kind === "rig" || job.kind === "render";
}

/** A rig or render job ended (ok, error, cancelled or lost). The Rig port fills this in. */
export async function finishRig(job: Job): Promise<void> {
  void job;
}

let wired = false;

/** Listen once. Called below at import; safe to call again. */
export function wireRigFinish(): void {
  if (wired) return;
  wired = true;
  onJobEnded((job) => {
    if (!isRigJob(job)) return;
    void finishRig(job).catch((err) => console.error(err));
  });
}

wireRigFinish();
