// Opening an asset: the asset screen picks its stage (Ideas, Concept, 2D, 3D, ...) from the flow.
import type { Stage } from "@/lib/flow";
import { useJobs, type Job } from "@/lib/jobs";
import { openStem } from "@/screens/asset/stage";
import { navigate } from "./router";

/** Open a stem; `stage` forces its stage (a rig job opens on Rig, a 2D job on its sheet). */
export function openAsset(stem: string, stage?: Stage): void {
  navigate({ name: "asset", stem });
  void openStem(stem, stage);
}

/** The stage a finished job opens on: a rig on Rig, a sprite draw on the 2D sheet, ideas on Ideas. */
export function jobStage(j: Pick<Job, "kind" | "spec">): Stage | undefined {
  if (j.kind === "rig") return "rig";
  if (j.kind === "ideas") return "ideas";
  if (j.kind === "flat2d") return (j.spec as { intent?: { output?: string } } | undefined)?.intent?.output === "sprite" ? "flat" : "concept";
  return undefined;
}

/**
 * The job chip (legacy onJobChip). Running or failed: show the job card, which lives on Home. Finished: open
 * what it made and drop the job. Cancelled: drop it.
 */
export function onJobChip(): void {
  const { job, setWatching, dismiss } = useJobs.getState();
  if (!job) return;
  if (job.state === "running" || job.state === "error" || job.next) {
    navigate({ name: "home" });
    setWatching(true);
    return;
  }
  dismiss();
  if (job.state === "ok") openAsset(job.stem, jobStage(job));
}
