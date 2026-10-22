// Where a rig job's finish goes (legacy finishJob's rig branch, index.html 6560-6606: the "Rig ready: …" toast,
// its "Inspect" / "Export →" / "Open" actions). Home's chain.ts skips rig and render jobs ("the viewer owns rig
// jobs"); this is that owner. Registered once at module scope, imported by App.tsx, so it hears a rig job end on
// any screen. A failed or cancelled rig stays on the chip and the job card, as legacy; a sheet render's result is
// the Sheet tab's (./actions.ts renderSheet), so its job is only cleared here.
import { readOutJson, type RigDoc } from "@/lib/api";
import { openAsset } from "@/app/open";
import { parseRoute } from "@/app/router";
import { degradedLine, displayStem, onJobEnded, useJobs, type Job } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { useStage } from "@/screens/asset/stage";
import type { RigSummary } from "@/viewer/useAssetDocs";
import { inspectRig } from "./actions";
import { finishAction, inspectKey, psxFacts, rigFacts, rigReadyText } from "./rig";

/** The jobs this file owns: exactly the ones chain.ts skips, so none is finished twice or not at all. */
export function isRigJob(job: Pick<Job, "kind">): boolean {
  return job.kind === "rig" || job.kind === "render";
}

/** What finishRig reads and does, injectable for the tests. */
export interface FinishDeps {
  read: <T>(stem: string, rel: string) => Promise<T | null>;
  /** The asset on screen ("" on Home or Settings). */
  onAsset: () => string;
  open: (stem: string, stage: "rig" | "export") => void;
  enter: (stage: "rig" | "export") => void;
  inspect: (stem: string, key: string) => void;
  toast: (text: string, actLabel?: string, act?: () => void) => void;
}

const deps0: FinishDeps = {
  read: (stem, rel) => readOutJson(stem, rel, Date.now()),
  onAsset: () => currentAsset(),
  open: (stem, stage) => openAsset(stem, stage),
  enter: (stage) => useStage.getState().enter(stage),
  inspect: inspectRig,
  toast,
};

/** The asset the screen shows: the stage store's stem while an asset route is open. */
function currentAsset(): string {
  if (typeof location === "undefined") return "";
  const r = parseRoute(location.search, location.hash);
  return r.name === "asset" && r.stem === useStage.getState().stem ? r.stem : "";
}

/** Drop the job from the chip when it is still the one there and has ended. */
function dropJob(job: Job): void {
  const cur = useJobs.getState().job;
  if (cur && cur.id === job.id && cur.state !== "running") useJobs.getState().dismiss();
}

/** A rig or render job ended (ok, error, cancelled or lost). */
export async function finishRig(job: Job, d: FinishDeps = deps0): Promise<void> {
  if (job.kind === "render") {
    dropJob(job);
    return;
  }
  if (job.state !== "ok") return;
  const stem = job.stem;
  const name = displayStem(stem);
  const intent = await d.read<{ output?: string }>(stem, "intent.json");
  // A ps1 asset's facts lead; rig.json's replace them when it can be read (legacy kept the psx ones otherwise).
  let facts = intent?.output === "ps1" ? psxFacts(await d.read<{ triangles?: number; atlas?: number }>(stem, "psx/psx.json")) : "";
  const r = await d.read<RigDoc>(stem, "rig/rig.json");
  if (r) facts = rigFacts(r);
  const summary = await d.read<RigSummary>(stem, "rig/summary.json");
  const text = rigReadyText(name, facts, summary?.line || "", degradedLine(job.degraded));
  const watching = useJobs.getState().watching;
  const here = d.onAsset() === stem;
  const action = finishAction({ here: here || watching, level: summary?.level });
  if (action === "Open") {
    d.toast(text, "Open", () => {
      dropJob(job);
      d.open(stem, "rig");
    });
    return;
  }
  // You were on this asset (or watching the job card): show the rig now, then offer what is next.
  dropJob(job);
  if (here) d.enter("rig");
  else d.open(stem, "rig");
  if (action === "Inspect") d.toast(text, "Inspect", () => d.inspect(stem, inspectKey(summary?.problems as never)));
  else d.toast(text, "Export →", () => (d.onAsset() === stem ? d.enter("export") : d.open(stem, "export")));
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
