// Jobs Home started, at module scope so they outlive the Home screen: the draw → 3D chain (a description with no
// picture is drawn as a four-view sheet first, then built like legacy buildFromConcept) and the finish toast.
import { flowFiles, listAssets, readIntent as readIntentJson, readOutJson, type Facing, type FacingSource } from "@/lib/api";
import { degradedLine, displayStem, onJobFinished, startJob, useJobs, type Job, type JobOpts } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { jobStage, openAsset } from "@/app/open";
import { buildFromDrawingBody, isThreeD, kindDef, NO_PICTURE, recoverDrawIntent, type NestedIntent } from "./make";

/** Specs Home posted (the store keeps the same object on the job, and Try again reuses it). */
const homeSpecs = new WeakSet<object>();
/** stem → the 3D intent a drawing is for, as picked in this page. Deleted when the draw ends. After a reload it
 * is gone, and out/<stem>/intent.json (which the draw saved) says the same. */
const drawThen3d = new Map<string, NestedIntent>();

export function startHomeJob(spec: Record<string, unknown>, opts: JobOpts): Promise<void> {
  homeSpecs.add(spec);
  return startJob(spec, opts);
}

/** Draw first, then build the 3D kind from the drawing when the draw finishes. */
export function startDrawThen3d(spec: Record<string, unknown>, opts: JobOpts, want: NestedIntent): Promise<void> {
  drawThen3d.set(String(spec.stem), want);
  return startHomeJob(spec, opts);
}

const onHome = () => !new URLSearchParams(location.search).get("asset") && !/^#\/asset\//.test(location.hash);

/** The finished draw's job, ended as an error with a plain line (the card or the chip shows it). */
function failChain(job: Job, error: string) {
  const cur = useJobs.getState().job;
  if (!cur || cur.id !== job.id) return;
  useJobs.setState({ job: { ...cur, state: "error", error, next: undefined } });
}

async function buildFromDrawing(job: Job, want: NestedIntent) {
  const stem = job.stem;
  const name = kindDef(want.output).name;
  // The card says what comes next while the reads run, not "Opening".
  const cur = useJobs.getState().job;
  if (cur && cur.id === job.id) useJobs.setState({ job: { ...cur, next: `Building the ${name}…` } });
  const files = await flowFiles(stem);
  // No picture came out of the draw (a fallback, a failed stage): say so plainly instead of a /generate error.
  if (files && !files.has("flat/source.png")) {
    failChain(job, degradedLine(job.degraded) || NO_PICTURE);
    return;
  }
  const [saved, key] = await Promise.all([
    readIntentJson(stem) as Promise<NestedIntent | null>,
    readOutJson<{ facing?: Facing; canonical?: boolean; facing_source?: FacingSource }>(stem, "flat/key.json"),
  ]);
  const spec = buildFromDrawingBody(stem, want, saved, key);
  await startHomeJob(spec, { title: `Building the ${name} · ${displayStem(stem)}`, watch: useJobs.getState().watching });
}

/** Legacy finishJob facts: ps1 " · N triangles, N px texture". */
async function facts(stem: string, output: string): Promise<string> {
  if (output !== "ps1") return "";
  const p = await readOutJson<{ triangles?: number; atlas?: number }>(stem, "psx/psx.json");
  if (!p || !p.triangles) return "";
  return ` · ${p.triangles.toLocaleString()} triangles${p.atlas ? `, ${p.atlas} px texture` : ""}`;
}

async function finish(job: Job) {
  const stem = job.stem;
  const name = displayStem(stem);
  let text: string;
  if (job.kind === "flat2d") {
    text = `${job.revise ? "Redrawn" : "Sprite ready"}: ${name}`;
  } else if (job.kind === "ideas") {
    return; // screens/asset/ideas/more.ts owns every ideas finish (legacy finishIdeasJob)
  } else if (job.kind === "rig" || job.kind === "render") {
    return; // the viewer owns rig jobs
  } else {
    const intent = (await readIntentJson(stem)) as NestedIntent | null;
    const output = intent?.output;
    // Legacy: "Asset ready" when there is no saved intent.
    text = `${output ? kindDef(output).name : "Asset"} ready: ${name}${output ? await facts(stem, output) : ""}`;
  }
  const deg = degradedLine(job.degraded);
  if (deg) text += ` · ${deg}`;
  const stage = jobStage(job);
  const open = () => {
    const cur = useJobs.getState().job;
    if (cur && cur.id === job.id && cur.state !== "running") useJobs.getState().dismiss();
    openAsset(stem, stage);
  };
  if (useJobs.getState().watching) {
    // You were watching: open it now. The toast then offers what is next (legacy "Export →").
    useJobs.getState().dismiss();
    open();
    toast(text, "Export →", () => openAsset(stem, "export"));
    return;
  }
  // In the background: the chip keeps "<name> is ready · Open" until you open or dismiss it.
  toast(text, "Open", open);
}

/** After a draw: build the 3D kind it was drawn for (this page's pick, else the intent the draw saved), unless the
 * asset already has a mesh. Otherwise it was a sprite: the plain finish. */
async function afterDraw(job: Job) {
  const known = drawThen3d.get(job.stem);
  drawThen3d.delete(job.stem);
  let want = known;
  if (!want && !job.revise) {
    const saved = (await readIntentJson(job.stem)) as NestedIntent | null;
    if (saved && isThreeD(saved.output)) {
      const a = (await listAssets().catch(() => null))?.assets.find((x) => x.id === job.stem);
      if (a && !a.has_mesh) want = recoverDrawIntent(saved, job.stem);
    }
  }
  if (want) return buildFromDrawing(job, want);
  return finish(job);
}

onJobFinished((job) => {
  const mine = !!job.spec && homeSpecs.has(job.spec);
  // A job the page picked back up after a reload has no spec. A draw's build follows wherever you are; anything
  // else gets its toast while you are on Home.
  const reattached = job.spec === undefined;
  if (!mine && !(reattached && (job.kind === "flat2d" || onHome()))) return;
  const run = job.kind === "flat2d" ? afterDraw(job) : finish(job);
  void run.catch((err) => {
    if (job.kind === "flat2d") failChain(job, `Could not start the build: ${err instanceof Error ? err.message : err}`);
    else console.error(err);
  });
});
