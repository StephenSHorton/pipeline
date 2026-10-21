// More ideas (legacy drawIdeas / followIdeas / finishIdeasJob): a background job that the chip and the grid follow,
// no overlay. Every Ideas job ends here, wherever it started (Home, this grid, or picked back up after a reload):
// watching opens it on Ideas; on this asset the chip goes and the grid reloads itself; anywhere else a toast offers
// to open them. Home's finish handler (screens/home/chain.ts) must leave ideas jobs alone (see the slice notes).
import { openAsset } from "@/app/open";
import { parseRoute } from "@/app/router";
import { displayStem, onJobFinished, startJob, useJobs } from "@/lib/jobs";
import { toast } from "@/lib/ui";
import { finishAction, MORE_COUNT, setIdeasDraft, shouldFollow } from "./ideas";

export function startMoreIdeas(stem: string, prompt: string, count = MORE_COUNT): Promise<void> {
  return startJob({ kind: "ideas", stem, prompt, count }, { path: "/2d/draw", kind: "ideas", title: "Drawing ideas", watch: false });
}

const routeStem = () => {
  const r = parseRoute(location.search, location.hash);
  return r.name === "asset" ? r.stem : null;
};

/** Jobs that already handed over from the overlay to the grid. */
const opened = new Set<string>();

// Legacy followIdeas: the first picture of a watched Ideas job (one started on Home) opens the grid, which then
// follows the rest of the job like one started here.
useJobs.subscribe((s) => {
  const j = s.job;
  if (!s.watching || j?.kind !== "ideas") return; // every poll lands here: skip the route read
  if (!shouldFollow(j, s.watching, routeStem(), opened)) return;
  opened.add(j.id);
  if (j.prompt) setIdeasDraft(j.stem, j.prompt);
  useJobs.getState().setWatching(false);
  openAsset(j.stem, "ideas");
});

onJobFinished((job) => {
  if (job.kind !== "ideas") return;
  const stem = job.stem;
  const jobs = useJobs.getState();
  const act = finishAction(jobs.watching, routeStem(), stem);
  if (act === "drop") {
    jobs.dismiss();
    return;
  }
  if (act === "open") {
    if (job.prompt) setIdeasDraft(stem, job.prompt);
    jobs.dismiss();
    openAsset(stem, "ideas");
    return;
  }
  toast(`Ideas ready: ${displayStem(stem)}`, "Open", () => {
    const cur = useJobs.getState().job;
    if (cur && cur.id === job.id && cur.state !== "running") useJobs.getState().dismiss();
    openAsset(stem, "ideas");
  });
});
