// What the Rig stage's buttons do (legacy rigAsset, recheckRig, renderRigSheet, inspectRig; index.html 3203-3234,
// 4793-4807, 6609-6615). Side effects only through startJob, toast, the flow store and ./api.ts, so the tests mock
// those. Rig it and the sheet render are jobs in the factory slot: the server answers 409 when it is taken, and
// startJob then follows the running job, so neither needs a jobOwns gate.
import { displayStem, onJobEnded, startJob, type Job } from "@/lib/jobs";
import { useFlow } from "@/lib/flow";
import { toast } from "@/lib/ui";
import { useStage } from "@/screens/asset/stage";
import { openViewerPanel, usePanels } from "@/screens/viewer/panels/store";
import { checkSheet, flowFileSet, loadRigCheck, postRecheck, useRig } from "./api";
import { choicesOf, planOptions, renderPath, rigBody, rigKeptHead, rigPath, rigTitle, shownChoice } from "./rig";

/**
 * Rig it / Re-rig, from the bar or from the panel (Rig it as made, Straighten and rig). Both send the plan the select
 * shows and this asset's Advanced flags: legacy's panel sent "" (auto) until a plan was picked by hand.
 */
export function rigAsset(stem: string, opts: { straighten?: string } = {}): void {
  const st = useRig.getState();
  const choices = choicesOf(st.choices, stem);
  const check = st.check && st.check.stem === stem ? st.check : null;
  const cls = shownChoice(choices.choice, check?.made ?? null, planOptions(st.plans));
  const body = rigBody(cls, choices.advanced, { straighten: opts.straighten, keptHead: rigKeptHead(check?.doc?.rig?.rest) });
  const spec = { ...body, stem };
  // Not awaited: startJob follows the job until it ends; the chip shows it meanwhile (the job card lives on Home).
  void startJob(spec, { path: rigPath(stem), kind: "rig", rig: true, title: rigTitle(displayStem(stem)), watch: false });
}

/**
 * Check this version / Measure (legacy recheckRig): POST /rig/<stem>/inspect, then the flow and the checks again.
 * An error is a toast and the reads still run, as legacy. With the panel open, it opens again on the new checks.
 */
export async function recheckRig(stem: string): Promise<void> {
  if (!stem || useRig.getState().checking) return;
  const wasOpen = usePanels.getState().open === "rig";
  useRig.setState({ checking: stem });
  try {
    await postRecheck(stem);
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err));
  } finally {
    useRig.setState({ checking: "" });
  }
  if (useStage.getState().stem !== stem) return;
  const flow = await useFlow.getState().load(stem);
  if (wasOpen && !useRig.getState().focus) useRig.setState({ focus: { stem, key: "" } });
  await loadRigCheck(stem, () => useStage.getState().stem === stem, flowFileSet(flow));
}

/**
 * Render the sheet (legacy renderRigSheet): POST /rig/<stem>/render {job: "default"} as a job in the slot. The
 * chip follows it; the Sheet tab says Rendering… and shows an error the way legacy did ("The render <state>").
 */
export async function renderSheet(stem: string): Promise<void> {
  const st = useRig.getState().sheet;
  if (st.busy) return;
  useRig.setState({ sheet: { ...st, stem, busy: true, error: "" } });
  const spec = { job: "default", stem };
  // How this render ended: the job leaves the chip (finish.ts) before startJob resolves, so listen for it.
  let ended: Job | null = null;
  const off = onJobEnded((j) => {
    if (j.spec === spec) ended = j;
  });
  let error = "";
  try {
    await startJob(spec, { path: renderPath(stem), kind: "render", title: `Rendering the sheet for ${displayStem(stem)}`, watch: false });
    const j = ended as Job | null;
    // A 409 followed someone else's job instead: nothing of ours to report.
    if (j && j.state !== "ok") error = j.error || `The render ${j.state}`;
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  } finally {
    off();
  }
  useRig.setState((s) => ({ sheet: { ...s.sheet, busy: false, ok: null, error } }));
  if (useStage.getState().stem === stem) await checkSheet(stem);
}

/**
 * Open the Rig check on one finding (legacy inspectRig): on the Rig stage of this asset with its checks loaded it
 * opens now; otherwise it waits in the store and the panel opens itself once the checks for `stem` have loaded.
 */
export function inspectRig(stem: string, key: string): void {
  useRig.setState({ focus: { stem, key } });
  const stage = useStage.getState();
  if (stage.stem === stem && stage.stage !== "rig") stage.enter("rig");
  const check = useRig.getState().check;
  if (stage.stem === stem && check && check.stem === stem && check.doc) openViewerPanel("rig");
}
