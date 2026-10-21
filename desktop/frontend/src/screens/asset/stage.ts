// The open asset's stage (legacy flowStage / workflow / data-flat-mode). openStem resets it for a new stem, asks
// /control/workflow (2d, 3d, or "" to keep the current one), reads the intent and the flow, then picks the
// default stage the legacy refreshFlow picked. In memory only: the query keeps just ?asset= and ?lod=.
import { create } from "zustand";
import { getWorkflow, readIntent, type Intent } from "@/lib/api";
import { defaultStage, STAGE_VIEW, useFlow, type FlatMode, type Stage, type Workflow } from "@/lib/flow";

interface StageStore {
  stem: string;
  stage: Stage | null;
  workflow: Workflow | "";
  flatMode: FlatMode;
  intent: (Intent & { made?: number }) | null;
  enter: (stage: Stage) => void;
  setFlatMode: (mode: FlatMode) => void;
  setIntent: (intent: (Intent & { made?: number }) | null) => void;
}

export const useStage = create<StageStore>((set) => ({
  stem: "",
  stage: null,
  workflow: "",
  flatMode: "picture",
  intent: null,
  enter(stage) {
    const view = STAGE_VIEW[stage];
    set((s) => ({ stage, workflow: view.workflow ?? s.workflow, flatMode: view.mode ?? s.flatMode }));
  },
  setFlatMode: (flatMode) => set({ flatMode }),
  setIntent: (intent) => set({ intent }),
}));

/** Open (or re-open) a stem. `stage` forces a stage (a rig job opens on Rig, a used idea on Concept). */
export async function openStem(stem: string, stage?: Stage): Promise<void> {
  const st = useStage.getState();
  if (st.stem !== stem) useStage.setState({ stem, stage: null, flatMode: "picture", intent: null });
  if (stage) useStage.getState().enter(stage);
  const [wf, intent, flow] = await Promise.all([
    getWorkflow(stem).catch(() => null),
    readIntent(stem).catch(() => null),
    useFlow.getState().load(stem),
  ]);
  if (useStage.getState().stem !== stem) return;
  const workflow = wf?.workflow ? wf.workflow : useStage.getState().workflow;
  useStage.setState({ workflow, intent });
  if (!useStage.getState().stage) useStage.getState().enter(defaultStage(flow, workflow));
}

/** Re-read the intent (after Build saves a new output). */
export async function reloadIntent(stem: string): Promise<void> {
  const intent = await readIntent(stem).catch(() => null);
  if (useStage.getState().stem === stem) useStage.setState({ intent });
}
