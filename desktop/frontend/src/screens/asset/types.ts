// The contract between the asset screen and each stage. A stage owns its own file(s) under ./<stage>/ and
// gets everything it needs from these props and the stores (useFlow, useStage, useJobs, useUi).
import type { Intent } from "@/lib/api";
import type { FlowState, Stage } from "@/lib/flow";

export interface StageProps {
  stem: string;
  stage: Stage;
  flow: FlowState | null;
  /** Bumped whenever the flow is re-read (a pick, a snapshot, a 2D write): re-read your files. */
  flowVersion: number;
  intent: (Intent & { made?: number }) | null;
  /** Go to another stage (Use an idea → concept; Build → mesh; a finished rig → rig). */
  enter: (stage: Stage) => void;
}
