// One open asset: the stage the rail picked, and the versions panel (not on Ideas). 3D and Rig are the clay
// viewer; Ideas, Concept/2D, Stack and Export are their own files under ./<stage>/. The versions panel sits on the
// left (VersionsPanel.tsx): the viewer puts it at the foot of its left column, under the asset card and the panel
// rail, so the left side is one stack; Stack floats it at the bottom-left corner of its canvas, clear of the title
// card; Concept/2D (whose sound panel opens at the bottom-left) and Export (a scrolling page) give it its own column
// at the left edge, so it never covers them. Phones: a row under the stage, folded to its heading.
import { useEffect, type ReactNode } from "react";
import { onJobFinished } from "@/lib/jobs";
import { useFlow, type Stage } from "@/lib/flow";
import { cn } from "@/lib/utils";
import { ViewerScreen } from "@/screens/viewer/ViewerScreen";
import { ExportStage } from "./export/ExportStage";
import { FlatStage } from "./flat/FlatStage";
import { IdeasStage } from "./ideas/IdeasStage";
import { openStem, useStage } from "./stage";
import { StackStage } from "./stack/StackStage";
import type { StageProps } from "./types";
import { VersionsPanel } from "./VersionsPanel";

export function AssetScreen({ stem, lod }: { stem: string; lod?: string }) {
  const stage = useStage((s) => (s.stem === stem ? s.stage : null));
  const intent = useStage((s) => (s.stem === stem ? s.intent : null));
  const enter = useStage((s) => s.enter);
  const flow = useFlow((s) => (s.stem === stem ? s.flow : null));
  const flowVersion = useFlow((s) => s.version);

  useEffect(() => {
    void openStem(stem);
  }, [stem]);
  // A job that wrote this asset changes its flow (a new take, a new stage holding files). Not a sheet render: it
  // writes rig/preview/ only, and a flow re-read would reload the model under the Rig check.
  useEffect(() => onJobFinished((j) => j.stem === stem && j.kind !== "render" && void openStem(stem)), [stem]);

  if (!stage) return null;
  const props: StageProps = { stem, stage, flow, flowVersion, intent, enter: (s: Stage) => enter(s) };
  const viewer = stage === "mesh" || stage === "rig";
  // In the viewer's left column the panel may shrink with the rail, so a short stage scrolls its body instead of
  // running it under the bottom pickers (the column is capped above them on phones).
  const versions = stage !== "ideas" ? <VersionsPanel key={stem} {...props} className={viewer ? "min-h-11 shrink" : undefined} /> : null;
  return (
    <div className="absolute inset-0 flex flex-col sm:flex-row">
      {versions && !viewer ? (
        // empty:hidden: the panel draws nothing when it has nothing to say, and its padding goes with it.
        <div
          className={cn(
            "shrink-0 px-3 pt-1 pb-3 empty:hidden max-sm:order-last",
            stage === "stack" ? "sm:pointer-events-none sm:absolute sm:bottom-3 sm:left-3 sm:z-10 sm:p-0" : "sm:p-3 sm:pr-0",
          )}
        >
          {versions}
        </div>
      ) : null}
      <div className="relative min-h-0 min-w-0 flex-1">
        <StageBody {...props} lod={lod} versions={viewer ? versions : null} />
      </div>
    </div>
  );
}

function StageBody({ versions, ...props }: StageProps & { lod?: string; versions: ReactNode }) {
  switch (props.stage) {
    case "ideas":
      return <IdeasStage {...props} />;
    case "concept":
    case "flat":
      return <FlatStage {...props} />;
    case "stack":
      return <StackStage {...props} />;
    case "export":
      return <ExportStage {...props} />;
    case "mesh":
    case "rig":
      // The stage props too, so the viewer's panels can tell 3D from Rig and read the flow.
      return <ViewerScreen key={props.stem} {...props} lod={props.stage === "rig" ? (props.lod ?? "rigged") : props.lod} versions={versions} />;
  }
}
