// One open asset: the stage the rail picked, and the versions strip under it (not on Ideas). 3D and Rig are the
// clay viewer; Ideas, Concept/2D, Stack and Export are their own files under ./<stage>/. Ideas and Export are
// full pages; every other stage keeps the strip at the bottom.
import { useEffect } from "react";
import { onJobFinished } from "@/lib/jobs";
import { useFlow, type Stage } from "@/lib/flow";
import { ViewerScreen } from "@/screens/viewer/ViewerScreen";
import { ExportStage } from "./export/ExportStage";
import { FlatStage } from "./flat/FlatStage";
import { IdeasStage } from "./ideas/IdeasStage";
import { openStem, useStage } from "./stage";
import { StackStage } from "./stack/StackStage";
import { TakesStrip } from "./TakesStrip";
import type { StageProps } from "./types";

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
  return (
    <div className="absolute inset-0 flex flex-col">
      <div className="relative min-h-0 flex-1">
        <StageBody {...props} lod={lod} />
      </div>
      {stage !== "ideas" ? <TakesStrip key={stem} {...props} /> : null}
    </div>
  );
}

function StageBody(props: StageProps & { lod?: string }) {
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
      return <ViewerScreen key={props.stem} {...props} lod={props.stage === "rig" ? (props.lod ?? "rigged") : props.lod} />;
  }
}
