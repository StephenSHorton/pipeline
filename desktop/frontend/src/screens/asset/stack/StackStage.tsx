import type { StageProps } from "../types";
import { StageStub } from "../StageStub";

export function StackStage(props: StageProps) {
  return <StageStub {...props} what="the sprite stack" />;
}
