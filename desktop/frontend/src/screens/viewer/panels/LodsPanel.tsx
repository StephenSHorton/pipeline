// The Detail levels panel (legacy lods cubby, index.html 2129-2142, 4273-4296, 5240-5262): the LOD list and the
// turnaround figures. Stub: the shell only, until the Detail levels port fills it.
import type { ViewerPanelProps } from "../types";
import { ViewerPanel } from "./ViewerPanel";

export function LodsPanel(p: ViewerPanelProps) {
  return <ViewerPanel panel="lods" id={p.id} open={p.open} onClose={p.onClose} />;
}
