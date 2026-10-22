// The Paint panel (legacy paint cubby, index.html 2106-2128, 5379-5541): the 3D brush and Save through /paint/*.
// Stub: the shell only, until the Paint port fills it.
import type { ViewerPanelProps } from "../types";
import { ViewerPanel } from "./ViewerPanel";

export function PaintPanel(p: ViewerPanelProps) {
  return <ViewerPanel panel="paint" id={p.id} open={p.open} onClose={p.onClose} />;
}
