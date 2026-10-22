// The Cleanup panel (legacy cleanup cubby, index.html 2090-2104, 5300-5370): Floaters and Carve through POST /clean.
// Stub: the shell only, until the Cleanup port fills it.
import type { ViewerPanelProps } from "../types";
import { ViewerPanel } from "./ViewerPanel";

export function CleanupPanel(p: ViewerPanelProps) {
  return <ViewerPanel panel="cleanup" id={p.id} open={p.open} onClose={p.onClose} />;
}
