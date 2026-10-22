// The Rig check dock (legacy rig-inspect.js panel, opened from the Rig stage's status button, index.html
// 2954-2966, 3175-3196). RigToggle is the dock's button: legacy labels it with the worst check ("Rig looks right"
// on pass), so it lives here with the panel. Stub: the shell and a plain "Rig check" toggle, until the Rig port.
import { PanelToggle, type PanelToggleProps } from "@/screens/viewer/panels/PanelToggle";
import { ViewerPanel } from "@/screens/viewer/panels/ViewerPanel";
import type { ViewerPanelProps } from "@/screens/viewer/types";

export function RigPanel(p: ViewerPanelProps) {
  return <ViewerPanel panel="rig" id={p.id} open={p.open} onClose={p.onClose} />;
}

export function RigToggle(p: Omit<PanelToggleProps, "panel"> & { props: ViewerPanelProps }) {
  return <PanelToggle panel="rig" open={p.open} controls={p.controls} onToggle={p.onToggle} ref={p.ref} title={p.title} className={p.className} />;
}
