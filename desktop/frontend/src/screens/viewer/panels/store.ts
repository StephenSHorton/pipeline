// Which viewer panel is open. Module-level, not ViewerScreen state, so code outside React can open one: the
// rig finish toast's "Inspect" (legacy inspectRig, index.html 6593, 6609-6615) opens the Rig check. The viewer
// clears it when it closes, not when it opens, so an open asked for just before the viewer mounts survives.
import { create } from "zustand";
import { togglePanel, type PanelId } from "./panels";

interface PanelStore {
  open: PanelId | null;
  toggle: (id: PanelId) => void;
  openPanel: (id: PanelId) => void;
  close: () => void;
}

export const usePanels = create<PanelStore>((set) => ({
  open: null,
  toggle: (id) => set((s) => ({ open: togglePanel(s.open, id) })),
  openPanel: (id) => set({ open: id }),
  close: () => set({ open: null }),
}));

/** Open one panel from anywhere (closes any other). */
export function openViewerPanel(id: PanelId): void {
  usePanels.getState().openPanel(id);
}
