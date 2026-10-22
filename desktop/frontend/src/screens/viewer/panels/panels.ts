// The viewer's panels: the left rail under the asset card (legacy #cubby-rail-left cubbies Cleanup, Paint,
// Detail levels, index.html 2015, 2090-2142) and the right dock under the camera stack (the Rig check, "a dock
// over the right of the mesh pane in the Rig stage", index.html 584). Legacy openStep (index.html 5817-5829):
// one open panel at a time, and clicking the open one closes it. Not ported: the "Hide X" buttons, "Hide panels"
// and the "Show X" tray (CSS-hidden, index.html 844-847), and hideAllCubbies, which left cubbies hidden for the
// rest of the session once a job started (index.html 3748, 6891). No React here.
import type { Stage } from "@/lib/flow";

export type PanelId = "cleanup" | "paint" | "lods" | "rig";
export type PanelSide = "left" | "right";
/** The stages the viewer draws. */
export type ViewerStage = Extract<Stage, "mesh" | "rig">;

/** The cubby titles, word for word (index.html 2092, 2108, 2131; the Rig check bar, rig-inspect.js 896). */
export const PANEL_TITLES: Record<PanelId, string> = {
  cleanup: "Cleanup",
  paint: "Paint",
  lods: "Detail levels",
  rig: "Rig check",
};

/** The Rig check's ×: title "Close the rig check", aria-label "Close" (rig-inspect.js 896). */
export const RIG_CLOSE_TITLE = "Close the rig check";
export const RIG_CLOSE_LABEL = "Close";

interface PanelDef {
  id: PanelId;
  side: PanelSide;
  stages: ViewerStage[];
}

/** Legacy rail order. The left cubbies sat in the mesh pane, shown on 3D and Rig; the Rig check only on Rig. */
export const PANELS: PanelDef[] = [
  { id: "cleanup", side: "left", stages: ["mesh", "rig"] },
  { id: "paint", side: "left", stages: ["mesh", "rig"] },
  { id: "lods", side: "left", stages: ["mesh", "rig"] },
  { id: "rig", side: "right", stages: ["rig"] },
];

/** The panels on one side for a stage, in rail order. */
export function panelsFor(stage: ViewerStage, side: PanelSide): PanelId[] {
  return PANELS.filter((p) => p.side === side && p.stages.includes(stage)).map((p) => p.id);
}

/** Clicking a panel's toggle: open it (closing any other), or close it when it is the open one (legacy openStep). */
export function togglePanel(open: PanelId | null, id: PanelId): PanelId | null {
  return open === id ? null : id;
}

/** The open panel as this stage sees it: one the stage does not show is closed (legacy: the Rig check off Rig). */
export function openOn(open: PanelId | null, stage: ViewerStage): PanelId | null {
  return open && PANELS.some((p) => p.id === open && p.stages.includes(stage)) ? open : null;
}

/** The close button's aria-label: the Rig check keeps legacy "Close"; the rail panels follow the Audio panel. */
export function closeLabel(id: PanelId): string {
  return id === "rig" ? RIG_CLOSE_LABEL : `Close ${PANEL_TITLES[id]}`;
}

/** The close button's tooltip. */
export function closeTitle(id: PanelId): string {
  return id === "rig" ? RIG_CLOSE_TITLE : `Close ${PANEL_TITLES[id]}`;
}

/** The DOM id a toggle's aria-controls points at. */
export function panelDomId(id: PanelId): string {
  return `viewer-panel-${id}`;
}
