// What the viewer shares with its HUD and its panels: the load state, the view toggles, and the props every
// panel gets (ViewerPanelProps). A panel owns its own file and gets everything from these props, the tool API
// (src/viewer/tools.ts) and the stores (useFlow, useStage, useJobs, useUi, usePanels).
import type { Lod } from "@/lib/api";
import type { StageProps } from "@/screens/asset/types";
import type { AssetDocs } from "@/viewer/useAssetDocs";
import type { ViewerTools } from "@/viewer/tools";
import type { ViewerStage } from "./panels/panels";

export interface ModelInfo {
  url: string;
  /** The last path segment ("rigged_allclips.glb"); currentFile() has the path under out/<stem>/. */
  file: string;
  lod: string;
  tag: string;
  ms: number;
  vertices: number;
  faces: number;
  hasMap: boolean;
}

export type LoadState = { state: "loading" } | { state: "ready"; info: ModelInfo } | { state: "error"; message: string };

/** The view toggles and the clip player, for panels that follow or drive them (the Rig check pauses Spin). */
export interface ViewerView {
  clips: string[];
  clip: string | null;
  onClip: (name: string) => void;
  playing: boolean;
  onPlaying: (on: boolean) => void;
  spin: boolean;
  onSpin: (on: boolean) => void;
  clay: boolean;
  onClay: (on: boolean) => void;
  wire: boolean;
  onWire: (on: boolean) => void;
}

/** Everything a viewer panel gets. The stage props come straight from AssetScreen. */
export interface ViewerPanelProps extends Omit<StageProps, "stage"> {
  stage: ViewerStage;
  /** The DOM id the toggle's aria-controls points at (panelDomId). */
  id: string;
  /** Panels stay mounted while closed (an armed tool or an undo stack can outlive a collapse); open says if shown. */
  open: boolean;
  /** Close this panel; focus goes back to its toggle when it was inside. */
  onClose: () => void;
  docs: AssetDocs | null;
  load: LoadState;
  /** lods.json minus stack and pixel* (viewableLods). */
  lods: Lod[];
  /** Show another LOD (the URL's ?lod=, as the HUD picker does). */
  onPickLod: (id: string) => void;
  /** Re-read the docs (lods.json, rig/*.json, inspect.json …) and reload what is on screen. */
  reloadDocs: () => void;
  view: ViewerView;
  tools: ViewerTools;
}

/** The panel props before the HUD adds the per-panel id, open and onClose. */
export type ViewerPanelBase = Omit<ViewerPanelProps, "id" | "open" | "onClose">;
