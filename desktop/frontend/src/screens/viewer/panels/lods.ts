// The Detail levels panel's pure logic and copy (legacy lods cubby, index.html 2129-2142; lodLabel 4969-4972; the
// list in load() 5249-5262; fillTurnaround 4274-4296). The panel only lists and switches the LODs lods.json already
// has: nothing here builds one (that is generate's lods stage, scripts/reduce.sh). LodsPanel.tsx lays these out.
// No React, no fetch.
import type { Lod } from "@/lib/api";
import { RIGGED } from "@/viewer/candidates";

/** The cubby's hint, word for word (index.html 2136). */
export const LODS_HINT = "Show another version of the model. PSX is the PS1 low-poly; Hero is the full sculpt; Game is the reduced one.";

/** The LOD list's aria-label (legacy had a bare div; the buttons need a group name). */
export const LODS_LIST_LABEL = "Detail levels";

/** The turnaround's aria-label (legacy aside#ref had none). */
export const TURNAROUND_LABEL = "Turnaround";

/** The fallback figure's caption when views.json lists no views (index.html 4284). */
export const INPUT_IMAGE = "Input image";

/** A button's text: the label, then the face count with thousands separators ("Game 5,000", "Hero"). Legacy lodLabel. */
export function lodLabel(item: Pick<Lod, "label" | "faces">): string {
  const faces = item.faces ? ` ${Number(item.faces).toLocaleString()}` : "";
  return `${item.label}${faces}`;
}

/** A button's tooltip, word for word (index.html 5255). */
export function lodTitle(item: Pick<Lod, "id" | "label" | "file">): string {
  return `Show ${item.label || item.id} in the left pane (${item.file}).`;
}

/** One view in out/<stem>/views.json. */
export interface TurnaroundView {
  id: string;
  file: string;
  label?: string;
  role?: string;
}

export interface ViewsDoc {
  views?: TurnaroundView[];
}

export interface Figure {
  id: string;
  file: string;
  /** The caption and the img alt: label, else id. */
  caption: string;
  /** role "shape": the view the mesh was built from, drawn with an accent border (index.html 1731). */
  shape: boolean;
}

/** The input picture names the flow versions (scripts/flow.py), in the order the fallback tries them. */
export const INPUT_FILES = ["input.jpg", "input.png", "input.jpeg", "input.webp"] as const;

/**
 * The fallback figure's file. Legacy always used input.jpg, which showed a broken image for a png or webp input;
 * with the flow's file list, the input that exists wins. No list (flow unreadable): input.jpg, as legacy.
 */
export function inputFile(files: ReadonlySet<string> | null): string {
  return (files && INPUT_FILES.find((f) => files.has(f))) || "input.jpg";
}

/** The figures, in views.json order; with no views, the single "Input image" figure (legacy fillTurnaround). */
export function turnaroundFigures(doc: ViewsDoc | null | undefined, input = "input.jpg"): Figure[] {
  const views: TurnaroundView[] = doc?.views?.length ? doc.views : [{ id: "input", file: input, role: "shape", label: INPUT_IMAGE }];
  return views
    .filter((v) => v && v.file)
    .map((v) => ({ id: v.id, file: v.file, caption: v.label || v.id, shape: v.role === "shape" }));
}

/** The Rigged button, kept from the new UI's model tabs (legacy's list had none; see lodButtons). */
export const RIGGED_LABEL = "Rigged";
export const RIGGED_TITLE = "The rigged model, with its clips.";

export interface LodButton {
  id: string;
  label: string;
  title: string;
}

/**
 * The list's buttons: one per LOD in lods.json order, even a single one (legacy had no minimum). Rigged comes
 * first when the asset has a rig: legacy went back to the rigged model on every stage change (preferLod "rigged",
 * index.html 3305-3307), but here the pick is the URL's ?lod=, which a stage change keeps, so without it a LOD
 * picked on Rig would leave no way back to the rigged model and its clips.
 */
export function lodButtons(lods: Lod[], hasRig: boolean): LodButton[] {
  return [
    ...(hasRig ? [{ id: RIGGED, label: RIGGED_LABEL, title: RIGGED_TITLE }] : []),
    ...lods.map((l) => ({ id: l.id, label: lodLabel(l), title: lodTitle(l) })),
  ];
}

/**
 * The pressed button: the one on screen. While the next one loads, the last shown stays pressed (legacy set
 * aria-pressed only after a load succeeded, index.html 5189-5191).
 */
export function pressedLod(buttons: LodButton[], shown: string): string | null {
  return buttons.some((b) => b.id === shown) ? shown : null;
}
