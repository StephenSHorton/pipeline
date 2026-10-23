// The Detail levels panel's pure logic and copy (legacy lods cubby, index.html 2129-2142; lodLabel 4969-4972; the
// list in load() 5249-5262; fillTurnaround 4274-4296). The panel only lists and switches the LODs lods.json already
// has: nothing here builds one (that is generate's lods stage, scripts/lods.py). The list is the finished models only
// (viewableLods: Photo bake, PSX, the app's own saves, plus Rigged), not detail levels: Stephen asked for no hero /
// lookdev / game / proxy versions. LodsPanel.tsx lays these out. No React, no fetch.
import type { Lod } from "@/lib/api";
import { RIGGED } from "@/viewer/candidates";

/** The cubby's hint: what each finished model is (legacy named Hero and Game, which are no longer offered). */
export const LODS_HINT = "Show another finished model. Photo bake is the textured model; PSX is the PS1 low-poly; Rigged has the clips.";

/** The LOD list's aria-label (legacy had a bare div). "Model", as the HUD's old tabs: the region is "Detail levels". */
export const LODS_LIST_LABEL = "Model";

/** The turnaround's aria-label (legacy aside#ref had none). */
export const TURNAROUND_LABEL = "Turnaround";

/** The fallback figure's caption when views.json lists no views (index.html 4284). */
export const INPUT_IMAGE = "Input image";

/** A button's text: the label, then the face count in words ("PSX · 1,184 faces", "Photo bake"). Legacy lodLabel
 *  showed a bare number. */
export function lodLabel(item: Pick<Lod, "label" | "faces">): string {
  const faces = item.faces ? ` · ${Number(item.faces).toLocaleString()} faces` : "";
  return `${item.label}${faces}`;
}

/** A button's tooltip (legacy index.html 5255 named a left pane the app no longer has). */
export function lodTitle(item: Pick<Lod, "id" | "label" | "file">): string {
  return `Show the ${item.label || item.id} model (${item.file}).`;
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
