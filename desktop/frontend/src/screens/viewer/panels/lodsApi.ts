// The Detail levels panel's reads: the turnaround's views.json (legacy fillTurnaround, index.html 4274-4296). The
// LOD list itself comes from the viewer's docs (lods.json, useAssetDocs), so it is not fetched again here.
import { flowFiles, outUrl, readOutJson } from "@/lib/api";
import { inputFile, turnaroundFigures, type Figure, type ViewsDoc } from "./lods";

/** A figure with its image URL. */
export interface ShownFigure extends Figure {
  src: string;
}

/**
 * The turnaround for a stem. views.json is always asked for, as legacy did: no flow stage lists a top-level
 * views.json (scripts/flow.py), so gating on the flow would hide an older asset's views. A miss gives the
 * "Input image" figure, from the input the flow lists. Image URLs carry no cache-bust, as legacy (the Go file
 * server sends no-store).
 */
export async function readTurnaround(stem: string): Promise<ShownFigure[]> {
  const [doc, files] = await Promise.all([readOutJson<ViewsDoc>(stem, "views.json"), flowFiles(stem)]);
  return turnaroundFigures(doc, inputFile(files)).map((f) => ({ ...f, src: outUrl(stem, f.file, false) }));
}
