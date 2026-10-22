// The Paint panel's factory call (legacy paint-save click, index.html 5488-5514): the PNG goes up raw, not as JSON.
// factoryJson re-reads the flow after any /paint/ POST (legacy queueFlowRefresh), so nothing here reloads it.
import { factoryJson } from "@/lib/api";
import { paintPath, type PaintKind } from "./paint";

/** What paint_save.py prints: stack/sheet.png (with layers, resolution) or textured_paint.glb (with source). */
export interface PaintSaved {
  ok: boolean;
  file?: string;
  stem?: string;
  layers?: number;
  resolution?: number;
  source?: string;
}

/** POST /paint/<kind>?stem=<stem> with the PNG as the body. Throws the server's error text. */
export function postPaint(kind: PaintKind, stem: string, png: Blob): Promise<PaintSaved> {
  return factoryJson<PaintSaved>(paintPath(kind, stem), { method: "POST", headers: { "Content-Type": "image/png" }, body: png });
}
