// The Stack stage's factory calls (legacy stack-lab.js load, 1704-1740, and the Paint Save, index.html 5488-5515).
// The sheet image itself is loaded by the view (an <img>), from sheetUrl.
import { enc, factoryJson, outUrl, readOutJson } from "@/lib/api";
import { isCharacterDoc, type CharacterDoc } from "./character/parts";
import type { StackMeta } from "./stack";

/** GET /stack/<stem>, falling back to out/<stem>/stack/stack.json; null when neither has a stack. */
export async function readStackMeta(stem: string, t: number | string = Date.now()): Promise<StackMeta | null> {
  try {
    const r = await fetch(`/stack/${enc(stem)}`);
    if (r.ok) return validMeta(await r.json());
  } catch {
    /* the fallback below */
  }
  return validMeta(await readOutJson<StackMeta>(stem, "stack/stack.json", t));
}

/** A stack.json the view can draw: whole layer and resolution counts, a pitch and the extents. */
function validMeta(m: unknown): StackMeta | null {
  const s = m as StackMeta | null;
  if (!s || !(s.layers > 0) || !(s.resolution > 0) || !(s.pitch > 0) || !Array.isArray(s.extents) || s.extents.length < 3) return null;
  return s;
}

/** The sheet, cache-busted: a paint save, a re-stack or a version pick rewrite the same path (legacy had no ?t=). */
export const sheetUrl = (stem: string, t: number | string) => outUrl(stem, "stack/sheet.png", t);

/** A walking character's character.json and its part strips (RGBA by part name); null when any is missing. */
export async function readCharacter(
  stem: string,
  file: string,
  t: number | string,
): Promise<{ doc: CharacterDoc; strips: Map<string, Uint8ClampedArray> } | null> {
  const doc = await readOutJson<unknown>(stem, `stack/${file}`, t);
  if (!isCharacterDoc(doc)) return null;
  const strips = new Map<string, Uint8ClampedArray>();
  for (const part of doc.parts) {
    const im = new Image();
    im.src = outUrl(stem, `stack/${part.file}`, t);
    try {
      await im.decode();
    } catch {
      return null;
    }
    const c = document.createElement("canvas");
    c.width = im.naturalWidth;
    c.height = im.naturalHeight;
    const ctx = c.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(im, 0, 0);
    strips.set(part.name, ctx.getImageData(0, 0, c.width, c.height).data);
  }
  return { doc, strips };
}

/** POST /paint/stack?stem= with the PNG. The server writes stack/sheet.png and preview.png and autosaves a take;
 * factoryJson re-reads the flow after any /paint/ POST, so the strip shows it. */
export function postStackSheet(stem: string, png: Blob): Promise<unknown> {
  return factoryJson(`/paint/stack?stem=${enc(stem)}`, {
    method: "POST",
    headers: { "Content-Type": "image/png" },
    body: png,
  });
}
