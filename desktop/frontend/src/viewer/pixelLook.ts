// The PS1/pixel look state, shared with the legacy page: localStorage "pipeline.pixelLook" (pixel-look.js
// bindPixelLookUi). Only on / pixelSize / lines are used here; the other legacy fields are kept as they are.
// ?game=0 turns it off for this load, any other ?game= turns it on.

export interface PixelLook {
  on: boolean;
  /** Screen pixels per game pixel, 1..8 (default 4). */
  pixelSize: number;
  /** A fixed internal height (240 = the PS1 preset); 0 = follow pixelSize. */
  lines: number;
}

const KEY = "pipeline.pixelLook";
export const PIXEL_DEFAULT: PixelLook = { on: false, pixelSize: 4, lines: 0 };

function readRaw(): Record<string, unknown> {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || "{}");
    return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

export function readPixelLook(search = typeof location !== "undefined" ? location.search : ""): PixelLook {
  const raw = readRaw();
  const px = Number(raw.pixelSize);
  const out: PixelLook = {
    on: raw.on === true,
    pixelSize: Number.isFinite(px) ? Math.max(1, Math.min(8, Math.round(px))) : PIXEL_DEFAULT.pixelSize,
    lines: Number(raw.lines) === 240 ? 240 : 0,
  };
  const game = new URLSearchParams(search).get("game");
  if (game !== null) out.on = game !== "0";
  return out;
}

export function writePixelLook(p: PixelLook): void {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...readRaw(), ...p }));
  } catch {
    /* private window or blocked storage: the look still works for this session */
  }
}

/** The PS1 240p button (pixel-look.js:396-401): on → back to the pixel look; off → the 240-line preset. */
export function toggle240(p: PixelLook): PixelLook {
  return p.on && p.lines === 240 ? { ...p, lines: 0 } : { ...p, on: true, lines: 240 };
}

/** Internal render size: lines tall (or height/px, at least 32), width matched to the canvas aspect. */
export function pixelTarget(p: PixelLook, width: number, height: number): { w: number; h: number } {
  const h = Math.max(1, p.lines || Math.max(32, Math.floor(height / p.pixelSize)));
  const w = Math.max(1, Math.round((h * width) / Math.max(1, height)));
  return { w, h };
}
