// The Paint panel's pure logic and copy (legacy paint cubby, index.html 2106-2128; hexToRgb .. stampUV and the
// drop pick, 5371-5452, 5542-5562). Word for word from the legacy page. PaintPanel.tsx only lays these out.
// No React, no fetch, no DOM here.

/** The cubby's hint line. */
export const PAINT_HINT = "Paint on the model with a 3D brush. Save keeps it.";

/** Every control's label and tooltip (index.html 2113-2124). */
export const PAINT_COPY = {
  paint: { label: "Brush", title: "Click-drag to paint. Right-drag still orbits." },
  drop: { label: "Drop", title: "Next click samples color from the surface." },
  erase: { label: "Erase", title: "Brush restores the original colors." },
  undo: { label: "Undo", title: "Undo the last stroke." },
  save: { label: "Save", title: "Write painted stack sheet and textured_paint.glb." },
  color: { label: "Color", title: "Brush color" },
  brush: { label: "Size", title: "Paint sphere size, as a fraction of the mesh." },
} as const;

/** #paint-color's default. */
export const PAINT_COLOR = "#2ecc71";

/** #paint-radius: a fraction of the mesh's bounding-box diagonal. */
export const BRUSH = { min: 0.01, max: 0.2, step: 0.005, value: 0.04 } as const;

/** #paint-radius-val before the slider moves: the markup's "0.04", not toFixed(3). */
export const BRUSH_TEXT = "0.04";

/** The hover ball (paintBall): a fixed green, not the brush colour. */
export const BALL = { color: 0x2ecc71, opacity: 0.28, widthSegments: 20, heightSegments: 12 } as const;

/** The server has no busy check on /paint/* (factory_http.py 1631-1670), so Save waits for a job on this asset. */
export const PAINT_BUSY = "Wait for the job on this asset to finish.";

/** The LOD paint_save.py puts first in lods.json ({id: "paint", file: "textured_paint.glb", label: "Painted"}). */
export const PAINTED_LOD = "paint";

/** The readout after the slider moves (legacy: Number(value).toFixed(3), so 0.045, 0.040). */
export function brushText(value: number | string): string {
  return Number(value).toFixed(3);
}

/** "#2ecc71" to [46, 204, 113] (legacy hexToRgb). */
export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(String(hex).replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** [46, 204, 113] to "#2ecc71" (legacy rgbToHex). */
export function rgbToHex(r: number, g: number, b: number): string {
  return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("");
}

/** The brush radius in world units (legacy worldPaintRadius: an empty model counts as a diagonal of 1). */
export function worldPaintRadius(diagonal: number, fraction: number): number {
  return (diagonal || 1) * Number(fraction);
}

/** The stamp's radius in texture pixels (legacy stampUV): at least 1.5 px. */
export function pixelRadius(radiusWorld: number, diagonal: number, width: number): number {
  return Math.max(1.5, (radiusWorld / (diagonal || 1)) * width * 1.6);
}

/** A uv as a point on the texture canvas; glTF maps are not flipped, so v runs down (legacy stampUV). */
export function uvToCanvas(uv: { x: number; y: number }, width: number, height: number, flipY: boolean): { x: number; y: number } {
  return { x: uv.x * width, y: (flipY ? 1 - uv.y : uv.y) * height };
}

/** The pixel the dropper reads (legacy: Math.floor of the canvas point), kept on the canvas. */
export function dropPixel(uv: { x: number; y: number }, width: number, height: number, flipY: boolean): { x: number; y: number } {
  const p = uvToCanvas(uv, width, height, flipY);
  return { x: Math.min(width - 1, Math.max(0, Math.floor(p.x))), y: Math.min(height - 1, Math.max(0, Math.floor(p.y))) };
}

export type PaintKind = "stack" | "mesh";

/** POST /paint/<kind>?stem=<stem> (legacy paint-save click). */
export function paintPath(kind: PaintKind, stem: string): string {
  return `/paint/${kind}?stem=${encodeURIComponent(stem)}`;
}
