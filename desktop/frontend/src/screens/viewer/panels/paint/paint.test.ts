// The Paint panel's copy and maths, word for word from the legacy paint cubby (index.html 2106-2128, 5371-5452).
// No Python test pins this cubby; these strings are the markup's.
import { describe, expect, it } from "vitest";
import {
  BALL,
  BRUSH,
  BRUSH_TEXT,
  PAINT_BUSY,
  PAINT_COLOR,
  PAINT_COPY,
  PAINT_HINT,
  PAINTED_LOD,
  brushText,
  dropPixel,
  hexToRgb,
  paintPath,
  pixelRadius,
  rgbToHex,
  uvToCanvas,
  worldPaintRadius,
} from "./paint";

describe("copy", () => {
  it("the hint, labels and tooltips (plain English; the mode is Brush, not the panel's name)", () => {
    expect(PAINT_HINT).toBe("Paint on the model with a 3D brush. Save keeps it.");
    expect(PAINT_COPY).toEqual({
      paint: { label: "Brush", title: "Click-drag to paint. Right-drag still orbits." },
      drop: { label: "Drop", title: "Next click samples color from the surface." },
      erase: { label: "Erase", title: "Brush restores the original colors." },
      undo: { label: "Undo", title: "Undo the last stroke." },
      save: { label: "Save", title: "Write painted stack sheet and textured_paint.glb." },
      color: { label: "Color", title: "Brush color" },
      brush: { label: "Size", title: "Paint sphere size, as a fraction of the mesh." },
    });
  });

  it("the defaults: #2ecc71, a 0.01-0.2 slider at 0.04 reading 0.04, the green see-through ball", () => {
    expect(PAINT_COLOR).toBe("#2ecc71");
    expect(BRUSH).toEqual({ min: 0.01, max: 0.2, step: 0.005, value: 0.04 });
    expect(BRUSH_TEXT).toBe("0.04");
    expect(BALL).toMatchObject({ color: 0x2ecc71, opacity: 0.28, widthSegments: 20, heightSegments: 12 });
  });

  it("the job gate and the Painted LOD", () => {
    expect(PAINT_BUSY).toBe("Wait for the job on this asset to finish.");
    expect(PAINTED_LOD).toBe("paint");
  });

  it("the readout after a move is toFixed(3)", () => {
    expect(brushText("0.045")).toBe("0.045");
    expect(brushText(0.04)).toBe("0.040");
    expect(brushText("0.2")).toBe("0.200");
  });
});

describe("maths", () => {
  it("hex and rgb round-trip", () => {
    expect(hexToRgb("#2ecc71")).toEqual([46, 204, 113]);
    expect(hexToRgb("000000")).toEqual([0, 0, 0]);
    expect(rgbToHex(46, 204, 113)).toBe("#2ecc71");
    expect(rgbToHex(0, 5, 255)).toBe("#0005ff");
  });

  it("the world radius is diagonal x fraction, with an empty model counted as 1", () => {
    expect(worldPaintRadius(10, 0.04)).toBeCloseTo(0.4);
    expect(worldPaintRadius(0, 0.04)).toBeCloseTo(0.04);
  });

  it("the stamp is (radius / diagonal) x width x 1.6 pixels, at least 1.5", () => {
    expect(pixelRadius(0.4, 10, 1024)).toBeCloseTo(0.04 * 1024 * 1.6);
    expect(pixelRadius(0.001, 10, 64)).toBe(1.5);
    expect(pixelRadius(0.04, 0, 100)).toBeCloseTo(6.4);
  });

  it("uv to canvas: v runs down unless the texture is flipped", () => {
    expect(uvToCanvas({ x: 0.25, y: 0.25 }, 100, 200, false)).toEqual({ x: 25, y: 50 });
    expect(uvToCanvas({ x: 0.25, y: 0.25 }, 100, 200, true)).toEqual({ x: 25, y: 150 });
  });

  it("the dropper floors to a pixel on the canvas", () => {
    expect(dropPixel({ x: 0.255, y: 0.5 }, 100, 100, false)).toEqual({ x: 25, y: 50 });
    expect(dropPixel({ x: 1, y: 0 }, 100, 100, true)).toEqual({ x: 99, y: 99 });
  });

  it("Save's paths carry the stem as a query", () => {
    expect(paintPath("stack", "hen")).toBe("/paint/stack?stem=hen");
    expect(paintPath("mesh", "a b")).toBe("/paint/mesh?stem=a%20b");
  });
});
