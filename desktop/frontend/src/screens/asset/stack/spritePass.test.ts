import { describe, expect, it } from "vitest";
import { slicePixels, sliceShade } from "./character/walkView";
import type { CharacterPart } from "./character/parts";
import { cellScale, lowSize } from "./spritePass";

describe("sprite-stack look", () => {
  it("draws one cell to about one low-resolution pixel, enlarged by a whole number", () => {
    // 600 px tall, 35 degree view, 4 world units away: one world unit is ~238 px, so a 0.04 cell is ~9.5 px
    expect(cellScale(0.04, 4, 35, 600)).toBe(10);
    expect(cellScale(0.0001, 4, 35, 600)).toBe(1);
    expect(lowSize(1000, 600, 9)).toEqual({ w: 112, h: 67 });
    expect(lowSize(1000, 600, 10)).toEqual({ w: 100, h: 60 });
  });

  it("darkens lower layers: the bottom at 0.62, the top at full", () => {
    expect(sliceShade(0, 63)).toBeCloseTo(0.62);
    expect(sliceShade(63, 63)).toBe(1);
    expect(sliceShade(31.5, 63)).toBeCloseTo(0.81);
  });

  it("cuts one slice from a part strip, shaded, empty cells clear", () => {
    const part: CharacterPart = { name: "head", file: "", cw: 2, ch: 1, n: 2, z0: 0, joint_layer: 0, side: null };
    // strip 4 x 1: slice 0 = [red, empty], slice 1 = [empty, blue]
    const strip = [200, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 255];
    expect(Array.from(slicePixels(part, strip, 0, 0.5))).toEqual([100, 0, 0, 255, 0, 0, 0, 0]);
    expect(Array.from(slicePixels(part, strip, 1, 1))).toEqual([0, 0, 0, 0, 0, 0, 100, 255]);
  });
});
