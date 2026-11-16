import { describe, expect, it } from "vitest";
import { FILL, fitZoom, viewExtents } from "./frame";

// A unit cube's corners, centred on the origin.
const cube: number[] = [];
for (const x of [-0.5, 0.5]) for (const y of [-0.5, 0.5]) for (const z of [-0.5, 0.5]) cube.push(x, y, z);

describe("viewExtents", () => {
  it("is the plain half size straight on", () => {
    const e = viewExtents(cube, 0, 0);
    expect(e.x).toBeCloseTo(0.5);
    expect(e.y).toBeCloseTo(0.5);
  });
  it("sees the cube's diagonal at 45 degrees", () => {
    expect(viewExtents(cube, 45, 0).x).toBeCloseTo(Math.SQRT1_2);
  });
  it("matches the camera's right vector: a point on +x moves right at azimuth 0, toward the viewer at 90", () => {
    expect(viewExtents([1, 0, 0], 0, 0).x).toBeCloseTo(1);
    expect(viewExtents([1, 0, 0], 90, 0).x).toBeCloseTo(0);
  });
  it("tilts height and depth into the vertical from above", () => {
    // top-front corner at elevation 30: 0.5 cos30 + 0.5 sin30
    expect(viewExtents(cube, 0, 30).y).toBeCloseTo(0.5 * Math.cos(Math.PI / 6) + 0.5 * Math.sin(Math.PI / 6));
  });
});

describe("fitZoom", () => {
  it("fits the limiting side to FILL", () => {
    const z = fitZoom({ x: 1, y: 2 }, 1000, 800);
    expect(z).toBeCloseTo((800 * FILL) / 4);
    expect(2 * 2 * z / 800).toBeCloseTo(FILL);
    expect(2 * 1 * z / 1000).toBeLessThan(FILL);
  });
  it("survives an empty model", () => {
    expect(fitZoom({ x: 0, y: 0 }, 100, 100)).toBe(1);
  });
});
