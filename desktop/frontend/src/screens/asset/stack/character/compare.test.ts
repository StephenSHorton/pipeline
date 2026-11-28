// The principles compare (pipeline#8): walk and punch, without and with, each in a column that holds it whole.
import { describe, expect, it } from "vitest";
import { columnFrame, COMPARE_COLUMNS, compareCycles, compareFromSearch, skeletonAt, toCanvas } from "./compare";
import { RIG_DEFAULTS } from "./rig";
import { bodyFromRig } from "./skeleton";

const RIG = {
  ...RIG_DEFAULTS, thigh: 9, shin: 9, ankle: 4, hipHalf: 4.5, hipFwd: -2, standHip: 21.28, stride: 28.6, lift: 3.06,
  bob: 0.55, twist: 0.14, spine: 26, neck: 28, shoulderHalf: 9.5, shoulderFwd: -3, upperArm: 12, foreArm: 13, armSwing: 0.45,
};

describe("the principles compare", () => {
  it("lays each pair next to each other: walk without, with, punch without, with", () => {
    expect(COMPARE_COLUMNS.map((c) => `${c.cycle}:${c.principles}`)).toEqual(["walk:false", "walk:true", "attack:false", "attack:true"]);
  });

  it("keeps every joint of every frame inside its column", () => {
    const body = bodyFromRig(RIG, 64);
    const cycles = compareCycles(body);
    cycles.forEach((c, i) => {
      const f = columnFrame(64, COMPARE_COLUMNS[i].cycle);
      const w = 300;
      const h = (w * f.height) / f.width;
      for (let k = 0; k < c.frames * 2; k++)
        for (const p of Object.values(skeletonAt(c, k))) {
          const [x, y] = toCanvas(p!, w, h, f);
          expect(x).toBeGreaterThan(0);
          expect(x).toBeLessThan(w);
          expect(y).toBeGreaterThan(0);
          expect(y).toBeLessThan(h);
        }
    });
  });

  it("opens from ?compare=principles", () => {
    expect(compareFromSearch("?asset=walker&compare=principles")).toBe(true);
    expect(compareFromSearch("?asset=walker")).toBe(false);
  });
});
