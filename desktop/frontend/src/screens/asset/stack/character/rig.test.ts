// The walk's promises (pipeline#2): a planted foot never moves, the bones keep their lengths, the lap loops.
import { describe, expect, it } from "vitest";
import { createWalker, lapPoint, place, pose, solveTwoBone, step, type V3 } from "./rig";

const dist = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// The walker asset's rig (out/walker/stack/character.json, 192 cells tall): long thighs, a long stride.
const WALKER = {
  thigh: 29, shin: 29, ankle: 12, hipHalf: 13.5, hipFwd: -2, kneeHint: 1, standHip: 67.68, stride: 91, duty: 0.6,
  lift: 9.86, bob: 1.75, twist: 0.14, spine: 75, neck: 82, arms: true, shoulderHalf: 29, shoulderFwd: -2,
  upperArm: 38, foreArm: 38, armSwing: 0.45,
};

function walkLap(strides = 10, lapFrames = 540, cfg = {}) {
  const w = createWalker(cfg);
  const start = lapPoint(0, w.c, strides, lapFrames);
  place(w, start.x, start.y, start.h, true);
  const frames = [];
  for (let f = 1; f <= lapFrames * 2; f++) {
    const p = lapPoint(f, w.c, strides, lapFrames);
    step(w, p.x, p.y, p.h);
    frames.push(pose(w));
  }
  return { w, frames };
}

describe("two-bone IK", () => {
  it("keeps both bone lengths and reaches the target when it can", () => {
    const root: V3 = [0, 0, 10];
    const target: V3 = [3, 1, 0];
    const knee = solveTwoBone(root, target, 7, 7, [1, 0, 0]);
    expect(dist(root, knee)).toBeCloseTo(7, 6);
    expect(dist(knee, target)).toBeCloseTo(7, 6);
    expect(knee[0]).toBeGreaterThan(1.5); // bends towards the hint
  });
});

describe.each([
  ["the kit's numbers", {}],
  ["the walker asset", WALKER],
])("the walk with %s", (_name, cfg) => {
  it("never slides a planted foot", () => {
    const { frames } = walkLap(10, 540, cfg);
    let checked = 0;
    for (let i = 1; i < frames.length; i++) {
      for (let s = 0; s < 2; s++) {
        const a = frames[i - 1].legs[s];
        const b = frames[i].legs[s];
        if (a.planted && b.planted) {
          expect(Math.hypot(a.sole[0] - b.sole[0], a.sole[1] - b.sole[1])).toBe(0);
          expect(b.sole[2]).toBe(0);
          checked++;
        }
      }
    }
    expect(checked).toBeGreaterThan(500);
  });

  it("keeps thigh and shin at their lengths every frame (a part stack cannot stretch)", () => {
    const { w, frames } = walkLap(10, 540, cfg);
    for (const P of frames)
      for (const L of P.legs) {
        expect(dist(L.hip, L.knee)).toBeCloseTo(w.c.thigh, 4);
        expect(dist(L.knee, L.ankle)).toBeCloseTo(w.c.shin, 4);
      }
  });

  it("takes two steps a stride and loops: one lap later the pose is the same", () => {
    const strides = 10;
    const lapFrames = 540;
    const { w, frames } = walkLap(strides, lapFrames, cfg);
    expect(w.steps).toBeGreaterThanOrEqual(4 * strides - 1);
    const a = frames[lapFrames - 1 + lapFrames];
    const b = frames[lapFrames - 1];
    for (let s = 0; s < 2; s++) expect(dist(a.legs[s].sole, b.legs[s].sole)).toBeLessThan(1e-6 + 0.05);
  });
});
