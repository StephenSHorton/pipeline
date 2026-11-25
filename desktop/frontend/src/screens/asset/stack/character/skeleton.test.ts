// The skeleton cycles' promises (pipeline#14): each loops, no bone stretches, a planted foot never slides, left is
// right mirrored, and a frame hangs a sprite stack's parts without stretching them.
import { describe, expect, it } from "vitest";
import { placeParts, type CharacterDoc } from "./parts";
import { RIG_DEFAULTS, type V3 } from "./rig";
import { bodyFromRig, cycleFrame, cycleFrameAt, CYCLES, JOINTS, makeCycle, rigPose, shot, type Body } from "./skeleton";

const dist = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// The walker asset's rig (out/walker/stack/character.json at 64 layers).
const RIG = {
  ...RIG_DEFAULTS, thigh: 9, shin: 9, ankle: 4, hipHalf: 4.5, hipFwd: -2, standHip: 21.28, stride: 28.6, lift: 3.06,
  bob: 0.55, twist: 0.14, spine: 26, neck: 28, shoulderHalf: 9.5, shoulderFwd: -3, upperArm: 12, foreArm: 13, armSwing: 0.45,
};
const BODY: Body = bodyFromRig(RIG, 64);
const SIDES = ["right", "left"] as const;

describe.each(CYCLES)("the %s cycle", (name) => {
  const c = makeCycle(name, BODY);

  it("loops: one loop on is the first frame moved by the travel", () => {
    const a = cycleFrameAt(name, BODY, 0);
    const b = cycleFrameAt(name, BODY, 1);
    for (const n of JOINTS) {
      expect(b.joints[n][0] - a.joints[n][0]).toBeCloseTo(c.travel, 9);
      expect(b.joints[n][1]).toBeCloseTo(a.joints[n][1], 9);
      expect(b.joints[n][2]).toBeCloseTo(a.joints[n][2], 9);
    }
    // and the last frame leads into the first: no joint jumps more there than it does inside the loop
    const step = (k: number) => Math.max(...JOINTS.map((n) => dist(cycleFrame(c, k).joints[n], cycleFrame(c, k + 1).joints[n])));
    let most = 0;
    for (let k = 0; k < c.frames - 1; k++) most = Math.max(most, step(k));
    expect(step(c.frames - 1)).toBeLessThanOrEqual(most + 1e-9);
  });

  it("keeps every bone its length", () => {
    for (let k = 0; k < c.frames * 2; k++) {
      const j = cycleFrame(c, k).joints;
      for (const s of SIDES) {
        expect(dist(j[`${s}_upLeg_joint`], j[`${s}_leg_joint`])).toBeCloseTo(BODY.thigh, 6);
        expect(dist(j[`${s}_leg_joint`], j[`${s}_foot_joint`])).toBeCloseTo(BODY.shin, 6);
        expect(dist(j[`${s}_shoulder_1_joint`], j[`${s}_forearm_joint`])).toBeCloseTo(BODY.upperArm, 6);
        expect(dist(j[`${s}_forearm_joint`], j[`${s}_hand_joint`])).toBeCloseTo(BODY.foreArm, 6);
      }
    }
  });

  it("never slides a planted foot, and keeps it on the ground", () => {
    let planted = 0;
    for (let k = 0; k < c.frames * 2; k++) {
      const a = cycleFrame(c, k);
      const b = cycleFrame(c, k + 1);
      SIDES.forEach((s, i) => {
        if (a.contact[i]) expect(a.joints[`${s}_foot_joint`][2]).toBeCloseTo(BODY.ankle, 9);
        if (a.contact[i] && b.contact[i]) {
          planted++;
          expect(dist(a.joints[`${s}_foot_joint`], b.joints[`${s}_foot_joint`])).toBeLessThan(1e-9);
        }
      });
    }
    expect(planted).toBeGreaterThan(0);
  });

  it("keeps planted feet still in a clip on the spot, for either facing", () => {
    for (const facing of ["right", "left"] as const) {
      const sh = shot(c, facing, c.frames * 2 + 16);
      for (let k = 16; k < sh.frames.length - 1; k++) {
        const a = sh.frames[k];
        const b = sh.frames[k + 1];
        SIDES.forEach((s, i) => {
          // on the spot the ground moves back under the body by the travel each frame, and no more
          if (a.frame.contact[i] && b.frame.contact[i]) {
            const moved = dist(a.joints[`${s}_foot_joint`], b.joints[`${s}_foot_joint`]);
            expect(moved).toBeCloseTo(c.travel / c.frames, 9);
          }
        });
      }
    }
  });

  it("faces left as the mirror of facing right", () => {
    const r = shot(c, "right", c.frames + 16);
    const l = shot(c, "left", c.frames + 16);
    for (let k = 16; k < r.frames.length; k++)
      for (const n of JOINTS) {
        expect(l.frames[k].joints[n][0]).toBeCloseTo(-r.frames[k].joints[n][0], 9);
        expect(l.frames[k].joints[n][2]).toBeCloseTo(r.frames[k].joints[n][2], 9);
      }
  });

  it("hangs a sprite stack's parts without stretching them", () => {
    const parts = ["head", "torso", "thigh_R", "thigh_L", "shin_R", "shin_L", "foot_R", "foot_L", "upperarm_R", "upperarm_L", "forearm_R", "forearm_L"];
    const doc = {
      rig: RIG,
      parts: parts.map((name) => ({ name, file: "", cw: 1, ch: 1, n: 1, z0: 22, joint_layer: 22, side: name.endsWith("_L") ? "L" : name.endsWith("_R") ? "R" : null })),
    } as unknown as CharacterDoc;
    for (let k = 0; k < c.frames; k++) {
      const placed = placeParts(doc, rigPose(cycleFrame(c, k), BODY, 3, -2, Math.PI));
      for (const [, at] of placed)
        for (const p of Array.isArray(at) ? at : [at]) expect(Math.hypot(...p.z)).toBeCloseTo(1, 6);
    }
  });
});

describe("the gaits", () => {
  it("walk has both feet down at once, the run a flight with neither", () => {
    const both = (n: "walk" | "run") => makeCycle(n, BODY).poses.some((f) => f.contact[0] && f.contact[1]);
    const none = (n: "walk" | "run") => makeCycle(n, BODY).poses.some((f) => !f.contact[0] && !f.contact[1]);
    expect(both("walk")).toBe(true);
    expect(none("walk")).toBe(false);
    expect(both("run")).toBe(false);
    expect(none("run")).toBe(true);
  });

  it("does not kick the walk's back foot up high", () => {
    const most = Math.max(...makeCycle("walk", BODY).poses.flatMap((f) => [f.joints.right_foot_joint[2], f.joints.left_foot_joint[2]]));
    expect(most - BODY.ankle).toBeLessThan(0.2 * (BODY.thigh + BODY.shin));
  });
});
