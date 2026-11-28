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

const BOTH = CYCLES.flatMap((name) => [{ name, principles: true }, { name, principles: false }]);

describe.each(BOTH)("the $name cycle (principles: $principles)", ({ name, principles }) => {
  const c = makeCycle(name, BODY, { principles });

  it("loops: one loop on is the first frame moved by the travel", () => {
    const a = cycleFrameAt(name, BODY, 0, principles);
    const b = cycleFrameAt(name, BODY, 1, principles);
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

// The animation principles (pipeline#8): built into the cycles on top of the same feet and timing, with the plain
// cycles kept exactly as pipeline#14 made them.
describe("the animation principles", () => {
  const weighed = (c: ReturnType<typeof makeCycle>) => {
    let s = 0;
    let w = 0;
    c.poses.forEach((f, k) => JOINTS.forEach((j, i) => f.joints[j].forEach((v, a) => {
      s += v;
      w += v * (((k * 31 + i * 7 + a) % 13) + 1);
    })));
    return [s, w];
  };
  // pipeline#14's cycles before this card, summed the same way
  const PLAIN = { walk: [23949.182575, 167576.599243], run: [19924.266174, 139985.971758], attack: [17325.563327, 121583.863193] };

  it.each(CYCLES)("leave the plain %s as pipeline#14 made it", (name) => {
    const [s, w] = weighed(makeCycle(name, BODY, { principles: false }));
    expect(s).toBeCloseTo(PLAIN[name][0], 5);
    expect(w).toBeCloseTo(PLAIN[name][1], 5);
  });

  it.each(CYCLES)("are on by default, and keep the %s's feet, frames and travel", (name) => {
    const on = makeCycle(name, BODY);
    const off = makeCycle(name, BODY, { principles: false });
    expect(on.principles).toBe(true);
    expect(off.principles).toBe(false);
    expect(on.frames).toBe(off.frames);
    expect(on.travel).toBe(off.travel);
    on.poses.forEach((f, k) => {
      expect(f.contact).toEqual(off.poses[k].contact);
      for (const s of SIDES)
        for (let a = 0; a < 3; a++) expect(f.joints[`${s}_foot_joint`][a]).toBeCloseTo(off.poses[k].joints[`${s}_foot_joint`][a], 9);
    });
  });

  // the side view (x, z) is all a video model sees, so the difference must show there
  const side = (f: { joints: Record<string, V3> }, n: string) => [f.joints[n][0], f.joints[n][2]];
  const moved = (a: { joints: Record<string, V3> }, b: { joints: Record<string, V3> }) =>
    Math.max(...JOINTS.map((n) => Math.hypot(side(a, n)[0] - side(b, n)[0], side(a, n)[1] - side(b, n)[1])));

  describe("in the punch", () => {
    const on = makeCycle("attack", BODY);
    const off = makeCycle("attack", BODY, { principles: false });
    const hand = (c: typeof on, k: number) => c.poses[k].joints.right_hand_joint[0] - c.poses[0].joints.right_hand_joint[0];

    it("winds up first (anticipation): the fist drawn back, the body back and down, further than the plain one", () => {
      expect(hand(on, 9)).toBeLessThan(-0.2 * (BODY.upperArm + BODY.foreArm));
      expect(hand(on, 9)).toBeLessThan(Math.min(...off.poses.map((_, k) => hand(off, k))));
      expect(on.poses[9].joints.root[0]).toBeLessThan(on.poses[0].joints.root[0] - 1);
      expect(on.poses[9].joints.root[2]).toBeLessThan(on.poses[0].joints.root[2] - 0.5);
    });

    it("holds the wind-up a frame, then stops dead on the hit for three frames (timing, hit-stop)", () => {
      expect(moved(on.poses[9], on.poses[10])).toBeLessThan(1e-9);
      expect(moved(on.poses[12], on.poses[13])).toBeLessThan(1e-9);
      expect(moved(on.poses[13], on.poses[14])).toBeLessThan(1e-9);
      expect(moved(on.poses[14], on.poses[15])).toBeGreaterThan(0.5);
      // the strike itself is fast: two frames from the wind-up to the hit
      expect(hand(on, 12) - hand(on, 10)).toBeGreaterThan(0.9 * (BODY.upperArm + BODY.foreArm));
    });

    it("carries on past the hit (follow-through) and the head settles after the body (overlap)", () => {
      expect(on.poses[16].shoulders).toBeGreaterThan(on.poses[12].shoulders + 0.1);
      expect(on.poses[16].lean).toBeGreaterThan(on.poses[12].lean);
      expect(hand(on, 16)).toBeGreaterThan(hand(on, 12));
      // the head is still nodding on after the shoulders have turned back
      expect(on.poses[17].head!.nod).toBeGreaterThan(on.poses[16].head!.nod);
      expect(on.poses[17].shoulders).toBeLessThan(on.poses[16].shoulders);
    });

    it("eases in and out of its key poses (slow in / slow out)", () => {
      // leaving the guard, and coming into the wind-up, the body moves less a frame than in the middle of the move
      expect(moved(on.poses[3], on.poses[4])).toBeLessThan(moved(on.poses[5], on.poses[6]));
      expect(moved(on.poses[8], on.poses[9])).toBeLessThan(moved(on.poses[5], on.poses[6]));
      expect(moved(on.poses[21], on.poses[22])).toBeLessThan(moved(on.poses[18], on.poses[19]));
    });

    it("brings the fist back on a low arc, not the line it went out on", () => {
      const z = (c: typeof on, k: number) => c.poses[k].joints.right_hand_joint[2];
      const out = (z(on, 11) + z(on, 12)) / 2;
      expect(z(on, 18)).toBeLessThan(Math.min(out, z(on, 12)) - 0.5);
    });

    it("differs from the plain punch where a video model can see it: the wind-up, the hit and the settle", () => {
      for (const k of [9, 12, 17]) expect(moved(on.poses[k], off.poses[k])).toBeGreaterThan(0.3);
    });
  });

  describe("in the walk", () => {
    const on = makeCycle("walk", BODY);
    const off = makeCycle("walk", BODY, { principles: false });
    const range = (c: typeof on, f: (p: (typeof on.poses)[number]) => number) => {
      const v = c.poses.map(f);
      return Math.max(...v) - Math.min(...v);
    };

    it("pushes the bob, the twist and the arm swing about a quarter further (exaggeration)", () => {
      const bob = range(on, (p) => p.joints.root[2]) / range(off, (p) => p.joints.root[2]);
      const twist = range(on, (p) => p.hips) / range(off, (p) => p.hips);
      const swing = (c: typeof on) => range(c, (p) => p.joints.right_hand_joint[0] - p.joints.right_shoulder_1_joint[0]);
      for (const k of [bob, twist, swing(on) / swing(off)]) {
        expect(k).toBeGreaterThan(1.15);
        expect(k).toBeLessThan(1.7);
      }
    });

    it("drops into the down pose just after a foot lands, and is highest passing (slow in / slow out)", () => {
      const z = on.poses.map((p) => p.joints.root[2]);
      const low = z.indexOf(Math.min(...z));
      const high = z.indexOf(Math.max(...z));
      // the right foot lands at frame 0 and the left at 14 (of 28): lowest a frame or two after, highest between
      expect([1, 2, 3, 15, 16, 17]).toContain(low);
      expect([8, 9, 10, 22, 23, 24]).toContain(high);
    });

    it("swings the hands behind the shoulders, and the forearms behind the upper arms (overlap)", () => {
      const peak = (c: typeof on) => {
        const v = c.poses.map((p) => p.joints.right_hand_joint[0] - p.joints.root[0]);
        return v.indexOf(Math.max(...v));
      };
      const turnPeak = on.poses.map((p) => -p.shoulders).indexOf(Math.max(...on.poses.map((p) => -p.shoulders)));
      expect(peak(on)).toBeGreaterThan(peak(off));
      expect(peak(on)).toBeGreaterThan(turnPeak);
    });

    it("nods the head after the body drops, and turns it back against the shoulders to look ahead", () => {
      for (const p of on.poses) {
        expect(p.head).toBeDefined();
        expect(Math.sign(p.head!.turn)).toBe(-Math.sign(p.shoulders) || 0);
      }
      expect(off.poses.every((p) => p.head === undefined)).toBe(true);
    });
  });

  it("hangs the walker's head on its own turn and nod", () => {
    const f = makeCycle("attack", BODY).poses[17];
    const P = rigPose(f, BODY, 0, 0, 0.3);
    expect(P.headAngle).toBeCloseTo(0.3 + f.shoulders + f.head!.turn, 12);
    expect(P.headLean).toBeCloseTo(f.lean + f.head!.nod, 12);
    const plain = rigPose(makeCycle("attack", BODY, { principles: false }).poses[17], BODY, 0, 0, 0.3);
    expect(plain.headAngle).toBeUndefined();
  });
});

// Animate makes the cycle for each character's own body, so the principles must hold on bodies other than the walker's.
describe.each([
  { label: "the rig's defaults", rig: RIG_DEFAULTS, height: 32 },
  { label: "near-straight standing legs", rig: { ...RIG, standHip: 21.9 }, height: 64 },
  { label: "short legs, long arms", rig: { ...RIG, thigh: 6, shin: 6, ankle: 2, standHip: 13.4, upperArm: 13, foreArm: 14 }, height: 52 },
])("on $label", ({ rig, height }) => {
  const body = bodyFromRig(rig, height);
  it.each(CYCLES)("the %s keeps its bones, its planted feet and its loop", (name) => {
    const c = makeCycle(name, body);
    for (let k = 0; k < c.frames * 2; k++) {
      const a = cycleFrame(c, k);
      const b = cycleFrame(c, k + 1);
      SIDES.forEach((s, i) => {
        expect(dist(a.joints[`${s}_upLeg_joint`], a.joints[`${s}_leg_joint`])).toBeCloseTo(body.thigh, 6);
        expect(dist(a.joints[`${s}_leg_joint`], a.joints[`${s}_foot_joint`])).toBeCloseTo(body.shin, 6);
        expect(dist(a.joints[`${s}_shoulder_1_joint`], a.joints[`${s}_forearm_joint`])).toBeCloseTo(body.upperArm, 6);
        expect(dist(a.joints[`${s}_forearm_joint`], a.joints[`${s}_hand_joint`])).toBeCloseTo(body.foreArm, 6);
        if (a.contact[i] && b.contact[i]) expect(dist(a.joints[`${s}_foot_joint`], b.joints[`${s}_foot_joint`])).toBeLessThan(1e-9);
      });
    }
    const f0 = cycleFrameAt(name, body, 0);
    const f1 = cycleFrameAt(name, body, 1);
    for (const n of JOINTS) expect(f1.joints[n][0] - f0.joints[n][0]).toBeCloseTo(c.travel, 9);
  });
});
