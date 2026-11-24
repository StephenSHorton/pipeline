// A walking character's parts (pipeline#2): only cubes with an open face are drawn, limbs hang from joint to joint
// without twisting off the walk's heading, and a standing pose puts every part back where the body was carved.
import { describe, expect, it } from "vitest";
import { limb, partCells, placeParts, placePoint, type CharacterDoc, type CharacterPart } from "./parts";
import { createWalker, place, pose, RIG_DEFAULTS, type V3 } from "./rig";

/** A strip of n solid slices, each cw x ch, one colour. */
function solidStrip(cw: number, ch: number, n: number): Uint8ClampedArray {
  const a = new Uint8ClampedArray(cw * n * ch * 4);
  for (let i = 0; i < a.length; i += 4) a.set([200, 40, 30, 255], i);
  return a;
}

const part = (o: Partial<CharacterPart>): CharacterPart => ({
  name: "thigh_R", file: "x.png", cw: 3, ch: 3, n: 3, z0: 10, joint_layer: 13, side: "R", ...o,
});

describe("partCells", () => {
  it("drops the cubes no face of which can show", () => {
    const c = partCells(part({}), solidStrip(3, 3, 3));
    expect(c.count).toBe(26); // a 3x3x3 block less its middle
  });

  it("puts each cube in the part's frame: x forward and y left of the axis cell, z up from the joint", () => {
    const c = partCells(part({ cw: 1, ch: 1, n: 1, z0: 10, joint_layer: 13 }), solidStrip(1, 1, 1));
    expect(Array.from(c.pos)).toEqual([0, 0, -2.5]);
    expect(c.layer[0]).toBe(10);
  });
});

describe("limb", () => {
  it("runs its up axis from the lower joint to the upper one and keeps forward square to it", () => {
    const hip: V3 = [0, 0, 20];
    const knee: V3 = [3, 0, 10];
    const p = limb(hip, knee, Math.hypot(3, 10), 0);
    const end = placePoint(p, [0, 0, -Math.hypot(3, 10)]);
    end.forEach((v, i) => expect(v).toBeCloseTo(knee[i], 6));
    const dot = p.x[0] * p.z[0] + p.x[1] * p.z[1] + p.x[2] * p.z[2];
    expect(dot).toBeCloseTo(0, 6);
    expect(p.x[0]).toBeGreaterThan(0.9);
  });
});

describe("placeParts", () => {
  it("in a straight standing pose, hangs every part with its axis upright on its joint", () => {
    const rig = { ...RIG_DEFAULTS, thigh: 6, shin: 6, ankle: 2, standHip: 14, spine: 10, neck: 12, bob: 0, twist: 0 };
    const doc: CharacterDoc = {
      format: "pipeline-sprite-character", version: 1, height: 30, rig,
      walk: { stride: 18, duty: 0.6, speed: 18, fps: 60, lap: { strides: 4, frames: 240 } },
      parts: ["torso", "head", "thigh_R", "thigh_L", "shin_R", "shin_L", "foot_R", "foot_L"].map((name) =>
        part({ name, side: name.endsWith("_L") ? "L" : name.endsWith("_R") ? "R" : null }),
      ),
    };
    const w = createWalker(rig);
    place(w, 0, 0, 0);
    const P = pose(w);
    const placed = placeParts(doc, P);
    const thigh = placed.get("thigh_L") as ReturnType<typeof limb>;
    expect(thigh.origin[1]).toBeCloseTo(rig.hipHalf, 6); // the left hip is on +y
    const torso = placed.get("torso") as ReturnType<typeof limb>[];
    expect(torso).toHaveLength(3);
    for (const s of torso) expect(s.x[0]).toBeCloseTo(1, 6); // facing +x, untwisted
    const head = placed.get("head") as ReturnType<typeof limb>;
    expect(head.origin[2]).toBeCloseTo(P.pelvis[2] + rig.neck, 6);
  });
});
