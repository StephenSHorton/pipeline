// The walker's view plays a skeleton cycle (pipeline#14): walk and run go round the lap, the attack stays put, and
// every part is placed somewhere real, in both looks.
import { Group, MeshBasicMaterial } from "three";
import { describe, expect, it } from "vitest";
import { CHARACTER_FORMAT, type CharacterDoc } from "./parts";
import { RIG_DEFAULTS } from "./rig";
import { CYCLES } from "./skeleton";
import { createWalkView } from "./walkView";

const NAMES = ["head", "torso", "thigh_R", "thigh_L", "shin_R", "shin_L", "foot_R", "foot_L", "upperarm_R", "upperarm_L", "forearm_R", "forearm_L"];
const doc: CharacterDoc = {
  format: CHARACTER_FORMAT, version: 1, height: 64,
  rig: { ...RIG_DEFAULTS, thigh: 9, shin: 9, ankle: 4, hipHalf: 4.5, hipFwd: -2, standHip: 21.28, spine: 26, neck: 28, shoulderHalf: 9.5, shoulderFwd: -3, upperArm: 12, foreArm: 13 },
  walk: { stride: 28.6, duty: 0.6, speed: 28.6, fps: 60, lap: { strides: 6, frames: 360 } },
  parts: NAMES.map((name) => ({ name, file: "", cw: 3, ch: 3, n: 2, z0: 22, joint_layer: 22, side: name.endsWith("_L") ? "L" : name.endsWith("_R") ? "R" : null })),
};
const strips = new Map(NAMES.map((n) => [n, new Uint8Array(3 * 2 * 3 * 4).fill(200)]));

describe.each(["cubes", "slices"] as const)("a cycle in the %s look", (look) => {
  it.each(CYCLES)("plays %s", (cycle) => {
    const parent = new Group();
    const v = createWalkView({ parent, doc, strips, pitch: 0.1, material: new MeshBasicMaterial(), color: (b) => b / 255, look, cycle });
    const from = v.where().clone();
    for (let i = 0; i < 60; i++) v.tick(1 / 30); // two seconds: more than a loop of each
    const to = v.where().clone();
    if (cycle === "attack") expect(to.distanceTo(from)).toBeLessThan(1e-9);
    else expect(to.distanceTo(from)).toBeGreaterThan(0.5);
    let placed = 0;
    parent.traverse((o) => {
      if (o instanceof Group && o !== parent && !o.matrixAutoUpdate) {
        expect(o.matrix.elements.every(Number.isFinite)).toBe(true);
        if (o.visible) placed++;
      }
    });
    expect(placed).toBeGreaterThanOrEqual(NAMES.length);
    v.dispose();
  });
});
