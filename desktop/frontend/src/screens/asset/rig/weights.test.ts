// The Weights panel's logic: grab scaling, the brush (add, erase, smooth with falloff), the four-bone limit, corner
// copies kept in step, undo / redo / reset, the bone list's chains and names, and Save's body.
import { describe, expect, it } from "vitest";
import { boneChains, boneWords, changedText, cornerGroups, falloff, heatRgb, neighbours, paintColours, savedToast, setShare, topFour, WeightEdit, weightsBody } from "./weights";

/** n points on a line x = 0..n-1 (y = z = 0), triangles fanned between consecutive pairs, every row as given. */
function line(rows: Array<Array<[number, number]>>, parents = [-1, 0, 1, 2, 3, 4, 5]) {
  const n = rows.length;
  const idx = new Uint16Array(n * 4);
  const w = new Float32Array(n * 4);
  rows.forEach((r, v) =>
    r.forEach(([j, x], k) => {
      idx[v * 4 + k] = j;
      w[v * 4 + k] = x;
    }),
  );
  const pos = new Float32Array(n * 3);
  for (let v = 0; v < n; v++) pos[v * 3] = v;
  const index: number[] = [];
  for (let v = 0; v + 2 < n; v++) index.push(v, v + 1, v + 2);
  const e = new WeightEdit({ idx, w, pos, index }, parents);
  return { e, idx, w, pos };
}

const sumRow = (w: Float32Array, v: number) => w[v * 4] + w[v * 4 + 1] + w[v * 4 + 2] + w[v * 4 + 3];
const bonesAt = (e: WeightEdit, v: number) => [0, 1, 2, 3].filter((k) => e.w[v * 4 + k] > 0).map((k) => e.idx[v * 4 + k]).sort();

describe("falloff", () => {
  it("is 1 at the centre, 0 at and past the rim, and falls smoothly", () => {
    expect(falloff(0, 1)).toBe(1);
    expect(falloff(1, 1)).toBe(0);
    expect(falloff(2, 1)).toBe(0);
    expect(falloff(0.5, 1)).toBeCloseTo(0.5625);
    expect(falloff(0.2, 1)).toBeGreaterThan(falloff(0.6, 1));
    expect(falloff(0, 0)).toBe(0);
  });
});

describe("rows", () => {
  it("keeps four bones, the painted one replacing the smallest, renormalised", () => {
    const m = new Map([
      [1, 0.4],
      [2, 0.3],
      [3, 0.2],
      [4, 0.1],
      [9, 0.05],
    ]);
    const r = topFour(m, 9);
    expect(r.j).toEqual([1, 2, 3, 9]);
    expect(r.w.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(topFour(m).j).toEqual([1, 2, 3, 4]);
  });

  it("setShare gives the others the rest in proportion; a lone bone lets go to its parent", () => {
    const m = setShare(new Map([[1, 0.5], [2, 0.25], [3, 0.25]]), 1, 0.8, 0);
    expect(m.get(1)).toBeCloseTo(0.8);
    expect(m.get(2)).toBeCloseTo(0.1);
    expect(m.get(3)).toBeCloseTo(0.1);
    const lone = setShare(new Map([[5, 1]]), 5, 0.3, 4);
    expect(lone.get(5)).toBeCloseTo(0.3);
    expect(lone.get(4)).toBeCloseTo(0.7);
    expect(setShare(new Map([[0, 1]]), 0, 0.3, -1).get(0)).toBe(1); // the root has no one to give to
  });
});

describe("grab", () => {
  it("scales a bone's weights and renormalises the others, from the drag's start", () => {
    const { e, w } = line([[[1, 0.5], [2, 0.5]], [[1, 1]], [[2, 1]]]);
    e.begin();
    e.grabTo(1, 150, 100);
    e.grabTo(1, 60, 100); // a drag back: from the start, not compounded
    e.end();
    expect(e.weightOf(0, 1)).toBeCloseTo(0.3);
    expect(e.weightOf(0, 2)).toBeCloseTo(0.7);
    // Point 1 had only bone 1: what it lets go of goes to bone 0, its parent.
    expect(e.weightOf(1, 1)).toBeCloseTo(0.6);
    expect(e.weightOf(1, 0)).toBeCloseTo(0.4);
    expect(e.weightOf(2, 2)).toBe(1); // bone 1 was not there
    for (let v = 0; v < 3; v++) expect(sumRow(w, v)).toBeCloseTo(1);
    expect(e.grabPercent(1)).toBeCloseTo(60);
    expect(e.canUndo).toBe(true);
    e.undo();
    expect(e.weightOf(0, 1)).toBeCloseTo(0.5);
    expect(e.weightOf(1, 1)).toBe(1);
    expect(e.grabPercent(1)).toBeCloseTo(100);
  });

  it("200% saturates at 1", () => {
    const { e } = line([[[1, 0.7], [2, 0.3]]]);
    e.grab(1, 2);
    expect(e.weightOf(0, 1)).toBe(1);
    expect(e.weightOf(0, 2)).toBe(0);
  });
});

describe("brush", () => {
  const five = () => line([[[2, 1]], [[2, 1]], [[2, 1]], [[2, 1]], [[2, 1]]]);

  it("Add pulls a bone in with falloff and renormalises", () => {
    const { e, w } = five();
    let changes = 0;
    e.onChange = () => changes++;
    e.dab("add", 3, e.pos(), [0, 0, 0], 2.5, 0.5);
    expect(changes).toBe(1);
    expect(e.weightOf(0, 3)).toBeCloseTo(0.5);
    expect(e.weightOf(1, 3)).toBeGreaterThan(0);
    expect(e.weightOf(1, 3)).toBeLessThan(e.weightOf(0, 3));
    expect(e.weightOf(2, 3)).toBeLessThan(e.weightOf(1, 3));
    expect(e.weightOf(3, 3)).toBe(0); // past the rim
    for (let v = 0; v < 5; v++) expect(sumRow(w, v)).toBeCloseTo(1);
  });

  it("Erase pushes it out to the other bones, else the parent", () => {
    const { e } = line([[[3, 0.5], [2, 0.5]], [[3, 1]]]);
    e.dab("erase", 3, e.pos(), [0, 0, 0], 3, 0.5);
    expect(e.weightOf(0, 3)).toBeCloseTo(0.25);
    expect(e.weightOf(0, 2)).toBeCloseTo(0.75);
    expect(e.weightOf(1, 3)).toBeLessThan(1);
    expect(e.weightOf(1, 2)).toBeCloseTo(1 - e.weightOf(1, 3)); // bone 3's parent is 2
  });

  it("Smooth evens the weights out with the neighbours", () => {
    const { e, w } = line([[[1, 1]], [[1, 1]], [[2, 1]], [[2, 1]], [[2, 1]]]);
    const before = Math.abs(e.weightOf(1, 1) - e.weightOf(2, 1));
    e.dab("smooth", 1, e.pos(), [1.5, 0, 0], 2, 1);
    expect(Math.abs(e.weightOf(1, 1) - e.weightOf(2, 1))).toBeLessThan(before);
    expect(e.weightOf(2, 1)).toBeGreaterThan(0);
    for (let v = 0; v < 5; v++) expect(sumRow(w, v)).toBeCloseTo(1);
  });

  it("a bone new to a full point replaces the smallest of its four", () => {
    const { e } = line([[[1, 0.4], [2, 0.3], [3, 0.2], [4, 0.1]]]);
    e.dab("add", 6, e.pos(), [0, 0, 0], 1, 0.3);
    expect(bonesAt(e, 0)).toEqual([1, 2, 3, 6]);
    expect(e.weightOf(0, 6)).toBeCloseTo(0.3);
    expect(sumRow(e.w, 0)).toBeCloseTo(1);
  });

  it("corners at the same position always get the same weights", () => {
    const idx = new Uint16Array([1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0]);
    const w = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    const pos = new Float32Array([0, 0, 0, 0, 0, 0, 5, 0, 0]); // 0 and 1 are copies of one point
    const e = new WeightEdit({ idx, w, pos, index: [0, 1, 2] }, [-1, 0, 1]);
    expect(e.members.length).toBe(2);
    e.dab("add", 1, e.pos(), [0, 0, 0], 1, 0.2);
    for (let k = 0; k < 4; k++) {
      expect(idx[k]).toBe(idx[4 + k]);
      expect(w[k]).toBeCloseTo(w[4 + k]);
    }
    expect(e.weightOf(0, 1)).toBeGreaterThan(0.5); // the mended split point leans to bone 1 after the add
  });

  it("a stroke is one undo step; redo puts it back; reset goes to the saved weights", () => {
    const { e } = five();
    e.begin();
    e.dab("add", 3, e.pos(), [0, 0, 0], 1.5, 0.5);
    e.dab("add", 3, e.pos(), [1, 0, 0], 1.5, 0.5);
    e.end();
    const after = e.weightOf(1, 3);
    expect(e.changedPoints()).toBeGreaterThan(1);
    e.undo();
    expect(e.weightOf(1, 3)).toBe(0);
    expect(e.changedPoints()).toBe(0);
    expect(e.canRedo).toBe(true);
    e.redo();
    expect(e.weightOf(1, 3)).toBeCloseTo(after);
    expect(e.reset()).toBe(true);
    expect(e.changedPoints()).toBe(0);
    e.undo(); // Reset is undoable too
    expect(e.weightOf(1, 3)).toBeCloseTo(after);
    e.markSaved();
    expect(e.changedPoints()).toBe(0);
    expect(e.grabPercent(3)).toBeCloseTo(100);
  });

  it("adopt carries the edit into a reloaded mesh's arrays", () => {
    const { e } = five();
    e.dab("add", 3, e.pos(), [0, 0, 0], 1, 1);
    const idx = new Uint16Array(20);
    const w = new Float32Array(20);
    e.adopt(idx, w);
    expect(w[0]).toBeCloseTo(1);
    expect(idx[0]).toBe(3);
    e.undo();
    expect(e.weightOf(0, 3)).toBe(0);
    expect(w[0]).toBe(1);
  });
});

describe("corners and neighbours", () => {
  it("groups same-position points and links them across copies", () => {
    const pos = [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0];
    const { group, members } = cornerGroups(pos, 4);
    expect([...group]).toEqual([0, 1, 0, 2]);
    expect(members).toEqual([[0, 2], [1], [3]]);
    const near = neighbours(group, 3, [0, 1, 3], 4);
    expect(near[0].sort()).toEqual([1, 2]);
    expect(near[2].sort()).toEqual([0, 1]);
  });
});

describe("bone list", () => {
  const sheep = ["root", "hips", "spine_1", "spine_2", "spine_3", "ribcage_middle", "ribcage_leaf", "spine_4", "spine_5", "spine_6", "head", "ear_1_r", "ear_2_r", "ear_leaf_r", "nose", "front_scapula_l", "front_leg_upper_l", "front_leg_lower_l", "front_leg_leaf_l", "tail_1", "tail_2", "tail_leaf", "back_leg_pelvis_l", "back_leg_upper_l", "back_leg_lower_l", "back_leg_foot_l"];
  const parent: Record<string, string> = {
    hips: "root", spine_1: "hips", spine_2: "spine_1", spine_3: "spine_2", ribcage_middle: "spine_2", ribcage_leaf: "ribcage_middle", spine_4: "spine_3", spine_5: "spine_4",
    spine_6: "spine_5", head: "spine_6", ear_1_r: "head", ear_2_r: "ear_1_r", ear_leaf_r: "ear_2_r", nose: "head", front_scapula_l: "spine_3", front_leg_upper_l: "front_scapula_l",
    front_leg_lower_l: "front_leg_upper_l", front_leg_leaf_l: "front_leg_lower_l", tail_1: "hips", tail_2: "tail_1", tail_leaf: "tail_2", back_leg_pelvis_l: "hips",
    back_leg_upper_l: "back_leg_pelvis_l", back_leg_lower_l: "back_leg_upper_l", back_leg_foot_l: "back_leg_lower_l",
  };
  const parents = sheep.map((n) => (parent[n] ? sheep.indexOf(parent[n]) : -1));

  it("groups the horse template by chain with plain names", () => {
    const chains = boneChains(sheep, parents);
    const by = Object.fromEntries(chains.map((c) => [c.label, c.bones.map((b) => b.name)]));
    expect(chains.map((c) => c.label)).toEqual(["Spine", "Neck & head", "Tail", "Front left leg", "Back left leg"]);
    expect(by.Spine).toEqual(["Hips", "Spine 1", "Spine 2", "Spine 3", "Middle ribcage"]);
    expect(by["Neck & head"]).toEqual(["Neck 1", "Neck 2", "Neck 3", "Head", "Right ear 1", "Right ear 2", "Nose"]);
    expect(by.Tail).toEqual(["Tail 1", "Tail 2"]);
    expect(by["Front left leg"]).toEqual(["Scapula", "Upper leg", "Lower leg"]);
    expect(by["Back left leg"]).toEqual(["Pelvis", "Upper leg", "Lower leg", "Foot"]);
  });

  it("uses inspect.json's labels when there are some, and shows a leaf that holds weight", () => {
    const chains = boneChains(sheep, parents, { labels: { back_leg_upper_l: "left back thigh" }, weighted: (i) => sheep[i] === "tail_leaf" });
    const back = chains.find((c) => c.key === "bl")!;
    expect(back.bones.map((b) => b.name)).toContain("Thigh");
    expect(chains.find((c) => c.key === "tail")!.bones.map((b) => b.name)).toEqual(["Tail 1", "Tail 2", "Tail leaf"]);
  });

  it("people: arms and legs by side", () => {
    const names = ["Hips", "Spine", "Neck", "Head", "LeftUpperArm", "LeftHand", "RightUpLeg", "RightFoot"];
    const ps = [-1, 0, 1, 2, 1, 4, 0, 6];
    const chains = boneChains(names, ps);
    expect(chains.map((c) => c.label)).toEqual(["Spine", "Neck & head", "Left arm", "Right leg"]);
    expect(boneWords("LeftUpperArm")).toEqual(["left", "upper", "arm"]);
  });
});

describe("colours and copy", () => {
  it("heat runs cold blue (none) to red (all)", () => {
    expect(heatRgb(0)).toEqual([0.05, 0.05, 0.35]);
    const [r, g, b] = heatRgb(1);
    expect(r).toBeCloseTo(1);
    expect(g).toBeCloseTo(0);
    expect(b).toBeCloseTo(0);
    expect(heatRgb(0.5)[1]).toBeGreaterThan(0.5); // green in the middle
  });

  it("paints heat for a bone and the parts by top bone", () => {
    const { e } = line([[[1, 1]], [[2, 1]]]);
    const out = new Float32Array(6);
    paintColours(e, "heat", 1, out);
    expect(out[0]).toBeCloseTo(1);
    expect(out[5]).toBeCloseTo(0.35);
    paintColours(e, "parts", -1, out);
    expect([out[0], out[1], out[2]]).not.toEqual([out[3], out[4], out[5]]);
  });

  it("says how much changed", () => {
    expect(changedText(0)).toBe("No changes");
    expect(changedText(1)).toBe("1 point changed, not saved");
    expect(savedToast({ changed: 12, blender: true })).toBe("Weights saved (12 points changed). The FBX was exported again.");
    expect(savedToast({ changed: 1, blender: false, fbx_note: "Blender not found." })).toBe("Weights saved (1 point changed). Blender not found.");
  });
});

describe("Save's body", () => {
  it("base64 of little-endian u16 joints and f32 weights", () => {
    const idx = new Uint16Array([1, 300, 0, 0]);
    const w = new Float32Array([0.25, 0.75, 0, 0]);
    const body = weightsBody([{ mesh: 0, primitive: 0, idx, w }], "x");
    expect(body.primitives[0].count).toBe(1);
    const jb = Uint8Array.from(atob(body.primitives[0].joints), (c) => c.charCodeAt(0));
    expect([...new Uint16Array(jb.buffer)]).toEqual([1, 300, 0, 0]);
    const wb = Uint8Array.from(atob(body.primitives[0].weights), (c) => c.charCodeAt(0));
    expect([...new Float32Array(wb.buffer)]).toEqual([0.25, 0.75, 0, 0]);
    expect(body.note).toBe("x");
  });
});
