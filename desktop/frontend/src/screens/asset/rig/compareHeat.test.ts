// Compare's left heat map: the earlier version's weights in the right skeleton's numbering, the same colours as the
// Weights panel, and the model's own material back after.
import { describe, expect, it } from "vitest";
import { Bone, BufferGeometry, Float32BufferAttribute, MeshLambertMaterial, Skeleton, SkinnedMesh, Uint16BufferAttribute } from "three";
import { CompareHeat, SkinWeights, skeletonNames } from "./compareHeat";
import { heatRgb } from "./weights";

/** Two points: the first all "neck" (left bone 1), the second split "root" / "neck". */
function mesh(names: string[]): SkinnedMesh {
  const geo = new BufferGeometry();
  geo.setAttribute("position", new Float32BufferAttribute([0, 0, 0, 1, 0, 0], 3));
  geo.setAttribute("skinIndex", new Uint16BufferAttribute([1, 0, 0, 0, 0, 1, 0, 0], 4));
  geo.setAttribute("skinWeight", new Float32BufferAttribute([1, 0, 0, 0, 0.25, 0.75, 0, 0], 4));
  const m = new SkinnedMesh(geo, new MeshLambertMaterial());
  m.bind(new Skeleton(names.map((n) => Object.assign(new Bone(), { name: n }))));
  return m;
}

describe("left weights", () => {
  it("reads the skin into the right's bone numbers", () => {
    const m = mesh(["root", "neck"]);
    const s = SkinWeights.read(m, Int32Array.from([1, 0]));
    expect(s.n).toBe(2);
    expect(s.topBone(0)).toBe(0); // neck is the right's bone 0
    expect(s.weightOf(1, 1)).toBeCloseTo(0.25); // root is the right's bone 1
    expect(s.weightOf(1, 0)).toBeCloseTo(0.75);
    expect(skeletonNames(m)).toEqual(["root", "neck"]);
  });

  it("paints the picked bone's heat and gives the material back", () => {
    const m = mesh(["root", "neck"]);
    const own = m.material;
    const heat = CompareHeat.bind(m, ["neck", "root"])!;
    heat.show("heat", 0);
    expect(m.material).not.toBe(own);
    const c = m.geometry.getAttribute("color");
    const want = heatRgb(1);
    expect([c.getX(0), c.getY(0), c.getZ(0)].map((x) => +x.toFixed(4))).toEqual(want.map((x) => +x.toFixed(4)));
    heat.release();
    expect(m.material).toBe(own);
    expect(m.geometry.getAttribute("color")).toBeUndefined();
  });
});
