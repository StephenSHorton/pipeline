// The sprite stack's pure core (stack-lab.js): texels from a slab, the Main island mask, marching squares and the
// contour shapes, OkLab Reduce, the brush (paint, erase back to rgb0, undo), Drop and the sheet Save writes.
import { describe, expect, it } from "vitest";
import {
  beginStroke,
  buildMergedVoxelSurface,
  cellKept,
  createCore,
  loopsToShapes,
  marchingSquares,
  nativeFromPixels,
  paintSphere,
  quantize,
  rebuildIslandMask,
  restBox,
  sampleVoxel,
  sheetPixels,
  signedArea,
  undoStroke,
  applyPaletteEntry,
  type Core,
} from "./core";
import { defaultsFromSearch, type StackMeta } from "./stack";

/** A res×res RGBA slab with the listed texels opaque in one colour. */
function slab(res: number, cells: [number, number][], rgb: [number, number, number] = [200, 40, 40]): Uint8ClampedArray {
  const d = new Uint8ClampedArray(res * res * 4);
  for (const [x, z] of cells) {
    const p = (z * res + x) * 4;
    d[p] = rgb[0];
    d[p + 1] = rgb[1];
    d[p + 2] = rgb[2];
    d[p + 3] = 255;
  }
  return d;
}

const meta = (layers: number, resolution: number): StackMeta => ({ layers, resolution, pitch: 0.01, extents: [resolution * 0.01, layers * 0.01, resolution * 0.01] });

/** A core whose slabs are the given texel lists, at full Texels (crunch = res). */
function coreOf(res: number, layers: [number, number][][], rgb?: [number, number, number]): Core {
  const state = { ...defaultsFromSearch(""), texels: res, layers: layers.length };
  const core = createCore(meta(layers.length, res), state);
  layers.forEach((cells, i) => (core.slices[i].native = nativeFromPixels(slab(res, cells, rgb), res)));
  return core;
}

describe("nativeFromPixels (cacheNative)", () => {
  it("keeps texels with alpha >= 24, row by row, and copies rgb into rgb0", () => {
    const d = slab(4, [[2, 0], [1, 3]]);
    d[(1 * 4 + 1) * 4 + 3] = 23; // too faint: dropped
    const n = nativeFromPixels(d, 4);
    expect(n.n).toBe(2);
    expect(Array.from(n.ix)).toEqual([2, 1]);
    expect(Array.from(n.iz)).toEqual([0, 3]);
    expect(Array.from(n.rgb)).toEqual([200, 40, 40, 200, 40, 40]);
    expect(Array.from(n.rgb0)).toEqual(Array.from(n.rgb));
    expect(n.rgb0).not.toBe(n.rgb);
  });
});

describe("Main: the island mask", () => {
  it("keeps only the largest 6-connected piece", () => {
    // A 3-texel bar on two stacked slabs (6 cells, joined through y) and a lone texel far away.
    const bar: [number, number][] = [[0, 0], [1, 0], [2, 0]];
    const core = coreOf(8, [[...bar, [7, 7]], bar]);
    rebuildIslandMask(core, 8);
    expect(cellKept(core, 0, 1, 0, 8)).toBe(true);
    expect(cellKept(core, 1, 2, 0, 8)).toBe(true);
    expect(cellKept(core, 0, 7, 7, 8)).toBe(false);
  });

  it("keeps ties, everything with Main off, and everything when the mask is for another Texels grid", () => {
    const tie = coreOf(8, [[[0, 0], [7, 7]]]);
    rebuildIslandMask(tie, 8);
    expect(cellKept(tie, 0, 7, 7, 8)).toBe(true);
    const core = coreOf(8, [[[0, 0], [1, 0], [7, 7]]]);
    rebuildIslandMask(core, 8);
    expect(cellKept(core, 0, 7, 7, 8)).toBe(false);
    expect(cellKept(core, 0, 7, 7, 4)).toBe(true);
    core.state.main = false;
    rebuildIslandMask(core, 8);
    expect(cellKept(core, 0, 7, 7, 8)).toBe(true);
  });

  it("drops slabs above Layers", () => {
    const core = coreOf(4, [[[0, 0]], [[0, 0]], [[0, 0]]]);
    core.state.layers = 2;
    rebuildIslandMask(core, 4);
    expect(cellKept(core, 1, 0, 0, 4)).toBe(true);
    expect(cellKept(core, 2, 0, 0, 4)).toBe(false);
  });
});

describe("Contour: marching squares", () => {
  it("one texel is one closed 4-point loop through the edge midpoints around it", () => {
    const occ = new Uint8Array(9);
    occ[4] = 1; // (1,1) on a 3×3 grid
    const loops = marchingSquares(occ, 3);
    expect(loops).toHaveLength(1);
    expect(loops[0]).toHaveLength(4);
    const xs = loops[0].map((p) => p[0]).sort();
    expect(xs).toEqual([0.5, 1, 1, 1.5]);
    expect(Math.abs(signedArea(loops[0]))).toBe(0.5);
  });

  it("a ring makes an outer shape with one hole", () => {
    const n = 8;
    const occ = new Uint8Array(n * n);
    for (let z = 1; z <= 6; z++) for (let x = 1; x <= 6; x++) if (x === 1 || x === 6 || z === 1 || z === 6) occ[z * n + x] = 1;
    const loops = marchingSquares(occ, n);
    expect(loops).toHaveLength(2);
    const shapes = loopsToShapes(loops, n, 1);
    expect(shapes).toHaveLength(1);
    expect(shapes[0].holes).toHaveLength(1);
  });

  it("an empty grid has no loops", () => {
    expect(marchingSquares(new Uint8Array(16), 4)).toEqual([]);
  });
});

describe("Round: the merged voxel surface", () => {
  it("one cube is 6 faces (36 corners); none shown is null", () => {
    const core = coreOf(4, [[[1, 1]]]);
    const geo = buildMergedVoxelSurface(core, 4, 0)!;
    expect(geo.getAttribute("position").count).toBe(36);
    expect(geo.getAttribute("normal").count).toBe(36);
    expect(buildMergedVoxelSurface(coreOf(4, [[]]), 4, 0.45)).toBeNull();
  });
});

describe("Reduce: OkLab quantize", () => {
  const colours: [number, number, number][] = [
    [220, 30, 30],
    [30, 200, 40],
    [40, 60, 210],
    [240, 240, 240],
    [20, 20, 20],
    [230, 200, 30],
  ];
  /** Six coloured columns of texels, two slabs. */
  function rainbow(): Core {
    const res = 6;
    const core = createCore(meta(2, res), { ...defaultsFromSearch(""), texels: res, layers: 2 });
    for (let s = 0; s < 2; s++) {
      const d = new Uint8ClampedArray(res * res * 4);
      for (let z = 0; z < res; z++)
        for (let x = 0; x < res; x++) {
          const p = (z * res + x) * 4;
          const c = colours[x];
          d[p] = c[0] + s * 3;
          d[p + 1] = c[1];
          d[p + 2] = c[2];
          d[p + 3] = 255;
        }
      core.slices[s].native = nativeFromPixels(d, res);
    }
    return core;
  }

  it("is deterministic: the same texels give the same palette and assignment", () => {
    const a = rainbow();
    const b = rainbow();
    const pa = quantize(a, 4);
    const pb = quantize(b, 4);
    expect(pa).toEqual(pb);
    expect(pa).toHaveLength(4);
    expect(Array.from(a.slices[0].native!.pal!)).toEqual(Array.from(b.slices[0].native!.pal!));
  });

  it("recolours rgb to the swatches, keeps rgb0, and clamps k to 2..32 and the texel count", () => {
    const core = rainbow();
    const n = core.slices[0].native!;
    const before = Array.from(n.rgb0);
    const pal = quantize(core, 1);
    expect(pal).toHaveLength(2);
    for (let i = 0; i < n.n; i++) {
      const e = pal[n.pal![i]];
      expect([n.rgb[i * 3], n.rgb[i * 3 + 1], n.rgb[i * 3 + 2]]).toEqual([e.r, e.g, e.b]);
    }
    expect(Array.from(n.rgb0)).toEqual(before);
    expect(core.palSelected).toBe(0);
    expect(quantize(rainbow(), 99).length).toBeLessThanOrEqual(32);
    expect(pal.every((e) => e.metal === 0 && e.rough === 0.7)).toBe(true);
  });

  it("is computed from rgb0, so a second Reduce ignores the first", () => {
    const a = rainbow();
    quantize(a, 2);
    const again = quantize(a, 6);
    expect(again).toEqual(quantize(rainbow(), 6));
  });

  it("a swatch change recolours its texels and says which slabs changed", () => {
    const core = rainbow();
    quantize(core, 3);
    const dirty = applyPaletteEntry(core, 0, { r: 1, g: 2, b: 3, metal: 0.5 });
    expect(dirty.length).toBeGreaterThan(0);
    const n = core.slices[0].native!;
    const i = n.pal!.indexOf(0);
    expect([n.rgb[i * 3], n.rgb[i * 3 + 1], n.rgb[i * 3 + 2]]).toEqual([1, 2, 3]);
    expect(core.palette![0].metal).toBe(0.5);
    expect(applyPaletteEntry(core, 99, { r: 0 })).toEqual([]);
  });
});

describe("the brush", () => {
  const res = 4;
  /** Texel (x, z) of slab i sits at x = ((x + 0.5) / res - 0.5) * world, y = i * gap. */
  const at = (core: Core, x: number, z: number, i: number) => {
    const world = res * core.meta.pitch;
    return { x: ((x + 0.5) / res - 0.5) * world, y: i * core.meta.pitch, z: ((z + 0.5) / res - 0.5) * world };
  };

  it("paints inside the sphere only, and Erase puts back rgb0", () => {
    const core = coreOf(res, [[[0, 0], [3, 3]]]);
    const r = paintSphere(core, at(core, 0, 0, 0), 0.001, [1, 2, 3], false);
    expect(r).toEqual({ painted: 1, dirty: [0] });
    const n = core.slices[0].native!;
    expect(Array.from(n.rgb)).toEqual([1, 2, 3, 200, 40, 40]);
    paintSphere(core, at(core, 0, 0, 0), 0.001, [9, 9, 9], true);
    expect(Array.from(n.rgb)).toEqual([200, 40, 40, 200, 40, 40]);
  });

  it("Erase after Reduce restores the pre-Reduce colours", () => {
    const core = coreOf(res, [[[0, 0], [1, 0]]], [123, 45, 67]);
    quantize(core, 2);
    paintSphere(core, at(core, 0, 0, 0), 1, [0, 0, 0], true);
    expect(Array.from(core.slices[0].native!.rgb)).toEqual([123, 45, 67, 123, 45, 67]);
  });

  it("after Reduce it paints with the selected swatch and moves the texel to it", () => {
    const core = coreOf(res, [[[0, 0]]]);
    core.palette = [
      { r: 10, g: 10, b: 10, metal: 0, rough: 0.7, emit: 0 },
      { r: 50, g: 60, b: 70, metal: 0, rough: 0.7, emit: 0 },
    ];
    core.slices[0].native!.pal = new Uint8Array([0]);
    core.palSelected = 1;
    paintSphere(core, at(core, 0, 0, 0), 0.001, [255, 255, 255], false);
    expect(Array.from(core.slices[0].native!.rgb)).toEqual([50, 60, 70]);
    expect(core.slices[0].native!.pal![0]).toBe(1);
  });

  it("Undo restores the colours from before the stroke, once", () => {
    const core = coreOf(res, [[[0, 0]]]);
    beginStroke(core);
    paintSphere(core, at(core, 0, 0, 0), 1, [5, 5, 5], false);
    expect(undoStroke(core)).toBe(true);
    expect(Array.from(core.slices[0].native!.rgb)).toEqual([200, 40, 40]);
    expect(undoStroke(core)).toBe(false);
  });

  it("Drop samples the nearest texel", () => {
    const core = coreOf(res, [[[0, 0]], [[3, 3]]]);
    core.slices[1].native!.rgb.set([7, 8, 9]);
    expect(sampleVoxel(core, at(core, 3, 3, 1))).toEqual([7, 8, 9]);
    expect(sampleVoxel(core, at(core, 0, 0, 0))).toEqual([200, 40, 40]);
  });
});

describe("the sheet Save writes (getSheetBlob)", () => {
  it("is (res*layers)×res, slab i at x = i*res, every texel opaque, at native resolution", () => {
    const res = 4;
    const core = coreOf(res, [[[1, 2]], [[0, 0], [3, 3]]]);
    core.slices[1].native!.rgb.set([1, 2, 3, 4, 5, 6]);
    // The display crops do not apply: Main off would keep these, Texels and Layers only change the view.
    core.state.texels = 16;
    core.state.layers = 1;
    const px = sheetPixels(core);
    expect(px.width).toBe(8);
    expect(px.height).toBe(4);
    expect(px.data.length).toBe(8 * 4 * 4);
    const rgba = (x: number, y: number) => Array.from(px.data.slice((y * 8 + x) * 4, (y * 8 + x) * 4 + 4));
    expect(rgba(1, 2)).toEqual([200, 40, 40, 255]);
    expect(rgba(4, 0)).toEqual([1, 2, 3, 255]);
    expect(rgba(7, 3)).toEqual([4, 5, 6, 255]);
    expect(rgba(0, 0)).toEqual([0, 0, 0, 0]);
    let opaque = 0;
    for (let i = 3; i < px.data.length; i += 4) if (px.data[i]) opaque++;
    expect(opaque).toBe(3);
  });
});

describe("restBox", () => {
  it("is the stack at rest: extents across, layers × pitch tall, standing on y = 0", () => {
    const b = restBox({ layers: 10, resolution: 4, pitch: 0.1, extents: [0.4, 1, 0.3] });
    expect(b.min).toEqual([-0.2, 0, -0.15]);
    expect(b.max).toEqual([0.2, 1, 0.15]);
    expect(b.diagonal).toBeCloseTo(Math.hypot(0.4, 1, 0.3));
  });
});
