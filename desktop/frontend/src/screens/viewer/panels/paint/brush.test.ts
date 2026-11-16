// The mesh brush against fake canvases (no DOM): what it takes, stamps, erases, undoes, samples and saves.
import { describe, expect, it } from "vitest";
import { BoxGeometry, Group, Mesh, MeshBasicMaterial, MeshLambertMaterial, Texture } from "three";
import { MeshBrush, type BrushDeps, type PaintCanvas, type PaintCtx } from "./brush";

interface Fake extends PaintCanvas {
  id: number;
  log: string[];
}

function fakes() {
  const made: Fake[] = [];
  const textures: Texture[] = [];
  const deps: BrushDeps = {
    makeCanvas(width, height) {
      const log: string[] = [];
      const ctx: PaintCtx = {
        fillStyle: "",
        drawImage: (img) => log.push(`draw ${(img as Fake).id ?? "img"}`),
        save: () => log.push("save"),
        restore: () => log.push("restore"),
        beginPath: () => {},
        arc: (x, y, r) => log.push(`arc ${x} ${y} ${r.toFixed(2)}`),
        clip: () => log.push("clip"),
        fill: () => log.push(`fill ${String(ctx.fillStyle)}`),
        getImageData: (x, y) => (log.push(`read ${x} ${y}`), { data: [46, 204, 113, 255] }),
      };
      const c: Fake = {
        id: made.length,
        log,
        width,
        height,
        getContext: () => ctx,
        toBlob: (done) => done(new Blob(["png"], { type: "image/png" })),
      };
      made.push(c);
      return c;
    },
    makeTexture(_canvas, from) {
      const t = new Texture();
      t.flipY = from.flipY;
      textures.push(t);
      return t;
    },
  };
  return { deps, made, textures };
}

function model(withMap = true) {
  const map = new Texture({ width: 100, height: 50 } as never);
  map.flipY = false;
  const mat = new MeshLambertMaterial(withMap ? { map } : {});
  const root = new Group();
  root.add(new Mesh(new BoxGeometry(), mat));
  return { root, map, mat };
}

describe("MeshBrush", () => {
  it("takes nothing from an untextured model (or Clay on): mesh paint and Save's mesh part do nothing", async () => {
    const { deps, made } = fakes();
    const b = new MeshBrush(deps);
    const { root } = model(false);
    b.stamp(root, { x: 0.5, y: 0.5 }, 1, 10, [1, 2, 3], false);
    expect(b.ensure(root)).toBe(false);
    expect(b.ensure(null)).toBe(false);
    expect(made).toHaveLength(0);
    expect(b.painted).toBe(false);
    expect(await b.toBlob()).toBeNull();
  });

  it("copies the map to a canvas (plus a pristine copy for Erase) and swaps it into the materials", () => {
    const { deps, made, textures } = fakes();
    const b = new MeshBrush(deps);
    const { root, mat } = model();
    expect(b.ensure(root)).toBe(true);
    expect(made.map((c) => [c.width, c.height])).toEqual([
      [100, 50],
      [100, 50],
    ]);
    expect(mat.map).toBe(textures[0]);
    expect(b.ensure(root)).toBe(true);
    expect(made).toHaveLength(2);
  });

  it("stamps a circle of the colour at the uv, or the original texture with erase", () => {
    const { deps, made } = fakes();
    const b = new MeshBrush(deps);
    const { root } = model();
    b.stamp(root, { x: 0.5, y: 0.25 }, 0.4, 10, [46, 204, 113], false);
    const paint = made[0];
    // (0.4 / 10) * 100 * 1.6 = 6.4 px; v not flipped: y = 0.25 * 50.
    expect(paint.log.slice(1)).toEqual(["save", "arc 50 12.5 6.40", "clip", "arc 50 12.5 6.40", "fill rgb(46,204,113)", "restore"]);
    paint.log.length = 0;
    b.stamp(root, { x: 0.5, y: 0.25 }, 0.4, 10, [0, 0, 0], true);
    expect(paint.log).toEqual(["save", "arc 50 12.5 6.40", "clip", `draw ${made[1].id}`, "restore"]);
    paint.log.length = 0;
    b.stamp(root, null, 0.4, 10, [0, 0, 0], false);
    expect(paint.log).toEqual([]);
  });

  it("Undo goes back one stroke, and again repeats the same snapshot", () => {
    const { deps, made } = fakes();
    const b = new MeshBrush(deps);
    const { root } = model();
    b.undo();
    b.begin(root);
    const snap = made[2];
    made[0].log.length = 0;
    b.undo();
    b.undo();
    expect(made[0].log).toEqual([`draw ${snap.id}`, `draw ${snap.id}`]);
  });

  it("the dropper reads the pixel under the uv as hex", () => {
    const { deps, made } = fakes();
    const b = new MeshBrush(deps);
    const { root } = model();
    expect(b.sample(root, { x: 0.5, y: 0.5 })).toBe("#2ecc71");
    expect(made[0].log.at(-1)).toBe("read 50 25");
    expect(b.sample(root, null)).toBeNull();
  });

  it("Save's PNG, and forget() drops it and disposes both textures", async () => {
    const { deps, textures } = fakes();
    const b = new MeshBrush(deps);
    const { root, map } = model();
    b.ensure(root);
    expect((await b.toBlob())?.type).toBe("image/png");
    let disposed = 0;
    map.addEventListener("dispose", () => disposed++);
    textures[0].addEventListener("dispose", () => disposed++);
    b.forget();
    expect(disposed).toBe(2);
    expect(b.painted).toBe(false);
    expect(await b.toBlob()).toBeNull();
  });

  it("takes the last mapped material, arrays included", () => {
    const { deps, made } = fakes();
    const b = new MeshBrush(deps);
    const { root } = model();
    const late = new Texture({ width: 8, height: 8 } as never);
    root.add(new Mesh(new BoxGeometry(), [new MeshBasicMaterial(), new MeshBasicMaterial({ map: late })]));
    b.ensure(root);
    expect(made[0].width).toBe(8);
  });
});
