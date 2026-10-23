// The Weights panel on a real (tiny) SkinnedMesh: edits land in the mesh's own skin attributes (so a clip shows them),
// the colours go on and come off, a reload of the same mesh carries the edit, undo spans the session, and Save posts
// the body and rechecks.
import { afterEach, describe, expect, it, vi } from "vitest";
import { Bone, BufferAttribute, BufferGeometry, Group, MeshLambertMaterial, Skeleton, SkinnedMesh, Uint16BufferAttribute, Uint8BufferAttribute, Vector3 } from "three";
import { saveWeights, useWeights } from "./weightsApi";
import { WeightsSession } from "./weightsView";

function sheepish(opts: { u8?: boolean } = {}) {
  const root = new Bone();
  root.name = "hips";
  const thigh = new Bone();
  thigh.name = "back_leg_upper_l";
  thigh.position.set(1, 0, 0);
  root.add(thigh);
  const geo = new BufferGeometry();
  // Points along x: 0..3, the middle two copies of one point at x = 1.
  geo.setAttribute("position", new BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0]), 3));
  geo.setIndex([0, 1, 3, 2, 3, 4]);
  const ji = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
  geo.setAttribute("skinIndex", opts.u8 ? new Uint8BufferAttribute(new Uint8Array(ji), 4) : new Uint16BufferAttribute(new Uint16Array(ji), 4));
  geo.setAttribute("skinWeight", new BufferAttribute(new Float32Array([1, 0, 0, 0, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 4));
  const mesh = new SkinnedMesh(geo, new MeshLambertMaterial());
  mesh.add(root);
  mesh.bind(new Skeleton([root, thigh]));
  mesh.userData.gltfPrimitive = [0, 0];
  const scene = new Group();
  scene.add(mesh);
  scene.updateMatrixWorld(true);
  return { scene, mesh };
}

afterEach(() => useWeights.setState({ saving: false }));

describe("WeightsSession", () => {
  it("edits the mesh's own skin attributes, in step across corner copies", () => {
    const { scene, mesh } = sheepish();
    const s = WeightsSession.bind(scene, "sheep", null)!;
    expect(s.names).toEqual(["hips", "back_leg_upper_l"]);
    expect(s.parents).toEqual([-1, 0]);
    const sw = mesh.geometry.getAttribute("skinWeight") as BufferAttribute;
    const v0 = sw.version;
    s.begin();
    s.dab("erase", 1, new Vector3(3, 0, 0), 1.5, 1);
    s.end();
    expect(sw.version).toBeGreaterThan(v0); // flagged for the GPU
    // Point 4 belonged to the thigh alone: Erase handed it to the hips (its parent).
    expect(s.parts[0].edit.weightOf(4, 0)).toBeCloseTo(1);
    // Corners 1 and 2 (same position) were split: mended to the same weights by the brush where it reached them.
    s.begin();
    s.dab("add", 1, new Vector3(1, 0, 0), 0.5, 0.5);
    s.end();
    const e = s.parts[0].edit;
    expect(e.weightOf(1, 1)).toBeCloseTo(e.weightOf(2, 1));
    expect(s.canUndo).toBe(true);
    s.undo();
    s.undo();
    expect(e.weightOf(4, 1)).toBe(1);
    expect(s.changedPoints()).toBe(0);
  });

  it("converts u8 joints to editable u16 and keeps them", () => {
    const { scene, mesh } = sheepish({ u8: true });
    const s = WeightsSession.bind(scene, "sheep", null)!;
    expect(mesh.geometry.getAttribute("skinIndex").array).toBeInstanceOf(Uint16Array);
    expect(s.parts[0].edit.weightOf(1, 1)).toBeCloseTo(0.5);
  });

  it("puts colours on and takes them off, the material given back", () => {
    const { scene, mesh } = sheepish();
    const mat = mesh.material;
    const s = WeightsSession.bind(scene, "sheep", null)!;
    s.show({ mode: "heat", bone: 1 });
    expect(mesh.material).not.toBe(mat);
    expect(mesh.geometry.getAttribute("color")).toBeTruthy();
    s.show({ mode: "parts", bone: -1 });
    s.release();
    expect(mesh.material).toBe(mat);
    expect(mesh.geometry.getAttribute("color")).toBeUndefined();
  });

  it("a reload of the same mesh carries the unsaved edit and its undo", () => {
    const a = sheepish();
    const s1 = WeightsSession.bind(a.scene, "sheep", null)!;
    s1.grabBegin(1);
    s1.grabTo(1, 50);
    s1.end();
    const b = sheepish();
    const s2 = WeightsSession.bind(b.scene, "sheep", s1)!;
    expect(s2.parts[0].edit.weightOf(4, 1)).toBeCloseTo(0.5);
    expect((b.mesh.geometry.getAttribute("skinWeight").array as Float32Array)[16]).toBeCloseTo(0.5);
    expect(s2.isOn(b.scene)).toBe(true);
    expect(s2.isOn(a.scene)).toBe(false);
    expect(Math.round(s2.grabAll()[1])).toBe(50);
    s2.undo();
    expect(s2.parts[0].edit.weightOf(4, 1)).toBe(1);
    // Another asset starts over.
    const s3 = WeightsSession.bind(sheepish().scene, "goat", s2)!;
    expect(s3.canUndo).toBe(false);
  });

  it("Save posts every point and rechecks; a refusal keeps the edit", async () => {
    const { scene } = sheepish();
    const s = WeightsSession.bind(scene, "sheep", null)!;
    s.grabBegin(1);
    s.grabTo(1, 50);
    s.end();
    const post = vi.fn(async () => ({ ok: true, stem: "sheep", take: "t2", files: ["rigged.glb"], changed: 4, fbx: null, fbx_note: "Blender not found.", blender: false }));
    const recheck = vi.fn(async () => {});
    const say = vi.fn();
    const out = await saveWeights(s, { post: post as never, recheck, say });
    expect(out?.take).toBe("t2");
    const [path, init] = post.mock.calls[0] as unknown as [string, RequestInit];
    expect(path).toBe("/rig/sheep/weights");
    const body = JSON.parse(String(init.body));
    expect(body.primitives[0]).toMatchObject({ mesh: 0, primitive: 0, count: 5 });
    expect(recheck).toHaveBeenCalledWith("sheep");
    expect(say).toHaveBeenCalledWith("Weights saved (4 points changed). Blender not found.");
    expect(s.changedPoints()).toBe(0); // what is saved now

    s.grabBegin(1);
    s.grabTo(1, 50);
    s.end();
    const busy = vi.fn(async () => {
      throw new Error("a job is writing this asset");
    });
    expect(await saveWeights(s, { post: busy as never, recheck, say })).toBeNull();
    expect(say).toHaveBeenLastCalledWith("a job is writing this asset");
    expect(s.changedPoints()).toBeGreaterThan(0);
    expect(useWeights.getState().saving).toBe(false);
  });
});
