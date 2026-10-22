// The tool API's pure parts: pointer to NDC and the legacy carvePick hit (index.html 4889-4898, 5346-5364),
// currentFile from a loaded URL (legacy loadFrom 5128-5129), and arming. three runs fine without a DOM.
import { afterEach, describe, expect, it, vi } from "vitest";
import { BoxGeometry, Group, Mesh, MeshBasicMaterial, OrthographicCamera, Vector2 } from "three";
import { clientToNdc, diagonalOf, outRelFile, raycastModel, resetViewerTools, setViewerModel, toToolHit, useViewerTools, viewerTools } from "./tools";
import type { PreparedModel } from "./prepareModel";

afterEach(() => resetViewerTools());

describe("clientToNdc", () => {
  it("maps the canvas rect to -1..1 with y up", () => {
    const rect = { left: 10, top: 20, width: 200, height: 100 };
    expect(clientToNdc(10, 20, rect).toArray()).toEqual([-1, 1]);
    expect(clientToNdc(110, 70, rect).toArray()).toEqual([0, 0]);
    expect(clientToNdc(210, 120, rect).toArray()).toEqual([1, -1]);
  });
});

/** A 2x2x2 box at (5, 0, 0) inside a parent scaled by 4 and moved, like prepareModel's root. */
function scene() {
  const root = new Group();
  root.scale.setScalar(4);
  root.position.set(1, 2, 3);
  const mesh = new Mesh(new BoxGeometry(2, 2, 2), new MeshBasicMaterial());
  mesh.position.set(5, 0, 0);
  root.add(mesh);
  root.updateMatrixWorld(true);
  return { root, mesh };
}

describe("raycastModel", () => {
  it("returns the hit in world space and in the mesh's own space, with its world scale (what /clean gets)", () => {
    const { root, mesh } = scene();
    // Box centre in world: root.position + 4 * (5,0,0) = (21, 2, 3). Look down -z at it from in front.
    const cam = new OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
    cam.position.set(21, 2, 100);
    cam.lookAt(21, 2, 3);
    cam.updateMatrixWorld(true);
    const hit = raycastModel(root, cam, new Vector2(0, 0))!;
    expect(hit).not.toBeNull();
    expect(hit.object).toBe(mesh);
    // Front face of the box: local z = +1, world z = 3 + 4 * 1.
    expect(hit.point.x).toBeCloseTo(21);
    expect(hit.point.z).toBeCloseTo(7);
    expect(hit.local.x).toBeCloseTo(0);
    expect(hit.local.y).toBeCloseTo(0);
    expect(hit.local.z).toBeCloseTo(1);
    expect(hit.scale).toBeCloseTo(4);
    expect(hit.normal!.z).toBeCloseTo(1);
    expect(hit.uv).not.toBeNull();
    expect(typeof hit.faceIndex).toBe("number");
  });

  it("misses give null", () => {
    const { root } = scene();
    const cam = new OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
    cam.position.set(-50, 2, 100);
    cam.lookAt(-50, 2, 3);
    cam.updateMatrixWorld(true);
    expect(raycastModel(root, cam, new Vector2(0, 0))).toBeNull();
  });

  it("skips hits that are not meshes", () => {
    expect(toToolHit([{ object: new Group() } as never])).toBeNull();
  });
});

describe("diagonalOf", () => {
  it("is the world bounding-box diagonal, 0 when empty", () => {
    const { root } = scene();
    expect(diagonalOf(root)).toBeCloseTo(Math.sqrt(3) * 8);
    expect(diagonalOf(new Group())).toBe(0);
    expect(diagonalOf(null)).toBe(0);
  });
});

describe("outRelFile", () => {
  it("is the path under out/<stem>/ without the query, GLBs only", () => {
    expect(outRelFile("/out/cow/mesh.glb?t=123", "cow")).toBe("mesh.glb");
    expect(outRelFile("/out/cow/psx.glb", "cow")).toBe("psx.glb");
    expect(outRelFile("/out/cow/rig/rigged_allclips.glb?t=1", "cow")).toBe("rig/rigged_allclips.glb");
    expect(outRelFile("/out/my%20cow/mesh_clean.glb?t=1", "my cow")).toBe("mesh_clean.glb");
  });
  it("is empty for the /glb fallbacks and other assets (legacy: currentFile = \"\" unless .glb)", () => {
    expect(outRelFile("/glb/cow?lod=lookdev&t=1", "cow")).toBe("");
    expect(outRelFile("/out/pig/mesh.glb", "cow")).toBe("");
    expect(outRelFile("/out/cow/lods.json", "cow")).toBe("");
  });
});

describe("armTool", () => {
  const model = (n: number) => ({ root: new Group(), n }) as unknown as PreparedModel;

  it("one tool at a time: arming another disarms the first (Paint on turns Carve off, and now the reverse too)", () => {
    const a = { onDisarm: vi.fn() };
    const b = { onDisarm: vi.fn() };
    viewerTools.armTool(a);
    expect(viewerTools.armed()).toBe(a);
    viewerTools.armTool(b);
    expect(a.onDisarm).toHaveBeenCalledTimes(1);
    expect(viewerTools.armed()).toBe(b);
    viewerTools.armTool(null);
    expect(b.onDisarm).toHaveBeenCalledTimes(1);
    expect(viewerTools.armed()).toBeNull();
  });

  it("the returned disarm only disarms its own tool", () => {
    const a = { onDisarm: vi.fn() };
    const b = {};
    const offA = viewerTools.armTool(a);
    viewerTools.armTool(b);
    offA();
    expect(viewerTools.armed()).toBe(b);
  });

  it("a new model disarms; the same model does not", () => {
    const m1 = model(1);
    setViewerModel(m1, "mesh.glb");
    const a = { onDisarm: vi.fn() };
    viewerTools.armTool(a);
    setViewerModel(m1, "mesh.glb");
    expect(viewerTools.armed()).toBe(a);
    setViewerModel(model(2), "mesh_clean.glb");
    expect(a.onDisarm).toHaveBeenCalledTimes(1);
    expect(viewerTools.armed()).toBeNull();
    expect(viewerTools.currentFile()).toBe("mesh_clean.glb");
  });

  it("leaving the viewer disarms and forgets the model", () => {
    setViewerModel(model(1), "mesh.glb");
    const a = { onDisarm: vi.fn() };
    viewerTools.armTool(a);
    resetViewerTools();
    expect(a.onDisarm).toHaveBeenCalledTimes(1);
    expect(viewerTools.root()).toBeNull();
    expect(viewerTools.currentFile()).toBe("");
    expect(useViewerTools.getState().overlays).toEqual({});
  });

  it("loadFile and reload reject while no viewer is open", async () => {
    await expect(viewerTools.loadFile("mesh.glb")).rejects.toThrow("no glb loaded");
    await expect(viewerTools.reload()).rejects.toThrow("no glb loaded");
  });
});
