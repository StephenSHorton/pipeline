import { describe, expect, it } from "vitest";
import {
  CARVE_BALL,
  CARVE_LABEL,
  CARVE_TITLE,
  cleanAsk,
  cleanBlock,
  CLEANUP_HINT,
  cleanInto,
  CUT_A_COPY,
  FLOATERS_LABEL,
  FLOATERS_TITLE,
  floatersBody,
  lodFor,
  nestedFileNote,
  RADIUS,
  RADIUS_LABEL,
  RADIUS_TITLE,
  radiusText,
  sphereBody,
  WAIT_FOR_JOB,
  worldCarveRadius,
  writtenFile,
} from "./cleanup";

// The legacy cleanup cubby's copy, word for word (index.html 2090-2104, 5311-5317).
describe("Cleanup copy", () => {
  it("matches the legacy cubby", () => {
    expect(CLEANUP_HINT).toBe("Knock off generator scraps. Carve is a sphere delete in 3D.");
    expect(FLOATERS_LABEL).toBe("Floaters");
    expect(FLOATERS_TITLE).toBe("Delete disconnected bits. Keeps the largest solid.");
    expect(CARVE_LABEL).toBe("Carve");
    expect(CARVE_TITLE).toBe("Click the mesh to delete a sphere of volume. Orbit with right-drag while this is on.");
    expect(RADIUS_LABEL).toBe("Radius");
    expect(RADIUS_TITLE).toBe("Carve sphere size, as a fraction of the mesh.");
    expect(CUT_A_COPY).toBe("Cut a copy");
    expect(WAIT_FOR_JOB).toBe("Wait for the job on this asset to finish.");
  });

  it("keeps the legacy slider range, default and readout", () => {
    expect(RADIUS).toEqual({ min: 0.02, max: 0.35, step: 0.01, value: 0.08 });
    expect(radiusText(RADIUS.value)).toBe("0.08");
    expect(radiusText(0.1)).toBe("0.10");
    expect(radiusText(0.35)).toBe("0.35");
  });

  it("keeps the legacy hover ball", () => {
    expect(CARVE_BALL).toEqual({ color: "#c45c3a", opacity: 0.3 });
  });
});

describe("the confirm", () => {
  it("names the file on screen and where the copy goes", () => {
    expect(cleanAsk("psx.glb")).toBe("Cleanup cuts a copy of psx.glb and saves it as psx_clean.glb. The original stays.");
    expect(cleanAsk("mesh_hero.glb")).toBe("Cleanup cuts a copy of mesh_hero.glb and saves it as mesh_clean.glb. The original stays.");
    expect(cleanAsk("")).toBe("Cleanup cuts a copy of the mesh and saves it as mesh_clean.glb. The original stays.");
  });

  it("sends psx* to psx_clean.glb and everything else to mesh_clean.glb (clean_mesh.py clean_name)", () => {
    expect(cleanInto("psx.glb")).toBe("psx_clean.glb");
    expect(cleanInto("psx_clean.glb")).toBe("psx_clean.glb");
    expect(cleanInto("mesh_clean.glb")).toBe("mesh_clean.glb");
    expect(cleanInto("rigged.glb")).toBe("mesh_clean.glb");
    expect(cleanInto("")).toBe("mesh_clean.glb");
  });
});

describe("cleanBlock", () => {
  it("waits for a job on this asset first", () => {
    expect(cleanBlock("mesh.glb", true)).toBe("Wait for the job on this asset to finish.");
    expect(cleanBlock("rig/rigged_allclips.glb", true)).toBe("Wait for the job on this asset to finish.");
  });

  it("refuses a file under a folder, which /clean's from cannot name", () => {
    expect(cleanBlock("rig/rigged_allclips.glb", false)).toBe(nestedFileNote("rig/rigged_allclips.glb"));
    expect(nestedFileNote("rig/rigged_allclips.glb")).toBe("Cleanup can't cut rig/rigged_allclips.glb. Show a detail level first.");
  });

  it("lets a bare file and the /glb fallback through", () => {
    expect(cleanBlock("mesh.glb", false)).toBe("");
    expect(cleanBlock("rigged.glb", false)).toBe("");
    expect(cleanBlock("", false)).toBe("");
  });
});

describe("POST /clean bodies (index.html 5326, 5357-5364)", () => {
  it("Floaters keeps the largest part of the file on screen", () => {
    expect(floatersBody("brown-hen", "psx.glb")).toEqual({ stem: "brown-hen", op: "floaters", keep: 1, from: "psx.glb" });
    expect(JSON.stringify(floatersBody("hen", ""))).toBe('{"stem":"hen","op":"floaters","keep":1,"from":""}');
  });

  it("a carve sends the mesh-local hit and the world radius over the mesh's world scale", () => {
    const body = sphereBody("hen", "mesh_clean.glb", { x: 0.1, y: -0.2, z: 0.3 }, 0.5, 2);
    expect(body).toEqual({ stem: "hen", op: "sphere", center: [0.1, -0.2, 0.3], radius: 0.25, from: "mesh_clean.glb" });
    expect(Object.keys(body)).toEqual(["stem", "op", "center", "radius", "from"]);
    expect(sphereBody("hen", "a.glb", { x: 0, y: 0, z: 0 }, 0.4, 0).radius).toBe(0.4);
  });
});

describe("worldCarveRadius", () => {
  it("is the diagonal times the slider", () => {
    expect(worldCarveRadius(2, 0.08)).toBeCloseTo(0.16);
  });

  it("counts an empty box as 1 (legacy carve sent radius 0 there)", () => {
    expect(worldCarveRadius(0, 0.08)).toBeCloseTo(0.08);
  });
});

describe("after a clean", () => {
  it("loads the file the server named, else mesh_clean.glb", () => {
    expect(writtenFile({ file: "psx_clean.glb" })).toBe("psx_clean.glb");
    expect(writtenFile({})).toBe("mesh_clean.glb");
    expect(writtenFile({ file: "" })).toBe("mesh_clean.glb");
    expect(writtenFile(null)).toBe("mesh_clean.glb");
  });

  it("finds the Detail levels entry patch_lods wrote", () => {
    const lods = [
      { id: "clean", file: "mesh_clean.glb", label: "Clean" },
      { id: "psx-clean", file: "psx_clean.glb", label: "PSX clean" },
      { id: "hero", file: "mesh_hero.glb", label: "Hero" },
    ];
    expect(lodFor(lods, "psx_clean.glb")?.id).toBe("psx-clean");
    expect(lodFor(lods, "mesh_clean.glb")?.id).toBe("clean");
    expect(lodFor(lods, "other.glb")).toBeNull();
  });
});
