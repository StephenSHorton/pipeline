import { describe, expect, it } from "vitest";
import type { Lod } from "@/lib/api";
import { fileOf, fileTag, modelCandidates, pickLod, preferredClip, riggedFiles, slugStem, viewableLods } from "./candidates";

const strip = (u: string) => u.replace(/([?&])t=\d+/, "$1").replace(/[?&]$/, "");

const CHICKEN: Lod[] = [
  { id: "photo", file: "textured_photo.glb", label: "Photo bake" },
  { id: "psx", file: "psx.glb", label: "PSX", faces: 1018 },
  { id: "hero", file: "mesh.glb", label: "Hero" },
  { id: "lookdev", file: "textured.glb", label: "Lookdev" },
  { id: "pixel-a", file: "pixel.glb", label: "Pixel" },
  { id: "stk", file: "stack.glb", label: "Stack", kind: "stack" },
];

describe("viewer candidates", () => {
  it("offers only finished models: no stack, pixel, hero, lookdev, game or proxy", () => {
    expect(viewableLods(CHICKEN).map((l) => l.id)).toEqual(["photo", "psx"]);
    const old: Lod[] = [
      { id: "paint", file: "textured_paint.glb", label: "Painted" },
      { id: "mv", file: "mesh_mv.glb", label: "2mv" },
      { id: "hero", file: "mesh.glb", label: "Hero" },
      { id: "lookdev", file: "textured.glb", label: "Lookdev", faces: 80000 },
      { id: "game", file: "mesh_game.glb", label: "Game" },
      { id: "proxy", file: "mesh_proxy.glb", label: "Proxy" },
      { id: "clean", file: "mesh_clean.glb", label: "Clean" },
      { id: "psx-clean", file: "psx_clean.glb", label: "PSX clean" },
    ];
    // An older asset with no photo bake: its paint (the lookdev textured.glb) stands in as the Photo bake.
    expect(viewableLods(old).map((l) => [l.id, l.label])).toEqual([
      ["paint", "Painted"],
      ["lookdev", "Photo bake"],
      ["clean", "Clean"],
      ["psx-clean", "PSX clean"],
    ]);
    // An untextured lookdev mesh is a step, not a finished model; PBR paint stands in when there is no textured.glb.
    expect(viewableLods([{ id: "lookdev", file: "mesh_lookdev.glb", label: "Lookdev" }])).toEqual([]);
    expect(viewableLods([{ id: "pbr", file: "pbr.glb", label: "PBR" }, { id: "hero", file: "mesh.glb", label: "Hero" }])).toEqual([
      { id: "pbr", file: "pbr.glb", label: "Photo bake" },
    ]);
    expect(viewableLods(null)).toEqual([]);
  });

  it("picks psx for a lookdev preference, else prefer > photo > lookdev > pbr > first", () => {
    const lods = viewableLods(CHICKEN);
    expect(pickLod(lods)?.id).toBe("psx");
    expect(pickLod(lods, "photo")?.id).toBe("photo");
    expect(pickLod(lods, "hero")?.id).toBe("photo");
    expect(pickLod(lods, "nope")?.id).toBe("photo");
    expect(pickLod([{ id: "x", file: "x.glb", label: "X" }, { id: "lookdev", file: "textured.glb", label: "Photo bake" }], "nope")?.id).toBe("lookdev");
    expect(pickLod([{ id: "x", file: "x.glb", label: "X" }], "nope")?.id).toBe("x");
    expect(pickLod([])).toBeNull();
  });

  it("puts the LOD file first, then the legacy meshUrls chain", () => {
    const c = modelCandidates({ stem: "qmv-new-chicken", lods: CHICKEN, hasRig: false, rigFiles: [], t: 1 });
    expect(c.lod).toBe("psx");
    expect(c.urls.map(strip)).toEqual([
      "/out/qmv-new-chicken/psx.glb",
      "/glb/qmv-new-chicken?lod=psx",
      "/glb/qmv-new-chicken?lod=lookdev",
      "/glb/qmv-new-chicken",
      "/out/qmv-new-chicken/mesh_lookdev.glb",
      "/out/qmv-new-chicken/mesh_hero.glb",
      "/out/qmv-new-chicken/mesh.glb",
      "/out/qmv-new-chicken/textured.glb",
    ]);
  });

  it("opens the legacy pick by default even when rigged; ?lod=rigged opens the rigged model, allclips first", () => {
    const files = riggedFiles({ allClips: true, rigJsonRigged: "rigged.glb" });
    expect(files).toEqual(["rig/rigged_allclips.glb", "rigged.glb"]);
    expect(modelCandidates({ stem: "s", lods: CHICKEN, hasRig: true, rigFiles: files, t: 1 }).lod).toBe("psx");
    const c = modelCandidates({ stem: "s", lod: "rigged", lods: CHICKEN, hasRig: true, rigFiles: files, t: 1 });
    expect(c.lod).toBe("rigged");
    expect(c.urls.slice(0, 3).map(strip)).toEqual(["/out/s/rig/rigged_allclips.glb", "/out/s/rigged.glb", "/glb/s?lod=lookdev"]);
  });

  it("an explicit LOD beats the rig; a hidden one (an old ?lod=hero) opens the photo bake; no lods.json leaves the /glb chain", () => {
    expect(modelCandidates({ stem: "s", lod: "photo", lods: CHICKEN, hasRig: true, rigFiles: ["rigged.glb"], t: 1 }).urls.map(strip)[0]).toBe("/out/s/textured_photo.glb");
    const hero = modelCandidates({ stem: "s", lod: "hero", lods: CHICKEN, hasRig: true, rigFiles: ["rigged.glb"], t: 1 });
    expect([hero.lod, strip(hero.urls[0])]).toEqual(["photo", "/out/s/textured_photo.glb"]);
    const bare = modelCandidates({ stem: "s", lods: [], hasRig: false, rigFiles: [], t: 1 });
    expect(bare.lod).toBe("");
    expect(bare.urls.map(strip)[0]).toBe("/glb/s?lod=lookdev");
    expect(bare.urls).toHaveLength(6);
  });

  it("plays Idle first, like the legacy page", () => {
    expect(preferredClip(["Walk", "Run", "Idle", "Death"])).toBe("Idle");
    expect(preferredClip(["Idle_A", "Idle"])).toBe("Idle_A");
    expect(preferredClip(["Run", "walk"])).toBe("walk");
    expect(preferredClip(["Peck"])).toBe("Peck");
    expect(preferredClip([])).toBeNull();
  });

  it("slugs a rename like the legacy page", () => {
    expect(slugStem("The Big Red Barn")).toBe("big-red-barn");
    expect(slugStem("chicken.png")).toBe("chicken");
    expect(slugStem("  !! ")).toBe("");
    expect(slugStem("alpha beta gamma delta epsilon zeta eta theta")).toBe("alpha-beta-gamma-delta-epsilon-zeta-eta");
  });

  it("tags and names files", () => {
    expect(fileOf("/out/s/rig/rigged_allclips.glb?t=1")).toBe("rigged_allclips.glb");
    expect(fileTag("rigged_allclips.glb", "rigged")).toBe("Rigged model");
    expect(fileTag("psx.glb", "psx")).toBe("PS1 model");
    expect(fileTag("textured_photo.glb", "photo")).toBe("Photo bake");
    expect(fileTag("mesh.glb", "hero")).toBe("3D model");
  });
});
