import { describe, expect, it } from "vitest";
import type { Asset, Project } from "./api";
import { assetProject, DELETE_PROJECT_MESSAGE, intoProject, membership, projectFilterName, projectOf } from "./projects";

const p = (id: string, name: string, assets: string[]): Project => ({ id, name, assets, created: 0, updated: 0 });
const farm = p("mlg-farming", "MLG Farming", ["hen", "cow"]);
const barn = p("barn", "Barn", ["hay"]);
const row = (id: string, project?: string | null): Asset => ({
  id, title: "", has_mesh: false, has_stack: false, has_flat: false, has_ideas: false, has_psx: false, output: null, input: null, preview: null, mtime: 0, project,
});

describe("projects", () => {
  it("membership and projectOf find the one project a stem is in", () => {
    const m = membership([farm, barn]);
    expect(m.get("cow")).toBe("mlg-farming");
    expect(m.get("hay")).toBe("barn");
    expect(m.has("pig")).toBe(false);
    expect(projectOf([farm, barn], "hay")?.name).toBe("Barn");
    expect(projectOf([farm, barn], "pig")).toBeNull();
    expect(projectOf(null, "hen")).toBeNull();
  });

  it("the loaded list wins over the row's own field (a move updates the list first)", () => {
    const m = membership([farm]);
    expect(assetProject(row("hen", null), m)).toBe("mlg-farming");
    expect(assetProject(row("hay", "barn"), m)).toBeNull();
    expect(assetProject(row("hay", "barn"), null)).toBe("barn");
    expect(assetProject(row("pig"), null)).toBeNull();
  });

  it("Make files into a real project only: Unsorted and a gone id are none", () => {
    expect(intoProject([farm], "mlg-farming")?.name).toBe("MLG Farming");
    expect(intoProject([farm], "_unsorted")).toBeNull();
    expect(intoProject([farm], "gone")).toBeNull();
    expect(intoProject([farm], undefined)).toBeNull();
  });

  it("names the filter", () => {
    expect(projectFilterName([farm], undefined)).toBe("All projects");
    expect(projectFilterName([farm], "_unsorted")).toBe("Unsorted");
    expect(projectFilterName([farm], "mlg-farming")).toBe("MLG Farming");
  });

  it("the delete message says the assets stay and become Unsorted", () => {
    expect(DELETE_PROJECT_MESSAGE("MLG Farming", 2)).toContain("Its 2 assets stay on disk and become Unsorted.");
    expect(DELETE_PROJECT_MESSAGE("Barn", 1)).toContain("Its asset stays on disk and becomes Unsorted.");
    expect(DELETE_PROJECT_MESSAGE("Empty", 0)).toContain("It has no assets.");
  });
});
