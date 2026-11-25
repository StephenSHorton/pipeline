import { describe, expect, it } from "vitest";
import { jobStage } from "@/app/open";
import type { Asset } from "@/lib/api";
import { jobDoing, jobDraws, jobVerb } from "@/lib/jobs";
import {
  buildingBody,
  buildingChips,
  buildingName,
  buildingProblem,
  buildingReadyText,
  DEFAULT_BUILDING,
  joinable,
  joinBody,
  joinName,
  joinProblem,
  MAX_TIERS,
  sourceBays,
  tiersText,
  towerable,
  towerBody,
  towerName,
  towerProblem,
  towerTiers,
  type BuildingChoice,
} from "./building";

const brick: BuildingChoice = {
  ...DEFAULT_BUILDING,
  wall: "  a three-storey red brick building, sash windows ",
  front: "Add a shop front and a green awning",
};

describe("a building from the app (POST /2d/building)", () => {
  it("sends the walls' words, the look and a flat roof, leaving width and depth as drawn", () => {
    expect(buildingBody(brick, "three-storey-red-brick", "city")).toEqual({
      stem: "three-storey-red-brick",
      wall: "a three-storey red brick building, sash windows",
      front: "Add a shop front and a green awning",
      look: "crisp",
      roof: "flat",
      project: "city",
    });
  });

  it("sends every choice the command line has: back, bays, a gable, its ridge and roof colour", () => {
    const c: BuildingChoice = { ...brick, back: "Add a door", look: "smooth", wide: 5, deep: 2, roof: "gable", ridge: "along", roofColour: "#8A4A34" };
    expect(buildingBody(c, "shop", null)).toEqual({
      stem: "shop",
      wall: "a three-storey red brick building, sash windows",
      front: "Add a shop front and a green awning",
      back: "Add a door",
      look: "smooth",
      wide: 5,
      deep: 2,
      roof: "gable",
      ridge: "along",
      roof_colour: "#8a4a34",
      project: null,
    });
    expect(buildingChips(c)).toEqual(["Smooth", "Wide 5 bays", "Deep 2 bays", "Gable roof, ridge along, #8A4A34"]);
  });

  it("says what is missing before it starts", () => {
    expect(buildingProblem(DEFAULT_BUILDING)).toBe("Describe the plain wall first.");
    expect(buildingProblem({ ...brick, front: " " })).toBe("Say what to add to the front.");
    expect(buildingProblem({ ...brick, roof: "gable", roofColour: "purple" })).toMatch(/roof colour/);
    expect(buildingProblem({ ...brick, roof: "gable", roofColour: "tile" })).toBe("");
  });

  it("is named from its name, else from the plain wall's first words", () => {
    expect(buildingName({ name: "The Corner Cafe", wall: "x" })).toBe("corner-cafe");
    expect(buildingName(brick)).toBe("three-storey-red-brick");
    expect(buildingName({ name: "", wall: "" })).toBe("building");
  });

  it("is a job the app knows: its words, its wall previews, and it opens on Stack", () => {
    const job = { stem: "corner-cafe", kind: "building" as const };
    expect(jobDoing(job)).toBe("Making the building corner cafe");
    expect(jobVerb(job)).toBe("make");
    expect(jobDraws(job)).toBe(true);
    expect(jobStage(job)).toBe("stack");
    expect(buildingReadyText("corner cafe", { stack: { file: "stack/stack.json", size: { width: 160, length: 96, height: 210 }, colors: 24, layers: 210 }, bays: { drawn: 5, wide: 3 } })).toBe(
      "Building ready: corner cafe · 160 by 96 cells, 210 layers · 3 by 5 window bays",
    );
  });
});

describe("joining buildings (POST /2d/join)", () => {
  const asset = (id: string, p: Partial<Asset>): Asset =>
    ({ id, title: id, has_mesh: false, has_stack: true, has_flat: false, has_ideas: false, has_psx: false, output: "stack", input: null, preview: null, mtime: 0, ...p }) as Asset;

  it("offers only buildings", () => {
    const list = [asset("cafe", { building: true }), asset("hen", {}), asset("old", { building: true, has_stack: false })];
    expect(joinable(list).map((a) => a.id)).toEqual(["cafe"]);
  });

  it("needs two or more for a row, exactly two for a corner", () => {
    expect(joinProblem("row", ["a"])).toMatch(/two or more/);
    expect(joinProblem("row", ["a", "b", "c"])).toBe("");
    expect(joinProblem("corner", ["a", "b", "c"])).toMatch(/two buildings/);
    expect(joinProblem("corner", ["a", "b"])).toBe("");
    expect(joinProblem("row", Array.from({ length: 9 }, (_, i) => `b${i}`))).toMatch(/at most 8/);
  });

  it("sends the parts in order, named from the first part when you give no name", () => {
    expect(joinName("", "row", ["city_cafe", "city_bakery"])).toBe("cafe-row");
    expect(joinName("", "corner", ["city_shop_wide", "city_townhouse"])).toBe("shop-wide-corner");
    expect(joinName("High Street", "row", ["a", "b"])).toBe("high-street");
    expect(joinBody("cafe-row", "row", ["city_cafe", "city_bakery"], null)).toEqual({ stem: "cafe-row", layout: "row", parts: ["city_cafe", "city_bakery"], project: null });
  });
});

describe("a tower from a building (Stack stage, POST /2d/tower)", () => {
  const drawn = { maker: "walls", shape: "code", bays: { drawn: 4, wide: 4, deep: 4 } };

  it("is offered only on a building drawn from walls", () => {
    expect(towerable(drawn)).toBe(true);
    expect(towerable({ maker: "walls", shape: "tower" })).toBe(false); // a tower has no drawn walls of its own
    expect(towerable({ maker: "walls" })).toBe(false); // nor has a joined building
    expect(towerable({ maker: "views" })).toBe(false);
    expect(towerable(null)).toBe(false);
  });

  it("starts from the building's own bays, as made", () => {
    expect(sourceBays(drawn)).toEqual([4, 4]);
    expect(sourceBays({ maker: "walls", shape: "code", bays: { drawn: 4, wide: 7, deep: 5 } })).toEqual([7, 5]);
    expect(sourceBays({ maker: "walls", shape: "code", bays: { drawn: 3 } })).toEqual([3, 3]);
    expect(sourceBays({ maker: "walls", shape: "code" })).toBeNull();
  });

  it("steps each tier one window bay in on every side", () => {
    expect(towerTiers(7, 5, 3)).toEqual([
      [7, 5],
      [5, 3],
      [3, 1],
    ]);
    expect(tiersText([4, 4], 2)).toBe("4 by 4 bays, then 2 by 2");
  });

  it("says plainly when the bays run out, never giving fewer tiers", () => {
    expect(towerProblem([7, 5], 3, 4)).toBe("");
    expect(towerProblem([4, 4], 3, 4)).toBe(
      "This building is 4 window bays across and 4 along. Each tier steps in one bay on every side, so it has room for 2 tiers, not 3.",
    );
    expect(towerProblem([4, 4], MAX_TIERS + 1, 4)).toMatch(/1 to 5 tiers/);
    expect(towerProblem([4, 4], 2, 0)).toMatch(/1 to 40 floors/);
    expect(towerProblem(null, 2, 4)).toMatch(/does not say its window bays/);
  });

  it("sends two numbers and the building, under a new name", () => {
    expect(towerName("city_bakery")).toBe("city-bakery-tower");
    expect(towerBody("city-bakery-tower", "city_bakery", 3, 4, "city")).toEqual({
      stem: "city-bakery-tower",
      from: "city_bakery",
      tiers: 3,
      floors: 4,
      project: "city",
    });
  });
});
