import { describe, expect, it } from "vitest";
import { INPUT_IMAGE, LODS_HINT, inputFile, lodButtons, lodLabel, lodTitle, pressedLod, turnaroundFigures } from "./lods";
import { PANEL_TITLES } from "./panels";

describe("Detail levels copy (legacy index.html 2131, 2136, 4969-4972, 5255)", () => {
  it("keeps the title and hint word for word", () => {
    expect(PANEL_TITLES.lods).toBe("Detail levels");
    expect(LODS_HINT).toBe("Show another version of the model. PSX is the PS1 low-poly; Hero is the full sculpt; Game is the reduced one.");
  });

  it("labels a LOD with its face count, as lodLabel did", () => {
    // Node ships full ICU and runs en-US here, so the legacy examples pin literally.
    expect(lodLabel({ label: "Game", faces: 5000 })).toBe("Game 5,000");
    expect(lodLabel({ label: "PSX", faces: 1184 })).toBe("PSX 1,184");
    expect(lodLabel({ label: "Hero" })).toBe("Hero");
    // A falsy count adds nothing (legacy `item.faces ? …`).
    expect(lodLabel({ label: "Proxy", faces: 0 })).toBe("Proxy");
  });

  it("titles a LOD with the legacy template, falling back to the id", () => {
    expect(lodTitle({ id: "game", label: "Game", file: "textured_game.glb" })).toBe("Show Game in the left pane (textured_game.glb).");
    expect(lodTitle({ id: "psx-clean", label: "", file: "psx_clean.glb" })).toBe("Show psx-clean in the left pane (psx_clean.glb).");
  });

  it("lists every LOD in lods.json order, even one, with Rigged first only when there is a rig", () => {
    const lods = [
      { id: "hero", file: "mesh.glb", label: "Hero" },
      { id: "game", file: "mesh_game.glb", label: "Game", faces: 5000 },
    ];
    expect(lodButtons(lods.slice(0, 1), false)).toEqual([{ id: "hero", label: "Hero", title: "Show Hero in the left pane (mesh.glb)." }]);
    expect(lodButtons(lods, true).map((b) => b.id)).toEqual(["rigged", "hero", "game"]);
    expect(lodButtons(lods, true)[0]).toEqual({ id: "rigged", label: "Rigged", title: "The rigged model, with its clips." });
    expect(lodButtons([], false)).toEqual([]);
  });

  it("presses the LOD on screen, and none for a file the list does not have", () => {
    const buttons = lodButtons(
      [
        { id: "hero", file: "mesh.glb", label: "Hero" },
        { id: "game", file: "mesh_game.glb", label: "Game" },
      ],
      false,
    );
    expect(pressedLod(buttons, "game")).toBe("game");
    expect(pressedLod(buttons, "rigged")).toBeNull();
    expect(pressedLod(buttons, "")).toBeNull();
  });
});

describe("turnaround figures (legacy fillTurnaround, index.html 4274-4296)", () => {
  it("falls back to one Input image figure, the shape view, when views.json lists none", () => {
    const fallback = [{ id: "input", file: "input.jpg", caption: "Input image", shape: true }];
    expect(INPUT_IMAGE).toBe("Input image");
    expect(turnaroundFigures(null)).toEqual(fallback);
    expect(turnaroundFigures({})).toEqual(fallback);
    expect(turnaroundFigures({ views: [] })).toEqual(fallback);
  });

  it("keeps views.json order, captions label or id, and marks role shape", () => {
    const doc = {
      views: [
        { id: "front", file: "views/front.png", role: "shape", label: "Front" },
        { id: "back", file: "views/back.png" },
      ],
    };
    expect(turnaroundFigures(doc)).toEqual([
      { id: "front", file: "views/front.png", caption: "Front", shape: true },
      { id: "back", file: "views/back.png", caption: "back", shape: false },
    ]);
  });

  it("takes the input the flow lists, else input.jpg as legacy", () => {
    expect(inputFile(null)).toBe("input.jpg");
    expect(inputFile(new Set(["mesh.glb"]))).toBe("input.jpg");
    expect(inputFile(new Set(["input.png", "mesh.glb"]))).toBe("input.png");
    expect(inputFile(new Set(["input.webp", "input.jpg"]))).toBe("input.jpg");
    expect(turnaroundFigures(null, "input.png")[0].file).toBe("input.png");
  });
});
