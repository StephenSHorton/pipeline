import { describe, expect, it } from "vitest";
import type { FlowEntry, FlowState, Stage } from "@/lib/flow";
import {
  exportAction,
  exportCards,
  fbxLine,
  finderFailed,
  LEAD_EMPTY,
  leadText,
  psxCard,
  rigCard,
  type ExportDoc,
  type PlainCard,
} from "./export";

const entry = (stage: Stage, live: boolean, stale: string[] = []): FlowEntry => ({
  stage,
  pick: null,
  takes: [],
  live,
  files: [],
  unsaved: false,
  stale,
  upstream: [],
});
const flowOf = (live: Partial<Record<Stage, boolean>>, stale: string[] = []): FlowState => ({
  ok: true,
  stem: "x",
  stages: (["ideas", "concept", "flat", "mesh", "rig", "stack", "export"] as const).map((s) =>
    entry(s, !!live[s], s === "export" ? stale : []),
  ),
});

// GET /export/chicken-canon (trimmed): rigged avian + PS1, no 2D, no stack.
const chicken: ExportDoc = {
  made: 1792522526,
  concept: "concept.png",
  "3d": {
    model: { file: "3d/model.glb" },
    game: { file: "3d/game.glb" },
    proxy: { file: "3d/proxy.glb", faces: 2000 },
    rigged: {
      file: "3d/rigged.glb",
      class: "avian",
      clips: ["Idle", "Walk", "Run", "Peck", "Flap", "Sleep", "Death"],
      joints: 55,
      fbx: "3d/rig/chicken.fbx",
      fbx_texture: "3d/rig/chicken.png",
    },
    psx: {
      file: "3d/psx.glb",
      obj: "3d/psx/chicken.obj",
      texture: "3d/psx/chicken.png",
      axes: { obj: "z-up, +x front" },
      triangles: 1200,
      atlas: 256,
      height: 48.0,
      size: [38.2, 19.5, 48.2],
    },
  },
};
const ps1 = { output: "ps1" as const };
const exported = flowOf({ concept: true, mesh: true, rig: true, export: true });

describe("lead", () => {
  it("before an export: the explainer, also when a doc exists but the export stage holds nothing", () => {
    expect(leadText("x", null, entry("export", false), "")).toBe(LEAD_EMPTY);
    expect(leadText("x", chicken, entry("export", false), "now")).toBe(LEAD_EMPTY);
    expect(LEAD_EMPTY).toBe(
      "Export copies what this asset made into one folder a game can load, with asset.json saying what each file is. Press Export below.",
    );
  });
  it("with a doc: the folder and the date, plus the stale note", () => {
    expect(leadText("chicken-canon", chicken, entry("export", true), "D")).toBe("out/chicken-canon/export/ · D");
    expect(leadText("chicken-canon", chicken, entry("export", true, ["rig"]), "D")).toBe(
      "out/chicken-canon/export/ · D · made from older picks, export again to update",
    );
  });
});

describe("cards", () => {
  it("chicken-canon: Rigged, PS1, then Concept and the other 3D files; no 2D or stack card for a ps1 intent", () => {
    const cards = exportCards(exported, ps1, chicken);
    expect(cards.map((c) => c.key)).toEqual(["rig", "psx", "concept", "mesh"]);
    const concept = cards[2] as PlainCard;
    expect(concept).toMatchObject({ title: "Concept", lines: ["concept.png"], img: "concept.png", missing: false });
    expect(cards[3]).toMatchObject({
      title: "Other 3D files",
      lines: ["Full detail: model.glb", "Game detail: game.glb", `Collision proxy: proxy.glb · ${(2000).toLocaleString()} faces`, "Rigged: rigged.glb"],
    });
  });

  it("no intent: every missing stage shows as Not in this export (dashed); plain 3D without psx", () => {
    const doc: ExportDoc = { made: 1, "3d": { model: { file: "3d/model.glb" } } };
    const cards = exportCards(exported, null, doc) as PlainCard[];
    expect(cards.map((c) => [c.title, c.lines, c.missing])).toEqual([
      ["2D sheet", ["Not in this export"], true],
      ["Sprite stack", ["Not in this export"], true],
      ["3D", ["Full detail: model.glb"], false],
    ]);
    // A sprite with no 3D in its export still lists the 3D slot only when the intent wants a mesh.
    expect(exportCards(exported, { output: "sprite" }, { made: 1 }).map((c) => c.title)).toEqual(["2D sheet"]);
    expect(exportCards(exported, null, { made: 1 }).map((c) => c.title)).toContain("3D");
  });

  it("2D sheet lines, motions, the first part as its picture, and empty", () => {
    const doc: ExportDoc = {
      made: 1,
      "2d": {
        parts: [{ image: "2d/parts/body.png", sound: "s" }, { image: "2d/parts/head.png", sound: "s", voice: "v" }],
        motions: { walk: { frames: 8, fps: 12 } },
      },
    };
    const flat = exportCards(exported, { output: "sprite" }, doc)[0] as PlainCard;
    expect(flat).toMatchObject({
      title: "2D sheet",
      lines: ["body.png + sound", "head.png + sound + voice", "walk: 8 frames at 12 fps"],
      img: "2d/parts/body.png",
    });
    expect((exportCards(exported, { output: "sprite" }, { made: 1, "2d": {} })[0] as PlainCard).lines).toEqual(["empty"]);
  });

  it("sprite stack lines", () => {
    const pic = exportCards(exported, { output: "stack" }, { made: 1, stack: { layers: 32, resolution: 64, maker: "picture", preview: "stack/preview.png" } });
    expect(pic[0]).toMatchObject({ lines: ["32 layers of 64×64", "built from the picture"], img: "stack/preview.png" });
    const mesh = exportCards(exported, { output: "stack" }, { made: 1, stack: { layers: 16, resolution: 32 } });
    expect((mesh[0] as PlainCard).lines[1]).toBe("sliced from the mesh");
  });

  it("before an export: Ready / Nothing yet placeholders for wanted stages, PS1 model for ps1", () => {
    const f = flowOf({ concept: true, mesh: true });
    expect(exportCards(f, ps1, null).map((c) => [c.title, (c as PlainCard).lines[0], (c as PlainCard).missing])).toEqual([["PS1 model", "Ready", false]]);
    // A stage the output does not need still shows once it holds files.
    expect(exportCards(flowOf({ stack: true }), { output: "model" }, null).map((c) => [c.title, (c as PlainCard).lines[0]])).toEqual([
      ["Sprite stack", "Ready"],
      ["3D", "Nothing yet"],
    ]);
    expect(exportCards(f, null, null).map((c) => c.title)).toEqual(["2D sheet", "Sprite stack", "3D"]);
  });

  it("no rig card without an FBX; psx-farmer_hat is PS1 then the rest", () => {
    const hat: ExportDoc = { made: 1, concept: "concept.png", "3d": { model: { file: "3d/model.glb" }, psx: { file: "3d/psx.glb", obj: "3d/psx/farmer_hat.obj" } } };
    expect(exportCards(exported, ps1, hat).map((c) => c.key)).toEqual(["psx", "concept", "mesh"]);
    const noFbx: ExportDoc = { made: 1, "3d": { rigged: { file: "3d/rigged.glb" } } };
    expect(exportCards(exported, ps1, noFbx).map((c) => c.key)).toEqual(["mesh"]);
  });
});

describe("hero cards", () => {
  it("PS1 facts: name from the obj, rounded size, default axes and atlas", () => {
    expect(psxCard(chicken["3d"]!.psx!)).toMatchObject({
      name: "chicken",
      texture: "3d/psx/chicken.png",
      atlas: "256",
      triangles: (1200).toLocaleString(),
      height: "48",
      size: "38 × 20 × 48",
      axes: "z-up, +x front",
    });
    expect(psxCard({ file: "3d/psx.glb" })).toMatchObject({ name: "model", atlas: "256", size: "", axes: "z-up, +x front", height: "" });
  });
  it("Rigged facts: body plan name, basenames, clip names with spaces", () => {
    expect(rigCard(chicken["3d"]!.rigged!)).toEqual({
      kind: "rig",
      key: "rig",
      title: "Rigged model · Bird",
      fbx: "chicken.fbx",
      texture: "chicken.png",
      bones: 55,
      clipCount: 7,
      clips: "Idle, Walk, Run, Peck, Flap, Sleep, Death",
    });
    expect(rigCard({ file: "f", class: "quadruped", clips: ["Look_Around"], fbx: "3d/rig/cow.fbx" })).toMatchObject({
      title: "Rigged model · Four legs",
      texture: "",
      clips: "Look Around",
    });
    expect(rigCard({ file: "f", class: "dragon", fbx: "a.fbx" }).title).toBe("Rigged model · dragon");
    expect(rigCard({ file: "f", fbx: "a.fbx" }).title).toBe("Rigged model · ");
    expect(rigCard({ file: "f", class: "", fbx: "a.fbx" }).title).toBe("Rigged model · Body plan: auto");
  });
  it("FBX line: checked (+ warnings), failed, the rig.json note, or nothing", () => {
    const ok = { ok: true, actions: 7, bones: 55, max_influences: 4, scene_fps: 30, warnings: [] };
    expect(fbxLine(ok, "ignored")).toEqual({ text: "FBX checked: 7 clips, 55 bones, 4 weights max, 30 fps", title: "", fail: false });
    expect(fbxLine({ ...ok, actions: ["a", "b"], bones: ["x"], warnings: ["w1"] }, null)).toEqual({
      text: "FBX checked: 2 clips, 1 bones, 4 weights max, 30 fps · 1 warning",
      title: "w1",
      fail: false,
    });
    expect(fbxLine({ ...ok, warnings: ["a", "b"] }, null)?.text).toMatch(/ · 2 warnings$/);
    expect(fbxLine({ ok: false, problems: ["no armature", "bad fps"] }, null)).toEqual({
      text: "FBX check failed: no armature; bad fps",
      title: "",
      fail: true,
    });
    expect(fbxLine({ ok: false }, null)?.text).toBe("FBX check failed: no report");
    expect(fbxLine(null, "FBX with 8 clips")).toEqual({ text: "FBX with 8 clips", title: "", fail: false });
    expect(fbxLine(null, undefined)).toBeNull();
  });
});

describe("action row", () => {
  it("Exporting… hides Show in Finder", () => {
    expect(exportAction(exported, true)).toEqual({ kind: "exporting", finder: false });
  });
  it("Export, Export again, or the reason", () => {
    expect(exportAction(flowOf({ mesh: true }), false)).toEqual({ kind: "button", label: "Export", finder: false });
    expect(exportAction(exported, false)).toEqual({ kind: "button", label: "Export again", finder: true });
    expect(exportAction(flowOf({ concept: true }), false)).toEqual({ kind: "none", finder: false });
    // An export left over after its sources went away still opens in Finder.
    expect(exportAction(flowOf({ export: true }), false)).toEqual({ kind: "none", finder: true });
    expect(exportAction(null, false)).toEqual({ kind: "none", finder: false });
  });
  it("the Finder failure names the folder", () => {
    expect(finderFailed("psx-cow")).toBe("Could not open Finder. The export is in out/psx-cow/export/.");
  });
});
