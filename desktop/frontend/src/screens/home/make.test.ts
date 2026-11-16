import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { Asset } from "@/lib/api";
import {
  assetKind,
  buildFromDrawingBody,
  drawFirstBody,
  drawTimeNote,
  flatName,
  guessSize,
  NO_PICTURE,
  photoDrawBody,
  recoverDrawIntent,
  matchesFilter,
  readIntent,
  relativeTime,
  slugStem,
  spriteDrawBody,
  type MakeChoice,
} from "./make";

const base: MakeChoice = { kind: "ps1", prompt: "", facing: "camera", height: 58, name: "", poly: "medium", steps: [] };

describe("request bodies", () => {
  it("a photo is a reference the four views are drawn from, never /generate on the photo itself", () => {
    const c: MakeChoice = { ...base, kind: "ps1", name: "speckled-hen" };
    expect(photoDrawBody(c, "speckled-hen", "in/hen.jpg")).toEqual({
      kind: "flat2d",
      stem: "speckled-hen",
      prompt: "",
      intent: { output: "ps1", height: 58, name: "speckled_hen" },
      steps: [],
      views: "4",
      reference: "in/hen.jpg",
    });
    // Words steer the drawing; 3D model and sprite stack take the same path with their poly.
    expect(photoDrawBody({ ...base, kind: "model", poly: "high", prompt: "rusty" }, "x", "in/x.png")).toMatchObject({
      prompt: "rusty", reference: "in/x.png", intent: { output: "model", poly: "high" },
    });
    expect(photoDrawBody({ ...base, kind: "stack" }, "crate", "in/x.png").intent).toEqual({ output: "stack", poly: "medium" });
  });

  it("a description with no picture is drawn first (flat2d, four views), never /generate", () => {
    const body = drawFirstBody({ ...base, prompt: "a speckled hen", name: "speckled-hen" }, "speckled-hen");
    expect(body).toEqual({
      kind: "flat2d",
      stem: "speckled-hen",
      prompt: "a speckled hen",
      intent: { output: "ps1", height: 58, name: "speckled_hen" },
      steps: [],
      views: "4",
    });
    // A user pick of the facing is kept; an explicit size pick names the pose subject.
    const picked = drawFirstBody({ ...base, prompt: "x", facing: "left", facingSource: "user", subject: "animal" }, "x");
    expect(picked.intent).toEqual({ output: "ps1", facing: "left", height: 58, facing_source: "user", subject: "animal" });
  });

  it("2D sprite: steps in the body and intent; source drop after /2d/drop", () => {
    expect(spriteDrawBody({ ...base, kind: "sprite", prompt: "a cat", steps: ["pieces"] }, "a-cat", true)).toEqual({
      kind: "flat2d", stem: "a-cat", prompt: "a cat", intent: { output: "sprite", steps: ["pieces"] }, steps: ["pieces"], pieces: "", source: "drop",
    });
    // Each picked step's words go along (legacy pickedStepFields); unpicked steps' words do not.
    const said = spriteDrawBody({ ...base, kind: "sprite", prompt: "a cat", steps: ["say", "sound"], stepText: { say: " hello ", sound: "meow", pieces: "tail" } }, "a-cat", false);
    expect(said).toMatchObject({ steps: ["say", "sound"], say: "hello", sound: "meow" });
    expect(said).not.toHaveProperty("pieces");
    expect(said).not.toHaveProperty("source");
  });

  it("drawing → 3D: the drawn facing from key.json, the still as the image", () => {
    const want = readIntent({ ...base, name: "hen" }, { drawn: true });
    const body = buildFromDrawingBody("hen", want, { output: "ps1", psx: { height: "48", name: "hen" }, subject: "creature" }, { facing: "right", canonical: true });
    expect(body).toEqual({
      image: "out/hen/flat/source.png",
      images: ["out/hen/flat/source.png"],
      stem: "hen",
      quality: "psx",
      facing: "right",
      psx_height: 48,
      psx_name: "hen",
      poly: "medium",
      intent: { output: "ps1", height: 48, name: "hen", facing: "right", facing_source: "drawn", subject: "creature" },
    });
  });
});

describe("names and guesses", () => {
  it("slugStem / flatName match the legacy page", () => {
    expect(slugStem("The Brass Lantern.jpg")).toBe("brass-lantern");
    expect(flatName("A weathered handheld radio!")).toBe("a-weathered-handheld-radio");
    expect(flatName("", "IMG 001.png")).toBe("img-001");
  });
  it("height from the kind of thing", () => {
    expect(guessSize("a brass lantern")).toBe(3);
    expect(guessSize("a knight")).toBe(0);
    expect(guessSize("a speckled hen")).toBe(1);
    // People before props; with no words, the photo's file name.
    expect(guessSize("a man holding a lantern")).toBe(0);
    expect(guessSize("a marble bust")).toBe(2);
    expect(guessSize("", "lantern.jpg")).toBe(3);
    expect(guessSize("", "old_farmer.png")).toBe(0);
    expect(guessSize("a hen", "lantern.jpg")).toBe(1);
  });
  it("NO_PICTURE is the factory's own line (factory_env.py and generate.sh)", () => {
    const root = path.resolve(import.meta.dirname, "../../../../..");
    const env = readFileSync(path.join(root, "scripts/factory_env.py"), "utf8").match(/^NO_PICTURE = "([^"]+)"/m);
    const sh = readFileSync(path.join(root, "scripts/generate.sh"), "utf8").match(/^NO_PICTURE="([^"]+)"/m);
    expect(env?.[1]).toBe(NO_PICTURE);
    expect(sh?.[1]).toBe(NO_PICTURE);
  });
  it("a draw found after a reload builds from its saved intent; a missing PS1 height is guessed", () => {
    const saved = { output: "ps1" as const, psx: { name: "chicken", facing: "right" as const }, subject: "creature", name: "chicken", facing_source: "drawn" as const };
    expect(recoverDrawIntent(saved, "qmv-fix-chicken")).toEqual({ ...saved, psx: { name: "chicken", facing: "right", height: 58 } });
    expect(recoverDrawIntent({ output: "ps1", psx: { height: 24 } }, "x").psx?.height).toBe(24);
    expect(recoverDrawIntent({ output: "model", poly: "high" }, "x")).toEqual({ output: "model", poly: "high" });
    const body = buildFromDrawingBody("qmv-fix-chicken", recoverDrawIntent(saved, "qmv-fix-chicken"), saved, { facing: "right", canonical: true });
    expect(body).toMatchObject({ quality: "psx", psx_height: 58, facing: "right", image: "out/qmv-fix-chicken/flat/source.png" });
  });
});

describe("library", () => {
  const a = (p: Partial<Asset>): Asset => ({
    id: "x", title: "", has_mesh: false, has_stack: false, has_flat: false, has_ideas: false, has_psx: false, output: null, input: null, preview: null, mtime: 0, ...p,
  });
  it("assetKind", () => {
    expect(assetKind(a({ output: "stack" }))).toBe("stack");
    expect(assetKind(a({ has_psx: true }))).toBe("ps1");
    expect(assetKind(a({ has_flat: true }))).toBe("sprite");
    expect(assetKind(a({ has_mesh: true }))).toBe("model");
    expect(assetKind(a({ has_ideas: true }))).toBe("ideas");
  });
  it("filter and search", () => {
    const hen = a({ id: "hen", title: "speckled hen", has_psx: true });
    expect(matchesFilter(hen, "ps1", "speck")).toBe(true);
    expect(matchesFilter(hen, "model", "")).toBe(false);
    expect(matchesFilter(hen, "", "PS1")).toBe(true);
  });
  it("project filter, and the project name in the search", () => {
    const hen = a({ id: "hen", has_psx: true, project: "mlg-farming" });
    const pig = a({ id: "pig", has_psx: true, project: null });
    const cow = a({ id: "cow", has_psx: true }); // an older factory's row: no field is Unsorted
    expect([hen, pig, cow].filter((x) => matchesFilter(x, "", "")).length).toBe(3);
    expect([hen, pig, cow].filter((x) => matchesFilter(x, "", "", "mlg-farming")).map((x) => x.id)).toEqual(["hen"]);
    expect([hen, pig, cow].filter((x) => matchesFilter(x, "", "", "_unsorted")).map((x) => x.id)).toEqual(["pig", "cow"]);
    expect(matchesFilter(hen, "model", "", "mlg-farming")).toBe(false);
    expect(matchesFilter(hen, "", "farm", undefined, "MLG Farming")).toBe(true);
    expect(matchesFilter(hen, "", "farm")).toBe(false);
  });
  it("relativeTime", () => {
    const now = 1_000_000_000_000;
    expect(relativeTime(now / 1000 - 120, now)).toBe("2 min ago");
    expect(relativeTime(now / 1000 - 7200, now)).toBe("2 h ago");
    expect(relativeTime(now / 1000 - 3 * 86400, now)).toBe("3 days ago");
    expect(relativeTime(now / 1000 - 30 * 3600, now)).toBe("yesterday");
  });
  it("drawTimeNote", () => {
    expect(drawTimeNote({ loaded: true, times: { attempt: 33, ground: 11 } })).toBe(" About 45 s last time.");
    expect(drawTimeNote({ loaded: false, times: { attempt: 33, ground: 11, load: 13 } })).toBe(" About 55 s last time, with loading the model.");
    expect(drawTimeNote(null)).toBe("");
  });
});
