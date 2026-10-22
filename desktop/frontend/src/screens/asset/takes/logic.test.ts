// The strip's copy and rules, word for word from the legacy renderFlow (index.html 2854-3031) and the intent merge
// of buildFromConcept (3542-3577).
import { describe, expect, it } from "vitest";
import type { FlowEntry, FlowState, FlowTake, Stage } from "@/lib/flow";
import { buildIntent, buildSpec, buildTitle, groupItems, NEXT_HEADING, panelHeading, stripItems, takeTitle, versionsSummary, withKeyFacing, type StripItem } from "./logic";

const take = (id: string, files: string[] = [], extra: Partial<FlowTake> = {}): FlowTake => ({ id, made: 0, files, ...extra });
const entry = (stage: Stage, e: Partial<FlowEntry> = {}): FlowEntry => ({
  stage,
  pick: null,
  takes: [],
  live: false,
  files: [],
  unsaved: false,
  stale: [],
  upstream: [],
  ...e,
});
const flowOf = (...stages: FlowEntry[]): FlowState => ({ ok: true, stem: "s", stages });
const concept = entry("concept", { live: true, files: ["flat/source.png"], pick: "t1", takes: [take("t1", ["flat/source.png"])] });
const texts = (items: StripItem[]) => items.map((it) => (it.type === "take" ? `#${it.n}${it.pressed ? "*" : ""}` : it.text));

describe("versions", () => {
  it("hides one version with nothing unsaved, shows two", () => {
    const one = flowOf(entry("mesh", { live: true, pick: "t1", takes: [take("t1")] }));
    expect(stripItems({ stem: "s", stage: "mesh", flow: one }).some((it) => it.type === "take")).toBe(false);
    const two = flowOf(entry("mesh", { live: true, pick: "t2", takes: [take("t1", ["preview.png"]), take("t2")] }));
    const items = stripItems({ stem: "s", stage: "mesh", flow: two });
    expect(texts(items).slice(0, 3)).toEqual(["Versions", "#1", "#2*"]);
    expect(items[0]).toMatchObject({ type: "label", title: "Each build is kept. Click one to go back to it; nothing is deleted." });
    expect(items[1]).toMatchObject({ src: "/out/s/takes/mesh/t1/preview.png" });
    expect(items[2]).toMatchObject({ src: "" });
  });

  it("one version plus unsaved: shown, nothing pressed, and Keep this version", () => {
    const f = flowOf(entry("rig", { live: true, pick: "t1", unsaved: true, takes: [take("t1")] }));
    const items = stripItems({ stem: "s", stage: "rig", flow: f });
    expect(texts(items)).toEqual(["Versions", "#1", "Keep this version"]);
    expect(items[2]).toMatchObject({ title: "Save what is on screen as a version you can come back to.", action: { do: "keep" } });
  });

  it("titles: Version N · note · date; a rig version names its check", () => {
    const made = 1792519222;
    const when = new Date(made * 1000).toLocaleString();
    expect(takeTitle("mesh", take("t1", [], { made, note: "faster" }), 0)).toBe(`Version 1 · faster · ${when}`);
    expect(takeTitle("mesh", take("t1"), 2)).toBe("Version 3");
    expect(takeTitle("rig", take("t1", [], { summary: { level: "warn", line: "2 clips stretch" } }), 0)).toBe("Version 1 · 2 clips stretch");
    expect(takeTitle("rig", take("t1", ["rig/inspect.json"]), 0)).toBe("Version 1 · Checked before clips were measured");
    expect(takeTitle("rig", take("t1"), 0)).toBe("Version 1 · Made before rig checks");
  });

  it("stale note joins the stage names with and", () => {
    const f = flowOf(concept, entry("mesh", { live: true, stale: ["concept", "rig"] }));
    expect(texts(stripItems({ stem: "s", stage: "mesh", flow: f }))[0]).toBe("Made from an older concept and rig.");
  });

  it("Ideas and a missing stage show nothing", () => {
    expect(stripItems({ stem: "s", stage: "ideas", flow: flowOf(entry("ideas", { live: true })) })).toEqual([]);
    expect(stripItems({ stem: "s", stage: "mesh", flow: null })).toEqual([]);
  });
});

describe("concept: what to make from the picture", () => {
  it("no intent: Make from this picture: and every output in order, none primary", () => {
    const items = stripItems({ stem: "s", stage: "concept", flow: flowOf(concept) });
    expect(texts(items)).toEqual(["Make from this picture:", "Build PS1 model", "Build 3D model", "Build sprite stack", "2D sprite"]);
    expect(items.some((it) => it.type === "act" && it.primary)).toBe(false);
    expect(items[1]).toMatchObject({ title: "Make the PS1 model from this picture." });
    expect(items[4]).toMatchObject({ title: "Cut this picture into pieces, add motion and sound.", action: { do: "build", output: "sprite" } });
  });

  it("intent first and primary; Rebuild once its mesh is live", () => {
    const f = flowOf(concept, entry("mesh", { live: true }));
    const items = stripItems({ stem: "s", stage: "concept", flow: f, output: "stack" });
    expect(texts(items)).toEqual(["Next:", "Rebuild sprite stack", "Build PS1 model", "Build 3D model", "2D sprite"]);
    expect(items[1]).toMatchObject({ primary: true, action: { do: "build", output: "stack" } });
  });

  it("empty stages", () => {
    expect(texts(stripItems({ stem: "s", stage: "concept", flow: flowOf(entry("concept")) }))).toEqual(["No picture yet."]);
    expect(texts(stripItems({ stem: "s", stage: "flat", flow: flowOf(entry("flat")) }))).toEqual(["Use a step below to start the 2D sheet."]);
  });
});

describe("mesh and stack rows", () => {
  it("stack with a picture and no mesh: Stack from picture, then Build <intent>", () => {
    const f = flowOf(concept, entry("stack", { live: true }));
    const items = stripItems({ stem: "s", stage: "stack", flow: f });
    expect(texts(items)).toEqual(["Stack from picture", "Build sprite stack"]);
    expect(items[0]).toMatchObject({ title: "Build the sprite stack straight from the concept picture. No 3D needed." });
    expect(items[1]).toMatchObject({ title: "Make the sprite stack from the picked concept.", primary: true });
    expect(texts(stripItems({ stem: "s", stage: "stack", flow: f, stacking: true }))[0]).toBe("Stacking the picture…");
  });

  it("mesh live and fresh: Change settings & remake… only on Mesh", () => {
    const f = flowOf(concept, entry("mesh", { live: true }), entry("stack", { live: true }));
    expect(texts(stripItems({ stem: "s", stage: "mesh", flow: f, output: "ps1" }))).toEqual(["Change settings & remake…"]);
    expect(texts(stripItems({ stem: "s", stage: "stack", flow: f, output: "ps1" }))).toEqual(["Stack from picture"]);
  });

  it("mesh stale: Rebuild <name>", () => {
    const f = flowOf(concept, entry("mesh", { live: true, stale: ["concept"] }));
    expect(texts(stripItems({ stem: "s", stage: "mesh", flow: f }))).toEqual(["Made from an older concept.", "Rebuild 3D model", "Change settings & remake…"]);
  });

  it("no concept picture and no mesh", () => {
    expect(texts(stripItems({ stem: "s", stage: "mesh", flow: flowOf(entry("concept"), entry("mesh")) }))).toEqual(["No concept picture yet."]);
  });
});

describe("panel sections", () => {
  it("splits the heading, the thumbnails and the buttons, in order", () => {
    const f = flowOf(concept, entry("mesh", { live: true, pick: "t2", stale: ["concept"], takes: [take("t1"), take("t2")] }));
    const g = groupItems(stripItems({ stem: "s", stage: "mesh", flow: f }));
    expect(panelHeading(g)).toBe("Versions");
    expect(texts(g.takes)).toEqual(["#1", "#2*"]);
    expect(texts(g.rest)).toEqual(["Made from an older concept.", "Rebuild 3D model", "Change settings & remake…"]);
    expect(g.busy).toBeNull();
    expect(versionsSummary(g.takes)).toBe("2 of 2");
  });

  it("no versions: Next step, no summary", () => {
    const f = flowOf(concept, entry("stack", { live: true }));
    const g = groupItems(stripItems({ stem: "s", stage: "stack", flow: f, stacking: true }));
    expect(g.label).toBeNull();
    expect(g.takes).toEqual([]);
    expect(panelHeading(g)).toBe(NEXT_HEADING);
    expect(NEXT_HEADING).toBe("Next step");
    expect(versionsSummary(g.takes)).toBe("");
    expect(g.busy).toMatchObject({ text: "Stacking the picture…", busy: true });
  });

  it("unsaved: no version is live, so the summary counts what is kept", () => {
    const f = flowOf(concept, entry("mesh", { live: true, pick: "t1", unsaved: true, takes: [take("t1"), take("t2"), take("t3")] }));
    const g = groupItems(stripItems({ stem: "s", stage: "mesh", flow: f }));
    expect(g.takes.some((t) => t.pressed)).toBe(false);
    expect(versionsSummary(g.takes)).toBe("3 kept");
    expect(texts(g.rest)[0]).toBe("Keep this version");
  });
});

describe("build intent", () => {
  it("keeps the saved intent for its own output", () => {
    const saved = { output: "ps1" as const, psx: { facing: "left" as const, height: 30 } };
    expect(buildIntent(saved, "ps1")).toBe(saved);
  });

  it("a new output carries subject, name, rig, facing source; PS1 gets a height", () => {
    const saved = { output: "model" as const, poly: "high" as const, subject: "animal", name: "cow", facing_source: "drawn" as const, kind: "x" };
    expect(buildIntent(saved, "ps1")).toEqual({ output: "ps1", subject: "animal", name: "cow", facing_source: "drawn", kind: "x", psx: { height: 48 } });
    expect(buildIntent(null, "stack")).toEqual({ output: "stack" });
  });

  it("PS1 from flat/source.png faces as key.json says, unless the user set it", () => {
    const still = "out/s/flat/source.png";
    const i = { output: "ps1" as const, psx: { height: 48 } };
    expect(withKeyFacing(i, still, { facing: "right", canonical: true })).toEqual({ output: "ps1", psx: { height: 48, facing: "right" }, facing_source: "drawn" });
    expect(withKeyFacing(i, still, { facing: "left" })).toMatchObject({ facing_source: "detected" });
    expect(withKeyFacing(i, "out/s/input.png", { facing: "left" })).toBe(i);
    const user = { ...i, psx: { height: 48, facing: "away" as const }, facing_source: "user" as const };
    expect(withKeyFacing(user, still, { facing: "left" })).toBe(user);
  });

  it("the /generate body and the job title", () => {
    const still = "out/chicken-canon/flat/source.png";
    expect(buildSpec("chicken-canon", still, { output: "ps1", psx: { facing: "right", height: "48", name: "chicken" }, facing_source: "drawn" })).toEqual({
      image: still,
      images: [still],
      stem: "chicken-canon",
      quality: "psx",
      facing: "right",
      psx_height: 48,
      psx_name: "chicken",
      poly: "medium",
      intent: { output: "ps1", facing: "right", height: "48", name: "chicken", facing_source: "drawn" },
    });
    expect(buildSpec("s", still, { output: "model", poly: "low" })).toMatchObject({ quality: "textured", poly: "low", intent: { output: "model", poly: "low" } });
    expect(buildTitle("a-mech-spider", "stack")).toBe("Building the sprite stack · a mech spider");
  });
});
