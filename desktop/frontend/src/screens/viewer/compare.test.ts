// Compare's rules: which version goes on the left, the side labels, the left file, and bones matched by name.
import { describe, expect, it } from "vitest";
import type { FlowEntry, FlowTake } from "@/lib/flow";
import { boneMap, compareTakes, comparing, currentTake, defaultLeftTake, leftTake, missingBone, motionSource, paneClears, paneLayout, roomShift, shiftLeft, sideLabels, sourceLabel, takeModelFiles, useCompare, versionNo } from "./compare";

const take = (id: string, files: string[] = []): FlowTake => ({ id, made: 0, files });
const rig = (e: Partial<FlowEntry> = {}): FlowEntry => ({
  stage: "rig",
  pick: "t3",
  takes: [take("t1"), take("t2"), take("t3")],
  live: true,
  files: [],
  unsaved: false,
  stale: [],
  upstream: [],
  ...e,
});

describe("the left version", () => {
  it("defaults to the version before the current one", () => {
    expect(defaultLeftTake(rig())).toBe("t2");
    expect(defaultLeftTake(rig({ pick: "t2" }))).toBe("t1");
  });

  it("takes the next one when the current is the first, the newest when the screen is unsaved", () => {
    expect(defaultLeftTake(rig({ pick: "t1" }))).toBe("t2");
    expect(defaultLeftTake(rig({ unsaved: true }))).toBe("t3");
    expect(defaultLeftTake(rig({ pick: null }))).toBe("t3");
  });

  it("has nothing to compare with one kept version on screen", () => {
    const one = rig({ pick: "t1", takes: [take("t1")] });
    expect(defaultLeftTake(one)).toBeNull();
    expect(compareTakes(one)).toEqual([]);
    expect(compareTakes(rig({ pick: "t1", takes: [take("t1")], unsaved: true })).map((t) => t.id)).toEqual(["t1"]);
    expect(defaultLeftTake(null)).toBeNull();
  });

  it("shift-clicking a version puts it on the left; the current one just opens compare", () => {
    expect(shiftLeft(rig(), "t1")).toBe("t1");
    expect(shiftLeft(rig(), "t3")).toBe("t2");
    expect(shiftLeft(rig({ pick: "t1", takes: [take("t1")] }), "t1")).toBeNull();
  });

  it("keeps a chosen version, but never the current one or one that is gone", () => {
    expect(leftTake(rig(), "t1")).toBe("t1");
    expect(leftTake(rig(), "t3")).toBe("t2");
    expect(leftTake(rig(), "t9")).toBe("t2");
    expect(leftTake(rig(), null)).toBe("t2");
  });

  it("knows the current take only when the screen is kept", () => {
    expect(currentTake(rig())).toBe("t3");
    expect(currentTake(rig({ unsaved: true }))).toBeNull();
    expect(currentTake(rig({ pick: "t7" }))).toBeNull();
  });
});

describe("labels", () => {
  it("numbers the sides as the versions panel does", () => {
    expect(sideLabels(rig(), "t2")).toEqual({ left: "Version 2", right: "Version 3 · current" });
    expect(sideLabels(rig({ unsaved: true }), "t3")).toEqual({ left: "Version 3", right: "Current · not kept" });
    expect(versionNo(rig(), "t1")).toBe(1);
    expect(versionNo(rig(), null)).toBe(0);
  });
});

describe("the store", () => {
  it("is on for one asset's stage, and starts on the default side", () => {
    useCompare.getState().start("sheep", "rig");
    useCompare.getState().setLeft("t1");
    expect(comparing(useCompare.getState(), "sheep", "rig")).toBe(true);
    expect(comparing(useCompare.getState(), "sheep", "mesh")).toBe(false);
    expect(comparing(useCompare.getState(), "goat", "rig")).toBe(false);
    useCompare.getState().start("sheep", "rig");
    expect(useCompare.getState().left).toBeNull();
    useCompare.getState().stop();
    expect(comparing(useCompare.getState(), "sheep", "rig")).toBe(false);
    expect(comparing(useCompare.getState(), "", "rig")).toBe(false);
  });

  it("shows the motion source on the left until a version is chosen", () => {
    useCompare.getState().startSource("sheep");
    expect(comparing(useCompare.getState(), "sheep", "rig")).toBe(true);
    expect(useCompare.getState().source).toBe(true);
    useCompare.getState().setLeft("t3");
    expect(useCompare.getState().source).toBe(false);
    useCompare.getState().startSource("sheep");
    useCompare.getState().start("sheep", "rig");
    expect(useCompare.getState().source).toBe(false);
    useCompare.getState().startSource("sheep");
    useCompare.getState().stop();
    expect(useCompare.getState().source).toBe(false);
  });
});

describe("the motion source", () => {
  it("offers rig.json's motion source only as a glb under the asset", () => {
    const horse = { file: "rig/source/motion.glb", name: "Horse", credit: "Mesh2Motion (CC0)" };
    expect(motionSource({ motion_source: horse })).toEqual(horse);
    expect(motionSource({ motion_source: null })).toBeNull();
    expect(motionSource(null)).toBeNull();
    expect(motionSource({ motion_source: { file: "../../etc/x.glb" } })).toBeNull();
    expect(motionSource({ motion_source: { file: "https://x/y.glb" } })).toBeNull();
    expect(motionSource({ motion_source: { file: "rig/source/motion.png" } })).toBeNull();
  });

  it("is labelled by its animal, with the credit under it", () => {
    expect(sourceLabel({ file: "a.glb", name: "Horse", credit: "Mesh2Motion (CC0)" })).toEqual({ label: "Motion source · Horse", note: "Clips from Mesh2Motion (CC0)" });
    expect(sourceLabel({ file: "a.glb" })).toEqual({ label: "Motion source", note: "" });
  });
});

describe("the left file", () => {
  const rigFiles = ["rig/rig.json", "rig/rigged_allclips.glb", "rig/skinned_raw.glb", "rigged.glb"];
  it("shows the file the right shows when the take has it", () => {
    expect(takeModelFiles("rig", rigFiles, "rig/rigged_allclips.glb")).toEqual(["rig/rigged_allclips.glb", "rigged.glb"]);
    expect(takeModelFiles("rig", rigFiles, "rigged.glb")).toEqual(["rigged.glb", "rig/rigged_allclips.glb"]);
    expect(takeModelFiles("mesh", ["psx.glb", "textured_photo.glb", "mesh.glb"], "textured_photo.glb")).toEqual(["textured_photo.glb", "psx.glb", "mesh.glb"]);
  });

  it("falls back to the stage's order, then any model outside the workbench", () => {
    expect(takeModelFiles("rig", ["rigged.glb"], "rig/rigged_allclips.glb")).toEqual(["rigged.glb"]);
    expect(takeModelFiles("mesh", ["workbench/edit_1.glb", "old_name.glb"], "psx.glb")).toEqual(["old_name.glb"]);
    expect(takeModelFiles("mesh", ["preview.png"], "psx.glb")).toEqual([]);
  });
});

describe("bones by name", () => {
  it("maps each left bone to the right's index, extras past the end", () => {
    expect([...boneMap(["root", "neck", "tail", "head"], ["root", "head", "neck"])]).toEqual([0, 2, 3, 1]);
    expect([...boneMap(["a", "x", "y"], ["a"])]).toEqual([0, 1, 2]);
  });

  it("says when the left lacks the picked bone", () => {
    expect(missingBone(["root", "neck"], ["root", "head"], 1)).toBe(true);
    expect(missingBone(["root", "head"], ["head", "root"], 0)).toBe(false);
    expect(missingBone(["root"], ["root"], -1)).toBe(false);
  });
});

describe("where the panes go", () => {
  const wide = (width: number, right = 342) => ({ width, height: 830, narrow: false, left: { width: 340, height: 430 }, right, top: 40, bottom: 50 });

  it("sits side by side between the HUD columns and under the view card", () => {
    expect(paneLayout(wide(1440))).toEqual({ inset: { top: 60, right: 362, bottom: 12, left: 360 }, stacked: false });
    expect(paneLayout(wide(1280)).stacked).toBe(false);
    // The 3D stage has only the camera stack on the right.
    expect(paneLayout(wide(1280, 47)).inset.right).toBe(67);
  });

  it("stacks when the halves would be too thin, and takes the dock's room back when the strip is", () => {
    expect(paneLayout(wide(1024))).toEqual({ inset: { top: 60, right: 362, bottom: 12, left: 360 }, stacked: true });
    expect(paneLayout(wide(800)).inset).toEqual({ top: 60, right: 12, bottom: 12, left: 360 });
    expect(paneLayout(wide(500)).inset).toEqual({ top: 60, right: 12, bottom: 12, left: 12 });
  });

  it("stacks between the left column and the bottom stack on phones, keeping room for both panes", () => {
    const phone = (left: number) => ({ width: 390, height: 728, narrow: true, left: { width: 366, height: left }, right: 0, top: 0, bottom: 180 });
    expect(paneLayout(phone(260))).toEqual({ inset: { top: 280, right: 12, bottom: 200, left: 12 }, stacked: true });
    expect(paneLayout(phone(500)).inset.top).toBe(288);
  });
});

describe("pane clears", () => {
  const layout = (stacked: boolean) => ({ inset: { top: 60, right: 362, bottom: 12, left: 360 }, stacked });

  it("gives each half of a side-by-side split its share of the HUD's room", () => {
    const c = paneClears(1440, 900, layout(false));
    expect(c.first).toEqual({ top: 60, right: 0, bottom: 12, left: 360 });
    expect(c.second).toEqual({ top: 60, right: 362, bottom: 12, left: 0 });
    expect(c.frame).toEqual({ width: 358, height: 828 });
    expect(roomShift(c.first)).toEqual({ x: 180, y: 24 });
    expect(roomShift(c.second)).toEqual({ x: -181, y: 24 });
  });

  it("splits top and bottom when stacked", () => {
    const c = paneClears(1024, 800, layout(true));
    expect(c.first).toEqual({ top: 60, right: 362, bottom: 0, left: 360 });
    expect(c.second).toEqual({ top: 0, right: 362, bottom: 12, left: 360 });
    expect(c.frame).toEqual({ width: 302, height: 340 });
  });

  it("keeps some of a pane clear when the HUD is wider than it", () => {
    const c = paneClears(600, 800, layout(false));
    expect(c.first.left).toBe(180);
    expect(c.second.right).toBe(180);
  });
});

