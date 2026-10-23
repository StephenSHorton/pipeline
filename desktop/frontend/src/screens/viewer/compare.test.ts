// Compare's rules: which version goes on the left, the side labels, the left file, and bones matched by name.
import { describe, expect, it } from "vitest";
import type { FlowEntry, FlowTake } from "@/lib/flow";
import { boneMap, compareTakes, comparing, currentTake, defaultLeftTake, leftTake, missingBone, paneLayout, sideLabels, takeModelFiles, useCompare, versionNo } from "./compare";

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
