import { beforeEach, describe, expect, it } from "vitest";
import { followStep } from "@/app/follow";
import { absorb, currentEntry, durationText, entryTime, imageUrl, mergeEntries, onActivity, splitOut, useActivity, type ActivityEntry } from "./activity";

const e = (id: string, seq: number, more: Partial<ActivityEntry> = {}): ActivityEntry => ({ id, seq, ts: seq, started: seq, ...more });

describe("mergeEntries", () => {
  it("keeps the newest copy of each id, newest action first", () => {
    const list = mergeEntries([e("a", 1, { state: "running" })], [e("b", 2), e("a", 3, { state: "ok", started: 1 })]);
    expect(list.map((x) => [x.id, x.state])).toEqual([["b", undefined], ["a", "ok"]]);
    expect(mergeEntries(list, [e("a", 2, { state: "running", started: 1 })]).find((x) => x.id === "a")?.state).toBe("ok");
  });
});

describe("currentEntry", () => {
  it("is Claude's running action before a job's, else null", () => {
    expect(currentEntry([e("j", 2, { state: "running", source: "job" }), e("c", 1, { state: "running", source: "claude" })])?.id).toBe("c");
    expect(currentEntry([e("j", 2, { state: "ok" })])).toBeNull();
  });
});

describe("times and paths", () => {
  it("reads like the job card", () => {
    expect([durationText(420), durationText(12_400), durationText(185_000), durationText(undefined)]).toEqual(["0.4 s", "12 s", "3 min 05 s", ""]);
    expect(entryTime(e("a", 1, { state: "running", started: 100 }), 103.2)).toBe("3.2 s");
    expect(entryTime(e("a", 1, { state: "ok", ms: 900 }), 999)).toBe("0.9 s");
  });
  it("splits out/-relative files and busts picture caches by seq", () => {
    expect(splitOut("mlg-sheep/workbench/clay.glb")).toEqual({ stem: "mlg-sheep", file: "workbench/clay.glb" });
    expect(splitOut("nofile")).toBeNull();
    expect(imageUrl(e("a", 7, { image: "mlg-sheep/flat/drawing.jpg" }))).toBe("/out/mlg-sheep/flat/drawing.jpg?v=7");
    expect(imageUrl(e("a", 7))).toBeNull();
  });
});

describe("absorb", () => {
  beforeEach(() => useActivity.setState({ entries: [], seq: 0, primed: false }));
  it("shows the first answer's history without telling listeners, then tells them each change", () => {
    const heard: string[] = [];
    const off = onActivity((x, prev) => heard.push(`${x.id}:${x.state}:${prev?.state ?? "-"}`));
    absorb({ ok: true, seq: 2, now: 10, entries: [e("old", 2, { state: "ok" })] });
    expect(heard).toEqual([]);
    absorb({ ok: true, seq: 4, now: 11, entries: [e("n", 3, { state: "running" }), e("n", 4, { state: "ok" })] });
    off();
    expect(heard).toEqual(["n:running:-", "n:ok:running"]);
    expect(useActivity.getState().seq).toBe(4);
  });
});

describe("followStep", () => {
  it("opens the asset when work starts and adopts a job", () => {
    expect(followStep(e("j", 1, { state: "running", source: "job", stem: "s" }), undefined)).toEqual({ open: "s", adopt: true });
    expect(followStep(e("j", 2, { state: "running", source: "job", stem: "s" }), e("j", 1, { state: "running" }))).toEqual({});
    // A quick action whose running copy the feed never sent.
    expect(followStep(e("m", 2, { state: "ok", source: "claude", stem: "s" }), undefined)).toEqual({ open: "s" });
    expect(followStep(e("m", 3, { state: "ok", source: "claude", stem: "s", glb: "s/psx.glb" }), undefined)).toEqual({
      open: "s",
      model: { stem: "s", file: "psx.glb" },
      refresh: true,
    });
  });
  it("shows each new model, and re-reads the versions after a workbench write", () => {
    const run = e("w", 1, { state: "running", source: "claude", stem: "s" });
    expect(followStep(e("w", 2, { state: "ok", source: "claude", stem: "s", glb: "s/workbench/clay.glb" }), run)).toEqual({
      model: { stem: "s", file: "workbench/clay.glb" },
      refresh: true,
    });
    const job = e("j", 1, { state: "running", source: "job", glb: "s/mesh.glb" });
    expect(followStep(e("j", 2, { state: "running", source: "job", glb: "s/mesh.glb" }), job)).toEqual({});
    expect(followStep(e("j", 3, { state: "running", source: "job", glb: "s/textured.glb" }), job)).toEqual({ model: { stem: "s", file: "textured.glb" } });
  });
});
