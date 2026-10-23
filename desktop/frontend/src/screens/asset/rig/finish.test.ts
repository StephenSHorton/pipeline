// The rig finish seam: it owns exactly the jobs Home's chain.ts skips ("the viewer owns rig jobs"), and says
// "Rig ready: …" with legacy finishJob's facts and actions (index.html 6560-6606). No call reaches a factory: the
// reads, the navigation and the toast are fakes.
import { afterEach, describe, expect, it } from "vitest";
import { useJobs, type Job } from "@/lib/jobs";
import { finishRig, isRigJob, type FinishDeps } from "./finish";

const job = (over: Partial<Job> = {}): Job => ({ id: "j1", stem: "big-cow", kind: "rig", state: "ok", progress: null, log: "", ...over });

function fakes(files: Record<string, unknown>, onAsset = "big-cow") {
  const calls: string[] = [];
  let act: (() => void) | undefined;
  const d: FinishDeps = {
    read: async <T,>(_stem: string, rel: string) => (files[rel] ?? null) as T | null,
    onAsset: () => onAsset,
    open: (stem, stage) => calls.push(`open ${stem} ${stage}`),
    enter: (stage) => calls.push(`enter ${stage}`),
    inspect: (stem, key) => calls.push(`inspect ${stem} ${key}`),
    toast: (text, label, fn) => {
      calls.push(`toast ${text} [${label ?? ""}]`);
      act = fn;
    },
  };
  return { d, calls, click: () => act?.() };
}

afterEach(() => useJobs.setState({ job: null, watching: false }));

describe("isRigJob", () => {
  it("rig and render jobs, nothing else", () => {
    expect(isRigJob({ kind: "rig" })).toBe(true);
    expect(isRigJob({ kind: "render" })).toBe(true);
    for (const kind of ["", "flat2d", "ideas"] as const) expect(isRigJob({ kind })).toBe(false);
  });
});

describe("finishRig", () => {
  it("on the asset: enter Rig, then Export → for a rig that passes", async () => {
    const j = job({ degraded: [{ stage: "rig", code: "clips_dropped", message: "Fly" }] });
    useJobs.setState({ job: j });
    const f = fakes({ "rig/rig.json": { class: "quadruped", clips: ["Walk", "Idle"], fbx: "x.fbx" }, "rig/summary.json": { level: "warn", line: "Head 9° off" } });
    await finishRig(j, f.d);
    expect(f.calls).toEqual(["enter rig", "toast Rig ready: big cow · Four legs, 2 clips, FBX for s&box · Head 9° off · Clips left out: Fly [Export →]"]);
    expect(useJobs.getState().job).toBeNull();
    f.click();
    expect(f.calls.at(-1)).toBe("enter export");
  });

  it("a failing rig offers Inspect on the worst problem", async () => {
    const f = fakes({ "rig/summary.json": { level: "fail", line: "Tears at 4 points", problems: [{ level: "warn", key: "planner" }, { level: "fail", key: "seam" }] } });
    await finishRig(job(), f.d);
    expect(f.calls).toEqual(["enter rig", "toast Rig ready: big cow · Tears at 4 points [Inspect]"]);
    f.click();
    expect(f.calls.at(-1)).toBe("inspect big-cow seam");
  });

  it("elsewhere: Open, which drops the chip and opens on Rig", async () => {
    const j = job();
    useJobs.setState({ job: j });
    const f = fakes({}, "");
    await finishRig(j, f.d);
    expect(f.calls).toEqual(["toast Rig ready: big cow [Open]"]);
    expect(useJobs.getState().job).toBe(j);
    f.click();
    expect(f.calls.at(-1)).toBe("open big-cow rig");
    expect(useJobs.getState().job).toBeNull();
  });

  it("watching the job card on Home: open on Rig now", async () => {
    useJobs.setState({ watching: true });
    const f = fakes({}, "");
    await finishRig(job(), f.d);
    expect(f.calls).toEqual(["open big-cow rig", "toast Rig ready: big cow [Export →]"]);
  });

  it("a ps1 asset keeps its psx facts when rig.json can't be read; rig.json's replace them", async () => {
    const f = fakes({ "intent.json": { output: "ps1" }, "psx/psx.json": { triangles: 1500, atlas: 128 } });
    await finishRig(job(), f.d);
    expect(f.calls[1]).toBe("toast Rig ready: big cow · 1,500 triangles, 128 px texture [Export →]");
    const g = fakes({ "intent.json": { output: "ps1" }, "psx/psx.json": { triangles: 1500, atlas: 128 }, "rig/rig.json": { class: "avian", clips: [] } });
    await finishRig(job(), g.d);
    expect(g.calls[1]).toBe("toast Rig ready: big cow · Bird, 0 clips [Export →]");
  });

  it("a failed or cancelled rig stays on the chip; a render only clears its job", async () => {
    const f = fakes({});
    await finishRig(job({ state: "error" }), f.d);
    await finishRig(job({ state: "cancelled" }), f.d);
    expect(f.calls).toEqual([]);
    const r = job({ id: "r1", kind: "render", state: "error" });
    useJobs.setState({ job: r });
    await finishRig(r, f.d);
    expect(f.calls).toEqual([]);
    expect(useJobs.getState().job).toBeNull();
  });
});
