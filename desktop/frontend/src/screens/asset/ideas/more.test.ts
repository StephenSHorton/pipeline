import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Job } from "@/lib/jobs";

const h = vi.hoisted(() => ({ finished: [] as ((j: Job) => void)[], openAsset: vi.fn(), toast: vi.fn() }));
vi.mock("@/app/open", () => ({ openAsset: h.openAsset }));
vi.mock("@/lib/ui", () => ({ toast: h.toast }));
vi.mock("@/lib/jobs", async (orig) => ({
  ...(await orig<typeof import("@/lib/jobs")>()),
  onJobFinished: (fn: (j: Job) => void) => {
    h.finished.push(fn);
    return () => {};
  },
}));

const { useJobs } = await import("@/lib/jobs");
await import("./more");
const { draftFor } = await import("./ideas");

const STEM = "a-stenciled-wooden-supply-crate";
const job = (p: Partial<Job> = {}): Job => ({ id: "j1", stem: STEM, kind: "ideas", state: "running", progress: null, log: "", ...p });
const at = (hash: string) => vi.stubGlobal("location", { search: "", hash });
const finish = (j: Job) => h.finished.forEach((fn) => fn(j));

beforeEach(() => {
  h.openAsset.mockClear();
  h.toast.mockClear();
  at("");
  useJobs.setState({ job: null, watching: false, busy: false });
});

describe("a watched Home ideas job", () => {
  it("hands over to the grid at its first picture, with its prompt, once", () => {
    useJobs.setState({ job: job({ id: "f1", prompt: "a crate" }), watching: true });
    expect(h.openAsset).not.toHaveBeenCalled();
    useJobs.setState({ job: job({ id: "f1", prompt: "a crate", result: { items: [{ file: "a.png" }] } }) });
    expect(h.openAsset).toHaveBeenCalledWith(STEM, "ideas");
    expect(useJobs.getState().watching).toBe(false);
    expect(draftFor(STEM)).toBe("a crate");
    useJobs.setState({ job: job({ id: "f1", result: { items: [{ file: "a.png" }, { file: "b.png" }] } }), watching: true });
    expect(h.openAsset).toHaveBeenCalledTimes(1);
  });
});

describe("an ideas job finishing", () => {
  it("watching: opens it on Ideas with no toast", () => {
    const j = job({ id: "o1", state: "ok", prompt: "a lantern" });
    useJobs.setState({ job: j, watching: true });
    finish(j);
    expect(h.openAsset).toHaveBeenCalledWith(STEM, "ideas");
    expect(h.toast).not.toHaveBeenCalled();
    expect(useJobs.getState().job).toBeNull();
  });
  it("on that asset: drops the job quietly, also a reattached one", () => {
    at(`#/asset/${STEM}`);
    const j = job({ id: "o2", state: "ok", spec: undefined });
    useJobs.setState({ job: j });
    finish(j);
    expect(h.toast).not.toHaveBeenCalled();
    expect(h.openAsset).not.toHaveBeenCalled();
    expect(useJobs.getState().job).toBeNull();
  });
  it("elsewhere: toasts Ideas ready with Open", () => {
    at("#/asset/other");
    const j = job({ id: "o3", state: "ok" });
    useJobs.setState({ job: j });
    finish(j);
    expect(h.toast).toHaveBeenCalledWith(expect.stringMatching(/^Ideas ready: /), "Open", expect.any(Function));
    expect(h.toast.mock.calls[0][0]).toMatch(/^Ideas ready: /);
    h.toast.mock.calls[0][2]();
    expect(h.openAsset).toHaveBeenCalledWith(STEM, "ideas");
  });
  it("leaves other kinds alone", () => {
    const j = job({ kind: "flat2d", state: "ok" });
    useJobs.setState({ job: j, watching: true });
    finish(j);
    expect(h.openAsset).not.toHaveBeenCalled();
  });
});
