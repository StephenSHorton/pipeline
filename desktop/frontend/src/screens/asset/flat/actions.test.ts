import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// No call here reaches a factory: startJob, askConfirm and fetch are mocks.
const startJob = vi.fn(async () => {});
const askConfirm = vi.fn(async () => true);
vi.mock("@/lib/jobs", async (orig) => ({ ...(await orig<typeof import("@/lib/jobs")>()), startJob }));
vi.mock("@/lib/ui", () => ({ askConfirm, toast: vi.fn() }));

const { useJobs } = await import("@/lib/jobs");
const { flatSpecs, jobOwns, panelAudio, redraw, reviseStep } = await import("./actions");
const { clipFields } = await import("./api");
const { emptyAudioAsset, mergeSheet } = await import("./flat");

const g = globalThis as unknown as { fetch: typeof fetch };
const realFetch = g.fetch;
let posted: { url: string; body: Record<string, unknown> }[] = [];
const answer = (data: unknown, status = 200) => {
  g.fetch = vi.fn(async (u: RequestInfo | URL, init?: RequestInit) => {
    posted.push({ url: String(u), body: JSON.parse(String(init?.body ?? "{}")) });
    return new Response(JSON.stringify(data), { status });
  }) as typeof fetch;
};
const running = (stem: string) =>
  useJobs.setState({ job: { id: "j1", stem, kind: "flat2d", state: "running", progress: null, log: "" } });

beforeEach(() => {
  posted = [];
  startJob.mockClear();
  askConfirm.mockClear();
  askConfirm.mockResolvedValue(true);
  useJobs.setState({ job: null });
});
afterEach(() => {
  g.fetch = realFetch;
});

describe("redraw (Picture mode)", () => {
  it("an empty prompt or a running job does not ask", async () => {
    expect(await redraw("hen", "")).toEqual({ status: "Describe the new picture first.", started: false });
    running("other");
    expect(await redraw("hen", "a hen")).toEqual({ status: "The factory is busy with another job.", started: false });
    expect(askConfirm).not.toHaveBeenCalled();
    expect(startJob).not.toHaveBeenCalled();
  });

  it("asks Redraw, then starts the redraw job on /2d/draw without the overlay", async () => {
    expect(await redraw("brown-hen", "a hen")).toEqual({ status: "", started: true });
    expect(askConfirm).toHaveBeenCalledWith("Draw a new picture from this prompt? It replaces the current one.", "Redraw");
    const [spec, opts] = startJob.mock.calls[0] as unknown as [Record<string, unknown>, Record<string, unknown>];
    expect(spec).toEqual({ kind: "flat2d", stem: "brown-hen", prompt: "a hen", revise: true });
    expect(opts).toEqual({ path: "/2d/draw", kind: "flat2d", title: "Redrawing brown hen", watch: false });
    expect(flatSpecs.has(spec)).toBe(true);
  });

  it("returns at once while the job runs", async () => {
    startJob.mockReturnValueOnce(new Promise(() => {}));
    expect(await redraw("hen", "a hen")).toEqual({ status: "", started: true });
  });

  it("No starts nothing", async () => {
    askConfirm.mockResolvedValue(false);
    expect(await redraw("hen", "a hen")).toEqual({ status: "", started: false });
    expect(startJob).not.toHaveBeenCalled();
  });
});

describe("reviseStep", () => {
  const sheet = { parts: mergeSheet({ parts: [{ id: "p1", file: "parts/p1.png" }] }).parts, index: 0, center: "source" };

  it("waits for a job on this asset", async () => {
    running("hen");
    expect(jobOwns("hen")).toBe(true);
    expect(await reviseStep("hen", "pieces", "x", sheet, null, [])).toEqual({ ok: false, status: "Wait for the job on this asset to finish." });
    expect(posted).toEqual([]);
  });

  it("posts one step, says the running text first, and applies the answer", async () => {
    answer({ ok: true, center: "sheets/walk.png", notes: ["Packed."] });
    const says: string[] = [];
    const r = await reviseStep("hen", "motion", "walk", sheet, emptyAudioAsset(), [], (t) => says.push(t));
    expect(says).toEqual(["Packing the motion."]);
    expect(posted[0]).toEqual({ url: "/2d/steps", body: { stem: "hen", when: "revise", steps: ["motion"], part: "p1", speaker: "af_heart", motion: "walk" } });
    expect(r).toMatchObject({ ok: true, center: "sheets/walk.png", status: "Packed." });
  });

  it("sends a video clip as {video}", async () => {
    answer({ ok: true });
    const clip = Object.assign(new Blob([new Uint8Array([1, 2, 3])], { type: "video/mp4" }), { name: "walk.mp4" });
    await reviseStep("hen", "motion", "walk", sheet, null, [clip]);
    expect(posted[0].body.video).toBe("AQID");
  });

  it("an error is the status line", async () => {
    answer({ ok: false, error: "no parts model" }, 500);
    expect(await reviseStep("hen", "pieces", "x", sheet, null, [])).toEqual({ ok: false, status: "no parts model" });
  });
});

describe("clipFields", () => {
  it("stills go as frames", async () => {
    const png = (b: number) => Object.assign(new Blob([new Uint8Array([b])], { type: "image/png" }), { name: `${b}.png` });
    expect(await clipFields([png(1), png(2)])).toEqual({ frames: ["AQ==", "Ag=="] });
  });
});

describe("panelAudio", () => {
  it("needs an asset; Sound waits for a job on it, Speak does not", async () => {
    expect(await panelAudio("", "say", "x", "af_heart")).toEqual({ rel: "", status: "Open an asset." });
    running("hen");
    expect(await panelAudio("hen", "sound", "bolt", "af_heart")).toEqual({ rel: "", status: "Wait for the job on this asset to finish." });
    answer({ ok: true, hear: "voices/asset.wav", notes: ["Spoke."] });
    const says: string[] = [];
    expect(await panelAudio("hen", "say", "Moo", "am_adam", (t) => says.push(t))).toEqual({ rel: "voices/asset.wav", status: "Spoke." });
    expect(says).toEqual(["Speaking."]);
    expect(posted[0].body).toEqual({ stem: "hen", when: "revise", steps: ["say"], part: "asset", speaker: "am_adam", say: "Moo" });
  });

  it("Sound says Writing the sound.", async () => {
    answer({ ok: true, hear: "sounds/asset.wav" });
    const says: string[] = [];
    expect(await panelAudio("hen", "sound", "bolt", "af_heart", (t) => says.push(t))).toEqual({ rel: "sounds/asset.wav", status: "" });
    expect(says).toEqual(["Writing the sound."]);
  });
});
