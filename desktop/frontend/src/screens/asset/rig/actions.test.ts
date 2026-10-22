// What the Rig stage's buttons send. No call here reaches a factory: startJob and toast are mocks, fetch a stub.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const h = vi.hoisted(() => ({ startJob: vi.fn(async (_spec: Record<string, unknown>, _opts: Record<string, unknown>) => {}), toast: vi.fn() }));
vi.mock("@/lib/jobs", async (orig) => ({ ...(await orig<typeof import("@/lib/jobs")>()), startJob: h.startJob }));
vi.mock("@/lib/ui", async (orig) => ({ ...(await orig<typeof import("@/lib/ui")>()), toast: h.toast }));

const { rigAsset, recheckRig, renderSheet, inspectRig } = await import("./actions");
const { useRig, setChoices, loadRigPlans, resetRigPlans } = await import("./api");
const { choicesFor } = await import("./rig");
const { SHEET0 } = await import("./inspect");
const { useStage } = await import("@/screens/asset/stage");
const { useJobs } = await import("@/lib/jobs");
const { usePanels } = await import("@/screens/viewer/panels/store");

// The select only offers (and so only sends) a plan GET /rig/plans listed, as the legacy bar did.
const PLANS = ["horse", "ground_bird"].map((id) => ({ id, label: id, blurb: "", class: "", aliases: [], built: false, fixed: false, clips: [] }));

let calls: string[] = [];
function serve(route: (method: string, url: string) => { status: number; body?: unknown } | null) {
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    const method = init?.method || "GET";
    calls.push(`${method} ${String(url).replace(/\?t=\d+/, "")}`);
    const r = route(method, String(url)) ?? { status: 404, body: { error: "nope" } };
    return new Response(r.body === undefined ? null : JSON.stringify(r.body), { status: r.status, headers: { "Content-Type": "application/json" } });
  });
}

beforeEach(() => {
  calls = [];
  h.startJob.mockClear();
  h.toast.mockClear();
  useRig.setState({ choices: choicesFor(""), plans: PLANS, check: null, checking: "", focus: null, sheet: { ...SHEET0 } });
  useStage.setState({ stem: "cow", stage: "rig" });
  usePanels.setState({ open: null });
});
afterEach(() => {
  vi.unstubAllGlobals();
  useJobs.setState({ job: null, watching: false });
});

describe("rigAsset", () => {
  it("POST /rig/<stem> as a rig job on the chip, with this asset's plan and Advanced", () => {
    setChoices("cow", { choice: "plan:horse", advanced: { template: "fixed", binder: "", pred: "" } });
    rigAsset("cow");
    expect(h.startJob).toHaveBeenCalledWith(
      { plan: "horse", template: "fixed", stem: "cow" },
      { path: "/rig/cow", kind: "rig", rig: true, title: "Rigging cow", watch: false },
    );
  });

  it("another asset starts from auto (legacy carried the choice over)", () => {
    setChoices("cow", { choice: "plan:horse" });
    rigAsset("pig");
    expect(h.startJob.mock.calls[0][0]).toEqual({ stem: "pig" });
  });

  it("the panel's Rig it as made sends the built plan the select shows", () => {
    useRig.setState({ check: { stem: "cow", doc: null, info: null, made: { stem: "cow", plan: "ground_bird", clips: {}, failed: [] } } });
    rigAsset("cow", { straighten: "off" });
    expect(h.startJob.mock.calls[0][0]).toEqual({ plan: "ground_bird", straighten: "off", stem: "cow" });
  });

  it("a head kept as made stays as made", () => {
    useRig.setState({ check: { stem: "cow", doc: { bones: [], rig: { rest: { measured: { turned: true, head_yaw_deg: 30 } } } }, info: null, made: null } });
    rigAsset("cow");
    expect(h.startJob.mock.calls[0][0]).toEqual({ straighten: "off", stem: "cow" });
  });
});

describe("loadRigPlans", () => {
  it("asks once; an error leaves [] for the RIG_NAMES fallback", async () => {
    resetRigPlans();
    const get = vi.fn(async () => {
      throw new Error("down");
    });
    await loadRigPlans(get);
    await loadRigPlans(get);
    expect(get).toHaveBeenCalledTimes(1);
    expect(useRig.getState().plans).toEqual([]);
  });
});

describe("recheckRig", () => {
  it("POST /rig/<stem>/inspect, then the flow and the checks; reopens on them when the panel was open", async () => {
    usePanels.setState({ open: "rig" });
    serve((m, url) => {
      if (m === "POST" && url === "/rig/cow/inspect") return { status: 200, body: { ok: true } };
      if (url.startsWith("/flow/cow")) return { status: 200, body: { ok: true, stem: "cow", stages: [{ stage: "rig", files: ["rig/inspect.json", "rig/rig.json"] }] } };
      if (url.startsWith("/out/cow/rig/inspect.json")) return { status: 200, body: { bones: [], checks: [] } };
      if (url.startsWith("/out/cow/rig/rig.json")) return { status: 200, body: { class: "avian" } };
      return null;
    });
    await recheckRig("cow");
    expect(calls).toEqual(["POST /rig/cow/inspect", "GET /flow/cow", "GET /out/cow/rig/inspect.json", "GET /out/cow/rig/rig.json"]);
    expect(useRig.getState().check).toMatchObject({ stem: "cow", doc: { bones: [], rig: { class: "avian", rest: null } }, made: null });
    expect(useRig.getState().focus).toEqual({ stem: "cow", key: "" });
    expect(useRig.getState().checking).toBe("");
  });

  it("an error is a toast and the reads still run, as legacy (only the files the flow lists)", async () => {
    const flow = { ok: true, stem: "cow", stages: [{ stage: "rig", files: ["rig/rig.json"] }] };
    serve((m, url) => (m === "POST" ? { status: 409, body: { error: "the factory is busy with another job" } } : url.startsWith("/flow") ? { status: 200, body: flow } : null));
    await recheckRig("cow");
    expect(h.toast).toHaveBeenCalledWith("the factory is busy with another job");
    expect(calls.slice(1)).toEqual(["GET /flow/cow", "GET /out/cow/rig/rig.json"]);
    expect(useRig.getState().check).toMatchObject({ stem: "cow", doc: null });
  });
});

describe("renderSheet", () => {
  // The real startJob and followJob, over the fetch stub: the job ends (and leaves the chip) before startJob resolves.
  const real = async () => (await vi.importActual<typeof import("@/lib/jobs")>("@/lib/jobs")).startJob;
  const factory = (state: string, error = "") => (m: string, url: string) => {
    if (m === "POST" && url === "/rig/cow/render") return { status: 202, body: { ok: true, accepted: true, job_id: "r1", state: "running", stem: "cow", kind: "render", status_url: "/status/r1" } };
    if (url === "/status/r1") return { status: 200, body: { state, stem: "cow", kind: "render", error } };
    if (m === "HEAD" && url.startsWith("/out/cow/rig/preview/contact_sheet.png")) return { status: state === "ok" ? 200 : 404 };
    return null;
  };

  it("POST /rig/<stem>/render {job: default} as a render job, then looks for the sheet", async () => {
    h.startJob.mockImplementationOnce(await real());
    serve(factory("ok"));
    await renderSheet("cow");
    expect(h.startJob).toHaveBeenCalledWith({ job: "default", stem: "cow" }, { path: "/rig/cow/render", kind: "render", title: "Rendering the sheet for cow", watch: false });
    expect(calls).toEqual(["POST /rig/cow/render", "GET /status/r1", "HEAD /out/cow/rig/preview/contact_sheet.png"]);
    expect(useRig.getState().sheet).toMatchObject({ stem: "cow", ok: true, busy: false, error: "" });
  });

  it("a failed render says so the legacy way, even once its job has left the chip", async () => {
    h.startJob.mockImplementationOnce(await real());
    serve(factory("cancelled"));
    const { onJobEnded } = await vi.importActual<typeof import("@/lib/jobs")>("@/lib/jobs");
    // finish.ts drops a render's job as it ends, before startJob resolves.
    const off = onJobEnded(() => useJobs.setState({ job: null }));
    await renderSheet("cow");
    off();
    expect(useRig.getState().sheet).toMatchObject({ ok: false, busy: false, error: "The render cancelled" });
  });

  it("the server's error when it will not start", async () => {
    h.startJob.mockImplementationOnce(await real());
    serve((m, url) => (m === "POST" && url === "/rig/cow/render" ? { status: 400, body: { ok: false, error: "this asset has no rig yet" } } : null));
    await renderSheet("cow");
    expect(useRig.getState().sheet).toMatchObject({ busy: false, error: "this asset has no rig yet" });
  });
});

describe("inspectRig", () => {
  it("waits in the store, enters Rig, and opens now when this asset's checks are in", () => {
    useStage.setState({ stem: "cow", stage: "mesh" });
    inspectRig("cow", "seam");
    expect(useRig.getState().focus).toEqual({ stem: "cow", key: "seam" });
    expect(useStage.getState().stage).toBe("rig");
    expect(usePanels.getState().open).toBeNull();
    useRig.setState({ check: { stem: "cow", doc: { bones: [] }, info: null, made: null } });
    inspectRig("cow", "seam");
    expect(usePanels.getState().open).toBe("rig");
  });
});
