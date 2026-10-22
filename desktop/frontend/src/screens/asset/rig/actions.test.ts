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
      if (url.startsWith("/flow/cow")) return { status: 200, body: { ok: true, stem: "cow", stages: [] } };
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

  it("an error is a toast and the reads still run, as legacy", async () => {
    serve((m, url) => (m === "POST" ? { status: 409, body: { error: "the factory is busy with another job" } } : url.startsWith("/flow") ? { status: 200, body: { ok: true, stem: "cow", stages: [] } } : null));
    await recheckRig("cow");
    expect(h.toast).toHaveBeenCalledWith("the factory is busy with another job");
    expect(calls.slice(1)).toEqual(["GET /flow/cow", "GET /out/cow/rig/inspect.json", "GET /out/cow/rig/rig.json"]);
    expect(useRig.getState().check).toMatchObject({ stem: "cow", doc: null });
  });
});

describe("renderSheet", () => {
  it("POST /rig/<stem>/render {job: default} as a render job, then looks for the sheet", async () => {
    h.startJob.mockImplementationOnce(async (spec) => {
      useJobs.setState({ job: { id: "r", stem: "cow", kind: "render", state: "ok", progress: null, log: "", spec } });
    });
    serve((m, url) => (m === "HEAD" && url.startsWith("/out/cow/rig/preview/contact_sheet.png") ? { status: 200 } : null));
    await renderSheet("cow");
    expect(h.startJob).toHaveBeenCalledWith({ job: "default", stem: "cow" }, { path: "/rig/cow/render", kind: "render", title: "Rendering the sheet for cow", watch: false });
    expect(calls).toEqual(["HEAD /out/cow/rig/preview/contact_sheet.png"]);
    expect(useRig.getState().sheet).toMatchObject({ stem: "cow", ok: true, busy: false, error: "" });
  });

  it("a failed render says so the legacy way", async () => {
    h.startJob.mockImplementationOnce(async (spec) => {
      useJobs.setState({ job: { id: "r", stem: "cow", kind: "render", state: "cancelled", progress: null, log: "", spec } });
    });
    serve(() => ({ status: 404 }));
    await renderSheet("cow");
    expect(useRig.getState().sheet).toMatchObject({ ok: false, busy: false, error: "The render cancelled" });
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
