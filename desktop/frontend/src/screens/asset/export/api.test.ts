import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useFlow } from "@/lib/flow";
import { useUi } from "@/lib/ui";
import { exportAsset, loadExport, readRigFacts, showInFinder, useExport } from "./api";

type Route = (url: string, init?: RequestInit) => Response | undefined;
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
let calls: { url: string; method: string; body?: unknown }[] = [];

function serve(route: Route) {
  calls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, method: init?.method ?? "GET", body: init?.body });
      return route(url, init) ?? new Response("404 page not found", { status: 404 });
    }),
  );
}

beforeEach(() => {
  useExport.setState({ stem: "", doc: null, exporting: false });
  useFlow.setState({ stem: "hen", flow: null, version: 0 });
  useUi.setState({ toast: null });
});
afterEach(() => vi.unstubAllGlobals());

describe("export calls", () => {
  it("GET /export/<s>: a 404 is no doc, not an error", async () => {
    serve(() => undefined);
    await loadExport("hen");
    expect(useExport.getState()).toMatchObject({ stem: "hen", doc: null });
    serve((u) => (u === "/export/hen" ? json({ ok: true, asset: { made: 5 } }) : undefined));
    await loadExport("hen");
    expect(useExport.getState().doc).toEqual({ made: 5 });
  });

  it("POST /export/<s> {} keeps the new doc and re-reads the flow", async () => {
    useExport.setState({ stem: "hen" });
    serve((u, init) => {
      if (u === "/export/hen" && init?.method === "POST") return json({ ok: true, asset: { made: 9 } });
      if (u === "/flow/hen") return json({ ok: true, stem: "hen", stages: [] });
    });
    await exportAsset("hen");
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual(["POST /export/hen", "GET /flow/hen"]);
    expect(calls[0].body).toBe("{}");
    expect(useExport.getState()).toMatchObject({ doc: { made: 9 }, exporting: false });
    expect(useFlow.getState().version).toBe(1);
  });

  it("a refused export is a toast with the factory's words", async () => {
    serve((u, init) =>
      init?.method === "POST" ? json({ error: "nothing to export yet: make a 2D sheet, a sprite stack or a 3D mesh first" }, 400) : u === "/flow/hen" ? json({ ok: true, stages: [] }) : undefined,
    );
    await exportAsset("hen");
    expect(useUi.getState().toast?.text).toBe("nothing to export yet: make a 2D sheet, a sprite stack or a 3D mesh first");
    expect(useExport.getState().exporting).toBe(false);
  });

  it("Show in Finder: POST /open-export without Wails; a failure names the folder", async () => {
    serve((u) => (u === "/open-export?stem=hen" ? json({ ok: true }) : undefined));
    await showInFinder("hen");
    expect(calls).toMatchObject([{ url: "/open-export?stem=hen", method: "POST" }]);
    expect(useUi.getState().toast).toBeNull();
    serve(() => json({ error: "hen is not exported yet" }, 400));
    await showInFinder("hen");
    expect(useUi.getState().toast?.text).toBe("Could not open Finder. The export is in out/hen/export/.");
  });

  it("Show in Finder: Wails OpenExport when there; a Wails refusal is a toast, never a POST", async () => {
    const byName = vi.fn(async () => {});
    vi.stubGlobal("window", { wails: { Call: { ByName: byName } } });
    serve(() => json({ ok: true }));
    await showInFinder("hen");
    expect(byName).toHaveBeenCalledWith(expect.stringMatching(/OpenExport$/), "hen");
    expect(calls).toEqual([]);
    expect(useUi.getState().toast).toBeNull();
    byName.mockRejectedValueOnce(new Error("no folder"));
    await showInFinder("hen");
    expect(calls).toEqual([]);
    expect(useUi.getState().toast?.text).toBe("Could not open Finder. The export is in out/hen/export/.");
  });

  it("rig facts: rig.json's fbx_check first, else fbx_check.json; HEAD on the contact sheet", async () => {
    serve((u, init) => {
      if (u.startsWith("/out/hen/rig/rig.json")) return json({ fbx_check: { ok: true, actions: 7 }, fbx_note: "n" });
      if (u.startsWith("/out/hen/rig/preview/contact_sheet.png") && init?.method === "HEAD") return new Response(null, { status: 200 });
    });
    expect(await readRigFacts("hen")).toEqual({ fbxCheck: { ok: true, actions: 7 }, fbxNote: "n", sheet: true });
    expect(calls.some((c) => c.url.includes("fbx_check.json"))).toBe(false);

    serve((u) => {
      if (u.startsWith("/out/hen/rig/rig.json")) return json({ fbx_note: "FBX with 8 clips" });
      if (u.startsWith("/out/hen/rig/fbx_check.json")) return json({ ok: false, problems: ["x"] });
    });
    expect(await readRigFacts("hen")).toEqual({ fbxCheck: { ok: false, problems: ["x"] }, fbxNote: "FBX with 8 clips", sheet: false });
  });

  it("with the flow's file list, asks only for the rig files that are there", async () => {
    serve((u) => (u.startsWith("/out/cow/rig/rig.json") ? json({ fbx_note: "FBX with 8 clips" }) : undefined));
    expect(await readRigFacts("cow", new Set(["rig/rig.json", "rig/cow.fbx"]))).toEqual({ fbxCheck: null, fbxNote: "FBX with 8 clips", sheet: false });
    expect(calls.map((c) => c.url.split("?")[0])).toEqual(["/out/cow/rig/rig.json"]);
  });
});
