// The Stack stage's calls: stack.json from GET /stack/<s> or out/, the cache-busted sheet, Paint's Save (POST
// /paint/stack with the PNG), and Save's job gate. No call here reaches a factory: fetch is a stub.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { onFlowDirty } from "@/lib/api";
import { useJobs } from "@/lib/jobs";
import { saveStackPaint, WAIT_FOR_JOB, jobOwns } from "./actions";
import { postStackSheet, readStackMeta, sheetUrl } from "./api";

type Route = (url: string, init?: RequestInit) => Response | undefined;
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
let calls: { url: string; method: string; body?: unknown; type?: string }[] = [];

function serve(route: Route) {
  calls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, method: init?.method ?? "GET", body: init?.body, type: (init?.headers as Record<string, string> | undefined)?.["Content-Type"] });
      return route(url, init) ?? new Response("404 page not found", { status: 404 });
    }),
  );
}

const stack = { layers: 87, resolution: 160, pitch: 0.01, extents: [1, 1, 1] };

beforeEach(() => useJobs.setState({ job: null }));
afterEach(() => vi.unstubAllGlobals());

describe("readStackMeta", () => {
  it("GET /stack/<s> first", async () => {
    serve((u) => (u === "/stack/hen" ? json({ ...stack, ok: true, stem: "hen" }) : undefined));
    expect((await readStackMeta("hen", 5))?.layers).toBe(87);
    expect(calls.map((c) => c.url)).toEqual(["/stack/hen"]);
  });

  it("falls back to out/<s>/stack/stack.json on a 404", async () => {
    serve((u) => (u === "/out/hen/stack/stack.json?t=5" ? json(stack) : json({ ok: false, error: "no sprite stack" }, 404)));
    expect((await readStackMeta("hen", 5))?.resolution).toBe(160);
    expect(calls.map((c) => c.url)).toEqual(["/stack/hen", "/out/hen/stack/stack.json?t=5"]);
  });

  it("is null when neither has a stack, or the doc cannot be drawn", async () => {
    serve(() => undefined);
    expect(await readStackMeta("hen", 1)).toBeNull();
    serve((u) => (u === "/stack/hen" ? json({ layers: 0, resolution: 160, pitch: 0.01, extents: [1, 1, 1] }) : undefined));
    expect(await readStackMeta("hen", 1)).toBeNull();
  });
});

describe("the sheet and Save", () => {
  it("sheet.png is cache-busted", () => {
    expect(sheetUrl("hen", 7)).toBe("/out/hen/stack/sheet.png?t=7");
  });

  it("Save POSTs the PNG to /paint/stack?stem= and the flow is re-read (factoryJson's /paint/ hook)", async () => {
    serve(() => json({ ok: true, file: "stack/sheet.png", layers: 87, resolution: 160, stem: "my hen" }));
    const dirty: string[] = [];
    const off = onFlowDirty((p) => dirty.push(p));
    const png = new Blob(["x"], { type: "image/png" });
    await postStackSheet("my hen", png);
    off();
    expect(calls).toEqual([{ url: "/paint/stack?stem=my%20hen", method: "POST", body: png, type: "image/png" }]);
    expect(dirty).toEqual(["/paint/stack?stem=my%20hen"]);
  });
});

describe("saveStackPaint", () => {
  const png = new Blob(["x"], { type: "image/png" });
  const fakes = (owns = false, fail = "") => {
    const log: string[] = [];
    return {
      log,
      deps: {
        owns: () => owns,
        post: async (stem: string) => {
          log.push(`post ${stem}`);
          if (fail) throw new Error(fail);
        },
        toast: (t: string) => log.push(`toast ${t}`),
      },
    };
  };

  it("posts the sheet", async () => {
    const f = fakes();
    expect(await saveStackPaint("hen", async () => png, f.deps)).toBe(true);
    expect(f.log).toEqual(["post hen"]);
  });

  it("waits for a job on this asset: /paint has no busy check on the server", async () => {
    const f = fakes(true);
    expect(await saveStackPaint("hen", async () => png, f.deps)).toBe(false);
    expect(f.log).toEqual(["toast Wait for the job on this asset to finish."]);
    expect(WAIT_FOR_JOB).toBe("Wait for the job on this asset to finish.");
  });

  it("an error is a toast with the factory's message", async () => {
    const f = fakes(false, "no such asset");
    expect(await saveStackPaint("hen", async () => png, f.deps)).toBe(false);
    expect(f.log).toEqual(["post hen", "toast no such asset"]);
  });

  it("nothing to save without a stack", async () => {
    const f = fakes();
    expect(await saveStackPaint("hen", async () => null, f.deps)).toBe(false);
    expect(f.log).toEqual([]);
  });

  it("jobOwns: a running job on this stem", () => {
    useJobs.setState({ job: { id: "1", stem: "hen", kind: "", state: "running", progress: null, log: "" } });
    expect(jobOwns("hen")).toBe(true);
    expect(jobOwns("cow")).toBe(false);
  });
});
