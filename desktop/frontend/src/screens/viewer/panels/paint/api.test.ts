import { afterEach, describe, expect, it, vi } from "vitest";
import { onFlowDirty } from "@/lib/api";
import { postPaint } from "./api";

// No call reaches a factory: fetch is a stub that records what was sent.
let calls: { url: string; method?: string; headers?: HeadersInit; body?: unknown }[] = [];
function serve(status: number, body: unknown) {
  calls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, method: init?.method, headers: init?.headers, body: init?.body });
      return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
    }),
  );
}
afterEach(() => vi.unstubAllGlobals());

describe("postPaint", () => {
  it("POSTs the PNG raw with Content-Type image/png and re-reads the flow", async () => {
    serve(200, { ok: true, file: "textured_paint.glb", source: "textured.glb", stem: "hen" });
    const dirty = vi.fn();
    const off = onFlowDirty(dirty);
    const png = new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" });
    const out = await postPaint("mesh", "hen", png);
    off();
    expect(out).toMatchObject({ ok: true, file: "textured_paint.glb" });
    expect(calls).toEqual([{ url: "/paint/mesh?stem=hen", method: "POST", headers: { "Content-Type": "image/png" }, body: png }]);
    expect(dirty).toHaveBeenCalledWith("/paint/mesh?stem=hen");
  });

  it("throws the server's error text (factory_http.py: empty body, stem query required)", async () => {
    serve(400, { ok: false, error: "empty body" });
    await expect(postPaint("stack", "hen", new Blob([]))).rejects.toThrow("empty body");
    serve(500, "no GLB in out/hen");
    await expect(postPaint("mesh", "hen", new Blob([]))).rejects.toThrow("no GLB in out/hen");
  });
});
