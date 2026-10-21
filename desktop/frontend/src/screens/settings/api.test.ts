import { afterEach, describe, expect, it, vi } from "vitest";
import { WAILS } from "@/lib/api";
import { inDesktop, openDataDirHere, readDataDir } from "./api";

const DESKTOP = { messageHandlers: { external: { postMessage: () => {} } } };
const setWindow = (w: unknown) => ((globalThis as { window?: unknown }).window = w);

afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
  vi.unstubAllGlobals();
});

describe("settings host calls", () => {
  it("is in the app only with the native bridge", () => {
    expect(inDesktop()).toBe(false);
    setWindow({ wails: { Call: { ByName: async () => "" } } });
    expect(inDesktop()).toBe(false);
    setWindow({ webkit: DESKTOP });
    expect(inDesktop()).toBe(true);
  });
  it("reads DataDir in the app", async () => {
    const byName = vi.fn(async (_m: string) => "/Users/me/Pipeline");
    setWindow({ webkit: DESKTOP, wails: { Call: { ByName: byName } } });
    expect(await readDataDir()).toBe("/Users/me/Pipeline");
    expect(byName).toHaveBeenCalledWith(WAILS.DATADIR);
  });
  it("falls back to /paths in a browser or on an empty answer", async () => {
    const fetchMock = vi.fn(async (_u: string) => new Response(JSON.stringify({ ok: true, dataDir: "/from/paths" })));
    vi.stubGlobal("fetch", fetchMock);
    setWindow({ wails: { Call: { ByName: async () => "" } } });
    expect(await readDataDir()).toBe("/from/paths");
    setWindow({ webkit: DESKTOP, wails: { Call: { ByName: async () => "" } } });
    expect(await readDataDir()).toBe("/from/paths");
    expect(fetchMock.mock.calls[0][0]).toBe("/paths");
  });
  it("opens over Wails in the app, else POST /open-data-dir, and throws the error", async () => {
    const byName = vi.fn(async (_m: string) => undefined);
    setWindow({ webkit: DESKTOP, wails: { Call: { ByName: byName } } });
    await openDataDirHere();
    expect(byName).toHaveBeenCalledWith(WAILS.OPEN_DIR);
    const fetchMock = vi.fn(async (_u: string, _o?: RequestInit) => new Response(JSON.stringify({ ok: false, error: "no folder" }), { status: 500 }));
    vi.stubGlobal("fetch", fetchMock);
    setWindow({});
    await expect(openDataDirHere()).rejects.toThrow("no folder");
    expect(fetchMock.mock.calls[0][0]).toBe("/open-data-dir");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST");
  });
});
