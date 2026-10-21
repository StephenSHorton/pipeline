import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WAILS } from "@/lib/api";
import {
  checkForUpdates,
  installingVersion,
  markInstalling,
  OUTSIDE_WAILS,
  overlayTitle,
  rawVersion,
  resetUpdateForTest,
  setVersion,
  startUpdateWatch,
  updateView,
  useUpdate,
  versionLabel,
  type UpdateState,
} from "./update";

const base: UpdateState = { status: "idle", version: "", updatingTo: "", error: "", buttonLabel: "Check for Updates" };
const view = (p: Partial<UpdateState>) => updateView({ ...base, ...p });

const DESKTOP = { messageHandlers: { external: { postMessage: () => {} } } };
type ByName = (method: string, ...args: unknown[]) => Promise<unknown>;
function setWails(byName?: ByName, on?: (name: string, cb: (ev: { data?: unknown }) => void) => void) {
  (globalThis as { window?: unknown }).window = { webkit: DESKTOP, wails: { Call: byName ? { ByName: byName } : undefined, Events: on ? { On: on } : undefined } };
}

beforeEach(() => resetUpdateForTest());
afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
  delete (globalThis as { document?: unknown }).document;
  vi.unstubAllGlobals();
});

describe("versions", () => {
  it("keeps the raw version like legacy applyVersion", () => {
    expect(rawVersion("Pipeline v1.4.2")).toBe("1.4.2");
    expect(rawVersion("v1.4.2")).toBe("1.4.2");
    expect(rawVersion("1.4.2")).toBe("1.4.2");
    expect(rawVersion("dev")).toBe("dev");
    expect(rawVersion("")).toBe("");
  });
  it("labels unknown as dev and adds one v", () => {
    expect(versionLabel("")).toBe("dev");
    expect(versionLabel("dev")).toBe("dev");
    expect(versionLabel("1.4.2")).toBe("v1.4.2");
    expect(versionLabel("v1.4.2")).toBe("v1.4.2");
  });
  it("keeps the first version it was given", () => {
    setVersion("Pipeline v1.0.0");
    setVersion("2.0.0");
    expect(useUpdate.getState().version).toBe("1.0.0");
  });
});

describe("update row", () => {
  it("reads Checking… before the first check", () => {
    expect(view({ status: "boot" })).toMatchObject({ hint: "Checking…", tone: "", buttonDisabled: false });
  });
  it("covers every state with the legacy copy and colour", () => {
    expect(view({ version: "1.2.0" })).toEqual({ hint: "Pipeline v1.2.0", tone: "", buttonLabel: "Check for Updates", buttonDisabled: false });
    expect(view({ status: "checking" })).toEqual({ hint: "Pipeline dev · Checking…", tone: "running", buttonLabel: "Checking…", buttonDisabled: true });
    expect(view({ status: "uptodate", version: "1.2.0" })).toMatchObject({ hint: "Pipeline v1.2.0 · Up to date", tone: "ok", buttonDisabled: false });
    expect(view({ status: "error", error: "boom" })).toMatchObject({ hint: "boom", tone: "error" });
    expect(view({ status: "error" })).toMatchObject({ hint: "Check failed", tone: "error" });
    expect(view({ status: "downloading", updatingTo: "v1.3.0" })).toMatchObject({
      hint: "Updating to v1.3.0… Pipeline will restart.",
      tone: "running",
      buttonDisabled: true,
    });
    expect(view({ status: "downloading" })).toMatchObject({ hint: "Updating… Pipeline will restart.", tone: "running", buttonDisabled: true });
  });
});

describe("overlay copy", () => {
  it("names the version when it has one", () => {
    expect(overlayTitle("1.3.0")).toBe("Updating to v1.3.0…");
    expect(overlayTitle("v1.3.0")).toBe("Updating to v1.3.0…");
    expect(overlayTitle("")).toBe("Updating…");
  });
  it("reads a string or [string] payload", () => {
    expect(installingVersion("1.3.0")).toBe("1.3.0");
    expect(installingVersion(["1.3.0"])).toBe("1.3.0");
    expect(installingVersion(undefined)).toBe("");
    expect(installingVersion([7])).toBe("");
  });
});

describe("checkForUpdates", () => {
  it("manual, outside Wails: says it runs in the desktop app", async () => {
    await checkForUpdates(true);
    expect(useUpdate.getState()).toMatchObject({ status: "error", error: OUTSIDE_WAILS });
    expect(updateView(useUpdate.getState()).hint).toBe("Check for Updates runs in the desktop app.");
  });
  it("silent, outside Wails: stays idle", async () => {
    await checkForUpdates(false);
    expect(useUpdate.getState().status).toBe("idle");
  });
  it("null means up to date", async () => {
    setWails(async () => null);
    await checkForUpdates(true);
    expect(useUpdate.getState().status).toBe("uptodate");
  });
  it("an update downloads and installs with the info it got", async () => {
    const calls: unknown[][] = [];
    const info = { version: "1.3.0" };
    setWails(async (m, ...a) => {
      calls.push([m, ...a]);
      return m === WAILS.CHECK ? info : undefined;
    });
    await checkForUpdates(true);
    expect(calls).toEqual([[WAILS.CHECK], [WAILS.INSTALL, info]]);
    const s = useUpdate.getState();
    expect(s).toMatchObject({ status: "downloading", updatingTo: "1.3.0" });
    // Legacy returns before resetting the button: it stays "Checking…", disabled.
    expect(updateView(s)).toMatchObject({ buttonLabel: "Checking…", buttonDisabled: true });
  });
  it("manual errors show; silent errors stay idle", async () => {
    setWails(async () => {
      throw new Error("no network");
    });
    await checkForUpdates(true);
    expect(useUpdate.getState()).toMatchObject({ status: "error", error: "no network" });
    await checkForUpdates(false);
    expect(useUpdate.getState().status).toBe("idle");
    expect(updateView(useUpdate.getState()).buttonLabel).toBe("Check for Updates");
  });
});

describe("in a plain browser", () => {
  it("the loaded runtime answering \"\" still counts as outside Wails", async () => {
    // The Wails runtime script loads in Chrome too; with no native bridge every call resolves to "".
    (globalThis as { window?: unknown }).window = { wails: { Call: { ByName: async () => "" } } };
    await checkForUpdates(true);
    expect(updateView(useUpdate.getState()).hint).toBe("Check for Updates runs in the desktop app.");
  });
});

describe("update:installing", () => {
  it("from idle keeps the button text, disabled", () => {
    markInstalling("");
    expect(updateView(useUpdate.getState())).toMatchObject({ hint: "Updating… Pipeline will restart.", buttonLabel: "Check for Updates", buttonDisabled: true });
  });
  it("subscribes at once when the runtime is there", async () => {
    let cb: ((ev: { data?: unknown }) => void) | undefined;
    setWails(async () => null, (name, fn) => {
      if (name === "update:installing") cb = fn;
    });
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ version: "1.2.0" })));
    startUpdateWatch();
    startUpdateWatch(); // idempotent
    await vi.waitFor(() => expect(useUpdate.getState().status).toBe("uptodate"));
    cb!({ data: ["1.3.0"] });
    expect(useUpdate.getState()).toMatchObject({ status: "downloading", updatingTo: "1.3.0" });
  });
  it("waits for wails:ready when the runtime is late", () => {
    const listeners: Record<string, () => void> = {};
    (globalThis as { document?: unknown }).document = { addEventListener: (n: string, fn: () => void) => (listeners[n] = fn) };
    vi.stubGlobal("fetch", async () => new Response("{}"));
    (globalThis as { window?: unknown }).window = {};
    startUpdateWatch();
    let cb: ((ev: { data?: unknown }) => void) | undefined;
    setWails(undefined, (_n, fn) => (cb = fn));
    listeners["wails:ready"]();
    cb!({ data: "2.0.0" });
    expect(useUpdate.getState()).toMatchObject({ status: "downloading", updatingTo: "2.0.0" });
  });
});
