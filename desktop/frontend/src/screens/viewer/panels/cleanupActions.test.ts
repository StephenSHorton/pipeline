import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Lod } from "@/lib/api";

// No call here reaches a factory: askConfirm, toast and fetch are mocks; the viewer seams are fakes.
const askConfirm = vi.fn(async (_m: string, _y?: string) => true);
const toast = vi.fn();
vi.mock("@/lib/ui", () => ({ askConfirm, toast }));

const { useJobs } = await import("@/lib/jobs");
const { carve, floaters, resetCleanup } = await import("./cleanupActions");
const { useCleaning } = await import("./cleanupApi");

const g = globalThis as unknown as { fetch: typeof fetch };
const realFetch = g.fetch;
let posted: { url: string; body: Record<string, unknown> }[] = [];
const answer = (data: unknown, status = 200) => {
  g.fetch = vi.fn(async (u: RequestInfo | URL, init?: RequestInit) => {
    posted.push({ url: String(u), body: JSON.parse(String(init?.body ?? "{}")) });
    return new Response(JSON.stringify(data), { status });
  }) as typeof fetch;
};

let calls: string[] = [];
let onScreen = "psx.glb";
let lods: Lod[] = [];
const deps = () => ({
  tools: {
    currentFile: () => onScreen,
    loadFile: vi.fn(async (file: string) => {
      calls.push(`load ${file}`);
      // loadFile re-reads lods.json: the new entry shows only after it.
      lods = [{ id: file.startsWith("psx") ? "psx-clean" : "clean", file, label: "Clean" }, ...lods];
      onScreen = file;
    }),
  },
  lods: () => lods,
  pickLod: vi.fn((id: string) => calls.push(`pick ${id}`)),
  onWritten: vi.fn((file: string) => calls.push(`wrote ${file}`)),
  reloadFlow: vi.fn(async (stem: string) => calls.push(`flow ${stem}`)),
});

beforeEach(() => {
  posted = [];
  calls = [];
  onScreen = "psx.glb";
  lods = [{ id: "psx", file: "psx.glb", label: "PSX" }];
  askConfirm.mockClear();
  askConfirm.mockResolvedValue(true);
  toast.mockClear();
  resetCleanup();
  useJobs.setState({ job: null });
});
afterEach(() => {
  g.fetch = realFetch;
});

describe("Floaters", () => {
  it("asks once, posts the legacy body, loads the copy, pins its LOD, then re-reads the flow", async () => {
    answer({ ok: true, stem: "hen", file: "psx_clean.glb", faces_in: 10, faces_out: 8 });
    const d = deps();
    expect(await floaters("hen", d)).toBe("psx_clean.glb");
    expect(askConfirm).toHaveBeenCalledWith("Cleanup cuts a copy of psx.glb and saves it as psx_clean.glb. The original stays.", "Cut a copy");
    expect(posted).toEqual([{ url: "/clean", body: { stem: "hen", op: "floaters", keep: 1, from: "psx.glb" } }]);
    expect(calls).toEqual(["wrote psx_clean.glb", "load psx_clean.glb", "pick psx-clean", "flow hen"]);
    expect(toast).not.toHaveBeenCalled();
    expect(useCleaning.getState().busy).toBe(false);
  });

  it("asks once per asset: the second clean cuts the cleaned copy without asking", async () => {
    answer({ ok: true, file: "psx_clean.glb" });
    await floaters("hen", deps());
    await floaters("hen", deps());
    expect(askConfirm).toHaveBeenCalledTimes(1);
    expect(posted[1].body.from).toBe("psx_clean.glb");
    onScreen = "mesh_hero.glb";
    await floaters("hen", deps());
    expect(askConfirm).toHaveBeenCalledTimes(1);
    await floaters("goat", deps());
    expect(askConfirm).toHaveBeenCalledTimes(2);
  });

  it("Cancel cuts nothing and asks again next time", async () => {
    answer({ ok: true });
    askConfirm.mockResolvedValueOnce(false);
    const d = deps();
    expect(await floaters("hen", d)).toBeNull();
    expect(posted).toEqual([]);
    expect(d.reloadFlow).not.toHaveBeenCalled();
    await floaters("hen", deps());
    expect(askConfirm).toHaveBeenCalledTimes(2);
  });

  it("falls back to mesh_clean.glb when the answer names no file", async () => {
    answer({ ok: true });
    onScreen = "mesh.glb";
    await floaters("hen", deps());
    expect(calls).toEqual(["wrote mesh_clean.glb", "load mesh_clean.glb", "pick clean", "flow hen"]);
  });

  it("does not pick a LOD lods.json does not list", async () => {
    answer({ ok: true, file: "mesh_clean.glb" });
    onScreen = "mesh.glb";
    const d = deps();
    d.tools.loadFile = vi.fn(async (file: string) => {
      calls.push(`load ${file}`);
    });
    await floaters("hen", d);
    expect(d.pickLod).not.toHaveBeenCalled();
    expect(calls).toEqual(["wrote mesh_clean.glb", "load mesh_clean.glb", "flow hen"]);
  });

  it("a refusal is a toast with the server's message, and nothing loads", async () => {
    answer({ ok: false, error: "no such mesh: psx.glb" }, 400);
    const d = deps();
    expect(await floaters("hen", d)).toBeNull();
    expect(toast).toHaveBeenCalledWith("no such mesh: psx.glb");
    expect(d.tools.loadFile).not.toHaveBeenCalled();
    expect(d.reloadFlow).not.toHaveBeenCalled();
    expect(useCleaning.getState().busy).toBe(false);
  });

  it("a load failure is a toast; the flow is still re-read (the file was written)", async () => {
    answer({ ok: true, file: "psx_clean.glb" });
    const d = deps();
    d.tools.loadFile = vi.fn(async () => {
      throw new Error("no glb loaded");
    });
    await floaters("hen", d);
    expect(toast).toHaveBeenCalledWith("no glb loaded");
    expect(d.pickLod).not.toHaveBeenCalled();
    expect(d.reloadFlow).toHaveBeenCalledWith("hen");
  });

  it("waits for a job on this asset, before asking", async () => {
    answer({ ok: true });
    useJobs.setState({ job: { id: "j1", stem: "hen", kind: "rig", state: "running", progress: null, log: "" } });
    expect(await floaters("hen", deps())).toBeNull();
    expect(toast).toHaveBeenCalledWith("Wait for the job on this asset to finish.");
    expect(askConfirm).not.toHaveBeenCalled();
    expect(posted).toEqual([]);
  });

  it("a job on another asset does not block", async () => {
    answer({ ok: true, file: "psx_clean.glb" });
    useJobs.setState({ job: { id: "j1", stem: "goat", kind: "rig", state: "running", progress: null, log: "" } });
    expect(await floaters("hen", deps())).toBe("psx_clean.glb");
  });

  it("refuses a file under a folder instead of cutting a different mesh", async () => {
    answer({ ok: true });
    onScreen = "rig/rigged_allclips.glb";
    expect(await floaters("hen", deps())).toBeNull();
    expect(toast).toHaveBeenCalledWith("Cleanup can't cut rig/rigged_allclips.glb. Show a detail level first.");
    expect(askConfirm).not.toHaveBeenCalled();
    expect(posted).toEqual([]);
  });

  it("drops a second clean while one runs", async () => {
    let release: (r: Response) => void = () => {};
    g.fetch = vi.fn(async (u: RequestInfo | URL, init?: RequestInit) => {
      posted.push({ url: String(u), body: JSON.parse(String(init?.body ?? "{}")) });
      return new Promise<Response>((r) => (release = r));
    }) as typeof fetch;
    const first = floaters("hen", deps());
    await vi.waitFor(() => expect(posted.length).toBe(1));
    expect(useCleaning.getState().busy).toBe(true);
    expect(await carve("hen", { local: { x: 0, y: 0, z: 0 } as never, scale: 1 }, 0.1, deps())).toBeNull();
    release(new Response(JSON.stringify({ ok: true, file: "psx_clean.glb" })));
    expect(await first).toBe("psx_clean.glb");
    expect(posted.length).toBe(1);
    expect(useCleaning.getState().busy).toBe(false);
  });
});

describe("Carve", () => {
  it("posts the legacy sphere body: mesh-local centre, world radius over the mesh scale", async () => {
    answer({ ok: true, file: "mesh_clean.glb" });
    onScreen = "mesh_clean.glb";
    const hit = { local: { x: 0.5, y: -0.25, z: 1 } as never, scale: 2 };
    expect(await carve("hen", hit, 0.3, deps())).toBe("mesh_clean.glb");
    expect(askConfirm).toHaveBeenCalledWith("Cleanup cuts a copy of mesh_clean.glb and saves it as mesh_clean.glb. The original stays.", "Cut a copy");
    expect(posted).toEqual([{ url: "/clean", body: { stem: "hen", op: "sphere", center: [0.5, -0.25, 1], radius: 0.15, from: "mesh_clean.glb" } }]);
    expect(calls).toEqual(["wrote mesh_clean.glb", "load mesh_clean.glb", "pick clean", "flow hen"]);
  });

  it("a server refusal is a toast", async () => {
    answer({ ok: false, error: "sphere would delete the entire mesh" }, 500);
    expect(await carve("hen", { local: { x: 0, y: 0, z: 0 } as never, scale: 1 }, 9, deps())).toBeNull();
    expect(toast).toHaveBeenCalledWith("sphere would delete the entire mesh");
  });
});
