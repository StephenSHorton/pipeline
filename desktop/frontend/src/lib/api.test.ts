import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createProject, factoryJson, meshUrls, moveAsset, outUrl, renameProject, uploadImage, WAILS, wailsByName } from "./api";

const g = globalThis as unknown as { window?: unknown; fetch: typeof fetch };
const realFetch = g.fetch;
afterEach(() => {
  g.fetch = realFetch;
  delete g.window;
});

const file = (bytes: number[], name = "hen.png") => Object.assign(new Blob([new Uint8Array(bytes)]), { name });

describe("factoryJson", () => {
  it("throws the JSON error text with status and payload", async () => {
    g.fetch = vi.fn(async () => new Response(JSON.stringify({ ok: false, error: "generate already running", busy: true }), { status: 409 }));
    const err = (await factoryJson("/generate", { method: "POST" }).catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe("generate already running");
    expect(err.status).toBe(409);
  });
});

describe("uploadImage", () => {
  it("posts base64 JSON (never a raw File body) to /upload?name=", async () => {
    const seen: RequestInit[] = [];
    g.fetch = vi.fn(async (_u: RequestInfo | URL, init?: RequestInit) => {
      seen.push(init!);
      return new Response(JSON.stringify({ ok: true, path: "in/hen.png", bytes: 3, stem: "hen" }), { status: 201 });
    }) as typeof fetch;
    const res = await uploadImage(file([1, 2, 3]));
    expect(res.path).toBe("in/hen.png");
    expect((g.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0]).toBe("/upload?name=hen.png");
    expect(JSON.parse(String(seen[0].body))).toEqual({ name: "hen.png", data: "AQID" });
  });

  it("falls back to the Wails SaveUpload binding when HTTP fails", async () => {
    g.fetch = vi.fn(async () => new Response("boom", { status: 500 }));
    const byName = vi.fn(async () => ({ ok: true, path: "in/hen.png" }));
    g.window = { wails: { Call: { ByName: byName } } };
    const res = await uploadImage(file([1, 2, 3]));
    expect(res.path).toBe("in/hen.png");
    expect(byName).toHaveBeenCalledWith("github.com/StephenSHorton/pipeline/desktop/internal/factory.Service.SaveUpload", "hen.png", "AQID");
    expect(WAILS.UPLOAD).toBe("github.com/StephenSHorton/pipeline/desktop/internal/factory.Service.SaveUpload");
  });

  it("rethrows the HTTP error outside Wails, and refuses an empty file", async () => {
    g.fetch = vi.fn(async () => new Response(JSON.stringify({ error: "upload too large" }), { status: 413 }));
    await expect(uploadImage(file([1]))).rejects.toThrow("upload too large");
    await expect(uploadImage(file([]))).rejects.toThrow("could not read image");
    expect(wailsByName("x")).toBeNull();
  });
});

describe("projects", () => {
  const project = { id: "mlg-farming", name: "MLG Farming", assets: ["hen"], created: 1, updated: 2 };
  const capture = (body: unknown) => {
    const seen: Array<{ url: string; init: RequestInit }> = [];
    g.fetch = vi.fn(async (u: RequestInfo | URL, init?: RequestInit) => {
      seen.push({ url: String(u), init: init! });
      return new Response(JSON.stringify(body), { status: 200 });
    }) as typeof fetch;
    return seen;
  };

  it("creates with a name, and moves the given assets in the same call", async () => {
    const seen = capture({ ok: true, project });
    expect((await createProject("MLG Farming")).project.id).toBe("mlg-farming");
    await createProject("MLG Farming", ["hen"]);
    expect(seen.map((s) => [s.url, s.init.method, JSON.parse(String(s.init.body))])).toEqual([
      ["/projects", "POST", { name: "MLG Farming" }],
      ["/projects", "POST", { name: "MLG Farming", assets: ["hen"] }],
    ]);
  });

  it("renames by PATCH on the id", async () => {
    const seen = capture({ ok: true, project });
    await renameProject("mlg-farming", "Farm");
    expect([seen[0].url, seen[0].init.method, JSON.parse(String(seen[0].init.body))]).toEqual(["/projects/mlg-farming", "PATCH", { name: "Farm" }]);
  });

  it("moves into a project by POST, unsorts by DELETE on the old one, and asks nothing for no project to none", async () => {
    const seen = capture({ ok: true, project, stem: "hen" });
    await moveAsset("hen", "mlg-farming", "barn");
    await moveAsset("hen", null, "mlg-farming");
    expect(await moveAsset("hen", null, null)).toBeNull();
    expect(seen.map((s) => [s.url, s.init.method])).toEqual([
      ["/projects/mlg-farming/assets", "POST"],
      ["/projects/mlg-farming/assets/hen", "DELETE"],
    ]);
    expect(JSON.parse(String(seen[0].init.body))).toEqual({ stem: "hen" });
  });
});

describe("urls", () => {
  it("encodes the stem the same way everywhere and keeps the mesh chain order", () => {
    expect(outUrl("hen", "rig/rig.json", false)).toBe("/out/hen/rig/rig.json");
    const urls = meshUrls("hen", "psx").map((u) => u.replace(/[?&]t=\d+/, ""));
    expect(urls).toEqual([
      "/glb/hen?lod=psx", "/glb/hen?lod=lookdev", "/glb/hen",
      "/out/hen/mesh_lookdev.glb", "/out/hen/mesh_hero.glb", "/out/hen/mesh.glb", "/out/hen/textured.glb",
    ]);
  });
});
