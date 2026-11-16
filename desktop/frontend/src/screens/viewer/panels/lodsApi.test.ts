import { afterEach, describe, expect, it, vi } from "vitest";
import { readTurnaround } from "./lodsApi";

const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
let urls: string[] = [];

function serve(route: (url: string) => Response | undefined) {
  urls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      urls.push(url);
      return route(url) ?? new Response("404 page not found", { status: 404 });
    }),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("readTurnaround", () => {
  it("reads out/<stem>/views.json and serves each view from out/<stem>/ without a cache-bust", async () => {
    serve((u) => (u.startsWith("/out/hen/views.json") ? json({ views: [{ id: "front", file: "views/front.png", role: "shape", label: "Front" }] }) : undefined));
    const figs = await readTurnaround("hen");
    expect(urls.some((u) => u.startsWith("/out/hen/views.json"))).toBe(true);
    expect(figs).toEqual([{ id: "front", file: "views/front.png", caption: "Front", shape: true, src: "/out/hen/views/front.png" }]);
  });

  it("a missing views.json and an unreadable flow give the legacy Input image figure from input.jpg", async () => {
    serve(() => undefined);
    expect(await readTurnaround("hen")).toEqual([{ id: "input", file: "input.jpg", caption: "Input image", shape: true, src: "/out/hen/input.jpg" }]);
  });

  it("the fallback uses the input the flow lists", async () => {
    serve((u) => (u === "/flow/hen" ? json({ stages: [{ files: ["input.png", "lifted.png"] }, { files: ["mesh.glb"] }] }) : undefined));
    const [fig] = await readTurnaround("hen");
    expect(fig.src).toBe("/out/hen/input.png");
  });
});
