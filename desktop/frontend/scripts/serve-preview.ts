// A headless stand-in for the Go host, for pictures and checks without opening the app window: serves the built
// React app (frontend/dist), out/<stem>/ files from the factory root, and proxies the factory routes to a
// factory_http.py it starts itself. Mirrors desktop/main.go assetHandler and sidecar.IsFactoryAPI.
//
//   bun run build && bun scripts/serve-preview.ts            # PORT (default 47801); the factory on PORT + 1
//   FACTORY_ROOT=/path/to/checkout bun scripts/serve-preview.ts
//
// Open http://127.0.0.1:$PORT/ like the app. Nothing here is used by the shipped app.
import { spawn } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import path from "node:path";

const here = import.meta.dirname;
const dist = path.resolve(here, "..", "dist");
const root = process.env.FACTORY_ROOT || path.resolve(here, "..", "..", "..");
const port = Number(process.env.PORT) || 47801;
const factoryPort = port + 1;

const python = existsSync(path.join(root, ".venv/bin/python")) ? path.join(root, ".venv/bin/python") : "python3";
const factory = spawn(python, [path.join(root, "scripts/factory_http.py"), "--host", "127.0.0.1", "--port", String(factoryPort), "--root", root], {
  cwd: root,
  env: { ...process.env, FACTORY_ROOT: root, PYTHONPATH: path.join(root, "scripts") },
  stdio: ["ignore", "ignore", "inherit"],
});
process.on("exit", () => factory.kill());
for (const sig of ["SIGINT", "SIGTERM"] as const) process.on(sig, () => process.exit(0));

const FACTORY_EXACT = new Set(["health", "assets", "status", "generate", "upload", "clean", "2d", "activity", "projects"]);
const FACTORY_PREFIX = ["inspect/", "status/", "glb/", "stack/", "assets/", "paint/", "2d/", "flow/", "export/", "cancel/", "intent/", "rig/", "facing/", "projects/"];
const isFactory = (p: string) => {
  const s = p.replace(/^\//, "");
  return FACTORY_EXACT.has(s) || FACTORY_PREFIX.some((pre) => s.startsWith(pre));
};

function fileIn(base: string, rel: string): string | null {
  const full = path.resolve(base, "." + path.posix.normalize("/" + decodeURIComponent(rel)));
  if (!full.startsWith(base)) return null;
  return existsSync(full) && statSync(full).isFile() ? full : null;
}

Bun.serve({
  hostname: "127.0.0.1",
  port,
  async fetch(req) {
    const url = new URL(req.url);
    const p = url.pathname;
    if (p === "/version" || p === "/paths") return Response.json({ ok: true, version: "preview", raw: "preview", dataDir: root });
    if (isFactory(p)) {
      const target = `http://127.0.0.1:${factoryPort}${p}${url.search}`;
      for (let i = 0; i < 50; i++) {
        try {
          return await fetch(target, { method: req.method, headers: req.headers, body: req.method === "GET" || req.method === "HEAD" ? undefined : await req.arrayBuffer() });
        } catch {
          await Bun.sleep(200); // the factory is still starting
        }
      }
      return new Response("factory not up", { status: 502 });
    }
    if (p.startsWith("/out/")) {
      const f = fileIn(path.join(root, "out"), p.slice(5));
      return f ? new Response(Bun.file(f)) : new Response("not found", { status: 404 });
    }
    const f = fileIn(dist, p === "/" ? "index.html" : p.slice(1));
    return new Response(Bun.file(f ?? path.join(dist, "index.html")));
  },
});
console.log(`preview on http://127.0.0.1:${port}/ (factory :${factoryPort}, root ${root})`);
