import { copyFileSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const here = import.meta.dirname;
const dist = path.resolve(here, "dist");
// The legacy page: its source stays at frontend/index.html (tests pin it); it is served at /legacy.html
// until every screen is ported.
const legacySrc = path.resolve(here, "index.html");

// Every path the Go host answers itself or proxies to the factory (desktop/main.go assetHandler,
// sidecar.IsFactoryAPI). The page uses relative URLs, so in `vite` dev these go to the running app on :8777.
const HOST = "http://127.0.0.1:8777";
// Matched as whole path segments, so /stack-lab.js (a public file) is not mistaken for /stack/.
const PROXIED = [
  "health", "assets", "status", "generate", "upload", "clean", "2d", "inspect", "glb", "stack",
  "paint", "flow", "export", "cancel", "intent", "rig", "facing", "out", "control", "version",
  "paths", "open-data-dir", "open-export", "diagnostics", "viewer", "wails",
];
const PROXY_RE = `^/(${PROXIED.join("|")})(/|\\?|$)`;

function legacyPage(): Plugin {
  return {
    name: "pipeline-legacy-page",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !/^\/legacy\.html(\?|$)/.test(req.url)) return next();
        res.setHeader("Content-Type", "text/html; charset=utf-8");
        res.setHeader("Cache-Control", "no-store");
        res.end(readFileSync(legacySrc));
      });
    },
    // After the bundle is written (emptyOutDir has already run), so the copy survives.
    writeBundle() {
      mkdirSync(dist, { recursive: true });
      copyFileSync(legacySrc, path.join(dist, "legacy.html"));
    },
  };
}

export default defineConfig({
  root: path.resolve(here, "app"),
  publicDir: path.resolve(here, "public"),
  plugins: [react(), tailwindcss(), legacyPage()],
  resolve: {
    alias: [
      { find: "@", replacement: path.resolve(here, "src") },
      // app/index.html loads /src/main.tsx; the HTML root is app/, the sources live in frontend/src.
      { find: /^\/src\//, replacement: `${path.resolve(here, "src")}/` },
    ],
    dedupe: ["three", "react", "react-dom"],
  },
  optimizeDeps: {
    include: ["stats.js", "three", "@react-three/fiber", "@react-three/drei", "@react-three/postprocessing"],
  },
  build: {
    outDir: dist,
    emptyOutDir: true,
    // Not "assets": the Go host proxies /assets* to the factory (sidecar.IsFactoryAPI).
    assetsDir: "static",
  },
  server: {
    host: "127.0.0.1",
    port: Number(process.env.WAILS_VITE_PORT) || 9245,
    strictPort: true,
    fs: { allow: [here] },
    proxy: { [PROXY_RE]: { target: HOST } },
  },
});
