import { copyFileSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const frontendScripts = dirname(fileURLToPath(import.meta.url));
const desktop = join(frontendScripts, "..", "..");
const repo = join(desktop, "..");
const src = join(repo, "scripts");
const dest = join(desktop, "internal", "runtime", "fs", "scripts");
const keepExt = new Set([".py", ".sh"]);

function copyDir(from, to) {
  mkdirSync(to, { recursive: true });
  for (const name of readdirSync(from)) {
    if (name === "__pycache__" || name.startsWith(".")) continue;
    const fp = join(from, name);
    const st = statSync(fp);
    if (st.isDirectory()) {
      copyDir(fp, join(to, name));
      continue;
    }
    if (!keepExt.has(extname(name).toLowerCase())) continue;
    copyFileSync(fp, join(to, name));
  }
}

rmSync(dest, { recursive: true, force: true });
copyDir(src, dest);
const fsRoot = join(desktop, "internal", "runtime", "fs");
mkdirSync(fsRoot, { recursive: true });
copyFileSync(join(desktop, "deps.json"), join(fsRoot, "deps.json"));
console.log("copied factory scripts -> internal/runtime/fs/scripts");
console.log("copied deps.json");
