import path from "node:path";
import { defineConfig } from "vitest/config";

// Separate from vite.config.ts on purpose: unit tests of pure helpers need neither React nor Tailwind.
export default defineConfig({
  root: import.meta.dirname,
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
