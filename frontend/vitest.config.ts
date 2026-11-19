// Test-only config. Kept separate from vite.config.ts because vitest bundles
// its own copy of vite, and mixing the two configs' types breaks `tsc -b`.
import { defineConfig } from "vitest/config";

export default defineConfig({
  esbuild: { jsx: "automatic" },
  test: {
    setupFiles: ["./src/test-setup.ts"],
  },
});
