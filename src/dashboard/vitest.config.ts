import { defineConfig } from "vitest/config";

// The same shape as the Quantum Codex desktop app's, so a test written for one
// reads in the other. `css: true` is what lets a layout test ask the real
// cascade what applies — the Logs pane's height is a cascade question, and a
// stubbed stylesheet answers it wrongly.
export default defineConfig({
  test: {
    environment: "jsdom",
    css: true,
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
