import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Fixed, strict port: `tauri.conf.json` points at it through devUrl, so a silent
// fallback to another port would start the app on nothing at all.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5273, strictPort: true },
  build: {
    target: "safari18",
    sourcemap: true,
    outDir: fileURLToPath(new URL("../../build/desktop/web", import.meta.url)),
    emptyOutDir: true,
  },
});
