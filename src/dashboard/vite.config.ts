import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// `base` matters as much as `outDir`: the server mounts this build under
// `/dashboard`, so every asset URL in index.html has to be written relative to
// that prefix. Built at the root instead, the page loads and then asks for
// `/assets/…`, which the server answers with the OpenAI 404 handler.
//
// The output lands *inside* the Python package, as `qds/_dashboard`, because
// that is what it is: package data the server serves at `/dashboard`. Building
// it elsewhere and copying it in at package time meant a relative path that had
// to resolve the same way from a checkout and from an unpacked sdist, and it
// does not — `uv build` builds the wheel from the sdist, where `../../build`
// exists nowhere.
export default defineConfig({
  plugins: [react()],
  base: "/dashboard/",
  clearScreen: false,
  server: {
    port: 5273,
    strictPort: true,
    // `npm run dev` serves the page; the API still belongs to the server, so
    // same-origin calls are forwarded to it rather than duplicated in config.
    proxy: {
      "/admin": "http://127.0.0.1:8765",
      "/v1": "http://127.0.0.1:8765",
      "/health": "http://127.0.0.1:8765",
      "/images": "http://127.0.0.1:8765",
      // The playground's API and its images. `/playground` itself is the page,
      // which Vite serves as `/dashboard/playground.html` in dev.
      "/playground/api": "http://127.0.0.1:8765",
      "/playground/images": "http://127.0.0.1:8765",
    },
  },
  build: {
    // Safari 18 was the floor while this only ever ran in a macOS WebView. It
    // is a web page now, so the floor is what the browsers people actually use
    // support.
    target: ["safari17", "chrome120", "firefox120"],
    sourcemap: true,
    outDir: fileURLToPath(new URL("../server/qds/_dashboard", import.meta.url)),
    emptyOutDir: true,
    // Two pages, one design system. Rollup needs both entries named, or
    // `playground.html` is simply not built and the server answers its route
    // with a missing file.
    rollupOptions: {
      input: {
        dashboard: fileURLToPath(new URL("./index.html", import.meta.url)),
        playground: fileURLToPath(new URL("./playground.html", import.meta.url)),
      },
    },
  },
});
