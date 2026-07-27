import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Port fixe et strict : `tauri.conf.json` pointe dessus via devUrl, donc un
// repli silencieux sur un autre port ferait démarrer l'app sur du vide.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5273, strictPort: true },
  build: { target: "safari18", sourcemap: true },
});
