/**
 * Copy the distributable Tauri artifacts out of Cargo's build tree.
 *
 * `build/` remains disposable compiler output. `dist/desktop/` contains only
 * the files intended to be installed or shared.
 */
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const bundle = join(repo, "build", "desktop", "tauri", "release", "bundle");
const dist = join(repo, "dist", "desktop");

const sources = [
  { directory: join(bundle, "macos"), extension: ".app" },
  { directory: join(bundle, "dmg"), extension: ".dmg" },
];

rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });

let copied = 0;
for (const { directory, extension } of sources) {
  if (!existsSync(directory)) {
    continue;
  }

  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (!entry.name.endsWith(extension)) {
      continue;
    }
    const source = join(directory, entry.name);
    const target = join(dist, basename(source));
    cpSync(source, target, { recursive: true });
    console.log(`✓ ${entry.name} copied to ${dist}`);
    copied += 1;
  }
}

if (copied === 0) {
  throw new Error(`No .app or .dmg artifacts found under ${bundle}`);
}
