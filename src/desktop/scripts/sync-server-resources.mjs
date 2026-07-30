/**
 * Copy the Python project into the Tauri bundle's resources.
 *
 * Nothing is duplicated in VCS: `build/desktop/staging/` is gitignored and
 * regenerated before every build (`beforeBuildCommand`). What ships is the bare
 * minimum for `uv sync --frozen` to rebuild the environment: the manifest, the
 * lock, the Python version, and the package itself.
 *
 * The `uv` binary is copied at the same time, under the name Tauri requires for
 * a sidecar: `uv-<target triple>`.
 */
import { execFileSync } from "node:child_process";
import { chmodSync, cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const desktop = resolve(here, "..");
const repo = resolve(desktop, "../..");
const server = join(repo, "src", "server");
const staging = join(repo, "build", "desktop", "staging");
const resources = join(staging, "resources", "server");

/**
 * Repo files and directories to embed, relative to its root.
 *
 * `README.md` is not documentation here: `pyproject.toml` declares it as
 * `readme`, and hatchling refuses to build the wheel without it.
 * `.python-version` pins the interpreter — without it, uv would take the newest
 * one satisfying `requires-python`, and the lock would resolve a different
 * package set.
 */
const PAYLOAD = ["pyproject.toml", "uv.lock", ".python-version", "README.md", "mflux_server"];

function syncPython() {
  rmSync(resources, { recursive: true, force: true });
  mkdirSync(resources, { recursive: true });

  for (const entry of PAYLOAD) {
    const source = join(server, entry);
    if (!existsSync(source)) {
      throw new Error(`Missing resource: ${source}`);
    }
    cpSync(source, join(resources, entry), {
      recursive: true,
      // No bytecode caches, no test artifacts: they bloat the bundle and would
      // be rewritten anyway.
      filter: (path) => !/(__pycache__|\.pyc$|\.pytest_cache|\.ruff_cache)/.test(path),
    });
  }
  console.log(`✓ Python project copied to ${resources}`);
}

function syncUv() {
  const triple = execFileSync("rustc", ["--print", "host-tuple"], { encoding: "utf8" }).trim();
  const binaries = join(staging, "binaries");
  mkdirSync(binaries, { recursive: true });

  let uv;
  try {
    uv = execFileSync("which", ["uv"], { encoding: "utf8" }).trim();
  } catch {
    throw new Error("`uv` not found in PATH: install it before building.");
  }

  // The triple suffix is mandated by Tauri for `externalBin` entries.
  const target = join(binaries, `uv-${triple}`);
  cpSync(uv, target);
  chmodSync(target, 0o755);
  const version = execFileSync(uv, ["--version"], { encoding: "utf8" }).trim();
  console.log(`✓ sidecar ${version} copied to uv-${triple}`);
}

syncPython();
syncUv();
