# Quantum Diffusion Server

A macOS control panel for [mflux-server](../README.md): a double-clickable app that installs its own Python, starts and supervises the server, exposes its configuration as a form, and drives model preparation.

Tauri 2 (Rust) + React 19 + Vite. **Apple Silicon only** — mlx is, by construction.

## Building

```sh
cd desktop
npm install
npm run app:build     # → src-tauri/target/release/bundle/
```

Prerequisites: Node, Rust, and `uv` on the `PATH` (it gets copied into the bundle as a sidecar).

The result: `Quantum Diffusion Server.app` (**57 MB**) and a `.dmg` (**25 MB**), ad-hoc signed (`signingIdentity: "-"`). No Developer ID certificate, no notarization: this is for personal use. After moving it between machines, `xattr -d com.apple.quarantine "Quantum Diffusion Server.app"` may be needed (the quotes matter: the name contains spaces).

For development: `npm run app:dev`. Beware that `tauri dev` inherits a comfortable working directory and therefore hides relative-path problems — serious checks happen on the built `.app`.

## Why not PyInstaller

The server's venv weighs **1.1 GB**: torch 501 MB (a hard mflux dependency, imported at module level in `weight_loader.py`), mlx 178 MB including a **150 MB** `mlx.metallib` of Metal shaders, opencv and matplotlib 142 MB pulled in by mflux without us ever using them. That is **186 native binaries** in total. That metallib and those 186 binaries are exactly what makes freezing fragile, and every one of them would need signing with the hardened runtime.

`uv`, by contrast, is a single 50 MB binary that can download CPython and rebuild the environment from `uv.lock` — 86 packages, identically. So the app stays at 57 MB, no third-party native binary enters the signed bundle, and the environment installs into the data directory on first launch.

## What the app does

```
~/Library/Application Support/com.exalandru.qds/
├── server/               writable copy of the Python project (source for uv sync)
├── env/                  virtual environment, ~1.0 GB
├── python/               uv-managed CPython 3.12, ~67 MB
├── uv-cache/             pruned after installation (1.5 GB reclaimed)
├── images/               images served for response_format="url"
└── server-config.json    driven by the Configuration tab
```

**First launch.** The app copies the bundle's resources into `server/` — the bundle is read-only — then runs:

```sh
uv sync --frozen --no-dev --no-editable --managed-python --python 3.12 --project <appdata>/server
```

Roughly 1.1 GB of download. `--no-editable` matters: without it the installed package would point back at the project copy. So does `--python 3.12`, see below.

**Starting the server.** The Rust side reads the port from the configuration, checks it is free, creates the write directories, then launches `<appdata>/env/bin/mflux-server` with everything absolute:

| variable | value |
|---|---|
| `MFLUX_SERVER_CONFIG` | `<appdata>/server-config.json` |
| `MFLUX_SERVER_HOST` / `PORT` | `127.0.0.1` / `server.port`, `8765` by default |
| `MFLUX_SERVER_IMAGE_STORE` | `<appdata>/images` |
| `MFLUX_SERVER_LOG_FILE` | `""` — the app captures instead |
| `MFLUX_SERVER_LOG_JSON` | `1` |
| `HF_HOME`, `HF_HUB_DISABLE_PROGRESS_BARS`, `PYTHONUNBUFFERED` | |

The port used to be drawn at random on every start, which made the address unstable and unconfigurable. Reading it from the file makes it fixed and editable from the Configuration tab — but it also makes collisions possible, so the port is test-bound before launching. On a collision the error is immediate and names the two likely causes (a server left over from a previous session, or `uv run mflux-server` running alongside) instead of surfacing 30 seconds later as a health-check timeout. There is no automatic fallback to another port: that would restore exactly the instability being fixed.

**Noticing a death.** `status()` calls `try_wait()` on each status poll, so a server that exits on its own is reaped within four seconds: the dashboard flips to *stopped*, shows the exit code or signal, and Start relaunches without needing Stop first. Lazily, at the rhythm of the existing poll — no watcher task, no channel.

**Downloading weights.** The Models tab reads the catalogue through `mflux-server-fetch --status`, a
third console script from the same wheel, and its **Install** button runs `mflux-server-fetch <key>` as
a sidecar whose output is relayed to the Logs tab like a conversion. Going through that script rather
than the HTTP API is what makes the list complete — disabled models included — and available with the
server stopped, which is exactly when you want to fetch weights. `HF_HUB_DISABLE_PROGRESS_BARS` is left
*on* for this one, unlike for the server: the progress bar is the feedback.

**Stopping.** SIGTERM to the process group, a wait bounded at `shutdown_grace_s + 8s`, then SIGKILL. The group rather than the pid alone, otherwise quitting the app would orphan the server: macOS does not reap grandchildren.

## Four traps we hit, and their fixes

Each was found by actually running the chain, not by reading it.

**The server would not start from a `.app`.** `image_store` and `log_file` were relative to the current directory and created during `create_app`, before the bind. Launched by Finder, the current directory is `/`, read-only → immediate failure. Fixed on the server side: both paths are now made absolute at validation time, and `setup_logging` creates the parent directories. The app additionally sets an explicit `current_dir`, belt and braces.

**`server-config.json` was silently ignored.** Its default path is relative to the Python *package*, so `site-packages/` in a wheel install, where the file does not exist — and a missing configuration is not an error. Every setting would have gone in the bin without a word. Hence `MFLUX_SERVER_CONFIG` always being explicit, plus a `warning` on the server side when no file is found.

**hatchling refused to build the wheel.** `pyproject.toml` declares `readme = "README.md"`: without that file in the resources, `uv sync` fails at the build step. So `README.md` is part of the embedded payload — it is not documentation here, it is a build dependency.

**uv installed Python 3.13 instead of 3.12.** Discovery of `.python-version` depends on the current directory, which is unpredictable for a sidecar. Since `requires-python` allows `>=3.12,<3.14`, uv took the newest — and `uv.lock`, whose markers tell 3.12 and 3.13 apart (`torch>=2.8`, `tokenizers`), resolved a different package set from the one that had been tested. The bootstrap now reads `.python-version` from the project copy and passes it as `--python`, which removes the ambiguity entirely. Incidentally, Tauri's `resources/server/**/*` glob **ignores dotfiles**: `.python-version` has to be listed explicitly in `tauri.conf.json`.

## Architecture

The Rust side does only what the browser cannot: install the environment, supervise a process, write a file.

| file | role |
|---|---|
| [src-tauri/src/paths.rs](src-tauri/src/paths.rs) | working locations, all absolute |
| [src-tauri/src/bootstrap.rs](src-tauri/src/bootstrap.rs) | resource copy, `uv sync`, cache pruning |
| [src-tauri/src/supervisor.rs](src-tauri/src/supervisor.rs) | server lifecycle, SIGTERM → SIGKILL ladder, output relay |
| [src-tauri/src/config.rs](src-tauri/src/config.rs) | `server-config.json`, atomic writes |
| [src-tauri/src/lib.rs](src-tauri/src/lib.rs) | commands exposed to React |

Everything else goes through the server's HTTP API, which React queries directly — including `/v1/progress` over Server-Sent Events, which it would be absurd to funnel across the IPC bridge. No schema is duplicated in Rust or TypeScript: the server already validates its configuration and rejects what makes no sense with a 400.

Progress uses `fetch` rather than `EventSource`, because the latter allows no `Authorization` header and `/v1/progress` is protected like the rest of `/v1`.

### The two output channels

The server runs with `MFLUX_SERVER_LOG_JSON=1`, which separates:

- **stdout**: the structured events, one line one valid JSON object, nothing else;
- **stderr**: the human-readable text, the tqdm bars, uvicorn's startup logs.

That separation is not cosmetic. mflux renders its denoising bar with tqdm, which writes carriage-return-terminated fragments to stderr **with no newline**: the JSON objects ended up glued to them on the same segment (`\r 0%| | 0/40 [...]{"ts": …}`) and a consumer splitting on `\n` missed all of them. tqdm offers no environment variable to silence itself, hence moving the JSON to stdout and disabling uvicorn's access log, which would have polluted it in turn.

The Logs tab shows both: level filtering on the structured events, raw text displayed as-is.

## Screens

- **Installation** — shown on its own as long as the environment is not ready, or was built by an earlier version of the app. uv's output goes by live: this is a gigabyte-scale download, an indeterminate spinner would not do.
- **Dashboard** — status, warm model, MLX memory, start/stop/restart, a progress bar fed by SSE, cancel, free memory, open `/docs`. "Server ready" and "model warm" are two distinct states: the server answers within a second but loads no weights at startup.
- **Configuration** — a form over `server-config.json`. Controls are greyed out based on `/v1/capabilities`: a distilled model's guidance is fixed, and the server already rejects any value with a 400. A note points out that the configuration is only read at startup.
- **Models** — the HuggingFace token (written where `hf auth login` writes it, so as not to duplicate a secret already sitting in plaintext next to it), the `flux2-dev` conversion wizard component by component, and the catalogue with the declared capabilities.
- **Logs** — both channels, filterable.

## Limitations

- No generation playground: the server is still driven by an OpenAI-compatible frontend.
- No automatic updates (`tauri-plugin-updater`).
- No hot reload of the configuration: the app restarts the process.
- Port selection has an unavoidable race window — `mflux-server` accepts no pre-opened socket and does not announce the port it got.
- The interface interactions were not clicked through automatically: screenshots and AppleScript automation require permissions that can only be granted by hand. What *is* verified is the chain beneath the interface — bootstrap, resulting environment, working server, no orphan processes.
