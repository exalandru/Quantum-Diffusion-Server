# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two applications in one repo:

- `src/server` — a Python/FastAPI server exposing [mflux](https://github.com/filipstrand/mflux) (MLX diffusion on Apple Silicon) behind an **OpenAI-Images-compatible API**, keeping one model warm in memory between requests.
- `src/desktop` — **Quantum Diffusion Server** (QDS), a Tauri 2 + React 19 macOS control panel that installs its own Python, supervises the server process, and edits its configuration.

Apple Silicon only (mlx). Python 3.12 is pinned by `src/server/.python-version` — this matters, see "Version pinning" below.

The two READMEs ([src/server/README.md](src/server/README.md), [src/desktop/README.md](src/desktop/README.md)) are long and current: they document the model catalogue, every config key, the endpoint list, and the reasoning behind most non-obvious choices. Read the relevant section before changing behaviour rather than re-deriving it.

## Commands

All commands run from the repo root via the [Makefile](Makefile), which sets `UV_PROJECT_ENVIRONMENT` to the root `.venv` and points `uv` at `src/server`.

```sh
make install        # uv sync (server) + npm install (desktop)
make dev-server     # uv run mflux-server  → http://127.0.0.1:8765, docs at /docs
make dev-desktop    # tauri dev
make test           # pytest — loads no weights
make lint           # ruff check + tsc --noEmit
make build          # wheel + sdist → dist/server/, QDS.app + .dmg → dist/desktop/
make clean          # rm -rf build/ dist/
```

Narrower invocations:

```sh
uv run --project src/server pytest tests/test_engine.py::test_name   # a single test
uv run --project src/server pytest -k idle                            # by keyword
uv run --project src/server ruff format .
npm --prefix src/desktop run typecheck
```

Three console scripts ship in the wheel (`pyproject.toml [project.scripts]`), all usable with the server stopped:

```sh
uv run --project src/server mflux-server              # the API server
uv run --project src/server mflux-server-fetch --status   # catalogue + cache state as JSON
uv run --project src/server mflux-server-fetch <key>      # download one model's weights
uv run --project src/server mflux-server-prequantize      # the flux2-dev 8-bit conversion
```

## Testing

`tests/conftest.py` swaps in a `FakeEngine` that records `GenerationJob`s and returns a 2×2 PNG. **No test loads weights or imports mflux at module scope** — that is the reason `registry.py` keeps every `mflux` import inside the loader functions (importing mflux pulls torch + transformers, several seconds). Preserve that: a top-level `import mflux` anywhere in the package makes the suite slow and CI-hostile.

What the suite covers: the registry and config overrides, OpenAI conformance and error shapes, engine caching/serialization/unload, the idle policy, log formatting, and the FLUX.2-dev port (numeric parity against transformers where feasible).

Real inference is verified **by hand** — the checklist is in [src/server/README.md](src/server/README.md) under "Development". The load-bearing item is: generate twice on the same model and confirm the second is noticeably faster, which is what validates the warm-model cache.

## Server architecture

Request flow: `app.py` (HTTP, validation, OpenAI shape) → `engine.py` (one warm model, one generation at a time) → `registry.py` (which mflux class to instantiate, with what config).

**`registry.py`** is the single source of truth for what each model can do. `ModelSpec` is a frozen dataclass carrying capabilities (`supports_guidance`, `supports_negative_prompt`, `prompt_formats`, `max_dimension`, `prequantized`, `edit`…). Its purpose is to let `app.py` return an explicit 400 *before* loading weights, rather than letting mflux raise a 500 after a 30 GB download. When adding or changing a model, the capability flags are the contract — `/v1/capabilities`, the desktop Configuration form's greyed-out controls, and the pre-flight validation all read from them. `load_model()` mirrors each mflux family's CLI `main()`; that CLI is the reference if results ever diverge.

**`settings.py`** layers configuration: code defaults → `server-config.json` → `MFLUX_SERVER_<KEY>` env vars (which override only the `server` section, plus `MFLUX_SERVER_DEFAULT_MODEL` / `_DEFAULT_SIZE`). `Settings.registry` applies the per-model overrides onto `BASE_SPECS` and is where an invalid combination (guidance on a distilled model, a preset on a model without presets) raises. Resolution precedence, highest first: request → `models.<key>.default_size` → top-level `default_size` → catalogue.

**`engine.py`** holds three invariants, all deliberate and all load-bearing on unified memory:
- one live model at a time (switching models unloads first);
- one generation at a time — an `asyncio.Lock` serializes, and inference runs on a single `ThreadPoolExecutor` worker, never on the event loop;
- one registered mflux callback per model — `CallbackRegistry` has no `unregister`.

Cancellation and the timeout both work only through `_ProgressCallback.call_in_loop`: MLX cannot be interrupted from outside and the worker thread cannot be killed, so the stop lands at the next denoising step. Unload is manual (`_UNLOADABLE_ATTRS` set to `None`, `gc.collect()`, `mx.clear_cache()`) because mflux exposes no teardown.

`ProgressSnapshot` is a single mutable object written from the worker thread and read from the event loop under the GIL — safe precisely *because* generations are serialized. That is what lets `/v1/progress` poll it for SSE without a cross-thread queue, and lets multiple SSE clients coexist. If concurrency ever stops being serialized, this breaks.

**`idle.py`** (`IdleUnloader`) is a context manager wrapped around a request's whole generation loop, not around a single image — an `n=3` request must load once and release once. It counts in-flight requests rather than using a flag, so overlapping requests do not re-arm early.

**`errors.py`** owns the OpenAI error envelope, the `APIError` exception carrying `status_code` / `error_type` / `param` / `code`, and `translate_mflux_exception`, which maps mflux and MLX exceptions onto that envelope (a cancelled generation becomes 499 `generation_stopped`). Pre-flight rejections raise `APIError` directly from where the rule lives — `prompt_must_be_json` in `app.py`, `model_not_prepared` in `registry.py` — but the envelope and the handler registration stay here.

**`flux2_dev/`** is a port, not a wrapper: mflux 0.18.0's FLUX.2 support is klein-only, so this package supplies the `ModelConfig`, an MLX `Mistral3` text encoder, a custom tokenizer, and the weight mapping. Two hazards documented in the server README's "mflux integration notes" have tests standing guard — guidance must be pre-multiplied by 1000 (canary test breaks if mflux fixes its heuristic), and left-padding under a causal mask produces silent NaN unless fully-masked rows are reopened.

**Logging** has two channels when `log_json` is on: structured JSON Lines on **stdout**, human text and tqdm bars on **stderr**. This split is not stylistic — tqdm writes `\r`-terminated fragments with no newline and used to glue itself onto the JSON objects. Do not move JSON back to stderr or re-enable uvicorn's access log in that mode.

## Desktop architecture

The Rust side does only what a browser cannot: install the environment, supervise a process, write a file. Everything else — including SSE progress — goes straight from React to the server's HTTP API. **No configuration schema is duplicated in Rust or TypeScript**; the server validates and returns 400. Keep it that way when adding config keys: the form reads `/v1/capabilities`, it does not encode model rules.

- [src-tauri/src/paths.rs](src/desktop/src-tauri/src/paths.rs) — every working location, always absolute
- [src-tauri/src/bootstrap.rs](src/desktop/src-tauri/src/bootstrap.rs) — copies bundle resources to app data, runs `uv sync --frozen --no-dev --no-editable --managed-python --python <read from the copied .python-version>`, prunes the uv cache
- [src-tauri/src/supervisor.rs](src/desktop/src-tauri/src/supervisor.rs) — spawn, SIGTERM-to-process-group → SIGKILL ladder, output relay, `try_wait()` death detection on each status poll
- [src-tauri/src/config.rs](src/desktop/src-tauri/src/config.rs) — atomic writes to `server-config.json`
- [src-tauri/src/lib.rs](src/desktop/src-tauri/src/lib.rs) — the `#[tauri::command]` surface consumed by [src/api.ts](src/desktop/src/api.ts)

Runtime data lives in `~/Library/Application Support/com.exalandru.qds/` (`server/`, `env/`, `python/`, `images/`, `server-config.json`). The app launches the server with everything passed as absolute paths in env vars, `MFLUX_SERVER_CONFIG` always explicit, `MFLUX_SERVER_LOG_JSON=1`, `MFLUX_SERVER_LOG_FILE=""`.

`scripts/sync-server-resources.mjs` runs as Tauri's `beforeDevCommand`/`beforeBuildCommand`. It stages the Python payload into `build/desktop/staging/` and copies the `uv` binary as a `uv-<target-triple>` sidecar. The payload list is deliberately minimal — but `README.md` is in it because `pyproject.toml` declares it as `readme` and hatchling refuses to build without it, and `.python-version` is in it (and listed explicitly in `tauri.conf.json`, since Tauri's resource glob skips dotfiles) because otherwise uv picks 3.13 and `uv.lock` resolves a different, untested package set.

`tauri dev` inherits a comfortable working directory and hides relative-path bugs. Anything touching paths, resources, or the bootstrap must be verified on the built `.app`.

## Conventions

- Comments explain *why*, especially where the code compensates for an mflux behaviour — most existing comments cite the upstream file and line (`config_resolution.py:57-64`). Keep that habit: it is what makes the compensations auditable when mflux is upgraded.
- Prose (comments, docstrings, READMEs) is in English. The repo was translated from French in commit `1df675e`; do not reintroduce French.
- Ruff: line length 110, rules `E,F,I,UP,B`, target py312.
- `build/` and `dist/` are generated and gitignored, as are `images/` and `*.log`.
- The mflux version is pinned exactly (`mflux==0.18.0`). Bumping it is a real task: the registry, the flux2-dev port, and the integration notes all encode assumptions about that release.
