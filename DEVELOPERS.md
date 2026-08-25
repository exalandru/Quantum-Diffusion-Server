# Developing Quantum Diffusion Server

Technical reference for building QDS from source, its full HTTP/MCP surface,
and the repository layout. For the product pitch, features and installation
as an end user, see [README.md](README.md).

## Common commands

```sh
make install
make test
make lint
make dev-server
make dev-dashboard
make build
make clean
```

Run `make help` for the complete list.

CI runs the same gates (`.github/workflows/ci.yml`) on every push to `main`
and every pull request.

`make build` does everything: the wheel, the app, and the disk image.

```sh
make install     # dependencies (needs Node, Swift and uv on the PATH)
make build-app   # → dist/app/QDS.app
make build-dmg   # → dist/app/QDS-<version>.dmg
```

The DMG holds the app beside a symlink to `/Applications`; it is compressed
and read-only, and takes its name and volume label from the bundle it
packages rather than from the source tree — a stale `dist/app/QDS.app` cannot
ship under a fresh version number.

The bundle is ad-hoc signed and not notarized, so after moving it between
machines macOS may need convincing: `xattr -d com.apple.quarantine "QDS.app"`
(the quotes matter, the name contains spaces). The same applies to the disk
image itself once it has been downloaded.

## Running the server standalone

Prefer the command line, or running on a headless Mac? The server works on
its own — the menubar app is a control panel over it, not a wrapper around
it:

```sh
make dev-server
# or, from a built wheel:
uv tool install ./dist/server/qds-<version>-py3-none-any.whl
qds serve      # then open http://127.0.0.1:8765/dashboard
```

## Repository layout

Three pieces, each owning one thing:

- [`src/server`](src/server/README.md): the Python API server exposing
  mflux through an OpenAI Images-compatible API. It also owns the control
  plane (`/admin`), the MCP surface (`/mcp`) and serves the dashboard.
- `src/dashboard`: the React interface, built into the Python package and
  served at `/dashboard`. It is a pure HTTP client — same origin, no
  privileges of its own.
- `src/menubar`: the Swift menubar app. It does the two things a web page
  cannot: install the server, and start and stop it.

```text
src/
├── server/       Python package, tests, and the built dashboard it ships
├── dashboard/    React sources for that interface
└── menubar/      Swift menubar app (SwiftPM)
dist/             the wheel, the source archive, QDS.app and its DMG
```

`dist/` is generated and ignored by Git.

The menubar app starts the server and gets out of the way; the server owns
its own configuration, its jobs and its interface. That split is why the same
dashboard works on a headless Mac with no app at all.

## API documentation

The server publishes its OpenAPI schema at `/openapi.json` and renders no
documentation page: `docs_url` and `redoc_url` are both `None`.

That is a deliberate constraint rather than an omission. FastAPI's `/docs` and
`/redoc` load Swagger UI and ReDoc from a CDN, and the server sends
`script-src 'self'` (see `CSP` in `qds/app.py`), so the browser refuses the
bundle and the page renders blank while still answering 200 — a broken page that
looks healthy to anything checking status codes. The alternatives were serving
those bundles from this package, which means carrying ~2.7 MB of third-party
browser code through the build, or opening a CSP hole for two pages. Neither is
worth a renderer for a schema that clients read directly.

To browse the surface interactively, point a local viewer at the schema, or
paste it into any OpenAPI client:

```sh
curl http://127.0.0.1:8765/openapi.json
```

`tests/test_no_docs_ui.py` holds the decision in place: it fails if a
documentation page reappears, and checks the schema is still served.

## API surface beyond OpenAI's shape

Beyond the OpenAI fields (`prompt`, `model`, `n`, `size`, `response_format`),
requests accept a few extras that the OpenAI SDKs pass through harmlessly:
`steps`, `seed`, `guidance`, `negative_prompt`, `strength` on edits, and
`response_format: "raw"` for the PNG bytes directly (`n=1` only). Parameters
with no local equivalent (`quality`, `style`, `user`…) are accepted and
ignored rather than rejected.

A few endpoints are additions rather than OpenAI shapes: `/v1/progress`
(Server-Sent Events), `/v1/cancel`, `/v1/unload`, `/v1/capabilities`, and a
`/health` that stays public even with an API key set.

All of it, plus the complete configuration reference and the full model
capability matrix, is documented in
[`src/server/README.md`](src/server/README.md).

## MCP

The server speaks MCP on the same port as the API (`/mcp`), or via the `qds
mcp` stdio bridge for clients that only launch subprocesses — see the
[README](README.md#using-it-via-mcp) for the two connection configs.

Tools: `generate_image`, `refine_image`, `vary_image`, `upscale_image`,
`wait_for_generation`, `cancel_generation`, `list_models`, `list_sessions`,
`open_session`; `delete_image`/`delete_group` only if `mcp.allow_destructive`
is set (off by default). Resources: `qds://models`, `qds://upscalers`,
`qds://sessions`, `qds://images/{filename}`.

Each MCP conversation lazily creates its own playground session on first
generation — results are visible and durable in the browser playground too.
Password-protected playground sessions are unreachable over MCP by design
(no unlock tool).

Worth knowing before building on it: **one generation at a time**. Concurrent
requests are queued rather than refused, and `n > 1` produces images
sequentially — deliberate, since on unified memory two live models saturate
the machine.

Full config reference (`mcp` section of `server-config.json`) and rationale
for the design in [`src/server/README.md#mcp`](src/server/README.md#mcp).
