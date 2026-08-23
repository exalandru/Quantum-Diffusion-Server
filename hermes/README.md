# Hermes ↔ QDS — `image_gen` provider

Local backend for Hermes' `image_generate` tool. When the Hermes model asks for
an image, this QDS server generates it — mflux on Apple Silicon, no API key, no
outbound network calls.

Code: `hermes/image_gen/qds/` (source of truth; Hermes loads it via a symlink).

## Installation

```bash
# 1. link the plugin into Hermes' user plugins directory
mkdir -p ~/.hermes/plugins/image_gen
ln -sfn ./hermes/image_gen/qds ~/.hermes/plugins/image_gen/qds

# 2. enable the plugin (opt-in)
hermes plugins enable image_gen/qds

# 3. make it the active backend for image_generate
hermes config set image_gen.provider qds
hermes config set image_gen.model anima-turbo   # optional, else the server default
```

`hermes plugins list` should show `qds | enabled | user`.
`hermes tools` → Image Generation lists "QDS (local)" and its installed models.

The QDS server must be running (`http://127.0.0.1:8765` by default):

```bash
qds serve            # or the repo's launch command
curl -s localhost:8765/health
```

## Usage

- **text-to-image**: `image_generate(prompt=…, aspect_ratio=…)`
  → `POST /v1/images/generations`.
- **image-to-image / edit**: `image_generate(prompt=…, image_url=…)`
  → `POST /v1/images/edits` (a single source image; extra references are
  ignored, the QDS endpoint only takes one). The server picks `edit` vs
  `img2img` based on the model — there is no `strength` knob on the Hermes
  side, by design.
- Sizes: `landscape` → `1344x768`, `square` → `1024x1024`,
  `portrait` → `768x1344` (mflux truncates to a multiple of 16).
- Output: always requested as `b64_json` and written to
  `$HERMES_HOME/cache/images/qds_<model>_<timestamp>_<uuid>.png`.
  Server URLs are never used: their store has a TTL (3600 s), a local file
  does not expire.

### Model selection (first hit wins)

1. `QDS_IMAGE_MODEL` (env var — escape hatch for scripts/tests);
2. the `model` kwarg forwarded by the tool (i.e. the picker's `image_gen.model`);
3. `image_gen.qds.model` in `~/.hermes/config.yaml`;
4. `image_gen.model` in `~/.hermes/config.yaml`;
5. the `default_model` returned by the server.

Config-derived candidates are only honoured when the server actually has that
model: `image_gen.model` is shared with every other backend and routinely holds
a foreign id (`gpt-image-2-medium`). `QDS_IMAGE_MODEL` is passed through as-is —
the server is the one that rejects it, with its own message.

### Server address

`QDS_BASE_URL` (default `http://127.0.0.1:8765`). Example:

```bash
QDS_BASE_URL=http://127.0.0.1:8770 hermes
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `image_gen.provider='qds' is set but no plugin registered that name` | plugin not enabled → `hermes plugins enable image_gen/qds`, or broken symlink → redo the `ln -sfn`. |
| `QDS is not reachable at http://…` | server stopped, or wrong port → check `curl $QDS_BASE_URL/health`. |
| `QDS returned HTTP 400: Unknown model: '…'` | model not installed → `curl $QDS_BASE_URL/v1/capabilities` for the real list. |
| Models missing from `hermes tools` | server was down at scan time (dynamic catalog); restart the server then reopen `hermes tools`. |
| First generation very slow | weights loading (krea-2-turbo ≈ 20 GB, qwen-image-2512 ≈ 55 GB). `anima-turbo` (10 steps) is the fast choice. |
| Generation cut off on the client side | provider HTTP timeout: 1800 s. Beyond that, look at the server (`/v1/progress`). |

The provider never raises an exception into Hermes: every failure becomes a
`{"success": false, "error_type": …}` — unreachable server, unknown model, empty
prompt, unreadable source image.
