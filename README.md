# Quantum Diffusion Server

OpenAI Image compatible server for running local Diffusion Models on Apple Silicon. Features a GUI control panel for model management, runtime quantization, pre-quantized variants, local model import, and Hugging Face downloads.

![QDS Screenshot](assets/app-01.jpg)


## What is QDS?

Quantum Diffusion Server turns a Mac into an image generation endpoint. It runs
FLUX, Qwen-Image, Z-Image, ERNIE-Image, FIBO and Ideogram locally on Apple
Silicon, and exposes them through the **same HTTP API as OpenAI's Images
endpoint** - so anything that already speaks to `images.generate` (Misty Studio,
Open WebUI, the `openai` SDK, your own script) can be pointed at
`http://127.0.0.1:8765/v1` and keep working. No account, no per-image cost, no
prompt or image leaving the machine.

A small **menubar app** installs the server on first launch and starts and
stops it; everything you interact with is a **web dashboard the server serves
itself**, at `http://127.0.0.1:8765/dashboard`. Nothing to install beforehand,
no terminal required, no global Python or Homebrew involvement — and because
the interface belongs to the server rather than to the app, a headless Mac gets
exactly the same one.

The same server also serves a **playground** at
`http://127.0.0.1:8765/playground`: a prompt box, a model picker, an optional
reference image, and a history of everything generated. Generations are recorded
server-side, so closing the tab mid-image loses neither the run nor the result.

What makes it different from calling a command-line tool per image: **the model
stays loaded in memory between requests**. A generation that costs 34s as a
fresh subprocess costs 18.5s here, and the saving repeats on every image. Since
that resident model also confiscates unified memory, an idle timer can hand it
back automatically when you stop generating - a knob rather than a compromise.

**Requirements:** an Apple Silicon Mac (M1 or later) on macOS 14 or newer. Count
roughly 1.1 GB for the Python runtime plus whatever the models weigh -
several GB each, tens of GB for the largest, all shown per model in the app
before you download anything. Memory is the real constraint rather than disk:
the two models enabled by default are small and quantized to 8 bits, while the
20B and 32B entries in the catalogue assume a large-memory machine - FLUX.2-dev
holds ~58 GB resident while it generates.


## Features

- **OpenAI-Images-compatible API** - `/v1/images/generations` and
  `/v1/images/edits`, standard request and error shapes, `/v1/models`, and
  interactive documentation at `/docs`. Existing clients need no adapter.
- **One warm model, ~1.8× faster per image** - weights are loaded once and
  reused, instead of paying a Python start-up and a weight rematerialization for
  every generation.
- **Ten models in one catalogue**, each with its own defaults taken from its own
  model card rather than a blanket setting. Switching model is a request field.
- **Weights on demand** - the Models tab lists the whole catalogue with its
  licence, local availability and disk usage. Install from Hugging Face, locate
  an existing compatible copy, or import a local model without copying its
  weights. Model management works with the generation server stopped.
- **Runtime and pre-quantization** - supported models can quantize while loading
  without writing another copy, or create reusable pre-quantized variants on
  disk. Large models can be converted component by component to keep peak memory
  bounded, and saved variants can be activated independently of the source model.
- **Gives the memory back** - `idle_unload_s` releases the model after a chosen
  period of inactivity, so a chat model or a build can have the RAM when you are
  not generating. Or press *Free memory*, from the dashboard or the menu bar.
- **Live progress, and a stop button** - step-by-step progress over
  Server-Sent Events, surfaced in the dashboard with elapsed time and a cancel
  that leaves the server usable.
- **Configuration as a form** - port, API key, CORS, timeouts, image retention,
  default model, Hugging Face access, and independent storage locations for HF
  source weights and QDS pre-quantized artifacts. Per-model controls live in the
  Models tab and are driven by the capabilities reported by the backend.
- **Editing and img2img** - instruction editing on the models that have an edit
  variant, image-to-image on the others, over the standard edits endpoint.
- **Readable logs** - structured events and raw output side by side, filterable
  by level, in the dashboard's Logs tab.
- **MCP, for a local chat model** - `/mcp` on the same port, or `qds mcp` for
  clients that speak only stdio. Generate, refine, vary and upscale from a
  conversation, with the results landing in the playground where you can carry
  on with them by hand.
- **Local by default** - binds to `127.0.0.1`; opening it to the network
  requires setting an API key, which the server enforces rather than suggests.


## Getting started

There is no signed release: build the app once, from a clone.

```sh
make install     # dependencies (needs Node, Swift and uv on the PATH)
make build-app   # → dist/app/QDS.app
make build-dmg   # → dist/app/QDS-<version>.dmg, to hand to someone else
```

`make build` does all of it: the wheel, the app, and the image.

The image holds the app beside a symlink to `/Applications`, so installing is
the usual drag. It is compressed and read-only, and it takes its name and
volume label from the bundle it packages rather than from the source tree —
a stale `dist/app/QDS.app` cannot ship under a fresh version number.

The bundle is ad-hoc signed and not notarized, so after moving it between
machines macOS may need convincing: `xattr -d com.apple.quarantine "QDS.app"`
(the quotes matter, the name contains spaces). The same applies to the image
itself once it has been downloaded.

QDS lives in the menu bar: it has no window and no Dock icon. The interface is
a **web dashboard the server itself serves**, at
`http://127.0.0.1:8765/dashboard` — *Open Dashboard* in the menu opens it.

Then:

1. **First launch** installs the server from the wheel the app carries. Only
   the Python runtime is downloaded, once.
2. **Models tab** - pick a model and press *Install* to fetch its weights.
   `z-image-turbo` and `ernie-image-turbo` are enabled out of the box: both are
   Apache-2.0 and ungated, so a fresh install generates with no HuggingFace
   token, no access request and no licence to accept.
3. The server answers within a second; the first generation is what actually
   loads the weights.
4. Point your client at `http://127.0.0.1:8765/v1`. Any API key value works
   unless you set one in the Configuration tab.

```sh
curl http://127.0.0.1:8765/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox in the snow", "size": "1280x720"}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
result = client.images.generate(prompt="a red fox in the snow", size="1280x720")
print(result.data[0].url)
```

Prefer the command line, or running on a headless Mac? The server works on its
own: `make dev-server`, or the `qds` console script from the wheel (`qds serve`).
The app is a control panel over it, not a wrapper around it.


## Models

![QDS Screenshot 2](assets/app-02.jpg)

| model | licence | enabled by default | notes |
|---|---|---|---|
| `z-image-turbo` | Apache-2.0 | ✅ | the default; 9 steps, fast |
| `ernie-image-turbo` | Apache-2.0 | ✅ | 8 steps, fast |
| `z-image` | Apache-2.0 | - | 20 steps, adjustable guidance |
| `ernie-image` | Apache-2.0 | - | 20 steps |
| `qwen-image-flash` | NVIDIA Open Model | - | Qwen-Image distilled to 4 steps |
| `qwen-image-2512` | Apache-2.0 | - | 20B, strong text rendering, optional editing |
| `flux2-klein` | FLUX Non-Commercial 🔒 | - | 9B distilled, 4 steps, instruction editing |
| `flux2-dev` | FLUX Non-Commercial 🔒 | - | 32B; raw HF source with bounded-memory pre-quantized variants |
| `anima-turbo` | CircleStone Non-Commercial | - | 2B anime-oriented, distilled to 10 steps; the variant its author recommends |
| `anima` | CircleStone Non-Commercial | - | the undistilled Aesthetic fine-tune, 30 steps |
| `krea-2-turbo` | Krea 2 Community 🔒 | - | 12B distilled, 8 steps, adjustable guidance |
| `fibo-lite` | CC-BY-NC-4.0 🔒 | - | prompts are structured JSON |
| `fibo` | CC-BY-NC-4.0 🔒 | - | prompts are structured JSON |
| `ideogram-4` | Ideogram Non-Commercial 🔒 | - | sampler presets instead of a step count |

🔒 = gated on HuggingFace: you need a token whose access has been approved. The
token is configured in the Configuration tab, and gated models ship disabled
because requesting that access - and accepting a non-commercial licence - is
your decision, not the app's. Enable or disable models directly from the Models
tab.

Built-in models can be installed from Hugging Face or pointed at an existing
compatible local directory with *Locate…*. The Models tab can also register
additional local models with a stable API name. QDS keeps source weights and
pre-quantized variants separate: creating or activating a saved variant does not
replace the original model source.

The full capability matrix - steps, guidance, negative prompts, prompt format,
img2img and editing per model - is in
[`src/server/README.md`](src/server/README.md#models), and the running server
reports its effective version at `GET /v1/capabilities`.


## Using it from your own code

Beyond the OpenAI fields (`prompt`, `model`, `n`, `size`, `response_format`),
requests accept a few extras that the OpenAI SDKs pass through harmlessly:
`steps`, `seed`, `guidance`, `negative_prompt`, `strength` on edits, and
`response_format: "raw"` for the PNG bytes. Parameters with no local equivalent
(`quality`, `style`, `user`…) are accepted and ignored rather than rejected.

A few endpoints are additions rather than OpenAI shapes: `/v1/progress` for
Server-Sent Events, `/v1/cancel`, `/v1/unload`, `/v1/capabilities`, and a
`/health` that stays public even with an API key set. All of it, plus the
complete configuration reference, is documented in
[`src/server/README.md`](src/server/README.md).

### From a chat with a local model

The server speaks MCP as well. Point an HTTP-capable client at
`http://127.0.0.1:8765/mcp`, or, for a client that only launches processes:

```json
{
  "mcpServers": {
    "quantum-diffusion": { "command": "qds", "args": ["mcp"] }
  }
}
```

The model gets `generate_image`, `refine_image`, `vary_image`, `upscale_image`
and a few others; it sees a thumbnail of what it made, and the full-resolution
file lands in a playground session you can open in the browser. The tools and
their bounds are documented in
[`src/server/README.md`](src/server/README.md#mcp).

Worth knowing before you build on it: **one generation at a time**. Concurrent
requests are queued rather than refused, and `n > 1` produces images
sequentially. That is deliberate - on unified memory two live models saturate
the machine.


## Licence

The code in this repository is MIT licensed - see [LICENSE](LICENSE).

**Model weights are not.** Each entry in the catalogue carries its own licence,
listed in the table above and enforced by nobody but you: the FLUX and Ideogram
models are non-commercial, FIBO is CC-BY-NC-4.0, and five of the ten are gated
behind an access request. Only the Apache-2.0 models are enabled out of the box.

QDS is a client of [mflux](https://github.com/filipstrand/mflux) (MIT), which
does the actual MLX inference.


## Development

### Common commands

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


### Repository layout

Three pieces, each owning one thing:

- [`src/server`](src/server/README.md): the Python API server exposing mflux
  through an OpenAI Images-compatible API. It also owns the control plane
  (`/admin`), the MCP surface (`/mcp`) and serves the dashboard.
- `src/dashboard`: the React interface, built into the Python package and
  served at `/dashboard`. It is a pure HTTP client — same origin, no privileges
  of its own.
- `src/menubar`: the Swift menubar app. It does the two things a web page
  cannot: install the server, and start and stop it.

```text
src/
├── server/       Python package, tests, and the built dashboard it ships
├── dashboard/    React sources for that interface
└── menubar/      Swift menubar app (SwiftPM)
docs/adr/         architecture decision records
dist/             the wheel, the source archive, QDS.app and its DMG
```

`dist/` is generated and ignored by Git.

The menubar app starts the server and gets out of the way; the server owns its
own configuration, its jobs and its interface. That split is why the same
dashboard works on a headless Mac with no app at all:

```sh
uv tool install ./dist/server/qds-2.0.0-py3-none-any.whl
qds serve      # then open http://127.0.0.1:8765/dashboard
```
