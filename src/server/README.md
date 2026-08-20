# qds — Quantum Diffusion Server

A local server exposing [mflux](https://github.com/filipstrand/mflux) — the MLX implementation of FLUX, Qwen-Image and Z-Image for Apple Silicon — behind an **OpenAI-Images-compatible API**. Enough to point any OpenAI-speaking frontend (Misty Studio, Open WebUI, the `openai` SDK…) at diffusion models running locally.

The model is **loaded once and kept in memory** between requests, instead of being reloaded by a fresh process for every image.

**Requirements: an Apple Silicon Mac (M1 or later) on macOS 14 or newer.** The inference stack is MLX, which has no Linux or Windows backend here; there is no headless build for those platforms and none is planned.

## Installation

From a checkout, for development:

```sh
uv sync
uv run qds serve
```

Or as a standalone tool, from a wheel built by `make build-server`:

```sh
uv tool install ./dist/server/qds-<version>-py3-none-any.whl
qds serve
```

mflux is a project dependency — no need for a separate `uv tool install mflux`. Weights already present in the HuggingFace cache are reused as-is.

Nothing else is required to generate: the two models enabled by default are ungated and Apache-2.0. Five of the ten catalogue entries *are* gated (`black-forest-labs/*`, `briaai/*`, `ideogram-ai/*`) and need a HuggingFace token with approved access (`hf auth login`) — they ship disabled for that reason. `flux2-dev` additionally requires a conversion step, see [FLUX.2-dev](#flux2-dev--32b-in-8-bit).

## The `qds` command

One command, with subcommands:

| Command | What it does |
|---|---|
| `qds serve` | run the server |
| `qds fetch <key>` | download a model's weights ahead of time |
| `qds fetch --status` | print the catalogue with cache state, as JSON |
| `qds prequantize` | convert a model into a saved, already-quantized artifact |
| `qds import …` | register a local model directory, without copying it |
| `qds status` | print a running server's `/health` document |

Every subcommand carries its own `--help`.

## Running

```sh
qds serve
```

The server listens on `http://127.0.0.1:8765`. Interactive docs at `/docs`.

```sh
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/v1/models

curl http://127.0.0.1:8765/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox in the snow", "size": "1024x1024"}'
```

With the official SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
result = client.images.generate(prompt="a red fox in the snow", size="1024x1024")
print(result.data[0].url)
```

## Performance

Real measurements on an M3 Ultra / 103 GB, weights already in the HuggingFace cache:

| scenario | subprocess (old) | server, warm model |
|---|---|---|
| `flux2-klein`, 1024×1024, 4 steps | 34.3s | **18.5s** |
| `z-image-turbo`, 1280×720, 9 steps | — | 45s on the 1st call, then **32s** |

That is roughly **1.8×**, and the gain repeats on every image. (Measured at 8 bits; the shipped config now asks for 4, which lowers memory further at some cost in fine detail.)

It is worth knowing where it comes from, because it is not what one would assume: mflux loads its weights **lazily / mmap'd**, so a 9B model is "ready" in half a second and the real cost is paid during the first generation. What the server saves is the startup of a full Python process (importing torch, transformers, mlx) and the rematerialization of the weights — not a multi-minute load. On a model where inference dominates (`z-image-turbo` at ~3.5s per step), the relative gain is therefore smaller.

Corollary: memory is the real limiting factor. Running another mflux process alongside the server evicts its pages and triples the time of the next generation.

### Giving the memory back — `idle_unload_s`

Keeping the model warm is what buys the numbers above, but on unified memory it also confiscates the machine: with a text LLM running alongside, one image makes the chat unusable until someone frees the memory by hand.

`idle_unload_s` releases the model on its own after that many seconds without a generation — `0` as soon as the request ends, `null` (the default) never. The countdown is armed **per request, not per image**, so a burst of `n=3` reloads once and releases once, at the end; and a request arriving before the deadline cancels it, so the model stays warm through a working session.

What it costs is the figure in the table above, read the other way: ~16 s of reload on `flux2-klein`, more on a larger model. Worth it when something else needs the memory, wasteful otherwise. The release is logged with the memory before and after, and `/health` reports the policy so that "no warm model" cannot be mistaken for a fault.

## Models

| key | repo | licence | gated | on | steps | guidance | prompt | negative | img2img | editing |
|---|---|---|---|---|---|---|---|---|---|---|
| `z-image-turbo` *(default)* | `mlx-community/Z-Image-Turbo-bf16` | Apache-2.0 | — | ✅ | 9 | forced to 0 | text | ✅ | ✅ | ❌ |
| `ernie-image-turbo` | `baidu/ERNIE-Image-Turbo` | Apache-2.0 | — | ✅ | 8 | fixed at 1.0 | text | ❌ | ✅ | ❌ |
| `z-image` | `mlx-community/Z-Image-bf16` | Apache-2.0 | — | ❌ | 50 | 4.0 | text | ✅ | ✅ | ❌ |
| `ernie-image` | `baidu/ERNIE-Image` | Apache-2.0 | — | ❌ | 50 | 4.0 | text | ✅ | ✅ | ❌ |
| `qwen-image-2512` | `Qwen/Qwen-Image-2512` | Apache-2.0 | — | ❌ | 50 | 4.0 | text | ✅ | ✅ | opt-in |
| `flux2-klein` | `black-forest-labs/FLUX.2-klein-9B` | **FLUX Non-Commercial** | 🔒 | ❌ | 4 | fixed at 1.0 | text | ❌ | ✅ | ✅ |
| `flux2-dev` | local 8-bit artifact | **FLUX Non-Commercial** | 🔒 | ❌ | 50 | 4.0 | text | ❌ | ✅ | ❌ |
| `fibo-lite` | `briaai/Fibo-lite` | **CC-BY-NC-4.0** | 🔒 | ❌ | 8 | fixed at 1.0 | **json** | ❌ | ✅ | ❌ |
| `fibo` | `briaai/FIBO` | **CC-BY-NC-4.0** | 🔒 | ❌ | 50 | 5.0 | **json** | ✅ | ✅ | ❌ |
| `ideogram-4` | `ideogram-ai/ideogram-4-fp8` | **Ideogram 4 Non-Commercial** | 🔒 | ❌ | 20 *(preset)* | preset | text + json | ❌ | ❌ | ❌ |

**The two models on by default are Apache-2.0 and ungated**, which is the point: a fresh install generates with no HuggingFace token, no access request, and no licence to accept. Everything else ships off — the gated and non-commercial ones because obtaining access is your decision, the rest because 50 steps is not a good first impression. Turn any of them on in the config, or in the app's Configuration tab.

Useful details:

- **Prompt format matters on three models.** The `prompt` column says what each one accepts. `fibo` and `fibo-lite` accept **only** a structured JSON caption: their prompt encoder opens with a bare `json.loads(prompt)` whose result is discarded, so plain text raises. The server refuses it up front — `400 prompt_must_be_json`, before any weights load — rather than let you discover it after several GB. `ideogram-4` accepts both, and prefers JSON: plain text works but, per Bria's and Ideogram's own docs, underperforms.

```bash
# fibo-lite: the prompt is a JSON object, passed as the prompt string
curl http://127.0.0.1:8765/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "fibo-lite", "prompt": "{\"high_level_description\": \"a red fox in the snow\"}"}'
```

  The full schemas live with the models: [FIBO's prompting guide](https://huggingface.co/briaai/FIBO) and [Ideogram's](https://github.com/ideogram-oss/ideogram-4/blob/main/docs/prompting.md). mflux ships `mflux-inspire-fibo` and `mflux-refine-fibo` to build those captions with Bria's VLM; the server does not call them — that would mean a second model resident alongside the first.
- **`ideogram-4` takes its step count from a sampler preset**, not from a number: `V4_DEFAULT_20` (20 steps), `V4_QUALITY_48`, `V4_TURBO_12`. Each preset carries a per-step guidance schedule and a noise schedule, so `guidance` is refused and `steps` is best left alone — passing it replaces the schedule with a constant. Pick one with `models.ideogram-4.preset`. Its dimensions are also capped at 2048, checked before loading.
- **`flux2-dev` requires a conversion step** and is not usable as-is: 32 billion parameters, a gated repo, and code mflux 0.18.0 does not provide. See [FLUX.2-dev](#flux2-dev--32b-in-8-bit). Expect ~113 GB of one-time download, a ~58 GB local artifact, and ~58 GB resident during generation.
- **The steps and guidance defaults come from each model's own card**, not from mflux's blanket defaults — 50/4.0 for `qwen-image-2512` where mflux would say 20/3.5, 5.0 for base FIBO where its signature says 4.0, and 8 steps for the ERNIE and FIBO turbos. The distilled models reject `guidance` and `negative_prompt` outright: at guidance 1.0 the negative pass is skipped, so accepting the parameter would mean accepting one that does nothing.
- **Editing is off in the default set.** `flux2-klein` is the only model with an instruction-editing variant sharing its weights, and it ships disabled. `/v1/images/edits` still works on the enabled models, as **img2img** — a different mechanic, so the result is a variation on your image rather than an edit of it. Enable `flux2-klein` for real editing.
- **Dimensions are truncated down to a multiple of 16** — an mflux constraint. `1920x1080` becomes `1920x1072`. The server applies the rounding itself and reports the effective size in the response's `mflux.size` field.

The table above is the catalogue — what each model is worth on its own. The shipped `server-config.json` then applies three policies on top: `default_size: "1280x720"` everywhere, `default_quantize: 4`, and the enabled set described above.

`GET /v1/capabilities` returns the effective values, config applied.

### Downloading weights ahead of time

The first generation on a model that is not in the HuggingFace cache pays the whole download — tens of gigabytes, inside an HTTP request that looks hung. `qds fetch` makes it an explicit step:

```sh
qds fetch --status          # what is cached, and how much room it takes
qds fetch ernie-image-turbo # download it now
```

`--status` prints one JSON object per catalogue entry (`cached`, `size_gb`, `gated`, `license`, `enabled`), which is what the app's Models tab shows next to an **Install** button — with the server stopped too, since it does not go through the HTTP API.

The download works by loading the model and exiting. That is deliberate: the download patterns live in each family's `WeightDefinition` inside mflux, and copying them here would be one more table to keep in step. Loading also proves the thing works — gated access granted, quantization applied — rather than only that files landed on disk. What it does not buy is the quantization: that is not persisted, so the first real generation pays it again. It is the download that was worth moving.

## Endpoints

| Route | Method | Note |
|---|---|---|
| `/v1/images/generations` | POST | text → image |
| `/v1/images/edits` | POST | multipart; instruction editing or img2img |
| `/v1/models` | GET | standard OpenAI shape |
| `/v1/models/{id}` | GET | + the model's capabilities under the `mflux` key |
| `/v1/capabilities` | GET | extension: the whole catalogue |
| `/v1/progress` | GET | extension: progress as Server-Sent Events |
| `/v1/cancel` | POST | extension: interrupts the running generation |
| `/v1/unload` | POST | extension: frees resident weights without restarting |
| `/health` | GET | public even with an API key set; reports the warm model and MLX memory |
| `/images/{name}.png` | GET | images served for `response_format="url"` |
| `/playground` | GET | the browser playground page (see below) |
| `/playground/api/…` | GET/POST/DELETE | its sessions, generations and cancellation |
| `/playground/api/preview` | GET | the running playground generation's latest partially-denoised frame (JPEG); 404 when there is none |
| `/playground/images/{name}.png` | GET | images owned by a playground session; **not** TTL-purged |

### Following, cancelling, freeing

`GET /v1/progress` streams a snapshot on every state change:

```sh
curl -N http://127.0.0.1:8765/v1/progress
# data: {"state":"loading","model":"z-image-turbo",...}
# data: {"state":"generating","model":"z-image-turbo","step":3,"total":9,"elapsed_s":4.2,...}
# data: {"state":"idle",...}
```

`state` is one of `idle`, `loading` or `generating`. The distinction matters: loading a model takes anywhere from a few seconds to several minutes depending on its size, and has no step-based progress. A `: ping` comment is emitted every 15s when nothing moves, so disconnects get noticed.

`POST /v1/cancel` interrupts the running generation. MLX cannot be cancelled from outside: the stop goes through the progress callback, so it takes effect at the next denoising step — not instantly. The in-flight request ends as a **499 `generation_stopped`** and the server stays usable, model still warm.

`POST /v1/unload` frees the weights without restarting — handy for giving a large model's tens of GB back to the machine. The route takes the engine lock: if a generation is running, it waits for it to finish rather than breaking it.

```sh
curl -s -X POST http://127.0.0.1:8765/v1/unload
# {"loaded_model":null,"memory":{"active_gb":0.0,"peak_gb":35.16,"cache_gb":0.0}}
```

### Parameters

OpenAI standards: `prompt`, `model`, `n`, `size`, `response_format`. Parameters with no equivalent (`quality`, `style`, `user`, `background`, `output_format`) are accepted and ignored rather than rejected.

Extensions — additional fields that the OpenAI SDKs ignore:

| field | effect |
|---|---|
| `steps` | number of denoising steps |
| `seed` | seed; with `n > 1`, incremented per image |
| `guidance` | CFG scale, rejected on distilled models |
| `negative_prompt` | rejected on `flux2-klein` |
| `strength` | *(edits only)* forces img2img instead of editing |
| `response_format: "raw"` | returns the PNG bytes directly, `n=1` only |

`size` accepts `"auto"` (the model's default size) and the `"WxH"` form.

### `/v1/images/edits`: editing or img2img?

Two genuinely different mechanics, and the server picks based on what you send:

- **`strength` provided** → img2img: the image is encoded then noised, and the loop starts at an intermediate step. The result is a variation on the source image.
- **`strength` absent, model has an edit variant** → instruction editing: the loop starts from pure noise and the image serves as conditioning tokens. This is what you want for "add a hat to this person".
- **`strength` absent, no edit variant** → img2img with `strength = 0.4`.

OpenAI's `mask` parameter is rejected with a 400: no model in the catalogue does inpainting.

### The playground — `/playground`

A prompt-driven studio the server serves itself, in the dashboard's design and
at the dashboard's auth level (the data-plane credential, plus a same-origin
check because it is a browser control surface).

What separates it from `/v1/images/generations` is durability. A submission is
accepted with `202` and a record, then runs on a single in-process FIFO worker
that calls the same engine — so generations still serialize, and an `n=3`
request still runs three images one after another:

- the record and its images live in `playground_store` — by default a
  `playground/` directory beside `server-config.json` — outside `image_store`,
  so no TTL purge can reach them. They are deleted with their session, and only
  then;
- closing the browser loses nothing: reopening `/playground` reconstructs the
  sessions, the transcript and the live status from the server alone;
- every accepted generation reaches a terminal status. One interrupted by a
  restart is marked `failed` with `Interrupted by server restart` at the next
  startup, rather than staying `running` forever.

Cancellation has three cases, because the engine can only be interrupted while
it is denoising:

- **queued** → cancelled by its record; it is never handed to the engine;
- **running, mid-denoise** → the engine's global stop, the same mechanism
  `/v1/cancel` exposes, taking effect at the next step;
- **running, but loading weights, waiting on the engine lock, or between the
  images of an `n>1` run** → the engine refuses the request there, so the runner
  holds it and applies it at the next image boundary. The image already being
  computed is kept; the record ends `cancelled` with the images it produced.

It is not per-job cancellation: the engine's stop is global, so with an external
`/v1` client holding the engine it is that request which stops.

While a generation runs, the page shows the image being denoised: every second
step the engine decodes the current latents into a small JPEG, keeps it in one
in-memory slot, and bumps `preview_seq` in the `/v1/progress` snapshot; the page
fetches the new frame from `GET /playground/api/preview` and fades it in over the
previous one. A slow accent-tinted sheen crosses the box the whole time — that,
not the frames, is what says "still working" while a partial image sits
unchanged for several seconds. One slot is enough because the engine runs one
generation at a time, and the bytes stay off the SSE stream, which the dashboard
page shares. A family whose latent layout is not mapped, or a decode that fails,
loses its previews for that run — never its generation.

## Configuration

`server-config.json` (JSON). Every key in the `server` section can be overridden by `QDS_SERVER_<KEY>` in upper case — `QDS_SERVER_PORT=9000`, `QDS_SERVER_API_KEY=…`, `QDS_SERVER_CORS_ORIGINS=https://a.example,https://b.example`. `QDS_SERVER_CONFIG` points at a different config file.

### The `server` section

| key | default | role |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8765` | binding |
| `api_key` | `null` | when set, `Authorization: Bearer` is required. **Mandatory as soon as `host` is not local** |
| `cors_origins` | `["*"]` | allowed origins |
| `max_n` | `4` | bounds OpenAI's `n` (generations are sequential) |
| `request_timeout_s` | `900` | interrupts the denoising loop past this point |
| `image_store` / `image_ttl_s` | `images` / `3600` | directory and lifetime of images served as `url` |
| `playground_store` | `null` | directory holding the playground's sessions database and its images. `null` = a `playground/` directory **beside this configuration file**, where the rest of the installation's state lives — not beside the working directory, which is read-only when the server is launched from the app bundle. A relative value set here is resolved against the working directory, like `image_store` |
| `max_upload_mb` | `25` | maximum size of an image sent to `/v1/images/edits` |
| `default_response_format` | `url` | value used when the client sends none. `url` is OpenAI's default: changing it breaks the SDKs |
| `log_level` / `log_file` | `INFO` / `mflux.log` | `log_file: null` disables the file. The path is made absolute and parent directories are created |
| `log_json` | `false` | one line, one JSON object, **on stdout**. See below |
| `progress_log_every` | `1` | a progress log every N steps; `0` to turn it off |
| `shutdown_grace_s` | `10` | bounds the graceful shutdown; without it a SIGTERM mid-generation would wait for `request_timeout_s` |
| `idle_unload_s` | `null` | releases the warm model after that many seconds without a generation. `null` = never, `0` = as soon as the request ends. See below |

### Top-level keys

Two keys sit outside `server`, because they are generation defaults rather than transport settings:

| key | default | role |
|---|---|---|
| `default_model` | `z-image-turbo` | model used when the request names none. Must be enabled |
| `default_quantize` | `null` | config-wide quantization: 3/4/5/6/8, or `0` for none. Skipped on models whose weights already carry their precision |
| `default_size` | `null` | config-wide resolution, `"WxH"`. Applies to every model; `null` leaves each on its catalogue size |

These are the *code* defaults, used when no config file is found. The shipped `server-config.json` is more opinionated: `1280x720`, 4 bits, and only the two fast ungated models enabled.

#### 4 bits, and where it silently does not apply

`default_quantize: 4` halves memory again over 8 bits, at some cost in fine detail and text rendering. It is reversible per model with `models.<key>.quantize`.

It only works from bf16 weights. mflux resolves a conflict in favour of what the file already contains — `QuantizationResolution`, rule `conflict` → action `STORED` — and prints "Model is pre-quantized at 8-bit. Ignoring -q 4". So a repo distributed pre-quantized keeps its own precision whatever you ask. Two entries are in that position and are skipped rather than misreported: `flux2-dev` (our own 8-bit artifact) and `ideogram-4` (fp8, every heavy component marked `skip_quantization`). `/v1/capabilities` reports the effective value plus a `prequantized` flag, so a client never has to guess.

That is also why `qwen-image-2512` and `fibo` point at their **raw bf16 repos** rather than the pre-quantized conversions: only raw weights can honour the setting. The trade is a bigger download and a load-time memory peak at bf16 before quantization — around 55 GB for the 20B Qwen, which fits on a 103 GB machine and not on a 32 GB one. Point `models.qwen-image-2512.model_path` at `mlx-community/Qwen-Image-2512-8bit` to go back.

`default_size` is the single knob for "one resolution everywhere", without repeating the value under each model. It is still overridable per model — see below — which is what you want when one model does not deserve the same area as the others: `flux2-dev` is a 32B, `flux2-klein` a distilled 9B.

Both accept an environment override too: `QDS_SERVER_DEFAULT_MODEL`, `QDS_SERVER_DEFAULT_SIZE`.

### The `models` section — per model

Each entry may contain these keys (all optional, `null` = the catalogue default):

| key | values | role |
|---|---|---|
| `enabled` | `true` / `false` | adds or removes the model from the exposed catalogue |
| `default_size` | `"WxH"` or `null` | default resolution (e.g. `"1024x1024"`). Truncated to a multiple of 16. |
| `default_steps` | integer ≥ 1 or `null` | default number of denoising steps |
| `default_guidance` | float ≥ 0 or `null` | default CFG scale. Rejected on distilled models (`flux2-klein`, `z-image-turbo`). |
| `quantize` | 3/4/5/6/8, 0 or `null` | quantization at load time. `0` = none (bf16). `null` = the catalogue default. |
| `model_path` | path, HF repo or `null` | weight source, in place of the catalogue's. Mostly useful for `flux2-dev`, whose pre-quantized artifact is machine-specific. |
| `enable_edit` | `true` / `false` or `null` | enables the instruction-editing variant. `null` = the catalogue default. |
| `preset` | `"V4_DEFAULT_20"`, `"V4_QUALITY_48"`, `"V4_TURBO_12"` | sampler preset. `ideogram-4` only; rejected on any other model, and it sets the step count too. |

The shipped file, in full:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "api_key": null,
    "cors_origins": [
      "*"
    ],
    "max_n": 4,
    "request_timeout_s": 2400,
    "image_store": "images",
    "image_ttl_s": 3600,
    "max_upload_mb": 25,
    "default_response_format": "url",
    "log_level": "INFO",
    "log_file": "mflux.log",
    "progress_log_every": 1,
    "idle_unload_s": null
  },
  "default_model": "z-image-turbo",
  "default_size": "1280x720",
  "default_quantize": 4,
  "models": {
    "z-image-turbo": {
      "enabled": true,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null
    },
    "ernie-image-turbo": {
      "enabled": true,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null
    },
    "z-image": {
      "enabled": false,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null
    },
    "ernie-image": {
      "enabled": false,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null
    },
    "qwen-image-2512": {
      "enabled": false,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null,
      "enable_edit": false
    },
    "flux2-klein": {
      "enabled": false,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null,
      "enable_edit": true
    },
    "flux2-dev": {
      "enabled": false,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": 8
    },
    "fibo-lite": {
      "enabled": false,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null
    },
    "fibo": {
      "enabled": false,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null
    },
    "ideogram-4": {
      "enabled": false,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null,
      "preset": "V4_DEFAULT_20"
    }
  }
}
```

#### Overriding `steps` and `size` — per request or in the config

`size` now has four levels of precedence, `steps` three. Highest wins:

| priority | `size` | `steps` |
|---|---|---|
| 1 | the request's `size` | the request's `steps` |
| 2 | `models.<key>.default_size` | `models.<key>.default_steps` |
| 3 | the top-level `default_size` | — |
| 4 | the catalogue | the catalogue |

A per-model `default_size` therefore wins over the global one, which is the point: it is the escape hatch for the one model that should not follow the config-wide resolution.

```bash
# Per request — ignores both the config and catalogue defaults
curl http://127.0.0.1:8765/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox in the snow", "size": "1024x1024", "steps": 30}'
```

In the config, to raise the resolution everywhere and change one model's default behaviour:

```json
{
  "default_size": "1536x1536",
  "models": {
    "z-image": {
      "default_size": "1024x1024",
      "default_steps": 30,
      "quantize": 8
    }
  }
}
```

Every model then generates at 1536², except `z-image` which stays at 1024².

#### Quantization — why `z-image` at 8-bit?

The `Z-Image-bf16` and `Z-Image-Turbo-bf16` repos are stored in bf16 (full precision), but the server quantizes them to 8 bits at load time. Reasons:

- **Memory**: a ~9B model in bf16 takes ~18 GB. In int8, ~9 GB. On unified memory (Mac), that is the difference between being able to keep the model warm plus run something else, or not.
- **Amortized cost**: quantization is paid once, on the first load, and then the model stays in memory.
- **Quality**: the visual loss is negligible for generative imagery.

To disable quantization and run in bf16:

```json
{
  "models": {
    "z-image": { "quantize": 0 }
  }
}
```

## FLUX.2-dev — 32B in 8-bit

`flux2-dev` is the only model in the catalogue that requires preparation, for two compounding reasons.

**mflux 0.18.0 cannot load FLUX.2-dev.** Its FLUX.2 family is *klein-only*: `AVAILABLE_MODELS` holds nothing but the klein variants, and `Flux2Initializer` hardwires `Qwen3TextEncoder` and `Flux2KleinWeightDefinition`. But the actual gap is narrow — verified tensor by tensor against the repo's weight indexes:

| component | verdict |
|---|---|
| transformer | the architecture is klein's, just bigger; every key of `transformer/config.json` is a kwarg of `Flux2Transformer`. mflux's weight mapping covers **329 of the 331** tensors. |
| missing weights | `time_guidance_embed.guidance_embedder.linear_{1,2}` — FLUX.2-dev is *guidance-distilled*, klein is not. Two `WeightTarget` entries added. |
| VAE | identical (`AutoencoderKLFlux2`, 32 latent channels): mflux's mapping applies unchanged. |
| scheduler | `ModelConfig`'s `sigma_*` defaults already match the repo's `scheduler_config.json`. |
| text encoder | **the only real gap.** FLUX.2-dev stacks three hidden states of a `Mistral3ForConditionalGeneration` (40 layers, hidden 5120 → 3 × 5120 = `joint_attention_dim` = 15360), where klein uses Qwen3. |

So the encoder is ported to MLX in [`qds/flux2_dev/mistral3.py`](qds/flux2_dev/mistral3.py). The Mistral decoder is standard dense and *the only* structural difference from mflux's `Qwen3VLAttention` is the absence of per-head `q_norm`/`k_norm` — the RMSNorm, the SwiGLU MLP, the RoPE and the GQA helpers are reused as-is. The result is validated against transformers' `MistralModel`: same weights, same hidden states to within 7e-7, across all three padding configurations.

A pleasant consequence: **no Torch at inference**. Everything stays in MLX, the encoder stays warm alongside the transformer, and there is nothing to reload between prompts.

**The repo ships bf16, and it does not fit.** A 64.5 GB transformer plus a 45.8 GB encoder plus the VAE, i.e. ~111 GB of resident weights. At 8 bits we drop to ~58 GB, comfortable on 96 GB of unified memory — but quantizing at load time requires holding the bf16 in memory first. Hence a one-time conversion up front:

```bash
qds prequantize            # → <app data>/cache/artifacts/
```

The repo is *gated*: you need a HuggingFace token that has been granted access (`hf auth login`). Expect ~113 GB of download and ~58 GB written. The script works **one component at a time**, and quantizes the transformer **block by block**: without that the memory peak would reach ~96 GB, against ~66 GB this way. The default order (transformer, encoder, VAE) lets you purge the bf16 from the HF cache between steps — the disk peak falls from ~169 GB to ~97 GB, and the script reminds you what to delete.

To convert a single component, for instance to validate the encoder before committing to the transformer's 64 GB:

```bash
qds prequantize --components text_encoder
```

Reloading needs no configuration at all: mflux detects the `quantization_level` written into the safetensors metadata and quantizes the structure before applying the weights. If the artifact lives elsewhere, set `model_path`:

```json
{
  "models": {
    "flux2-dev": { "model_path": "/Volumes/Assets/models/flux2-dev-mlx-8bit" }
  }
}
```

With no artifact, the server refuses to load and says so with a message that repeats the command — rather than silently falling back to the bf16 repo and attempting a 111 GB quantization.

Two limitations worth knowing:

- **No `negative_prompt`.** FLUX.2-dev is guidance-distilled: guidance is a scalar embedded in the transformer, not CFG. One pass per step (twice as fast as CFG), but no negative prompt is possible. Guidance stays adjustable, defaulting to 4.0.
- **No multi-image editing.** `/v1/images/edits` works in img2img mode, but conditioning on reference-image tokens is not implemented.

Finally, `request_timeout_s` goes to `2400`: 50 steps on a 32B model far exceed the original 900s.

### JSON logs for a supervisor

`"log_json": true` (or `QDS_SERVER_LOG_JSON=1`) switches the logs to JSON Lines, one object per line:

```json
{"ts":"2026-07-27T14:19:02","level":"INFO","logger":"qds","message":"z-image-turbo seed=42 1280x720 — step 3/9","event":"generation_step","fields":{"step":3,"total":9}}
```

`event` is one of `model_loading`, `model_ready`, `model_unload`, `generation_start`, `generation_step`, `generation_done`, `generation_cancel_requested`, or on the conversion side `prequantize_component_start`, `prequantize_progress`, `prequantize_component_done`. The human-readable `message` stays alongside the structured fields.

**In this mode the logs go to stdout, not stderr**, and uvicorn's access log is disabled. The reason is concrete: mflux renders its denoising bar with tqdm (`Config.time_steps`), which writes carriage-return-terminated fragments to stderr **with no newline**. The JSON objects ended up glued to them on the same segment — `\r 0%| | 0/40 [00:00<?, ?it/s]{"ts": …}` — and a consumer splitting on `\n` missed all of them. tqdm offers no environment variable to silence itself, hence the channel split:

- **stdout**: the structured events, one line one valid JSON object, nothing else;
- **stderr**: the human-readable text, the progress bars, and uvicorn's startup logs.

`qds prequantize --json-logs` applies the same configuration to the conversion.

### Access from the local network

```json
{"server": {"host": "0.0.0.0", "api_key": "a-long-random-key"}}
```

The server refuses to start with a non-local host and no API key.

## Known limitations

- **One generation at a time.** That is deliberate: on unified memory, two live models saturate the machine. Concurrent requests are queued, not rejected.
- **`n > 1` is sequential.** The model stays warm, but the images come out one after another.
- **The timeout does not cover weight loading.** It is only checked between denoising steps — the only interruption point mflux offers. So a first call that downloads 30 GB can exceed `request_timeout_s`.
- **No `partial_images` on `/v1`.** OpenAI's streamed partial images are not implemented; a `/v1` client gets step progress over `/v1/progress` and nothing else. The browser playground *does* show the image being denoised — every second step, decoded server-side and fetched from `/playground/api/preview` — and that is playground-only by construction: `/v1` jobs never ask for it.
- **No LoRA, ControlNet, inpainting or upscaling.** mflux offers them; they are not exposed here.

## Development

```sh
uv run pytest        # no weights are loaded
uv run ruff check .
uv run ruff format .
```

The tests cover the registry, OpenAI conformance and the engine (caching, serialization, unloading) with a fake model. **Real inference is verified by hand**:

1. `qds serve`
2. a first generation on `flux2-klein` — time it, loading included;
3. **run it again identically: it must be noticeably faster.** That is the test that validates the cache;
4. `negative_prompt` on `flux2-klein` → an explicit 400;
5. two simultaneous requests → serialized, memory stable in `/health`;
6. switch models → the unload shows up in the logs;
7. a dozen different prompts on `qwen-image` → memory must not drift;
8. point the frontend at it and check no CORS error appears in the browser console.

### mflux integration notes

Non-obvious points, verified in the mflux 0.18.0 source, that explain some of the choices:

- **`ModelConfig.from_name()` is avoided.** Its resolution loses `sigma_*` and `text_encoder_overrides` (`config_resolution.py:112-128`), which would change Qwen's scheduler. We pass the canonical factory plus `model_path`.
- **`CallbackManager.register_callbacks` is never called.** It installs a `MemorySaver` that destroys `text_encoder` on the very first generation when `num_seeds <= 1` (`memory_saver.py:45-47`): the second request would crash. It also installs a `BatterySaver` that runs `pmset` before every generation.
- **A single callback is registered, at load time.** `CallbackRegistry` has no `unregister` (`callback_registry.py:12-27`).
- **Qwen's `prompt_cache` is purged** past 16 entries: it is keyed by prompt and has no bound at all.
- **Unloading is manual** — mflux exposes no teardown method. We set the submodules back to `None`, then `gc.collect()` + `mx.clear_cache()`.

Specific to `flux2-dev`, where we leave the beaten path:

- **`ModelConfig.from_name("black-forest-labs/FLUX.2-dev")` does not even raise.** `can_infer_substring` finds the `"dev"` alias inside that name and silently fabricates a **FLUX.1**-dev config (`config_resolution.py:57-64`). So the config is built by hand, and `registry._LOCAL_MODEL_CONFIGS` resolves it alongside `ModelConfig`'s factories.
- **Guidance is pre-multiplied by 1000.** `Flux2Transformer.__call__` only scales guidance when it is 1.0 or less (`flux2/…/transformer.py:91`), whereas the FLUX.1 path — the only one exercised upstream with `guidance_embeds=True` — always multiplies by `num_train_steps` (`flux/…/transformer.py:155`). No model shipped by mflux enables `guidance_embeds` on the FLUX.2 transformer, so that path is untested there. `test_guidance_must_be_premultiplied_by_a_thousand` acts as the canary: if mflux fixes the heuristic, it breaks and the compensation must be removed.
- **The tokenizer pads on the left, which produced NaN.** Under a causal mask, a padding query at the head of the sequence has only itself to look at — and it is masked. Softmax returns NaN, and the next row propagates it (`0 × NaN = NaN`): by the second layer *every* position is contaminated and the whole prompt goes to NaN, without a single exception raised. Fully closed rows are therefore reopened, the way transformers' `AttentionMaskConverter._unmask_unattended` does it.
- **mflux's `LanguageTokenizer` does not fit.** With `use_chat_template=True` it sends `[{"role": "user", …}]` and `add_generation_prompt=True` (`tokenizer.py:86-92`), where FLUX.2-dev expects a system + user conversation, contents as lists of typed parts, and `add_generation_prompt=False`. A custom tokenizer is plugged in through `TokenizerDefinition.encoder_class`.
- **No `mx.compile` on the denoising loop**, unlike `Flux2Klein` which enables it outside M1/M2: at 32B the compiled graph can exceed Metal's GPU watchdog.
- **Components are quantized one at a time**, because `WeightApplier.apply_and_quantize` loads all the bf16 before quantizing anything. Reloading, by contrast, needs nothing: `WeightLoader._load_component` tries `_try_load_mflux_format` first (`weight_loader.py:89-92`) and reads the `quantization_level` written by `ModelSaver`.
