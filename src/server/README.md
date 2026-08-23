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
| `qwen-image-flash` | `nvidia/Qwen-Image-Flash` | **NVIDIA Open Model** | — | ❌ | 4 | 1.0 *(adjustable)* | text | ✅ | ✅ | ❌ |
| `qwen-image-2512` | `Qwen/Qwen-Image-2512` | Apache-2.0 | — | ❌ | 50 | 4.0 | text | ✅ | ✅ | opt-in |
| `flux2-klein` | `black-forest-labs/FLUX.2-klein-9B` | **FLUX Non-Commercial** | 🔒 | ❌ | 4 | fixed at 1.0 | text | ❌ | ✅ | ✅ |
| `flux2-dev` | local 8-bit artifact | **FLUX Non-Commercial** | 🔒 | ❌ | 50 | 4.0 | text | ❌ | ✅ | ❌ |
| `anima-turbo` | `circlestone-labs/Anima` | **CircleStone Non-Commercial** | — | ❌ | 10 | 1.0 *(adjustable)* | text | ✅ *(above 1.0)* | ✅ | ❌ |
| `anima` *(aesthetic)* | `circlestone-labs/Anima` | **CircleStone Non-Commercial** | — | ❌ | 30 | 4.5 | text | ✅ | ✅ | ❌ |
| `krea-2-turbo` | `krea/Krea-2-Turbo` | **Krea 2 Community** | 🔒 | ❌ | 8 | 1.0 *(adjustable)* | text | ✅ *(above 1.0)* | ✅ | ❌ |
| `fibo-lite` | `briaai/Fibo-lite` | **CC-BY-NC-4.0** | 🔒 | ❌ | 8 | fixed at 1.0 | **json** | ❌ | ✅ | ❌ |
| `fibo` | `briaai/FIBO` | **CC-BY-NC-4.0** | 🔒 | ❌ | 50 | 5.0 | **json** | ✅ | ✅ | ❌ |

The step and guidance columns are each model's own published defaults, taken
from its card rather than from mflux — which applies blanket values (20 steps,
guidance 3.5) that several of these authors never recommended. Where a card
gives a range, the column shows what this server picks: `z-image` publishes
"Inference steps: 28 - 50" and gets 50, the value its own example uses.

They are defaults, not requirements. Every one is overridable per model in
`server-config.json`, and the five 50-step rows are where that is worth doing
if a generation costs more time than it is worth to you.
| `ideogram-4` | `ideogram-ai/ideogram-4-fp8` | **Ideogram 4 Non-Commercial** | 🔒 | ❌ | 20 *(preset)* | preset | text + json | ❌ | ❌ | ❌ |

**The two models on by default are Apache-2.0 and ungated**, which is the point: a fresh install generates with no HuggingFace token, no access request, and no licence to accept. Everything else ships off — the gated and non-commercial ones because obtaining access is your decision, the rest because a 20-step base model is not a good first impression. Turn any of them on in the config, or in the app's Configuration tab.

Useful details:

- **Prompt format matters on three models.** The `prompt` column says what each one accepts. `fibo` and `fibo-lite` accept **only** a structured JSON caption: their prompt encoder opens with a bare `json.loads(prompt)` whose result is discarded, so plain text raises. The server refuses it up front — `400 prompt_must_be_json`, before any weights load — rather than let you discover it after several GB. `ideogram-4` accepts both, and prefers JSON: plain text works but, per Bria's and Ideogram's own docs, underperforms.

```bash
# fibo-lite: the prompt is a JSON object, passed as the prompt string
curl http://127.0.0.1:8765/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "fibo-lite", "prompt": "{\"high_level_description\": \"a red fox in the snow\"}"}'
```

  The full schemas live with the models: [FIBO's prompting guide](https://huggingface.co/briaai/FIBO) and [Ideogram's](https://github.com/ideogram-oss/ideogram-4/blob/main/docs/prompting.md). mflux ships `mflux-inspire-fibo` and `mflux-refine-fibo` to build those captions with Bria's VLM; the server still does not call them. Two reasons, and the first one used to be the whole answer: it would mean a second model resident alongside the first, and producing a caption against Bria's schema is a different job with a hard failure mode — the JSON check above rejects the output, after the weights have loaded. See [Enhancing a prompt](#enhancing-a-prompt) for the rewriting the server *does* do, on what terms, and why those terms do not extend to these two models.
- **`ideogram-4` takes its step count from a sampler preset**, not from a number: `V4_DEFAULT_20` (20 steps), `V4_QUALITY_48`, `V4_TURBO_12`. Each preset carries a per-step guidance schedule and a noise schedule, so `guidance` is refused and `steps` is best left alone — passing it replaces the schedule with a constant. Pick one with `models.ideogram-4.preset`. Its dimensions are also capped at 2048, checked before loading.
- **`flux2-dev` requires a conversion step** and is not usable as-is: 32 billion parameters, a gated repo, and code mflux 0.19.0 does not provide. See [FLUX.2-dev](#flux2-dev--32b-in-8-bit). Expect ~113 GB of one-time download, a ~58 GB local artifact, and ~58 GB resident during generation.
- **The guidance defaults come from each model's own card**, not from mflux's blanket defaults — 4.0 for `qwen-image-2512` where mflux would say 3.5, 5.0 for base FIBO where its signature says 4.0. The step counts are this server's own: 20 wherever a card asks for 50, because area and step count are what a base model actually costs, and 8 for the ERNIE and FIBO turbos, which is what their cards say.
- **"Distilled" does not by itself mean `guidance` is refused.** What the `guidance` column reports is whether another value is *rejected*: `flux2-klein` and the FIBO and ERNIE turbos fix it at 1.0 and error on anything else, and `z-image-turbo` has it forced to 0. `krea-2-turbo` is distilled too, but its guidance is a real knob — 1.0 is only the reference value — so it accepts both `guidance` and `negative_prompt`. The negative prompt is encoded only above 1.0 there: at exactly 1.0 there is no unconditional branch to put it in.
- **Two models ship in two trainings each.** `anima-turbo` and `anima` are one architecture and one repository, differing only in which checkpoint is loaded: Turbo is distilled to 10 steps at CFG 1, Aesthetic is the undistilled fine-tune at 30 steps and CFG 4.5. Its author recommends starting with Turbo — "only slightly worse on average, while being very fast". Likewise `qwen-image-flash` is `qwen-image-2512`'s architecture distilled to four steps; it is a separate row rather than a setting because the two do not share a noise schedule (Flash publishes a static shift of 3.0, the 2512 release dynamic shifting with a 0.02 terminal), and at four steps that difference is most of the trajectory.
- **Editing is off in the default set.** `flux2-klein` is the only model with an instruction-editing variant sharing its weights, and it ships disabled. `/v1/images/edits` still works on the enabled models, as **img2img** — a different mechanic, so the result is a variation on your image rather than an edit of it. Enable `flux2-klein` for real editing.
- **Dimensions are truncated down to a multiple of 16** — an mflux constraint. `1920x1080` becomes `1920x1072`. The server applies the rounding itself and reports the effective size in the response's `mflux.size` field.

The table above is the catalogue — what each model is worth on its own, precision included. The shipped `server-config.json` then applies two policies on top: `default_size: "1280x720"` everywhere, and the enabled set described above. It does **not** set a precision: that is the catalogue's to decide, per model.

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
| `/playground/api/…` | GET/POST/PATCH/DELETE | its sessions (create, rename, delete), generations and cancellation |
| `/playground/api/queue` | POST | hold or release the queue: `{"paused": true\|false}`. Global, and reported back on `GET /playground/api/sessions` |
| `/playground/api/groups/{id}` | DELETE | delete a whole feed entry — every generation of the lineage, its images, and the reference image its root was made with |
| `/playground/api/sessions/{id}/password` | POST/DELETE | set or change a session password (returns an unlock token) / remove it |
| `/playground/api/sessions/{id}/unlock`, `…/lock` | POST | redeem the password for an unlock token (sent as `X-QDS-Session-Token`; in-memory, 30 min idle, gone on restart) / give it back |
| `/playground/api/preview` | GET | the running playground generation's latest partially-denoised frame (JPEG); 404 when there is none |
| `/playground/images/{name}.png` | GET | images owned by a playground session; **not** TTL-purged. A locked session's images need its token (`?t=` accepted here, since `<img>` sends no header). The **only** playground route without the cross-site check — the `uuid4` filename is the capability, and MCP clients render in their own origin (see [MCP](#mcp)) |
| `/admin/playground/sessions/{id}/password` | DELETE | admin recovery: remove a session password without knowing it |
| `/mcp` | POST/GET/DELETE | the MCP surface (see below). Same credential as `/v1`; absent when `mcp.enabled` is false, and absent in recovery mode |

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

A **negative prompt** is accepted wherever the model has an unconditional
prediction to steer away from, and refused with a 400 where it does not —
guidance-distilled models embed their guidance in the transformer and make no
such prediction, so a negative prompt there would be silently inert. Which is
which is a catalogue fact (`supports_negative_prompt`, published per model on
`/v1/models/{id}` and `/v1/capabilities`); the playground reads it to grey the
field out, and the server refuses one regardless of what any client believes.

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

Cancellation has four cases, because the engine can only be interrupted at a
boundary it controls:

- **queued** → cancelled by its record; it is never handed to the engine;
- **running, mid-denoise** → the engine's global stop, the same mechanism
  `/v1/cancel` exposes, taking effect at the next step;
- **running, but loading weights, waiting on the engine lock, or between the
  images of an `n>1` run** → the engine refuses the request there, so the runner
  holds it and applies it at the next image boundary. The image already being
  computed is kept; the record ends `cancelled` with the images it produced;
- **an upscale** → the same global stop, but read *between tiles* rather than
  between denoising steps. The tile in flight finishes; nothing partial is
  written.

It is not per-job cancellation: the engine's stop is global, so with an external
`/v1` client holding the engine it is that request which stops.

**Pausing** holds the queue rather than stopping anything. It is deliberately
not a fourth kind of cancellation: the image already being denoised runs to
completion and is kept, because the engine can only be interrupted by raising at
a step and the alternative is throwing away work already paid for. The hold
takes effect at the two boundaries the runner owns — before a queued generation
is claimed, and between the images of an `n>1` run — so a paused `n=4` request
sits at `running` with the images it has finished, and produces the rest on
resume. Four consequences worth stating:

- it is **global**, one control for every session, because there is one FIFO
  worker behind all of them. It sits at this router's own auth level rather than
  admin's: it is reversible by anyone who can reach it, and that credential
  already permits `/v1/cancel` and unbounded submission. What it *is* that those
  are not is unbounded in time, which is why the state is published on the
  session list every open tab already polls;
- it does **not** pause `/v1`, which never touches the runner. A paused
  playground is not a paused server;
- a long pause lets the idle unloader release the model as usual — the worker
  parks outside `idle_unload_s`'s in-flight window on purpose, so holding the
  queue gives the machine back instead of pinning the weights;
- it is **runtime state**. A restart clears it, and anything still waiting in the
  queue is failed by the same `mark_interrupted()` that fails a generation caught
  mid-flight: the queue lives in memory and does not survive the process.

A 200 from `POST /playground/api/queue` therefore does not mean nothing is being
denoised — only that nothing more will start.

While a generation runs, the page shows the image being denoised: every second
step the engine decodes the current latents into a small JPEG, keeps it in one
in-memory slot, and bumps `preview_seq` in the `/v1/progress` snapshot; the page
fetches the new frame from `GET /playground/api/preview` and fades it in over the
previous one. Each frame is blurred in proportion to how far the run has got:
early on the latents are mostly noise, and a blur is what turns that into a
readable composition rather than a snowstorm, while by the end the denoiser has
done that job itself and any blur left is only hiding the picture. How much is
left at the end depends on the step count as well as the progress — an 8-step
schedule takes enormous jumps and its late previews are still coarse, where a
50-step one is nearly finished well before its last step. A slow accent-tinted sheen crosses the box the whole time — that,
not the frames, is what says "still working" while a partial image sits
unchanged for several seconds. One slot is enough because the engine runs one
generation at a time, and the bytes stay off the SSE stream, which the dashboard
page shares. A family whose latent layout is not mapped, or a decode that fails,
loses its previews for that run — never its generation.

### Enhancing a prompt

Off by default. Switched on, an **Enhance** button appears beside the model
picker in the composer: the prompt is expanded by a small local LLM before it
reaches the diffusion model, and both texts are kept.

```jsonc
// server-config.json
"rewrite": { "enabled": true }
```

Installed through the menubar app, the weights (2263 MB) arrive with the
server: `Bootstrap` runs `qds fetch --rewriter` after the wheel, so the first
Enhance costs nothing. Installed any other way, they are fetched on first use —
the composer says `First use downloads 2263 MB` before you press Generate — or
ahead of time with `qds fetch --rewriter`. Nothing else to install: `mlx-lm` is
an ordinary runtime dependency and arrives with the server, exactly as mflux
does. `qds rewrite "un chat sur un toit"` exercises the whole thing from the
terminal, with no browser.

**What it is for.** Local models reward a long, specific prompt and give little
back for three words. This closes that gap the way DALL·E 3 and Ideogram do; it
is not what makes Midjourney's images look the way they do, which is an
aesthetic fine-tune rather than any rewriting. A 20-word prompt comes back at
about 130 words of art direction — named rock, a light direction, a camera
angle. Measured at a fixed seed: a clear win on prompts of two or three words,
roughly a draw on ones the model already handles, and a decisive win on anything
not in English — `un chat sur un toit` as typed produces a *man* on a roof, no
cat anywhere, because the text encoder does not read French.

A **negative prompt is deliberately not generated**, and the reason is worth
knowing before writing one by hand: on the distilled models — `krea-2-turbo`,
`anima-turbo`, `z-image-turbo` — the unconditional branch is only built above
guidance 1.0, and those default to 1.0 or below. At the shipped settings a
negative prompt is never encoded at all. See the `negative` column above.

**Which model does it** is a configuration fact, not something the playground
shows: someone writing a prompt needs to know it will be improved and what a
first use costs, not which LLM does it. `rewrite.model` names it, the logs
record it, and the catalogue explains why it is that one — including why a
1.7B was not enough (a 46-word median, and a degenerate loop on a simple
subject) and why Ministral-3-3B was measured and rejected (it ignores the length
instruction, truncating eleven outputs in eighteen, and replaces the subject).

**What it will not touch.** At or over `rewrite.word_ceiling` words (40 by
default) the prompt is generated exactly as written, and the composer says so
before you press Generate. That ceiling is enforced in Python rather than asked
of the model, because asking did not work: told to leave long prompts alone, the
rewriter obeyed 8 times in 18 *and* got worse at everything else, the rule
having competed for a small model's attention. Models whose only prompt format
is JSON — `fibo`, `fibo-lite` — are refused rewriting outright rather than
silently skipped.

**What it records.** Your prompt is never overwritten. The feed titles the entry
with what you typed and folds the expansion away behind *Enhanced prompt*, with
a **Use this prompt** button that drops it into the composer as an ordinary
prompt — which is all that editing or pinning a rewrite needs. A refine or a
variation replays the recorded rewrite instead of asking for a new one: a
rewrite is sampled, so re-running it would produce different words and the
result would not be a variation of anything. If the rewriter fails, the image is
still generated from your prompt and the entry says why; throwing away a
generation you asked for, because an optional step that improves it did not
work, would be replacing detection with punishment.

**`/v1` never rewrites.** The surface is OpenAI-compatible and its Images API
has no rewrite parameter, so expanding a prompt there would quietly break the
contract every script relies on.

**Memory.** The rewriter is a *third* slot in the engine, and unlike the
upscaler's it is transient: loaded, decoded, and released in a `finally`, so
between two rewrites there is nothing in it. Bounded twice over —
`MAX_PROMPT_TOKENS` + `MAX_NEW_TOKENS` cap the KV cache, the only part of a
decode that grows with the input, and `MAX_REWRITER_FOOTPRINT_MB` refuses at
import any entry whose weights *plus that cache* exceed what an upscale already
costs transiently. A bound on weights alone was what shipped first, and it left
a loophole: a deep model with few parameters passes it and blows the KV budget. The prompt half is counted in **tokens**,
inside the engine, against the fully templated text — not at admission, which
has no tokenizer and where an earlier version counted words instead. That was
wrong in a way worth recording: a Chinese or Japanese prompt has no spaces, so
one of any length counted as a single word and cleared the bound entirely.
Admission keeps a character limit as triage, so an impossible prompt is refused
where the message can still name a parameter. Measured: 968 MB resident, 1289 MB peak,
against the 2630 MB this engine already accepts for an upscale and the 19281 MB
peak of a single 512×512 z-image generation. Over twenty consecutive
rewrite-then-generate cycles beside a warm diffusion model, each measured after
resetting MLX's peak counter, the per-cycle peak was identical from first to
last and the diffusion weights were never reloaded. End to end a rewrite costs about a
second, load and unload included.

The system prompt is overridable through `rewrite.system_prompt` — it is a
quality knob. The bounds around it are not: raising them is a decision about the
engine's memory invariant, not a setting.

### Upscaling

From the toolbar under any generated image: a factor, a model, and the enlarged
image lands in the same feed entry as the one it came from.

It is **Real-ESRGAN**, ported to MLX in `qds/upscale/` and checked against a
transcription of basicsr's `RRDBNet` in torch, tensor for tensor
(`tests/test_upscale.py`; the measured gap is around 1e-7 in fp32). Two entries:

| Key | For | Blocks | Weights |
|---|---|---|---|
| `realesrgan-x4plus` | photographic | 23 | 33.5 MB |
| `realesrgan-x4plus-anime` | illustration | 6 | 9 MB |

Both come from `mlx-community`, fp16 and already in MLX's NHWC layout, under
Real-ESRGAN's upstream BSD-3-Clause. The x4plus file is verified: bit-exactly
`fp16(t.transpose(0, 2, 3, 1))` of `Comfy-Org/Real-ESRGAN_repackaged`, checked
on five tensors spread across the network. The anime file has no independently
licensed counterpart to check against, and that is a known gap rather than an
oversight.

Things worth knowing before using it:

- **The network is always ×4.** ×2 is that ×4 resampled down with Lanczos,
  which is upstream's `--outscale` semantics. It costs exactly the same time and
  the same memory. The UI says so, because the shape of the control invites the
  opposite guess.
- **The weights download on first use**, inside the run and holding the queue,
  the way a diffusion model's do. `qds fetch realesrgan-x4plus` pulls them
  ahead of time.
- **Tiling is not optional.** The diffusion model stays resident, so the
  transient peak has to be bounded. Measured here — x4plus fp16, 1024×1024 →
  4096×4096, warm, a fresh process per row, best of three on an idle machine:

  | tile | tiles | time | MLX peak | host RSS |
  |---|---|---|---|---|
  | 256 | 16 | 5.2 s | 2.47 GB | 0.44 GB |
  | **192** | **36** | **6.0 s** | **1.52 GB** | **0.37 GB** |
  | 128 | 64 | 7.0 s | 1.14 GB | 0.35 GB |
  | 96 | 121 | 8.7 s | 0.76 GB | 0.37 GB |
  | 64 | 256 | 9.8 s | 0.42 GB | 0.36 GB |

  192 ships: 15% slower for a 38% smaller peak, which is the right side of that
  trade when 10–28 GB of diffusion model is sitting beside it. The same run at
  `realesrgan-x4plus-anime` takes 2.1 s — a quarter of the blocks.

  These timings are worth about as much as the machine was quiet. Measured
  while the GPU was also generating, the same rows came out three to six times
  higher; the memory columns did not move.

- **The size limit counts what is rendered, not what you asked for.** The
  network always works at ×4, so a ×2 request renders ×4 and throws three
  quarters of it away — meaning a 2048×2048 source at ×2 is four times the work
  of the same source at ×4 if you only look at the output. The limit is
  therefore `source × 4` and stands at 8192×8192. At that limit a run costs
  around 1.1 GB resident and about 21 s; the MLX allocator stays at 1.52 GB
  whatever the source size, which is the tiling doing its job.

  A consequence worth knowing before reaching for ×4 on a large source: an
  8192×8192 PNG of photographic content runs to about 80 MB. It goes into the
  playground's image directory, which is never purged, it is served over HTTP
  as-is, and the feed draws it as a thumbnail. The ceiling bounds the server's
  memory, not your disk.
- **Seams are not proven absent.** `tile_pad` is upstream's 10, while the
  23-block network's receptive radius is 347 pixels, so the tiled result is not
  bitwise identical to an untiled one — about 3e-08 away. The tests pin the
  tiling *geometry* exactly; whether a boundary is visible is a perceptual
  question they do not answer.
- **An upscale copies its source** into the entry's context file, so deleting
  the original does not break it. The copy is unlinked when the entry, group or
  session goes.
- **Cancelling during the weight download does not stop the work.** The engine
  refuses a cancellation while it is loading, so a request made in that window
  is held by the runner and applied afterwards: the record ends `cancelled`,
  but the tiles have already run. It is the same limitation `/v1` has had for
  diffusion weights, on a much shorter wait.
- **The engine keeps a second, bounded slot** for the upscaler, so enlarging an
  image never evicts a warm diffusion model. See `engine.py`'s module docstring
  for why that exception is safe and what bounds it.

## MCP

A third plane, for a language model in a chat client rather than for an
application or a browser. It reaches the same engine, the same queue and the
same durable sessions as everything else, over the same port and the same
credential — and if you are already running the server, it is already there.

### Connecting

Clients that speak HTTP MCP want one URL:

```
http://127.0.0.1:8765/mcp
```

Set `Authorization: Bearer <api_key>` if you configured one; on a loopback
install with no key, nothing is needed.

Clients that speak only stdio get a bridge:

```json
{
  "mcpServers": {
    "quantum-diffusion": { "command": "qds", "args": ["mcp"] }
  }
}
```

`qds mcp` is a relay, not a second server. It reads `server-config.json` for the
address and the API key, connects to the running server before it accepts
anything on stdin, and if nothing is listening it exits with a line saying so
rather than hanging. It never carries the local admin token: that credential
opens the control plane, and this needs the data plane.

### The tools

| Tool | What it does |
|---|---|
| `generate_image` | prompt → image(s). Optionally from a reference image, to edit or vary |
| `refine_image` | change an image this server made; joins its feed entry |
| `vary_image` | same settings, new seed; joins its feed entry |
| `upscale_image` | enlarge one, ×2 or ×4 |
| `wait_for_generation` | pick up a generation that outlived its tool call |
| `cancel_generation` | stop one |
| `list_models`, `list_sessions`, `open_session` | the catalogue, the sessions, and a new one |

`delete_image` and `delete_group` exist behind `mcp.allow_destructive`, off by
default: deleting in the playground is a click a person makes with the image in
front of them, and a tool has no equivalent of that confirmation. There is no
tool for session passwords and none for pausing the queue — see below.

Resources: `qds://models`, `qds://upscalers`, `qds://sessions`, and
`qds://images/{filename}` for an image at full resolution. `list_models`
duplicates `qds://models` deliberately: many clients never surface resources to
the model, while a tool is always reachable by it.

### What a tool call returns

**Text first**, then one resource link per image — anything that truncates a long
result drops what is at the end.

1. The **text** — the facts (file name, seed, dimensions), and a `full image:`
   line carrying the URL, one per image.
2. A **resource link** per image, pointing at the file's own http URL and
   annotated for the `user` audience.

**No pixels are sent to the model.** No image content block, no `data:` URI. Both
existed once and both are gone; the reasoning is worth keeping, because it is
what a reader will otherwise try to add back.

They were there so a *model* could look at the picture, or retype it into its
reply — the only way an image reaches a chat client's message body, since no MCP
block lands there. Retyping base64 is close to the worst task a language model
can be given: zero redundancy, so one wrong character invalidates the image; no
way for it to check its own output; and long high-entropy runs invite repetition
loops. Bounding that line to a length models would actually reproduce (1 300
characters) forced the preview down to **about 81px** on detailed output — mush,
not a picture anyone can judge — while costing the same context twice, once in
the retyped line and once in the image block.

So the trade was: pay thousands of tokens per image for something illegible that
a model may or may not copy correctly. Removed. **The person judges the image**,
in the playground or by opening the link, and neither costs a token nor can be
mis-copied.

What replaced it has to actually work, and that is now the tested property: each
image contributes a name in the text, a `full image:` URL beside it, a
`resource_link` block, and a resource that resolves to the full-resolution PNG.

The server's instructions tell the model to offer that URL as a link and not to
try to embed the picture.

*(Historical note: `mcp.thumbnail_px`, `mcp.thumbnail_quality` and
`mcp.preview_max_chars` sized this mechanism. They are gone. An existing
`server-config.json` that still lists them keeps loading — unknown keys are
ignored — but they no longer do anything.)*

### Why an image link works from a chat client at all

`GET /playground/images/{name}.png` is the one route on this server that grants
two things every other route refuses, and both exist for the same reason: a chat
client renders in an origin of its own, on a page that is not local.

- **No cross-site check.** Every other playground route refuses a request a
  browser marked cross-site. Here the origin was never the authority: the
  filename is a `uuid4().hex`, so naming a file is holding 122 bits of secret.
- **Private Network Access is granted.** Chromium asks permission before letting
  a page reach a more private address than its own, sending a preflight with
  `Access-Control-Request-Private-Network: true`. Starlette's CORS middleware
  refuses that outright with a 400, so an `<img>` in an Electron chat client
  never loaded — while `curl`, which sends no preflight, fetched the same URL
  perfectly. That asymmetry is why this looked like a client bug for a long time.

The grant is one route wide. `/v1`, `/admin`, `/playground/api` and `/mcp` all
still answer a private-network preflight with a 400, which matters most for
`/v1`: a keyless loopback install has an open data plane, and granting it there
would let any page in any tab spend this machine's GPU. That is precisely the
drive-by Private Network Access exists to stop, which is why the one-line
`allow_private_network=True` on the CORS middleware was **not** the fix — it
would have granted the same to all of them. See `qds/pna.py`.

What still holds on the image route: a password-protected session's images need
its token even so, a name no row holds is a 404, and `no-store` keeps a relocked
session out of the cache. A preflight only says the request may be *made*; every
check still runs on the GET that follows.

### What a result costs a model

A result is facts and URLs, so its size follows the *number* of images, not their
detail: a four-image call is a few hundred characters. That is the point of
removing the pixels.

The mechanism it replaced was measured on real 2880×1600 generations, and the
numbers are kept because they are the argument:

| what was sent | median tokens | worst seen |
|---|---|---|
| 512px preview / quality 82 (the first default, and wrong) | 16 800 | 22 100 |
| 256px / 70 (the last default) | 4 000 | 4 900 |
| **nothing (today)** | **~0** | ~0 |

The original 512/82 was validated against a flat-colour test image, came out at
3 KB, and looked free. It was not: on an 8B model with an 8k window it *is* the
context. Shrinking it to fit made it illegible instead. Neither is a good place
to be, which is why there is no preview now.

Full resolution is named two ways: the `full image:` URL in the text and the
`resource_link` block. The absolute filesystem path is deliberately not among
them: it used to be, and it put the operator's home directory — their username
with it — into a model's context on every generation, to save a lookup the file
name and the playground already answer.

### Waiting, and the ceiling

Generating blocks, because a chat client's model wants one call that comes back
with a picture. While it waits it emits MCP progress notifications.

After `mcp.tool_timeout_s` (default 600) the call **returns** rather than fails,
carrying the generation id, the status, and whether the queue is paused. The
work is queued and durable; `wait_for_generation` picks it up. A client that
disconnects or cancels mid-call cancels the generation with it.

**A caveat about progress, stated as one.** The engine keeps a single global
progress snapshot — one lock serializes every job, so one snapshot serves
`/v1/progress` for the whole process. A per-call notification therefore reports
a denoising step only when four facts agree: the runner has claimed *this*
generation, the engine is denoising, the model matches, and the seed is one of
this call's. Otherwise the notification carries lifecycle only — "queued",
"waiting for the engine" — rather than a step borrowed from another request.
What that buys is that progress is never *misleading*; what it does not buy is
precision, since two simultaneous jobs of the same model and the same seed would
still be conflated. Making it exact means a per-job progress channel in the
engine, which would cost the lock-free snapshot.

### Where the images go

Into playground sessions, which is the point rather than an implementation
detail: a chat generation is visible in the playground, survives the
conversation, and can be refined, varied, upscaled or deleted there. Each MCP
conversation gets its own session, created the first time it generates something
— connecting and listing tools leaves nothing behind. `open_session` starts a
fresh one; `session_id` names an existing one.

A **password-protected session is unreachable over MCP**, and there is no tool to
unlock one. A password is a decision someone made at a browser, and a tool that
could walk past it would make the control mean nothing. Unlock it in the
playground.

There is no tool to pause the queue either. Pausing is a person's control over
their own machine, and combined with a blocking tool it guarantees a timeout —
so `paused` is *reported* in what a timed-out call returns, rather than being
something a model can set.

### Settings

Under `mcp` in `server-config.json`. Like `rewrite`, it is not reachable through
`QDS_SERVER_*` — those cover the `server` section only.

| Key | Default | Note |
|---|---|---|
| `enabled` | `true` | `false` removes the route entirely |
| `tool_timeout_s` | `600` | when a blocking call gives back an id instead |
| `poll_interval_s` | `0.5` | how often the generation row is re-read while waiting |
| `image_roots` | `[]` | absolute directories a model-chosen `reference_path` may read from |
| `allow_destructive` | `false` | offer `delete_image` and `delete_group` |

There is deliberately no `mcp.max_n`: `server.max_n` is the one authority, and it
already applies here. There is no thumbnail setting either — a tool result sends
no pixels, so there is nothing to size.

**`image_roots` is a containment boundary, not a convenience.** `reference_path`
is an argument the *model* fills in, and a model is untrusted — what it asks for
may have been written into a prompt by someone you never met. Publishing an
arbitrary readable file into the playground store would put it behind an HTTP
route a loopback install serves without a credential. So it is empty by default,
every path outside it is refused without the file being opened, and symlinks are
resolved before the check. Point it at the one directory you meant.

## Configuration

`server-config.json` (JSON). Every key in the `server` section can be overridden by `QDS_SERVER_<KEY>` in upper case — `QDS_SERVER_PORT=9000`, `QDS_SERVER_API_KEY=…`, `QDS_SERVER_CORS_ORIGINS=https://a.example,https://b.example`. `QDS_SERVER_CONFIG` points at a different config file.

### The `server` section

| key | default | role |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8765` | binding |
| `api_key` | `null` | when set, `Authorization: Bearer` is required. **Mandatory as soon as `host` is not local** |
| `cors_origins` | `[]` | origins a **browser page on another origin** may read a response from. Empty means none, which is what a keyless `/v1` needs: the dashboard and playground are same-origin and need no entry here. Add one only for a browser client served from elsewhere, and prefer setting an `api_key` alongside a `"*"` |
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
| `default_size` | `null` | config-wide resolution, `"WxH"`. Applies to every model; `null` leaves each on its catalogue size |

These are the *code* defaults, used when no config file is found. The shipped `server-config.json` is more opinionated: `1280x720`, and only the two fast ungated models enabled.

#### The `rewrite` section

Prompt rewriting, described in [Enhancing a prompt](#enhancing-a-prompt). **On** by default. It shipped off, because the first Enhance fetched a gigabyte and nobody had asked for it; both halves of that reason are gone — the app fetches the weights at install, and the composer says what a first use costs before anything is pressed. The decoder itself ships with the server, like mflux.

| key | default | role |
|---|---|---|
| `enabled` | `true` | offer prompt rewriting at all. Off: the route refuses with 409, `/v1/capabilities` says why, the dashboard hides the control |
| `model` | `qwen3-4b-2507-4bit` | rewriter catalogue key. A separate catalogue from the image models; `qds fetch` accepts keys from either, or `--rewriter` for whichever this names |
| `word_ceiling` | `40` | prompts of this many words or more are generated as typed, without calling the rewriter. Must stay below `MAX_PROMPT_TOKENS` |
| `max_new_tokens` | `320` | longest rewrite to decode. Can only be lowered: it is what the engine's KV-cache bound is computed from |
| `temperature` | `0.7` | sampling temperature. Measured; 0.3 gave the same structures with less variety |
| `timeout_s` | `30` | wall-clock bound on one decode, checked between tokens. Generous against a measured p95 of 1.1 s |
| `system_prompt` | `null` | replaces the shipped instructions. A quality knob — the bounds above are not |

`MAX_REWRITER_WEIGHTS_MB`, `MAX_PROMPT_TOKENS` and the ceiling on `max_new_tokens` are deliberately **not** configurable: they are what make the engine's third slot safe, and raising one is a decision about that invariant rather than a preference.

There is deliberately **no config-wide quantization**. There was one, `default_quantize`, and it overwrote each catalogue row rather than standing behind it — so a single number decided the precision of every model, including the rows that had picked one on purpose. What a bit depth costs depends entirely on the model it is applied to: `anima` at 2B is visibly broken at 4 bits, where the same depth is unremarkable on a 20B. A config that still carries the key starts normally and logs that it was ignored.

#### Precision, per model, and where it silently does not apply

Each catalogue row carries its own bit depth — 8 for most, bf16 for `anima` — and `models.<key>.quantize` overrides it for one model (`0` means bf16). Nothing else touches it.

How far a model can be pushed is a property of that model, and it is worth measuring rather than assuming. On `anima`, rendered from one seed: bf16 and 8-bit are indistinguishable, 6-bit is clean, and **4-bit is broken** — illegible architecture and a scratchy overlay across the frame. On a 20B the same 4 bits merely costs fine detail. That asymmetry is why the setting lives on the row.

It only works from bf16 weights. mflux resolves a conflict in favour of what the file already contains — `QuantizationResolution`, rule `conflict` → action `STORED` — and prints "Model is pre-quantized at 8-bit. Ignoring -q". So a repo distributed pre-quantized keeps its own precision whatever you ask. Two entries are in that position and are skipped rather than misreported: `flux2-dev` (our own 8-bit artifact) and `ideogram-4` (fp8, every heavy component marked `skip_quantization`). `/v1/capabilities` reports the effective value plus a `prequantized` flag, so a client never has to guess.

That is also why `qwen-image-2512` and `fibo` point at their **raw bf16 repos** rather than the pre-quantized conversions: only raw weights can honour the setting, and both rows ask for 8 bits. The trade is a bigger download and a load-time memory peak at bf16 before quantization — around 55 GB for the 20B Qwen, which fits on a 103 GB machine and not on a 32 GB one. Point `models.qwen-image-2512.model_path` at `mlx-community/Qwen-Image-2512-8bit` to go back.

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
    "cors_origins": [],
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

**mflux 0.19.0 cannot load FLUX.2-dev.** Its FLUX.2 family is *klein-only*: `AVAILABLE_MODELS` holds nothing but the klein variants, and `Flux2Initializer` hardwires `Qwen3TextEncoder` and `Flux2KleinWeightDefinition`. But the actual gap is narrow — verified tensor by tensor against the repo's weight indexes:

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
- **No LoRA, ControlNet or inpainting.** mflux offers them; they are not exposed here.
- **Upscaling is playground-only.** There is no `/v1` route for it: the OpenAI
  Images API has no such endpoint, and inventing one is a decision nobody has
  asked for yet.
- **Upscalers are absent from the Models tab.** They are fetched on first use
  and with `qds fetch`, but the dashboard's model management does not list
  them or offer to delete them.

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

Non-obvious points, verified in the mflux 0.19.0 source, that explain some of the choices:

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
