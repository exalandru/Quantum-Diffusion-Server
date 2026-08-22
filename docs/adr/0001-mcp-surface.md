# 1. An MCP surface, mounted in the server and backed by playground sessions

- **Status**: Accepted
- **Date**: 2026-08-22

## Context

QDS had two HTTP planes. `/v1` is OpenAI-compatible, stateless and
request-scoped, and is called by applications. `/playground/api` is
SQLite-backed and durable, and is called by a person at a browser. Neither is
reachable from the third client that now matters: a language model running
locally in someone's chat client, choosing its own tools.

The requirement was a surface **as rich as the playground** — generation,
editing, refinement, variation, upscaling, prompt enhancement, sessions,
cancellation — reachable over MCP from a local chat client.

Four things about that client shaped every decision below. It is often small, so
it needs verbs rather than a parameter matrix and error *messages* rather than
codes. Its context window is small, so full-resolution images cannot go into a
tool result. It cannot be asked to confirm anything. And its arguments are
untrusted — not because the model is hostile, but because what it asks for may
have been written into a prompt by someone the operator never met.

## Decision

Five decisions, recorded because their reasons are not recoverable from the code.

### 1. `mcp` is a hard runtime dependency, enabled by default

`pyproject.toml` already carries a comment block refusing optional extras, for
mlx-lm. That argument rests on **two** premises, and only one transfers:

- *`uv tool install <wheel>` cannot reach an extra.* This is the whole
  installation for an app install, so a dependency has to be in the wheel's
  metadata or it never arrives. **This transfers.**
- *mlx-lm cost exactly one package*, because every transitive dependency it has
  already arrived with mflux. **This does not transfer.** Measured on this
  repository, `mcp==2.0.0` adds eight packages and about 11 MB to a 1.17 GB
  environment: `mcp-types`, `jsonschema`, `referencing`, `rpds-py`,
  `opentelemetry-api`, `pyjwt`, `pycparser`, `sse-starlette`.

That cost is real and is carried by every install, including ones that never
speak MCP. It was accepted anyway: 11 MB against 1.17 GB is under one percent,
and the alternative — an extra — cannot be installed by the application at all,
which is the failure mlx-lm's comment block already describes.

`enabled: true` by default, matching `rewrite`: nothing is downloaded on first
use, and a plane that has to be discovered in a config file is a plane nobody
finds. `enabled: false` removes the route rather than answering it with a
refusal.

Pinned `==2.0.0` rather than `>=`, like `mflux`. This SDK renamed its primary
class between majors and changed what a cancelled request returns, and QDS
depends on both. The next bump should be a deliberate act.

### 2. Mounted in the existing application, not a second process

`/mcp` is a Starlette mount inside `create_app`. A model therefore reaches the
same engine, the same FIFO queue, the same durable sessions and the same
credential as the browser, on the same port.

The alternative — a second process speaking HTTP to the server — was rejected on
mechanism, not effort. The engine is a hard singleton (one lock, one worker
thread, one model resident in unified memory), so a second process could not
generate anything itself; it would have to be an HTTP client of the first, which
is what `qds mcp` already is for the clients that need it. And a second process
would need a second port pre-flight, a second pid file, a second health check
and a second entry in the menubar app's supervisor — none of which the mount
requires. `src/menubar` and `src/dashboard` are unchanged by this work.

Two consequences were designed for rather than discovered:

- A mounted sub-application's lifespan **never runs**, so `create_app`'s lifespan
  enters the transport's session manager — entered last so it exits first, which
  is what unwinds an in-flight tool call before the runner and engine stop.
- The transport's session manager refuses to run twice, which would have made
  `create_app`'s result a single-use application. So the mount is an
  indirection (`MCPMount`) that builds a fresh transport per start. A request
  arriving while nothing is running gets a 503 naming the cause, rather than the
  SDK's "Task group is not initialized" as a 500.

### 3. Playground sessions, not a parallel MCP store

Every MCP generation is a row in the playground's SQLite database, in a session
visible in the browser.

This is the decision that delivers "as rich as the playground". Refinement and
variation need a **lineage** (`groupId`) to join; upscaling needs a source the
server already owns; cancellation and queue pausing need the shared runner. A
parallel store would have had to reproduce all four, and the two sets of images
would then have diverged in the one place a user looks at them.

It also makes the surface honest about what it is: images generated from a chat
are on someone's machine, in their playground, after the conversation ends.

Each MCP conversation binds to its own session, created lazily on first
generation — connecting and listing tools leaves nothing behind. The binding is
an in-memory cache keyed on the transport's session id; the *session* is the
durable thing, and losing the binding costs one new session rather than leaving
a dangling id.

### 4. Progress is fenced rather than forwarded — a knowingly partial guarantee

**This is the decision most likely to be "fixed" wrongly by a future reader.**

`ModelEngine` keeps one `ProgressSnapshot` for the whole process, and its own
docstring argues for that: one lock serializes every MLX job, so a global
display can poll one lock-free snapshot. That reasoning is sound for
`/v1/progress` and unsound for a per-call notification, because the playground
runner marks a row `running` *before* it awaits the engine's lock — so during
that window the snapshot may still describe a `/v1` request.

A per-tool-call notification therefore reports a denoising step only when four
facts agree: the runner has claimed this generation, the engine is `generating`,
the model matches `spec.key`, and the seed is one of this call's. Otherwise it
carries lifecycle only — "queued", "waiting for the engine" — and the emitted
value is clamped monotonic, because attribution can be lost mid-image and the
protocol requires progress to increase.

What this buys: progress is never *misleading*. What it does not buy: precision.
Two simultaneous jobs of the same model **and** the same seed would still be
conflated — which requires a `/v1` caller to pick, out of 2³², one of this
call's seeds.

The precise mechanism is a per-job progress channel in `ModelEngine`. It was not
built here because it retracts the lock-free-snapshot argument the engine
currently rests on, which is a decision about the engine rather than about this
surface. `tests/test_mcp_progress.py` holds the discriminating witness: an
implementation that forwards `engine.progress()["step"]` fails it.

### 5. MCP is deliberately less capable than the playground, in exactly two places

Both are places where the difference between a person and a model is the whole
point.

**A password-protected session is unreachable, and no tool can unlock one.** A
password is a decision someone made at a browser about their own machine. MCP
carries no unlock-token channel and must never grow one: a tool that could walk
past it would make the control mean nothing. The refusal names the playground
instead of offering a way through.

**A model-chosen filesystem path reads nothing by default.** `reference_path` is
filled in by the model, so `mcp.image_roots` is empty by default, every path
outside it is refused *without the file being opened*, and symlinks are resolved
before the containment check. The stake is concrete: publishing an arbitrary
readable file into the playground store would put it behind an HTTP route that a
default loopback install serves without a credential. `reference_image` — a file
this server generated — needs no such setting, because a database row must match
before any path is built, which is the traversal guard the upscale route already
relies on.

Two smaller refusals follow the same logic. `delete_image` and `delete_group`
are absent unless `mcp.allow_destructive` is set, because deleting in the
playground is a click made with the image in front of you. And there is no tool
to pause the queue: pausing is a person's control over their own machine, and
combined with a blocking tool it would guarantee a timeout — so `paused` is
*reported* in what a timed-out call returns instead.

## Consequences

**One admission authority.** MCP tools are handed the very validator callables
the playground route uses (`MCPDeps`), and the upscale route's body was
extracted into a closure both planes call. A check added to one plane refuses
the same input on the other, with the same error code — witnessed by a
parametrized test that runs each refusal through both surfaces side by side.
This is the long-term property this design is bought for.

**One authorization authority and one host allowlist.** `require_api`'s rule was
extracted into a predicate (`build_authorizer`) that the mounted app's ASGI
guard calls; `admin.deny_cross_site`'s rule likewise (`origin_matches`). The
SDK's own DNS-rebinding protection is switched **off**, because
`install_host_guard` already refuses an unknown `Host` on every route, driven by
`server.allowed_hosts` — a second allowlist would silently refuse `/mcp` the
first time an operator added a LAN name for `/v1`.

**One playground route lost its cross-site check, and it is `GET
/playground/images/{name}.png`.** Recorded here because it is a security
boundary that moved, not an implementation detail.

The tool result publishes that URL so a client which does not render image
content blocks can still show the picture. Every MCP client renders in an origin
of its own, so `deny_cross_site` turned the advertised link into a 403 whose
JSON body was saved as a `.png` — observed in Jan, and reproduced.

The relaxation is defensible because on that route the origin was never the
authority. The filename is a `uuid4().hex`: naming a file is holding 122 bits of
secret. The session lock is enforced per request underneath. So the origin check
refused exactly one thing — a page that already knew the name, which is a page
that already had what was being protected.

What was given up, stated plainly: a page that obtains a filename by some other
means can now display that image. What was kept: `/playground/api/*` still
refuses cross-site, so nothing can *enumerate* names; a protected session still
needs its token; `no-store` still prevents a relocked session being replayed
from cache. Four witnesses in `tests/test_playground_lock.py` pin both halves.

**Cross-site refusal applies to `/mcp`.** It is the playground's rule, not
`/v1`'s, because MCP writes what the playground writes. With `cors_origins`
defaulting to `["*"]` and a loopback install having no API key, without it a page
in any tab could spend the GPU and write durable state. It fires only when
`Origin` is present, so chat clients — which send none — are unaffected.

> **Amendment (2026-08-22).** `cors_origins` now defaults to `[]` rather than
> `["*"]`, so the premise above no longer describes a shipped default. The
> decision is unchanged and the reasoning is why: the refusal is enforced by
> `MCPGuard`, not by CORS, so an operator who widens `cors_origins` for `/v1`'s
> sake still cannot reopen `/mcp`.

**Blocking tools hold a connection for up to ten minutes.** A SIGTERM mid-call
drops it (`shutdown_grace_s` is 10s); the generation row survives, and
`mark_interrupted()` makes it visible as failed after restart.

**`mcp.*` is not environment-overridable**, exactly like `rewrite.*`:
`_env_overrides` covers `ServerSettings` only. The menubar app cannot toggle MCP
through `childEnvironmentMap`; it is set in the config file or the dashboard.

**The stdio relay has a fixed method set.** There is no generic proxy in the SDK,
and forwarding everything would mean reaching into transport internals. A test
asks the real server what it advertises and fails if `RELAYED` does not cover
it — failing *closed*, since an untested relay silently drops a new capability
for every stdio client.

**Concurrency is unchanged.** MCP submits to the same single-worker engine. It
adds a third caller to one queue, not a third lane.

## Known non-guarantees

- Progress is fenced, not per-job (decision 4).
- A resource read's *refusal reason* does not reach the caller: the SDK's
  resource-template layer replaces it with a generic message. The read is still
  refused, which is the containment; tools do carry their message.
- No streaming previews to the model. `engine.preview()` carries the same
  attribution hazard as the progress snapshot, for a much smaller benefit.
- Nothing here bounds engine memory.
- **The preview is a `data:` URI the model pastes, bounded by its encoded
  length rather than its dimensions.** This went round five times, and the loop
  is the finding worth keeping.

  `![](http://…)` cannot work: a chat client's `img-src` allows `https:` and
  `data:` but not `http:`, so it is refused before any request is made — which
  is why the access log stayed empty while the tile stayed broken, and why three
  correct server-side fixes (a cross-site refusal, a missing private-network
  grant, an oversized thumbnail) changed nothing. `![](data:…)` renders;
  verified in the client.

  Whether a model will reproduce it was got wrong twice, in both directions.
  From one refusal at 256px: "a model will not copy base64" — wrong, 2 263
  characters had been copied in a controlled test. From that: "it is a size
  limit, shrink it" — the same model then declined at 1 300. Then a different
  model (Hermes) reproduced 1 308 unprompted. What the evidence actually
  supports is narrow: *some* models reproduce *short* encodings, and the ceiling
  is per-model. So it became a setting with a measured default, not a constant.

  Dimensions were the wrong bound throughout. Across ten real generations at
  256px the encoding ranged 1 400–19 600 characters — cost follows detail, not
  pixels — so `preview_max_chars` bounds the encoding directly and the encoder
  walks quality then size down to fit. The consequence is a 60–110px preview on
  detailed output, accepted deliberately: a small picture that arrives beats a
  large one that does not.

  Two formatting rules are load-bearing and both were observed failing: the
  `![` row is flush left, because four leading spaces make markdown a code
  block and every neighbouring row is indented; and it stays on one line,
  because a newline inside `(...)` strands `![alt](` and `)` as prose.

- **Private Network Access is granted on `GET /playground/images/…`, and only
  there.** Chromium preflights any request from a page to a more private address
  than its own; Starlette's CORS middleware answers that preflight with a 400,
  so an `<img>` in an Electron chat client never loaded while `curl` fetched the
  same URL perfectly. Diagnosing it took an unreasonable amount of time
  precisely because every server-side measurement said the URL was fine — it
  was, for anything that does not preflight.

  `CORSMiddleware(allow_private_network=True)` would have been one line and was
  rejected: the flag is app-wide, and a keyless loopback install has an open
  `/v1`, so granting private-network access there lets any page in any tab spend
  the machine's GPU — the exact drive-by PNA exists to prevent. `qds/pna.py` is
  a middleware outside the CORS one that answers the preflight for that single
  path prefix. Witnesses assert both halves: the grant, and that `/v1`,
  `/admin`, `/playground/api` and `/mcp` still refuse.

  Same reasoning as the cross-site removal on the same route, and it composes
  with it: the `uuid4` filename is the capability, the session lock is enforced
  per request, and a preflight grants only the right to *make* the request.

- **The thumbnail is a context budget, and its first default was wrong by an
  order of magnitude.** `mcp.thumbnail_px` sizes a base64 block in a *model's*
  context, and its cost follows the detail surviving the downscale rather than
  the source file's size. 512/82 was validated against a flat-colour test image,
  produced 3 KB, and looked free; measured against real 2880x1600 generations it
  produced 40-54 KB, or 16 000 to 22 000 tokens -- on an 8B model with an 8k
  window, the whole context. The default is now 256/70 (~4 000 tokens on the
  same images) and `0` omits the block.

  There is deliberately no runtime detection. MCP's `ClientCapabilities` carries
  nothing about whether the client or its model can read an image, so a server
  that guesses spends thousands of tokens on a text-only model for nothing.
  It is a setting, and the documentation states the measured cost of each value.

  The lesson is about the witness, not the number: a synthetic flat-colour
  fixture is the least representative possible input for anything whose size is
  a function of entropy, and every test passed while the default was unusable.

- **Blocks are ordered text-first.** Anything that truncates a long result drops
  what is at the end, and the end must not be the markdown line the reply
  depends on.

- **No content block reaches the assistant's message body.** Observed in Jan and
  in Msty: both render the thumbnail inside the tool-call panel, which is
  collapsed by default, so the model sees its image, concludes it is displayed
  and does not repeat it -- leaving the person with prose. The protocol offers no
  block that lands in the reply, so only the model can put the image there, as
  markdown. The result therefore carries four channels per image: an annotated
  thumbnail (what the model looks at), a `resource_link` (the only route no
  client-side network policy can block), a ready-made `![alt](url)` line (the
  only route into the reply), and the facts in text. Handing over a finished
  markdown line rather than a bare url is deliberate: a small model copies more
  reliably than it composes.

  This reverses an earlier decision in this same design, which relabelled the
  url `download:` precisely to *discourage* re-embedding. That rested on the
  premise that a markdown tile duplicated a picture the person could already
  see. The premise was false, and the observation that falsified it is recorded
  here so the reversal is not re-reversed. The content block is
  spec-shaped (`image/jpeg`, 512px, a few KB) and some clients drop it silently;
  nothing server-side changes that, which is why the URL had to work.
