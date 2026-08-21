"""OpenAI-Images-compatible HTTP API.

Standard endpoints: `/v1/models`, `/v1/models/{id}`, `/v1/images/generations`,
`/v1/images/edits`. Local extensions: `/health`, `/v1/capabilities`,
`/v1/progress`, `/v1/cancel`, `/v1/unload`, plus the `steps`, `seed`,
`guidance`, `negative_prompt` and `strength` request fields — extra fields that
the OpenAI SDKs simply ignore.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from qds import __version__, admin, credential, logbuffer, playground_lock
from qds import settings as settings_module
from qds.auth import build_dependencies
from qds.engine import GenerationJob, ModelEngine
from qds.errors import APIError, error_payload, install_exception_handlers
from qds.hosts import allows as host_allows
from qds.hosts import build_allowlist
from qds.idle import IdleUnloader
from qds.jobs import JobManager
from qds.logbuffer import LogBuffer
from qds.logs import SERVER_LOGGER, setup_logging
from qds.playground import PlaygroundRunner, PlaygroundStore
from qds.registry import ModelSpec, edit_enabled, parse_size
from qds.session import SessionStore, discard_local_token, issue_local_token
from qds.settings import (
    RESPONSE_FORMATS,
    ConfigError,
    Settings,
    load_settings,
    recovery_settings,
)
from qds.store import ImageStore
from qds.upscale import catalogue as upscale_catalogue

logger = logging.getLogger(SERVER_LOGGER)

MAX_SEED = 2**32 - 1
#: Value used when `/v1/images/edits` falls back to img2img without the client
#: specifying `strength` (mflux/cli/defaults/defaults.py:14).
DEFAULT_IMAGE_STRENGTH = 0.4
#: Polling cadence of `/v1/progress`. A denoising step takes a few hundred
#: milliseconds at best, so there is no point going faster.
PROGRESS_POLL_S = 0.25
#: Heartbeat when nothing changes, so departed clients get noticed.
PROGRESS_PING_S = 15.0


#: An unlock token, as the playground sends it. Module-level on purpose: with
#: postponed annotations FastAPI resolves names in the module namespace, so a
#: local alias inside `create_app` would be read as a required body field.
SessionToken = Annotated[str | None, Header(alias=playground_lock.UNLOCK_HEADER)]


class RenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    #: `None` or blank clears the title back to "first prompt".
    title: str | None = Field(default=None, max_length=1000)


class QueueStateRequest(BaseModel):
    """Hold or release the playground queue."""

    model_config = ConfigDict(extra="forbid")
    paused: bool


class SessionPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str


class UpscaleRequest(BaseModel):
    """Enlarge an image the session already owns.

    No image bytes: `image` names a file the server wrote and can attribute.
    `model` and `scale` are checked against the catalogue in the route rather
    than by an enum here, so the error names the valid values.
    """

    model_config = ConfigDict(extra="forbid")
    #: Filename of a *generated* image, as served by `/playground/images/`.
    image: str = Field(max_length=255)
    model: str = Field(max_length=64)
    scale: int
    #: Feed entry to join. Defaults to the source's, so an upscale grows the
    #: entry its image came from rather than starting a new one.
    group: str | None = Field(default=None, max_length=64)


class ImageGenerationRequest(BaseModel):
    # `extra="ignore"`: quality, style, user, background, output_format,
    # moderation… are accepted and ignored rather than failing a standard
    # OpenAI client's request.
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    model: str | None = None
    n: int = Field(default=1, ge=1)
    size: str | None = None
    response_format: str | None = None

    # mflux extensions
    steps: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    guidance: float | None = Field(default=None, ge=0)
    negative_prompt: str | None = None


async def progress_events(
    engine: Any,
    *,
    poll_s: float = PROGRESS_POLL_S,
    ping_s: float = PROGRESS_PING_S,
) -> AsyncIterator[str]:
    """Progress SSE frames, until the consumer goes away.

    An infinite generator by design: it is the client disconnecting that
    cancels it, through `StreamingResponse`'s task group. Pulled out of the route
    so it can be tested without HTTP — `TestClient` does not propagate
    disconnects, so reading it through that would block forever.
    """
    last: str | None = None
    last_emit = time.monotonic()
    while True:
        payload = json.dumps(engine.progress(), ensure_ascii=False)
        now = time.monotonic()
        if payload != last:
            yield f"data: {payload}\n\n"
            last = payload
            last_emit = now
        elif now - last_emit >= ping_s:
            # An SSE comment: with no traffic, a client disconnect would stay
            # invisible until the next generation.
            yield ": ping\n\n"
            last_emit = now
        await asyncio.sleep(poll_s)


def _restart_unavailable() -> None:
    raise APIError(
        "This server was not started in a way that can restart itself.",
        status_code=501,
        error_type="server_error",
        code="restart_unavailable",
    )


def install_host_guard(app: FastAPI, settings: Settings) -> None:
    """Refuse requests whose `Host` header is not one this server answers to.

    This closes DNS rebinding, which authentication cannot: a page on
    `evil.example` whose name resolves to 127.0.0.1 is same-origin *to the
    browser*, so it may read the responses, and a default install has no API key
    to stop it. What it cannot fake is the `Host` header, which carries the name
    the browser dialled.

    **It no longer steps aside for a wildcard bind.** It used to, which meant
    turning on "listen on the local network" also turned this off — the
    protection disappearing exactly when it started to matter. The allowlist
    grows instead, to the addresses and names this machine answers to.
    """
    port = settings.server.port
    # Computed once, at startup, and never on the request path: resolving this
    # machine's own hostname can block on a network where it does not resolve.
    allowed = build_allowlist(settings.server.host, port, settings.server.allowed_hosts)

    @app.middleware("http")
    async def guard_host(request: Request, call_next):
        host = request.headers.get("host")
        if not host_allows(host, allowed, port):
            return JSONResponse(
                status_code=421,
                content=error_payload(
                    f"This server does not answer to the host {host!r}. "
                    f"Add it to server.allowed_hosts to permit it.",
                    error_type="invalid_request_error",
                    code="host_not_allowed",
                ),
            )
        return await call_next(request)


#: Built dashboard assets, put here by `make build-dashboard` and shipped inside
#: the wheel. Absent in a source checkout that has never built the front end.
DASHBOARD_DIR = Path(__file__).resolve().parent / "_dashboard"


def mount_dashboard(app: FastAPI) -> None:
    """Serve the dashboard at `/dashboard` and the playground at `/playground`.

    A missing build is answered with a 503 naming the command that fixes it,
    not with a 404: the difference between "this server has no dashboard" and
    "you typed the wrong path" is the whole diagnosis.
    """
    playground_page = DASHBOARD_DIR / "playground.html"
    if (DASHBOARD_DIR / "index.html").is_file():
        if playground_page.is_file():
            # An exact-match route, never a catch-all: `/playground/api` and
            # `/playground/images` are the same prefix and must keep working.
            @app.get("/playground", include_in_schema=False)
            async def playground_page_response() -> FileResponse:
                return FileResponse(playground_page)

        app.mount(
            "/dashboard",
            StaticFiles(directory=DASHBOARD_DIR, html=True),
            name="dashboard",
        )
        return

    @app.get("/dashboard", include_in_schema=False)
    @app.get("/dashboard/{path:path}", include_in_schema=False)
    @app.get("/playground", include_in_schema=False)
    async def dashboard_missing(path: str = "") -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=error_payload(
                "The dashboard was not built into this installation. "
                "Run `make build-dashboard` and reinstall, or use the API directly.",
                error_type="server_error",
                code="dashboard_not_built",
            ),
        )


def create_app(
    settings: Settings | None = None,
    engine: ModelEngine | None = None,
    *,
    request_restart: Callable[[], None] | None = None,
    local_token: str | None = None,
) -> FastAPI:
    """Build the application.

    `request_restart` is how `/admin/restart` reaches the process running this
    app. It is injected rather than reached for, because only the caller that
    owns the uvicorn server knows how to stop it — and an app embedded in a test
    client owns nothing, which is why the default refuses instead of pretending.
    """
    settings = settings or load_settings()
    setup_logging(settings.server.log_level, settings.server.log_file, settings.server.log_json)
    if settings_module.missing_config_path is not None:
        logger.warning(
            "No configuration file at %s: every default applies. "
            "Point QDS_SERVER_CONFIG at your server-config.json.",
            settings_module.missing_config_path,
        )

    registry = settings.registry()
    if not registry:
        raise ValueError("No model enabled: check the 'models' section of server-config.json.")

    engine = engine or ModelEngine(
        request_timeout_s=settings.server.request_timeout_s,
        progress_log_every=settings.server.progress_log_every,
    )
    store = ImageStore(
        Path(settings.server.image_store).expanduser(),
        ttl_s=settings.server.image_ttl_s,
    )
    idle_unloader = IdleUnloader(engine, settings.server.idle_unload_s)
    # Outside `image_store` on purpose: these images belong to a durable session
    # record, and the TTL purge must not be able to reach them. Where "outside"
    # is, is `playground_directory`'s to decide — never this process's CWD.
    playground = PlaygroundStore(settings_module.playground_directory(settings.server))
    scratch_dir = Path(tempfile.mkdtemp(prefix="mflux_scratch_"))
    created_at = int(time.time())
    jobs = JobManager()
    log_buffer = LogBuffer()
    pending = admin.PendingChanges()
    # A finished conversion rewrites the configuration, which is the same fact a
    # manual save reports: this process is now behind its file.
    jobs.on_config_changed = lambda: setattr(pending, "restart_required", True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.purge()
        # Before the runner starts: a generation left `running` by a previous
        # process has no way back, and a record stuck there would show as an
        # eternal spinner.
        playground.mark_interrupted()
        runner.start()
        buffer_handler = logbuffer.attach(log_buffer)
        logger.info(
            "qds %s - %d model(s): %s | default: %s",
            __version__,
            len(registry),
            ", ".join(sorted(registry)),
            settings.default_model,
        )
        if not settings.server.api_key and not settings.server.is_loopback:  # pragma: no cover
            logger.warning("Server exposed without an API key")
        yield
        # Before `engine.shutdown()`: the worker may be inside
        # `engine.generate()`, and stopping the engine under it would raise from
        # a task nobody is awaiting.
        await runner.shutdown()
        # Before `engine.shutdown()` too: a pending countdown would otherwise be
        # left dangling on a loop that is closing.
        idle_unloader.cancel()
        # Before the engine too, and not on a best-effort basis: a download or a
        # conversion that outlives this process becomes an orphan under launchd,
        # holding the HuggingFace cache and invisible to whatever starts next.
        await jobs.shutdown()
        engine.shutdown()
        shutil.rmtree(scratch_dir, ignore_errors=True)
        logbuffer.detach(buffer_handler)
        logger.info("Server stopped")

    app = FastAPI(
        title="Quantum Diffusion Server",
        version=__version__,
        lifespan=lifespan,
    )
    install_host_guard(app, settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)
    app.mount("/images", StaticFiles(directory=store.directory), name="images")
    # Playground images are *not* a mount: a session can be locked, and a file
    # must then not be served without the session's unlock token. See the
    # `/playground/images/{filename}` route below.

    # ── Authentication ─────────────────────────────────────────────────────
    #
    # One implementation, shared with `create_recovery_app`. See `qds/auth.py`
    # for why that sharing is load-bearing rather than merely tidy.

    sessions = SessionStore()
    throttle = admin.LoginThrottle()
    unlocks = playground_lock.UnlockStore()
    unlock_throttles = playground_lock.UnlockThrottles()
    require_api, require_admin = build_dependencies(settings, sessions, local_token)
    auth = Depends(require_api)

    # ── Control plane ──────────────────────────────────────────────────────

    app.include_router(
        admin.build_router(
            settings=settings,
            jobs=jobs,
            log_buffer=log_buffer,
            auth=Depends(require_admin),
            engine=engine,
            version=__version__,
            recovery_error=None,
            request_restart=request_restart or _restart_unavailable,
            pending=pending,
            sessions=sessions,
        )
    )
    app.include_router(
        admin.build_session_router(
            settings=settings, sessions=sessions, throttle=throttle, recovery_error=None
        )
    )
    mount_dashboard(app)

    # ── Helpers ────────────────────────────────────────────────────────────

    #: Public identifier → spec. Built-ins publish their catalogue key; an
    #: imported model publishes its `api_name` rather than the opaque
    #: `local-…` id it is stored under.
    by_public_name = {spec.public_name: spec for spec in registry.values()}

    def resolve_spec(key: str | None) -> ModelSpec:
        # The configured default may be an internal id — that is deliberate, and
        # `default_model` stays a durable reference rather than a friendly name
        # that would break the moment one were renamed.
        key = key or settings.default_model
        # Public name first, then the internal key. The second is an unadvertised
        # legacy path: an imported model's id was the only way to name it before
        # aliases existed, and silently breaking a script that used it would be a
        # poor trade for a listing that is already clean.
        spec = by_public_name.get(key) or registry.get(key)
        if spec is None:
            raise APIError(
                f"Unknown model: {key!r}. Available models: {sorted(by_public_name)}",
                param="model",
                code="model_not_found",
            )
        return spec

    # Constructed here rather than beside its store: the runner resolves a model
    # at execution time, not at submission time, so it needs `resolve_spec`.
    runner = PlaygroundRunner(
        playground, engine, idle_unloader, resolve_spec, upscale_catalogue.by_key
    )

    def resolve_size(spec: ModelSpec, size: str | None) -> tuple[int, int]:
        if size is None or size.lower() == "auto":
            width, height = spec.default_width, spec.default_height
        else:
            try:
                width, height = parse_size(size)
            except ValueError as exc:
                raise APIError(str(exc), param="size", code="invalid_size") from exc
        # Checked on the default too, not just on an explicit size: otherwise a
        # config-wide `default_size` outside the model's range would sail straight
        # through and fail inside mflux, after the weights were loaded.
        for label, value in (("width", width), ("height", height)):
            if value < spec.min_dimension or (spec.max_dimension and value > spec.max_dimension):
                bound = f"[{spec.min_dimension}, {spec.max_dimension or '∞'}]"
                raise APIError(
                    f"Model '{spec.key}' requires {label} in {bound}, got {value}.",
                    param="size",
                    code="invalid_size",
                )
        return width, height

    def resolve_response_format(value: str | None) -> str:
        fmt = value or settings.server.default_response_format
        if fmt not in RESPONSE_FORMATS:
            raise APIError(
                f"response_format must be one of {sorted(RESPONSE_FORMATS)}, got {fmt!r}",
                param="response_format",
                code="invalid_response_format",
            )
        return fmt

    def check_prompt(spec: ModelSpec, prompt: str) -> None:
        """Refuse a prompt the model cannot read, before any weights are loaded.

        FIBO's prompt encoder opens with a bare `json.loads(prompt)` whose result
        is discarded — a validation gate. Plain text raises a `JSONDecodeError`,
        which would reach the client as a 400 saying "Expecting value: line 1
        column 1" *after* several GB of weights had been loaded. So we say it here,
        and say what to do about it.
        """
        if "text" in spec.prompt_formats:
            # Accepting text means accepting anything: a JSON caption is text too.
            return
        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise APIError(
                f"Model '{spec.key}' only accepts a structured JSON caption as its prompt, "
                f"not plain text ({exc.msg}). Pass a JSON object describing the image - see the "
                f"model card for the schema.",
                param="prompt",
                code="prompt_must_be_json",
            ) from exc
        if not isinstance(parsed, dict):
            raise APIError(
                f"Model '{spec.key}' expects a JSON *object* as its prompt, got "
                f"{type(parsed).__name__}.",
                param="prompt",
                code="prompt_must_be_json",
            )

    def check_capabilities(spec: ModelSpec, *, negative_prompt: str | None, guidance: float | None) -> None:
        if negative_prompt and not spec.supports_negative_prompt:
            raise APIError(
                f"Model '{spec.key}' does not support negative_prompt. "
                f"Describe what you want in the prompt instead.",
                param="negative_prompt",
                code="unsupported_parameter",
            )
        if guidance is not None and not spec.supports_guidance:
            fixed = spec.default_guidance
            raise APIError(
                f"Model '{spec.key}' is distilled, so its guidance is fixed"
                + (f" at {fixed}." if fixed is not None else ".")
                + " Drop the guidance parameter.",
                param="guidance",
                code="unsupported_parameter",
            )

    def check_n(n: int) -> None:
        if n > settings.server.max_n:
            raise APIError(
                f"n={n} exceeds the server limit ({settings.server.max_n}). "
                f"Images are generated one at a time.",
                param="n",
                code="n_too_large",
            )

    def seeds_for(seed: int | None, n: int) -> list[int]:
        base = random.randint(0, MAX_SEED) if seed is None else seed
        return [(base + index) % (MAX_SEED + 1) for index in range(n)]

    def build_payload(
        request: Request,
        images: list[bytes],
        response_format: str,
        spec: ModelSpec,
        width: int,
        height: int,
        steps: int,
        seeds: list[int],
    ) -> JSONResponse:
        data: list[dict[str, Any]] = []
        for png in images:
            if response_format == "url":
                name = store.save(png)
                data.append({"url": f"{str(request.base_url).rstrip('/')}/images/{name}"})
            else:
                data.append({"b64_json": base64.b64encode(png).decode("ascii")})
        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
                # Extension: the effective size may differ from the requested
                # one (mflux truncates to a multiple of 16).
                "mflux": {
                    # The public name, matching what the request sent and what
                    # `/v1/models` lists. Echoing the internal `local-…` id here
                    # would hand a client the one identifier it must not learn.
                    "model": spec.public_name,
                    "size": f"{width}x{height}",
                    "steps": steps,
                    "seeds": seeds,
                },
            }
        )

    async def run_jobs(
        spec: ModelSpec,
        *,
        kind: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seeds: list[int],
        guidance: float | None,
        negative_prompt: str | None,
        image_path: Path | None = None,
        image_strength: float | None = None,
        steps_from_preset: bool = False,
    ) -> list[bytes]:
        # The idle countdown is armed here, on the way out, rather than inside
        # the engine: it must measure the gap between *requests*, otherwise a
        # delay of 0 would release the model between the images of a single one.
        with idle_unloader:
            images = []
            for seed in seeds:
                images.append(
                    await engine.generate(
                        GenerationJob(
                            spec=spec,
                            kind=kind,
                            prompt=prompt,
                            width=width,
                            height=height,
                            steps=steps,
                            seed=seed,
                            guidance=guidance,
                            negative_prompt=negative_prompt,
                            image_path=image_path,
                            image_strength=image_strength,
                            steps_from_preset=steps_from_preset,
                        )
                    )
                )
            return images

    # ── Endpoints ──────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "default_model": settings.default_model,
            "models": sorted(registry),
            "loaded_model": engine.loaded_model,
            # Without this, "no warm model" is indistinguishable from a bug.
            "idle_unload_s": settings.server.idle_unload_s,
            "memory": engine.memory_stats(),
        }

    @app.get("/v1/models")
    async def list_models(_: None = auth) -> dict:
        # Public names, not internal keys: `local-c1587aa663c4` is a storage
        # detail, and publishing it would make it the identifier every client
        # copied into its configuration.
        #
        # `display_name` is an extra field, like `mflux` on the single-model
        # route: the id is what a request must send, and an interface should not
        # have to invent something readable from it.
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": created_at,
                    "owned_by": "mflux",
                    "display_name": spec.display_name or name,
                }
                for name, spec in sorted(by_public_name.items())
            ],
        }

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, _: None = auth) -> dict:
        spec = resolve_spec(model_id)
        return {
            # The public name even when asked for by internal id, so a client
            # that follows the legacy alias still learns the current one.
            "id": spec.public_name,
            "object": "model",
            "created": created_at,
            "owned_by": "mflux",
            "mflux": _capabilities(spec),
        }

    @app.get("/v1/capabilities")
    async def capabilities(_: None = auth) -> dict:
        return {
            "default_model": settings.default_model,
            "max_n": settings.server.max_n,
            "response_formats": sorted(RESPONSE_FORMATS),
            "models": {key: _capabilities(spec) for key, spec in sorted(registry.items())},
        }

    @app.get("/v1/progress")
    async def progress_stream(_: None = auth) -> StreamingResponse:
        """Progress as Server-Sent Events.

        We poll `engine.progress()` rather than push from the worker: progress is
        produced on the inference thread, and a polled snapshot avoids any
        cross-thread queue, any backpressure risk and any leak when a consumer
        vanishes. Several clients can listen in parallel with no coordination.
        """
        return StreamingResponse(
            progress_events(engine),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/cancel")
    async def cancel_generation(_: None = auth) -> dict:
        """Interrupt the running generation at the next denoising step.

        MLX cannot be cancelled from outside: the stop goes through the progress
        callback, so it takes effect on the following step. The in-flight request
        ends as a 499 `generation_stopped`.
        """
        cancelled = engine.request_cancel()
        return {"cancelled": cancelled, "state": engine.progress()["state"]}

    @app.post("/v1/unload")
    async def unload_model(_: None = auth) -> dict:
        """Release the resident weights without restarting the server.

        Takes the engine lock: if a generation is running we wait for it to
        finish rather than breaking it.
        """
        await engine.unload()
        return {"loaded_model": engine.loaded_model, "memory": engine.memory_stats()}

    @app.post("/v1/images/generations")
    async def generate_images(request: Request, body: ImageGenerationRequest, _: None = auth):
        spec = resolve_spec(body.model)
        check_n(body.n)
        check_prompt(spec, body.prompt)
        check_capabilities(spec, negative_prompt=body.negative_prompt, guidance=body.guidance)
        response_format = resolve_response_format(body.response_format)
        if response_format == "raw" and body.n > 1:
            raise APIError(
                "response_format='raw' can only return a single image; use n=1 "
                "or response_format='b64_json'.",
                param="response_format",
                code="invalid_response_format",
            )

        width, height = resolve_size(spec, body.size)
        steps = body.steps or spec.default_steps
        # No explicit step count on a preset model means deferring to the preset,
        # schedule included.
        steps_from_preset = body.steps is None and spec.preset is not None
        seeds = seeds_for(body.seed, body.n)

        images = await run_jobs(
            spec,
            kind="txt2img",
            prompt=body.prompt,
            width=width,
            height=height,
            steps=steps,
            seeds=seeds,
            guidance=body.guidance,
            negative_prompt=body.negative_prompt,
            steps_from_preset=steps_from_preset,
        )

        if response_format == "raw":
            return Response(content=images[0], media_type="image/png")
        return build_payload(request, images, response_format, spec, width, height, steps, seeds)

    @app.post("/v1/images/edits")
    async def edit_images(
        request: Request,
        prompt: Annotated[str, Form()],
        image: Annotated[UploadFile, File()],
        mask: Annotated[UploadFile | None, File()] = None,
        model: Annotated[str | None, Form()] = None,
        n: Annotated[int, Form()] = 1,
        size: Annotated[str | None, Form()] = None,
        response_format: Annotated[str | None, Form()] = None,
        strength: Annotated[float | None, Form()] = None,
        steps: Annotated[int | None, Form()] = None,
        seed: Annotated[int | None, Form()] = None,
        guidance: Annotated[float | None, Form()] = None,
        negative_prompt: Annotated[str | None, Form()] = None,
        _: None = auth,
    ):
        if mask is not None:
            raise APIError(
                "No model on this server does inpainting: the mask parameter is not supported.",
                param="mask",
                code="unsupported_parameter",
            )

        spec = resolve_spec(model)
        check_n(n)
        check_prompt(spec, prompt)
        check_capabilities(spec, negative_prompt=negative_prompt, guidance=guidance)
        fmt = resolve_response_format(response_format)
        if fmt == "raw" and n > 1:
            raise APIError(
                "response_format='raw' can only return a single image.",
                param="response_format",
                code="invalid_response_format",
            )

        # img2img (noising the starting latent) and instruction editing
        # (images as conditioning tokens) are two different mechanics. An
        # explicit `strength` settles it in favour of the former.
        if strength is not None:
            kind, image_strength = "img2img", strength
        elif edit_enabled(spec):
            kind, image_strength = "edit", None
        else:
            kind, image_strength = "img2img", DEFAULT_IMAGE_STRENGTH

        if kind == "img2img" and not spec.supports_image_to_image:
            raise APIError(
                f"Model '{spec.key}' supports neither editing nor image-to-image.",
                param="model",
                code="unsupported_parameter",
            )

        width, height = resolve_size(spec, size)
        steps_val = steps or spec.default_steps
        steps_from_preset = steps is None and spec.preset is not None
        seeds = seeds_for(seed, n)

        in_file = scratch_dir / f"in_{uuid.uuid4().hex}{Path(image.filename or '').suffix or '.png'}"
        await _save_upload(image, in_file, settings.server.max_upload_mb)
        try:
            images = await run_jobs(
                spec,
                kind="edit" if kind == "edit" else "txt2img",
                prompt=prompt,
                width=width,
                height=height,
                steps=steps_val,
                seeds=seeds,
                guidance=guidance,
                negative_prompt=negative_prompt,
                image_path=in_file,
                image_strength=image_strength,
                steps_from_preset=steps_from_preset,
            )
        finally:
            in_file.unlink(missing_ok=True)

        if fmt == "raw":
            return Response(content=images[0], media_type="image/png")
        return build_payload(request, images, fmt, spec, width, height, steps_val, seeds)

    # ── Playground ─────────────────────────────────────────────────────────
    #
    # A browser control surface, so it carries the data-plane credential *and*
    # `deny_cross_site`, like the dashboard's own routes: a page on another
    # origin must not be able to spend this machine's GPU.

    playground_api = APIRouter(
        prefix="/playground/api",
        tags=["playground"],
        dependencies=[Depends(require_api), Depends(admin.deny_cross_site)],
    )

    def not_found(session_id: str) -> APIError:
        return APIError(
            f"No playground session {session_id!r}.", status_code=404, code="not_found"
        )

    def assert_unlocked(session_id: str, token: str | None) -> None:
        """404 for an unknown session, 403 `session_locked` for a protected one
        the token does not open. An open session passes with any token."""
        try:
            record = playground.password_record(session_id)
        except KeyError:
            raise not_found(session_id) from None
        if record is None:
            return
        if unlocks.session_for(token) != session_id:
            raise playground_lock.locked(session_id)

    def require_unlocked(session_id: str, x_qds_session_token: SessionToken = None) -> None:
        assert_unlocked(session_id, x_qds_session_token)

    unlocked = Depends(require_unlocked)

    @playground_api.post("/sessions", status_code=201)
    async def playground_create_session() -> dict:
        return playground.create_session()

    @playground_api.get("/sessions")
    async def playground_list_sessions() -> dict:
        # The pause rides on the list every tab already polls rather than on
        # `/v1/progress`: that stream is the *engine's* state and is shared with
        # `/v1` clients, and holding this queue is a playground control that has
        # no meaning there.
        return {"sessions": playground.list_sessions(), "paused": runner.paused}

    @playground_api.post("/queue")
    async def playground_set_queue_state(body: QueueStateRequest) -> dict:
        """Hold or release the playground queue, for every session at once.

        Not gated on a session, because it is not about one: there is a single
        FIFO worker behind every session's generations. It sits at the router's
        own auth level rather than admin's because it is reversible by anyone who
        can reach it, and that same credential already permits `/v1/cancel` and
        unbounded submission -- holding a queue is not more authority than
        emptying one. What it *is* that those are not is unbounded in time, which
        is why the state is published to every tab above.

        Idempotent, and deliberately not a claim about the engine: pausing takes
        effect at the runner's next boundary, so a 200 here does not mean nothing
        is being denoised.
        """
        await runner.set_paused(body.paused)
        return {"paused": runner.paused}

    @playground_api.get("/sessions/{session_id}", dependencies=[unlocked])
    async def playground_get_session(session_id: str) -> dict:
        detail = playground.get_session(session_id)
        if detail is None:
            raise not_found(session_id)
        return detail

    @playground_api.patch("/sessions/{session_id}", dependencies=[unlocked])
    async def playground_rename_session(session_id: str, body: RenameRequest) -> dict:
        session = playground.rename_session(session_id, body.title)
        if session is None:
            raise not_found(session_id)
        return session

    @playground_api.delete("/sessions/{session_id}", status_code=204, dependencies=[unlocked])
    async def playground_delete_session(session_id: str) -> None:
        # Stop the engine first: the worker is inside a generation whose record
        # is about to disappear, and it would otherwise keep the machine busy
        # producing images for a session nobody can see.
        await runner.cancel_running_in({session_id})
        playground.unlink(playground.delete_session(session_id))
        unlocks.revoke_session(session_id)
        unlock_throttles.forget(session_id)

    # ── Session passwords ──
    #
    # Setting, changing and removing all go through `unlocked`: on an open
    # session that passes trivially, on a protected one the token *is* the
    # proof of knowing the current password. The hash work runs in a thread —
    # ~100 ms of scrypt must not stall the event loop that serves previews.

    @playground_api.post("/sessions/{session_id}/password", dependencies=[unlocked])
    async def playground_set_password(session_id: str, body: SessionPasswordRequest) -> dict:
        try:
            record = await asyncio.to_thread(credential.hash_password, body.password)
        except credential.WeakPassword as exc:
            raise playground_lock.weak_password(str(exc)) from None
        if not playground.set_password(session_id, record):
            raise not_found(session_id)
        # Every earlier token was minted against the old password (or none);
        # the caller gets a fresh one so it stays where it is.
        unlocks.revoke_session(session_id)
        return {"token": unlocks.issue(session_id)}

    @playground_api.delete(
        "/sessions/{session_id}/password", status_code=204, dependencies=[unlocked]
    )
    async def playground_remove_password(session_id: str) -> None:
        if playground.password_record(session_id) is None:
            raise playground_lock.not_protected(session_id)
        playground.set_password(session_id, None)
        unlocks.revoke_session(session_id)
        unlock_throttles.forget(session_id)

    @playground_api.post("/sessions/{session_id}/unlock")
    async def playground_unlock(session_id: str, body: SessionPasswordRequest) -> dict:
        try:
            record = playground.password_record(session_id)
        except KeyError:
            raise not_found(session_id) from None
        if record is None:
            raise playground_lock.not_protected(session_id)
        throttle = unlock_throttles.for_session(session_id)
        wait = throttle.retry_after()
        if wait > 0:
            raise playground_lock.too_many_attempts(wait)
        if not await asyncio.to_thread(credential.verify_record, body.password, record):
            throttle.record_failure()
            logger.warning("playground: failed unlock attempt on session %s", session_id)
            raise playground_lock.invalid_password()
        throttle.record_success()
        return {"token": unlocks.issue(session_id), "session": playground.session_summary(session_id)}

    @playground_api.post("/sessions/{session_id}/lock", status_code=204)
    async def playground_lock_session(session_id: str, x_qds_session_token: SessionToken = None) -> None:
        """Give back the presented token. Only that one: another tab's unlock is
        its own; changing the password is what revokes them all."""
        try:
            playground.password_record(session_id)
        except KeyError:
            raise not_found(session_id) from None
        if unlocks.session_for(x_qds_session_token) == session_id:
            unlocks.revoke(x_qds_session_token)

    @playground_api.post("/sessions/{session_id}/generations", status_code=202, dependencies=[unlocked])
    async def playground_generate(
        session_id: str,
        prompt: Annotated[str, Form()],
        model: Annotated[str | None, Form()] = None,
        negative_prompt: Annotated[str | None, Form()] = None,
        n: Annotated[int, Form()] = 1,
        size: Annotated[str | None, Form()] = None,
        steps: Annotated[int | None, Form()] = None,
        seed: Annotated[int | None, Form()] = None,
        group: Annotated[str | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
    ) -> dict:
        spec = resolve_spec(model)
        if n < 1:
            raise APIError("n must be at least 1.", param="n", code="invalid_n")
        check_n(n)
        check_prompt(spec, prompt)
        # Blank is "none sent", not "an empty negative prompt": a browser form
        # posts the field whether or not it was typed in, and storing `""` would
        # both misreport the request and trip the capability check below on a
        # model that cannot take one.
        negative = (negative_prompt or "").strip() or None
        check_capabilities(spec, negative_prompt=negative, guidance=None)
        width, height = resolve_size(spec, size)
        if steps is not None and steps < 1:
            raise APIError("steps must be at least 1.", param="steps", code="invalid_steps")
        steps_val = steps or spec.default_steps
        if seed is not None and not (0 <= seed <= MAX_SEED):
            raise APIError(
                f"seed must be between 0 and {MAX_SEED}.", param="seed", code="invalid_seed"
            )
        seeds = seeds_for(seed, n)

        kind, image_strength = "txt2img", None
        if image is not None:
            # Same decision as `/v1/images/edits` with no explicit strength.
            if edit_enabled(spec):
                kind = "edit"
            elif spec.supports_image_to_image:
                image_strength = DEFAULT_IMAGE_STRENGTH
            else:
                raise APIError(
                    f"Model '{spec.key}' supports neither editing nor image-to-image.",
                    param="model",
                    code="unsupported_parameter",
                )

        # The upload lands directly in the playground's never-purged directory, so
        # anything that goes wrong between writing it and owning it by a row must
        # remove it: nothing else ever will. `/v1/images/edits` gets this from its
        # scratch directory; here it is explicit.
        destination: Path | None = None
        try:
            if image is not None:
                destination = playground.context_path(Path(image.filename or "").suffix)
                await _save_upload(image, destination, settings.server.max_upload_mb)
            record = playground.add_generation(
                session_id,
                prompt=prompt,
                negative_prompt=negative,
                model=spec.public_name,
                kind=kind,
                n=n,
                width=width,
                height=height,
                steps=steps_val,
                steps_from_preset=steps is None and spec.preset is not None,
                seeds=seeds,
                image_strength=image_strength,
                context_image=destination.name if destination else None,
                group=group,
            )
        except KeyError as exc:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise APIError(
                f"No playground session {session_id!r}.",
                status_code=404,
                code="not_found",
            ) from exc
        except ValueError as exc:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise APIError(
                f"No generation group {group!r} in this session.",
                param="group",
                code="invalid_group",
            ) from exc
        except BaseException:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise
        runner.submit(record["id"])
        return record

    @playground_api.get("/upscalers")
    def playground_upscalers() -> dict:
        """What the image toolbar can offer, and whether it will cost a wait.

        `downloaded` is asked of the *file*, not the repository:
        `availability.scan_repos` answers "this repo is in the cache", which is
        right for a status report and wrong here -- it would say present for a
        repo from which some other file had been pulled. This decides whether a
        click starts a download, so it asks the exact question.
        """
        from qds.upscale.weights import is_downloaded

        return {
            "upscalers": [
                {
                    "id": spec.key,
                    "name": spec.display_name,
                    "scales": list(upscale_catalogue.SCALES),
                    "downloaded": is_downloaded(spec),
                    "sizeMb": spec.size_mb,
                    "license": spec.license,
                }
                for spec in upscale_catalogue.SPECS
            ]
        }

    @playground_api.post("/sessions/{session_id}/upscales", status_code=202, dependencies=[unlocked])
    async def playground_upscale(session_id: str, body: UpscaleRequest) -> dict:
        """Enlarge an image this session already owns.

        A separate route rather than a field on `/generations`, and JSON rather
        than multipart, for the same reason: the source is by construction a
        file the server already holds and already knows the owner of. `refine`
        round-trips its PNG through the browser because a refinement may
        legitimately start from an image the server has never seen; an upscale
        cannot. Sending those bytes out and back would be pure work, and would
        put a trust boundary where there was none.
        """
        spec = upscale_catalogue.by_key(body.model)
        if spec is None:
            raise APIError(
                f"Unknown upscaler {body.model!r}. Available: "
                f"{', '.join(upscale_catalogue.KEYS)}.",
                param="model",
                code="invalid_model",
            )
        if body.scale not in upscale_catalogue.SCALES:
            raise APIError(
                f"scale must be one of {', '.join(str(s) for s in upscale_catalogue.SCALES)}.",
                param="scale",
                code="invalid_scale",
            )

        # Only a generated image, and only one of this session's. `not_found`
        # rather than `forbidden` for someone else's: the answer to "does this
        # exist" should not depend on who is asking.
        source_row = playground.generated_image(body.image)
        if source_row is None or source_row["session_id"] != session_id:
            raise APIError(
                f"No playground image {body.image!r} in this session.",
                status_code=404,
                param="image",
                code="not_found",
            )
        source = playground.images_dir / body.image
        if not source.is_file():
            raise APIError(
                f"No playground image {body.image!r} in this session.",
                status_code=404,
                param="image",
                code="not_found",
            )

        with Image.open(source) as opened:
            source_width, source_height = opened.size
        width, height = source_width * body.scale, source_height * body.scale
        # Bounded on what the network renders, which is not what was asked for:
        # it always works at `native_scale` and a smaller factor is that result
        # resampled down. See `MAX_RENDER_PIXELS`.
        rendered = source_width * source_height * spec.native_scale**2
        if rendered > upscale_catalogue.MAX_RENDER_PIXELS:
            side = source_width * spec.native_scale, source_height * spec.native_scale
            raise APIError(
                f"Upscaling {source_width}x{source_height} means rendering "
                f"{side[0]}x{side[1]} ({rendered} pixels), past the "
                f"{upscale_catalogue.MAX_RENDER_PIXELS} pixel limit. The network "
                f"always works at x{spec.native_scale}, so x{body.scale} costs the same.",
                param="image",
                code="image_too_large",
            )

        # A copy, not a reference: deleting the source image must not leave the
        # upscale pointing at a missing file. The copy lives and dies with the
        # row, through the same `context_image` cleanup every other path uses.
        destination = playground.context_path(".png")
        try:
            shutil.copyfile(source, destination)
            record = playground.add_generation(
                session_id,
                prompt=source_row["prompt"],
                model=spec.key,
                kind="upscale",
                n=1,
                width=width,
                height=height,
                steps=0,
                steps_from_preset=False,
                # `generation_images.seed` is NOT NULL, and the honest value is
                # the seed the source was generated with.
                seeds=[source_row["seed"]],
                context_image=destination.name,
                group=body.group or source_row["group_id"],
            )
        except KeyError as exc:
            destination.unlink(missing_ok=True)
            raise APIError(
                f"No playground session {session_id!r}.", status_code=404, code="not_found"
            ) from exc
        except ValueError as exc:
            destination.unlink(missing_ok=True)
            raise APIError(
                f"No generation group {body.group!r} in this session.",
                param="group",
                code="invalid_group",
            ) from exc
        except BaseException:
            # The images directory is never purged: a file no row owns stays
            # there for good.
            destination.unlink(missing_ok=True)
            raise
        runner.submit(record["id"])
        return record

    def no_generation(generation_id: str) -> APIError:
        return APIError(
            f"No playground generation {generation_id!r}.",
            status_code=404,
            code="not_found",
        )

    @playground_api.post("/generations/{generation_id}/cancel")
    async def playground_cancel(generation_id: str, x_qds_session_token: SessionToken = None) -> dict:
        session_id = playground.session_of_generation(generation_id)
        if session_id is None:
            raise no_generation(generation_id)
        assert_unlocked(session_id, x_qds_session_token)
        record = await runner.cancel(generation_id)
        if record is None:
            raise no_generation(generation_id)
        return record

    @playground_api.delete("/groups/{group_id}", status_code=204)
    async def playground_delete_group(group_id: str, x_qds_session_token: SessionToken = None) -> None:
        """Delete a whole feed entry: every generation of the lineage, and every
        file only it owned.

        The ordering is `playground_delete_session`'s, for the same reason: stop
        the engine first, because the worker may be inside a generation whose
        record is about to disappear and would otherwise keep the machine busy
        producing an image for an entry nobody can see.
        """
        session_id = playground.session_of_group(group_id)
        if session_id is None:
            raise APIError(
                f"No playground group {group_id!r}.", status_code=404, code="not_found"
            )
        assert_unlocked(session_id, x_qds_session_token)
        await runner.cancel_running_in_group(group_id)
        removed = playground.delete_group(group_id)
        if removed is None:  # pragma: no cover - deleted between the two calls
            raise APIError(
                f"No playground group {group_id!r}.", status_code=404, code="not_found"
            )
        playground.unlink(removed)

    @playground_api.get("/preview")
    async def playground_preview() -> Response:
        """The running generation's latest partially-denoised image, if there is one.

        A same-origin `<img>` sends the session cookie and no `Origin` header, so
        it satisfies both router dependencies — the same auth story as the feed's
        image fetches. 404 outside a run, or when the running job is a `/v1` one.

        The client's `?v=<preview_seq>` is a cache-buster, not a selector: the one
        slot always answers with its current frame, which can already be a newer
        one. Matching the counter exactly would fail-close a frame the client is
        entitled to whenever a fast model decodes the next one mid-fetch, and
        "latest" is what the caller wants either way.
        """
        payload = engine.preview()
        if payload is None:
            raise APIError("No preview is available.", status_code=404, code="not_found")
        return Response(payload, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def no_image(filename: str) -> APIError:
        return APIError(f"No playground image {filename!r}.", status_code=404, code="not_found")

    @playground_api.delete("/images/{filename}", status_code=204)
    async def playground_delete_image(filename: str, x_qds_session_token: SessionToken = None) -> None:
        # `unlink` runs only when a DB row matched, and rows only ever hold names
        # minted by `save_image` (`uuid4().hex + ".png"`) or `context_path`
        # (`ctx-<uuid><suffix>`), so a crafted path never reaches the filesystem.
        session_id = playground.session_of_image(filename)
        if session_id is None:
            raise no_image(filename)
        assert_unlocked(session_id, x_qds_session_token)
        matched = playground.delete_image(filename)
        if matched is None:
            raise APIError(
                f"No playground image {filename!r}.", status_code=404, code="not_found"
            )
        _session_id, group_id = matched
        # Deleting the group's last image dissolves the group itself, or an entry
        # that is nothing but a prompt would be left behind; an active member
        # keeps the group, since its image is still coming.
        playground.unlink([filename, *playground.dissolve_empty_group(group_id)])

    app.include_router(playground_api)

    @app.get(
        "/playground/images/{filename}",
        dependencies=[auth, Depends(admin.deny_cross_site)],
        include_in_schema=False,
    )
    async def playground_image(
        filename: str,
        x_qds_session_token: SessionToken = None,
        t: Annotated[str | None, Query(alias=playground_lock.UNLOCK_QUERY)] = None,
    ) -> FileResponse:
        """A session's image, behind its lock.

        A row lookup rather than a static mount: the row says which session the
        file belongs to, and whether that session is locked. It also means a
        name no row holds — a traversal, a guess — is a 404 before any path is
        built. `?t=` is accepted here and only here, because an `<img>` sends no
        headers. `no-store`, so a relocked session is not replayed from cache.
        """
        session_id = playground.session_of_image(filename)
        if session_id is None:
            raise no_image(filename)
        assert_unlocked(session_id, x_qds_session_token or t)
        path = playground.images_dir / filename
        if not path.is_file():
            raise no_image(filename)
        return FileResponse(path, headers={"Cache-Control": "private, no-store"})

    # ── Admin recovery ─────────────────────────────────────────────────────
    #
    # The one way past a session password without knowing it. Admin authority,
    # because the admin already owns the disk the database sits on; this only
    # saves them a sqlite shell.

    playground_admin = APIRouter(
        prefix="/admin/playground",
        tags=["admin"],
        dependencies=[Depends(require_admin), Depends(admin.deny_cross_site)],
    )

    @playground_admin.delete("/sessions/{session_id}/password", status_code=204)
    async def admin_strip_session_password(session_id: str) -> None:
        try:
            record = playground.password_record(session_id)
        except KeyError:
            raise not_found(session_id) from None
        if record is None:
            raise playground_lock.not_protected(session_id)
        playground.set_password(session_id, None)
        unlocks.revoke_session(session_id)
        unlock_throttles.forget(session_id)
        logger.warning("playground: session %s password removed by admin", session_id)

    app.include_router(playground_admin)

    return app


def _capabilities(spec: ModelSpec) -> dict:
    quantization = spec.quantization
    return {
        "repo": spec.repo,
        "default_size": spec.default_size,
        "default_steps": spec.default_steps,
        "default_guidance": spec.default_guidance,
        "quantize": spec.quantize,
        # Which saved representation *this running process* loaded its registry
        # with. The catalogue publishes the one the configuration currently
        # selects; the two disagreeing is precisely what a restart would fix, and
        # without this the interface had no way to tell that a variant it had
        # just activated was not yet the one being generated from.
        "active_variant": spec.prequantized_variant,
        # The quantization contract, published so the app stops keeping its own
        # copy of the bit-depth rules. `prequantized` used to stand in for all of
        # this and meant three different things at once.
        "supports_quantization": quantization.supports_quantization,
        "quantize_choices": list(quantization.quantize_choices),
        "supports_prequantize": quantization.supports_prequantize,
        "prequantize_choices": list(quantization.prequantize_choices),
        "prequantize_strategy": quantization.prequantize_strategy,
        "quantization_note": quantization.note,
        "license": spec.license,
        "gated": spec.gated,
        "prompt_formats": list(spec.prompt_formats),
        "preset": spec.preset,
        "min_dimension": spec.min_dimension,
        "max_dimension": spec.max_dimension,
        "scheduler": spec.scheduler,
        "supports_guidance": spec.supports_guidance,
        "supports_negative_prompt": spec.supports_negative_prompt,
        "supports_image_to_image": spec.supports_image_to_image,
        "supports_edit": edit_enabled(spec),
    }


async def _save_upload(upload: UploadFile, destination: Path, max_mb: float) -> None:
    """Write the upload in chunks, rejecting anything past the limit."""
    limit = int(max_mb * 1024 * 1024)
    written = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                handle.close()
                destination.unlink(missing_ok=True)
                raise APIError(
                    f"Image too large (limit: {max_mb:g} MB).",
                    status_code=413,
                    param="image",
                    code="file_too_large",
                )
            handle.write(chunk)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise APIError("The image file is empty.", param="image", code="invalid_image")


def effective_bind_host(settings: Settings, recovery_error: str | None) -> str:
    """Where the server may actually listen.

    A pure function so the decision can be tested without starting uvicorn — a
    test that cannot observe the bind proves nothing about it.

    The rule closes a hole the recovery path opened: `recovery_settings()` takes
    the host from the environment, so `QDS_SERVER_HOST=0.0.0.0` plus a config
    file that will not parse produced a **wildcard-bound, unauthenticated
    configuration writer**. Recovery mode deliberately leaves `/admin` open when
    no password is set — that is the first-run path — and the two together are a
    control plane on the network with no credential at all.

    So a recovery server binds loopback unless a password exists to protect it.
    The headless-repair case survives: a machine whose config went bad but whose
    password is intact keeps its configured address.
    """
    if recovery_error is None:
        return settings.server.host
    from qds import credential

    return settings.server.host if credential.is_set() else "127.0.0.1"


def _original_argv() -> list[str]:
    """The command to re-exec, rebuilt rather than remembered.

    `sys.argv[0]` is the console script when started as `qds serve` and
    `__main__.py` when started as `python -m qds`; neither is something to hand
    back to `execv`. `sys.executable -m qds` plus the original arguments names
    the same installation in both cases.
    """
    return [sys.executable, "-m", "qds", *sys.argv[1:]]


def create_recovery_app(
    settings: Settings,
    message: str,
    *,
    request_restart: Callable[[], None] | None = None,
    local_token: str | None = None,
) -> FastAPI:
    """The server a broken configuration gets: repairable, but not generating.

    Refusing to start at all is fail-closed and was also a trap — the screen
    that edits the configuration was served by the process the configuration
    stopped from starting, so the only way out was hand-editing JSON. This keeps
    the control plane and the dashboard up, and answers everything else with a
    503 that names the reason.
    """
    setup_logging(settings.server.log_level, settings.server.log_file, settings.server.log_json)
    jobs = JobManager()
    log_buffer = LogBuffer()
    pending = admin.PendingChanges()
    jobs.on_config_changed = lambda: setattr(pending, "restart_required", True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        buffer_handler = logbuffer.attach(log_buffer)
        logger.warning("qds %s - recovery mode: %s", __version__, message)
        yield
        await jobs.shutdown()
        logbuffer.detach(buffer_handler)
        logger.info("Server stopped")

    app = FastAPI(title="Quantum Diffusion Server (recovery)", version=__version__, lifespan=lifespan)
    install_host_guard(app, settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)

    sessions = SessionStore()
    throttle = admin.LoginThrottle()
    require_api, require_admin = build_dependencies(settings, sessions, local_token)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        # Same endpoint, same shape, different `status`. A client that polls
        # this — the menubar app, the dashboard — learns the server is up *and*
        # why it cannot generate, from the one call it already makes.
        return {
            "status": "config_error",
            "version": __version__,
            "error": message,
            "default_model": None,
            "models": [],
            "loaded_model": None,
            "idle_unload_s": None,
            "memory": {},
        }

    app.include_router(
        admin.build_router(
            settings=settings,
            jobs=jobs,
            log_buffer=log_buffer,
            auth=Depends(require_admin),
            engine=None,
            version=__version__,
            recovery_error=message,
            request_restart=request_restart or _restart_unavailable,
            pending=pending,
            sessions=sessions,
        )
    )
    app.include_router(
        admin.build_session_router(
            settings=settings, sessions=sessions, throttle=throttle, recovery_error=message
        )
    )
    app.include_router(admin.build_recovery_router(message=message, version=__version__))
    mount_dashboard(app)
    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    import argparse

    import uvicorn

    # An empty parser rather than none at all: `qds serve --port 9000` has to be
    # refused rather than ignored. Binding and configuration come from
    # `server-config.json` and the `QDS_SERVER_*` overrides, which is one
    # precedence rule; a second one spelled on the command line would be two.
    argparse.ArgumentParser(
        prog="qds serve",
        description=(
            "Run the server. Configuration comes from server-config.json "
            "(QDS_SERVER_CONFIG) and QDS_SERVER_* environment overrides."
        ),
    ).parse_args(argv)

    # *Any* unreadable configuration starts a recovery server, not only one that
    # breaks a runtime invariant. The distinction used to be load-bearing and was
    # exactly backwards: a disabled default model got a repair screen, while the
    # failures people actually produce by hand-editing JSON — a typo, an
    # out-of-range value — killed the process outright and left no way back in
    # except editing the same file again by hand.
    #
    # So three rungs, each falling to the next: strict, then lenient (which still
    # refuses a structurally invalid document), then the environment and the
    # defaults, which cannot fail and at least put the repair screen on the port
    # whatever launched this process is waiting on.
    recovery_error: str | None = None
    settings: Settings
    try:
        settings = load_settings()
    except (ConfigError, ValueError) as exc:
        recovery_error = str(exc)
        logger.error("Starting in recovery mode: %s", exc)
        try:
            settings = load_settings(strict=False)
        except (ConfigError, ValueError):
            settings = recovery_settings()

    # The server keeps whatever root it is given here for its whole lifetime:
    # mflux resolves the cache constant once, at import. Changing the setting
    # therefore takes effect for this process only on restart.
    settings.apply_hf_home()

    restart_wanted = False

    def request_restart() -> None:
        nonlocal restart_wanted
        restart_wanted = True
        server.should_exit = True

    # Issued before either app is built, and on every rung: it is the credential
    # of last resort. If the password is forgotten, or the file holding its hash
    # is the file that will not parse, this is what still lets the menubar app
    # and the CLI reach the control plane and repair things.
    local_token = issue_local_token()

    app = (
        create_recovery_app(
            settings, recovery_error, request_restart=request_restart, local_token=local_token
        )
        if recovery_error is not None
        else create_app(settings, request_restart=request_restart, local_token=local_token)
    )

    # `uvicorn.Server` rather than `uvicorn.run`, so `/admin/restart` has
    # something to set `should_exit` on and this function gets control back
    # afterwards.
    bind_host = effective_bind_host(settings, recovery_error)
    if bind_host != settings.server.host:
        logger.warning(
            "Recovery mode with no admin password: listening on %s instead of %s, "
            "because an unauthenticated control plane must not be reachable from the network.",
            bind_host,
            settings.server.host,
        )

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=bind_host,
            port=settings.server.port,
            log_level=settings.server.log_level.lower(),
            # Without this, uvicorn waits forever on in-flight connections: a
            # SIGTERM during a generation would block for up to
            # `request_timeout_s` (40 min in the shipped config). A supervisor
            # should still keep a SIGTERM → SIGKILL ladder, because a second
            # SIGTERM does not force the exit on uvicorn's side — only SIGINT
            # does.
            timeout_graceful_shutdown=settings.server.shutdown_grace_s,
            # In JSON mode, stdout is the structured-event channel: uvicorn's
            # access log, which writes plain text there, would make it
            # unparsable.
            access_log=not settings.server.log_json,
        )
    )
    server.run()

    # A token file that outlives its server would be a credential for a process
    # that no longer exists. Not before a re-exec, though: the replacement issues
    # its own, and removing it here would leave a window with none.
    if not restart_wanted:
        discard_local_token()

    if restart_wanted:
        # Re-exec rather than exit-and-be-restarted: the pid survives, so this
        # behaves the same whether the menubar app launched the server or
        # somebody typed `qds serve`. Listening sockets do not follow, because
        # Python marks its file descriptors close-on-exec (PEP 446) and uvicorn
        # has closed them by now anyway.
        logger.info("Restarting: re-executing %s", " ".join(_original_argv()))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, _original_argv())

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
