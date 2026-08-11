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
import random
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from mflux_server import __version__
from mflux_server import settings as mflux_settings
from mflux_server.engine import GenerationJob, ModelEngine
from mflux_server.errors import APIError, install_exception_handlers
from mflux_server.idle import IdleUnloader
from mflux_server.logs import SERVER_LOGGER, setup_logging
from mflux_server.registry import ModelSpec, edit_enabled, parse_size
from mflux_server.settings import RESPONSE_FORMATS, Settings, load_settings
from mflux_server.store import ImageStore

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


def create_app(settings: Settings | None = None, engine: ModelEngine | None = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.server.log_level, settings.server.log_file, settings.server.log_json)
    if mflux_settings.missing_config_path is not None:
        logger.warning(
            "No configuration file at %s: every default applies. "
            "Point MFLUX_SERVER_CONFIG at your server-config.json.",
            mflux_settings.missing_config_path,
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
    scratch_dir = Path(tempfile.mkdtemp(prefix="mflux_scratch_"))
    created_at = int(time.time())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.purge()
        logger.info(
            "mflux-server %s — %d model(s): %s | default: %s",
            __version__,
            len(registry),
            ", ".join(sorted(registry)),
            settings.default_model,
        )
        if not settings.server.api_key and not settings.server.is_loopback:  # pragma: no cover
            logger.warning("Server exposed without an API key")
        yield
        # Before `engine.shutdown()`: a pending countdown would otherwise be left
        # dangling on a loop that is closing.
        idle_unloader.cancel()
        engine.shutdown()
        shutil.rmtree(scratch_dir, ignore_errors=True)
        logger.info("Server stopped")

    app = FastAPI(
        title="mflux OpenAI-compatible server",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)
    app.mount("/images", StaticFiles(directory=store.directory), name="images")

    # ── Authentication ─────────────────────────────────────────────────────

    async def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = settings.server.api_key
        if not expected:
            return
        scheme, _, token = (authorization or "").partition(" ")
        # compare_digest rejects non-ASCII str, so we compare bytes.
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token.encode("utf-8"), expected.encode("utf-8")
        ):
            raise APIError(
                "Missing or invalid API key.",
                status_code=401,
                error_type="invalid_request_error",
                code="invalid_api_key",
            )

    auth = Depends(require_auth)

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
                f"not plain text ({exc.msg}). Pass a JSON object describing the image — see the "
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
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": created_at, "owned_by": "mflux"}
                for name in sorted(by_public_name)
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


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    settings = load_settings()
    # The server keeps whatever root it is given here for its whole lifetime:
    # mflux resolves the cache constant once, at import. Changing the setting
    # therefore takes effect for this process only on restart.
    settings.apply_hf_home()
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level.lower(),
        # Without this, uvicorn waits forever on in-flight connections: a
        # SIGTERM during a generation would block for up to `request_timeout_s`
        # (40 min in the shipped config). A supervisor should still keep a
        # SIGTERM → SIGKILL ladder, because a second SIGTERM does not force the
        # exit on uvicorn's side — only SIGINT does.
        timeout_graceful_shutdown=settings.server.shutdown_grace_s,
        # In JSON mode, stdout is the structured-event channel: uvicorn's access
        # log, which writes plain text there, would make it unparsable.
        access_log=not settings.server.log_json,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
