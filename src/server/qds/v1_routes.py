"""The OpenAI-compatible data plane: `/health` and everything under `/v1`.

Standard endpoints: `/v1/models`, `/v1/models/{id}`, `/v1/images/generations`,
`/v1/images/edits`. Local extensions: `/v1/capabilities`, `/v1/progress`,
`/v1/cancel`, `/v1/unload`, plus the `steps`, `seed`, `guidance`,
`negative_prompt` and `strength` request fields — extra fields that the OpenAI
SDKs simply ignore.

Synchronous by design, and that is the difference from the playground: a
generation here lives exactly as long as its HTTP request, and its images are
TTL-purged out of `ImageStore`. Nothing on this plane is durable.

Every refusal comes from `Admission`, never from a rule written here: the same
question asked over `/playground/api` or MCP must get the same answer.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile, params
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from qds import __version__
from qds.admission import (
    DEFAULT_IMAGE_STRENGTH,
    MAX_SEED,
    Admission,
    _capabilities,
    _rewrite_capabilities,
    _save_upload,
)
from qds.errors import APIError
from qds.registry import edit_enabled
from qds.settings import RESPONSE_FORMATS

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


def build_v1_router(admission: Admission, auth: params.Depends) -> APIRouter:
    """`/health` and the `/v1` plane, over one application's `Admission`.

    `auth` is carried per route rather than on the router because `/health` does
    not take it: it is the one endpoint a supervisor must be able to reach
    without a credential.
    """
    router = APIRouter()
    settings = admission.settings
    registry = admission.registry
    engine = admission.engine

    @router.get("/health")
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

    @router.get("/v1/models")
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
                    "created": admission.created_at,
                    "owned_by": "mflux",
                    "display_name": spec.display_name or name,
                }
                for name, spec in sorted(admission.by_public_name.items())
            ],
        }

    @router.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, _: None = auth) -> dict:
        spec = admission.resolve_spec(model_id)
        return {
            # The public name even when asked for by internal id, so a client
            # that follows the legacy alias still learns the current one.
            "id": spec.public_name,
            "object": "model",
            "created": admission.created_at,
            "owned_by": "mflux",
            "mflux": _capabilities(spec),
        }

    @router.get("/v1/capabilities")
    async def capabilities(_: None = auth) -> dict:
        return {
            "default_model": settings.default_model,
            "max_n": settings.server.max_n,
            "response_formats": sorted(RESPONSE_FORMATS),
            "models": {key: _capabilities(spec) for key, spec in sorted(registry.items())},
            "rewrite": _rewrite_capabilities(settings),
        }

    @router.get("/v1/progress")
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

    @router.post("/v1/cancel")
    async def cancel_generation(_: None = auth) -> dict:
        """Interrupt the running generation at the next denoising step.

        MLX cannot be cancelled from outside: the stop goes through the progress
        callback, so it takes effect on the following step. The in-flight request
        ends as a 499 `generation_stopped`.
        """
        cancelled = engine.request_cancel()
        return {"cancelled": cancelled, "state": engine.progress()["state"]}

    @router.post("/v1/unload")
    async def unload_model(_: None = auth) -> dict:
        """Release the resident weights without restarting the server.

        Takes the engine lock: if a generation is running we wait for it to
        finish rather than breaking it.
        """
        await engine.unload()
        return {"loaded_model": engine.loaded_model, "memory": engine.memory_stats()}

    @router.post("/v1/images/generations")
    async def generate_images(request: Request, body: ImageGenerationRequest, _: None = auth):
        spec = admission.resolve_spec(body.model)
        admission.check_n(body.n)
        admission.check_prompt(spec, body.prompt)
        admission.check_capabilities(
            spec, negative_prompt=body.negative_prompt, guidance=body.guidance
        )
        response_format = admission.resolve_response_format(body.response_format)
        if response_format == "raw" and body.n > 1:
            raise APIError(
                "response_format='raw' can only return a single image; use n=1 "
                "or response_format='b64_json'.",
                param="response_format",
                code="invalid_response_format",
            )

        width, height = admission.resolve_size(spec, body.size)
        steps = body.steps or spec.default_steps
        # No explicit step count on a preset model means deferring to the preset,
        # schedule included.
        steps_from_preset = body.steps is None and spec.preset is not None
        seeds = admission.seeds_for(body.seed, body.n)

        images = await admission.run_jobs(
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
        return admission.build_payload(
            request, images, response_format, spec, width, height, steps, seeds
        )

    @router.post("/v1/images/edits")
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

        spec = admission.resolve_spec(model)
        admission.check_n(n)
        admission.check_prompt(spec, prompt)
        admission.check_capabilities(spec, negative_prompt=negative_prompt, guidance=guidance)
        fmt = admission.resolve_response_format(response_format)
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

        width, height = admission.resolve_size(spec, size)
        steps_val = steps or spec.default_steps
        steps_from_preset = steps is None and spec.preset is not None
        seeds = admission.seeds_for(seed, n)

        in_file = (
            admission.scratch_dir
            / f"in_{uuid.uuid4().hex}{Path(image.filename or '').suffix or '.png'}"
        )
        await _save_upload(image, in_file, settings.server.max_upload_mb)
        try:
            images = await admission.run_jobs(
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
        return admission.build_payload(request, images, fmt, spec, width, height, steps_val, seeds)

    return router
