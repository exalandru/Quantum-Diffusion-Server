"""API HTTP compatible OpenAI Images.

Endpoints standards : `/v1/models`, `/v1/models/{id}`,
`/v1/images/generations`, `/v1/images/edits`.
Extensions maison : `/health`, `/v1/capabilities`, et les champs `steps`,
`seed`, `guidance`, `negative_prompt`, `strength` dans les requêtes — champs
additionnels que les SDK OpenAI ignorent.
"""

from __future__ import annotations

import base64
import logging
import random
import secrets
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from mflux_server import __version__
from mflux_server.engine import GenerationJob, ModelEngine
from mflux_server.errors import APIError, install_exception_handlers
from mflux_server.logs import SERVER_LOGGER, setup_logging
from mflux_server.registry import ModelSpec, edit_enabled, parse_size
from mflux_server.settings import RESPONSE_FORMATS, Settings, load_settings
from mflux_server.store import ImageStore

logger = logging.getLogger(SERVER_LOGGER)

MAX_SEED = 2**32 - 1
#: Valeur utilisée quand `/v1/images/edits` fait de l'img2img sans que le
#: client ait précisé `strength` (mflux/cli/defaults/defaults.py:14).
DEFAULT_IMAGE_STRENGTH = 0.4


class ImageGenerationRequest(BaseModel):
    # `extra="ignore"` : quality, style, user, background, output_format,
    # moderation… sont acceptés et ignorés plutôt que de faire échouer la
    # requête d'un client OpenAI standard.
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    model: str | None = None
    n: int = Field(default=1, ge=1)
    size: str | None = None
    response_format: str | None = None

    # Extensions mflux
    steps: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    guidance: float | None = Field(default=None, ge=0)
    negative_prompt: str | None = None


def create_app(settings: Settings | None = None, engine: ModelEngine | None = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.server.log_level, settings.server.log_file)

    registry = settings.registry()
    if not registry:
        raise ValueError("Aucun modèle activé : vérifie la section 'models' de server.config.")

    engine = engine or ModelEngine(
        request_timeout_s=settings.server.request_timeout_s,
        progress_log_every=settings.server.progress_log_every,
    )
    store = ImageStore(
        Path(settings.server.image_store).expanduser(),
        ttl_s=settings.server.image_ttl_s,
    )
    scratch_dir = Path(tempfile.mkdtemp(prefix="mflux_scratch_"))
    created_at = int(time.time())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.purge()
        logger.info(
            "mflux-server %s — %d modèle(s) : %s | défaut : %s",
            __version__,
            len(registry),
            ", ".join(sorted(registry)),
            settings.default_model,
        )
        if not settings.server.api_key and not settings.server.is_loopback:  # pragma: no cover
            logger.warning("Serveur exposé sans clé d'API")
        yield
        engine.shutdown()
        shutil.rmtree(scratch_dir, ignore_errors=True)
        logger.info("Serveur arrêté")

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

    # ── Authentification ───────────────────────────────────────────────────

    async def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = settings.server.api_key
        if not expected:
            return
        scheme, _, token = (authorization or "").partition(" ")
        # compare_digest refuse les str non-ASCII : on compare les octets.
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token.encode("utf-8"), expected.encode("utf-8")
        ):
            raise APIError(
                "Clé d'API manquante ou invalide.",
                status_code=401,
                error_type="invalid_request_error",
                code="invalid_api_key",
            )

    auth = Depends(require_auth)

    # ── Helpers ────────────────────────────────────────────────────────────

    def resolve_spec(key: str | None) -> ModelSpec:
        key = key or settings.default_model
        spec = registry.get(key)
        if spec is None:
            raise APIError(
                f"Modèle inconnu : {key!r}. Modèles disponibles : {sorted(registry)}",
                param="model",
                code="model_not_found",
            )
        return spec

    def resolve_size(spec: ModelSpec, size: str | None) -> tuple[int, int]:
        if size is None or size.lower() == "auto":
            return spec.default_width, spec.default_height
        try:
            return parse_size(size)
        except ValueError as exc:
            raise APIError(str(exc), param="size", code="invalid_size") from exc

    def resolve_response_format(value: str | None) -> str:
        fmt = value or settings.server.default_response_format
        if fmt not in RESPONSE_FORMATS:
            raise APIError(
                f"response_format doit valoir l'un de {sorted(RESPONSE_FORMATS)}, reçu : {fmt!r}",
                param="response_format",
                code="invalid_response_format",
            )
        return fmt

    def check_capabilities(spec: ModelSpec, *, negative_prompt: str | None, guidance: float | None) -> None:
        if negative_prompt and not spec.supports_negative_prompt:
            raise APIError(
                f"Le modèle '{spec.key}' ne supporte pas negative_prompt. "
                f"Décris plutôt ce que tu veux obtenir dans le prompt.",
                param="negative_prompt",
                code="unsupported_parameter",
            )
        if guidance is not None and not spec.supports_guidance:
            fixed = spec.default_guidance
            raise APIError(
                f"Le modèle '{spec.key}' est distillé : sa guidance est figée"
                + (f" à {fixed}." if fixed is not None else ".")
                + " Retire le paramètre guidance.",
                param="guidance",
                code="unsupported_parameter",
            )

    def check_n(n: int) -> None:
        if n > settings.server.max_n:
            raise APIError(
                f"n={n} dépasse la limite du serveur ({settings.server.max_n}). "
                f"Les images sont générées séquentiellement.",
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
                # Extension : la taille effective peut différer de la taille
                # demandée (mflux tronque au multiple de 16).
                "mflux": {
                    "model": spec.key,
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
    ) -> list[bytes]:
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
            "memory": engine.memory_stats(),
        }

    @app.get("/v1/models")
    async def list_models(_: None = auth) -> dict:
        return {
            "object": "list",
            "data": [
                {"id": key, "object": "model", "created": created_at, "owned_by": "mflux"}
                for key in sorted(registry)
            ],
        }

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, _: None = auth) -> dict:
        spec = resolve_spec(model_id)
        return {
            "id": spec.key,
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

    @app.post("/v1/images/generations")
    async def generate_images(request: Request, body: ImageGenerationRequest, _: None = auth):
        spec = resolve_spec(body.model)
        check_n(body.n)
        check_capabilities(spec, negative_prompt=body.negative_prompt, guidance=body.guidance)
        response_format = resolve_response_format(body.response_format)
        if response_format == "raw" and body.n > 1:
            raise APIError(
                "response_format='raw' ne peut renvoyer qu'une seule image ; utilise n=1 "
                "ou response_format='b64_json'.",
                param="response_format",
                code="invalid_response_format",
            )

        width, height = resolve_size(spec, body.size)
        steps = body.steps or spec.default_steps
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
                "Aucun modèle de ce serveur ne fait d'inpainting : le paramètre mask n'est pas supporté.",
                param="mask",
                code="unsupported_parameter",
            )

        spec = resolve_spec(model)
        check_n(n)
        check_capabilities(spec, negative_prompt=negative_prompt, guidance=guidance)
        fmt = resolve_response_format(response_format)
        if fmt == "raw" and n > 1:
            raise APIError(
                "response_format='raw' ne peut renvoyer qu'une seule image.",
                param="response_format",
                code="invalid_response_format",
            )

        # img2img (bruitage du latent de départ) et édition instructionnelle
        # (images en tokens de conditionnement) sont deux mécaniques
        # différentes. `strength` explicite tranche pour la première.
        if strength is not None:
            kind, image_strength = "img2img", strength
        elif edit_enabled(spec):
            kind, image_strength = "edit", None
        else:
            kind, image_strength = "img2img", DEFAULT_IMAGE_STRENGTH

        if kind == "img2img" and not spec.supports_image_to_image:
            raise APIError(
                f"Le modèle '{spec.key}' ne supporte ni l'édition ni l'image-to-image.",
                param="model",
                code="unsupported_parameter",
            )

        width, height = resolve_size(spec, size)
        steps_val = steps or spec.default_steps
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
            )
        finally:
            in_file.unlink(missing_ok=True)

        if fmt == "raw":
            return Response(content=images[0], media_type="image/png")
        return build_payload(request, images, fmt, spec, width, height, steps_val, seeds)

    return app


def _capabilities(spec: ModelSpec) -> dict:
    return {
        "repo": spec.repo,
        "default_size": spec.default_size,
        "default_steps": spec.default_steps,
        "default_guidance": spec.default_guidance,
        "quantize": spec.quantize,
        "scheduler": spec.scheduler,
        "supports_guidance": spec.supports_guidance,
        "supports_negative_prompt": spec.supports_negative_prompt,
        "supports_image_to_image": spec.supports_image_to_image,
        "supports_edit": edit_enabled(spec),
    }


async def _save_upload(upload: UploadFile, destination: Path, max_mb: float) -> None:
    """Écrit l'upload par blocs, en refusant au-delà de la limite."""
    limit = int(max_mb * 1024 * 1024)
    written = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                handle.close()
                destination.unlink(missing_ok=True)
                raise APIError(
                    f"Image trop volumineuse (limite : {max_mb:g} Mo).",
                    status_code=413,
                    param="image",
                    code="file_too_large",
                )
            handle.write(chunk)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise APIError("Le fichier image est vide.", param="image", code="invalid_image")


def main() -> None:  # pragma: no cover - point d'entrée
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
