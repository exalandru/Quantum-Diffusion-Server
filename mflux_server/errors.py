"""Format d'erreur OpenAI et traduction des exceptions mflux.

Un client OpenAI attend `{"error": {"message", "type", "param", "code"}}`.
Le prototype renvoyait le `detail` brut de FastAPI, que les SDK n'exploitent
pas.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mflux_server.logs import SERVER_LOGGER

logger = logging.getLogger(SERVER_LOGGER)

#: Le client a fermé la connexion / la génération a été interrompue.
HTTP_CLIENT_CLOSED_REQUEST = 499


class APIError(Exception):
    """Erreur métier déjà formatée pour l'API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code


class GenerationTimeout(APIError):
    def __init__(self, seconds: float):
        super().__init__(
            f"Génération interrompue après {seconds:.0f} s (request_timeout_s).",
            status_code=504,
            error_type="server_error",
            code="timeout",
        )


def error_payload(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def _json(status_code: int, **kwargs) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_payload(**kwargs))


def translate_mflux_exception(exc: BaseException) -> APIError:
    """Traduit une exception remontée de mflux en erreur d'API.

    Références : mflux/utils/exceptions.py, ainsi que les `ValueError` et
    `FileNotFoundError` levées par la résolution de chemin et le chargement
    des poids.
    """
    from mflux.utils.exceptions import (
        InvalidBaseModel,
        MFluxException,
        ModelConfigError,
        StopImageGenerationException,
    )

    if isinstance(exc, APIError):
        return exc

    if isinstance(exc, StopImageGenerationException):
        return APIError(
            str(exc) or "Génération interrompue.",
            status_code=HTTP_CLIENT_CLOSED_REQUEST,
            error_type="server_error",
            code="generation_stopped",
        )

    if isinstance(exc, (ModelConfigError, InvalidBaseModel)):
        return APIError(str(exc), status_code=400, param="model", code="invalid_model")

    if isinstance(exc, NotImplementedError):
        # Typiquement un scheduler inconnu (config.py:159).
        return APIError(str(exc), status_code=400, code="unsupported")

    if isinstance(exc, FileNotFoundError):
        return APIError(
            f"Poids ou modèle introuvable : {exc}",
            status_code=404,
            error_type="server_error",
            code="model_not_found",
        )

    try:
        from PIL import UnidentifiedImageError

        if isinstance(exc, UnidentifiedImageError):
            return APIError(
                "Le fichier fourni n'est pas une image exploitable.",
                status_code=400,
                param="image",
                code="invalid_image",
            )
    except ImportError:  # pragma: no cover - Pillow est une dépendance dure
        pass

    if isinstance(exc, MFluxException):
        return APIError(str(exc), status_code=500, error_type="server_error", code="mflux_error")

    return APIError(
        f"{type(exc).__name__}: {exc}",
        status_code=500,
        error_type="server_error",
        code="internal_error",
    )


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("%s (%s)", exc.message, exc.code)
        return _json(
            exc.status_code,
            message=exc.message,
            error_type=exc.error_type,
            param=exc.param,
            code=exc.code,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return _json(exc.status_code, message=str(exc.detail), error_type=error_type)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = [str(part) for part in first.get("loc", []) if part not in ("body", "query")]
        return _json(
            422,
            message=first.get("msg", "Requête invalide."),
            param=".".join(location) or None,
            code="invalid_parameter",
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Erreur non gérée")
        api_error = translate_mflux_exception(exc)
        return _json(
            api_error.status_code,
            message=api_error.message,
            error_type=api_error.error_type,
            param=api_error.param,
            code=api_error.code,
        )
