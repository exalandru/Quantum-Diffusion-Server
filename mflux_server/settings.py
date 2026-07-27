"""Loading and validation of `server-config.json`.

Every key in the `server` section can be overridden by a `MFLUX_SERVER_<KEY>`
environment variable, which makes it possible to deploy the same config with a
different binding and API key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mflux_server.registry import BASE_SPECS_BY_KEY, QUANTIZE_CHOICES, ModelSpec, build_registry

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "server-config.json"
ENV_PREFIX = "MFLUX_SERVER_"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
#: "raw" is a local extension: the PNG bytes straight in the response body.
RESPONSE_FORMATS = {"url", "b64_json", "raw"}


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    #: When set, required as `Authorization: Bearer <key>`.
    api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    #: Bounds the OpenAI `n` parameter; images are generated one at a time.
    max_n: int = Field(default=4, ge=1, le=32)
    #: Deadline past which the denoising loop is interrupted.
    request_timeout_s: float = Field(default=900.0, gt=0)
    #: Directory serving images for `response_format="url"`.
    image_store: str = "images"
    image_ttl_s: int = Field(default=3600, ge=0)
    #: Value used when the client sends no `response_format`.
    #: "url" is the OpenAI default: changing it breaks the SDKs, which read
    #: `data[0].url` and find `None`.
    default_response_format: str = "url"
    max_upload_mb: float = Field(default=25.0, gt=0)
    log_level: str = "INFO"
    log_file: str | None = "mflux.log"
    #: One line, one JSON object. Aimed at a supervisor (the desktop app)
    #: rather than a human in front of a terminal.
    log_json: bool = False
    #: One progress log every N denoising steps (0 = none).
    progress_log_every: int = Field(default=1, ge=0)
    #: Bounds uvicorn's graceful shutdown, whose default is to wait forever on
    #: in-flight connections — which here means up to `request_timeout_s`.
    shutdown_grace_s: float = Field(default=10.0, gt=0)

    @field_validator("image_store", "log_file")
    @classmethod
    def _absolute_path(cls, value: str | None) -> str | None:
        """Anchor the write paths, which are CWD-relative by default.

        `image_store` and `log_file` are created during `create_app`, before the
        server even binds. Launched from a `.app` (CWD = `/`, read-only), a
        relative path makes startup fail; launched from anywhere else, it
        scatters `images/` and `mflux.log` into the current directory. So we
        resolve them at validation time.
        """
        if value is None or value == "":
            return value
        return str(Path(value).expanduser().resolve())

    @field_validator("default_response_format")
    @classmethod
    def _check_response_format(cls, value: str) -> str:
        if value not in RESPONSE_FORMATS:
            raise ValueError(f"default_response_format must be one of {sorted(RESPONSE_FORMATS)}")
        return value

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log_level: {value!r}")
        return level

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS


class ModelOverride(BaseModel):
    # `model_path` falls into pydantic's reserved `model_` namespace; we free it
    # up rather than rename a field that speaks to mflux.
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = True
    quantize: int | None = None
    #: Local path or HF repo substituted for the catalogue's. Required for
    #: `flux2-dev`, whose pre-quantized artifact is machine-specific.
    model_path: str | None = None
    default_size: str | None = None
    default_steps: int | None = Field(default=None, ge=1)
    default_guidance: float | None = Field(default=None, ge=0)
    enable_edit: bool | None = None

    @field_validator("quantize")
    @classmethod
    def _check_quantize(cls, value: int | None) -> int | None:
        if value is not None and value != 0 and value not in QUANTIZE_CHOICES:
            raise ValueError(f"quantize must be 0 (none) or one of {list(QUANTIZE_CHOICES)}")
        return value


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    #: `qwen-image` rather than `flux2-klein`: it is already 8-bit quantized in
    #: its repo, so it works with no preparation step, and it is the only model in
    #: the catalogue that supports a negative prompt.
    default_model: str = "qwen-image"
    #: Config-wide generation resolution, `"WxH"`. Sits below the per-model
    #: `default_size` overrides and above the catalogue, so one knob covers every
    #: model while a single model can still be pinned. `null` keeps each model on
    #: its catalogue size. Lives here rather than under `server` because it is a
    #: generation default, not a transport setting — same level as `default_model`.
    default_size: str | None = None
    models: dict[str, ModelOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_default_model(self) -> Settings:
        if self.default_model not in BASE_SPECS_BY_KEY:
            raise ValueError(
                f"Unknown default_model: {self.default_model!r}. Valid keys: {sorted(BASE_SPECS_BY_KEY)}"
            )
        override = self.models.get(self.default_model)
        if override is not None and not override.enabled:
            raise ValueError(f"default_model {self.default_model!r} is disabled in the models section.")
        if not self.server.is_loopback and not self.server.api_key:
            raise ValueError(
                f"host={self.server.host!r} exposes the server beyond this machine: "
                f"an api_key is mandatory (config or {ENV_PREFIX}API_KEY)."
            )
        return self

    def registry(self) -> dict[str, ModelSpec]:
        return build_registry(self.models, default_size=self.default_size)


def _coerce_env(raw: str, field_name: str) -> Any:
    if field_name == "cors_origins":
        return [item.strip() for item in raw.split(",") if item.strip()]
    if raw == "" and field_name in {"api_key", "log_file"}:
        return None
    return raw


def _env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for field_name in ServerSettings.model_fields:
        raw = os.environ.get(f"{ENV_PREFIX}{field_name.upper()}")
        if raw is not None:
            overrides[field_name] = _coerce_env(raw, field_name)
    return overrides


def config_path() -> Path:
    return Path(os.environ.get(f"{ENV_PREFIX}CONFIG", DEFAULT_CONFIG_PATH)).expanduser()


#: Set by `load_settings` when no config file was found. `setup_logging` only
#: runs afterwards (in create_app), so we cannot log at the point of discovery —
#: `create_app` takes care of it.
missing_config_path: Path | None = None


def load_settings(path: Path | None = None) -> Settings:
    """Read `server-config.json`, apply environment overrides, validate.

    A missing config is not an error: the catalogue defaults are enough to
    start. But it is a trap once the package is installed as a wheel, where
    `DEFAULT_CONFIG_PATH` lands in `site-packages/`: we would silently fall back
    to every default. Hence `missing_config_path`, which `create_app` logs as a
    `warning`.
    """
    global missing_config_path

    path = path or config_path()
    raw: dict[str, Any] = {}
    if path.exists():
        missing_config_path = None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a JSON object at the root.")
    else:
        missing_config_path = path

    server_raw = dict(raw.get("server") or {})
    server_raw.update(_env_overrides())
    raw["server"] = server_raw

    # The generic `MFLUX_SERVER_*` loop above only covers `ServerSettings`
    # fields; these two live one level up and are special-cased.
    for field_name in ("default_model", "default_size"):
        value = os.environ.get(f"{ENV_PREFIX}{field_name.upper()}")
        if value:
            raw[field_name] = value

    try:
        settings = Settings.model_validate(raw)
        # Surface model-override errors right away rather than on the first
        # request.
        settings.registry()
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"Invalid configuration ({path}):\n{exc}") from exc
    return settings
