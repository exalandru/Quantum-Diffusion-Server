"""Loading and validation of `server-config.json`.

Every key in the `server` section can be overridden by a `MFLUX_SERVER_<KEY>`
environment variable, which makes it possible to deploy the same config with a
different binding and API key.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mflux_server.registry import QUANTIZE_CHOICES, ModelSpec, build_registry

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "server-config.json"
ENV_PREFIX = "MFLUX_SERVER_"
logger = logging.getLogger("mflux_server.settings")
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
    #: Release the warm model after this many seconds without a generation.
    #: `None` never releases — the historical behaviour. `0` releases as soon as
    #: the request ends. Meant for sharing unified memory with something else, a
    #: text LLM typically; the cost is paying the load again on the next image.
    idle_unload_s: float | None = Field(default=None, ge=0)

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


#: Where `huggingface_hub` keeps its cache when nothing says otherwise. Its own
#: default, and the one every QDS install has used so far.
DEFAULT_HF_HOME = "~/.cache/huggingface"


class StorageSettings(BaseModel):
    """Where large, long-lived files live. Currently only the HuggingFace cache.

    Separate from `server` on purpose: this is about the machine's disks, not
    about the HTTP surface, and the next things to land here — an import
    directory, a conversion destination — are the same kind of fact.
    """

    #: Root for downloaded weights, i.e. `HF_HOME`. `None` keeps
    #: `huggingface_hub`'s own default, which is what every install used before
    #: this setting existed. An absolute path is required: this is resolved by
    #: processes whose working directory is `/` when launched from Finder.
    hf_home: str | None = None

    @field_validator("hf_home")
    @classmethod
    def _absolute_storage_path(cls, value: str | None) -> str | None:
        if value is None or value.strip() == "":
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"storage.hf_home must be an absolute path (got {value!r}). A relative path "
                f"would resolve against the working directory, which is '/' for an app "
                f"launched from Finder."
            )
        # Deliberately not `.resolve()`: that walks symlinks and would fail on a
        # path whose external volume is not mounted, which is a state this app
        # must be able to hold and report rather than reject.
        return str(path)


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
    #: Which saved, already-quantized variant of *this source* to generate with.
    #: `None` uses the source itself. A bit depth selects the validated artifact
    #: for the current `model_path`/repo at that precision — deliberately a
    #: separate key from `model_path`, which stays the source's identity, so
    #: activating a conversion never rewrites what the model *is*.
    prequantized_variant: int | None = None
    #: Sampler preset, for the models that have any. `ideogram-4` only, whose step
    #: count and guidance schedule come as a named bundle.
    preset: str | None = None

    @field_validator("quantize")
    @classmethod
    def _check_quantize(cls, value: int | None) -> int | None:
        if value is not None and value != 0 and value not in QUANTIZE_CHOICES:
            raise ValueError(f"quantize must be 0 (none) or one of {list(QUANTIZE_CHOICES)}")
        return value


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    #: `z-image-turbo` because a default has to work with nothing set up: its repo
    #: is neither gated nor licence-restricted (Apache-2.0), it needs no
    #: preparation step, 9 steps make it the fastest usable model here, and its
    #: native 1280x720 matches the resolution the shipped config asks for.
    default_model: str = "z-image-turbo"
    #: Config-wide generation resolution, `"WxH"`. Sits below the per-model
    #: `default_size` overrides and above the catalogue, so one knob covers every
    #: model while a single model can still be pinned. `null` keeps each model on
    #: its catalogue size. Lives here rather than under `server` because it is a
    #: generation default, not a transport setting — same level as `default_model`.
    default_size: str | None = None
    #: Config-wide quantization, same precedence as `default_size`: below the
    #: per-model `quantize`, above the catalogue. Skipped on models whose weights
    #: already carry their precision — mflux would keep the stored value anyway.
    default_quantize: int | None = Field(default=None, ge=0, le=8)
    models: dict[str, ModelOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_default_model(self) -> Settings:
        # The effective catalogue, not the built-in one: an imported model is a
        # perfectly good default, and validating against `BASE_SPECS` alone made
        # it impossible to choose one. Membership decides — never the shape of
        # the key.
        effective = self.registry(include_disabled=True)
        if self.default_model not in effective:
            raise ValueError(
                f"Unknown default_model: {self.default_model!r}. Valid keys: {sorted(effective)}. "
                f"An imported model must still be registered; if it was forgotten, choose another."
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

    @property
    def effective_hf_home(self) -> str:
        """The HuggingFace root actually in force, resolved in one place.

        Precedence: the configuration, then an inherited `HF_HOME`, then
        huggingface_hub's default. Configuration wins over the environment
        because it is the setting the user chose in the app, whereas the
        environment is whatever the launcher happened to carry.
        """
        if self.storage.hf_home:
            return self.storage.hf_home
        inherited = os.environ.get("HF_HOME")
        if inherited:
            return inherited
        return str(Path(DEFAULT_HF_HOME).expanduser())

    @property
    def effective_hub_cache(self) -> str:
        """Where the cached repositories actually are, inside the storage root.

        `HF_HOME` and the hub cache are two different directories, and the
        setting names the first. Deriving the second by appending `hub`
        unconditionally is what made a storage folder that *is* a hub cache
        report every model as missing — see `availability.hub_cache_for`.
        """
        from mflux_server.availability import hub_cache_for

        return str(hub_cache_for(self.effective_hf_home))

    def apply_hf_home(self) -> str:
        """Publish the effective root into the environment, and return it.

        This has to happen before anything imports `huggingface_hub`, which
        freezes `HF_HUB_CACHE` at import time — so it belongs at the top of each
        entry point rather than at the point of use. Setting both variables keeps
        a stale inherited `HF_HUB_CACHE` from silently winning over the root the
        user configured, and makes every consumer — this app, mflux and
        huggingface_hub alike — agree on which directory holds the weights.
        """
        root = self.effective_hf_home
        os.environ["HF_HOME"] = root
        os.environ["HF_HUB_CACHE"] = self.effective_hub_cache
        return root

    def registry(self, *, include_disabled: bool = False) -> dict[str, ModelSpec]:
        """The configured catalogue.

        `include_disabled` keeps the entries the server will not expose. Downloading
        and reporting need it: the documented workflow is to fetch a model *before*
        turning it on, and dropping disabled entries there silently discarded their
        `model_path` and `quantize` overrides — so the Models tab named one repo and
        the Install button fetched another.
        """
        from mflux_server import library

        try:
            imported = library.load()
        except library.LibraryTooNew:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.warning("Imported-model library unavailable", exc_info=True)
            imported = []
        return build_registry(
            self.models,
            default_size=self.default_size,
            default_quantize=self.default_quantize,
            include_disabled=include_disabled,
            imported=imported,
        )


def _coerce_env(raw: str, field_name: str) -> Any:
    if field_name == "cors_origins":
        return [item.strip() for item in raw.split(",") if item.strip()]
    # An empty variable is how the environment says "leave this unset": the
    # desktop app already uses it for `log_file`, and `idle_unload_s` needs it to
    # be able to mean "never" rather than fail validation.
    if raw == "" and field_name in {"api_key", "log_file", "idle_unload_s"}:
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


def configured_default_model(path: Path | None = None) -> str:
    """The `default_model` currently written down, without validating the rest.

    Read raw rather than through `load_settings`, and for a specific reason: the
    caller is `mflux-server-import`, deciding whether removing a registration
    would leave this key dangling. Going through validation would make that answer
    depend on the whole config being loadable — and the one config guaranteed to
    fail validation is the one whose `default_model` names a model that no longer
    exists, which is exactly the state this check exists to prevent.

    Same precedence as `load_settings`: the environment override wins, then the
    file, then the field's own default.
    """
    env = os.environ.get(f"{ENV_PREFIX}DEFAULT_MODEL")
    if env:
        return env
    path = path or config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # No config yet: the app writes one carrying the field default, so that is
        # what the next start will use.
        raw = {}
    except (OSError, json.JSONDecodeError) as exc:
        # Unreadable is not "unset". Answering with the default here would let a
        # caller conclude "not the default model" about a file it never read.
        raise ValueError(f"{path} could not be read: {exc}") from exc
    value = raw.get("default_model") if isinstance(raw, dict) else None
    if isinstance(value, str) and value:
        return value
    return str(Settings.model_fields["default_model"].default)


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
    for field_name in ("default_model", "default_size", "default_quantize"):
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
