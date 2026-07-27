"""Chargement et validation de `server-config.json`.

Toute clé de la section `server` est surchargeable par une variable
d'environnement `MFLUX_SERVER_<CLÉ>`, ce qui permet de déployer la même
config avec un binding et une clé d'API différents.
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
#: "raw" est une extension maison : les octets PNG directement dans le corps.
RESPONSE_FORMATS = {"url", "b64_json", "raw"}


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    #: Si renseignée, exigée en `Authorization: Bearer <clé>`.
    api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    #: Borne le `n` de l'API OpenAI ; chaque image est générée séquentiellement.
    max_n: int = Field(default=4, ge=1, le=32)
    #: Délai au-delà duquel la boucle de débruitage est interrompue.
    request_timeout_s: float = Field(default=900.0, gt=0)
    #: Dossier servant les images en `response_format="url"`.
    image_store: str = "images"
    image_ttl_s: int = Field(default=3600, ge=0)
    #: Valeur retenue quand le client n'envoie pas `response_format`.
    #: "url" est le défaut de l'API OpenAI : le changer casse les SDK, qui
    #: liront `data[0].url` et trouveront `None`.
    default_response_format: str = "url"
    max_upload_mb: float = Field(default=25.0, gt=0)
    log_level: str = "INFO"
    log_file: str | None = "mflux.log"
    #: Une ligne = un objet JSON sur stderr. Destiné à un superviseur (l'app
    #: de bureau) plutôt qu'à un humain devant un terminal.
    log_json: bool = False
    #: Un log de progression toutes les N étapes de débruitage (0 = aucun).
    progress_log_every: int = Field(default=1, ge=0)
    #: Borne l'arrêt gracieux d'uvicorn, dont le défaut est une attente infinie
    #: sur les connexions en vol — soit, ici, jusqu'à `request_timeout_s`.
    shutdown_grace_s: float = Field(default=10.0, gt=0)

    @field_validator("image_store", "log_file")
    @classmethod
    def _absolute_path(cls, value: str | None) -> str | None:
        """Ancre les chemins d'écriture, qui sont relatifs au CWD par défaut.

        `image_store` et `log_file` sont créés pendant `create_app`, avant même
        que le serveur ne bind. Lancé depuis un `.app` (CWD = `/`, en lecture
        seule), un chemin relatif fait échouer le démarrage ; lancé de n'importe
        où ailleurs, il éparpille `images/` et `mflux.log` dans le dossier
        courant. On résout donc dès la validation.
        """
        if value is None or value == "":
            return value
        return str(Path(value).expanduser().resolve())

    @field_validator("default_response_format")
    @classmethod
    def _check_response_format(cls, value: str) -> str:
        if value not in RESPONSE_FORMATS:
            raise ValueError(f"default_response_format doit valoir l'un de {sorted(RESPONSE_FORMATS)}")
        return value

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"log_level invalide : {value!r}")
        return level

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS


class ModelOverride(BaseModel):
    # `model_path` tombe dans l'espace réservé `model_` de pydantic ; on le
    # libère plutôt que de renommer un champ qui parle à mflux.
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = True
    quantize: int | None = None
    #: Chemin local ou repo HF substitué à celui du catalogue. Indispensable pour
    #: `flux2-dev`, dont l'artefact pré-quantifié est propre à la machine.
    model_path: str | None = None
    default_size: str | None = None
    default_steps: int | None = Field(default=None, ge=1)
    default_guidance: float | None = Field(default=None, ge=0)
    enable_edit: bool | None = None

    @field_validator("quantize")
    @classmethod
    def _check_quantize(cls, value: int | None) -> int | None:
        if value is not None and value != 0 and value not in QUANTIZE_CHOICES:
            raise ValueError(f"quantize doit valoir 0 (aucune) ou l'un de {list(QUANTIZE_CHOICES)}")
        return value


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    default_model: str = "flux2-klein"
    models: dict[str, ModelOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_default_model(self) -> Settings:
        if self.default_model not in BASE_SPECS_BY_KEY:
            raise ValueError(
                f"default_model inconnu : {self.default_model!r}. Clés valides : {sorted(BASE_SPECS_BY_KEY)}"
            )
        override = self.models.get(self.default_model)
        if override is not None and not override.enabled:
            raise ValueError(f"default_model {self.default_model!r} est désactivé dans la section models.")
        if not self.server.is_loopback and not self.server.api_key:
            raise ValueError(
                f"host={self.server.host!r} expose le serveur hors de la machine locale : "
                f"une api_key est obligatoire (config ou {ENV_PREFIX}API_KEY)."
            )
        return self

    def registry(self) -> dict[str, ModelSpec]:
        return build_registry(self.models)


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


#: Renseigné par `load_settings` quand aucun fichier de config n'a été trouvé.
#: `setup_logging` n'est configuré qu'ensuite (create_app), donc on ne peut pas
#: journaliser au moment où on le découvre — `create_app` s'en charge.
missing_config_path: Path | None = None


def load_settings(path: Path | None = None) -> Settings:
    """Lit `server-config.json`, applique les variables d'environnement, valide.

    Une config absente n'est pas une erreur : les défauts du catalogue suffisent
    à démarrer. Mais c'est un piège une fois le paquet installé en wheel, où
    `DEFAULT_CONFIG_PATH` tombe dans `site-packages/` : on repartirait
    silencieusement sur tous les défauts. D'où `missing_config_path`, que
    `create_app` journalise en `warning`.
    """
    global missing_config_path

    path = path or config_path()
    raw: dict[str, Any] = {}
    if path.exists():
        missing_config_path = None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} n'est pas un JSON valide : {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path} doit contenir un objet JSON à la racine.")
    else:
        missing_config_path = path

    server_raw = dict(raw.get("server") or {})
    server_raw.update(_env_overrides())
    raw["server"] = server_raw

    default_model_env = os.environ.get(f"{ENV_PREFIX}DEFAULT_MODEL")
    if default_model_env:
        raw["default_model"] = default_model_env

    try:
        settings = Settings.model_validate(raw)
        # Provoque tout de suite les erreurs de surcharge de modèle plutôt
        # qu'à la première requête.
        settings.registry()
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"Configuration invalide ({path}) :\n{exc}") from exc
    return settings
