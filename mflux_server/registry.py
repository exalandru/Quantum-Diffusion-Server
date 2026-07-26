"""Catalogue des modèles exposés par le serveur.

Chaque entrée décrit la classe mflux à instancier, la factory `ModelConfig`
canonique, le repo HuggingFace, les défauts de génération et surtout les
*capacités* du modèle — c'est ce qui permet de renvoyer un 400 explicite
plutôt que de laisser mflux planter en 500.

Les imports de mflux sont volontairement faits à l'intérieur des loaders :
importer `mflux` tire torch et transformers (plusieurs secondes), et les
tests n'en ont pas besoin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

#: mflux tronque toute dimension au multiple de 16 inférieur
#: (mflux/models/common/config/config.py:41-47) sans borne min ni max.
DIMENSION_STEP = 16


@dataclass(frozen=True)
class EditSpec:
    """Variante « édition instructionnelle » d'un modèle."""

    family: str
    model_config_name: str
    model_path: str | None
    #: True si la variante réutilise les poids du modèle txt2img (pas de
    #: téléchargement supplémentaire au premier appel).
    shares_weights: bool
    enabled_by_default: bool


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    repo: str
    model_config_name: str
    #: Passé tel quel à mflux. `None` = repo canonique du `ModelConfig`.
    model_path: str | None
    default_width: int
    default_height: int
    default_steps: int
    default_guidance: float | None
    #: False pour les modèles distillés : la guidance est figée, toute autre
    #: valeur est refusée (cf. mflux/models/flux2/cli/flux2_generate.py:29-33).
    supports_guidance: bool
    supports_negative_prompt: bool
    supports_image_to_image: bool
    scheduler: str
    quantize: int | None = None
    edit: EditSpec | None = None
    enabled: bool = True

    @property
    def default_size(self) -> str:
        return f"{self.default_width}x{self.default_height}"


#: Défauts issus de mflux/cli/defaults/defaults.py (MODEL_INFERENCE_STEPS,
#: GUIDANCE_SCALE) et des README par modèle.
BASE_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="flux2-klein",
        family="flux2",
        repo="black-forest-labs/FLUX.2-klein-9B",
        model_config_name="flux2_klein_9b",
        model_path=None,
        default_width=1920,
        default_height=1072,
        default_steps=4,
        default_guidance=1.0,
        supports_guidance=False,  # modèle distillé : guidance figée à 1.0
        supports_negative_prompt=False,  # la CLI refuse explicitement le flag
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        edit=EditSpec(
            family="flux2-edit",
            model_config_name="flux2_klein_9b",
            model_path=None,
            shares_weights=True,
            enabled_by_default=True,
        ),
    ),
    ModelSpec(
        key="qwen-image",
        family="qwen",
        repo="mlx-community/Qwen-Image-2512-8bit",
        # Surtout pas ModelConfig.from_name() ici : la résolution par nom perd
        # les paramètres sigma_* du scheduler
        # (mflux/models/common/resolution/config_resolution.py:112-128).
        model_config_name="qwen_image",
        model_path="mlx-community/Qwen-Image-2512-8bit",
        default_width=1920,
        default_height=1072,
        default_steps=20,
        default_guidance=3.5,
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="linear",
        # Le repo est déjà quantifié 8 bits dans ses métadonnées safetensors :
        # passer quantize serait un no-op.
        quantize=None,
        edit=EditSpec(
            family="qwen-edit",
            model_config_name="qwen_image_edit",
            model_path=None,  # Qwen/Qwen-Image-Edit-2509
            shares_weights=False,  # téléchargement séparé de plusieurs Go
            enabled_by_default=False,
        ),
    ),
    ModelSpec(
        key="z-image",
        family="z-image",
        repo="mlx-community/Z-Image-bf16",
        model_config_name="z_image",
        model_path="mlx-community/Z-Image-bf16",
        default_width=1920,
        default_height=1072,
        default_steps=50,
        default_guidance=4.0,
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        quantize=8,
    ),
    ModelSpec(
        key="z-image-turbo",
        family="z-image",
        repo="mlx-community/Z-Image-Turbo-bf16",
        model_config_name="z_image_turbo",
        model_path="mlx-community/Z-Image-Turbo-bf16",
        default_width=1280,
        default_height=720,
        default_steps=9,
        default_guidance=None,
        supports_guidance=False,  # ModelConfig.supports_guidance=False → forcée à 0
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="linear",
        quantize=8,
    ),
)

BASE_SPECS_BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in BASE_SPECS}

#: Valeurs acceptées par nn.quantize via mflux (cli/defaults/defaults.py:59).
QUANTIZE_CHOICES = (3, 4, 5, 6, 8)


def normalize_dimension(value: int) -> int:
    """Tronque au multiple de 16 inférieur, comme le fait mflux en interne.

    On le fait ici pour pouvoir refuser explicitement ce que mflux
    accepterait en produisant un 0 (et un crash obscur plus loin).
    """
    if value < DIMENSION_STEP:
        raise ValueError(f"dimension trop petite : {value} (minimum {DIMENSION_STEP})")
    return DIMENSION_STEP * (value // DIMENSION_STEP)


def parse_size(size: str) -> tuple[int, int]:
    """Parse une taille OpenAI `"WxH"` et la normalise. `"auto"` non géré ici."""
    try:
        raw_width, raw_height = (int(part) for part in size.lower().split("x"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"size doit être au format 'WxH' (ex: 1024x1024), reçu : {size!r}") from exc
    return normalize_dimension(raw_width), normalize_dimension(raw_height)


def build_registry(overrides: dict[str, Any] | None = None) -> dict[str, ModelSpec]:
    """Applique les surcharges de `server.config` sur le catalogue de base."""
    overrides = overrides or {}
    unknown = set(overrides) - set(BASE_SPECS_BY_KEY)
    if unknown:
        raise ValueError(
            f"Modèles inconnus dans la config : {sorted(unknown)}. Clés valides : {sorted(BASE_SPECS_BY_KEY)}"
        )

    registry: dict[str, ModelSpec] = {}
    for key, base in BASE_SPECS_BY_KEY.items():
        spec = base
        override = overrides.get(key)
        if override is not None:
            spec = _apply_override(spec, override)
        if spec.enabled:
            registry[key] = spec
    return registry


def _apply_override(spec: ModelSpec, override: Any) -> ModelSpec:
    changes: dict[str, Any] = {"enabled": override.enabled}

    if override.default_size is not None:
        width, height = parse_size(override.default_size)
        changes["default_width"] = width
        changes["default_height"] = height
    if override.default_steps is not None:
        changes["default_steps"] = override.default_steps
    if override.default_guidance is not None:
        if not spec.supports_guidance:
            raise ValueError(
                f"Le modèle '{spec.key}' ne supporte pas une guidance configurable "
                f"(valeur figée : {spec.default_guidance})."
            )
        changes["default_guidance"] = override.default_guidance
    if override.quantize is not None:
        changes["quantize"] = override.quantize or None

    if override.enable_edit is not None and spec.edit is not None:
        changes["edit"] = replace(spec.edit, enabled_by_default=override.enable_edit)

    return replace(spec, **changes)


def edit_enabled(spec: ModelSpec) -> bool:
    return spec.edit is not None and spec.edit.enabled_by_default


# ── Chargement effectif des modèles ────────────────────────────────────────


def _model_config(name: str):
    from mflux.models.common.config import ModelConfig

    return getattr(ModelConfig, name)()


def load_model(spec: ModelSpec, *, kind: str = "txt2img") -> Any:
    """Instancie le modèle mflux correspondant au spec.

    Réplique fidèlement le `main()` de la CLI de chaque famille — c'est la
    référence à laquelle se comparer en cas de divergence de résultat.
    """
    if kind == "txt2img":
        family, model_config_name, model_path = spec.family, spec.model_config_name, spec.model_path
    elif kind == "edit":
        if spec.edit is None:
            raise ValueError(f"Le modèle '{spec.key}' n'a pas de variante d'édition.")
        family = spec.edit.family
        model_config_name = spec.edit.model_config_name
        model_path = spec.edit.model_path
    else:
        raise ValueError(f"kind inconnu : {kind!r}")

    model_config = _model_config(model_config_name)
    quantize = spec.quantize

    if family == "flux2":
        from mflux.models.flux2.variants import Flux2Klein

        return Flux2Klein(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "flux2-edit":
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

        return Flux2KleinEdit(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "qwen":
        # QwenImage n'est pas ré-exporté par mflux.models.qwen.variants.
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        return QwenImage(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "qwen-edit":
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

        return QwenImageEdit(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "z-image":
        # ZImage sert pour la base et pour le turbo ; c'est le ModelConfig qui
        # les distingue, et son défaut de constructeur est le turbo — d'où le
        # passage explicite.
        from mflux.models.z_image import ZImage

        return ZImage(model_config=model_config, model_path=model_path, quantize=quantize)

    raise ValueError(f"Famille de modèle inconnue : {family!r}")
