"""Catalogue of the models the server exposes.

Each entry describes the mflux class to instantiate, the canonical `ModelConfig`
factory, the HuggingFace repo, the generation defaults and above all the model's
*capabilities* — that is what lets us return an explicit 400 instead of letting
mflux blow up with a 500.

mflux imports are deliberately kept inside the loaders: importing `mflux` pulls
in torch and transformers (several seconds), and the tests do not need it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mflux_server.flux2_dev import config as flux2_dev_config

#: mflux truncates every dimension down to a multiple of 16
#: (mflux/models/common/config/config.py:41-47), with no lower or upper bound.
DIMENSION_STEP = 16


@dataclass(frozen=True)
class EditSpec:
    """A model's instruction-editing variant."""

    family: str
    model_config_name: str
    model_path: str | None
    #: True when the variant reuses the txt2img model's weights (no extra
    #: download on first call).
    shares_weights: bool
    enabled_by_default: bool


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    repo: str
    model_config_name: str
    #: Passed straight to mflux. `None` = the `ModelConfig`'s canonical repo.
    model_path: str | None
    default_width: int
    default_height: int
    default_steps: int
    default_guidance: float | None
    #: False for distilled models: guidance is fixed and any other value is
    #: rejected (see mflux/models/flux2/cli/flux2_generate.py:29-33).
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


#: Defaults taken from mflux/cli/defaults/defaults.py (MODEL_INFERENCE_STEPS,
#: GUIDANCE_SCALE) and from each model's README.
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
        supports_guidance=False,  # distilled model: guidance fixed at 1.0
        supports_negative_prompt=False,  # the CLI rejects the flag outright
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
        key="flux2-dev",
        family="flux2-dev",
        repo="black-forest-labs/FLUX.2-dev",
        # No factory on `ModelConfig`: mflux 0.18.0 does not know FLUX.2-dev.
        # Resolved through `_LOCAL_MODEL_CONFIGS`.
        model_config_name="flux2_dev",
        # The upstream repo ships bf16 (~111 GB of weights): only an already
        # quantized artifact fits in unified memory, see
        # `mflux-server-prequantize`.
        model_path=flux2_dev_config.DEFAULT_MODEL_PATH,
        # 1024² rather than 1920x1072: 32B over 50 steps, area is expensive.
        default_width=1024,
        default_height=1024,
        # Base model, not step-distilled (see mflux cli/defaults/defaults.py,
        # MODEL_INFERENCE_STEPS for the flux2-klein-base entries).
        default_steps=50,
        default_guidance=4.0,
        supports_guidance=True,
        # Guidance-distilled: the scalar is embedded in the transformer, so
        # there is no CFG and therefore no negative prompt.
        supports_negative_prompt=False,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        quantize=8,
    ),
    ModelSpec(
        key="qwen-image",
        family="qwen",
        repo="mlx-community/Qwen-Image-2512-8bit",
        # Definitely not ModelConfig.from_name() here: resolving by name loses
        # the scheduler's sigma_* parameters
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
        # The repo is already 8-bit quantized in its safetensors metadata, so
        # passing quantize would be a no-op.
        quantize=None,
        edit=EditSpec(
            family="qwen-edit",
            model_config_name="qwen_image_edit",
            model_path=None,  # Qwen/Qwen-Image-Edit-2509
            shares_weights=False,  # a separate multi-GB download
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
        supports_guidance=False,  # ModelConfig.supports_guidance=False → forced to 0
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="linear",
        quantize=8,
    ),
)

BASE_SPECS_BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in BASE_SPECS}

#: Values accepted by nn.quantize through mflux (cli/defaults/defaults.py:59).
QUANTIZE_CHOICES = (3, 4, 5, 6, 8)


def normalize_dimension(value: int) -> int:
    """Truncate down to a multiple of 16, the way mflux does internally.

    We do it here so we can explicitly reject what mflux would accept while
    producing a 0 (and an obscure crash further down).
    """
    if value < DIMENSION_STEP:
        raise ValueError(f"Dimension too small: {value} (minimum {DIMENSION_STEP})")
    return DIMENSION_STEP * (value // DIMENSION_STEP)


def parse_size(size: str) -> tuple[int, int]:
    """Parse an OpenAI `"WxH"` size and normalize it. `"auto"` is handled elsewhere."""
    try:
        raw_width, raw_height = (int(part) for part in size.lower().split("x"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"size must look like 'WxH' (e.g. 1024x1024), got {size!r}") from exc
    return normalize_dimension(raw_width), normalize_dimension(raw_height)


def build_registry(
    overrides: dict[str, Any] | None = None,
    *,
    default_size: str | None = None,
) -> dict[str, ModelSpec]:
    """Apply the `server-config.json` overrides on top of the base catalogue.

    `default_size` is the config-wide resolution. It sits **below** the per-model
    overrides, so the resulting precedence is:

        request `size` > models.<key>.default_size > default_size > catalogue

    Keeping the per-model level under the global one is the escape hatch that
    matters: a 32B does not want the same area as a distilled 4B, so pinning
    `flux2-dev` alone stays possible without giving up a single global knob.
    """
    overrides = overrides or {}
    unknown = set(overrides) - set(BASE_SPECS_BY_KEY)
    if unknown:
        raise ValueError(
            f"Unknown models in config: {sorted(unknown)}. Valid keys: {sorted(BASE_SPECS_BY_KEY)}"
        )

    global_size = parse_size(default_size) if default_size else None

    registry: dict[str, ModelSpec] = {}
    for key, base in BASE_SPECS_BY_KEY.items():
        spec = base
        if global_size is not None:
            spec = replace(spec, default_width=global_size[0], default_height=global_size[1])
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
                f"Model '{spec.key}' does not support configurable guidance "
                f"(fixed value: {spec.default_guidance})."
            )
        changes["default_guidance"] = override.default_guidance
    if override.quantize is not None:
        changes["quantize"] = override.quantize or None
    if override.model_path is not None:
        changes["model_path"] = override.model_path

    if override.enable_edit is not None and spec.edit is not None:
        changes["edit"] = replace(spec.edit, enabled_by_default=override.enable_edit)

    return replace(spec, **changes)


def edit_enabled(spec: ModelSpec) -> bool:
    return spec.edit is not None and spec.edit.enabled_by_default


# ── Actually loading the models ────────────────────────────────────────────


#: Configs mflux does not know about and that we build ourselves. These are
#: factories, not instances: they only import `ModelConfig` when called.
_LOCAL_MODEL_CONFIGS: dict[str, Any] = {"flux2_dev": flux2_dev_config.flux2_dev_model_config}


def _model_config(name: str):
    factory = _LOCAL_MODEL_CONFIGS.get(name)
    if factory is not None:
        return factory()

    from mflux.models.common.config import ModelConfig

    return getattr(ModelConfig, name)()


def _require_local_artifact(spec: ModelSpec, model_path: str | None) -> None:
    """Fail early and clearly when the pre-quantized artifact is missing.

    Without this guard, `PathResolution` would fall back to the bf16 HuggingFace
    repo and start an on-the-fly quantization of ~111 GB, which would fail much
    later and far less legibly.
    """
    if model_path and Path(model_path).expanduser().exists():
        return

    from mflux_server.errors import APIError

    raise APIError(
        f"Model '{spec.key}' requires a pre-quantized artifact, missing from {model_path!r}. "
        f"Run `mflux-server-prequantize --dest {model_path}` (a one-time download of about "
        f"113 GB from {spec.repo}, yielding an artifact of about 58 GB), or set "
        f"models.{spec.key}.model_path in server-config.json.",
        status_code=503,
        error_type="server_error",
        param="model",
        code="model_not_prepared",
    )


def load_model(spec: ModelSpec, *, kind: str = "txt2img") -> Any:
    """Instantiate the mflux model matching the spec.

    Faithfully mirrors each family's CLI `main()` — that is the reference to
    compare against if results ever diverge.
    """
    if kind == "txt2img":
        family, model_config_name, model_path = spec.family, spec.model_config_name, spec.model_path
    elif kind == "edit":
        if spec.edit is None:
            raise ValueError(f"Model '{spec.key}' has no editing variant.")
        family = spec.edit.family
        model_config_name = spec.edit.model_config_name
        model_path = spec.edit.model_path
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    model_config = _model_config(model_config_name)
    quantize = spec.quantize

    if family == "flux2":
        from mflux.models.flux2.variants import Flux2Klein

        return Flux2Klein(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "flux2-dev":
        # With no local artifact, `PathResolution` would fall back to the bf16
        # repo and attempt an on-the-fly quantization of ~111 GB: better to fail
        # right here.
        _require_local_artifact(spec, model_path)
        from mflux_server.flux2_dev import Flux2Dev

        return Flux2Dev(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "flux2-edit":
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

        return Flux2KleinEdit(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "qwen":
        # QwenImage is not re-exported by mflux.models.qwen.variants.
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        return QwenImage(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "qwen-edit":
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

        return QwenImageEdit(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "z-image":
        # ZImage covers both the base and the turbo model; the ModelConfig is
        # what tells them apart, and the constructor defaults to turbo — hence
        # passing it explicitly.
        from mflux.models.z_image import ZImage

        return ZImage(model_config=model_config, model_path=model_path, quantize=quantize)

    raise ValueError(f"Unknown model family: {family!r}")
