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
    #: License of the weights, as the model card states it. A string rather than a
    #: `commercial: bool`: we report a fact, we do not give legal advice.
    license: str
    #: True when the repo requires approved access. Drives the badge in the app,
    #: the warning before a download that would 401, and the explanation text —
    #: which used to hard-code `black-forest-labs/*` and drifted the moment
    #: another gated repo arrived.
    gated: bool = False
    #: Prompt formats the model actually accepts. `("json",)` means plain text is
    #: rejected — FIBO's prompt encoder starts with a bare `json.loads(prompt)`.
    #: A model that accepts text never refuses anything: a JSON string *is* text.
    prompt_formats: tuple[str, ...] = ("text",)
    quantize: int | None = None
    #: True when the weights arrive at a fixed precision — an already-quantized
    #: repo, or a layout `nn.quantize` cannot touch. Those models are skipped by
    #: the config-wide `default_quantize`: mflux keeps the stored precision and
    #: only prints a warning, so honouring the setting here would make
    #: `/v1/capabilities` lie.
    prequantized: bool = False
    #: Sampler preset, for the models whose step count and guidance schedule come
    #: as a named bundle rather than as numbers. Ideogram 4 only.
    preset: str | None = None
    #: Bounds enforced by the model itself. Checked before loading: on a 28 GB
    #: model, discovering the limit after the download is not an option.
    min_dimension: int = DIMENSION_STEP
    max_dimension: int | None = None
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
        # Non-commercial, and gated — which is why the shipped config leaves it
        # off despite it being the fastest model here.
        license="FLUX Non-Commercial",
        gated=True,
        # The BFL repo ships bf16 — 36 GB of blobs, every tensor BF16, no
        # `scales`/`biases` anywhere — so this quantization is real, not the no-op
        # it would be on a pre-quantized repo. Paid once on the first load, then
        # the model stays warm, and it roughly halves what a 9B plus its Mistral
        # text encoder hold in unified memory.
        quantize=8,
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
        license="FLUX Non-Commercial",
        gated=True,
        quantize=8,
        # The local artifact is already 8-bit: our own prequantize script made it.
        prequantized=True,
    ),
    ModelSpec(
        key="qwen-image-2512",
        family="qwen",
        repo="Qwen/Qwen-Image-2512",
        # Definitely not ModelConfig.from_name() here: resolving by name loses
        # the scheduler's sigma_* parameters
        # (mflux/models/common/resolution/config_resolution.py:112-128).
        model_config_name="qwen_image",
        # `ModelConfig.qwen_image()` points at `Qwen/Qwen-Image`, the original —
        # hence an explicit path for the 2512 release. The bf16 repo rather than
        # the 8-bit conversion: mflux keeps a stored quantization and ignores
        # `-q 4`, so only the raw weights can honour `default_quantize`.
        model_path="Qwen/Qwen-Image-2512",
        default_width=1920,
        default_height=1072,
        # The Qwen-Image-2512 card, not mflux: its MODEL_INFERENCE_STEPS says 20
        # and GUIDANCE_SCALE 3.5, which is the blanket default it applies to every
        # model. The card asks for 50 steps and cfg 4.0.
        #
        # The two are not paid for the same way here. `qwen_image.py:106-119` runs
        # the negative pass *unconditionally*, whatever the guidance, so raising
        # it to 4.0 costs nothing — unlike z-image, which skips that pass below
        # 1.0. Steps do cost: two transformer forwards each, so 50 steps on a 20B
        # is 100 passes per image. Lower `default_steps` in the config if that is
        # too slow; lowering the guidance would only cost quality.
        default_steps=50,
        default_guidance=4.0,
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="linear",
        license="Apache-2.0",
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
        # An mlx-community conversion; the license follows Tongyi-MAI upstream.
        default_width=1920,
        default_height=1072,
        default_steps=50,
        default_guidance=4.0,
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        license="Apache-2.0",
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
        license="Apache-2.0",
        quantize=8,
    ),
    ModelSpec(
        key="ernie-image",
        family="ernie",
        repo="baidu/ERNIE-Image",
        model_config_name="ernie_image",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # 50 steps and guidance 4.0: the model card, and what its own CLI forces
        # (`ernie_image_generate.py:31-33`). Its signature defaults to 8 and 1.0,
        # which are the *turbo* values — the constructor defaults to turbo too.
        default_steps=50,
        default_guidance=4.0,
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        # What the CLI pins, rather than the signature's default.
        scheduler="linear",
        license="Apache-2.0",
        quantize=8,
    ),
    ModelSpec(
        key="ernie-image-turbo",
        family="ernie",
        repo="baidu/ERNIE-Image-Turbo",
        model_config_name="ernie_image_turbo",
        model_path=None,
        default_width=1024,
        default_height=1024,
        default_steps=8,
        default_guidance=1.0,
        # mflux's own ModelConfig says `supports_guidance=True`, but its CLI calls
        # `parser.error()` on anything other than 1.0. The CLI is the behavioural
        # truth, so we reject it here with a clear 400 instead.
        supports_guidance=False,
        # At guidance <= 1.0 the negative pass is skipped outright
        # (`ernie_image.py:151`): a negative prompt would be silently inert, and
        # advertising an inert parameter is exactly what this registry avoids.
        supports_negative_prompt=False,
        supports_image_to_image=True,
        scheduler="linear",
        license="Apache-2.0",
        quantize=8,
    ),
    ModelSpec(
        key="fibo",
        family="fibo",
        repo="briaai/FIBO",
        # `ModelConfig.fibo().model_name` is already `briaai/FIBO`, so no path
        # override is needed — unlike the 8-bit conversion, which would pin the
        # precision and make `default_quantize` a no-op.
        model_config_name="fibo",
        model_path=None,
        default_width=1024,
        default_height=1024,
        default_steps=50,
        # 5.0, "base FIBO typical" per its CLI — not the 4.0 of the signature.
        default_guidance=5.0,
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        license="CC-BY-NC-4.0",
        gated=True,
        # JSON only. `PromptEncoder.encode_prompt` opens with a bare
        # `json.loads(prompt)` whose result is discarded — a validation gate. Plain
        # text raises a JSONDecodeError, so we refuse it before loading rather than
        # surface "Expecting value: line 1 column 1" after several GB.
        prompt_formats=("json",),
        quantize=8,
    ),
    ModelSpec(
        key="fibo-lite",
        family="fibo",
        repo="briaai/Fibo-lite",
        model_config_name="fibo_lite",
        model_path=None,
        default_width=1024,
        default_height=1024,
        default_steps=8,
        default_guidance=1.0,
        # `fibo.py:61-62` overrides guidance to 1.0 for this variant whatever we
        # pass: distilled, conditioning-only.
        supports_guidance=False,
        # Same as the other distilled models: at 1.0 the negative pass is skipped.
        supports_negative_prompt=False,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        license="CC-BY-NC-4.0",
        gated=True,
        prompt_formats=("json",),
        quantize=8,
    ),
    ModelSpec(
        key="ideogram-4",
        family="ideogram4",
        repo="ideogram-ai/ideogram-4-fp8",
        model_config_name="ideogram4_fp8",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # Mirrors V4_DEFAULT_20, so `/v1/capabilities` and the progress bar tell
        # the truth. `_apply_override` keeps the two in step if the preset changes.
        default_steps=20,
        # The preset carries a per-step guidance schedule (3.0 for the first
        # steps, then 7.0). Its CLI ignores `--guidance` with a warning, so we
        # refuse it with a 400 rather than accept a value that changes nothing.
        default_guidance=None,
        supports_guidance=False,
        supports_negative_prompt=False,
        # No image input at all: `generate_image` has no `image_path`.
        supports_image_to_image=False,
        # Unused — the engine drops `scheduler` for this family, since the
        # signature has no such parameter. mflux sets "linear" internally.
        scheduler="linear",
        license="Ideogram 4 Non-Commercial",
        gated=True,
        # JSON captions are what the model was trained on, but plain text is
        # accepted with a warning (`Ideogram4Caption.prepare`), unlike FIBO.
        prompt_formats=("text", "json"),
        # Every heavy component is `skip_quantization=True` and the FP8 layers are
        # `Fp8Linear(nn.Module)`, which `nn.quantize` does not touch. The knob is
        # inert, so the catalogue says so instead of advertising a bit depth.
        quantize=None,
        prequantized=True,
        preset="V4_DEFAULT_20",
        min_dimension=256,
        max_dimension=2048,
    ),
)

BASE_SPECS_BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in BASE_SPECS}

#: Step count of each Ideogram 4 sampler preset. Duplicated from
#: `mflux/models/ideogram4/model/ideogram4_scheduler/scheduler.py:50-70` rather
#: than imported, like `QUANTIZE_CHOICES`: importing mflux here would pull in
#: torch. The `--preset` flag of `mflux-generate-ideogram4` offers these three.
PRESET_STEPS: dict[str, int] = {"V4_DEFAULT_20": 20, "V4_QUALITY_48": 48, "V4_TURBO_12": 12}

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
    default_quantize: int | None = None,
) -> dict[str, ModelSpec]:
    """Apply the `server-config.json` overrides on top of the base catalogue.

    `default_size` and `default_quantize` are the config-wide settings. Both sit
    **below** the per-model overrides:

        request `size` > models.<key>.default_size > default_size > catalogue
                         models.<key>.quantize     > default_quantize > catalogue

    Keeping the per-model level under the global one is the escape hatch that
    matters: a 32B does not want the same area as a distilled 4B, so pinning
    `flux2-dev` alone stays possible without giving up a single global knob.

    `default_quantize` skips the `prequantized` models. mflux resolves a stored
    quantization in favour of the file
    (`QuantizationResolution`, rule `conflict` → action `STORED`) and merely prints
    "Ignoring -q 4", so applying the setting there would make the catalogue — and
    `/v1/capabilities` — claim a bit depth the model will not use.
    """
    overrides = overrides or {}
    unknown = set(overrides) - set(BASE_SPECS_BY_KEY)
    if unknown:
        raise ValueError(
            f"Unknown models in config: {sorted(unknown)}. Valid keys: {sorted(BASE_SPECS_BY_KEY)}"
        )

    global_size = parse_size(default_size) if default_size else None
    if default_quantize is not None and default_quantize != 0 and default_quantize not in QUANTIZE_CHOICES:
        raise ValueError(f"default_quantize must be 0 (none) or one of {list(QUANTIZE_CHOICES)}")

    registry: dict[str, ModelSpec] = {}
    for key, base in BASE_SPECS_BY_KEY.items():
        spec = base
        if global_size is not None:
            spec = replace(spec, default_width=global_size[0], default_height=global_size[1])
        if default_quantize is not None and not spec.prequantized:
            spec = replace(spec, quantize=default_quantize or None)
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
    if override.preset is not None:
        if spec.preset is None:
            raise ValueError(f"Model '{spec.key}' has no sampler presets.")
        if override.preset not in PRESET_STEPS:
            raise ValueError(
                f"Unknown preset {override.preset!r} for '{spec.key}'. "
                f"Valid presets: {sorted(PRESET_STEPS)}"
            )
        changes["preset"] = override.preset
        # The preset owns the step count, so the catalogue value has to follow —
        # otherwise the progress bar would announce 20 steps for a V4_QUALITY_48.
        # An explicit `default_steps` below still wins.
        changes["default_steps"] = PRESET_STEPS[override.preset]
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

    if family == "ernie":
        # Same trap as ZImage: one class for both variants, and the constructor
        # defaults to turbo.
        from mflux.models.ernie_image import ErnieImage

        return ErnieImage(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "fibo":
        # Not re-exported by its package — `mflux/models/fibo/__init__.py` is
        # empty — so the import goes all the way down, like QwenImage's. And the
        # class is `FIBO`, not `Fibo`.
        from mflux.models.fibo.variants.txt2img.fibo import FIBO

        return FIBO(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "ideogram4":
        from mflux.models.ideogram4 import Ideogram4

        return Ideogram4(model_config=model_config, model_path=model_path, quantize=quantize)

    raise ValueError(f"Unknown model family: {family!r}")
