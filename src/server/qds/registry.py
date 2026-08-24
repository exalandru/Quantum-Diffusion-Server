"""Catalogue of the models the server exposes.

Each entry describes the mflux class to instantiate, the canonical `ModelConfig`
factory, the HuggingFace repo, the generation defaults and above all the model's
*capabilities* — that is what lets us return an explicit 400 instead of letting
mflux blow up with a 500.

mflux imports are deliberately kept inside the loaders: importing `mflux` pulls
in torch and transformers (several seconds), and the tests do not need it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qds.anima import config as anima_config
from qds.flux2_dev import config as flux2_dev_config
from qds.logs import SERVER_LOGGER
from qds.qwen_flash import config as qwen_flash_config
from qds.sd35 import config as sd35_config

logger = logging.getLogger(f"{SERVER_LOGGER}.registry")

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
    #: Sampler preset, for the models whose step count and guidance schedule come
    #: as a named bundle rather than as numbers. Ideogram 4 only.
    preset: str | None = None
    #: Bounds enforced by the model itself. Checked before loading: on a 28 GB
    #: model, discovering the limit after the download is not an option.
    min_dimension: int = DIMENSION_STEP
    max_dimension: int | None = None
    edit: EditSpec | None = None
    enabled: bool = True
    #: Selected saved variant, in bits. `None` generates from the source.
    prequantized_variant: int | None = None
    #: Where this entry came from. Authoritative for anything that behaves
    #: differently for a local model — notably that HuggingFace download actions
    #: must key off this, not off the shape of `repo`.
    provenance: str = "built_in"
    #: For imported rows: the built-in profile the defaults were taken from.
    base_profile_key: str | None = None
    #: Human name. Every built-in sets one, because a key is an API identifier
    #: and not something to read: `qwen-image-2512` is what a request sends,
    #: `Qwen` is what a person is shown. An imported row sets it too, its key
    #: being an opaque id.
    display_name: str | None = None
    #: The public, machine-facing identifier for an imported model. `None` for a
    #: built-in, whose catalogue `key` already *is* its public name — which is
    #: exactly why an imported model may not take one.
    api_name: str | None = None
    #: Where this installation keeps generated artifacts, from
    #: `storage.cache_dir`. Carried on the spec because resolving a saved variant
    #: is a property of the model, and threading a directory through every call
    #: site is what keeps that location the configuration's rather than a
    #: module-level default nobody chose.
    cache_root: str | None = None

    @property
    def default_size(self) -> str:
        return f"{self.default_width}x{self.default_height}"

    @property
    def public_name(self) -> str:
        """What a client puts in `{"model": ...}`.

        A built-in's catalogue key doubles as its public name and always has. An
        imported model has an opaque internal id — `local-c1587aa663c4` — which is
        durable and a poor thing to type, so it publishes an alias instead. The
        id remains resolvable, quietly, for anything that already used it.
        """
        return self.api_name or self.key

    @property
    def source(self) -> str:
        """What this model *is*: its repo, or the path the user pointed it at."""
        return self.model_path or self.repo

    @property
    def effective_model_path(self) -> str | None:
        """What generation should load — the source, or a selected saved variant.

        Kept separate from `source` so the three stay distinguishable: the source
        model, the variants that exist for it, and the one representation actually
        in use.
        """
        if self.prequantized_variant is None:
            return self.model_path
        from qds import artifacts

        # Resolved through discovery rather than computed: activation has to
        # reach the artifact that *exists*, and a directory the user pointed the
        # model at can hold one that the layout would not have placed there.
        for variant in artifacts.discover_variants(self.key, self.source, base=self.cache_root):
            if variant.bits == self.prequantized_variant:
                return variant.path
        # Nothing validated at that depth: the canonical location is the honest
        # answer, and `_require_variant` names it when it refuses.
        return str(
            artifacts.artifact_dir(
                self.key, self.source, self.prequantized_variant, base=self.cache_root
            )
        )

    @property
    def quantization(self) -> QuantizationCapability:
        """Quantization facts, derived from the family rather than stored per row."""
        return capability_for(self.family)


#: Defaults taken from mflux/cli/defaults/defaults.py (MODEL_INFERENCE_STEPS,
#: GUIDANCE_SCALE) and from each model's README.
BASE_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="flux2-klein",
        display_name="Flux 2 Klein",
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
        display_name="Flux 2 Dev",
        family="flux2-dev",
        repo="black-forest-labs/FLUX.2-dev",
        # No factory on `ModelConfig`: mflux 0.19.0 does not know FLUX.2-dev.
        # Resolved through `_LOCAL_MODEL_CONFIGS`.
        model_config_name="flux2_dev",
        # The raw repository, like every other built-in. It used to be a
        # directory *our own* converter writes, which made a QDS-generated
        # artifact the model's identity: the catalogue reported the model as
        # installed when only the conversion existed, Install and Locate had
        # nothing to act on, and the 8-bit copy could never be listed as what it
        # is, one saved variant among the possible ones. Source and saved
        # representation are two facts, and this is the source.
        model_path=None,
        # 1024² rather than 1920x1072: 32B over 50 steps, area is expensive.
        default_width=1024,
        default_height=1024,
        # Base model, not step-distilled. The card's diffusers example says
        # `num_inference_steps=50` with `#28 steps can be a good trade-off`
        # beside it, and `guidance_scale=4`. mflux has no FLUX.2-dev entry to
        # follow — its `MODEL_INFERENCE_STEPS` covers only the klein rows, where
        # the base ones are 50 too (cli/defaults/defaults.py).
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
    ),
    ModelSpec(
        key="anima-turbo",
        display_name="Anima Turbo",
        family="anima",
        repo="circlestone-labs/Anima",
        # Same repository, same architecture, different checkpoint: the file is
        # chosen by `model_config_name` through `anima.config.WEIGHT_FILE_BY_CONFIG`.
        model_config_name="anima_turbo",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # "Use at CFG 1 and 8-12 steps", from the card. 10 is the middle of that
        # range. Its author recommends starting here rather than with Aesthetic:
        # "only slightly worse on average, while being very fast to generate".
        default_steps=10,
        default_guidance=1.0,
        # Distilled, and 1.0 is where it belongs — but nothing rejects another
        # value, so this reports what the model accepts rather than what the name
        # suggests. Same distinction Krea 2 Turbo makes.
        supports_guidance=True,
        # Honoured only above guidance 1.0, where an unconditional branch exists
        # to put it in. At the distilled default it is inert.
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        license="CircleStone Labs Non-Commercial",
        gated=False,
        # bf16, for the reason the Aesthetic row gives: 2B has no headroom for
        # 4-bit, and at this size there is nothing to reclaim.
        quantize=None,
        min_dimension=512,
        max_dimension=1536,
    ),
    ModelSpec(
        key="anima",
        display_name="Anima Aesthetic",
        family="anima",
        repo="circlestone-labs/Anima",
        # No factory on `ModelConfig`: mflux 0.19.0 does not know Anima, and its
        # DiT and text adapter are implemented in `qds/anima/`. Resolved through
        # `_LOCAL_MODEL_CONFIGS`.
        model_config_name="anima",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # The card asks for 30-50 at CFG 4-5 on the Aesthetic checkpoints. 30 is
        # the low end of its own range rather than this server's usual 20: at 2B
        # a step is cheap, and Anima is not step-distilled.
        default_steps=30,
        default_guidance=4.5,
        supports_guidance=True,
        # Real CFG, and therefore a real unconditional branch to put a negative
        # prompt in — two transformer passes per step above guidance 1.0.
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        # Non-commercial for the weights; the images they produce are not, per the
        # card. A fact reported, not advice — hence the plain name.
        license="CircleStone Labs Non-Commercial",
        # Ungated, unusually for a non-commercial release: the files download
        # without a token.
        gated=False,
        # bf16, chosen rather than defaulted to. At 2B the weights are ~4.2 GB,
        # so there is nothing to gain, and there is a great deal to lose: the
        # quality cliff on this model sits between 6 and 4 bits. Rendered from one
        # seed at 1280x720, bf16 and 8-bit are indistinguishable and 6-bit is
        # clean, while 4-bit produces illegible architecture and a scratchy
        # overlay across the whole frame. `models.anima.quantize` can still ask
        # for a depth; nothing config-wide can impose one.
        quantize=None,
        # 512-1536 per the card, and the DiT is patched 2x2 over an 8x downscale,
        # so a side must stay a multiple of 16 — which `DIMENSION_STEP` already is.
        min_dimension=512,
        max_dimension=1536,
    ),
    ModelSpec(
        key="sd35-medium",
        display_name="Stable Diffusion 3.5 Medium",
        family="sd35",
        repo="stabilityai/stable-diffusion-3.5-medium",
        # No factory on `ModelConfig`: mflux 0.19.0 does not know SD 3.5, and its
        # MMDiT-X transformer and both CLIP towers are implemented in `qds/sd35/`.
        # Resolved through `_LOCAL_MODEL_CONFIGS`.
        model_config_name="sd35_medium",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # `num_inference_steps=40, guidance_scale=4.5`, from the card's own diffusers
        # example. Higher than this server's usual 20-30 because Medium is 2.5B and
        # undistilled; the card asks for 40 and a step here is cheap.
        default_steps=40,
        default_guidance=4.5,
        supports_guidance=True,
        # Real CFG, and therefore a real unconditional branch to put a negative prompt
        # in — two transformer passes per step above guidance 1.0.
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        license="Stability AI Community",
        gated=True,
        # bf16. The whole model is ~16.3 GB and the transformer 4.94 of that, so
        # quantizing on the way in is affordable and nothing has to be reclaimed.
        # Pre-quantization is still offered — the family supports it — but it is a
        # convenience here rather than the only way to run the model.
        quantize=None,
        # `pos_embed_max_size` is 384 latent positions, so the positional table itself
        # bounds a side at 3072px; the card's progressive training tops out at 1440.
        # 1536 is this server's step-aligned bound inside both.
        min_dimension=512,
        max_dimension=1536,
    ),
    ModelSpec(
        key="sd35-large",
        display_name="Stable Diffusion 3.5 Large",
        family="sd35",
        repo="stabilityai/stable-diffusion-3.5-large",
        model_config_name="sd35_large",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # `num_inference_steps=28, guidance_scale=3.5`, from the card's example.
        default_steps=28,
        default_guidance=3.5,
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        license="Stability AI Community",
        gated=True,
        # 8-bit by default: ~27.6 GB at bf16, and quantizing on the fly needs the bf16
        # and the 8-bit copy resident together — about 41 GB. That fits 64 GB and does
        # not fit 32. A pre-quantized artifact avoids the peak entirely, which is why
        # this row is the one the conversion path is for.
        quantize=8,
        # `pos_embed_max_size` is 192 latent positions here, not Medium's 384, so
        # 192 * 2 * 8 = 3072 latent-to-pixel gives a *hard* 1536px ceiling: above it
        # the transformer has no positional row to crop and refuses.
        min_dimension=512,
        max_dimension=1536,
    ),
    ModelSpec(
        key="sd35-large-turbo",
        display_name="Stable Diffusion 3.5 Large Turbo",
        family="sd35",
        repo="stabilityai/stable-diffusion-3.5-large-turbo",
        model_config_name="sd35_large_turbo",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # `num_inference_steps=4, guidance_scale=0.0`, from the card. Adversarially
        # distilled: four steps and no classifier-free guidance at all.
        default_steps=4,
        default_guidance=0.0,
        # False, unlike Anima Turbo's: this is not a preference the model tolerates
        # being overridden. Guidance above 1.0 turns on an unconditional branch the
        # distillation removed the need for, and the result is visibly over-cooked.
        # The catalogue reports what the model accepts, so a request naming guidance
        # is refused with a 400 rather than quietly honoured.
        supports_guidance=False,
        # No unconditional branch exists at guidance 0.0, so a negative prompt would
        # have nowhere to go. Refused rather than silently ignored.
        supports_negative_prompt=False,
        supports_image_to_image=True,
        scheduler="flow_match_euler_discrete",
        license="Stability AI Community",
        gated=True,
        quantize=8,
        min_dimension=512,
        max_dimension=1536,
    ),
    ModelSpec(
        key="krea-2-turbo",
        display_name="Krea 2 Turbo",
        family="krea2",
        repo="krea/Krea-2-Turbo",
        # Turbo, not Raw. Krea publishes Raw as the base checkpoint to fine-tune
        # and post-train on, and says plainly it is not the one to run inference
        # with; Turbo is the distilled inference model. mflux carries configs for
        # both (`ModelConfig.krea2()` / `.krea2_raw()`), so Raw remains one row
        # away should a reason to serve it appear.
        model_config_name="krea2",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # 8 steps: mflux's own registry value for this model
        # (cli/defaults/defaults.py, MODEL_INFERENCE_STEPS["krea-2"]), and the
        # reference count for the distilled checkpoint.
        default_steps=8,
        # 1.0, from the family's CLI `DEFAULT_GUIDANCE` — the distilled default.
        default_guidance=1.0,
        # True, unlike the other distilled models here, and the distinction is
        # mflux's rather than a judgement call: `ModelConfig.krea2()` publishes
        # `supports_guidance=True`, and nothing rejects another value. FLUX.2
        # klein is the contrasting case — its CLI calls `parser.error()` on any
        # guidance but 1.0 — which is what this flag was introduced to report.
        supports_guidance=True,
        # Accepted, and honoured whenever guidance is not 1.0: the prompt encoder
        # builds the unconditional branch only then (`krea2_text_encoder/
        # prompt_encoder.py`). At the default of 1.0 it is inert, and mflux warns
        # rather than refusing. Reported as supported for the same reason z-image
        # does with the same conditional shape: the model does take one.
        supports_negative_prompt=True,
        supports_image_to_image=True,
        # `Krea2._resolve_scheduler` maps `None`/`"linear"` onto `er_sde`, the
        # sampler Krea publishes for this checkpoint. `"euler"` is the only other
        # value it accepts.
        scheduler="linear",
        # Free commercially below Krea's revenue and seat thresholds, which is a
        # condition this server cannot evaluate — hence the plain name, and the
        # shipped config leaving it off.
        license="Krea 2 Community",
        gated=True,
        quantize=8,
    ),
    ModelSpec(
        key="qwen-image-flash",
        display_name="Qwen Image Flash",
        family="qwen",
        repo="nvidia/Qwen-Image-Flash",
        # Qwen-Image's architecture with Flash's noise schedule. mflux's own
        # `qwen_image` config would apply dynamic shifting and a terminal stretch
        # this distillation was not trained against — see `qds/qwen_flash/`.
        model_config_name="qwen_image_flash",
        model_path=None,
        default_width=1280,
        default_height=720,
        # Four steps at CFG 1.0, from NVIDIA's own example — this is a
        # few-step distillation, and the card's code passes exactly these.
        default_steps=4,
        default_guidance=1.0,
        # `true_cfg_scale=1.0` in the card's example, but nothing rejects another
        # value: the Qwen pipeline runs its negative pass unconditionally.
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="linear",
        # NVIDIA Open Model License, per the repository. Ungated.
        license="NVIDIA Open Model",
        gated=False,
        # The raw bf16 release, so the setting below has something to act on:
        # unquantized this is a 20B at ~55 GB, the same arithmetic as the 2512 row.
        quantize=8,
    ),
    ModelSpec(
        key="qwen-image-2512",
        display_name="Qwen",
        family="qwen",
        repo="Qwen/Qwen-Image-2512",
        # Definitely not ModelConfig.from_name() here: resolving by name loses
        # the scheduler's sigma_* parameters
        # (mflux/models/common/resolution/config_resolution.py:112-128).
        model_config_name="qwen_image",
        # `ModelConfig.qwen_image()` points at `Qwen/Qwen-Image-2512` as of mflux
        # 0.19.0, which is this row's repo — so the explicit path now restates
        # the factory rather than correcting it. It stays: which weights a
        # catalogue row serves is this file's statement to make, and mflux moved
        # that name once already (0.18.0 resolved it to `Qwen/Qwen-Image`, the
        # original). The bf16 repo rather than the 8-bit conversion: mflux keeps
        # a stored quantization and ignores `-q`, so only the raw weights can
        # honour the `quantize` below.
        model_path="Qwen/Qwen-Image-2512",
        default_width=1920,
        default_height=1072,
        # The Qwen-Image-2512 card, not mflux: its MODEL_INFERENCE_STEPS says 20
        # and GUIDANCE_SCALE 3.5, which is the blanket default it applies to every
        # model and which Qwen never published. The card's own example is
        # `num_inference_steps=50, true_cfg_scale=4.0`, and that is what this row
        # follows.
        #
        # Note the parameter name: `true_cfg_scale`, not `guidance_scale`. This is
        # real CFG, a second pass — `qwen_image.py:106-119` runs the negative pass
        # *unconditionally*, whatever the value, unlike z-image, which skips it
        # below 1.0. So 50 steps on a 20B is 100 transformer forwards per image.
        # That is the card's price, and it is expensive; lower `default_steps` in
        # the config to trade quality for time.
        default_steps=50,
        default_guidance=4.0,
        supports_guidance=True,
        supports_negative_prompt=True,
        supports_image_to_image=True,
        scheduler="linear",
        license="Apache-2.0",
        # 8, stated here rather than left to a config-wide setting. This row
        # points at the raw bf16 repository *so that* the runtime setting has
        # something to quantize; unquantized it is ~55 GB resident, which fits a
        # 103 GB machine and not a 32 GB one. Leaving it unset would have made
        # that the default the day the global was removed.
        quantize=8,
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
        display_name="Z-Image",
        family="z-image",
        repo="mlx-community/Z-Image-bf16",
        model_config_name="z_image",
        model_path="mlx-community/Z-Image-bf16",
        # An mlx-community conversion; the license follows Tongyi-MAI upstream.
        default_width=1920,
        default_height=1072,
        # The only card here that publishes a *range* rather than an example:
        # "Inference steps: 28 - 50" and "Guidance scale: 3.0 - 5.0", with its
        # code sample at 50 and 4. This row takes the top of the step range, as
        # mflux does (`MODEL_INFERENCE_STEPS["z-image"] = 50`). A default below
        # 28 would sit outside what the model's authors say the model is for,
        # which is a different thing from spending less than an example suggests.
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
        display_name="Z-Image Turbo",
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
        display_name="Ernie",
        family="ernie",
        repo="baidu/ERNIE-Image",
        model_config_name="ernie_image",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # 50 steps and guidance 4.0: the card says both in prose ("typically 50
        # inference steps", "Guidance scale: 4.0") and in its diffusers example,
        # and mflux agrees. The two live in different places there, which is why
        # naming one file would be wrong: guidance is set in
        # `ernie_image/cli/ernie_image_generate.py:22,30-31`, the step count in
        # `cli/defaults/defaults.py:32`.
        #
        # The signature's 8 and 1.0 are the *turbo* values — the constructor
        # defaults to turbo too, so neither is this model's.
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
        display_name="Ernie Turbo",
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
        display_name="FIBO",
        family="fibo",
        repo="briaai/FIBO",
        # `ModelConfig.fibo().model_name` is already `briaai/FIBO`, so no path
        # override is needed — unlike the 8-bit conversion, which would pin the
        # precision and make this row's `quantize` a no-op.
        model_config_name="fibo",
        model_path=None,
        default_width=1024,
        default_height=1024,
        # The card has no prose recommendation at all; its three examples
        # (Generate, Refine, Inspire) all call `num_inference_steps=50,
        # guidance_scale=5`. mflux matches on both.
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
        display_name="FIBO Lite",
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
        display_name="Ideogram 4",
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

#: The conversion route. `QDS_MEMORY_BOUNDED` is now the only one: one component
#: loaded, quantized, saved and released at a time, so peak memory is bounded by
#: the largest single component rather than by the whole model.
#:
#: `MFLUX_SAVE` was the other — load the model at a bit depth and call its own
#: `save_model`, which holds every component resident to write them. Nothing
#: selects it any more, and the name survives because artifacts converted by it
#: recorded it in their completion markers; those artifacts are still valid, and
#: reading their strategy back must not become an unknown value.
STRATEGY_MFLUX_SAVE = "mflux_save"
STRATEGY_QDS_MEMORY_BOUNDED = "qds_memory_bounded"


@dataclass(frozen=True)
class QuantizationCapability:
    """What quantization means for one model family, as facts rather than a flag.

    `prequantized` used to carry three unrelated claims at once — "the runtime
    setting does nothing", "skip the global default", and "this came from our own
    converter" — which is why the Configuration form happily offered a bit depth
    for `ideogram-4` that mflux discards.
    """

    #: Does the runtime `quantize` setting change anything? False when
    #: `nn.quantize` cannot touch the weights that matter, and false when the
    #: weights already carry a stored precision that mflux resolves in their
    #: favour (`QuantizationResolution`, rule `conflict` → action `STORED`).
    supports_quantization: bool
    #: Bit depths worth offering at load time. Empty when unsupported.
    quantize_choices: tuple[int, ...]
    #: Can this be converted into a saved, already-quantized artifact?
    supports_prequantize: bool
    prequantize_choices: tuple[int, ...]
    prequantize_strategy: str | None
    #: Why, when a capability is off. Shown by the UI instead of a dead control.
    note: str | None = None


#: Runtime quantization is inert here: `Ideogram4WeightDefinition` marks
#: `conditional_transformer`, `unconditional_transformer` and `text_encoder`
#: `skip_quantization=True` — verified against mflux 0.19.0 — leaving only the VAE,
#: and the FP8 layers are `Fp8Linear`, which `nn.quantize` does not convert.
#: Saving would be worse than useless: `ModelSaver` would stamp
#: `quantization_level` onto weights nothing had quantized, and the reload path
#: would then try to build a quantized structure for them.
_IDEOGRAM4 = QuantizationCapability(
    supports_quantization=False,
    quantize_choices=(),
    supports_prequantize=False,
    prequantize_choices=(),
    prequantize_strategy=None,
    note="Ideogram 4 ships FP8 weights that mflux does not re-quantize.",
)

#: FLUX.2-dev is loaded from an artifact our own converter produced, so the
#: stored precision always wins over the runtime setting — hence
#: `supports_quantization=False`, for a different reason than Ideogram's.
#: Conversion itself is supported, by the memory-bounded path only: the generic
#: `mflux-save` dispatches `"flux.2"` to `Flux2Klein`, which is the wrong
#: architecture for this model.
_FLUX2_DEV = QuantizationCapability(
    supports_quantization=False,
    quantize_choices=(),
    supports_prequantize=True,
    prequantize_choices=QUANTIZE_CHOICES,
    prequantize_strategy=STRATEGY_QDS_MEMORY_BOUNDED,
    note="Loaded from a pre-quantized artifact, whose stored precision mflux keeps.",
)

#: Krea 2 quantizes at load like any other family — `WeightApplier` walks its
#: definition and `Krea2WeightDefinition.quantization_predicate` decides, keeping
#: the text encoder out of it (`skip_quantization=True`) and skipping the layers
#: whose last dimension is not a multiple of 64. Conversion is the part that does
#: not fit, and not for want of a `save_model`: the class has one.
#:
#: Its transformer ships as a single file at the repo root, so its
#: `ComponentDefinition.hf_subdir` is `""` — while the converter is built on one
#: component per subdirectory, named after the component. `single_component_definition`
#: would ask for `"/*.safetensors"`, which matches nothing; `availability` would
#: look for a `transformer/` directory the saver never wrote. Both are structural,
#: so this family refuses conversion rather than producing an artifact that cannot
#: be found again.
_SINGLE_FILE_COMPONENT = QuantizationCapability(
    supports_quantization=True,
    quantize_choices=QUANTIZE_CHOICES,
    supports_prequantize=False,
    prequantize_choices=(),
    prequantize_strategy=None,
    note="Quantized as it loads. This model keeps its transformer in one "
    "repository-root file, which the saved-artifact layout cannot represent.",
)

#: Anima quantizes at load like any other family, and its predicate skips the
#: layers MLX cannot group into 64s. Conversion is refused for a reason of the
#: same kind as Krea 2's, though the shape of it differs: Anima's components do
#: not sit one per subdirectory named after the component. Its transformer and
#: text adapter are two halves of *one* file under `split_files/diffusion_models`,
#: split by key prefix, and its VAE comes from a different repository altogether.
#: The converter writes and finds artifacts by component subdirectory, so there is
#: no layout for it to write this family into.
#:
#: It also has little to gain: 2B at bf16 is ~4.2 GB, and the whole point of the
#: saved-artifact path is models too large to quantize on the way in.
_ANIMA = QuantizationCapability(
    supports_quantization=True,
    quantize_choices=QUANTIZE_CHOICES,
    supports_prequantize=False,
    prequantize_choices=(),
    prequantize_strategy=None,
    note="No saved copy is needed here. A pre-quantized artifact and the runtime "
    "setting run the same quantizer over the same layers, so they produce the "
    "same weights — a saved copy only spares the conversion on models too large "
    "to quantize while loading, and this one is 4.2 GB. Set the bit depth under "
    "Runtime quantization instead; the default is bf16, and below 6 bits this "
    "model degrades badly. (Its components could not be saved in that layout "
    "anyway: two of them share one file, and its VAE comes from a second "
    "repository.)",
)

#: Families whose class defines its own `save_model` and whose components are
#: quantizable. Verified per family against mflux 0.19.0: `save_model` present on
#: the exact class `load_model` instantiates, no `skip_quantization` on the
#: transformer, and `ModelSaver`/`WeightLoader` round-trip the level generically.
#:
#: They convert component by component like FLUX.2-dev does. Whether a model
#: *fits* in memory was never the right test for how to convert it: loading a
#: 20 GB model whole to write it peaked at 20 GB when no component of it is more
#: than half that, and the components were independently convertible all along —
#: see `components.py` for the evidence, per family.
_GENERIC = QuantizationCapability(
    supports_quantization=True,
    quantize_choices=QUANTIZE_CHOICES,
    supports_prequantize=True,
    prequantize_choices=QUANTIZE_CHOICES,
    prequantize_strategy=STRATEGY_QDS_MEMORY_BOUNDED,
)

#: Qwen skips only its text encoder, so runtime quantization still applies to the
#: transformer and VAE; conversion is the ordinary route.
_CAPABILITIES: dict[str, QuantizationCapability] = {
    "flux2": _GENERIC,
    "qwen": _GENERIC,
    "z-image": _GENERIC,
    "ernie": _GENERIC,
    "fibo": _GENERIC,
    "ideogram4": _IDEOGRAM4,
    "flux2-dev": _FLUX2_DEV,
    "krea2": _SINGLE_FILE_COMPONENT,
    "anima": _ANIMA,
    # QDS's own five-component definition, for a model mflux 0.19.0 does not ship.
    # Converts exactly like the three-component families do: the converter walks
    # `components.components_for(family)`, and the only thing that changes with a
    # longer list is how many times it goes round.
    "sd35": _GENERIC,
}

#: Edit variants are reached through their parent model, and `Flux2KleinEdit` has
#: no `save_model` at all. Nothing here converts them, so they publish nothing.
_UNKNOWN = QuantizationCapability(
    supports_quantization=False,
    quantize_choices=(),
    supports_prequantize=False,
    prequantize_choices=(),
    prequantize_strategy=None,
    note="Quantization support for this family has not been verified.",
)


def capability_for(family: str) -> QuantizationCapability:
    """Quantization facts for a family.

    Keyed by family rather than by catalogue key on purpose: once a local model
    can be imported, identifying its family is enough to give it the same
    representation, with no new table.
    """
    return _CAPABILITIES.get(family, _UNKNOWN)


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
    include_disabled: bool = False,
    imported: list[Any] | None = None,
    cache_root: str | None = None,
) -> dict[str, ModelSpec]:
    """Apply the `server-config.json` overrides on top of the base catalogue.

    `default_size` is the one config-wide setting, and it sits **below** the
    per-model override:

        request `size` > models.<key>.default_size > default_size > catalogue
                         models.<key>.quantize                    > catalogue

    Size and precision are deliberately not symmetric. Area is a taste that
    applies across a catalogue — a single global is a reasonable thing to want.
    Precision is not: what a bit depth costs in quality depends entirely on the
    model it is applied to, so it is decided per row here and overridden per
    model in the config, with nothing in between.
    """
    overrides = overrides or {}
    imported_keys = {getattr(model, "id", None) for model in (imported or [])}
    unknown = set(overrides) - set(BASE_SPECS_BY_KEY) - imported_keys
    if unknown:
        raise ValueError(
            f"Unknown models in config: {sorted(unknown)}. Valid keys: {sorted(BASE_SPECS_BY_KEY)}"
        )

    global_size = parse_size(default_size) if default_size else None

    registry: dict[str, ModelSpec] = {}
    for key, base in BASE_SPECS_BY_KEY.items():
        # Every row learns where this installation keeps generated artifacts, so
        # that resolving a saved variant needs no second lookup and no default.
        spec = replace(base, cache_root=cache_root)
        if global_size is not None:
            spec = replace(spec, default_width=global_size[0], default_height=global_size[1])
        # Note what does *not* happen here: nothing config-wide touches
        # `quantize`. There used to be a `default_quantize`, and it overwrote the
        # catalogue rather than standing behind it, so one number decided the
        # precision of every model — including rows that had chosen one for a
        # reason. Anima is why that is no longer acceptable: at 4-bit a 2B model
        # produces visibly broken images, where the same setting is harmless on a
        # 20B. Precision belongs to the model.
        override = overrides.get(key)
        if override is not None:
            spec = _apply_override(spec, override)
        if spec.enabled or include_disabled:
            registry[key] = spec

    # Imported models are layered on afterwards: the built-in catalogue stays
    # source-code truth, and one unusable row may not cost the others.
    for model in imported or []:
        try:
            spec = replace(imported_spec(model), cache_root=cache_root)
        except ValueError as exc:
            logger.warning("Skipping imported model %s: %s", getattr(model, "id", "?"), exc)
            continue
        override = overrides.get(spec.key)
        if override is not None:
            spec = _apply_override(spec, override)
        if spec.enabled or include_disabled:
            registry[spec.key] = spec
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
        if spec.quantization.supports_quantization:
            changes["quantize"] = override.quantize or None
        else:
            # Normalised, not rejected: the shipped config carries
            # `flux2-dev.quantize = 8`, and refusing it would stop every existing
            # install from starting. The weights decide the precision here, so the
            # setting is dropped and said out loud rather than published as fact.
            logger.warning(
                "models.%s.quantize=%s ignored: %s",
                spec.key,
                override.quantize,
                spec.quantization.note or "this model's precision comes from its weights",
            )
    if override.model_path is not None:
        changes["model_path"] = override.model_path
    if override.prequantized_variant is not None:
        changes["prequantized_variant"] = override.prequantized_variant

    if override.enable_edit is not None and spec.edit is not None:
        changes["edit"] = replace(spec.edit, enabled_by_default=override.enable_edit)

    return replace(spec, **changes)


PROVENANCE_BUILT_IN = "built_in"
PROVENANCE_IMPORTED = "imported_local"


def imported_spec(model: Any) -> ModelSpec:
    """Turn one library row into a `ModelSpec`, borrowing a built-in's defaults.

    The imported directory establishes identity, family and location; everything
    else a `ModelSpec` requires — step counts, guidance, scheduler, capability
    flags — belongs to the *profile* the user chose, which is why the row records
    it rather than this function picking one.

    `repo` carries the local path purely because `ModelSpec` requires the field.
    It is not, and must not be presented as, a HuggingFace repository:
    `provenance` is what anything behavioural reads.
    """
    profile = BASE_SPECS_BY_KEY.get(model.base_profile_key)
    if profile is None:
        raise ValueError(
            f"Imported model {model.display_name!r} was based on the built-in profile "
            f"{model.base_profile_key!r}, which no longer exists in this version of QDS."
        )
    if profile.family != model.family:
        raise ValueError(
            f"Imported model {model.display_name!r} is a {model.family!r} model, but its base "
            f"profile {model.base_profile_key!r} is {profile.family!r}."
        )
    return replace(
        profile,
        key=model.id,
        repo=model.path,
        model_path=model.path,
        provenance=PROVENANCE_IMPORTED,
        base_profile_key=model.base_profile_key,
        display_name=model.display_name,
        api_name=model.api_name or None,
        edit=None,  # an edit variant belongs to the built-in's own weights
        prequantized_variant=None,
        enabled=True,
    )


def edit_enabled(spec: ModelSpec) -> bool:
    return spec.edit is not None and spec.edit.enabled_by_default


# ── Actually loading the models ────────────────────────────────────────────


#: Configs mflux does not know about and that we build ourselves. These are
#: factories, not instances: they only import `ModelConfig` when called.
_LOCAL_MODEL_CONFIGS: dict[str, Any] = {
    "flux2_dev": flux2_dev_config.flux2_dev_model_config,
    "anima": anima_config.anima_model_config,
    "anima_turbo": anima_config.anima_turbo_model_config,
    "sd35_medium": sd35_config.sd35_medium_model_config,
    "sd35_large": sd35_config.sd35_large_model_config,
    "sd35_large_turbo": sd35_config.sd35_large_turbo_model_config,
    "qwen_image_flash": qwen_flash_config.qwen_image_flash_model_config,
}


def _model_config(name: str):
    factory = _LOCAL_MODEL_CONFIGS.get(name)
    if factory is not None:
        return factory()

    from mflux.models.common.config import ModelConfig

    return getattr(ModelConfig, name)()


def generates_from_source(spec: ModelSpec) -> bool:
    """Whether generation can load this spec's source weights directly.

    False for FLUX.2-dev alone: its source ships bf16, ~111 GB, so generation
    needs a saved quantized copy and `load_model` refuses the source outright.

    Asked by anything that must not promise generation from a download. The
    download path in particular: everywhere else "download" means "load the model
    and exit", which for this one model would hit the guard below instead of
    fetching anything.
    """
    return spec.family != "flux2-dev"


def _require_local_artifact(spec: ModelSpec, model_path: str | None) -> None:
    """Fail early and clearly unless the pre-quantized artifact is actually complete.

    Without this guard, `PathResolution` would fall back to the bf16 HuggingFace
    repo and start an on-the-fly quantization of ~111 GB, which would fail much
    later and far less legibly.

    `exists()` used to be the whole test, and the converter creates its destination
    before downloading anything — so a conversion that died in its first minute
    sailed through here and failed deep inside mflux instead. The check is now the
    artifact's own completion contract.
    """
    from qds import availability as av

    if model_path:
        state, detail = av.flux2_dev_artifact_state(model_path)
        if state == av.PRESENT:
            return
    else:
        state, detail = av.MISSING, "no saved variant is selected"

    from qds.errors import APIError

    raise APIError(
        f"Model '{spec.key}' cannot be generated from its source: {spec.repo} ships bf16, about "
        f"111 GB of weights, which does not fit in unified memory. It needs a saved quantized "
        f"copy, and the one currently selected is {state} ({detail}). Convert it from the "
        f"Quantization dialog in the Models tab, or select an existing saved variant with "
        f"models.{spec.key}.prequantized_variant.",
        status_code=503,
        error_type="server_error",
        param="model",
        code="model_not_prepared",
    )


def _require_variant(spec: ModelSpec) -> None:
    """Refuse to generate from a saved variant that is not a valid one.

    The check is against the *current* source: selecting a variant and then
    pointing the model at a different repo must not quietly keep using the old
    conversion, which is the whole reason identity is recorded in the artifact.
    """
    from qds import artifacts
    from qds import availability as av
    from qds.errors import APIError

    path = Path(str(spec.effective_model_path)).expanduser()
    state, detail = artifacts.artifact_state(
        path, expect_source=spec.source, expect_bits=spec.prequantized_variant
    )
    if state == av.PRESENT:
        return
    raise APIError(
        f"Model '{spec.key}' is set to use its {spec.prequantized_variant}-bit saved variant, "
        f"but that artifact is {state} ({detail}). Convert it again, or clear "
        f"models.{spec.key}.prequantized_variant to generate from the source.",
        status_code=503,
        error_type="server_error",
        param="model",
        code="variant_not_prepared",
    )


def family_structure(family: str) -> tuple[Any, Any]:
    """The pair of mflux classes a family is built from: model, weight definition.

    The classes rather than an instance, because component-wise conversion needs
    the *structure* — which module class each component is, and what the family
    declares about loading and saving it — without materialising a model whose
    whole point is that it does not fit in memory.

    Family dispatch lives here, next to `load_model`, so there is one place that
    knows what `"z-image"` means. Imports stay inside the branches for the same
    reason they do there: this module is imported by the catalogue path, which
    must not pull in torch.
    """
    if family == "flux2":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

        return Flux2Klein, Flux2KleinWeightDefinition

    if family == "flux2-dev":
        from qds.flux2_dev import Flux2Dev, Flux2DevWeightDefinition

        return Flux2Dev, Flux2DevWeightDefinition

    if family == "qwen":
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

        return QwenImage, QwenWeightDefinition

    if family == "z-image":
        from mflux.models.z_image import ZImage
        from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

        return ZImage, ZImageWeightDefinition

    if family == "ernie":
        from mflux.models.ernie_image import ErnieImage
        from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition

        return ErnieImage, ErnieWeightDefinition

    if family == "fibo":
        from mflux.models.fibo.variants.txt2img.fibo import FIBO
        from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition

        return FIBO, FIBOWeightDefinition

    if family == "sd35":
        from qds.sd35 import SD35, SD35WeightDefinition

        # One definition for all three rows: they differ in transformer *shape*, which
        # `model_config_for(spec)` supplies, not in component layout.
        return SD35, SD35WeightDefinition

    raise ValueError(f"No component-wise conversion is established for family {family!r}")


def model_config_for(spec: ModelSpec) -> Any:
    """The `ModelConfig` this spec generates with.

    Exposed because conversion needs exactly the configuration generation uses:
    `flux2` and `ernie` build their transformer and text encoder with overrides
    read off it, and building those modules with the class defaults instead would
    produce a differently shaped module for a model whose weights then no longer
    fit it.
    """
    return _model_config(spec.model_config_name)


def load_model(spec: ModelSpec, *, kind: str = "txt2img") -> Any:
    """Instantiate the mflux model matching the spec.

    Faithfully mirrors each family's CLI `main()` — that is the reference to
    compare against if results ever diverge.
    """
    if kind == "txt2img":
        # `effective_model_path`, not `model_path`: a selected saved variant is
        # what generation loads, while `model_path` stays the source's identity.
        family, model_config_name = spec.family, spec.model_config_name
        model_path = spec.effective_model_path
        if spec.prequantized_variant is not None:
            _require_variant(spec)
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
        from qds.flux2_dev import Flux2Dev

        return Flux2Dev(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "flux2-edit":
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

        return Flux2KleinEdit(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "anima":
        from qds.anima import Anima

        # Which checkpoint, from the row's config name: Aesthetic and Turbo are
        # one architecture and two trainings, so they differ by file alone.
        return Anima(
            model_config=model_config,
            model_path=model_path,
            quantize=quantize,
            weight_file=anima_config.weight_file_for(model_config_name),
        )

    if family == "sd35":
        from qds.sd35 import SD35

        # Which of the three releases, entirely from the row's `ModelConfig`: the
        # repository it names and the transformer shape it carries. No extra argument,
        # unlike Anima — SD 3.5's variants are three shapes, not three files.
        return SD35(model_config=model_config, model_path=model_path, quantize=quantize)

    if family == "krea2":
        # Importing the package is also what registers `er_sde` as a scheduler —
        # a side effect of `mflux/models/krea2/__init__.py` — so the import has
        # to happen before `generate_image` resolves this spec's `"linear"`.
        # Doing it here rather than at module scope is the same rule as every
        # other family: the catalogue must not pull in torch.
        from mflux.models.krea2 import Krea2

        return Krea2(model_config=model_config, model_path=model_path, quantize=quantize)

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


def latent_creator_for(family: str) -> Any | None:
    """The latent unpacker for a family `load_model` knows, or None (no previews).

    Mirrors each family's CLI registration of mflux's `StepwiseHandler`: that is
    the reference for which creator belongs to which model. Only the unpacking
    side is used here — turning a mid-loop latent back into a decodable tensor —
    so the classes are returned, not instances (`unpack_latents` is a
    staticmethod on every one of them).

    Fail-closed: an unknown family yields `None`, which means "no preview", never
    an error. Imports stay inside the function for the same reason they do in
    `load_model`: this module is imported by the catalogue path, which must not
    pull in torch.
    """
    if family in ("flux2", "flux2-dev", "flux2-edit"):
        from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator

        return Flux2LatentCreator

    if family in ("qwen", "qwen-edit"):
        from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator

        return QwenLatentCreator

    if family == "krea2":
        from mflux.models.krea2.latent_creator import Krea2LatentCreator

        return Krea2LatentCreator

    if family == "anima":
        from qds.anima.latent_creator import AnimaLatentCreator

        return AnimaLatentCreator

    if family == "sd35":
        from qds.sd35.latent_creator import SD35LatentCreator

        return SD35LatentCreator

    if family == "z-image":
        from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator

        return ZImageLatentCreator

    if family == "ernie":
        from mflux.models.ernie_image.latent_creator.ernie_latent_creator import ErnieLatentCreator

        return ErnieLatentCreator

    if family == "fibo":
        from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator

        return FiboLatentCreator

    if family == "ideogram4":
        from mflux.models.ideogram4.latent_creator.ideogram4_latent_creator import Ideogram4LatentCreator

        return Ideogram4LatentCreator

    return None
