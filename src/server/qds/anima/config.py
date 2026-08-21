"""Anima's `ModelConfig` and module shapes, absent from mflux 0.19.0.

Every value here was read off the published artifacts rather than a model card:
the shapes from the tensor index of
`circlestone-labs/Anima/split_files/diffusion_models/anima-aesthetic-v1.1.safetensors`,
and the named fields from the `config.json` of each component in
`circlestone-labs/Anima-Base-v1.0-Diffusers`. `tests/test_anima.py` pins them
against both, so a value that drifts from the weights fails rather than
mis-generates.

This module must not import mflux or torch: `registry` imports it at module
scope to reach the `ModelConfig` factory, and the catalogue path may not pay for
that. The mflux import lives inside the factory function.
"""

from __future__ import annotations

#: The weights. Ungated, non-commercial; only this repository carries the
#: Aesthetic checkpoints.
REPO = "circlestone-labs/Anima"

#: Tokenizers and the diffusers-format VAE. A second repository because the
#: first ships neither: `split_files/` holds a ComfyUI-style bundle with no
#: tokenizer at all, and its VAE file uses ComfyUI's module names, which mflux's
#: Qwen VAE mapping does not describe. This one is a plain diffusers layout, so
#: both load with mflux's own machinery and no rename table.
#:
#: It is the Base v1.0 release rather than Aesthetic, and that is sound for
#: exactly these two components: the VAE is Qwen-Image's, frozen, and the
#: tokenizers are vocabularies. The transformer is *not* taken from here.
COMPANION_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"

#: The two checkpoints this catalogue serves, out of the eight the repository
#: publishes under `split_files/diffusion_models/`. They are the same
#: architecture and differ only in training: Aesthetic is the undistilled
#: fine-tune, Turbo the distilled one.
AESTHETIC_WEIGHT_FILE = "anima-aesthetic-v1.1.safetensors"
TURBO_WEIGHT_FILE = "anima-turbo-v1.0.safetensors"

#: Which checkpoint each catalogue row loads, keyed by its `model_config_name`.
#: One table, so the catalogue row, its `ModelConfig` and the file on disk cannot
#: drift apart: adding a row without an entry here fails loudly in
#: `weight_file_for` rather than quietly loading the other variant.
WEIGHT_FILE_BY_CONFIG: dict[str, str] = {
    "anima": AESTHETIC_WEIGHT_FILE,
    "anima_turbo": TURBO_WEIGHT_FILE,
}

#: What `AnimaWeightDefinition` and the tests use when no variant is named.
DEFAULT_WEIGHT_FILE = AESTHETIC_WEIGHT_FILE

#: The checkpoints whose naming this package translates.
#:
#: Not all eight are interchangeable, and the difference is invisible from the
#: file name. Verified by reading every published tensor index: the three
#: Aesthetic releases and Turbo v1.0 root their tensors at `model.diffusion_model.`
#: and translate completely (567 transformer + 118 adapter). **Base v1.0 and the
#: three previews root theirs at `net.` instead**, and this package's translation
#: maps none of their 685 tensors. Selecting one of those does not fail on its own
#: -- weights are applied non-strictly -- so `weights.verify_loaded` refuses it at
#: load time rather than letting the server generate noise.
SUPPORTED_WEIGHT_FILES = (
    "anima-aesthetic-v1.1.safetensors",
    "anima-aesthetic-v1.0b.safetensors",
    "anima-aesthetic-v1.0.safetensors",
    "anima-turbo-v1.0.safetensors",
)


def weight_file_for(model_config_name: str) -> str:
    """The checkpoint a catalogue row's `ModelConfig` name selects."""
    try:
        return WEIGHT_FILE_BY_CONFIG[model_config_name]
    except KeyError:
        raise ValueError(
            f"No Anima checkpoint is registered for {model_config_name!r}. "
            f"Known: {sorted(WEIGHT_FILE_BY_CONFIG)}."
        ) from None

#: Where the DiT and its text adapter live inside that single file.
CHECKPOINT_PREFIX = "model.diffusion_model."
ADAPTER_PREFIX = CHECKPOINT_PREFIX + "llm_adapter."

#: The Qwen tokenizer truncates here. Not a chat template: the pipeline encodes
#: the prompt raw (`diffusers/modular_pipelines/anima/encoders.py`).
MAX_SEQUENCE_LENGTH = 512

#: `scheduler/scheduler_config.json`: a *static* shift of 3.0, with
#: `use_dynamic_shifting` false. mflux expresses shift as `mu` through an
#: exponential time shift, and the two are the same function -- exponential
#: shift at `mu` equals static shift at `exp(mu)` -- so this is `ln(3)`.
#: `tests/test_anima.py` proves the identity rather than asserting it.
SIGMA_SHIFT = 3.0
NUM_TRAIN_TIMESTEPS = 1000

#: `transformer/config.json`. 16 heads x 128 = 2048 hidden; `in_channels` 16
#: plus the padding-mask channel is the 17 that, over a 2x2 patch, makes the
#: 68 columns of `patch_embed.proj`.
TRANSFORMER_OVERRIDES: dict = {
    "in_channels": 16,
    "out_channels": 16,
    "num_layers": 28,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "mlp_ratio": 4.0,
    "patch_size": (1, 2, 2),
    "rope_scale": (1.0, 4.0, 4.0),
    "max_size": (128, 240, 240),
    "concat_padding_mask": True,
}

#: `text_conditioner/config.json`. The adapter that turns Qwen3 hidden states
#: into the 1024-wide context the DiT cross-attends to, via learned T5 token
#: embeddings. 16 heads over `model_dim` 1024 gives a head dim of 64, which is
#: what the `q_norm`/`k_norm` shapes in the checkpoint show.
CONDITIONER_OVERRIDES: dict = {
    "source_dim": 1024,
    "target_dim": 1024,
    "model_dim": 1024,
    "num_layers": 6,
    "num_attention_heads": 16,
    "mlp_ratio": 4.0,
    "target_vocab_size": 32128,
    "min_sequence_length": 512,
    "use_self_attention": True,
    "use_layer_norm": False,
}

#: `text_encoder/config.json`: Qwen3-0.6B, unmodified. mflux already implements
#: this tower for FLUX.2 klein and its constructor takes every one of these, so
#: the encoder is reused rather than ported.
TEXT_ENCODER_OVERRIDES: dict = {
    "vocab_size": 151936,
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
}


def _model_config(alias: str):
    """An mflux `ModelConfig` for Anima, resolved through `_LOCAL_MODEL_CONFIGS`.

    `priority=999` keeps it out of mflux's own name resolution: this config is
    reached by name from the catalogue, never by mflux guessing at an alias.

    Aesthetic and Turbo share one config because they *are* one architecture:
    same 28 blocks, same adapter, same VAE, same sigma shift. What separates them
    is which checkpoint is loaded and how many steps it wants, and neither of
    those belongs here — the file comes from `WEIGHT_FILE_BY_CONFIG`, the step
    count from the catalogue row.
    """
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig(
        priority=999,
        aliases=[alias],
        model_name=REPO,
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=NUM_TRAIN_TIMESTEPS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        supports_guidance=True,
        # The shift is applied by this package, from `SIGMA_SHIFT`, because
        # mflux's own path derives `mu` from the image sequence length and
        # Anima's is a constant.
        requires_sigma_shift=False,
        transformer_overrides=TRANSFORMER_OVERRIDES,
        text_encoder_overrides=TEXT_ENCODER_OVERRIDES,
    )


def anima_model_config():
    """Anima Aesthetic v1.1 — the undistilled fine-tune."""
    return _model_config("anima")


def anima_turbo_model_config():
    """Anima Turbo v1.0 — distilled, and what the model card suggests starting with."""
    return _model_config("anima-turbo")
