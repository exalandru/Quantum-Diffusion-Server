"""FLUX.2-dev's `ModelConfig`, absent from mflux 0.18.0.

mflux's `AVAILABLE_MODELS` only contains the klein variants, so there is neither
a `ModelConfig.flux2_dev()` nor a usable alias. We build the config by hand from
the repo's `config.json` files.

Above all, do not go through `ModelConfig.from_name("black-forest-labs/FLUX.2-dev")`:
its substring resolution finds the `"dev"` alias in that name and silently
fabricates a **FLUX.1**-dev config
(mflux/models/common/resolution/config_resolution.py:57-64) — no error at all,
just the wrong architecture.
"""

from __future__ import annotations

from typing import Any

REPO = "black-forest-labs/FLUX.2-dev"

#: Default destination of `mflux-server-prequantize`. The upstream repo ships
#: bf16 (~111 GB of weights): on unified memory, only an already quantized
#: artifact is loadable.
DEFAULT_MODEL_PATH = "~/.cache/mflux-server/flux2-dev-mlx-8bit"

#: The repo's `transformer/config.json`. Every key is a kwarg of
#: `Flux2Transformer.__init__` — the architecture is klein's, just bigger.
#: `guidance_embeds` is not in the JSON, but the
#: `time_guidance_embed.guidance_embedder.*` weights are: FLUX.2-dev is
#: guidance-distilled, unlike klein.
TRANSFORMER_OVERRIDES: dict[str, Any] = {
    "num_layers": 8,
    "num_single_layers": 48,
    "num_attention_heads": 48,
    "attention_head_dim": 128,
    "joint_attention_dim": 15360,
    "guidance_embeds": True,
}

#: `text_encoder/config.json` → `text_config` (Mistral3's text tower).
TEXT_ENCODER_OVERRIDES: dict[str, Any] = {
    "vocab_size": 131072,
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 32768,
    "rms_norm_eps": 1e-5,
    "rope_theta": 1000000000.0,
    "max_position_embeddings": 131072,
}

#: Layers whose hidden states are stacked into the prompt embedding:
#: `(n//4, n//2, 3n//4)` over 40 layers, the same rule klein applies with
#: `(9, 18, 27)` over 36. 3 × 5120 = `joint_attention_dim`.
TEXT_ENCODER_OUT_LAYERS: tuple[int, ...] = (10, 20, 30)

MAX_SEQUENCE_LENGTH = 512


def flux2_dev_model_config():
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig(
        # Beyond mflux's own priorities: this config never competes in its
        # alias resolution.
        priority=999,
        aliases=["flux2-dev"],
        model_name=REPO,
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides=dict(TRANSFORMER_OVERRIDES),
        text_encoder_overrides=dict(TEXT_ENCODER_OVERRIDES),
        # ModelConfig's sigma_* defaults (0.5 / 1.15 / 256 / 4096) already
        # match the repo's scheduler_config.json.
    )
