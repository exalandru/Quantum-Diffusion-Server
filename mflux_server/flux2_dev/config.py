"""`ModelConfig` de FLUX.2-dev, absent de mflux 0.18.0.

`AVAILABLE_MODELS` de mflux ne contient que les variantes klein ; il n'existe
donc ni `ModelConfig.flux2_dev()` ni alias exploitable. On construit la config
à la main depuis les `config.json` du repo.

Surtout ne pas passer par `ModelConfig.from_name("black-forest-labs/FLUX.2-dev")` :
la résolution par sous-chaîne y trouve l'alias `"dev"` et fabrique
silencieusement une config **FLUX.1**-dev
(mflux/models/common/resolution/config_resolution.py:57-64) — aucune erreur,
juste la mauvaise architecture.
"""

from __future__ import annotations

from typing import Any

REPO = "black-forest-labs/FLUX.2-dev"

#: Destination par défaut de `mflux-server-prequantize`. Le repo amont est en
#: bf16 (~111 Go de poids) : sur mémoire unifiée, seul un artefact déjà
#: quantifié est chargeable.
DEFAULT_MODEL_PATH = "~/.cache/mflux-server/flux2-dev-mlx-8bit"

#: `transformer/config.json` du repo. Chaque clé est un kwarg de
#: `Flux2Transformer.__init__` — l'architecture est celle de klein, en plus
#: grand. `guidance_embeds` n'est pas dans le JSON mais les poids
#: `time_guidance_embed.guidance_embedder.*` sont bien là : FLUX.2-dev est
#: guidance-distilled, contrairement à klein.
TRANSFORMER_OVERRIDES: dict[str, Any] = {
    "num_layers": 8,
    "num_single_layers": 48,
    "num_attention_heads": 48,
    "attention_head_dim": 128,
    "joint_attention_dim": 15360,
    "guidance_embeds": True,
}

#: `text_encoder/config.json` → `text_config` (la tour texte de Mistral3).
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

#: Couches dont les états cachés sont empilés pour former l'embedding de
#: prompt : `(n//4, n//2, 3n//4)` sur 40 couches, la règle appliquée par klein
#: avec `(9, 18, 27)` sur 36. 3 × 5120 = `joint_attention_dim`.
TEXT_ENCODER_OUT_LAYERS: tuple[int, ...] = (10, 20, 30)

MAX_SEQUENCE_LENGTH = 512


def flux2_dev_model_config():
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig(
        # Au-delà des priorités de mflux : cette config n'entre jamais en
        # concurrence dans sa résolution par alias.
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
        # Les défauts sigma_* de ModelConfig (0.5 / 1.15 / 256 / 4096)
        # correspondent déjà au scheduler_config.json du repo.
    )
