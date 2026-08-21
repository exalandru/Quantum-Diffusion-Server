"""Support for `circlestone-labs/Anima`, absent from mflux 0.19.0.

Anima is a 2B text-to-image model: NVIDIA's Cosmos-Predict2 DiT, conditioned
through a Qwen3-0.6B encoder and a small learned adapter, decoding with the
Qwen-Image VAE. mflux implements the encoder and the VAE already; the DiT and the
adapter are ported here, and checked against diffusers in `tests/test_anima.py`.

Only `config` is imported when the package loads: it depends on nothing.
Everything else is resolved lazily, because `registry` imports this package at
module level and must stay light -- importing mflux pulls in torch and
transformers.
"""

from __future__ import annotations

from typing import Any

from qds.anima.config import (
    ADAPTER_PREFIX,
    CHECKPOINT_PREFIX,
    COMPANION_REPO,
    CONDITIONER_OVERRIDES,
    DEFAULT_WEIGHT_FILE,
    MAX_SEQUENCE_LENGTH,
    REPO,
    SIGMA_SHIFT,
    TEXT_ENCODER_OVERRIDES,
    TRANSFORMER_OVERRIDES,
    anima_model_config,
)

#: Exported name → the module that defines it. Each of these imports mflux or mlx.
_LAZY: dict[str, str] = {
    "Anima": "qds.anima.model",
    "AnimaTransformer": "qds.anima.transformer",
    "AnimaTextConditioner": "qds.anima.conditioner",
    "AnimaWeightDefinition": "qds.anima.weights",
    "AnimaLatentCreator": "qds.anima.latent_creator",
}

__all__ = [
    "ADAPTER_PREFIX",
    "CHECKPOINT_PREFIX",
    "COMPANION_REPO",
    "CONDITIONER_OVERRIDES",
    "DEFAULT_WEIGHT_FILE",
    "MAX_SEQUENCE_LENGTH",
    "REPO",
    "SIGMA_SHIFT",
    "TEXT_ENCODER_OVERRIDES",
    "TRANSFORMER_OVERRIDES",
    "Anima",
    "AnimaLatentCreator",
    "AnimaTextConditioner",
    "AnimaTransformer",
    "AnimaWeightDefinition",
    "anima_model_config",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module_name), name)
