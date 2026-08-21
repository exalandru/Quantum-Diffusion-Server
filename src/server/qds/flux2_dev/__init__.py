"""Support for `black-forest-labs/FLUX.2-dev`, absent from mflux 0.19.0.

Only `config` is imported when the package loads: it depends on nothing. Everything
else is resolved lazily, because `registry` imports this package at module level
and must stay light — importing mflux pulls in torch and transformers.
"""

from __future__ import annotations

from typing import Any

from qds.flux2_dev.config import (
    MAX_SEQUENCE_LENGTH,
    REPO,
    TEXT_ENCODER_OUT_LAYERS,
    TEXT_ENCODER_OVERRIDES,
    TRANSFORMER_OVERRIDES,
    flux2_dev_model_config,
)

#: Exported name → the module that defines it. Each of these imports mflux.
_LAZY: dict[str, str] = {
    "Flux2Dev": "qds.flux2_dev.model",
    "Flux2DevTokenizer": "qds.flux2_dev.tokenizer",
    "SYSTEM_MESSAGE": "qds.flux2_dev.tokenizer",
    "Flux2DevWeightDefinition": "qds.flux2_dev.weights",
    "single_component_definition": "qds.flux2_dev.weights",
    "Mistral3TextEncoder": "qds.flux2_dev.mistral3",
}

__all__ = [
    "MAX_SEQUENCE_LENGTH",
    "REPO",
    "SYSTEM_MESSAGE",
    "TEXT_ENCODER_OUT_LAYERS",
    "TEXT_ENCODER_OVERRIDES",
    "TRANSFORMER_OVERRIDES",
    "Flux2Dev",
    "Flux2DevTokenizer",
    "Flux2DevWeightDefinition",
    "Mistral3TextEncoder",
    "flux2_dev_model_config",
    "single_component_definition",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module_name), name)
