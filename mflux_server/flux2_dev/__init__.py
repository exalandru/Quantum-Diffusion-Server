"""Support de `black-forest-labs/FLUX.2-dev`, absent de mflux 0.18.0.

Seul `config` est importé au chargement du paquet : il ne dépend de rien. Tout
le reste est résolu paresseusement, parce que `registry` importe ce paquet au
niveau module et doit rester léger — importer mflux tire torch et transformers.
"""

from __future__ import annotations

from typing import Any

from mflux_server.flux2_dev.config import (
    DEFAULT_MODEL_PATH,
    MAX_SEQUENCE_LENGTH,
    REPO,
    TEXT_ENCODER_OUT_LAYERS,
    TEXT_ENCODER_OVERRIDES,
    TRANSFORMER_OVERRIDES,
    flux2_dev_model_config,
)

#: Nom exporté → module qui le définit. Chacun de ces modules importe mflux.
_LAZY: dict[str, str] = {
    "Flux2Dev": "mflux_server.flux2_dev.model",
    "Flux2DevTokenizer": "mflux_server.flux2_dev.tokenizer",
    "SYSTEM_MESSAGE": "mflux_server.flux2_dev.tokenizer",
    "Flux2DevWeightDefinition": "mflux_server.flux2_dev.weights",
    "single_component_definition": "mflux_server.flux2_dev.weights",
    "Mistral3TextEncoder": "mflux_server.flux2_dev.mistral3",
}

__all__ = [
    "DEFAULT_MODEL_PATH",
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
