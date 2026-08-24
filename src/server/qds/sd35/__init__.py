"""Stable Diffusion 3.5 — Medium, Large and Large Turbo — for mflux 0.19.0.

mflux 0.19.0 has no SD 3.5: no `mflux/models/sd3*`, no `sd3` factory on `ModelConfig`,
and no upstream work in progress. This package is the same answer `qds/anima/` and
`qds/flux2_dev/` already give for models mflux does not ship — local module classes, a
local weight definition and local `ModelConfig` factories, wired into the catalogue
through `registry.family_structure` / `load_model` / `_LOCAL_MODEL_CONFIGS`, reusing
mflux's common machinery for everything that is not model maths.

What is reused and what is local was decided per component, against the real
checkpoints rather than by family resemblance:

* **T5-XXL** is mflux's own `T5Encoder`, unchanged, with mflux's own mapping. Its
  hard-coded shape *is* SD 3.5's `text_encoder_3`.
* **the VAE** is mflux's FLUX.1 `VAE` with two constants overridden. Same diffusers
  `AutoencoderKL` topology, differently normalised latents.
* **both CLIP towers** are local: mflux's is fixed at 768/12/12 with no
  `text_projection` and returns only a pooled vector, while SD 3.5 conditions on the
  penultimate hidden states of a 768-wide *and* a 1280-wide tower.
* **the MMDiT-X transformer** is local, because nothing resembling it exists in mflux.

Scope is txt2img and img2img. No edit variant (SD 3.5 has none), no inpainting, no
LoRA, no fp8 path.

Only `config` is imported when the package loads: it depends on nothing. Everything
else is resolved lazily, because `registry` imports this package at module level and
must stay light — importing mflux pulls in torch and transformers, and the desktop
app reads the catalogue with the generation server stopped.
"""

from __future__ import annotations

from typing import Any

from qds.sd35.config import (
    LARGE_REPO,
    LARGE_TURBO_REPO,
    MEDIUM_REPO,
    NUM_TRAIN_TIMESTEPS,
    SIGMA_SHIFT,
    sd35_large_model_config,
    sd35_large_turbo_model_config,
    sd35_medium_model_config,
)

#: Exported name → the module that defines it. Each of these imports mflux or mlx.
_LAZY: dict[str, str] = {
    "SD35": "qds.sd35.model",
    "SD35ClipG": "qds.sd35.clip",
    "SD35ClipL": "qds.sd35.clip",
    "SD35ClipTower": "qds.sd35.clip",
    "SD35FlowMatchScheduler": "qds.sd35.scheduler",
    "SD35LatentCreator": "qds.sd35.latent_creator",
    "SD35Transformer": "qds.sd35.transformer",
    "SD35VAE": "qds.sd35.vae",
    "SD35WeightDefinition": "qds.sd35.weights",
}

__all__ = [
    "LARGE_REPO",
    "LARGE_TURBO_REPO",
    "MEDIUM_REPO",
    "NUM_TRAIN_TIMESTEPS",
    "SIGMA_SHIFT",
    "SD35",
    "SD35ClipG",
    "SD35ClipL",
    "SD35ClipTower",
    "SD35FlowMatchScheduler",
    "SD35LatentCreator",
    "SD35Transformer",
    "SD35VAE",
    "SD35WeightDefinition",
    "sd35_large_model_config",
    "sd35_large_turbo_model_config",
    "sd35_medium_model_config",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module_name), name)
