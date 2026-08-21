"""Qwen-Image-Flash's `ModelConfig` — Qwen-Image's architecture, a different schedule.

NVIDIA's Flash release is a four-step distillation of Qwen-Image. Its
`transformer/config.json` is `QwenImageTransformer2DModel` with 60 layers, 24
heads, 64 input channels and a 3584-wide joint attention — identical to
`Qwen/Qwen-Image-2512` — and its `model_index.json` names the same Qwen2.5-VL
text encoder, Qwen2 tokenizer and Qwen-Image VAE. So it loads through mflux's
existing `qwen` family with no new code, and this file exists for one reason:
the two models do **not** share a noise schedule.

    Qwen-Image-2512   use_dynamic_shifting: true,  max_shift 0.9, shift_terminal 0.02
    Qwen-Image-Flash  use_dynamic_shifting: false, shift 3.0,     shift_terminal null

`ModelConfig.qwen_image()` encodes the first, correctly, for the model it belongs
to. Reusing it for Flash would derive `mu` from the image sequence length and
then stretch the tail to 0.02 — a schedule the distillation was not trained
against. On a 50-step model that would cost some fidelity; on a four-step one,
where each sigma is a quarter of the whole trajectory, it is the difference
between the model working and not.

The fix needs no new scheduler. mflux's `LinearScheduler` builds `mu` linearly
between a base and a max shift over a sequence-length range, and **setting the
two shifts equal collapses that line to a constant** — which is precisely what a
static shift is. `ln(3)` is the constant, because mflux shifts exponentially
(`exp(mu)/(exp(mu) + (1/sigma - 1))`) where diffusers shifts multiplicatively
(`shift*sigma/(1 + (shift-1)*sigma)`), and the two are the same function at
`exp(mu) = shift`. `tests/test_qwen_flash.py` proves the identity against
diffusers rather than asserting it.
"""

from __future__ import annotations

import math

#: NVIDIA's own release. `mlx-community/Qwen-Image-Flash-bf16` mirrors it
#: byte for byte; the upstream publisher is preferred when the content is equal.
REPO = "nvidia/Qwen-Image-Flash"

#: `scheduler/scheduler_config.json`: `shift: 3.0`, `use_dynamic_shifting: false`,
#: `shift_terminal: null`.
SIGMA_SHIFT = 3.0

#: What mflux needs to express that shift: a `mu` that does not vary with the
#: image, i.e. a base and a max that are equal.
SIGMA_MU = math.log(SIGMA_SHIFT)


def qwen_image_flash_model_config():
    """Qwen-Image's config with Flash's schedule, via `_LOCAL_MODEL_CONFIGS`."""
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig(
        priority=999,
        aliases=["qwen-image-flash"],
        model_name=REPO,
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=None,
        # Shifted, but not by sequence length: equal base and max make
        # `LinearScheduler`'s slope zero, so `mu` is `SIGMA_MU` at every
        # resolution.
        requires_sigma_shift=True,
        sigma_base_shift=SIGMA_MU,
        sigma_max_shift=SIGMA_MU,
        # Left at their defaults on purpose: with the slope at zero they cannot
        # affect the result, and pinning them would suggest they do.
        # `sigma_shift_terminal` stays `None` — Flash publishes no terminal
        # stretch, unlike Qwen-Image-2512's 0.02.
        sigma_shift_terminal=None,
    )
