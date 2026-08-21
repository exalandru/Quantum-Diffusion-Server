"""Support for `nvidia/Qwen-Image-Flash`, absent from mflux 0.19.0's catalogue.

Unlike Anima, nothing here is a port. Flash *is* Qwen-Image — same
`QwenImageTransformer2DModel` at 60 layers and 24 heads, same Qwen2.5-VL text
encoder, same Qwen-Image VAE — distilled down to four steps. mflux already
implements every one of those, so this package supplies exactly one thing mflux's
own `qwen-image` entry gets wrong for it: the noise schedule.

Only `config` lives here, and it imports nothing heavy, so `registry` can reach
the factory at module scope.
"""

from __future__ import annotations

from qds.qwen_flash.config import (
    REPO,
    SIGMA_SHIFT,
    qwen_image_flash_model_config,
)

__all__ = ["REPO", "SIGMA_SHIFT", "qwen_image_flash_model_config"]
