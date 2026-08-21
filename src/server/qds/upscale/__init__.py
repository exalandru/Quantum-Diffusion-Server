"""Real-ESRGAN upscaling, ported to MLX.

Real-ESRGAN is a convolutional super-resolution network -- no diffusion, no
prompt, no sampling loop. That is why it is here rather than mflux's `SeedVR2`:
at 33 MB it can stay resident *beside* the diffusion model, so upscaling an
image does not evict a warm FLUX and pay a minute to reload it. See
`ModelEngine`'s module docstring for the bounded second slot this buys.

What is ported is basicsr's `RRDBNet` and nothing else: the checkpoints are
applied by their own names, so there is no translation table, and the tiling
and colour handling are this package's own. `tests/test_upscale.py` checks the
network tensor-for-tensor against a transcription of the reference in torch,
and checks the tiling against its own untiled result.

Only `catalogue` is imported when the package loads: it depends on nothing but
the standard library. Everything else is resolved lazily, because `fetch` reads
the catalogue on the `--status` path and `app` reads it at start-up, and
neither may pay for mlx.
"""

from __future__ import annotations

from typing import Any

from qds.upscale.catalogue import (
    KEYS,
    MAX_RENDER_PIXELS,
    MAX_WEIGHTS_MB,
    SCALES,
    SPECS,
    UpscalerSpec,
    by_key,
    tensor_names,
)

#: Exported name → the module that defines it. Each of these imports mlx.
_LAZY: dict[str, str] = {
    "RRDBNet": "qds.upscale.rrdbnet",
    "load_upscaler": "qds.upscale.weights",
    "resolve_file": "qds.upscale.weights",
    "is_downloaded": "qds.upscale.weights",
    "verify_state_dict": "qds.upscale.weights",
    "verify_loaded": "qds.upscale.weights",
    "upscale_image": "qds.upscale.pipeline",
    "tile_grid": "qds.upscale.pipeline",
}

__all__ = [
    "KEYS",
    "MAX_RENDER_PIXELS",
    "MAX_WEIGHTS_MB",
    "SCALES",
    "SPECS",
    "RRDBNet",
    "UpscalerSpec",
    "by_key",
    "is_downloaded",
    "load_upscaler",
    "resolve_file",
    "tensor_names",
    "tile_grid",
    "upscale_image",
    "verify_loaded",
    "verify_state_dict",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module_name), name)
