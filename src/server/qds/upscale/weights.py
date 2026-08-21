"""Resolving, loading and checking Real-ESRGAN checkpoints.

Two guards live here, and they guard different things.

`verify_state_dict` runs before anything is built: it asks whether the *file*
holds exactly the tensors this catalogue entry describes. `verify_loaded` runs
after `update`: it asks whether every parameter of the *module* received a
value, and whether the shapes are the ones NHWC convolutions need.

Both are necessary, and for reasons worth stating precisely rather than by
analogy with `qds/anima/weights.py`, whose guard exists because mflux applies
weights non-strictly.

`mlx.nn.Module.update` is half strict, measured on mlx 0.32.1: a key the module
does not have raises `ValueError`, but a parameter that *no* key reached keeps
the random initialisation it was constructed with, in silence. So a truncated
checkpoint would load cleanly and upscale into noise, with nothing anywhere to
explain it -- which is what `verify_state_dict` is for.

A *count* alone would not catch a layout mistake either: a torch-layout file has
exactly the right names and exactly the right number of tensors, and only its
axes are permuted. That is what `verify_loaded` is for.

The download is the ordinary Hugging Face one, so the file lands wherever
`settings.apply_hf_home()` already pointed `HF_HOME` and `HF_HUB_CACHE`. No
`cache_dir` is passed on purpose: a second opinion about where the cache lives
is how the two drift apart.
"""

from __future__ import annotations

import logging
from typing import Any

from qds.errors import APIError
from qds.logs import SERVER_LOGGER
from qds.upscale.catalogue import UpscalerSpec, tensor_names

logger = logging.getLogger(SERVER_LOGGER)


def cached_path(spec: UpscalerSpec) -> str | None:
    """The local file for `spec` if it is already downloaded, else `None`.

    Asks about the *file*, not the repository. `availability.scan_repos` answers
    "this repo is in the cache", which is the right question for a status
    report and the wrong one here: it would say `present` for a repository from
    which some other file had been pulled.
    """
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(spec.repo, spec.filename)
    return found if isinstance(found, str) else None


def is_downloaded(spec: UpscalerSpec) -> bool:
    """Whether using `spec` right now would avoid a download. Never hits the network."""
    try:
        return cached_path(spec) is not None
    except Exception:  # pragma: no cover - a broken cache is not a reason to 500
        logger.debug("Could not inspect the cache for %s", spec.key, exc_info=True)
        return False


def resolve_file(spec: UpscalerSpec, *, allow_download: bool = True) -> str:
    """The local path to `spec`'s weights, downloading them if allowed."""
    cached = cached_path(spec)
    if cached is not None:
        return cached
    if not allow_download:
        raise APIError(
            f"The weights for {spec.display_name!r} are not downloaded. "
            f"Run `qds fetch {spec.key}`.",
            status_code=409,
            code="upscaler_not_downloaded",
        )

    from huggingface_hub import hf_hub_download

    logger.info(
        "Downloading upscaler %s (%.1f MB) from %s",
        spec.key,
        spec.size_mb,
        spec.repo,
        extra={"model": spec.key, "repo": spec.repo},
    )
    return hf_hub_download(spec.repo, spec.filename)


def verify_state_dict(spec: UpscalerSpec, raw: dict[str, Any]) -> None:
    """Refuse a file that is not the checkpoint this catalogue entry describes."""
    expected = tensor_names(spec)
    found = frozenset(raw)
    if found == expected:
        return

    missing = sorted(expected - found)
    extra = sorted(found - expected)
    detail = f"{len(missing)} missing, {len(extra)} unexpected"
    if missing:
        detail += f"; first missing: {missing[0]}"
    if extra:
        detail += f"; first unexpected: {extra[0]}"
    raise ValueError(
        f"{spec.repo}/{spec.filename} does not hold the {spec.tensor_count} tensors "
        f"that {spec.key!r} ({spec.num_block} RRDB blocks) needs: {detail}. "
        "The catalogue entry and the published file have diverged."
    )


def verify_loaded(model: Any, spec: UpscalerSpec) -> None:
    """Refuse a module whose parameters did not all get the right values.

    Counting is not enough: a torch-layout checkpoint has every name and every
    count right and only its convolution axes permuted, so it passes
    `verify_state_dict` and then computes nonsense. `conv_last.weight` is the
    discriminator: `(3, 3, 3, num_feat)` here against torch's
    `(3, num_feat, 3, 3)`. `conv_first.weight` cannot serve -- it is
    `(num_feat, 3, 3, 3)` under both layouts, because its input channels and
    its kernel side are both 3.
    """
    from mlx.utils import tree_flatten

    loaded = dict(tree_flatten(model.parameters()))
    if len(loaded) != spec.tensor_count:
        raise ValueError(
            f"{spec.key!r} built {len(loaded)} parameters but its catalogue entry "
            f"declares {spec.tensor_count}."
        )

    wanted = {
        "conv_first.weight": (spec.num_feat, 3, 3, 3),
        "conv_last.weight": (3, 3, 3, spec.num_feat),
        "conv_body.weight": (spec.num_feat, 3, 3, spec.num_feat),
        "body.0.rdb1.conv1.weight": (spec.num_grow_ch, 3, 3, spec.num_feat),
        "conv_last.bias": (3,),
    }
    for name, shape in wanted.items():
        actual = tuple(loaded[name].shape)
        if actual != shape:
            raise ValueError(
                f"{spec.repo}/{spec.filename}: {name} has shape {actual}, expected {shape}. "
                f"The file's convolution layout is not {spec.layout!r} as the catalogue "
                "declares -- loading it would upscale into noise rather than fail."
            )


def load_upscaler(spec: UpscalerSpec, *, allow_download: bool = True) -> Any:
    """Build `spec`'s network with its published weights applied.

    Blocking: it may download tens of megabytes. `ModelEngine` calls it on the
    inference worker with the progress snapshot in its `loading` state, which
    is what stops the first click looking like a hang.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from qds.upscale.rrdbnet import RRDBNet

    path = resolve_file(spec, allow_download=allow_download)
    raw = mx.load(path)
    verify_state_dict(spec, raw)

    dtype = mx.float16 if spec.dtype == "float16" else mx.float32
    transpose = spec.layout == "nchw"
    flat = {
        key: (value.transpose(0, 2, 3, 1) if transpose and value.ndim == 4 else value).astype(dtype)
        for key, value in raw.items()
    }

    model = RRDBNet(
        num_block=spec.num_block, num_feat=spec.num_feat, num_grow_ch=spec.num_grow_ch
    )
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())
    verify_loaded(model, spec)
    return model
