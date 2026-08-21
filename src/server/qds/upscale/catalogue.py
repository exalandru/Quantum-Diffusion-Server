"""The upscaler catalogue: which Real-ESRGAN checkpoints this server serves.

Every value here was read off the published artifacts rather than a model card.
The tensor names, shapes and counts come from the safetensors headers of the
files named below, fetched by range request; the architecture parameters come
from each repository's `config.json` and are cross-checked against those
headers by `tensor_count` (see below). `tests/test_upscale.py` pins the
arithmetic, so a catalogue entry that drifts from its checkpoint fails at load
rather than mis-upscaling.

This module must not import mlx, mflux, torch or huggingface_hub. `fetch` reads
it on the `--status` path, which `tests/test_cli.py` holds to importing none of
those, and `app` reads it at start-up to publish the list. The catalogue
describes; it does not touch anything.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Upper bound on a catalogue entry's weights, in megabytes.
#:
#: This is the bound that makes `ModelEngine`'s second resident slot safe to
#: exist at all (see its module docstring). The exception to "one live model"
#: was argued from three orders of magnitude, not from "an upscaler is small",
#: so the number is enforced here, at import, rather than trusted. Raising it
#: is a decision about the engine's memory invariant, not a catalogue edit.
MAX_WEIGHTS_MB = 200.0

#: Refuse a job that would make the network render more than this, in pixels.
#:
#: Deliberately measured on what the network *renders*, not on what the caller
#: asked for. The two are not the same, and bounding the wrong one leaves the
#: guard open: RRDBNet always upsamples x4, so a x2 request renders x4 and
#: resamples down. A 2048x2048 source asked for x2 therefore renders 8192x8192
#: -- four times the work of the 2048->8192 x4 request that the same limit
#: refuses. Reachable in two clicks, because an upscale's output is itself a
#: generated image and can be upscaled again.
#:
#: The value bounds the transient this costs on the host. `upscale_array`
#: quantises each tile as it lands, so the full-resolution buffer is one byte
#: per channel rather than four across three live copies. Measured at this
#: limit -- a 2048x2048 source, rendering 8192x8192 -- the process peaks at
#: 1.11 GB resident, against 3.88 GB before that change. The MLX allocator
#: stays at 1.52 GB whatever the source size, which is the tiling doing its
#: job. Beside a 10-28 GB resident diffusion model, a gigabyte is affordable.
MAX_RENDER_PIXELS = 8192 * 8192


@dataclass(frozen=True)
class UpscalerSpec:
    """One Real-ESRGAN checkpoint and everything needed to run it.

    Deliberately *not* a `registry.ModelSpec`. That dataclass carries steps,
    guidance, schedulers, prompt formats, quantisation and an edit variant --
    around eighteen fields with no meaning here. Reusing it would mean inventing
    values for all of them, and everything that reads a `ModelSpec`
    (`_capabilities`, `resolve_size`, `check_capabilities`, `cache_status`,
    `load_model`'s family dispatch) would read those inventions as facts about a
    generative model.
    """

    key: str
    display_name: str
    #: Hugging Face repository and file. Both are catalogue data rather than
    #: constants so that switching source is an edit here, not in `weights`.
    repo: str
    filename: str
    #: Convolution weight layout in the file. "nhwc" is what mlx wants; "nchw"
    #: is torch's, and `weights.load_upscaler` transposes it.
    #:
    #: The field exists because the fallback source has the other layout: the
    #: `mlx-community` files ship pre-permuted, while `Comfy-Org` and the
    #: various re-uploads ship torch's. Without it, changing `repo` would
    #: silently produce garbage rather than fail.
    layout: str
    #: RRDB blocks. The only architectural difference between the two entries.
    num_block: int
    #: Exact number of tensors the file must hold: `num_block * 30 + 12`.
    #: Thirty per block (3 dense blocks x 5 convolutions x weight+bias) and
    #: twelve for the six convolutions outside the body.
    #:
    #: Redundant with `num_block` by construction, and that is the point: it is
    #: a checked assertion about the file, not a parameter. `__post_init__`
    #: refuses an entry where the two disagree.
    tensor_count: int
    #: Download size, megabytes. Shown before the first use costs the user a
    #: wait, and bounded by `MAX_WEIGHTS_MB`.
    size_mb: float
    #: Upstream licence, reported as a fact with its source. As with
    #: `ModelSpec.license`, this states what the publisher declares; it is not
    #: legal advice, and the re-packagers' declaration is not the upstream's.
    license: str
    num_feat: int = 64
    num_grow_ch: int = 32
    #: What the network does, always. RRDBNet upsamples x4 internally through
    #: two hardcoded nearest-neighbour doublings; a x2 or x1 checkpoint gets
    #: there by *pixel-unshuffling its input* instead, which changes
    #: `conv_first`'s input channels and imposes a divisibility constraint on
    #: the input size. Neither entry below is such a checkpoint, so no
    #: `mod_pad` is needed -- but that is a consequence of this field's value,
    #: which is why it is a field.
    native_scale: int = 4
    #: Inference dtype. fp16 is upstream's default (it offers `--fp32` to opt
    #: out) and is what both files ship as. fp32 raises the *compute* precision
    #: from fp16 weights, which is the useful knob: twenty-three residual
    #: blocks accumulating in fp16 is where saturation would come from.
    dtype: str = "float16"
    #: Output tile side, in input pixels, and the context added around each
    #: tile's input before inference. Tiling is not optional here: the
    #: diffusion model stays resident, so an untiled peak is not affordable.
    #:
    #: Measured on this port, `realesrgan-x4plus` fp16, 1024x1024 -> 4096x4096,
    #: warm, one fresh process per row, best of three on an otherwise idle
    #: machine (timings taken while the GPU was also generating came out three
    #: to six times higher and are not these):
    #:
    #:     tile  tiles   time   MLX peak   host RSS
    #:      256     16    5.2s     2.47 GB    0.44 GB
    #:      192     36    6.0s     1.52 GB    0.37 GB
    #:      128     64    7.0s     1.14 GB    0.35 GB
    #:       96    121    8.7s     0.76 GB    0.37 GB
    #:       64    256    9.8s     0.42 GB    0.36 GB
    #:
    #: 192 is the shipped default: a 15% slower pass for a 38% smaller MLX
    #: peak. The whole reason tiling is here is that a 10-28 GB diffusion model
    #: stays loaded, so a gigabyte off the peak is worth eight tenths of a
    #: second. Host RSS is flat because the assembled buffer is uint8 and does
    #: not depend on the tile size.
    #:
    #: These live on the spec rather than in `server-config.json` on purpose.
    #: A configuration value needs a user able to choose it, and nobody can
    #: choose a tile size without measuring.
    tile: int = 192
    #: Context around each tile, matching upstream. Note what it does *not*
    #: buy: `rrdbnet`'s receptive radius is `1 + 15 * num_block + 1`, which is
    #: 347 pixels for the 23-block entry, so at 10 the tiled result is not
    #: bitwise identical to an untiled one -- it is about 3e-08 away.
    #: Negligible numerically; seams are perceptual and untested.
    tile_pad: int = 10

    def __post_init__(self) -> None:
        expected = self.num_block * 30 + 12
        if self.tensor_count != expected:
            raise ValueError(
                f"Upscaler {self.key!r} declares {self.tensor_count} tensors but "
                f"{self.num_block} blocks imply {expected}."
            )
        if self.size_mb > MAX_WEIGHTS_MB:
            raise ValueError(
                f"Upscaler {self.key!r} weighs {self.size_mb} MB, over the "
                f"{MAX_WEIGHTS_MB} MB bound that makes the engine's second "
                "resident slot safe. Raising it is a decision about "
                "`ModelEngine`'s memory invariant, not a catalogue edit."
            )
        if self.layout not in ("nhwc", "nchw"):
            raise ValueError(f"Upscaler {self.key!r} has unknown layout {self.layout!r}.")


#: Real-ESRGAN, BSD-3-Clause upstream. Both files are the `mlx-community`
#: re-packagings: fp16, already permuted to NHWC, and the only safetensors
#: publications of these two checkpoints that declare a licence.
#:
#: The x4plus file is verified: it is bit-exactly
#: `fp16(comfy.transpose(0, 2, 3, 1))` of
#: `Comfy-Org/Real-ESRGAN_repackaged/RealESRGAN_x4plus.safetensors`, checked on
#: five tensors spread across the network. The anime file has no independently
#: licensed counterpart to check against; see the README.
_LICENSE = "BSD-3-Clause (upstream xinntao/Real-ESRGAN), as declared by the re-packager"

SPECS: tuple[UpscalerSpec, ...] = (
    UpscalerSpec(
        key="realesrgan-x4plus",
        display_name="Real-ESRGAN x4 (photo)",
        repo="mlx-community/Real-ESRGAN-x4plus",
        filename="model.safetensors",
        layout="nhwc",
        num_block=23,
        tensor_count=702,
        size_mb=33.5,
        license=_LICENSE,
    ),
    UpscalerSpec(
        key="realesrgan-x4plus-anime",
        display_name="Real-ESRGAN x4 (illustration)",
        repo="mlx-community/Real-ESRGAN-x4plus-anime-6B",
        filename="model.safetensors",
        layout="nhwc",
        num_block=6,
        tensor_count=192,
        size_mb=9.0,
        license=_LICENSE,
    ),
)

_BY_KEY: dict[str, UpscalerSpec] = {spec.key: spec for spec in SPECS}

#: Catalogue keys, in presentation order.
KEYS: tuple[str, ...] = tuple(spec.key for spec in SPECS)

#: Scale factors the server offers. The network is always x4; x2 is that x4
#: resampled down with Lanczos, which is upstream's `--outscale` semantics. It
#: costs exactly the same time and memory as x4, and the UI says so.
SCALES: tuple[int, ...] = (2, 4)


def by_key(key: str) -> UpscalerSpec | None:
    """The spec for `key`, or `None` if the catalogue does not have it."""
    return _BY_KEY.get(key)


def tensor_names(spec: UpscalerSpec) -> frozenset[str]:
    """Exactly the tensor names `spec`'s checkpoint must hold.

    Derived from `num_block` rather than written out, so the two cannot drift.
    These are upstream's own names (basicsr's `RRDBNet`), which every
    publication of these checkpoints keeps flat and unprefixed.
    """
    names: list[str] = []
    for stem in ("conv_first", "conv_body", "conv_up1", "conv_up2", "conv_hr", "conv_last"):
        names += [f"{stem}.weight", f"{stem}.bias"]
    for block in range(spec.num_block):
        for rdb in (1, 2, 3):
            for conv in range(1, 6):
                stem = f"body.{block}.rdb{rdb}.conv{conv}"
                names += [f"{stem}.weight", f"{stem}.bias"]
    return frozenset(names)
