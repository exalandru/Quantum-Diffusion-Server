"""From a PNG on disk to an upscaled PNG, through a tiled RRDBNet.

Tiling is not an optimisation here, it is the price of the engine's second
resident slot: the diffusion model stays loaded, so an untiled pass -- which
allocates on the order of gigabytes for its widest activation -- is not
affordable.

Be precise about what tiling bounds, because it is not everything. It bounds the
*activations*, to a function of `tile`: measured constant at 1.52 GB whatever the
source size. It does **not** bound the assembled image, which is `O(output
pixels)` however small the tiles are. That buffer is held to one byte per channel
by quantising each tile as it lands, and capped by
`catalogue.MAX_RENDER_PIXELS`. Two separate mechanisms, and
`tests/test_upscale.py` has a witness for each.

The geometry is upstream's (`realesrgan/utils.py`): output tiles do not overlap,
each tile's *input* is widened by `tile_pad` on all four sides and clamped to
the image, and after inference that padding is cropped away and the remainder
written straight in. There is no feathering, exactly as upstream, so the padding
buys receptive-field context and nothing else. `tests/test_upscale.py` pins the
geometry by asking for bitwise equality with the untiled result when the pad
covers the whole receptive field; at the shipped `tile_pad` it does not, and a
seam is a perceptual property that no test here establishes.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from qds.errors import APIError
from qds.logs import SERVER_LOGGER
from qds.upscale.catalogue import UpscalerSpec

logger = logging.getLogger(SERVER_LOGGER)

#: Called with (tiles done, tiles total) after each tile. It may raise to
#: abort the run: that is how cancellation and the request deadline reach in.
TileHook = Callable[[int, int], None]


def tile_grid(width: int, height: int, tile: int) -> list[tuple[int, int, int, int]]:
    """The non-overlapping output tiles, as `(x0, y0, x1, y1)` in input pixels.

    `tile <= 0` means one tile covering everything, which is also what a small
    enough image produces naturally.
    """
    if tile <= 0:
        return [(0, 0, width, height)]
    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def upscale_array(
    model: Any,
    image: np.ndarray,
    *,
    tile: int,
    tile_pad: int,
    scale: int,
    dtype: str = "float16",
    on_tile: TileHook | None = None,
) -> np.ndarray:
    """Run `model` over `image` (H, W, 3) float32 in [0, 1] and return uint8.

    uint8 out rather than float32: see the buffer comment below. The caller gets
    pixels ready to wrap in an image, and nothing full-resolution is held in a
    wider dtype at any point.
    """
    import mlx.core as mx

    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB array, got {channels} channels.")

    tiles = tile_grid(width, height, tile)
    # uint8, and each tile quantised as it arrives rather than at the end.
    #
    # This is what keeps the *host* side of an upscale bounded, and it is not a
    # micro-optimisation: assembling in float32 and converting afterwards costs
    # a full-resolution float32 buffer plus the two more that `np.clip` and the
    # scaling produce — four bytes per output pixel becoming sixteen, three
    # times over. At 8192x8192 that is 2.4 GB of host memory transient beside a
    # resident diffusion model, which is precisely the pressure the tiling
    # exists to avoid.
    out = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    compute = mx.float16 if dtype == "float16" else mx.float32

    for index, (x0, y0, x1, y1) in enumerate(tiles):
        # Widen the input for context, clamped to the image.
        px0, py0 = max(x0 - tile_pad, 0), max(y0 - tile_pad, 0)
        px1, py1 = min(x1 + tile_pad, width), min(y1 + tile_pad, height)

        patch = mx.array(image[py0:py1, px0:px1][None].astype(np.float32)).astype(compute)
        result = model(patch)
        # `np.array` is the materialisation point: converting to numpy forces
        # the graph for this tile, which is what stops the tiles accumulating
        # into one. An `mx.eval` on the line above would be redundant, and a
        # comment claiming otherwise was wrong -- see `tests/test_upscale.py`.
        rendered = np.array(result, dtype=np.float32)[0]

        # fp16 accumulating over the residual blocks can saturate; upstream
        # offers `--fp32` for the same reason. Checked per tile so a run that
        # has gone bad stops here rather than after every remaining tile.
        #
        # This covers the padded window, context margin included, so a value
        # that would have been cropped away still fails the run. Stricter than
        # checking the assembled image, deliberately: a tile whose margin went
        # non-finite is a tile whose interior is not to be trusted either.
        if not np.isfinite(rendered).all():
            raise APIError(
                "The upscaler produced non-finite values, which fp16 accumulation over "
                "the residual blocks can cause. Retry with float32 precision.",
                status_code=500,
                error_type="server_error",
                code="upscale_not_finite",
            )

        # Crop the context back off and write the tile's own region.
        cx0, cy0 = (x0 - px0) * scale, (y0 - py0) * scale
        out[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = _to_uint8(
            rendered[cy0 : cy0 + (y1 - y0) * scale, cx0 : cx0 + (x1 - x0) * scale]
        )
        del patch, result, rendered

        if on_tile is not None:
            on_tile(index + 1, len(tiles))

    return out


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Clip *then* quantise. The other order wraps: Real-ESRGAN overshoots
    [0, 1] on highlights, and `uint8` of a negative float comes out near 255."""
    return (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def upscale_png(
    model: Any,
    source: Path,
    *,
    spec: UpscalerSpec,
    target: tuple[int, int],
    on_tile: TileHook | None = None,
) -> bytes:
    """Upscale the PNG at `source` to exactly `target` and return PNG bytes.

    A target rather than a factor because the target is what the caller
    actually decided, and what the playground row records. Deriving a factor
    back out of a stored size -- by dividing it by the source's -- would put a
    computed number where an authoritative one already exists.

    The network is always `spec.native_scale`. A smaller target is that result
    resampled down with Lanczos -- upstream's `--outscale` semantics -- so it
    costs exactly the same time and memory as the native factor.

    Alpha is a deliberate divergence from upstream, which runs the whole network
    a second time over the alpha channel promoted to grey. Here alpha is
    resampled instead: the images this server upscales are the RGB PNGs mflux
    writes, so the case is unreachable in practice, and when it is reached,
    doubling the cost of the work for a channel that is almost always a hard
    mask is a poor trade. Dropping alpha with `convert("RGB")` would be silent
    data loss, which is the option actually worth refusing.

    It is resampled bicubically rather than with Lanczos, and that is not
    incidental. Lanczos has negative lobes, so enlarging a hard mask with it
    rings: measured on an 8x8 step blown up 4x, a fully opaque run dips to 247
    and a fully transparent one rises to 8. On colour that is a familiar
    sharpening halo; on a mask it is ghost pixels in nominally invisible
    regions and a haze over nominally solid ones. Bicubic gives the same soft
    two-pixel edge with no overshoot either side. Colour keeps Lanczos below,
    where the operation is a reduction and upstream's `--outscale` uses
    `INTER_LANCZOS4`.
    """
    with Image.open(source) as opened:
        opened.load()
        alpha = opened.getchannel("A") if "A" in opened.getbands() else None
        rgb = opened.convert("RGB")
        # RGB, not BGR: upstream reads BGR through OpenCV and converts, so the
        # weights expect RGB either way.
        image = np.asarray(rgb, dtype=np.float32) / 255.0

    rendered = upscale_array(
        model,
        image,
        tile=spec.tile,
        tile_pad=spec.tile_pad,
        scale=spec.native_scale,
        dtype=spec.dtype,
        on_tile=on_tile,
    )
    result = Image.fromarray(rendered, mode="RGB")

    if result.size != target:
        result = result.resize(target, Image.LANCZOS)
    if alpha is not None:
        result.putalpha(alpha.resize(target, Image.BICUBIC))

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()
