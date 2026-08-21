"""Anima's latents: 16 channels at one eighth of the image, and never packed.

FLUX and Qwen-Image fold each 2x2 latent patch into the channel axis before the
loop and unfold it afterwards, so their creators have real `pack`/`unpack` work
to do. Anima does not: the patching happens inside the transformer, at
`patch_embed`, and the loop carries plain `[B, 16, H/8, W/8]` latents throughout.

`unpack_latents` is therefore the identity, and it exists because that is the
interface previews are drawn through -- `engine._render_preview` unpacks a
mid-loop latent and hands it to the VAE. Returning the latent unchanged is the
correct unpacking here, not a stub.
"""

from __future__ import annotations

import mlx.core as mx

#: The VAE's spatial compression. Anima's is Qwen-Image's.
VAE_SCALE_FACTOR = 8

LATENT_CHANNELS = 16


class AnimaLatentCreator:
    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return mx.random.normal(
            shape=(
                1,
                LATENT_CHANNELS,
                height // VAE_SCALE_FACTOR,
                width // VAE_SCALE_FACTOR,
            ),
            key=mx.random.key(seed),
        )

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        return latents
