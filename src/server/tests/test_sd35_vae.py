"""SD 3.5's VAE and its latents.

The VAE is mflux's FLUX.1 autoencoder with two constants replaced, so the question
these tests answer is not "is the convolution stack right" — `test_sd35_weights.py`
already proves the parameter tree is the published 244-tensor set — but "is the
*normalisation* right, and did overriding it leave FLUX.1 alone".

`scaling_factor` and `shift_factor` are the affine map between the autoencoder's own
latent space and the one the transformer was trained in. Getting them wrong does not
crash: it produces a washed-out or over-saturated image from an otherwise correct
model, which is the hardest kind of error to attribute.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx

from qds.sd35 import config as sd35_config
from qds.sd35.latent_creator import SD35LatentCreator
from qds.sd35.vae import SD35VAE

INDEX = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "sd35_tensor_index.json").read_text()
)


def test_the_normalisation_constants_are_the_published_ones():
    """From `vae/config.json`, which is byte-identical across the three repositories."""
    published = INDEX["vae_config"]
    assert SD35VAE.scaling_factor == published["scaling_factor"] == 1.5305
    assert SD35VAE.shift_factor == published["shift_factor"] == 0.0609
    assert SD35VAE.latent_channels == published["latent_channels"] == 16
    assert sd35_config.VAE_SCALING_FACTOR == published["scaling_factor"]
    assert sd35_config.VAE_SHIFT_FACTOR == published["shift_factor"]
    # The published config says there is no quantisation convolution, and the tensor
    # index agrees — so the reused mapping's targets for it must find no source.
    assert published["use_quant_conv"] is False
    assert published["use_post_quant_conv"] is False
    assert not [key for key in INDEX["tensors"]["shared"]["vae"] if "quant_conv" in key]


def test_overriding_the_constants_left_flux1_alone():
    """A subclass, not a patch: mflux's own VAE keeps FLUX.1's normalisation."""
    from mflux.models.flux.model.flux_vae.vae import VAE

    assert VAE.scaling_factor == 0.3611
    assert VAE.shift_factor == 0.1159
    assert issubclass(SD35VAE, VAE)


def test_the_topology_is_the_one_the_checkpoint_describes():
    """Four down blocks of two resnets, four up blocks of three. Counted, not assumed.

    diffusers builds `layers_per_block + 1` resnets per *up* block and
    `layers_per_block` per down block, which is why `layers_per_block: 2` and twelve
    decoder resnets are consistent rather than contradictory.
    """
    published = INDEX["vae_config"]
    assert published["layers_per_block"] == 2
    assert published["block_out_channels"] == [128, 256, 512, 512]

    keys = INDEX["tensors"]["shared"]["vae"]
    down = {".".join(k.split(".")[:5]) for k in keys if ".down_blocks." in k and ".resnets." in k}
    up = {".".join(k.split(".")[:5]) for k in keys if ".up_blocks." in k and ".resnets." in k}
    assert len(down) == 4 * 2
    assert len(up) == 4 * 3

    vae = SD35VAE()
    assert len(vae.encoder.down_blocks) == 4
    assert len(vae.decoder.up_blocks) == 4
    assert all(len(block.resnets) == 2 for block in vae.encoder.down_blocks)
    assert all(len(block.resnets) == 3 for block in vae.decoder.up_blocks)


def test_decode_and_encode_round_trip_at_the_right_scale():
    """8x in each direction, 16 channels in, 3 out. Finite, and the right shape."""
    vae = SD35VAE()
    latents = mx.random.normal(shape=(1, 16, 8, 8), key=mx.random.key(0)) * 0.1

    decoded = vae.decode(latents)
    mx.eval(decoded)
    # mflux's VAE carries a frame axis for the video models that share the class:
    # `decode` returns `[B, 3, 1, H, W]`, which is what `ImageUtil.to_image` reads.
    assert decoded.shape == (1, 3, 1, 64, 64)
    assert bool(mx.all(mx.isfinite(decoded)))

    re_encoded = vae.encode(decoded[:, :, 0])
    mx.eval(re_encoded)
    assert re_encoded.shape == (1, 16, 1, 8, 8)
    assert bool(mx.all(mx.isfinite(re_encoded)))


def test_the_affine_map_is_applied_in_the_direction_diffusers_uses():
    """`(latent / scaling) + shift` on the way in, `(mean - shift) * scaling` on the way out.

    Checked by driving the decoder with a constant and reading what it was handed,
    rather than by re-deriving the arithmetic the implementation already contains.
    """
    vae = SD35VAE()
    seen = {}
    original = vae.decoder

    def spy(latents):
        seen["input"] = latents
        return original(latents)

    vae.decoder = spy
    latents = mx.ones((1, 16, 4, 4))
    vae.decode(latents)
    mx.eval(seen["input"])
    expected = (1.0 / sd35_config.VAE_SCALING_FACTOR) + sd35_config.VAE_SHIFT_FACTOR
    assert abs(float(seen["input"][0, 0, 0, 0]) - expected) < 1e-6


def test_the_latent_creator_produces_the_shape_the_loop_carries():
    """`[1, 16, h/8, w/8]`, unpacked, for every size the catalogue offers."""
    for height, width in ((1024, 1024), (768, 1344), (512, 512), (1536, 1024)):
        noise = SD35LatentCreator.create_noise(7, height, width)
        assert noise.shape == (1, 16, height // 8, width // 8)
        # Previews go through `unpack_latents`; SD 3.5 packs nothing, so it is identity.
        assert SD35LatentCreator.unpack_latents(noise, height, width) is noise


def test_the_noise_is_reproducible_from_the_seed():
    """Same seed, same latents — the precondition for every reference comparison."""
    first = SD35LatentCreator.create_noise(1234, 256, 256)
    second = SD35LatentCreator.create_noise(1234, 256, 256)
    other = SD35LatentCreator.create_noise(1235, 256, 256)
    mx.eval(first, second, other)
    assert bool(mx.array_equal(first, second))
    assert not bool(mx.array_equal(first, other))
