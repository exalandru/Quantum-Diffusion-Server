"""SD 3.5's VAE: FLUX.1's, differently normalised.

`vae/config.json` in all three repositories declares a plain diffusers `AutoencoderKL`
with 16 latent channels, `block_out_channels [128, 256, 512, 512]`,
`layers_per_block: 2` and `norm_num_groups: 32` — which is, tensor for tensor, the
encoder/decoder mflux already implements for FLUX.1. Checked against the real 244-key
tensor index rather than inferred from the config: 4 down blocks of 2 resnets with 3
downsamplers, 4 up blocks of 3 resnets with 3 upsamplers (diffusers decoders always
build `layers_per_block + 1`), one mid block with 2 resnets and 1 attention on each
side, and no `quant_conv`/`post_quant_conv` at all — `use_quant_conv` is false here,
as it is for FLUX.1.

So the whole of the difference is two constants. `scaling_factor` and `shift_factor`
are the affine map between the VAE's own latent space and the one the transformer was
trained in; FLUX.1's are 0.3611/0.1159 and SD 3.5's are 1.5305/0.0609. Inheriting the
module and overriding the pair keeps the maths mflux's and the normalisation ours,
which is the honest split: nothing about the convolutions changed.

mflux's `FluxWeightMapping.get_vae_mapping` is reused verbatim for the same reason. It
carries `quant_conv`/`post_quant_conv` targets that SD 3.5 has no source keys for, and
`WeightMapper` simply never emits a target whose source is absent — `test_sd35_weights.py`
asserts the mapped tree is exactly this module's parameter tree, so that claim is
checked rather than assumed.
"""

from __future__ import annotations

from mflux.models.flux.model.flux_vae.vae import VAE

from qds.sd35 import config as sd35_config


class SD35VAE(VAE):
    """FLUX.1's `AutoencoderKL` with SD 3.5's latent normalisation."""

    #: `vae/config.json`: `scaling_factor` 1.5305, `shift_factor` 0.0609. The base
    #: class applies them as `(latent / scaling) + shift` on decode and
    #: `(mean - shift) * scaling` on encode, which is diffusers' convention exactly.
    scaling_factor: float = sd35_config.VAE_SCALING_FACTOR
    shift_factor: float = sd35_config.VAE_SHIFT_FACTOR
    spatial_scale = sd35_config.VAE_SCALE_FACTOR
    latent_channels = sd35_config.LATENT_CHANNELS
