"""Real-ESRGAN's RRDBNet, ported to MLX.

The reference is basicsr's `RRDBNet` (`basicsr/archs/rrdbnet_arch.py`), and
this port mirrors its module names deliberately: the published checkpoints use
those names, flat and unprefixed, so `weights.py` needs no translation table at
all -- only a dtype cast and, for a torch-layout file, a transposition.
`tests/test_upscale.py` builds the same random weights into this port and into
a transcription of the reference in torch, and compares outputs.

Two things about the architecture are easy to get wrong and worth stating:

* **the network is always x4.** The two nearest-neighbour doublings below are
  hardcoded. Upstream's x2 and x1 checkpoints do not change them; they
  *pixel-unshuffle the input* first, which is why their `conv_first` takes 12
  or 48 channels instead of 3. This port serves only x4 checkpoints
  (`UpscalerSpec.native_scale`), so no unshuffle and no input-size divisibility
  constraint applies. Adding an x2 checkpoint would reintroduce both.
* **the 0.2 residual scalings are load-bearing.** Both the dense block and the
  RRDB scale their branch by 0.2 before adding. They are not initialisation
  details; the trained weights assume them.

Everything runs in NHWC, which is mlx's convolution layout, so the spatial
axes are 1 and 2 and channels are the last axis.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

#: Upstream's LeakyReLU slope. Every activation in the network uses it.
NEGATIVE_SLOPE = 0.2

#: Upstream's residual branch scaling, in both the dense block and the RRDB.
RESIDUAL_SCALE = 0.2


def nearest_upsample_2x(x: mx.array) -> mx.array:
    """Double both spatial axes by pixel replication, NHWC.

    Written out rather than delegated to `nn.Upsample(mode="nearest")`: at an
    integer factor this is exact by construction, with no dependence on how a
    resampler rounds sample positions or handles `align_corners`. The reference
    is `F.interpolate(scale_factor=2, mode="nearest")`, which is the same map.
    """
    b, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
    return x.reshape(b, h * 2, w * 2, c)


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    """Every convolution in RRDBNet: 3x3, stride 1, padding 1, with bias."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class ResidualDenseBlock(nn.Module):
    """Five convolutions, each seeing every earlier output concatenated."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = _conv3x3(num_feat, num_grow_ch)
        self.conv2 = _conv3x3(num_feat + num_grow_ch, num_grow_ch)
        self.conv3 = _conv3x3(num_feat + 2 * num_grow_ch, num_grow_ch)
        self.conv4 = _conv3x3(num_feat + 3 * num_grow_ch, num_grow_ch)
        self.conv5 = _conv3x3(num_feat + 4 * num_grow_ch, num_feat)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = nn.leaky_relu(self.conv1(x), NEGATIVE_SLOPE)
        x2 = nn.leaky_relu(self.conv2(mx.concatenate([x, x1], axis=-1)), NEGATIVE_SLOPE)
        x3 = nn.leaky_relu(self.conv3(mx.concatenate([x, x1, x2], axis=-1)), NEGATIVE_SLOPE)
        x4 = nn.leaky_relu(self.conv4(mx.concatenate([x, x1, x2, x3], axis=-1)), NEGATIVE_SLOPE)
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x5 * RESIDUAL_SCALE + x


class RRDB(nn.Module):
    """Three dense blocks in series, around one more residual connection."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * RESIDUAL_SCALE + x


class RRDBNet(nn.Module):
    """Real-ESRGAN's generator, x4, NHWC in and NHWC out.

    `body` is a plain Python list so that mlx names its parameters
    `body.{i}.rdb{j}.conv{k}.weight` -- exactly the checkpoint's own names, which
    is what lets `weights.py` apply a file with no rename table.
    """

    def __init__(self, num_block: int = 23, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.num_block = num_block
        self.num_feat = num_feat
        self.num_grow_ch = num_grow_ch
        self.conv_first = _conv3x3(3, num_feat)
        self.body = [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        self.conv_body = _conv3x3(num_feat, num_feat)
        self.conv_up1 = _conv3x3(num_feat, num_feat)
        self.conv_up2 = _conv3x3(num_feat, num_feat)
        self.conv_hr = _conv3x3(num_feat, num_feat)
        self.conv_last = _conv3x3(num_feat, 3)

    def __call__(self, x: mx.array) -> mx.array:
        feat = self.conv_first(x)
        body = feat
        for block in self.body:
            body = block(body)
        feat = feat + self.conv_body(body)
        feat = nn.leaky_relu(self.conv_up1(nearest_upsample_2x(feat)), NEGATIVE_SLOPE)
        feat = nn.leaky_relu(self.conv_up2(nearest_upsample_2x(feat)), NEGATIVE_SLOPE)
        return self.conv_last(nn.leaky_relu(self.conv_hr(feat), NEGATIVE_SLOPE))
