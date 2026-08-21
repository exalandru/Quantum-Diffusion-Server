"""Real-ESRGAN port: parity with the reference, and the tiling geometry.

The two oracles here are deliberately strong ones.

`test_rrdbnet_matches_the_torch_reference` transcribes basicsr's `RRDBNet` into
torch inside this file and compares outputs tensor-for-tensor. The reference is
a transcription of the published architecture, not a call into the code under
test, so the two cannot agree by construction: a flipped transposition, a wrong
padding, a mis-ordered concatenation, the wrong LeakyReLU slope, a missing 0.2
residual scaling, `bilinear` instead of `nearest`, or `conv_hr` and `conv_last`
swapped all break it.

`test_tiling_is_exact_when_the_pad_covers_the_receptive_field` asks for bitwise
equality rather than a tolerance, which is available because the tiled and
untiled paths compute the same function when each tile carries its full
receptive field as context. Its counter-test pins that the assertion has teeth.

torch is available because mflux declares it a hard dependency
(`torch>=2.7.1,<3.0`) and imports it at module scope in its weight loader, so it
is resident in the server process anyway. It is still `importorskip`ed, as
`test_anima.py` does, because the property being checked is about this port.
"""

from __future__ import annotations

import io
import tracemalloc

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from qds.errors import APIError
from qds.upscale.catalogue import SPECS, UpscalerSpec, tensor_names
from qds.upscale.pipeline import _to_uint8, tile_grid, upscale_array, upscale_png
from qds.upscale.rrdbnet import RRDBNet, nearest_upsample_2x

# A network small enough to build and run many times, but with every structural
# feature of the real thing: more than one block, all three dense blocks, both
# upsampling stages.
TOY = {"num_block": 2, "num_feat": 8, "num_grow_ch": 4}


def _torch_reference(torch):
    """basicsr's `RRDBNet`, transcribed. Deliberately verbose and literal."""
    nn = torch.nn
    F = torch.nn.functional

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat, num_grow_ch):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb3(self.rdb2(self.rdb1(x)))
            return out * 0.2 + x

    class Reference(nn.Module):
        def __init__(self, num_block, num_feat, num_grow_ch):
            super().__init__()
            self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return Reference


def _mlx_from_torch(state_dict, **config) -> RRDBNet:
    """Load a torch state dict into the port: transpose 4-D, keep 1-D."""
    from mlx.utils import tree_unflatten

    # `body.0.…` in the reference is `nn.Sequential`, whose parameter names are
    # the same `body.{i}.…` the port produces from a plain list.
    def convert(tensor):
        array = tensor.detach().numpy()
        return mx.array(array.transpose(0, 2, 3, 1) if array.ndim == 4 else array)

    flat = {key: convert(value) for key, value in state_dict.items()}
    model = RRDBNet(**config)
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())
    return model


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_rrdbnet_matches_the_torch_reference(torch):
    """The port computes the reference's function, in fp32, to ~1e-7."""
    torch.manual_seed(0)
    reference = _torch_reference(torch)(**TOY).eval()

    rng = np.random.default_rng(0)
    image = rng.standard_normal((1, 3, 16, 20), dtype=np.float32)

    with torch.no_grad():
        expected = reference(torch.from_numpy(image)).numpy()

    ported = _mlx_from_torch(reference.state_dict(), **TOY)
    actual = np.array(ported(mx.array(image.transpose(0, 2, 3, 1))))

    # Back to NCHW to compare against the reference's own layout.
    actual = actual.transpose(0, 3, 1, 2)
    assert actual.shape == expected.shape == (1, 3, 64, 80)
    assert np.abs(actual - expected).max() < 1e-5


def test_the_port_and_the_reference_have_the_same_parameter_names(torch):
    """No rename table: the checkpoint's names are the module's names."""
    from mlx.utils import tree_flatten

    reference = _torch_reference(torch)(**TOY)
    ported = RRDBNet(**TOY)
    assert {k for k, _ in tree_flatten(ported.parameters())} == set(reference.state_dict())


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_catalogue_entries_describe_the_module_they_build(spec: UpscalerSpec):
    """`tensor_count` and `tensor_names` are checked against the real module.

    This is what makes `verify_state_dict` meaningful: the names it demands of a
    checkpoint are the names the constructed network actually has.
    """
    from mlx.utils import tree_flatten

    model = RRDBNet(num_block=spec.num_block, num_feat=spec.num_feat, num_grow_ch=spec.num_grow_ch)
    names = {k for k, _ in tree_flatten(model.parameters())}
    assert names == tensor_names(spec)
    assert len(names) == spec.tensor_count


def test_conv_weights_are_nhwc():
    """`conv_last.weight` is the discriminating shape between the layouts.

    `conv_first.weight` is `(64, 3, 3, 3)` in *both* NCHW and NHWC because its
    input channels and its kernel side are both 3, so it cannot tell them
    apart. `conv_last` can: `(3, 3, 3, 64)` here against torch's `(3, 64, 3, 3)`.
    """
    model = RRDBNet(**TOY)
    assert model.conv_last.weight.shape == (3, 3, 3, TOY["num_feat"])


def test_nearest_upsample_replicates_pixels():
    x = mx.array(np.arange(2 * 3 * 1, dtype=np.float32).reshape(1, 2, 3, 1))
    out = np.array(nearest_upsample_2x(x))[0, :, :, 0]
    assert out.shape == (4, 6)
    assert np.array_equal(out[0], np.array([0, 0, 1, 1, 2, 2], dtype=np.float32))
    assert np.array_equal(out[0], out[1])


# --- The load guards -------------------------------------------------------


def _fake_state_dict(spec: UpscalerSpec, *, layout: str = "nhwc") -> dict[str, mx.array]:
    """A structurally valid checkpoint for `spec`, in either layout."""
    model = RRDBNet(num_block=spec.num_block, num_feat=spec.num_feat, num_grow_ch=spec.num_grow_ch)
    from mlx.utils import tree_flatten

    raw = {}
    for name, value in tree_flatten(model.parameters()):
        if layout == "nchw" and value.ndim == 4:
            value = value.transpose(0, 3, 1, 2)
        raw[name] = value
    return raw


SMALL = SPECS[1]  # the 6-block entry: same code path, a fifth of the tensors


def test_module_update_is_only_half_strict():
    """The measured behaviour `verify_state_dict` exists for.

    Pinned rather than asserted in prose, because the guard's whole
    justification is this asymmetry and an mlx release could change it: an
    unknown key raises, but a parameter no key reached keeps its random
    initialisation without a word. That second half is what would let a
    truncated checkpoint upscale into noise.
    """
    from mlx.utils import tree_unflatten

    model = RRDBNet(**TOY)
    with pytest.raises(ValueError):
        model.update(tree_unflatten([("nonsense.weight", mx.zeros((1,)))]))

    before = np.array(model.conv_last.weight)
    model.update(tree_unflatten([("conv_first.weight", mx.zeros_like(model.conv_first.weight))]))
    assert np.array_equal(np.array(model.conv_last.weight), before), (
        "an unreached parameter was not left alone -- if mlx now zeroes or "
        "errors on these, verify_state_dict's stated reason needs revisiting"
    )


def test_verify_state_dict_accepts_the_checkpoint_it_describes():
    from qds.upscale.weights import verify_state_dict

    verify_state_dict(SMALL, _fake_state_dict(SMALL))


def test_verify_state_dict_refuses_a_missing_tensor():
    from qds.upscale.weights import verify_state_dict

    raw = _fake_state_dict(SMALL)
    del raw["body.3.rdb2.conv4.weight"]
    with pytest.raises(ValueError, match="1 missing"):
        verify_state_dict(SMALL, raw)


def test_verify_state_dict_refuses_an_unexpected_tensor():
    from qds.upscale.weights import verify_state_dict

    raw = _fake_state_dict(SMALL)
    raw["conv_extra.weight"] = mx.zeros((1, 1, 1, 1))
    with pytest.raises(ValueError, match="1 unexpected"):
        verify_state_dict(SMALL, raw)


def test_verify_state_dict_refuses_the_wrong_block_count():
    """The 23-block file is a valid checkpoint -- just not this entry's."""
    from qds.upscale.weights import verify_state_dict

    with pytest.raises(ValueError, match="192 tensors"):
        verify_state_dict(SMALL, _fake_state_dict(SPECS[0]))


def test_verify_loaded_refuses_a_torch_layout_file():
    """The case a count cannot catch, and the reason `layout` is a spec field.

    A torch-layout checkpoint has every name right and every count right. If it
    were applied under `layout="nhwc"`, the network would run and upscale into
    noise. This is the guard that turns that into a load failure.
    """
    from mlx.utils import tree_unflatten

    from qds.upscale.weights import verify_loaded, verify_state_dict

    raw = _fake_state_dict(SMALL, layout="nchw")
    verify_state_dict(SMALL, raw)  # names and count are fine; that is the point

    model = RRDBNet(num_block=SMALL.num_block, num_feat=SMALL.num_feat)
    model.update(tree_unflatten(list(raw.items())))
    with pytest.raises(ValueError, match="convolution layout"):
        verify_loaded(model, SMALL)


def test_verify_loaded_accepts_a_correctly_built_module():
    from qds.upscale.weights import verify_loaded

    model = RRDBNet(num_block=SMALL.num_block, num_feat=SMALL.num_feat)
    verify_loaded(model, SMALL)


def test_importing_the_upscale_package_stays_light():
    """`fetch --status` and the app's start-up read the catalogue; neither may
    pay for mlx, and `tests/test_cli.py` holds the CLI to the same rule."""
    import subprocess
    import sys

    probe = (
        "import sys; import qds.upscale; "
        "heavy = [m for m in sys.modules "
        "if m.split('.')[0] in {'mflux', 'torch', 'transformers', 'mlx'}]; "
        "print(len(heavy))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "0", result.stdout


# --- Tiling ----------------------------------------------------------------


def receptive_radius(num_block: int) -> int:
    """Input pixels a single output pixel depends on, either side.

    One for `conv_first`, fifteen per RRDB (three dense blocks of five 3x3
    convolutions), one for `conv_body`. The convolutions after the two
    upsamplings run at 2x and 4x, so they do not widen this in input space.

    This is an upper bound on what a tile needs, and that is the only claim
    made for it. It is emphatically *not* the threshold below which tiling
    stops being exact: measured on the port at 97x97, equality with the untiled
    result still holds down to a pad of 9, 7 and 8 for one, two and three
    blocks -- 53%, 22% and 17% of the radius. Two reasons. Context beyond a few
    pixels changes a pixel by less than the quantisation step, so the uint8
    output hides it; and even in float32 the residual at one below the radius
    was around 4e-09 for a single block and exactly zero for two and three.

    The tests use the full radius because it is the value guaranteed sufficient
    by construction, not because anything measured says it is the boundary.
    """
    return 1 + 15 * num_block + 1


def _toy_net(num_block: int = 2):
    mx.random.seed(0)
    model = RRDBNet(num_block=num_block, num_feat=6, num_grow_ch=4)
    mx.eval(model.parameters())
    return model


def _noise(height: int, width: int) -> np.ndarray:
    return np.random.default_rng(1).random((height, width, 3), dtype=np.float32)


@pytest.mark.parametrize(
    ("height", "width", "label"),
    [
        (32, 32, "exact multiple of the tile"),
        (40, 44, "a partial last tile on both axes"),
        (12, 9, "smaller than one tile"),
        # 97, not 96: at 96 the middle tile's padded window reaches x0=16 on
        # the low side but `min(x1 + pad, width)` lands exactly on 96, so the
        # high side is still clamped. One more pixel and tile (48, 48) is
        # strictly inside on all four sides -- the only case here where none of
        # the four clamps fires, and so the only one that exercises the
        # unclamped arithmetic.
        (97, 97, "an interior tile, padded on all four sides"),
    ],
)
def test_tiling_is_exact_when_the_pad_covers_the_receptive_field(height, width, label):
    """Bitwise equality, not a tolerance.

    The tiled and untiled paths compute the same function once each tile carries
    its full receptive field as context, so anything less than exact equality
    means the geometry is wrong -- an off-by-one in the crop, a pad not clamped
    at the image edge, a tile written to the wrong place.

    What this does *not* establish: that the shipped settings are seam-free.
    `realesrgan-x4plus` has 23 blocks, so its receptive radius is 347 pixels
    against a shipped `tile_pad` of 10. The residual there is around 3e-08 --
    negligible numerically, but seams are a perceptual property and no test here
    speaks to them.
    """
    model = _toy_net()
    image = _noise(height, width)
    whole = upscale_array(model, image, tile=0, tile_pad=0, scale=4, dtype="float32")
    tiled = upscale_array(model, image, tile=16, tile_pad=receptive_radius(2), scale=4, dtype="float32")
    assert tiled.shape == whole.shape == (height * 4, width * 4, 3), label
    assert np.abs(tiled - whole).max() == 0.0, label


def test_an_ignored_tile_pad_does_not_pass_the_exactness_test():
    """The counter-test that gives the one above its teeth.

    Without it, a tiling that never widened its input at all would still satisfy
    "tiled equals untiled" on any image small enough to be a single tile.
    """
    model = _toy_net()
    image = _noise(40, 44)
    whole = upscale_array(model, image, tile=0, tile_pad=0, scale=4, dtype="float32")
    unpadded = upscale_array(model, image, tile=16, tile_pad=0, scale=4, dtype="float32")
    # Integers, so an integer threshold: these are quantised pixels, and a
    # float tolerance here would be measuring nothing. Note that quantisation
    # costs the exactness test above some sensitivity too -- an error smaller
    # than half a level is invisible now. A misplaced tile is nowhere near
    # that small, but the assertion is weaker than it was.
    assert np.abs(unpadded.astype(int) - whole.astype(int)).max() >= 1


def test_tile_grid_covers_the_image_exactly_once():
    tiles = tile_grid(40, 44, 16)
    assert len(tiles) == 3 * 3  # ceil(40/16) x ceil(44/16)
    covered = np.zeros((44, 40), dtype=np.int32)
    for x0, y0, x1, y1 in tiles:
        covered[y0:y1, x0:x1] += 1
    assert covered.min() == covered.max() == 1


def test_tile_grid_degenerates_to_one_tile():
    assert tile_grid(40, 44, 0) == [(0, 0, 40, 44)]
    assert tile_grid(10, 10, 256) == [(0, 0, 10, 10)]


def test_on_tile_reports_every_tile_and_can_abort():
    model = _toy_net()
    seen: list[tuple[int, int]] = []
    upscale_array(
        model,
        _noise(40, 44),
        tile=16,
        tile_pad=2,
        scale=4,
        dtype="float32",
        on_tile=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(i + 1, 9) for i in range(9)]

    class Stop(Exception):
        pass

    def abort(done: int, total: int) -> None:
        if done == 2:
            raise Stop

    with pytest.raises(Stop):
        upscale_array(model, _noise(40, 44), tile=16, tile_pad=2, scale=4, dtype="float32", on_tile=abort)


# --- PNG in, PNG out -------------------------------------------------------


class RepeatX4:
    """A stand-in for the network: nearest x4, so the pipeline is what is tested."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.repeat(mx.repeat(x, 4, axis=1), 4, axis=2)


def _write_png(tmp_path, image: np.ndarray, mode: str = "RGB"):
    path = tmp_path / f"source-{mode}.png"
    Image.fromarray(image, mode=mode).save(path)
    return path


@pytest.mark.parametrize("outscale", [2, 4])
def test_upscale_png_produces_the_requested_size(tmp_path, outscale):
    source = _write_png(tmp_path, np.full((12, 20, 3), 128, dtype=np.uint8))
    data = upscale_png(RepeatX4(), source, spec=SMALL, target=(20 * outscale, 12 * outscale))
    with Image.open(io.BytesIO(data)) as out:
        assert out.format == "PNG"
        assert out.size == (20 * outscale, 12 * outscale)


def test_upscale_png_clips_rather_than_wrapping(tmp_path):
    """A network that overshoots [0, 1] must saturate, not wrap around.

    Real-ESRGAN does overshoot on highlights. `uint8` of a negative float comes
    out near 255, so quantising before clipping turns dark pixels white.
    """

    class Overshoot:
        def __call__(self, x: mx.array) -> mx.array:
            return RepeatX4()(x) * 4.0 - 1.5

    source = _write_png(tmp_path, np.full((8, 8, 3), 10, dtype=np.uint8))
    data = upscale_png(Overshoot(), source, spec=SMALL, target=(32, 32))
    with Image.open(io.BytesIO(data)) as out:
        pixels = np.asarray(out)
    # 10/255 * 4 - 1.5 < 0 everywhere, so every pixel must be black.
    assert pixels.min() == pixels.max() == 0


def test_upscale_png_keeps_an_alpha_channel(tmp_path):
    """Alpha is resampled rather than dropped or run through the network."""
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    rgba[:4, :, 3] = 255  # top half opaque, bottom half transparent
    source = _write_png(tmp_path, rgba, mode="RGBA")

    data = upscale_png(RepeatX4(), source, spec=SMALL, target=(32, 32))
    with Image.open(io.BytesIO(data)) as out:
        assert out.mode == "RGBA"
        alpha = np.asarray(out.getchannel("A"))
    assert alpha.shape == (32, 32)
    # The step itself is spread over two or three rows: a resampled hard mask
    # has a soft boundary, which is expected and not something to assert away.
    # What must NOT happen is ringing. Lanczos, which colour uses, has negative
    # lobes and would put a 247 dip inside the opaque run and an 8 bump inside
    # the transparent one -- ghost pixels where the image is meant to be gone,
    # and that is what this rejects. It does not pin BICUBIC specifically:
    # NEAREST and BILINEAR do not ring either and would pass. What it pins is
    # the property, which is the right thing for a test to hold.
    assert alpha[:14].min() == 255
    assert alpha[18:].max() == 0
    boundary = alpha[13:19, 0].astype(int)
    assert all(earlier >= later for earlier, later in zip(boundary, boundary[1:], strict=False))


def test_non_finite_output_is_refused_rather_than_saved(tmp_path):
    """fp16 accumulating over 23 residual blocks can saturate. A black PNG
    delivered as a result is worse than a failure."""

    class Saturate:
        def __call__(self, x: mx.array) -> mx.array:
            return RepeatX4()(x) * mx.array(float("inf"))

    source = _write_png(tmp_path, np.full((8, 8, 3), 128, dtype=np.uint8))
    with pytest.raises(APIError) as caught:
        upscale_png(Saturate(), source, spec=SMALL, target=(32, 32))
    assert caught.value.code == "upscale_not_finite"


def test_tiling_bounds_the_mlx_peak(tmp_path):
    """Tiling actually costs less MLX memory than not tiling.

    Note what this does *not* witness, because an earlier version of it claimed
    to. It was written as the guard against dropping an `mx.eval` from the tile
    loop, on the theory that the tiles would otherwise accumulate into one lazy
    graph. That theory was wrong: `np.array(result)` materialises each tile on
    its own, so there is nothing for an `mx.eval` to force, and the test passed
    with the call removed. The call is gone; this measures the property it can
    actually see.
    """
    model = _toy_net()
    image = _noise(96, 96)  # 36 tiles at tile=16

    mx.clear_cache()
    mx.reset_peak_memory()
    upscale_array(model, image, tile=16, tile_pad=4, scale=4, dtype="float32")
    tiled_peak = mx.get_peak_memory()

    mx.clear_cache()
    mx.reset_peak_memory()
    upscale_array(model, image, tile=0, tile_pad=0, scale=4, dtype="float32")
    whole_peak = mx.get_peak_memory()

    assert tiled_peak < whole_peak / 2, (
        f"tiled peak {tiled_peak} vs untiled {whole_peak}: tiling is not bounding "
        "the MLX allocator at all."
    )


def test_quantising_per_tile_changes_no_pixel():
    """The memory fix must be invisible in the output.

    Quantising each tile as it lands, rather than assembling in float32 and
    converting once, is what keeps the host buffer to one byte per channel.
    `_to_uint8` is elementwise so the two orders agree by construction -- which
    is exactly the kind of "obviously equal" that stops being true the moment
    somebody adds a normalisation or a dither. Pinned rather than reasoned.
    """
    model = _toy_net()
    image = _noise(40, 44)
    per_tile = upscale_array(model, image, tile=16, tile_pad=32, scale=4, dtype="float32")

    # The same run, assembled wide and converted at the end.
    whole = np.empty((160, 176, 3), dtype=np.float32)
    for x0, y0, x1, y1 in tile_grid(44, 40, 16):
        px0, py0 = max(x0 - 32, 0), max(y0 - 32, 0)
        px1, py1 = min(x1 + 32, 44), min(y1 + 32, 40)
        rendered = np.array(model(mx.array(image[py0:py1, px0:px1][None])), dtype=np.float32)[0]
        cx0, cy0 = (x0 - px0) * 4, (y0 - py0) * 4
        whole[y0 * 4 : y1 * 4, x0 * 4 : x1 * 4] = rendered[
            cy0 : cy0 + (y1 - y0) * 4, cx0 : cx0 + (x1 - x0) * 4
        ]

    assert np.array_equal(per_tile, _to_uint8(whole))


def test_the_host_buffer_stays_near_one_byte_per_channel():
    """The host side of the same property, which the MLX peak cannot see.

    `mx.get_peak_memory` reports the MLX allocator only. The full-resolution
    buffer lives in numpy, and it is *not* bounded by the tile size -- it scales
    with the output. What keeps it small is that each tile is quantised as it
    lands, so the buffer is one byte per channel rather than four with two more
    full-size copies alive during the conversion. On an 8192x8192 render that
    is the difference between about 200 MB and about 2.4 GB, beside a resident
    diffusion model.

    Asserted on the measured peak, not on the returned dtype. An earlier version
    of this test checked `out.dtype == np.uint8` and called itself the witness
    for the whole property; it passed unchanged against a deliberate
    reimplementation of the float32 assembly, because that regression also
    *returns* uint8 -- it just builds a float32 image first.

    Where the bound sits, measured rather than assumed. As shipped: 2.15x the
    output, and 2.36x on the first call in a fresh process, because building
    the toy network and the noise happens inside the traced region. The
    *nearest* regression is not the float32 one -- it is assembling in uint16
    or float16, at 3.58x. So 3.0 sits between 2.36 and 3.58, with roughly a
    quarter of headroom below and a fifth above. That is a real margin but not
    a generous one, and halving the assembled dtype is the only intermediate
    mistake available.

    Two blind spots worth naming. `tracemalloc` sees CPython allocations, which
    covers numpy (it routes data allocations through the tracemalloc hooks from
    1.22) but not the MLX allocator at all -- a regression that assembled the
    full image into an `mx.array` would be invisible here. And the tile figures
    are hardcoded, so this says nothing about the shipped `tile`.
    """
    output_bytes = 1024 * 1024 * 3
    tracemalloc.start()
    try:
        out = upscale_array(
            _toy_net(), _noise(256, 256), tile=64, tile_pad=8, scale=4, dtype="float32"
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert out.dtype == np.uint8
    assert out.nbytes == output_bytes
    assert peak < 3 * output_bytes, (
        f"peak {peak / 1e6:.1f} MB is {peak / output_bytes:.1f}x the {output_bytes / 1e6:.1f} MB "
        "output. Assembling full-resolution pixels in anything wider than uint8 is what "
        "this rejects."
    )


# --- The catalogue's own guards --------------------------------------------
#
# `catalogue.py` says the tensor count is "enforced here, at import, rather
# than trusted", and that the weight bound "fails at import, not in
# production". Both were true and neither was witnessed.


def _spec(**patch):
    from dataclasses import replace

    return replace(SPECS[1], **patch)


def test_a_tensor_count_that_disagrees_with_the_block_count_is_refused():
    with pytest.raises(ValueError, match="declares 193 tensors but 6 blocks imply 192"):
        _spec(tensor_count=193)


def test_an_upscaler_over_the_weight_bound_is_refused():
    """The bound that makes the engine's second resident slot safe.

    A 2 GB entry added without rereading `engine.py`'s invariant is the way
    this exception stops being safe, so it fails where it is written down.
    """
    from qds.upscale.catalogue import MAX_WEIGHTS_MB

    with pytest.raises(ValueError, match="resident slot safe"):
        _spec(size_mb=MAX_WEIGHTS_MB + 1)


def test_an_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="unknown layout"):
        _spec(layout="nchw16")


def test_the_shipped_catalogue_satisfies_its_own_guards():
    """They run at import, so this passing at all is most of the assertion."""
    assert len(SPECS) == 2
    assert all(spec.size_mb < 200 for spec in SPECS)
    assert all(spec.layout in ("nhwc", "nchw") for spec in SPECS)
