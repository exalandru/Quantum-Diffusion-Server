"""The MMDiT-X transformer, checked against the real SD 3.5 checkpoints.

`fixtures/sd35_tensor_index.json` is a recording of the three gated
`stabilityai/stable-diffusion-3.5-*` repositories: every component's `config.json`,
and every tensor name with its shape, read from the safetensors headers themselves.
The point of testing against it rather than against a re-derivation is that a
re-derivation could not fail for the case that matters — a module whose structure has
drifted from the weights it is supposed to hold.

The central test here builds each variant's transformer and asserts its parameter
tree *is* the checkpoint's tensor set: same names, same shapes, nothing missing and
nothing extra. That single assertion covers the three structural facts that produce a
loadable-but-wrong model if got wrong — which blocks are dual-attention, which block
is `context_pre_only`, and how wide each modulation projection is — because each of
them changes the parameter tree in a way the checkpoint disagrees with.

Constructing an 8.1B-parameter module costs nothing here: MLX is lazy, so the
initialisers are graph nodes until something evaluates them, and only shapes are read.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from qds.sd35 import config as sd35_config
from qds.sd35.transformer import SD35Transformer, timestep_features

INDEX = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "sd35_tensor_index.json").read_text()
)

#: Which recorded transformer each catalogue row's overrides describe. Medium and the
#: two large releases are the only two shapes; Large Turbo's tensor set is identical
#: to Large's, which the fixture records once.
VARIANTS = {
    "medium": (sd35_config.MEDIUM_TRANSFORMER_OVERRIDES, "medium"),
    "large": (sd35_config.LARGE_TRANSFORMER_OVERRIDES, "large"),
}


def _recorded_config(name: str) -> dict:
    return INDEX["transformer_config"][name]


def test_the_overrides_are_the_published_configs():
    """Every number in `config.py` comes from `transformer/config.json`, not a card."""
    for name, (overrides, _) in VARIANTS.items():
        published = _recorded_config(name)
        for field in (
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "patch_size",
            "joint_attention_dim",
            "caption_projection_dim",
            "pooled_projection_dim",
            "pos_embed_max_size",
            "qk_norm",
        ):
            assert overrides[field] == published[field], (name, field)
        # Absent from the published config means no dual attention, which is a real
        # value and not a missing one — the two large releases genuinely have none.
        assert overrides["dual_attention_layers"] == tuple(
            published.get("dual_attention_layers", [])
        ), name


def test_large_turbo_is_the_same_transformer_as_large():
    """One shape, two trainings — which is why they share an overrides table."""
    assert _recorded_config("large-turbo") == _recorded_config("large")
    assert (
        sd35_config.sd35_large_turbo_model_config().transformer_overrides
        is sd35_config.LARGE_TRANSFORMER_OVERRIDES
    )


def test_dual_attention_is_a_medium_only_feature():
    """The plan assumed `(0..12)` everywhere. The repositories say otherwise.

    Medium declares thirteen MMDiT-X blocks; Large and Large Turbo declare none, and
    their checkpoints contain no `attn2.*` tensor at all. Asserted against the tensors
    rather than only the config, because the config could be a stale field.
    """
    assert sd35_config.MEDIUM_TRANSFORMER_OVERRIDES["dual_attention_layers"] == tuple(range(13))
    assert sd35_config.LARGE_TRANSFORMER_OVERRIDES["dual_attention_layers"] == ()

    medium = INDEX["tensors"]["medium"]["transformer"]
    large = INDEX["tensors"]["large"]["transformer"]
    medium_dual = {key.split(".")[1] for key in medium if ".attn2." in key}
    assert medium_dual == {str(index) for index in range(13)}
    assert not [key for key in large if ".attn2." in key]


def _expected_shape(name: str, recorded: list[int]) -> list[int]:
    """The recorded torch shape, in the layout MLX stores the same parameter in.

    One parameter differs, and only in layout: MLX convolutions are NHWC, so
    `pos_embed.proj.weight` is `[out, kh, kw, in]` where torch writes
    `[out, in, kh, kw]`. `weights.py` applies exactly this permutation on load, via
    mflux's own `transpose_conv2d`.
    """
    if name == "pos_embed.proj.weight":
        out_channels, in_channels, kh, kw = recorded
        return [out_channels, kh, kw, in_channels]
    return recorded


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_parameter_tree_is_the_checkpoints_tensor_set(variant):
    """Name for name and shape for shape, against the real safetensors headers."""
    overrides, recorded_key = VARIANTS[variant]
    transformer = SD35Transformer(**overrides)

    built = {name: list(param.shape) for name, param in tree_flatten(transformer.parameters())}
    recorded = INDEX["tensors"][recorded_key]["transformer"]

    missing = sorted(set(recorded) - set(built))
    extra = sorted(set(built) - set(recorded))
    assert not missing, f"{variant}: module has no home for {len(missing)} tensors, e.g. {missing[:5]}"
    assert not extra, f"{variant}: module has {len(extra)} parameters no tensor fills, e.g. {extra[:5]}"

    mismatched = {
        name: (built[name], recorded[name])
        for name in recorded
        if built[name] != _expected_shape(name, recorded[name])
    }
    assert not mismatched, (
        f"{variant}: {len(mismatched)} shape mismatches, e.g. {list(mismatched.items())[:3]}"
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_final_block_drops_its_text_stream(variant):
    """`context_pre_only` on the last block only, asserted structurally.

    A block that kept its text output would carry `attn.to_add_out` and `ff_context`,
    and its `norm1_context` would be six modulations wide rather than two. The
    checkpoint has neither, so building them would leave three randomly-initialised
    tensors in the model with nothing to fill them.
    """
    overrides, _ = VARIANTS[variant]
    transformer = SD35Transformer(**overrides)
    blocks = transformer.transformer_blocks
    inner = overrides["caption_projection_dim"]

    for block in blocks[:-1]:
        assert block.context_pre_only is False
        assert hasattr(block, "ff_context")
        assert hasattr(block.attn, "to_add_out")
        assert block.norm1_context.linear.weight.shape == (6 * inner, inner)

    last = blocks[-1]
    assert last.context_pre_only is True
    assert not hasattr(last, "ff_context")
    assert not hasattr(last.attn, "to_add_out")
    assert last.norm1_context.linear.weight.shape == (2 * inner, inner)


def test_a_dual_block_modulates_nine_ways_and_a_joint_block_six():
    """The structural difference MMDiT-X actually is, at the width that proves it."""
    overrides = sd35_config.MEDIUM_TRANSFORMER_OVERRIDES
    inner = overrides["caption_projection_dim"]
    transformer = SD35Transformer(**overrides)

    dual = transformer.transformer_blocks[0]
    joint = transformer.transformer_blocks[13]

    assert dual.use_dual_attention is True
    assert dual.norm1.linear.weight.shape == (9 * inner, inner)
    # A second attention over the image stream alone: no `add_*_proj`, no text output.
    assert dual.attn2.added is False
    assert not hasattr(dual.attn2, "add_q_proj")
    assert not hasattr(dual.attn2, "to_add_out")

    assert joint.use_dual_attention is False
    assert joint.norm1.linear.weight.shape == (6 * inner, inner)
    assert not hasattr(joint, "attn2")


def test_a_forward_pass_returns_a_velocity_of_the_latents_own_shape():
    """Shape only — no quality claim. A miniature of the real configuration."""
    transformer = SD35Transformer(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=64,
        caption_projection_dim=128,
        joint_attention_dim=64,
        pooled_projection_dim=32,
        pos_embed_max_size=16,
        dual_attention_layers=(0,),
    )
    latents = mx.random.normal(shape=(1, 16, 16, 24), key=mx.random.key(0))
    out = transformer(
        hidden_states=latents,
        timestep=mx.array([500.0]),
        encoder_hidden_states=mx.random.normal(shape=(1, 20, 64), key=mx.random.key(1)),
        pooled_projections=mx.random.normal(shape=(1, 32), key=mx.random.key(2)),
    )
    mx.eval(out)
    assert out.shape == latents.shape
    assert bool(mx.all(mx.isfinite(out)))


def test_a_resolution_larger_than_the_positional_table_is_refused():
    """Fail closed rather than crop garbage: the table bounds the resolution."""
    transformer = SD35Transformer(
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=64,
        caption_projection_dim=128,
        joint_attention_dim=64,
        pooled_projection_dim=32,
        pos_embed_max_size=8,
        dual_attention_layers=(),
    )
    with pytest.raises(ValueError, match="positional window"):
        transformer.pos_embed.cropped_pos_embed(64, 64)


def test_the_positional_window_is_centred():
    """Centre-cropped, so two aspect ratios share a frame instead of drifting apart."""
    transformer = SD35Transformer(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=64,
        caption_projection_dim=64,
        joint_attention_dim=64,
        pooled_projection_dim=32,
        pos_embed_max_size=8,
        dual_attention_layers=(),
    )
    # A distinguishable table: row-major position index, broadcast across the width.
    table = mx.arange(64, dtype=mx.float32).reshape(1, 64, 1) * mx.ones((1, 1, 64))
    transformer.pos_embed.pos_embed = table

    window = transformer.pos_embed.cropped_pos_embed(8, 8)  # 4x4 of an 8x8 table
    assert window.shape == (1, 16, 64)
    # top = left = (8 - 4) // 2 = 2, so the first row is positions 2*8+2 .. 2*8+5.
    assert [int(v) for v in window[0, :4, 0].tolist()] == [18, 19, 20, 21]


def test_timestep_features_lead_with_cosine():
    """`flip_sin_to_cos=True`. Getting the halves the wrong way round is silent."""
    features = timestep_features(mx.array([0.0]), dim=8)
    # At t = 0 every frequency is 0, so cos is 1 and sin is 0.
    assert [round(float(v), 6) for v in features[0].tolist()] == [1, 1, 1, 1, 0, 0, 0, 0]


def test_an_unsupported_qk_norm_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="qk_norm"):
        SD35Transformer(qk_norm="layer_norm")


def test_the_text_stream_must_be_carried_at_the_image_streams_width():
    with pytest.raises(ValueError, match="caption_projection_dim"):
        SD35Transformer(num_attention_heads=24, attention_head_dim=64, caption_projection_dim=2432)
