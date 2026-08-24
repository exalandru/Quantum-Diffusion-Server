"""SD 3.5's text encoders and the conditioning they assemble.

Structure is checked against the real checkpoints in `test_sd35_weights.py`; what is
checked here is *behaviour* — the four decisions inside the encoders that no shape
assertion can see, and each of which produces a plausible image that is not the
model's:

* CLIP attention is **causal**;
* the sequence conditioning is the **penultimate** hidden states, before
  `final_layer_norm`, not the final ones;
* pooling reads the **end-of-text position**, not the last position of the padded
  window;
* the two towers use **different activations** (`quick_gelu` for L, exact `gelu` for G).

Plus the conditioner's zero pad, which is what lets a 2048-wide CLIP pair share a
4096-wide sequence with T5.

The towers built here are miniatures at the real *configuration* — same class, same
code path — because the real ones are checked for shape elsewhere and running a 32-layer
1280-wide encoder to observe a causal mask buys nothing.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from qds.sd35 import conditioner
from qds.sd35 import config as sd35_config
from qds.sd35.clip import SD35ClipG, SD35ClipL, SD35ClipTower


def _tiny(**overrides) -> SD35ClipTower:
    kwargs = {
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "projection_dim": 16,
        "vocab_size": 64,
        "max_position_embeddings": 12,
    }
    kwargs.update(overrides)
    return SD35ClipTower(**kwargs)


def test_the_published_configurations_are_what_the_two_towers_default_to():
    """`SD35ClipL`/`SD35ClipG` are the two configs, not two hand-tuned guesses."""
    left = SD35ClipL()
    right = SD35ClipG()

    assert left.num_hidden_layers == sd35_config.CLIP_L_OVERRIDES["num_hidden_layers"]
    assert len(left.layers) == 12
    assert left.text_projection.weight.shape == (768, 768)
    assert left.text_model.encoder.layers[0].mlp.fc1.weight.shape == (3072, 768)

    assert right.num_hidden_layers == sd35_config.CLIP_G_OVERRIDES["num_hidden_layers"]
    assert len(right.layers) == 32
    assert right.text_projection.weight.shape == (1280, 1280)
    assert right.text_model.encoder.layers[0].mlp.fc1.weight.shape == (5120, 1280)

    # The overrides table and the class defaults are the same numbers; a row builds
    # CLIP-L through `text_encoder_overrides`, so they must not drift apart.
    explicit = SD35ClipL(**sd35_config.CLIP_L_OVERRIDES)
    assert explicit.text_projection.weight.shape == left.text_projection.weight.shape


def test_the_two_towers_use_different_activations():
    """`quick_gelu` and `gelu` differ by a few percent — enough to matter, not to crash."""
    x = mx.array([[[1.0, -1.0, 2.0, 0.5]]])
    quick = _tiny(hidden_size=4, num_attention_heads=1, intermediate_size=4, hidden_act="quick_gelu")
    exact = _tiny(hidden_size=4, num_attention_heads=1, intermediate_size=4, hidden_act="gelu")

    quick_act = quick.text_model.encoder.layers[0].mlp._activation
    exact_act = exact.text_model.encoder.layers[0].mlp._activation
    assert not bool(mx.allclose(quick_act(x), exact_act(x)))
    # quick_gelu is x * sigmoid(1.702x) exactly.
    assert bool(mx.allclose(quick_act(x), x * mx.sigmoid(1.702 * x)))

    assert sd35_config.CLIP_L_OVERRIDES["hidden_act"] == "quick_gelu"
    assert sd35_config.CLIP_G_OVERRIDES["hidden_act"] == "gelu"


def test_an_unknown_activation_is_refused():
    with pytest.raises(ValueError, match="quick_gelu"):
        _tiny(hidden_act="silu")


def test_the_attention_is_causal():
    """A token cannot see its successors. The counterfactual is what proves it.

    Change the *last* token and every earlier position must be untouched; change the
    *first* and later positions must move. A bidirectional implementation fails the
    first half, which is exactly the bug this guards against.
    """
    tower = _tiny()
    base = mx.array([[3, 4, 5, 6, 7, 63]])
    later = mx.array([[3, 4, 5, 6, 9, 63]])
    earlier = mx.array([[8, 4, 5, 6, 7, 63]])

    states, _ = tower(base)
    changed_later, _ = tower(later)
    changed_earlier, _ = tower(earlier)
    mx.eval(states, changed_later, changed_earlier)

    # Positions 0..3 precede the token that moved, so they must be identical.
    assert bool(mx.allclose(states[:, :4], changed_later[:, :4], atol=1e-6))
    # And a change at position 0 must reach position 4.
    assert not bool(mx.allclose(states[:, 4], changed_earlier[:, 4], atol=1e-6))


def test_the_sequence_conditioning_is_the_penultimate_layers_output():
    """`hidden_states[-2]`: all but the last layer, and before `final_layer_norm`."""
    tower = _tiny(num_hidden_layers=4)
    ids = mx.array([[3, 4, 5, 63]])
    captured, _ = tower(ids)

    hidden = tower.text_model.embeddings(ids)
    for layer in tower.text_model.encoder.layers[:3]:
        hidden = layer(hidden)
    mx.eval(captured, hidden)
    assert bool(mx.allclose(captured, hidden, atol=1e-6))

    # And it is emphatically not the final state: one more layer plus the final norm.
    final = tower.text_model.final_layer_norm(tower.text_model.encoder.layers[3](hidden))
    assert not bool(mx.allclose(captured, final, atol=1e-4))


def test_the_index_is_bounds_checked_against_the_towers_depth():
    with pytest.raises(ValueError, match="hidden_state_index"):
        _tiny(num_hidden_layers=4, hidden_state_index=-5)


def test_pooling_reads_the_end_of_text_position_not_the_last_padded_one():
    """The end-of-text token is the highest id, so `argmax` finds it — first occurrence.

    Padding a 77-position window and pooling at position 76 would read padding for
    every prompt shorter than the window, which is every prompt.
    """
    tower = _tiny(vocab_size=64)
    # id 63 is the highest, standing in for `<|endoftext|>`; 9 is padding after it.
    ids = mx.array([[3, 4, 63, 9, 9, 9]])
    _, pooled = tower(ids)

    hidden = tower.text_model.embeddings(ids)
    for layer in tower.text_model.encoder.layers:
        hidden = layer(hidden)
    last_hidden_state = tower.text_model.final_layer_norm(hidden)

    at_eos = tower.text_projection(last_hidden_state[:, 2])
    at_end = tower.text_projection(last_hidden_state[:, 5])
    mx.eval(pooled, at_eos, at_end)

    assert bool(mx.allclose(pooled, at_eos, atol=1e-6))
    assert not bool(mx.allclose(pooled, at_end, atol=1e-5))


def test_the_towers_return_the_widths_the_transformer_expects():
    """768 + 1280 = 2048 = `pooled_projection_dim`, and that is not a coincidence."""
    small = sd35_config.CLIP_L_OVERRIDES["projection_dim"]
    big = sd35_config.CLIP_G_OVERRIDES["projection_dim"]
    assert small + big == sd35_config.MEDIUM_TRANSFORMER_OVERRIDES["pooled_projection_dim"]
    assert small + big == sd35_config.LARGE_TRANSFORMER_OVERRIDES["pooled_projection_dim"]


def test_the_t5_tower_mflux_supplies_is_sd35s_own_shape():
    """Reuse is legal because the hard-coded model *is* this one. Asserted, not assumed."""
    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder

    published = sd35_config.T5_MAX_SEQUENCE_LENGTH
    assert published == 256

    encoder = T5Encoder()
    assert len(encoder.t5_blocks) == 24
    assert encoder.shared.weight.shape == (32128, 4096)
    assert encoder.final_layer_norm.weight.shape == (4096,)
    block = encoder.t5_blocks[0]
    assert block.attention.SelfAttention.q.weight.shape == (4096, 4096)
    assert block.ff.DenseReluDense.wi_0.weight.shape == (10240, 4096)
    # 4096 is what the transformer's `context_embedder` reads.
    assert sd35_config.JOINT_ATTENTION_DIM == 4096


def test_the_joint_context_pads_the_clip_pair_up_to_the_t5_width():
    """The pad is structural: `context_embedder` is one 4096-wide projection for both."""
    clip_l = mx.ones((1, 77, 768))
    clip_g = mx.full((1, 77, 1280), 2.0)
    t5 = mx.full((1, 256, 4096), 3.0)

    context = conditioner.joint_context(clip_l, clip_g, t5)
    mx.eval(context)
    assert context.shape == (1, 77 + 256, 4096)

    # CLIP first along the sequence axis, in feature order L then G, zero-padded after.
    assert float(context[0, 0, 0]) == 1.0
    assert float(context[0, 0, 768]) == 2.0
    assert float(context[0, 0, 2048]) == 0.0
    assert float(context[0, 0, 4095]) == 0.0
    # Then T5, unpadded.
    assert float(context[0, 77, 0]) == 3.0
    assert float(context[0, 77, 4095]) == 3.0


def test_a_clip_pair_wider_than_the_joint_stream_is_refused():
    with pytest.raises(ValueError, match="cannot be narrower"):
        conditioner.joint_context(mx.ones((1, 4, 768)), mx.ones((1, 4, 1280)), mx.ones((1, 4, 1024)))


def test_the_pooled_projection_is_the_two_towers_concatenated():
    pooled = conditioner.pooled_projections(mx.ones((1, 768)), mx.full((1, 1280), 2.0))
    mx.eval(pooled)
    assert pooled.shape == (1, 2048)
    assert float(pooled[0, 767]) == 1.0
    assert float(pooled[0, 768]) == 2.0


def test_the_quantization_unit_alias_is_the_same_list_the_weights_live_in():
    """`layers` is a second name for `text_model.encoder.layers`, not a second list."""
    tower = _tiny()
    assert tower.layers is tower.text_model.encoder.layers
    # And it is what `prequantize` walks to bound the conversion's memory peak.
    from qds.prequantize import _quantization_units

    assert _quantization_units(tower) == list(tower.text_model.encoder.layers)
