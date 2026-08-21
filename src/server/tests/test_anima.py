"""Anima's MLX port, checked against the implementation it was ported from.

QDS carries its own MLX implementation of Anima's DiT and text adapter because
mflux has neither. A port is worth what its evidence is worth, so these tests do
not check that the modules run: they load the *same weights* into this port and
into diffusers' reference classes and compare the outputs.

They import diffusers and torch, which the server never does. That is deliberate
and it is why diffusers is a dev dependency: the reference belongs in the tests,
not in the runtime.

Two more things are checked here that numbers alone would not catch:

* that the checkpoint-name translation in `weights.py` is *total* -- every name
  the real published checkpoint uses maps onto a parameter this port actually
  has, and every parameter is fed by exactly one name. The names come from
  `fixtures/anima_checkpoint_index.json`, recorded from the file itself, because
  a witness that rebuilds its input from the mapping under test cannot fail for
  a checkpoint whose names disagree with it -- which is the whole risk;
* that a checkpoint the translation does not recognise is refused rather than
  half-loaded, since weights are applied non-strictly and the untranslated case
  would otherwise generate noise in silence;
* that the sigma schedule matches the one Anima's scheduler config describes,
  which is a static shift of 3.0 and not the bare linspace the loop starts from.
"""

from __future__ import annotations

import json
import math
import pathlib

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from qds.anima import config as anima_config
from qds.anima.conditioner import AnimaTextConditioner
from qds.anima.transformer import AnimaTransformer
from qds.anima.weights import conditioner_key, transformer_key

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

#: Small enough to build in a test, same shape everywhere it matters: the patch
#: size, the RoPE scale and the adaLN structure are Anima's.
SMALL_DIT = dict(
    in_channels=16,
    out_channels=16,
    num_layers=2,
    num_attention_heads=4,
    attention_head_dim=24,
    text_embed_dim=32,
    adaln_lora_dim=16,
    mlp_ratio=4.0,
    patch_size=(1, 2, 2),
    rope_scale=(1.0, 4.0, 4.0),
    max_size=(128, 240, 240),
    concat_padding_mask=True,
)

SMALL_CONDITIONER = dict(
    source_dim=32,
    target_dim=32,
    model_dim=32,
    num_layers=2,
    num_attention_heads=4,
    mlp_ratio=4.0,
    target_vocab_size=64,
    min_sequence_length=16,
    use_self_attention=True,
    use_layer_norm=False,
)


def _load_reference_weights(mine, reference) -> dict[str, np.ndarray]:
    """Put the reference's weights into the port, and fail if the shapes disagree."""
    state = {k: v.detach().float().numpy() for k, v in reference.state_dict().items()}
    params = dict(tree_flatten(mine.parameters()))

    assert set(params) == set(state), (
        f"only in port: {sorted(set(params) - set(state))}; "
        f"only in reference: {sorted(set(state) - set(params))}"
    )
    mismatched = {
        k: (tuple(params[k].shape), tuple(state[k].shape))
        for k in params
        if tuple(params[k].shape) != tuple(state[k].shape)
    }
    assert not mismatched, mismatched

    mine.update(tree_unflatten([(k, mx.array(state[k])) for k in params]))
    mine.eval()
    return state


def _reference_transformer():
    from diffusers import CosmosTransformer3DModel

    torch.manual_seed(0)
    return CosmosTransformer3DModel(
        **SMALL_DIT,
        extra_pos_embed_type=None,
        use_crossattn_projection=False,
        img_context_dim_in=None,
        controlnet_block_every_n=None,
    ).eval()


def test_the_transformer_reproduces_the_reference():
    """The whole DiT, one forward, against `CosmosTransformer3DModel`."""
    reference = _reference_transformer()
    mine = AnimaTransformer(**SMALL_DIT)
    _load_reference_weights(mine, reference)

    batch, frames, height, width, text_len = 1, 1, 8, 12, 7
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((batch, 16, frames, height, width)).astype(np.float32)
    context = rng.standard_normal((batch, text_len, SMALL_DIT["text_embed_dim"])).astype(np.float32)
    timestep = np.array([0.7], dtype=np.float32)

    with torch.no_grad():
        expected = reference(
            hidden_states=torch.from_numpy(latents),
            timestep=torch.from_numpy(timestep),
            encoder_hidden_states=torch.from_numpy(context),
            padding_mask=torch.zeros(batch, 1, height, width),
            return_dict=False,
        )[0].numpy()

    actual = np.array(
        mine(
            mx.array(latents),
            mx.array(timestep),
            mx.array(context),
            padding_mask=mx.zeros((batch, 1, frames, height, width)),
        )
    )

    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 1e-4


def test_the_transformer_stays_faithful_across_timesteps():
    """One sample could agree by luck; the trajectory is what the loop walks."""
    reference = _reference_transformer()
    mine = AnimaTransformer(**SMALL_DIT)
    _load_reference_weights(mine, reference)

    rng = np.random.default_rng(3)
    latents = rng.standard_normal((1, 16, 1, 8, 8)).astype(np.float32)
    context = rng.standard_normal((1, 5, SMALL_DIT["text_embed_dim"])).astype(np.float32)

    for sigma in (1.0, 0.75, 0.3, 0.02):
        timestep = np.array([sigma], dtype=np.float32)
        with torch.no_grad():
            expected = reference(
                hidden_states=torch.from_numpy(latents),
                timestep=torch.from_numpy(timestep),
                encoder_hidden_states=torch.from_numpy(context),
                padding_mask=torch.zeros(1, 1, 8, 8),
                return_dict=False,
            )[0].numpy()
        actual = np.array(
            mine(mx.array(latents), mx.array(timestep), mx.array(context),
                 padding_mask=mx.zeros((1, 1, 1, 8, 8)))
        )
        assert np.abs(actual - expected).max() < 1e-4, f"diverges at sigma={sigma}"


@pytest.mark.parametrize("with_masks", [False, True])
def test_the_text_conditioner_reproduces_the_reference(with_masks):
    """Both branches: unmasked, and with padding on each side of the cross-attention."""
    from diffusers import AnimaTextConditioner as ReferenceConditioner

    torch.manual_seed(0)
    reference = ReferenceConditioner(**SMALL_CONDITIONER).eval()
    mine = AnimaTextConditioner(**SMALL_CONDITIONER)
    _load_reference_weights(mine, reference)

    rng = np.random.default_rng(1)
    source = rng.standard_normal((2, 9, SMALL_CONDITIONER["source_dim"])).astype(np.float32)
    ids = rng.integers(0, SMALL_CONDITIONER["target_vocab_size"], size=(2, 5)).astype(np.int64)
    target_mask = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]) if with_masks else None
    source_mask = np.array([[1] * 9, [1] * 6 + [0] * 3]) if with_masks else None

    with torch.no_grad():
        expected = reference(
            source_hidden_states=torch.from_numpy(source),
            target_input_ids=torch.from_numpy(ids),
            target_attention_mask=None if target_mask is None else torch.from_numpy(target_mask),
            source_attention_mask=None if source_mask is None else torch.from_numpy(source_mask),
        ).numpy()

    actual = np.array(
        mine(
            mx.array(source),
            mx.array(ids),
            None if target_mask is None else mx.array(target_mask),
            None if source_mask is None else mx.array(source_mask),
        )
    )

    assert actual.shape == expected.shape
    # Padded out to `min_sequence_length`, which the DiT cross-attends over whole.
    assert actual.shape[1] == SMALL_CONDITIONER["min_sequence_length"]
    assert np.abs(actual - expected).max() < 1e-4


def test_the_conditioner_refuses_shapes_anima_does_not_publish():
    """A silently-built variant would load weights that do not fit it."""
    with pytest.raises(ValueError, match="published shape"):
        AnimaTextConditioner(**{**SMALL_CONDITIONER, "use_layer_norm": True})
    with pytest.raises(ValueError, match="published shape"):
        AnimaTextConditioner(**{**SMALL_CONDITIONER, "use_self_attention": False})


#: Recorded from the tensor index of the real published checkpoint --
#: `circlestone-labs/Anima`, `anima-aesthetic-v1.1.safetensors` -- with block
#: indices collapsed to `{i}`. It is a fixture rather than a computation on
#: purpose: a witness that rebuilds its input from the mapping under test cannot
#: fail for the one thing it exists to catch, which is the mapping disagreeing
#: with a real file.
CHECKPOINT_INDEX = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "anima_checkpoint_index.json").read_text()
)


def _published_checkpoint_names(num_layers: int) -> list[str]:
    """Expand the recorded stems back to full names for `num_layers` blocks."""
    names = []
    for stem in CHECKPOINT_INDEX["distinct_stems"]:
        if "blocks.{i}." in stem:
            names.extend(stem.replace("{i}", str(index)) for index in range(num_layers))
        else:
            names.append(stem)
    return names


def test_the_recorded_index_is_the_checkpoint_the_catalogue_loads():
    """The fixture describes the file `config.DEFAULT_WEIGHT_FILE` names, and no other."""
    assert CHECKPOINT_INDEX["source_repo"] == anima_config.REPO
    assert CHECKPOINT_INDEX["source_file"].endswith(anima_config.DEFAULT_WEIGHT_FILE)
    # 685 = 567 transformer + 118 adapter, expanded at the real block counts.
    layers = anima_config.TRANSFORMER_OVERRIDES["num_layers"]
    conditioner_layers = anima_config.CONDITIONER_OVERRIDES["num_layers"]
    expanded = _published_checkpoint_names(max(layers, conditioner_layers))
    assert CHECKPOINT_INDEX["total_tensors"] == 685
    assert len(expanded) >= CHECKPOINT_INDEX["total_tensors"]


def test_every_published_name_reaches_the_component_it_belongs_to():
    """The translation is total against the real file's names, in both directions.

    A name that fails to map leaves a tensor on the floor; a parameter no name
    reaches keeps its random initialisation. Neither raises, and both generate
    noise, so neither may be left to inspection.
    """
    transformer = AnimaTransformer(**SMALL_DIT)
    conditioner = AnimaTextConditioner(**SMALL_CONDITIONER)
    parameters = {
        "transformer": set(dict(tree_flatten(transformer.parameters()))),
        "text_conditioner": set(dict(tree_flatten(conditioner.parameters()))),
    }

    # Expanded at the *small* block counts, since that is what the modules above
    # were built with; the mapping is index-agnostic, so this exercises the same
    # translation the 28-block file goes through.
    translated: dict[str, dict[str, str]] = {"transformer": {}, "text_conditioner": {}}
    for name in _published_checkpoint_names(SMALL_DIT["num_layers"]):
        target, component = transformer_key(name), "transformer"
        if target is None:
            target, component = conditioner_key(name), "text_conditioner"
        assert target is not None, f"published tensor reaches neither component: {name}"
        assert target not in translated[component], f"two names map onto {component}.{target}"
        translated[component][target] = name

    for component, names in translated.items():
        assert set(names) == parameters[component], (
            f"{component}: parameters nothing feeds "
            f"{sorted(parameters[component] - set(names))}; "
            f"names with no parameter {sorted(set(names) - parameters[component])}"
        )


def test_a_checkpoint_with_foreign_names_is_refused_rather_than_half_loaded():
    """The failure this guard exists for, and it is a reachable one.

    Four of the eight checkpoints published alongside the default -- Base v1.0 and
    the three previews -- root their tensors at `net.` instead of
    `model.diffusion_model.`, so none of their names translate. Weights are
    applied non-strictly, so without a guard the module would keep its random
    initialisation and the server would generate noise silently.
    """
    from qds.anima.weights import verify_loaded

    foreign = "net.blocks.0.self_attn.q_proj.weight"
    assert transformer_key(foreign) is None
    assert conditioner_key(foreign) is None

    transformer = AnimaTransformer(**SMALL_DIT)
    with pytest.raises(ValueError, match="received 0 tensors"):
        verify_loaded({"transformer": {}}, {"transformer": transformer})

    # And a partially-translated component is refused too, not only an empty one.
    partial = dict(list(tree_flatten(transformer.parameters()))[:5])
    with pytest.raises(ValueError, match="DEFAULT_WEIGHT_FILE"):
        verify_loaded({"transformer": partial}, {"transformer": transformer})


def test_a_fully_loaded_component_passes_the_guard():
    """The guard must not fire on the good case -- otherwise it proves nothing."""
    from mlx.utils import tree_unflatten

    from qds.anima.weights import verify_loaded

    transformer = AnimaTransformer(**SMALL_DIT)
    complete = tree_unflatten(list(tree_flatten(transformer.parameters())))
    verify_loaded({"transformer": complete}, {"transformer": transformer})


def test_the_adapter_is_separated_from_the_transformer_by_prefix_alone():
    """One file, two components. Each key belongs to exactly one of them."""
    dit = anima_config.CHECKPOINT_PREFIX + "blocks.0.self_attn.q_proj.weight"
    adapter = anima_config.ADAPTER_PREFIX + "blocks.0.self_attn.q_proj.weight"

    assert transformer_key(dit) == "transformer_blocks.0.attn1.to_q.weight"
    assert conditioner_key(dit) is None

    assert transformer_key(adapter) is None
    assert conditioner_key(adapter) == "blocks.0.self_attn.q_proj.weight"


def test_the_adapter_names_reach_the_conditioner_unchanged():
    """After the prefix, the checkpoint and the reference already agree."""
    mine = AnimaTextConditioner(**SMALL_CONDITIONER)
    parameters = set(dict(tree_flatten(mine.parameters())))
    for name in parameters:
        assert conditioner_key(anima_config.ADAPTER_PREFIX + name) == name


def test_the_text_encoder_prefix_is_stripped_and_nothing_else_is():
    from qds.anima.weights import text_encoder_key

    assert text_encoder_key("model.layers.0.self_attn.q_proj.weight") == (
        "layers.0.self_attn.q_proj.weight"
    )
    assert text_encoder_key("model.embed_tokens.weight") == "embed_tokens.weight"
    # The published file carries no `lm_head`; anything outside the tower is dropped
    # rather than loaded into a module that has no place for it.
    assert text_encoder_key("lm_head.weight") is None


def test_the_sigma_shift_is_the_one_the_scheduler_config_describes():
    """Anima shifts statically by 3.0; mflux shifts exponentially by `mu`.

    They are the same function -- `exp(mu)` is the static factor -- so the port
    uses mflux's machinery at `mu = ln(3)` rather than carrying a second
    implementation. If that identity ever stopped holding, the images would drift
    subtly rather than fail, so it is asserted rather than commented.
    """
    from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
        FlowMatchEulerDiscreteScheduler,
    )

    steps = 8
    linear = mx.linspace(1.0, 1.0 / steps, steps, dtype=mx.float32)
    through_mflux = np.array(
        FlowMatchEulerDiscreteScheduler._time_shift_exponential_array(
            math.log(anima_config.SIGMA_SHIFT), 1.0, linear
        )
    )

    raw = np.linspace(1.0, 1.0 / steps, steps)
    shift = anima_config.SIGMA_SHIFT
    static = shift * raw / (1 + (shift - 1) * raw)

    assert np.abs(through_mflux - static).max() < 1e-6
    # And it is a real change: the bare linspace is not the schedule.
    assert np.abs(through_mflux - raw).max() > 0.05


def test_the_configuration_matches_the_published_component_configs():
    """Pinned against `config.json` in `circlestone-labs/Anima-Base-v1.0-Diffusers`."""
    transformer = anima_config.TRANSFORMER_OVERRIDES
    assert transformer["num_attention_heads"] * transformer["attention_head_dim"] == 2048
    assert transformer["patch_size"] == (1, 2, 2)
    assert transformer["rope_scale"] == (1.0, 4.0, 4.0)
    assert transformer["adaln_lora_dim"] == 256
    assert transformer["text_embed_dim"] == 1024
    assert transformer["concat_padding_mask"] is True

    conditioner = anima_config.CONDITIONER_OVERRIDES
    assert conditioner["target_vocab_size"] == 32128
    assert conditioner["num_layers"] == 6
    assert conditioner["min_sequence_length"] == 512
    assert conditioner["model_dim"] // conditioner["num_attention_heads"] == 64

    encoder = anima_config.TEXT_ENCODER_OVERRIDES
    assert encoder["hidden_size"] == 1024
    assert encoder["num_hidden_layers"] == 28
    assert (encoder["num_attention_heads"], encoder["num_key_value_heads"]) == (16, 8)
    assert encoder["head_dim"] == 128


def test_the_patch_embedding_width_follows_from_the_padding_mask_channel():
    """68 columns is 17 channels over a 2x2 patch, and 17 is 16 latents plus the mask."""
    transformer = anima_config.TRANSFORMER_OVERRIDES
    p_t, p_h, p_w = transformer["patch_size"]
    channels = transformer["in_channels"] + (1 if transformer["concat_padding_mask"] else 0)
    assert channels * p_t * p_h * p_w == 68


def test_the_default_checkpoint_is_one_the_translation_supports():
    """The menu in `config.py` is not uniform, and the guard is not the only line.

    Four of the eight published checkpoints use a different tensor root and
    translate to nothing. `verify_loaded` catches that at load time; this catches
    it at edit time, which is cheaper than a 4 GB download.
    """
    assert anima_config.DEFAULT_WEIGHT_FILE in anima_config.SUPPORTED_WEIGHT_FILES


def test_the_two_catalogue_rows_select_different_checkpoints():
    """Aesthetic and Turbo are one architecture and two trainings.

    Nothing about the port differs between them -- same 28 blocks, same adapter,
    same shift -- so the variant is a file, and this is the table that says which.
    Getting it wrong would load the other model in silence, since both translate.
    """
    from qds.anima.weights import anima_weight_definition

    assert anima_config.weight_file_for("anima") == anima_config.AESTHETIC_WEIGHT_FILE
    assert anima_config.weight_file_for("anima_turbo") == anima_config.TURBO_WEIGHT_FILE
    assert anima_config.AESTHETIC_WEIGHT_FILE != anima_config.TURBO_WEIGHT_FILE

    for file in (anima_config.AESTHETIC_WEIGHT_FILE, anima_config.TURBO_WEIGHT_FILE):
        assert file in anima_config.SUPPORTED_WEIGHT_FILES
        definition = anima_weight_definition(file)
        # The one download this variant needs, and not its seven siblings.
        patterns = definition.get_download_patterns()
        assert f"split_files/diffusion_models/{file}" in patterns
        assert not any(p.endswith("/*.safetensors") for p in patterns), patterns
        # Both halves of the shared file follow the variant, or the adapter would
        # be read from a checkpoint the transformer is not.
        for component in definition.get_components():
            if component.name in ("transformer", "text_conditioner"):
                assert component.weight_files == [file], component.name


def test_an_unregistered_row_or_checkpoint_is_refused_rather_than_defaulted():
    """Falling back to the default file would load the wrong model, quietly."""
    from qds.anima.weights import anima_weight_definition

    with pytest.raises(ValueError, match="No Anima checkpoint is registered"):
        anima_config.weight_file_for("anima_something_else")

    # Base v1.0 is a real published file, and one this package cannot translate.
    with pytest.raises(ValueError, match="not a checkpoint this package translates"):
        anima_weight_definition("anima-base-v1.0.safetensors")
