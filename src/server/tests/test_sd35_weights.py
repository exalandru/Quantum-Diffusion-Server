"""Every tensor the repositories publish reaches a parameter, and every parameter is filled.

This is the test the whole port rests on. Three of SD 3.5's five components are loaded
in passthrough mode — no rename table at all — and mflux applies weights
*non-strictly*, so a name this package got wrong does not raise: the tensor is dropped,
the parameter keeps its random initialisation, and the server generates a plausible
image that is not the model's.

So the check here is exact and two-sided, per component, against
`fixtures/sd35_tensor_index.json` (the real safetensors headers of the three gated
repositories): run what `WeightLoader` would run — the component's `weight_transform`,
`key_transform` and `mapping_getter` — over the recorded tensor names, and assert the
result *is* the constructed module's parameter tree. Nothing unmapped, nothing
unfilled, and every shape agreeing.

No weights are downloaded. The recorded shapes are materialised as lazy zeros, which
MLX never evaluates, so an 8.1B-parameter component costs a graph and no memory.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from qds.sd35 import config as sd35_config
from qds.sd35.clip import SD35ClipG, SD35ClipL
from qds.sd35.transformer import SD35Transformer
from qds.sd35.vae import SD35VAE
from qds.sd35.weights import SD35WeightDefinition, verify_loaded

INDEX = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "sd35_tensor_index.json").read_text()
)


def _recorded(component: str, variant: str = "medium") -> dict[str, list[int]]:
    tensors = INDEX["tensors"]
    if component == "transformer":
        return tensors[variant]["transformer"]
    return tensors["shared"][component]


def _as_arrays(recorded: dict[str, list[int]]) -> dict[str, mx.array]:
    """The recorded tensor set, as lazy zeros of the right shape. Never evaluated."""
    return {name: mx.zeros(tuple(shape), dtype=mx.bfloat16) for name, shape in recorded.items()}


def _load_like_mflux(component, raw: dict[str, mx.array]) -> dict:
    """What `WeightLoader._load_component` does to a component's raw tensors.

    Only the steps this family uses: no prefix filters, no bulk transform, and
    precision conversion is a no-op because the sources are already bfloat16.
    """
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    from mlx.utils import tree_unflatten

    if component.key_transform is not None:
        raw = {
            component.key_transform(key): value
            for key, value in raw.items()
            if component.key_transform(key) is not None
        }
    if component.weight_transform is not None:
        raw = {key: component.weight_transform(key, value) for key, value in raw.items()}
    if component.mapping_getter is None:
        return tree_unflatten(list(raw.items()))
    return WeightMapper.apply_mapping(
        hf_weights=raw,
        mapping=component.mapping_getter(),
        num_blocks=component.num_blocks,
        num_layers=component.num_layers,
    )


def _module_for(name: str, variant: str):
    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder

    overrides = (
        sd35_config.MEDIUM_TRANSFORMER_OVERRIDES
        if variant == "medium"
        else sd35_config.LARGE_TRANSFORMER_OVERRIDES
    )
    return {
        "transformer": lambda: SD35Transformer(**overrides),
        "text_encoder": lambda: SD35ClipL(**sd35_config.CLIP_L_OVERRIDES),
        "text_encoder_2": SD35ClipG,
        "text_encoder_3": T5Encoder,
        "vae": SD35VAE,
    }[name]()


COMPONENTS = ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae")


@pytest.mark.parametrize("variant", ("medium", "large"))
@pytest.mark.parametrize("component_name", COMPONENTS)
def test_the_loaded_tree_is_exactly_the_modules_parameter_tree(component_name, variant):
    """Two-sided: no tensor is dropped, and no parameter is left unfilled."""
    component = {c.name: c for c in SD35WeightDefinition.get_components()}[component_name]

    recorded = _recorded(component_name, variant)
    loaded = _load_like_mflux(component, _as_arrays(recorded))
    module = _module_for(component_name, variant)

    got = {name: tuple(value.shape) for name, value in tree_flatten(loaded)}
    want = {name: tuple(value.shape) for name, value in tree_flatten(module.parameters())}

    unfilled = sorted(set(want) - set(got))
    unmapped = sorted(set(got) - set(want))
    assert not unfilled, (
        f"{variant}/{component_name}: {len(unfilled)} parameters no tensor reaches — "
        f"they would keep their random initialisation. e.g. {unfilled[:5]}"
    )
    assert not unmapped, (
        f"{variant}/{component_name}: {len(unmapped)} tensors land where no parameter "
        f"is — they would be silently discarded. e.g. {unmapped[:5]}"
    )
    mismatched = {name: (got[name], want[name]) for name in want if got[name] != want[name]}
    assert not mismatched, f"{variant}/{component_name}: {list(mismatched.items())[:3]}"


@pytest.mark.parametrize("component_name", COMPONENTS)
def test_every_published_tensor_is_consumed(component_name):
    """Counted at the source end too, so a mapping cannot quietly ignore a whole family.

    `WeightMapper` reports an unmapped source key by incrementing a counter nobody
    reads. The two-sided test above catches the consequence; this one names the cause.
    """
    component = {c.name: c for c in SD35WeightDefinition.get_components()}[component_name]
    recorded = _recorded(component_name)

    if component.mapping_getter is None:
        # Passthrough consumes every key by construction; the assertion that matters
        # is that they land somewhere real, which the test above makes.
        consumed = set(recorded)
    else:
        from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

        flat = WeightMapper._build_flat_mapping(
            component.mapping_getter(),
            num_blocks=component.num_blocks or 0,
            num_layers=component.num_layers or 0,
        )
        consumed = {key for key in recorded if key in flat}

    dropped = sorted(set(recorded) - consumed)
    assert not dropped, f"{component_name}: {len(dropped)} tensors unmapped, e.g. {dropped[:5]}"


def test_the_component_names_are_the_repository_subdirectories():
    """The converter's contract: component name == source subdir == artifact subdir."""
    for component in SD35WeightDefinition.get_components():
        assert component.hf_subdir == component.name
    assert {c.name for c in SD35WeightDefinition.get_components()} == set(COMPONENTS)


def test_one_definition_serves_all_three_releases():
    """The transformer is unpinned, so the directory decides how many shards there are.

    Medium publishes one 4.94 GB file and the large pair two shards; mflux's glob reads
    whichever is present. That is what lets `registry.family_structure`, which
    dispatches on the family alone, reach this definition at all.
    """
    components = {c.name: c for c in SD35WeightDefinition.get_components()}
    assert components["transformer"].weight_files is None
    assert components["vae"].weight_files is None
    # The text encoders are pinned, because those directories do hold duplicates.
    assert components["text_encoder"].weight_files == ["model.safetensors"]
    assert components["text_encoder_2"].weight_files == ["model.safetensors"]
    assert len(components["text_encoder_3"].weight_files) == 2

    # And the transformer directory really does hold one release's weights and nothing
    # else — otherwise globbing it would read the wrong shape.
    for variant in ("medium", "large"):
        recorded = _recorded("transformer", variant)
        assert recorded, variant


def test_the_download_patterns_name_files_rather_than_globbing_the_duplicates():
    """The fp16 copies are ~20 GB of the same tensors. Not fetching them is deliberate.

    Every text-encoder subdirectory in these repositories carries its weights twice.
    A `text_encoder_3/*.safetensors` pattern matches four shards where two are wanted,
    and mflux's `mlx_native` loader would then read all four.
    """
    patterns = SD35WeightDefinition.get_download_patterns()
    assert not [p for p in patterns if p.startswith("text_encoder") and p.endswith("*.safetensors")]
    assert "text_encoder_3/model-00001-of-00002.safetensors" in patterns
    assert "text_encoder_3/model-00002-of-00002.safetensors" in patterns
    assert "text_encoder_3/model.safetensors.index.json" in patterns
    for tokenizer in SD35WeightDefinition.get_tokenizers():
        assert f"{tokenizer.hf_subdir}/**" in patterns

    # The transformer and VAE are globbed, which is what makes one list serve three
    # releases — those two directories publish their weights exactly once.
    assert "transformer/*.safetensors" in patterns
    assert "transformer/*.json" in patterns
    assert "vae/*.safetensors" in patterns

    # And nothing reaches for the ComfyUI bundle or the root single-file checkpoints.
    assert not [p for p in patterns if p.startswith("text_encoders/")]
    assert not [p for p in patterns if p.startswith("sd3")]


def test_the_conv_transform_is_applied_to_the_patch_projection_and_nothing_else():
    """One tensor changes layout on load. Applying it twice, or not at all, is silent."""
    from qds.sd35.weights import transformer_weight_transform

    conv = mx.zeros((1536, 16, 2, 2))
    assert transformer_weight_transform("pos_embed.proj.weight", conv).shape == (1536, 2, 2, 16)

    linear = mx.zeros((1536, 4096))
    assert transformer_weight_transform("context_embedder.weight", linear).shape == (1536, 4096)
    # A four-dimensional tensor under any other name is left alone.
    assert transformer_weight_transform("proj_out.weight", conv).shape == (1536, 16, 2, 2)


def test_verify_loaded_refuses_a_component_that_did_not_fill_its_module():
    """Fail closed. Without this, an unrecognised checkpoint generates noise silently."""
    module = SD35ClipL(**sd35_config.CLIP_L_OVERRIDES)
    recorded = _recorded("text_encoder")
    component = {c.name: c for c in SD35WeightDefinition.get_components()}["text_encoder"]

    complete = _load_like_mflux(component, _as_arrays(recorded))
    verify_loaded({"text_encoder": complete}, {"text_encoder": module})

    missing_one = dict(recorded)
    missing_one.pop("text_projection.weight")
    partial = _load_like_mflux(component, _as_arrays(missing_one))
    with pytest.raises(ValueError, match="received 196 tensors but needs 197"):
        verify_loaded({"text_encoder": partial}, {"text_encoder": module})

    with pytest.raises(ValueError, match="received 0 tensors"):
        verify_loaded({}, {"text_encoder": module})


def test_the_quantization_predicate_skips_what_mlx_cannot_group():
    """A last axis that is not a multiple of 64 raises inside `nn.quantize`."""
    from mlx import nn

    predicate = SD35WeightDefinition.quantization_predicate
    assert predicate("context_embedder", nn.Linear(4096, 1536)) is True
    # `proj_out` maps 1536 -> 64; its input divides, so it quantizes.
    assert predicate("proj_out", nn.Linear(1536, 64)) is True
    assert predicate("odd", nn.Linear(100, 64)) is False
    assert predicate("norm", nn.RMSNorm(64)) is False
