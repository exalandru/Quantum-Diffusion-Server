"""FLUX.2-dev support: config, weight mappings, tokenizer, MLX encoder.

No real weights are loaded. The expected values come from the `config.json`
files of the `black-forest-labs/FLUX.2-dev` repo and from its weight indexes —
that is what makes these tests useful: they pin the architecture down against
any evolution of mflux.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from qds.flux2_dev import (
    TEXT_ENCODER_OUT_LAYERS,
    TEXT_ENCODER_OVERRIDES,
    TRANSFORMER_OVERRIDES,
    Flux2DevTokenizer,
    Flux2DevWeightDefinition,
    Mistral3TextEncoder,
    flux2_dev_model_config,
    single_component_definition,
)
from qds.flux2_dev.weights import TEXT_ENCODER_PREFIX

# ── Config ─────────────────────────────────────────────────────────────────


def test_config_mirrors_the_transformer_config_json():
    # The repo's transformer/config.json. `guidance_embeds` is not in it, but the
    # guidance embedder weights are: FLUX.2-dev is guidance-distilled, unlike
    # klein.
    assert TRANSFORMER_OVERRIDES == {
        "num_layers": 8,
        "num_single_layers": 48,
        "num_attention_heads": 48,
        "attention_head_dim": 128,
        "joint_attention_dim": 15360,
        "guidance_embeds": True,
    }


def test_joint_attention_dim_is_the_product_of_stacked_layers():
    # This is the invariant tying the encoder to the transformer: the
    # `context_embedder` contraction expects exactly n_layers × hidden_size.
    assert (
        TRANSFORMER_OVERRIDES["joint_attention_dim"]
        == len(TEXT_ENCODER_OUT_LAYERS) * (TEXT_ENCODER_OVERRIDES["hidden_size"])
    )


def test_output_layers_follow_the_klein_rule():
    # klein takes (9, 18, 27) over 36 layers, i.e. (n//4, n//2, 3n//4).
    n = TEXT_ENCODER_OVERRIDES["num_hidden_layers"]
    assert TEXT_ENCODER_OUT_LAYERS == (n // 4, n // 2, 3 * n // 4)


def test_model_config_exposes_overrides_and_guidance():
    config = flux2_dev_model_config()
    assert config.model_name == "black-forest-labs/FLUX.2-dev"
    assert config.transformer_overrides == TRANSFORMER_OVERRIDES
    assert config.text_encoder_overrides == TEXT_ENCODER_OVERRIDES
    assert config.supports_guidance is True
    assert config.max_sequence_length == 512
    # The sigma_* defaults must match the repo's scheduler_config.json.
    assert (config.sigma_base_shift, config.sigma_max_shift) == (0.5, 1.15)
    assert (config.sigma_base_seq_len, config.sigma_max_seq_len) == (256, 4096)


def test_model_config_is_not_shared_between_calls():
    # `ModelConfig` keeps the dicts by reference; mutating them would leak from
    # one instance to the next.
    first = flux2_dev_model_config()
    first.transformer_overrides["num_layers"] = 999
    assert flux2_dev_model_config().transformer_overrides["num_layers"] == 8


# ── Weight mappings ────────────────────────────────────────────────────────


def test_transformer_mapping_adds_the_guidance_embedder():
    from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

    base = Flux2WeightMapping.get_transformer_mapping()
    mapping = Flux2DevWeightDefinition.get_transformer_mapping()
    assert len(mapping) == len(base) + 2

    sources = {source for target in mapping for source in target.from_pattern}
    assert "time_guidance_embed.guidance_embedder.linear_1.weight" in sources
    assert "time_guidance_embed.guidance_embedder.linear_2.weight" in sources


def test_text_encoder_mapping_is_mistral_not_qwen():
    mapping = Flux2DevWeightDefinition.get_text_encoder_mapping()
    # 2 global tensors + 9 per layer.
    assert len(mapping) == 11
    # Mistral has no per-head q/k normalization, unlike Qwen3.
    assert not any("q_norm" in target.to_pattern or "k_norm" in target.to_pattern for target in mapping)
    # The checkpoint keys are prefixed: this is a wrapped Mistral3.
    assert all(source.startswith(TEXT_ENCODER_PREFIX) for target in mapping for source in target.from_pattern)


def test_every_expected_transformer_tensor_is_covered():
    # Replays the pattern expansion against the real checkpoint names without
    # downloading them: 11 global tensors, 16 per double block, 4 per single one.
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

    keys = _expected_transformer_keys()
    flat = WeightMapper._build_flat_mapping(
        Flux2DevWeightDefinition.get_transformer_mapping(),
        num_blocks=WeightMapper._detect_num_blocks(dict.fromkeys(keys)),
        num_layers=0,
    )
    assert len(keys) == 331
    assert sorted(key for key in keys if key not in flat) == []


def test_every_expected_encoder_tensor_is_covered():
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

    keys = _expected_text_encoder_keys()
    flat = WeightMapper._build_flat_mapping(
        Flux2DevWeightDefinition.get_text_encoder_mapping(),
        num_blocks=0,
        num_layers=WeightMapper._detect_num_layers(dict.fromkeys(keys)),
    )
    assert len(keys) == 362
    assert sorted(key for key in keys if key not in flat) == []


def test_download_patterns_avoid_the_root_monolith():
    patterns = Flux2DevWeightDefinition.get_download_patterns()
    # The repo exposes a 64.8 GB `flux2-dev.safetensors` that duplicates
    # `transformer/`: a root-level pattern would double the download.
    assert not any(pattern.startswith("*") for pattern in patterns)
    assert "transformer/*.safetensors" in patterns
    assert "tokenizer/**" in patterns


def test_encoder_filters_out_the_vision_tower():
    components = {component.name: component for component in Flux2DevWeightDefinition.get_components()}
    assert sorted(components) == ["text_encoder", "transformer", "vae"]
    # 585 tensors in `text_encoder/`, 223 of which are vision tower, multimodal
    # projector and lm_head that we do not load.
    assert components["text_encoder"].weight_prefix_filters == [TEXT_ENCODER_PREFIX]


def test_single_component_definition_for_prequantization():
    definition = single_component_definition("transformer")
    assert [component.name for component in definition.get_components()] == ["transformer"]
    assert definition.get_tokenizers() == []
    assert definition.get_download_patterns() == ["transformer/*.safetensors", "transformer/*.json"]

    # The tokenizer travels with the encoder: that is what needs it.
    encoder = single_component_definition("text_encoder")
    assert [tokenizer.name for tokenizer in encoder.get_tokenizers()] == ["mistral3"]
    assert "tokenizer/**" in encoder.get_download_patterns()

    with pytest.raises(ValueError, match="Unknown component"):
        single_component_definition("unet")


# ── Embedded guidance ──────────────────────────────────────────────────────


def _tiny_flux2_transformer():
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

    return Flux2Transformer(
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        joint_attention_dim=48,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=True,
    )


def test_transformer_exposes_the_expected_guidance_embedder():
    from mlx.utils import tree_flatten

    embeddings = _tiny_flux2_transformer().time_guidance_embed
    names = {path for path, _ in tree_flatten(embeddings.parameters())}
    # Our mapping's targets must exist on the module side, otherwise FLUX.2-dev's
    # two extra tensors would fall into the void.
    assert {"guidance_linear_1.weight", "guidance_linear_2.weight"} <= names


def test_guidance_must_be_premultiplied_by_a_thousand():
    """Canary on the mflux heuristic that `_guidance_embed` compensates for.

    `Flux2Transformer.__call__` only scales guidance when it is 1.0 or less
    (flux2/.../transformer.py:91), whereas the FLUX.1 path — the only one
    exercised upstream with `guidance_embeds=True` — always multiplies by
    `num_train_steps` (flux/.../transformer.py:155). No shipped mflux model
    enables `guidance_embeds` on the FLUX.2 transformer, so that path is untested
    there.

    If mflux fixes the heuristic, the equality below breaks: the
    pre-multiplication in `Flux2Dev._guidance_embed` will then have to go.
    """
    transformer = _tiny_flux2_transformer()
    mx.random.seed(0)
    arguments = {
        "hidden_states": mx.random.normal((1, 6, 8)).astype(mx.bfloat16),
        "encoder_hidden_states": mx.random.normal((1, 4, 48)).astype(mx.bfloat16),
        "timestep": mx.array(0.5),
        "img_ids": mx.zeros((1, 6, 4), dtype=mx.int32),
        "txt_ids": mx.zeros((1, 4, 4), dtype=mx.int32),
    }

    def run(guidance):
        output = transformer(**arguments, guidance=guidance)
        mx.eval(output)
        return output

    # The scalar is used as-is: pre-multiplying does change the result.
    assert not mx.allclose(run(4.0), run(4000.0))
    # And it is only scaled below 1.0 — hence the compensation on our side.
    assert mx.allclose(run(0.004), run(4.0))


def test_flux2_dev_premultiplies_guidance():
    from qds.flux2_dev.config import flux2_dev_model_config

    config = flux2_dev_model_config()
    # `Flux2Dev._guidance_embed` reproduces `config.guidance *
    # config.num_train_steps` from mflux's FLUX.1 path.
    assert config.num_train_steps == 1000


# ── Tokenizer ──────────────────────────────────────────────────────────────


class _RecordingTokenizer:
    """Captures `apply_chat_template`'s arguments without tokenizing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append({"conversations": conversations, **kwargs})
        batch = len(conversations)
        length = kwargs["max_length"]
        return {
            "input_ids": [[1] * length for _ in range(batch)],
            "attention_mask": [[1] * length for _ in range(batch)],
        }


def test_tokenizer_sends_a_system_message_and_no_generation_prompt():
    raw = _RecordingTokenizer()
    tokens = Flux2DevTokenizer(tokenizer=raw, max_length=16).tokenize("un renard roux")

    call = raw.calls[0]
    roles = [message["role"] for message in call["conversations"][0]]
    # mflux's `LanguageTokenizer` would send the user role only, with
    # add_generation_prompt=True: both departures are silent at runtime.
    assert roles == ["system", "user"]
    assert call["add_generation_prompt"] is False
    assert call["padding"] == "max_length"
    assert call["truncation"] is True
    assert call["max_length"] == 16

    # Contents as lists of typed parts, like Flux2Pipeline.
    user = call["conversations"][0][1]
    assert user["content"] == [{"type": "text", "text": "un renard roux"}]

    assert tokens.input_ids.shape == (1, 16)
    assert tokens.attention_mask.shape == (1, 16)


def test_tokenizer_strips_the_image_token():
    raw = _RecordingTokenizer()
    Flux2DevTokenizer(tokenizer=raw, max_length=8).tokenize("[IMG]un renard")
    user = raw.calls[0]["conversations"][0][1]
    assert user["content"][0]["text"] == "un renard"


def test_system_message_matches_diffusers():
    from qds.flux2_dev import SYSTEM_MESSAGE

    # The line break is in the upstream source and changes tokenization.
    assert SYSTEM_MESSAGE.startswith("You are an AI that reasons about image descriptions.")
    assert "object\nattribution" in SYSTEM_MESSAGE


# ── Mistral3 encoder ───────────────────────────────────────────────────────


def _tiny_encoder(**overrides) -> Mistral3TextEncoder:
    kwargs = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1e6,
        "max_position_embeddings": 64,
    }
    kwargs.update(overrides)
    return Mistral3TextEncoder(**kwargs)


#: A toy shape of the real model. `hidden_size` is deliberately not
#: `num_attention_heads * head_dim`, as on FLUX.2-dev (5120 vs 32 × 128), so
#: `q_proj` and `o_proj` are not square there.
_REFERENCE_SHAPE = {
    "vocab_size": 97,
    "hidden_size": 40,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 64,
    "rms_norm_eps": 1e-5,
    "rope_theta": 1e6,
    "max_position_embeddings": 64,
}


def _reference_pair():
    """A transformers `MistralModel` and our encoder, sharing the same weights."""
    import torch
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralModel

    torch.manual_seed(0)
    reference = MistralModel(MistralConfig(**_REFERENCE_SHAPE, sliding_window=None))
    reference.eval()

    encoder = Mistral3TextEncoder(**_REFERENCE_SHAPE)
    # The weights go through our own mapping table, so it is validated at the
    # same time as the architecture.
    encoder.update(
        WeightMapper.apply_mapping(
            hf_weights={
                f"{TEXT_ENCODER_PREFIX}{key}": mx.array(value.detach().float().numpy())
                for key, value in reference.state_dict().items()
            },
            mapping=Flux2DevWeightDefinition.get_text_encoder_mapping(),
            num_layers=_REFERENCE_SHAPE["num_hidden_layers"],
        )
    )
    return reference, encoder


@pytest.mark.parametrize(
    ("label", "input_ids", "attention_mask"),
    [
        # FLUX.2-dev's tokenizer pads on the left: that is the real case, and
        # the one where a query row ends up entirely masked.
        ("left padding", [0, 0, 0, 0, 5, 12, 7, 33], [0, 0, 0, 0, 1, 1, 1, 1]),
        ("right padding", [5, 12, 7, 33, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]),
        ("no padding", [5, 12, 7, 33, 2, 90, 4, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    ],
)
def test_mlx_encoder_matches_transformers_mistral(label, input_ids, attention_mask):
    """The decisive test: same architecture, same weights, same hidden states.

    A toy model with random weights is enough to validate everything that could
    diverge silently — RoPE, grouped-query attention, the causal mask, the
    padding mask, the order of the norms, the indexing of hidden states. An error
    in any of them would produce images that ignore the prompt, without ever
    raising.
    """
    import numpy as np
    import torch

    reference, encoder = _reference_pair()
    ids = np.array([input_ids])
    mask = np.array([attention_mask])

    with torch.no_grad():
        expected = reference(
            input_ids=torch.tensor(ids),
            attention_mask=torch.tensor(mask),
            output_hidden_states=True,
            use_cache=False,
        )
    _, hidden_states = encoder(mx.array(ids), mx.array(mask), output_hidden_states=True)
    mx.eval(*hidden_states)

    assert len(hidden_states) == len(expected.hidden_states)
    useful = np.array(attention_mask, dtype=bool)
    for index, ours in enumerate(hidden_states):
        # Padding positions are compared neither here nor upstream: nobody reads
        # their output. All that matters is that it stays finite.
        theirs = expected.hidden_states[index].numpy()[:, useful]
        assert np.abs(theirs - np.array(ours)[:, useful]).max() < 1e-5, f"mismatch at hidden_states[{index}]"


def test_left_padding_does_not_produce_nan():
    """Regression: left padding fully masked the leading rows.

    Under a causal mask, a padding query at the head of the sequence has only
    itself to look at — and it is masked. Softmax then returns NaN, and the next
    row propagates it (`0 × NaN = NaN`): by the second layer *every* position is
    contaminated and the whole prompt goes to NaN. That is exactly what
    FLUX.2-dev's tokenizer produces.
    """
    encoder = _tiny_encoder()
    input_ids = mx.array([[0, 0, 0, 0, 5, 12, 7, 33]])
    attention_mask = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]])

    embeds = encoder.get_prompt_embeds(input_ids, attention_mask, hidden_state_layers=(1, 2, 3))
    mx.eval(embeds)
    assert bool(mx.all(mx.isfinite(embeds)))


def test_encoder_produces_n_layers_times_hidden_embedding():
    encoder = _tiny_encoder()
    input_ids = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]])
    attention_mask = mx.array([[1, 1, 1, 1, 1, 0, 0, 0]])

    embeds = encoder.get_prompt_embeds(input_ids, attention_mask, hidden_state_layers=(1, 2, 3))
    mx.eval(embeds)
    assert embeds.shape == (1, 8, 3 * 32)
    assert bool(mx.all(mx.isfinite(embeds)))


def test_first_hidden_state_is_the_embedding_output():
    # HF-aligned indexing: `hidden_states[0]` precedes the first layer, otherwise
    # layers (10, 20, 30) would be off by one.
    encoder = _tiny_encoder()
    input_ids = mx.array([[3, 4, 5]])
    _, hidden_states = encoder(input_ids, output_hidden_states=True)
    assert len(hidden_states) == encoder.num_hidden_layers + 1
    assert mx.allclose(hidden_states[0], encoder.embed_tokens(input_ids))


def test_attention_is_causal():
    # A token must not see its successors: changing the last token leaves the
    # preceding positions untouched.
    encoder = _tiny_encoder()
    first, _ = encoder(mx.array([[3, 4, 5, 6]]))
    second, _ = encoder(mx.array([[3, 4, 5, 9]]))
    mx.eval(first, second)
    assert mx.allclose(first[:, :3], second[:, :3], atol=1e-5)
    assert not mx.allclose(first[:, 3], second[:, 3], atol=1e-5)


def test_padding_mask_isolates_the_useful_positions():
    # The prompt is padded up to max_length: without a mask, the padding would
    # pollute all 512 positions handed to the transformer.
    encoder = _tiny_encoder()
    input_ids = mx.array([[3, 4, 5, 7, 7]])
    attention_mask = mx.array([[1, 1, 1, 0, 0]])
    masked, _ = encoder(input_ids, attention_mask)
    reference, _ = encoder(mx.array([[3, 4, 5]]))
    mx.eval(masked, reference)
    assert mx.allclose(masked[:, :3], reference, atol=1e-5)


def test_attention_handles_grouped_query():
    # hidden_size is not num_heads * head_dim on this model: q_proj and o_proj
    # are not square, and the KV heads are repeated.
    encoder = _tiny_encoder(hidden_size=40, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    attention = encoder.layers[0].self_attn
    assert attention.q_proj.weight.shape == (32, 40)
    assert attention.k_proj.weight.shape == (16, 40)
    assert attention.o_proj.weight.shape == (40, 32)
    assert attention.num_key_value_groups == 2

    output, _ = encoder(mx.array([[1, 2, 3]]))
    mx.eval(output)
    assert output.shape == (1, 3, 40)


def test_architecture_matches_the_checkpoint_weight_names():
    # The MLX paths must be exactly the mapping's targets, otherwise
    # `model.update(..., strict=False)` would leave random weights in place.
    from mlx.utils import tree_flatten

    encoder = _tiny_encoder(num_hidden_layers=2)
    paths = {path for path, _ in tree_flatten(encoder.parameters())}

    expected = {"embed_tokens.weight", "norm.weight"}
    for layer in range(2):
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            expected.add(f"layers.{layer}.{suffix}")

    # `rotary_emb.inv_freq` is computed, not loaded.
    assert paths - {"rotary_emb.inv_freq"} == expected

    targets = {
        target.to_pattern.replace("{layer}", str(layer))
        for target in Flux2DevWeightDefinition.get_text_encoder_mapping()
        for layer in range(2)
    }
    assert targets == expected


# ── Expected checkpoint tensor names ───────────────────────────────────────


def _expected_transformer_keys() -> list[str]:
    keys = [
        "context_embedder.weight",
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.linear.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
        "single_stream_modulation.linear.weight",
        "time_guidance_embed.guidance_embedder.linear_1.weight",
        "time_guidance_embed.guidance_embedder.linear_2.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        "x_embedder.weight",
    ]
    double = (
        "attn.add_k_proj.weight",
        "attn.add_q_proj.weight",
        "attn.add_v_proj.weight",
        "attn.norm_added_k.weight",
        "attn.norm_added_q.weight",
        "attn.norm_k.weight",
        "attn.norm_q.weight",
        "attn.to_add_out.weight",
        "attn.to_k.weight",
        "attn.to_out.0.weight",
        "attn.to_q.weight",
        "attn.to_v.weight",
        "ff.linear_in.weight",
        "ff.linear_out.weight",
        "ff_context.linear_in.weight",
        "ff_context.linear_out.weight",
    )
    single = (
        "attn.norm_k.weight",
        "attn.norm_q.weight",
        "attn.to_out.weight",
        "attn.to_qkv_mlp_proj.weight",
    )
    keys += [f"transformer_blocks.{block}.{suffix}" for block in range(8) for suffix in double]
    keys += [f"single_transformer_blocks.{block}.{suffix}" for block in range(48) for suffix in single]
    return keys


def _expected_text_encoder_keys() -> list[str]:
    keys = [f"{TEXT_ENCODER_PREFIX}embed_tokens.weight", f"{TEXT_ENCODER_PREFIX}norm.weight"]
    suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    keys += [
        f"{TEXT_ENCODER_PREFIX}layers.{layer}.{suffix}"
        for layer in range(TEXT_ENCODER_OVERRIDES["num_hidden_layers"])
        for suffix in suffixes
    ]
    return keys
