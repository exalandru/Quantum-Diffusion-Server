"""FLUX.2-dev weight definition, modelled on `Flux2KleinWeightDefinition`.

Only two departures from klein:

* the FLUX.2-dev transformer carries two extra tensors
  (`time_guidance_embed.guidance_embedder.linear_{1,2}`), because it is
  guidance-distilled;
* the text encoder is a Mistral3 rather than a Qwen3 — hence a dedicated
  mapping, without `q_norm`/`k_norm` and prefixed with `language_model.`.

The VAE (`AutoencoderKLFlux2`) is the same one: `Flux2WeightMapping.get_vae_mapping`
applies unchanged.
"""

from __future__ import annotations

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

from qds.flux2_dev.config import MAX_SEQUENCE_LENGTH
from qds.flux2_dev.tokenizer import Flux2DevTokenizer

#: `text_encoder/` holds 585 tensors, 223 of which are useless here: the vision
#: tower (`vision_tower.*`), the multimodal projector and
#: `language_model.lm_head`. We filter them out before mapping so they never
#: reach memory.
TEXT_ENCODER_PREFIX = "language_model.model."


class Flux2DevWeightDefinition:
    @staticmethod
    def get_transformer_mapping() -> list[WeightTarget]:
        # The klein mapping covers 329 of the 331 tensors; it does not know
        # about the guidance embedder, which is disabled on klein models.
        mapping = list(Flux2WeightMapping.get_transformer_mapping())
        mapping.extend(
            [
                WeightTarget(
                    to_pattern="time_guidance_embed.guidance_linear_1.weight",
                    from_pattern=["time_guidance_embed.guidance_embedder.linear_1.weight"],
                ),
                WeightTarget(
                    to_pattern="time_guidance_embed.guidance_linear_2.weight",
                    from_pattern=["time_guidance_embed.guidance_embedder.linear_2.weight"],
                ),
            ]
        )
        return mapping

    @staticmethod
    def get_text_encoder_mapping() -> list[WeightTarget]:
        # `{layer}` is expanded automatically: WeightMapper._detect_num_layers
        # looks for `model\.layers\.(\d+)\.`, which matches inside
        # `language_model.model.layers.N.` → 40 layers detected.
        per_layer = (
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
        mapping = [
            WeightTarget(
                to_pattern="embed_tokens.weight",
                from_pattern=[f"{TEXT_ENCODER_PREFIX}embed_tokens.weight"],
            ),
            WeightTarget(
                to_pattern="norm.weight",
                from_pattern=[f"{TEXT_ENCODER_PREFIX}norm.weight"],
            ),
        ]
        mapping.extend(
            WeightTarget(
                to_pattern=f"layers.{{layer}}.{suffix}",
                from_pattern=[f"{TEXT_ENCODER_PREFIX}layers.{{layer}}.{suffix}"],
            )
            for suffix in per_layer
        )
        return mapping

    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=Flux2DevWeightDefinition.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=Flux2DevWeightDefinition.get_text_encoder_mapping,
                weight_prefix_filters=[TEXT_ENCODER_PREFIX],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> list[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="mistral3",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=Flux2DevTokenizer,
                max_length=MAX_SEQUENCE_LENGTH,
                download_patterns=["tokenizer/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> list[str]:
        # Definitely no root-level `*.safetensors`: the repo exposes a 64.8 GB
        # `flux2-dev.safetensors` monolith there that duplicates `transformer/`.
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/**",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")


def single_component_definition(name: str) -> type:
    """Build a single-component definition, for pre-quantization.

    `WeightApplier.apply_and_quantize` and `ModelSaver.save_model` operate on all
    of a definition's components; handling them one at a time is what keeps the
    111 GB of bf16 out of memory.
    """
    components = {c.name: c for c in Flux2DevWeightDefinition.get_components()}
    if name not in components:
        raise ValueError(f"Unknown component: {name!r}. Valid: {sorted(components)}")
    component = components[name]
    tokenizers = Flux2DevWeightDefinition.get_tokenizers() if name == "text_encoder" else []

    class _SingleComponentDefinition:
        @staticmethod
        def get_components() -> list[ComponentDefinition]:
            return [component]

        @staticmethod
        def get_tokenizers() -> list[TokenizerDefinition]:
            return tokenizers

        @staticmethod
        def get_download_patterns() -> list[str]:
            patterns = [f"{component.hf_subdir}/*.safetensors", f"{component.hf_subdir}/*.json"]
            if tokenizers:
                patterns.append("tokenizer/**")
            return patterns

        quantization_predicate = staticmethod(Flux2DevWeightDefinition.quantization_predicate)

    _SingleComponentDefinition.__name__ = f"Flux2Dev{name.title().replace('_', '')}WeightDefinition"
    return _SingleComponentDefinition
