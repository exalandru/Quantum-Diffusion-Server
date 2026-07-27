"""Définition des poids FLUX.2-dev, sur le modèle de `Flux2KleinWeightDefinition`.

Deux écarts seulement par rapport à klein :

* le transformer de FLUX.2-dev embarque deux tenseurs de plus
  (`time_guidance_embed.guidance_embedder.linear_{1,2}`), parce qu'il est
  guidance-distilled ;
* l'encodeur texte est un Mistral3 et non un Qwen3 — d'où un mapping dédié, sans
  `q_norm`/`k_norm` et préfixé `language_model.`.

Le VAE (`AutoencoderKLFlux2`) est le même : `Flux2WeightMapping.get_vae_mapping`
s'applique tel quel.
"""

from __future__ import annotations

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

from mflux_server.flux2_dev.config import MAX_SEQUENCE_LENGTH
from mflux_server.flux2_dev.tokenizer import Flux2DevTokenizer

#: `text_encoder/` contient 585 tenseurs dont 223 inutiles ici : tour vision
#: (`vision_tower.*`), projecteur multimodal et `language_model.lm_head`. On les
#: écarte avant le mapping pour ne pas les charger en mémoire.
TEXT_ENCODER_PREFIX = "language_model.model."


class Flux2DevWeightDefinition:
    @staticmethod
    def get_transformer_mapping() -> list[WeightTarget]:
        # Le mapping klein couvre 329 des 331 tenseurs ; il ne connaît pas le
        # guidance embedder, désactivé sur les modèles klein.
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
        # `{layer}` est étendu automatiquement : WeightMapper._detect_num_layers
        # cherche `model\.layers\.(\d+)\.`, qui matche dans
        # `language_model.model.layers.N.` → 40 couches détectées.
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
        # Surtout pas de `*.safetensors` à la racine : le repo y expose un
        # monolithe `flux2-dev.safetensors` de 64,8 Go qui duplique
        # `transformer/`.
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
    """Fabrique une définition à un seul composant, pour la pré-quantification.

    `WeightApplier.apply_and_quantize` et `ModelSaver.save_model` travaillent sur
    l'ensemble des composants d'une définition ; les traiter un par un est ce qui
    permet de ne jamais tenir les 111 Go de bf16 en mémoire.
    """
    components = {c.name: c for c in Flux2DevWeightDefinition.get_components()}
    if name not in components:
        raise ValueError(f"Composant inconnu : {name!r}. Valides : {sorted(components)}")
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
