"""Support FLUX.2-dev : config, mappings de poids, tokenizer, encodeur MLX.

Aucun poids réel n'est chargé. Les valeurs attendues viennent des `config.json`
du repo `black-forest-labs/FLUX.2-dev` et des index de poids — c'est ce qui rend
ces tests utiles : ils figent l'architecture face à une évolution de mflux.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mflux_server.flux2_dev import (
    TEXT_ENCODER_OUT_LAYERS,
    TEXT_ENCODER_OVERRIDES,
    TRANSFORMER_OVERRIDES,
    Flux2DevTokenizer,
    Flux2DevWeightDefinition,
    Mistral3TextEncoder,
    flux2_dev_model_config,
    single_component_definition,
)
from mflux_server.flux2_dev.weights import TEXT_ENCODER_PREFIX

# ── Config ─────────────────────────────────────────────────────────────────


def test_config_reprend_le_transformer_config_json():
    # transformer/config.json du repo. `guidance_embeds` n'y figure pas mais les
    # poids du guidance embedder sont présents : FLUX.2-dev est distillé sur la
    # guidance, contrairement à klein.
    assert TRANSFORMER_OVERRIDES == {
        "num_layers": 8,
        "num_single_layers": 48,
        "num_attention_heads": 48,
        "attention_head_dim": 128,
        "joint_attention_dim": 15360,
        "guidance_embeds": True,
    }


def test_joint_attention_dim_est_le_produit_des_couches_empilees():
    # C'est l'invariant qui relie l'encodeur au transformer : la contraction
    # `context_embedder` attend exactement n_couches × hidden_size.
    assert (
        TRANSFORMER_OVERRIDES["joint_attention_dim"]
        == len(TEXT_ENCODER_OUT_LAYERS) * (TEXT_ENCODER_OVERRIDES["hidden_size"])
    )


def test_couches_de_sortie_suivent_la_regle_de_klein():
    # klein prend (9, 18, 27) sur 36 couches, soit (n//4, n//2, 3n//4).
    n = TEXT_ENCODER_OVERRIDES["num_hidden_layers"]
    assert TEXT_ENCODER_OUT_LAYERS == (n // 4, n // 2, 3 * n // 4)


def test_model_config_expose_les_overrides_et_la_guidance():
    config = flux2_dev_model_config()
    assert config.model_name == "black-forest-labs/FLUX.2-dev"
    assert config.transformer_overrides == TRANSFORMER_OVERRIDES
    assert config.text_encoder_overrides == TEXT_ENCODER_OVERRIDES
    assert config.supports_guidance is True
    assert config.max_sequence_length == 512
    # Les défauts sigma_* doivent correspondre au scheduler_config.json du repo.
    assert (config.sigma_base_shift, config.sigma_max_shift) == (0.5, 1.15)
    assert (config.sigma_base_seq_len, config.sigma_max_seq_len) == (256, 4096)


def test_model_config_nest_pas_partagee_entre_appels():
    # `ModelConfig` garde les dicts par référence ; les muter fuiterait d'une
    # instance à l'autre.
    first = flux2_dev_model_config()
    first.transformer_overrides["num_layers"] = 999
    assert flux2_dev_model_config().transformer_overrides["num_layers"] == 8


# ── Mappings de poids ──────────────────────────────────────────────────────


def test_mapping_transformer_ajoute_le_guidance_embedder():
    from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

    base = Flux2WeightMapping.get_transformer_mapping()
    mapping = Flux2DevWeightDefinition.get_transformer_mapping()
    assert len(mapping) == len(base) + 2

    sources = {source for target in mapping for source in target.from_pattern}
    assert "time_guidance_embed.guidance_embedder.linear_1.weight" in sources
    assert "time_guidance_embed.guidance_embedder.linear_2.weight" in sources


def test_mapping_encodeur_texte_est_mistral_pas_qwen():
    mapping = Flux2DevWeightDefinition.get_text_encoder_mapping()
    # 2 tenseurs globaux + 9 par couche.
    assert len(mapping) == 11
    # Mistral n'a pas de normalisation par tête de q/k, contrairement à Qwen3.
    assert not any("q_norm" in target.to_pattern or "k_norm" in target.to_pattern for target in mapping)
    # Les clés du checkpoint sont préfixées : c'est un Mistral3 encapsulé.
    assert all(source.startswith(TEXT_ENCODER_PREFIX) for target in mapping for source in target.from_pattern)


def test_les_tenseurs_attendus_du_transformer_sont_tous_couverts():
    # Rejoue l'expansion des motifs sur les noms réels du checkpoint, sans les
    # télécharger : 11 tenseurs globaux, 16 par bloc double, 4 par bloc simple.
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

    keys = _expected_transformer_keys()
    flat = WeightMapper._build_flat_mapping(
        Flux2DevWeightDefinition.get_transformer_mapping(),
        num_blocks=WeightMapper._detect_num_blocks(dict.fromkeys(keys)),
        num_layers=0,
    )
    assert len(keys) == 331
    assert sorted(key for key in keys if key not in flat) == []


def test_les_tenseurs_attendus_de_lencodeur_sont_tous_couverts():
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

    keys = _expected_text_encoder_keys()
    flat = WeightMapper._build_flat_mapping(
        Flux2DevWeightDefinition.get_text_encoder_mapping(),
        num_blocks=0,
        num_layers=WeightMapper._detect_num_layers(dict.fromkeys(keys)),
    )
    assert len(keys) == 362
    assert sorted(key for key in keys if key not in flat) == []


def test_download_patterns_evitent_le_monolithe_racine():
    patterns = Flux2DevWeightDefinition.get_download_patterns()
    # Le repo expose un `flux2-dev.safetensors` de 64,8 Go qui duplique
    # `transformer/` : un pattern racine doublerait le téléchargement.
    assert not any(pattern.startswith("*") for pattern in patterns)
    assert "transformer/*.safetensors" in patterns
    assert "tokenizer/**" in patterns


def test_lencodeur_filtre_la_tour_vision():
    components = {component.name: component for component in Flux2DevWeightDefinition.get_components()}
    assert sorted(components) == ["text_encoder", "transformer", "vae"]
    # 585 tenseurs dans `text_encoder/`, dont 223 de tour vision, de projecteur
    # multimodal et de lm_head qu'on ne charge pas.
    assert components["text_encoder"].weight_prefix_filters == [TEXT_ENCODER_PREFIX]


def test_definition_mono_composant_pour_la_prequantification():
    definition = single_component_definition("transformer")
    assert [component.name for component in definition.get_components()] == ["transformer"]
    assert definition.get_tokenizers() == []
    assert definition.get_download_patterns() == ["transformer/*.safetensors", "transformer/*.json"]

    # Le tokenizer voyage avec l'encodeur : c'est lui qui en a besoin.
    encoder = single_component_definition("text_encoder")
    assert [tokenizer.name for tokenizer in encoder.get_tokenizers()] == ["mistral3"]
    assert "tokenizer/**" in encoder.get_download_patterns()

    with pytest.raises(ValueError, match="Composant inconnu"):
        single_component_definition("unet")


# ── Guidance embarquée ─────────────────────────────────────────────────────


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


def test_le_transformer_expose_le_guidance_embedder_attendu():
    from mlx.utils import tree_flatten

    embeddings = _tiny_flux2_transformer().time_guidance_embed
    names = {path for path, _ in tree_flatten(embeddings.parameters())}
    # Les cibles de notre mapping doivent exister côté module, sinon les deux
    # tenseurs supplémentaires de FLUX.2-dev tomberaient dans le vide.
    assert {"guidance_linear_1.weight", "guidance_linear_2.weight"} <= names


def test_la_guidance_doit_etre_premultipliee_par_mille():
    """Vigie sur l'heuristique de mflux, que `_guidance_embed` compense.

    `Flux2Transformer.__call__` ne met la guidance à l'échelle que si elle vaut
    1.0 ou moins (flux2/.../transformer.py:91), alors que le chemin FLUX.1 — le
    seul exercé en amont avec `guidance_embeds=True` — multiplie toujours par
    `num_train_steps` (flux/.../transformer.py:155). Aucun modèle mflux livré
    n'active `guidance_embeds` sur le transformer FLUX.2, donc ce chemin n'y est
    pas testé.

    Si mflux corrige l'heuristique, l'égalité ci-dessous casse : il faudra alors
    retirer la pré-multiplication de `Flux2Dev._guidance_embed`.
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

    # Le scalaire est utilisé tel quel : pré-multiplier change bien le résultat.
    assert not mx.allclose(run(4.0), run(4000.0))
    # Et il n'est mis à l'échelle que sous 1.0 — d'où la compensation côté serveur.
    assert mx.allclose(run(0.004), run(4.0))


def test_flux2_dev_premultiplie_la_guidance():
    from mflux_server.flux2_dev.config import flux2_dev_model_config

    config = flux2_dev_model_config()
    # `Flux2Dev._guidance_embed` reproduit `config.guidance * config.num_train_steps`
    # du chemin FLUX.1 de mflux.
    assert config.num_train_steps == 1000


# ── Tokenizer ──────────────────────────────────────────────────────────────


class _RecordingTokenizer:
    """Capture les arguments d'`apply_chat_template` sans tokeniser."""

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


def test_tokenizer_envoie_un_message_system_et_pas_de_generation_prompt():
    raw = _RecordingTokenizer()
    tokens = Flux2DevTokenizer(tokenizer=raw, max_length=16).tokenize("un renard roux")

    call = raw.calls[0]
    roles = [message["role"] for message in call["conversations"][0]]
    # `LanguageTokenizer` de mflux n'enverrait que le rôle user, avec
    # add_generation_prompt=True : les deux écarts sont silencieux à l'exécution.
    assert roles == ["system", "user"]
    assert call["add_generation_prompt"] is False
    assert call["padding"] == "max_length"
    assert call["truncation"] is True
    assert call["max_length"] == 16

    # Contenus en listes de parts typées, comme Flux2Pipeline.
    user = call["conversations"][0][1]
    assert user["content"] == [{"type": "text", "text": "un renard roux"}]

    assert tokens.input_ids.shape == (1, 16)
    assert tokens.attention_mask.shape == (1, 16)


def test_tokenizer_retire_le_token_dimage():
    raw = _RecordingTokenizer()
    Flux2DevTokenizer(tokenizer=raw, max_length=8).tokenize("[IMG]un renard")
    user = raw.calls[0]["conversations"][0][1]
    assert user["content"][0]["text"] == "un renard"


def test_le_message_system_est_celui_de_diffusers():
    from mflux_server.flux2_dev import SYSTEM_MESSAGE

    # Le retour à la ligne est dans la source amont et change la tokenisation.
    assert SYSTEM_MESSAGE.startswith("You are an AI that reasons about image descriptions.")
    assert "object\nattribution" in SYSTEM_MESSAGE


# ── Encodeur Mistral3 ──────────────────────────────────────────────────────


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


#: Forme jouet du vrai modèle. `hidden_size` n'est volontairement pas
#: `num_attention_heads * head_dim`, comme sur FLUX.2-dev (5120 vs 32 × 128) :
#: `q_proj` et `o_proj` n'y sont pas carrées.
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
    """Un `MistralModel` de transformers et notre encodeur, mêmes poids."""
    import torch
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralModel

    torch.manual_seed(0)
    reference = MistralModel(MistralConfig(**_REFERENCE_SHAPE, sliding_window=None))
    reference.eval()

    encoder = Mistral3TextEncoder(**_REFERENCE_SHAPE)
    # Les poids passent par notre propre table de correspondance : elle est donc
    # validée en même temps que l'architecture.
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
        # Le tokenizer de FLUX.2-dev complète à gauche : c'est le cas réel, et
        # celui où une ligne de requête se retrouve entièrement masquée.
        ("padding à gauche", [0, 0, 0, 0, 5, 12, 7, 33], [0, 0, 0, 0, 1, 1, 1, 1]),
        ("padding à droite", [5, 12, 7, 33, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]),
        ("sans padding", [5, 12, 7, 33, 2, 90, 4, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    ],
)
def test_lencodeur_mlx_reproduit_mistral_de_transformers(label, input_ids, attention_mask):
    """Le test décisif : même architecture, mêmes poids, mêmes états cachés.

    Un modèle jouet à poids aléatoires suffit à valider tout ce qui pourrait
    diverger silencieusement — RoPE, group-query attention, masque causal,
    masque de padding, ordre des normes, indexation des états cachés. Une erreur
    sur n'importe lequel produirait des images qui ignorent le prompt, sans
    jamais lever d'exception.
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
        # Les positions de padding ne sont comparées ni ici ni en amont : leur
        # sortie n'est lue par personne. Seul importe qu'elle reste finie.
        theirs = expected.hidden_states[index].numpy()[:, useful]
        assert np.abs(theirs - np.array(ours)[:, useful]).max() < 1e-5, f"écart à hidden_states[{index}]"


def test_le_padding_a_gauche_ne_produit_pas_de_nan():
    """Régression : le padding à gauche masquait entièrement les premières lignes.

    Sous masque causal, une requête de padding en tête de séquence n'a qu'elle-même
    à regarder — et elle est masquée. Le softmax renvoie alors NaN, et la ligne
    suivante le propage (`0 × NaN = NaN`) : dès la deuxième couche, *toutes* les
    positions sont contaminées et le prompt entier part en NaN. C'est exactement
    ce que produit le tokenizer de FLUX.2-dev.
    """
    encoder = _tiny_encoder()
    input_ids = mx.array([[0, 0, 0, 0, 5, 12, 7, 33]])
    attention_mask = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]])

    embeds = encoder.get_prompt_embeds(input_ids, attention_mask, hidden_state_layers=(1, 2, 3))
    mx.eval(embeds)
    assert bool(mx.all(mx.isfinite(embeds)))


def test_lencodeur_produit_un_embedding_de_n_couches_fois_hidden():
    encoder = _tiny_encoder()
    input_ids = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]])
    attention_mask = mx.array([[1, 1, 1, 1, 1, 0, 0, 0]])

    embeds = encoder.get_prompt_embeds(input_ids, attention_mask, hidden_state_layers=(1, 2, 3))
    mx.eval(embeds)
    assert embeds.shape == (1, 8, 3 * 32)
    assert bool(mx.all(mx.isfinite(embeds)))


def test_le_premier_etat_cache_est_la_sortie_de_lembedding():
    # Indexation alignée sur HF : `hidden_states[0]` précède la première couche,
    # sinon les couches (10, 20, 30) seraient décalées d'un cran.
    encoder = _tiny_encoder()
    input_ids = mx.array([[3, 4, 5]])
    _, hidden_states = encoder(input_ids, output_hidden_states=True)
    assert len(hidden_states) == encoder.num_hidden_layers + 1
    assert mx.allclose(hidden_states[0], encoder.embed_tokens(input_ids))


def test_lattention_est_causale():
    # Un token ne doit pas voir ses successeurs : changer le dernier token laisse
    # les positions précédentes intactes.
    encoder = _tiny_encoder()
    first, _ = encoder(mx.array([[3, 4, 5, 6]]))
    second, _ = encoder(mx.array([[3, 4, 5, 9]]))
    mx.eval(first, second)
    assert mx.allclose(first[:, :3], second[:, :3], atol=1e-5)
    assert not mx.allclose(first[:, 3], second[:, 3], atol=1e-5)


def test_le_masque_de_padding_isole_les_positions_utiles():
    # Le prompt est complété jusqu'à max_length : sans masque, le padding
    # polluerait les 512 positions transmises au transformer.
    encoder = _tiny_encoder()
    input_ids = mx.array([[3, 4, 5, 7, 7]])
    attention_mask = mx.array([[1, 1, 1, 0, 0]])
    masked, _ = encoder(input_ids, attention_mask)
    reference, _ = encoder(mx.array([[3, 4, 5]]))
    mx.eval(masked, reference)
    assert mx.allclose(masked[:, :3], reference, atol=1e-5)


def test_lattention_gere_le_group_query():
    # hidden_size n'est pas num_heads * head_dim sur ce modèle : q_proj et o_proj
    # ne sont pas carrées, et les têtes KV sont répétées.
    encoder = _tiny_encoder(hidden_size=40, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    attention = encoder.layers[0].self_attn
    assert attention.q_proj.weight.shape == (32, 40)
    assert attention.k_proj.weight.shape == (16, 40)
    assert attention.o_proj.weight.shape == (40, 32)
    assert attention.num_key_value_groups == 2

    output, _ = encoder(mx.array([[1, 2, 3]]))
    mx.eval(output)
    assert output.shape == (1, 3, 40)


def test_larchitecture_correspond_aux_noms_de_poids_du_checkpoint():
    # Les chemins MLX doivent être exactement les cibles du mapping, sinon
    # `model.update(..., strict=False)` laisserait des poids aléatoires en place.
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

    # `rotary_emb.inv_freq` est calculé, pas chargé.
    assert paths - {"rotary_emb.inv_freq"} == expected

    targets = {
        target.to_pattern.replace("{layer}", str(layer))
        for target in Flux2DevWeightDefinition.get_text_encoder_mapping()
        for layer in range(2)
    }
    assert targets == expected


# ── Noms de tenseurs attendus du checkpoint ────────────────────────────────


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
