"""Tour texte de Mistral3, portée en MLX.

FLUX.2 [dev] conditionne son transformer sur trois états cachés intermédiaires
de l'encodeur `Mistral3ForConditionalGeneration` empilés — exactement la même
mécanique que FLUX.2 klein avec Qwen3, mais mflux 0.18.0 ne fournit que le
`Qwen3TextEncoder`.

L'écart est mince : le décodeur Mistral est du dense standard (GQA + RoPE +
RMSNorm + SwiGLU) et *la seule* différence structurelle avec
`Qwen3VLAttention` est l'absence de `q_norm`/`k_norm` par tête. On ne peut pas
réutiliser la classe de mflux en neutralisant ces normes : un `Qwen3VLRMSNorm`
à poids unitaires reste une normalisation, pas une identité. Le reste
(RMSNorm, MLP, RoPE, `_repeat_kv`, `_apply_rotary_pos_emb`) est repris tel quel.

Ce qui est chargé : `language_model.model.*` uniquement. Ni `lm_head`, ni tour
vision, ni projecteur multimodal — on ne lit que des états cachés.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mflux.models.common_models.qwen3_vl.qwen3_vl_attention import Qwen3VLAttention
from mflux.models.common_models.qwen3_vl.qwen3_vl_mlp import Qwen3VLMLP
from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_rotary_embedding import Qwen3TextRotaryEmbedding
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class Mistral3Attention(nn.Module):
    """Attention GQA sans normalisation de q/k.

    Attention au fait que `hidden_size` (5120) n'est pas
    `num_attention_heads * head_dim` (4096) sur ce modèle : `q_proj` et
    `o_proj` ne sont donc pas carrées.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
        position_embeddings: tuple[mx.array, mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query_states, key_states = Qwen3VLAttention._apply_rotary_pos_emb(
            q=query_states,
            k=key_states,
            cos=cos,
            sin=sin,
        )

        if self.num_key_value_heads != self.num_attention_heads:
            key_states = Qwen3VLAttention._repeat_kv(key_states, self.num_key_value_groups)
            value_states = Qwen3VLAttention._repeat_kv(value_states, self.num_key_value_groups)

        # float32 pour l'attention, comme le fait mflux sur ses autres encodeurs.
        attn_output = scaled_dot_product_attention(
            query_states.astype(mx.float32),
            key_states.astype(mx.float32),
            value_states.astype(mx.float32),
            scale=self.scaling,
            mask=attention_mask,
        )

        attn_output = attn_output.astype(query_states.dtype)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attn_output)


class Mistral3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.input_layernorm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Mistral3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        self.post_attention_layernorm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3VLMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
        position_embeddings: tuple[mx.array, mx.array],
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask, position_embeddings)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class Mistral3TextEncoder(nn.Module):
    """Défauts issus de `text_encoder/config.json` de black-forest-labs/FLUX.2-dev."""

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 5120,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 32768,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000000.0,
        max_position_embeddings: int = 131072,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Mistral3DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3TextRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        attention_mask_4d = Mistral3TextEncoder._build_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=hidden_states.dtype,
        )

        position_ids = mx.broadcast_to(
            mx.expand_dims(mx.arange(seq_len, dtype=mx.int32), axis=0),
            (batch_size, seq_len),
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Indexation calquée sur `MistralModel.forward` : l'état est empilé
        # *avant* chaque couche — donc `[0]` est la sortie de l'embedding et
        # `[i]` l'entrée de la couche i — puis la norme finale est empilée en
        # dernier. Sans ça, les couches sélectionnées seraient décalées d'un
        # cran par rapport à `outputs.hidden_states` de transformers.
        hidden_states_list = [] if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                hidden_states_list.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            hidden_states_list.append(hidden_states)

        return hidden_states, hidden_states_list

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        hidden_state_layers: tuple[int, ...] = (10, 20, 30),
    ) -> mx.array:
        """Empile plusieurs états cachés en un embedding de `n * hidden_size`.

        `joint_attention_dim` du transformer FLUX.2-dev vaut 15360 = 3 × 5120,
        d'où les trois couches. Signature alignée sur `Qwen3TextEncoder` pour
        rester utilisable par `Flux2PromptEncoder.encode_prompt`.
        """
        _, hidden_states_list = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if hidden_states_list is None:  # pragma: no cover - défensif
            raise RuntimeError("États cachés indisponibles pour l'encodage du prompt.")

        stacked = mx.stack([hidden_states_list[i] for i in hidden_state_layers], axis=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return mx.transpose(stacked, (0, 2, 1, 3)).reshape(batch_size, seq_len, num_layers * hidden_dim)

    @staticmethod
    def _build_attention_mask(
        attention_mask: mx.array | None,
        batch_size: int,
        seq_len: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        """Masque additif causal + padding, en `(B, 1, S, S)`.

        Construit en booléens pour pouvoir rouvrir les lignes entièrement
        fermées. C'est indispensable, pas cosmétique : le tokenizer de
        FLUX.2-dev complète **à gauche**, donc les premières requêtes sont des
        tokens de padding qui, sous masque causal, n'ont qu'eux-mêmes à
        regarder — et se retrouvent tout masquées. Le softmax y renvoie alors
        des NaN, que la ligne suivante propage (`0 × NaN = NaN`) jusqu'à
        contaminer *toutes* les positions dès la deuxième couche. transformers
        applique le même correctif (`AttentionMaskConverter._unmask_unattended`).
        """
        indices = mx.arange(seq_len, dtype=mx.int32)
        attendable = mx.expand_dims(indices, axis=0) <= mx.expand_dims(indices, axis=1)
        attendable = mx.broadcast_to(
            mx.expand_dims(mx.expand_dims(attendable, axis=0), axis=0),
            (batch_size, 1, seq_len, seq_len),
        )

        if attention_mask is not None:
            not_padding = mx.expand_dims(mx.expand_dims(attention_mask == 1, axis=1), axis=1)
            attendable = mx.logical_and(attendable, not_padding)
            # Une ligne sans aucune clé visible est rouverte en entier : sa
            # sortie n'est jamais lue, mais elle doit rester finie.
            attendable = mx.logical_or(
                attendable,
                mx.all(mx.logical_not(attendable), axis=-1, keepdims=True),
            )

        return mx.where(
            attendable,
            mx.zeros((1,), dtype=dtype),
            mx.full((1,), -float("inf"), dtype=dtype),
        )
