"""SD 3.5's two CLIP text towers, ported to MLX.

The reference is HuggingFace's `CLIPTextModelWithProjection`
(`transformers/models/clip/modeling_clip.py`), which is what `model_index.json` names
for both `text_encoder` and `text_encoder_2`. Module and parameter names mirror it
exactly, so the checkpoints load without a rename table.

Written locally rather than reused from mflux, and that was a decision with a reason
rather than a preference. mflux ships a CLIP tower for FLUX.1
(`mflux/models/flux/model/flux_text_encoder/clip_encoder/`) whose dimensions happen to
match CLIP-L — but it is *hard-coded* to them (`clip_sdpa_attention.py` fixes 12 heads
of 64, `clip_mlp.py` fixes 768→3072 quick-GELU, `clip_text_model.py` fixes the final
norm at 768), it has no `text_projection`, and it returns only the pooled vector.
SD 3.5 needs all three of the things that are missing:

* a **1280-wide, 32-layer, plain-GELU** tower as well as the 768-wide quick-GELU one;
* the **penultimate hidden states**, not the final ones — `_get_clip_prompt_embeds`
  takes `hidden_states[-2]`, the output of the second-to-last layer, before
  `final_layer_norm`;
* the **projected** pooled vector, `text_projection(pooled)`, which is what
  `CLIPTextModelWithProjection` returns first and what half of the transformer's
  pooled conditioning is.

Parameterising mflux's classes for that would mean restructuring them in
site-packages. One local tower serves both instead, and `SD35ClipL` / `SD35ClipG` are
the two published configurations of it.

Two details that are silent when wrong: the attention is **causal** (CLIP is a
decoder-style text tower even though nothing is generated from it), and pooling takes
the position of the **highest token id**, which is the end-of-text token — the same
`input_ids.argmax(-1)` HuggingFace uses when `eos_token_id == 2`, as it is here.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

#: The activations the two published configs name, and nothing else. `quick_gelu` is
#: CLIP-L's; CLIP-G asks for plain `gelu`, which HuggingFace resolves to the exact
#: erf form, not the tanh approximation. One tower, two activations, and the
#: difference between them is invisible in the weights.
_ACTIVATIONS = {
    "quick_gelu": lambda x: x * mx.sigmoid(1.702 * x),
    "gelu": nn.gelu,
}


class _CLIPEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        positions = mx.arange(input_ids.shape[-1])
        return self.token_embedding(input_ids) + self.position_embedding(positions)[None]


class _CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _heads(self, x: mx.array) -> mx.array:
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq, _ = x.shape
        query = self._heads(self.q_proj(x))
        key = self._heads(self.k_proj(x))
        value = self._heads(self.v_proj(x))
        # Causal, always. CLIP's text tower is trained with a causal mask, and the
        # pooled vector is read off the end-of-text position precisely because every
        # earlier position is invisible to it.
        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask="causal"
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, self.num_heads * self.head_dim)
        return self.out_proj(out)


class _CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self._activation = _ACTIVATIONS[hidden_act]

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self._activation(self.fc1(x)))


class _CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, hidden_act: str, eps: float):
        super().__init__()
        self.self_attn = _CLIPAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = _CLIPMLP(hidden_size, intermediate_size, hidden_act)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class _CLIPEncoder(nn.Module):
    def __init__(self, num_layers: int, **layer_kwargs):
        super().__init__()
        self.layers = [_CLIPEncoderLayer(**layer_kwargs) for _ in range(num_layers)]


class _CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.embeddings = _CLIPEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = _CLIPEncoder(
            num_hidden_layers,
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            eps=layer_norm_eps,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)


class SD35ClipTower(nn.Module):
    """`CLIPTextModelWithProjection`: hidden states from layer -2, and a pooled projection.

    Returns both halves of what the pipeline needs from one forward pass, because both
    are read from the same run: the sequence conditioning comes from an intermediate
    layer, and the pooled conditioning from the end of the tower.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        projection_dim: int = 768,
        hidden_act: str = "quick_gelu",
        vocab_size: int = 49408,
        max_position_embeddings: int = 77,
        layer_norm_eps: float = 1e-5,
        hidden_state_index: int = -2,
    ):
        super().__init__()
        if hidden_act not in _ACTIVATIONS:
            raise ValueError(
                f"SD 3.5's CLIP configs declare 'quick_gelu' (L) or 'gelu' (G); got {hidden_act!r}."
            )
        if not -num_hidden_layers <= hidden_state_index < 0:
            raise ValueError(
                f"hidden_state_index {hidden_state_index} is outside this tower's "
                f"{num_hidden_layers} layers."
            )
        self.num_hidden_layers = num_hidden_layers
        self.hidden_state_index = hidden_state_index
        self.text_model = _CLIPTextTransformer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
        self.text_projection = nn.Linear(hidden_size, projection_dim, bias=False)

    @property
    def layers(self):
        """The encoder layers, under the name `prequantize._quantization_units` looks for.

        Quantizing this tower one layer at a time is what keeps the conversion's memory
        peak bounded, and that helper finds units by attribute name. The parameters
        still live at `text_model.encoder.layers.N`, which is where the checkpoint puts
        them — this is a second name for the same list, not a second list.
        """
        return self.text_model.encoder.layers

    def __call__(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        """`(hidden_states, pooled_projection)` for one prompt.

        `hidden_states` is HuggingFace's `hidden_states[-2]`: that tuple starts with the
        embedding output, so entry `-2` is the output of layer `num_layers - 2` — all
        but the last layer, and before `final_layer_norm`. Taking the final states
        instead is a plausible-looking image that is not the model's.
        """
        # Negative index into the tuple whose first entry is the embedding output; the
        # arithmetic is done once here rather than by slicing a list of every state.
        capture_at = self.num_hidden_layers + self.hidden_state_index
        hidden = self.text_model.embeddings(input_ids)
        captured = None
        for index, layer in enumerate(self.text_model.encoder.layers):
            hidden = layer(hidden)
            if index == capture_at:
                captured = hidden
        last_hidden_state = self.text_model.final_layer_norm(hidden)

        # The end-of-text token is the highest id in the vocabulary, so `argmax` finds
        # its first occurrence — which is what HuggingFace does for `eos_token_id == 2`,
        # the value both of SD 3.5's CLIP configs declare.
        eos = mx.argmax(input_ids, axis=-1)
        pooled = last_hidden_state[mx.arange(input_ids.shape[0]), eos]
        return captured, self.text_projection(pooled)


class SD35ClipL(SD35ClipTower):
    """`text_encoder`: CLIP ViT-L/14, 768 wide over 12 layers, quick-GELU."""


class SD35ClipG(SD35ClipTower):
    """`text_encoder_2`: OpenCLIP ViT-bigG/14, 1280 wide over 32 layers, plain GELU."""

    def __init__(
        self,
        *,
        hidden_size: int = 1280,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 20,
        intermediate_size: int = 5120,
        projection_dim: int = 1280,
        hidden_act: str = "gelu",
        vocab_size: int = 49408,
        max_position_embeddings: int = 77,
        layer_norm_eps: float = 1e-5,
        hidden_state_index: int = -2,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            projection_dim=projection_dim,
            hidden_act=hidden_act,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            hidden_state_index=hidden_state_index,
        )
