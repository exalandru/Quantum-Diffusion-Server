"""Anima's text adapter -- `llm_adapter` in the checkpoint -- ported to MLX.

This is the one module that is Anima's own. Everything else it sits between is
borrowed: Qwen3-0.6B produces hidden states on one side, and NVIDIA's Cosmos DiT
expects Cosmos-shaped text embeddings on the other. The adapter bridges them by
cross-attending from *learned T5 token embeddings* -- the T5 tokenizer supplies
ids, and this module owns the embedding table those ids index -- to the Qwen3
states.

Reference: diffusers' `AnimaTextConditioner`
(`diffusers/models/condition_embedders/condition_embedder_anima.py`, 0.39.0).
Module names mirror it, which is also what the checkpoint uses once the
`model.diffusion_model.llm_adapter.` prefix is stripped: the two agree tensor for
tensor, all 118 of them, so the mapping is the identity.

`in_proj` is deliberately absent. The reference makes it `nn.Identity()` whenever
`model_dim == target_dim`, which is Anima's case, and the checkpoint carries no
weights for it.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


def _rope_tables(
    seq_len: int, head_dim: int, theta: float = 10000.0, dtype: mx.Dtype = mx.float32
) -> tuple[mx.array, mx.array]:
    """cos/sin of shape [S, head_dim], each frequency duplicated across the halves."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = mx.outer(mx.arange(seq_len, dtype=mx.float32), inv_freq)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x is [B, H, S, D]; the rotation pairs element `i` with `i + D/2`."""
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class AnimaConditionerAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def _heads(self, x: mx.array) -> mx.array:
        b, s, _ = x.shape
        return x.reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rope: tuple[mx.array, mx.array] | None = None,
        encoder_rope: tuple[mx.array, mx.array] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = self.q_norm(self._heads(self.q_proj(hidden_states)))
        k = self.k_norm(self._heads(self.k_proj(context)))
        v = self._heads(self.v_proj(context))

        if rope is not None:
            q = _apply_rope(q, rope[0][None, None], rope[1][None, None])
            # Queries and keys are positioned independently: in cross-attention the
            # key side carries the *source* sequence's positions, not the target's.
            ek = encoder_rope if encoder_rope is not None else rope
            k = _apply_rope(k, ek[0][None, None], ek[1][None, None])

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        b, _, s, _ = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class AnimaConditionerBlock(nn.Module):
    def __init__(self, source_dim: int, model_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        head_dim = model_dim // num_heads
        self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.self_attn = AnimaConditionerAttention(model_dim, model_dim, num_heads, head_dim)
        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = AnimaConditionerAttention(model_dim, source_dim, num_heads, head_dim)
        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        inner = int(model_dim * mlp_ratio)
        # `mlp.0` and `mlp.2`, with the activation at index 1, because that is how
        # the reference's `nn.Sequential` numbers its weights.
        self.mlp = [nn.Linear(model_dim, inner), nn.GELU(), nn.Linear(inner, model_dim)]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        rope: tuple[mx.array, mx.array],
        source_rope: tuple[mx.array, mx.array],
        target_mask: mx.array | None = None,
        source_mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.norm_self_attn(hidden_states), rope=rope, encoder_rope=rope, mask=target_mask
        )
        hidden_states = hidden_states + self.cross_attn(
            self.norm_cross_attn(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            rope=rope,
            encoder_rope=source_rope,
            mask=source_mask,
        )
        normed = self.norm_mlp(hidden_states)
        return hidden_states + self.mlp[2](self.mlp[1](self.mlp[0](normed)))


class AnimaTextConditioner(nn.Module):
    def __init__(
        self,
        source_dim: int = 1024,
        target_dim: int = 1024,
        model_dim: int = 1024,
        num_layers: int = 6,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
        target_vocab_size: int = 32128,
        min_sequence_length: int = 512,
        use_self_attention: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        if not use_self_attention or use_layer_norm:
            # Anima ships neither variant; refusing beats silently building a
            # module whose weights would not fit the checkpoint.
            raise ValueError(
                "Only Anima's published shape is implemented: "
                "use_self_attention=True, use_layer_norm=False."
            )
        self.head_dim = model_dim // num_attention_heads
        self.min_sequence_length = min_sequence_length
        self.embed = nn.Embedding(target_vocab_size, target_dim)
        self.blocks = [
            AnimaConditionerBlock(source_dim, model_dim, num_attention_heads, mlp_ratio)
            for _ in range(num_layers)
        ]
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6)

    def __call__(
        self,
        source_hidden_states: mx.array,
        target_input_ids: mx.array,
        target_attention_mask: mx.array | None = None,
        source_attention_mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.embed(target_input_ids).astype(source_hidden_states.dtype)

        # Cast to the activation dtype: these are derived tables, and leaving
        # them float32 would promote every activation that touches them.
        dtype = hidden_states.dtype
        rope = _rope_tables(hidden_states.shape[1], self.head_dim, dtype=dtype)
        source_rope = _rope_tables(source_hidden_states.shape[1], self.head_dim, dtype=dtype)

        target_mask = _attention_mask(target_attention_mask, dtype)
        source_mask = _attention_mask(source_attention_mask, dtype)

        for block in self.blocks:
            hidden_states = block(
                hidden_states, source_hidden_states, rope, source_rope, target_mask, source_mask
            )

        hidden_states = self.norm(self.out_proj(hidden_states))

        if target_attention_mask is not None:
            # Padding positions are zeroed rather than left to whatever the norm
            # produced for them: the DiT cross-attends over this unmasked.
            hidden_states = hidden_states * target_attention_mask.astype(hidden_states.dtype)[..., None]

        pad = self.min_sequence_length - hidden_states.shape[1]
        if pad > 0:
            hidden_states = mx.pad(hidden_states, [(0, 0), (0, pad), (0, 0)])
        return hidden_states


def _attention_mask(mask: mx.array | None, dtype: mx.Dtype = mx.float32) -> mx.array | None:
    """A [B, S] keep-mask becomes the additive [B, 1, 1, S] form SDPA wants."""
    if mask is None:
        return None
    keep = mask.astype(mx.bool_)
    if keep.ndim == 2:
        keep = keep[:, None, None, :]
    return mx.where(keep, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))
