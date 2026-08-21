"""Anima's DiT -- NVIDIA's Cosmos-Predict2 transformer -- ported to MLX.

The reference is diffusers' `CosmosTransformer3DModel`
(`diffusers/models/transformers/transformer_cosmos.py`, 0.39.0), and this port
mirrors its module names deliberately. The checkpoint ships under ComfyUI's
names instead (`x_embedder`, `adaln_modulation_self_attn`, `mlp.layer1`), but
those two name sets are in total 1:1 correspondence -- 567 tensors, every shape
equal -- so `weights.py` translates once, on load, and everything downstream
reads like the reference it was checked against. `tests/test_anima.py` loads the
same weights into both implementations and compares outputs.

Anima is a still-image model, so the temporal patch size is 1 and every latent
arrives as a single frame. The 3D machinery is kept rather than folded away: the
RoPE tables are built per axis and the time axis simply has extent 1, which is
what makes this comparable to the reference tensor for tensor.

Two configuration branches of the reference are unreachable here and are not
ported: `extra_pos_embed_type` is null (no learnable positional embedding) and
`img_context_dim_in` is null with `use_crossattn_projection` false (no image
context, so cross-attention is the plain processor, not `CosmosAttnProcessor2_5`).
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn


def _timestep_embedding(timesteps: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """Sinusoidal features, matching `Timesteps(flip_sin_to_cos=True, shift=0.0)`.

    The flip is not cosmetic: it puts cosine first, and the `t_embedder` weights
    were trained against that order.
    """
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    freqs = timesteps.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
    return mx.concatenate([mx.cos(freqs), mx.sin(freqs)], axis=-1)


class CosmosPatchEmbed(nn.Module):
    """Fold each (p_t, p_h, p_w) latent patch into one token."""

    def __init__(self, in_channels: int, out_channels: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        p_t, p_h, p_w = patch_size
        self.proj = nn.Linear(in_channels * p_t * p_h * p_w, out_channels, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        b, c, t, h, w = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        x = hidden_states.reshape(b, c, t // p_t, p_t, h // p_h, p_h, w // p_w, p_w)
        # [B, T', H', W', C, p_t, p_h, p_w] -> flatten the trailing four
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(b, t // p_t, h // p_h, w // p_w, c * p_t * p_h * p_w)
        return self.proj(x)


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class CosmosEmbedding(nn.Module):
    """Produces both halves of the conditioning: `temb` and `embedded_timestep`.

    The adaLN layers consume the normalized projection; `temb` is added on top of
    what their own low-rank pair produces.
    """

    def __init__(self, embedding_dim: int, condition_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = nn.RMSNorm(embedding_dim, eps=1e-6)

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        # Built in float32 for the trigonometry, then cast to the activation
        # dtype — the reference does the same (`.type_as(hidden_states)`).
        # Leaving it float32 would promote every activation downstream of it.
        proj = _timestep_embedding(timestep, self.embedding_dim).astype(dtype)
        return self.t_embedder(proj), self.norm(proj)


def _layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """LayerNorm with no learnable affine -- the scale comes from adaLN instead."""
    return mx.fast.layer_norm(x, None, None, eps)


class CosmosAdaLayerNormZero(nn.Module):
    """Shift, scale and gate for one sub-block, from a low-rank projection."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 3 * in_features, bias=False)

    def __call__(
        self, hidden_states: mx.array, embedded_timestep: mx.array, temb: mx.array | None
    ) -> tuple[mx.array, mx.array]:
        emb = self.linear_2(self.linear_1(nn.silu(embedded_timestep)))
        if temb is not None:
            emb = emb + temb
        shift, scale, gate = mx.split(emb, 3, axis=-1)
        if emb.ndim == 2:
            shift, scale, gate = (v[:, None, :] for v in (shift, scale, gate))
        return _layer_norm(hidden_states) * (1 + scale) + shift, gate


class CosmosAdaLayerNorm(nn.Module):
    """The output norm: shift and scale only, no gate."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def __call__(
        self, hidden_states: mx.array, embedded_timestep: mx.array, temb: mx.array | None
    ) -> mx.array:
        emb = self.linear_2(self.linear_1(nn.silu(embedded_timestep)))
        if temb is not None:
            # Only the first two thirds: `temb` is sized for shift/scale/gate and
            # this norm has no gate.
            emb = emb + temb[..., : 2 * self.in_features]
        shift, scale = mx.split(emb, 2, axis=-1)
        if emb.ndim == 2:
            shift, scale = (v[:, None, :] for v in (shift, scale))
        return _layer_norm(hidden_states) * (1 + scale) + shift


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotary embedding, in the half-split convention Cosmos uses.

    diffusers reaches this through `apply_rotary_emb(..., use_real_unbind_dim=-2)`:
    the head dimension splits into two contiguous halves rather than into
    interleaved pairs, so the rotation pairs element `i` with element `i + D/2`.
    Using the interleaved convention here would load and run and quietly produce
    a different model.
    """
    half = x.shape[-1] // 2
    x_real, x_imag = x[..., :half], x[..., half:]
    x_rotated = mx.concatenate([-x_imag, x_real], axis=-1)
    return x * cos + x_rotated * sin


class CosmosAttention(nn.Module):
    def __init__(self, query_dim: int, cross_attention_dim: int | None, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner = heads * dim_head
        kv_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(kv_dim, inner, bias=False)
        self.to_v = nn.Linear(kv_dim, inner, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-6)
        # A list so the parameter path is `to_out.0.weight`, which is the name the
        # reference uses and therefore the name the mapping targets.
        self.to_out = [nn.Linear(inner, query_dim, bias=False)]

    def _heads(self, x: mx.array) -> mx.array:
        b, s, _ = x.shape
        return x.reshape(b, s, self.heads, self.dim_head).transpose(0, 2, 1, 3)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = self.norm_q(self._heads(self.to_q(hidden_states)))
        k = self.norm_k(self._heads(self.to_k(context)))
        v = self._heads(self.to_v(context))

        if rotary_emb is not None:
            cos, sin = rotary_emb
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        b, _, s, _ = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.heads * self.dim_head)
        return self.to_out[0](out)


class CosmosFeedForward(nn.Module):
    """`FeedForward(activation_fn="gelu")`: exact GELU, no gating, no bias."""

    def __init__(self, dim: int, mult: float):
        super().__init__()
        inner = int(dim * mult)
        proj = nn.Linear(dim, inner, bias=False)
        # `net.0.proj` and `net.2` -- index 1 is the reference's dropout, kept so
        # the numbering matches.
        self.net = [_GELUProj(proj), nn.Identity(), nn.Linear(inner, dim, bias=False)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[2](self.net[0](x))


class _GELUProj(nn.Module):
    def __init__(self, proj: nn.Linear):
        super().__init__()
        self.proj = proj

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.proj(x))


class CosmosTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float,
        adaln_lora_dim: int,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.norm1 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)
        self.attn1 = CosmosAttention(hidden_size, None, num_attention_heads, attention_head_dim)
        self.norm2 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)
        self.attn2 = CosmosAttention(
            hidden_size, cross_attention_dim, num_attention_heads, attention_head_dim
        )
        self.norm3 = CosmosAdaLayerNormZero(hidden_size, adaln_lora_dim)
        self.ff = CosmosFeedForward(hidden_size, mlp_ratio)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        embedded_timestep: mx.array,
        temb: mx.array | None,
        rotary_emb: tuple[mx.array, mx.array] | None,
    ) -> mx.array:
        normed, gate = self.norm1(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.attn1(normed, rotary_emb=rotary_emb)

        normed, gate = self.norm2(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.attn2(
            normed, encoder_hidden_states=encoder_hidden_states
        )

        normed, gate = self.norm3(hidden_states, embedded_timestep, temb)
        return hidden_states + gate * self.ff(normed)


class CosmosRotaryPosEmbed:
    """Per-axis RoPE tables. Not an `nn.Module`: it holds no parameters."""

    def __init__(
        self,
        head_dim: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        rope_scale: tuple[float, float, float],
        base_fps: int = 24,
    ):
        self.max_size = [s // p for s, p in zip(max_size, patch_size, strict=True)]
        self.patch_size = patch_size
        self.base_fps = base_fps
        self.dim_h = head_dim // 6 * 2
        self.dim_w = head_dim // 6 * 2
        # Whatever is left over goes to the time axis, so the three add up to the
        # head dimension exactly (44 + 42 + 42 = 128 at Anima's shape).
        self.dim_t = head_dim - self.dim_h - self.dim_w
        self.h_ntk = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def __call__(self, num_frames: int, height: int, width: int) -> tuple[mx.array, mx.array]:
        p_t, p_h, p_w = self.patch_size
        pe_t, pe_h, pe_w = num_frames // p_t, height // p_h, width // p_w

        seq = mx.arange(max(self.max_size), dtype=mx.float32)

        def axis(dim: int, theta: float, extent: int) -> mx.array:
            rng = mx.arange(0, dim, 2, dtype=mx.float32)[: dim // 2] / dim
            return mx.outer(seq[:extent], 1.0 / (theta**rng))

        emb_t = axis(self.dim_t, 10000.0 * self.t_ntk, pe_t)
        emb_h = axis(self.dim_h, 10000.0 * self.h_ntk, pe_h)
        emb_w = axis(self.dim_w, 10000.0 * self.w_ntk, pe_w)

        emb_t = mx.broadcast_to(emb_t[:, None, None, :], (pe_t, pe_h, pe_w, emb_t.shape[-1]))
        emb_h = mx.broadcast_to(emb_h[None, :, None, :], (pe_t, pe_h, pe_w, emb_h.shape[-1]))
        emb_w = mx.broadcast_to(emb_w[None, None, :, :], (pe_t, pe_h, pe_w, emb_w.shape[-1]))

        # Concatenated twice: the half-split rotation wants each frequency to
        # appear in both halves of the head dimension.
        freqs = mx.concatenate([emb_t, emb_h, emb_w] * 2, axis=-1).reshape(pe_t * pe_h * pe_w, -1)
        return mx.cos(freqs), mx.sin(freqs)


class AnimaTransformer(nn.Module):
    """The 2B Cosmos-Predict2 DiT, as Anima configures it."""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 28,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        mlp_ratio: float = 4.0,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] = (1.0, 4.0, 4.0),
        max_size: tuple[int, int, int] = (128, 240, 240),
        concat_padding_mask: bool = True,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.concat_padding_mask = concat_padding_mask

        patch_in = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_in, hidden_size, patch_size)
        self.rope = CosmosRotaryPosEmbed(attention_head_dim, max_size, patch_size, rope_scale)
        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)
        self.transformer_blocks = [
            CosmosTransformerBlock(
                num_attention_heads, attention_head_dim, text_embed_dim, mlp_ratio, adaln_lora_dim
            )
            for _ in range(num_layers)
        ]
        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False
        )

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        padding_mask: mx.array | None = None,
    ) -> mx.array:
        """`hidden_states` is [B, C, T, H, W]; `timestep` is the sigma, per batch item."""
        b, _, num_frames, height, width = hidden_states.shape

        if self.concat_padding_mask:
            if padding_mask is None:
                # What the pipeline passes for a full frame. Kept explicit because
                # the channel exists either way and zeros are a value, not an absence.
                padding_mask = mx.zeros((b, 1, num_frames, height, width), dtype=hidden_states.dtype)
            hidden_states = mx.concatenate([hidden_states, padding_mask], axis=1)

        # Same reasoning as the timestep projection: the tables are computed in
        # float32 and cast, so they do not silently widen the whole model.
        rotary_emb = self.rope(num_frames, height, width)
        cos, sin = (v[None, None, :, :].astype(hidden_states.dtype) for v in rotary_emb)

        p_t, p_h, p_w = self.patch_size
        post_t, post_h, post_w = num_frames // p_t, height // p_h, width // p_w

        x = self.patch_embed(hidden_states).reshape(b, post_t * post_h * post_w, -1)
        temb, embedded_timestep = self.time_embed(timestep, x.dtype)

        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states, embedded_timestep, temb, (cos, sin))

        x = self.proj_out(self.norm_out(x, embedded_timestep, temb))

        # The reference notes that this is not the mirror image of the patching
        # permutation, and it is not: the spatial patch axes come out ahead of the
        # grid axes. Following it literally is the point.
        x = x.reshape(b, post_t * post_h * post_w, p_h, p_w, p_t, self.out_channels)
        x = x.reshape(b, post_t, post_h, post_w, p_h, p_w, p_t, self.out_channels)
        x = x.transpose(0, 7, 1, 6, 2, 4, 3, 5)
        x = x.reshape(b, self.out_channels, post_t * p_t, post_h * p_h, post_w * p_w)
        return x
