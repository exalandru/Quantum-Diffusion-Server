"""SD 3.5's MMDiT-X transformer, ported to MLX.

The reference is diffusers' `SD3Transformer2DModel`
(`diffusers/models/transformers/transformer_sd3.py`) together with
`JointTransformerBlock`, `JointAttnProcessor2_0`, `SD35AdaLayerNormZeroX`,
`AdaLayerNormZero`, `AdaLayerNormContinuous` and `PatchEmbed`. Module and parameter
names mirror it exactly, because the published checkpoints are in that layout: the
weight mapping in `weights.py` is close to an identity, and every name below can be
checked directly against the tensor index.

Three structural facts, each read off the real checkpoints rather than assumed, and
each of which produces a loadable-but-wrong model if got wrong:

* **the last block is `context_pre_only`.** Its text stream is normalised by an
  `AdaLayerNormContinuous` (a 2x-wide modulation) and then dropped — it carries no
  `attn.to_add_out` and no `ff_context`, because nothing downstream reads the text
  stream again. Every earlier block's `norm1_context` is a 6x-wide `AdaLayerNormZero`.
* **dual attention is Medium-only.** `dual_attention_layers` is `[0..12]` on Medium and
  absent on both large releases. A dual block's `norm1` emits nine modulation vectors
  rather than six, and runs a second, image-only attention whose result is gated in
  before the feed-forward.
* **image tokens come first in the joint attention.** Query, key and value are the
  image stream concatenated with the text stream along the sequence axis, one softmax
  over the whole thing, then split back at the image length. Reversing that order
  loads and runs and quietly produces a different model.

The positional embedding is a learned table (`pos_embed.pos_embed`), stored at
`pos_embed_max_size` squared and *centre-cropped* to the requested resolution — which
is what lets one table serve every size the model accepts.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

#: Width of the sinusoidal timestep features, before `timestep_embedder`. Fixed by
#: the checkpoint: `time_text_embed.timestep_embedder.linear_1.weight` is [inner, 256].
TIME_PROJ_DIM = 256

#: diffusers builds every parameterless norm in this model with `eps=1e-6`.
NORM_EPS = 1e-6


def timestep_features(timesteps: mx.array, dim: int = TIME_PROJ_DIM) -> mx.array:
    """Sinusoidal features matching `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`.

    The flip is not cosmetic: it puts cosine first, and `timestep_embedder` was trained
    against that order. `downscale_freq_shift=0` is why the exponent divides by `half`
    rather than by `half - 1` — SD 3.5 differs from Stable Diffusion 1.x here.
    """
    half = dim // 2
    exponent = -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
    freqs = timesteps.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
    return mx.concatenate([mx.cos(freqs), mx.sin(freqs)], axis=-1)


def _layer_norm(x: mx.array) -> mx.array:
    """LayerNorm with no learnable affine — the scale comes from adaLN instead."""
    return mx.fast.layer_norm(x, None, None, NORM_EPS)


class SD35PatchEmbed(nn.Module):
    """Patchify to tokens, then add a centre-crop of the learned positional table."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, pos_embed_max_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # A plain array, not a submodule: the checkpoint stores it as
        # `pos_embed.pos_embed`, one row per position of a `max_size` square.
        self.pos_embed = mx.zeros((1, pos_embed_max_size * pos_embed_max_size, embed_dim))

    def cropped_pos_embed(self, height: int, width: int) -> mx.array:
        """The centre `h/p x w/p` window of the square table, flattened row-major.

        Centre rather than corner: this is what makes a 1024x1024 image and a
        1024x1536 one share a positional frame instead of drifting apart.
        """
        h = height // self.patch_size
        w = width // self.patch_size
        if h > self.pos_embed_max_size or w > self.pos_embed_max_size:
            raise ValueError(
                f"{height}x{width} needs a {h}x{w} positional window, but this variant's "
                f"table is {self.pos_embed_max_size}x{self.pos_embed_max_size}."
            )
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        table = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        window = table[:, top : top + h, left : left + w, :]
        return window.reshape(1, h * w, window.shape[-1])

    def __call__(self, latents: mx.array) -> mx.array:
        height, width = latents.shape[-2], latents.shape[-1]
        # MLX convolutions are NHWC; the latent arrives NCHW, as the loop carries it.
        x = self.proj(mx.transpose(latents, (0, 2, 3, 1)))
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x + self.cropped_pos_embed(height, width)


class _MLPEmbedder(nn.Module):
    """`linear_1 -> SiLU -> linear_2`.

    Both halves of `time_text_embed` have this shape: diffusers' `TimestepEmbedding`
    (`act_fn="silu"`) and its `PixArtAlphaTextProjection` (also SiLU) differ only in
    input width, so one class carries both and the names stay the checkpoint's.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(out_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class SD35TimeTextEmbed(nn.Module):
    """Timestep plus pooled text, summed into the single conditioning vector."""

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.timestep_embedder = _MLPEmbedder(TIME_PROJ_DIM, embedding_dim)
        self.text_embedder = _MLPEmbedder(pooled_projection_dim, embedding_dim)

    def __call__(self, timestep: mx.array, pooled_projections: mx.array) -> mx.array:
        # The trigonometry is built in float32 and cast to the activation dtype, as the
        # reference does. Leaving it float32 would promote everything downstream.
        dtype = pooled_projections.dtype
        features = timestep_features(timestep).astype(dtype)
        return self.timestep_embedder(features) + self.text_embedder(pooled_projections)


class AdaLayerNormZero(nn.Module):
    """Six modulation vectors: shift/scale/gate for attention, then for the MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim)

    def __call__(self, x: mx.array, temb: mx.array):
        emb = self.linear(nn.silu(temb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(emb, 6, axis=-1)
        normed = _layer_norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return normed, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroX(nn.Module):
    """Nine modulation vectors — MMDiT-X. The extra three drive the second attention.

    Both attention inputs come off *one* normalisation, scaled and shifted twice; they
    are not two independent norms. `SD35AdaLayerNormZeroX` in diffusers.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 9 * dim)

    def __call__(self, x: mx.array, temb: mx.array):
        emb = self.linear(nn.silu(temb))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = mx.split(emb, 9, axis=-1)
        normed = _layer_norm(x)
        hidden = normed * (1 + scale_msa[:, None]) + shift_msa[:, None]
        hidden2 = normed * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp, hidden2, gate_msa2


class AdaLayerNormContinuous(nn.Module):
    """Scale and shift only, no gate — and scale comes *first* out of the projection.

    The ordering is the opposite of `AdaLayerNormZero`'s, which is a genuine trap:
    swapping the two halves produces a model that loads and generates noise.
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(conditioning_embedding_dim, 2 * embedding_dim)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        emb = self.linear(nn.silu(conditioning))
        scale, shift = mx.split(emb, 2, axis=-1)
        return _layer_norm(x) * (1 + scale[:, None]) + shift[:, None]


class _GELUProj(nn.Module):
    """`net.0`: a projection whose activation is folded in, so the name stays `proj`."""

    def __init__(self, proj: nn.Linear):
        super().__init__()
        self.proj = proj

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(self.proj(x))


class SD35FeedForward(nn.Module):
    """`FeedForward(activation_fn="gelu-approximate")`, ratio 4, biases on."""

    def __init__(self, dim: int):
        super().__init__()
        inner = dim * 4
        # `net.0.proj` and `net.2` — index 1 is the reference's dropout, kept so the
        # numbering matches the checkpoint.
        self.net = [_GELUProj(nn.Linear(dim, inner)), nn.Identity(), nn.Linear(inner, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[2](self.net[0](x))


class SD35Attention(nn.Module):
    """Joint attention over the image and text streams, or image-only when `added=False`.

    `added=True` is `JointAttnProcessor2_0` with an encoder stream: the text tokens get
    their own q/k/v projections (`add_*_proj`) and their own qk-norms, are concatenated
    onto the image tokens, attended in one pass, then split back out. `added=False` is
    MMDiT-X's second attention, which sees only the image stream.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        *,
        added: bool = True,
        context_pre_only: bool = False,
    ):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.added = added
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, inner)
        self.to_k = nn.Linear(dim, inner)
        self.to_v = nn.Linear(dim, inner)
        self.norm_q = nn.RMSNorm(head_dim, eps=NORM_EPS)
        self.norm_k = nn.RMSNorm(head_dim, eps=NORM_EPS)
        # A list so the parameter path is `to_out.0.weight` — index 1 is dropout.
        self.to_out = [nn.Linear(inner, dim)]

        if added:
            self.add_q_proj = nn.Linear(dim, inner)
            self.add_k_proj = nn.Linear(dim, inner)
            self.add_v_proj = nn.Linear(dim, inner)
            self.norm_added_q = nn.RMSNorm(head_dim, eps=NORM_EPS)
            self.norm_added_k = nn.RMSNorm(head_dim, eps=NORM_EPS)
            if not context_pre_only:
                self.to_add_out = nn.Linear(inner, dim)

    def _heads(self, x: mx.array) -> mx.array:
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array | None = None
    ) -> tuple[mx.array, mx.array | None]:
        image_len = hidden_states.shape[1]
        query = self.norm_q(self._heads(self.to_q(hidden_states)))
        key = self.norm_k(self._heads(self.to_k(hidden_states)))
        value = self._heads(self.to_v(hidden_states))

        if encoder_hidden_states is not None:
            context_q = self.norm_added_q(self._heads(self.add_q_proj(encoder_hidden_states)))
            context_k = self.norm_added_k(self._heads(self.add_k_proj(encoder_hidden_states)))
            context_v = self._heads(self.add_v_proj(encoder_hidden_states))
            # Image first, text second. The split below relies on this order.
            query = mx.concatenate([query, context_q], axis=2)
            key = mx.concatenate([key, context_k], axis=2)
            value = mx.concatenate([value, context_v], axis=2)

        out = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        batch = out.shape[0]
        out = out.transpose(0, 2, 1, 3).reshape(batch, -1, self.heads * self.head_dim)

        if encoder_hidden_states is None:
            return self.to_out[0](out), None

        image, context = out[:, :image_len], out[:, image_len:]
        image = self.to_out[0](image)
        # The last block drops its text stream, so it has no output projection for it.
        context = None if self.context_pre_only else self.to_add_out(context)
        return image, context


class SD35TransformerBlock(nn.Module):
    """One MMDiT block: joint attention, optional second image attention, two MLPs."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        *,
        context_pre_only: bool,
        use_dual_attention: bool,
    ):
        super().__init__()
        self.context_pre_only = context_pre_only
        self.use_dual_attention = use_dual_attention

        self.norm1 = AdaLayerNormZeroX(dim) if use_dual_attention else AdaLayerNormZero(dim)
        self.norm1_context = (
            AdaLayerNormContinuous(dim, dim) if context_pre_only else AdaLayerNormZero(dim)
        )
        self.attn = SD35Attention(dim, heads, head_dim, context_pre_only=context_pre_only)
        if use_dual_attention:
            self.attn2 = SD35Attention(dim, heads, head_dim, added=False)
        self.ff = SD35FeedForward(dim)
        if not context_pre_only:
            self.ff_context = SD35FeedForward(dim)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array, temb: mx.array
    ) -> tuple[mx.array | None, mx.array]:
        if self.use_dual_attention:
            normed, gate_msa, shift_mlp, scale_mlp, gate_mlp, normed2, gate_msa2 = self.norm1(
                hidden_states, temb
            )
        else:
            normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            normed2 = gate_msa2 = None

        if self.context_pre_only:
            normed_context = self.norm1_context(encoder_hidden_states, temb)
            c_gate_msa = c_shift_mlp = c_scale_mlp = c_gate_mlp = None
        else:
            normed_context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, temb
            )

        attn_out, context_attn_out = self.attn(normed, encoder_hidden_states=normed_context)
        hidden_states = hidden_states + gate_msa[:, None] * attn_out

        if self.use_dual_attention:
            attn2_out, _ = self.attn2(normed2)
            hidden_states = hidden_states + gate_msa2[:, None] * attn2_out

        normed = _layer_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp[:, None] * self.ff(normed)

        if self.context_pre_only:
            # Nothing reads the text stream after the final block.
            return None, hidden_states

        encoder_hidden_states = encoder_hidden_states + c_gate_msa[:, None] * context_attn_out
        normed_context = (
            _layer_norm(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp[:, None] * self.ff_context(normed_context)
        )
        return encoder_hidden_states, hidden_states


class SD35Transformer(nn.Module):
    """`SD3Transformer2DModel`. Defaults are Medium's; the large rows pass overrides.

    Every constructor argument has a default, because `prequantize` builds this
    component on its own to convert it and `test_components.py` holds the family to
    that. The values a catalogue row actually uses come from
    `ModelConfig.transformer_overrides` — `config.MEDIUM_TRANSFORMER_OVERRIDES` or
    `config.LARGE_TRANSFORMER_OVERRIDES` — never from these defaults.
    """

    def __init__(
        self,
        *,
        num_layers: int = 24,
        num_attention_heads: int = 24,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_size: int = 2,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        pos_embed_max_size: int = 384,
        qk_norm: str = "rms_norm",
        dual_attention_layers: tuple[int, ...] = tuple(range(13)),
    ):
        super().__init__()
        if qk_norm != "rms_norm":
            raise ValueError(
                f"SD 3.5 declares qk_norm='rms_norm' in every published config; got {qk_norm!r}. "
                "Another value would need a different normalisation inside the attention."
            )
        inner_dim = num_attention_heads * attention_head_dim
        if caption_projection_dim != inner_dim:
            raise ValueError(
                f"caption_projection_dim {caption_projection_dim} must equal "
                f"num_attention_heads * attention_head_dim ({inner_dim}): the text stream "
                "is carried at the same width as the image stream through every block."
            )

        self.patch_size = patch_size
        self.out_channels = out_channels
        self.inner_dim = inner_dim
        self.dual_attention_layers = tuple(dual_attention_layers)

        self.pos_embed = SD35PatchEmbed(in_channels, inner_dim, patch_size, pos_embed_max_size)
        self.time_text_embed = SD35TimeTextEmbed(inner_dim, pooled_projection_dim)
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)
        dual = set(self.dual_attention_layers)
        self.transformer_blocks = [
            SD35TransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                # Only the last block: its text output is never read again.
                context_pre_only=index == num_layers - 1,
                use_dual_attention=index in dual,
            )
            for index in range(num_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        pooled_projections: mx.array,
    ) -> mx.array:
        """`[B, 16, H, W]` latents in, the same shape of velocity out.

        `timestep` is the raw flow-match timestep the pipeline passes — sigma scaled by
        `num_train_timesteps`, i.e. 0..1000 — not the sigma itself.
        """
        height, width = hidden_states.shape[-2], hidden_states.shape[-1]
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        return self._unpatchify(hidden_states, height, width)

    def _unpatchify(self, hidden_states: mx.array, height: int, width: int) -> mx.array:
        patch = self.patch_size
        channels = self.out_channels
        rows, cols = height // patch, width // patch
        batch = hidden_states.shape[0]
        x = hidden_states.reshape(batch, rows, cols, patch, patch, channels)
        # einsum("nhwpqc->nchpwq") in the reference.
        x = x.transpose(0, 5, 1, 3, 2, 4)
        return x.reshape(batch, channels, rows * patch, cols * patch)
