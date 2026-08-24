"""How SD 3.5's three text encoders become the transformer's two conditioning tensors.

The reference is `StableDiffusion3Pipeline.encode_prompt`
(`diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py`). There is no
module here and no weights: everything the encoders' outputs go through before the
transformer is concatenation and a zero pad. The projections that *do* have weights —
`context_embedder` and `time_text_embed.text_embedder` — live inside the transformer,
which is where the checkpoint puts them, so inventing a "conditioner" component with
parameters would be inventing a fifth thing the model does not have.

Two tensors come out:

* **`encoder_hidden_states`**, `[1, 77 + T5_len, 4096]` — the two CLIP towers'
  penultimate states concatenated along the feature axis (768 + 1280 = 2048), then
  **right-padded with zeros to 4096** so they can be concatenated with the T5 states
  along the *sequence* axis. The pad is not cosmetic: the transformer's
  `context_embedder` is a single 4096-wide projection applied to both halves, and the
  CLIP half occupies its first 2048 inputs with the rest held at zero.
* **`pooled_projections`**, `[1, 2048]` — the two towers' projected pooled vectors
  concatenated (768 + 1280), which is exactly `pooled_projection_dim`.

The CLIP half comes first along the sequence axis, and the order is load-bearing:
positions are not encoded in this stream, but the transformer was trained with the
CLIP tokens leading.
"""

from __future__ import annotations

import mlx.core as mx


def joint_context(
    clip_l_states: mx.array,
    clip_g_states: mx.array,
    t5_states: mx.array,
) -> mx.array:
    """`[1, clip_len + t5_len, joint_dim]` — the sequence the transformer cross-attends to."""
    clip_states = mx.concatenate([clip_l_states, clip_g_states], axis=-1)
    joint_dim = t5_states.shape[-1]
    padding = joint_dim - clip_states.shape[-1]
    if padding < 0:
        raise ValueError(
            f"The CLIP pair is {clip_states.shape[-1]} wide but the T5 stream is only "
            f"{joint_dim}; the joint context cannot be narrower than its CLIP half."
        )
    if padding:
        clip_states = mx.pad(clip_states, [(0, 0), (0, 0), (0, padding)])
    return mx.concatenate([clip_states, t5_states], axis=-2)


def pooled_projections(clip_l_pooled: mx.array, clip_g_pooled: mx.array) -> mx.array:
    """`[1, 2048]` — the timestep-conditioning half that comes from the prompt."""
    return mx.concatenate([clip_l_pooled, clip_g_pooled], axis=-1)
