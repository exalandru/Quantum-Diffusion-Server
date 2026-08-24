"""Stable Diffusion 3.5's `ModelConfig`s and module shapes, absent from mflux 0.19.0.

Every number here was read off the three gated repositories rather than a model card
— `transformer/config.json`, `vae/config.json`, `text_encoder*/config.json`,
`scheduler/scheduler_config.json` and the real safetensors tensor indexes — and
`tests/test_sd35_transformer.py` pins the shapes against
`tests/fixtures/sd35_config_index.json`, a recording of those files. A value that
drifts from the weights fails a test rather than mis-generating.

Two findings decided the shape of this module, and both contradict what the
architecture is usually described as:

* **the three rows do not share a transformer shape.** Medium is 24 layers of
  24x64; Large and Large Turbo are 38 layers of 38x64. So there are three
  `ModelConfig` factories rather than one config and a weight-file switch (the
  trick `qds/anima/` can use, because Anima's two checkpoints *are* one shape);
* **MMDiT-X dual attention is Medium-only.** Only `stable-diffusion-3.5-medium`
  declares `dual_attention_layers` (`[0..12]`); Large and Large Turbo omit the key
  and their checkpoints carry no `attn2.*` tensors at all. It is a per-variant
  field here, empty for two of the three rows.

This module must not import mflux or torch: `registry` imports it at module scope to
reach the `ModelConfig` factories, and the catalogue path may not pay for that. The
mflux import lives inside the factory function.
"""

from __future__ import annotations

#: The three published releases. All gated, all Stability AI Community licence.
MEDIUM_REPO = "stabilityai/stable-diffusion-3.5-medium"
LARGE_REPO = "stabilityai/stable-diffusion-3.5-large"
LARGE_TURBO_REPO = "stabilityai/stable-diffusion-3.5-large-turbo"

#: Which repository each catalogue row's `model_config_name` loads. One table, so a
#: row, its `ModelConfig` and the weights on disk cannot drift apart.
REPO_BY_CONFIG: dict[str, str] = {
    "sd35_medium": MEDIUM_REPO,
    "sd35_large": LARGE_REPO,
    "sd35_large_turbo": LARGE_TURBO_REPO,
}


def repo_for(model_config_name: str) -> str:
    """The repository a catalogue row's `ModelConfig` name selects."""
    try:
        return REPO_BY_CONFIG[model_config_name]
    except KeyError:
        raise ValueError(
            f"No SD 3.5 repository is registered for {model_config_name!r}. "
            f"Known: {sorted(REPO_BY_CONFIG)}."
        ) from None


#: `scheduler/scheduler_config.json`, byte-identical in all three repos: a *static*
#: shift of 3.0 over 1000 training steps, with no dynamic shifting. mflux expresses
#: shift as `mu` through an exponential time shift, and the two are the same function
#: — exponential shift at `mu` equals static shift at `exp(mu)` — so the model applies
#: `set_mu(log(3.0))`. `tests/test_sd35.py` proves the identity rather than asserting it.
SIGMA_SHIFT = 3.0
NUM_TRAIN_TIMESTEPS = 1000

#: `vae/config.json`. Not FLUX.1's 0.3611/0.1159: the same architecture, differently
#: normalised, which is exactly the pair `SD35VAE` overrides.
VAE_SCALING_FACTOR = 1.5305
VAE_SHIFT_FACTOR = 0.0609
VAE_SCALE_FACTOR = 8
LATENT_CHANNELS = 16

#: Both CLIP towers truncate at their trained position count; the T5 branch is what
#: `StableDiffusion3Pipeline` calls `max_sequence_length`, and 256 is its default.
CLIP_MAX_SEQUENCE_LENGTH = 77
T5_MAX_SEQUENCE_LENGTH = 256

#: The width of the joint context the transformer cross-attends to. The CLIP pair is
#: zero-padded from 2048 up to this before being concatenated with the T5 states.
JOINT_ATTENTION_DIM = 4096

#: `transformer/config.json` for `stable-diffusion-3.5-medium`. 24 heads x 64 = 1536,
#: which is also `caption_projection_dim`; `pos_embed_max_size` 384 is the side of the
#: square positional table (`pos_embed.pos_embed` is [1, 384*384, 1536]).
MEDIUM_TRANSFORMER_OVERRIDES: dict = {
    "num_layers": 24,
    "num_attention_heads": 24,
    "attention_head_dim": 64,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": 2,
    "joint_attention_dim": JOINT_ATTENTION_DIM,
    "caption_projection_dim": 1536,
    "pooled_projection_dim": 2048,
    "pos_embed_max_size": 384,
    "qk_norm": "rms_norm",
    # The MMDiT-X blocks: the first 13 attend over the image stream a second time,
    # separately from the joint pass. Verified against the checkpoint, where exactly
    # these 13 blocks carry `attn2.*` and their `norm1.linear` is 9x wide, not 6x.
    "dual_attention_layers": tuple(range(13)),
}

#: `transformer/config.json` for both large releases — byte-identical between them.
#: 38 heads x 64 = 2432. **No `dual_attention_layers` key**, and no `attn2.*` tensors.
LARGE_TRANSFORMER_OVERRIDES: dict = {
    "num_layers": 38,
    "num_attention_heads": 38,
    "attention_head_dim": 64,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": 2,
    "joint_attention_dim": JOINT_ATTENTION_DIM,
    "caption_projection_dim": 2432,
    "pooled_projection_dim": 2048,
    "pos_embed_max_size": 192,
    "qk_norm": "rms_norm",
    "dual_attention_layers": (),
}

#: `text_encoder/config.json`: the ordinary CLIP ViT-L/14 text tower, not OpenCLIP.
#: Passed to `SD35ClipL` through `ModelConfig.text_encoder_overrides`, which is where
#: `prequantize._module_kwargs` looks when it builds this component on its own.
CLIP_L_OVERRIDES: dict = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "projection_dim": 768,
    "hidden_act": "quick_gelu",
    "vocab_size": 49408,
    "max_position_embeddings": 77,
    "layer_norm_eps": 1e-5,
}

#: `text_encoder_2/config.json`: OpenCLIP ViT-bigG/14. Note `gelu`, not `quick_gelu`
#: — the one activation difference between the two towers, and a silent one.
CLIP_G_OVERRIDES: dict = {
    "hidden_size": 1280,
    "num_hidden_layers": 32,
    "num_attention_heads": 20,
    "intermediate_size": 5120,
    "projection_dim": 1280,
    "hidden_act": "gelu",
    "vocab_size": 49408,
    "max_position_embeddings": 77,
    "layer_norm_eps": 1e-5,
}

#: Which layer's output the pipeline conditions on. `_get_clip_prompt_embeds` takes
#: `hidden_states[-2]` — the penultimate entry of the tuple HuggingFace returns, whose
#: first entry is the embedding output — i.e. the states after all but the last layer,
#: and before `final_layer_norm`. Off by one here is a subtly wrong image, not a crash.
CLIP_HIDDEN_STATE_INDEX = -2


def _model_config(alias: str, repo: str, transformer_overrides: dict):
    """An mflux `ModelConfig` for one SD 3.5 row, resolved through `_LOCAL_MODEL_CONFIGS`.

    `priority=999` keeps it out of mflux's own name resolution: these configs are
    reached by name from the catalogue, never by mflux guessing at an alias.
    """
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig(
        priority=999,
        aliases=[alias],
        model_name=repo,
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=NUM_TRAIN_TIMESTEPS,
        max_sequence_length=T5_MAX_SEQUENCE_LENGTH,
        supports_guidance=True,
        # The shift is applied by this package, from `SIGMA_SHIFT`, because mflux's own
        # path derives `mu` from the image sequence length and SD 3.5's is a constant.
        requires_sigma_shift=False,
        transformer_overrides=transformer_overrides,
        text_encoder_overrides=CLIP_L_OVERRIDES,
    )


def sd35_medium_model_config():
    """SD 3.5 Medium — 2.5B, the only row with MMDiT-X dual attention."""
    return _model_config("sd35-medium", MEDIUM_REPO, MEDIUM_TRANSFORMER_OVERRIDES)


def sd35_large_model_config():
    """SD 3.5 Large — 8.1B, real CFG."""
    return _model_config("sd35-large", LARGE_REPO, LARGE_TRANSFORMER_OVERRIDES)


def sd35_large_turbo_model_config():
    """SD 3.5 Large Turbo — the distilled 8B, four steps at guidance 1.0."""
    return _model_config("sd35-large-turbo", LARGE_TURBO_REPO, LARGE_TRANSFORMER_OVERRIDES)
