"""Where SD 3.5's weights live, and what their tensors are called here.

All five components come from one repository, in the plain diffusers layout every
`stabilityai/stable-diffusion-3.5-*` release publishes: one subdirectory per component,
each with its own `config.json` and safetensors. That is also the layout QDS's
component-wise converter reads and writes, which is why this family can be
pre-quantized at all — the component name, the source subdirectory and the artifact
subdirectory are all the same string.

Three of the five need no rename table. `transformer`, `text_encoder` and
`text_encoder_2` are loaded in passthrough mode because this package's module names
*are* the checkpoint's — `tests/test_sd35_weights.py` asserts that against the recorded
tensor index rather than trusting it. The other two are mflux's own modules and so use
mflux's own mappings:

* `text_encoder_3` is `mflux...t5_encoder.T5Encoder`, whose hard-coded shape is exactly
  SD 3.5's T5-XXL, reached through `FluxWeightMapping.get_t5_encoder_mapping`. That
  mapping broadcasts block 0's `relative_attention_bias` to all 24 blocks, which is
  what T5 does anyway — the table is computed once and shared — and is why mflux's
  per-block copy of it is filled rather than left random;
* `vae` is `SD35VAE`, mflux's FLUX.1 autoencoder with SD 3.5's two normalisation
  constants, reached through `FluxWeightMapping.get_vae_mapping`.

**`weight_files` is pinned on the three text encoders and that is load-bearing.** Each
of those subdirectories ships the weights twice — `model.safetensors` beside
`model.fp16.safetensors`, and for T5 a second pair of shards — and mflux's
`mlx_native` mode globs `*.safetensors` when no files are named. Left unpinned, loading
T5 would read 19 GB to use 9.5 of it. The download patterns name the same files for
the same reason.

Weights are applied non-strictly, so a component whose names this package did not
recognise would arrive empty, apply cleanly, and leave a randomly-initialised module
behind. `verify_loaded` at the bottom of this module is what turns that into a refusal.
"""

from __future__ import annotations

import mlx.core as mx

from qds.sd35 import config as sd35_config

#: The one tensor whose layout differs between torch and MLX. Torch writes a 2D
#: convolution kernel as `[out, in, kh, kw]`; MLX convolutions are NHWC and want
#: `[out, kh, kw, in]`. Everything else in this model is a `Linear`, an `Embedding`,
#: a norm or the positional table, all of which agree.
_CONV_WEIGHT_KEYS = frozenset({"pos_embed.proj.weight"})

#: The exact files the three text encoders are read from, out of the several each of
#: those subdirectories publishes. See the module docstring: the fp16 duplicates are
#: the reason. `transformer/` and `vae/` need no such pin — they publish their weights
#: once, so globbing the directory reads Medium's single file and the large releases'
#: two shards without this class having to know which release it is looking at.
CLIP_L_WEIGHT_FILES = ["model.safetensors"]
CLIP_G_WEIGHT_FILES = ["model.safetensors"]
T5_WEIGHT_FILES = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def transformer_weight_transform(key: str, tensor: mx.array) -> mx.array:
    """Torch's conv layout to MLX's, for the one convolution in the transformer."""
    if key in _CONV_WEIGHT_KEYS and tensor.ndim == 4:
        return tensor.transpose(0, 2, 3, 1)
    return tensor


class SD35WeightDefinition:
    """The five components and three tokenizers, for any of the three releases.

    One definition rather than one per row, and that is a property of the publication
    rather than a simplification: the three releases differ in the *shape* of the
    transformer, which `ModelConfig.transformer_overrides` carries, and in how many
    shards it is written to, which the directory itself answers. Nothing here has to
    know which release it is describing.

    That also keeps `registry.family_structure(family)` honest. It dispatches on the
    family alone, so a definition that needed the repository could not be reached
    through it — and the converter reaches every definition through it.
    """

    @staticmethod
    def get_components() -> list:
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

        return [
            # Conversion order is `components.py`'s, not this list's; this one is the
            # model's own order, largest first, which happens to agree.
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                # Unpinned: this directory holds one release's weights and nothing
                # else, so mflux's glob reads Medium's single 4.94 GB file or the large
                # releases' two shards, whichever is on disk.
                weight_files=None,
                # No mapping: `SD35Transformer`'s parameter paths are the checkpoint's
                # tensor names, one for one. Proven in `test_sd35_weights.py` against
                # the recorded index, so this is a checked claim, not a convention.
                mapping_getter=None,
                weight_transform=transformer_weight_transform,
            ),
            ComponentDefinition(
                name="text_encoder_3",
                hf_subdir="text_encoder_3",
                precision=ModelConfig.precision,
                weight_files=T5_WEIGHT_FILES,
                # 24, explicitly: `WeightMapper` detects block counts from
                # `transformer_blocks.N`, which T5's `encoder.block.N` does not match,
                # and its fallback is 4 — which would silently load a sixth of the model.
                num_blocks=24,
                mapping_getter=FluxWeightMapping.get_t5_encoder_mapping,
            ),
            ComponentDefinition(
                name="text_encoder_2",
                hf_subdir="text_encoder_2",
                precision=ModelConfig.precision,
                weight_files=CLIP_G_WEIGHT_FILES,
                mapping_getter=None,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                weight_files=CLIP_L_WEIGHT_FILES,
                mapping_getter=None,
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=FluxWeightMapping.get_vae_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> list:
        from mflux.models.common.tokenizer import LanguageTokenizer
        from mflux.models.common.weights.loading.weight_definition import TokenizerDefinition

        return [
            TokenizerDefinition(
                name="clip_l",
                hf_subdir="tokenizer",
                tokenizer_class="CLIPTokenizer",
                encoder_class=LanguageTokenizer,
                # Both CLIP towers are trained with 77 positions and the pipeline pads
                # to all of them: the pooled vector is read at the end-of-text token,
                # and truncation shorter than 77 would move it.
                max_length=sd35_config.CLIP_MAX_SEQUENCE_LENGTH,
                download_patterns=["tokenizer/**"],
            ),
            TokenizerDefinition(
                name="clip_g",
                hf_subdir="tokenizer_2",
                tokenizer_class="CLIPTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=sd35_config.CLIP_MAX_SEQUENCE_LENGTH,
                download_patterns=["tokenizer_2/**"],
            ),
            TokenizerDefinition(
                name="t5",
                hf_subdir="tokenizer_3",
                tokenizer_class="T5Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=sd35_config.T5_MAX_SEQUENCE_LENGTH,
                download_patterns=["tokenizer_3/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns(model_name: str | None = None) -> list[str]:
        """Named files for the text encoders, directory globs for the rest.

        `text_encoder*/` each publish their weights twice — a bf16 copy and an fp16
        copy of the same tensors — so a `text_encoder_3/*.safetensors` pattern would
        fetch 19 GB to load 9.5 of them. Those three are named file by file.

        `transformer/` and `vae/` are globbed instead, and that is what lets one
        pattern list serve all three releases: Medium's transformer is a single file
        and the large pair's is two shards plus an index, and the glob is right for
        both without this class being told which release it is fetching.

        The repository-root `sd3.5_*.safetensors` single-file checkpoints and the
        `text_encoders/` ComfyUI bundle are matched by nothing here, deliberately:
        together they are another ~30 GB of weights this package does not read.
        """
        patterns = [
            "model_index.json",
            "scheduler/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "vae/*.safetensors",
            "vae/*.json",
        ]
        for subdir, files in (
            ("text_encoder", CLIP_L_WEIGHT_FILES),
            ("text_encoder_2", CLIP_G_WEIGHT_FILES),
            ("text_encoder_3", T5_WEIGHT_FILES),
        ):
            patterns.append(f"{subdir}/config.json")
            patterns += [f"{subdir}/{name}" for name in files]
        # T5 is sharded, and mflux reads the index to decide what a saved model is
        # made of; without it a resumed download cannot be told from a truncated one.
        patterns.append("text_encoder_3/model.safetensors.index.json")
        patterns += ["tokenizer/**", "tokenizer_2/**", "tokenizer_3/**"]
        return patterns

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        """MLX quantizes in groups of 64 along the last axis; skip what will not divide.

        The same guard Krea 2 and Anima apply, and for the same reason: a layer whose
        input dimension is not a multiple of 64 raises rather than quantizing. SD 3.5
        has one such layer per transformer — `proj_out`, whose 64-wide output is fine
        but whose siblings are not uniform — so this is a real filter, not a formality.
        """
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        return weight is not None and weight.shape[-1] % 64 == 0


def verify_loaded(components: dict, modules: dict) -> None:
    """Refuse a checkpoint whose tensors did not reach the modules they are for.

    mflux applies weights non-strictly, and three of these five components are loaded
    in passthrough mode — so a repository whose tensor names this package does not
    recognise would produce an *empty* component, apply cleanly, and leave the module
    with the random initialisation it was constructed with. The server would then
    generate noise, with no error anywhere to explain it.

    That is not hypothetical for this family. Medium and the large releases have
    different transformer shapes and different shard file names, and a row bound to the
    wrong `sd35_weight_definition` would find no files at all — or, worse, a directory
    imported by hand from a single-file `sd3.5_*.safetensors` checkpoint, whose tensors
    are named nothing like the diffusers layout.

    Counting against the constructed module rather than against a recorded number keeps
    this honest as the port changes: the question asked is "did every parameter get a
    value", which is the property that actually matters.
    """
    from mlx.utils import tree_flatten

    for name, module in modules.items():
        loaded = components.get(name)
        expected = len(tree_flatten(module.parameters()))
        found = len(tree_flatten(loaded)) if loaded else 0
        if found != expected:
            raise ValueError(
                f"SD 3.5's {name!r} received {found} tensors but needs {expected}. "
                "The source's tensor names are not the ones this package reads — check "
                "that the model path is a diffusers-layout stable-diffusion-3.5 "
                "repository and that the catalogue row matches the release on disk "
                "(Medium and the large variants are different shapes)."
            )
