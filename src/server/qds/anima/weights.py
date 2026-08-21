"""Where Anima's weights live, and what their tensors are called here.

Anima is published twice and neither publication is sufficient on its own.

`circlestone-labs/Anima` holds the ComfyUI bundle: one file per component under
`split_files/`, and the only place the Aesthetic checkpoints exist at all. Its
DiT file carries the text adapter too, under a `llm_adapter.` prefix, so one
download yields two components.

`circlestone-labs/Anima-Base-v1.0-Diffusers` holds a plain diffusers layout. It
is used for the VAE and the tokenizers only: the ComfyUI VAE file renames every
module (`conv1`, `decoder.head.0`) so mflux's Qwen VAE mapping does not describe
it, and the ComfyUI bundle ships no tokenizer at all. Both of those components
are variant-independent -- a frozen Qwen-Image VAE and two vocabularies -- so
taking them from the Base release is a fact about what they are, not a
compromise. The transformer is never taken from there.

The name translation below was derived by putting the two publications' tensor
indexes side by side: 567 DiT tensors correspond one-to-one with matching shapes,
and the adapter's 118 are already identical. `tests/test_anima.py` checks it
against `tests/fixtures/anima_checkpoint_index.json`, which is a recording of the
real checkpoint's tensor index -- not a re-derivation from this table, which
could not fail for the case that matters.

Names this table does not know translate to `None`, and mflux applies weights
non-strictly, so an unrecognised checkpoint yields an empty component rather than
an error. `verify_loaded` at the bottom of this module is what turns that into a
refusal; see its docstring for the four published checkpoints it catches.
"""

from __future__ import annotations

import re
from dataclasses import replace

from qds.anima import config as anima_config

#: ComfyUI name -> the reference name this port uses, for the tensors that sit
#: outside the repeated blocks.
_TOP_LEVEL: dict[str, str] = {
    "x_embedder.proj.1.weight": "patch_embed.proj.weight",
    "t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
    "t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
    "t_embedding_norm.weight": "time_embed.norm.weight",
    "final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
    "final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
    "final_layer.linear.weight": "proj_out.weight",
}

#: The same, per transformer block. `self_attn`/`cross_attn` become `attn1`/`attn2`,
#: and the three adaLN pairs are ordered by the sub-block they modulate.
_PER_BLOCK: dict[str, str] = {
    "self_attn.q_proj": "attn1.to_q",
    "self_attn.k_proj": "attn1.to_k",
    "self_attn.v_proj": "attn1.to_v",
    "self_attn.output_proj": "attn1.to_out.0",
    "self_attn.q_norm": "attn1.norm_q",
    "self_attn.k_norm": "attn1.norm_k",
    "cross_attn.q_proj": "attn2.to_q",
    "cross_attn.k_proj": "attn2.to_k",
    "cross_attn.v_proj": "attn2.to_v",
    "cross_attn.output_proj": "attn2.to_out.0",
    "cross_attn.q_norm": "attn2.norm_q",
    "cross_attn.k_norm": "attn2.norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
}

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)\.(weight|bias)$")


def transformer_key(key: str) -> str | None:
    """Translate one DiT tensor name, or `None` if this key is not the DiT's.

    Returning `None` is how the adapter's tensors are dropped from the
    transformer component: both live in one file, and each component reads only
    the half that belongs to it.
    """
    if key.startswith(anima_config.ADAPTER_PREFIX):
        return None
    if not key.startswith(anima_config.CHECKPOINT_PREFIX):
        return None
    key = key[len(anima_config.CHECKPOINT_PREFIX) :]
    if key in _TOP_LEVEL:
        return _TOP_LEVEL[key]
    match = _BLOCK_RE.match(key)
    if match is None:
        return None
    index, stem, suffix = match.groups()
    target = _PER_BLOCK.get(stem)
    if target is None:
        return None
    return f"transformer_blocks.{index}.{target}.{suffix}"


def conditioner_key(key: str) -> str | None:
    """Strip the adapter prefix. Beyond that the names already agree."""
    if not key.startswith(anima_config.ADAPTER_PREFIX):
        return None
    return key[len(anima_config.ADAPTER_PREFIX) :]


def text_encoder_key(key: str) -> str | None:
    """Qwen3-0.6B as HuggingFace writes it: a `model.` prefix and no `lm_head`."""
    if key.startswith("model."):
        return key[len("model.") :]
    return None


def anima_weight_definition(weight_file: str):
    """`AnimaWeightDefinition` bound to one of the published checkpoints.

    The variants share everything except which file the transformer and its
    adapter are read from, so the definition is generated rather than duplicated:
    one description of the model, parameterised by the checkpoint a catalogue row
    selected.
    """
    if weight_file not in anima_config.SUPPORTED_WEIGHT_FILES:
        raise ValueError(
            f"{weight_file!r} is not a checkpoint this package translates. "
            f"Known: {list(anima_config.SUPPORTED_WEIGHT_FILES)}."
        )

    class _Definition(AnimaWeightDefinition):
        @staticmethod
        def get_components() -> list:
            return [
                replace(component, weight_files=[weight_file])
                if component.name in ("transformer", "text_conditioner")
                else component
                for component in AnimaWeightDefinition.get_components()
            ]

        @staticmethod
        def get_download_patterns(model_name: str | None = None) -> list[str]:
            return [
                f"split_files/diffusion_models/{weight_file}",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
            ]

    _Definition.__name__ = "Anima" + weight_file.split(".")[0].title().replace("-", "") + "Weights"
    return _Definition


class AnimaWeightDefinition:
    """The four components, and the two repositories they come from.

    Describes the model as a whole, at the default checkpoint. A catalogue row
    reaches its own variant through `anima_weight_definition`.
    """

    @staticmethod
    def get_components() -> list:
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
        from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="split_files/diffusion_models",
                loading_mode="mlx_native",
                weight_files=[anima_config.DEFAULT_WEIGHT_FILE],
                precision=ModelConfig.precision,
                mapping_getter=None,
                key_transform=transformer_key,
            ),
            ComponentDefinition(
                # The same file as the transformer. mflux caches the raw read per
                # (path, mode, files), so naming it twice costs one load, not two.
                name="text_conditioner",
                hf_subdir="split_files/diffusion_models",
                loading_mode="mlx_native",
                weight_files=[anima_config.DEFAULT_WEIGHT_FILE],
                precision=ModelConfig.precision,
                mapping_getter=None,
                key_transform=conditioner_key,
                # Six blocks standing between the text encoder and every
                # cross-attention in the model: quantizing it buys ~12 MB and
                # risks the conditioning, which is the same trade Qwen and Krea
                # decline for their encoders.
                skip_quantization=True,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="split_files/text_encoders",
                loading_mode="mlx_native",
                weight_files=["qwen_3_06b_base.safetensors"],
                precision=ModelConfig.precision,
                mapping_getter=None,
                key_transform=text_encoder_key,
                skip_quantization=True,
            ),
            ComponentDefinition(
                # From the diffusers repository, so mflux's own Qwen VAE mapping
                # applies unchanged.
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=QwenWeightMapping.get_vae_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> list:
        from mflux.models.common.tokenizer import LanguageTokenizer
        from mflux.models.common.weights.loading.weight_definition import TokenizerDefinition

        return [
            TokenizerDefinition(
                # No chat template: the pipeline encodes the prompt raw
                # (`diffusers/modular_pipelines/anima/encoders.py`), so applying
                # one would prepend tokens the model never saw in training.
                name="qwen3",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=anima_config.MAX_SEQUENCE_LENGTH,
                padding="longest",
                download_patterns=["tokenizer/**"],
            ),
            TokenizerDefinition(
                # The T5 vocabulary indexes the adapter's own learned embedding
                # table. No T5 encoder is involved, and none is downloaded.
                name="t5",
                hf_subdir="t5_tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=anima_config.MAX_SEQUENCE_LENGTH,
                padding="longest",
                download_patterns=["t5_tokenizer/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns(model_name: str | None = None) -> list[str]:
        """Exactly the one checkpoint in use.

        `split_files/diffusion_models/` holds eight of them, ~25 GB together. A
        pattern over the directory would fetch all eight to run one.
        """
        return [
            f"split_files/diffusion_models/{anima_config.DEFAULT_WEIGHT_FILE}",
            "split_files/text_encoders/qwen_3_06b_base.safetensors",
        ]

    @staticmethod
    def get_companion_download_patterns() -> list[str]:
        """What the diffusers repository supplies: the VAE and both tokenizers."""
        return ["vae/*.safetensors", "vae/*.json", "tokenizer/**", "t5_tokenizer/**"]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        """MLX quantizes in groups of 64 along the last axis; skip what will not divide.

        Same guard Krea 2 applies, and for the same reason: a layer whose input
        dimension is not a multiple of 64 raises rather than quantizing.
        """
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        return weight is not None and weight.shape[-1] % 64 == 0


def component_subset(names: tuple[str, ...], definition=None):
    """`AnimaWeightDefinition` narrowed to the components one repository holds.

    `WeightLoader.load` reads every component of the definition it is handed
    against the single path it is handed, so the full definition can only be used
    where every component is present -- which, for Anima, is nowhere. Splitting
    it here keeps `AnimaWeightDefinition` the one description of the model, with
    these views deriving from it rather than restating it.

    The download patterns narrow with the components, so asking for the VAE does
    not also pull 4 GB of transformer, and asking for the transformer does not
    reach for a `vae/` directory the ComfyUI bundle does not have.
    """
    definition = definition or AnimaWeightDefinition
    missing = set(names) - {c.name for c in definition.get_components()}
    if missing:
        raise ValueError(f"Unknown Anima components: {sorted(missing)}")

    companion = set(names) - {"transformer", "text_conditioner", "text_encoder"}

    class _Subset:
        @staticmethod
        def get_components() -> list:
            return [c for c in definition.get_components() if c.name in names]

        @staticmethod
        def get_tokenizers() -> list:
            # Tokenizers are loaded separately, from the companion repository, so
            # no subset claims them: a definition that did would have the loader
            # look for them beside the weights.
            return []

        @staticmethod
        def get_download_patterns(model_name: str | None = None) -> list[str]:
            if companion:
                return AnimaWeightDefinition.get_companion_download_patterns()
            return definition.get_download_patterns(model_name)

        quantization_predicate = staticmethod(definition.quantization_predicate)

    _Subset.__name__ = "Anima" + "".join(n.title().replace("_", "") for n in names) + "Only"
    return _Subset


#: Parameters a module builds for itself rather than reading from a checkpoint,
#: per component. mflux's Qwen3 tower derives its rotary frequencies in
#: `__init__` from `rope_theta` and `head_dim`; HuggingFace marks the same buffer
#: non-persistent, so no published file carries it.
_DERIVED_PARAMETERS: dict[str, int] = {"text_encoder": 1}


def verify_loaded(components: dict, modules: dict) -> None:
    """Refuse a checkpoint whose tensors did not reach the modules they are for.

    Every component here is read through a `key_transform` that returns `None`
    for names it does not recognise, and mflux applies weights non-strictly. Those
    two facts compose badly: a checkpoint whose names this package does not know
    yields an *empty* component, applies cleanly, and leaves the module with the
    random initialisation it was constructed with. The server then generates
    noise, with no error anywhere to explain it.

    That is not hypothetical. `split_files/diffusion_models/` holds eight
    checkpoints, and four of them -- the Base and the three preview releases --
    root their tensors at `net.` rather than `model.diffusion_model.`, so this
    package's translation maps none of them. Selecting one through
    `config.DEFAULT_WEIGHT_FILE` is a documented option, which is exactly why it
    has to fail loudly rather than quietly.

    Counting against the constructed module rather than against a recorded number
    keeps this honest as the port changes: the question asked is "did every
    parameter get a value", which is the property that actually matters.
    """
    from mlx.utils import tree_flatten

    for name, module in modules.items():
        loaded = components.get(name)
        expected = len(tree_flatten(module.parameters())) - _DERIVED_PARAMETERS.get(name, 0)
        found = len(tree_flatten(loaded)) if loaded else 0
        if found != expected:
            raise ValueError(
                f"Anima's {name!r} received {found} tensors but needs {expected}. "
                "The checkpoint's tensor names are not the ones this package "
                "translates — check `config.DEFAULT_WEIGHT_FILE`: of the eight "
                "checkpoints published under `split_files/diffusion_models/`, only "
                "the Turbo and Aesthetic releases use the supported naming."
            )
