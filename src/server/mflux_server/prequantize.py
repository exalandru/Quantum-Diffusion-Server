"""Pre-quantization of FLUX.2-dev into a local MLX artifact.

The upstream repo ships bf16: a 64.5 GB transformer plus a 45.8 GB text encoder
plus the VAE. That is ~111 GB of resident weights, impossible on 96 GB of
unified memory. At 8 bits we drop to ~58 GB, comfortably within reach — but
quantizing on the fly requires holding the bf16 in memory first. Hence this
one-time conversion.

Two precautions, each bounding a different resource:

* **one component at a time.** `WeightApplier.apply_and_quantize` loads every
  component in bf16 before quantizing anything, so we go through mflux's
  "single" APIs and release between each.
* **block-by-block quantization.** A single `nn.quantize` over the whole
  transformer makes 64.5 GB of bf16 coexist with 34 GB of 8-bit, i.e. ~96 GB.
  Handling one block at a time and evaluating brings the peak down to ~66 GB.

The default order (transformer, then encoder, then VAE) lets you purge the HF
cache between components: the disk peak falls from ~169 GB to ~97 GB.

Reloading needs no code at all: `WeightLoader._load_component` tries
`_try_load_mflux_format` first, reads the `quantization_level` written here into
the safetensors metadata, and `WeightApplier` quantizes the structure before
applying the weights.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path

from mflux_server.logs import SERVER_LOGGER, setup_logging
from mflux_server.settings import ENV_PREFIX

logger = logging.getLogger(f"{SERVER_LOGGER}.prequantize")

#: Enforced order: biggest first, so the HF cache can be purged between steps
#: and the disk peak stays bounded.
COMPONENT_ORDER = ("transformer", "text_encoder", "vae")


def _build_module(name: str, model_config):
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

    from mflux_server.flux2_dev.mistral3 import Mistral3TextEncoder

    if name == "transformer":
        return Flux2Transformer(**model_config.transformer_overrides)
    if name == "text_encoder":
        return Mistral3TextEncoder(**model_config.text_encoder_overrides)
    if name == "vae":
        return Flux2VAE()
    raise ValueError(f"Unknown component: {name!r}")


def _quantization_units(module) -> list:
    """Submodules to quantize separately, to bound the memory peak.

    `transformer_blocks` / `single_transformer_blocks` for the transformer,
    `layers` for the text encoder. The VAE has none: it is small enough for the
    global pass.
    """
    units: list = []
    for attr in ("transformer_blocks", "single_transformer_blocks", "layers"):
        units.extend(getattr(module, attr, None) or [])
    return units


def _quantize_incrementally(module, *, bits: int, predicate) -> None:
    import mlx.core as mx
    from mlx import nn

    units = _quantization_units(module)
    for index, unit in enumerate(units, start=1):
        nn.quantize(unit, class_predicate=predicate, bits=bits)
        mx.eval(unit.parameters())
        mx.clear_cache()
        if index % 8 == 0 or index == len(units):
            logger.info(
                "  quantized %d/%d blocks — %s",
                index,
                len(units),
                _memory(),
                extra={
                    "event": "prequantize_progress",
                    "fields": {"block": index, "blocks": len(units)},
                },
            )

    # Final pass for the head layers (embeddings, projections, modulated
    # norms). No effect on what is already quantized: MLX's `Quantized*` modules
    # do not expose `to_quantized`, so mflux's predicate skips them.
    nn.quantize(module, class_predicate=predicate, bits=bits)
    mx.eval(module.parameters())
    mx.clear_cache()


def _memory() -> str:
    import mlx.core as mx

    return f"mlx active {mx.get_active_memory() / 1e9:.1f} GB, peak {mx.get_peak_memory() / 1e9:.1f} GB"


def _directory_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def convert_component(name: str, *, repo: str, dest: Path, bits: int) -> None:
    import mlx.core as mx
    from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.saving.model_saver import ModelSaver

    from mflux_server.flux2_dev import flux2_dev_model_config, single_component_definition
    from mflux_server.flux2_dev.weights import Flux2DevWeightDefinition

    definition = single_component_definition(name)
    component = definition.get_components()[0]
    model_config = flux2_dev_model_config()

    logger.info("── %s ──────────────────────────────────────", name)
    logger.info(
        "Downloading / reading %s/%s",
        repo,
        component.hf_subdir,
        extra={
            "event": "prequantize_component_start",
            "fields": {"component": name, "repo": repo, "bits": bits},
        },
    )
    # `load` rather than `load_single`: it goes through `PathResolution`, so
    # `--repo` also accepts a local directory, and the patterns come from the
    # single-component definition — only the intended subfolder is downloaded.
    weights = WeightLoader.load(weight_definition=definition, model_path=repo)

    resolved_bits, warning = QuantizationResolution.resolve(
        stored=weights.meta_data.quantization_level,
        requested=bits,
    )
    if warning:
        logger.warning(warning)
    if resolved_bits is None:
        raise ValueError(f"No quantization resolved for {name} (bits={bits!r})")

    module = _build_module(name, model_config)
    module.update(weights.components[component.name], strict=False)
    # Drop the loader's reference before quantizing: otherwise the bf16 arrays
    # stay alive for the whole conversion.
    weights.components.clear()
    del weights
    gc.collect()
    mx.eval(module.parameters())
    logger.info("bf16 weights applied — %s", _memory())

    _quantize_incrementally(
        module,
        bits=resolved_bits,
        predicate=Flux2DevWeightDefinition.quantization_predicate,
    )
    logger.info("Quantized to %d bits — %s", resolved_bits, _memory())

    shim = _ComponentShim(name, module)
    if definition.get_tokenizers():
        shim.tokenizers = TokenizerLoader.load_all(
            definitions=definition.get_tokenizers(),
            model_path=repo,
        )

    ModelSaver.save_model(
        model=shim,
        bits=resolved_bits,
        base_path=str(dest),
        weight_definition=definition,
    )

    written = _directory_size_gb(dest / component.hf_subdir)
    logger.info(
        "Written to %s (%.1f GB)",
        dest / component.hf_subdir,
        written,
        extra={
            "event": "prequantize_component_done",
            "fields": {"component": name, "bits": resolved_bits, "written_gb": round(written, 2)},
        },
    )

    del shim, module
    gc.collect()
    mx.clear_cache()
    logger.info("Released — %s", _memory())


class _ComponentShim:
    """Minimal attribute holder for `ModelSaver.save_model`.

    The saver reads `getattr(model, component.name)` and, when present,
    `model.tokenizers` — no need for the full model.
    """

    def __init__(self, name: str, module) -> None:
        setattr(self, name, module)


def main() -> int:
    from mflux_server.flux2_dev import DEFAULT_MODEL_PATH, REPO
    from mflux_server.registry import QUANTIZE_CHOICES

    parser = argparse.ArgumentParser(
        prog="mflux-server-prequantize",
        description="Convert black-forest-labs/FLUX.2-dev into a quantized MLX artifact, "
        "one component at a time so it fits in unified memory.",
    )
    parser.add_argument("--dest", default=DEFAULT_MODEL_PATH, help="output directory")
    parser.add_argument("--repo", default=REPO, help="source repo")
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=QUANTIZE_CHOICES,
        help="quantization bit width",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(COMPONENT_ORDER),
        choices=list(COMPONENT_ORDER),
        help="components to convert, in the given order (default: all, largest first)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        default=os.environ.get(f"{ENV_PREFIX}LOG_JSON", "").lower() in {"1", "true", "yes"},
        help="one line, one JSON object, so a supervisor can follow along",
    )
    args = parser.parse_args()

    # Same configuration as the server, so the desktop app follows the
    # conversion exactly as it follows a generation.
    setup_logging(level="INFO", log_file=None, json_lines=args.json_logs)
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    for name in args.components:
        convert_component(name, repo=args.repo, dest=dest, bits=args.bits)
        logger.info(
            "The bf16 for '%s' is no longer needed: purge it from the HF cache before the next "
            "component (`hf cache delete`, or delete %s/ inside the snapshot) to bound the disk peak.\n",
            name,
            name,
        )

    logger.info("Done — %s: %.1f GB", dest, _directory_size_gb(dest))
    logger.info(
        "If this path differs from the default, set it in server-config.json (models.flux2-dev.model_path)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
