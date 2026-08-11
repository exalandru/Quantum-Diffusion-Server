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
import shutil
from pathlib import Path

from mflux_server import artifacts
from mflux_server import availability as av
from mflux_server.logs import SERVER_LOGGER, setup_logging
from mflux_server.settings import ENV_PREFIX

logger = logging.getLogger(f"{SERVER_LOGGER}.prequantize")

#: Enforced order: biggest first, so the HF cache can be purged between steps
#: and the disk peak stays bounded.
COMPONENT_ORDER = ("transformer", "text_encoder", "vae")

#: Measured sizes of each component, in GB: the bf16 download, and the 8-bit
#: result. From this module's own header and confirmed by the artifact on disk —
#: these describe *our* conversion, not an mflux table.
COMPONENT_SIZE_GB: dict[str, tuple[float, float]] = {
    "transformer": (64.5, 34.0),
    "text_encoder": (45.8, 24.0),
    "vae": (0.4, 0.4),
}

#: Slack over the computed peak. Conversion writes shards while the source is
#: still cached, and neither figure is exact.
DISK_MARGIN_GB = 10.0

#: Floor for a generic conversion. Not an estimate of the output — the catalogue
#: has no per-model size — just enough that one cannot begin on a full volume.
GENERIC_MIN_FREE_GB = 20.0


def required_free_gb(components: tuple[str, ...] | list[str]) -> float:
    """Free space this conversion needs, in GB.

    The peak is reached while one component's bf16 source sits on disk next to
    every 8-bit result written so far — the conversion order is largest-first for
    exactly that reason. **This assumes the bf16 source is purged from the
    HuggingFace cache between components**, which is what the script tells you to
    do; leaving all three cached instead needs roughly 169 GB rather than ~104.
    That assumption is the reason for a margin rather than a bare sum.
    """
    ordered = [name for name in COMPONENT_ORDER if name in components]
    written = 0.0
    peak = 0.0
    for name in ordered:
        source_gb, result_gb = COMPONENT_SIZE_GB[name]
        peak = max(peak, written + source_gb + result_gb)
        written += result_gb
    return round(peak + DISK_MARGIN_GB, 1)


def free_gb(path: Path) -> float:
    """Free space on the filesystem holding `path`, walking up to what exists."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return usage.free / 1e9


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


def convert_generic(spec, *, dest: Path, bits: int) -> tuple[str, ...]:
    """Convert any model mflux can save, by loading it and calling `save_model`.

    Deliberately not the FLUX.2-dev algorithm: that one exists because ~111 GB of
    bf16 cannot be materialised at once, and paying its complexity for a model
    that loads fine would be inventing work. `load_model` is QDS's own family
    resolution, so this adds no second dispatch table — the requested bit depth
    simply rides in on the spec, which is where `save_model` reads it from
    (`self.bits`).
    """
    from dataclasses import replace as _replace

    from mflux_server.registry import load_model

    logger.info(
        "Loading %s at %d bits — this holds the whole model in memory once",
        spec.key,
        bits,
        extra={
            "event": "prequantize_phase",
            "fields": {"phase": "loading", "model": spec.key, "bits": bits},
        },
    )
    model = load_model(_replace(spec, quantize=bits))

    logger.info(
        "Saving to %s",
        dest,
        extra={"event": "prequantize_phase", "fields": {"phase": "saving", "model": spec.key}},
    )
    model.save_model(str(dest))
    del model

    components = written_components(dest)
    logger.info(
        "Wrote %s (%.1f GB)",
        dest,
        _directory_size_gb(dest),
        extra={
            "event": "prequantize_phase",
            "fields": {"phase": "written", "components": list(components)},
        },
    )
    return components


def written_components(dest: Path) -> tuple[str, ...]:
    """Component directories the save actually produced.

    Read off the filesystem rather than from a family table: `ModelSaver` decides
    the subdirectories from each family's own `WeightDefinition`, and restating
    that mapping in QDS is precisely the drift this project keeps removing. What
    landed on disk is also what later validation has to check.
    """
    return tuple(
        sorted(
            child.name
            for child in dest.iterdir()
            if child.is_dir() and (child / av.INDEX_FILE).is_file()
        )
    )


def main() -> int:
    from mflux_server.registry import QUANTIZE_CHOICES

    parser = argparse.ArgumentParser(
        prog="mflux-server-prequantize",
        description="Convert a catalogue model into a saved, already-quantized MLX artifact.",
    )
    parser.add_argument(
        "--model",
        default="flux2-dev",
        help="catalogue key to convert (default: flux2-dev, this tool's original subject)",
    )
    # Defaults to the artifact layout; an explicit path stays supported so the
    # legacy FLUX.2-dev destination can still be targeted by hand.
    parser.add_argument("--dest", default=None, help="output directory")
    parser.add_argument("--repo", default=None, help="source repo (FLUX.2-dev only)")
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=QUANTIZE_CHOICES,
        help="quantization bit width; must be one of the model's published choices",
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
    # Same ordering constraint as `fetch.main`: before mflux pulls in
    # huggingface_hub and its frozen cache constant.
    from mflux_server.settings import load_settings

    logger.info("HuggingFace storage: %s", load_settings().apply_hf_home())
    return run_guarded(lambda: convert(args), what="conversion")


def convert(args) -> int:
    """Run one conversion, dispatching on the model's published strategy."""
    from mflux_server.registry import (
        BASE_SPECS_BY_KEY,
        STRATEGY_MFLUX_SAVE,
        STRATEGY_QDS_MEMORY_BOUNDED,
    )
    from mflux_server.settings import load_settings

    settings = load_settings()
    spec = settings.registry(include_disabled=True).get(args.model) or BASE_SPECS_BY_KEY.get(
        args.model
    )
    if spec is None:
        raise ValueError(f"Unknown model {args.model!r}. Valid keys: {sorted(BASE_SPECS_BY_KEY)}")

    capability = spec.quantization
    # The capability contract decides, here as in the UI. A model marked
    # unsupported has no path through this function at all.
    if not capability.supports_prequantize:
        raise ValueError(
            f"{spec.key} cannot be pre-quantized: {capability.note or 'not supported'}"
        )
    if args.bits not in capability.prequantize_choices:
        raise ValueError(
            f"{args.bits}-bit is not available for {spec.key}. "
            f"Published choices: {list(capability.prequantize_choices)}"
        )

    source = spec.model_path or spec.repo
    dest = (
        Path(args.dest).expanduser()
        if args.dest
        else artifacts.artifact_dir(spec.key, source, args.bits)
    )

    if capability.prequantize_strategy == STRATEGY_QDS_MEMORY_BOUNDED:
        components = _run_memory_bounded(args, spec, dest=dest, source=source)
        # A subset conversion is a legitimate way to work through FLUX.2-dev in
        # stages, but the result is only an artifact once every component is
        # there — otherwise `partial output != valid artifact` would not hold.
        expected: tuple[str, ...] = av.REQUIRED_COMPONENTS
    elif capability.prequantize_strategy == STRATEGY_MFLUX_SAVE:
        # `save_model` writes every component of the family's definition in one
        # call, or fails; so what landed is the whole set.
        components = expected = _run_generic(spec, dest=dest, bits=args.bits)
    else:  # pragma: no cover - guarded by the capability check above
        raise ValueError(f"No conversion strategy for {spec.key}")

    return _finish(
        dest,
        spec=spec,
        source=source,
        bits=args.bits,
        strategy=capability.prequantize_strategy,
        components=components,
        expected=expected,
    )


def _run_memory_bounded(args, spec, *, dest: Path, source: str) -> tuple[str, ...]:
    """FLUX.2-dev, unchanged: component by component, block by block."""
    needed = required_free_gb(args.components)
    available = free_gb(dest)
    if available < needed:
        raise InsufficientDisk(
            f"needs about {needed:.0f} GB free to convert {', '.join(args.components)}, "
            f"but only {available:.0f} GB is available on {dest}. That figure assumes the bf16 "
            f"source is purged from the HuggingFace cache between components; keeping all of it "
            f"cached needs roughly 169 GB."
        )
    logger.info(
        "Disk check: %.0f GB free, about %.0f GB required — %s",
        available,
        needed,
        dest,
        extra={
            "event": "prequantize_disk_check",
            "fields": {"free_gb": round(available, 1), "required_gb": needed},
        },
    )
    dest.mkdir(parents=True, exist_ok=True)
    for name in args.components:
        convert_component(name, repo=args.repo or source, dest=dest, bits=args.bits)
        logger.info(
            "The bf16 for '%s' is no longer needed: purge it from the HF cache before the next "
            "component (`hf cache delete`, or delete %s/ inside the snapshot) to bound the disk "
            "peak.\n",
            name,
            name,
        )
    return written_components(dest)


def _run_generic(spec, *, dest: Path, bits: int) -> tuple[str, ...]:
    """Every other supported family, through mflux's own saved-model support.

    No honest per-model output estimate exists here — the catalogue carries no
    size — so the check is a floor rather than a promise: enough room that a
    conversion cannot start on a volume that is already full.
    """
    available = free_gb(dest)
    if available < GENERIC_MIN_FREE_GB:
        raise InsufficientDisk(
            f"only {available:.0f} GB free where {dest} would be written. A saved copy of "
            f"{spec.key} is a large fraction of its source, and QDS has no exact figure for it, "
            f"so it refuses to start below {GENERIC_MIN_FREE_GB:.0f} GB."
        )
    dest.mkdir(parents=True, exist_ok=True)
    return convert_generic(spec, dest=dest, bits=bits)


def _finish(
    dest: Path,
    *,
    spec,
    source: str,
    bits: int,
    strategy: str,
    components: tuple[str, ...],
    expected: tuple[str, ...],
) -> int:
    """Validate what was written, then record completion — in that order.

    A process that exited zero is not evidence: the marker goes down only after
    the shards referenced by every index are on disk and the precision mflux
    stamped matches the one that was asked for.
    """
    if not components:
        logger.warning("Nothing was written to %s", dest)
        return 1

    missing = [name for name in expected if name not in components]
    if missing:
        logger.warning(
            "Converted %s, but %s still missing — no completion marker written",
            ", ".join(components),
            ", ".join(missing),
        )
        return 1

    ok, detail = artifacts.components_are_complete(dest, components)
    if not ok:
        logger.warning("Conversion incomplete at %s (%s); no completion marker written", dest, detail)
        return 1

    stored = artifacts.stored_bits(dest, components)
    if stored != bits:
        # Never record a precision the artifact does not actually carry.
        logger.error(
            "%s was asked for %d-bit but the saved weights declare %s; refusing to mark complete",
            dest,
            bits,
            stored,
            extra={
                "event": "job_failed",
                "fields": {"reason": "bits_mismatch", "requested": bits, "stored": stored},
            },
        )
        return 1

    artifacts.write_record(
        dest,
        model_key=spec.key,
        family=spec.family,
        source=source,
        bits=bits,
        strategy=strategy,
        components=components,
    )
    logger.info(
        "Done — %s: %.1f GB, %d-bit",
        dest,
        _directory_size_gb(dest),
        bits,
        extra={
            "event": "prequantize_done",
            "fields": {"dest": str(dest), "bits": bits, "components": list(components)},
        },
    )
    logger.info("Select it for %s in the Models tab to generate with it.", spec.key)
    return 0


class InsufficientDisk(RuntimeError):
    """Not enough free space to attempt the conversion."""


def run_guarded(action, *, what: str) -> int:
    """Same contract as `fetch.run_guarded`: name expected failures on the stream."""
    from mflux_server.fetch import run_guarded as guarded

    return guarded(action, what=what, log=logger)


if __name__ == "__main__":
    raise SystemExit(main())
