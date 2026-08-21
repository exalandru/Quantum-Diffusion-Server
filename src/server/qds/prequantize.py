"""Pre-quantization into a local MLX artifact, one component at a time.

The technique was built for FLUX.2-dev, whose upstream repo ships bf16: a 64.5 GB
transformer plus a 45.8 GB text encoder plus the VAE. That is ~111 GB of resident
weights, impossible on 96 GB of unified memory. At 8 bits we drop to ~58 GB,
comfortably within reach — but quantizing on the fly requires holding the bf16 in
memory first. Hence this one-time conversion.

Three precautions, each bounding a different resource:

* **one component at a time.** `WeightApplier.apply_and_quantize` loads every
  component in bf16 before quantizing anything, so we go through mflux's
  "single" APIs and release between each.
* **block-by-block quantization.** A single `nn.quantize` over the whole
  transformer makes 64.5 GB of bf16 coexist with 34 GB of 8-bit, i.e. ~96 GB.
  Handling one block at a time and evaluating brings the peak down to ~66 GB.
* **no eager evaluation of the source weights.** MLX is lazy, and staying lazy
  between `update` and the block loop is what lets each block's source be
  materialised only when that block is quantized. Measured on z-image's
  transformer at 4 bits: 12.41 GB peak with an `mx.eval` after `update`, 4.68 GB
  without, byte-identical output.

The default order (transformer, then encoder, then VAE) lets you purge the HF
cache between components: the disk peak falls from ~169 GB to ~97 GB.

**Every supported family now takes this route**, not only FLUX.2-dev. The generic
alternative was `model.save_model(dest)` after `load_model`, which is one line and
holds the entire model resident to write it — so converting a 20 GB model peaked
at the size of the whole thing when no single component of it is more than half
that. Nothing about that was specific to being small; it was specific to nobody
having generalised the FLUX.2-dev path. What made generalising it possible is that
mflux writes a model as one directory per component (`ModelSaver.save_model` walks
`weight_definition.get_components()` and skips what the object does not carry), so
an artifact can be assembled across several runs — and even across several
processes.

What is family-specific is which module class each component is, and that comes
from the family's own variant class (its annotations) and `ModelConfig`, resolved
through `registry.family_structure`. No model maths is reimplemented here.

Reloading needs no code at all: `WeightLoader._load_component` tries
`_try_load_mflux_format` first, reads the `quantization_level` written here into
the safetensors metadata, and `WeightApplier` quantizes the structure before
applying the weights.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import logging
import os
import shutil
from pathlib import Path

from qds import artifacts, env
from qds import availability as av
from qds import components as comp
from qds.logs import SERVER_LOGGER, setup_logging

logger = logging.getLogger(f"{SERVER_LOGGER}.prequantize")

#: FLUX.2-dev's order, kept as a name because its disk arithmetic below is
#: written in terms of it. Every family's order — biggest first, for the same
#: reason — is `components.components_for(family)`.
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


def _module_class(variant_class, name: str):
    """The module class a family uses for one component.

    Read off the variant class's own annotations — `ZImage.transformer:
    ZImageTransformer` — which is where every family already states it, and which
    `test_components.py` checks stays true. The alternative was a second table
    naming three classes per family, i.e. exactly the kind of restatement that
    goes stale silently when mflux changes a class.

    `get_type_hints` rather than `__annotations__`: a module with `from __future__
    import annotations` (QDS's own `Flux2Dev`) stores them as strings.
    """
    from typing import get_type_hints

    try:
        hints = get_type_hints(variant_class)
    except Exception:  # pragma: no cover - defensive; falls back to raw annotations
        hints = dict(getattr(variant_class, "__annotations__", {}))
    target = hints.get(name)
    if not isinstance(target, type):
        raise ValueError(
            f"{variant_class.__name__} does not declare a module class for component {name!r}, "
            f"so it cannot be converted on its own."
        )
    return target


def _module_kwargs(model_config, name: str) -> dict:
    """Constructor overrides for one component, from the model's own configuration.

    Which model a family's class is building is carried by `ModelConfig`:
    `flux2` and `ernie` size their transformer and text encoder from
    `transformer_overrides` / `text_encoder_overrides`, and building those with
    the class defaults instead would silently produce a differently shaped module
    — one whose `update(..., strict=False)` then leaves tensors unassigned. Every
    initializer in mflux reads exactly these two attributes, so this mirrors them
    rather than inventing a convention.
    """
    if name == "transformer":
        return dict(getattr(model_config, "transformer_overrides", None) or {})
    if name == "text_encoder":
        return dict(getattr(model_config, "text_encoder_overrides", None) or {})
    return {}


def _build_module(variant_class, model_config, name: str):
    return _module_class(variant_class, name)(**_module_kwargs(model_config, name))


def single_component_definition(definition, name: str, *, with_tokenizers: bool):
    """A weight definition exposing one component, for a one-component run.

    `WeightLoader.load` and `ModelSaver.save_model` both walk *all* of a
    definition's components; handing them a definition that has one is what keeps
    the other components off the machine's memory and out of the download.

    The download patterns are narrowed to that component's own subdirectory for
    the same reason — a run converting the VAE must not pull 64 GB of transformer
    onto the disk first.
    """
    from mflux.models.common.weights.loading.weight_definition import (  # noqa: F401
        ComponentDefinition,
        TokenizerDefinition,
    )

    by_name = {c.name: c for c in definition.get_components()}
    if name not in by_name:
        raise ValueError(f"Unknown component: {name!r}. Valid: {sorted(by_name)}")
    component = by_name[name]
    tokenizers = list(definition.get_tokenizers()) if with_tokenizers else []

    class _SingleComponentDefinition:
        @staticmethod
        def get_components():
            return [component]

        @staticmethod
        def get_tokenizers():
            return tokenizers

        @staticmethod
        def get_download_patterns():
            patterns = [
                f"{component.hf_subdir}/*.safetensors",
                f"{component.hf_subdir}/*.json",
            ]
            for tokenizer in tokenizers:
                patterns.extend(tokenizer.download_patterns or [f"{tokenizer.hf_subdir}/**"])
            return patterns

        quantization_predicate = staticmethod(definition.quantization_predicate)

    _SingleComponentDefinition.__name__ = (
        f"{definition.__name__}{name.title().replace('_', '')}Only"
    )
    return _SingleComponentDefinition


def tokenizers_present(dest: Path, definition) -> bool:
    """Whether the artifact already carries every tokenizer the family declares.

    Tokenizers are small, and they are not components — nothing quantizes them —
    but an artifact without them cannot be loaded, so a run writes them whenever
    they are absent rather than only when the text encoder happens to be the
    component being converted.
    """
    return all(
        (dest / tokenizer.hf_subdir).is_dir() and any((dest / tokenizer.hf_subdir).iterdir())
        for tokenizer in definition.get_tokenizers()
    )


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


def _predicate_for_bits(predicate, bits: int | None):
    """Bind the requested bit depth into a family's quantization predicate.

    A family predicate takes either `(path, module)` or `(path, module, bits)`.
    The three-argument form is how a family varies precision per layer with the
    level asked for: Qwen keeps `.img_mod_linear` at 8-bit when the rest goes to
    4, because that is where 4-bit error compounds along the denoising
    trajectory. `nn.quantize` only ever calls a predicate with two arguments, so
    the third has to be bound here — and a three-argument predicate declares its
    `bits` parameter with a default, which means calling it blind loses the
    per-layer decision silently rather than raising.

    This mirrors `WeightApplier._predicate_with_bits`, which is what mflux
    applies on the load path. The two must agree: a prequantized artifact whose
    layer precisions disagree with the ones mflux would have chosen is an
    artifact mflux cannot rebuild faithfully. `test_prequantize_predicate.py`
    holds them to that, against the real family predicates.
    """
    if predicate is None:
        return None
    try:
        parameters = inspect.signature(predicate).parameters
    except (TypeError, ValueError):
        return predicate
    # Only parameters that can take `bits` positionally count: a keyword-only
    # third one would raise on the call below, and `*args` reports a single
    # parameter while happily accepting three.
    positional = [
        p
        for p in parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 3:
        return predicate
    return lambda path, module: predicate(path, module, bits)


def _quantize_incrementally(module, *, bits: int, predicate) -> None:
    import mlx.core as mx
    from mlx import nn

    predicate = _predicate_for_bits(predicate, bits)
    units = _quantization_units(module)
    for index, unit in enumerate(units, start=1):
        nn.quantize(unit, class_predicate=predicate, bits=bits)
        mx.eval(unit.parameters())
        mx.clear_cache()
        if index % 8 == 0 or index == len(units):
            logger.info(
                "  quantized %d/%d blocks - %s",
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



def convert_component(
    name: str,
    *,
    spec,
    repo: str,
    dest: Path,
    bits: int,
) -> int:
    """Load one component, quantize it, save it, and release it.

    The whole strategy in one function, and the order of the last two steps is
    the point: nothing else of the model is ever resident, and this component is
    gone before the next one is read. Returns the bytes written for it.

    Family-general. What differs per family — which module class this component
    is, what configuration sizes it, which predicate decides what may be
    quantized — is read from that family's own mflux structures, never from a
    table here.
    """
    import mlx.core as mx
    from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.saving.model_saver import ModelSaver

    from qds.registry import family_structure, model_config_for

    variant_class, family_definition = family_structure(spec.family)
    # Tokenizers are written by whichever run first finds them missing: they are
    # required to load the artifact and belong to no single component.
    with_tokenizers = not tokenizers_present(dest, family_definition)
    definition = single_component_definition(
        family_definition, name, with_tokenizers=with_tokenizers
    )
    component = definition.get_components()[0]
    model_config = model_config_for(spec)

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

    module = _build_module(variant_class, model_config, name)
    module.update(weights.components[component.name], strict=False)
    # Drop the loader's reference before quantizing: otherwise the bf16 arrays
    # stay alive for the whole conversion.
    weights.components.clear()
    del weights
    gc.collect()
    # Deliberately *not* `mx.eval(module.parameters())` here.
    #
    # MLX arrays are lazy, and that laziness is what makes this bounded: leaving
    # the source weights unevaluated lets `_quantize_incrementally` materialise
    # one block at a time, so the peak is the quantized result plus one block
    # rather than the whole component in its source precision. Measured on
    # z-image's transformer, 4-bit: 12.41 GB peak with an eval here, 4.68 GB
    # without, and the two runs produced byte-identical shards. The eval was
    # buying nothing but an honest-looking number in the log line below.
    logger.info("Source weights attached (lazily) - %s", _memory())

    # `skip_quantization` is the family's own statement that quantizing this
    # component degrades it — Qwen says so of its text encoder — and mflux
    # honours it on load as well. Quantizing it here would produce an artifact
    # whose weights disagree with the structure mflux rebuilds for them.
    if component.skip_quantization:
        logger.info(
            "%s is saved at its source precision: %s does not quantize it.",
            name,
            spec.family,
        )
    else:
        _quantize_incrementally(
            module,
            bits=resolved_bits,
            predicate=family_definition.quantization_predicate,
        )
        logger.info("Quantized to %d bits - %s", resolved_bits, _memory())

    shim = _ComponentShim(name, module)
    if definition.get_tokenizers():
        shim.tokenizers = TokenizerLoader.load_all(
            definitions=definition.get_tokenizers(),
            model_path=repo,
        )

    # Overwriting an existing component is the one case where a killed run could
    # leave a directory that still *validates* — the previous index next to a
    # truncated shard — so that case writes beside it and swaps. A first
    # conversion writes straight in, which keeps the disk peak exactly what the
    # arithmetic above assumes.
    target = dest / component.hf_subdir
    staging = dest / STAGING_DIRNAME if target.exists() else None
    if staging is not None:
        shutil.rmtree(staging, ignore_errors=True)

    ModelSaver.save_model(
        model=shim,
        bits=resolved_bits,
        base_path=str(staging or dest),
        weight_definition=definition,
    )
    if staging is not None:
        _swap_into_place(staging, dest)

    written_bytes = artifacts.directory_size(dest / component.hf_subdir)
    logger.info(
        "Written to %s (%.1f GB)",
        dest / component.hf_subdir,
        written_bytes / 1e9,
        extra={
            "event": "prequantize_component_done",
            "fields": {
                "component": name,
                "bits": resolved_bits,
                "written_gb": round(written_bytes / 1e9, 2),
            },
        },
    )

    del shim, module
    gc.collect()
    mx.clear_cache()
    logger.info("Released - %s", _memory())
    return written_bytes


class _ComponentShim:
    """Minimal attribute holder for `ModelSaver.save_model`.

    The saver reads `getattr(model, component.name)` and, when present,
    `model.tokenizers` — no need for the full model. That it accepts such an
    object is not incidental: it is the property that makes an artifact
    assemblable one run at a time.
    """

    def __init__(self, name: str, module) -> None:
        setattr(self, name, module)


#: Where a component is written when it is replacing one that is already there.
STAGING_DIRNAME = ".qds-staging"


def _swap_into_place(staging: Path, dest: Path) -> None:
    """Move everything a staged save produced over the live artifact.

    Directory by directory, and each one replaced only once it is complete in
    staging. The window in which a component is neither the old one nor the new
    one is a rename, rather than the length of a conversion.
    """
    for child in sorted(staging.iterdir()):
        if not child.is_dir():
            continue
        target = dest / child.name
        shutil.rmtree(target, ignore_errors=True)
        os.replace(child, target)
    shutil.rmtree(staging, ignore_errors=True)



def build_parser() -> argparse.ArgumentParser:
    """The command line this subcommand accepts. See `fetch.build_parser`."""
    from qds.registry import QUANTIZE_CHOICES

    parser = argparse.ArgumentParser(
        prog="qds prequantize",
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
    # No `choices`: which components exist is a property of the model, not of
    # this parser, and the model is not known until `--model` has been read. The
    # request is checked against that family's published components in `convert`,
    # where an unknown name is named rather than turned into a parser error
    # listing FLUX.2-dev's three.
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="components to convert (default: every component this model requires)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        default=env.flag("LOG_JSON"),
        help="one line, one JSON object, so a supervisor can follow along",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Same configuration as the server, so the menubar app follows the
    # conversion exactly as it follows a generation.
    setup_logging(level="INFO", log_file=None, json_lines=args.json_logs)
    # Same ordering constraint as `fetch.main`: before mflux pulls in
    # huggingface_hub and its frozen cache constant.
    from qds.settings import load_settings

    # Model management, like the catalogue: converting one model does not
    # depend on which model answers a request that names none.
    logger.info("HuggingFace storage: %s", load_settings(strict=False).apply_hf_home())
    return run_guarded(lambda: convert(args), what="conversion")


def convert(args) -> int:
    """Run one conversion of the requested components, and record what it did."""
    from qds.registry import BASE_SPECS_BY_KEY
    from qds.settings import load_settings

    settings = load_settings(strict=False)
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
    # Fails closed on the second contract too: a family with no established
    # component list has no conversion, whatever its capability says.
    if not comp.is_supported(spec.family):
        raise ValueError(
            f"No component-wise conversion is established for the {spec.family!r} family, "
            f"so {spec.key} cannot be converted."
        )

    required = comp.required_components(spec.family)
    requested = list(args.components) if args.components else list(required)
    unknown = comp.unknown(spec.family, requested)
    if unknown:
        raise ValueError(
            f"{spec.key} has no component called {', '.join(unknown)}. "
            f"Its components are: {', '.join(comp.component_keys(spec.family))}."
        )
    ordered = comp.ordered(spec.family, requested)

    source = spec.model_path or spec.repo
    _require_reachable_cache(settings)
    dest = (
        Path(args.dest).expanduser()
        if args.dest
        else artifacts.artifact_dir(spec.key, source, args.bits, base=settings.effective_cache_dir)
    )
    strategy = capability.prequantize_strategy

    _check_disk(spec, ordered, dest=dest)
    dest.mkdir(parents=True, exist_ok=True)
    _prepare_dest(dest, spec=spec, source=source, bits=args.bits, strategy=strategy)

    already = artifacts.component_states(
        dest, expected=required, source=source, bits=args.bits, strategy=strategy
    )
    done_before = sorted(n for n, state in already.items() if state == artifacts.COMPONENT_COMPLETE)
    if done_before:
        logger.info(
            "Continuing an existing %d-bit conversion: %s already converted.",
            args.bits,
            ", ".join(done_before),
            extra={
                "event": "prequantize_continue",
                "fields": {"bits": args.bits, "completed": done_before},
            },
        )

    for name in ordered:
        written = convert_component(
            name, spec=spec, repo=args.repo or source, dest=dest, bits=args.bits
        )
        # The component's own directory is validated before anything records it
        # as done: a run that exited is not evidence that a component landed.
        if not av.component_is_complete(dest / name):
            logger.error(
                "%s did not produce a complete %s; leaving it unrecorded",
                spec.key,
                name,
                extra={
                    "event": "job_failed",
                    "fields": {"reason": "component_incomplete", "component": name},
                },
            )
            return 1
        artifacts.record_component(
            dest,
            model_key=spec.key,
            family=spec.family,
            source=source,
            bits=args.bits,
            strategy=strategy or "",
            component=name,
            size_bytes=written,
        )
        if spec.family == "flux2-dev":
            logger.info(
                "The bf16 for '%s' is no longer needed: purge it from the HF cache before the "
                "next component (`hf cache delete`, or delete %s/ inside the snapshot) to bound "
                "the disk peak.\n",
                name,
                name,
            )

    return _finish(
        dest,
        spec=spec,
        source=source,
        bits=args.bits,
        strategy=strategy or "",
        required=required,
    )


def _require_reachable_cache(settings) -> None:
    """Refuse before any heavy work if the chosen cache cannot be reached.

    Only a directory the user explicitly chose is checked. Falling back to the
    derived default would be the worst possible response: the conversion would
    appear to succeed, tens of gigabytes would land somewhere the user did not
    choose, and the artifact would be invisible the moment the volume came back.
    An unplugged disk is a temporary fact about storage, and the honest answer is
    to stop and say so.

    A directory that merely does not exist yet is not unreachable — the first
    conversion creates it — so only an absent volume or an unreadable path
    refuses.
    """
    chosen = settings.storage.cache_dir
    if not chosen:
        return
    state, detail = av.local_path_availability(chosen)
    if state in (av.VOLUME_UNMOUNTED, av.UNREADABLE):
        raise UnavailableCache(
            f"the pre-quantized model cache is unavailable: {detail}. Reconnect it, or choose "
            f"another folder under Configuration → Storage. Nothing was converted and the "
            f"configured location was left unchanged."
        )


def _check_disk(spec, ordered: tuple[str, ...], *, dest: Path) -> None:
    """Refuse to start a conversion that cannot fit.

    FLUX.2-dev has measured figures per component, so its check is arithmetic.
    Nothing else does — the catalogue carries no per-component size — so theirs
    is a floor: enough room that a conversion cannot begin on a full volume.
    Inventing a number for the second case would be worse than admitting there
    is none.
    """
    available = free_gb(dest)
    if spec.family == "flux2-dev":
        needed = required_free_gb(ordered)
        if available < needed:
            raise InsufficientDisk(
                f"needs about {needed:.0f} GB free to convert {', '.join(ordered)}, "
                f"but only {available:.0f} GB is available on {dest}. That figure assumes the bf16 "
                f"source is purged from the HuggingFace cache between components; keeping all of it "
                f"cached needs roughly 169 GB."
            )
    elif available < GENERIC_MIN_FREE_GB:
        raise InsufficientDisk(
            f"only {available:.0f} GB free where {dest} would be written. A saved copy of "
            f"{spec.key} is a large fraction of its source, and QDS has no exact figure for it, "
            f"so it refuses to start below {GENERIC_MIN_FREE_GB:.0f} GB."
        )
    logger.info(
        "Disk check: %.0f GB free - %s",
        available,
        dest,
        extra={
            "event": "prequantize_disk_check",
            "fields": {"free_gb": round(available, 1)},
        },
    )


def _prepare_dest(dest: Path, *, spec, source: str, bits: int, strategy: str | None) -> None:
    """Make the destination safe to write one component into.

    Two things are cleared. Staging left by a killed run is garbage and is
    removed. And a *completion* marker is removed before the artifact is
    modified — with the components it vouched for carried into the progress
    record first, so nothing is forgotten. An artifact being rewritten is not a
    complete artifact, and leaving the marker in place would let a run cancelled
    halfway keep advertising a model that is no longer all there.
    """
    shutil.rmtree(dest / STAGING_DIRNAME, ignore_errors=True)

    record = artifacts.read_record(dest)
    if record is None:
        return

    for name in record.expected:
        if av.component_is_complete(dest / name):
            artifacts.record_component(
                dest,
                model_key=spec.key,
                family=spec.family,
                source=record.source or source,
                bits=record.bits if record.bits is not None else bits,
                strategy=record.strategy or strategy or "",
                component=name,
            )
    (dest / av.COMPLETION_MARKER).unlink(missing_ok=True)
    logger.info(
        "Rewriting an artifact that was complete; it is marked unusable until this run finishes.",
        extra={"event": "prequantize_reopen", "fields": {"dest": str(dest)}},
    )


def _finish(
    dest: Path,
    *,
    spec,
    source: str,
    bits: int,
    strategy: str,
    required: tuple[str, ...],
) -> int:
    """Validate what is on disk, then record completion — in that order.

    A process that exited zero is not evidence: the marker goes down only after
    the shards referenced by every index are present and the precision mflux
    stamped matches the one that was asked for.

    A run that converted only some of the required set is a **success**, not a
    failure: converting a model in stages is the point of the exercise. It
    leaves progress recorded and no completion marker, which is exactly the
    state that says "not usable yet, carry on from here".
    """
    states = artifacts.component_states(
        dest, expected=required, source=source, bits=bits, strategy=strategy
    )
    present = tuple(name for name in required if states.get(name) == artifacts.COMPONENT_COMPLETE)
    missing = [name for name in required if name not in present]

    if missing:
        logger.info(
            "Converted %s. Still missing: %s - %s stays partial until those are converted too.",
            ", ".join(present) or "nothing",
            ", ".join(missing),
            dest,
            extra={
                "event": "prequantize_partial",
                "fields": {
                    # The counterpart of `prequantize_done`, and the reason an
                    # exit code cannot be read as a result: this run succeeded
                    # and the artifact is still not usable.
                    "model": spec.key,
                    "variant_ready": False,
                    "dest": str(dest),
                    "bits": bits,
                    "completed": list(present),
                    "missing": missing,
                },
            },
        )
        return 0

    ok, detail = artifacts.components_are_complete(dest, present)
    if not ok:  # pragma: no cover - `component_states` already checked each one
        logger.warning("Conversion incomplete at %s (%s); no completion marker written", dest, detail)
        return 1

    missing_tokenizers = _missing_tokenizers(dest, spec)
    if missing_tokenizers:
        # An artifact without its tokenizer cannot be loaded, so it is not
        # complete however many components validated.
        logger.error(
            "%s has every component but no %s directory; refusing to mark it complete",
            dest,
            ", ".join(missing_tokenizers),
            extra={
                "event": "job_failed",
                "fields": {"reason": "tokenizer_missing", "missing": missing_tokenizers},
            },
        )
        return 1

    stored = artifacts.stored_bits(dest, present)
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

    size_bytes = artifacts.directory_size(dest)
    artifacts.write_record(
        dest,
        model_key=spec.key,
        family=spec.family,
        source=source,
        bits=bits,
        strategy=strategy,
        components=present,
        required=required,
        size_bytes=size_bytes,
    )
    artifacts.clear_progress(dest)
    logger.info(
        "Done - %s: %.1f GB, %d-bit",
        dest,
        size_bytes / 1e9,
        bits,
        extra={
            "event": "prequantize_done",
            "fields": {
                # Named here because this event is the *only* statement that an
                # artifact is complete and usable, and the supervisor acts on it:
                # a run that converted a subset emits `prequantize_partial`
                # instead and never reaches this line. Which model and which
                # depth are therefore part of the claim, not context — without
                # them the reader would have to parse a human-readable label to
                # know what became ready.
                "model": spec.key,
                "variant_ready": True,
                "dest": str(dest),
                "bits": bits,
                "components": list(present),
                "size_bytes": size_bytes,
            },
        },
    )
    logger.info("Select it for %s in the Models tab to generate with it.", spec.key)
    return 0


def _missing_tokenizers(dest: Path, spec) -> list[str]:
    """Tokenizer directories the family declares and the artifact does not have."""
    from qds.registry import family_structure

    _, definition = family_structure(spec.family)
    return [
        tokenizer.hf_subdir
        for tokenizer in definition.get_tokenizers()
        if not (dest / tokenizer.hf_subdir).is_dir()
    ]


class InsufficientDisk(RuntimeError):
    """Not enough free space to attempt the conversion."""


class UnavailableCache(RuntimeError):
    """The configured pre-quantization cache cannot be reached right now."""


def run_guarded(action, *, what: str) -> int:
    """Same contract as `fetch.run_guarded`: name expected failures on the stream."""
    from qds.fetch import run_guarded as guarded

    return guarded(action, what=what, log=logger)


if __name__ == "__main__":
    raise SystemExit(main())
