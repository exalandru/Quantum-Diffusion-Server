"""Downloading a model's weights on demand, and reporting what is already cached.

Without this, the first generation on a fresh model pays the whole download —
tens of gigabytes, with no progress anywhere in the app and a request that looks
hung. So it becomes an explicit step: a button in the Models tab, this script
behind it.

Two modes, both aimed at the desktop app but usable on their own:

* `--status` prints one JSON line per catalogue entry, saying whether its repo is
  in the HuggingFace cache and how much room it takes. Imports nothing heavier
  than `huggingface_hub`, so it answers instantly.
* `<key>` downloads that model, by **loading it and exiting**. That looks
  indirect, and it is deliberate: the download patterns live in each family's
  `WeightDefinition`, and duplicating them here would be one more table to keep
  in step with mflux. Loading also proves the thing works — gated access granted,
  quantization applied — instead of merely proving that files landed on disk.

What it does *not* buy: the quantization is not persisted, so the first real load
pays it again. It is the download that is worth moving, and the download is what
this moves.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mflux_server.logs import SERVER_LOGGER, setup_logging
from mflux_server.settings import ENV_PREFIX, load_settings

logger = logging.getLogger(f"{SERVER_LOGGER}.fetch")


def _cache_dir() -> str:
    """Where the weights live, read from the environment *now*.

    `huggingface_hub` freezes `HF_HUB_CACHE` at import time, so relying on its
    constant means whichever value was set when some other module first imported it
    — not the one this process was given. We resolve it per call instead, following
    the documented layout (`HF_HOME/hub`, or `HF_HUB_CACHE` when set outright).
    """
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return explicit
    from mflux_server.availability import hub_cache_for

    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    # The same normalisation `apply_hf_home` publishes, so a caller that reached
    # here without it cannot scan a different directory from the one downloads
    # land in.
    return str(hub_cache_for(home))


def resolved_target(spec: Any) -> str:
    """The repo or path this spec will actually load from.

    One function, so what the catalogue *reports* and what a download *fetches*
    cannot disagree — they used to, for any disabled model carrying a `model_path`
    override, because the reporting applied overrides and the fetch did not.
    """
    return spec.model_path or spec.repo


def _variants_of(spec: Any, source: str) -> list[Any]:
    """Saved variants for this spec, or none when the model cannot have any."""
    if not spec.quantization.supports_prequantize:
        return []
    from mflux_server import artifacts

    return artifacts.discover_variants(spec.key, source)


def cache_status() -> list[dict[str, Any]]:
    """Every catalogue entry with an explicit availability, not a boolean.

    Reads the config so `enabled` reflects what the server would expose, and so a
    `model_path` override is both what we report on and what we would fetch.

    See `availability.py` for exactly what `present` does and does not claim.
    """
    from mflux_server import availability as av
    from mflux_server.registry import PROVENANCE_BUILT_IN

    settings = load_settings()
    # Idempotent: `main` already applied it. Doing it here too keeps `cache_status`
    # correct when imported directly, as the tests do.
    settings.apply_hf_home()
    enabled = set(settings.registry())
    # Disabled entries included on purpose: you download a model before turning it
    # on, so this is the spec whose `model_path` the Install button will use.
    configured = settings.registry(include_disabled=True)

    root = Path(_cache_dir())
    repo_availability, repo_info, state = av.scan_repos(root)

    rows = []
    # The effective catalogue, not the built-in one: imported models are part of
    # what the user has, and iterating `BASE_SPECS` alone made them invisible in
    # the Models tab even though the registry knew about them.
    for key, spec in configured.items():
        target = resolved_target(spec)
        # Two different questions, deliberately kept apart. Whether the source is
        # a filesystem path decides *how availability is measured*; whether the
        # model came from HuggingFace at all decides *what may be offered*, and
        # that is provenance, never the shape of the string.
        is_path_source = not av.looks_like_repo_id(target)
        can_download = spec.provenance == PROVENANCE_BUILT_IN and not is_path_source

        if is_path_source:
            # A path is a filesystem question, answered as one. For our own
            # pre-quantized output the question is stricter — a directory holding a
            # half-finished conversion is not an artifact — and the spec's family is
            # what says which rule applies, rather than the shape of the string.
            if spec.family == "flux2-dev":
                status, detail = av.flux2_dev_artifact_state(target)
            else:
                status, detail = av.local_path_availability(target)
            info: dict[str, Any] = {}
        elif not state.usable:
            # The cache root itself is gone or unreadable: that is one fact about
            # the storage, not ten separate "not downloaded" verdicts.
            status, detail = state.availability, state.detail
            info = {}
        else:
            status = repo_availability.get(target, av.MISSING)
            detail = None
            info = repo_info.get(target, {})

        rows.append(
            {
                "key": key,
                "repo": target,
                "license": spec.license,
                "gated": spec.gated,
                "enabled": key in enabled,
                "availability": status,
                "detail": detail,
                # Retained for display only, and derived — never recomputed from
                # the string by a consumer.
                "local": is_path_source,
                "provenance": spec.provenance,
                "display_name": spec.display_name or key,
                # What an API request must send. A built-in publishes its key; an
                # imported model publishes its alias rather than its opaque id.
                "api_name": spec.public_name,
                "base_profile_key": spec.base_profile_key,
                "family": spec.family,
                # The single authority for offering Install/Resume.
                "can_download": can_download,
                "size_gb": info.get("size_gb", 0.0),
                "files": info.get("files", 0),
                # The quantization contract travels with the catalogue, not only
                # with `/v1/capabilities`: that endpoint publishes enabled models
                # only, while the Configuration form has to configure the disabled
                # ones too — and must keep working with the server stopped.
                # Which saved variants actually exist for this source, and which
                # one generation is set to use. Capability (what *could* be
                # converted), availability (what exists) and activation (what is
                # used) stay three separate facts.
                "variants": [
                    {"bits": v.bits, "path": v.path, "strategy": v.strategy, "legacy": v.legacy}
                    for v in _variants_of(spec, target)
                ],
                "active_variant": spec.prequantized_variant,
                "quantization": {
                    "supports_quantization": spec.quantization.supports_quantization,
                    "quantize_choices": list(spec.quantization.quantize_choices),
                    "supports_prequantize": spec.quantization.supports_prequantize,
                    "prequantize_choices": list(spec.quantization.prequantize_choices),
                    "prequantize_strategy": spec.quantization.prequantize_strategy,
                    "note": spec.quantization.note,
                },
            }
        )
    return rows


def fetch(key: str) -> int:
    from mflux_server.registry import BASE_SPECS_BY_KEY

    settings = load_settings()
    settings.apply_hf_home()
    # `include_disabled`: the documented workflow is to download a model *before*
    # enabling it, and the enabled-only view silently fell back to the raw
    # catalogue spec for those — dropping the very `model_path` the Models tab had
    # just reported. `resolved_target` is now the single answer to "which repo".
    spec = settings.registry(include_disabled=True).get(key) or BASE_SPECS_BY_KEY.get(key)
    if spec is None:
        logger.error(
            "Unknown model %r. Valid keys: %s",
            key,
            sorted(BASE_SPECS_BY_KEY),
            extra={"event": "job_failed", "fields": {"reason": "unknown_model", "model": key}},
        )
        return 2

    if spec.gated:
        logger.info(
            "%s is gated (%s): the download needs a HuggingFace token with approved access.",
            spec.repo,
            spec.license,
        )
    logger.info("Fetching %s (%s) — %s", key, resolved_target(spec), spec.license)

    from mflux_server.registry import load_model

    model = load_model(spec)
    # Dropping the reference is enough: the process exits right after, which is
    # the cheapest possible teardown.
    del model
    logger.info("%s is ready. The next generation will not wait for a download.", key)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mflux-server-fetch",
        description="Download a model's weights ahead of time, or report what is cached.",
    )
    parser.add_argument("model", nargs="?", help="catalogue key to download")
    parser.add_argument(
        "--status",
        action="store_true",
        help="print the catalogue with cache state as JSON, and exit",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        default=os.environ.get(f"{ENV_PREFIX}LOG_JSON", "").lower() in {"1", "true", "yes"},
        help="one line, one JSON object, so a supervisor can follow along",
    )
    args = parser.parse_args()

    # Before anything imports huggingface_hub: it freezes `HF_HUB_CACHE` at import
    # time, so the configured root has to be in the environment first.
    load_settings().apply_hf_home()

    if args.status:
        # Straight to stdout, without the logging setup: this output is parsed.
        json.dump(cache_status(), sys.stdout)
        sys.stdout.write("\n")
        return 0

    if not args.model:
        parser.error("give a model key, or --status")

    setup_logging(level="INFO", log_file=None, json_lines=args.json_logs)
    return run_guarded(lambda: fetch(args.model), what="download")


def run_guarded(action: Any, *, what: str, log: logging.Logger | None = None) -> int:
    """Run `action`, turning a known failure into a structured terminal event.

    The supervisor watching this stream had nothing to report but the exit code,
    so every failure — a gated repo, a full disk, a bad config — reached the user
    as "exited with code 1". An expected failure now names itself on the JSON
    stream before the non-zero exit; the traceback still goes to stderr, because
    for an unexpected one that is the useful artefact.
    """
    log = log or logger
    try:
        return action()
    except Exception as exc:
        reason = _known_reason(exc)
        log.error(
            "%s failed: %s",
            what,
            reason or f"{type(exc).__name__}: {exc}",
            exc_info=reason is None,
            extra={
                "event": "job_failed",
                "fields": {"reason": type(exc).__name__, "what": what},
            },
        )
        return 1


def _known_reason(exc: BaseException) -> str | None:
    """A plain sentence for the failures we expect, or None for a genuine bug."""
    name = type(exc).__name__
    if name == "InsufficientDisk":
        return str(exc)
    if name == "GatedRepoError":
        return (
            "this repository is gated. Request access on its model card, then save a "
            "HuggingFace token in the Models tab."
        )
    if name in {"RepositoryNotFoundError", "EntryNotFoundError", "RevisionNotFoundError"}:
        return f"the repository or file was not found ({exc})."
    if name in {"HfHubHTTPError", "LocalEntryNotFoundError", "ConnectionError"}:
        return f"HuggingFace could not be reached ({exc})."
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return "the disk is full."
    if isinstance(exc, ValueError):
        # `load_settings` raises this for an invalid server-config.json.
        return str(exc)
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
