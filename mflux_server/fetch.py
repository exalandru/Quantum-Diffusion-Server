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
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(home, "hub")


def cache_status() -> list[dict[str, Any]]:
    """Every catalogue entry, with its repo's presence and size in the HF cache.

    Reads the config so `enabled` reflects what the server would actually expose,
    and so a `model_path` override is the repo we report on.

    `scan_cache_dir` tells us what is cached, not whether every file mflux needs is
    there: a download interrupted halfway still shows up. That is acceptable —
    fetching again completes it, and mflux checks snapshot completeness itself at
    load time — but it means "cached" reads as "already downloaded, probably", not
    as a guarantee.
    """
    from huggingface_hub import scan_cache_dir

    from mflux_server.registry import BASE_SPECS_BY_KEY

    settings = load_settings()
    enabled = set(settings.registry())
    overrides = settings.models

    try:
        cache = {repo.repo_id: repo for repo in scan_cache_dir(cache_dir=_cache_dir()).repos}
    except Exception:  # pragma: no cover - a missing cache dir is not an error
        logger.debug("HuggingFace cache unreadable", exc_info=True)
        cache = {}

    rows = []
    for key, spec in BASE_SPECS_BY_KEY.items():
        override = overrides.get(key)
        repo = (override.model_path if override and override.model_path else spec.model_path) or spec.repo
        entry = cache.get(repo)
        rows.append(
            {
                "key": key,
                "repo": repo,
                "license": spec.license,
                "gated": spec.gated,
                "enabled": key in enabled,
                "cached": entry is not None,
                # A local artifact (flux2-dev) is not in the HF cache at all: the
                # app reports on it separately.
                "local": "/" not in repo or repo.startswith(("~", "/", ".")),
                "size_gb": round(entry.size_on_disk / 1e9, 1) if entry else 0.0,
                "files": entry.nb_files if entry else 0,
            }
        )
    return rows


def fetch(key: str) -> int:
    from mflux_server.registry import BASE_SPECS_BY_KEY

    settings = load_settings()
    # The configured spec, not the raw catalogue one: quantization and any
    # `model_path` override change what gets downloaded.
    spec = settings.registry().get(key) or BASE_SPECS_BY_KEY.get(key)
    if spec is None:
        logger.error("Unknown model %r. Valid keys: %s", key, sorted(BASE_SPECS_BY_KEY))
        return 2

    if spec.gated:
        logger.info(
            "%s is gated (%s): the download needs a HuggingFace token with approved access.",
            spec.repo,
            spec.license,
        )
    logger.info("Fetching %s (%s) — %s", key, spec.repo, spec.license)

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

    if args.status:
        # Straight to stdout, without the logging setup: this output is parsed.
        json.dump(cache_status(), sys.stdout)
        sys.stdout.write("\n")
        return 0

    if not args.model:
        parser.error("give a model key, or --status")

    setup_logging(level="INFO", log_file=None, json_lines=args.json_logs)
    return fetch(args.model)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
