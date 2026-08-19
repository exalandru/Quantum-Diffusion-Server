"""Environment-variable naming for the server.

One leaf module owns the contract, so no caller has to spell a prefix out twice
and the pre-2.0 fallback lives in exactly one place.
"""

from __future__ import annotations

import logging
import os

#: Canonical prefix: every key in the `server` config section is overridable as
#: `QDS_SERVER_<KEY>`.
ENV_PREFIX = "QDS_SERVER_"

#: The prefix used while the distribution was called `mflux-server`.
LEGACY_ENV_PREFIX = "MFLUX_SERVER_"

logger = logging.getLogger("qds.env")

#: Names already warned about, so a variable read on every request logs once.
_warned: set[str] = set()


def get(name: str, default: str | None = None) -> str | None:
    """Read `QDS_SERVER_<name>`, falling back to `MFLUX_SERVER_<name>`.

    The fallback is not politeness. Dropping the old prefix outright fails
    silently rather than loudly: a stale `MFLUX_SERVER_CONFIG` left behind in a
    launch agent or a shell profile would make the server start on packaged
    defaults instead of the operator's configuration, with nothing in the log to
    say why. Reading it and complaining is the loud version of the same story.
    """
    value = os.environ.get(ENV_PREFIX + name)
    if value is not None:
        return value
    value = os.environ.get(LEGACY_ENV_PREFIX + name)
    if value is not None:
        if name not in _warned:
            _warned.add(name)
            logger.warning(
                "%s%s is deprecated and will stop being read; rename it to %s%s.",
                LEGACY_ENV_PREFIX,
                name,
                ENV_PREFIX,
                name,
            )
        return value
    return default


def flag(name: str) -> bool:
    """A `QDS_SERVER_<name>` variable read as a boolean switch."""
    return (get(name) or "").strip().lower() in {"1", "true", "yes"}
