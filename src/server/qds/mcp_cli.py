"""`qds mcp`: bridge a stdio MCP client to this machine's running server.

A separate module from `qds/mcp/bridge.py` for the same reason every other
subcommand is separate from its implementation: this owns the *arguments*, and
`qds mcp --help` should print its own parser under its own `prog`.

Settings are read leniently, as `qds status` does and for the same reason: a
configuration the server would refuse to start on is exactly when someone runs
this, and answering "where would I even connect" is useful then.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def default_url() -> tuple[str, str | None]:
    """Where the local server is, and the key to present. Never the admin token.

    The local admin token is admin-equivalent -- it opens the control plane --
    and it is never needed here: a server with an `api_key` lets this read that
    key from the same configuration, and a server without one has an open data
    plane. Narrower is correct, so the stronger credential is not carried into
    a process a chat client launches.
    """
    from qds.settings import load_settings

    settings = load_settings(strict=False)
    host = settings.server.host
    # A wildcard bind is an address to listen on, not one to connect to.
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    return f"http://{host}:{settings.server.port}/mcp", settings.server.api_key


def key_for(url: str, derived_url: str, api_key: str | None) -> str | None:
    """The key, and only when the URL names this machine's own server.

    `--url` exists to reach a server on a non-default port or a second install,
    not to hand this machine's data-plane credential to an arbitrary host over
    plain HTTP -- which a typo or a pasted line of configuration is enough to do,
    because the header is set on the client before anything about the target is
    known. Loopback is accepted because that is what the flag is legitimately for.
    """
    if api_key is None or url == derived_url:
        return api_key
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").lower()
    return api_key if host in {"127.0.0.1", "::1", "localhost"} else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qds mcp",
        description=(
            "Speak MCP over stdin and stdout, forwarding to the MCP surface of the "
            "server running on this machine. For chat clients that cannot connect "
            "to an HTTP MCP server themselves."
        ),
    )
    parser.add_argument(
        "--url",
        default=None,
        help="the server's MCP endpoint (default: derived from server-config.json)",
    )
    args = parser.parse_args(argv)

    from qds.settings import ConfigError

    try:
        derived_url, api_key = default_url()
    except (ConfigError, ValueError) as exc:
        print(f"qds mcp: configuration could not be read: {exc}", file=sys.stderr)
        return 2
    url = args.url or derived_url

    import anyio

    from qds.mcp.bridge import Unreachable, describe_failure, relay, warn

    presented = key_for(url, derived_url, api_key)
    if api_key is not None and presented is None:
        warn(
            "--url names a host that is not this machine; the configured api_key "
            "was not sent."
        )

    try:
        anyio.run(lambda: relay(url, api_key=presented))
    except Unreachable as exc:
        warn(describe_failure(url, str(exc)))
        return 1
    except KeyboardInterrupt:  # pragma: no cover - a host stopping the child
        return 0
    return 0
