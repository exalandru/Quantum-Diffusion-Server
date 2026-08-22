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

    try:
        anyio.run(lambda: relay(url, api_key=api_key))
    except Unreachable as exc:
        warn(describe_failure(url, str(exc)))
        return 1
    except KeyboardInterrupt:  # pragma: no cover - a host stopping the child
        return 0
    return 0
