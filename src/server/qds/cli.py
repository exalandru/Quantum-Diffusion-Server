"""The `qds` command: one entry point over the server and its tools.

A dispatcher, deliberately, rather than one argparse tree covering everything.
Each subcommand's arguments stay owned by the module that implements it —
`qds fetch --help` is `fetch.main`'s own parser, printed under its own `prog` —
so there is no second copy of an option list to drift out of step with the
first. This module knows the names of the subcommands and nothing about their
arguments.

Every import of an implementation is lazy. `qds fetch --status` reads a
catalogue and must not pay for mflux and torch to do it, and the difference is
seconds on every call the menubar app makes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from qds import __version__

#: A subcommand implementation: `main(argv) -> exit status`.
Entry = Callable[[Sequence[str]], int]
#: A loader defers importing that implementation until the subcommand is chosen.
Loader = Callable[[], Entry]

#: name -> (one-line help, loader)
_COMMANDS: dict[str, tuple[str, Loader]] = {}


def _command(name: str, summary: str) -> Callable[[Loader], Loader]:
    def register(loader: Loader) -> Loader:
        _COMMANDS[name] = (summary, loader)
        return loader

    return register


@_command("serve", "run the server")
def _serve() -> Entry:
    from qds.app import main

    return main


@_command("fetch", "download a model's weights, or report what is cached")
def _fetch() -> Entry:
    from qds.fetch import main

    return main


@_command("rewrite", "expand a prompt with the local rewriter, and print it")
def _rewrite() -> Entry:
    from qds.rewrite_cli import main

    return main


@_command("prequantize", "convert a model into a saved, already-quantized artifact")
def _prequantize() -> Entry:
    from qds.prequantize import main

    return main


@_command("import", "register a local model directory, without copying it")
def _import() -> Entry:
    from qds.import_cli import main

    return main


@_command("status", "ask a running server how it is doing")
def _status() -> Entry:
    return status_main


def status_main(argv: Sequence[str] | None = None) -> int:
    """GET /health from the configured address and print the answer.

    Settings are read leniently: a configuration the *server* would refuse to
    start on is exactly when someone runs this, and answering "where would I
    even connect" is useful then.
    """
    import json
    import urllib.error
    import urllib.request

    parser = argparse.ArgumentParser(
        prog="qds status",
        description="Print the running server's /health document, or say why it could not be read.",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="seconds to wait for an answer (default: 5)"
    )
    args = parser.parse_args(argv)

    from qds.settings import ConfigError, load_settings

    try:
        settings = load_settings(strict=False)
    except (ConfigError, ValueError) as exc:
        print(f"qds status: configuration could not be read: {exc}", file=sys.stderr)
        return 2

    # A wildcard bind is an address to listen on, not one to connect to.
    host = settings.server.host
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    url = f"http://{host}:{settings.server.port}/health"

    # /health is never gated, so no key is sent: this stays a reachability check
    # rather than one that also depends on the key being right.
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        print(f"qds status: no answer from {url}: {exc.reason}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"qds status: no answer from {url}: {exc}", file=sys.stderr)
        return 1

    try:
        json.dump(json.loads(body), sys.stdout, indent=2)
    except json.JSONDecodeError:
        # Something answered on that port, but it was not this server.
        print(f"qds status: {url} did not answer with JSON.", file=sys.stderr)
        return 1
    sys.stdout.write("\n")
    return 0


def _usage() -> str:
    width = max(len(name) for name in _COMMANDS)
    lines = [f"  {name.ljust(width)}  {summary}" for name, (summary, _) in _COMMANDS.items()]
    return "\n".join(
        [
            "usage: qds <command> [options]",
            "",
            "commands:",
            *lines,
            "",
            "Run `qds <command> --help` for a command's own options.",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0
    if argv[0] in {"-V", "--version"}:
        print(__version__)
        return 0

    command, rest = argv[0], argv[1:]
    entry = _COMMANDS.get(command)
    if entry is None:
        print(f"qds: unknown command {command!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    _, loader = entry
    return loader()(rest)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
