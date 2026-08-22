"""stdio in, streamable-HTTP out: `qds mcp`.

Some clients speak only stdio. The engine cannot be shared between processes --
one MLX model, one lock, one machine's memory -- so a stdio server that
generated images itself would be a *second* server competing for the GPU with
the one already running. This relays instead: it is a client of the local HTTP
`/mcp`, and a server on its own stdin and stdout.

Written narrowly rather than generically. There is no generic MCP proxy in the
SDK, and a "forward everything" relay would have to reach into transport
internals to do it. So each relayed method is named, and a test asserts the
named set covers everything this server actually advertises -- adding a
capability without teaching the bridge fails that test rather than silently
dropping the capability for every stdio client.

Two things must survive the relay in opposite directions, and both are easy to
lose: progress notifications on the way back, and cancellation on the way in.
"""

from __future__ import annotations

import contextlib
import sys

#: Methods this relay forwards. Not a wildcard: see the module docstring, and
#: `tests/test_mcp_bridge.py`, which fails if the server offers something this
#: does not carry.
RELAYED = (
    "ping",
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/templates/list",
    "resources/read",
    "prompts/list",
    "prompts/get",
)


class Unreachable(RuntimeError):
    """The local server did not answer, and the reason is worth printing."""


async def relay(url: str, *, api_key: str | None = None) -> None:
    """Serve stdio, forwarding everything to the MCP server at `url`.

    Connects *before* serving stdio, deliberately. A host that launches this
    reports "the server failed to start" with whatever went to stderr, which is
    something a person can act on; a relay that accepted stdin and then hung
    waiting for a server that is not running is not.
    """
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.stdio import stdio_server

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    import httpx2

    # `follow_redirects=True` is load-bearing, not hygiene. `/mcp` is a mount,
    # so Starlette answers the bare path with a 307 to `/mcp/`; the SDK's own
    # HTTP client follows it, and a hand-built one -- which this has to be, to
    # carry the API key -- does not unless told to. Without it the relay fails
    # against a perfectly healthy server, and reports "could not reach" about a
    # server that is right there.
    async with httpx2.AsyncClient(headers=headers, follow_redirects=True) as http:
        try:
            async with Client(
                streamable_http_client(url, http_client=http), raise_exceptions=True
            ) as upstream:
                server = build_relay_server(upstream)
                async with stdio_server() as (read_stream, write_stream):
                    await server.run(read_stream, write_stream, server.create_initialization_options())
        except Unreachable:
            raise
        except BaseException as exc:  # every failure here carries the same advice
            raise Unreachable(root_cause(exc)) from exc


def root_cause(exc: BaseException) -> str:
    """The innermost message, not the task group's summary.

    anyio wraps a failed connection in an `ExceptionGroup` whose own message is
    "unhandled errors in a TaskGroup (1 sub-exception)" -- which tells a person
    reading their client's error panel nothing at all. The advice printed around
    this is only worth printing if the cause it names is the real one.
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def build_relay_server(upstream):
    """The downstream half of the relay, as a plain server object.

    Separate from `relay` so a test can drive it with an in-memory client on
    both sides. Testing the relay through two real transports would witness the
    transports; testing it here witnesses the *forwarding*, which is what has
    the interesting failure mode.
    """
    from mcp.server.lowlevel import Server

    async def on_list_tools(_ctx, _params):
        return await upstream.session.list_tools()

    async def on_call_tool(ctx, params):
        # Progress is what a naive relay loses: requests and responses forward
        # themselves, while the notifications that travel out-of-band during a
        # two-minute call do not, and the user watches nothing happen.
        #
        # `session.report_progress` rather than `send_progress_notification`,
        # and the difference is not cosmetic. The former is scoped to the
        # inbound request this session is serving: it addresses the caller's own
        # progress token, is a no-op when the caller asked for no progress, and
        # works on the in-process dispatcher as well as over JSON-RPC. Building
        # the notification by hand means guessing the token, and a wrong guess
        # is silent -- the notification is sent and simply matches nothing.
        async def forward(progress: float, total: float | None, message: str | None) -> None:
            with contextlib.suppress(Exception):
                await ctx.session.report_progress(progress, total, message)

        return await upstream.session.call_tool(
            params.name, params.arguments or {}, progress_callback=forward
        )

    async def on_list_resources(_ctx, _params):
        return await upstream.session.list_resources()

    async def on_list_resource_templates(_ctx, _params):
        return await upstream.session.list_resource_templates()

    async def on_read_resource(_ctx, params):
        return await upstream.session.read_resource(params.uri)

    async def on_list_prompts(_ctx, _params):
        return await upstream.session.list_prompts()

    async def on_get_prompt(_ctx, params):
        return await upstream.session.get_prompt(params.name, params.arguments)

    info = upstream.server_info
    server = Server(
        name=getattr(info, "name", "quantum-diffusion-server"),
        version=getattr(info, "version", ""),
        title=getattr(info, "title", None),
        # The upstream's own instructions, forwarded: they are how a model is
        # told this server's tools block and how sizes work, and a relay that
        # dropped them would leave stdio clients strictly worse guided.
        instructions=upstream.instructions,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_resources=on_list_resources,
        on_list_resource_templates=on_list_resource_templates,
        on_read_resource=on_read_resource,
        on_list_prompts=on_list_prompts,
        on_get_prompt=on_get_prompt,
    )

    return server


def describe_failure(url: str, reason: str) -> str:
    """One line a person can act on, for a host that only shows stderr."""
    return (
        f"qds mcp: could not reach the MCP surface at {url} ({reason}).\n"
        f"  Is the server running? Start it with `qds serve`, or open the QDS app.\n"
        f"  If it is running, check that mcp.enabled is true in server-config.json,\n"
        f"  and `qds status` for whether it came up in recovery mode."
    )


def warn(message: str) -> None:
    """stderr, never stdout: stdout is the protocol channel."""
    print(message, file=sys.stderr, flush=True)
