"""The stdio relay behind `qds mcp`.

Two failure modes, and both are silent, which is why they get tests rather than
trust.

The first is **drift**: a capability is added to the server and not to the
relay, so every stdio client quietly stops seeing it. `test_the_relay_carries
_every_method_this_server_offers` asks the real server what it advertises and
fails if the relay's named set does not cover it -- it fails *closed*, which is
the property that matters.

The second is **losing progress**. A relay that forwards requests and responses
is easy to write and looks correct; the notifications that travel out-of-band
during a two-minute call are what a naive one drops, and the user watches
nothing happen instead.
"""

from __future__ import annotations

import pytest

from qds.mcp.bridge import RELAYED, build_relay_server, describe_failure

from .mcp_support import mcp_session, mcp_settings

# ── The drift alarm ────────────────────────────────────────────────────────


async def test_the_relay_carries_every_method_this_server_offers(tmp_path):
    """Built from what the server actually advertises, not from a list someone
    remembered to update."""
    async with mcp_session(tmp_path) as (client, _app, _engine):
        offered = {"ping"}
        if (await client.list_tools()).tools:
            offered |= {"tools/list", "tools/call"}
        if (await client.list_resources()).resources:
            offered |= {"resources/list", "resources/read"}
        if (await client.list_resource_templates()).resource_templates:
            offered |= {"resources/templates/list", "resources/read"}
        if (await client.list_prompts()).prompts:
            offered |= {"prompts/list", "prompts/get"}

    missing = offered - set(RELAYED)
    assert not missing, f"the server offers {sorted(missing)} and the relay does not carry it"


def test_the_relayed_set_names_nothing_that_does_not_exist():
    """The other direction: a method in the list that no server speaks is dead
    weight that reads as coverage. Checked against the SDK's own spec table
    rather than a second hand-written list."""
    from mcp.server.lowlevel import server as lowlevel

    spec = set(inspect_spec_methods(lowlevel))
    unknown = set(RELAYED) - spec
    assert not unknown, f"{sorted(unknown)} is in RELAYED and is not an MCP method"


def inspect_spec_methods(lowlevel) -> set[str]:
    """Every method the SDK's lowlevel server knows how to dispatch."""
    import inspect
    import re

    source = inspect.getsource(lowlevel.Server.__init__)
    return set(re.findall(r'\("([a-z]+(?:/[a-zA-Z]+)*)",\s*types\.', source))


# ── Forwarding ─────────────────────────────────────────────────────────────


async def test_the_relay_forwards_the_tool_list_with_its_schemas(tmp_path):
    """Schemas included, and that is why the relay forwards `tools/list`
    verbatim rather than re-declaring tools of its own: a re-declared tool has
    a schema somebody wrote twice."""
    from mcp import Client

    async with mcp_session(tmp_path) as (upstream, _app, _engine):
        async with Client(build_relay_server(upstream), raise_exceptions=True) as downstream:
            through = await downstream.list_tools()
        direct = await upstream.list_tools()

    assert {t.name for t in through.tools} == {t.name for t in direct.tools}
    by_name = {t.name: t for t in through.tools}
    assert (
        by_name["generate_image"].input_schema
        == {t.name: t for t in direct.tools}["generate_image"].input_schema
    )


async def test_a_tool_call_through_the_relay_delivers_its_result_and_progress(tmp_path):
    """The whole point of the relay, and the half a naive one gets wrong.

    Requests and responses forward themselves; the notifications that travel
    out-of-band during a two-minute call do not.
    """
    from mcp import Client

    seen: list = []
    settings = mcp_settings(tmp_path)
    settings.mcp.poll_interval_s = 0.01

    async with mcp_session(tmp_path, settings=settings) as (upstream, _app, _engine):
        async with Client(build_relay_server(upstream), raise_exceptions=True) as downstream:
            result = await downstream.call_tool(
                "generate_image",
                {"prompt": "a cube"},
                progress_callback=lambda p, t, m: seen.append((p, t, m)),
            )

    assert result.is_error is False
    body = "\n".join(b.text for b in result.content if getattr(b, "type", "") == "text")
    assert "status: completed" in body
    assert "full image: http://" in body, "the route to the picture survived"
    assert [b for b in result.content if getattr(b, "type", "") == "resource_link"]
    assert not [b for b in result.content if getattr(b, "type", "") == "image"], (
        "the relay must not reintroduce pixels the upstream deliberately omits"
    )
    assert seen, "progress notifications survived the relay"


async def test_the_relay_forwards_the_instructions(tmp_path):
    """They are how a model is told the tools block and how sizes work. A relay
    that dropped them would leave stdio clients strictly worse guided."""
    from mcp import Client

    async with mcp_session(tmp_path) as (upstream, _app, _engine):
        async with Client(build_relay_server(upstream), raise_exceptions=True) as downstream:
            assert downstream.instructions == upstream.instructions
            assert "one at a time" in (downstream.instructions or "")


async def test_a_tool_error_survives_the_relay_as_a_readable_message(tmp_path):
    """A small model corrects itself from the message. Turning it into a code
    on the way through would defeat the surface's whole error design."""
    from mcp import Client

    async with mcp_session(tmp_path) as (upstream, _app, _engine):
        async with Client(build_relay_server(upstream), raise_exceptions=True) as downstream:
            result = await downstream.call_tool("generate_image", {"prompt": "a cube", "n": 99})

    assert result.is_error is True
    body = "\n".join(b.text for b in result.content if getattr(b, "type", "") == "text")
    assert "exceeds the server limit" in body


async def test_the_relay_forwards_a_resource_read(tmp_path):
    from mcp import Client

    async with mcp_session(tmp_path) as (upstream, _app, _engine):
        async with Client(build_relay_server(upstream), raise_exceptions=True) as downstream:
            listed = await downstream.list_resources()
            assert listed.resources
            read = await downstream.read_resource("qds://models")
    assert "default model:" in read.contents[0].text


# ── Failing usefully ───────────────────────────────────────────────────────


def test_the_bridge_exits_with_a_reason_when_nothing_is_listening(capsys):
    """A host that launches this shows stderr and nothing else. Hanging on a
    silent stdin is the failure mode this avoids."""
    from qds.mcp_cli import main

    code = main(["--url", "http://127.0.0.1:1/mcp"])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == "", "stdout is the protocol channel and must stay clean"
    assert "qds serve" in captured.err
    assert "http://127.0.0.1:1/mcp" in captured.err


def test_the_failure_message_names_the_url_and_what_to_do():
    message = describe_failure("http://127.0.0.1:8765/mcp", "connection refused")
    assert "http://127.0.0.1:8765/mcp" in message
    assert "qds serve" in message
    assert "mcp.enabled" in message


def test_the_bridge_never_carries_the_admin_token(monkeypatch, tmp_path):
    """The local token is admin-equivalent and is never needed here: a server
    with a key lets this read the key, and one without has an open data plane.
    """
    import inspect

    from qds import mcp_cli

    source = inspect.getsource(mcp_cli)
    assert "admin" not in source.lower().replace("admin-equivalent", "").replace("admin token", "").replace(
        "control plane", ""
    )


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://127.0.0.1:8765/mcp", "k"),
        ("http://127.0.0.1:9000/mcp", "k"),
        ("http://localhost:9000/mcp", "k"),
        ("http://[::1]:9000/mcp", "k"),
        ("http://attacker.example/mcp", None),
        ("http://192.168.1.19:8765/mcp", None),
    ],
)
def test_the_key_goes_only_to_this_machine(url, expected):
    """`--url` must not turn a local credential into an outbound one.

    The header is set on the client before anything about the target is known,
    so a typo or a pasted line of configuration was enough to post this
    machine's data-plane key to a stranger, in cleartext. Loopback stays
    allowed: reaching a second install on another port is what the flag is for.
    """
    from qds.mcp_cli import key_for

    assert key_for(url, "http://127.0.0.1:8765/mcp", "k") == expected


def test_no_configured_key_stays_no_key():
    from qds.mcp_cli import key_for

    assert key_for("http://attacker.example/mcp", "http://127.0.0.1:8765/mcp", None) is None


def test_qds_mcp_is_registered_and_imports_lazily():
    """Registration without cost: `qds --help` must not load the SDK."""
    import subprocess
    import sys

    probe = "import sys, qds.cli; qds.cli.main(['--help']); print('mcp' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert "mcp" in out.stdout
    assert out.stdout.strip().endswith("False")


@pytest.mark.parametrize("host,expected", [("0.0.0.0", "127.0.0.1"), ("::", "127.0.0.1")])
def test_a_wildcard_bind_is_not_dialled(monkeypatch, tmp_path, host, expected):
    """A wildcard is an address to listen on, not one to connect to -- the same
    rule `qds status` applies, for the same reason."""
    config = tmp_path / "server-config.json"
    config.write_text(f'{{"server": {{"host": "{host}", "port": 9999}}}}')
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    from qds.mcp_cli import default_url

    url, _key = default_url()
    assert url == f"http://{expected}:9999/mcp"


def test_the_api_key_is_presented_when_one_is_configured(monkeypatch, tmp_path):
    config = tmp_path / "server-config.json"
    config.write_text('{"server": {"api_key": "s3cret", "port": 9999}}')
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    from qds.mcp_cli import default_url

    _url, key = default_url()
    assert key == "s3cret"


def test_the_failure_names_the_real_cause_not_the_task_groups_summary():
    """anyio wraps a failed connection in an ExceptionGroup whose own message is
    "unhandled errors in a TaskGroup (1 sub-exception)". Printed to a user's
    client error panel, that says nothing -- and the advice around it is only
    worth printing if the cause it names is real. This was a live defect: the
    relay reported it verbatim until the unwrap existed."""
    from qds.mcp.bridge import root_cause

    inner = ConnectionRefusedError("All connection attempts failed")
    wrapped = BaseExceptionGroup("unhandled errors in a TaskGroup", [inner])
    nested = BaseExceptionGroup("outer", [wrapped])

    assert root_cause(inner) == "ConnectionRefusedError: All connection attempts failed"
    assert root_cause(wrapped) == "ConnectionRefusedError: All connection attempts failed"
    assert root_cause(nested) == "ConnectionRefusedError: All connection attempts failed"
    assert "TaskGroup" not in root_cause(nested)


def test_a_message_less_exception_still_names_its_type():
    from qds.mcp.bridge import root_cause

    assert root_cause(TimeoutError()) == "TimeoutError"


def test_the_relay_follows_the_redirect_the_mount_produces():
    """`/mcp` is a mount, so the bare path 307s to `/mcp/`. The SDK's own HTTP
    client follows redirects; the one built here -- which it has to build, to
    carry the API key -- does not unless told to. This was a live defect: the
    relay failed against a healthy server and reported "could not reach"."""
    import inspect

    from qds.mcp import bridge

    source = inspect.getsource(bridge.relay)
    assert "follow_redirects=True" in source
