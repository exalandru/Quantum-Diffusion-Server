"""What `/mcp` refuses, and the boundaries it does not widen.

Everything about the *guard* goes over HTTP, deliberately. An in-memory client
talks to the `MCPServer` object and never passes through the mounted
application's guard at all -- a security test written that way would pass
against a server with no guard, which makes it an oracle that cannot fail.

Everything about the *trust boundary* -- which files a model-chosen argument may
reach -- is exercised through the tools, because that is where the argument
arrives.
"""

from __future__ import annotations

import json

import pytest

from qds.app import create_app

from .conftest import FakeEngine, make_client, tiny_png
from .mcp_support import mcp_session, mcp_settings, text_of

# A well-formed `initialize`, so a refusal is the guard's and not the parser's.
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "witness", "version": "0"},
    },
}
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def app_with(tmp_path, **server):
    settings = mcp_settings(tmp_path, server=server)
    return create_app(settings, FakeEngine())


def post(client, **headers):
    return client.post("/mcp", content=json.dumps(INITIALIZE), headers={**MCP_HEADERS, **headers})


# ── Authorization: the same rule as /v1, not a second one ──────────────────


def test_mcp_is_open_when_the_data_plane_is(tmp_path):
    """A loopback install with no key generates an image without ceremony, and
    that has to be true on this plane too or the default is a lie."""
    with make_client(app_with(tmp_path)) as client:
        assert post(client).status_code == 200


def test_a_configured_key_gates_mcp(tmp_path):
    with make_client(app_with(tmp_path, api_key="s3cret")) as client:
        refused = post(client)
        assert refused.status_code == 401
        assert refused.json()["error"]["code"] == "invalid_api_key"


def test_the_configured_key_opens_mcp(tmp_path):
    """The counterfactual for the test above: without it, a guard that refused
    everything would look correct."""
    with make_client(app_with(tmp_path, api_key="s3cret")) as client:
        assert post(client, Authorization="Bearer s3cret").status_code == 200


def test_the_local_token_opens_mcp(tmp_path):
    """The menubar app and the CLI hold this one, not the API key."""
    settings = mcp_settings(tmp_path, server={"api_key": "s3cret"})
    app = create_app(settings, FakeEngine(), local_token="local-token")
    with make_client(app) as client:
        assert post(client, **{"X-QDS-Admin-Token": "local-token"}).status_code == 200


def test_the_refusal_is_shaped_like_every_other_refusal(tmp_path):
    """A mounted application is outside FastAPI's exception handlers, so the
    guard renders its own -- and it must render the same envelope, or a client
    that understands this server's errors stops understanding one plane's."""
    with make_client(app_with(tmp_path, api_key="s3cret")) as client:
        body = post(client).json()
    assert set(body["error"]) == {"message", "type", "param", "code"}


# ── Containment: the host guard first, the browser rule after ──────────────


def test_the_host_guard_still_runs_in_front_of_mcp(tmp_path):
    """`install_host_guard` wraps every route including this mount, which is why
    the SDK's own DNS-rebinding allowlist is switched off: one allowlist."""
    with make_client(app_with(tmp_path)) as client:
        refused = client.post(
            "/mcp", content=json.dumps(INITIALIZE), headers={**MCP_HEADERS, "Host": "evil.example"}
        )
    assert refused.status_code == 421
    assert refused.json()["error"]["code"] == "host_not_allowed"


def test_a_cross_site_origin_is_refused(tmp_path):
    """MCP writes durable playground state and spends the GPU. With
    `cors_origins` defaulting to `["*"]` and no key on loopback, without this a
    page in any tab could do both."""
    with make_client(app_with(tmp_path)) as client:
        refused = post(client, Origin="http://evil.example")
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "cross_site_denied"


def test_a_same_origin_request_is_not_refused(tmp_path):
    with make_client(app_with(tmp_path)) as client:
        assert post(client, Origin="http://127.0.0.1").status_code == 200


def test_a_client_that_sends_no_origin_is_unaffected(tmp_path):
    """Which is every chat client. The rule is about browsers, and only fires
    on the header a browser sets."""
    with make_client(app_with(tmp_path)) as client:
        assert post(client).status_code == 200


# ── The route's existence is a setting ─────────────────────────────────────


def test_mcp_is_absent_when_it_is_switched_off(tmp_path):
    settings = mcp_settings(tmp_path)
    settings.mcp.enabled = False
    with make_client(create_app(settings, FakeEngine())) as client:
        assert post(client).status_code == 404


def test_mcp_is_absent_from_the_recovery_app(tmp_path):
    """Recovery mode has no registry, no engine and no playground store, so
    every tool could only fail. A 404 is the honest answer, and `/health`
    already publishes why the server is in that state."""
    from qds.app import create_recovery_app

    settings = mcp_settings(tmp_path)
    with make_client(create_recovery_app(settings, "broken")) as client:
        assert post(client).status_code == 404


def test_mcp_answers_before_the_application_has_started(tmp_path):
    """Without the lifespan the SDK's transport raises "Task group is not
    initialized" as a 500. The mount answers 503 instead, which says what is
    actually wrong."""
    app = app_with(tmp_path)
    with make_client(app) as client:
        pass  # entering and leaving the lifespan stops the transport
    response = client.post("/mcp", content=json.dumps(INITIALIZE), headers=MCP_HEADERS)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "mcp_not_running"


# ── The trust boundary for a path a model chose ────────────────────────────


async def test_a_filesystem_path_is_refused_when_no_root_is_configured(tmp_path, monkeypatch):
    """Fail closed on the shipped configuration, and *without reading the file*.

    The monkeypatch is the real assertion: a refusal that happens after the
    bytes were opened has already done the thing it was meant to prevent.
    """
    outside = tmp_path / "private.png"
    outside.write_bytes(tiny_png())

    from PIL import Image as PILImage

    def refuse_to_open(*args, **kwargs):
        raise AssertionError("the file was opened despite being outside every root")

    monkeypatch.setattr(PILImage, "open", refuse_to_open)

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool(
            "generate_image", {"prompt": "a cube", "reference_path": str(outside)}
        )
    assert result.is_error is True
    assert "mcp.image_roots" in text_of(result)


async def test_a_filesystem_path_inside_a_configured_root_is_accepted(tmp_path):
    """The counterfactual: without it, a boundary that refused everything --
    including what an operator explicitly allowed -- would look correct."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "reference.png"
    source.write_bytes(tiny_png())

    settings = mcp_settings(tmp_path)
    settings.mcp.image_roots = [str(allowed)]
    async with mcp_session(tmp_path, settings=settings) as (client, _app, engine):
        result = await client.call_tool(
            "generate_image",
            {"prompt": "a cube", "model": "qwen-image-2512", "reference_path": str(source)},
        )
    assert result.is_error is False, text_of(result)
    assert engine.jobs[0].image_path is not None
    assert engine.jobs[0].image_path != source, "the row owns a copy, not the original"


async def test_a_symlink_out_of_a_root_is_refused(tmp_path):
    """Containment is a question about the file, not about the name -- and a
    symlink is exactly how a model would make those two differ."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret.png"
    secret.write_bytes(tiny_png())
    (allowed / "innocent.png").symlink_to(secret)

    settings = mcp_settings(tmp_path)
    settings.mcp.image_roots = [str(allowed)]
    async with mcp_session(tmp_path, settings=settings) as (client, _app, _engine):
        result = await client.call_tool(
            "generate_image",
            {"prompt": "a cube", "reference_path": str(allowed / "innocent.png")},
        )
    assert result.is_error is True
    assert "outside every directory" in text_of(result)


async def test_a_relative_path_is_refused(tmp_path):
    settings = mcp_settings(tmp_path)
    settings.mcp.image_roots = [str(tmp_path)]
    async with mcp_session(tmp_path, settings=settings) as (client, _app, _engine):
        result = await client.call_tool(
            "generate_image", {"prompt": "a cube", "reference_path": "../../etc/passwd"}
        )
    assert result.is_error is True
    assert "absolute" in text_of(result)


@pytest.mark.parametrize("name", ["../../../etc/passwd", "..%2Fsecret.png", "nope.png"])
async def test_a_reference_image_that_matches_no_row_is_a_404(tmp_path, name):
    """The traversal guard is that a *row* must match before a path is built --
    the same guard the upscale route relies on. There is no string to sanitise,
    which is why there is nothing here to get wrong."""
    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube", "reference_image": name})
    assert result.is_error is True
    assert "No image" in text_of(result)


async def test_the_two_ways_of_naming_a_reference_are_refused_together(tmp_path):
    """Two answers to one question. Quietly preferring one would make a model's
    mistake invisible to it."""
    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool(
            "generate_image",
            {"prompt": "a cube", "reference_image": "a.png", "reference_path": "/tmp/b.png"},
        )
    assert result.is_error is True
    assert "not both" in text_of(result)


# ── A locked session is unreachable, with no way to unlock one ─────────────


async def test_a_locked_session_is_refused_and_offers_no_way_through(tmp_path):
    """MCP carries no unlock token, and this is what makes that a guarantee.

    The password is set the way a person sets one -- through the playground
    route -- rather than by writing the record directly, so this witnesses the
    real interaction between the two planes rather than a fixture's idea of it.
    The `TestClient` is used *without* its context manager on purpose: the
    application's lifespan is already running, and entering a second one would
    restart the runner underneath the client we are holding.
    """
    async with mcp_session(tmp_path) as (client, app, _engine):
        first = await client.call_tool("generate_image", {"prompt": "a cube"})
        session_id = text_of(first).split("session: ")[1].split()[0]

        http = make_client(app)
        locked = http.post(
            f"/playground/api/sessions/{session_id}/password",
            json={"password": "a good password"},
        )
        assert locked.status_code == 200, locked.text

        refused = await client.call_tool("generate_image", {"prompt": "a cube", "session_id": session_id})
        reading = await client.call_tool("wait_for_generation", {"generation_id": "x"})

    body = text_of(refused)
    assert refused.is_error is True
    assert "password-protected" in body
    assert "playground" in body
    # No argument anywhere offers a way past it.
    assert "token" not in body and "password=" not in body
    assert reading.is_error is True


async def test_a_locked_session_hides_its_images_from_the_resource_too(tmp_path):
    """The resource is another door onto one library, not a wider one.

    Note what is asserted and what is not. The read is *refused* and no bytes
    come back -- that is the containment, and it is the property that matters.
    The refusal's *reason* does not survive: the SDK's resource-template layer
    catches whatever a template raises and reports "Error creating resource
    from template", discarding the message. Tools do carry their message (see
    the test above), and this asymmetry is documented rather than worked around
    -- a resource is host-mediated, so unlike a tool result nothing is reading
    the text in order to correct itself.
    """
    from .mcp_support import filenames_of

    async with mcp_session(tmp_path) as (client, app, _engine):
        made = await client.call_tool("generate_image", {"prompt": "a cube"})
        filename = filenames_of(made)[0]
        session_id = text_of(made).split("session: ")[1].split()[0]

        base = mcp_settings(tmp_path)
        uri = f"http://{base.server.host}:{base.server.port}/playground/images/{filename}"
        readable = await client.read_resource(uri)
        assert readable.contents, "an unlocked session's image is readable"

        make_client(app).post(
            f"/playground/api/sessions/{session_id}/password",
            json={"password": "a good password"},
        )
        with pytest.raises(Exception) as refusal:
            await client.read_resource(uri)
    # No bytes reached the caller; the exception carries the SDK's generic text.
    assert not hasattr(refusal.value, "contents")
    assert filename in str(refusal.value)


# ── The URL that is documented is the URL that works ───────────────────────


def test_both_the_bare_and_slashed_paths_reach_the_surface(tmp_path):
    """`/mcp` is what the documentation publishes and what people paste into a
    client's configuration. It is a Starlette mount, so the bare path is
    answered with a 307 to `/mcp/` -- which preserves method and body, and
    which every client tested follows. Pinned here so a change that leaves the
    advertised URL dead is caught rather than discovered by a user."""
    with make_client(app_with(tmp_path)) as client:
        assert post(client).status_code == 200, "the documented URL"
        assert client.post("/mcp/", content=json.dumps(INITIALIZE), headers=MCP_HEADERS).status_code == 200


def test_the_redirect_does_not_let_a_cross_site_request_past(tmp_path):
    """A redirect that happened *before* the guard would be a hole: the browser
    re-sends with `Origin` intact, so the check must still fire on the second
    hop. `follow_redirects` is explicit here because that is the whole point."""
    with make_client(app_with(tmp_path)) as client:
        refused = client.post(
            "/mcp",
            content=json.dumps(INITIALIZE),
            headers={**MCP_HEADERS, "Origin": "http://evil.example"},
            follow_redirects=True,
        )
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "cross_site_denied"
