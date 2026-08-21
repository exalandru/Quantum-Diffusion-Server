"""The control plane: who may write the configuration, and who may ask.

Two groups of properties here, and they fail in different ways.

The *write* path has to be all-or-nothing and has to refuse documents that would
leave the server unable to start — a half-written config and a config that names
a disabled default model are both states from which the only repair was editing
JSON by hand.

The *access* path exists because the client became a web page. A keyless install
on loopback was safe while only a native app could reach it; a browser changes
that, and the two checks below are what stand between a hostile page and this
server when there is no API key to present.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from qds import configfile
from qds.app import create_app, create_recovery_app
from qds.settings import ConfigError, Settings

from .conftest import make_client


@pytest.fixture
def configured(monkeypatch, tmp_path):
    """Point the server's configuration at a file the test owns."""
    path = tmp_path / "server-config.json"
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(path))
    return path


VALID = {
    "server": {"port": 8765, "log_file": None},
    "default_model": "z-image-turbo",
    "models": {"z-image-turbo": {"enabled": True}},
}


# ── Reading ────────────────────────────────────────────────────────────────


def test_overview_says_where_everything_lives(client, configured):
    payload = client.get("/admin/overview").json()
    assert payload["configPath"] == str(configured)
    assert payload["dataDir"] == str(configured.parent)
    assert payload["recoveryMode"] is False
    assert payload["server"]["port"] == 8765
    assert "effectiveHfHome" in payload and "effectiveCacheDir" in payload


def test_a_missing_configuration_reads_as_an_empty_document(client, configured):
    """Not as the server's defaults.

    Answering with defaults would mean the first Save wrote every one of them
    out as an explicit key, silently freezing today's defaults into the user's
    file forever.
    """
    assert not configured.exists()
    assert client.get("/admin/config").json() == {}


def test_the_document_is_returned_exactly_as_written(client, configured):
    configured.write_text(json.dumps({"default_model": "z-image", "extra": 1}), encoding="utf-8")
    assert client.get("/admin/config").json() == {"default_model": "z-image", "extra": 1}


# ── Writing ────────────────────────────────────────────────────────────────


def test_a_valid_document_is_written_and_says_a_restart_is_needed(client, configured):
    response = client.put("/admin/config", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    # The running process read its settings once, at start. Claiming otherwise
    # is how an interface comes to show a setting that is not in effect.
    assert body["restartRequired"] is True
    assert configfile.read(configured) == VALID


def test_saving_and_then_refreshing_tells_the_same_story(client, configured):
    """One flag, one meaning.

    `PUT` used to answer `restartRequired: true` from a hard-coded literal while
    `/admin/overview` answered `false` from a different field the write never
    touched — so a dashboard that saves and then refreshes, which is the natural
    order, was told both.
    """
    assert client.get("/admin/overview").json()["restartRequired"] is False

    assert client.put("/admin/config", json=VALID).json()["restartRequired"] is True

    assert client.get("/admin/overview").json()["restartRequired"] is True


def test_disabling_the_default_model_is_refused_and_names_the_fix(client, configured):
    document = {
        "default_model": "z-image-turbo",
        "models": {"z-image-turbo": {"enabled": False}},
    }
    response = client.put("/admin/config", json=document)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "disabled_default_model"
    assert "z-image-turbo" in error["message"]
    assert "another default model" in error["message"]
    assert not configured.exists(), "a refused write must write nothing"


def test_a_document_the_server_could_not_load_is_refused(client, configured):
    response = client.put("/admin/config", json={"server": {"port": 70000}})
    assert response.status_code == 400
    assert not configured.exists()


def test_a_refusal_leaves_the_previous_configuration_byte_identical(client, configured):
    configured.write_text(json.dumps(VALID, indent=2) + "\n", encoding="utf-8")
    before = configured.read_bytes()

    client.put("/admin/config", json={"server": {"port": -1}})

    assert configured.read_bytes() == before


def test_a_write_leaves_no_temporary_file_behind(client, configured):
    """The write goes through a temporary that is renamed into place; a stray
    one would be the next write's problem, not this one's."""
    client.put("/admin/config", json=VALID)
    assert [p.name for p in configured.parent.glob("*.tmp")] == []
    assert configured.is_file()


def test_the_written_file_is_not_world_readable(client, configured):
    """It can hold an API key."""
    client.put("/admin/config", json={**VALID, "server": {**VALID["server"], "api_key": "s3cret"}})
    assert configured.stat().st_mode & 0o077 == 0


def test_a_runtime_issue_is_saved_and_reported_rather_than_refused(client, configured):
    """Savable, because it has to be repairable.

    Binding beyond loopback without an API key is a runtime invariant the
    *server* refuses to start on, and a perfectly well-formed document. Refusing
    to save it would mean the screen that sets the key could not be used to
    reach a state where one exists — so it is written, and the problem comes
    back in the response.
    """
    document = {
        "server": {"host": "0.0.0.0", "api_key": None, "log_file": None},
        "default_model": "z-image-turbo",
        "models": {"z-image-turbo": {"enabled": True}},
    }
    response = client.put("/admin/config", json=document)

    assert response.status_code == 200
    codes = [issue["code"] for issue in response.json()["issues"]]
    assert "unauthenticated_host" in codes
    assert configfile.read(configured) == document


def test_the_token_is_written_private(tmp_path, engine):
    """A credential, so it must never exist world-readable — not even briefly."""
    home = tmp_path / "hf"
    settings = Settings.model_validate(
        {
            "server": {"image_store": str(tmp_path / "images"), "log_file": None},
            "storage": {"hf_home": str(home)},
        }
    )
    with make_client(create_app(settings, engine)) as configured_client:
        response = configured_client.post("/admin/hf-token", json={"token": "  hf_abc123  "})

    assert response.status_code == 200
    token_file = home / "token"
    assert token_file.is_file(), "no token file was written"
    assert token_file.read_text(encoding="utf-8").strip() == "hf_abc123"
    assert token_file.stat().st_mode & 0o077 == 0


def test_an_empty_token_is_refused(client, configured):
    assert client.post("/admin/hf-token", json={"token": "   "}).status_code == 400


# ── Who may ask ────────────────────────────────────────────────────────────


def test_a_host_this_server_does_not_answer_to_is_refused(settings, engine):
    """DNS rebinding, which an API key cannot stop when there is no API key.

    A page on `evil.example` whose name resolves to 127.0.0.1 is same-origin to
    the browser and may read the responses. The one thing it cannot change is
    the `Host` header carrying the name that was dialled.
    """
    with TestClient(create_app(settings, engine), base_url="http://evil.example") as rebound:
        response = rebound.get("/health")
    assert response.status_code == 421
    assert response.json()["error"]["code"] == "host_not_allowed"


def test_loopback_hosts_are_answered(settings, engine):
    for base in ("http://127.0.0.1", "http://localhost", "http://127.0.0.1:8765"):
        with TestClient(create_app(settings, engine), base_url=base) as dialled:
            assert dialled.get("/health").status_code == 200, base


def test_a_cross_site_write_is_refused(client, configured):
    """CSRF: a form post from another page carries an Origin it cannot forge."""
    response = client.put(
        "/admin/config", json=VALID, headers={"Origin": "http://evil.example"}
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "cross_site_denied"
    assert not configured.exists()


def test_a_same_origin_write_is_accepted(client, configured):
    response = client.put("/admin/config", json=VALID, headers={"Origin": "http://127.0.0.1"})
    assert response.status_code == 200


def test_a_request_with_no_origin_is_accepted(client, configured):
    """curl, the menubar app and the CLI send none, and are not browsers."""
    assert client.put("/admin/config", json=VALID).status_code == 200


@pytest.mark.parametrize(
    "route", ["/admin/config", "/admin/overview", "/admin/models", "/admin/logs", "/admin/jobs"]
)
def test_a_cross_site_read_is_refused_too(client, configured, route):
    """The check covers reads, and the reason is `cors_origins` defaulting to `*`.

    An earlier version guarded writes only, reasoning that CORS stopped
    cross-site reads. It does not — the wildcard is deliberate, so that an
    OpenAI-speaking front end on another origin can call `/v1` — and it left any
    page the user visited able to read this server's config path, its effective
    HuggingFace home, whether a token is present, and its whole log buffer.
    """
    response = client.get(route, headers={"Origin": "http://evil.example"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "cross_site_denied"


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example//127.0.0.1",  # suffix match, if compared as a string
        "https://127.0.0.1",  # right host, wrong scheme
        "null",  # what a sandboxed iframe sends
        "http://127.0.0.1.evil.example",
        "",
    ],
)
def test_an_origin_that_only_looks_like_ours_is_refused(client, configured, origin):
    """The check parses rather than pattern-matches, so it fails closed."""
    response = client.get("/admin/overview", headers={"Origin": origin})
    assert response.status_code == 403, origin


def test_the_data_plane_key_does_not_open_the_control_plane(tmp_path, engine, monkeypatch):
    """**This assertion is the inverse of the one it replaces, on purpose.**

    The previous test asserted that `Bearer <api_key>` returned 200 from
    `/admin/overview` — i.e. that the data-plane key was *sufficient* for the
    control plane. That was a deliberate decision at the time, and it is now a
    deliberate reversal: the key exists to be handed to other software so it can
    generate images, and handing it over should not also hand over the
    configuration writer, the log buffer and the restart button.

    A reviewer should be able to tell an intended inversion from a test bent to
    fit new code. The difference is this paragraph.
    """
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    from qds import credential

    credential.set_password("correct horse battery")

    settings = Settings.model_validate(
        {
            "server": {
                "api_key": "cle-secrete",
                "image_store": str(tmp_path / "images"),
                "log_file": None,
            }
        }
    )
    with make_client(create_app(settings, engine)) as secured:
        assert secured.get("/admin/overview").status_code == 401
        assert secured.put("/admin/config", json=VALID).status_code == 401

        with_key = secured.get("/admin/overview", headers={"Authorization": "Bearer cle-secrete"})
        assert with_key.status_code == 401
        assert with_key.json()["error"]["code"] == "admin_login_required"

        # The key still opens the plane it is for.
        assert secured.get("/v1/models", headers={"Authorization": "Bearer cle-secrete"}).status_code == 200

        # And a session opens the one it is for.
        secured.post("/admin/session", json={"password": "correct horse battery"})
        assert secured.get("/admin/overview").status_code == 200


# ── Logs ───────────────────────────────────────────────────────────────────


def test_logs_are_returned_after_a_cursor(client):
    first = client.get("/admin/logs").json()
    assert first["entries"], "the startup line should be in the buffer"
    assert first["lastSeq"] >= 1

    # Asking again from where we stopped returns nothing new, rather than the
    # same lines a second time.
    again = client.get("/admin/logs", params={"after": first["lastSeq"]}).json()
    assert again["entries"] == []
    assert again["lastSeq"] == first["lastSeq"]


def test_a_structured_event_keeps_its_fields_in_the_buffer(client):
    import logging

    logging.getLogger("qds.test").info(
        "converting", extra={"event": "prequantize_progress", "fields": {"block": 2}}
    )
    entries = client.get("/admin/logs").json()["entries"]
    matching = [entry for entry in entries if entry.get("event") == "prequantize_progress"]
    assert matching and matching[-1]["fields"] == {"block": 2}


# ── Jobs ───────────────────────────────────────────────────────────────────


def test_cancelling_with_nothing_running_is_a_conflict_not_a_success(client):
    response = client.post("/admin/jobs/cancel")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "no_job_running"


def test_the_job_status_starts_idle(client):
    payload = client.get("/admin/jobs").json()
    assert payload["state"] == "idle"
    assert payload["kind"] is None


def test_starting_and_reading_a_job_return_the_same_shape(client, monkeypatch):
    """One resource, one payload — whichever verb reached it."""
    import asyncio

    async def refuse(*_command, **_kwargs):
        raise _NoSpawn

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    with pytest.raises(_NoSpawn):
        client.post("/admin/jobs/fetch", json={"key": "z-image"})

    assert set(client.get("/admin/jobs").json()) == {
        "state",
        "kind",
        "target",
        "event",
        "fields",
        "message",
        "startedAtMs",
        "finishedAtMs",
    }


class _NoSpawn(Exception):
    """Stops a job before a real child is launched."""


# ── Restart ────────────────────────────────────────────────────────────────


def test_restart_is_refused_when_nobody_can_perform_it(client):
    """An app embedded in a test client owns no server to stop."""
    response = client.post("/admin/restart")
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "restart_unavailable"


def test_restart_reaches_the_process_that_owns_the_server(settings, engine):
    asked: list[bool] = []
    app = create_app(settings, engine, request_restart=lambda: asked.append(True))
    with make_client(app) as restartable:
        assert restartable.post("/admin/restart").status_code == 202
    assert asked == [True]


# ── Recovery mode ──────────────────────────────────────────────────────────


def test_a_broken_configuration_still_serves_the_control_plane(tmp_path, monkeypatch):
    """The trap this exists to avoid: the screen that repairs the configuration
    was served by the process the configuration stopped from starting."""
    path = tmp_path / "server-config.json"
    path.write_text(json.dumps({"default_model": "z-image-turbo"}), encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(path))
    settings = Settings.model_validate({"server": {"log_file": None}})

    app = create_recovery_app(settings, "z-image-turbo is disabled")
    with make_client(app) as recovering:
        health = recovering.get("/health").json()
        assert health["status"] == "config_error"
        assert "disabled" in health["error"]

        overview = recovering.get("/admin/overview").json()
        assert overview["recoveryMode"] is True
        assert overview["recoveryError"] == "z-image-turbo is disabled"

        # The repair path itself works.
        assert recovering.get("/admin/config").json() == {"default_model": "z-image-turbo"}
        saved = recovering.put("/admin/config", json=VALID)
        assert saved.status_code == 200
        assert configfile.read(path) == VALID


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("malformed JSON", '{"server": {'),
        ("out-of-range value", '{"models": {"z-image": {"quantize": 99}}}'),
        ("wrong type", '{"models": "not-an-object"}'),
        ("not an object", "[1, 2, 3]"),
    ],
)
def test_a_structurally_broken_configuration_still_reaches_a_repair_screen(
    tmp_path, monkeypatch, name, text
):
    """The failures people actually produce by hand-editing JSON.

    Recovery mode originally covered only *runtime* invariants — a disabled
    default model got a repair screen, while a typo or an out-of-range number
    killed the process outright and left no way back in except editing the same
    file again by hand, which is the exact trap the mode exists to prevent.
    """
    path = tmp_path / "server-config.json"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(path))

    from qds.settings import load_settings, recovery_settings

    with pytest.raises((ConfigError, ValueError)):
        load_settings()
    # Whatever the file says, there is always something to listen on.
    settings = recovery_settings()

    with make_client(create_recovery_app(settings, f"broken: {name}")) as recovering:
        assert recovering.get("/health").json()["status"] == "config_error"
        # The text is handed back, because a JSON error names a line and a
        # column and neither means anything without the line.
        document = recovering.get("/admin/config").json()
        if document.get("unparsed"):
            assert document["text"] == text
        # And the repair lands.
        assert recovering.put("/admin/config", json=VALID).status_code == 200
        assert configfile.read(path) == VALID


def test_the_recovery_server_listens_where_it_was_told_to(monkeypatch, tmp_path):
    """A repair screen nobody can reach is no better than no server at all."""
    from qds.settings import recovery_settings

    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "broken.json"))
    monkeypatch.setenv("QDS_SERVER_PORT", "9321")
    assert recovery_settings().server.port == 9321


def test_recovery_mode_refuses_to_generate_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    settings = Settings.model_validate({"server": {"log_file": None}})
    app = create_recovery_app(settings, "no model enabled")

    with make_client(app) as recovering:
        response = recovering.post("/v1/images/generations", json={"prompt": "a fox"})
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "config_error"
        assert "no model enabled" in error["message"]


def test_the_catalogue_is_readable_in_recovery_mode(tmp_path, monkeypatch):
    """Model management has to work from the state that needs repairing."""
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    settings = Settings.model_validate({"server": {"log_file": None}})

    with make_client(create_recovery_app(settings, "broken")) as recovering:
        payload = recovering.get("/admin/models").json()
    assert len(payload["models"]) > 5


# ── Dashboard ──────────────────────────────────────────────────────────────


def test_the_dashboard_is_served_when_it_has_been_built(client):
    """The whole point of shipping it inside the wheel."""
    from qds.app import DASHBOARD_DIR

    if not (DASHBOARD_DIR / "index.html").is_file():
        pytest.skip("run `make build-dashboard` to exercise this")

    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Built with `base: "/dashboard/"`; without it the page loads and then asks
    # for `/assets/…`, which this server answers with a 404.
    assert "/dashboard/assets/" in response.text


def test_an_unbuilt_dashboard_names_the_command_that_builds_it(settings, engine, monkeypatch, tmp_path):
    """A 404 would read as "wrong URL"; the diagnosis is "not built".

    Pointed at an empty directory rather than skipped when the checkout happens
    to have been built: a test that only runs on machines where it cannot fail
    is not a test.
    """
    monkeypatch.setattr("qds.app.DASHBOARD_DIR", tmp_path / "never-built")
    with make_client(create_app(settings, engine)) as unbuilt:
        response = unbuilt.get("/dashboard")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "dashboard_not_built"
    assert "build-dashboard" in response.json()["error"]["message"]


def test_the_import_library_is_listed_through_the_api(client, tmp_path, monkeypatch):
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    payload = client.get("/admin/import").json()
    assert payload["ok"] is True
    assert payload["models"] == []
