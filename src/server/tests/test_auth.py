"""Who may reach the control plane.

Three credentials, three audiences: a session for the person at a browser, a
local token for the menubar app and the CLI, and — during the transition — the
API key. Every positive here is paired with the negative that would catch a
check which simply returns "allowed".

The recovery app is exercised alongside the normal one wherever the rule is
supposed to be the same, because the duplicate `require_auth` these replaced is
exactly how the two came to differ, in the one state where being locked out is
unrecoverable.
"""

from __future__ import annotations

import pytest

from qds import credential, session
from qds.app import create_app, create_recovery_app
from qds.session import SessionStore
from qds.settings import Settings

from .conftest import FakeEngine, make_client

PASSWORD = "correct horse battery"


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    return tmp_path


def settings_for(tmp_path, **server) -> Settings:
    return Settings.model_validate(
        {"server": {"image_store": str(tmp_path / "images"), "log_file": None, **server}}
    )


def normal_app(tmp_path, *, token=None, **server):
    return create_app(settings_for(tmp_path, **server), FakeEngine(), local_token=token)


def recovery_app(tmp_path, *, token=None, **server):
    return create_recovery_app(settings_for(tmp_path, **server), "broken", local_token=token)


APPS = [normal_app, recovery_app]


# ── The session store ──────────────────────────────────────────────────────


def test_a_created_session_validates():
    store = SessionStore()
    assert store.validate(store.create()) is True


def test_a_token_the_store_never_issued_does_not_validate():
    """The negative without which the test above proves nothing."""
    store = SessionStore()
    store.create()
    assert store.validate("not-a-real-token") is False
    assert store.validate("") is False
    assert store.validate(None) is False


def test_revoking_ends_exactly_one_session():
    store = SessionStore()
    first, second = store.create(), store.create()
    store.revoke(first)
    assert store.validate(first) is False
    assert store.validate(second) is True


def test_changing_the_password_ends_every_session():
    store = SessionStore()
    tokens = [store.create() for _ in range(3)]
    store.revoke_all()
    assert not any(store.validate(token) for token in tokens)


def test_an_idle_session_expires(monkeypatch):
    store = SessionStore()
    token = store.create()
    monkeypatch.setattr(session, "IDLE_TIMEOUT_S", -1)
    assert store.validate(token) is False


def test_a_session_does_not_outlive_its_absolute_lifetime(monkeypatch):
    store = SessionStore()
    token = store.create()
    monkeypatch.setattr(session, "ABSOLUTE_LIFETIME_S", -1)
    assert store.validate(token) is False


def test_the_store_does_not_grow_without_bound(monkeypatch):
    monkeypatch.setattr(session, "MAX_SESSIONS", 4)
    store = SessionStore()
    for _ in range(20):
        store.create()
    assert len(store) <= 4


# ── The local token ────────────────────────────────────────────────────────


def test_the_token_file_is_not_world_readable(configured):
    session.issue_local_token()
    assert session.token_path().stat().st_mode & 0o077 == 0


def test_each_start_issues_a_different_token(configured):
    """A token read from a previous run must stop working."""
    first = session.issue_local_token()
    second = session.issue_local_token()
    assert first and second and first != second


def test_an_unwritable_directory_yields_no_token_rather_than_failing(tmp_path, monkeypatch):
    """A packaged install with a read-only config directory still serves."""
    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(unwritable / "server-config.json"))
    try:
        assert session.issue_local_token() is None
    finally:
        unwritable.chmod(0o700)


# ── Reaching /admin ────────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", APPS)
def test_with_no_password_the_control_plane_is_open(configured, factory):
    """The first-run state: the dashboard must be able to set a password."""
    with make_client(factory(configured)) as client:
        assert client.get("/admin/session").json()["passwordSet"] is False
        assert client.get("/admin/overview").status_code == 200


@pytest.mark.parametrize("factory", APPS)
def test_with_a_password_set_the_control_plane_demands_one(configured, factory):
    credential.set_password(PASSWORD)
    with make_client(factory(configured)) as client:
        response = client.get("/admin/overview")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "admin_login_required"


@pytest.mark.parametrize("factory", APPS)
def test_logging_in_opens_the_control_plane(configured, factory):
    credential.set_password(PASSWORD)
    with make_client(factory(configured)) as client:
        assert client.get("/admin/overview").status_code == 401

        login = client.post("/admin/session", json={"password": PASSWORD})
        assert login.status_code == 204
        assert client.get("/admin/overview").status_code == 200


@pytest.mark.parametrize("factory", APPS)
def test_the_wrong_password_sets_no_cookie(configured, factory):
    credential.set_password(PASSWORD)
    with make_client(factory(configured)) as client:
        response = client.post("/admin/session", json={"password": "wrong"})
        assert response.status_code == 401
        assert "set-cookie" not in {k.lower() for k in response.headers}
        assert client.get("/admin/overview").status_code == 401


def test_the_session_cookie_is_not_readable_by_scripts(configured):
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured)) as client:
        response = client.post("/admin/session", json={"password": PASSWORD})
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie
        # Strict, not Lax: `/v1` has permissive CORS and no Origin check, and a
        # session that also opens `/v1` must not ride a cross-site request.
        assert "samesite=strict" in cookie


def test_logging_out_ends_the_session(configured):
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured)) as client:
        client.post("/admin/session", json={"password": PASSWORD})
        assert client.get("/admin/overview").status_code == 200

        assert client.delete("/admin/session").status_code == 204
        assert client.get("/admin/overview").status_code == 401


def test_a_session_from_another_process_is_refused(configured):
    """Restarting ends every session, which is what an in-memory store buys."""
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured)) as first:
        first.post("/admin/session", json={"password": PASSWORD})
        stolen = first.cookies.get(session.COOKIE)
    assert stolen

    with make_client(normal_app(configured)) as second:
        second.cookies.set(session.COOKIE, stolen)
        assert second.get("/admin/overview").status_code == 401


def test_repeated_failures_are_throttled(configured):
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured)) as client:
        codes = [
            client.post("/admin/session", json={"password": "wrong"}).status_code
            for _ in range(6)
        ]
        assert 429 in codes, codes
        # And the throttle does not let the right password through either.
        assert client.post("/admin/session", json={"password": PASSWORD}).status_code == 429


def test_logging_in_is_not_reachable_from_another_origin(configured):
    """A hostile page must not be able to guess from the visitor's browser."""
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured)) as client:
        response = client.post(
            "/admin/session",
            json={"password": PASSWORD},
            headers={"Origin": "http://evil.example"},
        )
        assert response.status_code == 403


# ── The local token as a credential ────────────────────────────────────────


@pytest.mark.parametrize("factory", APPS)
def test_the_local_token_opens_the_control_plane(configured, factory):
    credential.set_password(PASSWORD)
    token = "a-local-token"
    with make_client(factory(configured, token=token)) as client:
        assert client.get("/admin/overview").status_code == 401
        authorised = client.get("/admin/overview", headers={"X-QDS-Admin-Token": token})
        assert authorised.status_code == 200


def test_a_wrong_local_token_does_not(configured):
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured, token="a-local-token")) as client:
        response = client.get("/admin/overview", headers={"X-QDS-Admin-Token": "guessed"})
        assert response.status_code == 401


def test_the_local_token_is_not_an_api_key(configured):
    """Two credentials, two meanings: neither is accepted in the other's place."""
    credential.set_password(PASSWORD)
    token = "a-local-token"
    with make_client(normal_app(configured, token=token, api_key="the-api-key")) as client:
        # The token in the API key's header opens nothing.
        assert (
            client.get("/admin/overview", headers={"Authorization": f"Bearer {token}"}).status_code
            == 401
        )
        # And the API key in the token's header likewise.
        assert (
            client.get("/admin/overview", headers={"X-QDS-Admin-Token": "the-api-key"}).status_code
            == 401
        )


# ── Changing the password ──────────────────────────────────────────────────


def test_the_first_password_can_be_set_without_one(configured):
    with make_client(normal_app(configured)) as client:
        assert client.post("/admin/password", json={"new": PASSWORD}).status_code == 200
        assert credential.verify(PASSWORD) is True


def test_changing_it_requires_the_current_one(configured):
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured, api_key=None)) as client:
        client.post("/admin/session", json={"password": PASSWORD})

        refused = client.post("/admin/password", json={"new": "a new password", "current": "no"})
        assert refused.status_code == 401
        # And the stored password is untouched.
        assert credential.verify(PASSWORD) is True

        accepted = client.post(
            "/admin/password", json={"new": "a new password", "current": PASSWORD}
        )
        assert accepted.status_code == 200
        assert credential.verify("a new password") is True


def test_changing_it_ends_every_session(configured):
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured)) as client:
        client.post("/admin/session", json={"password": PASSWORD})
        client.post("/admin/password", json={"new": "a new password", "current": PASSWORD})
        # The session was minted against a password that no longer exists.
        assert client.get("/admin/overview").status_code == 401


def test_a_weak_password_is_refused(configured):
    with make_client(normal_app(configured)) as client:
        response = client.post("/admin/password", json={"new": "short"})
        assert response.status_code == 400
        assert credential.is_set() is False


def test_the_password_cannot_be_removed_while_listening_beyond_this_machine(configured):
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured, host="0.0.0.0", api_key="k")) as client:
        client.post("/admin/session", json={"password": PASSWORD})
        response = client.delete("/admin/password")
        assert response.status_code == 409
        assert credential.is_set() is True


def test_it_can_be_removed_on_loopback(configured):
    credential.set_password(PASSWORD)
    with make_client(normal_app(configured)) as client:
        client.post("/admin/session", json={"password": PASSWORD})
        assert client.delete("/admin/password").status_code == 200
        assert credential.is_set() is False
