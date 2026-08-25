"""When each plane's gate applies, and what opens the playground.

Two settings, two credentials, two planes. The file exists because the defect a
careless implementation produces is invisible to a suite that tests the planes
one at a time: one shared flag, or one authorizer that cannot tell `/admin` from
`/playground/api`, moves both gates whenever either scope changes. Only the
asymmetric cell of the matrix below catches that — and it is the posture this
feature was asked for, `admin_auth_scope: always` with
`playground_auth_scope: network`.

Every positive here is paired with the negative that would catch a check which
simply returns "allowed", and the invariant the scopes may not break — that
neither value opens anything the server closes today — has tests of its own.
"""

from __future__ import annotations

import pytest

from qds import admin, credential, session
from qds.app import create_app, create_recovery_app
from qds.settings import Settings

from .conftest import FakeEngine, make_client

ADMIN_PASSWORD = "correct horse battery"
PLAYGROUND_PASSWORD = "open the picture door"
KEY = "a-long-random-key"


@pytest.fixture
def configured(monkeypatch, tmp_path):
    """Every credential file this test writes lands under `tmp_path`."""
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    return tmp_path


def settings_for(tmp_path, **server) -> Settings:
    return Settings.model_validate(
        {"server": {"image_store": str(tmp_path / "images"), "log_file": None, **server}}
    )


def app_for(tmp_path, *, token=None, **server):
    return create_app(settings_for(tmp_path, **server), FakeEngine(), local_token=token)


def codes(raw: dict) -> list[str]:
    return [issue.code for issue in Settings.model_validate(raw).runtime_issues()]


# ── Step 0: the settings, and refusing to serve a keyless gate ──────────────


def test_both_scopes_default_to_the_behaviour_that_shipped():
    server = Settings.model_validate({}).server
    assert (server.admin_auth_scope, server.playground_auth_scope) == ("network", "network")
    # The default posture on loopback: neither gate binds.
    assert (server.admin_gate_binds, server.playground_gate_binds) == (False, False)


def test_a_scope_only_accepts_the_two_values_it_has():
    with pytest.raises(ValueError):
        Settings.model_validate({"server": {"playground_auth_scope": "sometimes"}})


def test_network_still_binds_once_the_socket_leaves_this_machine():
    """The half of `network` that is not "off": it is "off *here*"."""
    server = Settings.model_validate({"server": {"host": "0.0.0.0"}}).server
    assert (server.admin_gate_binds, server.playground_gate_binds) == (True, True)


def test_always_binds_on_loopback_too():
    server = Settings.model_validate(
        {"server": {"admin_auth_scope": "always", "playground_auth_scope": "always"}}
    ).server
    assert (server.admin_gate_binds, server.playground_gate_binds) == (True, True)


def test_a_gate_set_to_always_without_its_password_refuses_to_serve(configured):
    """D4, per plane: the issue names the field of the plane that is wrong.

    Asserted as the whole list rather than by membership, because the failure
    mode being excluded is the *other* plane's issue appearing as well: a
    playground misconfiguration reported against `server.admin_auth_scope` sends
    whoever repairs it to the wrong screen.
    """
    assert codes({"server": {"admin_auth_scope": "always"}}) == ["admin_password_required_by_scope"]
    assert codes({"server": {"playground_auth_scope": "always"}}) == ["playground_password_required_by_scope"]

    issues = Settings.model_validate({"server": {"playground_auth_scope": "always"}}).runtime_issues()
    assert issues[0].field == "server.playground_auth_scope"

    # One plane configured correctly does not excuse the other, and does not
    # accuse it either.
    credential.set_password(ADMIN_PASSWORD)
    assert codes({"server": {"admin_auth_scope": "always", "playground_auth_scope": "always"}}) == [
        "playground_password_required_by_scope"
    ]
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    assert codes({"server": {"admin_auth_scope": "always", "playground_auth_scope": "always"}}) == []


def test_the_default_scope_asks_for_nothing_at_startup(configured):
    """`network` on loopback with no password anywhere: unchanged, and silent."""
    assert codes({}) == []


# ── Step 1: the playground's own credential file ────────────────────────────


def test_the_playground_password_round_trips(configured):
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    assert credential.PLAYGROUND.is_set() is True
    assert credential.PLAYGROUND.verify(PLAYGROUND_PASSWORD) is True
    assert credential.PLAYGROUND.verify("open the picture doo") is False
    assert credential.PLAYGROUND.verify("") is False


def test_the_playground_credential_is_not_the_admin_one(configured):
    """D1: two files, and neither verifies the other's password."""
    credential.set_password(ADMIN_PASSWORD)
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)

    assert credential.PLAYGROUND.path() != credential.credential_path()
    assert credential.PLAYGROUND.path().name == "playground-credential.json"
    assert credential.verify(PLAYGROUND_PASSWORD) is False
    assert credential.PLAYGROUND.verify(ADMIN_PASSWORD) is False

    # Clearing one leaves the other alone.
    credential.PLAYGROUND.clear()
    assert credential.PLAYGROUND.is_set() is False
    assert credential.is_set() is True


def test_the_playground_credential_file_is_not_world_readable(configured):
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    assert credential.PLAYGROUND.path().stat().st_mode & 0o077 == 0


def test_a_short_playground_password_is_refused(configured):
    with pytest.raises(credential.WeakPassword):
        credential.PLAYGROUND.set_password("short")
    assert credential.PLAYGROUND.is_set() is False


def test_a_damaged_playground_record_verifies_nothing(configured):
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    credential.PLAYGROUND.path().write_text("{not json", encoding="utf-8")
    assert credential.PLAYGROUND.is_set() is False
    assert credential.PLAYGROUND.verify(PLAYGROUND_PASSWORD) is False


# ── Step 3: the matrix ──────────────────────────────────────────────────────
#
# On loopback, with no password and no key anywhere: the two scopes are the only
# thing varying, which is what makes this a test of the scopes rather than of the
# credentials. Kept as one table compared in one assertion so that deleting half
# of it is deleting rows from a literal, not silently dropping a case.

MATRIX = {
    # (admin_auth_scope, playground_auth_scope): (/admin/config, /playground/api/sessions)
    ("network", "network"): (200, 200),
    ("always", "network"): (401, 200),
    ("network", "always"): (200, 401),
    ("always", "always"): (401, 401),
}


def test_each_plane_answers_to_its_own_scope_alone(configured):
    """The witness this whole change turns on.

    `always` + `network` is the asymmetric cell: an admin password demanded even
    on this machine, an open playground on it. A single shared scope, or one
    authorizer that cannot tell the planes apart, cannot produce that row — and
    would pass every test that looked at one plane at a time.
    """
    observed = {}
    for scopes in MATRIX:
        admin_scope, playground_scope = scopes
        app = app_for(
            configured,
            admin_auth_scope=admin_scope,
            playground_auth_scope=playground_scope,
        )
        with make_client(app) as client:
            observed[scopes] = (
                client.get("/admin/config").status_code,
                client.get("/playground/api/sessions").status_code,
            )

    assert observed == MATRIX


def test_the_refusal_names_the_credential_the_plane_wants(configured):
    """A 401 from one plane must not send the browser to the other's form."""
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        refused = client.get("/playground/api/sessions")
        assert refused.status_code == 401
        assert refused.json()["error"]["code"] == "playground_login_required"

    with make_client(app_for(configured, admin_auth_scope="always")) as client:
        refused = client.get("/admin/config")
        assert refused.status_code == 401
        assert refused.json()["error"]["code"] == "admin_login_required"


def test_a_gated_playground_says_nothing_about_the_password_in_its_refusal(configured):
    """The 401 body carries no credential state. The login screen asks for that
    from `/playground/api/session`, which is same-origin and cross-site denied."""
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        body = client.get("/playground/api/sessions").json()
    assert set(body) == {"error"}
    assert "passwordSet" not in body["error"]
    assert PLAYGROUND_PASSWORD not in repr(body)


# ── Step 3: what opens a gated playground ───────────────────────────────────


def gated(configured, **server):
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    return app_for(configured, playground_auth_scope="always", **server)


def test_the_api_key_still_opens_the_playground(configured):
    """D3. The Hermes plugin and Open WebUI reach this plane with a bearer token,
    and taking that away breaks them in Hermes rather than here."""
    with make_client(gated(configured, api_key=KEY)) as client:
        assert client.get("/playground/api/sessions").status_code == 401
        opened = client.get("/playground/api/sessions", headers={"Authorization": f"Bearer {KEY}"})
        assert opened.status_code == 200
        # And the wrong key is not "a key".
        assert (
            client.get(
                "/playground/api/sessions", headers={"Authorization": "Bearer nearly-right"}
            ).status_code
            == 401
        )


def test_an_admin_session_opens_the_playground(configured):
    """D2: admin is strictly stronger already — `GET /admin/config` hands out
    `server.api_key` — so refusing it would only mean two logins for one person."""
    credential.set_password(ADMIN_PASSWORD)
    with make_client(gated(configured)) as client:
        assert client.get("/playground/api/sessions").status_code == 401
        assert client.post("/admin/session", json={"password": ADMIN_PASSWORD}).status_code == 204
        assert client.cookies.get(session.COOKIE)
        assert client.get("/playground/api/sessions").status_code == 200


def test_the_local_token_opens_the_playground(configured):
    """The credential of last resort: it is how a forgotten playground password
    is recovered from by the menubar app and the CLI."""
    with make_client(gated(configured, token="local-secret")) as client:
        assert client.get("/playground/api/sessions").status_code == 401
        assert (
            client.get("/playground/api/sessions", headers={"X-QDS-Admin-Token": "local-secret"}).status_code
            == 200
        )


def test_the_playground_password_opens_the_playground_and_nothing_else(configured):
    """D5's other half: this cookie is not an admin credential.

    The playground plane's session must not reach the configuration writer, the
    logs or the restart button — that is the whole reason it is a second secret
    rather than the admin one.
    """
    credential.set_password(ADMIN_PASSWORD)
    with make_client(gated(configured, admin_auth_scope="always")) as client:
        assert (
            client.post("/playground/api/session", json={"password": PLAYGROUND_PASSWORD}).status_code == 204
        )
        assert client.cookies.get(session.PLAYGROUND_COOKIE)
        assert client.get("/playground/api/sessions").status_code == 200
        assert client.get("/admin/config").status_code == 401
        assert client.get("/admin/logs").status_code == 401


def test_neither_session_store_revokes_the_other(configured):
    """D5: two stores, so changing the admin password does not sign the
    playground out, and the playground's logout does not end the admin's."""
    credential.set_password(ADMIN_PASSWORD)
    with make_client(gated(configured, admin_auth_scope="always")) as client:
        client.post("/admin/session", json={"password": ADMIN_PASSWORD})
        client.post("/playground/api/session", json={"password": PLAYGROUND_PASSWORD})

        client.delete("/playground/api/session")
        # The admin session survived its sibling's logout.
        assert client.get("/admin/config").status_code == 200
        # And the playground cookie is genuinely dead, not merely unsent: the
        # admin session is what answers the playground now.
        assert client.cookies.get(session.PLAYGROUND_COOKIE) is None

        client.delete("/admin/session")
        assert client.get("/admin/config").status_code == 401
        assert client.get("/playground/api/sessions").status_code == 401


# ── Step 2: logging in ──────────────────────────────────────────────────────


def test_the_lock_screen_can_ask_what_it_needs_to_render(configured):
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        # Reachable while the plane it belongs to is refusing everything else:
        # a login endpoint behind the gate would be unreachable by construction.
        status = client.get("/playground/api/session")
        assert status.status_code == 200
        assert status.json() == {
            "passwordSet": True,
            "authenticated": False,
            "loopback": True,
            "gated": True,
        }
        client.post("/playground/api/session", json={"password": PLAYGROUND_PASSWORD})
        assert client.get("/playground/api/session").json()["authenticated"] is True


def test_the_lock_screen_knows_when_no_password_exists(configured):
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        assert client.get("/playground/api/session").json() == {
            "passwordSet": False,
            "authenticated": False,
            "loopback": True,
            "gated": True,
        }
        # Nothing to check it against, so there is nothing to log in to.
        refused = client.post("/playground/api/session", json={"password": "anything at all"})
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "no_password_set"


def test_an_ungated_playground_says_so(configured):
    """`gated: false` is what lets the surface stop asking on a loopback install
    whose scope is `network`, even though a password happens to exist."""
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured)) as client:
        assert client.get("/playground/api/session").json()["gated"] is False
        assert client.get("/playground/api/sessions").status_code == 200


def test_a_wrong_playground_password_is_refused_then_throttled(configured):
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        for _ in range(admin.LoginThrottle.ALLOWED):
            refused = client.post("/playground/api/session", json={"password": "not it"})
            assert refused.status_code == 401
            assert refused.json()["error"]["code"] == "invalid_password"
        assert client.cookies.get(session.PLAYGROUND_COOKIE) is None

        locked = client.post("/playground/api/session", json={"password": PLAYGROUND_PASSWORD})
        assert locked.status_code == 429
        assert locked.json()["error"]["code"] == "too_many_attempts"
        # Guessing here must not have cost anyone the control plane.
        assert client.get("/playground/api/sessions").status_code == 401


def test_guessing_the_playground_password_does_not_lock_the_admin_out(configured):
    """Separate throttles, because the admin login is the way back in.

    A shared one would let anybody who can reach the playground deny the
    operator their own control plane for fifteen minutes.
    """
    credential.set_password(ADMIN_PASSWORD)
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        for _ in range(admin.LoginThrottle.ALLOWED):
            client.post("/playground/api/session", json={"password": "not it"})
        assert client.post("/playground/api/session", json={"password": "not it"}).status_code == 429
        assert client.post("/admin/session", json={"password": ADMIN_PASSWORD}).status_code == 204


def test_a_hostile_page_cannot_post_a_guess(configured):
    """F2: the cookie is not the boundary on its own — `deny_cross_site` is what
    stops another origin using the browser's credentials, and the login route
    keeps it."""
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        for response in (
            client.post(
                "/playground/api/session",
                json={"password": PLAYGROUND_PASSWORD},
                headers={"Origin": "http://evil.example"},
            ),
            client.get("/playground/api/session", headers={"Origin": "http://evil.example"}),
        ):
            assert response.status_code == 403, response.text
            assert response.json()["error"]["code"] == "cross_site_denied"


def test_logging_out_revokes_the_token_rather_than_only_the_cookie(configured):
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        client.post("/playground/api/session", json={"password": PLAYGROUND_PASSWORD})
        token = client.cookies.get(session.PLAYGROUND_COOKIE)
        client.delete("/playground/api/session")
        # Presented again by hand, the way a stolen cookie would be.
        client.cookies.set(session.PLAYGROUND_COOKIE, token)
        assert client.get("/playground/api/sessions").status_code == 401


def test_the_playground_session_is_not_persisted_across_a_restart(configured):
    """In-memory, like the admin store: a cookie that outlived the process it
    was minted by would make logging out unenforceable."""
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        client.post("/playground/api/session", json={"password": PLAYGROUND_PASSWORD})
        token = client.cookies.get(session.PLAYGROUND_COOKIE)

    with make_client(app_for(configured, playground_auth_scope="always")) as restarted:
        restarted.cookies.set(session.PLAYGROUND_COOKIE, token)
        assert restarted.get("/playground/api/sessions").status_code == 401


# ── The invariant: a scope is a tightening knob only ────────────────────────


def test_network_does_not_open_a_playground_the_api_key_closes(configured):
    """The rule `network` keeps on loopback is exactly the one that shipped:
    open with no key configured, refused with one."""
    with make_client(app_for(configured, api_key=KEY)) as client:
        assert client.get("/playground/api/sessions").status_code == 401
        assert (
            client.get("/playground/api/sessions", headers={"Authorization": f"Bearer {KEY}"}).status_code
            == 200
        )


def test_network_does_not_open_a_playground_reachable_from_the_network(configured):
    """`network` means "bound off this machine", not "never"."""
    with make_client(app_for(configured, host="0.0.0.0")) as client:
        assert client.get("/playground/api/sessions").status_code == 401


def test_network_does_not_reopen_a_control_plane_that_has_a_password(configured):
    """The admin gate binds on any credential, whatever the scope says. A scope
    that could switch it off would be the loosening this feature may not be."""
    credential.set_password(ADMIN_PASSWORD)
    for scope in ("network", "always"):
        with make_client(app_for(configured, admin_auth_scope=scope)) as client:
            assert client.get("/admin/config").status_code == 401


def test_the_first_run_control_plane_is_still_open_by_default(configured):
    """Unchanged, and load-bearing: it is how the dashboard sets the first
    password. Only `always` withdraws it."""
    with make_client(app_for(configured)) as client:
        assert client.get("/admin/config").status_code == 200


def test_recovery_mode_ignores_the_admin_scope(configured):
    """The way back from `always` with no password.

    That configuration refuses to start, which lands the operator in the
    recovery app — the one screen that can repair it. Enforcing the scope there
    too would make the setting a one-way door, which is a STOP condition, not a
    trade-off. It is not a hole: `effective_bind_host` pins a passwordless
    recovery server to loopback whatever the file says.
    """
    settings = settings_for(configured, admin_auth_scope="always")
    with make_client(create_recovery_app(settings, "broken")) as client:
        assert client.get("/admin/config").status_code == 200

    # Once a password exists, recovery asks for it like anything else.
    credential.set_password(ADMIN_PASSWORD)
    with make_client(create_recovery_app(settings, "broken")) as client:
        assert client.get("/admin/config").status_code == 401


def test_the_overview_says_whether_a_playground_password_exists(configured):
    """What the Configuration screen reads to label its control. Beside
    `adminPasswordSet`, because it is the same kind of fact about the other
    plane — and behind admin, so it is not a disclosure."""
    with make_client(app_for(configured)) as client:
        assert client.get("/admin/overview").json()["playgroundPasswordSet"] is False
        client.post("/admin/playground/password", json={"new": PLAYGROUND_PASSWORD})
        assert client.get("/admin/overview").json()["playgroundPasswordSet"] is True


def test_the_v1_plane_does_not_move_with_the_playground_scope(configured):
    """Three planes, and the scopes name two of them. `/v1` keeps the rule it
    has: the api_key when one is configured, open on a keyless loopback install."""
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        assert client.get("/playground/api/sessions").status_code == 401
        assert client.get("/v1/models").status_code == 200
        assert client.get("/health").status_code == 200


# ── Admin authority over the playground's password ──────────────────────────


def test_an_admin_sets_the_playground_password(configured):
    """D1's consequence: setting it is an act of administration, so the person
    who may generate images is not the person who chooses the secret."""
    with make_client(app_for(configured)) as client:
        assert client.post("/admin/playground/password", json={"new": PLAYGROUND_PASSWORD}).status_code == 200
        assert credential.PLAYGROUND.verify(PLAYGROUND_PASSWORD) is True
        # No `current` is required — that is what makes this the recovery path —
        # but the length floor still is.
        weak = client.post("/admin/playground/password", json={"new": "short"})
        assert weak.status_code == 400
        assert weak.json()["error"]["code"] == "weak_password"


def test_changing_the_playground_password_ends_every_playground_session(configured):
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        client.post("/playground/api/session", json={"password": PLAYGROUND_PASSWORD})
        assert client.get("/playground/api/sessions").status_code == 200
        client.post("/admin/playground/password", json={"new": "a different long one"})
        assert client.get("/playground/api/sessions").status_code == 401


def test_the_playground_password_cannot_be_removed_while_the_scope_needs_it(configured):
    """A gate with no key is a gate that is off, and the server would refuse to
    restart in that state — so removing it here would only produce a
    configuration that cannot come back."""
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        refused = client.delete("/admin/playground/password")
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "password_required_by_scope"
        assert credential.PLAYGROUND.is_set() is True

    with make_client(app_for(configured)) as client:
        assert client.delete("/admin/playground/password").status_code == 200
        assert credential.PLAYGROUND.is_set() is False


def test_the_playground_password_routes_need_admin(configured):
    """Not the playground's own credential: it must not be able to change or
    remove itself."""
    credential.set_password(ADMIN_PASSWORD)
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    with make_client(app_for(configured, playground_auth_scope="always")) as client:
        client.post("/playground/api/session", json={"password": PLAYGROUND_PASSWORD})
        assert client.get("/playground/api/sessions").status_code == 200
        assert (
            client.post("/admin/playground/password", json={"new": "another long password"}).status_code
            == 401
        )


# ── The residual the API gate does not cover ────────────────────────────────


def test_the_image_routes_stay_on_the_data_plane_credential(configured):
    """Pinning the documented residual, so it cannot drift unnoticed.

    `/playground/images/{filename}` is fetched by an `<img>`, which presents no
    header, and `?view=plugin` is not always a context a `SameSite=Strict`
    cookie is sent in — so the bytes stay on the data-plane credential rather
    than moving onto the playground's cookie gate. `build_playground_router`
    argues that at length.

    The consequence is worth an assertion rather than a paragraph: with the
    playground gated and no credential, the *API* refuses, and a picture is
    still reachable to a caller who already knows its name. That is the trade,
    and a future change that quietly widened or narrowed it should have to edit
    this test and say why.
    """
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    app = app_for(configured, playground_auth_scope="always")

    with make_client(app) as client:
        # The API is shut: nothing can be enumerated.
        assert client.get("/playground/api/sessions").status_code == 401
        # A name nobody minted is a 404, not a 401: the route does not become an
        # oracle for which filenames exist.
        assert client.get("/playground/images/deadbeef.png").status_code == 404
        assert client.get("/playground/images/deadbeef.png/thumb").status_code == 404


def test_an_api_key_still_opens_the_playground_under_every_scope(configured):
    """D3, asserted across the matrix rather than once.

    The Hermes plugin and Open WebUI hold a bearer key, not a cookie. A gate
    that forgot them would break those surfaces silently — and the failure would
    surface in Hermes, not here.
    """
    credential.PLAYGROUND.set_password(PLAYGROUND_PASSWORD)
    for admin_scope in ("network", "always"):
        for playground_scope in ("network", "always"):
            app = app_for(
                configured,
                api_key="k" * 32,
                admin_auth_scope=admin_scope,
                playground_auth_scope=playground_scope,
            )
            with make_client(app) as client:
                answered = client.get(
                    "/playground/api/sessions",
                    headers={"Authorization": "Bearer " + "k" * 32},
                )
                assert answered.status_code == 200, (
                    f"api_key refused at admin={admin_scope} "
                    f"playground={playground_scope}: {answered.text}"
                )
