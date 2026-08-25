"""Who may reach which plane, in one place.

There are two planes and they have two audiences. `/v1` is meant to be called by
other applications — Open WebUI, the `openai` SDK, a script — and is gated by
`server.api_key`. `/admin` is the control plane: it edits the configuration,
starts downloads and restarts the process, and its audience is the person who
runs this server, not the software they point at it.

Keeping the rules here rather than inline in `create_app` is not tidiness. The
normal app and the recovery app each had their own copy of `require_auth`, and
a duplicate is exactly the mechanism by which recovery mode drifts into being
differently protected — in the one state where an authentication mistake is
unrecoverable, because the configuration that would fix it is the configuration
that failed.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Cookie, Header

from qds import credential
from qds import session as session_module
from qds.errors import APIError
from qds.session import SessionStore
from qds.settings import Settings


def invalid_key() -> APIError:
    """The data plane's refusal, so the ASGI guard can render the same one."""
    return APIError(
        "Missing or invalid API key.",
        status_code=401,
        error_type="invalid_request_error",
        code="invalid_api_key",
    )


def _presented_bearer(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    return token if scheme.lower() == "bearer" else ""


def matches_api_key(authorization: str | None, expected: str) -> bool:
    """Whether the presented bearer token is the configured key.

    `compare_digest` on bytes rather than on `str`: it rejects non-ASCII `str`
    outright, and a key with an accent in it is a key somebody will choose.
    """
    presented = _presented_bearer(authorization)
    return secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


#: The header a local client presents its token in. Deliberately *not*
#: `Authorization: Bearer`: the two credentials mean different things, and a
#: token pasted into an API-key field should authenticate nothing.
LOCAL_TOKEN_HEADER = "X-QDS-Admin-Token"


def _login_required() -> APIError:
    return APIError(
        "This server's control plane requires the admin password.",
        status_code=401,
        error_type="invalid_request_error",
        code="admin_login_required",
    )


def _playground_login_required() -> APIError:
    """The playground plane's refusal, distinct from the admin one.

    A separate code because a separate screen renders it: the dashboard shell
    asks for the admin password on `admin_login_required`, and the playground
    must not send someone to that form for a credential that would not open it.
    """
    return APIError(
        "This server's playground requires the playground password.",
        status_code=401,
        error_type="invalid_request_error",
        code="playground_login_required",
    )


def build_authorizer(
    settings: Settings,
    sessions: SessionStore | None = None,
    local_token: str | None = None,
):
    """The data-plane rule, as a predicate over three raw credential values.

    Extracted from `require_api` rather than copied, and for the reason this
    module exists at all: `/mcp` is a *mounted ASGI application*, so a FastAPI
    dependency cannot gate it, and the only other way to protect it would be a
    second implementation of these rules — the exact drift the module docstring
    names. `require_api` is a wrapper over this; there is one rule.

    Takes the values, not a `Request`: an ASGI guard reads them off `scope`
    while FastAPI extracts them from the signature, and neither should have to
    fabricate the other's argument.
    """

    def authorize(
        authorization: str | None,
        qds_admin: str | None,
        x_qds_admin_token: str | None,
    ) -> bool:
        # An admin is already strictly stronger than the data-plane key — they
        # can read `server.api_key` out of `GET /admin/config` — so accepting an
        # admin credential here grants nothing new, and it saves the dashboard
        # from holding two secrets to call two planes of one server.
        if sessions is not None and sessions.validate(qds_admin):
            return True
        if local_token and x_qds_admin_token and _same(x_qds_admin_token, local_token):
            return True

        expected = settings.server.api_key
        # No key configured is the default on loopback, and it means the data
        # plane is open. That is deliberate: a local install should generate an
        # image without ceremony.
        if not expected:
            return True
        return matches_api_key(authorization, expected)

    return authorize


def build_playground_authorizer(
    settings: Settings,
    sessions: SessionStore | None = None,
    playground_sessions: SessionStore | None = None,
    local_token: str | None = None,
):
    """The playground plane's rule, which is *not* the data plane's.

    `/v1` is called by other applications and gated by a machine credential;
    `/playground/api` is driven by a person in a browser, and this is what lets
    them be asked for a password they type rather than a key they paste. The two
    predicates are separate functions rather than one with a flag because the
    defect a shared implementation produces is silent: a single authorizer that
    cannot tell the planes apart would move both when one scope changed, and
    every test that exercises the planes separately would still pass.

    Four credentials open it, in the order they are cheapest to check:

    * an admin session — strictly stronger already (`GET /admin/config` hands
      out `server.api_key`), so refusing it would only mean two logins for one
      person;
    * the local token — the credential of last resort, and the way back in when
      the playground password is the one that was forgotten;
    * a playground session — the cookie this plane's own login mints;
    * `server.api_key` — unchanged, because it is how the Hermes plugin and Open
      WebUI reach this plane, and taking it away would break them silently.

    With none of those, the answer depends on whether this plane's gate binds —
    `always`, or a socket open beyond this machine. When it does not, the rule
    is exactly the one `/v1` has always applied on loopback: open when no key is
    configured, refused when one is. That is what keeps `network` from being a
    loosening of anything.
    """

    def authorize(
        authorization: str | None,
        qds_admin: str | None,
        x_qds_admin_token: str | None,
        qds_playground: str | None,
    ) -> bool:
        if sessions is not None and sessions.validate(qds_admin):
            return True
        if local_token and x_qds_admin_token and _same(x_qds_admin_token, local_token):
            return True
        if playground_sessions is not None and playground_sessions.validate(qds_playground):
            return True

        expected = settings.server.api_key
        if expected and matches_api_key(authorization, expected):
            return True

        if settings.server.playground_gate_binds:
            return False
        return not expected

    return authorize


def build_dependencies(
    settings: Settings,
    sessions: SessionStore | None = None,
    local_token: str | None = None,
    *,
    accept_api_key_for_admin: bool = False,
    playground_sessions: SessionStore | None = None,
    recovery: bool = False,
):
    """The three dependencies, sharing one set of rules.

    Returned as a tuple rather than exposed as module-level functions because
    they close over one running application's state — a process serving two
    configurations does not exist, but a *test* holding two apps at once does,
    and module state would have them share a session store.

    **The API key does not open `/admin`.** It used to, and that made the
    separation worthless: a key handed to Open WebUI so it could generate images
    also let it rewrite the configuration, read the logs and restart the process.
    Two planes, two audiences, two credentials. `accept_api_key_for_admin`
    remains only so a test can pin the old behaviour as *not* the current one.

    `recovery` exempts the control plane from `admin_auth_scope`, and that is
    the recovery path for the scope itself: a configuration asking for a
    password no file holds refuses to start (`runtime_issues`), which lands the
    operator in the recovery app — the one screen that can repair it. Enforcing
    the scope there too would make the setting unrecoverable, which is the STOP
    condition this feature was not allowed to hit. It is not a hole: a recovery
    server with no admin password binds loopback whatever the configuration says
    (`effective_bind_host`).
    """

    authorize = build_authorizer(settings, sessions, local_token)
    authorize_playground = build_playground_authorizer(
        settings, sessions, playground_sessions, local_token
    )

    async def require_api(
        authorization: Annotated[str | None, Header()] = None,
        qds_admin: Annotated[str | None, Cookie(alias=session_module.COOKIE)] = None,
        x_qds_admin_token: Annotated[str | None, Header()] = None,
    ) -> None:
        if not authorize(authorization, qds_admin, x_qds_admin_token):
            raise invalid_key()

    async def require_admin(
        authorization: Annotated[str | None, Header()] = None,
        qds_admin: Annotated[str | None, Cookie(alias=session_module.COOKIE)] = None,
        x_qds_admin_token: Annotated[str | None, Header()] = None,
    ) -> None:
        if sessions is not None and sessions.validate(qds_admin):
            return
        if local_token and x_qds_admin_token and _same(x_qds_admin_token, local_token):
            return

        # No password set is the first-run state, and it opens the control plane
        # so the dashboard can *set* one. That is only safe because a server
        # bound beyond loopback refuses to start without a password — the two
        # rules are one mechanism and must not be separated.
        #
        # `admin_auth_scope: always` withdraws that opening: a gate asked to bind
        # on loopback with no key behind it must close, not fall through to the
        # first-run path. `runtime_issues` already refuses to *start* such a
        # server; this is the same refusal one layer in, for an app that was
        # built without that check — and it is a tightening only, since the
        # default scope leaves this branch exactly as it was.
        if not credential.is_set():
            if settings.server.admin_auth_scope == "always" and not recovery:
                raise _login_required()
            if accept_api_key_for_admin and settings.server.api_key:
                # A key is configured, so it is still the gate during the
                # transition; without this an api_key-protected server would be
                # briefly open to anything that reached it.
                if not matches_api_key(authorization, settings.server.api_key):
                    raise invalid_key()
            return

        if accept_api_key_for_admin and settings.server.api_key:
            if matches_api_key(authorization, settings.server.api_key):
                return

        raise _login_required()

    async def require_playground(
        authorization: Annotated[str | None, Header()] = None,
        qds_admin: Annotated[str | None, Cookie(alias=session_module.COOKIE)] = None,
        x_qds_admin_token: Annotated[str | None, Header()] = None,
        qds_playground: Annotated[
            str | None, Cookie(alias=session_module.PLAYGROUND_COOKIE)
        ] = None,
    ) -> None:
        if authorize_playground(authorization, qds_admin, x_qds_admin_token, qds_playground):
            return
        # Which refusal, chosen by what is actually being asked for: a gate that
        # binds wants the playground password, and the lock screen renders on
        # that code. A gate that does not bind can only be refusing a missing or
        # wrong api_key, which is the answer this plane has always given.
        if settings.server.playground_gate_binds:
            raise _playground_login_required()
        raise invalid_key()

    return require_api, require_admin, require_playground


def _same(presented: str, expected: str) -> bool:
    return secrets.compare_digest(
        presented.strip().encode("utf-8"), expected.strip().encode("utf-8")
    )
