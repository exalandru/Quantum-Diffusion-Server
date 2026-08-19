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


def _invalid_key() -> APIError:
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


def build_dependencies(
    settings: Settings,
    sessions: SessionStore | None = None,
    local_token: str | None = None,
    *,
    accept_api_key_for_admin: bool = False,
):
    """The two dependencies, sharing one set of rules.

    Returned as a pair rather than exposed as module-level functions because
    they close over one running application's state — a process serving two
    configurations does not exist, but a *test* holding two apps at once does,
    and module state would have them share a session store.

    **The API key does not open `/admin`.** It used to, and that made the
    separation worthless: a key handed to Open WebUI so it could generate images
    also let it rewrite the configuration, read the logs and restart the process.
    Two planes, two audiences, two credentials. `accept_api_key_for_admin`
    remains only so a test can pin the old behaviour as *not* the current one.
    """

    async def require_api(
        authorization: Annotated[str | None, Header()] = None,
        qds_admin: Annotated[str | None, Cookie(alias=session_module.COOKIE)] = None,
        x_qds_admin_token: Annotated[str | None, Header()] = None,
    ) -> None:
        # An admin is already strictly stronger than the data-plane key — they
        # can read `server.api_key` out of `GET /admin/config` — so accepting an
        # admin credential here grants nothing new, and it saves the dashboard
        # from holding two secrets to call two planes of one server.
        if sessions is not None and sessions.validate(qds_admin):
            return
        if local_token and x_qds_admin_token and _same(x_qds_admin_token, local_token):
            return

        expected = settings.server.api_key
        # No key configured is the default on loopback, and it means the data
        # plane is open. That is deliberate: a local install should generate an
        # image without ceremony.
        if not expected:
            return
        if not matches_api_key(authorization, expected):
            raise _invalid_key()

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
        if not credential.is_set():
            if accept_api_key_for_admin and settings.server.api_key:
                # A key is configured, so it is still the gate during the
                # transition; without this an api_key-protected server would be
                # briefly open to anything that reached it.
                if not matches_api_key(authorization, settings.server.api_key):
                    raise _invalid_key()
            return

        if accept_api_key_for_admin and settings.server.api_key:
            if matches_api_key(authorization, settings.server.api_key):
                return

        raise _login_required()

    return require_api, require_admin


def _same(presented: str, expected: str) -> bool:
    return secrets.compare_digest(
        presented.strip().encode("utf-8"), expected.strip().encode("utf-8")
    )
