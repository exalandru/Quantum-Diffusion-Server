"""Per-session passwords for the playground: unlock tokens and their errors.

A session password gates *content* — a locked session's prompts, generations and
images are not served without proof of the password. The proof is an unlock
token: the password is redeemed once for an opaque, in-memory token bound to
one session, which the dashboard then presents on every request for that
session. Like admin sessions (`qds/session.py`), tokens live only in this
process: **a restart locks every session again**, which is the price of having
no signing key at rest and of a lock that actually revokes.

The hash itself is a `credential.hash_password` record in the session row; this
module never touches the database. It owns the token table and the guessing
throttle, and names the errors the routes raise so that the dashboard can tell
"locked" from every other 403.
"""

from __future__ import annotations

import secrets
import time

from qds.admin import LoginThrottle
from qds.errors import APIError

#: Header carrying an unlock token. A query parameter of the same token is
#: accepted on image GETs only: an `<img>` cannot send a header.
UNLOCK_HEADER = "X-QDS-Session-Token"
UNLOCK_QUERY = "t"

#: How long a token may sit unused. Shorter than an admin session's: this is a
#: lock the user asked for, and an unattended tab should not keep it open.
IDLE_TIMEOUT_S = 30 * 60

#: A ceiling on the token table, for the same reason `session.MAX_SESSIONS`
#: has one. Far above any real number of unlocked sessions.
MAX_TOKENS = 256


class UnlockStore:
    """Live unlock tokens, by token."""

    def __init__(self) -> None:
        #: token -> (session_id, last_seen)
        self._tokens: dict[str, tuple[str, float]] = {}

    def issue(self, session_id: str) -> str:
        self._purge()
        if len(self._tokens) >= MAX_TOKENS:
            oldest = min(self._tokens, key=lambda token: self._tokens[token][1])
            del self._tokens[oldest]
        token = secrets.token_urlsafe(32)
        self._tokens[token] = (session_id, time.monotonic())
        return token

    def session_for(self, token: str | None) -> str | None:
        """The session this token unlocks, refreshing its idle clock; `None`
        for an unknown or expired token."""
        if not token:
            return None
        self._purge()
        entry = self._tokens.get(token)
        if entry is None:
            return None
        session_id, _ = entry
        self._tokens[token] = (session_id, time.monotonic())
        return session_id

    def revoke(self, token: str | None) -> None:
        if token:
            self._tokens.pop(token, None)

    def revoke_session(self, session_id: str) -> None:
        """Every token for this session: the password changed or went away,
        or the session itself did."""
        for token in [t for t, (sid, _) in self._tokens.items() if sid == session_id]:
            del self._tokens[token]

    def _purge(self) -> None:
        now = time.monotonic()
        for token in [t for t, (_, seen) in self._tokens.items() if now - seen > IDLE_TIMEOUT_S]:
            del self._tokens[token]


class UnlockThrottles:
    """One `LoginThrottle` per session.

    Per session rather than global: a guess against one session should not
    lock the user out of unlocking another, and per-IP keying is free to
    evade on a LAN (the same reasoning `LoginThrottle` gives). Entries are
    dropped once they hold no state, so the table tracks sessions under
    attack, not sessions that exist.
    """

    def __init__(self) -> None:
        self._throttles: dict[str, LoginThrottle] = {}

    def for_session(self, session_id: str) -> LoginThrottle:
        for key, throttle in list(self._throttles.items()):
            if key != session_id and throttle.retry_after() == 0 and not throttle._failures:
                del self._throttles[key]
        return self._throttles.setdefault(session_id, LoginThrottle())

    def forget(self, session_id: str) -> None:
        self._throttles.pop(session_id, None)


# ── Errors ──────────────────────────────────────────────────────────────────
#
# All 403 or 409, never 401: a 401 is what makes the dashboard show the admin
# login, and a locked session is not a missing admin credential.


def locked(session_id: str) -> APIError:
    return APIError(
        f"Playground session {session_id!r} is locked.",
        status_code=403,
        code="session_locked",
    )


def invalid_password() -> APIError:
    return APIError("Wrong session password.", status_code=403, code="invalid_session_password")


def not_protected(session_id: str) -> APIError:
    return APIError(
        f"Playground session {session_id!r} has no password.",
        status_code=409,
        code="session_not_protected",
    )


def too_many_attempts(wait: float) -> APIError:
    return APIError(
        f"Too many attempts. Try again in {wait:.0f} seconds.",
        status_code=429,
        code="too_many_attempts",
    )


def weak_password(message: str) -> APIError:
    return APIError(message, status_code=400, param="password", code="weak_password")
