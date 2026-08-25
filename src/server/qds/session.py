"""Logged-in browsers, and the local process that does not log in.

Two credentials, because there are two kinds of client and only one of them has
a person behind it.

**Sessions** are for the dashboard. A password is redeemed once for an opaque
token held in an `HttpOnly` cookie, so the password itself is typed once and the
browser never holds something a script could read.

**The local token** is for the menubar app and the CLI. They run as the same
user on the same machine and cannot be asked for a password; they read a file
only that user can read. That is the boundary being claimed — not "this is the
tray", but "this is something already able to read the configuration directory",
which could already take over the server by editing the config and waiting for a
restart. The token grants the same authority without the wait.

The store is in memory, so **every restart ends every session**. That is a cost
— the dashboard's own Restart button logs you out — accepted against the
alternative: a signing key persisted to disk would be a second secret at rest,
and one that makes `DELETE /admin/session` unenforceable across a restart, since
a logged-out cookie would be honoured again by the next process.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from pathlib import Path

from qds.logs import SERVER_LOGGER
from qds.settings import config_path

logger = logging.getLogger(f"{SERVER_LOGGER}.session")

#: Name of the cookie holding an admin session token.
COOKIE = "qds_admin"

#: And the playground's, which is a different session in a different store. Two
#: names because they are two credentials: revoking every admin session must not
#: sign the playground out, and a browser that holds one holds only that one.
PLAYGROUND_COOKIE = "qds_playground"

#: How long a session may live at all, and how long it may sit idle. The
#: absolute bound is what limits replay of a cookie captured on a plain-HTTP
#: network; the idle bound is what closes an unattended browser.
ABSOLUTE_LIFETIME_S = 12 * 3600
IDLE_TIMEOUT_S = 2 * 3600

#: A ceiling, so a script hammering the login endpoint cannot grow this without
#: bound. Far above any real number of open dashboards.
MAX_SESSIONS = 64


class SessionStore:
    """Live sessions, by token."""

    def __init__(self) -> None:
        #: token -> (created_at, last_seen)
        self._sessions: dict[str, tuple[float, float]] = {}

    def create(self) -> str:
        self._purge()
        if len(self._sessions) >= MAX_SESSIONS:
            # Drop the least recently used rather than refuse: the person at the
            # keyboard is the one being served, and a full table is a symptom of
            # something else.
            oldest = min(self._sessions, key=lambda token: self._sessions[token][1])
            del self._sessions[oldest]
        token = secrets.token_urlsafe(32)
        now = time.monotonic()
        self._sessions[token] = (now, now)
        return token

    def validate(self, token: str | None) -> bool:
        """Whether this token is live, refreshing its idle clock if so."""
        if not token:
            return False
        self._purge()
        entry = self._sessions.get(token)
        if entry is None:
            return False
        created, _ = entry
        self._sessions[token] = (created, time.monotonic())
        return True

    def revoke(self, token: str | None) -> None:
        if token:
            self._sessions.pop(token, None)

    def revoke_all(self) -> None:
        """Used when the password changes: old sessions were minted against it."""
        self._sessions.clear()

    def _purge(self) -> None:
        now = time.monotonic()
        expired = [
            token
            for token, (created, seen) in self._sessions.items()
            if now - created > ABSOLUTE_LIFETIME_S or now - seen > IDLE_TIMEOUT_S
        ]
        for token in expired:
            del self._sessions[token]

    def __len__(self) -> int:  # pragma: no cover - diagnostics
        return len(self._sessions)


# ── The local token ────────────────────────────────────────────────────────


def token_path(path: Path | None = None) -> Path:
    return (path or config_path()).parent / "admin-token"


def issue_local_token(path: Path | None = None) -> str | None:
    """Write a fresh token for local clients, and return it.

    Rewritten at every start, so a token read from a previous run stops working
    — the same property the in-memory session store gives, for the same reason.

    A failure to write is not fatal. A packaged install whose configuration
    directory is read-only still serves; it just cannot offer this credential,
    and the menubar app will say so rather than the server refusing to start.
    """
    target = token_path(path)
    token = secrets.token_urlsafe(32)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # 0600 from creation, not applied afterwards: a window in which the
        # credential is world-readable is still a window.
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
    except OSError as exc:
        logger.warning("could not write %s: %s - local clients cannot authenticate", target, exc)
        return None
    return token


def discard_local_token(path: Path | None = None) -> None:
    """Best effort: a token file outliving its server authenticates nothing."""
    try:
        token_path(path).unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best effort
        pass
