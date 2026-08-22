"""Which playground session a conversation writes to.

MCP conversations and playground sessions are different things with different
lifetimes, and this module is the *cache* between them -- not an authority.
The playground session is durable, survives the chat that made it, and can be
renamed, locked or deleted by the person whose machine this is. The mapping here
is in memory and dies with the process, which is the right direction: losing it
costs one new session, while persisting it would let a deleted session come back
as a dangling id.

The key is the transport's `mcp-session-id`. Over stdio there is no such header,
because the bridge holds exactly one upstream session -- so every stdio client
gets one default session, correct by construction rather than by special case.

A locked session is never resolved here. MCP carries no unlock token and must
not grow one: a password on a session is a person's decision about a browser,
and a tool that could walk past it would make the control mean nothing.
"""

from __future__ import annotations

import threading

#: The header the streamable-HTTP transport assigns per client session.
SESSION_HEADER = "mcp-session-id"
#: Used when there is no such header -- stdio, and any client that omits it.
_SINGLE = "\x00single"


class SessionBinding:
    """Conversation → playground session, created on first use.

    Lazily, and that is deliberate: a client that connects, lists tools and
    disconnects should leave nothing behind. A session appears in the sidebar
    when a model actually generates something.
    """

    def __init__(self, store) -> None:
        self._store = store
        self._bound: dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key_for(headers) -> str:
        if not headers:
            return _SINGLE
        try:
            value = headers.get(SESSION_HEADER)
        except AttributeError:  # pragma: no cover - a mapping without .get
            value = None
        return value or _SINGLE

    def resolve(self, headers, *, explicit: str | None = None) -> str:
        """The session this call should write to.

        An explicit `session_id` wins and is *verified*, so naming a session
        that no longer exists is an error the model is told about rather than a
        silent write into a fresh one.
        """
        if explicit:
            if self._store.session_summary(explicit) is None:
                from qds.errors import APIError

                raise APIError(
                    f"No playground session {explicit!r}. Call list_sessions to see "
                    f"what exists, or omit session_id to use this conversation's own.",
                    status_code=404,
                    param="session_id",
                    code="not_found",
                )
            return explicit

        key = self.key_for(headers)
        with self._lock:
            bound = self._bound.get(key)
            # Deleted from the UI while a conversation was still pointing at it.
            # One silent re-creation, because the alternative is telling a model
            # about a session it never chose in the first place.
            if bound is not None and self._store.session_summary(bound) is not None:
                return bound
            created = self._store.create_session()["id"]
            self._bound[key] = created
            return created

    def bind(self, headers, session_id: str) -> None:
        """Point this conversation at an existing session (`open_session`)."""
        with self._lock:
            self._bound[self.key_for(headers)] = session_id


def require_unlocked(store, session_id: str) -> None:
    """Refuse a session someone put a password on.

    `password_record` returns None for an *open* session and raises `KeyError`
    for one that does not exist -- the two have to be told apart, which is what
    its own docstring asks of every caller. Any record at all means locked,
    including the empty dict it returns for a damaged one: fail closed.

    The refusal names the playground rather than offering a way through. The
    only correct place to unlock is the browser where the password was set, and
    a tool that could walk past it would make the control mean nothing.
    """
    from qds.errors import APIError

    try:
        record = store.password_record(session_id)
    except KeyError:
        raise APIError(
            f"No playground session {session_id!r}.",
            status_code=404,
            param="session_id",
            code="not_found",
        ) from None
    if record is not None:
        raise APIError(
            f"Playground session {session_id!r} is password-protected. Unlock it in "
            f"the playground; this server does not accept session passwords over MCP.",
            status_code=403,
            param="session_id",
            code="session_locked",
        )
