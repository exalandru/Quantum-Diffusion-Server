"""The guard in front of the mounted MCP application.

A mounted ASGI app is not a FastAPI route, so `Depends(require_api)` and
`Depends(admin.deny_cross_site)` cannot reach it. Writing the rules again here
would be the drift `qds/auth.py`'s docstring names, so this module holds no
rules: it reads three headers off `scope` and asks the same two predicates the
routes ask -- `auth.build_authorizer` and `admin.origin_matches`.

Pure ASGI rather than `BaseHTTPMiddleware`, and not as a preference. The
transport underneath streams a request body and a long-lived response, and
`BaseHTTPMiddleware` interposes a task and a queue on both. It is also outside
FastAPI's exception handlers -- a mounted app's raises never reach them -- so
this renders its own refusals in `error_payload` shape, which is what every
other refusal from this server looks like.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from qds import admin
from qds import session as session_module
from qds.auth import LOCAL_TOKEN_HEADER, invalid_key
from qds.errors import error_payload


def _cookie(headers: Headers, name: str) -> str | None:
    """One cookie, without building a `Request`.

    Deliberately tolerant of a malformed cookie header: a value we cannot parse
    is a value that does not authenticate, which is the same outcome as absent.
    """
    raw = headers.get("cookie")
    if not raw:
        return None
    for part in raw.split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value or None
    return None


class MCPGuard:
    """Authorize and contain, then hand over to the transport untouched.

    `authorize` is `auth.build_authorizer`'s predicate: the *same* rule
    `require_api` applies, so `/mcp` accepts exactly what `/v1` accepts -- an
    admin session, the local token, the configured API key, or anything at all
    when no key is configured.

    The cross-site check is the playground's, not `/v1`'s, because MCP writes
    what the playground writes: durable sessions, images on disk, GPU time. A
    loopback install has no API key, so without this any page in any tab could
    spend this machine's GPU and leave records in someone's playground -- and it
    holds independently of `cors_origins`, so widening that for `/v1`'s sake
    cannot reopen this. It fires only when `Origin` is present, so a chat client
    -- which sends none -- is unaffected.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        authorize: Callable[[str | None, str | None, str | None], bool],
        deny_cross_site: bool = True,
    ) -> None:
        self._app = app
        self._authorize = authorize
        self._deny_cross_site = deny_cross_site

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Lifespan is the transport's own business and carries nothing to judge.
        # Anything else -- a websocket, which this transport does not serve today
        # -- has no headers this guard can read, and a guard that cannot judge a
        # connection must not admit it. Passing it through would be an
        # unauthenticated path the moment the SDK grows one.
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
            return

        headers = Headers(scope=scope)

        if self._deny_cross_site:
            origin = headers.get("origin")
            if origin is not None and not admin.origin_matches(
                origin,
                scheme=scope.get("scheme", "http"),
                host=headers.get("host") or "",
            ):
                await self._refuse(send, admin.cross_site_denied())
                return

        allowed = self._authorize(
            headers.get("authorization"),
            _cookie(headers, session_module.COOKIE),
            headers.get(LOCAL_TOKEN_HEADER),
        )
        if not allowed:
            await self._refuse(send, invalid_key())
            return

        await self._app(scope, receive, send)

    @staticmethod
    async def _refuse(send: Send, error) -> None:
        body = json.dumps(
            error_payload(
                error.message,
                error_type=error.error_type,
                param=error.param,
                code=error.code,
            )
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": error.status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class MCPMount:
    """The mount point, whose transport exists only while the app is running.

    The MCP transport is not a plain ASGI app that can be built once and served
    forever: it owns a task group created by `session_manager.run()`, and its
    session manager refuses to be run twice. Mounting the transport directly
    would therefore make `create_app`'s result a *single-use* application --
    started once, never again -- which is not what the rest of this server
    promises and not what its tests do.

    So the mount is an indirection. Each time the application starts, `serving`
    builds a fresh transport, runs its session manager, and points this at it;
    each time it stops, it points back at nothing. Two consequences, both
    wanted: an app object can be started more than once, and a request that
    arrives while nothing is running gets a legible 503 rather than the SDK's
    "Task group is not initialized" as a 500.
    """

    def __init__(self, build: Callable[[], ASGIApp]) -> None:
        self._build = build
        self._target: ASGIApp | None = None

    @contextlib.asynccontextmanager
    async def serving(self, run_session_manager):
        """Build the transport, run it, and route to it for the duration.

        `run_session_manager` is called *after* the build, because the manager
        only exists once `streamable_http_app()` has made one -- and a fresh
        one per build is what lets this be entered again.
        """
        target = self._build()
        async with run_session_manager():
            self._target = target
            try:
                yield
            finally:
                self._target = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        target = self._target
        if target is None:
            if scope["type"] == "http":
                await MCPGuard._refuse(send, _not_running())
                return
            # A lifespan or websocket scope with nothing behind it: end it
            # rather than raise, so a probe does not look like a crash.
            return
        await target(scope, receive, send)


def _not_running():
    from qds.errors import APIError

    return APIError(
        "The MCP surface is not running on this server.",
        status_code=503,
        error_type="server_error",
        code="mcp_not_running",
    )
