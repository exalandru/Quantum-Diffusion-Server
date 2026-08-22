"""Answer the Private Network Access preflight for playground images, and only
for those.

Chromium asks permission before letting a page reach a *more private* address
than its own: a `<img src="http://127.0.0.1:8765/...">` in an Electron chat
client triggers a CORS preflight carrying
`Access-Control-Request-Private-Network: true`, and the image is blocked unless
the response grants it. Starlette's `CORSMiddleware` refuses that preflight
outright -- "Disallowed CORS private-network", 400 -- so the image never loads,
while the same URL fetched by `curl`, which sends no preflight, returns the file
perfectly. That asymmetry is exactly what made this hard to see.

**Why this is a separate middleware rather than `allow_private_network=True`.**
That flag is on `CORSMiddleware`, so it would grant private-network access to
*every* route: `/v1`, `/admin`, `/playground/api`, `/mcp`. PNA exists to stop a
web page reaching a service on someone's machine, and a keyless loopback install
has an open `/v1` -- granting it there would let any page in any tab spend this
machine's GPU. So the grant is scoped to the one route where it is defensible.

**Why it is defensible there.** The filename is a `uuid4().hex`: naming a file is
holding 122 bits of secret, the session lock is enforced per request underneath,
and `/playground/api` still refuses cross-site outright so nothing can enumerate
names. It is the same reasoning that removed the cross-site check from this route,
applied to the same route.

Outermost by construction: added after `CORSMiddleware`, so it sees the preflight
first and can answer it before that middleware rejects it.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

#: The one path prefix this grant covers.
IMAGES_PREFIX = "/playground/images/"

_REQUEST_HEADER = "access-control-request-private-network"


class PrivateNetworkImages:
    """Grant private-network access to `GET /playground/images/…`, and nothing else."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "OPTIONS":
            await self._app(scope, receive, send)
            return

        path = scope.get("path") or ""
        if not path.startswith(IMAGES_PREFIX):
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if headers.get(_REQUEST_HEADER, "").lower() != "true":
            await self._app(scope, receive, send)
            return

        # A preflight answers no data; it only says whether the real request may
        # be made. The real GET still goes through every check on the route --
        # the row lookup, the session lock, the host guard.
        origin = headers.get("origin") or "*"
        requested = headers.get("access-control-request-headers")
        response = [
            (b"access-control-allow-origin", origin.encode("latin-1")),
            (b"access-control-allow-private-network", b"true"),
            (b"access-control-allow-methods", b"GET, HEAD, OPTIONS"),
            (b"access-control-max-age", b"600"),
            (b"vary", b"Origin"),
            (b"content-length", b"0"),
        ]
        if requested:
            response.append((b"access-control-allow-headers", requested.encode("latin-1")))
        await send({"type": "http.response.start", "status": 200, "headers": response})
        await send({"type": "http.response.body", "body": b""})
