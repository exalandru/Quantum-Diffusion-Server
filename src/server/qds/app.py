"""The application assembler.

`create_app` builds the collaborators an application is made of — settings,
engine, image store, playground store, sessions, authentication and the shared
`Admission` — then mounts the planes over them:

* `qds.v1_routes` — `/health` and the OpenAI-compatible `/v1` plane;
* `qds.playground_routes` — `/playground/api` and the playground's images;
* `qds.admin` — the control plane and the session router;
* `qds.mcp` — a mounted ASGI application at `/mcp`.

What stays here is what belongs to the application rather than to a plane: the
middleware that bounds every response, the dashboard mounts, the lifespan, and
`create_recovery_app` — the deliberately minimal server a broken configuration
gets instead.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from qds import __version__, admin, credential, logbuffer, playground_lock
from qds import settings as settings_module
from qds.admission import Admission, _capabilities
from qds.auth import build_authorizer, build_dependencies
from qds.engine import ModelEngine
from qds.errors import APIError, error_payload, install_exception_handlers
from qds.hosts import allows as host_allows
from qds.hosts import build_allowlist
from qds.idle import IdleUnloader
from qds.jobs import JobManager
from qds.logbuffer import LogBuffer
from qds.logs import SERVER_LOGGER, setup_logging
from qds.playground import PlaygroundStore
from qds.playground_routes import build_playground_router, build_playground_session_router
from qds.pna import PrivateNetworkImages
from qds.session import SessionStore, discard_local_token, issue_local_token
from qds.settings import (
    ConfigError,
    Settings,
    load_settings,
    recovery_settings,
)
from qds.store import ImageStore
from qds.upscale import catalogue as upscale_catalogue
from qds.v1_routes import build_v1_router

logger = logging.getLogger(SERVER_LOGGER)

#: One decode bound for the process. Pillow only *warns* at its own default and
#: raises at twice it, so a 25 MB upload that decompresses to hundreds of
#: megapixels is otherwise accepted and allocated. Set to the render bound rather
#: than a new number: 8192x8192 is the largest image this server will produce, so
#: it is the largest one it ever has cause to read back.
Image.MAX_IMAGE_PIXELS = upscale_catalogue.MAX_RENDER_PIXELS


def _restart_unavailable() -> None:
    raise APIError(
        "This server was not started in a way that can restart itself.",
        status_code=501,
        error_type="server_error",
        code="restart_unavailable",
    )


#: Sent on every HTML document this server serves. The dashboard loads its script
#: and style as separate files, so `'self'` is the whole allowance; `'unsafe-inline'`
#: is needed for styles alone, because the playground sets inline `style` attributes
#: for the preview blur. It also contains anything that is served as a document and
#: should not have been -- the second lock on the stored-upload path, after
#: `context_path` normalises the name and the image route declares the type.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'"
)


def install_security_headers(app: FastAPI) -> None:
    """Add the headers that bound what a response is allowed to become.

    Registered last, therefore outermost, so it also covers the `StaticFiles`
    mounts -- which are ASGI apps a route dependency cannot reach.

    `nosniff` is what stops a byte stream being re-read as a document by a
    browser that disagrees with the declared type. `setdefault` throughout: a
    route that has already made a narrower promise keeps it.
    """

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers.setdefault("Content-Security-Policy", CSP)
            response.headers.setdefault("X-Frame-Options", "DENY")
        return response


def install_host_guard(app: FastAPI, settings: Settings) -> None:
    """Refuse requests whose `Host` header is not one this server answers to.

    This closes DNS rebinding, which authentication cannot: a page on
    `evil.example` whose name resolves to 127.0.0.1 is same-origin *to the
    browser*, so it may read the responses, and a default install has no API key
    to stop it. What it cannot fake is the `Host` header, which carries the name
    the browser dialled.

    **It no longer steps aside for a wildcard bind.** It used to, which meant
    turning on "listen on the local network" also turned this off — the
    protection disappearing exactly when it started to matter. The allowlist
    grows instead, to the addresses and names this machine answers to.
    """
    port = settings.server.port
    # Computed once, at startup, and never on the request path: resolving this
    # machine's own hostname can block on a network where it does not resolve.
    allowed = build_allowlist(settings.server.host, port, settings.server.allowed_hosts)

    @app.middleware("http")
    async def guard_host(request: Request, call_next):
        host = request.headers.get("host")
        if not host_allows(host, allowed, port):
            return JSONResponse(
                status_code=421,
                content=error_payload(
                    f"This server does not answer to the host {host!r}. "
                    f"Add it to server.allowed_hosts to permit it.",
                    error_type="invalid_request_error",
                    code="host_not_allowed",
                ),
            )
        return await call_next(request)


#: Built dashboard assets, put here by `make build-dashboard` and shipped inside
#: the wheel. Absent in a source checkout that has never built the front end.
DASHBOARD_DIR = Path(__file__).resolve().parent / "_dashboard"


def mount_dashboard(app: FastAPI) -> None:
    """Serve the dashboard at `/dashboard` and the playground at `/playground`.

    A missing build is answered with a 503 naming the command that fixes it,
    not with a 404: the difference between "this server has no dashboard" and
    "you typed the wrong path" is the whole diagnosis.
    """
    playground_page = DASHBOARD_DIR / "playground.html"
    if (DASHBOARD_DIR / "index.html").is_file():
        if playground_page.is_file():
            # An exact-match route, never a catch-all: `/playground/api` and
            # `/playground/images` are the same prefix and must keep working.
            @app.get("/playground", include_in_schema=False)
            async def playground_page_response() -> FileResponse:
                return FileResponse(playground_page)

        app.mount(
            "/dashboard",
            StaticFiles(directory=DASHBOARD_DIR, html=True),
            name="dashboard",
        )
        return

    @app.get("/dashboard", include_in_schema=False)
    @app.get("/dashboard/{path:path}", include_in_schema=False)
    @app.get("/playground", include_in_schema=False)
    async def dashboard_missing(path: str = "") -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=error_payload(
                "The dashboard was not built into this installation. "
                "Run `make build-dashboard` and reinstall, or use the API directly.",
                error_type="server_error",
                code="dashboard_not_built",
            ),
        )


def create_app(
    settings: Settings | None = None,
    engine: ModelEngine | None = None,
    *,
    request_restart: Callable[[], None] | None = None,
    local_token: str | None = None,
) -> FastAPI:
    """Build the application.

    `request_restart` is how `/admin/restart` reaches the process running this
    app. It is injected rather than reached for, because only the caller that
    owns the uvicorn server knows how to stop it — and an app embedded in a test
    client owns nothing, which is why the default refuses instead of pretending.
    """
    settings = settings or load_settings()
    setup_logging(settings.server.log_level, settings.server.log_file, settings.server.log_json)
    if settings_module.missing_config_path is not None:
        logger.warning(
            "No configuration file at %s: every default applies. "
            "Point QDS_SERVER_CONFIG at your server-config.json.",
            settings_module.missing_config_path,
        )

    registry = settings.registry()
    if not registry:
        raise ValueError("No model enabled: check the 'models' section of server-config.json.")

    engine = engine or ModelEngine(
        request_timeout_s=settings.server.request_timeout_s,
        progress_log_every=settings.server.progress_log_every,
    )
    store = ImageStore(
        Path(settings.server.image_store).expanduser(),
        ttl_s=settings.server.image_ttl_s,
    )
    idle_unloader = IdleUnloader(engine, settings.server.idle_unload_s)
    # Outside `image_store` on purpose: these images belong to a durable session
    # record, and the TTL purge must not be able to reach them. Where "outside"
    # is, is `playground_directory`'s to decide — never this process's CWD.
    playground = PlaygroundStore(settings_module.playground_directory(settings.server))
    scratch_dir = Path(tempfile.mkdtemp(prefix="mflux_scratch_"))
    jobs = JobManager()
    log_buffer = LogBuffer()
    pending = admin.PendingChanges()
    # A finished conversion rewrites the configuration, which is the same fact a
    # manual save reports: this process is now behind its file.
    jobs.on_config_changed = lambda: setattr(pending, "restart_required", True)

    # Every rule a generation is admitted through, in one object: `/v1`, the
    # playground and the MCP tools ask it the same questions and cannot get
    # different answers. It owns the playground runner, because the runner needs
    # its `resolve_spec` and its `submit_upscale` needs the runner.
    admission = Admission(
        settings,
        registry,
        engine=engine,
        store=store,
        playground=playground,
        idle_unloader=idle_unloader,
        scratch_dir=scratch_dir,
    )
    runner = admission.runner

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.purge()
        # Before the runner starts: a generation left `running` by a previous
        # process has no way back, and a record stuck there would show as an
        # eternal spinner.
        playground.mark_interrupted()
        runner.start()
        buffer_handler = logbuffer.attach(log_buffer)
        logger.info(
            "qds %s - %d model(s): %s | default: %s",
            __version__,
            len(registry),
            ", ".join(sorted(registry)),
            settings.default_model,
        )
        if not settings.server.api_key and not settings.server.is_loopback:  # pragma: no cover
            logger.warning("Server exposed without an API key")
        if "*" in settings.server.cors_origins and not settings.server.api_key:
            # Not a `RuntimeIssue`: that refuses to serve, and an operator who
            # deliberately configured a wildcard should not be locked out by an
            # upgrade that changed the default under them.
            logger.warning(
                "cors_origins allows every origin and no api_key is set: any page in "
                "any browser tab can drive this server's data plane and read the results."
            )
        # A mounted application's lifespan is never run by Starlette, and the
        # MCP transport's task group is created here or not at all — without
        # this the first request fails with "Task group is not initialized".
        #
        # Entered last so it exits first: on shutdown an in-flight tool call
        # must be unwound before `runner.shutdown()` and `engine.shutdown()`
        # below, or a tool task would be left touching a stopped runner.
        async with AsyncExitStack() as stack:
            if mcp_mount is not None:
                await stack.enter_async_context(
                    mcp_mount.serving(lambda: mcp_server.session_manager.run())
                )
            yield
        # Before `engine.shutdown()`: the worker may be inside
        # `engine.generate()`, and stopping the engine under it would raise from
        # a task nobody is awaiting.
        await runner.shutdown()
        # Before `engine.shutdown()` too: a pending countdown would otherwise be
        # left dangling on a loop that is closing.
        idle_unloader.cancel()
        # Before the engine too, and not on a best-effort basis: a download or a
        # conversion that outlives this process becomes an orphan under launchd,
        # holding the HuggingFace cache and invisible to whatever starts next.
        await jobs.shutdown()
        engine.shutdown()
        shutil.rmtree(scratch_dir, ignore_errors=True)
        logbuffer.detach(buffer_handler)
        logger.info("Server stopped")

    app = FastAPI(
        title="Quantum Diffusion Server",
        version=__version__,
        lifespan=lifespan,
    )
    install_host_guard(app, settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # After `CORSMiddleware`, therefore *outside* it: it must see the
    # private-network preflight first, because `CORSMiddleware` answers that one
    # with a 400 and an image a chat client asked for never loads. Scoped to
    # `/playground/images/` alone -- see `qds/pna.py` for why it is not the
    # `allow_private_network` flag, which would grant the same to `/v1`.
    app.add_middleware(PrivateNetworkImages)
    # Last, therefore outermost: it must see every response, including the ones
    # the static mounts produce and the ones the middlewares above short-circuit.
    install_security_headers(app)
    install_exception_handlers(app)
    app.mount("/images", StaticFiles(directory=store.directory), name="images")
    # Playground images are *not* a mount: a session can be locked, and a file
    # must then not be served without the session's unlock token. See the
    # `/playground/images/{filename}` route in `qds/playground_routes.py`.

    # ── Authentication ─────────────────────────────────────────────────────
    #
    # One implementation, shared with `create_recovery_app`. See `qds/auth.py`
    # for why that sharing is load-bearing rather than merely tidy.

    sessions = SessionStore()
    throttle = admin.LoginThrottle()
    # The playground's own store and its own throttle. Not the admin's, in both
    # cases and for the same reason: revoking every admin session must not sign
    # the playground out, and a housemate guessing the playground password must
    # not be able to lock the operator out of the control plane.
    playground_sessions = SessionStore()
    playground_throttle = admin.LoginThrottle()
    require_api, require_admin, require_playground = build_dependencies(
        settings, sessions, local_token, playground_sessions=playground_sessions
    )
    auth = Depends(require_api)

    # ── Control plane ──────────────────────────────────────────────────────

    app.include_router(
        admin.build_router(
            settings=settings,
            jobs=jobs,
            log_buffer=log_buffer,
            auth=Depends(require_admin),
            engine=engine,
            version=__version__,
            recovery_error=None,
            request_restart=request_restart or _restart_unavailable,
            pending=pending,
            sessions=sessions,
        )
    )
    app.include_router(
        admin.build_session_router(
            settings=settings, sessions=sessions, throttle=throttle, recovery_error=None
        )
    )
    mount_dashboard(app)

    # ── The OpenAI-compatible data plane ───────────────────────────────────

    app.include_router(build_v1_router(admission, auth))

    # ── The browser playground ─────────────────────────────────────────────

    # `require_playground`, not `auth`: this plane's gate is its own, so
    # `playground_auth_scope` moves it without moving `/v1` or `/admin`. The
    # image routes inside keep `auth` — see `build_playground_router`.
    app.include_router(build_playground_router(admission, Depends(require_playground), auth))
    app.include_router(
        build_playground_session_router(
            settings=settings, sessions=playground_sessions, throttle=playground_throttle
        )
    )

    # ── The playground's password, and the way past a project's ─────────────
    #
    # Both under admin authority, and for the same reason: the admin already owns
    # the disk these files and rows sit on, so this only saves them a text editor
    # and a sqlite shell. It is also what makes `playground_auth_scope: always`
    # recoverable — the plane's password is set, changed and removed from here,
    # by a credential that is not it.

    playground_admin = APIRouter(
        prefix="/admin/playground",
        tags=["admin"],
        dependencies=[Depends(require_admin), Depends(admin.deny_cross_site)],
    )

    @playground_admin.delete("/sessions/{session_id}/password", status_code=204)
    async def admin_strip_session_password(session_id: str) -> None:
        try:
            record = admission.playground.password_record(session_id)
        except KeyError:
            raise admission.not_found(session_id) from None
        if record is None:
            raise playground_lock.not_protected(session_id)
        admission.playground.set_password(session_id, None)
        admission.unlocks.revoke_session(session_id)
        admission.unlock_throttles.forget(session_id)
        logger.warning("playground: session %s password removed by admin", session_id)

    @playground_admin.post("/password")
    async def admin_set_playground_password(body: admin.PlaygroundPasswordRequest) -> dict:
        """Set or change the playground password.

        No `current`, unlike `/admin/password`: the credential being replaced is
        not the one that authorised this call, and demanding it would break the
        recovery path this route exists to be. An admin who has forgotten the
        playground password must still be able to set a new one.
        """
        try:
            await asyncio.to_thread(credential.PLAYGROUND.set_password, body.new)
        except credential.WeakPassword as exc:
            raise APIError(str(exc), status_code=400, code="weak_password") from exc
        # Every playground session was minted against the old password.
        playground_sessions.revoke_all()
        playground_throttle.record_success()
        logger.info("playground password changed; every playground session ended")
        return {"ok": True}

    @playground_admin.delete("/password")
    async def admin_clear_playground_password() -> dict:
        """Remove it. Refused while the scope demands one.

        `playground_auth_scope: always` with no password is a gate with no key:
        the server refuses to start in that state, so allowing this here would
        only produce a configuration that cannot come back. Same rule, same
        reason, as `DELETE /admin/password` off-loopback.
        """
        if settings.server.playground_auth_scope == "always":
            raise APIError(
                "The playground password cannot be removed while "
                "playground_auth_scope is 'always'.",
                status_code=409,
                code="password_required_by_scope",
                param="server.playground_auth_scope",
            )
        credential.PLAYGROUND.clear()
        playground_sessions.revoke_all()
        logger.info("playground password removed")
        return {"ok": True}

    app.include_router(playground_admin)

    # ── The MCP plane ──────────────────────────────────────────────────────
    #
    # A mounted ASGI application rather than a router: the SDK owns the
    # streamable-HTTP transport, and it is a whole app. Mounted *inside* this
    # one, so a model reaches the same engine, the same queue and the same
    # durable sessions as the browser, over the same port and the same
    # credential — and the menubar app, which knows how to start exactly one
    # process on exactly one port, needs to know nothing about it.
    #
    # `streamable_http_path="/"` because the mount already supplies `/mcp`; the
    # SDK's own default would make the route `/mcp/mcp`.
    #
    # The SDK's DNS-rebinding protection is switched off *here* because it is
    # not switched off in this server: `install_host_guard` already refuses an
    # unknown `Host` on every route, driven by `server.allowed_hosts`. Leaving
    # the SDK's on would be a second allowlist, silently refusing `/mcp` the
    # first time an operator adds a LAN name for `/v1`.
    mcp_server = None
    mcp_mount = None
    if settings.mcp.enabled:
        from mcp.server.transport_security import TransportSecuritySettings

        from qds.mcp.asgi import MCPGuard, MCPMount
        from qds.mcp.deps import MCPDeps
        from qds.mcp.server import build_server

        mcp_server = build_server(
            MCPDeps(
                settings=settings,
                store=playground,
                runner=runner,
                engine=engine,
                resolve_spec=admission.resolve_spec,
                resolve_size=admission.resolve_size,
                check_prompt=admission.check_prompt,
                check_capabilities=admission.check_capabilities,
                check_rewrite=admission.check_rewrite,
                check_n=admission.check_n,
                seeds_for=admission.seeds_for,
                submit_upscale=admission.submit_upscale,
                capabilities=_capabilities,
                models=admission.by_public_name,
                base_url=f"http://{settings.server.host}:{settings.server.port}",
            )
        )
        # The handle a test needs. Semantics are quicker to exercise against the
        # server object directly; anything about the *guard* must still go over
        # HTTP, because an in-memory client never passes through it.
        app.state.mcp_server = mcp_server
        mcp_mount = MCPMount(
            lambda: mcp_server.streamable_http_app(
                streamable_http_path="/",
                transport_security=TransportSecuritySettings(
                    enable_dns_rebinding_protection=False
                ),
            )
        )
        app.mount(
            "/mcp",
            MCPGuard(
                mcp_mount,
                authorize=build_authorizer(settings, sessions, local_token),
                deny_cross_site=True,
            ),
        )

    return app


def effective_bind_host(settings: Settings, recovery_error: str | None) -> str:
    """Where the server may actually listen.

    A pure function so the decision can be tested without starting uvicorn — a
    test that cannot observe the bind proves nothing about it.

    The rule closes a hole the recovery path opened: `recovery_settings()` takes
    the host from the environment, so `QDS_SERVER_HOST=0.0.0.0` plus a config
    file that will not parse produced a **wildcard-bound, unauthenticated
    configuration writer**. Recovery mode deliberately leaves `/admin` open when
    no password is set — that is the first-run path — and the two together are a
    control plane on the network with no credential at all.

    So a recovery server binds loopback unless a password exists to protect it.
    The headless-repair case survives: a machine whose config went bad but whose
    password is intact keeps its configured address.
    """
    if recovery_error is None:
        return settings.server.host
    return settings.server.host if credential.is_set() else "127.0.0.1"


def _original_argv() -> list[str]:
    """The command to re-exec, rebuilt rather than remembered.

    `sys.argv[0]` is the console script when started as `qds serve` and
    `__main__.py` when started as `python -m qds`; neither is something to hand
    back to `execv`. `sys.executable -m qds` plus the original arguments names
    the same installation in both cases.
    """
    return [sys.executable, "-m", "qds", *sys.argv[1:]]


def create_recovery_app(
    settings: Settings,
    message: str,
    *,
    request_restart: Callable[[], None] | None = None,
    local_token: str | None = None,
) -> FastAPI:
    """The server a broken configuration gets: repairable, but not generating.

    Refusing to start at all is fail-closed and was also a trap — the screen
    that edits the configuration was served by the process the configuration
    stopped from starting, so the only way out was hand-editing JSON. This keeps
    the control plane and the dashboard up, and answers everything else with a
    503 that names the reason.
    """
    setup_logging(settings.server.log_level, settings.server.log_file, settings.server.log_json)
    jobs = JobManager()
    log_buffer = LogBuffer()
    pending = admin.PendingChanges()
    jobs.on_config_changed = lambda: setattr(pending, "restart_required", True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        buffer_handler = logbuffer.attach(log_buffer)
        logger.warning("qds %s - recovery mode: %s", __version__, message)
        yield
        await jobs.shutdown()
        logbuffer.detach(buffer_handler)
        logger.info("Server stopped")

    app = FastAPI(title="Quantum Diffusion Server (recovery)", version=__version__, lifespan=lifespan)
    install_host_guard(app, settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # The recovery app serves the same login form as the real one, so it gets the
    # same containment: a weaker recovery mode is a way in, not a fallback.
    install_security_headers(app)
    install_exception_handlers(app)

    sessions = SessionStore()
    throttle = admin.LoginThrottle()
    # `recovery=True`: this app exists to repair a configuration, and
    # `admin_auth_scope: always` with no admin password is one of the
    # configurations that lands here. Enforcing the scope on the only screen
    # that can undo it would make the setting a one-way door. Safe because a
    # recovery server with no password binds loopback whatever the file says —
    # see `effective_bind_host`. There is no playground here, so the third
    # dependency has nothing to guard.
    require_api, require_admin, _ = build_dependencies(
        settings, sessions, local_token, recovery=True
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        # Same endpoint, same shape, different `status`. A client that polls
        # this — the menubar app, the dashboard — learns the server is up *and*
        # why it cannot generate, from the one call it already makes.
        return {
            "status": "config_error",
            "version": __version__,
            "error": message,
            "default_model": None,
            "models": [],
            "loaded_model": None,
            "idle_unload_s": None,
            "memory": {},
        }

    app.include_router(
        admin.build_router(
            settings=settings,
            jobs=jobs,
            log_buffer=log_buffer,
            auth=Depends(require_admin),
            engine=None,
            version=__version__,
            recovery_error=message,
            request_restart=request_restart or _restart_unavailable,
            pending=pending,
            sessions=sessions,
        )
    )
    app.include_router(
        admin.build_session_router(
            settings=settings, sessions=sessions, throttle=throttle, recovery_error=message
        )
    )
    app.include_router(admin.build_recovery_router(message=message, version=__version__))
    mount_dashboard(app)
    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    import argparse

    import uvicorn

    # An empty parser rather than none at all: `qds serve --port 9000` has to be
    # refused rather than ignored. Binding and configuration come from
    # `server-config.json` and the `QDS_SERVER_*` overrides, which is one
    # precedence rule; a second one spelled on the command line would be two.
    argparse.ArgumentParser(
        prog="qds serve",
        description=(
            "Run the server. Configuration comes from server-config.json "
            "(QDS_SERVER_CONFIG) and QDS_SERVER_* environment overrides."
        ),
    ).parse_args(argv)

    # *Any* unreadable configuration starts a recovery server, not only one that
    # breaks a runtime invariant. The distinction used to be load-bearing and was
    # exactly backwards: a disabled default model got a repair screen, while the
    # failures people actually produce by hand-editing JSON — a typo, an
    # out-of-range value — killed the process outright and left no way back in
    # except editing the same file again by hand.
    #
    # So three rungs, each falling to the next: strict, then lenient (which still
    # refuses a structurally invalid document), then the environment and the
    # defaults, which cannot fail and at least put the repair screen on the port
    # whatever launched this process is waiting on.
    recovery_error: str | None = None
    settings: Settings
    try:
        settings = load_settings()
    except (ConfigError, ValueError) as exc:
        recovery_error = str(exc)
        logger.error("Starting in recovery mode: %s", exc)
        try:
            settings = load_settings(strict=False)
        except (ConfigError, ValueError):
            settings = recovery_settings()

    # The server keeps whatever root it is given here for its whole lifetime:
    # mflux resolves the cache constant once, at import. Changing the setting
    # therefore takes effect for this process only on restart.
    settings.apply_hf_home()

    restart_wanted = False

    def request_restart() -> None:
        nonlocal restart_wanted
        restart_wanted = True
        server.should_exit = True

    # Issued before either app is built, and on every rung: it is the credential
    # of last resort. If the password is forgotten, or the file holding its hash
    # is the file that will not parse, this is what still lets the menubar app
    # and the CLI reach the control plane and repair things.
    local_token = issue_local_token()

    app = (
        create_recovery_app(
            settings, recovery_error, request_restart=request_restart, local_token=local_token
        )
        if recovery_error is not None
        else create_app(settings, request_restart=request_restart, local_token=local_token)
    )

    # `uvicorn.Server` rather than `uvicorn.run`, so `/admin/restart` has
    # something to set `should_exit` on and this function gets control back
    # afterwards.
    bind_host = effective_bind_host(settings, recovery_error)
    if bind_host != settings.server.host:
        logger.warning(
            "Recovery mode with no admin password: listening on %s instead of %s, "
            "because an unauthenticated control plane must not be reachable from the network.",
            bind_host,
            settings.server.host,
        )

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=bind_host,
            port=settings.server.port,
            log_level=settings.server.log_level.lower(),
            # Without this, uvicorn waits forever on in-flight connections: a
            # SIGTERM during a generation would block for up to
            # `request_timeout_s` (40 min in the shipped config). A supervisor
            # should still keep a SIGTERM → SIGKILL ladder, because a second
            # SIGTERM does not force the exit on uvicorn's side — only SIGINT
            # does.
            timeout_graceful_shutdown=settings.server.shutdown_grace_s,
            # In JSON mode, stdout is the structured-event channel: uvicorn's
            # access log, which writes plain text there, would make it
            # unparsable.
            access_log=not settings.server.log_json,
        )
    )
    server.run()

    # A token file that outlives its server would be a credential for a process
    # that no longer exists. Not before a re-exec, though: the replacement issues
    # its own, and removing it here would leave a window with none.
    if not restart_wanted:
        discard_local_token()

    if restart_wanted:
        # Re-exec rather than exit-and-be-restarted: the pid survives, so this
        # behaves the same whether the menubar app launched the server or
        # somebody typed `qds serve`. Listening sockets do not follow, because
        # Python marks its file descriptors close-on-exec (PEP 446) and uvicorn
        # has closed them by now anyway.
        logger.info("Restarting: re-executing %s", " ".join(_original_argv()))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, _original_argv())

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
