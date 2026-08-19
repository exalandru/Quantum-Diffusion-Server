"""The control plane: everything the dashboard needs that is not generation.

These endpoints replace the Tauri IPC commands the desktop app used to expose.
The move is not a translation exercise — it changes who may do what:

* **The server owns its configuration.** `PUT /admin/config` is the only writer,
  and it validates before replacing. The menubar app reads the file to learn a
  port and never writes it.
* **The server owns its jobs.** A download or a conversion is a child of this
  process, so it dies with it, and `POST /admin/jobs/cancel` reaches it through
  a process group rather than through a supervisor that lives elsewhere.
* **The server does not own its own lifetime.** There is no start or stop here:
  something must already be running for these routes to answer. `POST
  /admin/restart` is the one exception, and it is a re-exec in place rather than
  a stop — which is why it works identically under the menubar app and under a
  bare `qds serve`.

**Browsers make authentication insufficient on its own.** A default install is
keyless on loopback, which was safe while the only client was a native app and
is not once a web page can issue requests. Two checks close the two ways a
hostile page reaches 127.0.0.1: a Host allowlist (a rebound DNS name arrives
with its own Host header) and an Origin check (a cross-site request carries an
Origin the browser sets and the page cannot forge). Neither is a substitute for
the API key; both cover the case where there is not one.

**The Origin check covers reads, not only writes.** An earlier version of this
module checked writes only, on the reasoning that CORS already stopped a
cross-site *read*. It does not: `server.cors_origins` defaults to `["*"]` — on
purpose, so an OpenAI-speaking front end on another origin can call `/v1` — and
that default let any page the user happened to visit read this server's
configuration path, its effective HuggingFace home, whether a token is present,
and its entire log buffer. `/v1` stays deliberately open because being callable
from another origin is what it is for; `/admin` is for the dashboard this server
serves itself, so same-origin is the whole of its audience.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, Cookie, Depends, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from qds import configfile, credential, import_cli
from qds import session as session_module
from qds.errors import APIError
from qds.hosts import is_ip_literal, local_host_names
from qds.jobs import JobBusy, JobManager, NoJobRunning
from qds.logbuffer import LogBuffer
from qds.logs import SERVER_LOGGER
from qds.session import SessionStore
from qds.settings import ConfigError, Settings, config_path

logger = logging.getLogger(f"{SERVER_LOGGER}.admin")


# ── Request bodies ─────────────────────────────────────────────────────────


class HfTokenRequest(BaseModel):
    token: str


class FetchRequest(BaseModel):
    key: str


class PrequantizeRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    bits: int = Field(ge=1, le=16)
    components: list[str] | None = None
    dest: str | None = None


class InspectRequest(BaseModel):
    path: str


class LocateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    path: str
    model: str


class RegisterRequest(BaseModel):
    path: str
    base_profile: str
    name: str | None = None
    api_name: str | None = None


class ForgetRequest(BaseModel):
    id: str


# ── Cross-origin protection ────────────────────────────────────────────────


def _same_origin(origin: str, request: Request) -> bool:
    """Whether `Origin` names the very server handling this request.

    Parsed, not pattern-matched. A suffix test on `//<host>` reads
    `http://evil.example//127.0.0.1:8765` as same-origin, and ignores the scheme
    entirely; no browser would send either, but a check that fails open on input
    it did not anticipate is a check whose guarantee nobody can state.
    """
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    # The `Host` header carries no scheme, so the comparison takes it from the
    # request itself. Without this, `https://127.0.0.1` counts as same-origin
    # for a plain-http server — which it is not, and which a browser would never
    # send to one.
    if parsed.scheme != request.url.scheme:
        return False

    host = (request.headers.get("host") or "").lower()
    if not host:
        return False
    # A default port is implicit in one spelling and explicit in the other often
    # enough that both have to be accepted.
    authority = parsed.netloc.lower()
    default_port = "443" if parsed.scheme == "https" else "80"
    return host in {authority, f"{authority}:{default_port}"} or authority == f"{host}:{default_port}"


async def deny_cross_site(request: Request) -> None:
    """Refuse a request a browser marked as coming from another origin.

    Only when `Origin` is present: a browser sets it on exactly the requests
    that matter here, and curl, the menubar app and the CLI send none — so this
    keeps a hostile page out without asking a script for a header it has no
    reason to send.
    """
    origin = request.headers.get("origin")
    if origin is None or _same_origin(origin, request):
        return
    raise APIError(
        "Cross-site requests are not accepted on the control plane.",
        status_code=403,
        error_type="invalid_request_error",
        code="cross_site_denied",
    )


# ── Logging in ─────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    password: str


class PasswordRequest(BaseModel):
    new: str
    #: Required once a password exists. A hijacked session must not be able to
    #: change the credential it rode in on.
    current: str | None = None


class LoginThrottle:
    """A bound on guessing, on top of the one hashing already imposes.

    Verifying costs ~94 ms, so a single attacker gets about ten tries a second;
    this makes it five per minute. Keyed globally rather than per address: this
    is a single-user server, and per-IP keying is defeated by changing address,
    which is free on a LAN.
    """

    WINDOW_S = 60.0
    ALLOWED = 5
    MAX_LOCKOUT_S = 900.0

    def __init__(self) -> None:
        self._failures: list[float] = []
        self._locked_until = 0.0
        self._lockout = self.WINDOW_S

    def retry_after(self) -> float:
        remaining = self._locked_until - time.monotonic()
        return max(0.0, remaining)

    def record_failure(self) -> None:
        now = time.monotonic()
        self._failures = [at for at in self._failures if now - at < self.WINDOW_S]
        self._failures.append(now)
        if len(self._failures) >= self.ALLOWED:
            self._locked_until = now + self._lockout
            self._lockout = min(self.MAX_LOCKOUT_S, self._lockout * 2)
            self._failures.clear()

    def record_success(self) -> None:
        self._failures.clear()
        self._locked_until = 0.0
        self._lockout = 60.0


def build_session_router(
    *,
    settings: Settings,
    sessions: SessionStore,
    throttle: LoginThrottle,
    recovery_error: str | None,
) -> APIRouter:
    """Logging in, which by definition cannot require being logged in.

    A router of its own because FastAPI's router-level `dependencies` are not
    overridable per route: a login endpoint on the protected router would be
    unreachable by construction. It keeps `deny_cross_site`, so a hostile page
    still cannot post a guess from another origin.
    """
    router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(deny_cross_site)])

    @router.get("/session")
    async def session_status(
        qds_admin: Annotated[str | None, Cookie(alias=session_module.COOKIE)] = None,
    ) -> dict[str, Any]:
        """What the login screen needs, and nothing more.

        Whether a password exists is not a secret worth keeping from a
        same-origin caller — the alternative is a login form that cannot tell
        "set one" from "enter yours".
        """
        return {
            "passwordSet": credential.is_set(),
            "authenticated": sessions.validate(qds_admin),
            "loopback": settings.server.is_loopback,
            "recoveryMode": recovery_error is not None,
        }

    @router.post("/session", status_code=204)
    async def log_in(body: LoginRequest, response: Response) -> None:
        wait = throttle.retry_after()
        if wait > 0:
            raise APIError(
                f"Too many attempts. Try again in {wait:.0f} seconds.",
                status_code=429,
                error_type="invalid_request_error",
                code="too_many_attempts",
            )
        if not credential.is_set():
            raise APIError(
                "No admin password is set on this server.",
                status_code=409,
                code="no_password_set",
            )
        if not credential.verify(body.password):
            throttle.record_failure()
            # At WARNING so it lands in the ring buffer and shows up in the
            # dashboard's Logs view: repeated failures are worth seeing.
            logger.warning("admin login failed")
            raise APIError(
                "Incorrect password.",
                status_code=401,
                error_type="invalid_request_error",
                code="invalid_password",
            )

        throttle.record_success()
        token = sessions.create()
        response.set_cookie(
            session_module.COOKIE,
            token,
            httponly=True,
            samesite="strict",
            path="/",
            # `secure` is deliberately absent: this server speaks plain HTTP, and
            # a Secure cookie would simply never be sent over the LAN address the
            # network option exists to enable. The cost is real — on a LAN the
            # cookie is observable — and is stated in the interface rather than
            # papered over here.
        )
        # Nothing returned: FastAPI applies `status_code` and merges the
        # cookie set on the injected response. Returning it instead produces a
        # reply with no status at all.
        logger.info("admin logged in")

    @router.delete("/session", status_code=204)
    async def log_out(
        response: Response,
        qds_admin: Annotated[str | None, Cookie(alias=session_module.COOKIE)] = None,
    ) -> None:
        # No auth: presenting a cookie the server does not know and being told
        # you are logged out is the correct outcome, not an error.
        sessions.revoke(qds_admin)
        response.delete_cookie(session_module.COOKIE, path="/")

    return router


# ── Pending changes ────────────────────────────────────────────────────────


class PendingChanges:
    """Whether the running process is now behind its own configuration.

    One flag with one meaning, because there used to be two: `PUT /admin/config`
    answered `restartRequired: true` while `/admin/overview` answered `false`
    from a different field a config write never touched. A dashboard that saves
    and then refreshes — the natural order — was told both.

    Two things set it, and they are genuinely the same fact: someone replaced
    the configuration, or a finished conversion selected a variant in it. Either
    way the file on disk now says something this process did not read at start.
    """

    def __init__(self) -> None:
        self.restart_required = False


# ── Router ─────────────────────────────────────────────────────────────────


def build_router(
    *,
    settings: Settings,
    jobs: JobManager,
    log_buffer: LogBuffer,
    auth: Any,
    engine: Any | None,
    version: str,
    recovery_error: str | None,
    request_restart: Any,
    pending: PendingChanges,
    sessions: SessionStore,
) -> APIRouter:
    """Assemble `/admin`.

    `engine` is `None` in recovery mode, where no model stack was built; every
    route that would touch it says so rather than failing on an attribute.
    """
    # Resolved once, when the router is built.
    lan_addresses = sorted(
        name for name in local_host_names() if is_ip_literal(name) and not name.startswith("127.")
    )

    router = APIRouter(
        prefix="/admin",
        tags=["admin"],
        # Both on every route, reads included: the API key may be unset, and
        # a permissive `cors_origins` would otherwise let another origin read
        # everything here.
        dependencies=[auth, Depends(deny_cross_site)],
    )

    # ── Overview ───────────────────────────────────────────────────────────

    @router.get("/overview")
    async def overview() -> dict[str, Any]:
        hf_home = settings.effective_hf_home
        token_file = f"{hf_home.rstrip('/')}/token"
        from pathlib import Path

        return {
            "version": version,
            "configPath": str(config_path()),
            "dataDir": str(config_path().parent),
            "effectiveHfHome": hf_home,
            "effectiveCacheDir": settings.effective_cache_dir,
            "hfTokenPresent": Path(token_file).expanduser().is_file(),
            "recoveryMode": recovery_error is not None,
            "recoveryError": recovery_error,
            "restartRequired": pending.restart_required,
            "adminPasswordSet": credential.is_set(),
            # Computed once at startup, never per request: resolving this
            # machine's own name can block on a network where it does not.
            "lanAddresses": lan_addresses,
            "server": {"host": settings.server.host, "port": settings.server.port},
        }

    # ── Configuration ──────────────────────────────────────────────────────

    @router.get("/config")
    async def read_config() -> Any:
        """The document, or — when it will not parse — its text and the reason.

        The unparseable case is not hypothetical: it is precisely the state that
        starts the server in recovery mode, so this endpoint is the first thing
        the repair screen calls. Answering it with a 500 would leave the
        dashboard with nothing to show and nothing to edit, which is the trap
        recovery mode exists to avoid. The caller can tell the two apart because
        one is a JSON object and the other carries `unparsed: true`.
        """
        try:
            return configfile.read()
        except configfile.ConfigWriteError as exc:
            return JSONResponse(
                status_code=200,
                content={
                    "unparsed": True,
                    "reason": str(exc),
                    "text": configfile.read_text(),
                },
            )

    @router.put("/config")
    async def write_config(document: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        """Validate, then replace.

        Validation happens against the same `Settings` model the server starts
        with — not a second schema written for this endpoint, which would drift
        and would let the dashboard save a document that then refuses to boot.
        Structural validity is required; *runtime* validity is not, because a
        configuration whose default model is switched off has to be savable in
        order to be repairable, and `runtime_issues` reports it instead.
        """
        try:
            configfile.refuse_disabled_default(document)
        except configfile.DisabledDefaultModel as exc:
            raise APIError(
                str(exc), status_code=400, code="disabled_default_model", param="default_model"
            ) from exc

        try:
            candidate = Settings.model_validate(document)
            candidate.registry()
        except (ConfigError, ValueError) as exc:
            raise APIError(
                f"This configuration would not load: {exc}",
                status_code=400,
                code=getattr(exc, "code", "invalid_config"),
                param=getattr(exc, "field", None),
            ) from exc

        try:
            configfile.write(document)
        except configfile.ConfigWriteError as exc:
            raise APIError(str(exc), status_code=500, code="config_unwritable") from exc

        pending.restart_required = True
        logger.info("configuration replaced through the admin API")
        return {
            "ok": True,
            # Nothing here is applied to the running process: the server reads
            # its settings once, at start, and mflux freezes the cache root at
            # import. Saying so is the difference between a stale interface and
            # an honest one.
            "restartRequired": pending.restart_required,
            "issues": [issue.as_dict() for issue in candidate.runtime_issues()],
        }

    @router.post("/password")
    async def set_password(body: PasswordRequest) -> dict[str, Any]:
        """Set the admin password, or change it.

        On the protected router, so reaching it already required a credential —
        which on a fresh loopback install is "no password is set", and that is
        the whole of the first-run path. It is safe there only because a server
        bound beyond loopback refuses to start without one.
        """
        if credential.is_set():
            # Even with a valid session: a hijacked one must not be able to
            # change the credential it rode in on.
            if not body.current or not credential.verify(body.current):
                raise APIError(
                    "The current password is incorrect.",
                    status_code=401,
                    error_type="invalid_request_error",
                    code="invalid_password",
                )
        try:
            credential.set_password(body.new)
        except credential.WeakPassword as exc:
            raise APIError(str(exc), status_code=400, code="weak_password") from exc

        # Every session was minted against the old password.
        sessions.revoke_all()
        logger.info("admin password changed; all sessions ended")
        return {"ok": True, "loggedOut": True}

    @router.delete("/password")
    async def clear_password() -> dict[str, Any]:
        """Remove the password. Loopback only.

        Off-loopback the server refuses to start without one, so allowing this
        there would only produce a configuration that cannot come back.
        """
        if not settings.server.is_loopback:
            raise APIError(
                "The admin password cannot be removed while the server listens "
                "beyond this machine.",
                status_code=409,
                code="password_required_off_loopback",
            )
        credential.clear()
        sessions.revoke_all()
        logger.info("admin password removed")
        return {"ok": True}

    @router.post("/hf-token")
    async def write_hf_token(body: HfTokenRequest) -> dict[str, Any]:
        from pathlib import Path

        token = body.token.strip()
        if not token:
            raise APIError("The token is empty.", status_code=400, code="empty_token")
        home = Path(settings.effective_hf_home).expanduser()
        try:
            home.mkdir(parents=True, exist_ok=True)
            target = home / "token"
            # 0600 from creation: a credential must never exist world-readable,
            # not even for the moment between writing and chmod.
            import os

            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(token + "\n")
        except OSError as exc:
            raise APIError(
                f"could not write {home / 'token'}: {exc}",
                status_code=500,
                code="token_unwritable",
            ) from exc
        logger.info("HuggingFace token written to %s", home / "token")
        return {"ok": True}

    # ── Catalogue ──────────────────────────────────────────────────────────

    @router.get("/models")
    async def models() -> dict[str, Any]:
        """The catalogue with cache state, computed in this process.

        The desktop app shelled out to `qds fetch --status` because Rust could
        not call Python. Nothing here imports mflux or torch, so there is no
        reason for a subprocess now — and one fewer place for the two answers to
        disagree.
        """
        from qds.fetch import catalogue_status

        return catalogue_status()

    # ── Jobs ───────────────────────────────────────────────────────────────

    @router.get("/jobs")
    async def job_status() -> dict[str, Any]:
        status = await jobs.status()
        return status.as_dict()

    @router.post("/jobs/fetch", status_code=202)
    async def start_fetch(body: FetchRequest) -> dict[str, Any]:
        try:
            status = await jobs.start_fetch(body.key)
        except JobBusy as exc:
            raise APIError(str(exc), status_code=409, code="job_busy") from exc
        return status.as_dict()

    @router.post("/jobs/prequantize", status_code=202)
    async def start_prequantize(body: PrequantizeRequest) -> dict[str, Any]:
        # Free the resident model first. A conversion peaks around 66 GB, and
        # letting it start next to a warm model is how both end up swapping.
        if engine is not None:
            await engine.unload()
        try:
            status = await jobs.start_prequantize(
                body.model, body.bits, body.components, body.dest
            )
        except JobBusy as exc:
            raise APIError(str(exc), status_code=409, code="job_busy") from exc
        return status.as_dict()

    @router.post("/jobs/cancel")
    async def cancel_job() -> dict[str, Any]:
        try:
            status = await jobs.cancel()
        except NoJobRunning as exc:
            raise APIError(str(exc), status_code=409, code="no_job_running") from exc
        return status.as_dict()

    # ── Local models ───────────────────────────────────────────────────────
    #
    # In-process, unlike the jobs above: these are directory reads and a small
    # JSON file, they finish in milliseconds, and `import_cli` exposes each one
    # as a function returning the same payload its command prints.

    @router.post("/import/inspect")
    async def import_inspect(body: InspectRequest) -> dict[str, Any]:
        return import_cli.do_inspect(body.path)

    @router.post("/import/locate")
    async def import_locate(body: LocateRequest) -> dict[str, Any]:
        return import_cli.do_locate(body.path, body.model)

    @router.post("/import/register")
    async def import_register(body: RegisterRequest) -> dict[str, Any]:
        return import_cli.do_register(body.path, body.base_profile, body.name, body.api_name)

    @router.get("/import")
    async def import_list() -> dict[str, Any]:
        return import_cli.do_list()

    @router.post("/import/forget")
    async def import_forget(body: ForgetRequest) -> dict[str, Any]:
        return import_cli.do_forget(body.id)

    # ── Logs ───────────────────────────────────────────────────────────────

    @router.get("/logs")
    async def logs(
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=2000)] = 500,
    ) -> dict[str, Any]:
        return log_buffer.after(after, limit)

    # ── Restart ────────────────────────────────────────────────────────────

    @router.post("/restart", status_code=202)
    async def restart() -> dict[str, Any]:
        """Ask the process to re-exec itself once this response is out.

        Re-exec rather than exit: the same pid comes back, so nothing needs a
        supervisor's cooperation and the behaviour is identical whether the
        server was started by the menubar app or by hand.
        """
        request_restart()
        return {"ok": True, "restarting": True}

    return router


def build_recovery_router(*, message: str, version: str) -> APIRouter:
    """The `/v1` surface, in recovery mode: present, and honest about why not.

    Without this, a configuration the server refuses to start on would answer
    generation requests with a bare 404, which reads as "wrong URL" rather than
    "this server has no models loaded because its configuration is broken".
    """
    router = APIRouter(prefix="/v1", tags=["recovery"])

    async def unavailable() -> None:
        raise APIError(
            f"The server started in recovery mode and cannot generate: {message}",
            status_code=503,
            error_type="server_error",
            code="config_error",
        )

    for path in ("/models", "/capabilities", "/progress"):
        router.add_api_route(path, unavailable, methods=["GET"])
    for path in ("/images/generations", "/images/edits", "/cancel", "/unload"):
        router.add_api_route(path, unavailable, methods=["POST"])
    return router
