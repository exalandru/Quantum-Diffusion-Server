"""HTTP client for the QDS playground data plane.

One place owns the transport for every ``qds_*`` tool: base URL, auth
header, timeouts, error mapping, and the two non-JSON reads (a streamed
image file and the first frame of the progress SSE stream).

Deliberate choices:

- **No ``Origin`` header.** The playground router is wrapped in QDS'
  ``deny_cross_site`` guard, which exists to stop a *browser page* on
  another origin from driving this machine's GPU. A plain client sends no
  ``Origin`` and is therefore not cross-site. Faking one would be lying to
  a security guard, so we never send it.
- **No admin credential.** Every route used here lives on the data plane
  (``require_api``), which is keyless on a loopback install and takes a
  bearer key when one is configured. The admin cookie (install, quantize,
  restart, config) is out of scope for this plugin by design.
- **``trust_env=False``.** A generation submit is a call to
  ``127.0.0.1``; an ambient ``HTTP_PROXY`` or ``.netrc`` entry has no
  business in it, and both are a way for a loopback request to silently
  leave the machine.

Phase 1 (``hermes/image_gen/qds``) keeps its own client on purpose: the
two plugins are independently enabled and must not be coupled.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

#: Where QDS listens when nothing says otherwise.
DEFAULT_BASE_URL = "http://127.0.0.1:8765"

#: A health probe gates `hermes tools`; it must never stall the CLI.
HEALTH_TIMEOUT_S = 2.0
#: Catalogue / session reads are cheap and local.
META_TIMEOUT_S = 10.0
#: A submit only enqueues (HTTP 202), it does not wait for pixels. The
#: budget covers multipart upload of one source image, not a generation.
SUBMIT_TIMEOUT_S = 120.0
#: Downloading one finished PNG from the local filesystem-backed store.
FILE_TIMEOUT_S = 60.0
#: The progress stream emits a frame immediately; anything slower than
#: this is a stuck server, and blocking a tool call on it is worse than
#: reporting "no snapshot".
SSE_TIMEOUT_S = 5.0
#: Hard cap on a single downloaded image, so a misbehaving/renamed route
#: cannot make a tool call eat memory.
MAX_IMAGE_BYTES = 64 * 1024 * 1024


def base_url() -> str:
    """Return the QDS base URL (``QDS_BASE_URL``-overridable), no trailing slash."""
    raw = (os.environ.get("QDS_BASE_URL") or "").strip() or DEFAULT_BASE_URL
    return raw.rstrip("/")


def api_key() -> Optional[str]:
    """Return the data-plane bearer key, or ``None`` for a keyless server.

    **Environment only.** ``QDS_API_KEY`` (this plugin's own knob) wins, then
    QDS' own ``QDS_SERVER_API_KEY``. The value is never echoed into a tool
    result.

    This used to fall back to reading ``server.api_key`` out of QDS'
    ``server-config.json``, and that was wrong twice over:

    - **It assumes the client shares a filesystem with the server.** The moment
      QDS is exposed on the network and Hermes runs on another machine — the
      deployment the server's `playground_auth_scope` exists to make safe —
      there is no config file to read, and the plugin would report "keyless"
      while the server demanded a key.
    - **It behaved differently depending on where the file sat.** The lookup
      resolved `parents[2]/src/server/server-config.json`, which exists when the
      plugin runs from a checkout of this repo and does not exist once it is
      deployed as a copy under `~/.hermes/plugins/`. Identical code, two
      behaviours, and the one that worked was the one nobody ships.

    So the credential is configuration, stated once, the same way on every
    machine. A keyless loopback server still needs nothing set — that is the
    documented default and it is unchanged.
    """
    for var in ("QDS_API_KEY", "QDS_SERVER_API_KEY"):
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return None


class QdsError(RuntimeError):
    """Base error for every QDS transport failure."""


class QdsUnreachable(QdsError):
    """The server did not answer (not started, wrong port, timeout)."""


class QdsHttpError(QdsError):
    """The server answered with HTTP >= 400."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _error_message(resp: requests.Response) -> Tuple[str, Optional[str]]:
    """Extract ``(message, code)`` from a QDS error body.

    Two shapes exist: the playground/data-plane ``{"error": "...",
    "code": "..."}`` and the OpenAI-compatible ``{"error": {"message":
    ..., "code": ...}}``. Anything else degrades to the truncated body.
    """
    payload: Any = None
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        code = payload.get("code")
        if isinstance(err, dict):
            message = err.get("message")
            inner_code = err.get("code")
            if isinstance(message, str) and message:
                return message, inner_code if isinstance(inner_code, str) else None
        elif isinstance(err, str) and err:
            return err, code if isinstance(code, str) else None
    text = (resp.text or "").strip()
    return (text[:400] or f"HTTP {resp.status_code}"), None


class QdsClient:
    """Thin ``requests`` wrapper over the QDS data plane."""

    def __init__(self, base: Optional[str] = None, key: Optional[str] = None) -> None:
        self.base = (base or base_url()).rstrip("/")
        self._key = key if key is not None else api_key()
        self._session = requests.Session()
        # A loopback call must not be routed through an ambient proxy, and
        # must not pick up ~/.netrc credentials.
        self._session.trust_env = False

    # -- low level ----------------------------------------------------------

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        if extra:
            headers.update(extra)
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = META_TIMEOUT_S,
        stream: bool = False,
        accept_404: bool = False,
    ) -> requests.Response:
        """Perform one request, raising a typed error on failure.

        ``accept_404`` returns the 404 response instead of raising — the
        preview route uses 404 to mean "nothing rendering right now",
        which is a state, not a failure.
        """
        url = f"{self.base}{path}"
        try:
            resp = self._session.request(
                method,
                url,
                params=params,
                json=json_body,
                data=data,
                files=files,
                headers=self._headers(headers),
                timeout=timeout,
                stream=stream,
            )
        except requests.RequestException as exc:
            raise QdsUnreachable(f"QDS server unreachable at {self.base}: {exc}") from exc
        if resp.status_code == 404 and accept_404:
            return resp
        if resp.status_code == 401:
            # The one failure a user cannot diagnose from the server's own
            # wording. QDS says "login required", which is true and useless
            # here: this is not a browser, it will never see a login screen, and
            # the fix is an environment variable on *this* machine. Naming it —
            # and the URL it was aimed at — is the difference between a
            # two-minute fix and an afternoon.
            if stream:
                resp.close()
            _, code = _error_message(resp)
            held = "set" if self._key else "not set"
            raise QdsHttpError(
                f"QDS refused the request at {self.base} (HTTP 401). "
                f"This plugin's credential is {held}. "
                f"Set QDS_API_KEY to the server's server.api_key — and "
                f"QDS_BASE_URL if QDS runs on another machine. "
                f"The plugin reads credentials from the environment only; it "
                f"cannot read the server's config file, which would not exist "
                f"on this machine anyway.",
                status_code=401,
                code=code,
            )
        if resp.status_code >= 400:
            message, code = _error_message(resp)
            if stream:
                resp.close()
            raise QdsHttpError(
                f"QDS returned HTTP {resp.status_code}: {message}",
                status_code=resp.status_code,
                code=code,
            )
        return resp

    def json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Perform one request and decode its JSON body (``None`` for 204)."""
        resp = self.request(method, path, **kwargs)
        if resp.status_code == 204 or not (resp.content or b"").strip():
            return None
        try:
            return resp.json()
        except Exception as exc:
            raise QdsError(f"QDS returned a non-JSON response for {path}: {exc}") from exc

    # -- specialised reads --------------------------------------------------

    def health(self) -> bool:
        """Whether ``GET /health`` reports ``status == "ok"``."""
        try:
            payload = self.json("GET", "/health", timeout=HEALTH_TIMEOUT_S)
        except QdsError as exc:
            logger.debug("QDS health probe failed: %s", exc)
            return False
        return isinstance(payload, dict) and payload.get("status") == "ok"

    def download(self, path: str, *, timeout: float = FILE_TIMEOUT_S) -> Tuple[bytes, str]:
        """Download a binary body (an image), bounded by :data:`MAX_IMAGE_BYTES`."""
        resp = self.request("GET", path, timeout=timeout, stream=True)
        try:
            chunks = []
            total = 0
            for chunk in resp.iter_content(64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise QdsError(
                        f"{path} exceeded the {MAX_IMAGE_BYTES // (1024 * 1024)} MB "
                        "download limit"
                    )
                chunks.append(chunk)
            content_type = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        finally:
            resp.close()
        return b"".join(chunks), content_type

    def preview(self) -> Optional[bytes]:
        """Latest partial JPEG of the running session generation, or ``None``.

        A 404 is the idle/loading/``/v1``-job case: no frame exists yet.
        """
        resp = self.request(
            "GET",
            "/playground/api/preview",
            timeout=META_TIMEOUT_S,
            stream=True,
            accept_404=True,
        )
        try:
            if resp.status_code == 404:
                return None
            data = resp.content
        finally:
            resp.close()
        return data if data else None

    def progress_snapshot(self) -> Optional[Dict[str, Any]]:
        """First ``data:`` frame of ``GET /v1/progress``, then close the stream.

        ``/v1/progress`` is an endless SSE stream, so a tool call reads
        exactly one frame and hangs up. ``: ping`` comment lines are
        skipped. ``None`` means the server produced no frame inside
        :data:`SSE_TIMEOUT_S` — reported as such rather than blocking.
        """
        resp = self.request(
            "GET",
            "/v1/progress",
            headers={"Accept": "text/event-stream"},
            timeout=SSE_TIMEOUT_S,
            stream=True,
        )
        try:
            lines: Iterator[str] = resp.iter_lines(decode_unicode=True)
            for _ in range(64):  # bounded: ping comments before the first frame
                try:
                    line = next(lines)
                except StopIteration:
                    return None
                except requests.RequestException:
                    return None
                if not line or not line.startswith("data:"):
                    continue
                try:
                    frame = json.loads(line[5:].strip())
                except Exception:
                    return None
                return frame if isinstance(frame, dict) else None
            return None
        finally:
            resp.close()
