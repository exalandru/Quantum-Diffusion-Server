"""Model-callable tools for the QDS playground (toolset ``qds``).

Eight tools, all against a **local** QDS instance: nothing leaves the
machine, there is no API key on a loopback install, and no image is ever
fabricated. Every handler returns a JSON string — ``tool_result`` on
success, ``tool_error`` on any failure — and never lets an exception
escape.

Why the playground API rather than ``/v1``:

- generations are **durable** in a session (listable, upscalable,
  cancellable by id) instead of a one-shot request/response,
- a session generation is the only kind for which the server exposes a
  live partial frame (``GET /playground/api/preview``), which is what
  makes ``qds_progress`` meaningful.

Submission is asynchronous by design: ``qds_generate`` returns a
``generation_id`` (HTTP 202), and ``qds_wait`` polls a **bounded** number
of seconds and returns either the finished images or the latest snapshot,
for the model to re-call. A tool call never blocks for the whole length of
a 20 GB cold model load.

Finished images are downloaded and written under
``$HERMES_HOME/cache/images/``: the server's own ``/playground/images``
URL needs the server (and its auth) to resolve, while a local file does
not expire and can be handed to any downstream consumer as
``MEDIA:<path>``.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.registry import tool_error, tool_result

from .client import (
    MAX_IMAGE_BYTES,
    META_TIMEOUT_S,
    SUBMIT_TIMEOUT_S,
    QdsClient,
    QdsError,
    QdsHttpError,
    QdsUnreachable,
    base_url,
)

logger = logging.getLogger(__name__)

#: Sessions scanned when a generation's owning session was not supplied.
_MAX_SESSION_SCAN = 200
#: `qds_wait` upper bound. Long enough for a warm small model to finish in
#: one call, short enough that a tool call is never mistaken for a hang.
_MAX_WAIT_CAP_S = 300
_WAIT_POLL_INTERVAL_S = 1.0

#: The server's own vocabulary (`playground.ACTIVE_STATUSES`): a generation
#: that is not queued or running has reached a terminal state, and the only
#: successful terminal state is `completed`. Deriving "terminal" as "not
#: active" mirrors the module-level invariant QDS documents (every accepted
#: generation reaches a terminal state, even across a crash) instead of
#: keeping a second, drift-prone list of end states here.
_ACTIVE_STATUSES = {"queued", "running"}
_SUCCESS_STATUS = "completed"


# ---------------------------------------------------------------------------
# Availability gate + error funnel
# ---------------------------------------------------------------------------


def _check_qds_available() -> bool:
    """Whether the local QDS server answers ``GET /health`` with ``ok``."""
    try:
        return QdsClient().health()
    except Exception as exc:  # pragma: no cover - defensive, health never raises
        logger.debug("QDS availability check failed: %s", exc)
        return False


def _handler(fn: Callable[..., str]) -> Callable[..., str]:
    """Turn every failure into a clean ``tool_error`` JSON string.

    A leaking exception out of a tool handler is a broken turn for the
    agent; a typed error is a fact it can act on. Server errors keep the
    server's own status/code so the model can distinguish "unknown model"
    from "session locked".
    """

    @wraps(fn)
    def wrapper(args: Optional[dict] = None, **kwargs: Any) -> str:
        try:
            return fn(args or {}, **kwargs)
        except QdsUnreachable as exc:
            return tool_error(str(exc), success=False, base_url=base_url())
        except QdsHttpError as exc:
            extra: Dict[str, Any] = {"success": False, "status_code": exc.status_code}
            if exc.code:
                extra["code"] = exc.code
            return tool_error(str(exc), **extra)
        except QdsError as exc:
            return tool_error(str(exc), success=False)
        except ValueError as exc:
            return tool_error(str(exc), success=False)
        except Exception as exc:  # pragma: no cover - last resort
            logger.debug("QDS tool %s failed", fn.__name__, exc_info=True)
            return tool_error(f"QDS tool failed: {type(exc).__name__}: {exc}", success=False)

    return wrapper


# ---------------------------------------------------------------------------
# Argument coercion — a wrong type from the model is a clean error, not a 422
# ---------------------------------------------------------------------------


def _text(args: dict, key: str) -> Optional[str]:
    value = args.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _integer(args: dict, key: str, *, minimum: Optional[int] = None) -> Optional[int]:
    value = args.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {number}")
    return number


def _flag(args: dict, key: str, default: bool = False) -> bool:
    value = args.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
    return default


def _action(args: dict, allowed: Tuple[str, ...], default: str) -> str:
    action = (_text(args, "action") or default).lower()
    if action not in allowed:
        raise ValueError(f"Unknown action {action!r}. Expected one of: {', '.join(allowed)}.")
    return action


# ---------------------------------------------------------------------------
# Local materialisation
# ---------------------------------------------------------------------------


def _images_cache_dir() -> Path:
    """``$HERMES_HOME/cache/images/``, created on demand."""
    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    except Exception:  # pragma: no cover - Hermes always provides this
        home = Path(os.environ.get("HERMES_HOME") or "~/.hermes").expanduser()
    path = Path(home) / "cache" / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_bytes(data: bytes, *, prefix: str, extension: str) -> Path:
    """Write bytes into the Hermes image cache, mirroring ``save_b64_image``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _images_cache_dir() / f"{prefix}_{stamp}_{short}.{extension}"
    path.write_bytes(data)
    return path


def _load_source_image(image_path: Optional[str], image_url: Optional[str]) -> Tuple[bytes, str]:
    """Load the single source image for an image-to-image / edit generation.

    A local path goes through Hermes' shared credential-read guard: this
    file is uploaded to a server, and ``~/.ssh/id_rsa`` is not an image.
    """
    if image_path:
        from agent.file_safety import raise_if_read_blocked

        raise_if_read_blocked(image_path)
        resolved = Path(image_path).expanduser()
        data = resolved.read_bytes()
        return data, resolved.name or "image.png"

    assert image_url is not None
    import requests

    # External source URLs are model-supplied input: do not inherit ambient
    # proxy/netrc credentials, and do not read an unbounded body into memory.
    session = requests.Session()
    session.trust_env = False
    resp = None
    try:
        resp = session.get(image_url, timeout=60, stream=True)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        if length and int(length) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"source image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
            )
        chunks: List[bytes] = []
        total = 0
        for chunk in resp.iter_content(64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError(
                    f"source image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
                )
            chunks.append(chunk)
    finally:
        if resp is not None:
            resp.close()
        session.close()
    name = image_url.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
    return b"".join(chunks), name


# ---------------------------------------------------------------------------
# The attached session
# ---------------------------------------------------------------------------

#: The playground session this chat is driving, set by ``qds_playground``.
#:
#: This is what "the chat stays connected to the session" means: the pane shows
#: one session, and every tool here defaults to that same one, so the model does
#: not have to thread a session id through a conversation the user is steering in
#: plain language ("now make it wider", "upscale that one"). Process-local by
#: design — it is a pointer to what is on screen, not durable state. Restart
#: Hermes and there is no pane either.
_ATTACHED_SESSION: Optional[str] = None


def _attach_session(session_id: str) -> None:
    global _ATTACHED_SESSION
    _ATTACHED_SESSION = session_id


def _session_arg(args: dict) -> Optional[str]:
    """The session a call should act on: explicit argument, else the attached one."""
    return _text(args, "session_id") or _ATTACHED_SESSION


def _pane_url(session_id: str) -> str:
    """The embedded playground URL for a session — what goes in the preview pane."""
    return f"{base_url()}/playground?session={session_id}&view=plugin"


# ---------------------------------------------------------------------------
# Shared server reads
# ---------------------------------------------------------------------------


def _image_filename(url: Any) -> Optional[str]:
    """``/playground/images/<name>`` → ``<name>``."""
    if not isinstance(url, str) or not url.strip():
        return None
    return url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or None


def _session_generations(client: QdsClient, session_id: str) -> List[dict]:
    detail = client.json("GET", f"/playground/api/sessions/{session_id}", timeout=META_TIMEOUT_S)
    generations = detail.get("generations") if isinstance(detail, dict) else None
    return [g for g in generations if isinstance(g, dict)] if isinstance(generations, list) else []


def _find_generation(
    client: QdsClient,
    generation_id: str,
    session_id: Optional[str],
) -> Tuple[str, dict]:
    """Locate a generation record. QDS has no GET-by-id, so select from a session.

    With ``session_id`` this is one read. Without it, sessions are scanned
    (newest first, bounded) — the submit response carries ``sessionId``, so
    a model that passes it back pays for one request instead.
    """
    if session_id:
        for record in _session_generations(client, session_id):
            if record.get("id") == generation_id:
                return session_id, record
        raise ValueError(
            f"No generation {generation_id!r} in session {session_id!r}."
        )

    listing = client.json("GET", "/playground/api/sessions", timeout=META_TIMEOUT_S)
    sessions = listing.get("sessions") if isinstance(listing, dict) else None
    for entry in (sessions or [])[:_MAX_SESSION_SCAN]:
        candidate = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(candidate, str):
            continue
        try:
            records = _session_generations(client, candidate)
        except QdsHttpError as exc:
            if exc.status_code == 403:  # locked session: not ours to read
                continue
            raise
        for record in records:
            if record.get("id") == generation_id:
                return candidate, record
    raise ValueError(
        f"No generation {generation_id!r} found in any visible session. Pass "
        "session_id if the session is locked."
    )


def _generation_summary(record: dict) -> Dict[str, Any]:
    """The fields a model needs from a generation record, nothing else."""
    return {
        "generation_id": record.get("id"),
        "session_id": record.get("sessionId"),
        "group": record.get("groupId"),
        "status": record.get("status"),
        "kind": record.get("kind"),
        "model": record.get("model"),
        "prompt": record.get("prompt"),
        "rewritten_prompt": record.get("rewrittenPrompt"),
        "rewrite_error": record.get("rewriteError"),
        "n": record.get("n"),
        "size": record.get("size"),
        "steps": record.get("steps"),
        "seeds": record.get("seeds"),
        "context_image": record.get("contextImage"),
        "error": record.get("error"),
        "images": [
            {"filename": _image_filename(img.get("url")), "seed": img.get("seed")}
            for img in record.get("images") or []
            if isinstance(img, dict)
        ],
        "created_at": record.get("createdAt"),
        "started_at": record.get("startedAt"),
        "finished_at": record.get("finishedAt"),
    }


def _download_generation_images(client: QdsClient, record: dict) -> List[Dict[str, Any]]:
    """Fetch every finished image of a record into the Hermes image cache."""
    model = str(record.get("model") or "qds")
    prefix = f"qds_{model}".replace("/", "_")
    saved: List[Dict[str, Any]] = []
    for image in record.get("images") or []:
        if not isinstance(image, dict):
            continue
        filename = _image_filename(image.get("url"))
        if not filename:
            continue
        data, content_type = client.download(f"/playground/images/{filename}")
        extension = "png" if content_type in ("", "image/png") else content_type.split("/")[-1]
        path = _save_bytes(data, prefix=prefix, extension=extension)
        saved.append(
            {
                "path": str(path),
                "seed": image.get("seed"),
                "filename": filename,
                "bytes": len(data),
            }
        )
    return saved


# ---------------------------------------------------------------------------
# qds_models
# ---------------------------------------------------------------------------

QDS_MODELS_SCHEMA = {
    "name": "qds_models",
    "description": (
        "List the image models actually installed on the local QDS server, with "
        "their capabilities (default size/steps, image-to-image and edit support, "
        "dimension limits, license) and the server's default model. Read-only. "
        "Call this before qds_generate when the user names a model or asks what "
        "is available — the catalogue is a property of the machine, never assume it."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


@_handler
def _handle_qds_models(args: dict, **_kw: Any) -> str:
    client = QdsClient()
    caps = client.json("GET", "/v1/capabilities", timeout=META_TIMEOUT_S)
    models = caps.get("models") if isinstance(caps, dict) else None
    entries: List[Dict[str, Any]] = []
    for model_id, meta in sorted((models or {}).items()):
        meta = meta if isinstance(meta, dict) else {}
        entries.append(
            {
                "id": model_id,
                "license": meta.get("license"),
                "default_steps": meta.get("default_steps"),
                "default_size": meta.get("default_size"),
                "supports_image_to_image": bool(meta.get("supports_image_to_image")),
                "supports_edit": bool(meta.get("supports_edit")),
                "min_dimension": meta.get("min_dimension"),
                "max_dimension": meta.get("max_dimension"),
            }
        )
    return tool_result(
        {
            "success": True,
            "base_url": client.base,
            "default_model": caps.get("default_model") if isinstance(caps, dict) else None,
            "max_n": caps.get("max_n") if isinstance(caps, dict) else None,
            "count": len(entries),
            "models": entries,
        }
    )


# ---------------------------------------------------------------------------
# qds_sessions
# ---------------------------------------------------------------------------

QDS_SESSIONS_SCHEMA = {
    "name": "qds_sessions",
    "description": (
        "Manage QDS playground sessions — the durable containers that hold "
        "generations. Actions: 'list' (all sessions + whether the queue is "
        "paused), 'create' (a fresh session), 'view' (one session with its "
        "generations, statuses and image filenames), 'rename', 'delete' "
        "(removes the session and its images). A password-locked session "
        "returns an error: the user unlocks it in the QDS playground UI."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "create", "view", "rename", "delete"],
                "description": "What to do. Defaults to 'list'.",
            },
            "session_id": {
                "type": "string",
                "description": "Session id — required for view, rename and delete.",
            },
            "title": {
                "type": "string",
                "description": "New title for 'rename'. Empty resets it to the first prompt.",
            },
        },
        "required": [],
    },
}


@_handler
def _handle_qds_sessions(args: dict, **_kw: Any) -> str:
    action = _action(args, ("list", "create", "view", "rename", "delete"), "list")
    client = QdsClient()
    session_id = _text(args, "session_id")

    if action == "list":
        payload = client.json("GET", "/playground/api/sessions", timeout=META_TIMEOUT_S)
        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        return tool_result(
            {
                "success": True,
                "action": action,
                "paused": bool(payload.get("paused")) if isinstance(payload, dict) else None,
                "count": len(sessions or []),
                "sessions": sessions or [],
            }
        )

    if action == "create":
        session = client.json("POST", "/playground/api/sessions", timeout=META_TIMEOUT_S)
        return tool_result({"success": True, "action": action, "session": session})

    if not session_id:
        raise ValueError(f"session_id is required for action {action!r}.")

    if action == "view":
        detail = client.json(
            "GET", f"/playground/api/sessions/{session_id}", timeout=META_TIMEOUT_S
        )
        generations = detail.get("generations") if isinstance(detail, dict) else None
        return tool_result(
            {
                "success": True,
                "action": action,
                "session": detail.get("session") if isinstance(detail, dict) else None,
                "count": len(generations or []),
                "generations": [
                    _generation_summary(g) for g in (generations or []) if isinstance(g, dict)
                ],
            }
        )

    if action == "rename":
        title = _text(args, "title")
        session = client.json(
            "PATCH",
            f"/playground/api/sessions/{session_id}",
            json_body={"title": title},
            timeout=META_TIMEOUT_S,
        )
        return tool_result({"success": True, "action": action, "session": session})

    client.json("DELETE", f"/playground/api/sessions/{session_id}", timeout=META_TIMEOUT_S)
    return tool_result({"success": True, "action": "delete", "session_id": session_id})


# ---------------------------------------------------------------------------
# qds_generate
# ---------------------------------------------------------------------------


def _submit_generation(client: QdsClient, args: dict) -> Tuple[str, dict, bool]:
    """Submit one generation from tool arguments.

    Shared by ``qds_generate`` (fire-and-forget) and ``qds_image`` (submit and
    follow), so the two cannot drift on validation, field coercion or the
    single-source-image rule. Returns ``(session_id, record, created_session)``.
    """
    prompt = _text(args, "prompt")
    if not prompt:
        raise ValueError("prompt is required and must be a non-empty string.")

    image_path = _text(args, "image_path")
    image_url = _text(args, "image_url")
    if image_path and image_url:
        raise ValueError("Pass either image_path or image_url, not both — QDS takes one source image.")

    session_id = _session_arg(args)
    created_session = False
    if not session_id:
        session = client.json("POST", "/playground/api/sessions", timeout=META_TIMEOUT_S)
        session_id = session.get("id") if isinstance(session, dict) else None
        if not isinstance(session_id, str) or not session_id:
            raise QdsError("QDS did not return a session id for the new session.")
        created_session = True

    fields: Dict[str, str] = {"prompt": prompt, "rewrite": "true" if _flag(args, "rewrite") else "false"}
    for key in ("model", "negative_prompt", "size", "group"):
        value = _text(args, key)
        if value:
            fields[key] = value
    for key, minimum in (("n", 1), ("steps", 1), ("seed", 0)):
        number = _integer(args, key, minimum=minimum)
        if number is not None:
            fields[key] = str(number)

    files = None
    if image_path or image_url:
        try:
            data, filename = _load_source_image(image_path, image_url)
        except Exception as exc:
            raise ValueError(f"Could not load the source image: {exc}") from exc
        if not data:
            raise ValueError("The source image is empty.")
        files = {"image": (filename, data)}

    record = client.json(
        "POST",
        f"/playground/api/sessions/{session_id}/generations",
        data=fields,
        files=files,
        timeout=SUBMIT_TIMEOUT_S,
    )
    if not isinstance(record, dict):
        raise QdsError("QDS did not return a generation record.")
    return session_id, record, created_session


QDS_GENERATE_SCHEMA = {
    "name": "qds_generate",
    "description": (
        "Submit an image generation to the local QDS playground and return "
        "immediately with a generation_id (the job is queued, not finished). "
        "Follow with qds_progress to watch it and qds_wait to collect the "
        "images. Pass image_path or image_url to start from an existing image "
        "— the server decides per model whether that means an edit or "
        "image-to-image. If no session_id is given, a new session is created "
        "and its id is returned; reuse it for follow-up generations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What to render. Required."},
            "session_id": {
                "type": "string",
                "description": "Existing session to append to. Omit to create one.",
            },
            "model": {
                "type": "string",
                "description": "Installed model id (see qds_models). Omit for the server default.",
            },
            "n": {"type": "integer", "description": "Number of images (default 1)."},
            "size": {
                "type": "string",
                "description": "WIDTHxHEIGHT, e.g. '1024x1024'. Omit for the model default.",
            },
            "steps": {
                "type": "integer",
                "description": "Denoising steps. Omit for the model default.",
            },
            "seed": {
                "type": "integer",
                "description": "Fixed seed for a reproducible result. Omit for a random one.",
            },
            "negative_prompt": {"type": "string", "description": "What to avoid."},
            "group": {
                "type": "string",
                "description": "Feed entry id to join instead of starting a new one.",
            },
            "rewrite": {
                "type": "boolean",
                "description": (
                    "Let the server rewrite the prompt with its own LLM (default false). "
                    "Write your own optimized prompt instead unless the user asks for this."
                ),
            },
            "image_path": {
                "type": "string",
                "description": "Local file to use as the single source image.",
            },
            "image_url": {
                "type": "string",
                "description": "http(s) URL of the single source image (alternative to image_path).",
            },
        },
        "required": ["prompt"],
    },
}


@_handler
def _handle_qds_generate(args: dict, **_kw: Any) -> str:
    client = QdsClient()
    session_id, record, created_session = _submit_generation(client, args)

    summary = _generation_summary(record)
    summary.update(
        {
            "success": True,
            "created_session": created_session,
            "next": (
                "Call qds_wait with this generation_id (and session_id) to collect the "
                "images, or qds_progress for a live step/preview snapshot."
            ),
        }
    )
    return tool_result(summary)


# ---------------------------------------------------------------------------
# qds_progress
# ---------------------------------------------------------------------------

QDS_PROGRESS_SCHEMA = {
    "name": "qds_progress",
    "description": (
        "Live snapshot of what the QDS engine is doing right now: state "
        "(idle/loading/generating/upscaling/rewriting), model, current step of "
        "total, seed and elapsed seconds — plus the latest partially denoised "
        "preview frame saved locally as MEDIA:<path> when one exists. The frame "
        "carries the same progressive blur the playground applies (heavy early, "
        "easing away as the steps complete), reported as preview_blur_px. "
        "Engine-wide, not per generation: use qds_wait to follow a specific job."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "include_preview": {
                "type": "boolean",
                "description": "Save and return the partial preview frame (default true).",
            }
        },
        "required": [],
    },
}


@_handler
def _handle_qds_progress(args: dict, **_kw: Any) -> str:
    client = QdsClient()
    frame = client.progress_snapshot()
    payload: Dict[str, Any] = {"success": True, "base_url": client.base}
    if frame is None:
        payload["note"] = "QDS produced no progress frame in time; state unknown."
    else:
        for key in ("state", "model", "kind", "seed", "step", "total", "preview_seq", "elapsed_s"):
            payload[key] = frame.get(key)

    if _flag(args, "include_preview", default=True):
        data = client.preview()
        if data:
            # Raw, as the server renders it. The blur the playground applies is
            # a CSS effect for a human watching a composition emerge from noise;
            # a model reading the frame is better served by the actual pixels.
            path = _save_bytes(data, prefix="qds_preview", extension="jpg")
            payload["preview"] = f"MEDIA:{path}"
            payload["preview_path"] = str(path)
            payload["preview_bytes"] = len(data)
        else:
            payload["preview"] = None
            payload["preview_note"] = (
                "no preview frame yet (the engine is idle, still loading weights, "
                "or between decodes — a frame is rendered every few steps)"
            )
    return tool_result(payload)


# ---------------------------------------------------------------------------
# qds_wait
# ---------------------------------------------------------------------------

QDS_WAIT_SCHEMA = {
    "name": "qds_wait",
    "description": (
        "Poll a submitted QDS generation for up to max_wait_s and collect its "
        "result. When it finishes, every image is downloaded to a local path "
        "(also returned as MEDIA:<path>). When it fails or was cancelled, the "
        "server's error is returned. When it is still queued or running, the "
        "latest snapshot is returned and you should call this tool again — it "
        "is bounded on purpose and never blocks for a whole generation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "generation_id": {
                "type": "string",
                "description": "Id returned by qds_generate or qds_upscale. Required.",
            },
            "session_id": {
                "type": "string",
                "description": "Owning session id. Pass it to avoid scanning every session.",
            },
            "max_wait_s": {
                "type": "integer",
                "description": f"Seconds to poll before returning (default 60, max {_MAX_WAIT_CAP_S}).",
            },
        },
        "required": ["generation_id"],
    },
}


@_handler
def _handle_qds_wait(args: dict, **_kw: Any) -> str:
    generation_id = _text(args, "generation_id")
    if not generation_id:
        raise ValueError("generation_id is required.")
    session_id = _session_arg(args)
    max_wait_s = _integer(args, "max_wait_s", minimum=0)
    if max_wait_s is None:
        max_wait_s = 60
    max_wait_s = min(max_wait_s, _MAX_WAIT_CAP_S)

    client = QdsClient()
    started = time.monotonic()
    deadline = started + max_wait_s
    session_id, record = _find_generation(client, generation_id, session_id)
    while True:
        status = str(record.get("status") or "")
        if status not in _ACTIVE_STATUSES:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(_WAIT_POLL_INTERVAL_S)
        _, record = _find_generation(client, generation_id, session_id)

    summary = _generation_summary(record)
    status = str(record.get("status") or "")
    summary["success"] = True
    summary["waited_s"] = round(time.monotonic() - started, 1)

    if status == _SUCCESS_STATUS:
        saved = _download_generation_images(client, record)
        summary["images"] = saved
        summary["media"] = [f"MEDIA:{item['path']}" for item in saved]
        if not saved:
            summary["success"] = False
            summary["note"] = "QDS reported the generation completed but returned no image."
        return tool_result(summary)

    if status not in _ACTIVE_STATUSES:
        # failed / cancelled — the record carries the server's own reason.
        summary["success"] = False
        return tool_result(summary)

    summary["still_running"] = True
    summary["next"] = (
        "Not finished yet. Call qds_wait again with the same generation_id "
        "(and session_id), or qds_progress for the current step."
    )
    return tool_result(summary)


# ---------------------------------------------------------------------------
# qds_cancel
# ---------------------------------------------------------------------------

QDS_CANCEL_SCHEMA = {
    "name": "qds_cancel",
    "description": (
        "Cancel a queued or running QDS generation by id. A running job stops "
        "at the next denoising step, so the returned record may still show "
        "'running' for a moment; the final status is 'cancelled'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "generation_id": {
                "type": "string",
                "description": "Id returned by qds_generate or qds_upscale. Required.",
            }
        },
        "required": ["generation_id"],
    },
}


@_handler
def _handle_qds_cancel(args: dict, **_kw: Any) -> str:
    generation_id = _text(args, "generation_id")
    if not generation_id:
        raise ValueError("generation_id is required.")
    client = QdsClient()
    record = client.json(
        "POST",
        f"/playground/api/generations/{generation_id}/cancel",
        timeout=META_TIMEOUT_S,
    )
    if not isinstance(record, dict):
        raise QdsError("QDS did not return a generation record for the cancellation.")
    summary = _generation_summary(record)
    summary["success"] = True
    return tool_result(summary)


# ---------------------------------------------------------------------------
# qds_queue
# ---------------------------------------------------------------------------

QDS_QUEUE_SCHEMA = {
    "name": "qds_queue",
    "description": (
        "Hold or release the QDS playground queue, for every session at once. "
        "Actions: 'status' (is it paused?), 'pause' (queued generations stay "
        "queued; a running one finishes), 'resume'. Use this when the user "
        "needs the machine's GPU for something else."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "pause", "resume"],
                "description": "What to do. Defaults to 'status'.",
            }
        },
        "required": [],
    },
}


@_handler
def _handle_qds_queue(args: dict, **_kw: Any) -> str:
    action = _action(args, ("status", "pause", "resume"), "status")
    client = QdsClient()
    if action == "status":
        payload = client.json("GET", "/playground/api/sessions", timeout=META_TIMEOUT_S)
        paused = bool(payload.get("paused")) if isinstance(payload, dict) else None
        return tool_result({"success": True, "action": action, "paused": paused})

    payload = client.json(
        "POST",
        "/playground/api/queue",
        json_body={"paused": action == "pause"},
        timeout=META_TIMEOUT_S,
    )
    paused = bool(payload.get("paused")) if isinstance(payload, dict) else None
    return tool_result({"success": True, "action": action, "paused": paused})


# ---------------------------------------------------------------------------
# qds_upscale
# ---------------------------------------------------------------------------

QDS_UPSCALE_SCHEMA = {
    "name": "qds_upscale",
    "description": (
        "Enlarge an image the QDS playground already owns. Call it without "
        "arguments to read the upscaler catalogue (ids, scales, whether the "
        "weights are downloaded, size and license). Then pass image (the "
        "filename from a generation's images[]), model and scale to submit — it "
        "returns immediately with a generation_id of kind 'upscale'; collect it "
        "with qds_wait. An upscaler whose weights are not downloaded triggers a "
        "download on the server first, which can take a while."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": (
                    "Filename of a generated image (from qds_wait/qds_sessions view). "
                    "Omit to only read the catalogue."
                ),
            },
            "model": {"type": "string", "description": "Upscaler id from the catalogue."},
            "scale": {"type": "integer", "description": "Scale factor offered by that upscaler."},
            "session_id": {
                "type": "string",
                "description": (
                    "Session that owns the image. Resolved automatically when omitted."
                ),
            },
            "group": {
                "type": "string",
                "description": "Feed entry to join. Defaults to the source image's entry.",
            },
        },
        "required": [],
    },
}


def _catalogue(client: QdsClient) -> List[dict]:
    payload = client.json("GET", "/playground/api/upscalers", timeout=META_TIMEOUT_S)
    upscalers = payload.get("upscalers") if isinstance(payload, dict) else None
    return [u for u in upscalers if isinstance(u, dict)] if isinstance(upscalers, list) else []


def _session_owning_image(client: QdsClient, filename: str) -> str:
    """Find the session whose generations produced ``filename``.

    The upscale route only accepts an image the session owns, so guessing a
    session id would produce a 400 instead of an upscale.
    """
    listing = client.json("GET", "/playground/api/sessions", timeout=META_TIMEOUT_S)
    sessions = listing.get("sessions") if isinstance(listing, dict) else None
    for entry in (sessions or [])[:_MAX_SESSION_SCAN]:
        session_id = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(session_id, str):
            continue
        try:
            records = _session_generations(client, session_id)
        except QdsHttpError as exc:
            if exc.status_code == 403:
                continue
            raise
        for record in records:
            for image in record.get("images") or []:
                if isinstance(image, dict) and _image_filename(image.get("url")) == filename:
                    return session_id
    raise ValueError(
        f"No visible session owns the image {filename!r}. Pass session_id explicitly "
        "(a locked session is not scanned), or check the filename."
    )


@_handler
def _handle_qds_upscale(args: dict, **_kw: Any) -> str:
    client = QdsClient()
    image = _text(args, "image")
    if not image:
        return tool_result(
            {
                "success": True,
                "action": "catalogue",
                "upscalers": _catalogue(client),
                "next": "Call again with image, model and scale to submit an upscale.",
            }
        )

    model = _text(args, "model")
    scale = _integer(args, "scale", minimum=1)
    if not model or scale is None:
        available = [
            {"id": u.get("id"), "scales": u.get("scales"), "downloaded": u.get("downloaded")}
            for u in _catalogue(client)
        ]
        raise ValueError(
            f"model and scale are required to upscale {image!r}. Available: {available}"
        )

    session_id = _session_arg(args) or _session_owning_image(client, image)
    body: Dict[str, Any] = {"image": image, "model": model, "scale": scale}
    group = _text(args, "group")
    if group:
        body["group"] = group

    record = client.json(
        "POST",
        f"/playground/api/sessions/{session_id}/upscales",
        json_body=body,
        timeout=SUBMIT_TIMEOUT_S,
    )
    if not isinstance(record, dict):
        raise QdsError("QDS did not return a generation record for the upscale.")
    summary = _generation_summary(record)
    summary.update(
        {
            "success": True,
            "source_image": image,
            "upscaler": model,
            "scale": scale,
            "next": "Call qds_wait with this generation_id to collect the enlarged image.",
        }
    )
    return tool_result(summary)


# ---------------------------------------------------------------------------
# qds_playground
# ---------------------------------------------------------------------------

QDS_PLAYGROUND_SCHEMA = {
    "name": "qds_playground",
    "description": (
        "Open the QDS playground in the preview pane and attach this chat to "
        "it — the way to generate images when the user wants to WATCH them "
        "render. Creates a session, optionally starts a first generation, and "
        "returns pane_url. ALWAYS follow this call by opening pane_url with "
        "open_preview: the pane is the point, it shows the image forming step "
        "by step, live. After this, every other qds_* tool defaults to this "
        "session, so the user can steer in plain language ('make it wider', "
        "'upscale that one') and it lands in the pane they are looking at. For "
        "a quick one-off image with no pane, use image_generate instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Optional first generation to start immediately. Write a "
                    "full, optimized image prompt. Omit to just open an empty "
                    "session."
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Attach to an existing session instead of creating one "
                    "(from qds_sessions). Omit for a fresh session."
                ),
            },
            "title": {
                "type": "string",
                "description": "Name for the new session. Defaults to the prompt.",
            },
            "model": {"type": "string", "description": "Model id (see qds_models)."},
            "size": {"type": "string", "description": "WxH, e.g. 1024x1024."},
            "steps": {"type": "integer", "description": "Denoising steps."},
            "seed": {"type": "integer", "description": "Seed for a reproducible result."},
            "n": {"type": "integer", "description": "How many images (default 1)."},
            "negative_prompt": {"type": "string", "description": "What to avoid."},
            "image_path": {"type": "string", "description": "Local source image for edit / image-to-image."},
            "image_url": {"type": "string", "description": "Source image URL for edit / image-to-image."},
        },
        "required": [],
    },
}


@_handler
def _handle_qds_playground(args: dict, **_kw: Any) -> str:
    client = QdsClient()
    prompt = _text(args, "prompt")
    session_id = _text(args, "session_id")
    created_session = False

    if session_id:
        # Prove it exists before attaching: a bad id would otherwise surface as
        # a confusing failure on some later tool call instead of right here.
        _session_generations(client, session_id)
    else:
        body: Dict[str, Any] = {}
        title = _text(args, "title") or (prompt[:60] if prompt else None)
        if title:
            body["title"] = title
        session = client.json(
            "POST",
            "/playground/api/sessions",
            json_body=body or None,
            timeout=META_TIMEOUT_S,
        )
        candidate = session.get("id") if isinstance(session, dict) else None
        if not isinstance(candidate, str) or not candidate:
            raise QdsError("QDS did not return a session id for the new session.")
        session_id = candidate
        created_session = True

    payload: Dict[str, Any] = {
        "success": True,
        "session_id": session_id,
        "created_session": created_session,
        "pane_url": _pane_url(session_id),
        "attached": True,
    }

    if prompt:
        submit_args = dict(args)
        submit_args["session_id"] = session_id
        _, record, _ = _submit_generation(client, submit_args)
        payload["generation"] = _generation_summary(record)

    # Attach only once the work above succeeded: a failed open must not leave
    # the chat pointing at a session the user cannot see.
    _attach_session(session_id)

    payload["next"] = (
        f"Open {payload['pane_url']} with open_preview now — that pane is where "
        "the user watches the image render."
        + (
            " Then call qds_wait to collect the finished image."
            if prompt
            else " Then call qds_generate (no session_id needed) to start one."
        )
    )
    return tool_result(payload)

