"""QDS (Quantum Diffusion Server) image generation backend.

Routes Hermes' ``image_generate`` tool to a **local** QDS instance — an
OpenAI-images-compatible FastAPI server driving mflux on Apple Silicon.
Nothing leaves the machine: no API key, no cloud, no egress.

    text-to-image           POST /v1/images/generations   (JSON)
    image-to-image / edit   POST /v1/images/edits         (multipart)

The model catalog is **dynamic**: it is whatever the server reports at
``GET /v1/capabilities``, because the set of installed weights is a
property of the machine, not of this file. Server down → empty catalog
and clean ``error_response``s, never an exception and never a fake image.

Output is always requested as ``b64_json`` and materialised under
``$HERMES_HOME/cache/images/``. QDS' ``url`` response format is backed by
a store with a TTL (default 3600s); a cached local file has no expiry.

Selection precedence for the model (first hit wins):

1. ``QDS_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.qds.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when the server has that model)
4. the server's own ``default_model``

Server location: ``QDS_BASE_URL`` env var, default
``http://127.0.0.1:8765``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "qds"
DEFAULT_BASE_URL = "http://127.0.0.1:8765"

#: Health probe must not stall `hermes tools`; the server is local.
HEALTH_TIMEOUT_S = 2.0
#: Metadata endpoints are cheap and local.
META_TIMEOUT_S = 5.0
#: Generation on a cold model can mean loading tens of GB of weights from
#: disk before the first denoising step. Matches the server's own generous
#: request budget rather than imposing a client-side deadline on it.
GENERATE_TIMEOUT_S = 1800.0
#: Capabilities rarely change while the CLI is open; one refresh per half
#: minute keeps `hermes tools` from hammering the server.
_CAPS_TTL_S = 30.0

#: aspect_ratio → pixel size. Inside the 512–1536 window every installed
#: model accepts; mflux truncates to a multiple of 16 anyway.
_SIZES = {
    "landscape": "1344x768",
    "square": "1024x1024",
    "portrait": "768x1344",
}


def base_url() -> str:
    """Return the QDS base URL (env-overridable), without trailing slash."""
    raw = (os.environ.get("QDS_BASE_URL") or "").strip() or DEFAULT_BASE_URL
    return raw.rstrip("/")


# ---------------------------------------------------------------------------
# Server metadata
# ---------------------------------------------------------------------------


_caps_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _fetch_capabilities(*, use_cache: bool = True) -> Dict[str, Any]:
    """GET /v1/capabilities. Returns ``{}`` when the server is unreachable."""
    url = base_url()
    if use_cache:
        hit = _caps_cache.get(url)
        if hit is not None and (time.monotonic() - hit[0]) < _CAPS_TTL_S:
            return hit[1]

    try:
        import requests

        resp = requests.get(f"{url}/v1/capabilities", timeout=META_TIMEOUT_S)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.debug("QDS capabilities unavailable at %s: %s", url, exc)
        return {}

    if not isinstance(payload, dict):
        return {}
    _caps_cache[url] = (time.monotonic(), payload)
    return payload


def _speed_hint(steps: Any) -> str:
    """Turn a model's default step count into a coarse speed label."""
    if not isinstance(steps, int) or steps <= 0:
        return ""
    if steps <= 8:
        return f"{steps} steps — fastest"
    if steps <= 15:
        return f"{steps} steps — fast"
    if steps <= 30:
        return f"{steps} steps — balanced"
    return f"{steps} steps — slow"


def _strengths(meta: Dict[str, Any]) -> str:
    """One-line summary of what a model can do, from server metadata."""
    bits: List[str] = []
    if meta.get("supports_edit"):
        bits.append("edit")
    elif meta.get("supports_image_to_image"):
        bits.append("image-to-image")
    if meta.get("supports_negative_prompt"):
        bits.append("negative prompt")
    if meta.get("prompt_formats") == ["json"]:
        bits.append("JSON captions only")
    license_name = meta.get("license")
    if isinstance(license_name, str) and license_name:
        bits.append(license_name)
    if meta.get("gated"):
        bits.append("gated weights")
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model(caps: Dict[str, Any], requested: Optional[str] = None) -> Optional[str]:
    """Decide which QDS model to ask for; ``None`` when nothing is known.

    ``caps`` is a (possibly empty) ``/v1/capabilities`` payload. ``requested``
    is the ``model`` kwarg the tool layer forwards from ``image_gen.model``.
    Catalog-checked candidates are only honoured when the server actually has
    that model — ``image_gen.model`` is shared with every other backend and
    routinely holds an id from a different provider (``gpt-image-2-medium``).
    ``QDS_IMAGE_MODEL`` is an explicit escape hatch and is passed through
    as-is so the server, not this file, gets to reject it with its own error.
    """
    env_override = (os.environ.get("QDS_IMAGE_MODEL") or "").strip()
    if env_override:
        return env_override

    installed = caps.get("models") if isinstance(caps.get("models"), dict) else {}

    if isinstance(requested, str) and requested.strip() in installed:
        return requested.strip()

    cfg = _load_image_gen_config()
    qds_cfg = cfg.get(PROVIDER_NAME)
    if isinstance(qds_cfg, dict):
        value = qds_cfg.get("model")
        if isinstance(value, str) and value.strip():
            # Provider-scoped key: the user meant this backend, honour it even
            # when the catalog lookup failed (server momentarily down).
            if not installed or value.strip() in installed:
                return value.strip()

    top = cfg.get("model")
    if isinstance(top, str) and top.strip() in installed:
        return top.strip()

    default = caps.get("default_model")
    if isinstance(default, str) and default:
        return default
    return None


# ---------------------------------------------------------------------------
# Source-image loading (for image-to-image / edit)
# ---------------------------------------------------------------------------


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load image bytes from an http(s) URL, ``data:`` URI or local path.

    Returns ``(data, filename)``. Raises on any network / IO error so the
    caller can surface a clean error_response.
    """
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        import requests

        resp = requests.get(ref, timeout=60)
        resp.raise_for_status()
        name = ref.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
        return resp.content, name
    if lower.startswith("data:"):
        import base64

        header, _, b64 = ref.partition(",")
        ext = "png"
        if "image/" in header:
            ext = header.split("image/", 1)[1].split(";", 1)[0] or "png"
        return base64.b64decode(b64), f"image.{ext}"
    # Local file path — enforce the shared credential-read guard before reading.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read()
    name = os.path.basename(ref) or "image.png"
    return data, name


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _server_error_message(resp: Any) -> str:
    """Extract QDS' ``{"error": ...}`` message, falling back to the body."""
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            message = err.get("message")
            if isinstance(message, str) and message:
                return message
        elif isinstance(err, str) and err:
            return err
    text = (getattr(resp, "text", "") or "").strip()
    return text[:400] or f"HTTP {getattr(resp, 'status_code', '?')}"


def _error_type_for_status(status: int) -> str:
    if status in (401, 403):
        return "auth_required"
    if status in (400, 404, 413, 415, 422):
        return "invalid_argument"
    return "api_error"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class QdsImageGenProvider(ImageGenProvider):
    """Local Quantum Diffusion Server backend (mflux on Apple Silicon)."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "QDS (local)"

    def is_available(self) -> bool:
        try:
            import requests

            resp = requests.get(f"{base_url()}/health", timeout=HEALTH_TIMEOUT_S)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.debug("QDS health probe failed: %s", exc)
            return False
        return isinstance(payload, dict) and payload.get("status") == "ok"

    def list_models(self) -> List[Dict[str, Any]]:
        caps = _fetch_capabilities()
        models = caps.get("models")
        if not isinstance(models, dict):
            return []
        entries: List[Dict[str, Any]] = []
        for model_id, meta in models.items():
            if not isinstance(model_id, str):
                continue
            meta = meta if isinstance(meta, dict) else {}
            entries.append(
                {
                    "id": model_id,
                    "display": model_id,
                    "speed": _speed_hint(meta.get("default_steps")),
                    "strengths": _strengths(meta),
                    "price": "free (local)",
                }
            )
        entries.sort(key=lambda e: e["id"])
        return entries

    def default_model(self) -> Optional[str]:
        return _resolve_model(_fetch_capabilities())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": (
                "Quantum Diffusion Server on this machine (mflux / Apple "
                f"Silicon) — no API key. Set QDS_BASE_URL to move off "
                f"{DEFAULT_BASE_URL}."
            ),
            "env_vars": [],
        }

    def capabilities(self) -> Dict[str, Any]:
        # /v1/images/edits accepts exactly one source image, so references
        # beyond the primary one cannot be honoured.
        modalities = ["text"]
        caps = _fetch_capabilities()
        models = caps.get("models")
        if isinstance(models, dict) and models:
            if any(
                isinstance(m, dict)
                and (m.get("supports_image_to_image") or m.get("supports_edit"))
                for m in models.values()
            ):
                modalities.append("image")
        else:
            # Server unreachable: report the backend's own surface rather
            # than silently downgrading a capability the tool layer caches.
            modalities.append("image")
        return {"modalities": modalities, "max_reference_images": 1}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        size = _SIZES.get(aspect, _SIZES["square"])

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=PROVIDER_NAME,
                aspect_ratio=aspect,
            )

        try:
            import requests
        except ImportError:  # pragma: no cover - requests ships with Hermes
            return error_response(
                error="requests package not installed (pip install requests)",
                error_type="missing_dependency",
                provider=PROVIDER_NAME,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        caps = _fetch_capabilities()
        # The tool layer forwards the picker's ``image_gen.model`` here.
        model = _resolve_model(caps, kwargs.get("model"))
        if not model:
            return error_response(
                error=(
                    f"QDS is not reachable at {base_url()} and no model was "
                    "pinned (set QDS_IMAGE_MODEL or image_gen.qds.model). "
                    "Start the server, or point QDS_BASE_URL at it."
                ),
                error_type="api_error",
                provider=PROVIDER_NAME,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Only the primary source image can be honoured — /v1/images/edits
        # takes a single file. Extra references are dropped, loudly in logs.
        source: Optional[str] = None
        if isinstance(image_url, str) and image_url.strip():
            source = image_url.strip()
        if reference_image_urls:
            from agent.image_gen_provider import normalize_reference_images

            refs = normalize_reference_images(reference_image_urls) or []
            if source is None and refs:
                source = refs[0]
                refs = refs[1:]
            if refs:
                logger.info(
                    "QDS accepts one source image; ignoring %d extra reference(s).",
                    len(refs),
                )

        modality = "image" if source else "text"

        try:
            if source is not None:
                try:
                    data, filename = _load_image_bytes(source)
                except Exception as exc:
                    return error_response(
                        error=f"Could not load source image for editing: {exc}",
                        error_type="io_error",
                        provider=PROVIDER_NAME,
                        model=model,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                # No `strength`: which of edit / img2img a reference means is
                # a property of the model, and the server already decides it.
                resp = requests.post(
                    f"{base_url()}/v1/images/edits",
                    data={
                        "prompt": prompt,
                        "model": model,
                        "n": "1",
                        "size": size,
                        "response_format": "b64_json",
                    },
                    files={"image": (filename, data)},
                    timeout=GENERATE_TIMEOUT_S,
                )
            else:
                resp = requests.post(
                    f"{base_url()}/v1/images/generations",
                    json={
                        "prompt": prompt,
                        "model": model,
                        "n": 1,
                        "size": size,
                        "response_format": "b64_json",
                    },
                    timeout=GENERATE_TIMEOUT_S,
                )
        except Exception as exc:
            logger.debug("QDS request failed", exc_info=True)
            return error_response(
                error=f"QDS request to {base_url()} failed: {exc}",
                error_type="api_error",
                provider=PROVIDER_NAME,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if resp.status_code >= 400:
            return error_response(
                error=f"QDS returned HTTP {resp.status_code}: {_server_error_message(resp)}",
                error_type=_error_type_for_status(resp.status_code),
                provider=PROVIDER_NAME,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            payload = resp.json()
        except Exception as exc:
            return error_response(
                error=f"QDS returned a non-JSON response: {exc}",
                error_type="api_error",
                provider=PROVIDER_NAME,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        entries = payload.get("data") if isinstance(payload, dict) else None
        first = entries[0] if isinstance(entries, list) and entries else None
        b64 = first.get("b64_json") if isinstance(first, dict) else None
        if not isinstance(b64, str) or not b64:
            return error_response(
                error="QDS returned no image data",
                error_type="empty_response",
                provider=PROVIDER_NAME,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(b64, prefix=f"qds_{model}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider=PROVIDER_NAME,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # `size` is what we asked for; mflux truncates to a multiple of 16 and
        # reports the size actually rendered, plus the seeds it used.
        extra: Dict[str, Any] = {"size": size, "base_url": base_url()}
        mflux = payload.get("mflux") if isinstance(payload, dict) else None
        if isinstance(mflux, dict):
            for key in ("size", "steps", "seeds"):
                if key in mflux:
                    extra[f"mflux_{key}"] = mflux[key]

        return success_response(
            image=str(saved_path),
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=PROVIDER_NAME,
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``QdsImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(QdsImageGenProvider())
