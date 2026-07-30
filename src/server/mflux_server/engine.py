"""In-process inference engine, keeping one model warm in memory.

This is the heart of the server. The prototype spawned an `mflux-*` binary per
image, reloading the weights on every request. Here the model stays in memory
between calls, which takes a generation from several minutes down to a few
seconds once the model is loaded.

Three invariants:

* **one live model at a time** — on unified memory, keeping two (a 9B plus
  anything else) saturates the machine;
* **one generation at a time** — an `asyncio.Lock` serializes everything, and
  inference runs on a single worker thread, never on the event loop;
* **one registered callback per model** — mflux's `CallbackRegistry` has no
  `unregister`, so registering per request would grow the list without bound.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mflux_server.errors import APIError, GenerationTimeout, translate_mflux_exception
from mflux_server.logs import SERVER_LOGGER, capture_stdout
from mflux_server.registry import ModelSpec, load_model

logger = logging.getLogger(SERVER_LOGGER)

#: Submodules to release on unload. Mirrors what mflux's `MemorySaver` does
#: internally (callbacks/instances/memory_saver.py:76-107), for lack of a public
#: teardown method in the library.
_UNLOADABLE_ATTRS = (
    "transformer",
    "text_encoder",
    "clip_text_encoder",
    "t5_text_encoder",
    "vae",
    "image_encoder",
    "qwen_vl_encoder",
)

#: Past this point we purge Qwen's embedding cache: it is keyed by prompt and
#: has no bound at all (qwen_prompt_encoder.py:20-41).
_PROMPT_CACHE_MAX_ENTRIES = 16


@dataclass
class ProgressSnapshot:
    """The engine's current state, readable without a lock.

    Written from the worker thread, read from the event loop: these are plain
    attribute assignments, hence atomic under the GIL. A single snapshot is
    enough because `ModelEngine._lock` serializes generations — there are never
    two jobs in flight. That is what lets `/v1/progress` poll instead of
    building a cross-thread queue, and lets several SSE consumers coexist for
    free.
    """

    state: str = "idle"  # idle | loading | generating
    model: str | None = None
    kind: str | None = None
    seed: int | None = None
    step: int = 0
    total: int = 0
    started_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        elapsed = None if self.started_at is None else round(time.monotonic() - self.started_at, 1)
        return {
            "state": self.state,
            "model": self.model,
            "kind": self.kind,
            "seed": self.seed,
            "step": self.step,
            "total": self.total,
            "elapsed_s": elapsed,
        }

    def reset(self) -> None:
        self.state = "idle"
        self.model = None
        self.kind = None
        self.seed = None
        self.step = 0
        self.total = 0
        self.started_at = None


@dataclass
class GenerationJob:
    spec: ModelSpec
    kind: str  # "txt2img" | "edit"
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    guidance: float | None = None
    negative_prompt: str | None = None
    image_path: Path | None = None
    image_strength: float | None = None
    #: True when `steps` comes from the model's sampler preset rather than from
    #: the request. Only then may the step count be left out of the call, which is
    #: what lets the preset's guidance schedule apply.
    steps_from_preset: bool = False


class _ProgressCallback:
    """Persistent callback: logs progress, arms the timeout and the cancellation.

    Interrupting a generation is only possible from here: an in-flight MLX
    operation cannot be cancelled from outside, and `asyncio` cannot kill the
    worker thread. Raising from `call_in_loop` does propagate — mflux only
    catches `KeyboardInterrupt` in its loop. So a cancellation requested through
    `/v1/cancel` takes exactly the path the timeout already takes.
    """

    def __init__(self, log_every: int = 1, progress: ProgressSnapshot | None = None):
        self._log_every = log_every
        self._progress = progress or ProgressSnapshot()
        self.deadline: float | None = None
        self.label: str = ""
        self.timed_out: bool = False
        self.cancel_requested: bool = False
        self.cancelled: bool = False

    def arm(self, label: str, deadline: float | None) -> None:
        self.label = label
        self.deadline = deadline
        self.timed_out = False
        self.cancel_requested = False
        self.cancelled = False

    def call_in_loop(self, *, t: int, config: Any, time_steps: Any, **_: Any) -> None:
        from mflux.utils.exceptions import StopImageGenerationException

        step = t + 1
        total = config.num_inference_steps

        if self.cancel_requested:
            self.cancelled = True
            raise StopImageGenerationException(f"Generation cancelled at step {step}/{total}")

        if self.deadline is not None and time.monotonic() > self.deadline:
            self.timed_out = True
            raise StopImageGenerationException(f"Timed out at step {step}/{total}")

        self._progress.step = step
        self._progress.total = total
        if self._log_every and (step % self._log_every == 0 or step == total):
            logger.info(
                "%s — step %d/%d",
                self.label,
                step,
                total,
                extra={"event": "generation_step", "fields": {"step": step, "total": total}},
            )


class ModelEngine:
    def __init__(self, *, request_timeout_s: float = 900.0, progress_log_every: int = 1):
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mflux-inference")
        self._request_timeout_s = request_timeout_s
        self._progress_log_every = progress_log_every
        self._snapshot = ProgressSnapshot()
        self._callback = _ProgressCallback(progress_log_every, self._snapshot)
        self._model: Any | None = None
        self._loaded: tuple[str, str] | None = None  # (model key, kind)

    # ── State ──────────────────────────────────────────────────────────────

    @property
    def loaded_model(self) -> str | None:
        return None if self._loaded is None else f"{self._loaded[0]}:{self._loaded[1]}"

    def progress(self) -> dict[str, Any]:
        """Progress snapshot. Lock-free, callable from the event loop.

        Taking `self._lock` here would deadlock by construction: it is held for
        the whole generation, exactly when we want to observe it.
        """
        return {
            **self._snapshot.as_dict(),
            "loaded_model": self.loaded_model,
            "memory": self.memory_stats(),
        }

    def request_cancel(self) -> bool:
        """Request that the running generation stop. `False` if nothing is running.

        The flag is read by `_ProgressCallback.call_in_loop`, so the stop takes
        effect at the next denoising step — not instantly.
        """
        if self._snapshot.state != "generating":
            return False
        self._callback.cancel_requested = True
        logger.info(
            "Cancellation requested for %s",
            self._snapshot.model,
            extra={"event": "generation_cancel_requested", "fields": {"model": self._snapshot.model}},
        )
        return True

    def memory_stats(self) -> dict[str, float]:
        try:
            import mlx.core as mx
        except ImportError:  # pragma: no cover - mlx is a hard dependency
            return {}
        return {
            "active_gb": round(mx.get_active_memory() / 1e9, 2),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "cache_gb": round(mx.get_cache_memory() / 1e9, 2),
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def generate(self, job: GenerationJob) -> bytes:
        """Generate one PNG image. Serialized: one call at a time."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(self._executor, self._generate_sync, job)
            except GenerationTimeout as exc:
                raise exc
            except Exception as exc:
                raise translate_mflux_exception(exc) from exc

    async def unload(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._unload_sync)

    def shutdown(self) -> None:
        """Stop the engine, bounding the wait to a single denoising step.

        Two traps, in this precise order.

        First, request cancellation **before** joining the worker: otherwise
        `wait=True` waits for the running generation to finish, i.e. up to
        `request_timeout_s` (2400s in the shipped config). uvicorn's
        `timeout_graceful_shutdown` does not cover this: it only bounds the wait
        on HTTP connections, and the shutdown lifespan — hence this call — runs
        afterwards.

        Second, only release the submodules **after** the worker has stopped:
        `_unload_sync` runs on the calling thread, and setting
        `transformer = None` while a generation is using it is a race.
        """
        self._callback.cancel_requested = True
        self._executor.shutdown(wait=True)
        self._unload_sync()
        self._snapshot.reset()

    # ── Implementation (worker thread) ─────────────────────────────────────

    def _generate_sync(self, job: GenerationJob) -> bytes:
        try:
            model = self._ensure_model(job.spec, job.kind)
            label = f"{job.spec.key} seed={job.seed} {job.width}x{job.height}"
            deadline = time.monotonic() + self._request_timeout_s if self._request_timeout_s else None
            self._callback.arm(label, deadline)

            self._snapshot.state = "generating"
            self._snapshot.model = job.spec.key
            self._snapshot.kind = job.kind
            self._snapshot.seed = job.seed
            self._snapshot.step = 0
            self._snapshot.total = job.steps
            self._snapshot.started_at = time.monotonic()

            logger.info(
                "▶ %s — %d steps",
                label,
                job.steps,
                extra={
                    "event": "generation_start",
                    "fields": {
                        "model": job.spec.key,
                        "kind": job.kind,
                        "seed": job.seed,
                        "steps": job.steps,
                        "width": job.width,
                        "height": job.height,
                    },
                },
            )
            started = time.monotonic()
            with capture_stdout():
                generated = model.generate_image(**self._generate_kwargs(job))
            elapsed = time.monotonic() - started
            logger.info(
                "✓ %s — %.1f s",
                label,
                elapsed,
                extra={
                    "event": "generation_done",
                    "fields": {"model": job.spec.key, "seed": job.seed, "elapsed_s": round(elapsed, 1)},
                },
            )

            self._trim_prompt_cache(model)
            return _to_png_bytes(generated)
        except Exception as exc:
            if self._callback.timed_out:
                raise GenerationTimeout(self._request_timeout_s) from exc
            raise translate_mflux_exception(exc) from exc
        finally:
            # Whatever became of the job — success, cancellation, timeout,
            # crash — the engine is available again and SSE consumers must see
            # it. The loaded model, however, stays warm.
            self._snapshot.reset()

    def _generate_kwargs(self, job: GenerationJob) -> dict[str, Any]:
        spec = job.spec
        kwargs: dict[str, Any] = {
            "seed": job.seed,
            "prompt": job.prompt,
            "num_inference_steps": job.steps,
            "width": job.width,
            "height": job.height,
            "scheduler": spec.scheduler,
        }

        guidance = job.guidance if job.guidance is not None else spec.default_guidance
        if guidance is not None:
            kwargs["guidance"] = guidance

        if spec.supports_negative_prompt and job.negative_prompt is not None:
            kwargs["negative_prompt"] = job.negative_prompt

        if job.kind == "edit":
            # Instruction editing: the reference images are conditioning
            # tokens, not a noised starting latent.
            kwargs["image_paths"] = [str(job.image_path)] if job.image_path else []
        elif job.image_path is not None:
            kwargs["image_path"] = str(job.image_path)
            kwargs["image_strength"] = job.image_strength

        if spec.preset is not None:
            # Ideogram 4. Two departures from every other family, both of which
            # bite silently if ignored.
            #
            # Its `generate_image` has no `scheduler` parameter at all, so passing
            # ours would be a TypeError rather than an ignored argument.
            kwargs.pop("scheduler", None)
            # And `preset` goes in *always*, even when the step count is explicit:
            # the sampler also supplies `mu` and `std` to the noise schedule
            # (`make_timesteps`), not just the step count.
            kwargs["preset"] = spec.preset
            if job.steps_from_preset:
                # Passing `num_inference_steps` replaces the preset's per-step
                # guidance schedule with a constant (`ideogram4.py:67-72`), so we
                # leave it out unless the client asked for a specific count.
                kwargs.pop("num_inference_steps", None)

        return kwargs

    def _ensure_model(self, spec: ModelSpec, kind: str) -> Any:
        target = (spec.key, kind)
        if self._loaded == target and self._model is not None:
            return self._model

        if self._model is not None:
            logger.info(
                "Unloading %s",
                self.loaded_model,
                extra={"event": "model_unload", "fields": {"model": self.loaded_model}},
            )
            self._unload_sync()

        # "loading" rather than "generating": on flux2-dev this takes a minute
        # or more, and the UI should be able to say so instead of showing 0/50.
        self._snapshot.state = "loading"
        self._snapshot.model = spec.key
        self._snapshot.kind = kind
        self._snapshot.started_at = time.monotonic()

        logger.info(
            "Loading %s (%s) — %s",
            spec.key,
            kind,
            spec.repo,
            extra={"event": "model_loading", "fields": {"model": spec.key, "kind": kind, "repo": spec.repo}},
        )
        started = time.monotonic()
        with capture_stdout():
            model = load_model(spec, kind=kind)
        # A single callback, registered at load time and never removed:
        # CallbackRegistry cannot unregister.
        model.callbacks.register(self._callback)

        self._model = model
        self._loaded = target
        memory = self.memory_stats()
        logger.info(
            "Model %s ready in %.1fs — memory %s",
            spec.key,
            time.monotonic() - started,
            memory,
            extra={
                "event": "model_ready",
                "fields": {
                    "model": spec.key,
                    "kind": kind,
                    "load_s": round(time.monotonic() - started, 1),
                    **memory,
                },
            },
        )
        return model

    def _unload_sync(self) -> None:
        model = self._model
        self._model = None
        self._loaded = None
        if model is None:
            return

        try:
            from mflux.callbacks.callback_registry import CallbackRegistry

            model.callbacks = CallbackRegistry()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not reset the callbacks", exc_info=True)

        for attr in _UNLOADABLE_ATTRS:
            if hasattr(model, attr):
                try:
                    setattr(model, attr, None)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("Could not release %s", attr, exc_info=True)

        prompt_cache = getattr(model, "prompt_cache", None)
        if isinstance(prompt_cache, dict):
            prompt_cache.clear()

        del model
        gc.collect()
        _clear_mlx_cache()

    def _trim_prompt_cache(self, model: Any) -> None:
        prompt_cache = getattr(model, "prompt_cache", None)
        if isinstance(prompt_cache, dict) and len(prompt_cache) > _PROMPT_CACHE_MAX_ENTRIES:
            prompt_cache.clear()
            gc.collect()
            _clear_mlx_cache()
            logger.debug("prompt_cache purged")


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:  # pragma: no cover - mlx is a hard dependency
        pass


def _to_png_bytes(generated: Any) -> bytes:
    """Extract the PNG from a `GeneratedImage` without going through disk.

    `ZImage.generate_image` is annotated `-> Image.Image` but actually returns a
    `GeneratedImage` (z_image.py:59 vs :137) — hence the `getattr`.
    """
    image = getattr(generated, "image", generated)
    if image is None:
        raise APIError(
            "mflux produced no image.", status_code=500, error_type="server_error", code="empty_result"
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
