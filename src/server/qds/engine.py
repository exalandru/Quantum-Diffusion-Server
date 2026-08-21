"""In-process inference engine, keeping one model warm in memory.

This is the heart of the server. The prototype spawned an `mflux-*` binary per
image, reloading the weights on every request. Here the model stays in memory
between calls, which takes a generation from several minutes down to a few
seconds once the model is loaded.

Four invariants:

* **one live diffusion model at a time** — on unified memory, keeping two (a 9B
  plus anything else) saturates the machine;
* **one bounded second slot, for an upscaler** — what makes this exception safe
  is not that Real-ESRGAN's weights are small (33 MB against 10-28 GB), it is
  that the cost of *running* one is bounded and transient. Two bounds, and they
  are separate:

  - the MLX allocator sees one tile's activations at a time, bounded by
    `UpscalerSpec.tile` — measured constant at 1.52 GB whatever the source
    size, which is the tiling working;
  - the host sees the assembled image, which is *not* bounded by the tile and
    scales with the output. It is held at one byte per channel (`pipeline`
    quantises each tile as it lands) and capped by `MAX_RENDER_PIXELS`,
    which is measured on what the network renders rather than on what was
    asked for. At that cap a run peaks around 1.11 GB resident.

  Evicting a warm FLUX to enlarge an image and reloading it (a minute on
  flux2-dev) for the next generation would cost more average memory and far
  more time than holding those 33 MB. The slot itself is bounded by
  construction: the catalogue holds only RRDBNets and refuses an entry over
  `MAX_WEIGHTS_MB` at import, and `_ensure_upscaler` keeps exactly one;
* **one MLX job at a time** — an `asyncio.Lock` serializes everything,
  generations and upscales alike, and both run on the same single worker
  thread, never on the event loop;
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

from PIL import Image

from qds.errors import APIError, GenerationTimeout, translate_mflux_exception
from qds.logs import SERVER_LOGGER, capture_stdout
from qds.registry import ModelSpec, latent_creator_for, load_model
from qds.upscale import UpscalerSpec

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
    # Anima's text adapter. Small, but a submodule this tuple does not name is a
    # submodule unload leaves resident, and the point of unloading is that
    # nothing stays.
    "text_conditioner",
)

#: Past this point we purge Qwen's embedding cache: it is keyed by prompt and
#: has no bound at all (qwen_prompt_encoder.py:20-41).
_PROMPT_CACHE_MAX_ENTRIES = 16


@dataclass
class ProgressSnapshot:
    """The engine's current state, readable without a lock.

    Written from the worker thread, read from the event loop: these are plain
    attribute assignments, hence atomic under the GIL. A single snapshot is
    enough because `ModelEngine._lock` serializes every MLX job — generations
    and upscales alike — so there are never two in flight. That is what lets `/v1/progress` poll instead of
    building a cross-thread queue, and lets several SSE consumers coexist for
    free.
    """

    state: str = "idle"  # idle | loading | generating | upscaling
    model: str | None = None
    kind: str | None = None
    seed: int | None = None
    step: int = 0
    total: int = 0
    #: 0 = no preview frame for the current run; otherwise the engine-lifetime
    #: count of the latest frame. Monotonic across runs, so it doubles as the
    #: cache-buster the client puts on the preview URL.
    preview_seq: int = 0
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
            "preview_seq": self.preview_seq,
            "elapsed_s": elapsed,
        }

    def reset(self) -> None:
        self.state = "idle"
        self.model = None
        self.kind = None
        self.seed = None
        self.step = 0
        self.total = 0
        self.preview_seq = 0
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
    #: Decode a preview image every N denoising steps; 0 disables previews.
    #: Set by the playground runner only — the `/v1` plane never opts in.
    preview_every: int = 0


@dataclass
class _PreviewPlan:
    """What one armed run needs in order to render previews."""

    every: int
    creator: Any
    model: Any


#: Previews are shown on the same 512px track as the feed's finished images, and
#: are JPEG because they are transient frames, not the deliverable (which stays PNG).
_PREVIEW_MAX_PX = 512
_PREVIEW_JPEG_QUALITY = 80


def _render_preview(*, model: Any, creator: Any, latents: Any, config: Any, seed: int, prompt: str) -> bytes:
    """Decode a mid-loop latent into a small JPEG.

    Mirrors mflux's own `StepwiseHandler._save_image`
    (callbacks/instances/stepwise_handler.py) — that is the reference for how a
    partially-denoised latent becomes an image. Standalone rather than a method
    so tests can replace it without a model.
    """
    from mflux.utils.image_util import ImageUtil

    unpacked = creator.unpack_latents(latents=latents, height=config.height, width=config.width)
    if hasattr(model.vae, "decode_packed_latents"):
        decoded = model.vae.decode_packed_latents(unpacked)
    else:
        decoded = model.vae.decode(unpacked)
    image = ImageUtil.to_image(
        decoded_latents=decoded,
        config=config,
        seed=seed,
        prompt=prompt,
        quantization=getattr(model, "bits", 0) or 0,
        generation_time=0.0,
    ).image
    image = image.convert("RGB")
    image.thumbnail((_PREVIEW_MAX_PX, _PREVIEW_MAX_PX))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_PREVIEW_JPEG_QUALITY)
    return buffer.getvalue()


@dataclass
class UpscaleJob:
    """One image to enlarge. Carries none of `GenerationJob`'s sampling fields.

    They would all be lies: an upscale has no prompt, no seed, no steps, no
    guidance and no scheduler. `kind` is not here either — an upscale is its
    own kind of work, not a variant of generation.
    """

    spec: UpscalerSpec
    #: The source PNG, on disk. Never bytes in memory: the route already has a
    #: file it owns, and Pillow reads the header without decoding the pixels.
    image_path: Path
    #: Exactly the size to produce. Not a factor: this is what the caller
    #: decided and what the playground row stores, so nothing has to be
    #: divided back out of it. See `upscale_png`.
    target: tuple[int, int]


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
        self._preview_plan: _PreviewPlan | None = None
        #: Latest preview frame of the running generation, or None. Written here
        #: on the worker thread, read from the event loop; see `ProgressSnapshot`
        #: for why that needs no lock.
        self.preview_jpeg: bytes | None = None
        self._preview_count: int = 0  # never reset: monotonic across runs

    def arm(self, label: str, deadline: float | None, preview: _PreviewPlan | None = None) -> None:
        self.label = label
        self.deadline = deadline
        self.timed_out = False
        self.cancel_requested = False
        self.cancelled = False
        self._preview_plan = preview
        self.preview_jpeg = None

    def disarm_preview(self) -> None:
        """Drop the frame and the plan. The plan holds the model: it must not outlive the run."""
        self.preview_jpeg = None
        self._preview_plan = None

    def call_in_loop(
        self,
        *,
        t: int,
        config: Any,
        time_steps: Any,
        latents: Any = None,
        seed: int = 0,
        prompt: str = "",
        **_: Any,
    ) -> None:
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
                "%s - step %d/%d",
                self.label,
                step,
                total,
                extra={"event": "generation_step", "fields": {"step": step, "total": total}},
            )

        plan = self._preview_plan
        if plan is not None and step < total and step % plan.every == 0:
            # `step < total`: the final step's image arrives moments later
            # through the normal path, at full resolution.
            try:
                self.preview_jpeg = _render_preview(
                    model=plan.model,
                    creator=plan.creator,
                    latents=latents,
                    config=config,
                    seed=seed,
                    prompt=prompt,
                )
                # Bytes before the counter: the reader fetches on a counter
                # change, so the frame must already be there.
                self._preview_count += 1
                self._progress.preview_seq = self._preview_count
            except Exception:
                logger.debug("Preview decode failed; previews disabled for this run", exc_info=True)
                self._preview_plan = None


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
        #: The second, bounded slot. Held independently of `_model` on purpose:
        #: an upscale must never evict a warm diffusion model.
        self._upscaler: Any | None = None
        self._upscaler_key: str | None = None

    # ── State ──────────────────────────────────────────────────────────────

    @property
    def loaded_model(self) -> str | None:
        """The resident *diffusion* model, `key:kind`, or None.

        Deliberately still only that: `/health` and `/v1/progress` publish it,
        and the dashboard reads it as "is a model loaded". The upscaler has its
        own property rather than widening this one's meaning.
        """
        return None if self._loaded is None else f"{self._loaded[0]}:{self._loaded[1]}"

    @property
    def loaded_upscaler(self) -> str | None:
        """The resident upscaler's catalogue key, or None."""
        return self._upscaler_key

    def progress(self) -> dict[str, Any]:
        """Progress snapshot. Lock-free, callable from the event loop.

        Taking `self._lock` here would deadlock by construction: it is held for
        the whole generation, exactly when we want to observe it.
        """
        return {
            **self._snapshot.as_dict(),
            "loaded_model": self.loaded_model,
            "upscaler": self.loaded_upscaler,
            "memory": self.memory_stats(),
        }

    def preview(self) -> bytes | None:
        """Latest preview JPEG of the running generation, if any. Lock-free.

        Empty outside a run and whenever the running job did not opt in, which is
        every `/v1` request.
        """
        return self._callback.preview_jpeg

    def request_cancel(self) -> bool:
        """Request that the running job stop. `False` if nothing is running.

        The flag is read by `_ProgressCallback.call_in_loop` during a
        generation, and between tiles during an upscale, so the stop takes
        effect at the next step or the next tile — not instantly.

        One flag, not two: `_ProgressCallback` already *is* the control block
        for the running job, and only its `call_in_loop` is mflux-specific.
        A second flag would open a window where this method reads `state` and
        then arms the wrong one.
        """
        if self._snapshot.state not in ("generating", "upscaling"):
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

    async def upscale(self, job: UpscaleJob) -> bytes:
        """Enlarge one PNG. Serialized against generation: same lock, same worker.

        Sharing them is not tidiness, it is three separate requirements:

        * unified memory — a tile's activations run to hundreds of megabytes,
          and letting those overlap a denoising step is precisely the exhaustion
          the lock exists to prevent;
        * one Metal stream — two threads submitting to MLX parallelize nothing
          measurable and add their peaks;
        * one `ProgressSnapshot` — its docstring justifies a single lock-free
          snapshot *by* there never being two jobs in flight. An unlocked path
          would retract that argument and cost a second progress mechanism, plus
          the SSE surface to publish it.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(self._executor, self._upscale_sync, job)
            except (APIError, GenerationTimeout):
                raise
            except Exception as exc:
                raise translate_mflux_exception(exc) from exc

    async def unload(self) -> None:
        """Release everything the engine holds — both slots.

        Both, because this is what `/v1/unload` means to whoever pressed "Free
        memory". Leaving 33 MB behind a button named that is a cheap lie.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._unload_all_sync)

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
        self._unload_all_sync()
        self._snapshot.reset()

    # ── Implementation (worker thread) ─────────────────────────────────────

    def _generate_sync(self, job: GenerationJob) -> bytes:
        try:
            model = self._ensure_model(job.spec, job.kind)
            label = f"{job.spec.key} seed={job.seed} {job.width}x{job.height}"
            deadline = time.monotonic() + self._request_timeout_s if self._request_timeout_s else None

            preview = None
            if job.preview_every > 0:
                family = job.spec.edit.family if job.kind == "edit" and job.spec.edit else job.spec.family
                creator = latent_creator_for(family)
                if creator is not None:
                    preview = _PreviewPlan(every=job.preview_every, creator=creator, model=model)
            self._callback.arm(label, deadline, preview)

            self._snapshot.state = "generating"
            self._snapshot.model = job.spec.key
            self._snapshot.kind = job.kind
            self._snapshot.seed = job.seed
            self._snapshot.step = 0
            self._snapshot.total = job.steps
            self._snapshot.started_at = time.monotonic()

            logger.info(
                "▶ %s - %d steps",
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
                "✓ %s - %.1f s",
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
            # The endpoint 404s again the moment the run is over, and the plan
            # goes with it: holding it would keep the model alive past `unload()`.
            self._callback.disarm_preview()

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
            "Loading %s (%s) - %s",
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
            "Model %s ready in %.1fs - memory %s",
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

    def _ensure_upscaler(self, spec: UpscalerSpec) -> Any:
        """The resident upscaler for `spec`, loading it if needed.

        Symmetric with `_ensure_model` but for one deliberate difference:
        loading an upscaler releases any *other* upscaler, and never touches
        `self._model`. That is the whole point of the second slot, and the line
        `tests/test_engine.py` pins.
        """
        if self._upscaler_key == spec.key and self._upscaler is not None:
            return self._upscaler

        if self._upscaler is not None:
            logger.info(
                "Unloading upscaler %s",
                self._upscaler_key,
                extra={"event": "upscaler_unload", "fields": {"upscaler": self._upscaler_key}},
            )
            self._unload_upscaler_sync()

        # "loading", as for a diffusion model: the first use of an upscaler
        # downloads tens of megabytes, and the UI should say so rather than sit
        # at tile 0.
        self._snapshot.state = "loading"
        self._snapshot.model = spec.key
        self._snapshot.kind = "upscale"
        self._snapshot.started_at = time.monotonic()

        from qds.upscale.weights import load_upscaler

        started = time.monotonic()
        model = load_upscaler(spec)
        self._upscaler = model
        self._upscaler_key = spec.key
        logger.info(
            "Upscaler %s ready in %.1fs - memory %s",
            spec.key,
            time.monotonic() - started,
            self.memory_stats(),
            extra={
                "event": "upscaler_ready",
                "fields": {"upscaler": spec.key, "elapsed_s": round(time.monotonic() - started, 1)},
            },
        )
        return model

    def _upscale_sync(self, job: UpscaleJob) -> bytes:
        from qds.upscale.pipeline import tile_grid, upscale_png

        try:
            model = self._ensure_upscaler(job.spec)
            label = f"{job.spec.key} {job.target[0]}x{job.target[1]} {job.image_path.name}"
            deadline = time.monotonic() + self._request_timeout_s if self._request_timeout_s else None
            # No preview plan: there are no latents to decode, so `preview_seq`
            # stays 0 and the client shows the tile counter instead. Arming
            # still matters — it clears the previous run's cancel flag and drops
            # its frame, which would otherwise keep being served.
            self._callback.arm(label, deadline, None)

            with Image.open(job.image_path) as opened:
                width, height = opened.size
            total = len(tile_grid(width, height, job.spec.tile))

            self._snapshot.state = "upscaling"
            self._snapshot.model = job.spec.key
            self._snapshot.kind = "upscale"
            self._snapshot.seed = None
            self._snapshot.step = 0
            self._snapshot.total = total
            self._snapshot.started_at = time.monotonic()

            logger.info(
                "▶ %s - %d tiles",
                label,
                total,
                extra={
                    "event": "upscale_start",
                    "fields": {
                        "upscaler": job.spec.key,
                        "target": f"{job.target[0]}x{job.target[1]}",
                        "width": width,
                        "height": height,
                        "tiles": total,
                    },
                },
            )
            started = time.monotonic()
            png = upscale_png(
                model, job.image_path, spec=job.spec, target=job.target, on_tile=self._on_tile
            )
            elapsed = time.monotonic() - started
            logger.info(
                "✓ %s - %.1f s",
                label,
                elapsed,
                extra={
                    "event": "upscale_done",
                    "fields": {"upscaler": job.spec.key, "elapsed_s": round(elapsed, 1)},
                },
            )
            return png
        except Exception as exc:
            if self._callback.timed_out:
                raise GenerationTimeout(self._request_timeout_s) from exc
            raise
        finally:
            self._snapshot.reset()
            self._callback.disarm_preview()

    def _on_tile(self, done: int, total: int) -> None:
        """Between tiles: publish progress, and honour a stop.

        This is the upscale's equivalent of `call_in_loop`'s head, and it
        raises the same exception for the same reason: `StopImageGenerationException`
        already means "the run was interrupted" everywhere downstream, so
        `translate_mflux_exception` maps it to `generation_stopped` and the
        playground runner records `cancelled` with no extra code.
        """
        from mflux.utils.exceptions import StopImageGenerationException

        if self._callback.cancel_requested:
            self._callback.cancelled = True
            raise StopImageGenerationException(f"Upscale cancelled at tile {done}/{total}")
        if self._callback.deadline is not None and time.monotonic() > self._callback.deadline:
            self._callback.timed_out = True
            raise StopImageGenerationException(f"Timed out at tile {done}/{total}")
        self._snapshot.step = done
        self._snapshot.total = total

    def _unload_upscaler_sync(self) -> None:
        """Release the second slot only. Never touches the diffusion model."""
        model = self._upscaler
        self._upscaler = None
        self._upscaler_key = None
        if model is None:
            return
        del model
        gc.collect()
        _clear_mlx_cache()

    def _unload_all_sync(self) -> None:
        """Both slots. `_unload_sync` stays strictly the diffusion model.

        Kept apart because they are not the same operation: `_unload_sync`
        resets a `CallbackRegistry`, walks `_UNLOADABLE_ATTRS` and clears a
        prompt cache, none of which an RRDBNet has. `_ensure_model` calls that
        one, and it must not take the upscaler with it.
        """
        self._unload_sync()
        self._unload_upscaler_sync()

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
