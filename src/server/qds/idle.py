"""Releasing the warm model after a period of inactivity.

Keeping the model warm is the whole point of this server — it takes a generation
from 34s to 18s. But on unified memory a warm model confiscates the machine: with
a text LLM running alongside, one image makes the chat unusable until someone
thinks to hit "Free memory".

Hence `server.idle_unload_s`: `None` never releases (the historical behaviour),
`0` releases as soon as the request ends, `N` after N seconds with no generation.

**The countdown is armed per request, not per image.** `ModelEngine.generate()`
takes its lock once per image, so arming it there would fire *between* the images
of an `n=3` request with a delay of 0 — three loads for three images. Keeping the
policy at the request boundary also keeps the engine ignorant of the HTTP request
lifecycle, which is what makes both testable on their own.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from qds.logs import SERVER_LOGGER

logger = logging.getLogger(SERVER_LOGGER)


class IdleUnloader:
    """Frees the engine's model once nothing has run for `delay` seconds.

    Used as a synchronous context manager around a request's generation loop:
    entering cancels any pending countdown, leaving re-arms it.
    """

    def __init__(self, engine: Any, delay: float | None):
        self._engine = engine
        self._delay = delay
        self._task: asyncio.Task[None] | None = None
        #: Requests currently generating. A plain flag would not do: two
        #: overlapping requests would leave the countdown armed while the second
        #: is still working, and with a delay of 0 it would slip between its
        #: images. We only re-arm on the way down to zero.
        self._inflight = 0

    @property
    def enabled(self) -> bool:
        return self._delay is not None

    def __enter__(self) -> IdleUnloader:
        self._inflight += 1
        self.cancel()
        return self

    def __exit__(self, *_: object) -> None:
        self._inflight -= 1
        if self._inflight <= 0:
            self._inflight = 0
            self._arm()

    def cancel(self) -> None:
        """Drop a pending countdown. Also the shutdown hook."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def _arm(self) -> None:
        if self._delay is None:
            return
        self.cancel()
        self._task = asyncio.create_task(self._wait_then_release())

    async def _wait_then_release(self) -> None:
        try:
            await asyncio.sleep(self._delay or 0)
            before = self._engine.memory_stats()
            model = self._engine.loaded_model
            upscaler = self._engine.loaded_upscaler
            # Both, or an upscale done without generating first — a reopened
            # session, or a model already released — would leave the upscaler
            # resident forever: the countdown would arm, wake, see no diffusion
            # model, and give up.
            if model is None and upscaler is None:
                return
            await self._engine.unload()
            after = self._engine.memory_stats()
            # Logged, and with figures: a silent release would read as the model
            # having leaked away on its own.
            released = " + ".join(filter(None, (model, upscaler))) or "nothing"
            logger.info(
                "Released %s after %.10gs idle - memory %s → %s",
                released,
                self._delay,
                before.get("active_gb"),
                after.get("active_gb"),
                extra={
                    "event": "model_idle_unload",
                    "fields": {
                        "model": model,
                        "upscaler": upscaler,
                        "idle_unload_s": self._delay,
                        "active_gb_before": before.get("active_gb"),
                        "active_gb_after": after.get("active_gb"),
                    },
                },
            )
        except asyncio.CancelledError:
            # A new request came in, or the server is shutting down. Neither is
            # worth a word.
            raise
        except Exception:  # pragma: no cover - defensive
            logger.warning("Automatic release failed", exc_info=True)
