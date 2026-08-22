"""Submit, wait with a ceiling, and cancel without losing the job.

A generation takes tens of seconds to minutes, and a chat client's model wants
one tool call that comes back with a picture. So the tools block. Three things
follow, and this module is all three:

* **Progress.** While waiting, notifications go out -- honestly, which is
  `progress.Attribution`'s job, not this one's.
* **A ceiling.** Waiting forever is not an option a client can recover from, so
  the wait ends and *returns* the generation id. Returns, not raises: nothing
  failed, the work is queued and durable, and `wait_for_generation` resumes it.
  The returned text says whether the queue is paused, because a human who paused
  it in the browser is otherwise the invisible cause of every timeout.
* **Cancellation.** A client that disconnects or cancels the call must not leave
  a generation running on somebody's GPU.
"""

from __future__ import annotations

import asyncio
import contextlib

TERMINAL = {"completed", "failed", "cancelled"}


async def wait_for(
    deps,
    ctx,
    *,
    generation_id: str,
    attribution,
    timeout_s: float | None = None,
) -> dict:
    """Poll this generation's row until it settles, or the ceiling elapses.

    Returns the row either way; the caller reads `status` to find out which.

    The row is polled rather than the engine subscribed to, and that is the
    honest arrangement rather than a lazy one: the row is authoritative for
    *this* job, while the engine's snapshot is global (see `progress.py`). A
    subscription would have to answer "is this event mine?" on every frame, and
    would answer it with the same four-way check for more machinery.
    """
    ceiling = deps.settings.mcp.tool_timeout_s
    if timeout_s is not None:
        ceiling = min(ceiling, max(timeout_s, 0.0))
    interval = deps.settings.mcp.poll_interval_s

    loop = asyncio.get_running_loop()
    deadline = loop.time() + ceiling

    try:
        while True:
            record = deps.store.get_generation(generation_id)
            if record is None:
                # Deleted from the playground while we waited. Not an error to
                # raise at the model: someone with the machine in front of them
                # decided this should not exist.
                return {"id": generation_id, "status": "cancelled", "images": [], "error": None}

            progress, total, message = attribution.sample(
                record=record, runner=deps.runner, engine=deps.engine
            )
            with contextlib.suppress(Exception):
                # A notification is a courtesy; a client that cannot take one
                # must not lose the generation over it.
                await ctx.report_progress(progress, total, message)

            if record["status"] in TERMINAL:
                return record
            if loop.time() >= deadline:
                return record

            await asyncio.sleep(min(interval, max(deadline - loop.time(), 0.0)))
    except asyncio.CancelledError:
        _cancel_detached(deps, generation_id)
        raise


def _cancel_detached(deps, generation_id: str) -> None:
    """Cancel on a task that outlives the handler being torn down.

    `ensure_future` rather than `await`: this runs while the handler task is
    unwinding, and an await there is not reliably completed. The independent
    task survives that unwinding and does the real work.

    And `runner.cancel`, never a local copy of its body. That method already
    reasons about the three states a generation can be cancelled in -- still
    queued, mid-denoise, or parked on the pause gate -- and a second copy of
    that reasoning is a second thing to get wrong when one of them changes.
    """
    with contextlib.suppress(RuntimeError):
        task = asyncio.ensure_future(deps.runner.cancel(generation_id))
        # Without a reference the loop may drop it before it runs; without the
        # callback a failure would surface as "exception never retrieved".
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)


#: Detached cancellations still in flight. A module-level set is the smallest
#: thing that keeps them referenced for the few milliseconds they need.
_PENDING: set = set()
