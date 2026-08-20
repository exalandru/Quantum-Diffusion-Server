"""Lifecycle of the long model operations: downloading weights, and conversions.

A port of what the desktop app's Rust `job.rs` did, moved into the server now
that the server is what the dashboard talks to. The shape is kept deliberately:
one slot, one lock, a process group, and a SIGTERM → grace → SIGKILL ladder.

**Why a subprocess and not a task.** Both operations are blocking work that ends
up in threads — a HuggingFace download, an MLX conversion. Python cannot kill a
thread, so an in-process job could be *asked* to stop and nothing more, while
`killpg` on a child actually stops one. A conversion also peaks around 66 GB;
a child hands all of that back to the operating system when it exits, where an
in-process one would leave MLX holding it alongside a possibly-resident model.

**Why the child is `python -m qds`.** `sys.executable -m qds` names the
interpreter already running, so the child is guaranteed to be the same
installation as its parent — no PATH lookup that could find a different `qds`.
It inherits this process's environment, so it resolves `QDS_SERVER_CONFIG` and
`HF_HOME` exactly as the server did, rather than through a second, parallel
assembly of the same variables.

**Signalling a pid is only safe before it is reaped**, and this module narrows
that window rather than closing it. Once the child is reaped the operating
system may hand its number to somebody else, and a late SIGKILL would land on a
stranger. Every signal below is therefore guarded, under the lock, by
`returncode is None`.

That guard is weaker than it looks, and the weaker statement is the true one.
CPython's default `ThreadedChildWatcher` calls `os.waitpid` on a helper *thread*
and only then schedules the assignment of `returncode` onto the loop
(`asyncio/unix_events.py`). So `returncode is None` means "this loop has not
been told yet", not "not yet reaped": there is a window, bounded by one
`call_soon_threadsafe` hop, in which the child is gone and the guard still
passes.

What keeps that from being a real hazard is the *shape* of the signal, not the
guard: `killpg` addresses a process **group**, so a stray signal would need the
kernel to have handed that exact pid to a new process-group leader — a process
that called `setsid`/`setpgid` — inside a window measured in microseconds, on a
platform whose pids increment monotonically through a large space. The residual
risk is accepted and stated here rather than claimed away. It would be closed
properly by a pidfd, which macOS does not provide.

Falsifier: a signal delivered to an unrelated process group would show up as an
unexplained termination elsewhere on the machine, correlated with a cancelled
job. Nothing of the sort has been observed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from qds.logs import SERVER_LOGGER

logger = logging.getLogger(f"{SERVER_LOGGER}.jobs")

#: Grace given to a cancelled child before SIGKILL. Shorter than the server's own
#: shutdown ladder, because these children have no in-flight HTTP to drain — the
#: cost of cutting one off is a partial download, which was already the outcome.
CANCEL_GRACE_S = 10.0


class JobKind(StrEnum):
    FETCH = "fetch"
    PREQUANTIZE = "prequantize"

    @property
    def label(self) -> str:
        return {
            JobKind.FETCH: "a model download",
            JobKind.PREQUANTIZE: "a model conversion",
        }[self]


class JobState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    #: SIGTERM sent, child not yet reaped. Distinct from `running` so the
    #: interface can stop offering Cancel twice, and from `cancelled` so it does
    #: not claim the process is gone before it is.
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_active(self) -> bool:
        return self in {JobState.RUNNING, JobState.CANCELLING}


class JobBusy(RuntimeError):
    """A job is already running. Carries the message the interface shows."""


class NoJobRunning(RuntimeError):
    """Nothing to cancel."""


@dataclass
class JobStatus:
    """What a client needs to reconstruct the operation after a reload.

    Field names match what the Rust implementation serialised, so the dashboard
    keeps reading one shape across the move.
    """

    state: JobState = JobState.IDLE
    kind: JobKind | None = None
    #: Model key for a fetch; the model and components for a conversion.
    target: str | None = None
    #: Name of the last structured event on the child's stdout, verbatim.
    event: str | None = None
    #: Its `fields` object, verbatim — the child's schema is not reshaped here.
    fields: dict[str, Any] | None = None
    #: Latest human-readable line while running; the terminal reason once done.
    message: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "kind": self.kind.value if self.kind else None,
            "target": self.target,
            "event": self.event,
            "fields": self.fields,
            "message": self.message,
            "startedAtMs": self.started_at_ms,
            "finishedAtMs": self.finished_at_ms,
        }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _failure_message(error_message: str | None, code: int | None) -> str:
    """What a failed job reports: what the child said, then the exit code.

    The fallback used to be the whole story — every failure read "exited with
    code 1", whatever had actually gone wrong.
    """
    if error_message:
        return error_message
    if code is None:  # pragma: no cover - only reachable if called before exit
        return "terminated by a signal"
    if code < 0:
        return f"terminated by signal {-code}"
    return f"exited with code {code}"


def child_command(argv: list[str]) -> list[str]:
    """The command that runs a subcommand as a child of this server.

    `sys.executable -m qds` rather than a `qds` binary from PATH: it names the
    interpreter already running, so the child cannot be a different
    installation than its parent.
    """
    return [sys.executable, "-m", "qds", *argv]


def _signal_group(pid: int, sig: int) -> None:
    """Signal the child's whole process group.

    The group, not the process: a download spawns its own workers, and
    signalling only the parent leaves them running and holding the cache. The
    child is started with `start_new_session=True`, so its pid *is* its group id.
    """
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        # Already gone; the waiter will settle the status.
        pass
    except PermissionError:  # pragma: no cover - defensive
        logger.warning("not permitted to signal process group %d", pid)


@dataclass
class _Running:
    process: asyncio.subprocess.Process
    pid: int
    #: Set by `cancel`, read when settling to tell a cancellation apart from a
    #: failure — the exit code cannot, since a SIGTERM'd Python exits non-zero
    #: either way.
    cancelling: bool = False
    #: Last ERROR-level message from the structured stream, kept as the terminal
    #: reason candidate.
    error_message: str | None = None
    pumps: list[asyncio.Task] = field(default_factory=list)

    @property
    def reaped(self) -> bool:
        """Whether this loop has been told the child exited.

        Not quite "whether the pid is safe to signal" — see the module
        docstring for the window this leaves and why it is accepted.
        """
        return self.process.returncode is not None


class JobManager:
    """One heavy model operation at a time, cancellable, in its own process group.

    Every mutation happens under `_lock`, and every signal is guarded by
    `_Running.reaped` inside that critical section — together, those are what
    make cancelling and finishing race-free.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._status = JobStatus()
        self._running: _Running | None = None
        self._waiter: asyncio.Task | None = None
        #: Called when a completed conversion has changed the configuration.
        #: The manager states the fact; whoever wired it decides what it means
        #: for the interface — one flag, one owner.
        self.on_config_changed: Callable[[], None] | None = None

    # ── Reading ────────────────────────────────────────────────────────────

    async def status(self) -> JobStatus:
        async with self._lock:
            return self._status

    # ── Starting ───────────────────────────────────────────────────────────

    async def start_fetch(self, key: str) -> JobStatus:
        return await self._start(JobKind.FETCH, key, ["fetch", key, "--json-logs"])

    async def start_prequantize(
        self,
        model: str,
        bits: int,
        components: list[str] | None = None,
        dest: str | None = None,
    ) -> JobStatus:
        argv = ["prequantize", "--json-logs", "--model", model, "--bits", str(bits)]
        if dest:
            argv += ["--dest", dest]
        if components:
            argv += ["--components", *components]
        target = f"{model} {bits}-bit"
        if components:
            target += f" ({', '.join(components)})"
        return await self._start(JobKind.PREQUANTIZE, target, argv)

    async def _start(self, kind: JobKind, target: str, argv: list[str]) -> JobStatus:
        async with self._lock:
            self._ensure_free()

            process = await asyncio.create_subprocess_exec(
                *child_command(argv),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Its own process group, so the ladder can signal the child and
                # everything it spawns as one.
                start_new_session=True,
            )
            running = _Running(process=process, pid=process.pid)
            self._running = running
            self._status = JobStatus(
                state=JobState.RUNNING,
                kind=kind,
                target=target,
                started_at_ms=_now_ms(),
            )
            logger.info("%s started: %s (pid %d)", kind.label, target, process.pid)

            running.pumps = [
                asyncio.create_task(self._pump(process.stdout, structured=True)),
                asyncio.create_task(self._pump(process.stderr, structured=False)),
            ]
            self._waiter = asyncio.create_task(self._await_exit(running))
            return self._status

    def _ensure_free(self) -> None:
        """Raise when a job is already active, naming it so the message helps."""
        if not self._status.state.is_active:
            return
        what = self._status.kind.label if self._status.kind else "an operation"
        target = f" ({self._status.target})" if self._status.target else ""
        raise JobBusy(
            f"{what}{target} is already running. Only one heavy model operation runs at a "
            f"time - they compete for the same unified memory and the same HuggingFace "
            f"cache. Wait for it, or cancel it first."
        )

    # ── Output ─────────────────────────────────────────────────────────────

    async def _pump(self, stream: asyncio.StreamReader | None, *, structured: bool) -> None:
        r"""Read one of the child's streams to EOF, folding lines into the status.

        Split on `\r` as well as `\n`: progress bars rewrite one line with
        carriage returns, and reading by `\n` alone would hold a whole
        download's worth of output in a single unbounded "line".
        """
        if stream is None:  # pragma: no cover - both pipes are always requested
            return
        buffer = b""
        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                buffer = (buffer + chunk).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                *lines, buffer = buffer.split(b"\n")
                for raw in lines:
                    await self._note_line(raw.decode("utf-8", "replace"), structured=structured)
            if buffer:
                await self._note_line(buffer.decode("utf-8", "replace"), structured=structured)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception:  # pragma: no cover - a broken pipe must not kill the job
            logger.debug("job output stream ended abruptly", exc_info=True)

    async def _note_line(self, line: str, *, structured: bool) -> None:
        line = line.strip()
        if not line:
            return
        # Every line reaches the log as it came: the Logs view shows the child's
        # real output rather than this module's summary of it.
        if structured:
            logger.info("%s", line)
        else:
            logger.debug("%s", line)
            return
        async with self._lock:
            self._fold(line)

    def _fold(self, line: str) -> None:
        """Fold one structured stdout line into the status.

        Only the fields the interface needs: the event name and its `fields` for
        progress, `message` for the current step, and an ERROR-level `message`
        as the terminal reason. Everything else on the line is ignored, and the
        raw line still reaches the log untouched — this is not a second protocol.
        """
        if not self._status.state.is_active or self._running is None:
            return
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(record, dict):
            return

        event = record.get("event")
        if isinstance(event, str):
            self._status.event = event
            fields = record.get("fields")
            self._status.fields = fields if isinstance(fields, dict) else None

        message = record.get("message")
        if isinstance(message, str):
            self._status.message = message
            if record.get("level") in {"ERROR", "CRITICAL"}:
                self._running.error_message = message

    # ── Finishing ──────────────────────────────────────────────────────────

    async def _await_exit(self, running: _Running) -> None:
        """Wait for the child, drain its output, then settle the terminal state.

        The draining is not tidiness. A conversion emits `prequantize_done` and
        exits immediately after; settling as soon as the process is gone would
        race that last line, and the variant it announces would never be
        selected. Waiting for both pumps to reach EOF happens *outside* the lock
        precisely because folding a line takes it.
        """
        with contextlib.suppress(asyncio.CancelledError):
            await running.process.wait()
        await asyncio.gather(*running.pumps, return_exceptions=True)

        async with self._lock:
            if self._running is not running:
                # Superseded by `shutdown`, which already settled this job.
                return
            self._settle(running)

    def _settle(self, running: _Running) -> None:
        """Record the terminal state. Caller holds `_lock`."""
        code = running.process.returncode
        self._running = None
        self._status.finished_at_ms = _now_ms()
        if running.cancelling:
            self._status.state = JobState.CANCELLED
            self._status.message = "Cancelled."
        elif code == 0:
            self._status.state = JobState.COMPLETED
            self._status.message = None
        else:
            self._status.state = JobState.FAILED
            self._status.message = _failure_message(running.error_message, code)

        logger.info(
            "job %d finished: %s (%s)",
            running.pid,
            self._status.state.value,
            self._status.message or "no message",
        )

        # Still under the lock, and the status is already terminal: whatever this
        # does is done before any reader can be told the job finished. Failure
        # and cancellation do nothing — there is nothing to record about a
        # conversion that produced no artifact.
        if self._status.state is JobState.COMPLETED:
            self._activate_if_variant_ready()

    def _activate_if_variant_ready(self) -> None:
        """Select a variant the conversion has just declared complete.

        **Who decides, and where.** Only the conversion can say an artifact is
        whole: it validates every required component, checks the precision the
        weights actually carry, and only then writes the completion marker and
        emits `prequantize_done`. A run that converted a subset emits
        `prequantize_partial` and never reaches that line — so "exited 0" is
        *not* the signal, and completeness is not re-derived here. This reads
        the claim and acts on it.

        Writing the file under the lock is deliberate: the selection has to be
        recorded before any reader can observe the job as finished, or the first
        status poll wins the race and the interface refreshes against a
        configuration that has not been written yet. The write is synchronous
        and includes two `fsync`s, so it stalls the event loop — for a file of a
        few kilobytes, which is the price of the ordering above. Moving it to a
        thread would keep the lock but hand the loop back mid-write, and the
        ordering is exactly what must not be interruptible.

        A failure to write is logged and dropped on purpose: the conversion
        succeeded and its artifact is on disk and valid, so refusing to record
        the selection is a smaller loss than failing a job that did what it was
        asked.
        """
        if self._status.event != "prequantize_done":
            return
        fields = self._status.fields or {}
        model = fields.get("model")
        bits = fields.get("bits")
        # `bool` is an `int`; a JSON `true` here is not a bit depth.
        if not isinstance(model, str) or not isinstance(bits, int) or isinstance(bits, bool):
            logger.warning("a completed conversion named no model and bit depth; nothing selected")
            return

        from qds import configfile

        try:
            configfile.select_variant(model, bits)
        except Exception as exc:
            logger.warning(
                "%s's %d-bit copy is ready but could not be selected: %s", model, bits, exc
            )
            return
        if self.on_config_changed is not None:
            self.on_config_changed()
        logger.info("%s now uses its %d-bit copy on the next server start", model, bits)

    # ── Cancelling ─────────────────────────────────────────────────────────

    async def cancel(self) -> JobStatus:
        """SIGTERM the child's process group, and arm the SIGKILL that follows."""
        async with self._lock:
            running = self._running
            if not self._status.state.is_active or running is None or running.reaped:
                # `reaped` included on purpose: the child is gone and its waiter
                # is about to settle the status, so there is nothing to signal —
                # and signalling anyway could land on a recycled pid.
                raise NoJobRunning("No model operation is running.")

            running.cancelling = True
            self._status.state = JobState.CANCELLING
            self._status.message = "Stopping…"
            _signal_group(running.pid, signal.SIGTERM)
            logger.info("cancelling job (pid %d)", running.pid)
            asyncio.create_task(self._escalate_after_grace(running))
            return self._status

    async def _escalate_after_grace(self, running: _Running) -> None:
        """SIGKILL a cancellation that SIGTERM did not finish.

        No-op if the child already exited or a different job has since started.
        Both are checked against the job this escalation was armed for rather
        than against whatever is running now, so a slow cancellation can never
        kill its successor.
        """
        await asyncio.sleep(CANCEL_GRACE_S)
        async with self._lock:
            if self._running is not running or running.reaped:
                return
            logger.warning(
                "job %d did not stop within %.0fs; killing", running.pid, CANCEL_GRACE_S
            )
            _signal_group(running.pid, signal.SIGKILL)

    async def shutdown(self) -> None:
        """Terminate without waiting, for server shutdown.

        SIGKILL rather than the ladder: the server is going away, and a child
        left behind becomes an orphan under launchd holding the HuggingFace
        cache, invisible to whatever starts next.
        """
        async with self._lock:
            running = self._running
            if running is None:
                return
            if not running.reaped:
                logger.info("killing job %d on shutdown", running.pid)
                _signal_group(running.pid, signal.SIGKILL)
            for pump in running.pumps:
                pump.cancel()
            self._running = None
            self._status.state = JobState.CANCELLED
            self._status.message = "The server stopped."
            self._status.finished_at_ms = _now_ms()

        if self._waiter is not None:
            self._waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._waiter
