"""A bounded, readable tail of the server's own log.

The desktop app used to receive every log line as a Tauri event and keep the
last 2000 in React state. The dashboard is a web page with no such channel, so
the buffer moves here: the server keeps the same bounded tail, and `/admin/logs`
hands out whatever a client has not seen yet.

**A cursor, not a stream.** Each entry carries a monotonically increasing `seq`,
and a reader asks for everything after the last one it holds. That makes a poll
idempotent and a missed poll harmless, and it means a client that was closed for
a minute resumes exactly where it stopped — up to the buffer's own bound, which
it can detect because the first `seq` it receives is higher than the one it
asked for. A second streaming protocol alongside `/v1/progress` would buy none
of that.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

from qds.logs import LIBRARY_LOGGER, SERVER_LOGGER

#: How much history is kept. The same bound the desktop app applied in React,
#: for the same reason: a conversion emits a line per block for hours, and an
#: unbounded buffer is a slow memory leak with a log-shaped excuse.
CAPACITY = 2000


class LogBuffer:
    """The last `capacity` records, each with a cursor.

    Written from whatever thread emitted the log record — uvicorn's loop, the
    engine's executor thread, a job's output pump — so the deque is guarded by a
    plain lock rather than an asyncio one. The critical section is an append.
    """

    def __init__(self, capacity: int = CAPACITY) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._next_seq = 1

    def append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            entry["seq"] = self._next_seq
            self._next_seq += 1
            self._entries.append(entry)

    def after(self, seq: int, limit: int) -> dict[str, Any]:
        """Entries newer than `seq`, oldest first, and where the tail now ends.

        `dropped` says how many entries fell out of the buffer between what the
        caller last saw and the oldest one still held — the difference between
        "nothing happened" and "you missed it", which a client cannot otherwise
        tell.
        """
        with self._lock:
            entries = [entry for entry in self._entries if entry["seq"] > seq]
            oldest_held = self._entries[0]["seq"] if self._entries else self._next_seq
            last_seq = self._next_seq - 1
        dropped = max(0, oldest_held - seq - 1) if seq else 0
        truncated = entries[-limit:] if limit and len(entries) > limit else entries
        return {
            "entries": truncated,
            "lastSeq": last_seq,
            "dropped": dropped + (len(entries) - len(truncated)),
        }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class BufferHandler(logging.Handler):
    """Feeds a `LogBuffer` from the logging machinery.

    Records are reduced to what a reader can use, keeping the structured
    `event`/`fields` the job protocol carries — the same two attributes
    `JsonFormatter` publishes, read the same way, so the buffer and the JSON
    Lines on stdout cannot describe an event differently.
    """

    def __init__(self, buffer: LogBuffer) -> None:
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry: dict[str, Any] = {
                "ts": self.formatter.formatTime(record) if self.formatter else record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            event = getattr(record, "event", None)
            if event is not None:
                entry["event"] = event
            fields = getattr(record, "fields", None)
            if fields:
                entry["fields"] = fields
            if record.exc_info:
                entry["traceback"] = logging.Formatter().formatException(record.exc_info)
            self._buffer.append(entry)
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def attach(buffer: LogBuffer, level: int = logging.INFO) -> BufferHandler:
    """Add the buffer to the two loggers the server writes through.

    Called *after* `setup_logging`, which clears the handler list on both
    loggers: attaching first would be attaching to something about to be thrown
    away.
    """
    handler = BufferHandler(buffer)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    for name in (SERVER_LOGGER, LIBRARY_LOGGER):
        logging.getLogger(name).addHandler(handler)
    return handler


def detach(handler: BufferHandler) -> None:
    for name in (SERVER_LOGGER, LIBRARY_LOGGER):
        logging.getLogger(name).removeHandler(handler)
