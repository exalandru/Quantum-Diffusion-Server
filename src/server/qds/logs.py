"""Logging configuration.

Two distinct namespaces, and the distinction matters: `"mflux"` is the
*library's* logger (it never calls `basicConfig`, so attaching a handler is
enough to capture its messages), `"qds"` is ours. The prototype used
`"mflux"` for its own logs, which would have mixed the two together once mflux
was imported in-process.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

SERVER_LOGGER = "qds"
LIBRARY_LOGGER = "mflux"

logger = logging.getLogger(SERVER_LOGGER)


class JsonFormatter(logging.Formatter):
    """One line, one JSON object, for a supervisor reading stdout.

    Lifecycle events additionally carry an `event` name and the contents of
    `fields`, attached through `extra=` at the call site — the human-readable
    message stays alongside them. One caveat for consumers: tqdm writes to
    stderr without going through logging (see `capture_stdout`), so any line
    that fails to parse as JSON should be ignored rather than treated as an
    error.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = "mflux.log",
    json_lines: bool = False,
) -> None:
    # In JSON mode we write to **stdout** rather than stderr: mflux renders its
    # denoising progress with tqdm (`Config.time_steps`), which emits
    # carriage-return-terminated fragments to stderr with no newline. Our JSON
    # objects would end up glued to them — `{"ts": …}` preceded by
    # `\r 0%| | 0/40` on the same segment — and any reasonable consumer would
    # miss all of them. tqdm offers no environment variable to silence itself,
    # hence the channel split: stdout carries structured events, stderr carries
    # human-readable text and progress bars. `main()` disables uvicorn's access
    # log accordingly, since it would otherwise pollute stdout in turn.
    console = logging.StreamHandler(sys.stdout if json_lines else sys.stderr)
    console.setFormatter(
        JsonFormatter()
        if json_lines
        else logging.Formatter("[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )

    handlers: list[logging.Handler] = [console]
    if log_file:
        # `FileHandler` neither expands `~` nor creates parent directories:
        # without this, a path like `~/Library/Logs/qds/mflux.log`
        # would create a directory literally named `~`, or raise
        # FileNotFoundError at startup.
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(
            JsonFormatter()
            if json_lines
            else logging.Formatter(
                "%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    for name in (SERVER_LOGGER, LIBRARY_LOGGER):
        target = logging.getLogger(name)
        for handler in list(target.handlers):
            target.removeHandler(handler)
        for handler in handlers:
            target.addHandler(handler)
        target.setLevel(level)
        # Without this, messages would also reach uvicorn's root handler and be
        # printed twice.
        target.propagate = False


class _LoggerWriter(io.TextIOBase):
    """File-like object that re-emits each complete line to a logger."""

    def __init__(self, target: logging.Logger, level: int = logging.INFO):
        self._logger = target
        self._level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.strip())
        self._buffer = ""


@contextlib.contextmanager
def capture_stdout() -> Iterator[None]:
    """Redirect mflux's `print()` calls to the library logger.

    mflux emits a few on the hot path (quantization warnings, "Downloading
    model from HuggingFace…"). Its tqdm bar, however, writes to stderr and is
    deliberately left visible in the terminal: it is the only live progress
    readout for someone watching the console.

    Safe despite mutating the global `sys.stdout`: the engine serializes
    generations onto a single worker.
    """
    writer = _LoggerWriter(logging.getLogger(LIBRARY_LOGGER))
    previous = sys.stdout
    sys.stdout = writer  # type: ignore[assignment]
    try:
        yield
    finally:
        writer.flush()
        sys.stdout = previous
