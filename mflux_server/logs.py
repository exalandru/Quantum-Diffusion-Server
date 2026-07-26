"""Configuration du logging.

Deux namespaces distincts, et c'est important : `"mflux"` est le logger
*de la librairie* (elle n'appelle jamais `basicConfig`, donc y poser un
handler suffit à capturer ses messages), `"mflux_server"` est le nôtre. Le
prototype utilisait `"mflux"` pour ses propres logs, ce qui aurait mélangé
les deux une fois mflux importé en process.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

SERVER_LOGGER = "mflux_server"
LIBRARY_LOGGER = "mflux"

logger = logging.getLogger(SERVER_LOGGER)


def setup_logging(level: str = "INFO", log_file: str | Path | None = "mflux.log") -> None:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))

    handlers: list[logging.Handler] = [console]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
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
        # Sans ça, les messages remonteraient aussi au handler racine
        # d'uvicorn et seraient affichés en double.
        target.propagate = False


class _LoggerWriter(io.TextIOBase):
    """Fichier-like qui réémet chaque ligne complète vers un logger."""

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
    """Redirige les `print()` de mflux vers le logger de la librairie.

    mflux en émet quelques-uns sur le chemin chaud (warning de quantization,
    « Downloading model from HuggingFace… »). La barre tqdm, elle, écrit sur
    stderr et reste volontairement visible dans le terminal : c'est la seule
    progression lisible en direct pour qui regarde la console.

    Sûr malgré la mutation globale de `sys.stdout` : le moteur sérialise les
    générations sur un unique worker.
    """
    writer = _LoggerWriter(logging.getLogger(LIBRARY_LOGGER))
    previous = sys.stdout
    sys.stdout = writer  # type: ignore[assignment]
    try:
        yield
    finally:
        writer.flush()
        sys.stdout = previous
