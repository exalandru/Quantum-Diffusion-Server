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
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

SERVER_LOGGER = "mflux_server"
LIBRARY_LOGGER = "mflux"

logger = logging.getLogger(SERVER_LOGGER)


class JsonFormatter(logging.Formatter):
    """Une ligne = un objet JSON, pour un superviseur qui lit stderr.

    Les événements de cycle de vie portent en plus un `event` et les champs de
    `fields`, posés par `extra=` à l'appel — le message humain reste inchangé à
    côté. Attention côté consommateur : tqdm écrit sur stderr sans passer par
    le logging (cf. `capture_stdout`), donc toute ligne qui ne parse pas en
    JSON doit être ignorée plutôt que traitée comme une erreur.
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
    # En mode JSON, on sort sur **stdout** et pas stderr : mflux affiche sa barre
    # de progression de débruitage avec tqdm (`Config.time_steps`), qui écrit sur
    # stderr des fragments terminés par `\r` sans retour à la ligne. Nos objets
    # JSON s'y colleraient — `{"ts": …}` précédé de `\r 0%| | 0/40` sur le même
    # segment — et un consommateur raisonnable les manquerait tous. tqdm n'offre
    # aucune variable d'environnement pour se taire, d'où la séparation des
    # canaux : stdout porte les événements structurés, stderr le texte humain et
    # les barres. `main()` coupe l'access log d'uvicorn en conséquence, sinon il
    # polluerait stdout à son tour.
    console = logging.StreamHandler(sys.stdout if json_lines else sys.stderr)
    console.setFormatter(
        JsonFormatter()
        if json_lines
        else logging.Formatter("[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )

    handlers: list[logging.Handler] = [console]
    if log_file:
        # `FileHandler` n'expanse pas `~` et ne crée pas les dossiers parents :
        # sans ça, un chemin type `~/Library/Logs/mflux-server/mflux.log`
        # échouerait en créant un dossier littéral `~`, ou lèverait un
        # FileNotFoundError au démarrage.
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
