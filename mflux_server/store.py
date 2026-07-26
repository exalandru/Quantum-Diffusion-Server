"""Stockage éphémère des images servies en `response_format="url"`.

L'API OpenAI renvoie par défaut une URL, pas du base64 : il faut donc
persister l'image quelque part. Les fichiers sont purgés au-delà de leur
TTL, au démarrage et après chaque écriture.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from mflux_server.logs import SERVER_LOGGER

logger = logging.getLogger(SERVER_LOGGER)


class ImageStore:
    def __init__(self, directory: Path, ttl_s: int = 3600):
        self.directory = Path(directory)
        self.ttl_s = ttl_s
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes) -> str:
        name = f"{uuid.uuid4().hex}.png"
        (self.directory / name).write_bytes(data)
        self.purge()
        return name

    def purge(self) -> int:
        """Supprime les images expirées. `ttl_s = 0` désactive la purge."""
        if not self.ttl_s:
            return 0
        cutoff = time.time() - self.ttl_s
        removed = 0
        for path in self.directory.glob("*.png"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:  # pragma: no cover - course avec une autre purge
                logger.debug("Purge impossible pour %s", path, exc_info=True)
        if removed:
            logger.info("%d image(s) expirée(s) supprimée(s)", removed)
        return removed
