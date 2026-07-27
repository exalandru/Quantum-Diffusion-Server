"""Ephemeral storage for images served with `response_format="url"`.

The OpenAI API returns a URL by default, not base64, so the image has to be
persisted somewhere. Files are purged once past their TTL, at startup and after
every write.
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
        """Delete expired images. `ttl_s = 0` disables purging."""
        if not self.ttl_s:
            return 0
        cutoff = time.time() - self.ttl_s
        removed = 0
        for path in self.directory.glob("*.png"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:  # pragma: no cover - race with a concurrent purge
                logger.debug("Could not purge %s", path, exc_info=True)
        if removed:
            logger.info("Removed %d expired image(s)", removed)
        return removed
