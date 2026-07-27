"""Shared fixtures.

No test loads weights: the engine is replaced by a double that returns a tiny
PNG. The tests validate the HTTP layer and the registry, not inference — that is
verified by hand (see the README).
"""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mflux_server.app import create_app
from mflux_server.engine import GenerationJob
from mflux_server.settings import Settings


def wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll a condition that the app's own event loop has to satisfy.

    `TestClient` drives the app from another thread, so sleeping here lets it
    progress. Polling rather than awaiting: the automatic release happens on a
    task the test does not hold.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def tiny_png(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    """Stands in for `ModelEngine`: records jobs, loads nothing."""

    def __init__(self) -> None:
        self.jobs: list[GenerationJob] = []
        self.loaded_model: str | None = None
        self.shutdown_called = False
        self.unload_called = False
        #: Counts them, not just "did it happen": the point of the idle policy is
        #: that an n=3 request releases *once*, at the end.
        self.unload_count = 0
        #: Simulates a running generation, to exercise `/v1/cancel`.
        self.busy = False
        self.cancel_requested = False

    async def generate(self, job: GenerationJob) -> bytes:
        self.jobs.append(job)
        self.loaded_model = f"{job.spec.key}:{job.kind}"
        return tiny_png()

    def memory_stats(self) -> dict:
        return {"active_gb": 0.0, "peak_gb": 0.0, "cache_gb": 0.0}

    def progress(self) -> dict:
        return {
            "state": "generating" if self.busy else "idle",
            "model": "z-image-turbo" if self.busy else None,
            "kind": "txt2img" if self.busy else None,
            "seed": 42 if self.busy else None,
            "step": 3 if self.busy else 0,
            "total": 9 if self.busy else 0,
            "elapsed_s": 1.5 if self.busy else None,
            "loaded_model": self.loaded_model,
            "memory": self.memory_stats(),
        }

    def request_cancel(self) -> bool:
        if not self.busy:
            return False
        self.cancel_requested = True
        return True

    async def unload(self) -> None:
        self.unload_called = True
        self.unload_count += 1
        self.loaded_model = None

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings.model_validate(
        {
            "server": {
                "image_store": str(tmp_path / "images"),
                "log_file": None,
                "progress_log_every": 0,
                "max_n": 4,
            },
            "default_model": "flux2-klein",
            "models": {"qwen-image": {"enable_edit": True}},
        }
    )


@pytest.fixture
def client(settings, engine):
    with TestClient(create_app(settings, engine)) as test_client:
        yield test_client
