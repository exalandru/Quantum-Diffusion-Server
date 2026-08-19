"""Shared fixtures.

No test loads weights: the engine is replaced by a double that returns a tiny
PNG. The tests validate the HTTP layer and the registry, not inference — that is
verified by hand (see the README).
"""

from __future__ import annotations

import io
import os
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from qds.app import create_app
from qds.engine import GenerationJob
from qds.env import ENV_PREFIX, LEGACY_ENV_PREFIX
from qds.settings import Settings


@pytest.fixture(autouse=True)
def _neutral_environment(monkeypatch):
    """Start every test from an environment that names no configuration.

    Both prefixes, and that is the whole point: a test that clears only
    `QDS_SERVER_CONFIG` is still exposed to a `MFLUX_SERVER_CONFIG` left in the
    developer's shell, because the deprecated spelling is deliberately still
    read. The result was a suite whose greenness depended on who ran it — and,
    worse, tests that silently read the real `server-config.json` instead of
    their `tmp_path` one without asserting on the difference.
    """
    for name in list(os.environ):
        if name.startswith((ENV_PREFIX, LEGACY_ENV_PREFIX)):
            monkeypatch.delenv(name, raising=False)


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
            "models": {"qwen-image-2512": {"enable_edit": True}},
        }
    )


def make_client(app) -> TestClient:
    """A `TestClient` dialling a host the server answers to.

    `TestClient` defaults to `http://testserver`, and the server refuses a
    `Host` header it does not recognise — that guard is what closes DNS
    rebinding against a keyless loopback install. Adding "testserver" to the
    allowlist would weaken the shipped server to spare the tests, so the tests
    dial a loopback address instead, like a real client does.
    """
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def client(settings, engine):
    with make_client(create_app(settings, engine)) as test_client:
        yield test_client
