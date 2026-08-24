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

#: MLX needs a Metal device. On a virtualised macOS CI runner there is none, and
#: `import mlx.core` hangs or crashes there (ml-explore/mlx#3148), which stalls
#: collection before a single test runs. `QDS_CI_NO_MLX=1` drops the files that
#: need MLX. The name deliberately carries neither the `QDS_SERVER_` nor the
#: `MFLUX_SERVER_` prefix: `_neutral_environment` strips those before every test,
#: and this flag has to survive — it is read here, at collection time.
NO_MLX = os.environ.get("QDS_CI_NO_MLX") == "1"

#: Whole files, not just their MLX imports: they also import `qds` submodules
#: that pull in MLX at module level.
collect_ignore = (
    ["test_anima.py", "test_upscale.py", "test_flux2_dev.py"] if NO_MLX else []
)


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
        self.upscales: list = []
        self.loaded_model: str | None = None
        self.loaded_upscaler: str | None = None
        self.shutdown_called = False
        self.unload_called = False
        #: Counts them, not just "did it happen": the point of the idle policy is
        #: that an n=3 request releases *once*, at the end.
        self.unload_count = 0
        #: Simulates a running generation, to exercise `/v1/cancel`.
        self.busy = False
        #: What `progress()` should report while busy. Reflects the last job
        #: handed to `generate`, so a test that asks "whose step is this?" can
        #: get a truthful answer -- and `pretend_busy_with` can make it somebody
        #: else's, which is the only way to witness that MCP does not forward
        #: another job's progress as its own.
        self._busy_with: dict = {"model": "z-image-turbo", "kind": "txt2img", "seed": 42,
                                 "step": 3, "total": 9}
        self.cancel_requested = False
        #: Whatever `/playground/api/preview` should serve right now.
        self.preview_bytes: bytes | None = None
        #: Rewrite jobs seen, and what `rewrite` should do with the next one.
        #: The list is what proves a rewrite did *not* happen, which is the
        #: assertion a passing-by-luck test would miss.
        self.rewrites: list = []
        self.rewrite_result: object = "an expanded prompt, rich in detail"
        self.loaded_rewriter: str | None = None

    async def generate(self, job: GenerationJob) -> bytes:
        self.jobs.append(job)
        self.loaded_model = f"{job.spec.key}:{job.kind}"
        # `spec.key`, not the public name: that is what the real engine writes
        # into its snapshot, and an attribution check compares against it.
        self._busy_with = {
            "model": job.spec.key,
            "kind": job.kind,
            "seed": job.seed,
            "step": self._busy_with.get("step", 3),
            "total": job.steps,
        }
        return tiny_png()

    async def upscale(self, job) -> bytes:
        self.upscales.append(job)
        self.loaded_upscaler = job.spec.key
        return tiny_png()

    async def rewrite(self, job) -> str:
        """Records the job and returns `rewrite_result`, or raises it.

        Assigning an exception rather than patching the method keeps the two
        outcomes a test needs -- a rewrite that works and one that does not --
        one line apart, which matters because the interesting property is that
        the *generation* survives the second.
        """
        self.rewrites.append(job)
        if isinstance(self.rewrite_result, Exception):
            raise self.rewrite_result
        return self.rewrite_result

    def memory_stats(self) -> dict:
        return {"active_gb": 0.0, "peak_gb": 0.0, "cache_gb": 0.0}

    def pretend_busy_with(
        self, model: str, *, seed: int, step: int, total: int, kind: str = "txt2img"
    ) -> None:
        """Report a job that is *not* the one under test.

        The real engine's snapshot is global, so "the engine is denoising" and
        "the engine is denoising *my* image" are different facts. Without a way
        to make them differ, a test cannot tell an implementation that checks
        from one that simply forwards whatever the engine last said.
        """
        self.busy = True
        self._busy_with = {"model": model, "kind": kind, "seed": seed, "step": step, "total": total}

    def progress(self) -> dict:
        busy = self._busy_with
        return {
            "state": "generating" if self.busy else "idle",
            "model": busy["model"] if self.busy else None,
            "kind": busy["kind"] if self.busy else None,
            "seed": busy["seed"] if self.busy else None,
            "step": busy["step"] if self.busy else 0,
            "total": busy["total"] if self.busy else 0,
            "preview_seq": 0,
            "elapsed_s": 1.5 if self.busy else None,
            "loaded_model": self.loaded_model,
            "upscaler": self.loaded_upscaler,
            "memory": self.memory_stats(),
        }

    def preview(self) -> bytes | None:
        return self.preview_bytes

    def request_cancel(self) -> bool:
        if not self.busy:
            return False
        self.cancel_requested = True
        return True

    async def unload(self) -> None:
        self.unload_called = True
        self.unload_count += 1
        self.loaded_model = None
        self.loaded_upscaler = None

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
                "playground_store": str(tmp_path / "playground"),
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
