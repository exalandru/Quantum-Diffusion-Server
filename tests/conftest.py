"""Fixtures partagées.

Aucun test ne charge de poids : le moteur est remplacé par un double qui
renvoie un PNG minuscule. Les tests valident la couche HTTP et le registre,
pas l'inférence — celle-ci se vérifie à la main (voir le README).
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mflux_server.app import create_app
from mflux_server.engine import GenerationJob
from mflux_server.settings import Settings


def tiny_png(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    """Remplace `ModelEngine` : enregistre les jobs, ne charge rien."""

    def __init__(self) -> None:
        self.jobs: list[GenerationJob] = []
        self.loaded_model: str | None = None
        self.shutdown_called = False
        self.unload_called = False
        #: Simule une génération en cours, pour tester `/v1/cancel`.
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
