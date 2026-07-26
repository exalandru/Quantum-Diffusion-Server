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

    async def generate(self, job: GenerationJob) -> bytes:
        self.jobs.append(job)
        self.loaded_model = f"{job.spec.key}:{job.kind}"
        return tiny_png()

    def memory_stats(self) -> dict:
        return {"active_gb": 0.0, "peak_gb": 0.0, "cache_gb": 0.0}

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
