"""Automatic release of the warm model.

The policy is tested on its own, with a stub engine: no weights, no HTTP, and
delays of a few tens of milliseconds. What matters here is *when* it fires, which
is exactly what a fake engine can pin down.
"""

from __future__ import annotations

import asyncio

from mflux_server.idle import IdleUnloader
from tests.conftest import wait_until


class StubEngine:
    def __init__(self) -> None:
        self.loaded_model: str | None = "z-image-turbo:txt2img"
        self.unload_count = 0

    def memory_stats(self) -> dict[str, float]:
        return {"active_gb": 0.0 if self.loaded_model is None else 9.0}

    async def unload(self) -> None:
        self.unload_count += 1
        self.loaded_model = None


def test_the_model_is_released_after_the_delay():
    async def scenario():
        engine = StubEngine()
        unloader = IdleUnloader(engine, 0.05)
        with unloader:
            pass
        # Before the deadline: still warm. This half of the assertion is what
        # distinguishes a delay from an immediate release.
        await asyncio.sleep(0.02)
        assert engine.loaded_model is not None
        await asyncio.sleep(0.08)
        return engine

    engine = asyncio.run(scenario())
    assert engine.unload_count == 1
    assert engine.loaded_model is None


def test_a_new_request_cancels_the_countdown():
    async def scenario():
        engine = StubEngine()
        unloader = IdleUnloader(engine, 0.05)
        with unloader:
            pass
        await asyncio.sleep(0.02)
        # A second request arrives before the deadline: the model must stay warm,
        # which is the whole point of a delay rather than an immediate release.
        with unloader:
            await asyncio.sleep(0.06)
        assert engine.unload_count == 0
        await asyncio.sleep(0.08)
        return engine

    engine = asyncio.run(scenario())
    assert engine.unload_count == 1


def test_overlapping_requests_hold_the_model():
    """The reason for a counter rather than a flag.

    With a flag, the first request leaving would re-arm the countdown while the
    second is still generating — and with a delay of 0 it would release the model
    between two of its images.
    """

    async def scenario():
        engine = StubEngine()
        unloader = IdleUnloader(engine, 0)
        outer = unloader.__enter__()
        inner = unloader.__enter__()
        outer.__exit__()
        # Zero delay, so one turn of the loop is enough for a countdown to fire.
        await asyncio.sleep(0)
        assert engine.unload_count == 0, "released while a request was in flight"
        inner.__exit__()
        await asyncio.sleep(0.02)
        return engine

    engine = asyncio.run(scenario())
    assert engine.unload_count == 1


def test_a_zero_delay_releases_as_soon_as_the_request_ends():
    async def scenario():
        engine = StubEngine()
        with IdleUnloader(engine, 0):
            pass
        await asyncio.sleep(0.02)
        return engine

    assert asyncio.run(scenario()).loaded_model is None


def test_no_delay_means_never():
    async def scenario():
        engine = StubEngine()
        unloader = IdleUnloader(engine, None)
        assert unloader.enabled is False
        with unloader:
            pass
        await asyncio.sleep(0.05)
        # No task was even created: the default path costs nothing.
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        return engine

    engine = asyncio.run(scenario())
    assert engine.unload_count == 0
    assert engine.loaded_model is not None


def test_cancel_leaves_no_pending_task():
    """What the shutdown hook is for: no "Task was destroyed but it is pending"."""

    async def scenario():
        unloader = IdleUnloader(StubEngine(), 30)
        with unloader:
            pass
        unloader.cancel()
        # One turn so the cancellation is actually delivered.
        await asyncio.sleep(0)
        return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    assert asyncio.run(scenario()) == []


def test_a_real_engine_loads_once_for_a_multi_image_request(monkeypatch, tmp_path):
    """The claim the whole design rests on, against the real engine.

    The stub tests above count releases; this one counts *loads*. With the
    countdown armed inside `ModelEngine.generate()` — which takes its lock per
    image — a delay of 0 would release between the two images of an `n=2` request
    and reload for the second. Here it must load once and release once.
    """
    from fastapi.testclient import TestClient

    from mflux_server import engine as engine_module
    from mflux_server.app import create_app
    from mflux_server.engine import ModelEngine
    from mflux_server.settings import Settings
    from tests.test_engine import FakeModel

    loads: list[str] = []

    def fake_load_model(spec, *, kind="txt2img"):
        loads.append(spec.key)
        return FakeModel(spec.key, kind)

    monkeypatch.setattr(engine_module, "load_model", fake_load_model)
    settings = Settings.model_validate(
        {
            "server": {
                "image_store": str(tmp_path / "images"),
                "log_file": None,
                "progress_log_every": 0,
                "idle_unload_s": 0,
            }
        }
    )
    engine = ModelEngine(progress_log_every=0)
    with TestClient(create_app(settings, engine)) as client:
        response = client.post(
            "/v1/images/generations",
            json={"prompt": "un renard", "n": 2, "model": "z-image-turbo"},
        )
        assert response.status_code == 200
        assert len(loads) == 1, "the model was reloaded between the two images"
        assert wait_until(lambda: engine.loaded_model is None), "never released"


def test_releasing_an_empty_engine_is_a_no_op():
    async def scenario():
        engine = StubEngine()
        engine.loaded_model = None
        with IdleUnloader(engine, 0):
            pass
        await asyncio.sleep(0.02)
        return engine

    # Nothing to release: `unload()` is not even called.
    assert asyncio.run(scenario()).unload_count == 0
