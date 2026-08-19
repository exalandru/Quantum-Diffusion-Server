"""Engine tests: caching, serialization, unloading, callbacks.

No weights are loaded — `load_model` is replaced by a double. The mflux objects
actually used (`CallbackRegistry`, `StopImageGenerationException`) are the real
ones: pinning down their behaviour is exactly the point.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image

from qds import engine as engine_module
from qds.engine import GenerationJob, ModelEngine
from qds.errors import APIError
from qds.registry import BASE_SPECS_BY_KEY

#: What the double uses when no step count is passed, standing in for the step
#: count a sampler preset would supply.
_PRESET_FALLBACK_STEPS = 20


class FakeGenerated:
    def __init__(self):
        self.image = Image.new("RGB", (2, 2), "green")


class FakeModel:
    """Imitates only what the engine touches on an mflux model."""

    def __init__(self, key: str, kind: str):
        from mflux.callbacks.callback_registry import CallbackRegistry

        self.key = key
        self.kind = kind
        self.callbacks = CallbackRegistry()
        self.transformer = object()
        self.text_encoder = object()
        self.vae = object()
        self.prompt_cache: dict[str, object] = {}
        self.calls: list[dict] = []
        self.delay = 0.0
        #: Called after each step, with the step index. Lets a test trigger a
        #: cancellation at a deterministic moment.
        self.step_hook = None

    def generate_image(self, **kwargs):
        import time

        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        # The real model notifies its callbacks on every denoising step.
        # `num_inference_steps` may be absent: Ideogram 4 takes its step count from
        # the sampler preset, and its signature defaults the parameter to None.
        steps = kwargs.get("num_inference_steps") or _PRESET_FALLBACK_STEPS
        config = _FakeConfig(steps)
        for t in range(steps):
            for callback in self.callbacks.in_loop_callbacks():
                callback.call_in_loop(
                    t=t,
                    seed=kwargs["seed"],
                    prompt=kwargs["prompt"],
                    latents=None,
                    config=config,
                    time_steps=None,
                )
            if self.step_hook is not None:
                self.step_hook(t)
        return FakeGenerated()


class _FakeConfig:
    def __init__(self, steps: int):
        self.num_inference_steps = steps


@pytest.fixture
def loaded(monkeypatch):
    created: list[FakeModel] = []

    def fake_load_model(spec, *, kind="txt2img"):
        model = FakeModel(spec.key, kind)
        created.append(model)
        return model

    monkeypatch.setattr(engine_module, "load_model", fake_load_model)
    return created


def job(key: str = "flux2-klein", kind: str = "txt2img", **kwargs) -> GenerationJob:
    spec = BASE_SPECS_BY_KEY[key]
    defaults = dict(
        prompt="un renard",
        width=1024,
        height=1024,
        steps=spec.default_steps,
        seed=42,
    )
    defaults.update(kwargs)
    return GenerationJob(spec=spec, kind=kind, **defaults)


def test_model_stays_warm_between_generations(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())
        await eng.generate(job(seed=43))
        return eng

    eng = asyncio.run(scenario())
    assert len(loaded) == 1, "the weights were reloaded"
    assert eng.loaded_model == "flux2-klein:txt2img"
    eng.shutdown()


def test_switching_model_unloads_the_previous_one(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("flux2-klein"))
        first = loaded[0]
        await eng.generate(job("z-image-turbo"))
        return eng, first

    eng, first = asyncio.run(scenario())
    assert len(loaded) == 2
    assert eng.loaded_model == "z-image-turbo:txt2img"
    # The first model's submodules were indeed released.
    assert first.transformer is None
    assert first.text_encoder is None
    assert first.vae is None
    eng.shutdown()


def test_edit_variant_is_a_separate_load(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("flux2-klein", kind="txt2img"))
        await eng.generate(job("flux2-klein", kind="edit"))
        return eng

    eng = asyncio.run(scenario())
    assert [model.kind for model in loaded] == ["txt2img", "edit"]
    eng.shutdown()


def test_callback_is_registered_only_once(loaded):
    """`CallbackRegistry` has no unregister: registering per request would grow
    the list without bound."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        for seed in range(5):
            await eng.generate(job(seed=seed))
        return eng

    eng = asyncio.run(scenario())
    assert len(loaded[0].callbacks.in_loop_callbacks()) == 1
    eng.shutdown()


def test_generations_are_serialized(loaded):
    """Two concurrent requests must never overlap: on unified memory, two live
    models saturate the machine."""
    overlaps = 0
    running = 0

    async def scenario():
        nonlocal overlaps, running
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())  # loads the model
        original = loaded[0].generate_image

        def instrumented(**kwargs):
            nonlocal overlaps, running
            running += 1
            if running > 1:
                overlaps += 1
            try:
                return original(**kwargs)
            finally:
                running -= 1

        loaded[0].generate_image = instrumented
        loaded[0].delay = 0.02
        await asyncio.gather(*(eng.generate(job(seed=index)) for index in range(4)))
        return eng

    eng = asyncio.run(scenario())
    assert overlaps == 0
    eng.shutdown()


def test_qwen_prompt_cache_is_purged(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("qwen-image-2512"))
        model = loaded[0]
        model.prompt_cache.update({f"prompt-{index}": object() for index in range(50)})
        await eng.generate(job("qwen-image-2512", seed=7))
        return eng, model

    eng, model = asyncio.run(scenario())
    assert model.prompt_cache == {}
    eng.shutdown()


# ── Ideogram 4: the one family whose call shape differs ────────────────────


def test_ideogram_gets_its_preset_and_no_scheduler(loaded):
    """The kwargs actually handed to mflux, which is where this breaks silently.

    `Ideogram4.generate_image` has no `scheduler` parameter — passing ours would be
    a TypeError, not an ignored argument. And `preset` must go in even though the
    step count is left out, because the sampler also supplies `mu` and `std` to the
    noise schedule.
    """

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("ideogram-4", steps_from_preset=True))
        return eng, loaded[0].calls[0]

    eng, kwargs = asyncio.run(scenario())
    assert "scheduler" not in kwargs
    assert kwargs["preset"] == "V4_DEFAULT_20"
    # Absent on purpose: passing it would flatten the preset's per-step guidance
    # schedule into a constant.
    assert "num_inference_steps" not in kwargs
    eng.shutdown()


def test_an_explicit_step_count_reaches_ideogram(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        # steps_from_preset stays False: the client asked for a number.
        await eng.generate(job("ideogram-4", steps=12))
        return eng, loaded[0].calls[0]

    eng, kwargs = asyncio.run(scenario())
    assert kwargs["num_inference_steps"] == 12
    # Still passed: the preset governs the noise schedule regardless.
    assert kwargs["preset"] == "V4_DEFAULT_20"
    eng.shutdown()


def test_every_other_family_still_gets_its_scheduler(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("z-image-turbo"))
        return eng, loaded[0].calls[0]

    eng, kwargs = asyncio.run(scenario())
    assert kwargs["scheduler"] == "linear"
    assert kwargs["num_inference_steps"] == 9
    assert "preset" not in kwargs
    eng.shutdown()


def test_timeout_interrupts_the_denoising_loop(loaded):
    """Only the in-loop callback can stop a generation: neither asyncio nor a
    thread can cancel an in-flight MLX operation."""

    async def scenario():
        eng = ModelEngine(request_timeout_s=0.05, progress_log_every=0)
        await eng.generate(job())  # loads the model, within the deadline
        loaded[0].delay = 0.2  # the next step will land past the deadline
        with pytest.raises(APIError) as excinfo:
            await eng.generate(job(seed=1))
        return eng, excinfo.value

    eng, error = asyncio.run(scenario())
    assert error.status_code == 504
    assert error.code == "timeout"
    eng.shutdown()


def test_mflux_errors_are_translated(loaded, monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    def exploding_load(spec, *, kind="txt2img"):
        raise ModelConfigError("base model introuvable")

    monkeypatch.setattr(engine_module, "load_model", exploding_load)

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        with pytest.raises(APIError) as excinfo:
            await eng.generate(job())
        eng.shutdown()
        return excinfo.value

    error = asyncio.run(scenario())
    assert error.status_code == 400
    assert error.param == "model"


def test_png_is_produced_in_memory(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        data = await eng.generate(job())
        eng.shutdown()
        return data

    data = asyncio.run(scenario())
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_arguments_passed_to_mflux(loaded):
    """Checks the per-family wiring: negative_prompt and guidance are only
    forwarded when the model accepts them."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("flux2-klein", negative_prompt="flou", guidance=None))
        flux_kwargs = loaded[0].calls[0]
        await eng.generate(job("z-image", negative_prompt="flou", guidance=6.0))
        z_kwargs = loaded[1].calls[0]
        await eng.generate(job("flux2-klein", kind="edit", image_path="/tmp/in.png"))
        edit_kwargs = loaded[2].calls[0]
        await eng.generate(job("z-image", image_path="/tmp/in.png", image_strength=0.6))
        img2img_kwargs = loaded[3].calls[0]
        eng.shutdown()
        return flux_kwargs, z_kwargs, edit_kwargs, img2img_kwargs

    flux_kwargs, z_kwargs, edit_kwargs, img2img_kwargs = asyncio.run(scenario())

    # FLUX.2 Klein has no negative_prompt parameter and its guidance is fixed.
    assert "negative_prompt" not in flux_kwargs
    assert flux_kwargs["guidance"] == 1.0
    assert flux_kwargs["scheduler"] == "flow_match_euler_discrete"

    assert z_kwargs["negative_prompt"] == "flou"
    assert z_kwargs["guidance"] == 6.0

    # Editing: a list of conditioning images, no image_strength.
    assert edit_kwargs["image_paths"] == ["/tmp/in.png"]
    assert "image_strength" not in edit_kwargs

    # img2img: a noised starting latent.
    assert img2img_kwargs["image_path"] == "/tmp/in.png"
    assert img2img_kwargs["image_strength"] == 0.6


# ── Progress and cancellation ──────────────────────────────────────────────


def test_engine_is_idle_when_at_rest():
    eng = ModelEngine(progress_log_every=0)
    snapshot = eng.progress()
    assert snapshot["state"] == "idle"
    assert snapshot["loaded_model"] is None
    assert (snapshot["step"], snapshot["total"]) == (0, 0)
    assert snapshot["elapsed_s"] is None
    eng.shutdown()


def test_progress_tracks_the_steps(loaded):
    """The snapshot is written from the worker thread and read without a lock."""
    seen: list[tuple[str, int, int]] = []

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())  # loads the model
        loaded[0].step_hook = lambda _: seen.append(
            (eng.progress()["state"], eng.progress()["step"], eng.progress()["total"])
        )
        await eng.generate(job(seed=1))
        return eng

    eng = asyncio.run(scenario())
    # flux2-klein is distilled: 4 steps.
    assert seen == [("generating", 1, 4), ("generating", 2, 4), ("generating", 3, 4), ("generating", 4, 4)]
    # Back to idle, but the model stays warm.
    assert eng.progress()["state"] == "idle"
    assert eng.loaded_model == "flux2-klein:txt2img"
    eng.shutdown()


def test_cancelling_at_rest_does_nothing():
    eng = ModelEngine(progress_log_every=0)
    assert eng.request_cancel() is False
    eng.shutdown()


def test_cancellation_interrupts_the_loop(loaded):
    """Same path as the timeout: the in-loop callback is the only handle."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("z-image"))  # 50 steps, loaded
        # Request cancellation from the first step; it takes effect on the next
        # one, as in production.
        loaded[0].step_hook = lambda t: eng.request_cancel() if t == 0 else None
        with pytest.raises(APIError) as excinfo:
            await eng.generate(job("z-image", seed=1))
        return eng, excinfo.value

    eng, error = asyncio.run(scenario())
    assert error.status_code == 499
    assert error.code == "generation_stopped"
    # The engine stays usable and the model warm.
    assert eng.progress()["state"] == "idle"
    assert eng.loaded_model == "z-image:txt2img"
    eng.shutdown()


def test_engine_stays_usable_after_a_cancellation(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("z-image"))
        loaded[0].step_hook = lambda t: eng.request_cancel() if t == 0 else None
        with pytest.raises(APIError):
            await eng.generate(job("z-image", seed=1))
        # The flag must be reset by `arm()`, otherwise the next generation would
        # be cancelled too.
        loaded[0].step_hook = None
        data = await eng.generate(job("z-image", seed=2))
        eng.shutdown()
        return data

    data = asyncio.run(scenario())
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_unload_releases_the_model(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())
        model = loaded[0]
        await eng.unload()
        eng.shutdown()
        return eng, model

    eng, model = asyncio.run(scenario())
    assert eng.loaded_model is None
    assert eng.progress()["loaded_model"] is None
    assert model.transformer is None
