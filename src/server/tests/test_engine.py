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


# ── Step previews (playground only) ────────────────────────────────────────


def test_preview_rendered_every_n_steps(loaded, monkeypatch):
    """One frame every `preview_every` steps, and none for the last step.

    The decode itself is replaced: what matters here is the cadence, the slot and
    the counter the client watches, not mflux's VAE.
    """
    monkeypatch.setattr(engine_module, "_render_preview", lambda **_: b"jpeg")
    seen: list[tuple[int, bytes | None]] = []

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        # The warm-up run loads the model and asks for no previews, so the
        # engine-lifetime counter starts this run at zero.
        await eng.generate(job(steps=9))
        loaded[0].step_hook = lambda _: seen.append((eng.progress()["preview_seq"], eng.preview()))
        await eng.generate(job(steps=9, preview_every=2, seed=1))
        return eng

    eng = asyncio.run(scenario())
    # Steps 1..9: the counter advances at 2, 4, 6 and 8. Not at 9 — the finished
    # image is moments away through the normal path.
    assert [seq for seq, _ in seen] == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert [payload for _, payload in seen[1:]] == [b"jpeg"] * 8
    # Nothing is running any more: the endpoint must 404 and the bar must be bare.
    assert eng.preview() is None
    assert eng.progress()["preview_seq"] == 0
    eng.shutdown()


def test_preview_counter_keeps_climbing_across_runs(loaded, monkeypatch):
    """The counter doubles as a cache-buster, so it must never repeat a value."""
    monkeypatch.setattr(engine_module, "_render_preview", lambda **_: b"jpeg")
    seen: list[int] = []

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job(steps=4))  # loads the model, no previews
        loaded[0].step_hook = lambda _: seen.append(eng.progress()["preview_seq"])
        await eng.generate(job(steps=4, preview_every=2, seed=1))
        await eng.generate(job(steps=4, preview_every=2, seed=2))
        return eng

    eng = asyncio.run(scenario())
    assert seen == [0, 1, 1, 1, 0, 2, 2, 2]
    eng.shutdown()


def test_preview_failure_disables_previews_for_the_run(loaded, monkeypatch):
    """A broken decode is not a broken generation."""
    calls = 0

    def exploding(**_):
        nonlocal calls
        calls += 1
        raise RuntimeError("no vae here")

    monkeypatch.setattr(engine_module, "_render_preview", exploding)
    seen: list[int] = []

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job(steps=9))  # loads the model, no previews
        loaded[0].step_hook = lambda _: seen.append(eng.progress()["preview_seq"])
        data = await eng.generate(job(steps=9, preview_every=2, seed=1))
        return eng, data

    eng, data = asyncio.run(scenario())
    assert Image.open(io.BytesIO(data)).format == "PNG"
    assert seen == [0] * 9
    # Tried once, at step 2, then given up on for the rest of the run.
    assert calls == 1
    eng.shutdown()


def test_no_preview_without_opt_in(loaded, monkeypatch):
    """`/v1` never asks for previews, so nothing must be decoded for it."""
    calls: list[dict] = []
    monkeypatch.setattr(engine_module, "_render_preview", lambda **kwargs: calls.append(kwargs) or b"jpeg")

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job(steps=9))
        return eng

    eng = asyncio.run(scenario())
    assert calls == []
    assert eng.preview() is None
    eng.shutdown()


def test_preview_is_skipped_for_a_family_without_a_latent_creator(loaded, monkeypatch):
    """Fail-closed: an unmapped family loses previews, not its generation."""
    monkeypatch.setattr(engine_module, "latent_creator_for", lambda _family: None)
    calls: list[dict] = []
    monkeypatch.setattr(engine_module, "_render_preview", lambda **kwargs: calls.append(kwargs) or b"jpeg")

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        data = await eng.generate(job(steps=9, preview_every=2))
        return eng, data

    eng, data = asyncio.run(scenario())
    assert Image.open(io.BytesIO(data)).format == "PNG"
    assert calls == []
    eng.shutdown()


class _StubCreator:
    """Records how the renderer calls an unpacker, and hands the latents back."""

    calls: list[dict] = []

    @staticmethod
    def unpack_latents(**kwargs):
        _StubCreator.calls.append({k: v for k, v in kwargs.items() if k != "latents"})
        return kwargs["latents"]


class _PackedVae:
    """The mflux VAEs that expose the packed entry point; the renderer must prefer it."""

    def __init__(self):
        self.used = []

    def decode_packed_latents(self, latents):
        self.used.append("packed")
        return latents

    def decode(self, latents):  # pragma: no cover - must never be reached
        self.used.append("plain")
        return latents


class _PlainVae:
    def __init__(self):
        self.used = []

    def decode(self, latents):
        self.used.append("plain")
        return latents


class _VaeOnlyModel:
    bits = 8

    def __init__(self, vae):
        self.vae = vae


@pytest.mark.parametrize("vae_class", [_PackedVae, _PlainVae])
def test_the_real_renderer_produces_a_bounded_jpeg(vae_class):
    """The one path the other preview tests monkeypatch away.

    No weights: the latent unpacker and the VAE are stubs, but `mflux`'s own
    `Config` and `ImageUtil` are the real ones, so a wrong keyword, a wrong VAE
    entry point or a wrong `to_image` argument fails here instead of turning into
    a debug line and silently missing previews at runtime.
    """
    import mlx.core as mx
    from mflux.models.common.config.config import Config

    from qds.registry import model_config_for

    _StubCreator.calls.clear()
    config = Config(
        model_config=model_config_for(BASE_SPECS_BY_KEY["z-image-turbo"]),
        num_inference_steps=6,
        height=704,
        width=1280,
        guidance=1.0,
    )
    vae = vae_class()
    data = engine_module._render_preview(
        model=_VaeOnlyModel(vae),
        creator=_StubCreator,
        latents=mx.zeros((1, 3, 704, 1280)),
        config=config,
        seed=7,
        prompt="un renard",
    )

    # The unpacker is given the run's dimensions, by keyword: every mflux creator
    # takes `latents`, `height`, `width` and nothing else.
    assert _StubCreator.calls == [{"height": 704, "width": 1280}]
    assert vae.used == ["packed" if vae_class is _PackedVae else "plain"]
    image = Image.open(io.BytesIO(data))
    assert image.format == "JPEG"
    assert image.mode == "RGB"
    # Scaled down to the feed's track, aspect ratio kept.
    assert max(image.size) == engine_module._PREVIEW_MAX_PX
    assert image.size == (512, 282)


# ── The second slot: upscaling ─────────────────────────────────────────────


class FakeUpscaler:
    """Nearest x4, with the hooks a test needs to time a cancellation."""

    def __init__(self, key: str):
        self.key = key
        self.tiles = 0
        self.delay = 0.0
        self.tile_hook = None

    def __call__(self, x):
        import time

        import mlx.core as mx

        self.tiles += 1
        if self.delay:
            time.sleep(self.delay)
        if self.tile_hook is not None:
            self.tile_hook(self.tiles)
        return mx.repeat(mx.repeat(x, 4, axis=1), 4, axis=2)


@pytest.fixture
def upscalers(monkeypatch):
    """Replace weight loading, and record every instance built."""
    from qds.upscale import weights as weights_module

    built: list[FakeUpscaler] = []

    def fake_load_upscaler(spec, **_kwargs):
        model = FakeUpscaler(spec.key)
        built.append(model)
        return model

    monkeypatch.setattr(weights_module, "load_upscaler", fake_load_upscaler)
    return built


def upscale_job(tmp_path, *, key: str = "realesrgan-x4plus", outscale: int = 4, size=(8, 8)):
    from qds.engine import UpscaleJob
    from qds.upscale import by_key

    path = tmp_path / f"{key}-{size[0]}x{size[1]}.png"
    Image.new("RGB", size, "blue").save(path)
    return UpscaleJob(
        spec=by_key(key),
        image_path=path,
        target=(size[0] * outscale, size[1] * outscale),
    )


def test_an_upscale_does_not_disturb_the_diffusion_model(loaded, upscalers, tmp_path):
    """I1, the reason the second slot exists at all.

    If `_ensure_upscaler` reached for `_unload_sync` the way `_ensure_model`
    does, this would build the weights twice and the user would pay a reload
    for every enlarged image.
    """

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())
        warm = eng.loaded_model
        await eng.upscale(upscale_job(tmp_path))
        assert eng.loaded_model == warm, "the upscale evicted the diffusion model"
        await eng.generate(job(seed=43))
        return eng

    eng = asyncio.run(scenario())
    assert len(loaded) == 1, "the diffusion weights were reloaded around the upscale"
    assert len(upscalers) == 1
    assert eng.loaded_model == "flux2-klein:txt2img"
    assert eng.loaded_upscaler == "realesrgan-x4plus"
    eng.shutdown()


def test_the_upscaler_stays_warm_between_upscales(upscalers, tmp_path):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.upscale(upscale_job(tmp_path))
        await eng.upscale(upscale_job(tmp_path, outscale=2))
        return eng

    eng = asyncio.run(scenario())
    assert len(upscalers) == 1
    eng.shutdown()


def test_switching_upscaler_releases_the_previous_one(upscalers, tmp_path):
    """One upscaler, as with diffusion models -- but only among upscalers."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.upscale(upscale_job(tmp_path))
        await eng.upscale(upscale_job(tmp_path, key="realesrgan-x4plus-anime"))
        return eng

    eng = asyncio.run(scenario())
    assert len(upscalers) == 2
    assert eng.loaded_upscaler == "realesrgan-x4plus-anime"
    eng.shutdown()


def test_upscales_and_generations_are_serialized(loaded, upscalers, tmp_path):
    """I2: one MLX job at a time, across both kinds of work."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())  # load once, so the delay is the generation's
        loaded[0].delay = 0.15
        order: list[str] = []

        async def generation():
            await eng.generate(job(seed=44))
            order.append("generate")

        async def upscale():
            await asyncio.sleep(0.02)
            await eng.upscale(upscale_job(tmp_path))
            order.append("upscale")

        await asyncio.gather(generation(), upscale())
        return eng, order

    eng, order = asyncio.run(scenario())
    # The upscale started second and could only run once the lock was free.
    assert order == ["generate", "upscale"]
    eng.shutdown()


def test_progress_reports_the_upscale_and_its_tiles(upscalers, tmp_path):
    """I3. `state` is its own value, not `generating` with a different model."""
    seen: list[dict] = []

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        # Prime the slot so the observed run is the upscale, not the load.
        await eng.upscale(upscale_job(tmp_path, size=(4, 4)))
        upscalers[0].delay = 0.05
        upscalers[0].tile_hook = lambda _n: seen.append(eng.progress())
        # 3x3 tiles at tile=192 needs a source larger than one tile.
        await eng.upscale(upscale_job(tmp_path, size=(400, 400)))
        return eng

    eng = asyncio.run(scenario())
    assert seen, "no tile was observed"
    assert {snapshot["state"] for snapshot in seen} == {"upscaling"}
    assert {snapshot["kind"] for snapshot in seen} == {"upscale"}
    assert {snapshot["model"] for snapshot in seen} == {"realesrgan-x4plus"}
    assert seen[-1]["total"] == 9  # ceil(400/192) squared
    assert seen[0]["seed"] is None
    assert eng.progress()["state"] == "idle"
    eng.shutdown()


def test_an_upscale_can_be_cancelled_between_tiles(upscalers, tmp_path):
    """I4. `request_cancel` answers False today; this is what extends it."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.upscale(upscale_job(tmp_path, size=(4, 4)))

        armed: list[bool] = []
        upscalers[0].tile_hook = lambda _n: armed.append(eng.request_cancel())

        with pytest.raises(APIError) as caught:
            await eng.upscale(upscale_job(tmp_path, size=(400, 400)))
        return eng, caught.value, armed

    eng, error, armed = asyncio.run(scenario())
    assert armed[0] is True, "request_cancel refused a running upscale"
    assert error.code == "generation_stopped"
    # And the engine is usable afterwards: the cancel flag was re-armed clean
    # by the next `arm()`, and the upscaler is still resident.
    assert eng.progress()["state"] == "idle"
    upscalers[0].tile_hook = None
    asyncio.run(eng.upscale(upscale_job(tmp_path, size=(4, 4))))
    assert len(upscalers) == 1, "the cancelled run cost the resident upscaler"
    eng.shutdown()


def test_request_cancel_still_refuses_when_nothing_runs(upscalers):
    eng = ModelEngine(progress_log_every=0)
    assert eng.request_cancel() is False
    eng.shutdown()


def test_unload_releases_both_slots(loaded, upscalers, tmp_path):
    """I5, first half: "Free memory" must mean all of it."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())
        await eng.upscale(upscale_job(tmp_path))
        assert eng.loaded_model is not None and eng.loaded_upscaler is not None
        await eng.unload()
        return eng

    eng = asyncio.run(scenario())
    assert eng.loaded_model is None
    assert eng.loaded_upscaler is None
    eng.shutdown()


def test_unloading_a_diffusion_model_leaves_the_upscaler_alone(loaded, upscalers, tmp_path):
    """I5, second half: the test that resists a future "simplification".

    `_ensure_model` calls `_unload_sync` on every model switch. If that ever
    grew to release the upscaler too, every alternation between generating and
    upscaling would reload 33 MB, and nothing else would notice.
    """

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.upscale(upscale_job(tmp_path))
        await eng.generate(job())
        await eng.generate(job(key="z-image-turbo"))  # forces a model switch
        return eng

    eng = asyncio.run(scenario())
    assert len(loaded) == 2, "the model switch did not happen"
    assert len(upscalers) == 1, "_unload_sync took the upscaler with it"
    assert eng.loaded_upscaler == "realesrgan-x4plus"
    eng.shutdown()


def test_shutdown_releases_both_slots(loaded, upscalers, tmp_path):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())
        await eng.upscale(upscale_job(tmp_path))
        return eng

    eng = asyncio.run(scenario())
    eng.shutdown()
    assert eng.loaded_model is None
    assert eng.loaded_upscaler is None


@pytest.mark.parametrize("outscale", [2, 4])
def test_upscale_returns_a_png_at_the_requested_scale(upscalers, tmp_path, outscale):
    data = asyncio.run(
        ModelEngine(progress_log_every=0).upscale(
            upscale_job(tmp_path, outscale=outscale, size=(8, 12))
        )
    )
    with Image.open(io.BytesIO(data)) as out:
        assert out.size == (8 * outscale, 12 * outscale)
