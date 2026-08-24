"""The SD 3.5 generation loop, and the schedule it walks.

The centre of this file is `test_the_sigma_schedule_is_diffusers_own`. SD 3.5's
scheduler is the one part of the port where the obvious implementation — mflux's
`set_mu(log(shift))`, which `qds/anima/` uses correctly for its own model — silently
produces a different schedule, and a different schedule is a different image with no
error anywhere. So it is compared against **diffusers' real scheduler object**,
element by element, rather than against a formula re-derived here: a re-derivation
would agree with the implementation by construction and could not fail for the case
that matters.

`diffusers` is a real dependency of this environment (mflux pulls it in), so the
reference is the library, not a transcription of it.

The rest of the file checks the loop's contract with the engine — the argument names
`engine._generate_kwargs` sends, the callback sequence, the prompt cache, and when the
unconditional branch exists — using a stand-in transformer rather than 8.1B real
parameters. What those tests cannot establish is image quality; that is Step 8's
reference comparison on real weights.
"""

from __future__ import annotations

import inspect
import math

import mlx.core as mx
import numpy as np
import pytest

from qds.sd35 import config as sd35_config
from qds.sd35.model import SD35
from qds.sd35.scheduler import SD35FlowMatchScheduler, static_shift


class _Config:
    """The two fields `_compute_timesteps_and_sigmas` reads off `Config`."""

    def __init__(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        self.model_config = None


def _ours(num_steps: int) -> SD35FlowMatchScheduler:
    scheduler = SD35FlowMatchScheduler.__new__(SD35FlowMatchScheduler)
    scheduler.config = _Config(num_steps)
    scheduler.num_train_timesteps = 1000
    scheduler.shift_terminal = 0.02
    scheduler._sigmas, scheduler._timesteps = scheduler._compute_timesteps_and_sigmas()
    return scheduler


@pytest.mark.parametrize("num_steps", (1, 4, 20, 28, 50))
def test_the_sigma_schedule_is_diffusers_own(num_steps):
    """Against `FlowMatchEulerDiscreteScheduler(shift=3.0)`, the object, not a formula."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    reference = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=sd35_config.NUM_TRAIN_TIMESTEPS,
        shift=sd35_config.SIGMA_SHIFT,
    )
    reference.set_timesteps(num_inference_steps=num_steps)

    ours = _ours(num_steps)
    np.testing.assert_allclose(
        np.array(ours.sigmas.tolist()), reference.sigmas.numpy(), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(ours.timesteps.tolist()), reference.timesteps.numpy(), rtol=0, atol=1e-3
    )


def test_mfluxs_own_shift_helper_is_not_this_schedule():
    """The falsified premise, kept as a test so it cannot be quietly reintroduced.

    `set_mu(log(3.0))` shifts identically but spaces differently: it ends the linspace
    at `1/num_steps` instead of at the shifted smallest training sigma. At the four
    steps Large Turbo is distilled for, that leaves the last sigma at 0.5 rather than
    0.0089 — the loop would stop halfway through denoising.
    """
    from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
        FlowMatchEulerDiscreteScheduler as MfluxScheduler,
    )

    mflux_scheduler = MfluxScheduler.__new__(MfluxScheduler)
    mflux_scheduler.config = _Config(4)
    mflux_scheduler.num_train_timesteps = 1000
    mflux_scheduler.shift_terminal = 0.02
    mflux_scheduler.set_mu(math.log(sd35_config.SIGMA_SHIFT))

    theirs = np.array(mflux_scheduler.sigmas.tolist())
    ours = np.array(_ours(4).sigmas.tolist())
    assert theirs[-2] == pytest.approx(0.5, abs=1e-6)
    assert ours[-2] == pytest.approx(0.00893, abs=1e-4)
    assert np.abs(theirs - ours).max() > 0.4

    # And calling it on this scheduler is refused rather than silently accepted.
    with pytest.raises(NotImplementedError, match="fixed shift"):
        _ours(4).set_mu(1.0)
    with pytest.raises(NotImplementedError, match="image sequence length"):
        _ours(4).set_image_seq_len(4096)


def test_the_shift_is_the_published_one_and_applied_the_published_way():
    published = {"shift": 3.0, "num_train_timesteps": 1000}
    assert sd35_config.SIGMA_SHIFT == published["shift"]
    assert sd35_config.NUM_TRAIN_TIMESTEPS == published["num_train_timesteps"]
    # shift(1.0) == 1.0 for any shift, which is why the linspace starts there.
    assert float(static_shift(mx.array(1.0))) == pytest.approx(1.0)
    assert float(static_shift(mx.array(0.5))) == pytest.approx(3 * 0.5 / (1 + 2 * 0.5))


def test_the_schedule_ends_at_zero_and_steps_monotonically_down():
    """`step` reads `sigmas[t + 1]`, so the array is one longer than the loop."""
    scheduler = _ours(28)
    sigmas = np.array(scheduler.sigmas.tolist())
    assert len(sigmas) == 29
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == 0.0
    assert np.all(np.diff(sigmas) < 0)
    assert len(np.array(scheduler.timesteps.tolist())) == 28


# ── The engine's call contract ──────────────────────────────────────────────


def test_generate_image_accepts_exactly_what_the_engine_sends():
    """`engine._generate_kwargs` builds these by name; a rename here is a TypeError there."""
    parameters = inspect.signature(SD35.generate_image).parameters
    for name in (
        "seed",
        "prompt",
        "num_inference_steps",
        "width",
        "height",
        "scheduler",
        "guidance",
        "negative_prompt",
        "image_path",
        "image_strength",
    ):
        assert name in parameters, name
    # No `preset` and no `image_paths`: this is a standard txt2img/img2img family, so
    # the engine needs no special case for it (`engine.py`'s Ideogram branch).
    assert "preset" not in parameters
    assert "image_paths" not in parameters


def test_the_catalogue_scheduler_word_resolves_to_this_packages_scheduler():
    from qds.sd35.scheduler import SCHEDULER_PATH

    for accepted in (None, "linear", "flow_match_euler_discrete", SCHEDULER_PATH):
        assert SD35._resolve_scheduler(accepted) == SCHEDULER_PATH
    with pytest.raises(ValueError, match="Unknown SD 3.5 scheduler"):
        SD35._resolve_scheduler("er_sde")


def test_config_resolves_the_dotted_scheduler_path():
    """The dotted path is only useful if mflux's `Config` actually imports it."""
    from mflux.models.common.config.config import Config

    from qds.sd35.scheduler import SCHEDULER_PATH

    config = Config(
        model_config=sd35_config.sd35_medium_model_config(),
        num_inference_steps=4,
        height=512,
        width=512,
        guidance=1.0,
        scheduler=SCHEDULER_PATH,
    )
    assert isinstance(config.scheduler, SD35FlowMatchScheduler)
    # `requires_sigma_shift` is False, so `Config` must not have called the setter that
    # this scheduler refuses.
    assert config.model_config.requires_sigma_shift is False
    assert len(np.array(config.scheduler.sigmas.tolist())) == 5


def test_the_five_annotated_components_are_the_ones_the_converter_needs():
    """`prequantize._module_class` reads these; `test_components.py` re-checks them."""
    from typing import get_type_hints

    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder

    from qds.sd35.clip import SD35ClipG, SD35ClipL
    from qds.sd35.transformer import SD35Transformer
    from qds.sd35.vae import SD35VAE

    hints = get_type_hints(SD35)
    assert hints["transformer"] is SD35Transformer
    assert hints["text_encoder"] is SD35ClipL
    assert hints["text_encoder_2"] is SD35ClipG
    assert hints["text_encoder_3"] is T5Encoder
    assert hints["vae"] is SD35VAE


# ── The loop, on a stand-in transformer ─────────────────────────────────────


class _RecordingTransformer:
    """Returns a velocity of the latents' shape and records what it was called with."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, hidden_states, timestep, encoder_hidden_states, pooled_projections):
        self.calls.append(
            {
                "timestep": float(timestep[0]),
                "context": encoder_hidden_states,
                "pooled": pooled_projections,
            }
        )
        return mx.zeros(hidden_states.shape) + 0.01


def _loop_model(guidance: float, num_steps: int = 3) -> tuple[SD35, _RecordingTransformer]:
    """An `SD35` with the loop's collaborators replaced. No weights are loaded."""
    model = SD35.__new__(SD35)
    nn_init = mx.zeros((1, 8, 4096))
    model.model_config = sd35_config.sd35_medium_model_config()
    model.prompt_cache = {}
    from mflux.callbacks.callback_registry import CallbackRegistry

    model.callbacks = CallbackRegistry()
    model.tiling_config = None
    model.lora_paths = None
    model.lora_scales = None
    model.bits = None
    transformer = _RecordingTransformer()
    model.transformer = transformer

    def condition(prompt: str):
        # Distinguishable per prompt, so the CFG branches can be told apart.
        marker = float(len(prompt))
        return nn_init + marker, mx.zeros((1, 2048)) + marker

    model._condition = condition
    return model, transformer


def test_the_unconditional_branch_exists_only_above_guidance_one():
    """Two passes per step with CFG, one without. This is the distilled path's cost."""
    model, _ = _loop_model(guidance=4.5)
    context, pooled, negative = model._encode_prompts("a cat", None, 4.5)
    assert negative is not None
    assert context.shape == (1, 8, 4096)
    assert pooled.shape == (1, 2048)

    model, _ = _loop_model(guidance=1.0)
    _, _, negative = model._encode_prompts("a cat", None, 1.0)
    assert negative is None, "guidance 1.0 must not pay for a second transformer pass"


def test_an_absent_negative_prompt_still_gets_an_unconditional_branch():
    """CFG needs somewhere to put the unconditional pass; the reference encodes ""."""
    model, _ = _loop_model(guidance=4.5)
    _, _, negative = model._encode_prompts("a cat", None, 4.5)
    empty_context, _ = negative
    # `""` has length 0, which the stand-in conditioner encodes as marker 0.0.
    assert float(empty_context[0, 0, 0]) == 0.0

    model, _ = _loop_model(guidance=4.5)
    _, _, negative = model._encode_prompts("a cat", "blurry", 4.5)
    assert float(negative[0][0, 0, 0]) == float(len("blurry"))


def test_the_prompt_cache_is_keyed_on_everything_that_changes_the_conditioning():
    model, _ = _loop_model(guidance=4.5)
    calls = []
    inner = model._condition
    model._condition = lambda prompt: (calls.append(prompt), inner(prompt))[1]

    model._encode_prompts("a cat", None, 4.5)
    model._encode_prompts("a cat", None, 4.5)
    assert len(calls) == 2, "one conditional and one unconditional pass, then cached"

    model._encode_prompts("a cat", "blurry", 4.5)
    assert len(calls) == 4
    model._encode_prompts("a cat", None, 1.0)
    assert len(calls) == 5, "guidance 1.0 encodes the prompt only"


def test_the_loop_walks_the_schedule_and_reports_to_the_callbacks():
    """Timesteps come off the scheduler in order, and the callback contract is honoured."""
    model, transformer = _loop_model(guidance=1.0)

    seen: dict[str, list] = {"before": [], "in": [], "after": []}

    class _Callback:
        def call_before_loop(self, seed, prompt, latents, config, **kwargs):
            seen["before"].append(latents.shape)

        def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
            seen["in"].append(t)

        def call_after_loop(self, seed, prompt, latents, config):
            seen["after"].append(latents.shape)

    model.callbacks.register(_Callback())

    class _VAE:
        @staticmethod
        def decode(latents):
            return mx.zeros((1, 3, 1, 64, 64))

    model.vae = _VAE()

    image = model.generate_image(
        seed=3, prompt="a cat", num_inference_steps=4, height=64, width=64, guidance=1.0
    )
    assert image is not None

    assert seen["before"] == [(1, 16, 8, 8)]
    assert seen["in"] == [0, 1, 2, 3]
    assert seen["after"] == [(1, 16, 8, 8)]

    # One pass per step at guidance 1.0, and the timesteps are the schedule's own.
    assert len(transformer.calls) == 4
    expected = np.array(_ours(4).timesteps.tolist())
    got = np.array([call["timestep"] for call in transformer.calls])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3)


def test_classifier_free_guidance_runs_both_branches_per_step():
    model, transformer = _loop_model(guidance=4.5)

    class _VAE:
        @staticmethod
        def decode(latents):
            return mx.zeros((1, 3, 1, 64, 64))

    model.vae = _VAE()
    model.generate_image(
        seed=3, prompt="a cat", num_inference_steps=3, height=64, width=64, guidance=4.5
    )
    assert len(transformer.calls) == 6
    # Conditional first, unconditional second, at the same timestep.
    assert transformer.calls[0]["timestep"] == transformer.calls[1]["timestep"]
    assert float(transformer.calls[0]["pooled"][0, 0]) == float(len("a cat"))
    assert float(transformer.calls[1]["pooled"][0, 0]) == 0.0


def test_an_empty_prompt_is_tokenized_to_a_full_window_rather_than_nothing():
    """The regression this cost a 16 GB download to find.

    mflux's `LanguageTokenizer.tokenize` short-circuits an all-empty batch to a
    `(1, 0)` array. Classifier-free guidance encodes exactly `""` for its
    unconditional branch, and the reference pipeline does not treat that as nothing:
    it produces a full padded window whose start/end/padding tokens are a real
    embedding. Left unhandled, CLIP's `argmax` over a zero-width axis raises — which
    is what happened on the first real generation, on the *default* path of two of
    the three catalogue rows.
    """
    import numpy as np

    class _RawTokenizer:
        def __call__(self, prompts, padding, max_length, truncation, add_special_tokens, return_tensors):
            assert padding == "max_length"
            assert truncation is True
            assert return_tensors == "np"
            # Start, end, then padding — what a CLIP tokenizer returns for "".
            row = [49406, 49407] + [49407] * (max_length - 2)
            return {"input_ids": np.array([row] * len(prompts), dtype=np.int32)}

    class _Wrapper:
        max_length = 77
        tokenizer = _RawTokenizer()

        def tokenize(self, prompt):
            width = 0 if prompt == "" else 77

            class _Out:
                input_ids = mx.zeros((1, width), dtype=mx.int32)

            return _Out()

    model = SD35.__new__(SD35)
    model.tokenizers = {"clip_l": _Wrapper()}

    empty = model._tokenize("clip_l", "")
    assert empty.shape == (1, 77)
    assert int(empty[0, 0]) == 49406
    assert int(empty[0, 1]) == 49407

    # A non-empty prompt still goes through mflux's wrapper untouched.
    assert model._tokenize("clip_l", "a cat").shape == (1, 77)
