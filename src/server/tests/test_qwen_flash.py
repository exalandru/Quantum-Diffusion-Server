"""Qwen-Image-Flash's noise schedule, checked against the scheduler it comes from.

Flash needs no ported architecture — it is Qwen-Image, and mflux implements that.
What it needs is its own schedule: NVIDIA publishes `shift: 3.0` with
`use_dynamic_shifting: false` and no terminal stretch, where `Qwen/Qwen-Image-2512`
publishes dynamic shifting with a terminal of 0.02. mflux's `qwen_image` config
encodes the second, correctly, for the model it belongs to.

`qds/qwen_flash/config.py` expresses the first without adding a scheduler, by
setting mflux's base and max shift equal so the interpolation collapses to a
constant. That is a trick, and a trick deserves a witness: these tests compare the
sigmas mflux produces against the ones diffusers produces from Flash's own
published `scheduler_config.json`, at the four steps the model is distilled for.
"""

from __future__ import annotations

import numpy as np
import pytest

from qds.qwen_flash import config as flash_config
from qds.registry import BASE_SPECS_BY_KEY

pytest.importorskip("diffusers")

#: `nvidia/Qwen-Image-Flash/scheduler/scheduler_config.json`, fetched from the
#: repository. Recorded here so the comparison needs no network.
FLASH_SCHEDULER_CONFIG = dict(
    base_image_seq_len=256,
    base_shift=0.5,
    invert_sigmas=False,
    max_image_seq_len=8192,
    max_shift=0.9,
    num_train_timesteps=1000,
    shift=3.0,
    shift_terminal=None,
    stochastic_sampling=False,
    time_shift_type="exponential",
    use_beta_sigmas=False,
    use_dynamic_shifting=False,
    use_exponential_sigmas=False,
    use_karras_sigmas=False,
)


def _mflux_sigmas(model_config, steps: int, width: int, height: int) -> np.ndarray:
    from mflux.models.common.config.config import Config

    config = Config(
        model_config=model_config,
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=1.0,
        scheduler="linear",
    )
    return np.array(config.scheduler.sigmas)


def _diffusers_sigmas(steps: int) -> np.ndarray:
    """What `Krea2Pipeline`-style `retrieve_timesteps(sigmas=linspace(1, 1/N, N))` gives."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(**FLASH_SCHEDULER_CONFIG)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps))
    return scheduler.sigmas.numpy()


@pytest.mark.parametrize("steps", [4, 8])
@pytest.mark.parametrize(("width", "height"), [(1280, 720), (1024, 1024), (1920, 1072)])
def test_the_flash_schedule_matches_the_published_scheduler(steps, width, height):
    """The whole point: the same sigmas diffusers would produce, at any resolution.

    Resolution is a parameter here because that is exactly what the trick has to
    survive — mflux's `mu` normally varies with the image sequence length, and
    Flash's shift does not vary at all.
    """
    ours = _mflux_sigmas(flash_config.qwen_image_flash_model_config(), steps, width, height)
    theirs = _diffusers_sigmas(steps)

    assert ours.shape == theirs.shape
    assert np.abs(ours - theirs).max() < 1e-6, (steps, width, height, ours, theirs)
    # Terminal zero, and no stretch before it: Flash publishes `shift_terminal: null`.
    assert ours[-1] == 0.0
    assert ours[-2] > 0.1


def test_the_schedule_would_be_wrong_under_qwen_images_own_config():
    """The discriminating case, so the test above cannot pass for a trivial reason.

    If reusing `ModelConfig.qwen_image()` produced the same sigmas, this whole
    module would be unnecessary. It does not: dynamic shifting plus a 0.02
    terminal is a different trajectory, and at four steps that is most of the
    trajectory.
    """
    from mflux.models.common.config.model_config import ModelConfig

    flash = _mflux_sigmas(flash_config.qwen_image_flash_model_config(), 4, 1280, 720)
    reused = _mflux_sigmas(ModelConfig.qwen_image(), 4, 1280, 720)

    assert np.abs(flash - reused).max() > 0.05, (flash, reused)
    # And the reused one really does stretch its tail, where Flash ends at zero.
    assert reused[-2] != pytest.approx(flash[-2], abs=1e-3)


def test_the_static_shift_is_the_published_one_expressed_as_mu():
    """`exp(mu)` is the shift, because mflux shifts exponentially and diffusers does not."""
    import math

    assert flash_config.SIGMA_SHIFT == 3.0
    assert math.exp(flash_config.SIGMA_MU) == pytest.approx(3.0)

    config = flash_config.qwen_image_flash_model_config()
    # Equal endpoints are what makes the interpolation constant; if they ever
    # diverge, `mu` becomes resolution-dependent again and the model drifts.
    assert config.sigma_base_shift == config.sigma_max_shift == flash_config.SIGMA_MU
    assert config.sigma_shift_terminal is None
    assert config.requires_sigma_shift is True


def test_the_flash_row_uses_qwens_family_and_its_own_config():
    """No port: the architecture is mflux's, only the schedule is ours."""
    spec = BASE_SPECS_BY_KEY["qwen-image-flash"]
    assert spec.family == "qwen"
    assert spec.repo == flash_config.REPO
    assert spec.model_config_name == "qwen_image_flash"
    # Four steps at guidance 1.0, from NVIDIA's own example.
    assert (spec.default_steps, spec.default_guidance) == (4, 1.0)
    assert spec.scheduler == "linear"
    assert spec.gated is False
