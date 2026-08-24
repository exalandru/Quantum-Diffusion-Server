"""SD 3.5's sigma schedule — diffusers' static shift, not mflux's `set_mu`.

**This module exists because a plan premise was false.** The obvious move, and the one
`qds/anima/` makes, is `FlowMatchEulerDiscreteScheduler.set_mu(log(shift))`: mflux
expresses shift as `mu` over an exponential time shift, and exponential shift at `mu`
equals static shift at `exp(mu)`, so for Anima the two are the same function. For
SD 3.5 they are not, and the reason is the *spacing*, not the shift.

Both schedules shift the same way. They disagree on where the linspace ends:

* mflux's `set_mu` walks `linspace(1.0, 1/num_steps, num_steps)` — Anima's schedule;
* diffusers' `FlowMatchEulerDiscreteScheduler.set_timesteps(num_steps)`, which is what
  `StableDiffusion3Pipeline` calls with `sigmas=None`, walks
  `linspace(sigma_max, sigma_min, num_steps)` where `sigma_min` is the *shifted*
  smallest training sigma, `shift(1/1000) ≈ 0.002994`.

At 28 steps that puts mflux's last sigma at 0.1 where the reference has 0.00893. At the
4 steps SD 3.5 Large Turbo is distilled for it is far worse: 0.5 against 0.00893, so the
loop would stop halfway through denoising and return something visibly unfinished.
`tests/test_sd35.py` compares this class against diffusers' real scheduler element by
element rather than re-deriving the formula, so the claim is checked against the
reference and not against my reading of it.

Everything else is inherited. `step` is the same Euler update
(`latents + (sigma_next - sigma) * velocity`), which mflux and diffusers already agree
on exactly; only the schedule it walks is replaced.
"""

from __future__ import annotations

import mlx.core as mx
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler,
)

from qds.sd35.config import SIGMA_SHIFT

#: Dotted path `Config` resolves this class through. mflux's `Config.scheduler`
#: imports any dotted name it is handed, which is how a local family supplies its own
#: schedule without registering anything globally.
SCHEDULER_PATH = "qds.sd35.scheduler.SD35FlowMatchScheduler"


def static_shift(sigmas: mx.array, shift: float = SIGMA_SHIFT) -> mx.array:
    """`shift * s / (1 + (shift - 1) * s)` — diffusers' non-dynamic shift, verbatim."""
    return shift * sigmas / (1 + (shift - 1) * sigmas)


class SD35FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """`FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=False)`.

    The shift is 3.0 in all three repositories' `scheduler/scheduler_config.json`, and
    it is fixed rather than resolution-dependent — which is why nothing here consults
    the image sequence length.
    """

    shift: float = SIGMA_SHIFT

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        num_steps = self.config.num_inference_steps
        # `sigma_max` is `shift(1.0)`, which is 1.0 for every shift; `sigma_min` is
        # `shift(1/num_train_timesteps)`, already shifted once — diffusers derives the
        # endpoints from its shifted training schedule and then shifts the linspace
        # again. Reproducing that twice-shifted lower bound is the whole difference.
        sigma_min = float(static_shift(mx.array(1.0 / self.num_train_timesteps), self.shift))
        if num_steps == 1:
            sigmas = mx.array([1.0], dtype=mx.float32)
        else:
            sigmas = mx.linspace(1.0, sigma_min, num_steps, dtype=mx.float32)
            sigmas = static_shift(sigmas, self.shift)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        return sigmas, timesteps

    def set_mu(self, mu: float) -> None:
        raise NotImplementedError(
            "SD 3.5's schedule is a fixed shift of 3.0 over diffusers' own sigma "
            "spacing, which `set_mu` does not reproduce — see this module's docstring. "
            "The schedule is built in the constructor; nothing needs to set it."
        )

    def set_image_seq_len(self, image_seq_len: int) -> None:
        raise NotImplementedError(
            "SD 3.5 does not shift by image sequence length: `use_dynamic_shifting` is "
            "false in every published scheduler config. Its `ModelConfig` sets "
            "`requires_sigma_shift=False`, so nothing should call this."
        )
