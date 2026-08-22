"""Which engine progress is honestly *this* call's.

`ModelEngine` keeps one `ProgressSnapshot` for the whole process, and says why:
its single lock serializes every MLX job, so there are never two in flight, and
one lock-free snapshot is enough for `/v1/progress` to poll. That reasoning is
sound for a *global* progress display. It is not sound for a per-call
notification, and the gap is this module's entire subject.

The hazard is concrete. A playground row is marked `running` by the runner
*before* it awaits `engine.generate`, and the engine may still be finishing a
`/v1` request under its lock. Forwarding `snapshot["step"]` during that window
reports another request's denoising step as this call's progress -- plausible,
monotonic, and wrong.

So a step is reported only when four facts agree, and otherwise the notification
carries what *is* known: where the job sits in its own lifecycle. `images_done`
comes from the generation row, which is authoritative for this job and for no
other, and that is the part that is exactly true. The attributed step is the
part that is a judgement, and it is fenced accordingly.

**What this does not establish.** Attribution rests on `(model, seed)` matching
a global snapshot, so two simultaneous jobs of the same model *and* the same
seed would be conflated. That needs a `/v1` caller to pick, out of 2**32, one of
this call's seeds. The claim is therefore "not misleading", not "precise": a
per-job progress channel in `ModelEngine` would be the precise mechanism, and it
would cost the lock-free snapshot the engine currently argues for.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Attribution:
    """One tool call's view of the engine, and what it may claim from it."""

    generation_id: str
    #: `spec.key`, the *internal* catalogue key -- which is what the engine
    #: writes into the snapshot, not the public name the row stores.
    model_key: str
    seeds: frozenset[int]
    n: int
    steps: int
    #: Progress already reported. The protocol requires a progress value to
    #: increase, and attribution can be lost mid-image (the engine moves on to
    #: another job's work between two polls), so the emitted number is clamped
    #: rather than allowed to fall back.
    high_water: float = field(default=0.0)

    @property
    def total(self) -> float:
        """Denoising steps across the whole run, not images."""
        return float(self.n * self.steps)

    def attributable_step(self, snapshot: dict, *, running: bool) -> int:
        """The engine's step, if and only if it is ours.

        Four conditions, each of which can fail on its own:

        * `running` -- the runner says this generation is the one it claimed;
        * the engine is denoising rather than loading weights, upscaling or
          rewriting, none of which have a step this run can count;
        * the model matches, which separates us from a different model's job;
        * the seed is one of ours, which separates us from the same model's job.
        """
        if not running:
            return 0
        if snapshot.get("state") != "generating":
            return 0
        if snapshot.get("model") != self.model_key:
            return 0
        seed = snapshot.get("seed")
        if seed is None or seed not in self.seeds:
            return 0
        return int(snapshot.get("step") or 0)

    def sample(self, *, record: dict, runner, engine) -> tuple[float, float, str]:
        """`(progress, total, message)` for one notification.

        `record` is the generation row -- the authoritative account of this job.
        `runner` and `engine` are the shared machinery, consulted only for facts
        the row cannot carry: which job is claimed, and where the denoiser is.
        """
        status = record.get("status")
        images_done = len(record.get("images") or [])
        running = getattr(runner, "current_id", None) == self.generation_id
        snapshot = engine.progress()
        step = self.attributable_step(snapshot, running=running)

        # The row records the steps that were *asked for*; the engine reports
        # the ones it is actually running. They normally agree, and when they do
        # not the engine is right -- a sampler preset drops the requested count
        # and decides for itself (`steps_from_preset`), so the row's number is a
        # request rather than a fact. Believing the row there produced
        # "step 9 of 4" and a progress value past its own total, which is both
        # nonsense to read and a protocol violation.
        if step > 0:
            reported = int(snapshot.get("total") or 0)
            if reported > 0:
                self.steps = reported

        progress = float(images_done * self.steps + step)
        # Never backwards, even when attribution is lost between two polls, and
        # never past the end: both are things the protocol forbids and a client
        # renders as a broken bar.
        progress = min(max(progress, self.high_water), self.total)
        self.high_water = progress

        return (
            progress,
            self.total,
            self._message(
                status=status,
                images_done=images_done,
                step=step,
                running=running,
                paused=bool(getattr(runner, "paused", False)),
            ),
        )

    def _message(
        self, *, status: str | None, images_done: int, step: int, running: bool, paused: bool
    ) -> str:
        """Say what is true, and no more.

        Every branch below is reachable, and the uninformative ones are the
        point: "waiting for the engine" is what a caller should be told while
        another job holds the lock, rather than a step number borrowed from it.
        """
        if status == "queued":
            return "queued (the queue is paused)" if paused else "queued"
        if status in {"completed", "failed", "cancelled"}:
            return str(status)
        if not running:
            # Claimed by nobody we recognise: either the runner has not picked
            # this row up yet, or it just finished it.
            return "starting"
        if step > 0:
            of_n = f"image {images_done + 1} of {self.n}, " if self.n > 1 else ""
            return f"{of_n}step {step} of {self.steps}"
        if images_done:
            return f"{images_done} of {self.n} done"
        return "waiting for the engine"
