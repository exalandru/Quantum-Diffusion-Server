"""Progress that does not lie.

`ModelEngine` keeps one snapshot for the whole process, and says why: its single
lock serializes every MLX job, so a global display can poll it. That reasoning
does not carry to a per-call notification, because a playground row is marked
`running` *before* the engine's lock is acquired -- so during that window the
snapshot may still describe a `/v1` request.

The first test below is the one this file exists for. Everything else guards a
property that only matters once the first one holds.
"""

from __future__ import annotations

import asyncio

from qds.mcp.progress import Attribution

from .mcp_support import mcp_session, mcp_settings, text_of


def attribution(**overrides) -> Attribution:
    base = dict(generation_id="gen-1", model_key="flux2-klein", seeds=frozenset({7}), n=1, steps=4)
    base.update(overrides)
    return Attribution(**base)


def snapshot(**overrides) -> dict:
    base = {"state": "generating", "model": "flux2-klein", "seed": 7, "step": 3, "total": 4}
    base.update(overrides)
    return base


# ── The discrimination ─────────────────────────────────────────────────────


def test_a_step_is_claimed_only_when_all_four_facts_agree():
    """The positive, so the negatives below mean something."""
    assert attribution().attributable_step(snapshot(), running=True) == 3


def test_another_jobs_step_is_never_claimed_as_this_calls():
    """The counterfactual an implementation that forwards `snapshot["step"]`
    fails: each of these snapshots is a *plausible* one to read, and each
    belongs to somebody else."""
    subject = attribution()
    assert subject.attributable_step(snapshot(seed=999), running=True) == 0, "another seed"
    assert subject.attributable_step(snapshot(model="z-image"), running=True) == 0, "another model"
    assert subject.attributable_step(snapshot(), running=False) == 0, "not the claimed row"
    assert subject.attributable_step(snapshot(state="loading"), running=True) == 0, "weights"
    assert subject.attributable_step(snapshot(state="upscaling"), running=True) == 0, "an upscale"
    assert subject.attributable_step(snapshot(state="rewriting"), running=True) == 0, "a rewrite"
    assert subject.attributable_step(snapshot(seed=None), running=True) == 0, "no seed at all"


class Runner:
    def __init__(self, current_id=None, paused=False):
        self.current_id = current_id
        self.paused = paused


class Engine:
    def __init__(self, snap):
        self._snap = snap

    def progress(self):
        return self._snap


def sample(subject, *, status, images, current_id, snap, paused=False):
    record = {"status": status, "images": [{"url": f"/x/{i}.png", "seed": 0} for i in range(images)]}
    return subject.sample(record=record, runner=Runner(current_id, paused), engine=Engine(snap))


def test_progress_reports_no_step_while_the_engine_serves_another_job():
    """The whole file in one assertion: a job of the same *model*, at step 7,
    that is not this one. A forwarding implementation reports 7."""
    subject = attribution(steps=9)
    progress, total, message = sample(
        subject,
        status="running",
        images=0,
        current_id="gen-1",
        snap=snapshot(seed=999, step=7, total=9),
    )
    assert progress == 0.0
    assert total == 9.0
    assert "step" not in message
    assert message == "waiting for the engine"


def test_progress_is_monotonic_when_attribution_is_lost_mid_image():
    """The protocol requires progress to increase, and attribution can be lost
    between two polls -- the engine moves to another job's work, or the row is
    handed on. Falling back to 0 would be a protocol violation *and* would read
    to a user as the work restarting."""
    subject = attribution(steps=4)
    first, _, _ = sample(subject, status="running", images=0, current_id="gen-1", snap=snapshot(step=3))
    second, _, _ = sample(
        subject, status="running", images=0, current_id="gen-1", snap=snapshot(model="z-image")
    )
    assert first == 3.0
    assert second == 3.0, "not 0: progress may not go backwards"


def test_a_multi_image_run_is_carried_by_finished_images_not_by_the_step():
    """`images_done` comes from the generation row, which is authoritative for
    this job and for no other. That is the part of the number that is exactly
    true; the attributed step is the part that is a judgement."""
    subject = attribution(n=2, steps=4, seeds=frozenset({7, 8}))
    progress, total, message = sample(
        subject, status="running", images=1, current_id="gen-1", snap=snapshot(seed=8, step=2)
    )
    assert (progress, total) == (6.0, 8.0)
    assert message == "image 2 of 2, step 2 of 4"


def test_a_queued_generation_says_so_and_says_whether_the_queue_is_paused():
    """A human who paused the queue in the browser is otherwise the invisible
    cause of every timeout on this plane."""
    subject = attribution()
    _, _, running = sample(subject, status="queued", images=0, current_id=None, snap=snapshot())
    _, _, paused = sample(subject, status="queued", images=0, current_id=None, snap=snapshot(), paused=True)
    assert running == "queued"
    assert paused == "queued (the queue is paused)"


# ── End to end ─────────────────────────────────────────────────────────────


async def test_a_tool_call_reports_progress_that_never_claims_another_jobs_step(tmp_path):
    """The same property, through the real tool, the real runner and the real
    notification path -- because a predicate can be right while the code that
    calls it reads the wrong snapshot."""
    seen: list[tuple[float, float | None, str | None]] = []

    settings = mcp_settings(tmp_path)
    settings.mcp.poll_interval_s = 0.01
    settings.mcp.tool_timeout_s = 0.4

    async with mcp_session(tmp_path, settings=settings) as (client, app, engine):
        # Park the queue so the generation cannot start, then put the engine to
        # work on a job that is emphatically not ours.
        from .conftest import make_client

        make_client(app).post("/playground/api/queue", json={"paused": True})
        engine.pretend_busy_with("flux2-klein", seed=999, step=7, total=9)

        result = await client.call_tool(
            "generate_image",
            {"prompt": "a cube"},
            progress_callback=lambda p, t, m: seen.append((p, t, m)),
        )

    assert seen, "the tool reported progress at all"
    assert all(progress == 0.0 for progress, _, _ in seen), seen
    assert all("step" not in (message or "") for _, _, message in seen), seen
    # And it came back rather than hanging, naming what to do next.
    body = text_of(result)
    assert result.is_error is False
    assert "wait_for_generation" in body
    assert "paused" in body


async def test_the_ceiling_returns_an_id_and_wait_for_generation_resumes(tmp_path):
    """Timing out is not failing: the work is queued and durable, and the
    second call picks it up."""
    settings = mcp_settings(tmp_path)
    settings.mcp.poll_interval_s = 0.01
    settings.mcp.tool_timeout_s = 0.2

    async with mcp_session(tmp_path, settings=settings) as (client, app, _engine):
        from .conftest import make_client

        http = make_client(app)
        http.post("/playground/api/queue", json={"paused": True})

        early = await client.call_tool("generate_image", {"prompt": "a cube"})
        assert early.is_error is False, "a ceiling is not a failure"
        generation_id = text_of(early).split("generation: ")[1].split()[0]
        assert not [b for b in early.content if getattr(b, "type", "") == "image"]

        http.post("/playground/api/queue", json={"paused": False})
        settings.mcp.tool_timeout_s = 5.0
        finished = await client.call_tool("wait_for_generation", {"generation_id": generation_id})

    assert finished.is_error is False, text_of(finished)
    assert "status: completed" in text_of(finished)
    assert [b for b in finished.content if getattr(b, "type", "") == "resource_link"], (
        "and the resumed call carries the route to the picture"
    )
    assert "full image: http://" in text_of(finished)


async def test_a_cancelled_tool_call_cancels_its_generation(tmp_path):
    """INV-5. A client that walks away must not leave a generation on the GPU,
    and the cancellation runs on a task that outlives the dying handler."""
    settings = mcp_settings(tmp_path)
    settings.mcp.poll_interval_s = 0.01
    settings.mcp.tool_timeout_s = 30.0

    async with mcp_session(tmp_path, settings=settings) as (client, app, _engine):
        from .conftest import make_client, wait_until

        http = make_client(app)
        http.post("/playground/api/queue", json={"paused": True})

        call = asyncio.ensure_future(client.call_tool("generate_image", {"prompt": "a cube"}))
        # Let it reach the queue and start polling.
        await asyncio.sleep(0.2)
        call.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError, Exception):
            await call

        listed = http.get("/playground/api/sessions").json()
        session_id = listed["sessions"][0]["id"]

        def cancelled() -> bool:
            detail = http.get(f"/playground/api/sessions/{session_id}").json()
            return all(g["status"] == "cancelled" for g in detail["generations"])

        await asyncio.sleep(0.3)
        detail = http.get(f"/playground/api/sessions/{session_id}").json()
        assert detail["generations"], "a row was written"
        assert cancelled(), [g["status"] for g in detail["generations"]]
        del wait_until


def test_the_engines_step_count_wins_over_the_rows_requested_one():
    """The row records the steps that were *asked for*; the engine reports the
    ones it is running, and a sampler preset makes those differ
    (`steps_from_preset` drops the requested count and lets the preset decide).

    This was a live defect, found by watching a real 90-second generation over
    HTTP: the notifications read "step 9 of 4", and progress climbed past its own
    total -- nonsense to read, and a protocol violation a client renders as a
    broken bar.
    """
    subject = attribution(steps=4)  # what the row asked for
    progress, total, message = sample(
        subject,
        status="running",
        images=0,
        current_id="gen-1",
        snap=snapshot(step=9, total=9),  # what the engine is actually running
    )
    assert message == "step 9 of 9"
    assert (progress, total) == (9.0, 9.0)


def test_progress_never_exceeds_its_own_total():
    """The other half of the same defect, asserted independently: even if the
    arithmetic went wrong again, the emitted value stays inside its bounds."""
    subject = attribution(n=1, steps=4)
    for step in (1, 4, 40):
        progress, total, _ = sample(
            subject, status="running", images=1, current_id="gen-1", snap=snapshot(step=step)
        )
        assert progress <= total, f"step={step}: {progress} > {total}"


def test_the_row_is_still_believed_when_the_engine_is_not_ours():
    """The counterfactual for the fix: another job's `total` must not redefine
    this run's length. Only an *attributed* step may correct the step count."""
    subject = attribution(steps=4)
    sample(
        subject,
        status="running",
        images=0,
        current_id="gen-1",
        snap=snapshot(seed=999, step=7, total=50),  # somebody else's 50-step run
    )
    assert subject.steps == 4
    assert subject.total == 4.0
