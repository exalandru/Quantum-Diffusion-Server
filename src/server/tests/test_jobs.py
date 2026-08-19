"""The single-flight job manager: what it folds, what it settles, what it kills.

Children here are real processes, and every structured line they emit goes
through `qds.logs`'s own `JsonFormatter` with the same `extra={"event", "fields"}`
call shape the real `fetch` and `prequantize` use. Hand-written JSON fixtures
would have let the producer and the consumer drift apart silently, which is the
one failure this layer cannot afford: the manager's whole job is to understand
what those two commands say.

What is *not* exercised: an actual download or conversion. The children stand in
for the work, not for the protocol.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from qds import configfile, jobs
from qds.jobs import JobManager, JobState

# ── Children ───────────────────────────────────────────────────────────────


#: Emits its lines through the real formatter, so the bytes on the wire are the
#: ones `fetch --json-logs` and `prequantize --json-logs` actually produce.
EMITTER = """
import logging, sys, time
from qds.logs import setup_logging

setup_logging(level="DEBUG", log_file=None, json_lines=True)
logger = logging.getLogger("qds.test-child")
{body}
sys.exit({code})
"""


def child_script(body: str, code: int = 0) -> str:
    return EMITTER.format(body=textwrap.indent(textwrap.dedent(body), ""), code=code)


@pytest.fixture
def run_child(monkeypatch):
    """Point the manager at a scripted child instead of a real subcommand."""

    def install(script: str):
        monkeypatch.setattr(
            jobs, "child_command", lambda argv: [sys.executable, "-c", script, *argv]
        )

    return install


async def settle(manager: JobManager, timeout: float = 20.0) -> jobs.JobStatus:
    """Wait for the job to reach a terminal state, whatever it turns out to be."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        status = await manager.status()
        if not status.state.is_active:
            return status
        await asyncio.sleep(0.02)
    raise AssertionError(f"job never settled; last state {status.state}")


async def wait_for_event(manager: JobManager, event: str, timeout: float = 20.0) -> jobs.JobStatus:
    """Observe the job *while it runs*, rather than racing its exit.

    A completed job clears its message, so anything asserting on in-flight
    progress has to catch it in flight — with a child that outlives the
    assertion, not one whose sleep happens to be longer than the test's.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        status = await manager.status()
        if status.event == event:
            return status
        assert status.state.is_active, f"job ended before emitting {event!r}: {status}"
        await asyncio.sleep(0.02)
    raise AssertionError(f"never saw event {event!r}")


# ── Folding the child's structured output ──────────────────────────────────


@pytest.mark.asyncio
async def test_a_progress_event_is_kept_verbatim(run_child):
    run_child(
        child_script(
            """
            logger.info("converting block 3", extra={"event": "prequantize_progress",
                                                     "fields": {"block": 3, "blocks": 57}})
            time.sleep(30)
            """
        )
    )
    manager = JobManager()
    await manager.start_prequantize("z-image", 4)

    status = await wait_for_event(manager, "prequantize_progress")
    assert status.fields == {"block": 3, "blocks": 57}
    assert status.message == "converting block 3"

    await manager.cancel()
    await settle(manager)


@pytest.mark.asyncio
async def test_output_that_is_not_json_is_ignored_rather_than_guessed_at(run_child):
    """tqdm writes to stderr without going through logging; it must not confuse us."""
    run_child(
        child_script(
            """
            logger.info("step one", extra={"event": "fetch_start", "fields": {"key": "z-image"}})
            print("  42%|####      | 4.2G/10G [00:03<00:07]", file=sys.stderr)
            print("not json either")
            time.sleep(30)
            """
        )
    )
    manager = JobManager()
    await manager.start_fetch("z-image")

    status = await wait_for_event(manager, "fetch_start")
    assert status.message == "step one"
    # The tqdm bar and the bare line arrive after it and change nothing.
    await asyncio.sleep(0.3)
    status = await manager.status()
    assert status.event == "fetch_start"
    assert status.message == "step one"

    await manager.cancel()
    await settle(manager)


@pytest.mark.asyncio
async def test_a_failure_reports_what_the_child_said_not_its_exit_code(run_child):
    run_child(
        child_script(
            """
            logger.error("z-image is gated: accept the licence on HuggingFace first.")
            """,
            code=1,
        )
    )
    manager = JobManager()
    await manager.start_fetch("z-image")

    status = await settle(manager)
    assert status.state is JobState.FAILED
    assert status.message == "z-image is gated: accept the licence on HuggingFace first."
    assert "code 1" not in (status.message or "")


@pytest.mark.asyncio
async def test_a_silent_failure_falls_back_to_the_exit_code(run_child):
    run_child(child_script("pass", code=3))
    manager = JobManager()
    await manager.start_fetch("z-image")

    status = await settle(manager)
    assert status.state is JobState.FAILED
    assert status.message == "exited with code 3"


@pytest.mark.asyncio
async def test_an_info_line_updates_the_message_without_becoming_the_reason(run_child):
    """Only ERROR and CRITICAL are terminal reasons; progress chatter is not."""
    run_child(
        child_script(
            """
            logger.info("downloading shard 2 of 9")
            """,
            code=1,
        )
    )
    manager = JobManager()
    await manager.start_fetch("z-image")

    status = await settle(manager)
    assert status.state is JobState.FAILED
    assert status.message == "exited with code 1"


# ── Single flight ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_second_job_is_refused_and_the_message_names_the_first(run_child):
    run_child(child_script("time.sleep(2)"))
    manager = JobManager()
    await manager.start_fetch("z-image")

    with pytest.raises(jobs.JobBusy) as raised:
        await manager.start_prequantize("flux2-dev", 8)
    assert "z-image" in str(raised.value)
    assert "one heavy model operation" in str(raised.value)

    await manager.cancel()
    await settle(manager)


@pytest.mark.asyncio
async def test_a_finished_job_no_longer_blocks_the_next_one(run_child):
    run_child(child_script("pass"))
    manager = JobManager()
    await manager.start_fetch("z-image")
    assert (await settle(manager)).state is JobState.COMPLETED

    await manager.start_fetch("ernie-image")
    assert (await manager.status()).target == "ernie-image"
    await settle(manager)


# ── Cancellation ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancelling_stops_the_child_and_says_so(run_child):
    run_child(child_script("time.sleep(30)"))
    manager = JobManager()
    await manager.start_fetch("z-image")

    cancelling = await manager.cancel()
    assert cancelling.state is JobState.CANCELLING

    status = await settle(manager)
    assert status.state is JobState.CANCELLED
    assert status.message == "Cancelled."


@pytest.mark.asyncio
async def test_a_child_ignoring_sigterm_is_killed_after_the_grace(run_child, monkeypatch):
    """The second rung of the ladder, with the grace shortened so it is testable."""
    monkeypatch.setattr(jobs, "CANCEL_GRACE_S", 0.5)
    run_child(
        child_script(
            """
            import signal
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            logger.info("ignoring SIGTERM")
            time.sleep(30)
            """
        )
    )
    manager = JobManager()
    await manager.start_fetch("z-image")
    await asyncio.sleep(0.3)

    await manager.cancel()
    status = await settle(manager, timeout=10.0)
    assert status.state is JobState.CANCELLED


@pytest.mark.asyncio
async def test_the_whole_process_group_goes_not_just_the_child(run_child, tmp_path):
    """A download spawns workers; signalling only the parent orphans them."""
    marker = tmp_path / "grandchild-alive"
    run_child(
        child_script(
            f"""
            import subprocess
            grandchild = subprocess.Popen([
                sys.executable, "-c",
                "import time, pathlib\\n"
                "p = pathlib.Path({str(marker)!r})\\n"
                "\\nwhile True:\\n    p.write_text('alive')\\n    time.sleep(0.05)\\n",
            ])
            logger.info("worker started", extra={{"event": "fetch_start", "fields": {{}}}})
            time.sleep(30)
            """
        )
    )
    manager = JobManager()
    await manager.start_fetch("z-image")

    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.05)
    assert marker.exists(), "the grandchild never started; the test proves nothing"

    await manager.cancel()
    await settle(manager)

    # If the grandchild survived, it keeps rewriting the marker.
    marker.unlink()
    await asyncio.sleep(0.5)
    assert not marker.exists(), "the grandchild outlived the cancelled job"


@pytest.mark.asyncio
async def test_cancelling_nothing_is_an_error_not_a_silent_success():
    manager = JobManager()
    with pytest.raises(jobs.NoJobRunning):
        await manager.cancel()


# ── Completion work ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_completed_conversion_is_selected_before_anyone_can_see_it_finish(
    run_child, monkeypatch, tmp_path
):
    """The ordering invariant, stated as the race it prevents.

    The selection is written while the manager still holds its lock, so the
    first status read that reports `completed` already reflects it. Asserting
    "the file changed eventually" would pass even if the write happened after
    the interface had refreshed — which is the bug.
    """
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"models": {"z-image": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    run_child(
        child_script(
            """
            logger.info("done", extra={"event": "prequantize_done",
                                       "fields": {"model": "z-image", "bits": 4}})
            """
        )
    )
    manager = JobManager()
    changed: list[bool] = []
    manager.on_config_changed = lambda: changed.append(True)
    await manager.start_prequantize("z-image", 4)

    status = await settle(manager)
    assert status.state is JobState.COMPLETED
    # Read at the very moment the state is first observed as terminal.
    document = configfile.read(config)
    assert document["models"]["z-image"]["prequantized_variant"] == 4
    assert changed == [True]


@pytest.mark.asyncio
async def test_a_partial_conversion_claims_no_variant(run_child, monkeypatch, tmp_path):
    """`prequantize_partial` exits 0 too; only `_done` means a usable artifact."""
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"models": {"z-image": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    run_child(
        child_script(
            """
            logger.info("partial", extra={"event": "prequantize_partial",
                                          "fields": {"model": "z-image", "bits": 4}})
            """
        )
    )
    manager = JobManager()
    changed: list[bool] = []
    manager.on_config_changed = lambda: changed.append(True)
    await manager.start_prequantize("z-image", 4, components=["vae"])

    assert (await settle(manager)).state is JobState.COMPLETED
    assert "prequantized_variant" not in configfile.read(config).get("models", {}).get(
        "z-image", {}
    )
    assert changed == []


@pytest.mark.asyncio
async def test_a_cancelled_conversion_claims_no_variant(run_child, monkeypatch, tmp_path):
    config = tmp_path / "server-config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    run_child(
        child_script(
            """
            logger.info("done", extra={"event": "prequantize_done",
                                       "fields": {"model": "z-image", "bits": 4}})
            time.sleep(30)
            """
        )
    )
    manager = JobManager()
    changed: list[bool] = []
    manager.on_config_changed = lambda: changed.append(True)
    await manager.start_prequantize("z-image", 4)
    await asyncio.sleep(0.4)

    await manager.cancel()
    assert (await settle(manager)).state is JobState.CANCELLED
    assert configfile.read(config) == {}
    assert changed == []


@pytest.mark.asyncio
async def test_a_conversion_naming_no_bit_depth_selects_nothing(
    run_child, monkeypatch, tmp_path, caplog
):
    config = tmp_path / "server-config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    run_child(
        child_script(
            """
            logger.info("done", extra={"event": "prequantize_done", "fields": {"model": "z-image"}})
            """
        )
    )
    manager = JobManager()
    with caplog.at_level("WARNING", logger="qds.jobs"):
        await manager.start_prequantize("z-image", 4)
        assert (await settle(manager)).state is JobState.COMPLETED

    assert configfile.read(config) == {}
    assert "nothing selected" in caplog.text


# ── Shutdown ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_server_shutdown_leaves_no_orphan(run_child, tmp_path):
    marker = tmp_path / "still-running"
    run_child(
        child_script(
            f"""
            import pathlib, time
            p = pathlib.Path({str(marker)!r})
            logger.info("working", extra={{"event": "fetch_start", "fields": {{}}}})
            while True:
                p.write_text("alive")
                time.sleep(0.05)
            """
        )
    )
    manager = JobManager()
    await manager.start_fetch("z-image")
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.05)
    assert marker.exists(), "the child never started; the test proves nothing"

    await manager.shutdown()

    marker.unlink()
    await asyncio.sleep(0.5)
    assert not marker.exists(), "the job outlived the server"
    assert (await manager.status()).state is JobState.CANCELLED


# ── The real commands still speak this protocol ────────────────────────────


@pytest.mark.parametrize(
    ("start", "args", "kwargs"),
    [
        ("start_fetch", ("z-image",), {}),
        ("start_prequantize", ("z-image", 4), {}),
        ("start_prequantize", ("z-image", 8), {"components": ["vae", "transformer"]}),
        ("start_prequantize", ("z-image", 4), {"dest": "/tmp/elsewhere"}),
    ],
)
async def test_the_real_parsers_accept_the_argv_the_manager_builds(
    monkeypatch, start, args, kwargs
):
    """The exact argv, through the exact parser that will receive it.

    An earlier version of this ran `main(["--help"])`, which exits before
    parsing anything: renaming `--json-logs` or dropping `--components` would
    have left it green while every job started from the dashboard died on
    launch.
    """
    from qds import fetch, prequantize

    built: list[list[str]] = []

    async def capture(*command, **_):
        built.append(list(command))
        raise _Spawned

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    manager = JobManager()
    with pytest.raises(_Spawned):
        await getattr(manager, start)(*args, **kwargs)

    command = built[0]
    assert command[:3] == [sys.executable, "-m", "qds"], command
    subcommand, *tail = command[3:]

    parser = {"fetch": fetch, "prequantize": prequantize}[subcommand].build_parser()
    # `parse_args` exits 2 on anything the parser does not accept, so reaching
    # the assertions below is itself the proof that this argv is acceptable.
    parsed = parser.parse_args(tail)
    assert parsed.json_logs is True, "the manager always asks for JSON lines"


class _Spawned(Exception):
    """Raised instead of launching a child, once the command is known."""


def test_the_child_is_always_this_interpreter():
    """Never a `qds` from PATH, which could be a different installation."""
    assert jobs.child_command(["fetch", "z-image"]) == [
        sys.executable,
        "-m",
        "qds",
        "fetch",
        "z-image",
    ]
    assert Path(sys.executable).is_file()
