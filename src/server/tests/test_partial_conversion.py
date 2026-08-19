"""Partial conversions: what they may claim, and what they must never claim.

A conversion can now be done a component at a time, across separate runs and
separate processes. That makes "half an artifact" a state the system has to hold
correctly rather than an accident to be cleaned up — and the one thing it must
never do is let half an artifact be generated from.

Nothing here converts a model; components are synthesised in the shape
`ModelSaver` writes, the same shape `test_artifacts.py` uses.
"""

from __future__ import annotations

from qds import artifacts, components
from qds import availability as av
from qds.prequantize import convert
from qds.registry import STRATEGY_QDS_MEMORY_BOUNDED

from .test_artifacts import _Args, component_writer, write_component, write_tokenizer

SOURCE = "mlx-community/Z-Image-bf16"
REQUIRED = components.required_components("z-image")


def convert_one(monkeypatch, dest, component, *, bits=4, model="z-image"):
    """One run converting one component, with the heavy part stubbed out."""
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)
    seen: list[str] = []
    monkeypatch.setattr("qds.prequantize.convert_component", component_writer(seen))
    code = convert(_Args(model, bits, dest=dest, components=[component]))
    return code, seen


# ── One component at a time ────────────────────────────────────────────────


def test_converting_one_component_does_not_require_a_complete_artifact(monkeypatch, tmp_path):
    """The run succeeds; it simply has not produced a usable model yet."""
    dest = tmp_path / "artifact"
    code, seen = convert_one(monkeypatch, dest, "transformer")

    assert code == 0, "a partial run is a legitimate run, not a failure"
    assert seen == ["transformer"]
    assert not (dest / av.COMPLETION_MARKER).exists()

    progress = artifacts.read_progress(dest)
    assert progress is not None
    assert progress.completed() == ("transformer",)


def test_a_partial_component_set_is_not_activatable(monkeypatch, tmp_path):
    """The property everything else here exists to protect."""
    dest = artifacts.artifact_dir("z-image", SOURCE, 4, base=str(tmp_path))
    convert_one(monkeypatch, dest, "transformer")

    # Not a variant: `discover_variants` is what activation chooses from.
    assert artifacts.discover_variants("z-image", SOURCE, base=str(tmp_path)) == []
    # And it reads as partial rather than as present.
    state, detail = artifacts.artifact_state(dest, expect_source=SOURCE, expect_bits=4)
    assert state == av.PARTIAL
    assert "transformer" in (detail or "")

    # It *is* reported, separately, as work in progress.
    partials = artifacts.discover_partials(
        "z-image", SOURCE, expected=REQUIRED, base=str(tmp_path)
    )
    assert [p.bits for p in partials] == [4]
    assert partials[0].components["transformer"] == artifacts.COMPONENT_COMPLETE
    assert partials[0].components["vae"] == artifacts.COMPONENT_MISSING


def test_completing_the_last_required_component_makes_the_variant_valid(monkeypatch, tmp_path):
    dest = artifacts.artifact_dir("z-image", SOURCE, 4, base=str(tmp_path))
    for component in REQUIRED[:-1]:
        convert_one(monkeypatch, dest, component)
        assert artifacts.discover_variants("z-image", SOURCE, base=str(tmp_path)) == []

    code, _ = convert_one(monkeypatch, dest, REQUIRED[-1])
    assert code == 0

    variants = artifacts.discover_variants("z-image", SOURCE, base=str(tmp_path))
    assert [v.bits for v in variants] == [4]
    assert variants[0].size_bytes and variants[0].size_bytes > 0
    # And the partial record is gone: one authority for a finished artifact.
    assert artifacts.read_progress(dest) is None
    assert (
        artifacts.discover_partials("z-image", SOURCE, expected=REQUIRED, base=str(tmp_path)) == []
    )

    record = artifacts.read_record(dest)
    assert record is not None
    assert set(record.components) == set(REQUIRED)
    assert set(record.required) == set(REQUIRED)
    assert record.strategy == STRATEGY_QDS_MEMORY_BOUNDED


def test_an_already_converted_component_survives_the_next_run(monkeypatch, tmp_path):
    """Continuing must not cost the hours the previous run already spent."""
    dest = tmp_path / "artifact"
    convert_one(monkeypatch, dest, "transformer")
    first = (dest / "transformer" / "0.safetensors").read_bytes()
    stamp = (dest / "transformer" / "0.safetensors").stat().st_mtime_ns

    convert_one(monkeypatch, dest, "vae")

    assert (dest / "transformer" / "0.safetensors").read_bytes() == first
    assert (dest / "transformer" / "0.safetensors").stat().st_mtime_ns == stamp
    assert set(artifacts.read_progress(dest).completed()) == {"transformer", "vae"}


# ── Identity of partial work ───────────────────────────────────────────────


def test_partial_work_for_another_bit_depth_is_not_continued(monkeypatch, tmp_path):
    """Half a 4-bit conversion is not half of an 8-bit one."""
    dest = tmp_path / "artifact"
    convert_one(monkeypatch, dest, "transformer", bits=4)

    # The same directory, asked for a different precision. Whatever it holds is
    # not progress towards this request.
    states = artifacts.component_states(
        dest, expected=REQUIRED, source=SOURCE, bits=8, strategy=STRATEGY_QDS_MEMORY_BOUNDED
    )
    assert set(states.values()) == {artifacts.COMPONENT_MISSING}


def test_partial_work_from_another_source_is_not_continued(monkeypatch, tmp_path):
    dest = tmp_path / "artifact"
    convert_one(monkeypatch, dest, "transformer", bits=4)

    states = artifacts.component_states(
        dest,
        expected=REQUIRED,
        source="someone/else",
        bits=4,
        strategy=STRATEGY_QDS_MEMORY_BOUNDED,
    )
    assert set(states.values()) == {artifacts.COMPONENT_MISSING}
    # And it is not offered as partial work towards that other source either.
    assert (
        artifacts.discover_partials(
            "z-image", "someone/else", expected=REQUIRED, base=str(tmp_path)
        )
        == []
    )


def test_recording_a_component_for_different_work_replaces_rather_than_merges(tmp_path):
    """Mixing two sources' components would produce an artifact of neither."""
    dest = tmp_path / "artifact"
    dest.mkdir()
    artifacts.record_component(
        dest,
        model_key="z-image",
        family="z-image",
        source=SOURCE,
        bits=4,
        strategy=STRATEGY_QDS_MEMORY_BOUNDED,
        component="transformer",
    )
    artifacts.record_component(
        dest,
        model_key="z-image",
        family="z-image",
        source="someone/else",
        bits=4,
        strategy=STRATEGY_QDS_MEMORY_BOUNDED,
        component="vae",
    )

    progress = artifacts.read_progress(dest)
    assert progress.completed() == ("vae",)
    assert progress.source == "someone/else"


# ── Cancellation ───────────────────────────────────────────────────────────


def test_a_cancelled_component_is_never_recorded_as_complete(monkeypatch, tmp_path):
    """What a kill leaves behind, and what it must not leave behind.

    The Rust job manager kills the child; the child records a component only
    *after* that component has been written and validated, so there is no path
    on which an interrupted component is marked done.
    """
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)

    dest = tmp_path / "artifact"
    dest.mkdir()
    # One component already done by an earlier run.
    write_component(dest, "transformer", bits="4")
    write_tokenizer(dest)
    artifacts.record_component(
        dest,
        model_key="z-image",
        family="z-image",
        source=SOURCE,
        bits=4,
        strategy=STRATEGY_QDS_MEMORY_BOUNDED,
        component="transformer",
    )

    class Killed(BaseException):
        """Stands in for SIGKILL: nothing after this point in the run happens."""

    def killed(name, *, spec, repo, dest, bits):
        (dest / name).mkdir(parents=True, exist_ok=True)
        (dest / name / "0.safetensors").write_bytes(b"half a shard")
        raise Killed()

    monkeypatch.setattr("qds.prequantize.convert_component", killed)
    try:
        convert(_Args("z-image", 4, dest=dest, components=["vae"]))
    except Killed:
        pass

    progress = artifacts.read_progress(dest)
    # The earlier component is still recorded; the killed one is not.
    assert progress.completed() == ("transformer",)
    assert not (dest / av.COMPLETION_MARKER).exists()
    # And the half-written directory does not validate, so nothing reads it as
    # done even without the record.
    assert not av.component_is_complete(dest / "vae")
    assert artifacts.discover_variants("z-image", SOURCE, base=str(tmp_path)) == []


def test_a_rewrite_of_a_complete_artifact_drops_its_completion_first(monkeypatch, tmp_path):
    """An artifact being rewritten is not a complete artifact while it is.

    Reconverting one component in place used to be safe only because nobody did
    it. If the run were cancelled halfway, the completion marker would still be
    vouching for a component that is now half-written.
    """
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)

    dest = tmp_path / "artifact"
    dest.mkdir()
    for name in REQUIRED:
        write_component(dest, name, bits="4")
    write_tokenizer(dest)
    artifacts.write_record(
        dest,
        model_key="z-image",
        family="z-image",
        source=SOURCE,
        bits=4,
        strategy=STRATEGY_QDS_MEMORY_BOUNDED,
        components=REQUIRED,
        required=REQUIRED,
    )

    seen: list[str] = []

    def observe(name, *, spec, repo, dest, bits):
        # What the world looks like *while* the component is being rewritten.
        seen.append(name)
        assert not (dest / av.COMPLETION_MARKER).exists(), (
            "the artifact still claimed to be complete while being rewritten"
        )
        # The components that are still valid are remembered, so a kill here
        # leaves continuable progress rather than an empty slate.
        assert set(artifacts.read_progress(dest).completed()) == set(REQUIRED)
        write_component(dest, name, bits="4")
        return 6

    monkeypatch.setattr("qds.prequantize.convert_component", observe)
    assert convert(_Args("z-image", 4, dest=dest, components=["vae"])) == 0
    assert seen == ["vae"]
    # Re-validated and re-recorded at the end of the run.
    assert (dest / av.COMPLETION_MARKER).is_file()
    assert artifacts.read_progress(dest) is None


# ── Legacy artifacts ───────────────────────────────────────────────────────


def test_a_v2_marker_is_still_complete_and_is_not_reconverted(tmp_path):
    """Markers written before component state existed said everything by saying less."""
    import json

    dest = tmp_path / "old"
    for name in REQUIRED:
        write_component(dest, name, bits="4")
    (dest / av.COMPLETION_MARKER).write_text(
        json.dumps(
            {
                "marker_version": 2,
                "model_key": "z-image",
                "family": "z-image",
                "source": SOURCE,
                "bits": 4,
                "strategy": "mflux_save",
                "components": list(REQUIRED),
            }
        ),
        encoding="utf-8",
    )

    record = artifacts.read_record(dest)
    assert record.marker_version == 2
    # No `required` field, so what it wrote *is* what it needed — it was only
    # ever written when the conversion was complete.
    assert set(record.expected) == set(REQUIRED)
    assert record.size_bytes is None, "an old marker measured nothing, and none is invented"
    assert artifacts.artifact_state(dest, expect_source=SOURCE, expect_bits=4)[0] == av.PRESENT


def test_a_v1_marker_still_reads_as_a_finished_artifact(tmp_path):
    import json

    dest = tmp_path / "ancient"
    for name in av.REQUIRED_COMPONENTS:
        write_component(dest, name, bits="8")
    (dest / av.COMPLETION_MARKER).write_text(
        json.dumps({"marker_version": 1, "bits": 8, "components": list(av.REQUIRED_COMPONENTS)}),
        encoding="utf-8",
    )

    record = artifacts.read_record(dest)
    assert record.legacy is True
    assert artifacts.artifact_state(dest, expect_bits=8)[0] == av.PRESENT


def test_an_artifact_with_no_marker_at_all_is_still_recognised(tmp_path):
    """The FLUX.2-dev artifact that predates markers entirely."""
    dest = tmp_path / "flux2-dev-mlx-8bit"
    for name in av.REQUIRED_COMPONENTS:
        write_component(dest, name, bits="8")

    assert artifacts.artifact_state(dest)[0] == av.PRESENT
    variants = artifacts.discover_variants("flux2-dev", str(dest), base=str(tmp_path / "other"))
    assert [v.bits for v in variants] == [8]
    assert variants[0].legacy is True
    # Measured rather than guessed, because the marker never recorded a size.
    assert variants[0].size_bytes == artifacts.directory_size(dest)


def test_directory_size_does_not_follow_symlinks(tmp_path):
    """What keeps a HuggingFace snapshot from counting its blobs twice."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "weight").write_bytes(b"x" * 1000)
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "weight").symlink_to(blobs / "weight")

    assert artifacts.directory_size(tmp_path) == 1000
    assert artifacts.directory_size(snapshot) == 0


def test_partial_and_complete_conversions_of_one_model_stay_separate(monkeypatch, tmp_path):
    """A finished 4-bit and a half-done 3-bit are two different lists."""
    complete = artifacts.artifact_dir("z-image", SOURCE, 4, base=str(tmp_path))
    for name in REQUIRED:
        convert_one(monkeypatch, complete, name, bits=4)

    started = artifacts.artifact_dir("z-image", SOURCE, 3, base=str(tmp_path))
    convert_one(monkeypatch, started, "transformer", bits=3)

    assert [v.bits for v in artifacts.discover_variants("z-image", SOURCE, base=str(tmp_path))] == [4]
    assert [
        p.bits
        for p in artifacts.discover_partials(
            "z-image", SOURCE, expected=REQUIRED, base=str(tmp_path)
        )
    ] == [3]


# ── The terminal event the supervisor acts on ──────────────────────────────


def terminal_event(caplog):
    """The last structured event a run emitted, as the supervisor reads it."""
    events = [
        (record.__dict__.get("event"), record.__dict__.get("fields"))
        for record in caplog.records
        if record.__dict__.get("event")
    ]
    return events[-1] if events else (None, None)


def test_a_complete_run_declares_which_variant_became_ready(monkeypatch, tmp_path, caplog):
    """The one statement that an artifact is usable, and it names the artifact.

    Nothing downstream may infer this from an exit code: a run that converted a
    subset exits zero too. The supervisor selects a variant on the strength of
    this event, so the event has to carry what it is talking about.
    """
    caplog.set_level("INFO")
    dest = tmp_path / "artifact"
    for component in REQUIRED:
        convert_one(monkeypatch, dest, component)

    event, fields = terminal_event(caplog)
    assert event == "prequantize_done"
    assert fields["model"] == "z-image"
    assert fields["bits"] == 4
    assert fields["variant_ready"] is True


def test_a_partial_run_declares_that_no_variant_is_ready(monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO")
    dest = tmp_path / "artifact"
    convert_one(monkeypatch, dest, "transformer")

    event, fields = terminal_event(caplog)
    assert event == "prequantize_partial"
    assert fields["model"] == "z-image"
    assert fields["variant_ready"] is False
    assert fields["completed"] == ["transformer"]
    assert set(fields["missing"]) == set(REQUIRED) - {"transformer"}
