"""Pre-quantization guards. Nothing here converts anything.

`prequantize.py` had no tests at all, and it is the module that writes tens of
gigabytes and used to leave a directory the rest of the app called ready.
"""

from __future__ import annotations

import pytest

from mflux_server import artifacts
from mflux_server import availability as av
from mflux_server.prequantize import (
    COMPONENT_ORDER,
    DISK_MARGIN_GB,
    InsufficientDisk,
    convert,
    free_gb,
    required_free_gb,
)


def test_the_peak_is_the_largest_source_beside_what_is_already_written():
    # Largest-first ordering exists so the peak is bounded; the estimate has to
    # model that, not just sum everything.
    full = required_free_gb(COMPONENT_ORDER)
    assert 100 <= full <= 125, full
    assert required_free_gb(["vae"]) < required_free_gb(["text_encoder"]) < full


def test_the_margin_is_explicit_and_included():
    assert required_free_gb(["vae"]) == pytest.approx(0.4 + 0.4 + DISK_MARGIN_GB, abs=0.1)


def test_component_order_is_respected_whatever_order_is_requested():
    assert required_free_gb(["vae", "transformer"]) == required_free_gb(["transformer", "vae"])


def test_free_space_is_measured_on_a_destination_that_does_not_exist_yet(tmp_path):
    # The destination is created by the conversion, so the check has to walk up.
    assert free_gb(tmp_path / "not" / "created" / "yet") > 0


class _Args:
    """The parsed CLI shape `convert` dispatches on."""

    def __init__(self, dest, components, *, model="flux2-dev", bits=8):
        self.model = model
        self.dest = str(dest)
        self.components = list(components)
        self.repo = "black-forest-labs/FLUX.2-dev"
        self.bits = bits


def test_the_disk_check_refuses_before_anything_heavy_starts(tmp_path, monkeypatch):
    """The point of a preflight: fail now, not two hours and 60 GB in."""
    monkeypatch.setattr("mflux_server.prequantize.free_gb", lambda _: 1.0)
    monkeypatch.delenv("MFLUX_SERVER_CONFIG", raising=False)
    called = []
    monkeypatch.setattr(
        "mflux_server.prequantize.convert_component",
        lambda *a, **k: called.append(a),
    )

    dest = tmp_path / "artifact"
    with pytest.raises(InsufficientDisk) as raised:
        convert(_Args(dest, COMPONENT_ORDER))

    assert not called, "no component may be converted after the check fails"
    assert not dest.exists(), "the destination must not even be created"
    assert "GB free" in str(raised.value)
    # The assumption behind the number is stated rather than implied.
    assert "purged" in str(raised.value)


def test_a_finished_conversion_writes_the_marker_last(tmp_path, monkeypatch):
    monkeypatch.setattr("mflux_server.prequantize.free_gb", lambda _: 10_000.0)
    monkeypatch.delenv("MFLUX_SERVER_CONFIG", raising=False)

    dest = tmp_path / "artifact"

    def fake_convert(name, *, spec, repo, dest, bits):
        component = dest / name
        component.mkdir(parents=True, exist_ok=True)
        (component / "0.safetensors").write_bytes(b"tensor")
        (component / av.INDEX_FILE).write_text(
            '{"metadata": {"quantization_level": "8"}, "weight_map": {"w": "0.safetensors"}}',
            encoding="utf-8",
        )
        # Every family declares one, and an artifact without it cannot load.
        (dest / "tokenizer").mkdir(parents=True, exist_ok=True)
        (dest / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return 6

    monkeypatch.setattr("mflux_server.prequantize.convert_component", fake_convert)

    assert convert(_Args(dest, COMPONENT_ORDER)) == 0
    assert (dest / av.COMPLETION_MARKER).is_file()
    record = artifacts.read_record(dest)
    assert record is not None
    # Identity, not just completion: which model, which source, which precision.
    assert record.model_key == "flux2-dev"
    assert record.bits == 8
    assert record.strategy == "qds_memory_bounded"
    assert artifacts.artifact_state(dest, expect_bits=8)[0] == av.PRESENT


def test_a_partial_conversion_leaves_no_marker_and_is_not_present(tmp_path, monkeypatch):
    """Cancelled or failed output is preserved, but must never read as ready."""
    monkeypatch.setattr("mflux_server.prequantize.free_gb", lambda _: 10_000.0)
    monkeypatch.delenv("MFLUX_SERVER_CONFIG", raising=False)

    dest = tmp_path / "artifact"

    def fake_convert(name, *, spec, repo, dest, bits):
        component = dest / name
        component.mkdir(parents=True, exist_ok=True)
        (component / "0.safetensors").write_bytes(b"tensor")
        (component / av.INDEX_FILE).write_text(
            '{"metadata": {"quantization_level": "8"}, "weight_map": {"w": "0.safetensors"}}',
            encoding="utf-8",
        )
        # Every family declares one, and an artifact without it cannot load.
        (dest / "tokenizer").mkdir(parents=True, exist_ok=True)
        (dest / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return 6

    monkeypatch.setattr("mflux_server.prequantize.convert_component", fake_convert)

    # Only one component requested: a real artifact needs all three. Converting
    # a subset is a legitimate way to work through a model in stages, so the run
    # *succeeds* — what it must not do is leave anything claiming to be usable.
    assert convert(_Args(dest, ["vae"])) == 0
    assert not (dest / av.COMPLETION_MARKER).exists()
    assert av.flux2_dev_artifact_state(str(dest))[0] == av.PARTIAL
    assert artifacts.artifact_state(dest)[0] == av.PARTIAL

    # And the one component that did convert is recorded, so the next run can
    # continue rather than start again.
    progress = artifacts.read_progress(dest)
    assert progress is not None
    assert progress.completed() == ("vae",)
    assert progress.bits == 8
