"""Saved prequantized artifacts: identity, dispatch, activation.

Nothing here converts a model. Artifacts are synthesised in the shape
`ModelSaver` writes, which is the same shape Slice 3 validated against a real
58 GB artifact on disk.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from qds import artifacts, components
from qds import availability as av
from qds.prequantize import convert
from qds.registry import (
    BASE_SPECS_BY_KEY,
    STRATEGY_MFLUX_SAVE,
    STRATEGY_QDS_MEMORY_BOUNDED,
)
from qds.settings import Settings


def write_component(root, name, *, bits="8"):
    component = root / name
    component.mkdir(parents=True, exist_ok=True)
    (component / "0.safetensors").write_bytes(b"tensor")
    (component / av.INDEX_FILE).write_text(
        json.dumps(
            {
                "metadata": {"quantization_level": bits, "mflux_version": "0.18.0"},
                "weight_map": {"w": "0.safetensors"},
            }
        ),
        encoding="utf-8",
    )


def make_artifact(base, key, source, bits, *, components=("vae", "transformer", "text_encoder")):
    dest = artifacts.artifact_dir(key, source, bits, base=str(base))
    dest.mkdir(parents=True, exist_ok=True)
    for name in components:
        write_component(dest, name, bits=str(bits))
    artifacts.write_record(
        dest,
        model_key=key,
        family="z-image",
        source=source,
        bits=bits,
        strategy=STRATEGY_MFLUX_SAVE,
        components=tuple(components),
    )
    return dest


def write_tokenizer(root):
    """What every family declares and every real artifact carries.

    Not a component and never quantized, but an artifact without it cannot be
    loaded — so a conversion is not complete until it is there.
    """
    tokenizer = root / "tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _Args:
    def __init__(self, model, bits, *, dest=None, components=None):
        self.model = model
        self.bits = bits
        self.dest = str(dest) if dest else None
        self.components = list(components) if components else None
        self.repo = None


def component_writer(seen, *, stored_bits=None, tokenizer=True):
    """A stand-in for one component's conversion, writing what the real one writes.

    `stored_bits` differing from the requested depth is the case where mflux
    stamped something other than what was asked for, which must never be recorded
    as the request.
    """

    def fake_component(name, *, spec, repo, dest, bits):
        seen.append(name)
        write_component(dest, name, bits=str(bits if stored_bits is None else stored_bits))
        if tokenizer:
            write_tokenizer(dest)
        return 6

    return fake_component


# ── Dispatch ───────────────────────────────────────────────────────────────


def test_an_unsupported_model_cannot_start_a_conversion(monkeypatch):
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    with pytest.raises(ValueError) as raised:
        convert(_Args("ideogram-4", 8))
    assert "cannot be pre-quantized" in str(raised.value)


def test_an_unpublished_bit_depth_is_refused_before_any_work(monkeypatch, tmp_path):
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    called = []
    monkeypatch.setattr("qds.prequantize.convert_component", component_writer(called))
    with pytest.raises(ValueError) as raised:
        convert(_Args("z-image", 7, dest=tmp_path / "out"))
    assert "not available" in str(raised.value)
    assert not called


def test_a_component_this_model_does_not_have_is_named_rather_than_ignored(monkeypatch, tmp_path):
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    called = []
    monkeypatch.setattr("qds.prequantize.convert_component", component_writer(called))
    with pytest.raises(ValueError) as raised:
        convert(_Args("z-image", 4, dest=tmp_path / "out", components=["unet"]))
    assert "unet" in str(raised.value)
    assert not called


def test_a_generic_model_converts_component_by_component(monkeypatch, tmp_path):
    """The slice's whole point: not only FLUX.2-dev takes the bounded route.

    z-image used to be loaded whole and written with `save_model`. It now goes
    through the same one-component-at-a-time path, and records that it did.
    """
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)
    seen: list[str] = []
    monkeypatch.setattr("qds.prequantize.convert_component", component_writer(seen))

    assert convert(_Args("z-image", 4, dest=tmp_path / "out")) == 0
    # Every required component, one call each, largest first.
    assert seen == list(components.required_components("z-image"))
    record = artifacts.read_record(tmp_path / "out")
    assert record.strategy == STRATEGY_QDS_MEMORY_BOUNDED
    assert set(record.components) == set(components.required_components("z-image"))


def test_flux2_dev_still_converts_component_by_component(monkeypatch, tmp_path):
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)
    seen: list[str] = []
    monkeypatch.setattr("qds.prequantize.convert_component", component_writer(seen))

    assert convert(_Args("flux2-dev", 8, dest=tmp_path / "out")) == 0
    assert set(seen) == set(av.REQUIRED_COMPONENTS)
    assert artifacts.read_record(tmp_path / "out").strategy == STRATEGY_QDS_MEMORY_BOUNDED


def test_capability_choices_and_the_accepted_choices_cannot_drift(monkeypatch, tmp_path):
    """Anything the capability publishes must be accepted, and nothing else."""
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)
    monkeypatch.setattr(
        "qds.prequantize.convert_component",
        component_writer([], tokenizer=False),
    )
    spec = BASE_SPECS_BY_KEY["z-image"]
    for bits in spec.quantization.prequantize_choices:
        # Refused later for the missing tokenizer, but never for the bit depth.
        assert convert(_Args("z-image", bits, dest=tmp_path / f"out{bits}")) == 1
    with pytest.raises(ValueError) as raised:
        convert(_Args("z-image", 7, dest=tmp_path / "out7"))
    assert "not available" in str(raised.value)


# ── Identity ───────────────────────────────────────────────────────────────


def test_identity_includes_source_and_bits(tmp_path):
    dest = make_artifact(tmp_path, "z-image", "mlx-community/Z-Image-bf16", 4)
    record = artifacts.read_record(dest)
    assert record.model_key == "z-image"
    assert record.source == "mlx-community/Z-Image-bf16"
    assert record.bits == 4
    assert record.strategy == STRATEGY_MFLUX_SAVE
    assert record.marker_version == artifacts.MARKER_VERSION


def test_a_changed_source_invalidates_the_old_artifact(tmp_path):
    dest = make_artifact(tmp_path, "z-image", "mlx-community/Z-Image-bf16", 4)

    same, _ = artifacts.artifact_state(dest, expect_source="mlx-community/Z-Image-bf16", expect_bits=4)
    other, detail = artifacts.artifact_state(dest, expect_source="someone/else", expect_bits=4)

    assert same == av.PRESENT
    assert other == av.MISSING
    assert "no longer this model's source" in (detail or "")


def test_two_bit_depths_of_one_source_do_not_collide(tmp_path):
    four = make_artifact(tmp_path, "z-image", "mlx-community/Z-Image-bf16", 4)
    eight = make_artifact(tmp_path, "z-image", "mlx-community/Z-Image-bf16", 8)

    assert four != eight
    assert artifacts.artifact_state(four, expect_bits=4)[0] == av.PRESENT
    assert artifacts.artifact_state(eight, expect_bits=8)[0] == av.PRESENT
    # Asking one for the other's precision is a miss, not a silent substitution.
    assert artifacts.artifact_state(four, expect_bits=8)[0] == av.MISSING


def test_different_sources_get_different_directories(tmp_path):
    a = artifacts.artifact_dir("z-image", "mlx-community/Z-Image-bf16", 4, base=str(tmp_path))
    b = artifacts.artifact_dir("z-image", "someone/else", 4, base=str(tmp_path))
    assert a != b


# ── Validation ─────────────────────────────────────────────────────────────


def test_partial_output_never_becomes_valid(tmp_path):
    dest = artifacts.artifact_dir("z-image", "src", 4, base=str(tmp_path))
    dest.mkdir(parents=True)
    write_component(dest, "vae", bits="4")  # one component, no marker
    assert artifacts.artifact_state(dest)[0] != av.PRESENT


def test_the_marker_is_only_written_after_validation(monkeypatch, tmp_path):
    """A conversion that writes nothing usable must not be recorded as done."""
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)
    monkeypatch.setattr(
        "qds.prequantize.convert_component",
        lambda name, *, spec, repo, dest, bits: 0,  # returns, writes nothing
    )

    dest = tmp_path / "out"
    assert convert(_Args("z-image", 4, dest=dest)) == 1
    assert not (dest / av.COMPLETION_MARKER).exists()
    # And nothing was recorded as converted either.
    assert artifacts.read_progress(dest) is None


def test_stored_bits_must_equal_the_requested_bits(monkeypatch, tmp_path):
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)
    # Asked for 4, the weights say 8: never record the request as truth.
    monkeypatch.setattr(
        "qds.prequantize.convert_component",
        component_writer([], stored_bits=8),
    )
    dest = tmp_path / "out"
    assert convert(_Args("z-image", 4, dest=dest)) == 1
    assert not (dest / av.COMPLETION_MARKER).exists()


def test_a_legacy_flux2_dev_artifact_is_still_recognized(tmp_path):
    """No marker, pre-identity layout — the Slice 3 rules still accept it."""
    legacy = tmp_path / "flux2-dev-mlx-8bit"
    for name in av.REQUIRED_COMPONENTS:
        write_component(legacy, name, bits="8")
    assert not (legacy / av.COMPLETION_MARKER).exists()
    assert artifacts.artifact_state(legacy)[0] == av.PRESENT

    variants = artifacts.discover_variants("flux2-dev", str(legacy), base=str(tmp_path / "other"))
    assert [v.bits for v in variants] == [8]
    assert variants[0].legacy is True


# ── Discovery and activation ───────────────────────────────────────────────


def test_discovery_reports_only_validated_artifacts(tmp_path):
    source = "mlx-community/Z-Image-bf16"
    make_artifact(tmp_path, "z-image", source, 4)
    # A half-written 8-bit attempt, and an artifact of a different source.
    partial = artifacts.artifact_dir("z-image", source, 8, base=str(tmp_path))
    partial.mkdir(parents=True)
    write_component(partial, "vae", bits="8")
    make_artifact(tmp_path, "z-image", "someone/else", 6)

    found = artifacts.discover_variants("z-image", source, base=str(tmp_path))
    assert [v.bits for v in found] == [4]


def test_activation_resolves_only_the_matching_current_source(monkeypatch, tmp_path):
    settings = Settings.model_validate({"models": {"z-image": {"prequantized_variant": 4}}})
    spec = settings.registry(include_disabled=True)["z-image"]

    assert spec.source == "mlx-community/Z-Image-bf16"  # `model_path` untouched
    assert spec.prequantized_variant == 4
    assert str(artifacts.source_digest(spec.source)) in str(spec.effective_model_path)


def test_clearing_the_variant_restores_the_source_representation():
    active = Settings.model_validate({"models": {"z-image": {"prequantized_variant": 8}}})
    cleared = Settings.model_validate({"models": {"z-image": {"prequantized_variant": None}}})

    assert cleared.registry(include_disabled=True)["z-image"].effective_model_path == (
        BASE_SPECS_BY_KEY["z-image"].model_path
    )
    assert active.registry(include_disabled=True)["z-image"].effective_model_path != (
        BASE_SPECS_BY_KEY["z-image"].model_path
    )


def test_a_missing_active_variant_is_an_explicit_error_not_a_fallback():
    """Silently generating from the source would be the dangerous outcome."""
    from qds.errors import APIError
    from qds.registry import _require_variant

    settings = Settings.model_validate({"models": {"z-image": {"prequantized_variant": 5}}})
    spec = settings.registry(include_disabled=True)["z-image"]

    with pytest.raises(APIError) as raised:
        _require_variant(spec)
    assert raised.value.code == "variant_not_prepared"
    assert "5-bit" in raised.value.message


def test_react_owns_no_prequantization_bit_depth_truth():
    src = pathlib.Path(__file__).resolve().parents[2] / "dashboard" / "src"
    if not src.is_dir():  # pragma: no cover
        pytest.skip("dashboard sources not present")
    offenders = [
        str(path)
        for path in src.rglob("*.tsx")
        if "prequantize_choices" not in path.read_text(encoding="utf-8")
        and "3, 4, 5, 6, 8" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders
