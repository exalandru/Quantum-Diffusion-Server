"""The quantization contract the app consumes.

Declarative only: nothing here converts or saves a model. What is pinned is that
the published facts match what mflux will actually do, because the previous
single `prequantized` flag let the Configuration form offer bit depths that mflux
discards.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from qds.registry import (
    BASE_SPECS,
    BASE_SPECS_BY_KEY,
    QUANTIZE_CHOICES,
    STRATEGY_MFLUX_SAVE,
    STRATEGY_QDS_MEMORY_BOUNDED,
    build_registry,
    capability_for,
)
from qds.settings import ModelOverride, Settings


def test_every_builtin_model_publishes_an_internally_consistent_capability():
    for spec in BASE_SPECS:
        q = spec.quantization
        # Choices exist exactly when the capability they belong to is on.
        assert bool(q.quantize_choices) == q.supports_quantization, spec.key
        assert bool(q.prequantize_choices) == q.supports_prequantize, spec.key
        assert (q.prequantize_strategy is not None) == q.supports_prequantize, spec.key
        # Nothing may be offered that mflux does not accept.
        assert set(q.quantize_choices) <= set(QUANTIZE_CHOICES), spec.key
        assert set(q.prequantize_choices) <= set(QUANTIZE_CHOICES), spec.key
        # A capability that is off owes the user a reason.
        if not q.supports_quantization:
            assert q.note, f"{spec.key} disables the control without saying why"


def test_models_where_quantization_is_inert_publish_no_choices():
    for key in ("ideogram-4", "flux2-dev"):
        q = BASE_SPECS_BY_KEY[key].quantization
        assert q.supports_quantization is False
        assert q.quantize_choices == ()


def test_supported_models_publish_only_mflux_backed_choices():
    for key in ("flux2-klein", "qwen-image-2512", "z-image", "ernie-image", "fibo"):
        q = BASE_SPECS_BY_KEY[key].quantization
        assert q.supports_quantization is True
        assert q.quantize_choices == QUANTIZE_CHOICES


def test_flux2_dev_publishes_prequantization_with_its_own_strategy():
    """Its bf16 is ~111 GB, and the generic save CLI would route it to Flux2Klein."""
    q = BASE_SPECS_BY_KEY["flux2-dev"].quantization
    assert q.supports_prequantize is True
    assert q.prequantize_strategy == STRATEGY_QDS_MEMORY_BOUNDED
    assert q.prequantize_choices == QUANTIZE_CHOICES


def test_every_convertible_family_declares_the_memory_bounded_strategy():
    """One route, for the same reason FLUX.2-dev has always had it.

    These used to declare `mflux_save`: load the model whole, call `save_model`,
    hold every component resident to write them. Whether a model *fits* was never
    the right test for how to convert it, and their components were independently
    convertible all along — see `components.py`.
    """
    for key in ("flux2-klein", "qwen-image-2512", "z-image-turbo", "ernie-image-turbo", "fibo-lite"):
        assert (
            BASE_SPECS_BY_KEY[key].quantization.prequantize_strategy
            == STRATEGY_QDS_MEMORY_BOUNDED
        )


def test_the_superseded_strategy_name_survives_for_artifacts_that_recorded_it():
    """Markers written before this slice say `mflux_save`, and stay readable."""
    assert STRATEGY_MFLUX_SAVE == "mflux_save"


def test_ideogram_declares_no_conversion_at_all():
    """Saving it would stamp a level onto weights nothing quantized."""
    q = BASE_SPECS_BY_KEY["ideogram-4"].quantization
    assert q.supports_prequantize is False
    assert q.prequantize_strategy is None


def test_an_unknown_family_is_unsupported_rather_than_assumed():
    """Future local imports must fail closed until their family is identified."""
    q = capability_for("something-we-have-not-verified")
    assert q.supports_quantization is False
    assert q.supports_prequantize is False
    assert q.note


def test_a_per_model_override_cannot_advertise_ineffective_quantization():
    """P8: the override path used to bypass the check the global default made."""
    registry = build_registry(
        {"ideogram-4": ModelOverride(quantize=8), "flux2-dev": ModelOverride(quantize=4)},
        include_disabled=True,
    )
    assert registry["ideogram-4"].quantize is None
    # Normalised to the catalogue value, not to the requested 4.
    assert registry["flux2-dev"].quantize == BASE_SPECS_BY_KEY["flux2-dev"].quantize


def test_nothing_config_wide_can_change_a_model_s_precision():
    """The invariant whose absence let a global silently break a model.

    There used to be a `default_quantize`, and it *overwrote* each row rather
    than standing behind it — so a single number decided the precision of every
    model, including the ones that had chosen one. Anima at 4-bit produces
    visibly broken images while the same depth is harmless on a 20B, which is
    what makes precision a property of the model rather than of the install.

    `build_registry` therefore takes no config-wide quantization at all. This
    asserts the shape of the call, not just today's behaviour: a future
    keyword would fail here before it could reach a catalogue row.
    """
    import inspect

    assert "quantize" not in inspect.signature(build_registry).parameters

    registry = build_registry({}, include_disabled=True)
    for key, base in BASE_SPECS_BY_KEY.items():
        assert registry[key].quantize == base.quantize, key


def test_a_config_that_still_carries_the_removed_global_starts_and_says_so(
    tmp_path, monkeypatch, caplog
):
    """Pydantic ignores unknown keys, which is how a removed setting becomes a mystery."""
    import json as _json
    import logging

    config = tmp_path / "server-config.json"
    config.write_text(_json.dumps({"default_quantize": 4}), encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))
    from qds.settings import load_settings

    with caplog.at_level(logging.WARNING):
        settings = load_settings()

    # Started, rather than refused: an install that set this must not be bricked.
    assert settings.registry(include_disabled=True)["z-image"].quantize == (
        BASE_SPECS_BY_KEY["z-image"].quantize
    )
    assert any("default_quantize" in record.message for record in caplog.records)


def test_an_unsupported_bit_depth_is_rejected_by_the_backend(tmp_path, monkeypatch):
    """React is UX; a hand-edited config must not reach runtime unchecked.

    The per-model value is a field validator, so an impossible bit depth is a
    startup error rather than a silent no-op. `load_settings` builds the registry
    eagerly so that it reaches the user at that point and not on first request.
    """
    with pytest.raises(ValueError):
        Settings.model_validate({"models": {"z-image": {"quantize": 7}}})

    # ...and `load_settings` is what makes that reach a real config file.
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"models": {"z-image": {"quantize": 7}}}), encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))
    from qds.settings import load_settings

    with pytest.raises(ValueError):
        load_settings()


def test_capabilities_endpoint_matches_the_spec(client):
    published = client.get("/v1/capabilities").json()["models"]
    for key, entry in published.items():
        q = BASE_SPECS_BY_KEY[key].quantization
        assert entry["supports_quantization"] == q.supports_quantization, key
        assert entry["quantize_choices"] == list(q.quantize_choices), key
        assert entry["supports_prequantize"] == q.supports_prequantize, key
        assert entry["prequantize_choices"] == list(q.prequantize_choices), key
        assert entry["prequantize_strategy"] == q.prequantize_strategy, key


def test_react_keeps_no_quantization_table_of_its_own():
    """The whole point of the slice: one source of truth, and it is not the app."""
    # tests/ -> server/ -> src/ ; the dashboard is src/dashboard/src.
    src = pathlib.Path(__file__).resolve().parents[2] / "dashboard" / "src"
    if not src.is_dir():  # pragma: no cover - server-only checkout
        pytest.skip("dashboard sources not present")
    offenders = []
    for path in src.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        if "QUANTIZE_CHOICES" in text:
            offenders.append(str(path))
        # A literal bit-depth list is the shape that used to drift.
        if "3, 4, 5, 6, 8" in text:
            offenders.append(f"{path} (literal bit-depth list)")
    assert not offenders, f"React still owns quantization truth: {offenders}"


def test_the_catalogue_carries_capability_for_disabled_models_too(monkeypatch, tmp_path):
    """`/v1/capabilities` publishes enabled models only.

    The Configuration form lists the whole catalogue, so taking capability from
    that endpoint left every disabled row with a dead control and no reason —
    which is most of the catalogue on a default install.
    """
    from qds.fetch import cache_status

    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"models": {"ideogram-4": {"enabled": False}, "z-image": {"enabled": False}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    rows = {row["key"]: row for row in cache_status()}
    assert len(rows) == len(BASE_SPECS_BY_KEY)
    for key, row in rows.items():
        q = BASE_SPECS_BY_KEY[key].quantization
        assert row["quantization"]["supports_quantization"] == q.supports_quantization, key
        assert row["quantization"]["quantize_choices"] == list(q.quantize_choices), key
        assert row["quantization"]["prequantize_strategy"] == q.prequantize_strategy, key

    # Disabled, and still fully described.
    assert rows["ideogram-4"]["enabled"] is False
    assert rows["ideogram-4"]["quantization"]["note"]
    assert rows["z-image"]["enabled"] is False
    assert rows["z-image"]["quantization"]["quantize_choices"] == list(QUANTIZE_CHOICES)
