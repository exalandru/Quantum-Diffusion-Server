"""Ideogram 4 is a model QDS ships, so its sources must be recognisable.

The defect: pointing Locate at a genuine Ideogram 4 directory answered
`Unsupported model architecture 'Ideogram4Transformer2DModel'`, listing five
families that did not include the one the catalogue already contains. The class
was simply missing from the identity table.

The architecture string here is not assumed. It was read from
`ideogram-ai/ideogram-4-fp8`'s own `transformer/config.json` — the repository the
catalogue names — by fetching that one file, and it matches the value the failing
application reported from a real local directory.

Recognising the family says nothing about quantization, and the tests below hold
that line: Ideogram 4 keeps exactly the capability Slice 5 verified.
"""

from __future__ import annotations

import json

from mflux_server import importing
from mflux_server.registry import BASE_SPECS_BY_KEY, capability_for
from mflux_server.settings import Settings

CLASS_NAME = "Ideogram4Transformer2DModel"


def source_dir(root, class_name=CLASS_NAME, *, name="ideogram-4"):
    path = root / name
    (path / "transformer").mkdir(parents=True)
    (path / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": class_name}), encoding="utf-8"
    )
    return path


# ── Identity ───────────────────────────────────────────────────────────────


def test_the_architecture_maps_to_the_family_the_catalogue_already_has():
    assert importing.CLASS_NAME_TO_FAMILY[CLASS_NAME] == "ideogram4"
    assert BASE_SPECS_BY_KEY["ideogram-4"].family == "ideogram4"


def test_a_real_ideogram_source_is_detected_rather_than_refused(tmp_path):
    verdict = importing.inspect(str(source_dir(tmp_path)))
    assert verdict.ok is True
    assert verdict.family == "ideogram4"
    assert verdict.class_name == CLASS_NAME
    # A family with no built-in profile could not supply generation defaults;
    # this one has one, which is why the mapping is usable at all.
    assert "ideogram-4" in verdict.profiles


def test_locating_an_ideogram_source_binds_the_built_in(tmp_path):
    verdict = importing.locate(str(source_dir(tmp_path)), "ideogram-4")
    assert verdict.ok is True
    assert verdict.model == "ideogram-4"
    assert verdict.family == "ideogram4"
    assert verdict.reason is None


def test_an_incompatible_source_is_still_refused(tmp_path):
    verdict = importing.locate(
        str(source_dir(tmp_path, "ZImageTransformer2DModel")), "ideogram-4"
    )
    assert verdict.ok is False
    assert verdict.availability == "incompatible"
    assert "z-image" in (verdict.reason or "")


def test_an_unknown_architecture_still_names_itself(tmp_path):
    verdict = importing.inspect(str(source_dir(tmp_path, "SomeFutureTransformer")))
    assert verdict.ok is False
    assert "SomeFutureTransformer" in (verdict.reason or "")
    # And the list it offers now includes the family that was missing.
    assert "ideogram4" in (verdict.reason or "")


def test_locating_creates_no_imported_row(tmp_path):
    path = source_dir(tmp_path)
    importing.locate(str(path), "ideogram-4")
    registry = Settings.model_validate(
        {"models": {"ideogram-4": {"model_path": str(path)}}}
    ).registry(include_disabled=True)

    assert [key for key in registry if key.startswith(importing.KEY_PREFIX)] == []
    assert registry["ideogram-4"].provenance == "built_in"
    assert registry["ideogram-4"].source == str(path)


# ── Capability is a different question ─────────────────────────────────────


def test_recognising_the_family_grants_it_no_quantization():
    """Slice 5's verdict, unchanged: the knob is inert and conversion is refused."""
    capability = capability_for("ideogram4")
    assert capability.supports_quantization is False
    assert capability.quantize_choices == ()
    assert capability.supports_prequantize is False
    assert capability.prequantize_choices == ()
    assert capability.prequantize_strategy is None
    assert capability.note and "FP8" in capability.note


def test_the_catalogue_entry_publishes_no_component_list():
    """Fails closed on conversion while being perfectly locatable."""
    from mflux_server import components

    assert components.is_supported("ideogram4") is False
    assert components.payload("ideogram4") == []


def test_a_located_ideogram_reports_fixed_precision_and_no_conversion(monkeypatch, tmp_path):
    from mflux_server.fetch import cache_status

    path = source_dir(tmp_path)
    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"models": {"ideogram-4": {"model_path": str(path)}}}), encoding="utf-8"
    )
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    row = {entry["key"]: entry for entry in cache_status()}["ideogram-4"]
    assert row["local"] is True
    assert row["availability"] == "present"
    assert row["quantization"]["supports_quantization"] is False
    assert row["quantization"]["supports_prequantize"] is False
    assert row["quantization"]["prequantize_components"] == []
    assert row["variants"] == []
