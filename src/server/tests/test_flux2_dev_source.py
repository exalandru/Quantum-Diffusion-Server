"""FLUX.2-dev's source is a repository, and its 8-bit artifact is a variant.

The defect these pin: the catalogue named a QDS-generated 8-bit artifact as the
model's `model_path`. That directory is written by our own converter, so the
model's *identity* was one of its own outputs — which meant the raw weights could
never be installed or located, the model read as installed when only a conversion
existed, and a saved copy could never be shown as what it is: one representation
among the possible ones.

Nothing here converts or downloads anything.
"""

from __future__ import annotations

import json

import pytest

from mflux_server import artifacts, importing
from mflux_server import availability as av
from mflux_server.fetch import cache_status
from mflux_server.registry import BASE_SPECS_BY_KEY, STRATEGY_QDS_MEMORY_BOUNDED
from mflux_server.settings import Settings

from .test_artifacts import write_component


def registry_with(models: dict) -> dict:
    """The effective catalogue for one `models` section, overrides applied."""
    return Settings.model_validate({"models": models}).registry(include_disabled=True)


RAW_REPO = "black-forest-labs/FLUX.2-dev"


def saved_variant(cache_root, bits=8, source=RAW_REPO):
    """A complete artifact in the configured cache directory."""
    dest = artifacts.artifact_dir("flux2-dev", source, bits, base=str(cache_root))
    for name in av.REQUIRED_COMPONENTS:
        write_component(dest, name, bits=str(bits))
    artifacts.write_record(
        dest,
        model_key="flux2-dev",
        family="flux2-dev",
        source=source,
        bits=bits,
        strategy=STRATEGY_QDS_MEMORY_BOUNDED,
        components=av.REQUIRED_COMPONENTS,
        required=av.REQUIRED_COMPONENTS,
    )
    return dest


# ── Source identity ────────────────────────────────────────────────────────


def test_the_catalogue_source_is_the_raw_repository():
    spec = BASE_SPECS_BY_KEY["flux2-dev"]
    assert spec.repo == RAW_REPO
    assert spec.model_path is None
    assert spec.source == RAW_REPO


# ── Install and Locate ─────────────────────────────────────────────────────


def test_a_missing_raw_source_offers_install_and_locate(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(tmp_path / "absent.json"))

    row = {entry["key"]: entry for entry in cache_status()}["flux2-dev"]
    assert row["availability"] == av.MISSING
    assert row["repo"] == RAW_REPO
    # `can_download` is the single authority the interface reads for both
    # Install and Locate.
    assert row["can_download"] is True
    assert row["local"] is False
    assert row["provenance"] == "built_in"


def test_locating_a_raw_flux2_dev_source_binds_the_built_in(tmp_path):
    """A raw FLUX.2-dev directory declares klein's class — and must still bind.

    Both models are `Flux2Transformer2DModel`; what separates them is
    configuration a directory need not declare. Refusing the comparison would
    make FLUX.2-dev the one built-in that can never be located.
    """
    source = tmp_path / "flux2-dev-weights"
    (source / "transformer").mkdir(parents=True)
    (source / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Flux2Transformer2DModel"}), encoding="utf-8"
    )

    verdict = importing.locate(str(source), "flux2-dev")
    assert verdict.ok is True
    assert verdict.model == "flux2-dev"
    assert verdict.class_name == "Flux2Transformer2DModel"
    # No proof of provenance from a loose directory, and it says so rather than
    # claiming the repository.
    assert verdict.repo_verified is False


def test_locating_the_wrong_family_is_still_refused(tmp_path):
    source = tmp_path / "z-image-weights"
    (source / "transformer").mkdir(parents=True)
    (source / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "ZImageTransformer2DModel"}), encoding="utf-8"
    )

    verdict = importing.locate(str(source), "flux2-dev")
    assert verdict.ok is False
    assert verdict.availability == "incompatible"
    assert "z-image" in (verdict.reason or "")


def test_locating_creates_no_imported_row(tmp_path):
    """Locate tells a built-in where its weights are. It mints nothing."""
    source = tmp_path / "flux2-dev-weights"
    (source / "transformer").mkdir(parents=True)
    (source / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Flux2Transformer2DModel"}), encoding="utf-8"
    )
    importing.locate(str(source), "flux2-dev")

    registry = registry_with({"flux2-dev": {"model_path": str(source)}})
    assert [key for key in registry if key.startswith(importing.KEY_PREFIX)] == []
    assert registry["flux2-dev"].provenance == "built_in"
    assert registry["flux2-dev"].key == "flux2-dev"


def test_clearing_the_override_restores_the_raw_repository(tmp_path):
    located = str(tmp_path / "weights")
    registry = registry_with({"flux2-dev": {"model_path": located}})
    assert registry["flux2-dev"].source == located

    registry = registry_with({"flux2-dev": {"model_path": None}})
    assert registry["flux2-dev"].source == RAW_REPO


# ── Source and variant are two different facts ─────────────────────────────


def test_a_saved_variant_does_not_make_the_source_installed(monkeypatch, tmp_path):
    """The state that must stay expressible: source absent, variant present."""
    config = tmp_path / "server-config.json"
    cache = tmp_path / "cache"
    saved_variant(cache)
    config.write_text(json.dumps({"storage": {"cache_dir": str(cache)}}), encoding="utf-8")
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    row = {entry["key"]: entry for entry in cache_status()}["flux2-dev"]
    # The raw weights are not in this cache, and an artifact is not the source.
    assert row["availability"] == av.MISSING
    assert row["can_download"] is True
    # And the 8-bit copy is listed for what it is.
    assert [v["bits"] for v in row["variants"]] == [8]
    # Source size unknown, variant size measured: two different questions.
    assert row["disk"]["source_bytes"] is None
    assert row["disk"]["total_bytes"] > 0


def test_a_saved_variant_is_not_selected_merely_by_existing(tmp_path):
    cache = tmp_path / "cache"
    saved_variant(cache)
    settings = Settings.model_validate({"storage": {"cache_dir": str(cache)}})
    spec = settings.registry(include_disabled=True)["flux2-dev"]

    assert spec.prequantized_variant is None
    assert spec.effective_model_path is None, "nothing may be activated by existing"


def test_activating_a_variant_resolves_to_the_configured_cache(tmp_path):
    cache = tmp_path / "cache"
    dest = saved_variant(cache)
    settings = Settings.model_validate(
        {
            "storage": {"cache_dir": str(cache)},
            "models": {"flux2-dev": {"prequantized_variant": 8}},
        }
    )
    spec = settings.registry(include_disabled=True)["flux2-dev"]

    assert spec.source == RAW_REPO
    assert spec.effective_model_path == str(dest)


# ── Capability is unchanged ────────────────────────────────────────────────


def test_quantization_capability_is_exactly_what_it_was():
    """Fixed precision at runtime, convertible into saved copies. Both, still."""
    capability = BASE_SPECS_BY_KEY["flux2-dev"].quantization
    assert capability.supports_quantization is False
    assert capability.quantize_choices == ()
    assert capability.supports_prequantize is True
    assert capability.prequantize_strategy == STRATEGY_QDS_MEMORY_BOUNDED
    assert capability.note and "pre-quantized artifact" in capability.note


def test_generating_from_the_raw_source_is_refused_with_the_reason(monkeypatch):
    """111 GB of bf16 does not fit, and the refusal says what to do instead."""
    from mflux_server.errors import APIError
    from mflux_server.registry import _require_local_artifact

    spec = BASE_SPECS_BY_KEY["flux2-dev"]
    with pytest.raises(APIError) as raised:
        _require_local_artifact(spec, None)
    message = raised.value.message
    assert "does not fit" in message
    assert "Quantization dialog" in message or "prequantized_variant" in message
