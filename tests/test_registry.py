from __future__ import annotations

from dataclasses import replace

import pytest

from mflux_server.errors import APIError
from mflux_server.registry import (
    BASE_SPECS_BY_KEY,
    build_registry,
    edit_enabled,
    load_model,
    normalize_dimension,
    parse_size,
)
from mflux_server.settings import ModelOverride, Settings


def test_default_registry_exposes_the_five_models():
    registry = build_registry({})
    assert set(registry) == {"flux2-dev", "flux2-klein", "qwen-image", "z-image", "z-image-turbo"}


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1024x1024", (1024, 1024)),
        # The historical bug: mflux truncated 1080 to 1072 without saying so.
        ("1920x1080", (1920, 1072)),
        ("1280x720", (1280, 720)),
        ("512X512", (512, 512)),
    ],
)
def test_parse_size_normalizes_to_a_multiple_of_16(size, expected):
    assert parse_size(size) == expected


@pytest.mark.parametrize("size", ["auto", "1024", "axb", "1024x", ""])
def test_parse_size_rejects_invalid_formats(size):
    with pytest.raises(ValueError):
        parse_size(size)


def test_dimension_below_the_minimum_step_is_rejected():
    # mflux would settle for a warning and then produce 0.
    with pytest.raises(ValueError):
        normalize_dimension(15)


def test_flux2_klein_is_distilled():
    spec = BASE_SPECS_BY_KEY["flux2-klein"]
    assert spec.supports_negative_prompt is False
    assert spec.supports_guidance is False
    assert spec.default_guidance == 1.0
    # Distilled model: 4 steps, not the prototype's 20.
    assert spec.default_steps == 4
    assert spec.scheduler == "flow_match_euler_discrete"


def test_flux2_edit_shares_weights_and_is_on_by_default():
    spec = BASE_SPECS_BY_KEY["flux2-klein"]
    assert spec.edit is not None
    assert spec.edit.shares_weights is True
    assert edit_enabled(spec) is True


def test_qwen_edit_is_off_by_default_due_to_separate_download():
    spec = BASE_SPECS_BY_KEY["qwen-image"]
    assert spec.edit is not None
    assert spec.edit.shares_weights is False
    assert edit_enabled(spec) is False


def test_qwen_does_not_use_from_name():
    # from_name() would lose the scheduler's sigma_* values; we pass the
    # canonical factory plus the repo as model_path.
    spec = BASE_SPECS_BY_KEY["qwen-image"]
    assert spec.model_config_name == "qwen_image"
    assert spec.model_path == "mlx-community/Qwen-Image-2512-8bit"


def test_flux2_dev_is_guidance_distilled_and_prequantized():
    spec = BASE_SPECS_BY_KEY["flux2-dev"]
    # Guidance is an embedded scalar, not CFG: configurable but with no
    # negative prompt.
    assert spec.supports_guidance is True
    assert spec.supports_negative_prompt is False
    assert spec.default_guidance == 4.0
    # Base model: 50 steps, and 1024² to avoid paying for area on a 32B.
    assert spec.default_steps == 50
    assert (spec.default_width, spec.default_height) == (1024, 1024)
    # The upstream repo ships bf16: loading goes through a local artifact.
    assert spec.quantize == 8
    assert spec.model_path is not None and spec.model_path.endswith("flux2-dev-mlx-8bit")
    # Multi-image editing is not implemented.
    assert spec.edit is None


def test_flux2_dev_refuses_to_load_without_a_prequantized_artifact(tmp_path):
    # Without this guard, mflux would fall back to the bf16 repo and attempt an
    # on-the-fly quantization of ~111 GB.
    spec = replace(BASE_SPECS_BY_KEY["flux2-dev"], model_path=str(tmp_path / "absent"))
    with pytest.raises(APIError, match="prequantize") as raised:
        load_model(spec)
    assert raised.value.status_code == 503
    assert raised.value.code == "model_not_prepared"


def test_model_path_override():
    registry = build_registry({"flux2-dev": ModelOverride(model_path="/models/flux2-dev")})
    assert registry["flux2-dev"].model_path == "/models/flux2-dev"


def test_overrides_are_applied():
    registry = build_registry(
        {
            "z-image": ModelOverride(default_size="1024x1024", default_steps=30, default_guidance=5.0),
            "qwen-image": ModelOverride(enabled=False),
        }
    )
    assert "qwen-image" not in registry
    spec = registry["z-image"]
    assert (spec.default_width, spec.default_height) == (1024, 1024)
    assert spec.default_steps == 30
    assert spec.default_guidance == 5.0


# ── Config-wide resolution ─────────────────────────────────────────────────


def test_global_default_size_applies_to_every_model():
    registry = build_registry({}, default_size="1024x1024")
    assert {(spec.default_width, spec.default_height) for spec in registry.values()} == {(1024, 1024)}


def test_a_per_model_size_wins_over_the_global_one():
    """The escape hatch: one knob for everything, minus the model you pin.

    A 32B does not want the same area as a distilled 4B, so `flux2-dev` has to
    remain overridable without giving up the global setting.
    """
    registry = build_registry(
        {"flux2-dev": ModelOverride(default_size="768x768")},
        default_size="1920x1080",
    )
    assert (registry["flux2-dev"].default_width, registry["flux2-dev"].default_height) == (768, 768)
    # 1080 is truncated to 1072, like any other size.
    assert (registry["z-image"].default_width, registry["z-image"].default_height) == (1920, 1072)


def test_no_global_size_leaves_the_catalogue_untouched():
    catalogue = build_registry({})
    assert build_registry({}, default_size=None) == catalogue


@pytest.mark.parametrize("size", ["auto", "1024", "axb", "1024x"])
def test_an_invalid_global_size_is_rejected(size):
    # Rejected at startup rather than on the first request: `load_settings` calls
    # `registry()` eagerly for exactly this reason.
    with pytest.raises(ValueError):
        Settings.model_validate({"default_size": size}).registry()


def test_an_empty_global_size_means_unset():
    # Same convention as `api_key` and `log_file`: an empty string is how an
    # environment variable says "leave this alone", so it must not be read as a
    # malformed size.
    assert Settings.model_validate({"default_size": ""}).registry() == build_registry({})


def test_the_global_size_reaches_settings_registry():
    settings = Settings.model_validate({"default_size": "512x512"})
    spec = settings.registry()["flux2-klein"]
    assert (spec.default_width, spec.default_height) == (512, 512)
    assert spec.default_size == "512x512"


def test_overriding_guidance_on_a_distilled_model_fails():
    with pytest.raises(ValueError, match="guidance"):
        build_registry({"flux2-klein": ModelOverride(default_guidance=3.5)})


def test_unknown_model_in_the_config():
    with pytest.raises(ValueError, match="Unknown models"):
        build_registry({"sdxl": ModelOverride()})


def test_default_model_must_exist():
    with pytest.raises(ValueError):
        Settings.model_validate({"default_model": "sdxl"})


def test_default_model_cannot_be_disabled():
    with pytest.raises(ValueError):
        Settings.model_validate({"default_model": "z-image", "models": {"z-image": {"enabled": False}}})


def test_non_local_binding_requires_an_api_key():
    with pytest.raises(ValueError, match="api_key"):
        Settings.model_validate({"server": {"host": "0.0.0.0"}})

    settings = Settings.model_validate({"server": {"host": "0.0.0.0", "api_key": "secret"}})
    assert settings.server.is_loopback is False
