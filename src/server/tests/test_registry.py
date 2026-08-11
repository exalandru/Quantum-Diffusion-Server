from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
from mflux_server.settings import ModelOverride, Settings, load_settings


def test_the_catalogue_exposes_ten_models():
    registry = build_registry({})
    assert set(registry) == {
        "ernie-image",
        "ernie-image-turbo",
        "fibo",
        "fibo-lite",
        "flux2-dev",
        "flux2-klein",
        "ideogram-4",
        "qwen-image-2512",
        "z-image",
        "z-image-turbo",
    }


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


def test_quantization_is_only_advertised_where_it_happens():
    """The runtime setting is published only where it changes something.

    `prequantized` used to carry this, plus two unrelated claims. The facts are
    now separate: whether the load-time setting does anything, and whether the
    model can be converted into a saved artifact.
    """
    # bf16 upstream: quantizing at load is real work.
    assert BASE_SPECS_BY_KEY["flux2-klein"].quantize == 8
    assert BASE_SPECS_BY_KEY["flux2-klein"].quantization.supports_quantization is True

    # Ideogram's fp8 layout: the heavy components are all `skip_quantization=True`.
    assert BASE_SPECS_BY_KEY["ideogram-4"].quantization.supports_quantization is False
    assert BASE_SPECS_BY_KEY["ideogram-4"].quantize is None

    # FLUX.2-dev loads a stored-quantized artifact, so the setting is equally
    # inert — but for a different reason, and unlike Ideogram it *can* be
    # converted, by the memory-bounded path.
    flux2_dev = BASE_SPECS_BY_KEY["flux2-dev"].quantization
    assert flux2_dev.supports_quantization is False
    assert flux2_dev.supports_prequantize is True
    assert flux2_dev.prequantize_strategy == "qds_memory_bounded"
    assert BASE_SPECS_BY_KEY["ideogram-4"].quantization.supports_prequantize is False


def test_flux2_edit_shares_weights_and_is_on_by_default():
    spec = BASE_SPECS_BY_KEY["flux2-klein"]
    assert spec.edit is not None
    assert spec.edit.shares_weights is True
    assert edit_enabled(spec) is True


def test_qwen_edit_is_off_by_default_due_to_separate_download():
    spec = BASE_SPECS_BY_KEY["qwen-image-2512"]
    assert spec.edit is not None
    assert spec.edit.shares_weights is False
    assert edit_enabled(spec) is False


def test_qwen_does_not_use_from_name():
    # from_name() would lose the scheduler's sigma_* values; we pass the
    # canonical factory plus the repo as model_path. And the path is needed at all
    # because `ModelConfig.qwen_image()` points at `Qwen/Qwen-Image`, the original
    # release, not the 2512 this key is named after.
    spec = BASE_SPECS_BY_KEY["qwen-image-2512"]
    assert spec.model_config_name == "qwen_image"
    assert spec.model_path == "Qwen/Qwen-Image-2512"
    # The raw bf16 repo, not the 8-bit conversion: only raw weights can honour
    # `default_quantize`.
    assert spec.quantization.supports_quantization is True


def test_qwen_follows_its_own_card_not_the_mflux_defaults():
    """The Qwen-Image-2512 card asks for 50 steps and cfg 4.0.

    mflux would give 20 steps (`MODEL_INFERENCE_STEPS["qwen"]`) and guidance 3.5
    (`GUIDANCE_SCALE`, its blanket default for every model). qwen-image is the one
    entry in the catalogue where the two disagree, so it is also the one that would
    silently regress if someone "harmonized" the table against mflux.
    """
    spec = BASE_SPECS_BY_KEY["qwen-image-2512"]
    assert spec.default_steps == 50
    assert spec.default_guidance == 4.0


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
            "qwen-image-2512": ModelOverride(enabled=False),
        }
    )
    assert "qwen-image-2512" not in registry
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


def test_the_code_default_model_is_usable_with_no_config():
    """A config-less run must serve its own default with nothing set up at all.

    That rules out more models than it looks: no gated repo (no HuggingFace token
    to obtain), no non-commercial licence to accept, no preparation step, and no
    JSON-only prompt format — a default that rejects "a red fox" is not a default.
    """
    settings = Settings()
    spec = settings.registry()[settings.default_model]
    assert settings.default_model == "z-image-turbo"
    assert spec.gated is False
    assert spec.license == "Apache-2.0"
    assert "text" in spec.prompt_formats


def test_the_shipped_config_says_what_the_readme_says():
    """Load the real `server-config.json`, not a fixture.

    This is the only test that catches a typo in the file we actually ship — the
    policies it applies on top of the catalogue live nowhere else.
    """
    settings = load_settings(Path(__file__).resolve().parent.parent / "server-config.json")
    registry = settings.registry()

    assert settings.default_model == "z-image-turbo"
    assert set(registry) == {"z-image-turbo", "ernie-image-turbo"}
    assert {spec.default_size for spec in registry.values()} == {"1280x720"}

    # The whole point of the enabled set: nothing to obtain, nothing to accept,
    # and nothing that refuses a plain-text prompt. Re-enabling `fibo-lite` — which
    # is gated, non-commercial *and* JSON-only — would fail right here.
    for spec in registry.values():
        assert spec.gated is False, spec.key
        assert spec.license == "Apache-2.0", spec.key
        assert "text" in spec.prompt_formats, spec.key
        assert spec.quantize == 4, spec.key


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
