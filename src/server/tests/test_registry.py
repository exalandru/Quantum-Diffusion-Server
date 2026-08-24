from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qds.errors import APIError
from qds.registry import (
    BASE_SPECS,
    BASE_SPECS_BY_KEY,
    build_registry,
    edit_enabled,
    family_structure,
    latent_creator_for,
    load_model,
    normalize_dimension,
    parse_size,
)
from qds.settings import ModelOverride, Settings, load_settings


def test_the_catalogue_exposes_seventeen_models():
    registry = build_registry({})
    assert set(registry) == {
        "anima",
        "anima-turbo",
        "ernie-image",
        "ernie-image-turbo",
        "fibo",
        "fibo-lite",
        "flux2-dev",
        "flux2-klein",
        "ideogram-4",
        "krea-2-turbo",
        "qwen-image-2512",
        "qwen-image-flash",
        "sd35-large",
        "sd35-large-turbo",
        "sd35-medium",
        "z-image",
        "z-image-turbo",
    }


def test_every_built_in_declares_the_release_it_is_read_under():
    """A row with no `group_label` would have no section to render in."""
    missing = [spec.key for spec in BASE_SPECS if not spec.group_label]
    assert missing == []


def test_each_release_is_one_contiguous_run_of_the_catalogue():
    """Grouping reads the catalogue in order; it never reorders it.

    The dashboard groups adjacent rows and keeps what the backend published. A
    release split into two runs would therefore render as two sections with the
    same legend — so contiguity here is what makes one section per release a
    property of the data rather than of the renderer.
    """
    runs = []
    for spec in BASE_SPECS:
        if not runs or runs[-1] != spec.group_label:
            runs.append(spec.group_label)
    assert len(runs) == len(set(runs))


def test_a_release_never_straddles_the_open_gated_split():
    """The two tabs each render whole sections, so a split release would break one.

    `gated` decides which tab a row is listed in. If one release held both a
    gated and an ungated model, its section would appear cut in half on each
    tab, with a legend claiming more than the tab shows.
    """
    tabs: dict[str, set[bool]] = {}
    for spec in BASE_SPECS:
        tabs.setdefault(spec.group_label or spec.key, set()).add(spec.gated)
    straddling = sorted(label for label, gates in tabs.items() if len(gates) > 1)
    assert straddling == []


def test_each_release_lists_its_lightest_model_first():
    """Order is the catalogue's own, and this is the intent it encodes.

    A pinned expectation rather than a derived one, deliberately: nothing
    measurable ranks these. Repository size is unknown until a model is
    downloaded, and the ordering wanted here is generation cost — the distilled,
    few-step row first — which no field on the spec reports. So this states the
    intended reading order and fails when a new row is filed in the wrong place.
    """
    order: dict[str, list[str]] = {}
    for spec in BASE_SPECS:
        order.setdefault(spec.group_label or "", []).append(spec.key)
    assert order == {
        "FLUX.2": ["flux2-klein", "flux2-dev"],
        "Anima": ["anima-turbo", "anima"],
        "Stable Diffusion 3.5": ["sd35-large-turbo", "sd35-large", "sd35-medium"],
        "Krea 2": ["krea-2-turbo"],
        "Qwen Image": ["qwen-image-flash", "qwen-image-2512"],
        "Z-Image": ["z-image-turbo", "z-image"],
        "ERNIE Image": ["ernie-image-turbo", "ernie-image"],
        "FIBO": ["fibo-lite", "fibo"],
        "Ideogram 4": ["ideogram-4"],
    }


def test_the_release_label_is_not_the_architecture_family():
    """Two facts, and FLUX.2 is where they visibly disagree.

    `family` decides quantization capability, latent creation and which profiles
    an imported directory may bind to. `group_label` decides nothing but how the
    list reads. klein and dev ship as one release and load as two families, so a
    label that followed `family` would split the section.
    """
    klein = BASE_SPECS_BY_KEY["flux2-klein"]
    dev = BASE_SPECS_BY_KEY["flux2-dev"]
    assert klein.group_label == dev.group_label == "FLUX.2"
    assert klein.family != dev.family


def test_an_imported_model_is_filed_under_no_release():
    """It borrows a built-in's defaults, not its identity.

    `imported_spec` copies the base profile wholesale, so every field it must not
    inherit has to be cleared explicitly. Left alone, a local Z-Image directory
    would arrive labelled "Z-Image" and claim membership of a release it is not
    part of — and Local models is an ungrouped list with nowhere to show it.
    """
    from qds.library import ImportedModel
    from qds.registry import imported_spec

    spec = imported_spec(
        ImportedModel(
            id="local-c1587aa663c4",
            display_name="My Z-Image",
            path="/models/my-z-image",
            family="z-image",
            base_profile_key="z-image",
            imported_at="2026-01-01T00:00:00Z",
        )
    )
    assert BASE_SPECS_BY_KEY["z-image"].group_label == "Z-Image"
    assert spec.group_label is None


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


def test_krea_2_turbo_is_distilled_but_still_takes_guidance():
    """Distillation and a fixed guidance are two facts, and Krea 2 separates them.

    Every other distilled model here refuses another value — `flux2-klein`'s CLI
    calls `parser.error()`, `z-image-turbo`'s `ModelConfig` forces it to 0 — which
    is what `supports_guidance=False` reports. Krea 2 Turbo does neither: 1.0 is
    the reference value, not a constraint, so the flags follow mflux rather than
    the word "turbo".
    """
    spec = BASE_SPECS_BY_KEY["krea-2-turbo"]
    assert spec.supports_guidance is True
    assert spec.supports_negative_prompt is True
    assert spec.default_guidance == 1.0
    assert spec.default_steps == 8
    # `Krea2._resolve_scheduler` maps "linear" onto er_sde, the published sampler.
    assert spec.scheduler == "linear"
    assert spec.gated is True


def test_krea_2_flags_match_the_mflux_config_they_were_read_from():
    """The derivation, not just the result: if mflux changes, this fails loudly."""
    from mflux.models.common.config.model_config import ModelConfig

    spec = BASE_SPECS_BY_KEY["krea-2-turbo"]
    config = ModelConfig.krea2()
    assert config.model_name == spec.repo
    assert config.supports_guidance is spec.supports_guidance

    # The contrasting case, so the assertion above cannot pass by both being True
    # for unrelated reasons: z-image-turbo is where mflux does force the value.
    assert ModelConfig.z_image_turbo().supports_guidance is False
    assert BASE_SPECS_BY_KEY["z-image-turbo"].supports_guidance is False


def test_krea_2_refuses_conversion_because_of_its_layout():
    """Not for want of a `save_model` -- `Krea2` has one. The layout is the reason.

    Its transformer is a single file at the repository root, so the component's
    `hf_subdir` is empty, and the converter is built on one component per
    subdirectory named after it. Publishing the refusal keeps the dashboard from
    offering a conversion that would write an artifact nothing could find again.
    """
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

    transformer = next(c for c in Krea2WeightDefinition.get_components() if c.name == "transformer")
    assert transformer.hf_subdir == ""

    capability = BASE_SPECS_BY_KEY["krea-2-turbo"].quantization
    assert capability.supports_prequantize is False
    assert capability.prequantize_choices == ()
    # Runtime quantization is unaffected: that path never needs a subdirectory.
    assert capability.supports_quantization is True
    assert BASE_SPECS_BY_KEY["krea-2-turbo"].quantize == 8

    with pytest.raises(ValueError, match="krea2"):
        family_structure("krea2")


def test_anima_is_configured_from_its_own_card_and_ships_off():
    spec = BASE_SPECS_BY_KEY["anima"]
    assert spec.family == "anima"
    assert spec.repo == "circlestone-labs/Anima"
    # Real CFG with a real unconditional branch, unlike the distilled models.
    assert spec.supports_guidance is True
    assert spec.supports_negative_prompt is True
    assert (spec.default_steps, spec.default_guidance) == (30, 4.5)
    # 512-1536 per the card, checked before a 4 GB download rather than after.
    assert (spec.min_dimension, spec.max_dimension) == (512, 1536)
    # Ungated — unusually for a non-commercial release — so the licence, not the
    # access request, is the reason it ships disabled.
    assert spec.gated is False
    assert "Non-Commercial" in spec.license


def test_anima_loads_unquantized_and_qwen_does_not():
    """The two rows that used to inherit their precision from a config-wide setting.

    Anima at 4-bit produces visibly broken images -- illegible architecture and a
    scratchy overlay -- while bf16 and 8-bit are indistinguishable from each
    other. At 2B the weights are ~4.2 GB, so bf16 costs almost nothing and is the
    row's answer. Qwen is the opposite case at 20B: unquantized it is ~55 GB, and
    this row points at raw bf16 weights precisely so the setting has something to
    act on, so leaving it unset would have made 55 GB the default.
    """
    registry = build_registry({}, include_disabled=True)
    assert registry["anima"].quantize is None
    assert registry["qwen-image-2512"].quantize == 8

    # And a config can still ask for something else, per model.
    asked = build_registry({"anima": ModelOverride(quantize=8)}, include_disabled=True)
    assert asked["anima"].quantize == 8


def test_anima_refuses_conversion_because_its_components_do_not_map_to_the_layout():
    """Two components share one file and a third comes from another repository."""
    capability = BASE_SPECS_BY_KEY["anima"].quantization
    assert capability.supports_prequantize is False
    assert capability.prequantize_choices == ()
    # Runtime quantization is still offered: that path needs no artifact layout.
    assert capability.supports_quantization is True

    with pytest.raises(ValueError, match="anima"):
        family_structure("anima")


def test_anima_does_not_pull_mflux_into_the_catalogue_path():
    """`registry` imports `qds.anima.config` eagerly, so it must stay light.

    Importing mflux drags in torch and transformers — seconds of start-up on a
    path that only needs to answer what the catalogue contains.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import qds.registry; "
        "heavy = [m for m in sys.modules if m.split('.')[0] in {'mflux', 'torch', 'transformers'}]; "
        "print(len(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "0", result.stdout


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


def test_every_catalogued_family_has_a_latent_creator():
    """Add a family without its unpacker and previews silently stop working.

    Edit families count: the playground submits edits too, and they run the same
    denoising loop.
    """
    families = {spec.family for spec in BASE_SPECS_BY_KEY.values()}
    families |= {spec.edit.family for spec in BASE_SPECS_BY_KEY.values() if spec.edit is not None}
    assert {family: latent_creator_for(family) is not None for family in sorted(families)} == {
        family: True for family in sorted(families)
    }


def test_an_unknown_family_gets_no_previews_rather_than_an_error():
    assert latent_creator_for("stable-diffusion-1.5") is None


def test_qwen_does_not_use_from_name():
    # from_name() would lose the scheduler's sigma_* values; we pass the
    # canonical factory plus the repo as model_path. As of mflux 0.19.0 that path
    # agrees with `ModelConfig.qwen_image()`, which now resolves to the 2512
    # release; it kept its place because mflux moved that name once already
    # (0.18.0 resolved it to `Qwen/Qwen-Image`), and which weights a key serves is
    # the catalogue's statement rather than a dependency's default.
    spec = BASE_SPECS_BY_KEY["qwen-image-2512"]
    assert spec.model_config_name == "qwen_image"
    assert spec.model_path == "Qwen/Qwen-Image-2512"
    # The raw bf16 repo, not the 8-bit conversion: only raw weights can honour
    # `default_quantize`.
    assert spec.quantization.supports_quantization is True


def test_qwen_follows_its_own_card_not_the_mflux_defaults():
    """Both numbers come from the card, and neither matches mflux.

    mflux would give 20 steps and guidance 3.5 — its blanket defaults, applied
    to every model and never published by Qwen. The card's own example is
    `num_inference_steps=50, true_cfg_scale=4.0`.
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
    # Base model: the card's 50 steps, and 1024² to avoid paying for area on a 32B.
    assert spec.default_steps == 50
    assert (spec.default_width, spec.default_height) == (1024, 1024)
    # The upstream repo ships bf16: loading goes through a saved variant, which
    # is now a *variant* rather than the model's identity. Its source is the raw
    # repository, like every other built-in, so it can be installed and located.
    assert spec.quantize == 8
    assert spec.model_path is None
    assert spec.repo == "black-forest-labs/FLUX.2-dev"
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
    settings = Settings.model_validate({"default_size": ""})
    assert settings.registry() == build_registry({}, cache_root=settings.effective_cache_dir)


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
        # The catalogue's own precision, reaching the shipped install untouched.
        # It used to be 4 here for every model, from a config-wide setting that
        # overwrote each row; the setting is gone, and this is what proves the
        # shipped file no longer overrides what the catalogue chose.
        assert spec.quantize == BASE_SPECS_BY_KEY[spec.key].quantize, spec.key
        assert spec.quantize == 8, spec.key


def codes(raw: dict) -> list[str]:
    """The runtime invariants a configuration breaks, without loading a file.

    These are no longer construction errors: a document that cannot serve
    generations is still a document model management has to be able to read and
    repair. `load_settings(strict=True)` — what the server uses — is what turns
    them back into refusals, and `test_catalogue_resilience.py` holds that end.
    """
    return [issue.code for issue in Settings.model_validate(raw).runtime_issues()]


def test_default_model_must_exist():
    assert codes({"default_model": "sdxl"}) == ["unknown_default_model"]
    assert codes({"default_model": "z-image"}) == []


def test_default_model_cannot_be_disabled():
    assert codes({"default_model": "z-image", "models": {"z-image": {"enabled": False}}}) == [
        "default_model_disabled"
    ]
    # Disabling anything else is none of this invariant's business.
    assert codes({"default_model": "z-image", "models": {"flux2-dev": {"enabled": False}}}) == []


def test_non_local_binding_requires_both_credentials(tmp_path, monkeypatch):
    """Two credentials, because there are two planes.

    This assertion gained a second code deliberately: an API key protects `/v1`,
    and it no longer opens `/admin` at all — so a server reachable from the
    network needs an admin password as well, or its control plane would be open
    to everyone who can reach the port.

    Kept as an exact list rather than relaxed to a membership check: the point of
    comparing the whole thing is to notice a *third* issue appearing by accident.
    """
    from qds import credential

    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))

    assert codes({"server": {"host": "0.0.0.0"}}) == [
        "unauthenticated_admin",
        "unauthenticated_host",
    ]
    messages = " ".join(
        issue.message
        for issue in Settings.model_validate({"server": {"host": "0.0.0.0"}}).runtime_issues()
    )
    assert "api_key" in messages
    assert "admin password" in messages

    # An API key alone is no longer enough.
    assert codes({"server": {"host": "0.0.0.0", "api_key": "secret"}}) == ["unauthenticated_admin"]

    credential.set_password("correct horse battery")
    settings = Settings.model_validate({"server": {"host": "0.0.0.0", "api_key": "secret"}})
    assert settings.server.is_loopback is False
    assert settings.runtime_issues() == []


def test_base_models_default_to_the_step_count_their_cards_publish():
    """No base model is quietly cheaper than what its authors recommend.

    These five were dropped to 20 together, which made the catalogue disagree
    with every one of their model cards at once. They are checked as a group
    because that is how they drifted: a blanket number applied across rows whose
    only shared property was being expensive.

    z-image is the one that mattered most. Its card publishes a *range* —
    "Inference steps: 28 - 50" — so 20 was not a thriftier reading of an
    example, it was outside what the model is documented to do.
    """
    published = {
        "flux2-dev": 50,  # card: num_inference_steps=50 (with "#28 ... trade-off")
        "qwen-image-2512": 50,  # card: num_inference_steps=50, true_cfg_scale=4.0
        "z-image": 50,  # card: "Inference steps: 28 - 50", example at 50
        "ernie-image": 50,  # card: "typically 50 inference steps"
        "fibo": 50,  # card: every example calls num_inference_steps=50
    }
    assert {key: BASE_SPECS_BY_KEY[key].default_steps for key in published} == published


def test_ideogram_keeps_the_step_count_of_its_preset():
    """Not one of the five: 20 here is a fact about the sampler, not a budget.

    Ideogram 4 takes its schedule from a preset, and `V4_DEFAULT_20` carries 20.
    Raising this number alone would not buy any steps — it would only make
    `/v1/capabilities` and the progress bar report a count the sampler is not
    running. Changing it means changing `preset`.
    """
    spec = BASE_SPECS_BY_KEY["ideogram-4"]
    assert spec.default_steps == 20
    assert spec.preset == "V4_DEFAULT_20"


def test_the_three_stable_diffusion_35_rows_report_what_the_models_accept():
    """Defaults are the cards' own, and the distilled row refuses what it cannot use."""
    registry = build_registry({})

    medium = registry["sd35-medium"]
    large = registry["sd35-large"]
    turbo = registry["sd35-large-turbo"]

    for spec in (medium, large, turbo):
        assert spec.family == "sd35"
        assert spec.gated is True
        assert spec.license == "Stability AI Community"
        assert spec.scheduler == "flow_match_euler_discrete"
        assert spec.supports_image_to_image is True
        assert spec.edit is None
        assert spec.prompt_formats == ("text",)
        # `pos_embed_max_size` bounds the large releases at 1536px structurally; the
        # same ceiling is applied to Medium, whose card trains to 1440.
        assert (spec.min_dimension, spec.max_dimension) == (512, 1536)

    # From each card's own diffusers example, not from mflux's blanket defaults.
    assert (medium.default_steps, medium.default_guidance) == (40, 4.5)
    assert (large.default_steps, large.default_guidance) == (28, 3.5)
    assert (turbo.default_steps, turbo.default_guidance) == (4, 0.0)

    # Medium and Large run real classifier-free guidance, so a negative prompt has an
    # unconditional branch to go in. Large Turbo is adversarially distilled to run at
    # guidance 0.0: there is no such branch, so both are refused rather than ignored.
    assert medium.supports_guidance is True and medium.supports_negative_prompt is True
    assert large.supports_guidance is True and large.supports_negative_prompt is True
    assert turbo.supports_guidance is False and turbo.supports_negative_prompt is False

    # bf16 for Medium (~16 GB whole), 8-bit for the large pair: their bf16 peak while
    # quantizing on the way in is ~41 GB, which fits 64 GB and not 32.
    assert medium.quantize is None
    assert large.quantize == 8 and turbo.quantize == 8


def test_stable_diffusion_35_ships_disabled_like_every_other_gated_model():
    registry = build_registry({})
    for key in ("sd35-medium", "sd35-large", "sd35-large-turbo"):
        assert BASE_SPECS_BY_KEY[key].gated is True
        # Enabled-by-default is the shipped config's decision, and it says no.
        assert registry[key].enabled is True  # the bare catalogue enables everything
    settings = load_settings(Path(__file__).resolve().parent.parent / "server-config.json")
    shipped = settings.registry()
    assert not [key for key in shipped if key.startswith("sd35")]


def test_stable_diffusion_35_publishes_the_generic_quantization_capability():
    """Keyed by family, so all three rows get it — and all five components with it."""
    from qds import components
    from qds.registry import STRATEGY_QDS_MEMORY_BOUNDED, capability_for

    capability = capability_for("sd35")
    assert capability.supports_quantization is True
    assert capability.supports_prequantize is True
    assert capability.prequantize_strategy == STRATEGY_QDS_MEMORY_BOUNDED
    assert len(components.components_for("sd35")) == 5


def test_stable_diffusion_35_needs_no_engine_special_case():
    """A standard family: `latent_creator_for` answers, and nothing else is bespoke."""
    from qds.registry import latent_creator_for

    creator = latent_creator_for("sd35")
    assert creator is not None
    assert hasattr(creator, "unpack_latents")
    # Previews carry plain `[1, 16, h/8, w/8]` latents; unpacking is the identity.
    import mlx.core as mx

    latents = mx.zeros((1, 16, 8, 8))
    assert creator.unpack_latents(latents, 64, 64) is latents
