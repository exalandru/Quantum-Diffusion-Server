from __future__ import annotations

import pytest

from mflux_server.registry import (
    BASE_SPECS_BY_KEY,
    build_registry,
    edit_enabled,
    normalize_dimension,
    parse_size,
)
from mflux_server.settings import ModelOverride, Settings


def test_default_registry_expose_les_quatre_modeles():
    registry = build_registry({})
    assert set(registry) == {"flux2-klein", "qwen-image", "z-image", "z-image-turbo"}


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1024x1024", (1024, 1024)),
        # Le bug historique : mflux tronquait 1080 en 1072 sans le dire.
        ("1920x1080", (1920, 1072)),
        ("1280x720", (1280, 720)),
        ("512X512", (512, 512)),
    ],
)
def test_parse_size_normalise_au_multiple_de_16(size, expected):
    assert parse_size(size) == expected


@pytest.mark.parametrize("size", ["auto", "1024", "axb", "1024x", ""])
def test_parse_size_refuse_les_formats_invalides(size):
    with pytest.raises(ValueError):
        parse_size(size)


def test_dimension_sous_le_pas_minimal_est_refusee():
    # mflux se contenterait d'un warning puis produirait 0.
    with pytest.raises(ValueError):
        normalize_dimension(15)


def test_flux2_klein_est_distille():
    spec = BASE_SPECS_BY_KEY["flux2-klein"]
    assert spec.supports_negative_prompt is False
    assert spec.supports_guidance is False
    assert spec.default_guidance == 1.0
    # Modèle distillé : 4 étapes, pas les 20 du prototype.
    assert spec.default_steps == 4
    assert spec.scheduler == "flow_match_euler_discrete"


def test_edit_flux2_partage_les_poids_et_est_actif_par_defaut():
    spec = BASE_SPECS_BY_KEY["flux2-klein"]
    assert spec.edit is not None
    assert spec.edit.shares_weights is True
    assert edit_enabled(spec) is True


def test_edit_qwen_desactive_par_defaut_car_telechargement_separe():
    spec = BASE_SPECS_BY_KEY["qwen-image"]
    assert spec.edit is not None
    assert spec.edit.shares_weights is False
    assert edit_enabled(spec) is False


def test_qwen_nutilise_pas_from_name():
    # from_name() perdrait les sigma_* du scheduler ; on passe la factory
    # canonique et le repo en model_path.
    spec = BASE_SPECS_BY_KEY["qwen-image"]
    assert spec.model_config_name == "qwen_image"
    assert spec.model_path == "mlx-community/Qwen-Image-2512-8bit"


def test_surcharges_appliquees():
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


def test_surcharger_la_guidance_dun_modele_distille_echoue():
    with pytest.raises(ValueError, match="guidance"):
        build_registry({"flux2-klein": ModelOverride(default_guidance=3.5)})


def test_modele_inconnu_dans_la_config():
    with pytest.raises(ValueError, match="Modèles inconnus"):
        build_registry({"sdxl": ModelOverride()})


def test_default_model_doit_exister():
    with pytest.raises(ValueError):
        Settings.model_validate({"default_model": "sdxl"})


def test_default_model_ne_peut_pas_etre_desactive():
    with pytest.raises(ValueError):
        Settings.model_validate({"default_model": "z-image", "models": {"z-image": {"enabled": False}}})


def test_binding_non_local_exige_une_cle_api():
    with pytest.raises(ValueError, match="api_key"):
        Settings.model_validate({"server": {"host": "0.0.0.0"}})

    settings = Settings.model_validate({"server": {"host": "0.0.0.0", "api_key": "secret"}})
    assert settings.server.is_loopback is False
