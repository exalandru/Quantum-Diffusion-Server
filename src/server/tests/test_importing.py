"""Identifying a local model directory. No weights are read anywhere here."""

from __future__ import annotations

import json

from mflux_server import availability as av
from mflux_server.importing import (
    CLASS_NAME_TO_FAMILY,
    KEY_PREFIX,
    compatible_profiles,
    inspect,
    new_id,
    normalise_path,
)
from mflux_server.registry import BASE_SPECS_BY_KEY, capability_for


def make_source(root, class_name="ZImageTransformer2DModel", *, components=("transformer", "vae")):
    """A source model directory, shaped the way diffusers writes one."""
    for name in components:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(
            json.dumps({"_class_name": class_name if name == "transformer" else "AutoencoderKL"}),
            encoding="utf-8",
        )
        (d / "diffusion_pytorch_model.safetensors").write_bytes(b"w")
    return root


def make_saved_artifact(root):
    """What `ModelSaver` writes: shards + index, and no source configuration."""
    for name in ("transformer", "vae"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "0.safetensors").write_bytes(b"w")
        (d / av.INDEX_FILE).write_text(
            json.dumps({"metadata": {"quantization_level": "4"}, "weight_map": {"w": "0.safetensors"}}),
            encoding="utf-8",
        )
    return root


def test_a_valid_source_directory_is_identified(tmp_path):
    verdict = inspect(str(make_source(tmp_path / "zimage")))
    assert verdict.ok
    assert verdict.family == "z-image"
    assert verdict.class_name == "ZImageTransformer2DModel"
    assert verdict.availability == av.PRESENT


def test_a_saved_artifact_is_refused_with_an_explanation(tmp_path):
    """Slice 6 owns these as variants of a source; they carry no family."""
    verdict = inspect(str(make_saved_artifact(tmp_path / "artifact")))
    assert not verdict.ok
    assert verdict.availability == "invalid"
    assert "saved/pre-quantized" in (verdict.reason or "")
    assert "original/source model" in (verdict.reason or "")


def test_a_directory_without_a_transformer_config_is_refused(tmp_path):
    plain = tmp_path / "plain"
    (plain / "transformer").mkdir(parents=True)
    verdict = inspect(str(plain))
    assert not verdict.ok
    assert "config.json" in (verdict.reason or "")


def test_an_unknown_architecture_fails_closed_and_names_itself(tmp_path):
    verdict = inspect(str(make_source(tmp_path / "x", class_name="SomeFutureTransformer")))
    assert not verdict.ok
    assert verdict.availability == "incompatible"
    assert "SomeFutureTransformer" in (verdict.reason or "")


def test_a_config_without_class_name_is_refused(tmp_path):
    root = tmp_path / "y"
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    assert not inspect(str(root)).ok


def test_storage_states_stay_distinguishable(tmp_path):
    assert inspect("/Volumes/QDSNeverMounted/model").availability == av.VOLUME_UNMOUNTED
    assert inspect(str(tmp_path / "absent")).availability == av.MISSING


def test_the_identity_is_opaque_and_not_derived_from_the_path(tmp_path):
    """Identity and location are different facts.

    A path-derived key would silently make a moved model a different
    registration; duplicate detection is answered by comparing paths instead.
    """
    source = str(make_source(tmp_path / "m"))
    first, second = new_id(), new_id()
    assert first != second
    assert source not in first
    assert first.startswith(KEY_PREFIX)


def test_an_imported_id_cannot_collide_with_a_builtin():
    assert new_id() not in BASE_SPECS_BY_KEY
    assert not any(builtin.startswith(KEY_PREFIX) for builtin in BASE_SPECS_BY_KEY)


def test_a_family_with_several_profiles_offers_them_all(tmp_path):
    """Nothing may silently pick one: Z-Image and Turbo differ in defaults."""
    verdict = inspect(str(make_source(tmp_path / "z")))
    assert verdict.ok
    assert set(verdict.profiles) == set(compatible_profiles("z-image"))
    assert len(verdict.profiles) > 1


def test_paths_are_absolute_but_not_symlink_resolved(tmp_path):
    """An unplugged volume must stay representable, so no canonicalisation."""
    target = make_source(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(target)
    assert normalise_path(str(link)) == str(link)
    assert normalise_path("/Volumes/Gone/m") == "/Volumes/Gone/m"


def test_every_mapped_family_has_a_slice5_capability():
    """Detection may only produce families the capability contract knows."""
    for family in set(CLASS_NAME_TO_FAMILY.values()):
        capability = capability_for(family)
        assert capability.note is None or capability.supports_quantization is not None
        assert any(spec.family == family for spec in BASE_SPECS_BY_KEY.values()), family
