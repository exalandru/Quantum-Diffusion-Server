"""Pointing a built-in catalogue entry at weights already on this machine.

Locating is not importing, and the tests below are mostly about keeping the two
apart: importing mints a new entry the user owns, locating tells an entry QDS
already ships where its weights are. The catalogue key, family and profile stay
exactly as shipped.
"""

from __future__ import annotations

import json

from qds import importing, library
from qds.registry import BASE_SPECS_BY_KEY, build_registry
from qds.settings import ModelOverride

CLASS_OF = {
    "z-image": "ZImageTransformer2DModel",
    "flux2": "Flux2Transformer2DModel",
    "qwen": "QwenImageTransformer2DModel",
}


def source_dir(root, family: str = "z-image"):
    """A minimal source model: what `inspect` actually reads."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "transformer").mkdir(exist_ok=True)
    (root / "transformer" / "config.json").write_text(json.dumps({"_class_name": CLASS_OF[family]}))
    (root / "vae").mkdir(exist_ok=True)
    return root


def cached_source(base, org: str, name: str, family: str = "z-image"):
    """The same, but inside a hub cache so its repository is provable."""
    revision = "0123456789abcdef0123456789abcdef01234567"
    return source_dir(base / f"models--{org}--{name}" / "snapshots" / revision, family)


def test_locating_a_compatible_directory_succeeds(tmp_path):
    verdict = importing.locate(str(source_dir(tmp_path / "weights")), "z-image")
    assert verdict.ok
    assert verdict.family == "z-image"
    assert verdict.model == "z-image"


def test_an_incompatible_family_fails_closed(tmp_path):
    # Binding these weights would produce a model that cannot load, discovered
    # minutes into the first generation rather than here.
    verdict = importing.locate(str(source_dir(tmp_path / "weights")), "flux2-klein")
    assert not verdict.ok
    assert verdict.availability == "incompatible"
    assert "z-image" in verdict.reason and "flux2" in verdict.reason


def test_an_unknown_built_in_is_refused(tmp_path):
    verdict = importing.locate(str(source_dir(tmp_path / "weights")), "not-a-model")
    assert not verdict.ok
    assert "built-in" in verdict.reason


def test_a_directory_that_is_not_a_model_is_refused(tmp_path):
    # Present on disk, but carrying no `transformer/config.json` — so there is
    # nothing that says which model it is, and guessing is what this refuses.
    other = tmp_path / "photos"
    other.mkdir()
    (other / "holiday.jpg").write_text("not weights")
    verdict = importing.locate(str(other), "z-image")
    assert not verdict.ok
    assert verdict.availability == "invalid"
    assert "transformer/config.json" in verdict.reason


def test_an_absent_volume_is_reported_as_such(tmp_path):
    verdict = importing.locate("/Volumes/NotMounted/models/z", "z-image")
    assert not verdict.ok
    assert verdict.availability == "volume_unmounted"


def test_a_cached_repository_proves_its_identity(tmp_path):
    located = cached_source(tmp_path, "mlx-community", "Z-Image-bf16")
    verdict = importing.locate(str(located), "z-image")
    assert verdict.ok
    assert verdict.detected_repo == "mlx-community/Z-Image-bf16"
    assert verdict.repo_verified is True


def test_a_raw_directory_claims_no_provenance(tmp_path):
    # Compatible is not the same as "this is that repository", and the difference
    # is shown to the user rather than smoothed over.
    verdict = importing.locate(str(source_dir(tmp_path / "somewhere")), "z-image")
    assert verdict.ok
    assert verdict.detected_repo is None
    assert verdict.repo_verified is False


def test_a_cache_entry_for_a_different_repository_is_compatible_but_unverified(tmp_path):
    # Same family, provably a different repository: allowed, and said out loud.
    located = cached_source(tmp_path, "someone-else", "Z-Image-fork")
    verdict = importing.locate(str(located), "z-image")
    assert verdict.ok
    assert verdict.detected_repo == "someone-else/Z-Image-fork"
    assert verdict.repo_verified is False


def test_locating_copies_nothing(tmp_path):
    weights = source_dir(tmp_path / "weights")
    before = sorted(p.name for p in weights.rglob("*"))
    importing.locate(str(weights), "z-image")
    assert sorted(p.name for p in weights.rglob("*")) == before
    # And it writes nothing anywhere else either: the whole answer is a verdict.
    assert not (tmp_path / library.LIBRARY_FILENAME).exists()


def test_locating_creates_no_imported_row(tmp_path, monkeypatch):
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    importing.locate(str(source_dir(tmp_path / "weights")), "z-image")
    assert library.load() == []


def test_the_override_keeps_the_built_in_identity(tmp_path):
    """What binding the path actually does, through the ordinary override."""
    weights = str(source_dir(tmp_path / "weights"))
    registry = build_registry({"z-image": ModelOverride(model_path=weights)}, include_disabled=True)
    spec = registry["z-image"]

    assert spec.key == "z-image"  # catalogue identity, unchanged
    assert spec.provenance == "built_in"
    assert spec.family == BASE_SPECS_BY_KEY["z-image"].family
    assert spec.public_name == "z-image"
    # The source is what moved.
    assert spec.model_path == weights
    assert spec.source == weights


def test_resetting_the_location_restores_the_catalogue_repository(tmp_path):
    weights = str(source_dir(tmp_path / "weights"))
    located = build_registry({"z-image": ModelOverride(model_path=weights)}, include_disabled=True)["z-image"]
    assert located.source == weights

    # `Reset location` writes `model_path: null` — nothing else. Note what the
    # catalogue restores to: several built-ins carry a `model_path` of their own,
    # so "no override" means the catalogue's value, not the absence of one.
    reset = build_registry({"z-image": ModelOverride(model_path=None)}, include_disabled=True)["z-image"]
    assert reset.source == BASE_SPECS_BY_KEY["z-image"].repo
    assert reset.model_path == BASE_SPECS_BY_KEY["z-image"].model_path
    # And the files are still there: the override never owned them.
    assert (tmp_path / "weights" / "transformer" / "config.json").is_file()


def test_a_located_built_in_is_no_longer_downloadable(tmp_path, monkeypatch):
    from qds.fetch import cache_status

    weights = str(source_dir(tmp_path / "weights"))
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"models": {"z-image": {"model_path": weights}}}))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    row = next(r for r in cache_status() if r["key"] == "z-image")
    assert row["availability"] == "present"
    assert row["repo"] == weights  # the located path is what the row shows
    assert row["can_download"] is False  # so no Install/Resume can be offered
    assert row["provenance"] == "built_in"
    assert row["local"] is True


def test_a_located_path_on_an_absent_volume_reports_volume_unmounted(tmp_path, monkeypatch):
    from qds.fetch import cache_status

    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"models": {"z-image": {"model_path": "/Volumes/Gone/models/z-image"}}})
    )
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    row = next(r for r in cache_status() if r["key"] == "z-image")
    assert row["availability"] == "volume_unmounted"
    assert row["can_download"] is False
