"""Durable imported-model registrations, and what must never destroy them."""

from __future__ import annotations

import json

import pytest

from mflux_server import importing, library
from mflux_server.registry import (
    BASE_SPECS_BY_KEY,
    PROVENANCE_BUILT_IN,
    PROVENANCE_IMPORTED,
    build_registry,
    capability_for,
)
from mflux_server.settings import Settings


def row(
    tmp_path,
    *,
    id="local-aaa",
    path=None,
    family="z-image",
    profile="z-image-turbo",
    api_name="fixture",
):
    return library.ImportedModel(
        id=id,
        display_name="Fixture",
        api_name=api_name,
        path=str(path or (tmp_path / "model")),
        family=family,
        base_profile_key=profile,
        imported_at="2026-01-01T00:00:00",
    )


def test_persistence_round_trips(tmp_path):
    original = [row(tmp_path)]
    library.save(original, base=str(tmp_path))
    back = library.load(base=str(tmp_path))
    assert [m.as_dict() for m in back] == [m.as_dict() for m in original]
    # Schema 2 added `api_name`; a saved file always carries the current version.
    assert json.loads((tmp_path / library.LIBRARY_FILENAME).read_text())["version"] == 2


def test_a_malformed_row_is_skipped_without_taking_the_others(tmp_path):
    (tmp_path / library.LIBRARY_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {"id": "local-ok", "path": "/x", "family": "z-image", "base_profile_key": "z-image"},
                    {"nonsense": True},
                    "not even an object",
                ],
            }
        ),
        encoding="utf-8",
    )
    models = library.load(base=str(tmp_path))
    assert [m.id for m in models] == ["local-ok"]
    assert library.last_load_error  # and the user is told


def test_an_unreadable_library_is_reported_rather_than_looking_empty(tmp_path):
    (tmp_path / library.LIBRARY_FILENAME).write_text("{ truncated", encoding="utf-8")
    assert library.load(base=str(tmp_path)) == []
    assert "could not be read" in (library.last_load_error or "")
    # And the file is left alone for a future version to read.
    assert (tmp_path / library.LIBRARY_FILENAME).read_text() == "{ truncated"


def test_a_future_schema_version_is_refused_not_rewritten(tmp_path):
    payload = {"version": library.SCHEMA_VERSION + 1, "models": []}
    (tmp_path / library.LIBRARY_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(library.LibraryTooNew):
        library.load(base=str(tmp_path))
    assert json.loads((tmp_path / library.LIBRARY_FILENAME).read_text()) == payload


def test_the_same_path_finds_the_existing_registration(tmp_path):
    existing = row(tmp_path)
    models = [existing]
    assert library.find_by_path(models, existing.path).id == existing.id
    assert library.find_by_path(models, existing.path + "/").id == existing.id
    assert library.find_by_path(models, str(tmp_path / "other")) is None


def test_a_refresh_never_removes_a_registration(tmp_path):
    """An unplugged disk is a fact about storage, not a decision to unregister."""
    absent = row(tmp_path, path="/Volumes/QDSNotMounted/model")
    library.save([absent], base=str(tmp_path))
    state, _ = importing.availability_of(absent.path)
    assert state == "volume_unmounted"
    assert [m.id for m in library.load(base=str(tmp_path))] == [absent.id]


# ── Merge into the registry ────────────────────────────────────────────────


def test_an_imported_row_becomes_a_spec_with_its_profile_defaults(tmp_path):
    registry = build_registry({}, imported=[row(tmp_path)], include_disabled=True)
    spec = registry["local-aaa"]
    profile = BASE_SPECS_BY_KEY["z-image-turbo"]

    assert spec.provenance == PROVENANCE_IMPORTED
    assert spec.base_profile_key == "z-image-turbo"
    assert spec.default_steps == profile.default_steps
    assert spec.source == str(tmp_path / "model")
    # Capability is the family's, not a copy stored on the row.
    assert spec.quantization == capability_for("z-image")


def test_a_profile_from_the_wrong_family_is_rejected(tmp_path):
    bad = row(tmp_path, family="z-image", profile="qwen-image-2512")
    registry = build_registry({}, imported=[bad], include_disabled=True)
    assert "local-aaa" not in registry
    assert "z-image" in registry  # built-ins untouched


def test_a_vanished_profile_does_not_break_the_catalogue(tmp_path):
    gone = row(tmp_path, profile="a-profile-removed-in-a-later-version")
    registry = build_registry({}, imported=[gone], include_disabled=True)
    assert "local-aaa" not in registry
    assert set(BASE_SPECS_BY_KEY) <= set(registry)


def test_builtins_stay_source_code_owned(tmp_path):
    registry = build_registry({}, imported=[row(tmp_path)], include_disabled=True)
    for key, spec in registry.items():
        if key in BASE_SPECS_BY_KEY:
            assert spec.provenance == PROVENANCE_BUILT_IN


# ── Provenance vs string shape ─────────────────────────────────────────────


def test_a_repo_shaped_local_path_cannot_pass_for_a_downloadable_model(tmp_path, monkeypatch):
    """The old heuristic looked at the string; provenance looks at the fact."""
    from mflux_server.fetch import cache_status

    # A directory whose name looks exactly like `org/model`.
    tricky = tmp_path / "mlx-community"
    (tricky / "Z-Image-bf16").mkdir(parents=True)
    imported = row(tmp_path, path=str(tricky / "Z-Image-bf16"))
    library.save([imported], base=str(tmp_path))

    config = tmp_path / "server-config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    rows = {r["key"]: r for r in cache_status()}
    entry = rows[imported.id]
    assert entry["provenance"] == PROVENANCE_IMPORTED
    assert entry["can_download"] is False  # never an Install/Resume action
    assert rows["z-image"]["can_download"] is True


# ── default_model ──────────────────────────────────────────────────────────


def test_an_imported_model_may_be_the_default(tmp_path, monkeypatch):
    imported = row(tmp_path)
    library.save([imported], base=str(tmp_path))
    monkeypatch.setattr(library, "library_path", lambda base=None: tmp_path / library.LIBRARY_FILENAME)

    settings = Settings.model_validate({"default_model": imported.id})
    assert settings.default_model == imported.id


def test_a_forgotten_imported_default_is_refused_with_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "library_path", lambda base=None: tmp_path / library.LIBRARY_FILENAME)
    library.save([], base=str(tmp_path))
    with pytest.raises(ValueError) as raised:
        Settings.model_validate({"default_model": "local-forgotten"})
    assert "still be registered" in str(raised.value)


def test_the_imported_source_is_what_slice6_identifies_variants_by(tmp_path):
    from mflux_server import artifacts

    spec = build_registry({}, imported=[row(tmp_path)], include_disabled=True)["local-aaa"]
    expected = artifacts.artifact_dir(spec.key, spec.source, 4)
    assert artifacts.source_digest(str(tmp_path / "model")) in str(expected)


# ── Public API names ───────────────────────────────────────────────────────


def test_a_version_1_row_gains_a_stable_api_name(tmp_path):
    """The migration: an old file is readable, and its rows become addressable.

    Deterministic on purpose — the alias a client used yesterday has to be the
    one it gets today, whether or not anything has rewritten the file since.
    """
    (tmp_path / library.LIBRARY_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "id": "local-c1587aa663c4",
                        "display_name": "My Z-Image",
                        "path": "/models/z",
                        "family": "z-image",
                        "base_profile_key": "z-image-turbo",
                    }
                ],
            }
        )
    )

    first = library.load(base=str(tmp_path))
    assert [m.api_name for m in first] == ["my-z-image"]
    # The durable identity is untouched by the migration.
    assert first[0].id == "local-c1587aa663c4"
    # And it is stable across loads.
    assert [m.api_name for m in library.load(base=str(tmp_path))] == ["my-z-image"]


def test_reading_an_old_file_does_not_rewrite_it(tmp_path):
    # A read that writes is a read that can race an import, and can turn an
    # unreadable-but-recoverable file into a rewritten one.
    path = tmp_path / library.LIBRARY_FILENAME
    path.write_text(json.dumps({"version": 1, "models": []}))
    before = path.read_bytes()
    library.load(base=str(tmp_path))
    assert path.read_bytes() == before


def test_migrated_names_avoid_each_other_and_the_built_in_catalogue(tmp_path):
    (tmp_path / library.LIBRARY_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "id": "local-1",
                        "display_name": "z-image",  # a built-in's public key
                        "path": "/a",
                        "family": "z-image",
                        "base_profile_key": "z-image-turbo",
                    },
                    {
                        "id": "local-2",
                        "display_name": "Shared Name",
                        "path": "/b",
                        "family": "z-image",
                        "base_profile_key": "z-image-turbo",
                    },
                    {
                        "id": "local-3",
                        "display_name": "Shared Name",
                        "path": "/c",
                        "family": "z-image",
                        "base_profile_key": "z-image-turbo",
                    },
                ],
            }
        )
    )

    names = [m.api_name for m in library.load(base=str(tmp_path))]
    assert names == ["z-image-2", "shared-name", "shared-name-2"]
    assert "z-image" in BASE_SPECS_BY_KEY  # the premise of the first assertion
    assert len(set(names)) == 3


def test_a_newer_library_is_still_refused(tmp_path):
    (tmp_path / library.LIBRARY_FILENAME).write_text(json.dumps({"version": 99, "models": []}))
    with pytest.raises(library.LibraryTooNew):
        library.load(base=str(tmp_path))
