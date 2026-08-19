"""Three identities, kept apart.

An imported model has an opaque internal id (`local-c1587aa663c4`), a display
name for people, and a public API name clients send. Collapsing any two of them
is the failure this guards: the id was being published as the OpenAI model id, so
the identifier every client copied into its configuration was a storage detail.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from qds import importing, library
from qds.app import create_app
from qds.settings import Settings

from .conftest import FakeEngine, make_client


def imported(tmp_path, *, id="local-c1587aa663c4", display="My Z-Image", api_name="my-z-image"):
    weights = tmp_path / "weights"
    (weights / "transformer").mkdir(parents=True, exist_ok=True)
    (weights / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "ZImageTransformer2DModel"})
    )
    return library.ImportedModel(
        id=id,
        display_name=display,
        api_name=api_name,
        path=str(weights),
        family="z-image",
        base_profile_key="z-image-turbo",
        imported_at="2026-01-01T00:00:00",
    )


def settings_with(tmp_path, monkeypatch, rows, **overrides) -> Settings:
    library.save(rows, base=str(tmp_path))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    return Settings.model_validate({"models": {}, **overrides})


# ── Identity ───────────────────────────────────────────────────────────────


def test_the_internal_id_stays_opaque_and_is_not_derived_from_anything(tmp_path):
    first, second = importing.new_id(), importing.new_id()
    assert first.startswith("local-") and first != second
    # Not a slug of a name, not a path: identity and location are different facts.
    assert "my-z-image" not in first


def test_an_imported_model_publishes_its_alias_not_its_id(tmp_path, monkeypatch):
    settings = settings_with(tmp_path, monkeypatch, [imported(tmp_path)])
    spec = settings.registry(include_disabled=True)["local-c1587aa663c4"]

    assert spec.key == "local-c1587aa663c4"  # durable, internal
    assert spec.display_name == "My Z-Image"  # for people
    assert spec.public_name == "my-z-image"  # for machines


def test_a_built_in_publishes_its_catalogue_key(tmp_path, monkeypatch):
    settings = settings_with(tmp_path, monkeypatch, [])
    assert settings.registry(include_disabled=True)["z-image"].public_name == "z-image"


def test_the_api_name_defaults_from_the_display_name():
    assert importing.unique_api_name("My Z-Image", set()) == "my-z-image"
    assert importing.unique_api_name("Flux 2 — Dev!", set()) == "flux-2-dev"


def test_a_duplicate_api_name_is_rejected(tmp_path):
    taken = importing.taken_api_names([imported(tmp_path)])
    assert "my-z-image" in taken
    assert importing.unique_api_name("My Z-Image", taken) == "my-z-image-2"


def test_an_api_name_cannot_collide_with_a_built_in_key(tmp_path):
    taken = importing.taken_api_names([])
    assert "z-image" in taken and "flux2-dev" in taken
    assert importing.unique_api_name("z-image", taken) == "z-image-2"


@pytest.mark.parametrize("bad", ["", "Has Spaces", "UPPER", "-leading", "a" * 65, "sla/sh"])
def test_invalid_api_names_are_named_as_such(bad):
    assert importing.api_name_problem(bad) is not None


@pytest.mark.parametrize("good", ["my-z-image", "z.image", "a", "model_2"])
def test_valid_api_names_pass(good):
    assert importing.api_name_problem(good) is None


# ── The public API boundary ────────────────────────────────────────────────


def client_for(tmp_path, monkeypatch, rows, **overrides) -> TestClient:
    settings = settings_with(tmp_path, monkeypatch, rows, **overrides)
    return make_client(create_app(settings=settings, engine=FakeEngine()))


def test_v1_models_lists_the_alias_and_never_the_internal_id(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch, [imported(tmp_path)]) as client:
        ids = [row["id"] for row in client.get("/v1/models").json()["data"]]

    assert "my-z-image" in ids
    # The whole point: the opaque id is not what clients are told to use.
    assert not any(name.startswith("local-") for name in ids)
    assert "z-image-turbo" in ids  # built-ins keep their catalogue keys


def test_a_model_can_be_retrieved_by_its_alias(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch, [imported(tmp_path)]) as client:
        body = client.get("/v1/models/my-z-image").json()
    assert body["id"] == "my-z-image"
    assert body["mflux"]["repo"].endswith("weights")


def test_the_internal_id_still_resolves_but_answers_with_the_alias(tmp_path, monkeypatch):
    # An unadvertised legacy path: the id was the only way to name an imported
    # model before aliases existed, and a script that used it should keep working
    # — while being told the current name.
    with client_for(tmp_path, monkeypatch, [imported(tmp_path)]) as client:
        body = client.get("/v1/models/local-c1587aa663c4").json()
    assert body["id"] == "my-z-image"


def test_an_unknown_model_lists_public_names_in_its_error(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch, [imported(tmp_path)]) as client:
        body = client.get("/v1/models/nope").json()
    message = body["error"]["message"]
    assert "my-z-image" in message
    assert "local-c1587aa663c4" not in message


def test_a_generation_request_by_alias_resolves_the_imported_model(tmp_path, monkeypatch):
    """The request path, not just the listing: `model: "my-z-image"` must land."""
    settings = settings_with(tmp_path, monkeypatch, [imported(tmp_path)])
    engine = FakeEngine()

    with make_client(create_app(settings=settings, engine=engine)) as client:
        response = client.post(
            "/v1/images/generations",
            json={
                "prompt": "a cat",
                "model": "my-z-image",
                "size": "64x64",
                "response_format": "b64_json",
            },
        )

    assert response.status_code == 200, response.text
    # Resolved to the durable internal identity, which is what the engine loads.
    assert [job.spec.key for job in engine.jobs] == ["local-c1587aa663c4"]
    assert engine.jobs[0].spec.public_name == "my-z-image"
    # And the response names it the way the request did.
    assert response.json()["mflux"]["model"] == "my-z-image"


def test_a_generation_request_by_internal_id_also_resolves(tmp_path, monkeypatch):
    settings = settings_with(tmp_path, monkeypatch, [imported(tmp_path)])
    engine = FakeEngine()

    with make_client(create_app(settings=settings, engine=engine)) as client:
        response = client.post(
            "/v1/images/generations",
            json={
                "prompt": "a cat",
                "model": "local-c1587aa663c4",
                "size": "64x64",
                "response_format": "b64_json",
            },
        )

    assert response.status_code == 200, response.text
    assert [job.spec.key for job in engine.jobs] == ["local-c1587aa663c4"]


def test_the_default_model_may_stay_an_internal_id(tmp_path, monkeypatch):
    # `default_model` is a durable reference on purpose: a friendly name would
    # break the moment one were renamed.
    settings = settings_with(
        tmp_path, monkeypatch, [imported(tmp_path)], default_model="local-c1587aa663c4"
    )
    registry = settings.registry(include_disabled=True)
    assert settings.default_model in registry
    assert registry[settings.default_model].public_name == "my-z-image"
