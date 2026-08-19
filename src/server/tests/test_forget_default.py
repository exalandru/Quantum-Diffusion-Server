"""Forgetting an imported model must never leave `default_model` dangling.

The invariant under test is not "Forget shows a nicer error". It is that the two
durable files this app owns cannot be driven into disagreement: after any Forget,
successful or refused, `server-config.json` still names a model the registry can
resolve. The witness is `load_settings`, which is what the server itself runs.
"""

from __future__ import annotations

import json

import pytest

from qds import import_cli, library, settings

IMPORTED_ID = "local-1234abcd"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A config and a library of our own. Nothing here reads the real ones."""
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    monkeypatch.delenv("QDS_SERVER_DEFAULT_MODEL", raising=False)
    return tmp_path


def write_config(tmp_path, default_model: str) -> None:
    (tmp_path / "server-config.json").write_text(
        json.dumps({"default_model": default_model, "models": {}}), encoding="utf-8"
    )


def register(tmp_path, model_id: str = IMPORTED_ID) -> library.ImportedModel:
    model = library.ImportedModel(
        id=model_id,
        display_name="Imported fixture",
        path=str(tmp_path / "weights"),
        family="z-image",
        base_profile_key="z-image-turbo",
        imported_at="2026-01-01T00:00:00",
    )
    library.save([model])
    return model


def test_forgetting_the_current_default_is_refused_before_anything_is_written(
    isolated, capsys
):
    register(isolated)
    write_config(isolated, IMPORTED_ID)
    # The premise: this config is valid *because* the model is registered.
    assert settings.load_settings().default_model == IMPORTED_ID

    code = import_cli.main(["forget", IMPORTED_ID])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["ok"] is False
    assert payload["code"] == "is_default_model"
    # Actionable: it names what the user has to do, and where.
    assert "another default model" in payload["reason"]
    assert "Configuration" in payload["reason"]


def test_a_refused_forget_leaves_the_registration_and_the_config_intact(isolated, capsys):
    register(isolated)
    write_config(isolated, IMPORTED_ID)
    before = (isolated / library.LIBRARY_FILENAME).read_bytes()

    import_cli.main(["forget", IMPORTED_ID])
    capsys.readouterr()

    # Byte-identical: the refusal happens before `library.save`, so there is no
    # rewrite to be partially correct about.
    assert (isolated / library.LIBRARY_FILENAME).read_bytes() == before
    assert [model.id for model in library.load()] == [IMPORTED_ID]
    # And the server would still start.
    assert settings.load_settings().default_model == IMPORTED_ID


def test_forget_succeeds_once_another_valid_default_is_chosen(isolated, capsys):
    register(isolated)
    write_config(isolated, IMPORTED_ID)
    import_cli.main(["forget", IMPORTED_ID])
    capsys.readouterr()

    # The user does what the message asked, in the Configuration tab.
    write_config(isolated, "z-image-turbo")

    code = import_cli.main(["forget", IMPORTED_ID])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["forgotten"] == IMPORTED_ID
    assert library.load() == []
    # The complete invariant: the config is still valid with the model gone.
    assert settings.load_settings().default_model == "z-image-turbo"


def test_forgetting_a_model_that_is_not_the_default_is_unaffected(isolated, capsys):
    register(isolated)
    write_config(isolated, "z-image-turbo")

    assert import_cli.main(["forget", IMPORTED_ID]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert library.load() == []


def test_the_check_follows_the_environment_override(isolated, capsys):
    # `QDS_SERVER_DEFAULT_MODEL` outranks the file in `load_settings`, so a
    # check reading only the file would clear a model that is still the default.
    register(isolated)
    write_config(isolated, "z-image-turbo")

    import os

    os.environ["QDS_SERVER_DEFAULT_MODEL"] = IMPORTED_ID
    try:
        assert import_cli.main(["forget", IMPORTED_ID]) == 1
        assert json.loads(capsys.readouterr().out)["code"] == "is_default_model"
    finally:
        del os.environ["QDS_SERVER_DEFAULT_MODEL"]
    assert [model.id for model in library.load()] == [IMPORTED_ID]


def test_an_unreadable_config_refuses_rather_than_assuming_the_built_in_default(
    isolated, capsys
):
    # Fail closed: not being able to read the config is not evidence that this
    # model is not the default. Answering with the field default would have been
    # a decision made about a file nobody read.
    register(isolated)
    (isolated / "server-config.json").write_text("{ this is not json", encoding="utf-8")

    assert import_cli.main(["forget", IMPORTED_ID]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "config_unreadable"
    assert [model.id for model in library.load()] == [IMPORTED_ID]


def test_a_missing_config_uses_the_field_default(isolated, capsys):
    # A fresh install has no file yet; the app writes one carrying this default,
    # which is a built-in key and so never an imported id.
    assert settings.configured_default_model() == "z-image-turbo"
    register(isolated)
    assert import_cli.main(["forget", IMPORTED_ID]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_an_unknown_id_is_still_reported_before_the_default_check(isolated, capsys):
    write_config(isolated, IMPORTED_ID)
    assert import_cli.main(["forget", "local-nonexistent"]) == 1
    assert "No imported model" in json.loads(capsys.readouterr().out)["reason"]
