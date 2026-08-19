"""The catalogue survives a configuration the generation server would refuse.

The defect: `default_model` naming a model that is switched off is a real
invariant violation — the server must not start — and it was enforced at
`Settings` construction. Every reader died with it, including
`qds fetch --status`, so the Models view became a traceback and the
switches that would have repaired the configuration went with it.

Two contracts now, and the split is what these tests hold:

* **model management** reads the same file leniently and reports the problem;
* **the generation server** still refuses to start, unchanged.
"""

from __future__ import annotations

import json

import pytest

from qds.fetch import cache_status, catalogue_status
from qds.settings import ConfigError, Settings, load_settings

BROKEN = {"default_model": "z-image-turbo", "models": {"z-image-turbo": {"enabled": False}}}


def write_config(tmp_path, document) -> str:
    path = tmp_path / "server-config.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def with_config(monkeypatch, tmp_path, document):
    monkeypatch.setenv("QDS_SERVER_CONFIG", write_config(tmp_path, document))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


# ── The invariant is still enforced where it belongs ───────────────────────


def test_the_generation_server_still_refuses_an_invalid_default(monkeypatch, tmp_path):
    """Strict is strict: this configuration cannot serve generations."""
    with_config(monkeypatch, tmp_path, BROKEN)
    with pytest.raises(ConfigError) as raised:
        load_settings()
    assert raised.value.code == "default_model_disabled"
    assert "z-image-turbo" in str(raised.value)


def test_repairing_the_default_restores_strict_validity(monkeypatch, tmp_path):
    with_config(monkeypatch, tmp_path, BROKEN)
    with pytest.raises(ConfigError):
        load_settings()

    # Either repair works, and neither is chosen for the user.
    with_config(
        monkeypatch,
        tmp_path,
        {"default_model": "z-image-turbo", "models": {"z-image-turbo": {"enabled": True}}},
    )
    assert load_settings().default_model == "z-image-turbo"

    with_config(
        monkeypatch,
        tmp_path,
        {"default_model": "z-image", "models": {"z-image-turbo": {"enabled": False}}},
    )
    assert load_settings().default_model == "z-image"


def test_a_structural_failure_is_still_fatal_for_everyone(monkeypatch, tmp_path):
    """Leniency covers runtime invariants, not a file nobody can read."""
    path = tmp_path / "server-config.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(path))
    with pytest.raises(ValueError):
        load_settings(strict=False)


def test_an_invalid_model_override_is_still_fatal(monkeypatch, tmp_path):
    with_config(monkeypatch, tmp_path, {"models": {"z-image": {"quantize": 7}}})
    with pytest.raises(ValueError):
        load_settings(strict=False)


# ── Model management stays usable ──────────────────────────────────────────


def test_the_catalogue_renders_with_an_invalid_default(monkeypatch, tmp_path):
    """The whole point: every row, from the configuration that broke the server."""
    with_config(monkeypatch, tmp_path, BROKEN)

    rows = cache_status()
    keys = {row["key"] for row in rows}
    assert "z-image-turbo" in keys
    assert "flux2-dev" in keys
    assert len(rows) > 5
    # The disabled model is reported as disabled — not hidden, not repaired.
    assert {row["key"]: row["enabled"] for row in rows}["z-image-turbo"] is False


def test_the_catalogue_reports_the_problem_beside_the_rows(monkeypatch, tmp_path):
    with_config(monkeypatch, tmp_path, BROKEN)

    payload = catalogue_status()
    assert len(payload["models"]) > 5
    assert [warning["code"] for warning in payload["warnings"]] == ["default_model_disabled"]
    message = payload["warnings"][0]["message"]
    # Actionable, and it names both the model and the two ways out.
    assert "z-image-turbo" in message
    assert "Enable it" in message and "another default model" in message
    assert "Traceback" not in message


def test_a_healthy_configuration_warns_about_nothing(monkeypatch, tmp_path):
    with_config(monkeypatch, tmp_path, {"default_model": "z-image-turbo"})
    payload = catalogue_status()
    assert payload["warnings"] == []
    assert payload["models"]


def test_status_never_repairs_the_configuration(monkeypatch, tmp_path):
    """Reporting is not rewriting: the file is exactly as the user left it."""
    path = write_config(tmp_path, BROKEN)
    monkeypatch.setenv("QDS_SERVER_CONFIG", path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    before = open(path, encoding="utf-8").read()

    catalogue_status()

    assert open(path, encoding="utf-8").read() == before


def test_an_unknown_default_model_is_a_warning_too(monkeypatch, tmp_path):
    with_config(monkeypatch, tmp_path, {"default_model": "sdxl"})
    payload = catalogue_status()
    assert [warning["code"] for warning in payload["warnings"]] == ["unknown_default_model"]
    assert payload["models"], "an unknown default does not empty the catalogue either"


def test_locating_a_model_cannot_take_the_catalogue_down(monkeypatch, tmp_path):
    """The sequence that surfaced the defect, as a property.

    A `model_path` override is a structural fact about one row. Writing one — as
    Locate does — must leave the rest of the catalogue readable, whatever the
    configuration says about a default model.
    """
    located = tmp_path / "weights"
    located.mkdir()
    document = dict(BROKEN)
    document["models"] = dict(BROKEN["models"], **{"ernie-image": {"model_path": str(located)}})
    with_config(monkeypatch, tmp_path, document)

    payload = catalogue_status()
    rows = {row["key"]: row for row in payload["models"]}
    assert rows["ernie-image"]["repo"] == str(located)
    assert rows["ernie-image"]["local"] is True
    # Every unrelated row is still there, and the warning is still local.
    assert len(rows) > 5
    assert [warning["code"] for warning in payload["warnings"]] == ["default_model_disabled"]


def test_the_catalogue_needs_no_generation_server(monkeypatch, tmp_path):
    """Nothing above talks to a server, and that is the point of this path.

    The companion property — that reading the catalogue does not even *import*
    mflux — cannot be asserted here, because this suite has already imported it
    in another module. It is checked in a subprocess instead, by
    `test_cli.test_reading_the_catalogue_does_not_import_mflux_or_torch`.
    """
    with_config(monkeypatch, tmp_path, BROKEN)
    assert catalogue_status()["models"]


# ── Structured failure rather than a traceback ─────────────────────────────


def test_an_expected_config_failure_carries_a_code(monkeypatch, tmp_path):
    with_config(monkeypatch, tmp_path, BROKEN)
    with pytest.raises(ConfigError) as raised:
        load_settings()
    assert raised.value.code == "default_model_disabled"
    assert raised.value.field == "default_model"


def test_the_status_command_answers_a_broken_file_with_structured_json(monkeypatch, tmp_path, capsys):
    """What the supervisor reads instead of a traceback."""
    from qds.fetch import main

    path = tmp_path / "server-config.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr("sys.argv", ["qds", "--status"])

    code = main()
    captured = capsys.readouterr()

    assert code == 2
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "invalid_config"
    assert "is not valid JSON" in payload["error"]["message"]
    # The traceback still exists, on the stream meant for it.
    assert "Traceback" in captured.err
    assert "Traceback" not in payload["error"]["message"]


def test_the_status_command_prints_rows_and_warnings_together(monkeypatch, tmp_path, capsys):
    from qds.fetch import main

    with_config(monkeypatch, tmp_path, BROKEN)
    monkeypatch.setattr("sys.argv", ["qds", "--status"])

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"]
    assert payload["warnings"][0]["code"] == "default_model_disabled"


# ── The invariant, as data ─────────────────────────────────────────────────


def test_disabling_a_non_default_model_breaks_nothing():
    settings = Settings.model_validate(
        {"default_model": "z-image-turbo", "models": {"flux2-dev": {"enabled": False}}}
    )
    assert settings.runtime_issues() == []
