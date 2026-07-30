"""Cache reporting behind the Install button.

Pointed at an empty `HF_HOME` so the result does not depend on what happens to be
downloaded on the machine running the tests. The download path itself is not
tested: it loads real weights, which is precisely what the whole suite avoids.
"""

from __future__ import annotations

import json

from mflux_server.fetch import cache_status


def test_status_lists_the_whole_catalogue_disabled_models_included(monkeypatch, tmp_path):
    """Every entry, whether the server would expose it or not.

    That is the point of going through Rust rather than `/v1/capabilities`: you
    download a model *before* turning it on, so a list of enabled models would be
    the wrong list.
    """
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"models": {"ideogram-4": {"enabled": False}}}), encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    rows = {row["key"]: row for row in cache_status()}

    assert len(rows) == 10
    assert rows["ideogram-4"]["enabled"] is False
    assert rows["z-image-turbo"]["enabled"] is True

    # Nothing is cached in an empty cache, and no exception either — a missing
    # cache directory is the state of a fresh install.
    assert all(row["cached"] is False for row in rows.values())
    assert all(row["size_gb"] == 0.0 for row in rows.values())


def test_status_reports_the_licence_and_the_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(tmp_path / "absent.json"))
    rows = {row["key"]: row for row in cache_status()}

    # The app warns before starting a download that would 401, so it needs both.
    assert rows["fibo"]["gated"] is True
    assert rows["fibo"]["license"] == "CC-BY-NC-4.0"
    assert rows["ernie-image-turbo"]["gated"] is False
    assert rows["ernie-image-turbo"]["license"] == "Apache-2.0"


def test_a_local_artifact_is_not_a_download(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(tmp_path / "absent.json"))
    rows = {row["key"]: row for row in cache_status()}

    # flux2-dev points at a directory produced by `mflux-server-prequantize`, not
    # at a repo: offering to "install" it would send the user down the wrong path.
    assert rows["flux2-dev"]["local"] is True
    assert rows["z-image-turbo"]["local"] is False


def test_a_model_path_override_is_what_gets_reported(monkeypatch, tmp_path):
    """The reported repo has to be the one that would actually be downloaded."""
    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"models": {"z-image": {"model_path": "mlx-community/Z-Image-4bit"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    rows = {row["key"]: row for row in cache_status()}

    assert rows["z-image"]["repo"] == "mlx-community/Z-Image-4bit"
