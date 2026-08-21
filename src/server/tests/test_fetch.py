"""Cache reporting behind the Install button.

Pointed at an empty `HF_HOME` so the result does not depend on what happens to be
downloaded on the machine running the tests. The download path itself is not
tested: it loads real weights, which is precisely what the whole suite avoids.
"""

from __future__ import annotations

import json

from qds.fetch import cache_status


def test_status_lists_the_whole_catalogue_disabled_models_included(monkeypatch, tmp_path):
    """Every entry, whether the server would expose it or not.

    That is the point of going through Rust rather than `/v1/capabilities`: you
    download a model *before* turning it on, so a list of enabled models would be
    the wrong list.
    """
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"models": {"ideogram-4": {"enabled": False}}}), encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))
    rows = {row["key"]: row for row in cache_status()}

    assert len(rows) == 14
    assert rows["ideogram-4"]["enabled"] is False
    assert rows["z-image-turbo"]["enabled"] is True

    # An empty cache on a fresh install: every remote model is genuinely absent,
    # which is `missing` — not an error, and not the unmounted-volume state.
    from qds import availability as av

    remote = [row for row in rows.values() if not row["local"]]
    assert all(row["availability"] == av.MISSING for row in remote)
    assert all(row["size_gb"] == 0.0 for row in rows.values())


def test_status_reports_the_licence_and_the_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "absent.json"))
    rows = {row["key"]: row for row in cache_status()}

    # The app warns before starting a download that would 401, so it needs both.
    assert rows["fibo"]["gated"] is True
    assert rows["fibo"]["license"] == "CC-BY-NC-4.0"
    assert rows["ernie-image-turbo"]["gated"] is False
    assert rows["ernie-image-turbo"]["license"] == "Apache-2.0"


def test_a_local_artifact_is_not_a_download(monkeypatch, tmp_path):
    """A path source is not a download, whichever model carries it.

    The subject used to be flux2-dev, whose catalogue entry pointed at a
    directory *our own converter* writes. That was the defect, not the example:
    a built-in's source is a repository, and a saved artifact is a variant of it.
    The property still holds — it is now witnessed by an entry that genuinely
    reads from a local path, because the user pointed it there.
    """
    config = tmp_path / "server-config.json"
    located = tmp_path / "weights"
    located.mkdir()
    (located / "config.json").write_text("{}", encoding="utf-8")
    config.write_text(
        json.dumps({"models": {"flux2-dev": {"model_path": str(located)}}}), encoding="utf-8"
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))
    rows = {row["key"]: row for row in cache_status()}

    assert rows["flux2-dev"]["local"] is True
    assert rows["flux2-dev"]["can_download"] is False
    assert rows["z-image-turbo"]["local"] is False


def test_flux2_dev_is_a_downloadable_repository_like_every_other_built_in(monkeypatch, tmp_path):
    """The corrective slice's subject, as the catalogue reports it.

    Its source used to be a QDS-generated 8-bit artifact under the old
    development cache, so the model could never be installed, located, or shown
    as anything but present-or-broken.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "absent.json"))
    rows = {row["key"]: row for row in cache_status()}

    assert rows["flux2-dev"]["repo"] == "black-forest-labs/FLUX.2-dev"
    assert rows["flux2-dev"]["local"] is False
    # Install and Locate are both offered off this one field.
    assert rows["flux2-dev"]["can_download"] is True
    assert rows["flux2-dev"]["availability"] == "missing"


def test_a_model_path_override_is_what_gets_reported(monkeypatch, tmp_path):
    """The reported repo has to be the one that would actually be downloaded."""
    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"models": {"z-image": {"model_path": "mlx-community/Z-Image-4bit"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))
    rows = {row["key"]: row for row in cache_status()}

    assert rows["z-image"]["repo"] == "mlx-community/Z-Image-4bit"


def test_the_reported_repo_is_the_repo_install_would_fetch(monkeypatch, tmp_path):
    """M7: status and fetch must never name different repos.

    They did, for exactly the models the workflow tells you to download first:
    `cache_status` applied the config override, while `fetch` resolved through the
    *enabled-only* registry and silently fell back to the raw catalogue spec for
    anything switched off. So the Models tab advertised `Z-Image-4bit` and the
    Install button fetched `Z-Image-bf16`.
    """
    from qds.fetch import resolved_target
    from qds.registry import BASE_SPECS_BY_KEY
    from qds.settings import load_settings

    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps(
            {
                "models": {
                    # Disabled on purpose: that is the case that used to diverge.
                    "z-image": {"enabled": False, "model_path": "mlx-community/Z-Image-4bit"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    reported = {row["key"]: row["repo"] for row in cache_status()}
    settings = load_settings()

    for key in BASE_SPECS_BY_KEY:
        # Precisely what `fetch()` resolves.
        spec = settings.registry(include_disabled=True).get(key) or BASE_SPECS_BY_KEY[key]
        assert reported[key] == resolved_target(spec), f"{key} reports a different repo than it fetches"

    assert reported["z-image"] == "mlx-community/Z-Image-4bit"


def test_an_unmounted_cache_volume_is_not_reported_as_ten_missing_models(monkeypatch, tmp_path):
    """The state that used to offer to re-download the entire catalogue."""
    from qds import availability as av

    absent = av.VOLUMES_ROOT / "QDSWitnessVolumeThatIsNotMounted" / "hf"
    assert not absent.exists(), "this test needs a volume that is genuinely not mounted"
    monkeypatch.setenv("HF_HOME", str(absent))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "absent.json"))

    rows = {row["key"]: row for row in cache_status()}
    remote = [row for row in rows.values() if not row["local"]]

    assert remote, "the catalogue has remote models"
    assert all(row["availability"] == av.VOLUME_UNMOUNTED for row in remote)
    assert all("not mounted" in (row["detail"] or "") for row in remote)


def test_a_local_artifact_path_is_checked_on_disk_not_guessed_from_its_shape(monkeypatch, tmp_path):
    """`local` used to be a claim about the string, never about the filesystem."""
    from qds import availability as av

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "absent.json"))

    rows = {row["key"]: row for row in cache_status()}
    # A repo id is not a local path, and the raw weights are not in this cache.
    assert rows["flux2-dev"]["local"] is False
    assert rows["flux2-dev"]["availability"] == av.MISSING


def test_a_completed_artifact_makes_flux2_dev_present(monkeypatch, tmp_path):
    from qds import availability as av

    dest = tmp_path / "artifact"
    dest.mkdir()
    av.write_completion_marker(dest, bits=8, components=av.REQUIRED_COMPONENTS)

    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"models": {"flux2-dev": {"model_path": str(dest)}}}), encoding="utf-8"
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    rows = {row["key"]: row for row in cache_status()}
    assert rows["flux2-dev"]["availability"] == av.PRESENT


def test_an_empty_artifact_directory_never_reports_present(monkeypatch, tmp_path):
    from qds import availability as av

    dest = tmp_path / "artifact"
    dest.mkdir()  # what the converter creates before downloading anything

    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"models": {"flux2-dev": {"model_path": str(dest)}}}), encoding="utf-8"
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))

    rows = {row["key"]: row for row in cache_status()}
    assert rows["flux2-dev"]["availability"] != av.PRESENT
