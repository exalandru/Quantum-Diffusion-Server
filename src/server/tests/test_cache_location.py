"""Where QDS keeps what it generates, and who decides.

The old location was `~/.cache/mflux-server` — a dot-cache directory named after
the Python package, from before this was an application. It is gone: the default
is the app's own data directory, derived rather than written down, and a setting
can point it anywhere.

Two directories, two independent settings: `storage.hf_home` holds weights that
can be downloaded again, `storage.cache_dir` holds conversions that cannot.
"""

from __future__ import annotations

import json

from mflux_server import artifacts
from mflux_server.fetch import catalogue_status
from mflux_server.settings import Settings, load_settings

from .test_artifacts import write_component


def complete_artifact(cache_root, *, key="z-image", source="mlx-community/Z-Image-bf16", bits=4):
    dest = artifacts.artifact_dir(key, source, bits, base=str(cache_root))
    for name in ("transformer", "text_encoder", "vae"):
        write_component(dest, name, bits=str(bits))
    artifacts.write_record(
        dest,
        model_key=key,
        family="z-image",
        source=source,
        bits=bits,
        strategy="qds_memory_bounded",
        components=("transformer", "text_encoder", "vae"),
        required=("transformer", "text_encoder", "vae"),
    )
    return dest


# ── The default ────────────────────────────────────────────────────────────


def test_the_default_is_the_applications_own_data_directory(monkeypatch):
    """Derived from the bundle identifier and this user's home, not a constant."""
    monkeypatch.delenv("MFLUX_SERVER_CONFIG", raising=False)
    from pathlib import Path

    root = artifacts.default_cache_root()
    assert root == Path.home() / "Library" / "Application Support" / "com.exalandru.qds" / "cache"
    # Home-relative, so two accounts on one Mac get their own.
    assert str(root).startswith(str(Path.home()))


def test_the_packaged_app_gets_its_own_data_directory(monkeypatch, tmp_path):
    """What the desktop app actually hands its children.

    Every child process is launched with `MFLUX_SERVER_CONFIG` pointing at a file
    *inside* the application's data directory, whatever `app_data_dir()` resolved
    to. Deriving from it means the app and its children cannot disagree, and a
    development checkout pointed at its own config gets its own cache.
    """
    data = tmp_path / "Application Support" / "com.exalandru.qds"
    data.mkdir(parents=True)
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(data / "server-config.json"))

    assert artifacts.default_cache_root() == data / "cache"


def test_nothing_resolves_into_the_retired_development_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MFLUX_SERVER_CONFIG", raising=False)
    assert "mflux-server" not in str(artifacts.default_cache_root())
    assert "mflux-server" not in str(artifacts.artifacts_root())

    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    settings = load_settings(strict=False)
    assert "mflux-server" not in settings.effective_cache_dir
    spec = settings.registry(include_disabled=True)["z-image"]
    assert spec.cache_root and "mflux-server" not in spec.cache_root


def test_the_source_tree_names_the_retired_path_nowhere():
    """A grep, because a default that comes back is a default nobody notices."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "mflux_server"
    offenders = [
        str(path)
        for path in root.rglob("*.py")
        if ".cache/mflux-server" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# ── The setting ────────────────────────────────────────────────────────────


def test_a_configured_directory_is_what_gets_used(tmp_path):
    settings = Settings.model_validate({"storage": {"cache_dir": str(tmp_path / "elsewhere")}})
    assert settings.effective_cache_dir == str(tmp_path / "elsewhere")
    assert settings.registry(include_disabled=True)["z-image"].cache_root == str(
        tmp_path / "elsewhere"
    )


def test_the_setting_persists_through_the_configuration_file(monkeypatch, tmp_path):
    config = tmp_path / "server-config.json"
    chosen = tmp_path / "on-an-external-disk"
    config.write_text(json.dumps({"storage": {"cache_dir": str(chosen)}}), encoding="utf-8")
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))

    assert load_settings(strict=False).effective_cache_dir == str(chosen)


def test_a_relative_directory_is_refused(tmp_path):
    """The working directory is `/` for an app launched from Finder."""
    import pytest

    with pytest.raises(ValueError, match="cache_dir"):
        Settings.model_validate({"storage": {"cache_dir": "relative/cache"}})


def test_the_two_storage_settings_are_independent(tmp_path):
    weights = tmp_path / "weights"
    cache = tmp_path / "cache"
    settings = Settings.model_validate(
        {"storage": {"hf_home": str(weights), "cache_dir": str(cache)}}
    )
    assert settings.effective_hf_home == str(weights)
    assert settings.effective_cache_dir == str(cache)

    # Setting one leaves the other on its own default.
    only_weights = Settings.model_validate({"storage": {"hf_home": str(weights)}})
    assert only_weights.effective_hf_home == str(weights)
    assert only_weights.effective_cache_dir == str(artifacts.default_cache_root())

    only_cache = Settings.model_validate({"storage": {"cache_dir": str(cache)}})
    assert only_cache.effective_cache_dir == str(cache)
    assert only_cache.effective_hf_home != str(cache)


# ── What changing it does, and does not do ─────────────────────────────────


def test_discovery_follows_the_configured_directory(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    complete_artifact(cache)
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"storage": {"cache_dir": str(cache)}}), encoding="utf-8")
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    row = {entry["key"]: entry for entry in catalogue_status()["models"]}["z-image"]
    assert [variant["bits"] for variant in row["variants"]] == [4]
    assert row["variants"][0]["path"].startswith(str(cache))


def test_changing_the_directory_moves_nothing(monkeypatch, tmp_path):
    """Tens of gigabytes are not relocated as a side effect of a form field."""
    old = tmp_path / "old-cache"
    dest = complete_artifact(old)
    before = sorted(str(path.relative_to(old)) for path in old.rglob("*"))

    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"storage": {"cache_dir": str(tmp_path / "new-cache")}}), encoding="utf-8"
    )
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    row = {entry["key"]: entry for entry in catalogue_status()["models"]}["z-image"]
    # The new directory is empty, so nothing is found there — and the old
    # directory is exactly as it was, deleted from nothing and copied nowhere.
    assert row["variants"] == []
    assert dest.is_dir()
    assert sorted(str(path.relative_to(old)) for path in old.rglob("*")) == before
    assert not (tmp_path / "new-cache").exists()


def test_a_conversion_writes_into_the_configured_directory(monkeypatch, tmp_path):
    from mflux_server.prequantize import convert

    from .test_artifacts import _Args, component_writer

    cache = tmp_path / "cache"
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"storage": {"cache_dir": str(cache)}}), encoding="utf-8")
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setattr("mflux_server.prequantize.free_gb", lambda _: 10_000.0)
    monkeypatch.setattr("mflux_server.prequantize.convert_component", component_writer([]))

    assert convert(_Args("z-image", 4)) == 0

    written = artifacts.artifact_dir(
        "z-image", "mlx-community/Z-Image-bf16", 4, base=str(cache)
    )
    assert written.is_dir()
    assert str(written).startswith(str(cache))
    assert artifacts.read_record(written) is not None


# ── Unavailable storage ────────────────────────────────────────────────────


def test_an_unmounted_cache_volume_is_reported_rather_than_read_as_empty(monkeypatch, tmp_path):
    """An unplugged disk and an empty one produce the same "no variants"."""
    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"storage": {"cache_dir": "/Volumes/NotMounted/qds-cache"}}), encoding="utf-8"
    )
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    payload = catalogue_status()
    codes = [warning["code"] for warning in payload["warnings"]]
    assert "cache_dir_unavailable" in codes
    message = next(w for w in payload["warnings"] if w["code"] == "cache_dir_unavailable")["message"]
    assert "not mounted" in message
    # And the catalogue still renders: this is a warning, not a failure.
    assert len(payload["models"]) > 5


def test_a_directory_that_does_not_exist_yet_is_not_a_problem(monkeypatch, tmp_path):
    """The first conversion creates it; warning about that teaches nothing."""
    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"storage": {"cache_dir": str(tmp_path / "not-created-yet")}}), encoding="utf-8"
    )
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    assert catalogue_status()["warnings"] == []


def test_the_default_directory_is_never_warned_about(monkeypatch, tmp_path):
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert catalogue_status()["warnings"] == []


# ── The real write path ────────────────────────────────────────────────────
#
# The release blocker these exist for: every test above proved that the
# *setting* resolved correctly, and none of them proved where a conversion
# actually writes. It wrote to the derived default, because the child process
# was spawned without `MFLUX_SERVER_CONFIG` and never saw the setting at all.
# So these drive `convert()` — the real entry point, with only the weight maths
# stubbed — and assert on the directory that receives the bytes.


def run_conversion(monkeypatch, tmp_path, *, cache_dir=None, components=("vae",), bits=4):
    """One real `convert()`, with the heavy component work stubbed out."""
    from mflux_server.prequantize import convert

    from .test_artifacts import _Args, component_writer

    document = {"models": {"z-image": {"enabled": True}}}
    if cache_dir is not None:
        document["storage"] = {"cache_dir": str(cache_dir)}
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr("mflux_server.prequantize.free_gb", lambda _: 10_000.0)

    seen: list[str] = []
    monkeypatch.setattr("mflux_server.prequantize.convert_component", component_writer(seen))
    code = convert(_Args("z-image", bits, components=list(components)))
    return code, seen


def files_under(root) -> list[str]:
    from pathlib import Path

    root = Path(root)
    if not root.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def test_with_no_setting_a_conversion_writes_to_the_derived_default(monkeypatch, tmp_path):
    code, _ = run_conversion(monkeypatch, tmp_path, cache_dir=None)
    assert code == 0
    # The default is derived from the configuration's own directory, which is
    # what the app hands every child.
    assert files_under(tmp_path / "cache")
    assert any("4bit-" in name for name in files_under(tmp_path / "cache"))


def test_a_configured_cache_receives_the_conversion(monkeypatch, tmp_path):
    custom = tmp_path / "custom-cache"
    code, seen = run_conversion(monkeypatch, tmp_path, cache_dir=custom)

    assert code == 0
    assert seen == ["vae"]
    written = files_under(custom)
    assert written, "the configured cache received nothing"
    assert any(name.startswith("artifacts/z-image/4bit-") for name in written)


def test_nothing_at_all_reaches_the_default_when_a_cache_is_configured(monkeypatch, tmp_path):
    """The blocker, stated as the property it violated."""
    custom = tmp_path / "custom-cache"
    run_conversion(monkeypatch, tmp_path, cache_dir=custom)

    assert files_under(tmp_path / "cache") == [], "the derived default was written to"
    assert files_under(custom)


def test_partial_progress_is_recorded_under_the_configured_cache(monkeypatch, tmp_path):
    custom = tmp_path / "custom-cache"
    code, _ = run_conversion(monkeypatch, tmp_path, cache_dir=custom, components=("transformer",))

    assert code == 0
    written = files_under(custom)
    assert any(name.endswith(artifacts.PROGRESS_FILE) for name in written), written
    # Not a variant yet, and not in the default either.
    assert files_under(tmp_path / "cache") == []


def test_a_completed_artifact_lands_entirely_under_the_configured_cache(monkeypatch, tmp_path):
    from mflux_server import availability as av

    custom = tmp_path / "custom-cache"
    code, _ = run_conversion(
        monkeypatch, tmp_path, cache_dir=custom, components=("transformer", "text_encoder", "vae")
    )

    assert code == 0
    written = files_under(custom)
    assert any(name.endswith(av.COMPLETION_MARKER) for name in written)
    assert any(name.endswith("0.safetensors") for name in written)
    assert files_under(tmp_path / "cache") == []


def test_discovery_activation_and_size_all_read_the_configured_cache(monkeypatch, tmp_path):
    custom = tmp_path / "custom-cache"
    run_conversion(
        monkeypatch, tmp_path, cache_dir=custom, components=("transformer", "text_encoder", "vae")
    )

    # Discovery.
    row = {entry["key"]: entry for entry in catalogue_status()["models"]}["z-image"]
    assert [variant["bits"] for variant in row["variants"]] == [4]
    assert row["variants"][0]["path"].startswith(str(custom))
    # Size accounting reads the artifact that is actually there.
    assert row["disk"]["total_bytes"] > 0
    assert any(entry["path"].startswith(str(custom)) for entry in row["disk"]["breakdown"])

    # Activation resolves to it.
    settings = Settings.model_validate(
        {
            "storage": {"cache_dir": str(custom)},
            "models": {"z-image": {"prequantized_variant": 4}},
        }
    )
    spec = settings.registry(include_disabled=True)["z-image"]
    assert spec.effective_model_path.startswith(str(custom))


def test_an_unreachable_configured_cache_refuses_instead_of_falling_back(monkeypatch, tmp_path):
    """Silently writing tens of gigabytes elsewhere is the worst answer here."""
    import pytest

    from mflux_server.prequantize import UnavailableCache

    with pytest.raises(UnavailableCache) as raised:
        run_conversion(monkeypatch, tmp_path, cache_dir="/Volumes/NotMounted/qds-cache")

    assert "unavailable" in str(raised.value)
    assert "not mounted" in str(raised.value)
    # No fallback, and nothing started.
    assert files_under(tmp_path / "cache") == []


def test_the_refusal_happens_before_any_component_is_touched(monkeypatch, tmp_path):
    import pytest

    from mflux_server.prequantize import UnavailableCache, convert

    from .test_artifacts import _Args

    config = tmp_path / "server-config.json"
    config.write_text(
        json.dumps({"storage": {"cache_dir": "/Volumes/NotMounted/qds-cache"}}), encoding="utf-8"
    )
    monkeypatch.setenv("MFLUX_SERVER_CONFIG", str(config))
    monkeypatch.setattr("mflux_server.prequantize.free_gb", lambda _: 10_000.0)
    monkeypatch.setattr(
        "mflux_server.prequantize.convert_component",
        lambda *a, **k: pytest.fail("no component may be converted after the check fails"),
    )

    with pytest.raises(UnavailableCache):
        convert(_Args("z-image", 4))


def test_moving_the_cache_leaves_the_previous_one_untouched(monkeypatch, tmp_path):
    first = tmp_path / "cache-a"
    second = tmp_path / "cache-b"
    run_conversion(
        monkeypatch, tmp_path, cache_dir=first, components=("transformer", "text_encoder", "vae")
    )
    before = files_under(first)
    assert before

    run_conversion(monkeypatch, tmp_path, cache_dir=second, components=("vae",))

    # B receives the new work; A is exactly as it was.
    assert files_under(second)
    assert files_under(first) == before

    # And discovery now follows B, which holds no complete variant.
    row = {entry["key"]: entry for entry in catalogue_status()["models"]}["z-image"]
    assert row["variants"] == []
    assert [partial["bits"] for partial in row["partials"]] == [4]
    assert row["partials"][0]["path"].startswith(str(second))
