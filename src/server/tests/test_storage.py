"""Where weights are stored, and who decides.

The hazard this guards is a split brain: status scanning one cache while a
download writes another. Nothing here downloads anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qds import availability as av
from qds.settings import DEFAULT_HF_HOME, Settings, load_settings


def settings_with(tmp_path, storage=None, **extra):
    config = tmp_path / "server-config.json"
    payload = dict(extra)
    if storage is not None:
        payload["storage"] = storage
    config.write_text(json.dumps(payload), encoding="utf-8")
    os.environ["QDS_SERVER_CONFIG"] = str(config)
    return config


def test_unset_hf_home_keeps_the_previous_default(monkeypatch, tmp_path):
    settings_with(tmp_path, storage=None)
    monkeypatch.delenv("HF_HOME", raising=False)

    settings = load_settings()
    assert settings.storage.hf_home is None
    assert settings.effective_hf_home == str(Path(DEFAULT_HF_HOME).expanduser())


def test_an_explicit_absolute_path_is_respected(monkeypatch, tmp_path):
    chosen = tmp_path / "Models" / "HuggingFace"
    settings_with(tmp_path, storage={"hf_home": str(chosen)})
    monkeypatch.setenv("HF_HOME", "/somewhere/else")

    settings = load_settings()
    # Configuration beats an inherited environment: it is what the user chose in
    # the app, the variable is whatever the launcher happened to carry.
    assert settings.effective_hf_home == str(chosen)


def test_an_inherited_environment_is_used_only_when_nothing_is_configured(monkeypatch, tmp_path):
    settings_with(tmp_path, storage=None)
    monkeypatch.setenv("HF_HOME", "/inherited/root")
    assert load_settings().effective_hf_home == "/inherited/root"


def test_a_relative_path_is_refused_with_a_reason(tmp_path):
    settings_with(tmp_path, storage={"hf_home": "models/hf"})
    with pytest.raises(ValueError) as raised:
        load_settings()
    assert "absolute" in str(raised.value)
    # The reason matters: a relative path would resolve against `/` under Finder.
    assert "Finder" in str(raised.value)


def test_a_tilde_path_is_expanded_not_taken_literally(tmp_path):
    settings_with(tmp_path, storage={"hf_home": "~/Models/hf"})
    resolved = load_settings().storage.hf_home
    assert resolved is not None and resolved.startswith("/") and "~" not in resolved


def test_applying_it_publishes_one_root_for_every_child(monkeypatch, tmp_path):
    """§4: status, fetch, prequantize and the server must agree by construction.

    They all reach the root the same way — `apply_hf_home` at the entry point,
    then the environment — so this pins that one function's output rather than
    four call sites.
    """
    chosen = tmp_path / "Volumes-stand-in" / "hf"
    settings_with(tmp_path, storage={"hf_home": str(chosen)})
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    settings = load_settings()
    applied = settings.apply_hf_home()

    from qds.fetch import _cache_dir

    assert applied == str(chosen)
    assert os.environ["HF_HOME"] == str(chosen)
    # What `cache_status` scans, and what huggingface_hub will download into.
    assert _cache_dir() == str(chosen / "hub")


def test_a_stale_inherited_hub_cache_cannot_win_over_the_configured_root(monkeypatch, tmp_path):
    """The split-brain case: HF_HUB_CACHE outranks HF_HOME inside the library."""
    chosen = tmp_path / "chosen"
    settings_with(tmp_path, storage={"hf_home": str(chosen)})
    monkeypatch.setenv("HF_HUB_CACHE", "/stale/hub")

    load_settings().apply_hf_home()
    from qds.fetch import _cache_dir

    assert _cache_dir() == str(chosen / "hub")


def test_changing_the_root_does_not_touch_the_old_one(monkeypatch, tmp_path):
    """Version 1 semantics: nothing is moved, copied, deleted or rewritten."""
    old = tmp_path / "old" / "hub"
    (old / "models--org--kept").mkdir(parents=True)
    (old / "models--org--kept" / "marker").write_text("original", encoding="utf-8")
    before = sorted(p.relative_to(old).as_posix() for p in old.rglob("*"))

    settings_with(tmp_path, storage={"hf_home": str(tmp_path / "new")})
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    settings = load_settings()
    settings.apply_hf_home()

    from qds.fetch import cache_status

    cache_status()

    after = sorted(p.relative_to(old).as_posix() for p in old.rglob("*"))
    assert after == before
    assert (old / "models--org--kept" / "marker").read_text(encoding="utf-8") == "original"
    # And the new root is not conjured into existence just by being configured.
    assert not (tmp_path / "new").exists()


def test_a_configured_path_on_a_mounted_volume_behaves_normally(monkeypatch, tmp_path):
    root = tmp_path / "mounted" / "hf"
    (root / "hub").mkdir(parents=True)
    settings_with(tmp_path, storage={"hf_home": str(root)})
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    load_settings().apply_hf_home()

    from qds.fetch import cache_status

    rows = {row["key"]: row for row in cache_status()}
    remote = [row for row in rows.values() if not row["local"]]
    assert all(row["availability"] == av.MISSING for row in remote)


def test_an_unmounted_configured_volume_reports_volume_unmounted(monkeypatch, tmp_path):
    """§7, through Slice 3's rules rather than a second storage-state model."""
    absent = av.VOLUMES_ROOT / "QDSSlice4VolumeNotMounted" / "HuggingFace"
    assert not absent.exists(), "this test needs a volume that is genuinely not mounted"

    settings_with(tmp_path, storage={"hf_home": str(absent)})
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    load_settings().apply_hf_home()

    from qds.fetch import cache_status

    rows = {row["key"]: row for row in cache_status()}
    remote = [row for row in rows.values() if not row["local"]]
    assert remote
    assert all(row["availability"] == av.VOLUME_UNMOUNTED for row in remote)


def test_reconnecting_restores_the_previous_state_with_no_rewrite(monkeypatch, tmp_path):
    """Recovery is a re-read, not a re-registration."""
    root = tmp_path / "removable" / "hf"
    settings_with(tmp_path, storage={"hf_home": str(root)})
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    load_settings().apply_hf_home()

    from qds.fetch import cache_status

    # Absent, but not under /Volumes: a plain fresh root, so models read `missing`.
    missing_first = {row["key"]: row["availability"] for row in cache_status()}

    (root / "hub").mkdir(parents=True)
    restored = {row["key"]: row["availability"] for row in cache_status()}

    assert missing_first == restored  # nothing re-registered, nothing rewritten


def test_an_unreadable_configured_root_is_not_reported_as_missing(monkeypatch, tmp_path):
    root = tmp_path / "locked" / "hf"
    (root / "hub").mkdir(parents=True)
    (root / "hub").chmod(0o000)
    settings_with(tmp_path, storage={"hf_home": str(root)})
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    load_settings().apply_hf_home()

    from qds.fetch import cache_status

    try:
        rows = {row["key"]: row for row in cache_status()}
    finally:
        (root / "hub").chmod(0o755)

    if os.geteuid() == 0:  # pragma: no cover - root ignores the mode
        pytest.skip("running as root: permissions are not enforced")
    remote = [row for row in rows.values() if not row["local"]]
    assert all(row["availability"] == av.UNREADABLE for row in remote)


def test_unknown_configuration_keys_still_round_trip(tmp_path):
    """Rust writes the file back verbatim, so Python must not be the one to drop keys."""
    settings_with(tmp_path, storage={"hf_home": str(tmp_path / "x")}, futureKey={"a": 1})
    settings = load_settings()
    # Python ignores what it does not know rather than refusing to start...
    assert settings.storage.hf_home is not None
    # ...and the file itself is untouched by loading it.
    raw = json.loads((tmp_path / "server-config.json").read_text(encoding="utf-8"))
    assert raw["futureKey"] == {"a": 1}


def test_the_settings_model_accepts_a_storage_section_directly():
    settings = Settings.model_validate({"storage": {"hf_home": "/tmp/qds-hf"}})
    assert settings.storage.hf_home == "/tmp/qds-hf"
    assert settings.effective_hf_home == "/tmp/qds-hf"
