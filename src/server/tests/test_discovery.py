"""Where QDS looks for downloaded weights, and what it finds there.

The bug these pin: a storage folder that *is* a hub cache reported every model as
missing. QDS treated the configured value as `HF_HOME` and derived `<root>/hub`,
which did not exist — observed against a real cache of five repositories, none of
which were recognised.
"""

from __future__ import annotations

import json

from qds import availability as av
from qds.fetch import cache_status
from qds.settings import Settings


def hub_repo(root, org: str, name: str, *, incomplete: bool = False) -> None:
    """One repository in huggingface_hub's on-disk layout."""
    repo = root / f"models--{org}--{name}"
    revision = "0123456789abcdef0123456789abcdef01234567"
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    (repo / "snapshots" / revision).mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(revision)
    (repo / "blobs" / "aaaa").write_text("{}")
    (repo / "snapshots" / revision / "config.json").symlink_to("../../blobs/aaaa")
    if incomplete:
        (repo / "blobs" / "bbbb.incomplete").write_text("half a file")


def configure(tmp_path, monkeypatch, root) -> None:
    config = tmp_path / "server-config.json"
    config.write_text(json.dumps({"storage": {"hf_home": str(root)}, "models": {}}))
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(config))


def status_of(key: str) -> dict:
    return next(row for row in cache_status() if row["key"] == key)


# ── Which directory is the cache ───────────────────────────────────────────


def test_an_hf_home_root_resolves_to_its_hub_subdirectory(tmp_path):
    (tmp_path / "hub").mkdir()
    assert av.hub_cache_for(tmp_path) == tmp_path / "hub"


def test_a_hub_cache_chosen_directly_resolves_to_itself(tmp_path):
    # The shape that was broken: the user picked the folder their models are in.
    hub_repo(tmp_path, "mlx-community", "Z-Image-bf16")
    assert av.hub_cache_for(tmp_path) == tmp_path


def test_an_empty_root_takes_the_documented_layout(tmp_path):
    # A fresh folder must fill in `hub/` rather than scatter `models--*` at the
    # top level, so a storage directory made by QDS looks like every other one.
    assert av.hub_cache_for(tmp_path) == tmp_path / "hub"


def test_a_root_with_both_prefers_the_hub_subdirectory(tmp_path):
    (tmp_path / "hub").mkdir()
    hub_repo(tmp_path, "mlx-community", "Z-Image-bf16")
    assert av.hub_cache_for(tmp_path) == tmp_path / "hub"


def test_a_folder_of_loose_files_is_not_a_cache(tmp_path):
    # "Do not treat arbitrary raw directories as cache entries": without the
    # `models--` evidence this is just a folder, and downloads belong in `hub/`.
    (tmp_path / "weights.safetensors").write_text("x")
    (tmp_path / "transformer").mkdir()
    assert not av.looks_like_hub_cache(tmp_path)
    assert av.hub_cache_for(tmp_path) == tmp_path / "hub"


def test_an_absent_root_does_not_probe_the_filesystem(tmp_path):
    # An unmounted volume must stay absent so the availability rules can say so,
    # rather than being resolved into something that exists.
    missing = tmp_path / "nope"
    assert av.hub_cache_for(missing) == missing / "hub"
    assert not missing.exists()


# ── What the catalogue then reports ────────────────────────────────────────


def test_an_exact_repo_already_in_the_active_cache_is_present(tmp_path, monkeypatch):
    hub_repo(tmp_path, "mlx-community", "Z-Image-bf16")
    configure(tmp_path, monkeypatch, tmp_path)

    row = status_of("z-image")
    assert row["availability"] == "present"
    # Matched by exact repository identity, never by name resemblance.
    assert row["repo"] == "mlx-community/Z-Image-bf16"
    assert status_of("z-image-turbo")["availability"] == "missing"


def test_a_near_miss_repo_is_not_mistaken_for_the_catalogue_entry(tmp_path, monkeypatch):
    # `Z-Image-Turbo-bf16` contains `Z-Image`, and substring matching would have
    # called `z-image` installed on the strength of it.
    hub_repo(tmp_path, "mlx-community", "Z-Image-Turbo-bf16")
    configure(tmp_path, monkeypatch, tmp_path)

    assert status_of("z-image-turbo")["availability"] == "present"
    assert status_of("z-image")["availability"] == "missing"


def test_the_same_repos_under_an_hf_home_root_are_found_too(tmp_path, monkeypatch):
    hub_repo(tmp_path / "hub", "mlx-community", "Z-Image-bf16")
    configure(tmp_path, monkeypatch, tmp_path)
    assert status_of("z-image")["availability"] == "present"


def test_changing_the_active_root_changes_discovery_immediately(tmp_path, monkeypatch):
    full = tmp_path / "full"
    empty = tmp_path / "empty"
    empty.mkdir()
    hub_repo(full, "mlx-community", "Z-Image-bf16")

    configure(tmp_path, monkeypatch, full)
    assert status_of("z-image")["availability"] == "present"

    # No restart, no cached scan: the next status call reads the new root.
    configure(tmp_path, monkeypatch, empty)
    assert status_of("z-image")["availability"] == "missing"

    configure(tmp_path, monkeypatch, full)
    assert status_of("z-image")["availability"] == "present"


def test_an_interrupted_download_is_still_partial(tmp_path, monkeypatch):
    hub_repo(tmp_path, "mlx-community", "Z-Image-bf16", incomplete=True)
    configure(tmp_path, monkeypatch, tmp_path)
    assert status_of("z-image")["availability"] == "partial"


def test_an_unmounted_volume_is_still_reported_as_such(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, "/Volumes/DefinitelyNotMounted/models")
    row = status_of("z-image")
    assert row["availability"] == "volume_unmounted"
    assert "not mounted" in (row["detail"] or "")


def test_the_environment_published_to_children_names_the_resolved_cache(tmp_path, monkeypatch):
    # mflux and huggingface_hub read `HF_HUB_CACHE`; the scan and the download
    # have to agree about which directory that is.
    hub_repo(tmp_path, "mlx-community", "Z-Image-bf16")
    settings = Settings.model_validate({"storage": {"hf_home": str(tmp_path)}})
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    settings.apply_hf_home()

    import os

    assert os.environ["HF_HOME"] == str(tmp_path)
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path)


def test_an_unreadable_root_is_reported_not_raised(tmp_path, monkeypatch):
    """A root that cannot be stat'ed must not take the catalogue down with it.

    Found in QA: `Path.is_dir()` swallows "not there" but propagates `EACCES`, so
    resolving the hub cache under a permission-denied root raised out of
    `apply_hf_home` and the Models view failed wholesale — instead of reporting
    the one thing that is true, which is that the root is unreadable.
    """
    root = tmp_path / "locked"
    hub_repo(root, "mlx-community", "Z-Image-bf16")
    root.chmod(0o000)
    try:
        configure(tmp_path, monkeypatch, root)
        # Resolution answers rather than raising…
        assert av.hub_cache_for(root) == root / "hub"
        # …and the catalogue is still readable, with the truth about the root.
        row = status_of("z-image")
        assert row["availability"] == "unreadable"
        assert row["detail"]
        # Never an Install button for a root we cannot even read.
        assert row["size_gb"] == 0.0
    finally:
        root.chmod(0o755)


def test_an_unreadable_root_does_not_masquerade_as_a_hub_cache(tmp_path):
    root = tmp_path / "locked2"
    root.mkdir()
    root.chmod(0o000)
    try:
        assert av.looks_like_hub_cache(root) is False
    finally:
        root.chmod(0o755)
