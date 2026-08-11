"""Availability of weights, against synthetic cache trees.

Everything here builds its own HuggingFace cache on a tmp_path: no real weights,
no network, no credentials. That matters beyond speed — the states being tested
are precisely the ones that used to make the app offer to re-download tens of
gigabytes.
"""

from __future__ import annotations

import json
import os

import pytest

from mflux_server import availability as av

# ── Building synthetic caches ──────────────────────────────────────────────


def make_repo(root, repo_id: str, *, files=("model.safetensors",), incomplete=False, dangle=False):
    """A repo laid out the way huggingface_hub lays one out."""
    name = f"models--{repo_id.replace('/', '--')}"
    repo = root / name
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / "abc123"
    refs = repo / "refs"
    for directory in (blobs, snapshot, refs):
        directory.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text("abc123", encoding="utf-8")
    for index, filename in enumerate(files):
        blob = blobs / f"deadbeef{index}"
        blob.write_bytes(b"x" * 16)
        link = snapshot / filename
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(blob, link)
        if dangle:
            # The other shape an interrupted download leaves: the snapshot entry
            # survives, the blob it points at does not.
            blob.unlink()
    if incomplete:
        (blobs / "cafebabe.incomplete").write_bytes(b"half a file")
    return repo


def test_reachable_cache_without_the_model_is_missing(tmp_path):
    root = tmp_path / "hub"
    make_repo(root, "someone/else")
    availability, _, state = av.scan_repos(root)

    assert state.usable
    assert "org/wanted" not in availability


def test_a_complete_repo_is_present(tmp_path):
    root = tmp_path / "hub"
    make_repo(root, "org/wanted")
    availability, info, state = av.scan_repos(root)

    assert state.usable
    assert availability["org/wanted"] == av.PRESENT
    assert info["org/wanted"]["files"] == 1


def test_an_interrupted_download_is_partial_not_present(tmp_path):
    """The regression that hid the only control able to finish the download."""
    root = tmp_path / "hub"
    make_repo(root, "org/wanted", incomplete=True)
    availability, _, _ = av.scan_repos(root)

    assert availability["org/wanted"] == av.PARTIAL


def test_a_snapshot_pointing_at_a_missing_blob_is_partial(tmp_path):
    root = tmp_path / "hub"
    make_repo(root, "org/wanted", dangle=True)
    availability, _, _ = av.scan_repos(root)

    assert availability["org/wanted"] == av.PARTIAL


def test_an_unmounted_volume_is_not_everything_missing(tmp_path):
    """The state that used to offer to re-download the whole catalogue."""
    absent = av.VOLUMES_ROOT / "QDSWitnessVolumeThatIsNotMounted" / "hf" / "hub"
    assert not absent.exists(), "this test needs a volume that is genuinely not mounted"

    state = av.root_state(absent)

    assert state.availability == av.VOLUME_UNMOUNTED
    assert "not mounted" in (state.detail or "")


def test_a_missing_root_that_is_not_on_a_volume_is_just_a_fresh_machine(tmp_path):
    # Absence alone must not be escalated: on a new install the cache simply does
    # not exist yet, and every model really is missing.
    state = av.root_state(tmp_path / "never-created")
    assert state.usable


def test_an_unreadable_root_is_reported_as_such(tmp_path):
    root = tmp_path / "hub"
    root.mkdir()
    root.chmod(0o000)
    try:
        state = av.root_state(root)
    finally:
        root.chmod(0o755)

    if os.geteuid() == 0:  # pragma: no cover - root ignores the mode
        pytest.skip("running as root: permissions are not enforced")
    assert state.availability == av.UNREADABLE


def test_a_root_that_is_a_file_is_unreadable(tmp_path):
    root = tmp_path / "hub"
    root.write_text("not a directory", encoding="utf-8")
    assert av.root_state(root).availability == av.UNREADABLE


# ── Local paths ────────────────────────────────────────────────────────────


def test_a_local_path_that_does_not_exist_is_not_present(tmp_path):
    """`local` used to be a guess about the string, never a look at the disk."""
    status, detail = av.local_path_availability(str(tmp_path / "nope"))
    assert status == av.MISSING
    assert "does not exist" in (detail or "")


def test_an_empty_local_directory_is_not_present(tmp_path):
    target = tmp_path / "weights"
    target.mkdir()
    assert av.local_path_availability(str(target))[0] == av.MISSING


def test_repo_ids_and_paths_are_told_apart_the_way_mflux_does():
    assert av.looks_like_repo_id("mlx-community/Z-Image-bf16")
    assert not av.looks_like_repo_id("~/Library/Application Support/com.exalandru.qds/cache")
    assert not av.looks_like_repo_id("/tmp/weights")
    assert not av.looks_like_repo_id("./weights")
    assert not av.looks_like_repo_id("org/name/extra")


# ── The FLUX.2-dev artifact ────────────────────────────────────────────────


def write_component(dest, name: str, *, shards=("0.safetensors",), complete=True, bits="8"):
    """A component directory shaped like `ModelSaver._save_weights` writes it."""
    component = dest / name
    component.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        if complete:
            (component / shard).write_bytes(b"tensor")
    (component / av.INDEX_FILE).write_text(
        json.dumps(
            {
                "metadata": {"quantization_level": bits, "mflux_version": "0.1.0"},
                "weight_map": {f"w{i}": shard for i, shard in enumerate(shards)},
            }
        ),
        encoding="utf-8",
    )
    return component


def test_a_missing_destination_is_not_ready(tmp_path):
    assert av.flux2_dev_artifact_state(str(tmp_path / "absent"))[0] == av.MISSING


def test_an_empty_destination_is_never_ready(tmp_path):
    """The exact false positive: the converter makes this directory up front."""
    dest = tmp_path / "flux2-dev-mlx-8bit"
    dest.mkdir()
    status, _ = av.flux2_dev_artifact_state(str(dest))
    assert status == av.MISSING


def test_an_interrupted_conversion_is_partial(tmp_path):
    dest = tmp_path / "flux2-dev-mlx-8bit"
    write_component(dest, "vae")
    status, detail = av.flux2_dev_artifact_state(str(dest))

    assert status == av.PARTIAL
    assert "transformer" in (detail or "")


def test_a_component_whose_shards_are_missing_is_not_counted(tmp_path):
    dest = tmp_path / "flux2-dev-mlx-8bit"
    for name in av.REQUIRED_COMPONENTS:
        write_component(dest, name, complete=(name != "transformer"))
    status, detail = av.flux2_dev_artifact_state(str(dest))

    assert status == av.PARTIAL
    assert "transformer" in (detail or "")


def test_a_legacy_artifact_without_a_marker_is_still_recognized(tmp_path):
    """Artifacts built by earlier QDS versions must not be declared missing.

    Validated the way mflux's own saver writes them — index file, weight map,
    quantization metadata, shards present — which is what a real 58 GB artifact on
    disk was confirmed to carry.
    """
    dest = tmp_path / "flux2-dev-mlx-8bit"
    for name in av.REQUIRED_COMPONENTS:
        write_component(dest, name)
    assert not (dest / av.COMPLETION_MARKER).exists()

    assert av.flux2_dev_artifact_state(str(dest))[0] == av.PRESENT


def test_a_marker_written_by_a_finished_conversion_is_recognized(tmp_path):
    dest = tmp_path / "flux2-dev-mlx-8bit"
    dest.mkdir()
    av.write_completion_marker(dest, bits=8, components=av.REQUIRED_COMPONENTS)

    assert av.flux2_dev_artifact_state(str(dest))[0] == av.PRESENT


def test_an_index_that_does_not_parse_is_not_complete(tmp_path):
    dest = tmp_path / "flux2-dev-mlx-8bit"
    component = dest / "vae"
    component.mkdir(parents=True)
    (component / av.INDEX_FILE).write_text("{ truncated", encoding="utf-8")

    assert not av.component_is_complete(component)


def test_an_index_without_quantization_metadata_is_not_complete(tmp_path):
    dest = tmp_path / "flux2-dev-mlx-8bit"
    component = dest / "vae"
    component.mkdir(parents=True)
    (component / "0.safetensors").write_bytes(b"tensor")
    (component / av.INDEX_FILE).write_text(
        json.dumps({"metadata": {}, "weight_map": {"w0": "0.safetensors"}}), encoding="utf-8"
    )

    assert not av.component_is_complete(component)
