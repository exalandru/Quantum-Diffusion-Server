"""Truthful size accounting: three questions, three answers, none invented.

The catalogue used to publish one number — the HuggingFace repository's size —
and the interface showed it as the model's size whatever representation was
actually in use. A model set to its 4-bit copy loads 5.9 GB; saying 20.5 GB is
answering a question nobody asked.
"""

from __future__ import annotations

from qds import artifacts
from qds import availability as av
from qds.fetch import _disk_report

from .test_artifacts import write_component


class _Spec:
    """Only what `_disk_report` reads."""

    def __init__(self, prequantized_variant=None):
        self.prequantized_variant = prequantized_variant


def variant(bits, path, size):
    return artifacts.Variant(
        bits=bits, path=path, strategy=None, legacy=False, size_bytes=size
    )


def partial(bits, path, size):
    return artifacts.PartialConversion(
        bits=bits, path=path, strategy=None, components={}, size_bytes=size
    )


def test_the_active_size_is_the_representation_generation_will_load():
    report = _disk_report(
        _Spec(prequantized_variant=4),
        source="mlx-community/Z-Image-bf16",
        availability=av.PRESENT,
        info={"size_bytes": 20_500_000_000, "path": "/hf/z"},
        variants=[variant(3, "/a/3bit", 3_900_000_000), variant(4, "/a/4bit", 5_900_000_000)],
        partials=[],
    )
    assert report["active_bytes"] == 5_900_000_000
    assert report["source_bytes"] == 20_500_000_000
    assert report["total_bytes"] == 20_500_000_000 + 3_900_000_000 + 5_900_000_000


def test_the_active_size_is_the_source_when_no_variant_is_selected():
    report = _disk_report(
        _Spec(),
        source="mlx-community/Z-Image-bf16",
        availability=av.PRESENT,
        info={"size_bytes": 20_500_000_000, "path": "/hf/z"},
        variants=[variant(4, "/a/4bit", 5_900_000_000)],
        partials=[],
    )
    assert report["active_bytes"] == report["source_bytes"] == 20_500_000_000


def test_a_selected_variant_that_is_not_there_has_no_size_rather_than_a_guess():
    """Activation and existence are different facts, and this is where they meet."""
    report = _disk_report(
        _Spec(prequantized_variant=8),
        source="mlx-community/Z-Image-bf16",
        availability=av.PRESENT,
        info={"size_bytes": 20_500_000_000, "path": "/hf/z"},
        variants=[variant(4, "/a/4bit", 5_900_000_000)],
        partials=[],
    )
    assert report["active_bytes"] is None


def test_a_model_that_is_not_here_reports_no_disk_usage():
    """Its catalogue size is what a download would cost, not storage in use."""
    report = _disk_report(
        _Spec(),
        source="mlx-community/Z-Image-bf16",
        availability=av.MISSING,
        info={"size_bytes": 20_500_000_000, "path": "/hf/z"},
        variants=[],
        partials=[],
    )
    assert report["source_bytes"] is None
    assert report["active_bytes"] is None
    assert report["total_bytes"] == 0
    assert report["breakdown"] == []


def test_one_directory_that_is_both_source_and_variant_is_counted_once():
    """FLUX.2-dev: its source *is* its 8-bit artifact."""
    report = _disk_report(
        _Spec(),
        source="/artifacts/flux2-dev-mlx-8bit",
        availability=av.PRESENT,
        info={},  # a path, not a cached repo
        variants=[variant(8, "/artifacts/flux2-dev-mlx-8bit", 58_700_000_000)],
        partials=[],
    )
    assert report["total_bytes"] == 58_700_000_000
    assert len(report["breakdown"]) == 1
    assert report["breakdown"][0]["is_source"] is True
    assert report["breakdown"][0]["bits"] == 8


def test_work_in_progress_counts_towards_disk_but_never_towards_usable():
    report = _disk_report(
        _Spec(),
        source="mlx-community/Z-Image-bf16",
        availability=av.PRESENT,
        info={"size_bytes": 20_000_000_000, "path": "/hf/z"},
        variants=[],
        partials=[partial(3, "/a/3bit", 1_000_000_000)],
    )
    assert report["total_bytes"] == 21_000_000_000
    kinds = [entry["kind"] for entry in report["breakdown"]]
    assert kinds == ["source", "partial"]
    # The partial is not what generation would load.
    assert report["active_bytes"] == 20_000_000_000


def test_a_variant_whose_size_nobody_measured_is_left_out_rather_than_estimated():
    report = _disk_report(
        _Spec(),
        source="mlx-community/Z-Image-bf16",
        availability=av.PRESENT,
        info={"size_bytes": 20_000_000_000, "path": "/hf/z"},
        variants=[artifacts.Variant(bits=4, path="/a/4bit", strategy=None, legacy=True, size_bytes=None)],
        partials=[],
    )
    assert [entry["kind"] for entry in report["breakdown"]] == ["source"]
    assert report["total_bytes"] == 20_000_000_000


def test_a_local_source_is_measured_rather_than_taken_from_a_cache_record(tmp_path):
    """An imported or located model has no HuggingFace bookkeeping to read."""
    write_component(tmp_path, "transformer", bits="4")
    report = _disk_report(
        _Spec(),
        source=str(tmp_path),
        availability=av.PRESENT,
        info={},
        variants=[],
        partials=[],
    )
    assert report["source_bytes"] == artifacts.directory_size(tmp_path)
    assert report["source_bytes"] > 0


def test_the_variant_size_recorded_at_completion_is_the_one_reported(tmp_path):
    """Measured once, when the conversion validated — not on every status read."""
    dest = artifacts.artifact_dir("z-image", "src", 4, base=str(tmp_path))
    dest.mkdir(parents=True)
    for name in ("transformer", "text_encoder", "vae"):
        write_component(dest, name, bits="4")
    artifacts.write_record(
        dest,
        model_key="z-image",
        family="z-image",
        source="src",
        bits=4,
        strategy="qds_memory_bounded",
        components=("transformer", "text_encoder", "vae"),
        required=("transformer", "text_encoder", "vae"),
    )

    recorded = artifacts.read_record(dest).size_bytes
    # What the conversion weighed: the artifact's contents at the moment it
    # validated, which is everything except the marker written immediately after.
    assert recorded > 0
    assert recorded == artifacts.directory_size(dest) - (dest / av.COMPLETION_MARKER).stat().st_size

    # And that recorded figure is what is reported, rather than a fresh walk.
    found = artifacts.discover_variants("z-image", "src", base=str(tmp_path))
    assert found[0].size_bytes == recorded
