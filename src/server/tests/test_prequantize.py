"""Pre-quantization guards. Nothing here converts anything.

`prequantize.py` had no tests at all, and it is the module that writes tens of
gigabytes and used to leave a directory the rest of the app called ready.
"""

from __future__ import annotations

import pytest

from qds import artifacts
from qds import availability as av
from qds.prequantize import (
    COMPONENT_ORDER,
    DISK_MARGIN_GB,
    InsufficientDisk,
    convert,
    free_gb,
    required_free_gb,
)


def test_the_peak_is_the_largest_source_beside_what_is_already_written():
    # Largest-first ordering exists so the peak is bounded; the estimate has to
    # model that, not just sum everything.
    full = required_free_gb(COMPONENT_ORDER)
    assert 100 <= full <= 125, full
    assert required_free_gb(["vae"]) < required_free_gb(["text_encoder"]) < full


def test_the_margin_is_explicit_and_included():
    assert required_free_gb(["vae"]) == pytest.approx(0.4 + 0.4 + DISK_MARGIN_GB, abs=0.1)


def test_component_order_is_respected_whatever_order_is_requested():
    assert required_free_gb(["vae", "transformer"]) == required_free_gb(["transformer", "vae"])


def test_free_space_is_measured_on_a_destination_that_does_not_exist_yet(tmp_path):
    # The destination is created by the conversion, so the check has to walk up.
    assert free_gb(tmp_path / "not" / "created" / "yet") > 0


class _Args:
    """The parsed CLI shape `convert` dispatches on."""

    def __init__(self, dest, components, *, model="flux2-dev", bits=8):
        self.model = model
        self.dest = str(dest)
        self.components = list(components)
        self.repo = "black-forest-labs/FLUX.2-dev"
        self.bits = bits


def test_the_disk_check_refuses_before_anything_heavy_starts(tmp_path, monkeypatch):
    """The point of a preflight: fail now, not two hours and 60 GB in."""
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 1.0)
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)
    called = []
    monkeypatch.setattr(
        "qds.prequantize.convert_component",
        lambda *a, **k: called.append(a),
    )

    dest = tmp_path / "artifact"
    with pytest.raises(InsufficientDisk) as raised:
        convert(_Args(dest, COMPONENT_ORDER))

    assert not called, "no component may be converted after the check fails"
    assert not dest.exists(), "the destination must not even be created"
    assert "GB free" in str(raised.value)
    # The assumption behind the number is stated rather than implied.
    assert "purged" in str(raised.value)


def test_a_finished_conversion_writes_the_marker_last(tmp_path, monkeypatch):
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)

    dest = tmp_path / "artifact"

    def fake_convert(name, *, spec, repo, dest, bits):
        component = dest / name
        component.mkdir(parents=True, exist_ok=True)
        (component / "0.safetensors").write_bytes(b"tensor")
        (component / av.INDEX_FILE).write_text(
            '{"metadata": {"quantization_level": "8"}, "weight_map": {"w": "0.safetensors"}}',
            encoding="utf-8",
        )
        # Every family declares one, and an artifact without it cannot load.
        (dest / "tokenizer").mkdir(parents=True, exist_ok=True)
        (dest / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return 6

    monkeypatch.setattr("qds.prequantize.convert_component", fake_convert)

    assert convert(_Args(dest, COMPONENT_ORDER)) == 0
    assert (dest / av.COMPLETION_MARKER).is_file()
    record = artifacts.read_record(dest)
    assert record is not None
    # Identity, not just completion: which model, which source, which precision.
    assert record.model_key == "flux2-dev"
    assert record.bits == 8
    assert record.strategy == "qds_memory_bounded"
    assert artifacts.artifact_state(dest, expect_bits=8)[0] == av.PRESENT


def test_a_partial_conversion_leaves_no_marker_and_is_not_present(tmp_path, monkeypatch):
    """Cancelled or failed output is preserved, but must never read as ready."""
    monkeypatch.setattr("qds.prequantize.free_gb", lambda _: 10_000.0)
    monkeypatch.delenv("QDS_SERVER_CONFIG", raising=False)

    dest = tmp_path / "artifact"

    def fake_convert(name, *, spec, repo, dest, bits):
        component = dest / name
        component.mkdir(parents=True, exist_ok=True)
        (component / "0.safetensors").write_bytes(b"tensor")
        (component / av.INDEX_FILE).write_text(
            '{"metadata": {"quantization_level": "8"}, "weight_map": {"w": "0.safetensors"}}',
            encoding="utf-8",
        )
        # Every family declares one, and an artifact without it cannot load.
        (dest / "tokenizer").mkdir(parents=True, exist_ok=True)
        (dest / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return 6

    monkeypatch.setattr("qds.prequantize.convert_component", fake_convert)

    # Only one component requested: a real artifact needs all three. Converting
    # a subset is a legitimate way to work through a model in stages, so the run
    # *succeeds* — what it must not do is leave anything claiming to be usable.
    assert convert(_Args(dest, ["vae"])) == 0
    assert not (dest / av.COMPLETION_MARKER).exists()
    assert av.flux2_dev_artifact_state(str(dest))[0] == av.PARTIAL
    assert artifacts.artifact_state(dest)[0] == av.PARTIAL

    # And the one component that did convert is recorded, so the next run can
    # continue rather than start again.
    progress = artifacts.read_progress(dest)
    assert progress is not None
    assert progress.completed() == ("vae",)
    assert progress.bits == 8


# ── A family with more than three components ────────────────────────────────
#
# SD 3.5 is the first, and the reason these tests exist is that `components.py`
# used to record "every supported family has exactly three components" as a
# *finding*. It was true and it was never an invariant — nothing counted them —
# but a finding written down long enough starts being relied on. These pin the
# thing that is actually true: the converter is driven by whatever list a family
# declares.


def test_sd35_declares_five_components_in_conversion_order():
    """Largest first, so the disk peak is reached while the volume is emptiest."""
    from qds import components

    assert components.is_supported("sd35")
    assert components.component_keys("sd35") == (
        "transformer",
        "text_encoder_3",
        "text_encoder_2",
        "text_encoder",
        "vae",
    )
    # All five are required: an artifact missing any one of them cannot be loaded,
    # and must therefore never be advertised as usable.
    assert components.required_components("sd35") == components.component_keys("sd35")
    assert all(spec.independently_convertible for spec in components.components_for("sd35"))
    assert all(spec.quantized for spec in components.components_for("sd35"))

    # The table's order wins over the caller's, as it does for every other family.
    assert components.ordered("sd35", ["vae", "text_encoder_2", "transformer"]) == (
        "transformer",
        "text_encoder_2",
        "vae",
    )
    assert components.unknown("sd35", ["text_encoder_4"]) == ("text_encoder_4",)


def test_the_three_component_families_are_untouched_by_the_generalisation():
    """A recorded snapshot, so widening the table cannot quietly move an existing row.

    The payload is what `/v1/capabilities` publishes and what the Quantization
    dialog renders, so a changed label or a flipped flag here is a user-visible
    change to a model nobody was working on.
    """
    from qds import components

    standard = [
        {
            "key": "transformer",
            "label": "Transformer",
            "required": True,
            "independently_convertible": True,
            "quantized": True,
            "note": None,
        },
        {
            "key": "text_encoder",
            "label": "Text encoder",
            "required": True,
            "independently_convertible": True,
            "quantized": True,
            "note": None,
        },
        {
            "key": "vae",
            "label": "VAE",
            "required": True,
            "independently_convertible": True,
            "quantized": True,
            "note": None,
        },
    ]
    for family in ("flux2", "z-image", "ernie", "fibo", "flux2-dev"):
        assert components.payload(family) == standard, family

    qwen = components.payload("qwen")
    assert [entry["key"] for entry in qwen] == ["transformer", "text_encoder", "vae"]
    assert qwen[1]["quantized"] is False
    assert qwen[0] == standard[0] and qwen[2] == standard[2]


def test_the_t5_tower_is_quantized_block_by_block_rather_than_all_at_once():
    """9.5 GB in 24 blocks. Without a name here it would go through in one pass.

    `_quantize_incrementally` falls back to a single global sweep for a module
    whose block list it cannot find — correct, but not memory-bounded, which is
    the entire point of converting a component at a time.
    """
    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder

    from qds.prequantize import _quantization_units

    encoder = T5Encoder()
    assert _quantization_units(encoder) == list(encoder.t5_blocks)
    assert len(_quantization_units(encoder)) == 24

    # And both CLIP towers, which reach the same helper under `layers`.
    from qds.sd35.clip import SD35ClipG, SD35ClipL

    assert len(_quantization_units(SD35ClipL())) == 12
    assert len(_quantization_units(SD35ClipG())) == 32

    # The VAE has no block list under any of these names and is small enough not
    # to need one; the head sweep covers it.
    from qds.sd35.vae import SD35VAE

    assert _quantization_units(SD35VAE()) == []


def test_every_sd35_component_builds_on_its_own_from_the_rows_configuration():
    """What `prequantize` actually does before converting one component.

    `_build_module` reads the module class off `SD35`'s annotations and sizes it
    from the row's `ModelConfig`. Building the transformer with class defaults
    instead would produce Medium's 24 blocks for a Large conversion — a module
    that then loads non-strictly and leaves most of itself unassigned.
    """
    from qds import components
    from qds.prequantize import _build_module
    from qds.registry import BASE_SPECS_BY_KEY, family_structure, model_config_for

    variant_class, _ = family_structure("sd35")
    for key, expected_blocks in (("sd35-medium", 24), ("sd35-large", 38)):
        model_config = model_config_for(BASE_SPECS_BY_KEY[key])
        built = {
            name: _build_module(variant_class, model_config, name)
            for name in components.component_keys("sd35")
        }
        assert len(built) == 5
        assert len(built["transformer"].transformer_blocks) == expected_blocks
        assert len(built["text_encoder"].layers) == 12
        assert len(built["text_encoder_2"].layers) == 32
        assert len(built["text_encoder_3"].t5_blocks) == 24


def test_a_single_component_download_asks_only_for_the_files_it_reads():
    """SD 3.5's encoder directories hold their weights twice; the glob would fetch both."""
    from qds.prequantize import single_component_definition
    from qds.sd35 import SD35WeightDefinition

    t5 = single_component_definition(SD35WeightDefinition, "text_encoder_3", with_tokenizers=False)
    patterns = t5.get_download_patterns()
    assert "text_encoder_3/model-00001-of-00002.safetensors" in patterns
    assert "text_encoder_3/model-00002-of-00002.safetensors" in patterns
    assert "text_encoder_3/*.safetensors" not in patterns
    assert "text_encoder_3/*.json" in patterns

    # A component that names no files still gets the directory glob, which is what
    # every three-component family relies on — none of them pin `weight_files`.
    transformer = single_component_definition(
        SD35WeightDefinition, "transformer", with_tokenizers=False
    )
    assert "transformer/*.safetensors" in transformer.get_download_patterns()

    from qds.registry import family_structure

    _, z_image = family_structure("z-image")
    for name in ("transformer", "text_encoder", "vae"):
        single = single_component_definition(z_image, name, with_tokenizers=False)
        assert f"{name}/*.safetensors" in single.get_download_patterns()
