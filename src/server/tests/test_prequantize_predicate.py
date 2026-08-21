"""The converter's predicate handling, held against the mflux it must agree with.

QDS does not quantize through `WeightApplier`: the converter walks a component
block by block, so that the memory peak is one block plus the quantized result
rather than the whole component in source precision (see `prequantize.py`). That
loop calls `nn.quantize` itself, which means it also inherits a contract it did
not write — how a family's `quantization_predicate` is invoked.

The contract has two forms, `(path, module)` and `(path, module, bits)`, and the
three-argument one is the interesting half: it is how a family varies precision
per layer with the requested level. Qwen uses it to hold `.img_mod_linear` at
8-bit while the rest of the transformer goes to 4. Because such a predicate
declares `bits` with a default, calling it with two arguments does not raise --
it quietly returns the uniform answer, and the artifact loses a per-layer
decision that mflux would have made on the load path.

So these tests do not check that QDS *has* a shim. They check that QDS's shim
and mflux's own produce the same answers, for the real family predicates, at
every bit depth the converter offers.
"""

from __future__ import annotations

import pytest

from qds.prequantize import _predicate_for_bits
from qds.registry import BASE_SPECS_BY_KEY, QUANTIZE_CHOICES, capability_for, family_structure

#: Only families the converter actually reaches. A family that publishes no
#: prequantize capability has no `family_structure` branch to ask for.
CONVERTIBLE = sorted(
    {
        spec.family
        for spec in BASE_SPECS_BY_KEY.values()
        if capability_for(spec.family).supports_prequantize
    }
)

#: Paths chosen to exercise both the uniform answer and the per-layer exception.
#: `.img_mod_linear` is Qwen's protected layer (upstream #484); the others stand
#: for ordinary transformer and text-encoder weights.
PROBE_PATHS = (
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.img_mod_linear",
    "transformer_blocks.7.img_mod_linear",
    "single_transformer_blocks.3.proj_out",
    "layers.0.self_attn.q_proj",
    "proj_out",
)


class _Quantizable:
    """Stands in for an `nn.Linear`: what a predicate inspects, and nothing else."""

    def __init__(self, last_dim: int = 128):
        self.weight = _Shape((last_dim, last_dim))

    def to_quantized(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("the predicate must not quantize anything")


class _Shape:
    def __init__(self, shape):
        self.shape = shape


def _mflux_reference(predicate, bits):
    """mflux's own resolution, which the converter's must match.

    Imported inside the test because it pulls in torch, and because the import
    failing *is* a result: if mflux stops resolving predicates this way, the
    converter's assumption needs re-reading, not silently keeping.
    """
    from mflux.models.common.weights.loading.weight_applier import WeightApplier

    return WeightApplier._predicate_with_bits(predicate, bits)


@pytest.mark.parametrize("family", CONVERTIBLE)
@pytest.mark.parametrize("bits", QUANTIZE_CHOICES)
def test_the_converter_resolves_predicates_exactly_as_mflux_does(family, bits):
    """Same predicate, same bit depth, same answer -- per layer, not on average."""
    _, definition = family_structure(family)
    predicate = definition.quantization_predicate

    ours = _predicate_for_bits(predicate, bits)
    theirs = _mflux_reference(predicate, bits)

    for path in PROBE_PATHS:
        module = _Quantizable()
        assert ours(path, module) == theirs(path, module), (family, bits, path)


def test_a_per_layer_exception_survives_the_converter():
    """The case the two-argument call used to lose.

    Without binding `bits`, Qwen's 4-bit predicate sees `bits=None`, never takes
    its `.img_mod_linear` branch, and the converter writes a uniformly 4-bit
    transformer -- loadable, but not the artifact mflux describes.
    """
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

    predicate = QwenWeightDefinition.quantization_predicate
    module = _Quantizable()
    path = "transformer_blocks.0.img_mod_linear"

    assert predicate(path, module) is True, "the unbound call is the bug, not the fix"
    assert _predicate_for_bits(predicate, 4)(path, module) == {"bits": 8}
    assert _predicate_for_bits(predicate, 8)(path, module) is True


def test_a_two_argument_predicate_is_left_alone():
    """Most families do not vary by level, and must not be handed a third argument."""

    def predicate(path, module):
        return hasattr(module, "to_quantized")

    assert _predicate_for_bits(predicate, 4) is predicate


def test_no_predicate_stays_no_predicate():
    """`nn.quantize` treats `None` as its own default; wrapping it would not."""
    assert _predicate_for_bits(None, 4) is None
