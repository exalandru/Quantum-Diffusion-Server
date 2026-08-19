"""Which parts of a model can be converted on their own.

A saved quantized copy is not one object. mflux writes a model as one directory
per component, each with its own shards and its own `model.safetensors.index.json`
— so a conversion can be done a component at a time, and that is the whole reason
peak memory can be bounded: `transformer` need never be resident while
`text_encoder` is being quantized.

**This table is the contract, and it is Python's.** Two independent constraints
put it here as data rather than derived at call time:

* the desktop app reads the catalogue with the generation server stopped, through
  `qds fetch --status`. That path deliberately imports neither mflux nor
  torch — a multi-second import on every visit to the Models tab — so it cannot
  ask mflux what a family's components are;
* the interface must not own a family→component table, which is what the React
  `COMPONENTS` constant was: three FLUX.2-dev components hard-coded in a `.tsx`
  and shown for every model that reached that branch.

A table restating another library's structure is only as good as its proof, so
`test_components.py` imports each family's real `WeightDefinition` and asserts
this file matches it exactly — name, subdirectory, quantizability and all. Drift
is a test failure rather than a silent lie, which is the same arrangement
`registry._CAPABILITIES` already uses for quantization capability.

Established against mflux 0.18.0, by inspecting each family's
`WeightDefinition.get_components()`, its initializer's `_init_models`, and its
variant class's annotations:

* every supported family has exactly three components, each in its own
  subdirectory, with no shared source between them;
* every component's module class takes no required constructor argument (the
  overrides `flux2` and `ernie` pass come from `ModelConfig`, and have defaults);
* `ModelSaver.save_model` writes only the components present on the object it is
  given, so an incremental save is the library's own behaviour rather than a
  trick played on it.

Families absent from the table publish nothing and convert nothing. That is the
fail-closed direction: an unverified family must not be offered a conversion that
would produce an artifact nobody has checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Conversion order, and the order these are published in.
#:
#: Largest first. For FLUX.2-dev that is load-bearing — the bf16 source of a
#: finished component can be purged from the HuggingFace cache before the next
#: one starts, which is what keeps the disk peak near 97 GB instead of 169 GB —
#: and for everything else it means the run that is most likely to be interrupted
#: happens while the disk is emptiest.
TRANSFORMER = "transformer"
TEXT_ENCODER = "text_encoder"
VAE = "vae"


@dataclass(frozen=True)
class ComponentSpec:
    """One independently convertible part of a model."""

    #: mflux's own component name, which is also the artifact subdirectory and
    #: the value `--components` takes. Never translated.
    key: str
    label: str
    #: Part of the set a *usable* artifact must carry. A conversion that leaves a
    #: required component missing produces partial work, not a variant.
    required: bool
    #: Can be loaded, converted and saved without any other component resident.
    independently_convertible: bool
    #: False when mflux declares `skip_quantization` for it: the component is
    #: still written into the artifact, at its source precision, because a usable
    #: artifact needs it — it just does not shrink.
    quantized: bool
    #: Why, when one of the flags above is off.
    note: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "required": self.required,
            "independently_convertible": self.independently_convertible,
            "quantized": self.quantized,
            "note": self.note,
        }


def _component(key: str, label: str, **overrides: Any) -> ComponentSpec:
    return ComponentSpec(
        key=key,
        label=label,
        required=overrides.pop("required", True),
        independently_convertible=overrides.pop("independently_convertible", True),
        quantized=overrides.pop("quantized", True),
        note=overrides.pop("note", None),
    )


#: The shape every supported family happens to have today — which is a finding,
#: not an assumption the code is allowed to make. Each family names its own list
#: below, and `test_components.py` checks each against mflux separately, so a
#: family that grows a fourth component or drops one fails a test instead of
#: silently converting two thirds of itself.
def _standard() -> tuple[ComponentSpec, ...]:
    return (
        _component(TRANSFORMER, "Transformer"),
        _component(TEXT_ENCODER, "Text encoder"),
        _component(VAE, "VAE"),
    )


_COMPONENTS: dict[str, tuple[ComponentSpec, ...]] = {
    "flux2": _standard(),
    "z-image": _standard(),
    "ernie": _standard(),
    "fibo": _standard(),
    # `QwenWeightDefinition` marks the text encoder `skip_quantization=True`
    # ("Quantization causes significant semantic degradation"), and
    # `WeightApplier` honours that on both save and load. It is still required
    # and still written — at bf16, so this component alone does not get smaller.
    "qwen": (
        _component(TRANSFORMER, "Transformer"),
        _component(
            TEXT_ENCODER,
            "Text encoder",
            quantized=False,
            note="mflux does not quantize this encoder: it is copied at its source precision.",
        ),
        _component(VAE, "VAE"),
    ),
    # QDS's own definition rather than mflux's, for a model mflux 0.18.0 does not
    # ship. Same three components, and the one this whole mechanism was built for.
    "flux2-dev": _standard(),
}


def is_supported(family: str) -> bool:
    """Whether component-wise conversion is established for this family."""
    return family in _COMPONENTS


def components_for(family: str) -> tuple[ComponentSpec, ...]:
    """The convertible components of `family`, in conversion order.

    Empty for a family nobody has verified — which is what makes the conversion
    path fail closed rather than guess a component list from a similar family.
    """
    return _COMPONENTS.get(family, ())


def required_components(family: str) -> tuple[str, ...]:
    """Component keys a complete, usable artifact of `family` must carry."""
    return tuple(spec.key for spec in components_for(family) if spec.required)


def component_keys(family: str) -> tuple[str, ...]:
    return tuple(spec.key for spec in components_for(family))


def spec_for(family: str, key: str) -> ComponentSpec | None:
    for spec in components_for(family):
        if spec.key == key:
            return spec
    return None


def ordered(family: str, keys: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """`keys` in this family's conversion order, dropping anything it does not have.

    The order is the table's, never the caller's: a request that asks for the VAE
    before the transformer still converts the transformer first, because the disk
    argument for that order does not depend on what the user typed.
    """
    wanted = set(keys)
    return tuple(spec.key for spec in components_for(family) if spec.key in wanted)


def unknown(family: str, keys: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Requested components this family does not have. Never silently ignored."""
    known = set(component_keys(family))
    return tuple(key for key in keys if key not in known)


def payload(family: str) -> list[dict[str, Any]]:
    """The published form, for the catalogue and `/v1/capabilities`."""
    return [spec.payload() for spec in components_for(family)]
