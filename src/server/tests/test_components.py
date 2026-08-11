"""The component table, checked against the mflux it claims to describe.

`components.py` is a table because the catalogue path may not import mflux — see
its header. A table restating another library's structure is worth exactly as
much as its proof, so this file is that proof: it imports each family's real
`WeightDefinition` and variant class and asserts, per family, that the published
components *are* mflux's components.

These tests import mflux, and therefore torch. They are the only ones here that
do, and they earn it: without them the whole slice rests on a hand-written list.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

from mflux_server import components
from mflux_server.registry import BASE_SPECS_BY_KEY, capability_for, family_structure

#: Every family the catalogue actually ships, so a new one cannot be added to the
#: registry without either an established component list or an explicit refusal.
CATALOGUE_FAMILIES = sorted({spec.family for spec in BASE_SPECS_BY_KEY.values()})

SUPPORTED = sorted(
    {
        spec.family
        for spec in BASE_SPECS_BY_KEY.values()
        if capability_for(spec.family).supports_prequantize
    }
)


def test_every_family_that_may_be_converted_has_an_established_component_list():
    """Capability and components are the same claim, and must not disagree.

    A family that publishes `supports_prequantize` with no component list would
    reach the converter and find nothing to convert; a family with a list but no
    capability would advertise components for a conversion nothing offers.
    """
    assert SUPPORTED, "the catalogue publishes no convertible family at all"
    for family in CATALOGUE_FAMILIES:
        supported = capability_for(family).supports_prequantize
        assert components.is_supported(family) == supported, family
        assert bool(components.components_for(family)) == supported, family


@pytest.mark.parametrize("family", SUPPORTED)
def test_published_components_are_the_families_own_components(family):
    """Name for name, subdirectory for subdirectory, against mflux itself."""
    _, definition = family_structure(family)
    mflux_components = definition.get_components()

    published = components.components_for(family)
    assert {spec.key for spec in published} == {c.name for c in mflux_components}, family

    by_name = {c.name: c for c in mflux_components}
    for spec in published:
        component = by_name[spec.key]
        # The published key is the artifact subdirectory, which is what
        # `--components` selects and what completion validation looks for. A
        # component whose subdirectory differed from its name would break both.
        assert component.hf_subdir == spec.key, (family, spec.key)
        # `skip_quantization` is the library's statement that this component is
        # written but not shrunk. Publishing it as quantized would promise a size
        # reduction that cannot happen.
        assert spec.quantized is not component.skip_quantization, (family, spec.key)


@pytest.mark.parametrize("family", SUPPORTED)
def test_each_component_can_be_built_without_the_others(family):
    """`independently_convertible` is a claim about construction, and it is checked.

    Two things have to hold for a component to be convertible on its own: the
    variant class must say which module class it is, and that class must be
    constructible without the rest of the model. Both are checked here rather
    than discovered at conversion time, when the answer would arrive after a
    download.
    """
    variant, _ = family_structure(family)
    annotations = dict(getattr(variant, "__annotations__", {}))

    for spec in components.components_for(family):
        if not spec.independently_convertible:
            continue
        module_class = annotations.get(spec.key)
        assert module_class is not None, f"{family}.{spec.key} has no annotated module class"
        required = [
            parameter
            for parameter in list(inspect.signature(module_class.__init__).parameters.values())[1:]
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        assert not required, f"{family}.{spec.key} needs {[p.name for p in required]} to construct"


@pytest.mark.parametrize("family", SUPPORTED)
def test_each_family_can_save_one_component_at_a_time(family):
    """The saver writes what the object has, which is what makes a partial run possible."""
    variant, _ = family_structure(family)
    assert hasattr(variant, "save_model"), family

    from mflux.models.common.weights.saving.model_saver import ModelSaver

    # `getattr(model, attr, None)` and `if component is not None` is the line
    # this whole strategy depends on: a model object carrying one component
    # writes one component, rather than failing or writing empty directories.
    source = inspect.getsource(ModelSaver.save_model)
    assert "getattr(model, attr_name, None)" in source
    assert "if component is not None" in source


def test_an_unverified_family_publishes_nothing_and_converts_nothing():
    """Fail closed: the answer for a family nobody checked is not a guess."""
    assert components.components_for("no-such-family") == ()
    assert components.required_components("no-such-family") == ()
    assert components.is_supported("no-such-family") is False
    assert components.payload("no-such-family") == []
    with pytest.raises(ValueError, match="no-such-family"):
        family_structure("no-such-family")


def test_ideogram_is_excluded_on_purpose_rather_than_forgotten():
    """The one catalogue family that must never be offered a conversion."""
    assert capability_for("ideogram4").supports_prequantize is False
    assert components.is_supported("ideogram4") is False


@pytest.mark.parametrize("family", SUPPORTED)
def test_the_required_set_is_explicit_and_complete(family):
    """Every family names the components a usable artifact must carry."""
    required = components.required_components(family)
    assert required, family
    # Nothing may be required that is not also published as a component.
    assert set(required) <= set(components.component_keys(family))


def test_requested_components_are_ordered_by_the_table_not_the_caller():
    """Largest first, whatever order the request arrived in."""
    assert components.ordered("z-image", ["vae", "transformer"]) == ("transformer", "vae")
    assert components.ordered("z-image", ["vae"]) == ("vae",)
    # A component this family does not have is reported, never quietly dropped.
    assert components.unknown("z-image", ["vae", "unet"]) == ("unet",)
    assert components.ordered("z-image", ["unet"]) == ()


def test_qwens_text_encoder_is_published_as_saved_but_not_quantized():
    """The one component whose conversion does not shrink it, and it says so."""
    spec = components.spec_for("qwen", "text_encoder")
    assert spec is not None
    assert spec.required is True
    assert spec.quantized is False
    assert spec.note and "precision" in spec.note


def test_react_owns_no_component_table_of_its_own():
    """The interface renders the published components; it does not remember them.

    `Models.tsx` used to carry FLUX.2-dev's three component names in a `const`,
    shown for whatever model reached that branch. The same shape of check as
    `test_react_keeps_no_quantization_table_of_its_own`, for the same reason: a
    list of model parts in a `.tsx` is a table that goes stale silently.
    """
    src = pathlib.Path(__file__).resolve().parents[2] / "desktop" / "src"
    if not src.is_dir():  # pragma: no cover - server-only checkout
        pytest.skip("desktop sources not present")

    offenders = []
    for path in sorted(src.rglob("*.tsx")) + sorted(src.rglob("*.ts")):
        if path.name.endswith(".test.tsx") or path.name.endswith(".test.ts"):
            continue
        if path.name == "test-fixtures.ts":
            continue
        text = path.read_text(encoding="utf-8")
        # The names only ever appeared together as a hard-coded component list.
        if "text_encoder" in text and "transformer" in text and "vae" in text:
            offenders.append(str(path))
    assert not offenders, f"React still names a family's components: {offenders}"


def test_the_published_payload_is_what_the_interface_needs():
    """Shape, not just content: the fields the dialog renders must be there."""
    published = components.payload("z-image")
    assert published
    for entry in published:
        assert set(entry) == {
            "key",
            "label",
            "required",
            "independently_convertible",
            "quantized",
            "note",
        }
        assert entry["label"] and entry["label"] != entry["key"] or entry["key"] == entry["label"]
