"""Saved, already-quantized copies of a model, and how QDS identifies them.

A converted artifact is expensive — tens of gigabytes and, for the largest model,
hours — so the question "is this directory the conversion of *this* source at
*these* bits" has to be answerable from the artifact itself. A path and an
`is_dir()` cannot answer it: change `models.<key>.model_path` to a different repo
and the old directory is still sitting there, still parses, and is no longer the
conversion of anything the user asked for.

So identity is recorded, not inferred:

    <root>/<model key>/<bits>bit-<digest of the effective source>/

The digest in the path means a source change cannot collide with, or quietly
inherit, the previous conversion; the bit depth in the path means a 4-bit
conversion does not destroy a valid 8-bit one. The completion record inside
repeats all of it, so a directory copied elsewhere still declares what it is.

Nothing here imports mflux or torch: the desktop app reads artifact state on
every visit to the Models tab, with the generation server stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from mflux_server import availability as av

#: Root for everything QDS converts. Sibling of the legacy FLUX.2-dev directory
#: rather than inside it, and deliberately outside huggingface_hub's cache, which
#: owns its own layout.
DEFAULT_ARTIFACT_ROOT = "~/.cache/mflux-server"
ARTIFACTS_DIRNAME = "artifacts"

#: Written last, and only after the output validates. Version 1 was the flat
#: `{components, bits}` record Slice 3 introduced for FLUX.2-dev; version 2 adds
#: the identity fields, so a v1 marker is read as "8-bit FLUX.2-dev, source
#: unverified" rather than rejected.
MARKER_VERSION = 2


@dataclass(frozen=True)
class ArtifactRecord:
    """What a completion marker asserts about a directory."""

    marker_version: int
    model_key: str | None
    family: str | None
    source: str | None
    bits: int | None
    strategy: str | None
    components: tuple[str, ...]

    @property
    def legacy(self) -> bool:
        """A marker predating identity, or an artifact with none at all."""
        return self.marker_version < MARKER_VERSION


def source_digest(source: str) -> str:
    """Short, stable digest of the effective source a conversion was made from.

    `~` is expanded first so the same directory named two ways does not read as
    two different sources.
    """
    normalised = os.path.expanduser(source).rstrip("/")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


def artifacts_root(base: str | None = None) -> Path:
    return Path(base or DEFAULT_ARTIFACT_ROOT).expanduser() / ARTIFACTS_DIRNAME


def artifact_dir(model_key: str, source: str, bits: int, *, base: str | None = None) -> Path:
    """Where the conversion of `source` at `bits` for `model_key` belongs."""
    return artifacts_root(base) / model_key / f"{bits}bit-{source_digest(source)}"


def read_record(path: Path) -> ArtifactRecord | None:
    """The completion record in `path`, or None when there is no valid marker."""
    marker = path / av.COMPLETION_MARKER
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    components = payload.get("components")
    return ArtifactRecord(
        marker_version=int(payload.get("marker_version", 1) or 1),
        model_key=payload.get("model_key"),
        family=payload.get("family"),
        source=payload.get("source"),
        bits=payload.get("bits"),
        strategy=payload.get("strategy"),
        components=tuple(components) if isinstance(components, list) else (),
    )


def write_record(
    path: Path,
    *,
    model_key: str,
    family: str,
    source: str,
    bits: int,
    strategy: str,
    components: tuple[str, ...],
) -> None:
    """Record completion. Called last, and only after the output has validated."""
    payload = {
        "marker_version": MARKER_VERSION,
        "model_key": model_key,
        "family": family,
        "source": source,
        "source_digest": source_digest(source),
        "bits": bits,
        "strategy": strategy,
        "components": list(components),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (path / av.COMPLETION_MARKER).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def components_are_complete(path: Path, components: tuple[str, ...]) -> tuple[bool, str | None]:
    """Every named component holds a complete mflux-saved tensor set.

    The same contract Slice 3 established for FLUX.2-dev, applied to whichever
    components the conversion actually wrote — which is why the marker records
    them rather than this module keeping a per-family table.
    """
    missing = [name for name in components if not av.component_is_complete(path / name)]
    if missing:
        return False, f"incomplete or unreadable: {', '.join(missing)}"
    return True, None


def stored_bits(path: Path, components: tuple[str, ...]) -> int | None:
    """The bit depth mflux actually stamped into the saved shards.

    Read back rather than trusted from the request: a conversion that silently
    produced a different precision must not be recorded as the requested one.
    """
    found: set[int] = set()
    for name in components:
        index = path / name / av.INDEX_FILE
        try:
            record = json.loads(index.read_text(encoding="utf-8"))
            level = record.get("metadata", {}).get("quantization_level")
            if level is not None:
                found.add(int(level))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
    if len(found) != 1:
        return None
    return found.pop()


def artifact_state(
    path: Path,
    *,
    expect_source: str | None = None,
    expect_bits: int | None = None,
) -> tuple[av.Availability, str | None]:
    """Whether `path` is a usable artifact for the expected source and bits.

    A directory that validates but was built from a different source is not
    `present`: it is a real artifact of something else, and using it silently is
    exactly what changing `model_path` must never do.
    """
    if not path.is_dir():
        return av.MISSING, f"{path} does not exist"

    record = read_record(path)
    if record is None:
        # No marker at all — fall back to the proven FLUX.2-dev validation, which
        # is what keeps artifacts built before markers existed usable.
        return av.flux2_dev_artifact_state(str(path))

    ok, detail = components_are_complete(path, record.components or av.REQUIRED_COMPONENTS)
    if not ok:
        return av.PARTIAL, detail

    if expect_bits is not None and record.bits is not None and record.bits != expect_bits:
        return av.MISSING, f"this artifact is {record.bits}-bit, not {expect_bits}-bit"

    if expect_source is not None and record.source is not None:
        if source_digest(record.source) != source_digest(expect_source):
            return (
                av.MISSING,
                f"built from {record.source!r}, which is no longer this model's source",
            )

    return av.PRESENT, None


@dataclass(frozen=True)
class Variant:
    """A saved artifact that exists and validates for the current source."""

    bits: int
    path: str
    strategy: str | None
    #: True for an artifact recognised by the Slice 3 rules rather than a marker.
    legacy: bool


def discover_variants(model_key: str, source: str, *, base: str | None = None) -> list[Variant]:
    """Validated saved variants of `source` for `model_key`, cheapest first.

    Derived from the filesystem each time rather than remembered: these are QDS's
    own artifacts, a directory is the whole record, and a list that could go stale
    is exactly the thing the marker exists to avoid. Only artifacts that validate
    *against this source* are returned — one built from a different repo is a real
    artifact of something else, not a variant of this model.
    """
    found: list[Variant] = []
    model_root = artifacts_root(base) / model_key
    try:
        children = sorted(model_root.iterdir())
    except OSError:
        children = []

    for child in children:
        if not child.is_dir():
            continue
        record = read_record(child)
        bits = record.bits if record else None
        state, _ = artifact_state(child, expect_source=source, expect_bits=bits)
        if state == av.PRESENT and bits is not None:
            found.append(
                Variant(
                    bits=bits,
                    path=str(child),
                    strategy=record.strategy if record else None,
                    legacy=bool(record and record.legacy),
                )
            )

    # The legacy FLUX.2-dev artifact lives outside the layout, at the path the
    # catalogue points at, and predates variants entirely. It is still a valid
    # 8-bit representation and must not vanish from the list.
    legacy_path = Path(source).expanduser()
    if not any(variant.path == str(legacy_path) for variant in found):
        state, _ = av.flux2_dev_artifact_state(str(legacy_path))
        if state == av.PRESENT:
            record = read_record(legacy_path)
            found.append(
                Variant(
                    bits=(record.bits if record and record.bits else 8),
                    path=str(legacy_path),
                    strategy=(record.strategy if record else None),
                    legacy=True,
                )
            )

    return sorted(found, key=lambda variant: variant.bits)
