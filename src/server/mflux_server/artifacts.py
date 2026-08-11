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

#: The application's bundle identifier, and the directory name macOS gives it
#: under Application Support. The one place either is written down on this side.
BUNDLE_ID = "com.exalandru.qds"

#: What QDS generates, kept under the application's own data directory rather
#: than in a dot-cache folder named after the Python package. Deliberately
#: outside huggingface_hub's cache, which owns its own layout: downloaded weights
#: and generated copies are different kinds of thing with different lifetimes,
#: and one of them can be re-downloaded.
CACHE_DIRNAME = "cache"
ARTIFACTS_DIRNAME = "artifacts"


def default_cache_root() -> Path:
    """Where generated artifacts go when the configuration names no directory.

    Derived rather than written down as a path. The desktop app tells every child
    process where its data lives by handing it `MFLUX_SERVER_CONFIG`, which is a
    file *inside* that directory — so the directory holding the configuration is
    the application's data directory, whatever `app_data_dir()` resolved to on
    this machine and for this user. That covers the packaged app, a development
    checkout pointed at its own config, and two accounts on one Mac, without any
    of them agreeing on a constant.

    The fallback is for a bare `python -m` with nothing configured: the same
    location the app would have used, composed from the home directory and the
    bundle identifier rather than assumed.
    """
    configured = os.environ.get("MFLUX_SERVER_CONFIG")
    if configured:
        return Path(configured).expanduser().parent / CACHE_DIRNAME
    return Path.home() / "Library" / "Application Support" / BUNDLE_ID / CACHE_DIRNAME

#: Written last, and only after the output validates. Version 1 was the flat
#: `{components, bits}` record Slice 3 introduced for FLUX.2-dev; version 2 adds
#: the identity fields, so a v1 marker is read as "8-bit FLUX.2-dev, source
#: unverified" rather than rejected. Version 3 adds per-component state and the
#: measured size, and changes nothing a v1 or v2 marker asserted: both are still
#: read, still valid, and — because they were only ever written after a complete
#: conversion validated — still complete. Nothing is reconverted to gain the new
#: fields.
MARKER_VERSION = 3

#: Component-level progress of a conversion that is *not* finished.
#:
#: A separate file from `COMPLETION_MARKER`, and that separation is the point.
#: The completion marker's meaning — "every required component is here and
#: validated" — is relied on in several places, including
#: `availability.flux2_dev_artifact_state`, which treats its mere presence as
#: proof. Recording partial work in that same file would make a one-component run
#: look like a finished model. So partial work gets its own name, and the
#: completion marker keeps meaning exactly what it always did.
PROGRESS_FILE = ".qds-prequantize-progress"

#: What a component's directory is, once its run has validated.
COMPONENT_COMPLETE = "complete"
#: Nothing on disk for it yet. There is deliberately no third state: a component
#: whose run was cancelled or died leaves whatever the saver had written, which
#: is validated as incomplete and therefore reads as missing.
COMPONENT_MISSING = "missing"


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
    #: The set this artifact needed to be usable. Absent from v1/v2 markers,
    #: which recorded only what they wrote — and wrote only when complete.
    required: tuple[str, ...] = ()
    #: Bytes on disk when the conversion completed. `None` for older markers,
    #: which is a fact about the marker, not a size of zero.
    size_bytes: int | None = None

    @property
    def legacy(self) -> bool:
        """A marker predating identity, or an artifact with none at all."""
        return self.marker_version < 2

    @property
    def expected(self) -> tuple[str, ...]:
        """Components this artifact must carry to be usable.

        v3 records it; v1 and v2 did not, and for them what was written *is* the
        required set, because neither was ever written before the conversion was
        complete.
        """
        return self.required or self.components


@dataclass(frozen=True)
class ConversionProgress:
    """Partial work: which components of one source-and-bits are already done.

    Identity is repeated here for the same reason the completion marker repeats
    it. Continuing a conversion means writing into a directory whose earlier
    contents were produced by an earlier run, and "the path matches" is not
    evidence that the earlier run converted the same weights at the same
    precision by the same route.
    """

    model_key: str | None
    family: str | None
    source: str | None
    bits: int | None
    strategy: str | None
    #: Component key → `COMPONENT_COMPLETE`. Only completed components appear.
    components: dict[str, str]
    #: Component key → bytes written, for the ones that recorded it.
    sizes: dict[str, int]

    def matches(self, *, source: str, bits: int, strategy: str | None) -> bool:
        """Whether this partial work may be continued for the given request.

        A mismatch is not an error and not something to repair: it is other
        work, and the correct response is to ignore it rather than to build half
        an artifact out of one source and half out of another.
        """
        if self.bits is not None and self.bits != bits:
            return False
        if self.strategy is not None and strategy is not None and self.strategy != strategy:
            return False
        if self.source is not None and source_digest(self.source) != source_digest(source):
            return False
        return True

    def completed(self) -> tuple[str, ...]:
        return tuple(
            name for name, state in self.components.items() if state == COMPONENT_COMPLETE
        )


def source_digest(source: str) -> str:
    """Short, stable digest of the effective source a conversion was made from.

    `~` is expanded first so the same directory named two ways does not read as
    two different sources.
    """
    normalised = os.path.expanduser(source).rstrip("/")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


def artifacts_root(base: str | None = None) -> Path:
    """Where this installation's artifacts live.

    `base` is the configured cache directory when there is one. It is threaded
    through every caller rather than read from a global: the same process reads
    the catalogue for one configuration, and a value fetched from the environment
    halfway down would be a second source of truth for a location the user chose.
    """
    return (Path(base).expanduser() if base else default_cache_root()) / ARTIFACTS_DIRNAME


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
    required = payload.get("required")
    size = payload.get("size_bytes")
    return ArtifactRecord(
        marker_version=int(payload.get("marker_version", 1) or 1),
        model_key=payload.get("model_key"),
        family=payload.get("family"),
        source=payload.get("source"),
        bits=payload.get("bits"),
        strategy=payload.get("strategy"),
        components=tuple(components) if isinstance(components, list) else (),
        required=tuple(required) if isinstance(required, list) else (),
        size_bytes=int(size) if isinstance(size, (int, float)) else None,
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
    required: tuple[str, ...] = (),
    size_bytes: int | None = None,
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
        # What "complete" meant for this artifact, recorded rather than left to
        # be re-derived later from a table that may have changed underneath it.
        "required": list(required or components),
        # Measured once, here, so nothing has to walk the artifact to answer
        # "how big is the 4-bit copy" on every visit to the Models tab.
        "size_bytes": directory_size(path) if size_bytes is None else size_bytes,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (path / av.COMPLETION_MARKER).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


# ── Partial work ───────────────────────────────────────────────────────────


def read_progress(path: Path) -> ConversionProgress | None:
    """Component progress recorded in `path`, or None when there is none."""
    try:
        payload = json.loads((path / PROGRESS_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    components = payload.get("components")
    sizes = payload.get("sizes")
    return ConversionProgress(
        model_key=payload.get("model_key"),
        family=payload.get("family"),
        source=payload.get("source"),
        bits=payload.get("bits"),
        strategy=payload.get("strategy"),
        components=dict(components) if isinstance(components, dict) else {},
        sizes={k: int(v) for k, v in (sizes or {}).items()} if isinstance(sizes, dict) else {},
    )


def record_component(
    path: Path,
    *,
    model_key: str,
    family: str,
    source: str,
    bits: int,
    strategy: str,
    component: str,
    size_bytes: int | None = None,
) -> ConversionProgress:
    """Record that one component finished, merging with work already recorded.

    Merging, never replacing: converting the text encoder must not erase the
    knowledge that the transformer is already there, or the next run would redo
    hours of work it did not need to. The merge happens only when the existing
    record is *this* work — same source, same bit depth, same strategy. When it
    is not, the record is replaced outright, because half an artifact of one
    source plus half of another is not an artifact of anything.
    """
    existing = read_progress(path)
    keep = existing is not None and existing.matches(
        source=source, bits=bits, strategy=strategy
    )
    components = dict(existing.components) if keep and existing else {}
    sizes = dict(existing.sizes) if keep and existing else {}
    components[component] = COMPONENT_COMPLETE
    if size_bytes is not None:
        sizes[component] = size_bytes

    payload = {
        "model_key": model_key,
        "family": family,
        "source": source,
        "source_digest": source_digest(source),
        "bits": bits,
        "strategy": strategy,
        "components": components,
        "sizes": sizes,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (path / PROGRESS_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ConversionProgress(
        model_key=model_key,
        family=family,
        source=source,
        bits=bits,
        strategy=strategy,
        components=components,
        sizes=sizes,
    )


def clear_progress(path: Path) -> None:
    """Drop the progress record. Called once completion has been written."""
    try:
        (path / PROGRESS_FILE).unlink()
    except OSError:
        pass


def component_states(
    path: Path,
    *,
    expected: tuple[str, ...],
    source: str | None = None,
    bits: int | None = None,
    strategy: str | None = None,
) -> dict[str, str]:
    """Per-component state of `path`, judged from the filesystem.

    The progress file says what a run *claimed*; `component_is_complete` says
    what is actually on disk. This reports the second, and uses the first only to
    decide whether the work in this directory belongs to the requested source and
    bits at all — a directory full of another conversion's components is not
    partial progress towards this one.
    """
    progress = read_progress(path)
    if progress is not None and source is not None and bits is not None:
        if not progress.matches(source=source, bits=bits, strategy=strategy):
            return {name: COMPONENT_MISSING for name in expected}

    return {
        name: (
            COMPONENT_COMPLETE
            if av.component_is_complete(path / name)
            else COMPONENT_MISSING
        )
        for name in expected
    }


def directory_size(path: Path) -> int:
    """Bytes of real files under `path`.

    Symlinks are not followed and not counted, which is what keeps this honest
    over a HuggingFace snapshot: those entries are links into `blobs/`, and
    following them would count the same bytes twice. Only file metadata is read —
    measured at about half a millisecond for a 59 GB artifact of 36 shards — so
    this is not the "walk tens of gigabytes" that a size like this sounds like.
    """
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


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
        # No completion marker. Partial work recorded by a component run is not a
        # failure and not an artifact: it is a conversion someone has not
        # finished, and saying so is the difference between "convert it again"
        # and "carry on where you left off".
        progress = read_progress(path)
        if progress is not None:
            done = progress.completed()
            return av.PARTIAL, (
                f"conversion in progress: {', '.join(sorted(done))} converted so far"
                if done
                else "conversion started, nothing completed yet"
            )
        # Otherwise fall back to the proven FLUX.2-dev validation, which is what
        # keeps artifacts built before markers existed usable.
        return av.flux2_dev_artifact_state(str(path))

    ok, detail = components_are_complete(path, record.expected or av.REQUIRED_COMPONENTS)
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
    #: Bytes this representation actually occupies. Taken from the marker where a
    #: conversion measured it, measured here otherwise — never estimated from the
    #: source size and a bit depth, which would be a number nobody weighed.
    size_bytes: int | None = None


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
                    size_bytes=(record.size_bytes if record and record.size_bytes else None)
                    or directory_size(child),
                )
            )

    # A model the user has pointed at an artifact directory themselves: that
    # directory is a valid representation of this source, and it is outside the
    # layout, so nothing above would have found it.
    for candidate in (source,):
        path = Path(candidate).expanduser()
        if any(variant.path == str(path) for variant in found):
            continue
        state, _ = av.flux2_dev_artifact_state(str(path))
        if state != av.PRESENT:
            continue
        record = read_record(path)
        found.append(
            Variant(
                bits=(record.bits if record and record.bits else 8),
                path=str(path),
                strategy=(record.strategy if record else None),
                legacy=True,
                size_bytes=(record.size_bytes if record and record.size_bytes else None)
                or directory_size(path),
            )
        )

    return sorted(found, key=lambda variant: variant.bits)


@dataclass(frozen=True)
class PartialConversion:
    """Component work towards a variant that is not usable yet."""

    bits: int
    path: str
    strategy: str | None
    #: Component key → `complete` / `missing`, judged from disk.
    components: dict[str, str]
    size_bytes: int


def discover_partials(
    model_key: str,
    source: str,
    *,
    expected: tuple[str, ...],
    strategy: str | None = None,
    base: str | None = None,
) -> list[PartialConversion]:
    """Unfinished conversions of `source` for `model_key`, cheapest bits first.

    Reported separately from variants, and never mixed into them: a variant is
    something generation can be pointed at, and a partial conversion is
    explicitly something it cannot. Merging the two lists is precisely how a
    half-converted model would come to be offered as usable.
    """
    found: list[PartialConversion] = []
    model_root = artifacts_root(base) / model_key
    try:
        children = sorted(model_root.iterdir())
    except OSError:
        return found

    for child in children:
        if not child.is_dir():
            continue
        # A finished artifact is a variant, not partial work — even when its
        # marker says it was built by a different route.
        if read_record(child) is not None:
            continue
        progress = read_progress(child)
        if progress is None or progress.bits is None:
            continue
        if not progress.matches(source=source, bits=progress.bits, strategy=strategy):
            continue
        if source_digest(str(progress.source or "")) != source_digest(source):
            continue
        found.append(
            PartialConversion(
                bits=progress.bits,
                path=str(child),
                strategy=progress.strategy,
                components=component_states(
                    child,
                    expected=expected,
                    source=source,
                    bits=progress.bits,
                    strategy=strategy,
                ),
                size_bytes=directory_size(child),
            )
        )
    return sorted(found, key=lambda partial: partial.bits)
