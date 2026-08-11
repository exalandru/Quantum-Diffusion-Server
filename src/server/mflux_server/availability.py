"""Whether a model's weights are actually usable, and if not, why not.

This replaces a single `cached: bool`, which collapsed five different situations
into one word and got two of them dangerously wrong: an unreadable or unmounted
cache root reported *every* model as "not downloaded" and offered to re-download
tens of gigabytes, and a download interrupted halfway reported as cached, which
hid the only control that could have retried it.

**What `PRESENT` claims, precisely.** "In the HuggingFace cache, with no evidence
on disk of an interrupted download." It is deliberately *not* a proof that mflux
can load the model without fetching anything more. Proving that means resolving
each family's `WeightDefinition.get_download_patterns()` through
`PathResolution._find_complete_cached_snapshot`, and importing those definitions
pulls in torch — which would put a multi-second import on the path of every visit
to the Models tab and cost this module its independence from mflux. So the
evidence used here is huggingface_hub's own on-disk bookkeeping and nothing else;
a download killed cleanly between two files still reads as present. Widening this
is a separate decision, not a file rule to invent here.

Nothing in this module imports mflux or torch. That is what keeps model
management working with the generation server stopped.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mflux_server.logs import SERVER_LOGGER

logger = logging.getLogger(f"{SERVER_LOGGER}.availability")

Availability = Literal["present", "partial", "missing", "volume_unmounted", "unreadable"]

PRESENT: Availability = "present"
PARTIAL: Availability = "partial"
MISSING: Availability = "missing"
VOLUME_UNMOUNTED: Availability = "volume_unmounted"
UNREADABLE: Availability = "unreadable"

#: macOS mounts removable and external media under here. QDS is macOS-only, so
#: this is the whole of the "is the storage plugged in" question — no mount
#: framework, no platform abstraction.
VOLUMES_ROOT = Path("/Volumes")


#: huggingface_hub names every cached repository `models--<org>--<name>`, and that
#: prefix is the only thing that makes a directory recognisable as its cache.
REPO_PREFIX = "models--"

#: Where huggingface_hub puts its cache inside `HF_HOME`.
HUB_SUBDIR = "hub"


def looks_like_hub_cache(path: Path) -> bool:
    """True when `path` is itself a HuggingFace hub cache.

    The evidence is one `models--*` directory, which is huggingface_hub's own
    naming and the same thing `scan_cache_dir` looks for. Deliberately narrow: a
    folder of loose weight files is not a cache, and treating one as though it
    were would report models as installed that nothing can load.
    """
    try:
        return any(
            entry.name.startswith(REPO_PREFIX) and entry.is_dir() for entry in path.iterdir()
        )
    except OSError:
        return False


def hub_cache_for(root: Path | str) -> Path:
    """The hub cache inside a configured storage root.

    Two shapes reach this setting, and both are things a user reasonably picks in
    a folder chooser:

    * an `HF_HOME` root, whose cache is `<root>/hub` — huggingface_hub's own
      layout, and what a fresh QDS install creates;
    * the hub cache itself, `<root>/models--org--name/…` — what you get by
      pointing at a folder you have been downloading models into, or by picking
      the `hub` directory of an existing cache.

    Resolving this in one place is the difference between the two working and the
    second silently reporting every model as missing: QDS assumed the first,
    derived `<root>/hub`, and scanned a directory that did not exist. Observed on
    a real cache of five repositories, every one of which read as *not
    installed*.

    An empty or absent root resolves to `<root>/hub`, so a new storage folder
    fills in the documented layout rather than scattering `models--*` at its top
    level. Nothing here creates a directory; an unmounted volume stays absent so
    the availability rules can say so.
    """
    base = Path(root)
    nested = base / HUB_SUBDIR
    # Every probe below is allowed to fail. `Path.is_dir()` swallows "not there"
    # but *propagates* `EACCES`, so a root the process cannot stat — a wrong-owner
    # mount, a directory with its permissions removed — used to raise out of
    # `apply_hf_home` and take the whole catalogue down with it. That is precisely
    # the state `root_state` exists to report as `unreadable`, so resolution falls
    # back to the documented layout and lets it do so.
    if _is_dir(nested):
        return nested
    if looks_like_hub_cache(base):
        return base
    return nested


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


@dataclass(frozen=True)
class RootState:
    """State of the cache root itself, before any individual model is judged."""

    availability: Availability | None  #: None = usable; otherwise applies to every model
    detail: str | None = None

    @property
    def usable(self) -> bool:
        return self.availability is None


def _on_absent_volume(path: Path) -> bool:
    """True when `path` lives on a `/Volumes` mount that is not currently there.

    The discriminator that matters: a cache root that does not exist because the
    machine is fresh is *not* the same as one that does not exist because the SSD
    was unplugged. Only the second may not offer a download button.
    """
    parts = path.parts
    if len(parts) < 3 or Path(parts[0]) != Path("/") or parts[1] != VOLUMES_ROOT.name:
        return False
    # Same guard as everywhere else here: a question about the filesystem is
    # allowed to fail, and a mount point we cannot stat is not a mounted one.
    return not _is_dir(VOLUMES_ROOT / parts[2])


def root_state(root: Path) -> RootState:
    """Classify the cache root. A usable root leaves each model judged on its own.

    Every filesystem question here is asked defensively, because the interesting
    roots are the broken ones. `Path.exists()` and `Path.is_dir()` swallow "not
    there" but *propagate* `EACCES`, so a root whose parent directory cannot be
    stat'ed — a wrong-owner mount, a folder with its permissions removed — used to
    raise straight out of this function and take the whole catalogue with it. The
    honest answer for a root nobody can look at is `unreadable`, which is a state
    this module already has a word for.
    """
    if _on_absent_volume(root):
        volume = VOLUMES_ROOT / root.parts[2]
        return RootState(VOLUME_UNMOUNTED, f"the volume {volume} is not mounted")
    try:
        if not root.exists():
            # A fresh machine: the root appears on the first download. Every model
            # is genuinely absent, which is exactly `missing` — not an error.
            return RootState(None)
        if not root.is_dir():
            return RootState(UNREADABLE, f"{root} exists but is not a directory")
        os.listdir(root)
    except OSError as exc:
        return RootState(UNREADABLE, f"{root} cannot be read: {exc}")
    return RootState(None)


def scan_repos(root: Path) -> tuple[dict[str, Availability], dict[str, dict[str, Any]], RootState]:
    """Availability of every repo in the HuggingFace cache at `root`.

    Returns `(availability_by_repo, info_by_repo, root_state)`. `info_by_repo`
    carries the size and file count for display; it is not used for decisions.
    """
    state = root_state(root)
    if not state.usable:
        return {}, {}, state

    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        cache = scan_cache_dir(cache_dir=root)
    except CacheNotFound:
        # The root exists but holds no cache yet — same meaning as a fresh machine.
        return {}, {}, RootState(None)
    except OSError as exc:
        return {}, {}, RootState(UNREADABLE, f"{root} cannot be read: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        # A cache we cannot parse is not a cache we may call empty.
        logger.debug("Unreadable HuggingFace cache at %s", root, exc_info=True)
        return {}, {}, RootState(UNREADABLE, f"{root} is not a readable HuggingFace cache: {exc}")

    # `<root>/models--org--name/blobs/<etag>.incomplete` is huggingface_hub's own
    # marker for a download that was cut off. Attributing it back to its repo is
    # a path lookup, not a heuristic.
    interrupted: set[str] = set()
    for item in cache.incomplete_files:
        try:
            interrupted.add(item.file_path.parent.parent.name)
        except Exception:  # pragma: no cover - defensive
            continue

    availability: dict[str, Availability] = {}
    info: dict[str, dict[str, Any]] = {}
    for repo in cache.repos:
        info[repo.repo_id] = {
            "size_gb": round(repo.size_on_disk / 1e9, 1),
            "files": repo.nb_files,
            "path": str(repo.repo_path),
        }
        availability[repo.repo_id] = _repo_availability(repo, interrupted)

    # A repo directory `scan_cache_dir` refused to parse — a snapshot pointing at
    # a blob that is gone, for instance — never reaches `cache.repos` at all; it
    # is reported in `cache.warnings` instead. Left at that, such a repo would
    # read as `missing` and offer a fresh multi-gigabyte download over the top of
    # it. Something is on disk, so the honest answer is `partial`.
    for directory in _repo_dirs_on_disk(root):
        availability.setdefault(directory, PARTIAL)
    return availability, info, state


def _repo_dirs_on_disk(root: Path) -> list[str]:
    """Repo ids for every `models--*` directory present, parsed or not."""
    found: list[str] = []
    try:
        entries = list(root.glob("models--*"))
    except OSError:  # pragma: no cover - defensive
        return found
    for entry in entries:
        if entry.is_dir():
            found.append(entry.name.removeprefix("models--").replace("--", "/"))
    return found


def _repo_availability(repo: Any, interrupted: set[str]) -> Availability:
    """Judge one cached repo on huggingface_hub's evidence alone."""
    if repo.repo_path.name in interrupted:
        return PARTIAL
    if not repo.revisions:
        # A repo directory with no snapshot at all: something started and stopped.
        return PARTIAL
    for revision in repo.revisions:
        for cached_file in revision.files:
            # A snapshot entry whose blob is gone is a broken cache link, which is
            # the other shape an interrupted download leaves behind.
            try:
                if not cached_file.blob_path.exists():
                    return PARTIAL
            except OSError:
                return UNREADABLE
    return PRESENT


# ── Local artifacts ────────────────────────────────────────────────────────


def looks_like_repo_id(value: str) -> bool:
    """True when `value` is a HuggingFace `org/name`, not a filesystem path.

    Mirrors mflux's own rule (`PathResolution._is_hf_format`) rather than
    inventing a second one: exactly one slash, and not an explicitly local prefix.
    """
    return "/" in value and value.count("/") == 1 and not value.startswith(("./", "../", "~/", "/"))


def local_path_availability(raw: str) -> tuple[Availability, str | None]:
    """Availability of a weights directory named by path rather than by repo.

    Presence is a filesystem question and is answered as one. The previous
    reporting called any non-repo-shaped string a "local artifact" and stopped
    there, so a path that had never existed still displayed as if it were on disk.
    """
    path = Path(raw).expanduser()
    if _on_absent_volume(path):
        return VOLUME_UNMOUNTED, f"the volume {VOLUMES_ROOT / path.parts[2]} is not mounted"
    if not path.exists():
        return MISSING, f"{path} does not exist"
    if not path.is_dir():
        return UNREADABLE, f"{path} is not a directory"
    try:
        entries = os.listdir(path)
    except OSError as exc:
        return UNREADABLE, f"{path} cannot be read: {exc}"
    if not entries:
        return MISSING, f"{path} is empty"
    return PRESENT, None


# ── The FLUX.2-dev pre-quantized artifact ──────────────────────────────────

#: Written by `mflux-server-prequantize` as its **last** act, so its presence
#: means every requested component converted and validated. Absence does not mean
#: the artifact is bad — artifacts built before this marker existed are validated
#: from mflux's own output instead, see `flux2_dev_artifact_state`.
COMPLETION_MARKER = ".qds-prequantize-complete"

#: The component subdirectories a loadable artifact must carry. Same names, and
#: same order, as `prequantize.COMPONENT_ORDER` — this is QDS's own conversion
#: output, not a restatement of an mflux table.
REQUIRED_COMPONENTS = ("transformer", "text_encoder", "vae")

#: What `ModelSaver._save_weights` writes next to each component's shards.
INDEX_FILE = "model.safetensors.index.json"


def component_is_complete(component_dir: Path) -> bool:
    """True when a component directory holds a complete mflux-saved tensor set.

    This is the legacy witness, and it is mflux's own contract rather than a guess:
    `ModelSaver._save_weights` writes `model.safetensors.index.json` carrying a
    `weight_map` of every tensor to its shard, plus `quantization_level` metadata.
    Verified against a real artifact built by an earlier QDS: 266 mapped tensors
    across one shard, all present. So an artifact predating the completion marker
    still validates, and a half-written one cannot.
    """
    index = component_dir / INDEX_FILE
    try:
        record = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(record, dict):
        return False
    weight_map = record.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("quantization_level"):
        return False
    return all((component_dir / shard).is_file() for shard in set(weight_map.values()))


def flux2_dev_artifact_state(raw: str) -> tuple[Availability, str | None]:
    """Whether the pre-quantized FLUX.2-dev artifact at `raw` is usable.

    A directory existing used to be the whole test, on both sides — `is_dir()` in
    Rust for the dashboard badge, `.exists()` in the registry's pre-flight. Both
    are satisfied by the empty directory the converter creates *before* it
    downloads anything, so a conversion that died in its first minute still
    advertised "artifact present", and the failure surfaced a minute into the next
    generation instead.
    """
    path = Path(raw).expanduser()
    if _on_absent_volume(path):
        return VOLUME_UNMOUNTED, f"the volume {VOLUMES_ROOT / path.parts[2]} is not mounted"
    if not path.is_dir():
        return MISSING, f"{path} does not exist"
    try:
        if (path / COMPLETION_MARKER).is_file():
            return PRESENT, None
        complete = [name for name in REQUIRED_COMPONENTS if component_is_complete(path / name)]
    except OSError as exc:
        return UNREADABLE, f"{path} cannot be read: {exc}"

    if len(complete) == len(REQUIRED_COMPONENTS):
        # Built before the marker existed, and still valid on mflux's own terms.
        return PRESENT, None
    if complete:
        missing = [name for name in REQUIRED_COMPONENTS if name not in complete]
        return PARTIAL, f"incomplete conversion: {', '.join(missing)} missing or unfinished"
    return MISSING, f"{path} holds no completed component"


def write_completion_marker(dest: Path, *, bits: int, components: tuple[str, ...]) -> None:
    """Record that a conversion finished. Called last, and only after validation."""
    payload = {
        "components": list(components),
        "bits": bits,
        "marker_version": 1,
    }
    (dest / COMPLETION_MARKER).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
