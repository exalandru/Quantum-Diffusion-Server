"""Registering a model the user already has on disk.

Importing is deliberately narrow: QDS records *where* a model is and *what* it is,
and never copies the weights. The hard part is the second half — deciding what a
directory contains — and mflux is no help there. Its resolution is entirely
name-based (`ConfigResolution.resolve(model_name=...)`, and the same substring
matching in `mflux-save`), so nothing in the library can look at a folder and say
which family it belongs to.

What *is* reliable is the diffusers metadata a source model carries:
`transformer/config.json` names the implementing class — `ZImageTransformer2DModel`,
and so on. That is a standard field, it costs one small JSON read, and it needs no
weights. The mapping below turns it into a QDS family: one entry per family, not
per catalogue model, and anything unrecognised is refused rather than guessed at.

A saved/pre-quantized artifact is refused outright. `ModelSaver` writes shards and
an index whose metadata is only `{quantization_level, mflux_version}` — no family,
no architecture, and component directories named identically for every family. So
such a directory cannot be identified, and Slice 6 already owns QDS-created
artifacts as *variants of their source model* rather than as models in their own
right.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mflux_server import availability as av

#: huggingface_hub's directory prefix for a cached repository.
REPO_PREFIX_DIR = av.REPO_PREFIX

#: Prefix for generated keys. Keeps imported models in a namespace of their own,
#: so one can never collide with a `BASE_SPECS` key.
KEY_PREFIX = "local-"

#: The diffusers class each supported family implements, read from
#: `transformer/config.json`. One entry per family — deliberately not per
#: catalogue model — and the only place a directory is turned into a family.
CLASS_NAME_TO_FAMILY: dict[str, str] = {
    "ZImageTransformer2DModel": "z-image",
    "ErnieImageTransformer2DModel": "ernie",
    "QwenImageTransformer2DModel": "qwen",
    "FluxTransformer2DModel": "flux2",
    "Flux2Transformer2DModel": "flux2",
    "FIBOTransformer2DModel": "fibo",
}

#: Where the architecture is declared in a source model.
CONFIG_FILE = "config.json"
TRANSFORMER_DIR = "transformer"


@dataclass(frozen=True)
class ImportVerdict:
    """The answer to "can QDS register this directory, and as what?"."""

    ok: bool
    path: str
    availability: av.Availability
    #: Populated only when `ok`.
    family: str | None = None
    class_name: str | None = None
    suggested_name: str | None = None
    reason: str | None = None
    components: tuple[str, ...] = field(default_factory=tuple)
    #: Built-in entries of the same family, to choose the defaults from.
    profiles: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path": self.path,
            "availability": self.availability,
            "family": self.family,
            "class_name": self.class_name,
            "suggested_name": self.suggested_name,
            "reason": self.reason,
            "components": list(self.components),
            "profiles": list(self.profiles),
        }


def new_id() -> str:
    """A fresh opaque identity for a newly registered model.

    Deliberately *not* derived from the path. Identity and location are different
    facts: a model that moves — a volume remounted elsewhere, a directory renamed
    — is still the same registration, and a path-derived key would silently make
    it a different one. Duplicate detection is a separate question, answered by
    comparing normalised paths against the rows already registered.
    """
    return f"{KEY_PREFIX}{uuid.uuid4().hex[:12]}"


#: What a public model identifier may contain. Deliberately narrow: this string
#: travels in JSON request bodies and in URLs, and an OpenAI client should be able
#: to paste it anywhere without quoting.
_API_NAME_SAFE = re.compile(r"[^a-z0-9._-]+")
API_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MAX_API_NAME = 64


def slugify(value: str) -> str:
    """A conservative API-safe name derived from a display name.

    Lossy on purpose. "My Z-Image ✨" becomes `my-z-image`, and if nothing
    survives, the caller supplies a fallback — an empty public identifier is not
    something to paper over silently.
    """
    slug = _API_NAME_SAFE.sub("-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return slug[:MAX_API_NAME]


def api_name_problem(name: str) -> str | None:
    """Why this public name is unusable, or None."""
    if not name:
        return "The API name cannot be empty."
    if len(name) > MAX_API_NAME:
        return f"The API name cannot be longer than {MAX_API_NAME} characters."
    if not API_NAME_PATTERN.match(name):
        return (
            f"{name!r} is not a valid API name. Use lowercase letters, digits, "
            f"'.', '_' or '-', starting with a letter or digit."
        )
    return None


def taken_api_names(models: list[Any]) -> set[str]:
    """Every public identifier already spoken for.

    Both halves matter. A built-in's catalogue key *is* its public name, so an
    imported model may not take one — the request would become ambiguous. And an
    imported model's own internal id stays resolvable as a legacy alias, so that
    too is reserved.
    """
    from mflux_server.registry import BASE_SPECS_BY_KEY

    taken = set(BASE_SPECS_BY_KEY)
    for model in models:
        taken.add(model.id)
        api_name = getattr(model, "api_name", None)
        if api_name:
            taken.add(api_name)
    return taken


def unique_api_name(desired: str, taken: set[str], *, fallback: str = "model") -> str:
    """`desired` if it is free, else the first `-2`, `-3`… that is.

    Deterministic, which is what makes it safe to use for migrating a library
    written before public names existed: the same rows in the same order always
    produce the same names.
    """
    base = slugify(desired) or fallback
    if base not in taken:
        return base
    for suffix in range(2, 1000):
        candidate = f"{base}-{suffix}"[:MAX_API_NAME]
        if candidate not in taken:
            return candidate
    raise ValueError(f"could not find a free API name based on {base!r}")


def compatible_profiles(family: str) -> list[str]:
    """Built-in catalogue entries whose family matches, as base-profile candidates.

    An imported directory tells us its family, not its generation defaults, and a
    family can carry more than one profile — Z-Image and Z-Image Turbo differ in
    step count and guidance. Taking "the first entry for this family" would quietly
    pick one, so the caller is handed the candidates and must choose.
    """
    from mflux_server.registry import BASE_SPECS

    return [spec.key for spec in BASE_SPECS if spec.family == family]


def normalise_path(path: str) -> str:
    """Absolute, `~`-expanded, not symlink-resolved.

    `Path.resolve()` would be wrong here: it canonicalises through the filesystem,
    which cannot represent a directory on a volume that is currently unplugged —
    and holding exactly that state is the point of an imported entry.
    """
    return os.path.abspath(os.path.expanduser(path)).rstrip("/")


def _looks_like_saved_artifact(path: Path) -> bool:
    """Shards and an index, but no source configuration to identify it by."""
    for child in path.iterdir():
        if child.is_dir() and (child / av.INDEX_FILE).is_file():
            if not (child / CONFIG_FILE).is_file():
                return True
    return False


def inspect(raw_path: str) -> ImportVerdict:
    """Decide whether `raw_path` is a model QDS can register, and as what.

    Structural only: no weights are read, and the generation server is not
    involved, so this answers with the server stopped.
    """
    path = Path(normalise_path(raw_path))
    reject = lambda state, reason: ImportVerdict(  # noqa: E731
        ok=False, path=str(path), availability=state, reason=reason
    )

    # Storage first, in Slice 3's vocabulary: an unplugged volume is a different
    # answer from a directory that is not there.
    state, detail = av.local_path_availability(str(path))
    if state == av.VOLUME_UNMOUNTED:
        return reject(av.VOLUME_UNMOUNTED, detail or "the volume is not mounted")
    if state == av.MISSING:
        return reject(av.MISSING, detail or f"{path} does not exist")
    if state == av.UNREADABLE:
        return reject(av.UNREADABLE, detail or f"{path} cannot be read")

    try:
        if _looks_like_saved_artifact(path):
            return reject(
                "invalid",
                "This directory appears to be a saved/pre-quantized MFlux artifact, but it "
                "does not contain enough model-family metadata to import safely as a "
                "standalone model. Import the original/source model instead.",
            )
        components = tuple(sorted(child.name for child in path.iterdir() if child.is_dir()))
    except OSError as exc:
        return reject(av.UNREADABLE, f"{path} cannot be read: {exc}")

    config = path / TRANSFORMER_DIR / CONFIG_FILE
    if not config.is_file():
        return reject(
            "invalid",
            f"No {TRANSFORMER_DIR}/{CONFIG_FILE} in {path}. A source model directory carries "
            f"its architecture there; without it QDS cannot tell which model this is.",
        )

    try:
        declared = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return reject("invalid", f"{config} could not be read: {exc}")

    class_name = declared.get("_class_name") if isinstance(declared, dict) else None
    if not class_name:
        return reject(
            "invalid",
            f"{config} declares no `_class_name`, so the model family cannot be identified.",
        )

    family = CLASS_NAME_TO_FAMILY.get(class_name)
    if family is None:
        # Named, so the value can be reported and the mapping widened knowingly —
        # rather than guessed at from a directory or file name.
        return reject(
            "incompatible",
            f"Unsupported model architecture {class_name!r}. QDS supports: "
            f"{', '.join(sorted(set(CLASS_NAME_TO_FAMILY.values())))}.",
        )

    profiles = compatible_profiles(family)
    if not profiles:
        return reject(
            "incompatible",
            f"{class_name!r} maps to the {family!r} family, but no built-in profile of that "
            f"family exists to take generation defaults from.",
        )

    return ImportVerdict(
        ok=True,
        path=str(path),
        availability=av.PRESENT,
        family=family,
        class_name=class_name,
        suggested_name=path.name,
        components=components,
        profiles=tuple(profiles),
    )


def repo_id_from_cache_path(path: Path | str) -> str | None:
    """The repository a hub-cache directory belongs to, or None.

    huggingface_hub stores `org/name` as `models--org--name`, so a path anywhere
    under such a directory names its repository exactly. That is a *proof* of
    identity — the directory was written by a download of that repo — as opposed
    to a folder of weights that merely looks compatible.

    Returns None for anything else, and None is the honest answer: a raw
    directory of the right shape is not evidence of provenance.
    """
    for part in reversed(Path(path).parts):
        if not part.startswith(REPO_PREFIX_DIR):
            continue
        pieces = part.split("--")
        if len(pieces) == 3 and pieces[1] and pieces[2]:
            return f"{pieces[1]}/{pieces[2]}"
        return None
    return None


@dataclass(frozen=True)
class LocateVerdict:
    """Whether a directory may be bound to a built-in catalogue entry."""

    ok: bool
    path: str
    #: The built-in key this was checked against.
    model: str
    availability: av.Availability | str
    family: str | None = None
    class_name: str | None = None
    reason: str | None = None
    #: The repository the path proves it came from, when it proves one.
    detected_repo: str | None = None
    #: True only when `detected_repo` equals the catalogue's repository. False
    #: means "structurally compatible, provenance unproven" — never "wrong".
    repo_verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path": self.path,
            "model": self.model,
            "availability": self.availability,
            "family": self.family,
            "class_name": self.class_name,
            "reason": self.reason,
            "detected_repo": self.detected_repo,
            "repo_verified": self.repo_verified,
        }


def locate(raw_path: str, model_key: str) -> LocateVerdict:
    """Check a local directory against one built-in catalogue entry.

    Locating is not importing. Importing mints a new catalogue entry the user
    owns; locating tells an entry QDS already ships where its weights are, and the
    built-in's key, family and profile stay exactly what the catalogue says. So
    this validates against *that* entry rather than offering a choice of profiles.

    Fails closed on family: a Z-Image directory offered for `flux2-klein` is
    refused rather than bound and left to fail at load time.
    """
    from mflux_server.registry import BASE_SPECS_BY_KEY

    spec = BASE_SPECS_BY_KEY.get(model_key)
    path = normalise_path(raw_path)
    if spec is None:
        return LocateVerdict(
            ok=False,
            path=path,
            model=model_key,
            availability="invalid",
            reason=f"{model_key!r} is not a built-in model.",
        )

    verdict = inspect(raw_path)
    if not verdict.ok:
        return LocateVerdict(
            ok=False,
            path=verdict.path,
            model=model_key,
            availability=verdict.availability,
            family=verdict.family,
            class_name=verdict.class_name,
            reason=verdict.reason,
        )

    if verdict.family != spec.family:
        return LocateVerdict(
            ok=False,
            path=verdict.path,
            model=model_key,
            availability="incompatible",
            family=verdict.family,
            class_name=verdict.class_name,
            reason=(
                f"This directory holds a {verdict.family!r} model, but {model_key!r} is "
                f"{spec.family!r}. Locating it here would bind weights the model cannot load."
            ),
        )

    detected = repo_id_from_cache_path(verdict.path)
    return LocateVerdict(
        ok=True,
        path=verdict.path,
        model=model_key,
        availability=av.PRESENT,
        family=verdict.family,
        class_name=verdict.class_name,
        detected_repo=detected,
        repo_verified=detected is not None and detected.lower() == spec.repo.lower(),
    )


def availability_of(raw_path: str) -> tuple[av.Availability, str | None]:
    """Current state of an already-registered import.

    Reused by status refreshes, which update availability and never unregister:
    an unplugged disk is a temporary fact about storage, not a decision by the
    user to remove the model.
    """
    return av.local_path_availability(normalise_path(raw_path))
