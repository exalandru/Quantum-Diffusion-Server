"""The user's library of imported models: durable, user-created, and small.

Kept in its own file rather than in `server-config.json` because it is a different
kind of state. The config is settings — things with defaults, which the app may
rewrite wholesale and which can be reconstructed by shipping new defaults. These
rows are records of a decision the user made, and losing one means losing the
knowledge that a model exists at all.

Two rules shape the reading code. A row QDS cannot understand is **skipped and
reported**, never dropped from the file and never fatal: one bad entry must not
make the other imports — or the built-in catalogue — unavailable. And a file
written by a *newer* QDS is refused outright rather than parsed optimistically,
because rewriting it with an older schema is how user data disappears quietly.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mflux_server import importing
from mflux_server.importing import normalise_path
from mflux_server.logs import SERVER_LOGGER

logger = logging.getLogger(f"{SERVER_LOGGER}.library")

#: Bumped only for a change older versions could not read correctly.
#:
#: 2 added `api_name`, the public identifier an OpenAI client sends. A version-1
#: file is read without complaint — every row simply gets a name derived from its
#: display name, deterministically, so the same file always yields the same
#: aliases. See `_migrate`.
SCHEMA_VERSION = 2

LIBRARY_FILENAME = "imported-models.json"


class LibraryTooNew(RuntimeError):
    """The file was written by a newer QDS than this one."""


@dataclass(frozen=True)
class ImportedModel:
    """One registered local model. Identity is the `id`, never the path."""

    id: str
    display_name: str
    path: str
    family: str
    #: The built-in entry whose generation defaults this model borrows. Recorded,
    #: because "the first entry of this family" is a guess and families have more
    #: than one profile.
    base_profile_key: str
    imported_at: str
    last_seen: str | None = None
    #: The public, machine-facing identifier: what a client puts in
    #: `{"model": ...}`. Distinct from `id`, which is durable and opaque, and from
    #: `display_name`, which is for people and may one day be renameable. Empty
    #: only in a version-1 file, and filled in deterministically on load.
    api_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "api_name": self.api_name,
            "path": self.path,
            "family": self.family,
            "base_profile_key": self.base_profile_key,
            "imported_at": self.imported_at,
            "last_seen": self.last_seen,
        }


def library_path(base: str | None = None) -> Path:
    """Where the library lives. Beside the config, in the app's data directory."""
    if base:
        return Path(base).expanduser() / LIBRARY_FILENAME
    from mflux_server.settings import config_path

    return config_path().parent / LIBRARY_FILENAME


def _row(payload: Any) -> ImportedModel | None:
    """One row, or None when it cannot be trusted."""
    if not isinstance(payload, dict):
        return None
    required = ("id", "path", "family", "base_profile_key")
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in required):
        return None
    return ImportedModel(
        id=payload["id"],
        display_name=payload.get("display_name") or Path(payload["path"]).name,
        # Absent in a version-1 file; `_migrate` fills it in.
        api_name=payload.get("api_name") or "",
        path=payload["path"],
        family=payload["family"],
        base_profile_key=payload["base_profile_key"],
        imported_at=payload.get("imported_at") or "",
        last_seen=payload.get("last_seen"),
    )


#: Set by `load` when the file existed but could not be read. Checked by status
#: so an unreadable library cannot masquerade as "you have imported nothing" —
#: the registrations may all still be there, behind a permissions or disk fault.
last_load_error: str | None = None


def load(base: str | None = None) -> list[ImportedModel]:
    """Every readable row. Unreadable ones are warned about and skipped."""
    global last_load_error
    last_load_error = None
    path = library_path(base)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        # An unreadable library must not take the built-in catalogue down with it,
        # but it must not pass for an empty one either.
        logger.warning("Imported-model library at %s is unreadable: %s", path, exc)
        last_load_error = f"{path} could not be read: {exc}"
        return []

    if not isinstance(payload, dict):
        logger.warning("Imported-model library at %s is not an object; ignoring", path)
        last_load_error = f"{path} does not contain a JSON object"
        return []

    version = payload.get("version")
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise LibraryTooNew(
            f"{path} was written by a newer version of QDS (schema {version}, this build "
            f"understands {SCHEMA_VERSION}). Update QDS rather than letting it rewrite the file."
        )

    models: list[ImportedModel] = []
    skipped: list[int] = []
    for index, raw in enumerate(payload.get("models") or []):
        row = _row(raw)
        if row is None:
            logger.warning("Skipping malformed imported model at index %d in %s", index, path)
            skipped.append(index)
            continue
        models.append(row)
    if skipped:
        last_load_error = f"{len(skipped)} imported model(s) in {path} could not be read"
    return _migrate(models)


def _migrate(models: list[ImportedModel]) -> list[ImportedModel]:
    """Give every row a public name, without touching anything else.

    Read-path only: nothing is written here. A deterministic derivation means the
    alias is the same on every load whether or not the file has been rewritten
    since, so a client that used it yesterday can use it today — and the file
    picks up the new schema the next time something saves it, rather than having a
    read silently rewrite a file it was only asked to read.

    Rows that already carry a name keep it, and are claimed first so a migrated
    neighbour cannot take a name that is already in use.
    """
    if all(model.api_name for model in models):
        return models
    taken = importing.taken_api_names(models)
    migrated: list[ImportedModel] = []
    for model in models:
        if model.api_name:
            migrated.append(model)
            continue
        name = importing.unique_api_name(model.display_name, taken, fallback=model.id)
        taken.add(name)
        migrated.append(replace(model, api_name=name))
    return migrated


def save(models: list[ImportedModel], base: str | None = None) -> None:
    """Write the library atomically and durably, the way the config is written."""
    path = library_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": SCHEMA_VERSION, "models": [model.as_dict() for model in models]}

    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    # And the directory, so the replace itself survives a power cut.
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def find_by_path(models: list[ImportedModel], path: str) -> ImportedModel | None:
    """An existing registration for this directory, if there is one.

    Duplicate detection lives here rather than in the identity: importing the same
    directory twice selects what is already registered instead of minting a second
    row for the same model.
    """
    wanted = normalise_path(path)
    return next((model for model in models if normalise_path(model.path) == wanted), None)


def touch_last_seen(
    models: list[ImportedModel], model_id: str, when: str | None = None
) -> list[ImportedModel]:
    """Record that a model was observed. Returns a new list; does not write."""
    stamp = when or time.strftime("%Y-%m-%dT%H:%M:%S")
    return [replace(model, last_seen=stamp) if model.id == model_id else model for model in models]
