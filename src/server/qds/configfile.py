"""Reading and writing `server-config.json` as a document.

Distinct from `settings.py`, which turns that document into validated settings.
This module is about the *file*: how it is replaced without ever being observed
half-written, and which documents are refused outright.

**One writer.** The server process is the only thing that writes this file. It
used to be the desktop app, with the server merely reading it — and a conversion
finishing meant a Rust process and a Python subprocess both had a reason to
touch it. The menubar app now reads it and never writes it (except to seed one
when none exists, before any server is running), so there is no second writer to
race with.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qds.settings import config_path


class ConfigWriteError(RuntimeError):
    """The document could not be persisted. Nothing was changed."""


def read(path: Path | None = None) -> dict[str, Any]:
    """The document as written, or `{}` when there is no file yet.

    Deliberately raw: no defaults are filled in and nothing is validated. A
    caller editing the configuration has to see what is actually in the file,
    because whatever it sends back is what will replace it. Merging the server's
    defaults in here would silently promote every default into an explicit key
    the first time anyone pressed Save.
    """
    path = path or config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigWriteError(f"could not read {path}: {exc}") from exc

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigWriteError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigWriteError(f"{path} must contain a JSON object at the root.")
    return document


def read_text(path: Path | None = None) -> str:
    """The file's raw bytes as text, for showing a document that will not parse.

    The repair screen has to display what is actually there — a JSON error names
    a line and a column, and neither means anything without the line.
    """
    path = path or config_path()
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def write(document: dict[str, Any], path: Path | None = None) -> None:
    """Replace the configuration atomically.

    Write to a temporary file, flush it to disk, then rename over the target.
    The rename is what makes a reader see all-or-nothing; the `fsync` before it
    is what makes the *contents* durable first, which rename alone does not
    promise. Without both, an interruption at the wrong moment leaves a
    truncated document, and the next start fails on a configuration nobody
    edited.

    The file can hold an API key, so it is created 0600 before it is put in
    place rather than after — a window in which the key is world-readable is
    still a window.
    """
    path = path or config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigWriteError(f"could not create {path.parent}: {exc}") from exc

    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    # Not `with_suffix`, which would replace `.json` rather than extend it.
    temporary = path.parent / (path.name + ".tmp")
    try:
        # 0600 from the moment it exists, not applied afterwards.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        # A failed write must not leave a stray `.tmp` behind for the next one
        # to trip over.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort
            pass
        raise ConfigWriteError(f"could not write {path}: {exc}") from exc

    # And fsync the directory, so the rename itself survives a power cut.
    try:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:  # pragma: no cover - not every filesystem allows it
        pass


class DisabledDefaultModel(ValueError):
    """The document switches off the model it also nominates as default."""


def refuse_disabled_default(document: dict[str, Any]) -> None:
    """Refuse a document that would switch off the model it nominates by default.

    The invariant is the generation server's — it will not start with a default
    model that is disabled — and the Enabled switch can create exactly that
    state with one click, from the one screen that could then no longer be
    trusted to render. Checked at the write, because that is where every path
    meets: the switch, the Configuration form, and anything added later.

    A pure property of the document, deliberately: no catalogue is consulted and
    no replacement default is chosen. Silently promoting another model to
    default would be this process deciding something the user did not ask for,
    so the refusal names the fix instead.
    """
    default_model = document.get("default_model")
    if not isinstance(default_model, str):
        # No default named: this document cannot break the invariant. A missing
        # key means the server's own default applies, and that model is not the
        # one being disabled here.
        return

    models = document.get("models")
    entry = models.get(default_model) if isinstance(models, dict) else None
    enabled = entry.get("enabled") if isinstance(entry, dict) else None
    # Absence is not disabled: `enabled` defaults to true in `ModelOverride`.
    if enabled is False:
        raise DisabledDefaultModel(
            f'"{default_model}" is currently the default model. Choose another default '
            f"model in Configuration before disabling it."
        )


def select_variant(model: str, bits: int, path: Path | None = None) -> None:
    """Write `models.<key>.prequantized_variant`, leaving every other key alone.

    Read-modify-write of the whole document, because that is what `write` takes.
    Only this one key changes: other bit depths' artifacts are untouched on disk
    and unmentioned here, and nothing else in the model's entry is rewritten —
    activating a conversion must never rewrite what the model *is*.
    """
    path = path or config_path()
    document = read(path)

    models = document.setdefault("models", {})
    if not isinstance(models, dict):
        raise ConfigWriteError("the configuration's 'models' is not an object.")
    entry = models.setdefault(model, {})
    if not isinstance(entry, dict):
        raise ConfigWriteError(f"the configuration's 'models.{model}' is not an object.")

    entry["prequantized_variant"] = bits
    write(document, path)
