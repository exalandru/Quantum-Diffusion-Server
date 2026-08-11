"""`mflux-server-import`: inspect, register, list and forget local models.

A separate console script for the same reason `mflux-server-fetch` is one: model
management has to work with the generation server stopped, and nothing here
imports mflux or torch.

Registration revalidates rather than trusting what the caller was told earlier —
an inspection result can be minutes old, and the directory may have moved or the
volume gone in between.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from mflux_server import importing, library, settings
from mflux_server.registry import BASE_SPECS_BY_KEY


def _emit(payload: dict) -> int:
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0 if payload.get("ok", True) else 1


def cmd_inspect(path: str) -> int:
    verdict = importing.inspect(path)
    payload = verdict.as_dict()
    if verdict.ok:
        models = library.load()
        existing = library.find_by_path(models, verdict.path)
        # Importing the same directory twice selects what is already there.
        payload["already_imported"] = existing.as_dict() if existing else None
        # A starting point for the public identifier, free at the time of asking.
        # The user may edit it, and `register` checks again — this is a
        # suggestion, not a reservation.
        payload["suggested_api_name"] = importing.unique_api_name(
            verdict.suggested_name or verdict.path,
            importing.taken_api_names(models),
        )
    return _emit(payload)


def cmd_register(path: str, base_profile: str, name: str | None, api_name: str | None) -> int:
    verdict = importing.inspect(path)
    if not verdict.ok:
        return _emit(verdict.as_dict())

    models = library.load()
    existing = library.find_by_path(models, verdict.path)
    if existing is not None:
        return _emit({"ok": True, "already_imported": True, "model": existing.as_dict()})

    profile = BASE_SPECS_BY_KEY.get(base_profile)
    if profile is None or profile.family != verdict.family:
        return _emit(
            {
                "ok": False,
                "reason": f"{base_profile!r} is not a compatible base profile for a "
                f"{verdict.family!r} model. Choose one of: {list(verdict.profiles)}",
            }
        )

    display_name = name or verdict.suggested_name or verdict.path
    taken = importing.taken_api_names(models)
    # Chosen or derived, but always checked here: `register` is the boundary that
    # persists it, so it is the boundary that owns uniqueness. A name suggested
    # minutes ago may have been taken by another import in between.
    public = api_name.strip() if api_name else importing.unique_api_name(display_name, taken)
    problem = importing.api_name_problem(public)
    if problem:
        return _emit({"ok": False, "code": "invalid_api_name", "reason": problem})
    if public in taken:
        return _emit(
            {
                "ok": False,
                "code": "api_name_taken",
                "reason": (
                    f"The API name {public!r} is already used by another model. Choose a "
                    f"different one — this is the identifier API requests will send."
                ),
            }
        )

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    model = library.ImportedModel(
        id=importing.new_id(),
        display_name=display_name,
        api_name=public,
        path=verdict.path,
        family=verdict.family or "",
        base_profile_key=base_profile,
        imported_at=now,
        last_seen=now,
    )
    library.save(models + [model])
    return _emit({"ok": True, "already_imported": False, "model": model.as_dict()})


def cmd_locate(path: str, model_key: str) -> int:
    """Check a directory against one built-in entry. Binds nothing.

    Writing the override is the app's job, through the same configuration path
    every other per-model setting uses. This only answers whether it may.
    """
    return _emit(importing.locate(path, model_key).as_dict())


def cmd_list() -> int:
    models = library.load()
    rows = []
    for model in models:
        state, detail = importing.availability_of(model.path)
        # A refresh reports what it saw. It never unregisters: an unplugged disk
        # is a fact about storage, not a decision to remove the model.
        rows.append({**model.as_dict(), "availability": state, "detail": detail})
    return _emit({"ok": True, "models": rows, "warning": library.last_load_error})


def cmd_forget(model_id: str) -> int:
    """Remove the registration. Never the user's files.

    Refused outright while the model is the configured `default_model`. Removing
    it would leave that key naming something no registry can resolve, and the
    server refuses to start on an unknown default — a fail-closed outcome, but one
    reached *after* the registration is gone, so the Models tab no longer offers
    the row and the only repair is editing JSON by hand.

    The alternative, rewriting `default_model` to some other model, is rejected on
    purpose: which model generates by default is the user's decision, and silently
    making it for them is how an app changes behaviour nobody asked it to change.
    """
    models = library.load()
    remaining = [model for model in models if model.id != model_id]
    if len(remaining) == len(models):
        return _emit({"ok": False, "reason": f"No imported model with id {model_id!r}."})

    # Before `library.save`, so a refusal is not a partial removal: nothing has
    # been written when this returns.
    try:
        current_default = settings.configured_default_model()
    except ValueError as exc:
        return _emit({"ok": False, "code": "config_unreadable", "reason": str(exc)})
    if current_default == model_id:
        return _emit(
            {
                "ok": False,
                "code": "is_default_model",
                "reason": (
                    "This model is the current default model. Choose another default model "
                    "in the Configuration tab before removing this one from the library."
                ),
            }
        )

    library.save(remaining)
    return _emit(
        {
            "ok": True,
            "forgotten": model_id,
            "note": "The registration was removed. The model files were not touched.",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mflux-server-import",
        description="Register a local model directory with QDS, without copying it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="identify a directory, without registering it")
    inspect_parser.add_argument("path")

    register_parser = sub.add_parser("register", help="register a directory as a model")
    register_parser.add_argument("path")
    register_parser.add_argument("--base-profile", required=True)
    register_parser.add_argument("--name", default=None)
    register_parser.add_argument(
        "--api-name",
        default=None,
        help="public identifier API requests send; derived from --name when omitted",
    )

    locate_parser = sub.add_parser(
        "locate", help="check a directory against a built-in model, without binding it"
    )
    locate_parser.add_argument("path")
    locate_parser.add_argument("--model", required=True)

    sub.add_parser("list", help="list registered models with their availability")

    forget_parser = sub.add_parser("forget", help="remove a registration; files are untouched")
    forget_parser.add_argument("id")

    args = parser.parse_args()
    try:
        if args.command == "inspect":
            return cmd_inspect(args.path)
        if args.command == "register":
            return cmd_register(args.path, args.base_profile, args.name, args.api_name)
        if args.command == "locate":
            return cmd_locate(args.path, args.model)
        if args.command == "list":
            return cmd_list()
        if args.command == "forget":
            return cmd_forget(args.id)
    except library.LibraryTooNew as exc:
        return _emit({"ok": False, "reason": str(exc)})
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
