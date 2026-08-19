"""The admin password: hashed, and kept out of the configuration document.

**Its own file, deliberately.** `GET /admin/config` returns the configuration
document verbatim, and the dashboard's Configuration form reads it and writes
`server` back wholesale. A credential living in there would be handed to the
browser on every read and round-tripped on every save — so a password changed
while that form sat open would be reverted by pressing Save. Putting it beside
the configuration rather than inside it removes the whole class of problem
instead of adding a rule to remember.

**scrypt, from the standard library.** `passlib`, `bcrypt`, `argon2` and
`cryptography` are all absent from this project's dependency set, and adding one
for a single-user local server is a supply chain nobody asked for. scrypt is
memory-hard, which pbkdf2 is not, and both are free.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any

from qds import configfile
from qds.settings import config_path

#: Work factors, **measured on the target machine** rather than copied from a
#: blog: n=16384 verifies in 27 ms and n=65536 in 94 ms. A tenth of a second is
#: imperceptible when you log in and makes guessing an order of magnitude more
#: expensive.
SCRYPT_N = 65536
SCRYPT_R = 8
SCRYPT_P = 1
DK_LEN = 32
SALT_BYTES = 16

#: `hashlib.scrypt` refuses anything over OpenSSL's 32 MiB default, and
#: `128 * r * n` is exactly 64 MiB at these parameters — so omitting this does
#: not make it slower, it makes it *raise*. A test that used a smaller `n` would
#: never see it, which is why the regression guard uses the constants above.
MAXMEM = 128 * SCRYPT_R * SCRYPT_N + 1024 * 1024

#: Minimum length. A floor, not a policy: it stops a one-character password
#: without pretending that a rule about punctuation would make anyone safer.
MIN_LENGTH = 8


class WeakPassword(ValueError):
    """The password is too short to be worth hashing."""


def credential_path(path: Path | None = None) -> Path:
    """Beside the configuration, because it belongs to the same installation."""
    return (path or config_path()).parent / "admin-credential.json"


def _derive(password: str, salt: bytes, *, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=DK_LEN,
        maxmem=128 * r * n + 1024 * 1024,
    )


def is_set(path: Path | None = None) -> bool:
    return _read(path) is not None


def _read(path: Path | None = None) -> dict[str, Any] | None:
    try:
        raw = credential_path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("algorithm") != "scrypt":
        return None
    return record


def set_password(password: str, path: Path | None = None) -> None:
    """Replace the stored credential.

    Written through `configfile.write`, so it inherits the atomic replace and
    the 0600-from-creation this file needs at least as much as the
    configuration does.
    """
    if len(password) < MIN_LENGTH:
        raise WeakPassword(f"The password must be at least {MIN_LENGTH} characters.")

    salt = secrets.token_bytes(SALT_BYTES)
    derived = _derive(password, salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    configfile.write(
        {
            "version": 1,
            "algorithm": "scrypt",
            # The parameters travel *with* the hash, so raising the defaults
            # later does not invalidate a password somebody already set.
            "n": SCRYPT_N,
            "r": SCRYPT_R,
            "p": SCRYPT_P,
            "salt": base64.b64encode(salt).decode("ascii"),
            "hash": base64.b64encode(derived).decode("ascii"),
        },
        credential_path(path),
    )


def clear(path: Path | None = None) -> None:
    credential_path(path).unlink(missing_ok=True)


def verify(password: str, path: Path | None = None) -> bool:
    """Whether this is the stored password.

    Fails closed on anything it cannot read: a truncated, hand-edited or
    unparseable record verifies nothing rather than raising, because a
    credential file that has been damaged should lock the door, not crash the
    endpoint that guards it.
    """
    record = _read(path)
    if record is None:
        return False
    try:
        salt = base64.b64decode(record["salt"], validate=True)
        expected = base64.b64decode(record["hash"], validate=True)
        n, r, p = int(record["n"]), int(record["r"]), int(record["p"])
    except (KeyError, TypeError, ValueError, base64.binascii.Error):
        return False
    if not salt or not expected:
        return False

    try:
        derived = _derive(password, salt, n=n, r=r, p=p)
    except (ValueError, OverflowError):
        # A record naming parameters this machine will not run — a hand-written
        # n, or one raised past what memory allows.
        return False
    return hmac.compare_digest(derived, expected)


def restrict_if_needed(path: Path | None = None) -> None:
    """Make sure an existing record is not readable by anyone else.

    `configfile.write` creates it 0600, so this only matters for a file that
    arrived some other way — restored from a backup, or copied by hand.
    """
    target = credential_path(path)
    try:
        if target.is_file() and target.stat().st_mode & 0o077:
            os.chmod(target, 0o600)
    except OSError:  # pragma: no cover - best effort
        pass
