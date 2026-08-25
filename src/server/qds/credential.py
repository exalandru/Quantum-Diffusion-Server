"""The typed-by-a-human passwords: hashed, and kept out of the configuration document.

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


class Credential:
    """One plane's password, in its own file beside the configuration.

    A class rather than a second module of copied functions, and rather than a
    `filename=` keyword on the module-level helpers: a forgotten keyword would
    silently check the *admin* password where the playground's was meant, which
    is a security bug that reads as correct code. Binding the filename to an
    object makes the plane part of the call — `PLAYGROUND.verify(...)` cannot be
    mistaken for the admin one.

    The algorithm, the work factors and the length floor are the module's, not
    the instance's: two planes with two scrypt configurations is exactly the
    drift this file exists to prevent.
    """

    def __init__(self, filename: str, plane: str) -> None:
        self.filename = filename
        #: What the logs call it, so "password changed" says which one.
        self.plane = plane

    def path(self, path: Path | None = None) -> Path:
        """Beside the configuration, because it belongs to the same installation."""
        return (path or config_path()).parent / self.filename

    def is_set(self, path: Path | None = None) -> bool:
        return self._read(path) is not None

    def _read(self, path: Path | None = None) -> dict[str, Any] | None:
        try:
            raw = self.path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(record, dict) or record.get("algorithm") != "scrypt":
            return None
        return record

    def set_password(self, password: str, path: Path | None = None) -> None:
        """Replace the stored credential.

        Written through `configfile.write`, so it inherits the atomic replace and
        the 0600-from-creation this file needs at least as much as the
        configuration does.
        """
        configfile.write(hash_password(password), self.path(path))

    def clear(self, path: Path | None = None) -> None:
        self.path(path).unlink(missing_ok=True)

    def verify(self, password: str, path: Path | None = None) -> bool:
        """Whether this is the stored password. See `verify_record` for the
        failure posture."""
        return verify_record(password, self._read(path))

    def restrict_if_needed(self, path: Path | None = None) -> None:
        """Make sure an existing record is not readable by anyone else.

        `configfile.write` creates it 0600, so this only matters for a file that
        arrived some other way — restored from a backup, or copied by hand.
        """
        target = self.path(path)
        try:
            if target.is_file() and target.stat().st_mode & 0o077:
                os.chmod(target, 0o600)
        except OSError:  # pragma: no cover - best effort
            pass


#: The control plane's password: `/admin`, and therefore the configuration, the
#: logs and the restart button.
ADMIN = Credential("admin-credential.json", "admin")

#: The playground's own, deliberately not the admin's. An admin credential opens
#: the configuration writer; a housemate who may generate images should not get
#: it, and the person who owns the machine should not have to hand it over to
#: let them.
PLAYGROUND = Credential("playground-credential.json", "playground")


# ── The admin credential, as this module has always exposed it ──────────────
#
# Delegations rather than aliases: every existing caller and test names these,
# and `credential.is_set()` reading as "is the admin password set" is right —
# `/admin` is what this module was written for. The playground says which plane
# it means by asking `PLAYGROUND` for it.


def credential_path(path: Path | None = None) -> Path:
    return ADMIN.path(path)


def is_set(path: Path | None = None) -> bool:
    return ADMIN.is_set(path)


def set_password(password: str, path: Path | None = None) -> None:
    ADMIN.set_password(password, path)


def clear(path: Path | None = None) -> None:
    ADMIN.clear(path)


def verify(password: str, path: Path | None = None) -> bool:
    return ADMIN.verify(password, path)


def restrict_if_needed(path: Path | None = None) -> None:
    ADMIN.restrict_if_needed(path)


# ── The algorithm, shared by every password this server stores ───────────────
#
# The two credential files above and the playground's per-session passwords all
# go through these: one algorithm, one set of work factors, one length floor.


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


def hash_password(password: str) -> dict[str, Any]:
    """A fresh scrypt record for this password, ready to persist anywhere.

    Shared by the two credential files and the playground's per-session
    passwords, so there is one algorithm, one set of work factors and one
    length floor — not three that drift apart.
    """
    if len(password) < MIN_LENGTH:
        raise WeakPassword(f"The password must be at least {MIN_LENGTH} characters.")

    salt = secrets.token_bytes(SALT_BYTES)
    derived = _derive(password, salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return {
        "version": 1,
        "algorithm": "scrypt",
        # The parameters travel *with* the hash, so raising the defaults
        # later does not invalidate a password somebody already set.
        "n": SCRYPT_N,
        "r": SCRYPT_R,
        "p": SCRYPT_P,
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(derived).decode("ascii"),
    }


def verify_record(password: str, record: dict[str, Any] | None) -> bool:
    """Whether `password` matches a record made by `hash_password`.

    Fails closed on anything it cannot read: a truncated, hand-edited or
    unparseable record verifies nothing rather than raising, because a
    credential that has been damaged should lock the door, not crash the
    endpoint that guards it.
    """
    if not isinstance(record, dict) or record.get("algorithm") != "scrypt":
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
