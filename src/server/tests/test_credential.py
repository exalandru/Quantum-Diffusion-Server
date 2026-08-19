"""The admin password: what is stored, and what refuses.

The properties worth pinning here are the ones whose failure is silent. A hash
that verifies its own password proves almost nothing on its own — a function
returning `True` unconditionally passes it — so every positive below is paired
with the negative that would catch that.
"""

from __future__ import annotations

import json

import pytest

from qds import credential
from qds.credential import WeakPassword


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    return tmp_path / "admin-credential.json"


# ── Round trip ─────────────────────────────────────────────────────────────


def test_the_stored_credential_verifies_its_own_password(configured):
    credential.set_password("correct horse battery")
    assert credential.verify("correct horse battery") is True


def test_a_wrong_password_does_not_verify(configured):
    """The negative without which the test above proves nothing."""
    credential.set_password("correct horse battery")
    assert credential.verify("correct horse batteries") is False
    assert credential.verify("") is False
    assert credential.verify("CORRECT HORSE BATTERY") is False


def test_no_stored_credential_means_nothing_verifies(configured):
    """Not "any password works" — the distinction the whole scheme rests on."""
    assert credential.is_set() is False
    assert credential.verify("anything") is False
    assert credential.verify("") is False


def test_clearing_removes_the_credential(configured):
    credential.set_password("correct horse battery")
    assert credential.is_set() is True
    credential.clear()
    assert credential.is_set() is False
    assert credential.verify("correct horse battery") is False


# ── What is on disk ────────────────────────────────────────────────────────


def test_the_password_is_not_in_the_file(configured):
    """Hashed means the plaintext is absent, not merely encoded."""
    password = "correct horse battery"
    credential.set_password(password)

    raw = configured.read_bytes()
    assert password.encode() not in raw
    # Nor a bare digest of it: a salt-less hash is a rainbow-table lookup.
    import hashlib

    assert hashlib.sha256(password.encode()).hexdigest().encode() not in raw


def test_two_records_for_one_password_differ(configured):
    """The salt is real. Without it, equal passwords would store equal bytes."""
    credential.set_password("correct horse battery")
    first = configured.read_bytes()
    credential.set_password("correct horse battery")
    second = configured.read_bytes()

    assert first != second
    # And both still verify: a salt that broke verification would also fail here.
    assert credential.verify("correct horse battery") is True


def test_the_file_is_not_world_readable(configured):
    credential.set_password("correct horse battery")
    assert configured.stat().st_mode & 0o077 == 0


def test_the_credential_lives_beside_the_configuration_not_inside_it(
    configured, monkeypatch, tmp_path
):
    """The reason it is a separate file at all.

    `GET /admin/config` returns the configuration document verbatim, so a hash
    kept in there would be handed to the browser on every read, and the
    Configuration form — which writes `server` back wholesale — would revert a
    password changed while it sat open.
    """
    from qds import configfile

    config = tmp_path / "server-config.json"
    configfile.write({"server": {"port": 8765}}, config)
    credential.set_password("correct horse battery")

    assert configured.exists()
    assert "admin" not in json.dumps(configfile.read(config))
    assert "hash" not in json.dumps(configfile.read(config))


# ── Parameters ─────────────────────────────────────────────────────────────


def test_hashing_at_the_shipped_parameters_does_not_raise(configured):
    """The regression guard for `maxmem`.

    `hashlib.scrypt` refuses anything over OpenSSL's 32 MiB default, and these
    parameters need 64 MiB — so omitting `maxmem` does not make this slower, it
    makes every login return 500. A test using a smaller, test-local `n` would
    never see it, which is why this one uses the production constants.
    """
    assert credential.SCRYPT_N >= 32768, "below this the bug cannot reproduce"
    credential.set_password("correct horse battery")
    assert credential.verify("correct horse battery") is True


def test_a_record_written_at_weaker_parameters_still_verifies(configured, monkeypatch):
    """Raising the default must not lock anyone out of their own password."""
    monkeypatch.setattr(credential, "SCRYPT_N", 16384)
    credential.set_password("correct horse battery")

    monkeypatch.setattr(credential, "SCRYPT_N", 65536)
    # Verified with the parameters in the record, not the current defaults.
    assert credential.verify("correct horse battery") is True
    assert json.loads(configured.read_text())["n"] == 16384


# ── Failing closed ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json at all",
        "[]",
        '{"algorithm": "scrypt"}',
        '{"algorithm": "scrypt", "salt": "!!!", "hash": "!!!", "n": 1, "r": 1, "p": 1}',
        '{"algorithm": "md5", "salt": "AAAA", "hash": "AAAA", "n": 1, "r": 1, "p": 1}',
        '{"algorithm": "scrypt", "salt": "", "hash": "", "n": 16384, "r": 8, "p": 1}',
    ],
)
def test_a_damaged_record_verifies_nothing_and_does_not_raise(configured, content):
    """A broken credential locks the door; it does not crash the guard."""
    configured.write_text(content, encoding="utf-8")
    assert credential.verify("anything") is False
    assert credential.verify("") is False


def test_a_record_naming_impossible_parameters_verifies_nothing(configured):
    """A hand-edited `n` past what memory allows must not become an exception."""
    configured.write_text(
        json.dumps(
            {
                "algorithm": "scrypt",
                "salt": "AAAAAAAAAAAAAAAAAAAAAA==",
                "hash": "AAAAAAAAAAAAAAAAAAAAAA==",
                "n": 2**30,
                "r": 8,
                "p": 1,
            }
        ),
        encoding="utf-8",
    )
    assert credential.verify("anything") is False


def test_a_password_shorter_than_the_floor_is_refused(configured):
    with pytest.raises(WeakPassword):
        credential.set_password("short")
    # And refusing wrote nothing.
    assert credential.is_set() is False
