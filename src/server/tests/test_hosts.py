"""The `Host` allowlist, and the hole a network bind used to open in it.

DNS rebinding is the one attack authentication cannot stop, because the page
doing it is same-origin as far as the browser is concerned: `Origin` and `Host`
both read `evil.example`, so the same-origin check agrees with it. Only the
`Host` header distinguishes the name that was dialled from the address it
resolved to.

The guard used to return early for `0.0.0.0` — so enabling "listen on the local
network" switched off the protection at the exact moment it began to matter.
The test that would have caught that is the first one below, and it did not
exist.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qds import hosts
from qds.app import create_app
from qds.settings import Settings

from .conftest import FakeEngine


@pytest.fixture(autouse=True)
def _password(monkeypatch, tmp_path):
    """A network bind now needs an admin password, so give every case one."""
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    from qds import credential

    credential.set_password("correct horse battery")


def app_bound_to(tmp_path, host: str, allowed: list[str] | None = None):
    settings = Settings.model_validate(
        {
            "server": {
                "host": host,
                "api_key": "k",
                "allowed_hosts": allowed or [],
                "image_store": str(tmp_path / "images"),
                "log_file": None,
            }
        }
    )
    return create_app(settings, FakeEngine())


def dial(app, host: str):
    with TestClient(app, base_url=f"http://{host}") as client:
        return client.get("/health")


# ── The regression this exists for ─────────────────────────────────────────


def test_a_network_bound_server_still_refuses_an_unknown_host(tmp_path):
    """The property that did not exist.

    Bound to `0.0.0.0` the guard used to disable itself entirely, so a rebound
    `evil.example` reached every endpoint.
    """
    response = dial(app_bound_to(tmp_path, "0.0.0.0"), "evil.example")
    assert response.status_code == 421
    assert response.json()["error"]["code"] == "host_not_allowed"


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "127.0.0.1.evil.example"])
def test_a_loopback_bound_server_refuses_unknown_hosts_too(tmp_path, host):
    assert dial(app_bound_to(tmp_path, "127.0.0.1"), host).status_code == 421


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.1:8765"])
def test_loopback_is_always_answered(tmp_path, host):
    assert dial(app_bound_to(tmp_path, "0.0.0.0"), host).status_code == 200


def test_the_refusal_names_the_setting_that_would_permit_it(tmp_path):
    """The derivation is best-effort; this message is what closes the gap."""
    body = dial(app_bound_to(tmp_path, "0.0.0.0"), "nas.example").json()
    assert "allowed_hosts" in body["error"]["message"]


# ── Reaching it by address ─────────────────────────────────────────────────


@pytest.mark.parametrize("host", ["192.168.1.19", "192.168.1.19:8765", "10.0.0.4"])
def test_an_address_is_answered(tmp_path, host):
    """Any IP literal, and that is not a loosening.

    A rebinding attack cannot produce one — the `Host` comes from the URL
    authority and the attack's URL is a *name*. Meanwhile a browser pointed
    straight at `http://192.168.1.19:8765` is the dashboard. It also removes the
    DHCP failure: an address list derived at startup goes stale the moment the
    router hands out a different lease, and the server would then refuse the very
    address it is reachable at.
    """
    assert dial(app_bound_to(tmp_path, "0.0.0.0"), host).status_code == 200


def test_a_bracketed_ipv6_address_is_allowed():
    """Checked at the helper, not over HTTP: `httpx` cannot take
    `http://[::1]:8765` as a base URL, which is a limitation of the test client
    rather than of the guard."""
    assert hosts.allows("[::1]:8765", {"127.0.0.1"}, 8765) is True
    assert hosts.allows("[2a01:cb1d::1]", {"127.0.0.1"}, 8765) is True


def test_a_name_that_merely_contains_an_address_is_refused(tmp_path):
    """The negative for the rule above."""
    assert dial(app_bound_to(tmp_path, "0.0.0.0"), "192.168.1.19.evil.example").status_code == 421


# ── The explicit list ──────────────────────────────────────────────────────


def test_a_listed_name_is_answered(tmp_path):
    app = app_bound_to(tmp_path, "0.0.0.0", allowed=["nas.example"])
    assert dial(app, "nas.example").status_code == 200
    assert dial(app, "nas.example:8765").status_code == 200


def test_an_unlisted_name_is_still_refused(tmp_path):
    """A list is an allowlist, not a hint: adding one name permits one name."""
    app = app_bound_to(tmp_path, "0.0.0.0", allowed=["nas.example"])
    assert dial(app, "other.example").status_code == 421


def test_a_list_cannot_lock_its_author_out_of_loopback(tmp_path):
    app = app_bound_to(tmp_path, "0.0.0.0", allowed=["nas.example"])
    assert dial(app, "127.0.0.1").status_code == 200
    assert dial(app, "localhost").status_code == 200


# ── The derivation ─────────────────────────────────────────────────────────


def test_this_machine_names_itself_somehow():
    names = hosts.local_host_names()
    assert names, "no local name could be derived at all"
    assert all(name == name.lower() for name in names)


def test_the_bonjour_name_is_looked_up_separately_from_the_hostname():
    """They are unrelated strings on macOS, which is the whole reason.

    Measured: `gethostname()` gives `macstudodecorin.home` while the Bonjour
    name is `MacStudio-de-Corin`. Deriving from the hostname alone would refuse
    `MacStudio-de-Corin.local`, the most likely way anyone reaches a Mac on a
    LAN — so this asserts the second lookup happens at all.
    """
    import socket

    bonjour = hosts._bonjour_name()
    if bonjour is None:
        pytest.skip("scutil is unavailable in this environment")
    names = hosts.local_host_names()
    assert f"{bonjour.lower()}.local" in names
    # And it really is a different source from the hostname.
    assert socket.gethostname().lower() in names


def test_a_missing_scutil_degrades_rather_than_raises(monkeypatch):
    monkeypatch.setattr(hosts, "_bonjour_name", lambda: None)
    assert hosts.local_host_names()


# ── The pure helpers ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("192.168.1.19", True),
        ("127.0.0.1", True),
        ("[::1]", True),
        ("::1", True),
        ("nas.example", False),
        ("192.168.1.19.evil.example", False),
        ("", False),
    ],
)
def test_is_ip_literal(host, expected):
    assert hosts.is_ip_literal(host) is expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("192.168.1.19:8765", "192.168.1.19"),
        ("192.168.1.19", "192.168.1.19"),
        ("[::1]:8765", "[::1]"),
        ("nas.example:80", "nas.example"),
    ],
)
def test_split_port(host, expected):
    assert hosts.split_port(host) == expected


def test_a_request_with_no_host_is_allowed():
    """HTTP/1.0 or a raw socket — not a browser, so not a rebinding attack."""
    assert hosts.allows(None, {"127.0.0.1"}, 8765) is True


# ── What a broken configuration may listen on ──────────────────────────────


def test_a_healthy_server_binds_what_it_was_configured_to(tmp_path):
    from qds.app import effective_bind_host

    settings = Settings.model_validate({"server": {"host": "0.0.0.0"}})
    assert effective_bind_host(settings, None) == "0.0.0.0"


def test_a_broken_configuration_without_a_password_falls_back_to_loopback(
    tmp_path, monkeypatch
):
    """The hole this closes.

    `recovery_settings()` reads the host from the environment, and recovery mode
    leaves `/admin` open when no password is set — so `QDS_SERVER_HOST=0.0.0.0`
    plus an unparseable config file produced a wildcard-bound, unauthenticated
    configuration writer.
    """
    from qds import credential
    from qds.app import effective_bind_host

    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "gone.json"))
    credential.clear()

    settings = Settings.model_validate({"server": {"host": "0.0.0.0"}})
    assert effective_bind_host(settings, "the config will not parse") == "127.0.0.1"


def test_a_broken_configuration_with_a_password_keeps_its_address(tmp_path, monkeypatch):
    """Headless repair survives: a bad config and an intact password is fine."""
    from qds.app import effective_bind_host

    settings = Settings.model_validate({"server": {"host": "0.0.0.0"}})
    # The autouse fixture already set a password for this test's config path.
    assert effective_bind_host(settings, "the config will not parse") == "0.0.0.0"
