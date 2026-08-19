"""Which `Host` headers this server answers to.

The guard exists to close DNS rebinding, and rebinding is the one attack an API
key cannot stop: a page on `evil.example` whose name resolves to `127.0.0.1` is
same-origin *to the browser*, so `Origin` and `Host` both read `evil.example`
and the same-origin check returns true. Only the `Host` header — the name the
browser actually dialled — tells the two apart.

**It used to step aside for a wildcard bind**, which is exactly what "listen on
the local network" produces: the protection switched itself off at the moment it
started to matter. It no longer does. Instead the allowlist grows to include the
addresses and names this machine actually answers to.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import subprocess

from qds.logs import SERVER_LOGGER

logger = logging.getLogger(f"{SERVER_LOGGER}.hosts")

#: Always allowed, whatever else is configured, so a hand-written
#: `allowed_hosts` cannot lock its author out of the machine it runs on.
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def local_host_names() -> set[str]:
    """Names and addresses this machine plausibly answers to.

    Best effort, and it says so. Three sources, because none of them is enough:

    * `gethostname`/`getfqdn` — the Unix hostname;
    * `scutil --get LocalHostName` + `.local` — the **Bonjour** name, which on
      macOS is an unrelated string. Measured on one machine: `gethostname()`
      returns `macstudodecorin.home` while the Bonjour name is
      `MacStudio-de-Corin`, so deriving from the first alone would still refuse
      `MacStudio-de-Corin.local` — the single most likely way anyone reaches a
      Mac over a LAN.
    * the addresses the hostname resolves to.

    Whatever this misses, `server.allowed_hosts` is the way to add it, and the
    refusal message says so.
    """
    names: set[str] = set()

    for lookup in (socket.gethostname, socket.getfqdn):
        try:
            value = lookup()
        except OSError:  # pragma: no cover - defensive
            continue
        if value:
            names.add(value.lower())

    if bonjour := _bonjour_name():
        names.add(bonjour.lower())
        names.add(f"{bonjour.lower()}.local")

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            names.add(str(info[4][0]).lower())
    except OSError:
        # A hostname that does not resolve is normal on some networks; the IP
        # literal rule below covers the addresses anyway.
        logger.debug("could not resolve this machine's own hostname", exc_info=True)

    return names


def _bonjour_name() -> str | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/scutil", "--get", "LocalHostName"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        # Not available in every launch context. Degrading is the point: the
        # allowlist is best-effort and `allowed_hosts` is the guarantee.
        return None
    name = result.stdout.strip()
    return name if result.returncode == 0 and name else None


def is_ip_literal(host: str) -> bool:
    """Whether the authority is a bare address rather than a name.

    Any IP literal is allowed, and that is not a loosening. A rebinding attack
    cannot produce one: the `Host` header comes from the URL's authority, and
    the attack's URL is `http://evil.example/`. A browser pointed straight at
    `http://192.168.1.19:8765` *is* the dashboard.

    It also removes a failure mode no amount of re-derivation would: an address
    list computed at startup goes stale the moment DHCP hands out a different
    one, and the server would then refuse the very address it is reachable at.
    """
    candidate = host
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return True


def split_port(host: str) -> str:
    """The authority without its port. `[::1]:8765` keeps its brackets."""
    if host.startswith("["):
        closing = host.find("]")
        return host[: closing + 1] if closing != -1 else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def build_allowlist(host: str, port: int, configured: list[str] | None) -> set[str]:
    """Every spelling this server answers to, bare and with `:port`."""
    names = set(LOOPBACK)
    names.add(host.lower())
    names |= {name.lower() for name in (configured or [])}
    if not configured:
        # Only derived when nothing was written down: an explicit list is an
        # allowlist, and silently adding to it would make it a suggestion.
        names |= local_host_names()
    names.discard("")
    names.discard("0.0.0.0")
    names.discard("::")
    return names | {f"{name}:{port}" for name in names}


def allows(host: str | None, allowlist: set[str], port: int) -> bool:
    if host is None:
        # HTTP/1.0 or a raw socket, not a browser. uvicorn accepts it and so do
        # we; a request with no Host cannot be a rebinding attack either.
        return True
    lowered = host.lower()
    if lowered in allowlist:
        return True
    return is_ip_literal(split_port(lowered))
