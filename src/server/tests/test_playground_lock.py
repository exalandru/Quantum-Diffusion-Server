"""Renaming a playground session, and locking one behind a password.

The property under test is **content authority**: once a session has a password,
nothing of it — detail, images, a new generation, its deletion — is served
without a live unlock token for that very session, and the token dies with a
lock, a password change, a deletion, an idle timeout or a restart. The list
endpoint keeps showing the row, and only the row.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from qds import credential, playground_lock
from qds.app import create_app
from qds.playground import PlaygroundStore
from qds.settings import Settings
from tests.conftest import make_client, tiny_png, wait_until
from tests.test_playground import generations, new_session, status_of, submit

PASSWORD = "correct horse battery"
HEADER = playground_lock.UNLOCK_HEADER


def protect(client: TestClient, session_id: str, password: str = PASSWORD) -> str:
    response = client.post(f"/playground/api/sessions/{session_id}/password", json={"password": password})
    assert response.status_code == 200, response.text
    return response.json()["token"]


def unlock(client: TestClient, session_id: str, password: str = PASSWORD):
    return client.post(f"/playground/api/sessions/{session_id}/unlock", json={"password": password})


def listed(client: TestClient, session_id: str) -> dict:
    rows = client.get("/playground/api/sessions").json()["sessions"]
    return next(row for row in rows if row["id"] == session_id)


def with_token(token: str) -> dict[str, str]:
    return {HEADER: token}


def finished_session(client: TestClient) -> tuple[str, str, str]:
    """A session with one completed generation: (session, image url, context url)."""
    session_id = new_session(client)
    submit(client, session_id, files={"image": ("ctx.png", tiny_png(), "image/png")})
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    record = generations(client, session_id)[0]
    return session_id, record["images"][0]["url"], record["contextImage"]


# ── Renaming ───────────────────────────────────────────────────────────────


def test_a_session_can_be_renamed(client):
    session_id = new_session(client)
    renamed = client.patch(f"/playground/api/sessions/{session_id}", json={"title": "  Foxes  "})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Foxes"
    assert renamed.json()["locked"] is False
    assert listed(client, session_id)["title"] == "Foxes"

    # A user-chosen title survives the first prompt.
    submit(client, session_id, prompt="a badger")
    assert listed(client, session_id)["title"] == "Foxes"


def test_clearing_the_title_hands_it_back_to_the_first_prompt(client):
    session_id = new_session(client)
    client.patch(f"/playground/api/sessions/{session_id}", json={"title": "Foxes"})
    cleared = client.patch(f"/playground/api/sessions/{session_id}", json={"title": ""})
    assert cleared.json()["title"] is None
    submit(client, session_id, prompt="a badger")
    assert listed(client, session_id)["title"] == "a badger"


def test_a_title_is_clamped_to_the_limit(client):
    session_id = new_session(client)
    renamed = client.patch(f"/playground/api/sessions/{session_id}", json={"title": "x" * 200})
    assert len(renamed.json()["title"]) == 80


def test_renaming_an_unknown_session_is_a_404(client):
    assert client.patch("/playground/api/sessions/nope", json={"title": "x"}).status_code == 404


# ── Locking ────────────────────────────────────────────────────────────────


def test_a_weak_password_is_refused(client):
    session_id = new_session(client)
    refused = client.post(f"/playground/api/sessions/{session_id}/password", json={"password": "short"})
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "weak_password"
    assert listed(client, session_id)["locked"] is False


def test_a_locked_session_serves_nothing_without_its_token(client):
    session_id, image_url, context_url = finished_session(client)
    other = new_session(client)
    token = protect(client, session_id)

    row = listed(client, session_id)
    assert row["locked"] is True
    assert not any("password" in key.lower() for key in row)

    def is_locked(response) -> bool:
        return response.status_code == 403 and response.json()["error"]["code"] == "session_locked"

    detail = client.get(f"/playground/api/sessions/{session_id}", headers=with_token(token))
    generation_id = detail.json()["generations"][0]["id"]
    image_name = image_url.rsplit("/", 1)[-1]

    assert is_locked(client.get(f"/playground/api/sessions/{session_id}"))
    assert is_locked(client.patch(f"/playground/api/sessions/{session_id}", json={"title": "x"}))
    assert is_locked(client.delete(f"/playground/api/sessions/{session_id}"))
    assert is_locked(submit(client, session_id))
    assert is_locked(client.get(image_url))
    assert is_locked(client.get(context_url))
    assert is_locked(client.post(f"/playground/api/generations/{generation_id}/cancel"))
    assert is_locked(client.delete(f"/playground/api/images/{image_name}"))
    assert is_locked(client.delete(f"/playground/api/sessions/{session_id}/password"))
    assert is_locked(
        client.post(f"/playground/api/sessions/{session_id}/password", json={"password": PASSWORD})
    )

    # Another session's token opens nothing here.
    foreign = protect(client, other)
    assert is_locked(client.get(f"/playground/api/sessions/{session_id}", headers=with_token(foreign)))

    # The session's own token opens everything; the query form only the image.
    assert client.get(f"/playground/api/sessions/{session_id}", headers=with_token(token)).status_code == 200
    assert client.get(image_url, headers=with_token(token)).status_code == 200
    served = client.get(image_url, params={"t": token})
    assert served.status_code == 200
    assert served.headers["cache-control"] == "private, no-store"
    assert client.get(context_url, params={"t": token}).status_code == 200
    assert is_locked(client.get(f"/playground/api/sessions/{session_id}", params={"t": token}))
    assert submit(client, session_id, headers=with_token(token)).status_code == 202
    assert (
        client.patch(
            f"/playground/api/sessions/{session_id}", json={"title": "x"}, headers=with_token(token)
        ).status_code
        == 200
    )


def test_an_open_session_ignores_tokens(client):
    session_id = new_session(client)
    assert client.get(f"/playground/api/sessions/{session_id}", headers=with_token("junk")).status_code == 200
    refused = unlock(client, session_id)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "session_not_protected"
    assert client.delete(f"/playground/api/sessions/{session_id}/password").status_code == 409


def test_unlocking_redeems_the_password_for_a_token(client):
    session_id = new_session(client)
    protect(client, session_id)

    wrong = unlock(client, session_id, "not the password")
    assert wrong.status_code == 403
    assert wrong.json()["error"]["code"] == "invalid_session_password"

    right = unlock(client, session_id)
    assert right.status_code == 200
    token = right.json()["token"]
    assert right.json()["session"]["locked"] is True
    assert client.get(f"/playground/api/sessions/{session_id}", headers=with_token(token)).status_code == 200


def test_guessing_is_throttled_per_session(client):
    session_id = new_session(client)
    other = new_session(client)
    protect(client, session_id)
    protect(client, other)
    for _ in range(5):
        assert unlock(client, session_id, "nope-nope").status_code == 403
    throttled = unlock(client, session_id)
    assert throttled.status_code == 429
    assert throttled.json()["error"]["code"] == "too_many_attempts"
    # The other session is not punished for it.
    assert unlock(client, other).status_code == 200


def test_locking_gives_back_the_presented_token_only(client):
    session_id = new_session(client)
    first = protect(client, session_id)
    second = unlock(client, session_id).json()["token"]

    assert (
        client.post(f"/playground/api/sessions/{session_id}/lock", headers=with_token(first)).status_code
        == 204
    )
    assert client.get(f"/playground/api/sessions/{session_id}", headers=with_token(first)).status_code == 403
    assert client.get(f"/playground/api/sessions/{session_id}", headers=with_token(second)).status_code == 200
    # Without a token it is a no-op, not an error.
    assert client.post(f"/playground/api/sessions/{session_id}/lock").status_code == 204


def test_changing_the_password_revokes_every_token(client):
    session_id = new_session(client)
    first = protect(client, session_id)
    second = unlock(client, session_id).json()["token"]

    changed = client.post(
        f"/playground/api/sessions/{session_id}/password",
        json={"password": "another password"},
        headers=with_token(first),
    )
    assert changed.status_code == 200
    fresh = changed.json()["token"]
    for dead in (first, second):
        assert (
            client.get(f"/playground/api/sessions/{session_id}", headers=with_token(dead)).status_code == 403
        )
    assert client.get(f"/playground/api/sessions/{session_id}", headers=with_token(fresh)).status_code == 200
    assert unlock(client, session_id, PASSWORD).status_code == 403
    assert unlock(client, session_id, "another password").status_code == 200


def test_removing_the_password_opens_the_session(client):
    session_id = new_session(client)
    token = protect(client, session_id)
    removed = client.delete(f"/playground/api/sessions/{session_id}/password", headers=with_token(token))
    assert removed.status_code == 204
    assert listed(client, session_id)["locked"] is False
    assert client.get(f"/playground/api/sessions/{session_id}").status_code == 200
    assert client.delete(f"/playground/api/sessions/{session_id}/password").status_code == 409


def test_deleting_a_locked_session_needs_its_token_and_kills_it(client):
    session_id = new_session(client)
    token = protect(client, session_id)
    assert client.delete(f"/playground/api/sessions/{session_id}").status_code == 403
    assert (
        client.delete(f"/playground/api/sessions/{session_id}", headers=with_token(token)).status_code == 204
    )
    assert client.get(f"/playground/api/sessions/{session_id}", headers=with_token(token)).status_code == 404


def test_an_idle_token_expires(client, monkeypatch):
    import time

    session_id = new_session(client)
    token = protect(client, session_id)
    real = time.monotonic
    monkeypatch.setattr(
        playground_lock.time, "monotonic", lambda: real() + playground_lock.IDLE_TIMEOUT_S + 1
    )
    assert client.get(f"/playground/api/sessions/{session_id}", headers=with_token(token)).status_code == 403


def test_a_restart_locks_every_session_again(settings, engine):
    with make_client(create_app(settings, engine)) as first:
        session_id = new_session(first)
        token = protect(first, session_id)
    with make_client(create_app(settings, engine)) as second:
        assert listed(second, session_id)["locked"] is True
        assert (
            second.get(f"/playground/api/sessions/{session_id}", headers=with_token(token)).status_code == 403
        )
        assert unlock(second, session_id).status_code == 200


def test_the_hash_never_leaves_the_server(client, settings):
    session_id = new_session(client)
    token = protect(client, session_id)
    for response in (
        client.get("/playground/api/sessions"),
        client.get(f"/playground/api/sessions/{session_id}", headers=with_token(token)),
        unlock(client, session_id),
    ):
        assert "scrypt" not in response.text and "salt" not in response.text


# ── Images ─────────────────────────────────────────────────────────────────


def test_an_image_no_row_holds_is_a_404(client, settings):
    stray = PlaygroundStore(settings.server.playground_store).images_dir / "stray.png"
    stray.write_bytes(tiny_png())
    assert client.get("/playground/images/stray.png").status_code == 404
    assert client.get("/playground/images/../playground.db").status_code == 404


def test_images_carry_the_data_plane_credential(engine, tmp_path):
    settings = Settings.model_validate(
        {
            "server": {
                "api_key": "cle-secrete",
                "image_store": str(tmp_path / "images"),
                "playground_store": str(tmp_path / "playground"),
                "log_file": None,
            }
        }
    )
    bearer = {"Authorization": "Bearer cle-secrete"}
    with make_client(create_app(settings, engine)) as secured:
        session_id = secured.post("/playground/api/sessions", headers=bearer).json()["id"]
        secured.post(
            f"/playground/api/sessions/{session_id}/generations", data={"prompt": "a fox"}, headers=bearer
        )
        assert wait_until(
            lambda: (
                secured.get(f"/playground/api/sessions/{session_id}", headers=bearer).json()["generations"][
                    0
                ]["status"]
                == "completed"
            )
        )
        url = secured.get(f"/playground/api/sessions/{session_id}", headers=bearer).json()["generations"][0][
            "images"
        ][0]["url"]
        assert secured.get(url).status_code == 401
        assert secured.get(url, headers=bearer).status_code == 200
        assert secured.get(f"{url}/thumb").status_code == 401
        assert secured.get(f"{url}/thumb", headers=bearer).status_code == 200


# ── Thumbnails, behind the same lock ───────────────────────────────────────
#
# Asserted on the thumbnail route itself and never inferred from the image
# route passing. The two are separate routes with separate code: the lock check
# on this one is its own statement, and a copy-paste that dropped it would leave
# every test above green while publishing a locked session's pictures -- small,
# but pictures.


def test_a_locked_sessions_thumbnail_needs_that_sessions_token(client, settings):
    session_id, image_url, context_url = finished_session(client)
    other = new_session(client)
    token = protect(client, session_id)
    thumbs = PlaygroundStore(settings.server.playground_store).thumbs_dir

    def is_locked(response) -> bool:
        return response.status_code == 403 and response.json()["error"]["code"] == "session_locked"

    assert is_locked(client.get(f"{image_url}/thumb"))
    assert is_locked(client.get(f"{context_url}/thumb"))
    assert is_locked(client.get(f"{image_url}/thumb", params={"t": "junk"}))
    # Another session's live token opens nothing here either.
    assert is_locked(client.get(f"{image_url}/thumb", headers=with_token(protect(client, other))))
    # Refused before any work: the lock is checked before the derivation, so a
    # refusal leaves nothing on disk for a later request to find already made.
    assert list(thumbs.iterdir()) == []

    # Both token forms, because an `<img>` can only send the query one.
    with_header = client.get(f"{image_url}/thumb", headers=with_token(token))
    assert with_header.status_code == 200
    assert with_header.headers["content-type"] == "image/webp"
    assert client.get(f"{image_url}/thumb", params={"t": token}).status_code == 200
    assert client.get(f"{context_url}/thumb", params={"t": token}).status_code == 200

    # Re-locking stops serving it again, whatever the browser may still hold in
    # its own cache -- see the route's `TEMPORARY:` note.
    relocked = client.post(f"/playground/api/sessions/{session_id}/lock", headers=with_token(token))
    assert relocked.status_code == 204
    assert is_locked(client.get(f"{image_url}/thumb", params={"t": token}))


def test_a_thumbnail_no_row_holds_is_a_404(client, settings):
    stray = PlaygroundStore(settings.server.playground_store).images_dir / "stray.png"
    stray.write_bytes(tiny_png())
    assert client.get("/playground/images/stray.png/thumb").status_code == 404
    assert client.get("/playground/images/../playground.db/thumb").status_code == 404
    assert client.get("/playground/images/nope.png/thumb").status_code == 404
    # The row lookup is the traversal guard, so nothing was read or written.
    assert stray.is_file()
    assert list(PlaygroundStore(settings.server.playground_store).thumbs_dir.iterdir()) == []


def test_a_private_network_preflight_for_a_thumbnail_is_granted(client):
    """The grant is a path-prefix one, and the thumbnail route lives under it on
    purpose: an embedded client renders tiles from a private address too, and a
    sibling path would have failed the same preflight the image route needed."""
    response = client.options("/playground/images/anything.png/thumb", headers=PNA)
    assert response.status_code == 200
    assert response.headers["access-control-allow-private-network"] == "true"


# ── Recovery ───────────────────────────────────────────────────────────────


def test_the_admin_can_strip_a_password_without_knowing_it(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    credential.set_password("the admin password")
    settings = Settings.model_validate(
        {
            "server": {
                "image_store": str(tmp_path / "images"),
                "playground_store": str(tmp_path / "playground"),
                "log_file": None,
            }
        }
    )
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        token = protect(client, session_id)
        path = f"/admin/playground/sessions/{session_id}/password"

        assert client.delete(path).status_code == 401
        assert listed(client, session_id)["locked"] is True

        client.post("/admin/session", json={"password": "the admin password"})
        assert client.delete(path).status_code == 204
        assert listed(client, session_id)["locked"] is False
        assert client.get(f"/playground/api/sessions/{session_id}").status_code == 200
        assert client.delete(path).status_code == 409
        assert client.delete("/admin/playground/sessions/nope/password").status_code == 404
        # Old tokens are gone too, not that an open session checks them.
        assert (
            client.post(f"/playground/api/sessions/{session_id}/lock", headers=with_token(token)).status_code
            == 204
        )


# ── Migration ──────────────────────────────────────────────────────────────


def test_a_store_written_before_passwords_existed_is_migrated(tmp_path):
    directory = tmp_path / "playground"
    directory.mkdir()
    old = sqlite3.connect(directory / "playground.db", isolation_level=None)
    old.executescript(
        """
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY, title TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        INSERT INTO sessions VALUES ('s1', 'old', 1.0, 1.0);
        """
    )
    old.close()
    store = PlaygroundStore(directory)
    try:
        assert store.list_sessions()[0]["locked"] is False
        assert store.password_record("s1") is None
        store.set_password("s1", credential.hash_password(PASSWORD))
        assert store.list_sessions()[0]["locked"] is True
        assert credential.verify_record(PASSWORD, store.password_record("s1"))
    finally:
        store.close()


def test_a_damaged_record_stays_locked(tmp_path):
    store = PlaygroundStore(tmp_path / "playground")
    try:
        session_id = store.create_session()["id"]
        store._db.execute("UPDATE sessions SET password = 'not json' WHERE id = ?", (session_id,))
        assert store.list_sessions()[0]["locked"] is True
        assert credential.verify_record(PASSWORD, store.password_record(session_id)) is False
    finally:
        store.close()


# ── Cross-site, on the two routes where the origin never was the authority ──
#
# `deny_cross_site` guards every playground route except `GET
# /playground/images/{filename}` and its `/thumb` sibling. That exception is
# deliberate and narrow, and these tests are what make it a decision rather than
# an oversight: the first pins what changed, and the others pin everything that
# must not.


def image_of(client: TestClient) -> str:
    session_id = new_session(client)
    submit(client, session_id)
    wait_until(lambda: status_of(client, session_id) == "completed")
    url = generations(client, session_id)[0]["images"][0]["url"]
    return session_id, url


def test_an_image_is_readable_from_another_origin(client):
    """What changed, and why.

    The filename is a `uuid4().hex`, so naming a file is holding 122 bits of
    secret; the session lock is checked underneath regardless. An origin check
    on top of that only refused a page that already knew the name.

    What it did break is real: `/mcp` publishes this URL to models, and every
    MCP client renders in an origin of its own -- so the guard turned the
    advertised link into a 403 whose JSON body got saved as a `.png`.
    """
    _session_id, url = image_of(client)
    response = client.get(url, headers={"Origin": "tauri://localhost"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n", "the body is the image, not an error"


def test_a_locked_sessions_image_is_still_refused_from_another_origin(client):
    """The lock, not the origin, is what protects a protected session -- so
    relaxing the origin check must not reach it. This is the assertion that
    would fail if the relaxation had been applied one layer too high."""
    session_id, url = image_of(client)
    protect(client, session_id)
    response = client.get(url, headers={"Origin": "tauri://localhost"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "session_locked"


def test_a_locked_sessions_thumbnail_is_still_refused_from_another_origin(client):
    """The same assertion on the thumbnail route, made on the route itself.

    The two routes share a reasoning, not an implementation: this one derives a
    file rather than serving a stored one, and the lock check on it is its own
    line of code. A passing test on the image route says nothing about whether
    that line is there.
    """
    session_id, url = image_of(client)
    protect(client, session_id)
    response = client.get(f"{url}/thumb", headers={"Origin": "tauri://localhost"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "session_locked"


def test_the_api_still_refuses_another_origin(client):
    """Scope: one route, not the plane. A hostile page still cannot list what
    exists, so it cannot learn a filename to ask for in the first place."""
    session_id, _url = image_of(client)
    for response in (
        client.get("/playground/api/sessions", headers={"Origin": "http://evil.example"}),
        client.get(f"/playground/api/sessions/{session_id}", headers={"Origin": "http://evil.example"}),
        client.post("/playground/api/sessions", headers={"Origin": "http://evil.example"}),
    ):
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == "cross_site_denied"


def test_an_unguessable_name_is_the_capability(client):
    """The relaxation rests on the name being secret, so a name that holds no
    row stays a 404 from any origin -- there is nothing to enumerate."""
    for name in ("stray.png", "../playground.db", "00000000000000000000000000000000.png"):
        response = client.get(f"/playground/images/{name}", headers={"Origin": "tauri://localhost"})
        assert response.status_code == 404, name


# ── Private Network Access, granted on one route only ──────────────────────
#
# Chromium asks permission before letting a page reach a more private address
# than its own. Starlette's CORSMiddleware refuses that preflight outright, so
# an `<img src="http://127.0.0.1:8765/playground/images/…">` in an Electron chat
# client never loaded -- while `curl`, which sends no preflight, fetched the
# same URL perfectly. That asymmetry is what made it hard to see, and these
# tests are what stop it coming back.

PNA = {
    "Origin": "https://app.example",
    "Access-Control-Request-Method": "GET",
    "Access-Control-Request-Private-Network": "true",
}


def test_a_private_network_preflight_for_an_image_is_granted(client):
    response = client.options("/playground/images/anything.png", headers=PNA)
    assert response.status_code == 200
    assert response.headers["access-control-allow-private-network"] == "true"


@pytest.mark.parametrize(
    "route",
    [
        "/v1/images/generations",
        "/v1/models",
        "/admin/config",
        "/playground/api/sessions",
        "/mcp",
    ],
)
def test_every_other_route_still_refuses_a_private_network_preflight(client, route):
    """The grant is one route wide, and this is what says so.

    `allow_private_network=True` on the CORS middleware would have been one
    line and would have granted the same to all of these. A keyless loopback
    install has an open `/v1`; granting it there lets any page in any tab spend
    this machine's GPU, which is the exact drive-by that PNA exists to stop.
    """
    response = client.options(route, headers=PNA)
    assert response.status_code == 400, response.text
    assert "private-network" in response.text


def test_the_grant_does_not_answer_an_ordinary_preflight(client):
    """Only the private-network question is answered here; everything else is
    still the CORS middleware's to decide, unchanged."""
    ordinary = {"Origin": "https://app.example", "Access-Control-Request-Method": "GET"}
    response = client.options("/playground/images/anything.png", headers=ordinary)
    assert "access-control-allow-private-network" not in response.headers


def test_granting_the_preflight_does_not_grant_the_image(client):
    """A preflight says whether the request may be *made*. Every check on the
    route still runs on the GET that follows -- so a locked session's image is
    still refused, and a name no row holds is still a 404."""
    session_id, url = image_of(client)
    protect(client, session_id)

    assert client.options(url, headers=PNA).status_code == 200
    refused = client.get(url, headers={"Origin": "https://app.example"})
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "session_locked"

    assert client.get("/playground/images/nope.png", headers=PNA).status_code == 404


def test_a_locked_projects_in_flight_frame_is_not_served(client, engine, monkeypatch):
    """The lock covers the picture being denoised, not just the finished file.

    A project can be locked *while* it generates -- locking is allowed at any
    time and does not stop the run -- and `/playground/api/preview` serves
    whatever the engine currently holds. Before this check the route asked
    nothing: a caller with no token got the in-flight frame with a 200, byte for
    byte what the token holder got, while `GET /sessions/{id}` refused them 403.
    A half-denoised picture is still the picture.

    The run is held open rather than left to the fake engine, which finishes
    instantly: the whole question is what happens *while* a generation is on the
    worker, so `current_id` has to still be set when the frame is asked for.
    """
    session_id = new_session(client)
    engine.preview_bytes = b"\xff\xd8\xff-in-flight"

    started = threading.Event()
    release = threading.Event()
    original = engine.generate

    async def held(job):
        started.set()
        # Bounded: a wedged test is worse than a failing one.
        await asyncio.get_running_loop().run_in_executor(None, release.wait, 10)
        return await original(job)

    monkeypatch.setattr(engine, "generate", held)

    assert submit(client, session_id).status_code == 202
    assert started.wait(10), "the generation never reached the engine"

    try:
        token = protect(client, session_id)

        refused = client.get("/playground/api/preview")
        assert refused.status_code == 403, refused.text
        assert refused.json()["error"]["code"] == "session_locked"
        assert b"in-flight" not in refused.content

        # And the entitled caller still sees it: the lock withholds, it does not
        # break the feature for the person who set it.
        allowed = client.get("/playground/api/preview", headers=with_token(token))
        assert allowed.status_code == 200
        assert allowed.content == engine.preview_bytes
    finally:
        release.set()
