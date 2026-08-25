"""The cover image `/playground/api/sessions` publishes for each project.

The rail draws a project as a picture rather than as a letter, so the list
payload carries one image per row. That list is the one playground endpoint
deliberately reachable *without* an unlock token, which makes the cover a
content-authority question rather than a cosmetic one.

The contract these tests defend is `test_playground_lock.py`'s, quoted here
because this file is where it can now be broken: once a session has a password,
nothing of it is served without a live unlock token for that very session, and
**the list endpoint keeps showing the row, and only the row**.

A cover is content. Worse, the cover URL *is* the capability: the filename is a
`uuid4().hex` and the image route's whole traversal guard is that naming a file
means holding those 122 bits. Publishing one for a locked project on an
endpoint that asks for no token would therefore hand out the secret the lock
exists to withhold — not merely reveal that a picture exists.

So a locked project carries `null`, and it carries the same `null` a project
with no images at all carries: nothing in the payload distinguishes "locked"
from "empty", which is the point. `test_a_locked_project_publishes_no_cover`
below is the witness that fails if any of that regresses.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from qds.playground import PlaygroundStore
from tests.conftest import wait_until
from tests.test_playground import generations, new_session, status_of, submit
from tests.test_playground_lock import PASSWORD, protect, with_token


def cover_of(client: TestClient, session_id: str) -> str | None:
    rows = client.get("/playground/api/sessions").json()["sessions"]
    return next(row for row in rows if row["id"] == session_id)["cover"]


def finished(client: TestClient, session_id: str, **fields) -> list[dict]:
    """Run one generation to completion and return its images, newest run last."""
    assert submit(client, session_id, **fields).status_code == 202
    index = len(generations(client, session_id)) - 1
    assert wait_until(lambda: status_of(client, session_id, index) == "completed")
    return generations(client, session_id)[index]["images"]


def test_an_open_project_publishes_a_fetchable_thumbnail_cover(client):
    session_id = new_session(client)
    images = finished(client, session_id)

    cover = cover_of(client, session_id)
    # The thumbnail route, never the full image: a rail tile is ~40px, and the
    # stored files average megabytes.
    assert cover == f"{images[0]['url']}/thumb"

    # Fetchable *by this caller*, which is the only claim that matters — a URL
    # in a payload the browser cannot then load is a broken tile, not a cover.
    served = client.get(cover)
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/webp"


def test_a_project_with_no_images_publishes_null(client):
    # `null` rather than a URL derived from the session id: a route that 404s is
    # a broken image in the rail, and the rail's letter-and-hue landmark is what
    # `null` selects.
    assert cover_of(client, new_session(client)) is None


def test_a_locked_project_publishes_no_cover(client):
    """The one that matters. A leak here is a leak of the capability itself."""
    session_id = new_session(client)
    images = finished(client, session_id)
    assert cover_of(client, session_id) is not None

    token = protect(client, session_id)

    # Asserted on the payload directly, not on what the rail chooses to draw:
    # the client is not the enforcement point, and a second tab of the same
    # browser would receive whatever this row holds.
    row = next(
        entry
        for entry in client.get("/playground/api/sessions").json()["sessions"]
        if entry["id"] == session_id
    )
    assert row["locked"] is True
    assert row["cover"] is None
    # And the filename is nowhere else in the row either — the point is that the
    # 122-bit name never crosses this endpoint, by any key.
    assert images[0]["url"].rsplit("/", 1)[-1] not in repr(row)

    # Holding the token changes nothing about the *list*: the row stays a row.
    # The unlocked caller reaches the image by the detail endpoint, which is
    # where content lives, so nothing is lost by refusing it here.
    listed = client.get("/playground/api/sessions", headers=with_token(token))
    assert next(
        entry for entry in listed.json()["sessions"] if entry["id"] == session_id
    )["cover"] is None
    assert client.get(f"{images[0]['url']}/thumb", headers=with_token(token)).status_code == 200


def test_locking_a_project_withdraws_the_cover_it_had(client):
    session_id = new_session(client)
    finished(client, session_id)
    assert cover_of(client, session_id) is not None

    token = protect(client, session_id)
    assert cover_of(client, session_id) is None

    # And removing the password gives it back: the lock withholds, it does not
    # destroy. Asserted so that an implementation which simply stopped
    # publishing covers at all would fail this test rather than pass it.
    removed = client.delete(
        f"/playground/api/sessions/{session_id}/password", headers=with_token(token)
    )
    assert removed.status_code == 204
    assert cover_of(client, session_id) is not None


def test_deleting_the_cover_image_promotes_the_next_one(client):
    session_id = new_session(client)
    first = finished(client, session_id)
    second = finished(client, session_id)
    # The most recent image, so the rail shows what the project last produced.
    assert cover_of(client, session_id) == f"{second[0]['url']}/thumb"

    assert client.delete(second[0]["url"].replace("/images/", "/api/images/")).status_code == 204
    assert cover_of(client, session_id) == f"{first[0]['url']}/thumb"

    assert client.delete(first[0]["url"].replace("/images/", "/api/images/")).status_code == 204
    assert cover_of(client, session_id) is None


def test_the_list_is_still_one_statement_per_call(tmp_path):
    """No N+1: covers cost the same single statement the list already cost.

    A per-row query would be invisible in every assertion above — the payload
    would be identical — and would multiply the cost of a rail refresh by the
    number of projects, on a list every open tab polls.

    Asked of the store rather than through HTTP, and with sqlite3's own trace
    callback rather than a wrapper: `Connection.execute` is a read-only
    attribute, so a monkeypatched counter cannot see the statements this method
    actually issues. The trace fires once per statement the connection runs,
    which is exactly the quantity under test.
    """
    store = PlaygroundStore(tmp_path / "store")
    try:
        for _ in range(3):
            session = store.create_session()["id"]
            record = store.add_generation(
                session,
                prompt="a fox",
                model="m",
                kind="text",
                n=1,
                width=64,
                height=64,
                steps=1,
                steps_from_preset=False,
                seeds=[1],
            )
            assert store.add_image(record["id"], 0, f"{record['id']}.png", 1)

        statements: list[str] = []
        store._db.set_trace_callback(statements.append)
        rows = store.list_sessions()
        store._db.set_trace_callback(None)

        assert len(rows) == 3
        assert all(row["cover"] is not None for row in rows)
        assert len(statements) == 1, statements
    finally:
        store.close()


def test_a_locked_project_and_an_empty_one_are_indistinguishable(client):
    """Same payload for both, so the row does not report whether a locked
    project holds images. `null` is not a placeholder that says "there is
    something here you may not see"."""
    locked = new_session(client)
    finished(client, locked)
    protect(client, locked, PASSWORD)
    empty = new_session(client)

    rows = {
        row["id"]: row for row in client.get("/playground/api/sessions").json()["sessions"]
    }
    assert rows[locked]["cover"] is None
    assert rows[empty]["cover"] is None
    assert set(rows[locked]) == set(rows[empty])
