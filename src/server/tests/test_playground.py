"""The playground's durable generations.

What is tested here is the property the `/v1` plane does not have: a generation
accepted by the playground reaches a terminal status and keeps its images, even
when the browser closes or the process restarts. Inference is still faked — the
subject is the record, not the pixels.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from qds import playground as playground_module
from qds import settings as settings_module
from qds.app import create_app
from qds.playground import PlaygroundStore
from qds.settings import Settings
from tests.conftest import FakeEngine, make_client, tiny_png, wait_until


def images_dir(settings) -> Path:
    return Path(settings.server.playground_store, "images")


class BlockingEngine(FakeEngine):
    """A generation that does not finish until the test says so.

    A polled flag rather than an `asyncio.Event`: `TestClient` drives the app
    from another thread, and setting an event from there would touch a loop the
    test does not own.
    """

    def __init__(self) -> None:
        super().__init__()
        self.release = False

    async def generate(self, job):
        self.busy = True
        try:
            while not self.release:
                if self.cancel_requested:
                    from mflux.utils.exceptions import StopImageGenerationException

                    raise StopImageGenerationException("Generation stopped by user.")
                await asyncio.sleep(0.005)
        finally:
            self.busy = False
        return await super().generate(job)


class LoadingEngine(FakeEngine):
    """Busy, but not denoising — the real engine while it loads weights.

    The distinction is the whole point: `ModelEngine.request_cancel()` returns
    False and sets nothing unless its snapshot says `generating`, so a double
    that reports itself busy would hide the window in which a cancel has to be
    honoured by the runner rather than by the engine.
    """

    def __init__(self) -> None:
        super().__init__()
        self.release = False

    async def generate(self, job):
        while not self.release:
            await asyncio.sleep(0.005)
        return await super().generate(job)


def new_session(client: TestClient) -> str:
    response = client.post("/playground/api/sessions")
    assert response.status_code == 201
    return response.json()["id"]


def submit(client: TestClient, session_id: str, **fields):
    body = {"prompt": "a fox", **fields}
    files = body.pop("files", None)
    return client.post(
        f"/playground/api/sessions/{session_id}/generations", data=body, files=files
    )


def generations(client: TestClient, session_id: str) -> list[dict]:
    response = client.get(f"/playground/api/sessions/{session_id}")
    assert response.status_code == 200
    return response.json()["generations"]


def status_of(client: TestClient, session_id: str, index: int = 0) -> str:
    return generations(client, session_id)[index]["status"]


def updated_at(client: TestClient, session_id: str) -> float:
    listed = client.get("/playground/api/sessions").json()["sessions"]
    return next(entry for entry in listed if entry["id"] == session_id)["updatedAt"]


# ── Lifecycle ──────────────────────────────────────────────────────────────


def test_a_generation_runs_and_keeps_its_images(client, settings):
    session_id = new_session(client)
    accepted = submit(client, session_id, n=2)
    assert accepted.status_code == 202
    record = accepted.json()
    assert record["status"] == "queued"
    assert record["n"] == 2
    assert len(record["seeds"]) == 2
    assert record["images"] == []
    # An untouched composer sends no size and no step count, and must still get
    # the model's own defaults — the property the advanced fields must not move.
    assert record["size"] == "1920x1072"
    assert record["steps"] == 4

    assert wait_until(lambda: status_of(client, session_id) == "completed")
    done = generations(client, session_id)[0]
    assert len(done["images"]) == 2
    assert [image["seed"] for image in done["images"]] == done["seeds"]
    assert done["finishedAt"] is not None and done["startedAt"] is not None

    # Served, and out of `image_store`'s reach: the TTL purge cannot touch a
    # session's images.
    for image in done["images"]:
        served = client.get(image["url"])
        assert served.status_code == 200
        assert served.content[:4] == b"\x89PNG"
        name = image["url"].rsplit("/", 1)[-1]
        assert images_dir(settings).joinpath(name).is_file()
        assert not Path(settings.server.image_store, name).exists()


def test_the_session_takes_its_title_from_the_first_prompt(client):
    session_id = new_session(client)
    submit(client, session_id, prompt="a fox in the snow")
    assert wait_until(lambda: status_of(client, session_id) == "completed")

    listed = client.get("/playground/api/sessions").json()["sessions"]
    row = next(entry for entry in listed if entry["id"] == session_id)
    assert row["title"] == "a fox in the snow"
    assert row["generating"] is False

    # A second prompt does not rename the session.
    submit(client, session_id, prompt="a bear")
    assert wait_until(lambda: len(generations(client, session_id)) == 2)
    listed = client.get("/playground/api/sessions").json()["sessions"]
    assert next(e for e in listed if e["id"] == session_id)["title"] == "a fox in the snow"


def test_a_queued_generation_is_reported_as_generating(client, engine, settings):
    """`generating` is what the sidebar's live dot reads."""
    with make_client(create_app(settings, BlockingEngine())) as blocked:
        session_id = new_session(blocked)
        submit(blocked, session_id)
        assert wait_until(lambda: status_of(blocked, session_id) == "running")
        listed = blocked.get("/playground/api/sessions").json()["sessions"]
        assert next(e for e in listed if e["id"] == session_id)["generating"] is True


# ── Validation ─────────────────────────────────────────────────────────────


def test_an_unknown_session_is_a_404(client):
    assert client.get("/playground/api/sessions/nope").status_code == 404
    response = submit(client, "nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_validation_mirrors_the_v1_plane(client):
    session_id = new_session(client)
    assert submit(client, session_id, model="sdxl").json()["error"]["code"] == "model_not_found"
    assert submit(client, session_id, n=99).json()["error"]["code"] == "n_too_large"
    assert submit(client, session_id, n=0).json()["error"]["code"] == "invalid_n"
    # FIBO reads JSON captions only, and says so before any weights load.
    fibo = submit(client, session_id, model="fibo", prompt="a fox")
    assert fibo.json()["error"]["code"] == "prompt_must_be_json"
    assert generations(client, session_id) == []


def test_advanced_fields_reach_the_engine(client, engine):
    session_id = new_session(client)
    record = submit(client, session_id, size="640x384", steps=12, seed=7, n=2).json()
    assert record["size"] == "640x384"
    assert record["steps"] == 12
    assert record["seeds"] == [7, 8]

    assert wait_until(lambda: status_of(client, session_id) == "completed")
    job = engine.jobs[-1]
    assert (job.width, job.height) == (640, 384)
    assert job.steps == 12
    assert job.seed == 8
    assert job.steps_from_preset is False
    # The playground is the only submitter that asks for step previews; `/v1`
    # jobs leave this at 0.
    assert job.preview_every == playground_module.PREVIEW_EVERY > 0


def test_deleting_an_image_removes_the_row_and_its_file(client, settings):
    session_id = new_session(client)
    submit(client, session_id, prompt="a fox", n=2)
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    images = generations(client, session_id)[0]["images"]
    assert len(images) == 2
    names = [image["url"].rsplit("/", 1)[-1] for image in images]
    before = updated_at(client, session_id)

    assert client.delete(f"/playground/api/images/{names[0]}").status_code == 204

    # The generation survives its image: the prompt history is the transcript.
    entry = generations(client, session_id)[0]
    assert entry["prompt"] == "a fox"
    assert [image["url"] for image in entry["images"]] == [images[1]["url"]]
    assert not images_dir(settings).joinpath(names[0]).exists()
    assert images_dir(settings).joinpath(names[1]).is_file()
    # The sidebar sorts by this and prints it: a deletion is a session change.
    assert updated_at(client, session_id) > before


def test_deleting_an_unknown_image_is_a_404(client, settings):
    response = client.delete("/playground/api/images/nope.png")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"

    # Fail-closed unlink, witnessed by a file that really is in `images_dir` and
    # really has no row: an uploaded reference image. Unlinking before the row
    # lookup would delete it — and `missing_ok=True` means a name from outside
    # that directory could never have shown the difference.
    session_id = new_session(client)
    accepted = submit(
        client,
        session_id,
        model="qwen-image-2512",
        files={"image": ("ctx.png", tiny_png(), "image/png")},
    )
    assert accepted.status_code == 202
    context = accepted.json()["contextImage"].rsplit("/", 1)[-1]
    reference = images_dir(settings) / context
    assert reference.is_file()

    assert client.delete(f"/playground/api/images/{context}").status_code == 404
    assert reference.is_file()


def test_deleting_the_last_image_of_a_group_dissolves_the_group(client, settings):
    """An entry that is nothing but a prompt is noise, so it goes with the image."""
    session_id = new_session(client)
    submit(client, session_id, prompt="a fox")
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    name = generations(client, session_id)[0]["images"][0]["url"].rsplit("/", 1)[-1]

    assert client.delete(f"/playground/api/images/{name}").status_code == 204

    # The session is the transcript and survives; its generation does not.
    assert client.get(f"/playground/api/sessions/{session_id}").status_code == 200
    assert generations(client, session_id) == []
    assert not images_dir(settings).joinpath(name).exists()


def test_dissolving_a_group_unlinks_the_reference_its_root_was_made_with(client, settings):
    session_id = new_session(client)
    accepted = submit(
        client,
        session_id,
        model="qwen-image-2512",
        files={"image": ("ctx.png", tiny_png(), "image/png")},
    )
    assert accepted.status_code == 202
    reference = images_dir(settings) / accepted.json()["contextImage"].rsplit("/", 1)[-1]
    assert reference.is_file()
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    name = generations(client, session_id)[0]["images"][0]["url"].rsplit("/", 1)[-1]

    assert client.delete(f"/playground/api/images/{name}").status_code == 204

    # The reference image is a column, not a row: the dissolve is the only thing
    # that removes it.
    assert generations(client, session_id) == []
    assert not reference.exists()


def test_a_group_with_a_member_still_running_is_not_dissolved(settings):
    """A queued/running member will bring the group's next image: keep the prompt."""
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as blocked:
        session_id = new_session(blocked)
        first = submit(blocked, session_id, prompt="a fox").json()
        engine.release = True
        assert wait_until(lambda: status_of(blocked, session_id) == "completed")
        name = generations(blocked, session_id)[0]["images"][0]["url"].rsplit("/", 1)[-1]

        # A refine of the group, still running: its image has not landed yet.
        engine.release = False
        submit(blocked, session_id, prompt="a fox", group=first["groupId"])
        assert wait_until(lambda: status_of(blocked, session_id, 1) == "running")

        assert blocked.delete(f"/playground/api/images/{name}").status_code == 204

        entries = generations(blocked, session_id)
        assert len(entries) == 2
        assert entries[0]["images"] == []
        assert not images_dir(settings).joinpath(name).exists()

        # And the running member still lands in the group it belongs to.
        engine.release = True
        assert wait_until(
            lambda: len(generations(blocked, session_id)[1]["images"]) == 1
        )


# ── Groups ─────────────────────────────────────────────────────────────────


def test_a_generation_without_a_group_starts_its_own(client):
    session_id = new_session(client)
    record = submit(client, session_id).json()
    assert record["groupId"] == record["id"]


def test_a_grouped_generation_joins_the_lineage_it_names(client):
    """What the feed renders as one entry: several generations, one group."""
    session_id = new_session(client)
    first = submit(client, session_id, prompt="a fox").json()
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    second = submit(client, session_id, prompt="a fox", group=first["groupId"]).json()
    assert second["id"] != first["id"]
    assert second["groupId"] == first["groupId"]
    # A third one may name the group through the member that joined it, not just
    # through the root: the group id is the lineage, not a parent pointer.
    third = submit(client, session_id, prompt="a fox", group=second["groupId"]).json()
    assert third["groupId"] == first["groupId"]

    # Durable, and read back from the database rather than from the accepted
    # record: the grouping survives the reload the feed rebuilds itself from.
    assert wait_until(lambda: len(generations(client, session_id)) == 3)
    listed = generations(client, session_id)
    assert [entry["groupId"] for entry in listed] == [first["groupId"]] * 3
    # Still three independent records: grouping is presentation, not a merge.
    assert len({entry["id"] for entry in listed}) == 3


def test_a_group_survives_reopening_the_store(tmp_path):
    """Grouping is durable state, not a fact about one live connection."""
    directory = tmp_path / "playground"
    store = PlaygroundStore(directory)
    fields = dict(
        prompt="a fox",
        model="qwen-image-2512",
        kind="txt2img",
        n=1,
        width=512,
        height=512,
        steps=4,
        steps_from_preset=False,
        seeds=[1],
    )
    try:
        session_id = store.create_session()["id"]
        root = store.add_generation(session_id, **fields)
        store.add_generation(session_id, group=root["groupId"], **fields)
    finally:
        store.close()

    reopened = PlaygroundStore(directory)
    try:
        listed = reopened.get_session(session_id)["generations"]
        assert [entry["groupId"] for entry in listed] == [root["groupId"]] * 2
        # And the reopen ran no second migration on an already-migrated store.
        third = reopened.add_generation(session_id, group=root["groupId"], **fields)
        assert third["groupId"] == root["groupId"]
    finally:
        reopened.close()


def test_a_group_from_another_session_is_refused(client):
    """A client-supplied group id must not graft one session onto another."""
    session_id = new_session(client)
    other_id = new_session(client)
    theirs = submit(client, other_id).json()

    refused = submit(client, session_id, group=theirs["groupId"])
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_group"
    assert refused.json()["error"]["param"] == "group"
    unknown = submit(client, session_id, group="nope")
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "invalid_group"
    # Fail-closed: nothing was queued, in either session.
    assert generations(client, session_id) == []
    assert len(generations(client, other_id)) == 1


def test_a_refused_group_removes_the_upload_it_arrived_with(client, settings):
    """The reference image of a rejected submission must not stay on disk."""
    session_id = new_session(client)
    before = set(images_dir(settings).iterdir())
    refused = submit(
        client,
        session_id,
        model="qwen-image-2512",
        group="nope",
        files={"image": ("ctx.png", tiny_png(), "image/png")},
    )
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_group"
    assert set(images_dir(settings).iterdir()) == before


def test_a_store_written_before_groups_existed_is_migrated(tmp_path):
    """Every pre-existing generation is its own group, and stays readable."""
    directory = tmp_path / "playground"
    directory.mkdir()
    (directory / "images").mkdir()
    old = sqlite3.connect(directory / "playground.db", isolation_level=None)
    old.executescript(
        """
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY, title TEXT, created_at REAL NOT NULL,
          updated_at REAL NOT NULL
        );
        CREATE TABLE generations (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id)
            ON DELETE CASCADE,
          prompt TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
          n INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
          steps INTEGER NOT NULL, steps_from_preset INTEGER NOT NULL,
          seeds TEXT NOT NULL, image_strength REAL, context_image TEXT,
          status TEXT NOT NULL, error TEXT, created_at REAL NOT NULL,
          started_at REAL, finished_at REAL
        );
        CREATE TABLE generation_images (
          generation_id TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
          position INTEGER NOT NULL, filename TEXT NOT NULL, seed INTEGER NOT NULL,
          PRIMARY KEY (generation_id, position)
        );
        INSERT INTO sessions VALUES ('s1', 'old', 1.0, 1.0);
        INSERT INTO generations VALUES (
          'g1', 's1', 'an old fox', 'qwen-image-2512', 'txt2img', 1, 512, 512, 4,
          0, '[7]', NULL, NULL, 'completed', NULL, 1.0, 1.0, 2.0
        );
        """
    )
    old.close()

    store = PlaygroundStore(directory)
    try:
        payload = store.get_session("s1")
        assert payload is not None
        [entry] = payload["generations"]
        assert entry["groupId"] == "g1"
        # And the migrated table takes new members of that same group.
        joined = store.add_generation(
            "s1",
            prompt="an old fox",
            model="qwen-image-2512",
            kind="txt2img",
            n=1,
            width=512,
            height=512,
            steps=4,
            steps_from_preset=False,
            seeds=[8],
            group="g1",
        )
        assert joined["groupId"] == "g1"
    finally:
        store.close()


def test_an_explicit_step_count_overrides_the_preset(client, engine):
    """`ideogram-4`'s preset owns the schedule only while nobody names a number."""
    session_id = new_session(client)
    assert submit(client, session_id, model="ideogram-4", steps=12).status_code == 202
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    assert engine.jobs[-1].steps == 12
    assert engine.jobs[-1].steps_from_preset is False

    assert submit(client, session_id, model="ideogram-4").status_code == 202
    assert wait_until(lambda: len(engine.jobs) == 2)
    assert engine.jobs[-1].steps == 20
    assert engine.jobs[-1].steps_from_preset is True


def test_invalid_advanced_fields_are_rejected(client):
    session_id = new_session(client)
    steps = submit(client, session_id, steps=0)
    assert steps.status_code == 400
    assert steps.json()["error"]["param"] == "steps"
    seed = submit(client, session_id, seed=-1)
    assert seed.status_code == 400
    assert seed.json()["error"]["param"] == "seed"
    assert submit(client, session_id, size="abc").json()["error"]["code"] == "invalid_size"
    too_small = submit(client, session_id, size="64x64", model="ideogram-4")
    assert too_small.json()["error"]["code"] == "invalid_size"
    assert generations(client, session_id) == []


def test_a_context_image_becomes_an_edit_when_the_model_edits(client, engine):
    session_id = new_session(client)
    response = submit(
        client,
        session_id,
        model="qwen-image-2512",
        files={"image": ("ctx.png", tiny_png(), "image/png")},
    )
    assert response.status_code == 202
    record = response.json()
    assert record["kind"] == "edit"
    assert record["contextImage"].startswith("/playground/images/ctx-")

    assert wait_until(lambda: status_of(client, session_id) == "completed")
    job = engine.jobs[-1]
    assert job.kind == "edit"
    assert job.image_path is not None and job.image_path.is_file()
    assert job.image_strength is None
    # The reference image is served back, for the thumbnail in the feed.
    assert client.get(record["contextImage"]).status_code == 200


def test_a_context_image_falls_back_to_img2img(client, engine):
    session_id = new_session(client)
    record = submit(
        client,
        session_id,
        model="z-image-turbo",
        files={"image": ("ctx.png", tiny_png(), "image/png")},
    ).json()
    assert record["kind"] == "txt2img"
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    assert engine.jobs[-1].image_strength == 0.4


def test_a_context_image_is_refused_by_a_text_only_model(client, settings):
    session_id = new_session(client)
    response = submit(
        client,
        session_id,
        model="ideogram-4",
        files={"image": ("ctx.png", tiny_png(), "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert generations(client, session_id) == []
    # And the upload it refused left nothing behind.
    assert list(images_dir(settings).iterdir()) == []


# ── Cancellation ───────────────────────────────────────────────────────────


def test_a_queued_generation_is_cancelled_without_running(settings):
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        first = submit(client, session_id).json()
        second = submit(client, session_id).json()
        assert wait_until(lambda: status_of(client, session_id) == "running")

        cancelled = client.post(f"/playground/api/generations/{second['id']}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        engine.release = True
        assert wait_until(lambda: status_of(client, session_id, 0) == "completed")
        # Never handed to the engine: one job for the first generation only.
        assert len(engine.jobs) == 1
        assert status_of(client, session_id, 1) == "cancelled"
        assert first["id"] != second["id"]


def test_a_running_generation_is_cancelled_through_the_engine(settings):
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        record = submit(client, session_id).json()
        assert wait_until(lambda: status_of(client, session_id) == "running")

        response = client.post(f"/playground/api/generations/{record['id']}/cancel")
        assert response.status_code == 200
        assert wait_until(lambda: status_of(client, session_id) == "cancelled")
        assert engine.jobs == []
        # The server stays usable: a later generation still runs.
        engine.release = True
        again = submit(client, session_id).json()
        assert wait_until(
            lambda: next(
                g["status"] for g in generations(client, session_id) if g["id"] == again["id"]
            )
            == "completed"
        )


def test_a_generation_is_cancelled_while_the_model_loads(settings):
    """The window the engine cannot close, and the runner must.

    `engine.request_cancel()` does nothing outside a denoising loop, so a Cancel
    pressed during the weight load used to answer 200 and change nothing: the
    record stayed `running` and the generation ran to completion.
    """
    engine = LoadingEngine()
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        record = submit(client, session_id).json()
        assert wait_until(lambda: status_of(client, session_id) == "running")
        assert engine.cancel_requested is False

        response = client.post(f"/playground/api/generations/{record['id']}/cancel")
        assert response.status_code == 200
        # The image being computed is finished and kept — it is paid for — and the
        # record is terminal rather than running.
        engine.release = True
        assert wait_until(lambda: status_of(client, session_id) == "cancelled")
        assert len(generations(client, session_id)[0]["images"]) == 1


def test_cancelling_between_images_stops_the_remaining_ones(settings):
    engine = LoadingEngine()
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        record = submit(client, session_id, n=3).json()
        assert wait_until(lambda: status_of(client, session_id) == "running")
        client.post(f"/playground/api/generations/{record['id']}/cancel")

        engine.release = True
        assert wait_until(lambda: status_of(client, session_id) == "cancelled")
        # One image, not three: the request was honoured at the next boundary.
        assert len(engine.jobs) == 1
        assert len(generations(client, session_id)[0]["images"]) == 1


def test_cancelling_an_unknown_generation_is_a_404(client):
    assert client.post("/playground/api/generations/nope/cancel").status_code == 404


def test_the_preview_endpoint_serves_the_engine_slot(client, engine):
    """Empty slot is a 404, which is what the client's stale fetch races into."""
    assert client.get("/playground/api/preview").status_code == 404

    engine.preview_bytes = b"\xff\xd8fake"
    response = client.get("/playground/api/preview")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # Never cached: one URL per frame already, and the slot is overwritten.
    assert response.headers["cache-control"] == "no-store"
    assert response.content == b"\xff\xd8fake"


# ── Where the store lives ──────────────────────────────────────────────────


def test_the_default_store_sits_beside_the_configuration(tmp_path, monkeypatch):
    """The app-bundle crash, as an assertion.

    A field default bypasses pydantic's validators, so a CWD-relative default
    reached `mkdir` unresolved and was created — or refused — wherever the
    process happened to be. Launched from a bundle that is `/`, startup died with
    `Read-only file system: 'playground'`. The anchor is the configuration file,
    which identifies the installation.
    """
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    server = Settings.model_validate({"models": {}}).server
    assert server.playground_store is None
    assert settings_module.playground_directory(server) == tmp_path / "playground"


def test_the_store_does_not_follow_the_working_directory(tmp_path, monkeypatch):
    """The negative of the above: the CWD must not move the store."""
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "conf" / "server-config.json"))
    server = Settings.model_validate({"models": {}}).server
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    assert settings_module.playground_directory(server) == tmp_path / "conf" / "playground"


def test_an_explicit_relative_store_is_made_absolute(tmp_path, monkeypatch):
    """A written-down relative path is a choice, and resolves like `image_store`."""
    monkeypatch.chdir(tmp_path)
    server = Settings.model_validate({"server": {"playground_store": "pg"}, "models": {}}).server
    assert Path(server.playground_store).is_absolute()
    assert settings_module.playground_directory(server) == tmp_path / "pg"


def test_a_server_with_no_configured_store_starts(tmp_path, monkeypatch, engine):
    """End to end: `create_app` must not touch the working directory."""
    monkeypatch.setenv("QDS_SERVER_CONFIG", str(tmp_path / "server-config.json"))
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    settings = Settings.model_validate(
        {
            "server": {"image_store": str(tmp_path / "images"), "log_file": None},
            "default_model": "z-image-turbo",
        }
    )
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        assert submit(client, session_id).status_code == 202
        assert wait_until(lambda: status_of(client, session_id) == "completed")
    assert (tmp_path / "playground" / "playground.db").is_file()
    assert not (tmp_path / "cwd" / "playground").exists()


# ── Restart ────────────────────────────────────────────────────────────────


def test_a_generation_interrupted_by_a_restart_becomes_failed(settings):
    """Invariant: no record stays non-terminal across a restart."""
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        submit(client, session_id)
        assert wait_until(lambda: status_of(client, session_id) == "running")

    with make_client(create_app(settings, FakeEngine())) as restarted:
        record = generations(restarted, session_id)[0]
        assert record["status"] == "failed"
        assert record["error"] == "Interrupted by server restart"
        assert record["finishedAt"] is not None
        listed = restarted.get("/playground/api/sessions").json()["sessions"]
        assert next(e for e in listed if e["id"] == session_id)["generating"] is False


# ── Deletion ───────────────────────────────────────────────────────────────


def test_deleting_a_session_removes_its_rows_and_files(client, settings):
    session_id = new_session(client)
    submit(client, session_id, files={"image": ("ctx.png", tiny_png(), "image/png")})
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    urls = [image["url"] for image in generations(client, session_id)[0]["images"]]
    context = generations(client, session_id)[0]["contextImage"]

    assert client.delete(f"/playground/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/playground/api/sessions/{session_id}").status_code == 404
    assert session_id not in {e["id"] for e in client.get("/playground/api/sessions").json()["sessions"]}
    for url in [*urls, context]:
        assert client.get(url).status_code == 404
    assert list(images_dir(settings).iterdir()) == []


def test_deleting_a_session_stops_its_running_generation(settings):
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        submit(client, session_id)
        assert wait_until(lambda: status_of(client, session_id) == "running")
        assert client.delete(f"/playground/api/sessions/{session_id}").status_code == 204
        assert engine.cancel_requested is True
        assert client.get(f"/playground/api/sessions/{session_id}").status_code == 404


def test_deleting_an_unknown_session_is_a_404(client):
    assert client.delete("/playground/api/sessions/nope").status_code == 404
