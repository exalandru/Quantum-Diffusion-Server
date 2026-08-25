"""The playground's durable generations.

What is tested here is the property the `/v1` plane does not have: a generation
accepted by the playground reaches a terminal status and keeps its images, even
when the browser closes or the process restarts. Inference is still faked — the
subject is the record, not the pixels.
"""

from __future__ import annotations

import asyncio
import io
import sqlite3
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

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
        await self._block()
        return await super().generate(job)

    async def upscale(self, job):
        await self._block()
        return await super().upscale(job)

    async def _block(self):
        self.busy = True
        try:
            while not self.release:
                if self.cancel_requested:
                    from mflux.utils.exceptions import StopImageGenerationException

                    raise StopImageGenerationException("Generation stopped by user.")
                await asyncio.sleep(0.005)
        finally:
            self.busy = False


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
    headers = body.pop("headers", None)
    return client.post(
        f"/playground/api/sessions/{session_id}/generations", data=body, files=files, headers=headers
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


def test_an_upload_cannot_choose_the_type_it_is_served_as(client):
    """A reference image is stored under a suffix this server picked.

    The filename comes from the client, and the image route derives its content
    type from the stored name. Left alone, an upload called `payload.html` came
    back as `text/html` from this server's own origin -- a script holding the
    dashboard's cookies, one click away in the viewer's file link. The name is
    normalised on the way in and the type is declared on the way out; this pins
    both, and `nosniff` so a browser cannot overrule the second.
    """
    session_id = new_session(client)
    accepted = submit(
        client,
        session_id,
        model="qwen-image-2512",
        files={"image": ("payload.html", b"<script>alert(1)</script>", "text/html")},
    )
    assert accepted.status_code == 202

    context = accepted.json()["contextImage"]
    assert context.endswith(".png")

    served = client.get(context)
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.headers["x-content-type-options"] == "nosniff"


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


# ── Negative prompts ───────────────────────────────────────────────────────


def test_a_negative_prompt_reaches_the_engine(client, engine):
    session_id = new_session(client)
    submit(client, session_id, model="qwen-image-2512", negative_prompt="blurry, watermark")
    assert wait_until(lambda: engine.jobs)
    assert engine.jobs[-1].negative_prompt == "blurry, watermark"
    assert generations(client, session_id)[0]["negativePrompt"] == "blurry, watermark"


def test_a_negative_prompt_is_refused_by_a_model_without_one(client, engine):
    """The UI greys the field out; this is what makes that advice rather than
    the enforcement. `flux2-klein` is guidance-distilled — it has no
    unconditional branch to apply a negative prompt to."""
    session_id = new_session(client)
    response = submit(client, session_id, model="flux2-klein", negative_prompt="blurry")

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "negative_prompt"
    assert error["code"] == "unsupported_parameter"
    assert generations(client, session_id) == []


def test_a_blank_negative_prompt_is_no_negative_prompt(client, engine):
    """A browser posts the field whether or not it was typed in. Storing `""`
    would both misreport the request and refuse a model that cannot take one."""
    session_id = new_session(client)
    assert submit(client, session_id, model="flux2-klein", negative_prompt="   ").status_code == 202

    assert wait_until(lambda: engine.jobs)
    assert engine.jobs[-1].negative_prompt is None
    assert generations(client, session_id)[0]["negativePrompt"] is None


def test_a_store_written_before_negative_prompts_existed_is_migrated(tmp_path):
    """The column is added to a database an older build created, and the rows it
    already holds keep working — a negative prompt nobody sent is `NULL`."""
    directory = tmp_path / "playground"
    directory.mkdir()
    (directory / "images").mkdir()
    database = sqlite3.connect(directory / "playground.db", isolation_level=None)
    database.executescript(
        """
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY, title TEXT, created_at REAL NOT NULL,
          updated_at REAL NOT NULL, password TEXT
        );
        CREATE TABLE generations (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, group_id TEXT,
          prompt TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
          n INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
          steps INTEGER NOT NULL, steps_from_preset INTEGER NOT NULL,
          seeds TEXT NOT NULL, image_strength REAL, context_image TEXT,
          status TEXT NOT NULL, error TEXT, created_at REAL NOT NULL,
          started_at REAL, finished_at REAL
        );
        CREATE TABLE generation_images (
          generation_id TEXT NOT NULL, position INTEGER NOT NULL,
          filename TEXT NOT NULL, seed INTEGER NOT NULL,
          PRIMARY KEY (generation_id, position)
        );
        INSERT INTO sessions VALUES ('s1', 'old', 1.0, 1.0, NULL);
        INSERT INTO generations VALUES (
          'g1', 's1', 'g1', 'a fox', 'z-image-turbo', 'txt2img', 1, 64, 64, 4, 0,
          '[7]', NULL, NULL, 'completed', NULL, 1.0, 1.0, 1.0
        );
        """
    )
    database.close()

    store = PlaygroundStore(directory)
    try:
        entry = store.get_session("s1")["generations"][0]
        assert entry["prompt"] == "a fox"
        assert entry["negativePrompt"] is None
    finally:
        store.close()


# ── Deleting a group ───────────────────────────────────────────────────────


def test_deleting_a_group_removes_its_generations_and_files(client, settings):
    session_id = new_session(client)
    first = submit(client, session_id, prompt="a fox").json()
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    submit(client, session_id, prompt="a fox", group=first["groupId"])
    assert wait_until(lambda: len(generations(client, session_id)) == 2)
    assert wait_until(lambda: status_of(client, session_id, 1) == "completed")

    entries = generations(client, session_id)
    names = [image["url"].rsplit("/", 1)[-1] for entry in entries for image in entry["images"]]
    assert len(names) == 2

    assert client.delete(f"/playground/api/groups/{first['groupId']}").status_code == 204

    # The session is the transcript and survives; the entry does not.
    assert client.get(f"/playground/api/sessions/{session_id}").status_code == 200
    assert generations(client, session_id) == []
    assert [name for name in names if images_dir(settings).joinpath(name).exists()] == []


def test_deleting_a_group_removes_the_reference_image_its_root_was_made_with(client, settings):
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

    assert client.delete(f"/playground/api/groups/{accepted.json()['groupId']}").status_code == 204

    # A column, not a row: nothing else would ever unlink it.
    assert not reference.exists()


def test_deleting_a_group_cancels_the_members_it_had_not_run_yet(settings):
    """Unlike `dissolve_empty_group`, which refuses a group with work still to
    come: there the user deleted one image, here they deleted the entry."""
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as blocked:
        session_id = new_session(blocked)
        first = submit(blocked, session_id, prompt="a fox").json()
        assert wait_until(lambda: status_of(blocked, session_id) == "running")
        submit(blocked, session_id, prompt="a fox", group=first["groupId"])
        assert wait_until(lambda: len(generations(blocked, session_id)) == 2)

        assert blocked.delete(f"/playground/api/groups/{first['groupId']}").status_code == 204

        assert engine.cancel_requested is True
        assert generations(blocked, session_id) == []
        # And the worker does not go on producing images for an entry that is gone.
        engine.release = True
        assert not wait_until(lambda: generations(blocked, session_id), timeout=0.3)


def test_deleting_an_unknown_group_is_a_404(client):
    assert client.delete("/playground/api/groups/nope").status_code == 404


def test_deleting_a_group_of_a_locked_session_needs_its_token(client, settings):
    session_id = new_session(client)
    accepted = submit(client, session_id).json()
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    name = generations(client, session_id)[0]["images"][0]["url"].rsplit("/", 1)[-1]
    assert client.post(
        f"/playground/api/sessions/{session_id}/password", json={"password": "hunter2hunter2"}
    ).status_code == 200

    refused = client.delete(f"/playground/api/groups/{accepted['groupId']}")
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "session_locked"
    # Refused, not merely unreported: the entry's image is still on disk.
    assert images_dir(settings).joinpath(name).is_file()


# ── Pausing the queue ──────────────────────────────────────────────────────


def pause(client: TestClient, paused: bool) -> None:
    response = client.post("/playground/api/queue", json={"paused": paused})
    assert response.status_code == 200
    assert response.json()["paused"] is paused


def test_pausing_holds_the_queue(client, engine):
    """The witness that fails if the gate is a no-op.

    `FakeEngine` completes within a turn of the loop, so an unheld submission
    reaches `engine.jobs` immediately — the negative assertion is the whole test.
    """
    session_id = new_session(client)
    pause(client, True)
    assert submit(client, session_id).status_code == 202

    assert not wait_until(lambda: engine.jobs, timeout=0.3)
    assert status_of(client, session_id) == "queued"
    assert client.get("/playground/api/sessions").json()["paused"] is True

    pause(client, False)
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    assert len(engine.jobs) == 1


def test_resuming_runs_what_was_held_in_order(client, engine):
    """A held id has been taken off the queue already: this proves it comes back,
    and comes back in the order it was submitted."""
    session_id = new_session(client)
    pause(client, True)
    submit(client, session_id, prompt="first", seed=11)
    submit(client, session_id, prompt="second", seed=22)
    assert not wait_until(lambda: engine.jobs, timeout=0.3)

    pause(client, False)
    assert wait_until(lambda: len(engine.jobs) == 2)
    assert [job.prompt for job in engine.jobs] == ["first", "second"]


def test_pausing_does_not_stop_the_image_already_being_denoised(settings):
    """The decision the whole design rests on: the engine can only be
    interrupted by raising at a step, so a pause that stopped the current image
    would throw away work already paid for."""
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as blocked:
        session_id = new_session(blocked)
        submit(blocked, session_id)
        assert wait_until(lambda: status_of(blocked, session_id) == "running")

        pause(blocked, True)
        engine.release = True

        assert wait_until(lambda: status_of(blocked, session_id) == "completed")
        assert engine.cancel_requested is False
        assert len(generations(blocked, session_id)[0]["images"]) == 1


def test_pausing_stops_an_n_greater_than_one_run_between_its_images(settings):
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as blocked:
        session_id = new_session(blocked)
        submit(blocked, session_id, n=3)
        assert wait_until(lambda: status_of(blocked, session_id) == "running")

        pause(blocked, True)
        engine.release = True

        # The image in flight lands; the other two are held.
        assert wait_until(lambda: len(generations(blocked, session_id)[0]["images"]) == 1)
        assert not wait_until(
            lambda: len(generations(blocked, session_id)[0]["images"]) > 1, timeout=0.3
        )
        assert status_of(blocked, session_id) == "running"

        pause(blocked, False)
        assert wait_until(lambda: len(generations(blocked, session_id)[0]["images"]) == 3)
        assert status_of(blocked, session_id) == "completed"


def test_a_held_generation_can_still_be_cancelled(client, engine):
    session_id = new_session(client)
    pause(client, True)
    accepted = submit(client, session_id).json()

    cancelled = client.post(f"/playground/api/generations/{accepted['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    pause(client, False)
    assert not wait_until(lambda: engine.jobs, timeout=0.3)


def test_cancelling_a_generation_held_between_images_settles_it_at_once(settings):
    """Not on resume: the parked worker is the only thing that can settle the
    record, so the gate is woken and the response the caller reads is terminal."""
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as blocked:
        session_id = new_session(blocked)
        accepted = submit(blocked, session_id, n=3).json()
        assert wait_until(lambda: status_of(blocked, session_id) == "running")
        pause(blocked, True)
        engine.release = True
        assert wait_until(lambda: len(generations(blocked, session_id)[0]["images"]) == 1)

        cancelled = blocked.post(f"/playground/api/generations/{accepted['id']}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        # And the images it did produce are kept.
        assert len(generations(blocked, session_id)[0]["images"]) == 1


def test_deleting_a_session_while_paused_settles_its_generation(settings):
    """A session can be deleted out from under a held run, and nothing is left
    behind: no further images, no orphaned files.

    Note what this does *not* prove. `_stop_current` records the cancellation
    rather than only asking the engine — which is the honest expression of "the
    runner honours what the engine cannot" — but with the park sitting outside
    the idle context there is no observable difference here either way: the
    worker that is not woken simply lets go on the next resume, holding nothing
    in the meantime. The record is kept because it is correct and because it is
    what stops that from being true only by accident, not because this test
    catches its absence.
    """
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as blocked:
        session_id = new_session(blocked)
        submit(blocked, session_id, n=3)
        assert wait_until(lambda: status_of(blocked, session_id) == "running")
        pause(blocked, True)
        engine.release = True
        assert wait_until(lambda: len(generations(blocked, session_id)[0]["images"]) == 1)

        assert blocked.delete(f"/playground/api/sessions/{session_id}").status_code == 204
        assert blocked.get(f"/playground/api/sessions/{session_id}").status_code == 404

        # The worker lets go rather than generating the rest for a dead session.
        before = len(engine.jobs)
        assert not wait_until(lambda: len(engine.jobs) > before, timeout=0.3)
        assert list(images_dir(settings).iterdir()) == []


def test_the_pause_does_not_survive_a_restart(settings):
    """Stated, not assumed: the queue is in memory, so a held backlog is failed
    by `mark_interrupted()` like anything else caught mid-flight."""
    engine = FakeEngine()
    with make_client(create_app(settings, engine)) as first:
        session_id = new_session(first)
        pause(first, True)
        submit(first, session_id)
        assert not wait_until(lambda: engine.jobs, timeout=0.3)

    with make_client(create_app(settings, FakeEngine())) as second:
        assert second.get("/playground/api/sessions").json()["paused"] is False
        entry = generations(second, session_id)[0]
        assert entry["status"] == "failed"
        assert entry["error"] == "Interrupted by server restart"


def test_shutting_down_while_paused_does_not_hang(settings):
    """The witness is termination. A lost wakeup would hang the suite rather
    than fail an assertion, so the bound is the assertion."""
    started = time.monotonic()
    engine = FakeEngine()
    with make_client(create_app(settings, engine)) as blocked:
        session_id = new_session(blocked)
        pause(blocked, True)
        submit(blocked, session_id)
        assert not wait_until(lambda: engine.jobs, timeout=0.2)
    assert time.monotonic() - started < 5


def _paused_client(settings, engine, **overrides):
    """An app whose settings differ from the shared fixture's."""
    raw = settings.model_dump(mode="json")
    raw["server"].update(overrides)
    return make_client(create_app(Settings.model_validate(raw), engine))


def test_a_held_queue_lets_the_model_be_released(settings):
    """The reason the park sits *outside* `with self._idle:`.

    `IdleUnloader.__enter__` destroys the pending countdown and only `__exit__`
    recreates it, so parking inside the context would hold the weights resident
    for the whole pause — the exact failure the idle policy exists to prevent.

    The run has to be genuinely *held between images* for this to mean anything,
    which is why it takes the blocking double: with an engine that completes
    within a turn of the loop the run is over before the pause lands, the context
    is left for the ordinary reason, and the test would pass either way.
    """
    engine = BlockingEngine()
    with _paused_client(settings, engine, idle_unload_s=0) as client:
        session_id = new_session(client)
        submit(client, session_id, n=3)
        assert wait_until(lambda: status_of(client, session_id) == "running")
        pause(client, True)
        engine.release = True

        # One image lands, then the worker parks with two seeds still to run.
        assert wait_until(lambda: len(generations(client, session_id)[0]["images"]) == 1)
        assert status_of(client, session_id) == "running"

        # Held, and the machine gets its memory back anyway.
        assert wait_until(lambda: engine.unload_count >= 1)


def test_a_v1_request_still_releases_the_model_while_the_queue_is_paused(settings):
    """The unloader is one instance shared with the `/v1` plane, and it re-arms
    only on the way down to zero in flight. A playground worker parked inside it
    would disable automatic release for the whole server, `/v1` included — so
    this asks `/v1` the question while a playground run is held mid-flight."""
    engine = BlockingEngine()
    with _paused_client(settings, engine, idle_unload_s=0) as client:
        session_id = new_session(client)
        submit(client, session_id, n=3)
        assert wait_until(lambda: status_of(client, session_id) == "running")
        pause(client, True)
        engine.release = True
        assert wait_until(lambda: len(generations(client, session_id)[0]["images"]) == 1)
        assert status_of(client, session_id) == "running"

        before = engine.unload_count
        assert client.post(
            "/v1/images/generations",
            json={"prompt": "a fox", "model": "z-image-turbo", "response_format": "b64_json"},
        ).status_code == 200
        assert wait_until(lambda: engine.unload_count > before)


# ── Upscaling ──────────────────────────────────────────────────────────────


def generated_image_name(client: TestClient, session_id: str) -> str:
    """Submit one generation, wait for it, and return its image's filename."""
    assert submit(client, session_id).status_code == 202
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    return generations(client, session_id)[0]["images"][0]["url"].rsplit("/", 1)[-1]


def upscale(client: TestClient, session_id: str, image: str, **fields):
    body = {"image": image, "model": "realesrgan-x4plus", "scale": 2, **fields}
    headers = body.pop("headers", None)
    return client.post(
        f"/playground/api/sessions/{session_id}/upscales", json=body, headers=headers
    )


def test_the_catalogue_of_upscalers_is_published(client):
    listed = client.get("/playground/api/upscalers")
    assert listed.status_code == 200
    entries = listed.json()["upscalers"]
    assert [entry["id"] for entry in entries] == [
        "realesrgan-x4plus",
        "realesrgan-x4plus-anime",
    ]
    for entry in entries:
        assert entry["scales"] == [2, 4]
        assert isinstance(entry["downloaded"], bool)
        assert entry["sizeMb"] > 0
        assert "BSD-3-Clause" in entry["license"]


def test_an_upscale_is_recorded_and_runs(client, engine, settings):
    session_id = new_session(client)
    name = generated_image_name(client, session_id)
    source = generations(client, session_id)[0]

    accepted = upscale(client, session_id, name, scale=4)
    assert accepted.status_code == 202
    record = accepted.json()
    assert record["kind"] == "upscale"
    assert record["n"] == 1
    assert record["steps"] == 0
    assert record["model"] == "realesrgan-x4plus"
    # The row records the *output* size, which is what the runner hands the
    # engine -- no factor is derived back out of it anywhere.
    # The fake source is 2x2, so x4 is 8x8.
    assert record["size"] == "8x8"
    # It joins the entry its source came from rather than starting a new one.
    assert record["groupId"] == source["groupId"]
    assert record["prompt"] == source["prompt"]
    assert record["seeds"] == [source["images"][0]["seed"]]

    assert wait_until(lambda: status_of(client, session_id, 1) == "completed")
    done = generations(client, session_id)[1]
    assert len(done["images"]) == 1
    assert client.get(done["images"][0]["url"]).status_code == 200
    # The engine was asked for an upscale, not a generation.
    assert len(engine.upscales) == 1
    assert engine.upscales[0].target == (8, 8)


def test_an_upscale_copies_its_source_rather_than_referencing_it(client, settings):
    """Deleting the source must not leave the upscale pointing at nothing."""
    session_id = new_session(client)
    name = generated_image_name(client, session_id)
    accepted = upscale(client, session_id, name)
    assert accepted.status_code == 202
    assert wait_until(lambda: status_of(client, session_id, 1) == "completed")

    context = generations(client, session_id)[1]["contextImage"]
    assert context is not None
    copy_name = context.rsplit("/", 1)[-1]
    assert copy_name != name, "the source was referenced, not copied"
    assert images_dir(settings).joinpath(copy_name).is_file()

    assert client.delete(f"/playground/api/images/{name}").status_code == 204
    upscaled = generations(client, session_id)[1]
    assert client.get(upscaled["images"][0]["url"]).status_code == 200
    assert client.get(upscaled["contextImage"]).status_code == 200


def test_deleting_the_group_unlinks_the_copy(client, settings):
    session_id = new_session(client)
    name = generated_image_name(client, session_id)
    assert upscale(client, session_id, name).status_code == 202
    assert wait_until(lambda: status_of(client, session_id, 1) == "completed")

    copy_name = generations(client, session_id)[1]["contextImage"].rsplit("/", 1)[-1]
    group_id = generations(client, session_id)[1]["groupId"]
    assert client.delete(f"/playground/api/groups/{group_id}").status_code == 204
    # The assertion is on the directory, not the return value: what matters is
    # that a directory which is never purged did not keep the file.
    assert not images_dir(settings).joinpath(copy_name).exists()


# ── Upscaling: authority ───────────────────────────────────────────────────


def test_an_image_from_another_session_is_not_upscalable(client):
    owner = new_session(client)
    name = generated_image_name(client, owner)
    intruder = new_session(client)
    refused = upscale(client, intruder, name)
    assert refused.status_code == 404
    # "unknown", not "not yours": existence must not depend on who asks.
    assert refused.json()["error"]["code"] == "not_found"


def test_an_invented_filename_is_a_404(client):
    session_id = new_session(client)
    assert upscale(client, session_id, "../../etc/passwd").status_code == 404
    assert upscale(client, session_id, "nope.png").status_code == 404


def test_a_context_image_is_not_upscalable(client, settings):
    """`session_of_image` would accept one; `generated_image` must not.

    A context file keeps the suffix it was uploaded with, so it need not be a
    PNG, and it has no seed -- while `generation_images.seed` is NOT NULL.
    """
    session_id = new_session(client)
    assert submit(
        client, session_id, model="qwen-image-2512", files={"image": ("ref.png", tiny_png())}
    ).status_code == 202
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    context = generations(client, session_id)[0]["contextImage"]
    assert context is not None

    refused = upscale(client, session_id, context.rsplit("/", 1)[-1])
    assert refused.status_code == 404


def test_an_unknown_upscaler_names_the_valid_keys(client):
    session_id = new_session(client)
    name = generated_image_name(client, session_id)
    refused = upscale(client, session_id, name, model="realesrgan-x9000")
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_model"
    assert "realesrgan-x4plus" in refused.json()["error"]["message"]


def test_an_unsupported_scale_is_refused(client):
    session_id = new_session(client)
    name = generated_image_name(client, session_id)
    refused = upscale(client, session_id, name, scale=3)
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_scale"


def test_an_oversized_job_is_refused_and_leaves_no_copy(client, settings, monkeypatch):
    """The guard between one click and a multi-gigabyte allocation.

    The directory assertion is the one that matters: it is what proves the
    cleanup protocol ran, in a directory nothing ever purges.
    """
    from qds.upscale import catalogue as upscale_catalogue

    monkeypatch.setattr(upscale_catalogue, "MAX_RENDER_PIXELS", 4)
    session_id = new_session(client)
    name = generated_image_name(client, session_id)
    before = set(p.name for p in images_dir(settings).iterdir())

    refused = upscale(client, session_id, name, scale=4)
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "image_too_large"
    assert set(p.name for p in images_dir(settings).iterdir()) == before


def test_the_size_guard_measures_what_is_rendered_not_what_is_asked_for(
    client, settings, monkeypatch
):
    """The hole a target-based guard leaves open.

    RRDBNet always renders at x4, so a x2 request does the same work as a x4 one
    and then throws three quarters of it away. A guard on the *target* therefore
    lets a x2 job through that is four times heavier than the x4 job it refuses
    on the same source -- and it is reachable, because an upscale's output is a
    generated image and can be upscaled again.

    The source here is 2x2, so either factor renders 8x8 -- 64 pixels. With the
    limit one pixel below that, *both* requests must be refused, because both
    do the same work. A target-based guard would refuse only the x4.
    """
    from qds.upscale import catalogue as upscale_catalogue

    monkeypatch.setattr(upscale_catalogue, "MAX_RENDER_PIXELS", 63)
    session_id = new_session(client)
    name = generated_image_name(client, session_id)

    for scale in (4, 2):
        refused = upscale(client, session_id, name, scale=scale)
        assert refused.status_code == 400, f"x{scale} slipped past the guard"
        assert refused.json()["error"]["code"] == "image_too_large"
        # The rendered size, named, so the limit does not look arbitrary.
        assert "8x8" in refused.json()["error"]["message"]

    # One pixel of headroom and both are allowed, for the same reason.
    monkeypatch.setattr(upscale_catalogue, "MAX_RENDER_PIXELS", 64)
    assert upscale(client, session_id, name, scale=2).status_code == 202
    assert upscale(client, session_id, name, scale=4).status_code == 202


def test_upscaling_a_locked_session_needs_the_token(client):
    session_id = new_session(client)
    name = generated_image_name(client, session_id)
    password = "correct horse battery"
    assert client.post(
        f"/playground/api/sessions/{session_id}/password", json={"password": password}
    ).status_code in (200, 204)

    refused = upscale(client, session_id, name)
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "session_locked"

    token = client.post(
        f"/playground/api/sessions/{session_id}/unlock", json={"password": password}
    ).json()["token"]
    accepted = upscale(client, session_id, name, headers={"X-QDS-Session-Token": token})
    assert accepted.status_code == 202


def test_an_upscale_needs_no_schema_change(tmp_path):
    """I9: a store written before upscales existed takes them without migrating."""
    store = PlaygroundStore(tmp_path)
    before = {row[1] for row in store._db.execute("PRAGMA table_info(generations)")}
    store.create_session()
    sessions = store.list_sessions()
    store.add_generation(
        sessions[0]["id"],
        prompt="a fox",
        model="realesrgan-x4plus",
        kind="upscale",
        n=1,
        width=2048,
        height=2048,
        steps=0,
        steps_from_preset=False,
        seeds=[7],
        context_image="ctx-abc.png",
    )
    after = {row[1] for row in store._db.execute("PRAGMA table_info(generations)")}
    assert after == before, "an upscale row required a new column"

    reopened = PlaygroundStore(tmp_path)
    record = reopened.get_session(sessions[0]["id"])["generations"][0]
    assert record["kind"] == "upscale"
    assert record["size"] == "2048x2048"


def test_an_upscale_in_flight_can_be_cancelled(settings):
    """Cancelled, not failed, and with nothing half-written on disk."""
    engine = BlockingEngine()
    with make_client(create_app(settings, engine)) as client:
        session_id = new_session(client)
        # The source generation must finish; only the upscale is held.
        engine.release = True
        assert submit(client, session_id).status_code == 202
        assert wait_until(lambda: status_of(client, session_id) == "completed")
        name = generations(client, session_id)[0]["images"][0]["url"].rsplit("/", 1)[-1]
        engine.release = False

        record = upscale(client, session_id, name).json()
        assert wait_until(lambda: status_of(client, session_id, 1) == "running")
        assert client.post(
            f"/playground/api/generations/{record['id']}/cancel"
        ).status_code == 200
        assert wait_until(lambda: status_of(client, session_id, 1) == "cancelled")
        assert generations(client, session_id)[1]["images"] == []


def test_a_vanished_source_fails_with_an_answer_about_the_source(client, settings):
    """A real failure path, and it used to answer about model weights.

    `Image.open` on a missing file raises `FileNotFoundError`, which
    `translate_mflux_exception` renders as "Weights or model not found". The
    row reached a terminal status either way; the message was about the wrong
    thing.
    """
    session_id = new_session(client)
    name = generated_image_name(client, session_id)

    # Held, so the file is gone before the runner ever claims the row: without
    # this the upscale usually wins the race and the test proves nothing.
    assert client.post("/playground/api/queue", json={"paused": True}).status_code == 200
    accepted = upscale(client, session_id, name)
    assert accepted.status_code == 202
    images_dir(settings).joinpath(accepted.json()["contextImage"].rsplit("/", 1)[-1]).unlink()
    assert client.post("/playground/api/queue", json={"paused": False}).status_code == 200

    assert wait_until(lambda: status_of(client, session_id, 1) == "failed")
    error = generations(client, session_id)[1]["error"]
    assert "source image" in error
    assert "Weights" not in error, f"answered about the wrong thing: {error}"


# ── Thumbnails ─────────────────────────────────────────────────────────────
#
# A derived artifact, and every test here is about that word: it is produced on
# demand from the stored file, it is bounded, it dies with what it was derived
# from, and its absence costs quality rather than the picture.


def thumbs_dir(settings) -> Path:
    return Path(settings.server.playground_store, "thumbnails")


def thumb_names(settings) -> set[str]:
    return {path.name for path in thumbs_dir(settings).iterdir()}


def finished_image_url(client: TestClient, session_id: str, **fields) -> str:
    submit(client, session_id, **fields)
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    return generations(client, session_id)[0]["images"][0]["url"]


def test_a_thumbnail_is_derived_on_first_request_and_then_reused(client, settings):
    session_id = new_session(client)
    url = finished_image_url(client, session_id)
    name = url.rsplit("/", 1)[-1]
    assert thumb_names(settings) == set(), "nothing is derived until it is asked for"

    response = client.get(f"{url}/thumb")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.headers["cache-control"] == "private, max-age=86400"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert Image.open(io.BytesIO(response.content)).format == "WEBP"

    assert thumb_names(settings) == {f"{name}.webp"}
    derived = thumbs_dir(settings) / f"{name}.webp"
    stamp = derived.stat().st_mtime_ns

    again = client.get(f"{url}/thumb")
    assert again.status_code == 200
    assert again.content == response.content
    assert derived.stat().st_mtime_ns == stamp, "the second request re-derived it"


def test_a_thumbnail_is_bounded_to_its_longest_edge(client, settings):
    """The whole point of the route: what it serves is small.

    The stored file is replaced with a large one *behind* its row, which is the
    only way to get a realistic source out of a fake engine. The row is
    untouched, so this exercises exactly the path a real 5120x2880 upscale takes.
    """
    session_id = new_session(client)
    url = finished_image_url(client, session_id)
    name = url.rsplit("/", 1)[-1]
    buffer = io.BytesIO()
    Image.new("RGB", (2048, 1024), "green").save(buffer, format="PNG")
    images_dir(settings).joinpath(name).write_bytes(buffer.getvalue())

    response = client.get(f"{url}/thumb")
    assert response.status_code == 200
    thumbnail = Image.open(io.BytesIO(response.content))
    # Aspect ratio preserved, longest edge at the bound, not merely "smaller".
    assert thumbnail.size == (playground_module.THUMBNAIL_EDGE, playground_module.THUMBNAIL_EDGE // 2)
    assert len(response.content) < len(buffer.getvalue()) / 10


def test_an_image_that_cannot_be_thumbnailed_is_served_whole(client, settings):
    """T4's second half: a missing thumbnail degrades to the full image.

    Witnessed on bytes that are not a picture, because that is the failure a
    derivation can hit at request time and cannot be retried out of. The caller
    was already authorized for these bytes -- the derived copy was an
    optimisation -- so the answer is the file, not a 500 and not a broken tile.
    """
    session_id = new_session(client)
    url = finished_image_url(client, session_id)
    name = url.rsplit("/", 1)[-1]
    images_dir(settings).joinpath(name).write_bytes(b"not an image at all")

    response = client.get(f"{url}/thumb")
    assert response.status_code == 200
    assert response.content == b"not an image at all"
    assert response.headers["content-type"] == "image/png"
    # The fallback is the full-resolution file, so it keeps that file's régime.
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert thumb_names(settings) == set(), "a failed derivation left a file behind"


def test_a_thumbnail_of_a_vanished_file_is_a_404(client, settings):
    """A row without its file is a 404, as it is on the image route: there is
    nothing to fall back to, and answering 200 with nothing would be the broken
    tile the fallback exists to avoid."""
    session_id = new_session(client)
    url = finished_image_url(client, session_id)
    images_dir(settings).joinpath(url.rsplit("/", 1)[-1]).unlink()
    assert client.get(f"{url}/thumb").status_code == 404


def test_deleting_an_image_reclaims_its_thumbnail(client, settings):
    session_id = new_session(client)
    submit(client, session_id, prompt="a fox", n=2)
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    urls = [image["url"] for image in generations(client, session_id)[0]["images"]]
    names = [url.rsplit("/", 1)[-1] for url in urls]
    for url in urls:
        assert client.get(f"{url}/thumb").status_code == 200
    assert thumb_names(settings) == {f"{name}.webp" for name in names}

    assert client.delete(f"/playground/api/images/{names[0]}").status_code == 204

    assert thumb_names(settings) == {f"{names[1]}.webp"}, "the thumbnail outlived its image"
    assert client.get(f"{urls[0]}/thumb").status_code == 404
    assert client.get(f"{urls[1]}/thumb").status_code == 200


def test_deleting_a_session_reclaims_the_thumbnails_of_its_images(client, settings):
    session_id = new_session(client)
    submit(client, session_id, prompt="a fox", n=2, files={"image": ("ctx.png", tiny_png(), "image/png")})
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    entry = generations(client, session_id)[0]
    urls = [image["url"] for image in entry["images"]] + [entry["contextImage"]]
    for url in urls:
        assert client.get(f"{url}/thumb").status_code == 200
    assert len(thumb_names(settings)) == 3, "the context image is a stored file too"

    assert client.delete(f"/playground/api/sessions/{session_id}").status_code == 204

    assert thumb_names(settings) == set()
    for url in urls:
        assert client.get(f"{url}/thumb").status_code == 404


def test_deleting_a_group_reclaims_the_thumbnails_of_its_images(client, settings):
    session_id = new_session(client)
    first = submit(client, session_id, prompt="a fox").json()
    assert wait_until(lambda: status_of(client, session_id) == "completed")
    submit(client, session_id, prompt="a fox", group=first["groupId"])
    assert wait_until(lambda: len(generations(client, session_id)) == 2)
    assert wait_until(lambda: status_of(client, session_id, 1) == "completed")
    urls = [
        image["url"] for entry in generations(client, session_id) for image in entry["images"]
    ]
    assert len(urls) == 2
    for url in urls:
        assert client.get(f"{url}/thumb").status_code == 200
    assert len(thumb_names(settings)) == 2

    assert client.delete(f"/playground/api/groups/{first['groupId']}").status_code == 204

    assert thumb_names(settings) == set()
