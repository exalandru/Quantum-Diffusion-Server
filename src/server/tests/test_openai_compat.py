"""Conformance of the HTTP layer to what an OpenAI client expects."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qds.app import create_app, progress_events
from qds.settings import Settings, load_settings
from tests.conftest import make_client, tiny_png, wait_until


def generate(client: TestClient, **body):
    return client.post("/v1/images/generations", json={"prompt": "un renard", **body})


# ── /v1/models ─────────────────────────────────────────────────────────────


def test_model_list_is_conformant(client):
    payload = client.get("/v1/models").json()
    assert payload["object"] == "list"
    # Every catalogue entry, since the fixture disables none of them.
    assert {entry["id"] for entry in payload["data"]} == {
        "ernie-image",
        "ernie-image-turbo",
        "fibo",
        "fibo-lite",
        "flux2-dev",
        "flux2-klein",
        "ideogram-4",
        "qwen-image-2512",
        "z-image",
        "z-image-turbo",
    }
    for entry in payload["data"]:
        assert entry["object"] == "model"
        assert isinstance(entry["created"], int)
        assert entry["owned_by"] == "mflux"


def test_model_list_carries_a_readable_name(client):
    """The id is what a request sends; the name is what an interface shows.

    Without this, every client has to invent something readable from an API
    identifier, and they all invent something different.
    """
    rows = {entry["id"]: entry for entry in client.get("/v1/models").json()["data"]}
    assert rows["qwen-image-2512"]["display_name"] == "Qwen"
    assert rows["z-image-turbo"]["display_name"] == "Z-Image Turbo"
    for name, entry in rows.items():
        assert entry["display_name"], f"{name} publishes no readable name"


def test_retrieving_a_model(client):
    payload = client.get("/v1/models/z-image-turbo").json()
    assert payload["id"] == "z-image-turbo"
    assert payload["mflux"]["default_size"] == "1280x720"
    assert payload["mflux"]["supports_guidance"] is False


def test_unknown_model_returns_an_openai_error(client):
    response = client.get("/v1/models/sdxl")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "model"
    assert error["code"] == "model_not_found"
    assert "sdxl" in error["message"]


# ── response_format ────────────────────────────────────────────────────────


def test_url_is_the_default_and_is_served(client):
    """Prototype bug #1: `url` was silently ignored."""
    payload = generate(client).json()
    url = payload["data"][0]["url"]
    assert url.endswith(".png")
    assert "b64_json" not in payload["data"][0]

    served = client.get(url)
    assert served.status_code == 200
    assert served.content == tiny_png()


def test_b64_json(client):
    payload = generate(client, response_format="b64_json").json()
    assert base64.b64decode(payload["data"][0]["b64_json"]) == tiny_png()


def test_raw_returns_the_png(client):
    response = generate(client, response_format="raw")
    assert response.headers["content-type"] == "image/png"
    assert response.content == tiny_png()


def test_raw_with_n_above_one_is_rejected(client):
    # The prototype silently fell back to b64.
    response = generate(client, response_format="raw", n=2)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "response_format"


def test_unknown_response_format(client):
    response = generate(client, response_format="jpeg")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_response_format"


# ── Parameters ─────────────────────────────────────────────────────────────


def test_model_default_size(client):
    payload = generate(client, size="auto").json()
    assert payload["mflux"]["size"] == "1920x1072"


def test_size_rounded_to_multiple_of_16_and_reported(client):
    payload = generate(client, size="1920x1080").json()
    assert payload["mflux"]["size"] == "1920x1072"


def test_invalid_size(client):
    response = generate(client, size="grand")
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "size"


def test_unsupported_openai_parameters_are_ignored(client):
    response = generate(client, quality="hd", style="vivid", user="corin", background="opaque")
    assert response.status_code == 200


def test_n_is_bounded_by_the_config(client, engine):
    assert generate(client, n=4).status_code == 200
    assert len(engine.jobs) == 4

    response = generate(client, n=5)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "n_too_large"


def test_seed_increments_per_image(client, engine):
    generate(client, n=3, seed=100)
    assert [job.seed for job in engine.jobs] == [100, 101, 102]


def test_missing_seed_is_random_but_stays_consecutive(client, engine):
    generate(client, n=2)
    first, second = (job.seed for job in engine.jobs)
    assert second == first + 1


def test_model_default_steps(client, engine):
    generate(client, model="flux2-klein")
    assert engine.jobs[0].steps == 4  # not the prototype's 20
    generate(client, model="z-image-turbo")
    assert engine.jobs[1].steps == 9


# ── Model capabilities ─────────────────────────────────────────────────────


def test_negative_prompt_rejected_on_flux2_klein(client):
    """Prototype bug #2: a systematic 500 on the default model."""
    response = generate(client, negative_prompt="flou")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "negative_prompt"
    assert error["code"] == "unsupported_parameter"


def test_negative_prompt_accepted_on_qwen(client, engine):
    assert generate(client, model="qwen-image-2512", negative_prompt="flou").status_code == 200
    assert engine.jobs[0].negative_prompt == "flou"


def test_guidance_rejected_on_a_distilled_model(client):
    response = generate(client, guidance=3.5)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "guidance"


def test_guidance_accepted_on_z_image(client, engine):
    assert generate(client, model="z-image", guidance=5.0).status_code == 200
    assert engine.jobs[0].guidance == 5.0


# ── /v1/images/edits ───────────────────────────────────────────────────────


def edit(client: TestClient, **data):
    return client.post(
        "/v1/images/edits",
        data={"prompt": "ajoute un chapeau", **data},
        files={"image": ("source.png", tiny_png("blue"), "image/png")},
    )


def test_edits_uses_the_edit_variant_when_available(client, engine):
    assert edit(client).status_code == 200
    job = engine.jobs[0]
    assert job.kind == "edit"
    assert job.image_strength is None


def test_edits_switches_to_img2img_when_strength_is_given(client, engine):
    assert edit(client, strength=0.7).status_code == 200
    job = engine.jobs[0]
    assert job.kind == "txt2img"
    assert job.image_strength == 0.7


def test_edits_falls_back_to_img2img_without_an_edit_variant(client, engine):
    assert edit(client, model="z-image").status_code == 200
    job = engine.jobs[0]
    assert job.kind == "txt2img"
    assert job.image_strength == 0.4


def test_mask_is_explicitly_rejected(client):
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "remove the background"},
        files={
            "image": ("source.png", tiny_png(), "image/png"),
            "mask": ("mask.png", tiny_png("black"), "image/png"),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "mask"


def test_empty_image_is_rejected(client):
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "x"},
        files={"image": ("vide.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_oversized_upload_is_rejected(tmp_path, engine):
    settings = Settings.model_validate(
        {
            "server": {
                "image_store": str(tmp_path / "images"),
                "log_file": None,
                "max_upload_mb": 0.001,  # ~1 Ko
            }
        }
    )
    with make_client(create_app(settings, engine)) as client:
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "x"},
            files={"image": ("gros.png", b"0" * 5000, "image/png")},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


# ── Transport : CORS, auth, health ─────────────────────────────────────────


def test_cors_preflight_is_allowed(client):
    response = client.options(
        "/v1/images/generations",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def secured_app(tmp_path, engine, api_key: str):
    settings = Settings.model_validate(
        {
            "server": {
                "image_store": str(tmp_path / "images"),
                "log_file": None,
                "api_key": api_key,
            },
            # Pinned rather than inherited from the code default, like the main
            # `client` fixture: these tests assert on the value, so they should
            # own it.
            "default_model": "flux2-klein",
        }
    )
    return create_app(settings, engine)


@pytest.fixture
def secured_client(tmp_path, engine):
    with make_client(secured_app(tmp_path, engine, "cle-secrete")) as test_client:
        yield test_client


def test_auth_required_when_a_key_is_configured(secured_client):
    assert secured_client.get("/v1/models").status_code == 401
    assert secured_client.get("/v1/models", headers={"Authorization": "Bearer mauvaise"}).status_code == 401
    assert secured_client.get("/v1/models", headers={"Authorization": "Basic cle-secrete"}).status_code == 401
    assert (
        secured_client.get("/v1/models", headers={"Authorization": "Bearer cle-secrete"}).status_code == 200
    )


def test_non_ascii_key_is_rejected_without_crashing(tmp_path, engine):
    """Such a key is unreachable (HTTP headers are latin-1), but it must not
    surface a 500 out of compare_digest.

    The non-ASCII key is the fixture, not an oversight: `secrets.compare_digest`
    rejects non-ASCII `str` outright, which is why the comparison is done on
    bytes.
    """
    with make_client(secured_app(tmp_path, engine, "secret-\u65e5\u672c")) as test_client:
        assert test_client.get("/v1/models", headers={"Authorization": "Bearer x"}).status_code == 401


def test_health_stays_public(secured_client):
    payload = secured_client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["default_model"] == "flux2-klein"
    assert payload["loaded_model"] is None


def test_health_exposes_the_warm_model(client):
    generate(client, model="z-image")
    assert client.get("/health").json()["loaded_model"] == "z-image:txt2img"


def test_capabilities(client):
    payload = client.get("/v1/capabilities").json()
    assert payload["max_n"] == 4
    assert payload["models"]["flux2-klein"]["supports_edit"] is True
    assert payload["models"]["z-image"]["supports_edit"] is False


def test_empty_prompt_returns_a_formatted_error(client):
    response = client.post("/v1/images/generations", json={"prompt": ""})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["param"] == "prompt"


def test_shutdown_releases_the_engine(settings, engine):
    with make_client(create_app(settings, engine)):
        pass
    assert engine.shutdown_called is True


# ── Prompt formats and dimension bounds ────────────────────────────────────


def test_a_plain_prompt_is_refused_before_the_model_loads(client, engine):
    """FIBO's prompt encoder opens with a bare `json.loads(prompt)`.

    Plain text would raise a JSONDecodeError *after* several GB had been loaded, and
    reach the client as "Expecting value: line 1 column 1". The empty `engine.jobs`
    is the whole point of this test: the refusal has to come first.
    """
    response = generate(client, model="fibo-lite")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "prompt"
    assert error["code"] == "prompt_must_be_json"
    assert engine.jobs == [], "the model was loaded before the prompt was checked"


def test_a_json_caption_is_accepted(client, engine):
    caption = json.dumps({"high_level_description": "a red fox in the snow"})
    response = client.post("/v1/images/generations", json={"prompt": caption, "model": "fibo-lite"})
    assert response.status_code == 200
    assert engine.jobs[0].prompt == caption


def test_a_json_array_is_not_a_caption(client):
    response = client.post("/v1/images/generations", json={"prompt": "[1, 2]", "model": "fibo"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "prompt_must_be_json"


def test_a_text_model_accepts_a_json_looking_prompt(client, engine):
    # Accepting text means accepting anything: a JSON string is text too. No model
    # should reject a prompt for being *more* structured than it needs.
    response = generate(client, model="z-image-turbo", prompt='{"not": "a caption"}')
    assert response.status_code == 200
    assert engine.jobs[0].prompt == '{"not": "a caption"}'


def test_a_size_beyond_the_model_bound_is_refused_before_loading(client, engine):
    # Ideogram 4 caps at 2048. Its weights are 28 GB: finding out afterwards is not
    # an option.
    response = generate(client, model="ideogram-4", size="2560x1440")
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "size"
    assert engine.jobs == []


def test_the_bound_also_covers_the_model_default(client, engine):
    """A config-wide `default_size` out of range must not sail through.

    The check runs on the resolved size, not just on an explicit one — otherwise a
    request with no `size` would reach mflux and fail there.
    """
    settings = Settings.model_validate(
        {
            "server": {"log_file": None, "progress_log_every": 0},
            "default_size": "2560x1440",
        }
    )
    with make_client(create_app(settings, engine)) as scoped:
        assert generate(scoped, model="ideogram-4").status_code == 400
        # Every other model is happy with it.
        assert generate(scoped, model="z-image-turbo").status_code == 200


def test_capabilities_publish_the_new_flags(client):
    models = client.get("/v1/capabilities").json()["models"]
    assert models["fibo"]["prompt_formats"] == ["json"]
    assert models["ideogram-4"]["prompt_formats"] == ["text", "json"]
    assert models["z-image-turbo"]["prompt_formats"] == ["text"]
    assert models["ideogram-4"]["preset"] == "V4_DEFAULT_20"
    assert models["ideogram-4"]["max_dimension"] == 2048
    assert models["z-image-turbo"]["max_dimension"] is None
    assert models["flux2-klein"]["gated"] is True
    assert models["z-image-turbo"]["gated"] is False
    assert models["z-image-turbo"]["license"] == "Apache-2.0"
    assert models["ideogram-4"]["supports_quantization"] is False
    assert models["ideogram-4"]["quantize_choices"] == []


# ── Automatic release of the warm model ────────────────────────────────────


def _client_with(settings_kwargs: dict, engine) -> TestClient:
    settings = Settings.model_validate(
        {"server": {"log_file": None, "progress_log_every": 0, **settings_kwargs}}
    )
    return make_client(create_app(settings, engine))


def test_a_zero_delay_releases_once_per_request_not_per_image(engine, tmp_path):
    """The reason the countdown is armed per request.

    Armed inside the engine, which locks per image, an n=2 request would release
    the model between the two — and reload it for the second.
    """
    with _client_with({"image_store": str(tmp_path / "i"), "idle_unload_s": 0}, engine) as client:
        generate(client, n=2)
        assert wait_until(lambda: engine.unload_count >= 1), "never released"
        # Give a second release the chance to show up before ruling it out.
        time.sleep(0.05)
        assert engine.unload_count == 1
        assert client.get("/health").json()["loaded_model"] is None


def test_by_default_the_model_stays_warm(client, engine):
    generate(client)
    time.sleep(0.05)
    assert engine.unload_count == 0
    assert engine.loaded_model is not None


def test_health_reports_the_release_policy(engine, tmp_path):
    with _client_with({"image_store": str(tmp_path / "i"), "idle_unload_s": 30}, engine) as client:
        assert client.get("/health").json()["idle_unload_s"] == 30


def test_the_default_policy_is_never(client):
    assert client.get("/health").json()["idle_unload_s"] is None


@pytest.mark.parametrize(("raw", "expected"), [("", None), ("0", 0.0), ("45", 45.0)])
def test_the_delay_is_settable_from_the_environment(monkeypatch, raw, expected):
    # An empty variable means "never", the same convention as log_file: without
    # it there would be no way to turn the setting back off from the environment.
    monkeypatch.setenv("QDS_SERVER_IDLE_UNLOAD_S", raw)
    assert load_settings(Path("/nonexistent")).server.idle_unload_s == expected


# ── Progress, cancellation, unloading ──────────────────────────────────────


async def _take(generator, count: int) -> list[str]:
    frames = []
    try:
        async for frame in generator:
            frames.append(frame)
            if len(frames) == count:
                break
    finally:
        await generator.aclose()
    return frames


def test_progress_emits_one_sse_frame_then_goes_quiet(engine):
    # Tested on the generator, not over HTTP: the stream is infinite by design
    # and `TestClient` does not propagate the disconnect, so reading it through
    # that would block forever.
    engine.busy = True
    frames = asyncio.run(_take(progress_events(engine, poll_s=0, ping_s=0.05), 2))

    assert frames[0].startswith("data: ") and frames[0].endswith("\n\n")
    payload = json.loads(frames[0].removeprefix("data: ").rstrip())
    assert payload["state"] == "generating"
    assert (payload["step"], payload["total"]) == (3, 9)
    assert payload["memory"]["active_gb"] == 0.0

    # The state has not changed meanwhile: the second frame is a heartbeat, not
    # a repeat of the snapshot.
    assert frames[1] == ": ping\n\n"


def test_progress_re_emits_when_state_changes(engine):
    async def scenario():
        generator = progress_events(engine, poll_s=0, ping_s=3600)
        try:
            first = await anext(generator)
            engine.busy = True
            second = await anext(generator)
            return first, second
        finally:
            await generator.aclose()

    first, second = asyncio.run(scenario())
    assert json.loads(first.removeprefix("data: ").rstrip())["state"] == "idle"
    assert json.loads(second.removeprefix("data: ").rstrip())["state"] == "generating"


#: No HTTP test of an established SSE stream: `TestClient` blocks on leaving
#: `with client.stream(...)`, even without reading the body, because it waits for
#: a generator that is infinite by design to finish. The frames and the heartbeat
#: are covered above on `progress_events`, the route wiring by the auth test
#: below, and the MIME type is checked by hand:
#: `curl -N -i http://127.0.0.1:8765/v1/progress`.


def test_cancel_on_an_idle_server_does_nothing(client, engine):
    payload = client.post("/v1/cancel").json()
    assert payload == {"cancelled": False, "state": "idle"}
    assert engine.cancel_requested is False


def test_cancel_during_a_generation(client, engine):
    engine.busy = True
    payload = client.post("/v1/cancel").json()
    assert payload["cancelled"] is True
    assert engine.cancel_requested is True


def test_unload_releases_the_model(client, engine):
    generate(client, model="z-image")
    assert client.get("/health").json()["loaded_model"] == "z-image:txt2img"

    payload = client.post("/v1/unload").json()
    assert engine.unload_called is True
    assert payload["loaded_model"] is None
    assert client.get("/health").json()["loaded_model"] is None


def test_progress_and_cancel_require_the_api_key(secured_client):
    # Unlike /health, these routes live under /v1: they must be protected like
    # the rest.
    assert secured_client.get("/v1/progress").status_code == 401
    assert secured_client.post("/v1/cancel").status_code == 401
    assert secured_client.post("/v1/unload").status_code == 401
