"""The MCP surface: what it offers a model, and what it refuses one.

MCP is the third plane, and the tests here are organised around the one thing
that makes it different from the other two: `/v1` serves an application and
`/playground/api` serves a person, while `/mcp` serves a *model* -- an untrusted
party whose arguments may have been written by someone else's prompt.

Security and containment live in `test_mcp_security.py`, progress attribution in
`test_mcp_progress.py`, and the stdio relay in `test_mcp_bridge.py`. This file
holds configuration, tool semantics and the shapes a tool returns.
"""

from __future__ import annotations

import pytest

from qds.settings import Settings

# --------------------------------------------------------------------------
# Configuration.
# --------------------------------------------------------------------------


def test_mcp_is_on_in_a_default_configuration():
    """The default is a decision, not an accident.

    On, matching `rewrite`: the SDK arrives with the server, a first call
    downloads nothing, and a plane that has to be discovered in a config file is
    a plane nobody finds.
    """
    assert Settings().mcp.enabled is True


def test_mcp_can_be_switched_off_entirely():
    settings = Settings.model_validate({"mcp": {"enabled": False}})
    assert settings.mcp.enabled is False


def test_destructive_tools_are_off_in_a_default_configuration():
    """Deleting in the playground is a click a human makes with the image in
    front of them. A tool has no equivalent of that confirmation."""
    assert Settings().mcp.allow_destructive is False


def test_image_roots_are_empty_by_default():
    """The containment boundary for `reference_path`, and it fails closed."""
    assert Settings().mcp.image_roots == []


def test_an_image_root_must_be_absolute():
    """A root that resolves against the CWD contains nothing -- and the CWD is
    '/' for an app launched from Finder."""
    with pytest.raises(ValueError, match="absolute"):
        Settings.model_validate({"mcp": {"image_roots": ["pictures"]}})


def test_an_image_root_is_expanded_and_kept_absolute(tmp_path):
    settings = Settings.model_validate({"mcp": {"image_roots": [str(tmp_path), "~/Pictures"]}})
    assert settings.mcp.image_roots[0] == str(tmp_path)
    assert settings.mcp.image_roots[1].startswith("/")


def test_blank_image_roots_are_dropped_rather_than_becoming_a_root(tmp_path):
    """An empty string is `Path("")` is `Path(".")`, which is not a root anyone
    meant -- and would be the CWD if it survived validation."""
    settings = Settings.model_validate({"mcp": {"image_roots": ["", "  ", str(tmp_path)]}})
    assert settings.mcp.image_roots == [str(tmp_path)]


def test_there_is_no_second_ceiling_on_n():
    """`server.max_n` is the one authority, enforced by `check_n` on every
    plane. Two numbers for one rule is how the lower one silently wins."""
    assert not hasattr(Settings().mcp, "max_n")


def test_the_thumbnail_bound_is_a_context_budget_not_a_free_number():
    """It sizes what enters a model's context, so it is bounded on both sides."""
    with pytest.raises(ValueError):
        Settings.model_validate({"mcp": {"thumbnail_px": 8192}})
    with pytest.raises(ValueError):
        Settings.model_validate({"mcp": {"thumbnail_px": 8}})


def test_mcp_settings_are_not_reachable_through_the_environment(monkeypatch):
    """`_env_overrides` covers `ServerSettings` only, as it does for `rewrite`.

    Pinned as a *decision*: someone reading `QDS_SERVER_ENABLED` in a launch
    script should find out here that it does nothing, rather than in production.
    """
    monkeypatch.setenv("QDS_SERVER_ENABLED", "false")
    monkeypatch.setenv("QDS_SERVER_ALLOW_DESTRUCTIVE", "true")
    settings = Settings()
    assert settings.mcp.enabled is True
    assert settings.mcp.allow_destructive is False


# --------------------------------------------------------------------------
# Import weight.
# --------------------------------------------------------------------------


def test_importing_the_settings_stays_free_of_the_sdk():
    """The SDK is a runtime dependency, not a startup cost for every command.

    `qds fetch --status` and `qds status` read configuration and exit, and the
    menubar app calls them often enough that seconds matter. Mirrors the same
    assertion `test_rewrite.py` makes about mlx.
    """
    import subprocess
    import sys

    probe = (
        "import sys, qds.settings, qds.cli, qds.upscale.catalogue, qds.rewrite.catalogue;"
        "print('mcp' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------
# What the surface offers.
# --------------------------------------------------------------------------


async def test_the_tools_a_default_configuration_offers(tmp_path):
    """Pinned as a list, so adding one is a decision somebody made on purpose.

    Nine verbs. `refine_image` and `vary_image` are both expressible as
    `generate_image` with the right arguments, and exist anyway because for a
    model choosing from a list the name is the instruction.
    """
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        names = sorted(tool.name for tool in (await client.list_tools()).tools)
    assert names == [
        "cancel_generation",
        "generate_image",
        "list_models",
        "list_sessions",
        "open_session",
        "refine_image",
        "upscale_image",
        "vary_image",
        "wait_for_generation",
    ]


async def test_destructive_tools_are_absent_with_the_shipped_configuration(tmp_path):
    """Absent from `tools/list`, not present-and-refusing.

    A model cannot call a tool it was never shown, which is a stronger
    guarantee than one that argues with it.
    """
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert "delete_image" not in names
    assert "delete_group" not in names


async def test_destructive_tools_appear_when_they_are_switched_on(tmp_path):
    """The counterfactual: without it, the test above would pass on a typo."""
    from .mcp_support import mcp_session, mcp_settings

    settings = mcp_settings(tmp_path)
    settings.mcp.allow_destructive = True
    async with mcp_session(tmp_path, settings=settings) as (client, _app, _engine):
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert {"delete_image", "delete_group"} <= names


async def test_there_is_no_tool_for_a_session_password(tmp_path):
    """A password on a session is a person's decision about their browser.

    A tool that could set, remove or present one would make the control mean
    nothing -- so MCP carries no token channel at all, and this is the witness
    that none was added later "just for convenience".
    """
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        names = " ".join(tool.name for tool in (await client.list_tools()).tools)
    assert "password" not in names
    assert "unlock" not in names


async def test_there_is_no_tool_for_pausing_the_queue(tmp_path):
    """Pausing is a human's control over their own machine, and combined with a
    blocking tool it guarantees a timeout. It is *reported* instead."""
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        names = " ".join(tool.name for tool in (await client.list_tools()).tools)
    assert "pause" not in names


# --------------------------------------------------------------------------
# Generating.
# --------------------------------------------------------------------------


async def test_a_generation_attaches_the_picture_and_names_the_file(tmp_path):
    """The image block is what a client renders, and it needs nothing from the
    model to do it.

    It was removed once, on the reasoning that only the person judges the
    picture -- which left a markdown `data:` URI as the sole way it could
    appear. A real model will not reproduce thousands of tokens of base64: one
    was observed reasoning "I can't directly attach the image again" while the
    encoding sat unused in its context. Attaching it is the server's job.
    """
    from .mcp_support import filenames_of, images_of, mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, engine):
        result = await client.call_tool("generate_image", {"prompt": "a red cube on grass"})

    assert result.is_error is False
    assert len(images_of(result)) == 1
    assert images_of(result)[0].mime_type == "image/jpeg"
    body = text_of(result)
    assert "status: completed" in body
    assert filenames_of(result)[0].endswith(".png"), "the file named is the PNG, not the preview"
    assert engine.jobs, "the job reached the engine"


def links_of(result) -> list:
    return [b for b in result.content if getattr(b, "type", "") == "resource_link"]


async def test_each_image_carries_a_resource_link_to_its_own_url(tmp_path):
    """The link names the file's own http URL, not a private `qds://` scheme.

    Both are valid MCP -- a resource URI is opaque, resolved by
    `resources/read` -- but only one is also an address a client can open by
    itself, and a model shown an unresolvable scheme was observed concluding it
    had nothing to offer. `file://` was not an option: it would put the
    operator's home directory into a model's context on every generation.
    """
    from .mcp_support import filenames_of, mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube", "n": 2})
        links = links_of(result)
        assert len(links) == 2, "one per image"
        assert {str(link.uri).rsplit("/", 1)[-1] for link in links} == set(filenames_of(result))
        assert all(str(link.uri).startswith("http://") for link in links)
        assert all(link.mime_type == "image/png" for link in links)

        # And it still resolves, which is what makes it a route and not a label.
        payload = await client.read_resource(str(links[0].uri))
    assert payload.contents[0].mime_type == "image/png"


async def test_the_result_never_carries_the_operators_filesystem_path(tmp_path):
    """It did, and that put a home directory -- a username with it -- into a
    model's context on every generation, to save a lookup the file name and the
    playground already answer."""
    from .mcp_support import mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube"})

    body = text_of(result) + " ".join(str(link.uri) for link in links_of(result))
    assert "path:" not in body
    assert str(tmp_path) not in body, "no absolute path anywhere in the result"
    assert "file://" not in body, "which is why the resource URI is the http one"


async def test_the_result_offers_a_link_and_not_a_markdown_image(tmp_path):
    """A link, because a markdown image cannot be made to work here.

    With the server's URL, a chat client's `img-src` refuses `http:` outright.
    With a `data:` URI the model has to reproduce the encoding, and it declined
    at every size tried -- down to 1 300 characters, well under the 2 263 it had
    been observed copying successfully once. The link is twenty tokens, it gets
    reproduced, and it opens the full file rather than a preview.
    """
    from .mcp_support import filenames_of, mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a red cube on grass"})

    body = text_of(result)
    row = next(r for r in body.splitlines() if "link:" in r).strip()
    link = row.partition("link: ")[2]
    assert link == f"[Open the image](http://127.0.0.1:8765/playground/images/{filenames_of(result)[0]})"
    assert "![" not in body, "never a markdown image: it cannot render from this server"
    assert "data:" not in body, "and never the encoding, which the model will not retype"


async def test_the_attached_preview_is_bounded_by_the_setting(tmp_path):
    """One encode, and it is the size `thumbnail_px` asks for."""
    import base64
    import io

    from PIL import Image as PILImage

    from .mcp_support import images_of, mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube"})

    with PILImage.open(io.BytesIO(base64.b64decode(images_of(result)[0].data))) as preview:
        assert preview.format == "JPEG"
        assert max(preview.size) <= Settings().mcp.thumbnail_px


async def test_with_no_thumbnail_the_facts_and_the_link_survive(tmp_path):
    """`thumbnail_px: 0` removes a view, never a fact."""
    from .mcp_support import images_of, mcp_session, mcp_settings, text_of

    settings = mcp_settings(tmp_path)
    settings.mcp.thumbnail_px = 0
    async with mcp_session(tmp_path, settings=settings) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube"})

    assert images_of(result) == []
    assert links_of(result), "the file is still reachable"
    assert "size: 2x2" in text_of(result), "the real dimensions survive"


async def test_the_instructions_ask_for_the_link_and_nothing_more(tmp_path):
    """One thing has to travel through the model, and it is small enough that
    it does."""
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        instructions = client.instructions or ""
    assert "`link:` line" in instructions
    assert "data:" not in instructions


async def test_the_resource_link_is_annotated_for_the_user(tmp_path):
    """The protocol's own way of saying "this is meant to be shown", on the one
    block that is still a block."""
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube"})

    link = links_of(result)[0]
    assert link.annotations is not None
    assert "user" in link.annotations.audience


async def test_a_generation_lands_in_a_playground_session(tmp_path):
    """MCP writes what the browser writes. That is the whole persistence design:
    a chat generation is visible, resumable and refinable in the playground."""
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        await client.call_tool("generate_image", {"prompt": "a red cube"})
        listed = await client.call_tool("list_sessions", {})
    assert "No playground sessions yet." not in listed.content[0].text


async def test_listing_tools_leaves_no_session_behind(tmp_path):
    """Sessions are created lazily, on first *generation*.

    A client that connects, reads the tool list and disconnects -- which is what
    every client does on startup -- must not litter someone's sidebar.
    """
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        await client.list_tools()
        listed = await client.call_tool("list_sessions", {})
    assert "No playground sessions yet." in listed.content[0].text


async def test_n_produces_that_many_images(tmp_path):
    from .mcp_support import filenames_of, mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube", "n": 3})
    assert len(set(filenames_of(result))) == 3
    assert len(links_of(result)) == 3
    assert text_of(result).count("image ") >= 3, "one row each"


async def test_a_seed_is_reported_so_it_can_be_reproduced(tmp_path):
    from .mcp_support import mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube", "seed": 1234})
    assert "seed: 1234" in text_of(result)
    assert engine.jobs[0].seed == 1234


async def test_width_and_height_reach_the_engine_as_a_size(tmp_path):
    """The tool takes two integers where the shared validator takes "WxH".

    Formatting the string at one point removes a class of error a small model
    makes, while `resolve_size` stays the one that decides what is allowed.
    """
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, engine):
        await client.call_tool("generate_image", {"prompt": "a cube", "width": 1024, "height": 768})
    assert (engine.jobs[0].width, engine.jobs[0].height) == (1024, 768)


async def test_one_of_width_and_height_alone_is_refused(tmp_path):
    """Half a size is not a size, and silently completing it would be a guess."""
    from .mcp_support import mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube", "width": 1024})
    assert result.is_error is True
    assert "width and height together" in text_of(result)


# --------------------------------------------------------------------------
# One admission authority.
# --------------------------------------------------------------------------
#
# The long-term failure this guards against is drift: someone adds a check to
# `playground_generate` and not to the MCP tool, and the two planes quietly
# start accepting different things. So the assertion is not "MCP refuses bad
# input" -- it is that MCP refuses *the same input with the same code* as the
# HTTP route, compared side by side in one test.

REFUSALS = [
    pytest.param(
        {"prompt": "a cube", "n": 99},
        {"prompt": "a cube", "n": "99"},
        "n_too_large",
        id="n over the server limit",
    ),
    pytest.param(
        {"prompt": "a cube", "model": "no-such-model"},
        {"prompt": "a cube", "model": "no-such-model"},
        "model_not_found",
        id="an unknown model",
    ),
    pytest.param(
        {"prompt": "a cube", "model": "fibo"},
        {"prompt": "a cube", "model": "fibo"},
        "prompt_must_be_json",
        id="plain text to a JSON-only model",
    ),
    pytest.param(
        {"prompt": "a cube", "model": "flux2-klein", "negative_prompt": "blurry"},
        {"prompt": "a cube", "model": "flux2-klein", "negative_prompt": "blurry"},
        "unsupported_parameter",
        id="a negative prompt a model cannot take",
    ),
    pytest.param(
        {"prompt": "a cube", "model": "anima", "width": 256, "height": 256},
        {"prompt": "a cube", "model": "anima", "size": "256x256"},
        "invalid_size",
        id="a size under the model's minimum",
    ),
    pytest.param(
        {"prompt": "a cube", "model": "anima", "width": 4096, "height": 4096},
        {"prompt": "a cube", "model": "anima", "size": "4096x4096"},
        "invalid_size",
        id="a size over the model's maximum",
    ),
    pytest.param(
        {"prompt": "a cube", "model": "fibo", "enhance_prompt": True},
        {"prompt": "a cube", "model": "fibo", "rewrite": "true"},
        "prompt_must_be_json",
        id="enhancing a JSON-only model",
    ),
]


@pytest.mark.parametrize("tool_args,form,expected_code", REFUSALS)
async def test_a_tool_refuses_exactly_what_the_playground_route_refuses(
    tmp_path, tool_args, form, expected_code
):
    from .conftest import make_client
    from .mcp_support import mcp_session, text_of

    async with mcp_session(tmp_path) as (client, app, _engine):
        http = make_client(app)
        session_id = http.post("/playground/api/sessions").json()["id"]
        over_http = http.post(f"/playground/api/sessions/{session_id}/generations", data=form)
        over_mcp = await client.call_tool("generate_image", tool_args)

    assert over_http.status_code >= 400, "the route refused it"
    assert over_http.json()["error"]["code"] == expected_code
    assert over_mcp.is_error is True, "and so did the tool"
    # The message the model reads is the server's own, not a generic wrapper --
    # which is what lets a small model correct its own call.
    assert over_http.json()["error"]["message"] in text_of(over_mcp)


async def test_an_error_reaches_the_model_as_readable_text(tmp_path):
    """Never `MCPError`, which hides the message from the model. A tool error
    that says only "invalid request" is one a model cannot act on."""
    from .mcp_support import mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube", "n": 99})
    body = text_of(result)
    assert "exceeds the server limit" in body
    assert "4" in body, "and it names the limit, so the next call can be right"


# --------------------------------------------------------------------------
# Refine, vary, upscale: lineage, and files a row owns.
# --------------------------------------------------------------------------


async def test_refining_joins_the_originals_group(tmp_path):
    """One feed entry in the playground, which is what makes a chat refinement
    look like a refinement rather than an unrelated second picture."""
    from .mcp_support import filenames_of, mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        first = await client.call_tool("generate_image", {"prompt": "a cube", "model": "qwen-image-2512"})
        group = text_of(first).split("group: ")[1].split()[0]
        refined = await client.call_tool(
            "refine_image", {"image": filenames_of(first)[0], "prompt": "make it blue"}
        )
    assert refined.is_error is False, text_of(refined)
    assert f"group: {group}" in text_of(refined)


async def test_a_variation_keeps_the_settings_and_changes_the_seed(tmp_path):
    from .mcp_support import filenames_of, mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, engine):
        first = await client.call_tool("generate_image", {"prompt": "a cube", "seed": 4242})
        varied = await client.call_tool("vary_image", {"image": filenames_of(first)[0]})

    assert varied.is_error is False, text_of(varied)
    assert "seed: 4242" not in text_of(varied), "a variation is a new seed"
    original, variation = engine.jobs[0], engine.jobs[-1]
    assert (variation.width, variation.height) == (original.width, original.height)
    assert variation.steps == original.steps
    assert variation.prompt == original.prompt


async def test_a_variation_survives_deleting_the_image_it_came_from(tmp_path):
    """The row owns a *copy* of its reference, never the original.

    An implementation that reuses the source file name passes every "does it
    work" test and fails only here -- and fails in production the first time
    someone tidies up their playground.
    """
    from .conftest import make_client
    from .mcp_support import filenames_of, mcp_session, text_of

    async with mcp_session(tmp_path) as (client, app, _engine):
        first = await client.call_tool("generate_image", {"prompt": "a cube", "model": "qwen-image-2512"})
        source = filenames_of(first)[0]
        refined = await client.call_tool("refine_image", {"image": source, "prompt": "make it blue"})
        assert refined.is_error is False, text_of(refined)

        http = make_client(app)
        assert http.delete(f"/playground/api/images/{source}").status_code == 204

        # The refinement's own reference must still be on disk.
        again = await client.call_tool("vary_image", {"image": filenames_of(refined)[0]})
    assert again.is_error is False, text_of(again)


async def test_upscaling_goes_through_the_same_closure_as_the_route(tmp_path):
    from .mcp_support import filenames_of, mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, engine):
        first = await client.call_tool("generate_image", {"prompt": "a cube"})
        bigger = await client.call_tool("upscale_image", {"image": filenames_of(first)[0], "scale": 2})
    assert bigger.is_error is False, text_of(bigger)
    assert engine.upscales, "the upscale reached the engine"


async def test_upscaling_an_image_this_server_never_made_is_refused(tmp_path):
    from .mcp_support import mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("upscale_image", {"image": "elsewhere.png"})
    assert result.is_error is True
    assert "No image" in text_of(result)


# --------------------------------------------------------------------------
# Sessions.
# --------------------------------------------------------------------------


async def test_naming_a_session_that_does_not_exist_is_an_error_not_a_new_one(tmp_path):
    """Silently writing into a fresh session would hide the model's mistake and
    scatter someone's work across sessions they never asked for."""
    from .mcp_support import mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube", "session_id": "not-a-session"})
    assert result.is_error is True
    assert "list_sessions" in text_of(result)


async def test_open_session_redirects_what_follows_into_it(tmp_path):
    from .mcp_support import mcp_session, text_of

    async with mcp_session(tmp_path) as (client, _app, _engine):
        opened = await client.call_tool("open_session", {"title": "Cathedrals"})
        session_id = text_of(opened).split("Session ")[1].split()[0]
        made = await client.call_tool("generate_image", {"prompt": "a cube"})
        listed = await client.call_tool("list_sessions", {})

    assert f"session: {session_id}" in text_of(made)
    assert "Cathedrals" in text_of(listed)


async def test_the_resource_serves_the_png_while_the_attached_preview_is_a_jpeg(tmp_path):
    """The two halves of the budget: a small preview in the result, the real
    file one `resources/read` away."""
    import base64
    import io

    from PIL import Image as PILImage

    from .mcp_support import images_of, mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        made = await client.call_tool("generate_image", {"prompt": "a cube"})
        resource = await client.read_resource(str(links_of(made)[0].uri))

    assert resource.contents[0].mime_type == "image/png"
    with PILImage.open(io.BytesIO(base64.b64decode(images_of(made)[0].data))) as preview:
        assert preview.format == "JPEG"


# --------------------------------------------------------------------------
# The thumbnail is a context budget.
# --------------------------------------------------------------------------


def test_the_preview_default_is_a_measured_context_budget():
    """Pinned because it was got wrong by an order of magnitude once.

    512/82 was validated against a flat-colour test image, came out at 3 KB and
    looked free; real 2880x1600 generations produced 40-54 KB -- 16 000 to
    22 000 tokens, which on an 8B model with an 8k window is the whole context.
    Cost follows the detail surviving the downscale, not the source file's size.
    """
    mcp = Settings().mcp
    assert mcp.thumbnail_px == 256
    assert mcp.thumbnail_quality == 70


def test_the_thumbnail_can_be_switched_off_entirely():
    """For a text-only model it is pure cost, and MCP offers no way to detect
    one -- `ClientCapabilities` says nothing about what the model can read."""
    assert Settings.model_validate({"mcp": {"thumbnail_px": 0}}).mcp.thumbnail_px == 0


def test_a_thumbnail_too_small_to_be_worth_its_tokens_is_refused():
    """0 means "omit"; 32 means "spend context on something unreadable"."""
    with pytest.raises(ValueError, match="omit the thumbnail"):
        Settings.model_validate({"mcp": {"thumbnail_px": 32}})


# --------------------------------------------------------------------------
# Block order.
# --------------------------------------------------------------------------


async def test_the_text_block_comes_first(tmp_path):
    """Anything that truncates a long result drops what is at the end, and the
    end must not be the facts the reply is built from."""
    from .mcp_support import mcp_session

    async with mcp_session(tmp_path) as (client, _app, _engine):
        result = await client.call_tool("generate_image", {"prompt": "a cube", "n": 2})

    kinds = [getattr(block, "type", "") for block in result.content]
    assert kinds[0] == "text", kinds
    assert set(kinds[1:]) == {"image", "resource_link"}, kinds
