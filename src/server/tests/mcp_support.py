"""Shared scaffolding for the MCP tests.

Its own module rather than `conftest.py` because importing it imports the MCP
SDK, and `test_mcp.py::test_importing_the_settings_stays_free_of_the_sdk` is a
witness that the rest of the server does not. A fixture that everything loads
would make that witness impossible to write.

Two ways in, and choosing between them is the point:

* `mcp_session` drives the server object directly, in memory. Fast, and right
  for *semantics* -- what a tool accepts, what it returns, what it refuses.
* `make_client` (from `conftest`) goes over HTTP. Required for anything about
  the *guard*, because an in-memory client never passes through it, and a test
  that cannot fail is not a witness.
"""

from __future__ import annotations

import contextlib

from qds.app import create_app
from qds.settings import Settings

from .conftest import FakeEngine


def mcp_settings(tmp_path, **overrides) -> Settings:
    """The `settings` fixture's configuration, with an `mcp` section to vary."""
    document = {
        "server": {
            "image_store": str(tmp_path / "images"),
            "playground_store": str(tmp_path / "playground"),
            "log_file": None,
            "progress_log_every": 0,
            "max_n": 4,
            **overrides.pop("server", {}),
        },
        "default_model": "flux2-klein",
        "models": {"qwen-image-2512": {"enable_edit": True}},
    }
    if overrides:
        document.update(overrides)
    return Settings.model_validate(document)


@contextlib.asynccontextmanager
async def mcp_session(tmp_path, *, engine=None, settings=None, **overrides):
    """`(client, app, engine)` with the application actually running.

    The lifespan is entered rather than skipped, and not as ceremony: the
    playground runner is started there, so a generation submitted without it
    would sit in the queue for ever and every tool call would hit its ceiling.
    """
    from mcp import Client

    engine = engine or FakeEngine()
    app = create_app(settings or mcp_settings(tmp_path, **overrides), engine)
    async with app.router.lifespan_context(app):
        async with Client(app.state.mcp_server, raise_exceptions=False) as client:
            yield client, app, engine


def text_of(result) -> str:
    """Every text block of a tool result, joined.

    Tool results are mixed: thumbnails first, then one text block. Asserting on
    `content[0]` would silently start reading an image the day a tool returns
    one, so the text is collected by *kind* rather than by position.
    """
    return "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text")


def images_of(result) -> list:
    return [block for block in result.content if getattr(block, "type", "") == "image"]


def filenames_of(result) -> list[str]:
    """The image file names a tool reported, read out of its text block."""
    names = []
    for line in text_of(result).splitlines():
        if "file: " in line:
            names.append(line.split("file: ", 1)[1].split()[0])
    return names
