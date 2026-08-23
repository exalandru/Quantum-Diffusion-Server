"""QDS playground plugin — the chat's remote control for the playground.

Phase 2 of the QDS ↔ Hermes integration, and deliberately NOT a second way
to generate an image. Phase 1 (``hermes/image_gen/qds``) is the backend of
the built-in ``image_generate`` tool: ask for an image, get an image. That
is the right tool for a quick one-off, and this plugin does not replace it.

What this plugin adds is the thing a chat message cannot be: a **live
surface**. ``qds_playground`` creates a session, hands back a
``?view=plugin`` URL for Hermes' preview pane, and attaches the chat to
that session. The user then watches the image actually form — the real
playground feed, its own live preview, its own progressive blur — while
driving it in plain language from the chat.

The split matters, because the chat transcript is a stream of messages and
the playground is a surface that updates in place. Reproducing the second
inside the first only ever produces a stack of stills; opening the real one
beside the chat gives the user exactly what they already trust.

So: ``qds_playground`` opens and attaches, and every other tool here
defaults to the attached session, which is what lets "make it wider" or
"upscale that one" land in the pane the user is looking at without the
model threading an id through the conversation.

Every tool is gated by ``_check_qds_available`` (``GET /health``), so with
the server down they stay listed in ``hermes tools`` but never dispatch a
doomed call — and each handler still answers with a clean error if the
server dies between the check and the call.
"""

from __future__ import annotations

from .tools import (
    QDS_CANCEL_SCHEMA,
    QDS_GENERATE_SCHEMA,
    QDS_MODELS_SCHEMA,
    QDS_PLAYGROUND_SCHEMA,
    QDS_PROGRESS_SCHEMA,
    QDS_QUEUE_SCHEMA,
    QDS_SESSIONS_SCHEMA,
    QDS_UPSCALE_SCHEMA,
    QDS_WAIT_SCHEMA,
    _check_qds_available,
    _handle_qds_cancel,
    _handle_qds_generate,
    _handle_qds_models,
    _handle_qds_playground,
    _handle_qds_progress,
    _handle_qds_queue,
    _handle_qds_sessions,
    _handle_qds_upscale,
    _handle_qds_wait,
)

_TOOLS = (
    # The entry point: open the pane, attach the chat. Everything below it
    # operates on the session it attached to.
    ("qds_playground", QDS_PLAYGROUND_SCHEMA, _handle_qds_playground, "🎛"),
    ("qds_models",     QDS_MODELS_SCHEMA,     _handle_qds_models,     "📦"),
    ("qds_sessions",   QDS_SESSIONS_SCHEMA,   _handle_qds_sessions,   "🗂"),
    ("qds_generate",   QDS_GENERATE_SCHEMA,   _handle_qds_generate,   "🎨"),
    ("qds_progress",   QDS_PROGRESS_SCHEMA,   _handle_qds_progress,   "📊"),
    ("qds_wait",       QDS_WAIT_SCHEMA,       _handle_qds_wait,       "⏳"),
    ("qds_cancel",     QDS_CANCEL_SCHEMA,     _handle_qds_cancel,     "🛑"),
    ("qds_queue",      QDS_QUEUE_SCHEMA,      _handle_qds_queue,      "⏸"),
    ("qds_upscale",    QDS_UPSCALE_SCHEMA,    _handle_qds_upscale,    "🔍"),
)


def register(ctx) -> None:
    """Register every QDS playground tool. Called once by the plugin loader."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="qds",
            schema=schema,
            handler=handler,
            check_fn=_check_qds_available,
            emoji=emoji,
        )
