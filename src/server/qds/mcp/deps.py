"""What the MCP tools are handed, instead of what they could reach for.

Every validator here already exists as a closure inside `create_app`, and this
bundle exists so that MCP calls *those* rather than its own. That is INV-1, and
it is the difference between a second plane and a second set of rules: a check
added to `playground_generate` tomorrow reaches MCP for free, because there is
one implementation, not two that look alike.

The pattern is `PlaygroundRunner`'s. It takes `resolve_spec`, `resolve_upscaler`
and `build_rewrite_job` for the reason its own docstring gives -- it is handed
how to look things up rather than reaching into a catalogue itself -- and this
is the same arrangement one layer out. Nothing moves out of `create_app`; `/v1`
is untouched.

Values, not a `Request` or an app: the tools run on a mounted application that
never sees FastAPI's dependency machinery, so anything they need must be here at
construction time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from qds.settings import Settings


@dataclass(frozen=True)
class MCPDeps:
    """One application's generative capability, addressable without HTTP."""

    settings: Settings
    #: The durable record. MCP writes what the playground writes.
    store: Any
    #: The FIFO worker. MCP submits to the same queue as the browser, and is
    #: paused and cancelled by the same controls.
    runner: Any
    engine: Any

    # ── The admission rules, borrowed rather than reproduced ───────────────
    resolve_spec: Callable[[str | None], Any]
    resolve_size: Callable[[Any, str | None], tuple[int, int]]
    check_prompt: Callable[[Any, str], None]
    check_capabilities: Callable[..., None]
    check_rewrite: Callable[..., bool]
    check_n: Callable[[int], None]
    seeds_for: Callable[[int | None, int], list[int]]

    #: Admit, copy and enqueue an upscale -- the very closure
    #: `POST /playground/api/sessions/{id}/upscales` calls. The render-budget
    #: arithmetic and the "only this session's images" rule live there, once.
    submit_upscale: Callable[..., dict]

    # ── What the catalogue says, for the listing tools and resources ───────
    capabilities: Callable[[Any], dict]
    #: Public name → spec, for `list_models`. The same mapping `resolve_spec`
    #: consults, passed rather than rebuilt so the two cannot disagree about
    #: which models exist.
    models: dict[str, Any]

    #: Absolute root of this server's own URLs, e.g. `http://127.0.0.1:8765`.
    #: Tool output names files by URL as well as by path, because a chat client
    #: can render the first and a script wants the second, and neither can
    #: reconstruct the origin from a relative path.
    base_url: str
