"""The MCP plane: this server's capability, offered to a model.

`/v1` is called by applications, `/playground/api` by a person at a browser, and
`/mcp` by a *model* -- a language model in someone's chat client, deciding for
itself which tool to call and what to put in its arguments. That third audience
is what every design choice in this package answers to: the arguments are chosen
by an untrusted party, the results are consumed by a limited context window, and
the caller cannot be asked to confirm anything.

**Naming.** This package is `qds.mcp`; the SDK is the top-level `mcp`. Python 3
absolute imports keep them apart, so `import mcp` inside these modules is the
SDK, every time -- the local package is only ever reached as `qds.mcp.*`. Never
add a relative `from . import mcp`-shaped import here; it would make that
sentence false and the failure would be silent.

**What lives where.**

- `asgi`     -- the guard that authorizes and contains the mounted application.
- `deps`     -- the bundle of validators and stores the tools are handed.
- `images`   -- thumbnails, and the trust boundary for a model-chosen path.
- `progress` -- attribution: which engine progress is honestly this call's.
- `run`      -- submit, wait with a ceiling, cancel.
- `sessions` -- which playground session a conversation writes to.
- `server`   -- the tools, resources and prompts themselves.
- `bridge`   -- the stdio relay behind `qds mcp`.

Nothing here re-implements admission. A generation reaching the engine through
MCP is validated by the very callables `/playground/api` uses, handed over in
`deps.MCPDeps`, because a second admission path is a path that drifts laxer.
"""

from __future__ import annotations
