"""`python -m qds` — the same command as the `qds` console script.

The server spawns its own long-running jobs this way rather than by looking for
a `qds` binary on PATH: `sys.executable -m qds` names the interpreter already
running, so a child is guaranteed to be the same installation as its parent.
"""

from qds.cli import main

raise SystemExit(main())
