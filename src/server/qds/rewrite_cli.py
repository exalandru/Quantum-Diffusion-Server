"""`qds rewrite` -- expand one prompt and print the result.

This exists so the feature can be exercised, and its quality judged, without a
browser. That is not a convenience: the evaluation that decides whether prompt
rewriting is worth showing to a user has to compare images generated with and
without it, over a fixed prompt set, and driving that through the playground UI
would make the sample size the thing that gives out first.

It also makes the failure modes reachable by hand. A rewrite can fail in three
ways that look identical from the dashboard -- the dependency is missing, the
weights are not downloaded, or the model produced something `sanitise` refused
-- and each prints a different line here.

Deliberately not wired to the queue or the store: this loads the rewriter,
decodes, unloads, and exits. It never touches the playground database, so it
cannot be used to inspect or alter a session.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from qds.errors import APIError
from qds.logs import SERVER_LOGGER, setup_logging
from qds.settings import load_settings

logger = logging.getLogger(SERVER_LOGGER)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qds rewrite",
        description="Expand a prompt with the local rewriter and print the result.",
    )
    parser.add_argument("prompt", help="the prompt to expand")
    parser.add_argument(
        "--model",
        default=None,
        help="rewriter catalogue key (default: the configured `rewrite.model`)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite even when the prompt is at or over `rewrite.word_ceiling`",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampler seed (default: 0)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature (default: the configured `rewrite.temperature`)",
    )
    args = parser.parse_args(list(argv or []))

    setup_logging(level="INFO")
    # `strict=False`: expanding a prompt does not need a servable configuration,
    # and refusing to run because `default_model` is disabled would be the same
    # blast-radius mistake `Settings.runtime_issues` exists to avoid.
    settings = load_settings(strict=False)
    settings.apply_hf_home()

    from qds.rewrite.catalogue import KEYS, by_key
    from qds.rewrite.prompt import DEFAULT_SYSTEM_PROMPT, RewriteRejected, should_rewrite

    key = args.model or settings.rewrite.model
    spec = by_key(key)
    if spec is None:
        logger.error("Unknown rewriter %r. Valid keys: %s", key, sorted(KEYS))
        return 2

    # The ceiling is reported rather than silently applied: on the CLI the user
    # named this prompt explicitly, so "nothing happened" needs a reason.
    if not args.force and not should_rewrite(args.prompt, word_ceiling=settings.rewrite.word_ceiling):
        logger.info(
            "%d words is at or over the %d-word ceiling: generated as typed. Use --force to rewrite anyway.",
            len(args.prompt.split()),
            settings.rewrite.word_ceiling,
        )
        print(args.prompt)
        return 0

    # `--force` skips the *quality* ceiling, never the triage that stands ahead
    # of the memory bound. The HTTP route refuses past this too, and the bound
    # itself -- in tokens -- is enforced in the engine, which both paths reach.
    from qds.rewrite.catalogue import MAX_PROMPT_CHARS

    if len(args.prompt) > MAX_PROMPT_CHARS:
        logger.error(
            "This prompt is %d characters, past the %d the rewriter accepts.",
            len(args.prompt),
            MAX_PROMPT_CHARS,
        )
        return 2

    import asyncio

    from qds.engine import ModelEngine, RewriteJob

    engine = ModelEngine(progress_log_every=0)
    job = RewriteJob(
        spec=spec,
        prompt=args.prompt,
        system_prompt=settings.rewrite.system_prompt or DEFAULT_SYSTEM_PROMPT,
        max_new_tokens=settings.rewrite.max_new_tokens,
        temperature=args.temperature if args.temperature is not None else settings.rewrite.temperature,
        timeout_s=settings.rewrite.timeout_s,
        seed=args.seed,
    )
    try:
        print(asyncio.run(engine.rewrite(job)))
        return 0
    except RewriteRejected as exc:
        # The decode worked and the output was unusable. In the playground this
        # is the case that falls back to the typed prompt and records why; here
        # it is worth seeing on its own.
        logger.error("The rewriter produced an unusable result: %s", exc)
        return 1
    except APIError as exc:
        logger.error("%s", exc.message)
        return 1
    finally:
        engine.shutdown()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
