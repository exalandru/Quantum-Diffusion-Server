"""Prompt rewriting: a small local LLM that expands a short prompt before it
reaches a diffusion model.

Local models reward a long, specific prompt and give little back for three
words. This package closes that gap the way DALL-E 3 and Ideogram do -- by
expanding the prompt with a language model first -- and not the way Midjourney
does, which owes its results to an aesthetic fine-tune rather than to any
rewriting. What this buys is "three words become usable"; it is not a style.

`src/server/README.md` used to record the opposite decision: that
`mflux-inspire-fibo` and `mflux-refine-fibo` are not called because that "would
mean a second model resident alongside the first". That reasoning is superseded
here, and only on these terms:

* the rewriter is **transient**, not resident. `ModelEngine.rewrite` loads it,
  decodes, and unloads it in a `finally` -- so between two rewrites there is
  nothing in the slot, and `tests/test_engine.py` pins that on the exception
  path too, which is where a `raise` would otherwise strand a gigabyte;
* what it costs while it *is* live is bounded by construction, in two separate
  places, exactly as the upscaler's second slot is: `MAX_PROMPT_TOKENS` and
  `MAX_NEW_TOKENS` bound the KV cache -- the only part of a decode that grows
  with the input -- and `MAX_REWRITER_FOOTPRINT_MB` bounds weights plus that
  cache plus one step's logits, enforced at import;
* measured, on the shipped entry: 968 MB resident, 1289 MB peak with a full
  prompt. Against the 2630 MB transient the engine already accepts for an
  upscale, and against the 19281 MB peak a single 512x512 z-image generation
  reaches on the same machine. Over twenty consecutive rewrite-then-generate
  cycles beside a warm diffusion model, each measured after
  `mx.reset_peak_memory()` -- without that reset the figure is a monotonic
  high-water mark and would show nothing -- the per-cycle peak was identical
  from the first to the twentieth, and the diffusion weights were never
  reloaded.

`mflux-inspire-fibo` and `mflux-refine-fibo` are still not called. Those build
FIBO's structured JSON captions with Bria's VLM, which is a different job with a
hard failure mode -- `check_prompt` rejects malformed JSON outright -- and
models whose only prompt format is JSON are refused rewriting rather than
silently handed a caption this package cannot validate.

Two findings from the evaluation are load-bearing and easy to undo by accident:

* **zero-shot only.** Shown few-shot turns, the shipped 1.7B reproduces an
  exemplar verbatim for any input it cannot read: "закат над горами" comes back
  as the diving-helmet example, identically across seeds, 22 times in 108.
  Moving the examples into the system prompt made it worse, not better.
* **"leave a long prompt alone" is not a rule for the model.** Asked to return
  already-detailed prompts unchanged, the model obeyed 8 times in 18 -- and its
  quality on everything *else* dropped, because the rule competed for a small
  model's attention. The ceiling is enforced in Python instead
  (`RewriteSettings.word_ceiling`): above it the rewriter is not called at all,
  which makes the property true by construction rather than by hope.

`mlx-lm` is a plain runtime dependency rather than an optional extra, and that
was a correction. The extra's argument -- a feature that ships switched off
should not make everyone carry its dependency -- turned out to be wrong twice:
resolving mlx-lm alongside mflux adds exactly one package, since every
transitive dependency it has already arrives with mflux; and the menubar app
installs the server by `uv tool install <wheel>`, which cannot reach an extra
at all, so the first user to press Enhance was told to run a command that only
exists inside a checkout they do not have.

Only `catalogue` is imported when the package loads: it depends on nothing but
the standard library. Everything else is resolved lazily, because `fetch` reads
the catalogue on the `--status` path and `app` reads it at start-up, and
neither may pay for mlx.
"""

from __future__ import annotations

from typing import Any

from qds.rewrite.catalogue import (
    ALLOWED_BITS,
    KEYS,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_REWRITER_FOOTPRINT_MB,
    SPECS,
    RewriterSpec,
    by_key,
    kv_cache_bytes,
)

#: Exported name → the module that defines it. `weights` imports mlx_lm;
#: `prompt` does not, but is listed here so the package's import surface is one
#: table rather than two rules.
_LAZY: dict[str, str] = {
    "load_rewriter": "qds.rewrite.weights",
    "resolve_repo": "qds.rewrite.weights",
    "is_downloaded": "qds.rewrite.weights",
    "verify_loaded": "qds.rewrite.weights",
    "DEFAULT_SYSTEM_PROMPT": "qds.rewrite.prompt",
    "build_messages": "qds.rewrite.prompt",
    "sanitise": "qds.rewrite.prompt",
    "strip_thinking": "qds.rewrite.prompt",
    "trim_to_last_clause": "qds.rewrite.prompt",
}

__all__ = [
    "ALLOWED_BITS",
    "DEFAULT_SYSTEM_PROMPT",
    "KEYS",
    "MAX_NEW_TOKENS",
    "MAX_PROMPT_TOKENS",
    "MAX_REWRITER_FOOTPRINT_MB",
    "SPECS",
    "RewriterSpec",
    "build_messages",
    "by_key",
    "is_downloaded",
    "kv_cache_bytes",
    "load_rewriter",
    "resolve_repo",
    "sanitise",
    "strip_thinking",
    "trim_to_last_clause",
    "verify_loaded",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module_name), name)
