"""Resolving, loading and checking the rewriter's weights.

Two guards live here, and they guard different things -- the same split as
`qds/upscale/weights.py`, for a different reason.

There is no `verify_state_dict` equivalent: `mlx_lm.load` builds the module from
the repository's own `config.json` and applies the checkpoint itself, so a
truncated file fails there rather than loading into silence. What *can* still
be wrong is the catalogue: nothing stops an entry from declaring 28 layers
against a repository that has 36. That matters more than it looks, because
`catalogue.kv_cache_bytes` -- the execution half of the argument that lets a
third slot exist at all -- is computed from those declared numbers. An
unchecked catalogue turns a bound into an estimate.

So `verify_loaded` compares the loaded model's architecture against the
catalogue and refuses a mismatch, and separately requires the tokenizer to
carry a chat template, without which `build_messages` would be silently
concatenated into a bare completion prompt.

The download is the ordinary Hugging Face one, so the files land wherever
`settings.apply_hf_home()` already pointed `HF_HOME` and `HF_HUB_CACHE`. No
`cache_dir` is passed on purpose: a second opinion about where the cache lives
is how the two drift apart.
"""

from __future__ import annotations

import logging
from typing import Any

from qds.errors import APIError
from qds.logs import SERVER_LOGGER
from qds.rewrite.catalogue import RewriterSpec

logger = logging.getLogger(SERVER_LOGGER)

#: Files `mlx_lm.load` needs before it will build anything.
#:
#: Asking about these rather than about the repository is the same distinction
#: `upscale.weights.cached_path` draws: `availability.scan_repos` answers "this
#: repo is in the cache", which would say `present` for a repository from which
#: only the README had been pulled.
_REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def cached_files(spec: RewriterSpec) -> list[str] | None:
    """The local metadata files for `spec` if present, else `None`.

    Weight shards are deliberately not checked by name: their filenames depend
    on how the repository was sharded, which is not catalogue data. `mlx_lm`
    resolves them from `config.json`, and a repository whose metadata is cached
    but whose shards are not simply downloads them.
    """
    from huggingface_hub import try_to_load_from_cache

    found = []
    for name in _REQUIRED_FILES:
        path = try_to_load_from_cache(spec.repo, name)
        if not isinstance(path, str):
            return None
        found.append(path)
    return found


def is_downloaded(spec: RewriterSpec) -> bool:
    """Whether using `spec` right now would avoid a download. Never hits the network."""
    try:
        return cached_files(spec) is not None
    except Exception:  # pragma: no cover - a broken cache is not a reason to 500
        logger.debug("Could not inspect the cache for %s", spec.key, exc_info=True)
        return False


def require_mlx_lm() -> Any:
    """The `mlx_lm` module, or a 409 saying the installation is incomplete.

    `mlx-lm` is a plain runtime dependency, so on any installation this server
    made of itself it is present and this never raises. It was an optional
    extra first, and the message here told the user to run `uv sync --extra
    rewrite` -- which is not a thing someone who installed the app can do, or
    should have to. Dependencies install themselves; that is what a dependency
    is.

    The guard stays because reaching it now means something else is true: the
    environment was assembled by hand, or a partial install was interrupted. So
    it reports *that*, rather than naming a step nobody skipped.
    """
    try:
        import mlx_lm
    except ImportError as exc:
        raise APIError(
            "This server's installation is incomplete: `mlx-lm`, a required "
            "dependency, is missing. Reinstalling the server repairs it.",
            status_code=409,
            code="rewriter_unavailable",
        ) from exc
    return mlx_lm


def resolve_repo(spec: RewriterSpec, *, allow_download: bool = True) -> str:
    """`spec`'s repository id, checking first that using it will not surprise.

    Returns the repo rather than a path because `mlx_lm.load` takes a repo id
    and does its own resolution. What this adds is the refusal: without it, a
    first rewrite on a cold cache would block the engine's single worker thread
    for the length of a download, with the diffusion model resident and the
    queue stalled behind it.
    """
    if is_downloaded(spec):
        return spec.repo
    if not allow_download:
        raise APIError(
            f"The weights for {spec.display_name!r} are not downloaded. "
            f"Run `qds fetch {spec.key}`.",
            status_code=409,
            code="rewriter_not_downloaded",
        )
    logger.info(
        "Downloading rewriter %s (%.0f MB) from %s",
        spec.key,
        spec.size_mb,
        spec.repo,
        extra={"model": spec.key, "repo": spec.repo},
    )
    return spec.repo


def verify_loaded(model: Any, tokenizer: Any, spec: RewriterSpec) -> None:
    """Refuse a model whose architecture contradicts the catalogue.

    The three architecture fields checked here are exactly the three
    `kv_cache_bytes` multiplies. Checking anything else would be decoration;
    checking fewer would leave the bound unverified in the dimension that
    drifted.
    """
    args = getattr(model, "args", None)
    if args is None:  # pragma: no cover - every mlx_lm model carries this
        raise ValueError(f"Rewriter {spec.key!r} exposes no model arguments to check.")

    declared = {
        "num_hidden_layers": spec.num_hidden_layers,
        "num_key_value_heads": spec.num_key_value_heads,
        "vocab_size": spec.vocab_size,
    }
    for field, expected in declared.items():
        actual = getattr(args, field, None)
        # An *absent* field used to pass here, and that was a hole rather than a
        # nicety. Measured on `mlx-community/Ministral-3-3B-Instruct-2512-4bit`,
        # whose `config.json` nests everything under `text_config`: every one of
        # these reads `None`, the whole cross-check evaluates to nothing, and
        # `kv_cache_bytes` -- the execution half of the argument that lets a
        # third slot exist at all -- silently reverts to an unverified estimate.
        # A bound that cannot be checked is not a bound, so absence raises.
        if actual is None:
            raise ValueError(
                f"Rewriter {spec.key!r} loads from {spec.repo} without declaring "
                f"{field}, so the catalogue's value cannot be checked against it. "
                "`kv_cache_bytes` is computed from the catalogue and would be an "
                "estimate rather than a bound. Checkpoints that nest their "
                "architecture (multimodal `text_config`) land here."
            )
        if actual != expected:
            raise ValueError(
                f"Rewriter {spec.key!r} declares {field}={expected} but "
                f"{spec.repo} loads with {actual}. The catalogue and the "
                "published model have diverged, and `kv_cache_bytes` is "
                "computed from the catalogue."
            )

    # `head_dim` is optional in mlx_lm's argument dataclasses -- some families
    # derive it -- so it is checked against its derivation when absent rather
    # than skipped, which would leave a factor of the bound unverified.
    head_dim = getattr(args, "head_dim", None)
    if head_dim is None:
        # A real derivation, not an absence: some families compute it rather
        # than store it, and `hidden_size // num_attention_heads` is that
        # computation. If neither is there either, we are back to the unchecked
        # case above and it raises for the same reason.
        hidden = getattr(args, "hidden_size", None)
        heads = getattr(args, "num_attention_heads", None)
        head_dim = hidden // heads if hidden and heads else None
    if head_dim is None:
        raise ValueError(
            f"Rewriter {spec.key!r} neither declares nor allows deriving "
            "head_dim, one of the three numbers `kv_cache_bytes` multiplies."
        )
    if head_dim != spec.head_dim:
        raise ValueError(
            f"Rewriter {spec.key!r} declares head_dim={spec.head_dim} but "
            f"{spec.repo} loads with {head_dim}."
        )

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Rewriter {spec.key!r} has no chat template. `build_messages` "
            "would be flattened into a bare completion prompt, and "
            "`enable_thinking=False` -- one of the two barriers against "
            "reasoning reaching a diffusion model -- would have nowhere to "
            "apply."
        )


def load_rewriter(spec: RewriterSpec, *, allow_download: bool = True) -> tuple[Any, Any]:
    """Build `spec`'s model and tokenizer, checked against the catalogue.

    Blocking, and called on the engine's worker thread with the snapshot in
    `rewriting` -- never on the event loop.
    """
    mlx_lm = require_mlx_lm()
    repo = resolve_repo(spec, allow_download=allow_download)
    model, tokenizer = mlx_lm.load(repo)
    verify_loaded(model, tokenizer, spec)
    return model, tokenizer
