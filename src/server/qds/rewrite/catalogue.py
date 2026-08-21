"""The rewriter catalogue: which small LLM this server uses to expand prompts.

Every value here was read off the published `config.json` of the repository
named below and is re-checked against the loaded model by
`weights.verify_loaded`. That cross-check is not decoration: the KV-cache bound
in `MAX_PROMPT_TOKENS`/`MAX_NEW_TOKENS` is computed from `num_hidden_layers`,
`num_key_value_heads` and `head_dim`, so a catalogue entry that drifts from its
checkpoint would turn the engine's third-slot memory argument into a fiction.
`tests/test_rewrite.py` pins the arithmetic.

This module must not import mlx, mlx_lm, mflux, torch or huggingface_hub.
`fetch` reads it on the `--status` path, which `tests/test_cli.py` holds to
importing none of those, and `app` reads it at start-up to publish
`/v1/capabilities`. The catalogue describes; it does not touch anything.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Upper bound on what one rewrite may cost, in megabytes: weights, plus the KV
#: cache the token bounds allow, plus one step's logits.
#:
#: This is the bound that makes `ModelEngine`'s third slot safe to exist at all
#: (see its module docstring). As with `upscale.catalogue.MAX_WEIGHTS_MB`, the
#: exception to "one live model" was argued from measured, bounded, transient
#: cost -- not from "a small model is small" -- so it is enforced here, at
#: import, rather than trusted. Raising it is a decision about the engine's
#: memory invariant, not a catalogue edit.
#:
#: The number is what an *upscale* already costs transiently beside a resident
#: diffusion model: 1520 MB of MLX activations plus 1110 MB of assembled image
#: on the host, from `ModelEngine`'s docstring. A rewrite must cost less than
#: the exception this engine already grants, or the third slot needs an argument
#: of its own rather than borrowing the second's.
#:
#: Measured against it: the shipped entry is 2263 MB resident and its bounded
#: footprint is 2386 MB.
#:
#: *Rejected alternative, recorded because it was what shipped first:* a bound
#: on `size_mb` alone (`MAX_REWRITER_WEIGHTS_MB = 1200.0`). One line, but it
#: enforces a proxy for the property that actually matters, and it leaves a
#: loophole -- a deep model with few parameters passes a weights bound and blows
#: the KV budget. The footprint form self-adjusts across architectures.
MAX_REWRITER_FOOTPRINT_MB = 2630.0

#: Longest prompt the rewriter will accept, in tokens.
#:
#: Half of the execution bound (`MAX_NEW_TOKENS` is the other half): together
#: they cap the KV cache, which is the only part of a decode whose size depends
#: on the input. A prompt past it is *refused*, never truncated -- a very long
#: prompt is a signal about what the user wants, not an input to mutilate
#: silently.
#:
#: Enforced in *tokens*, by `ModelEngine._rewrite_sync`, against the fully
#: templated text -- system prompt included, because the KV cache holds that
#: too. That is the bound; everything else is triage.
#:
#: It has to live there rather than at admission, and the reason is worth
#: stating because an earlier version got it wrong. Admission has no tokenizer,
#: so it counted `len(prompt.split())` -- and a Chinese or Japanese prompt has
#: no spaces in it, so one of 200,000 characters counted as a single word and
#: sailed past this bound. Not an exotic input: the feature's clearest win is on
#: prompts that are not in English. A word count is not an approximation of a
#: token count for those scripts; it is unrelated to it.
#:
#: Nothing about this is about quality. The ceiling that decides whether a
#: prompt is worth rewriting at all is `RewriteSettings.word_ceiling`, and the
#: two are independent on purpose.
MAX_PROMPT_TOKENS = 512

#: Longest prompt admission will accept, in characters.
#:
#: Triage for `MAX_PROMPT_TOKENS`, which cannot be checked without a tokenizer.
#: Deliberately loose: at four characters per token this is about 512 tokens of
#: English, and at roughly one character per token about 2048 of Chinese -- so
#: it does not *establish* the token bound for every script, and does not claim
#: to. What it does is refuse a prompt that could never fit, at the boundary
#: where the refusal can name a parameter, before a gigabyte of weights loads to
#: discover the same thing.
MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS * 4

#: Longest rewrite the model may produce, in tokens.
#:
#: Measured against what the shipped entry actually emits under the shipped
#: system prompt: 120 words median, 179 at the observed maximum, so roughly 245
#: tokens. Nothing truncates at 320.
#:
#: The margin is not decoration. Removing the length target from the system
#: prompt -- an intermediate version -- put four of eighteen outputs over this
#: bound, and one rejected candidate put eleven of eighteen over it. The bound
#: exists to make a runaway decode terminate; `prompt.trim_to_last_clause` is
#: what makes hitting it survivable rather than silently corrupting a prompt.
MAX_NEW_TOKENS = 320

#: Quantisations a catalogue entry may declare.
#:
#: Without this, the footprint bound is bypassable by format rather than
#: by decision: the same Qwen3-1.7B checkpoint is 968 MB at 4 bits and about
#: 3.4 GB in bf16, and only the first fits the bound this module enforces.
ALLOWED_BITS: tuple[int, ...] = (4, 8)


def kv_cache_bytes(spec: RewriterSpec, *, context_tokens: int | None = None) -> int:
    """Bytes the KV cache holds at `context_tokens`, bf16.

    This is the whole execution-bound argument in one function, which is why it
    is here rather than inline in the engine: two tensors (K and V), two bytes
    each, for every layer, every key/value head and every head dimension, over
    the longest context the two token bounds allow.

    Derived from the catalogue rather than measured, and `verify_loaded`
    refuses a checkpoint whose architecture contradicts the catalogue -- so the
    number is an assertion about what will be allocated, not an estimate.
    """
    if context_tokens is None:
        context_tokens = MAX_PROMPT_TOKENS + MAX_NEW_TOKENS
    return context_tokens * spec.num_hidden_layers * spec.num_key_value_heads * spec.head_dim * 2 * 2


@dataclass(frozen=True)
class RewriterSpec:
    """One small causal LLM and everything needed to bound running it.

    Deliberately *not* a `registry.ModelSpec`, for the same reason
    `UpscalerSpec` is not: steps, guidance, schedulers, prompt formats, image
    sizes and an edit variant have no meaning for a text decoder, and
    everything that reads a `ModelSpec` would read invented values for them as
    facts about a generative image model.
    """

    key: str
    display_name: str
    #: Hugging Face repository. Catalogue data rather than a constant so that
    #: switching source is an edit here, not in `weights`.
    repo: str
    #: Quantisation the repository declares in `config.json`. Checked against
    #: the file by `verify_loaded`, and constrained to `ALLOWED_BITS` below.
    bits: int
    #: Parameter count in billions, as the publisher states it. Used only to
    #: sanity-check `size_mb` against `bits`; see `__post_init__`.
    params_b: float
    #: Resident size, megabytes, measured rather than computed -- 968 MB for
    #: the shipped entry, against the 900 MB its parameter count alone would
    #: predict, because embeddings and norms are not quantised.
    size_mb: float
    #: Architecture, read from `config.json`. These four are what bound the KV
    #: cache, and `weights.verify_loaded` refuses a checkpoint that contradicts
    #: them -- see `kv_cache_bytes`.
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    #: Upstream licence, reported as a fact with its source. As with
    #: `ModelSpec.license`, this states what the publisher declares; it is not
    #: legal advice, and a re-packager's declaration is not the upstream's.
    license: str
    #: Whether the model has a hybrid "thinking" mode that must be turned off.
    #:
    #: A field rather than an assumption because the two barriers against it
    #: cost something: `enable_thinking=False` on the chat template, and
    #: `prompt.strip_thinking` refusing a malformed block. Measured on the
    #: shipped entry: 0 leaks in 281 generations with both in place.
    hybrid_thinking: bool = True

    def footprint_mb(self) -> float:
        """What one rewrite costs at its bounds: weights + KV cache + logits.

        The whole third-slot argument in one method, which is why it is here
        rather than inline: the weights are resident for the decode, the KV
        cache is the only part that grows with the input and is capped by the
        two token bounds, and the logits are one step's worth. Nothing else in
        a decode scales.
        """
        return self.size_mb + kv_cache_bytes(self) / 1e6 + self.vocab_size * 2 / 1e6

    def __post_init__(self) -> None:
        if self.bits not in ALLOWED_BITS:
            raise ValueError(
                f"Rewriter {self.key!r} declares {self.bits}-bit weights; only "
                f"{ALLOWED_BITS} are allowed. An unquantised checkpoint would "
                f"clear {MAX_REWRITER_FOOTPRINT_MB} MB by format rather than by "
                "decision."
            )
        footprint = self.footprint_mb()
        if footprint > MAX_REWRITER_FOOTPRINT_MB:
            raise ValueError(
                f"Rewriter {self.key!r} has a bounded footprint of {footprint:.0f} MB "
                f"({self.size_mb} MB of weights plus its KV cache), over the "
                f"{MAX_REWRITER_FOOTPRINT_MB} MB the engine's third slot borrows "
                "from what an upscale already costs. Raising it is a decision "
                "about `ModelEngine`'s memory invariant, not a catalogue edit."
            )
        # Redundant with `bits` and `params_b` by construction, and that is the
        # point: it is a checked assertion about the artifact, not a parameter.
        # The window is wide because what is *not* quantised -- embeddings,
        # norms, and on some publications the lm_head -- varies by repacking.
        predicted = self.params_b * 1000.0 * self.bits / 8.0
        if not predicted <= self.size_mb <= predicted * 1.6:
            raise ValueError(
                f"Rewriter {self.key!r} declares {self.size_mb} MB for "
                f"{self.params_b}B at {self.bits} bits, outside the "
                f"{predicted:.0f}-{predicted * 1.6:.0f} MB this implies."
            )
        if self.head_dim <= 0 or self.num_key_value_heads <= 0 or self.num_hidden_layers <= 0:
            raise ValueError(f"Rewriter {self.key!r} has a non-positive architecture parameter.")


#: Qwen3-4B-Instruct-2507, 4-bit, as re-packaged by mlx-community.
#:
#: Chosen on measured images, against the 1.7B that shipped first and against
#: Ministral-3-3B, on a fixed set of six prompts at three seeds each.
#:
#: The 1.7B is kept below as a smaller option, not deleted, because it is a
#: real trade someone may want: 968 MB and a sub-second decode. But it does not
#: do this job. Asked for a rich prompt it returns a **46-word median** and
#: falls into a degenerate loop on a simple subject -- "a white ceramic bowl
#: with a cloudy sky" repeated until the token bound cut it off, twice in
#: eighteen. Its ceiling is capacity, not instruction: the same prompt on the
#: 4B returns 126 words median with nothing truncated.
#:
#: Ministral-3-3B-Instruct-2512-4bit was measured too, on the reputation that
#: Mistral writes better, and it lost on mechanism rather than taste. It ignores
#: the length instruction (207-word median against a stated 160 ceiling), so
#: **eleven of eighteen outputs were truncated mid-clause**; it emits Markdown
#: emphasis that would reach a diffusion model as literal asterisks; and it
#: replaces the subject -- "a bowl of ramen" came back as a copper ladle on a
#: chopping block, which is the one thing `prompt.py`'s framing rule exists to
#: prevent. It is also 2745 MB for a 3B, because the extra bytes are a vision
#: tower that would be loaded, never used, and paid for on every Enhance; its
#: footprint of 2834 MB exceeds the bound above on its own.
#:
#: No entry may be given examples: shown two few-shot turns, the 1.7B
#: reproduces one of them verbatim for any input it cannot read -- "закат над
#: горами" returned the diving-helmet exemplar, identically across seeds -- so
#: `prompt.build_messages` is zero-shot by construction and
#: `tests/test_rewrite.py` pins that.
SPECS: tuple[RewriterSpec, ...] = (
    RewriterSpec(
        key="qwen3-4b-2507-4bit",
        display_name="Qwen3 4B Instruct 2507 (4-bit)",
        repo="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        bits=4,
        params_b=4.0,
        size_mb=2263.0,
        num_hidden_layers=36,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        license="Apache-2.0 (upstream Qwen/Qwen3-4B-Instruct-2507), as declared by the re-packager",
        # An Instruct checkpoint: no hybrid thinking mode, so the chat template
        # takes no `enable_thinking`. Passing it would raise in Jinja rather
        # than be ignored, which is why this is a field and not an assumption.
        hybrid_thinking=False,
    ),
    RewriterSpec(
        key="qwen3-1.7b-4bit",
        display_name="Qwen3 1.7B (4-bit)",
        repo="mlx-community/Qwen3-1.7B-4bit",
        bits=4,
        params_b=1.7,
        size_mb=968.0,
        num_hidden_layers=28,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        license="Apache-2.0 (upstream Qwen/Qwen3-1.7B), as declared by the re-packager",
    ),
)

_BY_KEY: dict[str, RewriterSpec] = {spec.key: spec for spec in SPECS}

#: Catalogue keys, in presentation order.
KEYS: tuple[str, ...] = tuple(spec.key for spec in SPECS)


def by_key(key: str) -> RewriterSpec | None:
    """The spec for `key`, or `None` if the catalogue does not have it."""
    return _BY_KEY.get(key)
