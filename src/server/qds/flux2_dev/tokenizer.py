"""Tokenization of FLUX.2-dev prompts.

mflux's `LanguageTokenizer` does not fit: with `use_chat_template=True` it sends
`[{"role": "user", "content": prompt}]` and `add_generation_prompt=True`
(mflux/models/common/tokenizer/tokenizer.py:86-92), whereas FLUX.2-dev expects a
**system + user** conversation, contents as lists of typed parts, and
`add_generation_prompt=False`. A mismatch here breaks nothing visibly: the model
simply generates images that ignore the prompt.

`TokenizerDefinition.encoder_class` lets us plug in a custom class, which
`TokenizerLoader._create_tokenizer` instantiates as `(tokenizer=…, max_length=…)`.
"""

from __future__ import annotations

import mlx.core as mx
from mflux.models.common.tokenizer import BaseTokenizer, TokenizerOutput

#: Copied verbatim from diffusers
#: (src/diffusers/pipelines/flux2/system_messages.py), itself taken from
#: black-forest-labs/flux2. The line break after "object" is in the source: it
#: changes tokenization, so it is preserved exactly.
SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object\nattribution and "
    "actions without speculation."
)


def format_messages(prompt: str, system_message: str = SYSTEM_MESSAGE) -> list[dict]:
    """Mirror `Flux2Pipeline`'s `format_input`."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        # `[IMG]` is the template's image token: leaving it in a text prompt
        # would shift the positions.
        {"role": "user", "content": [{"type": "text", "text": prompt.replace("[IMG]", "")}]},
    ]


class Flux2DevTokenizer(BaseTokenizer):
    def __init__(self, tokenizer, max_length: int = 512):
        super().__init__(tokenizer, max_length)

    def tokenize(
        self,
        prompt: str | list[str],
        images=None,
        max_length: int | None = None,
        **kwargs,
    ) -> TokenizerOutput:
        max_length = max_length or self.max_length
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [p if p is not None else "" for p in prompts]

        encoded = self.tokenizer.apply_chat_template(
            [format_messages(p) for p in prompts],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return TokenizerOutput(
            input_ids=mx.array(encoded["input_ids"]),
            attention_mask=mx.array(encoded["attention_mask"]),
        )
