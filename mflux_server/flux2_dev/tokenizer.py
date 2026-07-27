"""Tokenisation des prompts FLUX.2-dev.

`LanguageTokenizer` de mflux ne convient pas : avec `use_chat_template=True` il
envoie `[{"role": "user", "content": prompt}]` et `add_generation_prompt=True`
(mflux/models/common/tokenizer/tokenizer.py:86-92), alors que FLUX.2-dev attend
une conversation **system + user**, des contenus en listes de parts typées, et
`add_generation_prompt=False`. Un décalage ici ne casse rien visiblement : le
modèle génère simplement des images qui ignorent le prompt.

`TokenizerDefinition.encoder_class` permet de brancher une classe maison, que
`TokenizerLoader._create_tokenizer` instancie en `(tokenizer=…, max_length=…)`.
"""

from __future__ import annotations

import mlx.core as mx
from mflux.models.common.tokenizer import BaseTokenizer, TokenizerOutput

#: Repris à l'identique de diffusers
#: (src/diffusers/pipelines/flux2/system_messages.py), lui-même repris de
#: black-forest-labs/flux2. Le retour à la ligne après « object » est dans la
#: source : il change la tokenisation, donc on le conserve tel quel.
SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object\nattribution and "
    "actions without speculation."
)


def format_messages(prompt: str, system_message: str = SYSTEM_MESSAGE) -> list[dict]:
    """Reproduit le `format_input` de `Flux2Pipeline`."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        # `[IMG]` est le token d'image du template : le laisser dans un prompt
        # texte décalerait les positions.
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
