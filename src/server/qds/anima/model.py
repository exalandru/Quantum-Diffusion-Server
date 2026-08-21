"""Anima: the Cosmos DiT and its text adapter, driven the way its pipeline drives them.

The reference for this loop is diffusers' `AnimaModularPipeline`
(`diffusers/modular_pipelines/anima/`), and the departures from it are only those
mflux's own conventions require -- MLX arrays, mflux's scheduler, and the
callback contract the server's engine registers against.

Four components, from two repositories, wired here:

    T5 tokenizer  ->  ids  ------------------.
                                             v
    Qwen tokenizer -> ids -> Qwen3-0.6B -> adapter -> text embeddings -> DiT -> VAE

The Qwen3 tower is mflux's own -- the one it ships for FLUX.2 klein -- because
Anima uses Qwen3-0.6B unmodified and that class is parameterised for it. The DiT
and the adapter are this package's, since mflux has neither.

`mx.compile` is deliberately not applied: at 2B the loop is already fast, and the
value of matching the reference step for step while the port is young is higher
than the throughput a compiled graph would add.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mlx import nn

from qds.anima import config as anima_config
from qds.anima.conditioner import AnimaTextConditioner
from qds.anima.latent_creator import AnimaLatentCreator
from qds.anima.transformer import AnimaTransformer
from qds.anima.weights import (
    AnimaWeightDefinition,
    anima_weight_definition,
    component_subset,
    verify_loaded,
)


class Anima(nn.Module):
    # These annotations are load-bearing beyond documentation: the converter reads
    # a family's module classes off them (`prequantize._module_class`).
    vae: QwenVAE
    transformer: AnimaTransformer
    text_encoder: Qwen3TextEncoder
    text_conditioner: AnimaTextConditioner

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        weight_file: str | None = None,
    ):
        super().__init__()
        self.model_config = model_config or anima_config.anima_model_config()
        # Which of the published checkpoints this instance is. Aesthetic and Turbo
        # are one architecture and two trainings, so the variant is a file rather
        # than a class — the catalogue row names it, and everything else here is
        # identical between them.
        self.weight_file = weight_file or anima_config.DEFAULT_WEIGHT_FILE
        definition = anima_weight_definition(self.weight_file)
        self.prompt_cache: dict[tuple[str, str | None, float], tuple[mx.array, mx.array | None]] = {}
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        # LoRA is not supported; the server passes these for every model.
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

        path = model_path or self.model_config.model_name
        companion = anima_config.COMPANION_REPO

        # Two loads, because the components live in two repositories and
        # `WeightLoader` reads every component of the definition it is given
        # against the one path it is given. Each side is asked only for what it
        # actually holds; the results are merged before anything is applied.
        weights = WeightLoader.load(
            weight_definition=component_subset(_MAIN_REPO_COMPONENTS, definition), model_path=path
        )
        companion_weights = WeightLoader.load(
            weight_definition=component_subset(_COMPANION_COMPONENTS, definition),
            model_path=companion,
        )
        weights.components.update(companion_weights.components)
        self.tokenizers = TokenizerLoader.load_all(
            definitions=AnimaWeightDefinition.get_tokenizers(),
            model_path=companion,
        )

        self.vae = QwenVAE()
        self.transformer = AnimaTransformer(**self.model_config.transformer_overrides)
        self.text_encoder = Qwen3TextEncoder(**self.model_config.text_encoder_overrides)
        self.text_conditioner = AnimaTextConditioner(**anima_config.CONDITIONER_OVERRIDES)

        models = {
            "vae": self.vae,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_conditioner": self.text_conditioner,
        }
        # Before applying, not after: weights are applied non-strictly, so a
        # component whose names this package does not recognise would arrive
        # empty, apply cleanly, and leave a randomly-initialised module behind.
        verify_loaded(weights.components, models)

        self.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=definition,
            models=models,
        )

    # ── Generation ─────────────────────────────────────────────────────────

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.5,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
    ) -> GeneratedImage:
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=self._resolve_scheduler(scheduler),
        )
        # Anima shifts its sigmas by a constant 3.0 rather than by image sequence
        # length. mflux expresses shift as `mu` over an exponential time shift,
        # which is the same function at `exp(mu)`; `tests/test_anima.py` proves
        # the identity. Without this the schedule is the unshifted linspace and
        # every image is subtly wrong rather than obviously broken.
        config.scheduler.set_mu(math.log(anima_config.SIGMA_SHIFT))

        embeds, negative_embeds = self._encode_prompts(prompt, negative_prompt, guidance)
        latents = self._prepare_latents(seed=seed, config=config)
        mx.eval(latents, embeds)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                sigma = config.scheduler.sigmas[t]
                noise = self._predict(latents, sigma, embeds)
                if negative_embeds is not None:
                    uncond = self._predict(latents, sigma, negative_embeds)
                    noise = uncond + guidance * (noise - uncond)
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                ) from None

        ctx.after_loop(latents)

        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    @staticmethod
    def _resolve_scheduler(scheduler: str | None) -> str:
        """Anima is flow-matched; `"linear"` is the catalogue's word for a default."""
        if scheduler in (None, "linear", "flow_match_euler_discrete"):
            return "flow_match_euler_discrete"
        raise ValueError(
            f"Unknown Anima scheduler {scheduler!r}. Expected 'flow_match_euler_discrete'."
        )

    def _predict(self, latents: mx.array, sigma: mx.array, embeds: mx.array) -> mx.array:
        """One transformer pass. The DiT is 3D, so a still image is one frame."""
        hidden = mx.expand_dims(latents, axis=2)
        noise = self.transformer(
            hidden_states=hidden,
            # The pipeline divides the timestep by `num_train_timesteps` before
            # the call, which leaves exactly the sigma.
            timestep=mx.array([sigma], dtype=mx.float32).reshape(1),
            encoder_hidden_states=embeds,
        )
        return mx.squeeze(noise, axis=2)

    def _encode_prompts(
        self, prompt: str, negative_prompt: str | None, guidance: float
    ) -> tuple[mx.array, mx.array | None]:
        key = (prompt, negative_prompt, guidance)
        cached = self.prompt_cache.get(key)
        if cached is not None:
            return cached

        embeds = self._condition(prompt)
        negative_embeds = None
        if guidance != 1.0:
            # An empty negative prompt still gets encoded: classifier-free
            # guidance needs an unconditional branch, and " " is what the other
            # families use for it.
            text = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            negative_embeds = self._condition(text)

        mx.eval(embeds)
        if negative_embeds is not None:
            mx.eval(negative_embeds)
        self.prompt_cache[key] = (embeds, negative_embeds)
        return embeds, negative_embeds

    def _condition(self, prompt: str) -> mx.array:
        """Qwen3 states plus T5 ids, through the adapter, into DiT context."""
        qwen_ids, qwen_mask = self._tokenize("qwen3", prompt)
        t5_ids, t5_mask = self._tokenize("t5", prompt)

        hidden, _ = self.text_encoder(input_ids=qwen_ids, attention_mask=qwen_mask)
        # Padding positions are zeroed before the adapter sees them, as the
        # pipeline does; the adapter also masks, and the two together are what
        # keeps a short prompt from being conditioned on its own padding.
        hidden = hidden * qwen_mask.astype(hidden.dtype)[..., None]

        return self.text_conditioner(
            source_hidden_states=hidden,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=qwen_mask,
        )

    def _tokenize(self, name: str, prompt: str) -> tuple[mx.array, mx.array]:
        encoded = self.tokenizers[name].tokenize(prompt)
        return mx.array(encoded.input_ids), mx.array(encoded.attention_mask)

    def _prepare_latents(self, *, seed: int, config: Config) -> mx.array:
        noise = AnimaLatentCreator.create_noise(seed, config.height, config.width)
        if config.image_path is None or not config.image_strength:
            return noise

        image = ImageUtil.to_array(
            ImageUtil.scale_to_dimensions(
                ImageUtil.load_image(config.image_path).convert("RGB"),
                config.width,
                config.height,
            )
        )
        clean = self.vae.encode(image)
        if clean.ndim == 5:
            clean = mx.squeeze(clean, axis=2)
        return LatentCreator.add_noise_by_interpolation(
            clean=clean,
            noise=noise,
            sigma=config.scheduler.sigmas[config.init_time_step],
        )


#: Which repository holds which component. `AnimaWeightDefinition` describes the
#: model as a whole -- all four, which is what the applier maps against -- and
#: these two names split that description by where the files actually are.
_MAIN_REPO_COMPONENTS = ("transformer", "text_conditioner", "text_encoder")
_COMPANION_COMPONENTS = ("vae",)
