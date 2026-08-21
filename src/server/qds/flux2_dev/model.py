"""FLUX.2 [dev]: mflux's transformer and VAE, with our own Mistral3 text encoder.

mflux's `Flux2Initializer` hardwires `Qwen3TextEncoder` and
`Flux2KleinWeightDefinition`, hence this class rather than a call to
`Flux2Klein`. The denoising loop faithfully reproduces
`Flux2Klein.generate_image` (mflux 0.19.0) — that is the reference to compare
against if results ever diverge.

Two deliberate differences from klein:

* **embedded guidance.** FLUX.2-dev is guidance-distilled: the scalar goes
  through `time_guidance_embed`, there is no Classifier-Free Guidance and hence
  only one transformer pass per step (klein passes `guidance=None` to the
  transformer and does real CFG against a negative prompt).
* **no `mx.compile`.** At 32B the compiled graph trips Metal GPU timeouts; the
  uncompiled path is slower but reliable.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mlx import nn

from qds.flux2_dev.config import (
    MAX_SEQUENCE_LENGTH,
    TEXT_ENCODER_OUT_LAYERS,
    flux2_dev_model_config,
)
from qds.flux2_dev.mistral3 import Mistral3TextEncoder
from qds.flux2_dev.weights import Flux2DevWeightDefinition


class Flux2Dev(nn.Module):
    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Mistral3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        self.model_config = model_config or flux2_dev_model_config()
        self.prompt_cache: dict[str, tuple[mx.array, mx.array]] = {}
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        # LoRA unsupported: the server passes `lora_paths` for no model.
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

        path = model_path or self.model_config.model_name
        weights = WeightLoader.load(weight_definition=Flux2DevWeightDefinition, model_path=path)
        self.tokenizers = TokenizerLoader.load_all(
            definitions=Flux2DevWeightDefinition.get_tokenizers(),
            model_path=path,
        )

        self.vae = Flux2VAE()
        self.transformer = Flux2Transformer(**self.model_config.transformer_overrides)
        self.text_encoder = Mistral3TextEncoder(**self.model_config.text_encoder_overrides)

        self.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Flux2DevWeightDefinition,
            models={
                "vae": self.vae,
                "transformer": self.transformer,
                "text_encoder": self.text_encoder,
            },
        )

    # ── Generation ─────────────────────────────────────────────────────────

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
    ) -> GeneratedImage:
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )

        prompt_embeds, text_ids = self._encode_prompt(prompt)
        latents, latent_ids, latent_height, latent_width = self._prepare_generation_latents(
            seed=seed,
            config=config,
        )

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)
        guidance_embed = self._guidance_embed(config.guidance)

        for t in config.time_steps:
            try:
                noise = predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    guidance=guidance_embed,
                    timestep=config.scheduler.timesteps[t],
                )
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
                )
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                ) from None

        ctx.after_loop(latents)

        packed_latents = latents.reshape(
            latents.shape[0], latent_height, latent_width, latents.shape[-1]
        ).transpose(0, 3, 1, 2)
        decoded = self.vae.decode_packed_latents(packed_latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=None,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def _guidance_embed(self, guidance: float) -> mx.array:
        """Scale guidance the way the guidance embedder expects.

        `Flux2Transformer.__call__` only multiplies by 1000 when
        `max(guidance) <= 1.0` (flux2/.../transformer.py:91) — a heuristic
        written for the timestep and never exercised upstream, since no shipped
        mflux model enables `guidance_embeds` on the FLUX.2 transformer. The
        FLUX.1 path does do `guidance * num_train_steps`
        (flux/.../transformer.py:155). So we pre-multiply here, which makes the
        heuristic a no-op (×1.0) and restores the right value.
        """
        return mx.array(guidance * self.model_config.num_train_steps, dtype=ModelConfig.precision)

    def _encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        cached = self.prompt_cache.get(prompt)
        if cached is not None:
            return cached

        # `Flux2PromptEncoder.encode_prompt` is generic: all it requires of the
        # encoder is a `get_prompt_embeds(input_ids, attention_mask,
        # hidden_state_layers)`, which Mistral3TextEncoder exposes like Qwen3.
        embeds = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["mistral3"],
            text_encoder=self.text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            text_encoder_out_layers=TEXT_ENCODER_OUT_LAYERS,
        )
        self.prompt_cache[prompt] = embeds
        return embeds

    def _prepare_generation_latents(
        self,
        *,
        seed: int,
        config: Config,
    ) -> tuple[mx.array, mx.array, int, int]:
        if config.image_path is None or config.image_strength is None or config.image_strength <= 0.0:
            return Flux2LatentCreator.prepare_packed_latents(
                seed=seed,
                height=config.height,
                width=config.width,
                batch_size=1,
            )
        return self._prepare_img2img_latents(seed=seed, config=config)

    def _prepare_img2img_latents(self, *, seed: int, config: Config) -> tuple[mx.array, mx.array, int, int]:
        noise_latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )

        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
        encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
        encoded = Flux2Klein._match_latent_spatial_size(
            encoded=encoded,
            target_height=latent_height * 2,
            target_width=latent_width * 2,
        )
        encoded = Flux2LatentCreator.patchify_latents(encoded)
        encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=self.vae)
        clean_latents = Flux2LatentCreator.pack_latents(encoded)

        sigma = config.scheduler.sigmas[config.init_time_step]
        latents = LatentCreator.add_noise_by_interpolation(
            clean=clean_latents,
            noise=noise_latents,
            sigma=sigma,
        )
        return latents, latent_ids, latent_height, latent_width

    @staticmethod
    def _predict(transformer):
        def predict(
            *,
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            guidance: mx.array,
            timestep: mx.array,
        ) -> mx.array:
            return transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=guidance,
            )

        # No mx.compile here, unlike Flux2Klein: at 32B the compiled graph can
        # exceed Metal's GPU watchdog.
        return predict

    # ── Saving ─────────────────────────────────────────────────────────────

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Flux2DevWeightDefinition,
        )
