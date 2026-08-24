"""SD 3.5: the five components, driven the way `StableDiffusion3Pipeline` drives them.

The reference for this loop is diffusers'
`StableDiffusion3Pipeline.__call__`/`encode_prompt`, and the departures from it are
only those mflux's conventions require — MLX arrays, mflux's scheduler and `Config`,
and the callback contract the server's engine registers against.

Five components, one repository:

    CLIP-L  ─ penultimate states (768) ─┐
                                        ├─ concat ─ zero-pad to 4096 ─┐
    CLIP-G  ─ penultimate states (1280) ┘                             ├─ context
    T5-XXL  ─ states (4096) ──────────────────────────────────────────┘
    CLIP-L pooled (768) ─┐
                         ├─ concat (2048) ─ pooled conditioning ─ MMDiT ─ VAE
    CLIP-G pooled (1280) ┘

Real classifier-free guidance: above guidance 1.0 the loop runs two transformer passes
per step. Large Turbo is distilled to run at 1.0, where the unconditional branch is
skipped entirely and a step costs one pass.

`mx.compile` is deliberately not applied: matching the reference step for step while
the port is young is worth more than the throughput a compiled graph would add.
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
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mlx import nn

from qds.sd35 import conditioner
from qds.sd35 import config as sd35_config
from qds.sd35.clip import SD35ClipG, SD35ClipL
from qds.sd35.latent_creator import SD35LatentCreator
from qds.sd35.scheduler import SCHEDULER_PATH
from qds.sd35.transformer import SD35Transformer
from qds.sd35.vae import SD35VAE
from qds.sd35.weights import SD35WeightDefinition, verify_loaded

#: What the catalogue calls this family's scheduler, and mflux's word for "the
#: default". Both resolve to `qds.sd35.scheduler.SD35FlowMatchScheduler`.
_ACCEPTED_SCHEDULERS = (None, "linear", "flow_match_euler_discrete")


class SD35(nn.Module):
    # These annotations are load-bearing beyond documentation: the converter reads a
    # family's module classes off them (`prequantize._module_class`), and
    # `test_components.py` holds all five to being constructible on their own.
    transformer: SD35Transformer
    text_encoder_3: T5Encoder
    text_encoder_2: SD35ClipG
    text_encoder: SD35ClipL
    vae: SD35VAE

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        self.model_config = model_config or sd35_config.sd35_medium_model_config()
        # Which of the three releases this instance is comes entirely from the
        # `ModelConfig` a catalogue row named. Unlike Anima's two checkpoints, these
        # are three *shapes* — Medium is 24 blocks of 1536, the large pair 38 of 2432 —
        # and `transformer_overrides` carries that. The weight definition is the same
        # for all three: the components sit in the same subdirectories, and how many
        # shards the transformer is written to is a question the directory answers.
        definition = SD35WeightDefinition
        self.prompt_cache: dict[tuple[str, str | None, float], tuple] = {}
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        # LoRA is not supported; the server passes these for every model.
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

        path = model_path or self.model_config.model_name
        weights = WeightLoader.load(weight_definition=definition, model_path=path)
        self.tokenizers = TokenizerLoader.load_all(
            definitions=definition.get_tokenizers(), model_path=path
        )

        self.transformer = SD35Transformer(**self.model_config.transformer_overrides)
        self.text_encoder = SD35ClipL(**self.model_config.text_encoder_overrides)
        self.text_encoder_2 = SD35ClipG()
        self.text_encoder_3 = T5Encoder()
        self.vae = SD35VAE()

        models = {
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "text_encoder_3": self.text_encoder_3,
            "vae": self.vae,
        }
        # Before applying, not after: weights are applied non-strictly, and three of
        # these five components are read with no rename table, so a source whose names
        # this package does not recognise would arrive empty, apply cleanly, and leave
        # a randomly-initialised module behind.
        #
        # Skipped for a saved artifact, and only there: a quantized component stores
        # three tensors per linear layer (weight, scales, biases) against the module's
        # one, so counting parameters would refuse the very artifacts this family's
        # conversion path produces. Their structure is guaranteed by the converter and
        # by `WeightLoader`, which refuses a saved model whose index names a shard that
        # is not on disk.
        if weights.meta_data.quantization_level is None:
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
        num_inference_steps: int = 28,
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

        context, pooled, negative = self._encode_prompts(prompt, negative_prompt, guidance)
        latents = self._prepare_latents(seed=seed, config=config)
        mx.eval(latents, context, pooled)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # The transformer takes the raw flow-match timestep — sigma scaled by
                # `num_train_timesteps` — not the sigma. Sliced rather than indexed so
                # it stays a `[1]` array and no device sync happens in the loop.
                timestep = config.scheduler.timesteps[t : t + 1]
                noise = self._predict(latents, timestep, context, pooled)
                if negative is not None:
                    uncond = self._predict(latents, timestep, *negative)
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
        """SD 3.5 is flow-matched; `"linear"` is the catalogue's word for a default.

        Resolves to this package's own scheduler rather than mflux's, because mflux's
        sigma spacing is not SD 3.5's — see `qds/sd35/scheduler.py`.
        """
        if scheduler in _ACCEPTED_SCHEDULERS or scheduler == SCHEDULER_PATH:
            return SCHEDULER_PATH
        raise ValueError(
            f"Unknown SD 3.5 scheduler {scheduler!r}. Expected 'flow_match_euler_discrete'."
        )

    def _predict(
        self, latents: mx.array, timestep: mx.array, context: mx.array, pooled: mx.array
    ) -> mx.array:
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=context,
            pooled_projections=pooled,
        )

    def _encode_prompts(
        self, prompt: str, negative_prompt: str | None, guidance: float
    ) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array] | None]:
        key = (prompt, negative_prompt, guidance)
        cached = self.prompt_cache.get(key)
        if cached is not None:
            return cached

        context, pooled = self._condition(prompt)
        negative = None
        # `> 1.0`, exactly as the reference decides it
        # (`do_classifier_free_guidance = guidance_scale > 1`). Not `!= 1.0`: SD 3.5
        # Large Turbo's published default is guidance **0.0**, which under `!=` would
        # buy a second transformer pass per step and then scale the difference by
        # zero — twice the cost for the unconditional image.
        if guidance > 1.0:
            # An empty negative prompt still gets encoded: classifier-free guidance
            # needs an unconditional branch, and the reference pipeline encodes "" for
            # it rather than skipping the pass.
            negative = self._condition(negative_prompt if negative_prompt else "")

        mx.eval(context, pooled)
        if negative is not None:
            mx.eval(*negative)
        self.prompt_cache[key] = (context, pooled, negative)
        return context, pooled, negative

    def _condition(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Three encoders into the two tensors the transformer reads."""
        clip_l_states, clip_l_pooled = self.text_encoder(self._tokenize("clip_l", prompt))
        clip_g_states, clip_g_pooled = self.text_encoder_2(self._tokenize("clip_g", prompt))
        t5_states = self.text_encoder_3(self._tokenize("t5", prompt))
        return (
            conditioner.joint_context(clip_l_states, clip_g_states, t5_states),
            conditioner.pooled_projections(clip_l_pooled, clip_g_pooled),
        )

    def _tokenize(self, name: str, prompt: str) -> mx.array:
        # Padded to the tokenizer's full length, never `longest`: CLIP's pooled vector
        # is read at the end-of-text position and T5's states are consumed as a
        # fixed-width block, both of which the reference pads for.
        tokenizer = self.tokenizers[name]
        input_ids = tokenizer.tokenize(prompt).input_ids
        if input_ids.shape[-1]:
            return input_ids
        # `LanguageTokenizer.tokenize` short-circuits an all-empty batch to a
        # zero-width array. The unconditional branch of classifier-free guidance
        # encodes exactly that — `""` — and the reference does *not* treat it as
        # nothing: it runs the tokenizer normally and gets a full padded window of
        # start, end and padding tokens, which is a real and non-trivial embedding.
        # Without this, CLIP's `argmax` over an empty axis raises, and substituting
        # `" "` would silently condition on a space.
        raw = tokenizer.tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        return mx.array(raw["input_ids"])

    def _prepare_latents(self, *, seed: int, config: Config) -> mx.array:
        noise = SD35LatentCreator.create_noise(seed, config.height, config.width)
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
            # mflux's VAE carries a frame axis for the video models that share the class.
            clean = mx.squeeze(clean, axis=2)
        return LatentCreator.add_noise_by_interpolation(
            clean=clean,
            noise=noise,
            sigma=config.scheduler.sigmas[config.init_time_step],
        )

    # ── Saving ─────────────────────────────────────────────────────────────

    def save_model(self, base_path: str) -> None:
        """Write this model as one directory per component.

        `ModelSaver.save_model` walks the definition's components and skips the ones
        the object does not carry, which is what makes a component-at-a-time
        conversion possible: `prequantize` builds an `SD35`-shaped object holding a
        single component and calls this, five times, into the same directory.
        """
        from mflux.models.common.weights.saving.model_saver import ModelSaver

        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=SD35WeightDefinition,
        )
