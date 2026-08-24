"""The rules a generation is admitted through, and the state they consult.

Every generative plane asks the same questions before a job exists: is this
model real, can it read this prompt, is `n` inside the server's limit, what
size does that come to, may this session have this image. `/v1`, the
playground's `/playground/api` and the MCP tools must answer them identically —
a check added on one plane and missed on another is not a smaller bug for
having been an omission.

This used to be ~30 closures inside `create_app`, and the MCP plane already
borrowed them one by one through `MCPDeps`. That borrowing is what makes the
object rather than the closures the right shape: the callables were already a
bundle, passed as a bundle, with `create_app`'s local scope as its accidental
container.

Nothing here knows about HTTP request objects or FastAPI dependencies. Routes
translate; this decides.
"""

from __future__ import annotations

import base64
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from qds import playground_lock
from qds.engine import GenerationJob
from qds.errors import APIError
from qds.idle import IdleUnloader
from qds.playground import PlaygroundRunner, PlaygroundStore
from qds.registry import ModelSpec, edit_enabled, parse_size
from qds.rewrite.catalogue import MAX_PROMPT_CHARS
from qds.rewrite.prompt import should_rewrite
from qds.settings import RESPONSE_FORMATS, Settings
from qds.store import ImageStore
from qds.upscale import catalogue as upscale_catalogue

MAX_SEED = 2**32 - 1
#: Value used when `/v1/images/edits` falls back to img2img without the client
#: specifying `strength` (mflux/cli/defaults/defaults.py:14).
DEFAULT_IMAGE_STRENGTH = 0.4


def _rewrite_capabilities(settings) -> dict:
    """What the client needs to decide whether to offer an "Enhance" control.

    `available` decides whether the control exists at all; `reason` is why it
    does not, so "where is the button" is answerable without reading a log;
    `word_ceiling` lets the composer say "generated as typed" *before*
    submitting rather than after -- the same number the route enforces,
    published rather than duplicated as a constant on both sides. `downloaded`
    and `sizeMb` are the pair `playground_upscalers` publishes, asked the same
    way: of the *files*, not the repository.

    Deliberately **not** published: which model does the rewriting, or its
    licence. Someone using the playground needs to know their prompt will be
    improved and what a first use costs; which LLM does it is an operator fact.
    Removed rather than merely left unrendered -- a field that is published and
    unused is an invitation to render it again. The identity is not lost, it is
    re-addressed: `rewrite.model` in the configuration, the `rewriter_ready` and
    `rewrite_done` log events, the README, and the catalogue itself.
    """
    spec = settings.rewriter()
    if spec is None:
        return {
            "available": False,
            "reason": settings.rewrite_unavailable_reason(),
            "downloaded": False,
            "sizeMb": None,
            "word_ceiling": settings.rewrite.word_ceiling,
        }

    from qds.rewrite.weights import is_downloaded

    return {
        "available": True,
        "reason": None,
        "downloaded": is_downloaded(spec),
        "sizeMb": spec.size_mb,
        "word_ceiling": settings.rewrite.word_ceiling,
    }


def _capabilities(spec: ModelSpec) -> dict:
    quantization = spec.quantization
    return {
        "repo": spec.repo,
        "default_size": spec.default_size,
        "default_steps": spec.default_steps,
        "default_guidance": spec.default_guidance,
        "quantize": spec.quantize,
        # Which saved representation *this running process* loaded its registry
        # with. The catalogue publishes the one the configuration currently
        # selects; the two disagreeing is precisely what a restart would fix, and
        # without this the interface had no way to tell that a variant it had
        # just activated was not yet the one being generated from.
        "active_variant": spec.prequantized_variant,
        # The quantization contract, published so the app stops keeping its own
        # copy of the bit-depth rules. `prequantized` used to stand in for all of
        # this and meant three different things at once.
        "supports_quantization": quantization.supports_quantization,
        "quantize_choices": list(quantization.quantize_choices),
        "supports_prequantize": quantization.supports_prequantize,
        "prequantize_choices": list(quantization.prequantize_choices),
        "prequantize_strategy": quantization.prequantize_strategy,
        "quantization_note": quantization.note,
        "license": spec.license,
        "gated": spec.gated,
        "prompt_formats": list(spec.prompt_formats),
        "preset": spec.preset,
        "min_dimension": spec.min_dimension,
        "max_dimension": spec.max_dimension,
        "scheduler": spec.scheduler,
        "supports_guidance": spec.supports_guidance,
        "supports_negative_prompt": spec.supports_negative_prompt,
        "supports_image_to_image": spec.supports_image_to_image,
        "supports_edit": edit_enabled(spec),
    }


async def _save_upload(upload: UploadFile, destination: Path, max_mb: float) -> None:
    """Write the upload in chunks, rejecting anything past the limit."""
    limit = int(max_mb * 1024 * 1024)
    written = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                handle.close()
                destination.unlink(missing_ok=True)
                raise APIError(
                    f"Image too large (limit: {max_mb:g} MB).",
                    status_code=413,
                    param="image",
                    code="file_too_large",
                )
            handle.write(chunk)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise APIError("The image file is empty.", param="image", code="invalid_image")


class Admission:
    """One application's generative capability, minus the HTTP.

    Built once by `create_app` and handed whole to the `/v1` router, the
    playground router and `MCPDeps`. Holding the collaborators rather than
    reaching for them keeps the direction of dependency the same as
    `PlaygroundRunner`'s: it is told what it may use.
    """

    def __init__(
        self,
        settings: Settings,
        registry: dict[str, ModelSpec],
        *,
        engine: Any,
        store: ImageStore,
        playground: PlaygroundStore,
        idle_unloader: IdleUnloader,
        scratch_dir: Path,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.engine = engine
        self.store = store
        self.playground = playground
        self.idle_unloader = idle_unloader
        self.scratch_dir = scratch_dir
        #: `created` on every `/v1/models` entry: when this process started
        #: serving them, which is the only honest answer a local server has.
        self.created_at = int(time.time())
        #: Public identifier → spec. Built-ins publish their catalogue key; an
        #: imported model publishes its `api_name` rather than the opaque
        #: `local-…` id it is stored under.
        self.by_public_name = {spec.public_name: spec for spec in registry.values()}
        self.unlocks = playground_lock.UnlockStore()
        self.unlock_throttles = playground_lock.UnlockThrottles()
        #: Constructed here rather than by the caller: the runner needs
        #: `resolve_spec` and `build_rewrite_job`, and `submit_upscale` needs
        #: the runner. The cycle closes inside the one object that owns both
        #: ends, instead of being handed to `create_app` to wire up in order.
        self.runner = PlaygroundRunner(
            playground,
            engine,
            idle_unloader,
            self.resolve_spec,
            upscale_catalogue.by_key,
            self.build_rewrite_job,
        )

    # ── The model catalogue ────────────────────────────────────────────────

    def resolve_spec(self, key: str | None) -> ModelSpec:
        # The configured default may be an internal id — that is deliberate, and
        # `default_model` stays a durable reference rather than a friendly name
        # that would break the moment one were renamed.
        key = key or self.settings.default_model
        # Public name first, then the internal key. The second is an unadvertised
        # legacy path: an imported model's id was the only way to name it before
        # aliases existed, and silently breaking a script that used it would be a
        # poor trade for a listing that is already clean.
        spec = self.by_public_name.get(key) or self.registry.get(key)
        if spec is None:
            raise APIError(
                f"Unknown model: {key!r}. Available models: {sorted(self.by_public_name)}",
                param="model",
                code="model_not_found",
            )
        return spec

    def build_rewrite_job(self, prompt: str) -> Any:
        """A `RewriteJob` for `prompt`, or None if rewriting is not configured.

        Resolved at execution rather than captured at admission on purpose: the
        settings object is the one the app was built with, so this reflects a
        configuration reload, and a runner holding a job built minutes earlier
        would not.

        The seed is drawn here, per rewrite, and that is a behaviour change with
        a measured cause. `RewriteJob.seed` defaults to 0 and nothing was passing
        one, so every rewrite of the same prompt was byte-identical -- confirmed
        in a playground database where one prompt enhanced six times produced six
        identical expansions. Measured across the evaluation set, 13 of 31
        prompts change defect status between seeds, so a fixed seed does not mean
        "the best expansion", it means "the same one, good or bad, forever".

        No setting and no control for it, deliberately: the user toggles Enhance
        and it works. Reproducing a *generation* does not go through here at all
        -- it replays `generations.rewritten_prompt` on the carried path in
        `check_rewrite`, which never calls the rewriter.
        """
        spec = self.settings.rewriter()
        if spec is None:
            return None
        from qds.engine import RewriteJob
        from qds.rewrite.prompt import DEFAULT_SYSTEM_PROMPT

        rewrite = self.settings.rewrite
        return RewriteJob(
            spec=spec,
            prompt=prompt,
            system_prompt=rewrite.system_prompt or DEFAULT_SYSTEM_PROMPT,
            max_new_tokens=rewrite.max_new_tokens,
            temperature=rewrite.temperature,
            timeout_s=rewrite.timeout_s,
            seed=random.randint(0, MAX_SEED),
        )

    # ── Admission ──────────────────────────────────────────────────────────

    def resolve_size(self, spec: ModelSpec, size: str | None) -> tuple[int, int]:
        if size is None or size.lower() == "auto":
            width, height = spec.default_width, spec.default_height
        else:
            try:
                width, height = parse_size(size)
            except ValueError as exc:
                raise APIError(str(exc), param="size", code="invalid_size") from exc
        # Checked on the default too, not just on an explicit size: otherwise a
        # config-wide `default_size` outside the model's range would sail straight
        # through and fail inside mflux, after the weights were loaded.
        for label, value in (("width", width), ("height", height)):
            if value < spec.min_dimension or (spec.max_dimension and value > spec.max_dimension):
                bound = f"[{spec.min_dimension}, {spec.max_dimension or '∞'}]"
                raise APIError(
                    f"Model '{spec.key}' requires {label} in {bound}, got {value}.",
                    param="size",
                    code="invalid_size",
                )
        return width, height

    def resolve_response_format(self, value: str | None) -> str:
        fmt = value or self.settings.server.default_response_format
        if fmt not in RESPONSE_FORMATS:
            raise APIError(
                f"response_format must be one of {sorted(RESPONSE_FORMATS)}, got {fmt!r}",
                param="response_format",
                code="invalid_response_format",
            )
        return fmt

    def check_prompt(self, spec: ModelSpec, prompt: str) -> None:
        """Refuse a prompt the model cannot read, before any weights are loaded.

        FIBO's prompt encoder opens with a bare `json.loads(prompt)` whose result
        is discarded — a validation gate. Plain text raises a `JSONDecodeError`,
        which would reach the client as a 400 saying "Expecting value: line 1
        column 1" *after* several GB of weights had been loaded. So we say it here,
        and say what to do about it.
        """
        if "text" in spec.prompt_formats:
            # Accepting text means accepting anything: a JSON caption is text too.
            return
        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise APIError(
                f"Model '{spec.key}' only accepts a structured JSON caption as its prompt, "
                f"not plain text ({exc.msg}). Pass a JSON object describing the image - see the "
                f"model card for the schema.",
                param="prompt",
                code="prompt_must_be_json",
            ) from exc
        if not isinstance(parsed, dict):
            raise APIError(
                f"Model '{spec.key}' expects a JSON *object* as its prompt, got "
                f"{type(parsed).__name__}.",
                param="prompt",
                code="prompt_must_be_json",
            )

    def check_capabilities(
        self, spec: ModelSpec, *, negative_prompt: str | None, guidance: float | None
    ) -> None:
        if negative_prompt and not spec.supports_negative_prompt:
            raise APIError(
                f"Model '{spec.key}' does not support negative_prompt. "
                f"Describe what you want in the prompt instead.",
                param="negative_prompt",
                code="unsupported_parameter",
            )
        if guidance is not None and not spec.supports_guidance:
            fixed = spec.default_guidance
            raise APIError(
                f"Model '{spec.key}' is distilled, so its guidance is fixed"
                + (f" at {fixed}." if fixed is not None else ".")
                + " Drop the guidance parameter.",
                param="guidance",
                code="unsupported_parameter",
            )

    def check_rewrite(
        self, spec: ModelSpec, prompt: str, *, requested: bool, carried: str | None
    ) -> bool:
        """Decide whether this generation will be rewritten, refusing at the door.

        Returns whether the runner should expand the prompt. Every reason it
        might not is settled *here*, before a row exists and before any weights
        load, so the runner's only remaining question is "was one asked for".

        Fail-closed, and separately for each reason:

        * a model whose only prompt format is JSON is **refused**, not silently
          skipped. Producing a caption against Bria's schema is a different job
          with a hard failure mode -- `check_prompt` would reject the output --
          so the honest answer is that this model cannot be enhanced;
        * rewriting switched off, or naming a key this build lacks, is a 409
          carrying the reason `/v1/capabilities` publishes;
        * a prompt over `MAX_PROMPT_CHARS` is refused rather than truncated.
          Mutilating a long prompt silently is the one outcome nobody asked
          for. This is triage, not the memory bound: that one is in tokens and
          lives in `ModelEngine._rewrite_sync`, which has a tokenizer;
        * at or over `word_ceiling`, the request is simply not a rewrite. Not an
          error: the user asked for the best result and, measured, that is their
          own prompt. The UI says so before submitting, from the same number.

        Rewritten text -- requested or supplied -- is validated exactly once,
        here. A *requested* rewrite needs no check on its output because it is
        only offered where `check_prompt` is vacuous: every model that accepts
        `"text"` accepts anything, and the ones that do not are refused above. A
        *supplied* one is checked directly, because nothing else on that path
        looks at the text that will actually reach the model.
        """
        if carried is not None and requested:
            raise APIError(
                "A generation cannot both request a rewrite and supply one.",
                param="rewrite",
                code="invalid_rewrite",
            )
        if carried is not None:
            # A carried rewrite is replay, not work: it was produced under
            # whatever configuration held at the time, and refusing it now
            # because the feature was since switched off would make old
            # generations unrepeatable. So the availability checks below do not
            # apply to it -- but the *prompt format* check does, and must.
            #
            # Supplying text is a rewriting path too, and it is the one path on
            # which nothing else validates: `check_prompt` above ran against
            # `prompt`, while what actually reaches the model is this. Without
            # the line below, `rewritten_prompt="a plain sentence"` on a
            # JSON-only model is accepted, several GB load, and FIBO's encoder
            # raises on a bare `json.loads` -- exactly the failure `check_prompt`
            # exists to prevent, moved past it.
            self.check_prompt(spec, carried)
            return False
        if not requested:
            return False

        if "text" not in spec.prompt_formats:
            raise APIError(
                f"Model '{spec.key}' takes only a structured JSON caption, which "
                "the prompt rewriter does not produce.",
                param="rewrite",
                code="rewrite_unsupported_for_model",
            )
        reason = self.settings.rewrite_unavailable_reason()
        if reason is not None:
            raise APIError(reason, status_code=409, code="rewriter_unavailable")

        # Characters, not words. A word count is not an approximation of a
        # token count for a script without spaces -- a Chinese prompt of any
        # length is one "word" -- and this runs before any tokenizer exists.
        # The real bound is checked in `ModelEngine._rewrite_sync`, in tokens;
        # this refuses what could never fit, while the message can still name a
        # parameter and nothing has loaded.
        if len(prompt) > MAX_PROMPT_CHARS:
            raise APIError(
                f"This prompt is {len(prompt)} characters, past the "
                f"{MAX_PROMPT_CHARS} the rewriter accepts. Generate it as "
                "written instead.",
                param="prompt",
                code="prompt_too_long_to_rewrite",
            )
        return should_rewrite(prompt, word_ceiling=self.settings.rewrite.word_ceiling)

    def check_n(self, n: int) -> None:
        if n > self.settings.server.max_n:
            raise APIError(
                f"n={n} exceeds the server limit ({self.settings.server.max_n}). "
                f"Images are generated one at a time.",
                param="n",
                code="n_too_large",
            )

    def seeds_for(self, seed: int | None, n: int) -> list[int]:
        base = random.randint(0, MAX_SEED) if seed is None else seed
        return [(base + index) % (MAX_SEED + 1) for index in range(n)]

    # ── Execution, for the synchronous `/v1` plane ─────────────────────────

    def build_payload(
        self,
        request: Request,
        images: list[bytes],
        response_format: str,
        spec: ModelSpec,
        width: int,
        height: int,
        steps: int,
        seeds: list[int],
    ) -> JSONResponse:
        data: list[dict[str, Any]] = []
        for png in images:
            if response_format == "url":
                name = self.store.save(png)
                data.append({"url": f"{str(request.base_url).rstrip('/')}/images/{name}"})
            else:
                data.append({"b64_json": base64.b64encode(png).decode("ascii")})
        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
                # Extension: the effective size may differ from the requested
                # one (mflux truncates to a multiple of 16).
                "mflux": {
                    # The public name, matching what the request sent and what
                    # `/v1/models` lists. Echoing the internal `local-…` id here
                    # would hand a client the one identifier it must not learn.
                    "model": spec.public_name,
                    "size": f"{width}x{height}",
                    "steps": steps,
                    "seeds": seeds,
                },
            }
        )

    async def run_jobs(
        self,
        spec: ModelSpec,
        *,
        kind: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seeds: list[int],
        guidance: float | None,
        negative_prompt: str | None,
        image_path: Path | None = None,
        image_strength: float | None = None,
        steps_from_preset: bool = False,
    ) -> list[bytes]:
        # The idle countdown is armed here, on the way out, rather than inside
        # the engine: it must measure the gap between *requests*, otherwise a
        # delay of 0 would release the model between the images of a single one.
        with self.idle_unloader:
            images = []
            for seed in seeds:
                images.append(
                    await self.engine.generate(
                        GenerationJob(
                            spec=spec,
                            kind=kind,
                            prompt=prompt,
                            width=width,
                            height=height,
                            steps=steps,
                            seed=seed,
                            guidance=guidance,
                            negative_prompt=negative_prompt,
                            image_path=image_path,
                            image_strength=image_strength,
                            steps_from_preset=steps_from_preset,
                        )
                    )
                )
            return images

    # ── Playground ownership and locks ─────────────────────────────────────

    def not_found(self, session_id: str) -> APIError:
        return APIError(
            f"No playground session {session_id!r}.", status_code=404, code="not_found"
        )

    def no_generation(self, generation_id: str) -> APIError:
        return APIError(
            f"No playground generation {generation_id!r}.",
            status_code=404,
            code="not_found",
        )

    def no_image(self, filename: str) -> APIError:
        return APIError(
            f"No playground image {filename!r}.", status_code=404, code="not_found"
        )

    def assert_unlocked(self, session_id: str, token: str | None) -> None:
        """404 for an unknown session, 403 `session_locked` for a protected one
        the token does not open. An open session passes with any token."""
        try:
            record = self.playground.password_record(session_id)
        except KeyError:
            raise self.not_found(session_id) from None
        if record is None:
            return
        if self.unlocks.session_for(token) != session_id:
            raise playground_lock.locked(session_id)

    def submit_upscale(
        self, session_id: str, *, image: str, model: str, scale: int, group: str | None = None
    ) -> dict:
        """Enlarge an image this session already owns: admit, copy, enqueue.

        Here rather than in the route because there are two callers --
        `POST /playground/api/sessions/{id}/upscales` and the MCP
        `upscale_image` tool -- and every refusal below is an admission rule.
        A second copy of the render-budget arithmetic, or of the "only this
        session's images" check, is a second thing to get wrong the next time
        the catalogue changes.

        The source is by construction a file the server already holds and
        already knows the owner of, which is why nothing is uploaded here.
        `refine` round-trips its bytes because a refinement may legitimately
        start from an image the server has never seen; an upscale cannot.
        """
        spec = upscale_catalogue.by_key(model)
        if spec is None:
            raise APIError(
                f"Unknown upscaler {model!r}. Available: "
                f"{', '.join(upscale_catalogue.KEYS)}.",
                param="model",
                code="invalid_model",
            )
        if scale not in upscale_catalogue.SCALES:
            raise APIError(
                f"scale must be one of {', '.join(str(s) for s in upscale_catalogue.SCALES)}.",
                param="scale",
                code="invalid_scale",
            )

        # Only a generated image, and only one of this session's. `not_found`
        # rather than `forbidden` for someone else's: the answer to "does this
        # exist" should not depend on who is asking.
        source_row = self.playground.generated_image(image)
        if source_row is None or source_row["session_id"] != session_id:
            raise APIError(
                f"No playground image {image!r} in this session.",
                status_code=404,
                param="image",
                code="not_found",
            )
        source = self.playground.images_dir / image
        if not source.is_file():
            raise APIError(
                f"No playground image {image!r} in this session.",
                status_code=404,
                param="image",
                code="not_found",
            )

        with Image.open(source) as opened:
            source_width, source_height = opened.size
        width, height = source_width * scale, source_height * scale
        # Bounded on what the network renders, which is not what was asked for:
        # it always works at `native_scale` and a smaller factor is that result
        # resampled down. See `MAX_RENDER_PIXELS`.
        rendered = source_width * source_height * spec.native_scale**2
        if rendered > upscale_catalogue.MAX_RENDER_PIXELS:
            side = source_width * spec.native_scale, source_height * spec.native_scale
            raise APIError(
                f"Upscaling {source_width}x{source_height} means rendering "
                f"{side[0]}x{side[1]} ({rendered} pixels), past the "
                f"{upscale_catalogue.MAX_RENDER_PIXELS} pixel limit. The network "
                f"always works at x{spec.native_scale}, so x{scale} costs the same.",
                param="image",
                code="image_too_large",
            )

        # A copy, not a reference: deleting the source image must not leave the
        # upscale pointing at a missing file. The copy lives and dies with the
        # row, through the same `context_image` cleanup every other path uses.
        destination = self.playground.context_path(".png")
        try:
            shutil.copyfile(source, destination)
            record = self.playground.add_generation(
                session_id,
                prompt=source_row["prompt"],
                model=spec.key,
                kind="upscale",
                n=1,
                width=width,
                height=height,
                steps=0,
                steps_from_preset=False,
                # `generation_images.seed` is NOT NULL, and the honest value is
                # the seed the source was generated with.
                seeds=[source_row["seed"]],
                context_image=destination.name,
                group=group or source_row["group_id"],
            )
        except KeyError as exc:
            destination.unlink(missing_ok=True)
            raise APIError(
                f"No playground session {session_id!r}.", status_code=404, code="not_found"
            ) from exc
        except ValueError as exc:
            destination.unlink(missing_ok=True)
            raise APIError(
                f"No generation group {group!r} in this session.",
                param="group",
                code="invalid_group",
            ) from exc
        except BaseException:
            # The images directory is never purged: a file no row owns stays
            # there for good.
            destination.unlink(missing_ok=True)
            raise
        self.runner.submit(record["id"])
        return record
