"""The tools, resources and prompts a model is offered.

Designed for the client that actually calls it: a language model running on the
same machine, often small, deciding without supervision which tool to use and
what to put in its arguments. Three consequences run through everything here.

**Verbs, not a parameter matrix.** `refine_image` and `vary_image` are both
expressible as `generate_image` with the right arguments -- which is exactly why
they exist as named tools. For a model choosing from a list, the name *is* the
instruction, and "refine this" is a thing it can recognise wanting to do.

**Closed sets are types.** A `Literal` is rejected by the SDK before the handler
runs, and the model reads the rejection and corrects itself. A string validated
in the body produces the same refusal one round trip later and with less to go on.

**Errors are messages, not codes.** A failing tool re-raises the server's own
`APIError` text so the SDK renders it as a tool error the model can read. Never
`MCPError`, which hides the message from the model -- the opposite of what a
small model needs in order to fix its own call.

The result of a generating tool is one text block per call plus a resource link
per image. Prose with keys rather than JSON: `file: 3f2c….png` is read correctly
by models that would mis-parse a nested object. No pixels are sent to the model
-- the person judges the picture, in the playground or through the link.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp import types
from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from qds.errors import APIError
from qds.mcp import images as image_tools
from qds.mcp import run as run_tools
from qds.mcp.progress import Attribution
from qds.mcp.sessions import SessionBinding, require_unlocked

MAX_SEED = 2**32 - 1

INSTRUCTIONS = """\
This server generates images locally with diffusion models. Generating one takes
tens of seconds to a few minutes; the tools block until the image exists, so call
one at a time and wait rather than issuing several.

Start from generate_image. To change a picture it produced, use refine_image with
that picture's file name; for another take on the same idea, vary_image; to make
one larger, upscale_image.

Sizes are given as width and height in pixels, and models have their own limits --
call list_models when unsure, or omit them for the model's own default. Leave seed
unset for something new; pass a previous image's seed to reproduce it.

A tool result gives you each image's file name and a `full image:` URL. Say what
you made and offer that URL as a link so the person can open it; do not try to
embed the picture in your reply. The images also appear in the playground on this
machine as they are made.

Every image is kept in a playground session on this machine and stays there after
the conversation ends. The person running this server can see them.
"""


def build_server(deps) -> Any:
    """Assemble the MCP server over one application's `MCPDeps`.

    A factory rather than a module-level server for the same reason
    `build_dependencies` is: everything closes over one running application, and
    a test holding two apps at once must not have them share a session binding.
    """
    from qds import __version__

    mcp = MCPServer(
        name="quantum-diffusion-server",
        title="Quantum Diffusion Server",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    binding = SessionBinding(deps.store)

    # ── Shared plumbing ────────────────────────────────────────────────────

    def session_for(ctx, explicit: str | None) -> str:
        session_id = binding.resolve(getattr(ctx, "headers", None), explicit=explicit)
        require_unlocked(deps.store, session_id)
        return session_id

    def admit(
        *,
        prompt: str,
        model: str | None,
        n: int,
        width: int | None,
        height: int | None,
        steps: int | None,
        seed: int | None,
        negative_prompt: str | None,
        enhance_prompt: bool,
        rewritten_prompt: str | None = None,
    ) -> dict:
        """Run the *playground route's* admission, and return what it settled.

        Not a re-implementation and not a subset: these are the very callables
        `POST /playground/api/sessions/{id}/generations` runs, handed over in
        `MCPDeps`. A check added there tomorrow refuses the same input here,
        with the same error code, without anyone remembering to come back.
        """
        spec = deps.resolve_spec(model)
        if n < 1:
            raise APIError("n must be at least 1.", param="n", code="invalid_n")
        deps.check_n(n)
        deps.check_prompt(spec, prompt)
        negative = (negative_prompt or "").strip() or None
        deps.check_capabilities(spec, negative_prompt=negative, guidance=None)
        carried = (rewritten_prompt or "").strip() or None
        rewrite = deps.check_rewrite(spec, prompt, requested=enhance_prompt, carried=carried)

        # Width and height are separate integers here while the shared validator
        # takes "WxH". Formatting the string at this one point removes a whole
        # class of error a small model makes, and the validator stays the one
        # that decides whether the numbers are allowed.
        if (width is None) != (height is None):
            raise APIError(
                "Pass width and height together, or neither.",
                param="width" if width is None else "height",
                code="invalid_size",
            )
        size = f"{width}x{height}" if width is not None else None
        resolved_width, resolved_height = deps.resolve_size(spec, size)

        if steps is not None and steps < 1:
            raise APIError("steps must be at least 1.", param="steps", code="invalid_steps")
        if seed is not None and not (0 <= seed <= MAX_SEED):
            raise APIError(f"seed must be between 0 and {MAX_SEED}.", param="seed", code="invalid_seed")

        return {
            "spec": spec,
            "negative": negative,
            "carried": carried,
            "rewrite": rewrite,
            "width": resolved_width,
            "height": resolved_height,
            "steps": steps or spec.default_steps,
            "steps_from_preset": steps is None and spec.preset is not None,
            "seeds": deps.seeds_for(seed, n),
        }

    def submit(
        session_id: str,
        admitted: dict,
        *,
        prompt: str,
        n: int,
        context: Any = None,
        group: str | None = None,
        kind: str | None = None,
        image_strength: float | None = None,
    ) -> dict:
        """Write the row and hand it to the queue, cleaning up on any failure.

        The reference copy already sits in the playground's never-purged
        directory, so anything that goes wrong between writing it and a row
        owning it must remove it -- nothing else ever will. Same reasoning, and
        the same `try`, as the playground route.
        """
        spec = admitted["spec"]
        try:
            record = deps.store.add_generation(
                session_id,
                prompt=prompt,
                negative_prompt=admitted["negative"],
                rewrite=admitted["rewrite"],
                rewritten_prompt=admitted["carried"],
                model=spec.public_name,
                kind=kind or "txt2img",
                n=n,
                width=admitted["width"],
                height=admitted["height"],
                steps=admitted["steps"],
                steps_from_preset=admitted["steps_from_preset"],
                seeds=admitted["seeds"],
                image_strength=image_strength,
                context_image=context.name if context else None,
                group=group,
            )
        except KeyError as exc:
            _unlink(context)
            raise APIError(
                f"No playground session {session_id!r}.", status_code=404, code="not_found"
            ) from exc
        except ValueError as exc:
            _unlink(context)
            raise APIError(
                f"No generation group {group!r} in this session.",
                param="group_id",
                code="invalid_group",
            ) from exc
        except BaseException:
            _unlink(context)
            raise
        deps.runner.submit(record["id"])
        return record

    def _unlink(context) -> None:
        if context is not None:
            context.unlink(missing_ok=True)

    def kind_for(spec, has_reference: bool) -> tuple[str, float | None]:
        """Edit, image-to-image, or a refusal -- the playground route's rule.

        There is deliberately no `strength` argument anywhere in this package.
        The playground has none either: which of the two a reference means is a
        property of the model, not a knob, and offering one would invite a model
        to set it on a model that has no such mode.
        """
        if not has_reference:
            return "txt2img", None
        from qds.registry import edit_enabled

        if edit_enabled(spec):
            return "edit", None
        if spec.supports_image_to_image:
            from qds.admission import DEFAULT_IMAGE_STRENGTH

            return "txt2img", DEFAULT_IMAGE_STRENGTH
        raise APIError(
            f"Model '{spec.key}' supports neither editing nor image-to-image, so it "
            f"cannot start from a reference image.",
            param="model",
            code="unsupported_parameter",
        )

    async def deliver(ctx, record: dict, admitted: dict) -> list:
        """Wait for the row to settle, then render it for a model to read."""
        attribution = Attribution(
            generation_id=record["id"],
            model_key=admitted["spec"].key,
            seeds=frozenset(admitted["seeds"]),
            n=record["n"],
            steps=record["steps"] or 1,
        )
        final = await run_tools.wait_for(deps, ctx, generation_id=record["id"], attribution=attribution)
        return render(final)

    def render(record: dict) -> list:
        """Per image: the facts in text, the file linked. No pixels.

        Two channels, and the absence of a third is the design.

        **No image block, and no `data:` URI.** Both were tried, at length, and
        removed. What they were
        for was letting a *model* judge, or retype, the picture -- and the
        second is a task models perform badly: a base64 line has no redundancy,
        so one wrong character loses the image, and the model cannot check its
        own copy. Bounding that line's length to what a model would reproduce
        drove the preview down to about 81px on detailed output, which is not
        an image anyone can judge. Paying context twice for something illegible
        is worse than not sending it.

        The person judges the image. They have the playground, and they have
        the link below; neither costs a token and neither can be mis-copied.

        The **resource link** names the file's own http URL rather than a
        private `qds://` scheme. Both are valid MCP -- a resource URI is opaque,
        resolved by `resources/read` -- but only one of them is also an address
        the client can open on its own, and a model shown an unresolvable
        scheme was observed concluding it had nothing to offer. The resource is
        registered under that same URI, so reading it still works: a link
        nothing can dereference is a label.

        The **text** carries what a model reasons about -- which file, which
        seed, what size -- and nothing it is asked to recite. Text first,
        because anything that truncates a long result drops the end.

        The absolute filesystem path is deliberately absent, and is why the
        resource URI is the http one rather than `file://`: it would put the
        operator's home directory -- their username with it -- into a model's
        context on every generation.
        """
        status = record.get("status")
        if status == "failed":
            raise APIError(
                record.get("error") or "The generation failed.",
                status_code=500,
                error_type="server_error",
                code="generation_failed",
            )
        if status == "cancelled":
            return [f"Generation {record['id']} was cancelled."]

        blocks: list = []
        lines = [
            f"generation: {record['id']}  status: {status}  "
            f"session: {record['sessionId']}  group: {record['groupId']}"
        ]

        if status != "completed":
            lines.append(
                f"Still {status}. The queue is "
                + ("paused; resume it in the playground." if deps.runner.paused else "working.")
                + f" Call wait_for_generation with generation_id {record['id']!r} to pick it up."
            )
            return blocks + ["\n".join(lines)]

        for index, image in enumerate(record.get("images") or [], start=1):
            filename = image["url"].rsplit("/", 1)[-1]
            path = deps.store.images_dir / filename
            url = f"{deps.base_url}{image['url']}"
            try:
                width, height = image_tools.dimensions(path)
            except (OSError, ValueError):
                # The record is the deliverable; its dimensions are a detail of
                # it. Losing them must not lose the file's name or its link.
                size = None
                lines.append(f"image {index}  file: {filename}  (size unavailable)")
            else:
                size = f"{width}x{height}"
                lines.append(f"image {index}  file: {filename}  seed: {image['seed']}  size: {size}")
            lines.append(f"full image: {url}")
            blocks.append(
                types.ResourceLink(
                    type="resource_link",
                    uri=url,
                    name=filename,
                    title=f"{size} image, seed {image['seed']}" if size else f"image, seed {image['seed']}",
                    mime_type="image/png",
                    size=path.stat().st_size if path.is_file() else None,
                    annotations=types.Annotations(audience=["user"], priority=0.8),
                )
            )
        if record.get("rewrittenPrompt"):
            lines.append(f"prompt used: {record['rewrittenPrompt']}")
        if record.get("rewriteError"):
            lines.append(f"note: enhancing failed ({record['rewriteError']}); your prompt was used.")
        # Text first: anything that truncates a long result drops what is at the
        # end, and the text is what tells the model which file it made.
        return ["\n".join(lines)] + blocks

    # ── Generation ─────────────────────────────────────────────────────────

    @mcp.tool(
        description=(
            "Generate one or more images from a text description. Blocks until the "
            "images exist. Optionally start from a reference image to edit or vary it."
        )
    )
    async def generate_image(
        prompt: Annotated[str, Field(min_length=1, description="What to draw, in plain English.")],
        ctx: Context,
        model: Annotated[
            str | None, Field(description="Model id from list_models. Omit for the default.")
        ] = None,
        n: Annotated[int, Field(ge=1, description="How many images to make.")] = 1,
        width: Annotated[
            int | None, Field(ge=64, description="Width in pixels. Omit for the model's default.")
        ] = None,
        height: Annotated[
            int | None, Field(ge=64, description="Height in pixels. Pass with width or not at all.")
        ] = None,
        steps: Annotated[
            int | None,
            Field(ge=1, description="Denoising steps: more is slower. Omit for the model's own."),
        ] = None,
        seed: Annotated[
            int | None,
            Field(
                ge=0,
                le=MAX_SEED,
                description="Same seed and same settings reproduce the same image. Omit for a new one.",
            ),
        ] = None,
        negative_prompt: Annotated[
            str | None, Field(description="What to keep out. Not every model accepts one.")
        ] = None,
        enhance_prompt: Annotated[
            bool,
            Field(description="Expand a short prompt into a detailed one before generating."),
        ] = False,
        reference_image: Annotated[
            str | None,
            Field(description="File name of an image THIS server generated, to edit or vary."),
        ] = None,
        reference_path: Annotated[
            str | None,
            Field(
                description=(
                    "Absolute path to an image on this machine. Usually refused: the "
                    "server only reads from directories set in mcp.image_roots."
                )
            ),
        ] = None,
        session_id: Annotated[
            str | None,
            Field(description="Playground session to save into. Omit to use this conversation's."),
        ] = None,
        group_id: Annotated[
            str | None, Field(description="Attach to an existing feed entry rather than a new one.")
        ] = None,
    ) -> list:
        session = session_for(ctx, session_id)
        admitted = admit(
            prompt=prompt,
            model=model,
            n=n,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            negative_prompt=negative_prompt,
            enhance_prompt=enhance_prompt,
        )
        context = image_tools.resolve_reference(
            deps,
            session_id=session,
            reference_image=reference_image,
            reference_path=reference_path,
        )
        try:
            kind, strength = kind_for(admitted["spec"], context is not None)
        except BaseException:
            _unlink(context)
            raise
        record = submit(
            session,
            admitted,
            prompt=prompt,
            n=n,
            context=context,
            group=group_id,
            kind=kind,
            image_strength=strength,
        )
        return await deliver(ctx, record, admitted)

    @mcp.tool(
        description=(
            "Change an image this server made: it becomes the starting point and your "
            "prompt says what to change. Joins the same feed entry as the original."
        )
    )
    async def refine_image(
        image: Annotated[str, Field(description="File name of the image to work from.")],
        prompt: Annotated[str, Field(min_length=1, description="What to change about it.")],
        ctx: Context,
        model: Annotated[str | None, Field(description="Omit to keep the original's model.")] = None,
        steps: Annotated[int | None, Field(ge=1)] = None,
        seed: Annotated[
            int | None, Field(ge=0, le=MAX_SEED, description="Omit to keep the original's seed.")
        ] = None,
    ) -> list:
        source = _replayable(image)
        session = session_for(ctx, source["session_id"])
        admitted = admit(
            prompt=prompt,
            model=model or source["model"],
            n=1,
            width=source["width"],
            height=source["height"],
            steps=steps or source["steps"],
            seed=source["seed"] if seed is None else seed,
            negative_prompt=source["negative_prompt"],
            enhance_prompt=False,
        )
        context = image_tools.resolve_reference(
            deps, session_id=session, reference_image=image, reference_path=None
        )
        try:
            kind, strength = kind_for(admitted["spec"], True)
        except BaseException:
            _unlink(context)
            raise
        record = submit(
            session,
            admitted,
            prompt=prompt,
            n=1,
            context=context,
            group=source["group_id"],
            kind=kind,
            image_strength=strength,
        )
        return await deliver(ctx, record, admitted)

    @mcp.tool(
        description=(
            "Another take on an image this server made: the same prompt and settings, "
            "a new random seed. Joins the same feed entry as the original."
        )
    )
    async def vary_image(
        image: Annotated[str, Field(description="File name of the image to vary.")],
        ctx: Context,
        n: Annotated[int, Field(ge=1, description="How many variations.")] = 1,
    ) -> list:
        source = _replayable(image)
        session = session_for(ctx, source["session_id"])
        # The *typed* prompt, and the recorded rewrite replayed rather than
        # redone: a variation that re-enhanced would vary the words as well as
        # the seed, and stop being a variation of anything.
        admitted = admit(
            prompt=source["prompt"],
            model=source["model"],
            n=n,
            width=source["width"],
            height=source["height"],
            steps=source["steps"],
            seed=None,
            negative_prompt=source["negative_prompt"],
            enhance_prompt=False,
            rewritten_prompt=source["rewritten_prompt"],
        )
        # The original's own reference, if it had one -- copied again, so this
        # row owns its file and deleting the source breaks nothing.
        context = None
        if source["context_image"]:
            import shutil

            original = deps.store.images_dir / source["context_image"]
            if original.exists():
                context = deps.store.context_path(original.suffix or ".png")
                shutil.copyfile(original, context)
        try:
            kind, strength = kind_for(admitted["spec"], context is not None)
        except BaseException:
            _unlink(context)
            raise
        record = submit(
            session,
            admitted,
            prompt=source["prompt"],
            n=n,
            context=context,
            group=source["group_id"],
            kind=kind,
            image_strength=strength,
        )
        return await deliver(ctx, record, admitted)

    def _replayable(filename: str) -> dict:
        source = deps.store.generation_of_image(filename)
        if source is None:
            raise APIError(
                f"No image {filename!r} on this server. Only an image generated here "
                f"can be refined or varied.",
                status_code=404,
                param="image",
                code="not_found",
            )
        return source

    @mcp.tool(
        description=(
            "Enlarge an image this server made, without changing what is in it. "
            "Slower than generating, and it produces a much larger file."
        )
    )
    async def upscale_image(
        image: Annotated[str, Field(description="File name of the image to enlarge.")],
        ctx: Context,
        scale: Annotated[Literal[2, 4], Field(description="How many times larger.")] = 2,
        upscaler: Annotated[
            str | None, Field(description="Upscaler id. Omit for this server's default.")
        ] = None,
    ) -> list:
        from qds.upscale import catalogue as upscale_catalogue

        spec = (
            upscale_catalogue.by_key(upscaler)
            if upscaler
            else (upscale_catalogue.SPECS[0] if upscale_catalogue.SPECS else None)
        )
        if spec is None:
            raise APIError(
                f"Unknown upscaler {upscaler!r}. Available: {', '.join(upscale_catalogue.KEYS)}.",
                param="upscaler",
                code="invalid_model",
            )
        source = _replayable(image)
        session = session_for(ctx, source["session_id"])
        # Admission, the render budget and the copy all belong to the method
        # the HTTP route uses. This tool chooses a default upscaler and nothing
        # more; every refusal below it is the same refusal a browser would get.
        record = deps.submit_upscale(
            session, image=image, model=spec.key, scale=scale, group=source["group_id"]
        )
        attribution = Attribution(
            generation_id=record["id"],
            # An upscale has no denoising step to attribute, so nothing will
            # ever match: the notifications carry lifecycle only, which is the
            # truth about a tiled upscale rather than a borrowed step count.
            model_key="",
            seeds=frozenset(),
            n=1,
            steps=1,
        )
        final = await run_tools.wait_for(deps, ctx, generation_id=record["id"], attribution=attribution)
        return render(final)

    # ── Following work already started ─────────────────────────────────────

    @mcp.tool(
        description=(
            "Wait for a generation that was still running when a tool returned its id, "
            "and give back its images."
        )
    )
    async def wait_for_generation(
        generation_id: Annotated[str, Field(description="The id a previous tool returned.")],
        ctx: Context,
        timeout_s: Annotated[
            float | None, Field(gt=0, description="Seconds to wait. Omit for the server's own.")
        ] = None,
    ) -> list:
        record = deps.store.get_generation(generation_id)
        if record is None:
            raise APIError(
                f"No generation {generation_id!r}.",
                status_code=404,
                param="generation_id",
                code="not_found",
            )
        require_unlocked(deps.store, record["sessionId"])
        spec = deps.resolve_spec(record["model"])
        attribution = Attribution(
            generation_id=generation_id,
            model_key=spec.key,
            seeds=frozenset(record.get("seeds") or []),
            n=record["n"],
            steps=record["steps"] or 1,
        )
        final = await run_tools.wait_for(
            deps, ctx, generation_id=generation_id, attribution=attribution, timeout_s=timeout_s
        )
        return render(final)

    @mcp.tool(description="Stop a generation that is queued or running.")
    async def cancel_generation(
        generation_id: Annotated[str, Field(description="The id a previous tool returned.")],
    ) -> str:
        record = deps.store.get_generation(generation_id)
        if record is None:
            raise APIError(
                f"No generation {generation_id!r}.",
                status_code=404,
                param="generation_id",
                code="not_found",
            )
        require_unlocked(deps.store, record["sessionId"])
        cancelled = await deps.runner.cancel(generation_id)
        if cancelled:
            return f"Generation {generation_id} was cancelled."
        return (
            f"Generation {generation_id} was already {deps.store.status_of(generation_id)}; "
            f"nothing to cancel."
        )

    # ── Catalogue and sessions ─────────────────────────────────────────────

    @mcp.tool(
        description=(
            "List the image models this server can use, with what each one accepts. "
            "Also available as the resource qds://models."
        )
    )
    async def list_models() -> str:
        return _models_text()

    def _models_text() -> str:
        lines = [f"default model: {deps.resolve_spec(None).public_name}", ""]
        for name in sorted(deps.models):
            spec = deps.models[name]
            caps = deps.capabilities(spec)
            accepts = []
            if caps.get("supports_negative_prompt"):
                accepts.append("negative_prompt")
            if caps.get("supports_edit"):
                accepts.append("reference image (edit)")
            elif caps.get("supports_image_to_image"):
                accepts.append("reference image (variation)")
            if caps.get("prompt_formats") == ["json"]:
                accepts.append("JSON captions only, not plain text")
            lines.append(
                f"{name}\n"
                f"  default size: {caps.get('default_size')}   steps: {caps.get('default_steps')}\n"
                f"  size range:   {caps.get('min_dimension')}-{caps.get('max_dimension') or 'any'} px\n"
                f"  accepts:      {', '.join(accepts) if accepts else 'prompt only'}"
            )
        return "\n".join(lines)

    @mcp.tool(description="List the playground sessions on this server, newest first.")
    async def list_sessions() -> str:
        sessions = deps.store.list_sessions()
        if not sessions:
            return "No playground sessions yet."
        lines = []
        for row in sessions:
            marks = []
            if row.get("locked"):
                marks.append("locked")
            if row.get("generating"):
                marks.append("generating")
            suffix = f"  [{', '.join(marks)}]" if marks else ""
            lines.append(f"{row['id']}  {row.get('title') or '(untitled)'}{suffix}")
        return "\n".join(lines)

    @mcp.tool(
        description=(
            "Start a new playground session and make it the one this conversation "
            "saves into. Use it to keep a new subject apart from what came before."
        )
    )
    async def open_session(
        ctx: Context,
        title: Annotated[
            str | None, Field(max_length=80, description="A name. Omit to title it from the prompt.")
        ] = None,
    ) -> str:
        record = deps.store.create_session()
        if title and title.strip():
            deps.store.rename_session(record["id"], title.strip())
        binding.bind(getattr(ctx, "headers", None), record["id"])
        return f"Session {record['id']} created, and this conversation now saves into it."

    if deps.settings.mcp.allow_destructive:

        @mcp.tool(description="Permanently delete one image this server made.")
        async def delete_image(
            image: Annotated[str, Field(description="File name of the image to delete.")],
        ) -> str:
            row = deps.store.generated_image(image)
            if row is None:
                raise APIError(
                    f"No image {image!r} on this server.",
                    status_code=404,
                    param="image",
                    code="not_found",
                )
            require_unlocked(deps.store, row["session_id"])
            deps.store.delete_image(image)
            return f"Deleted {image}."

        @mcp.tool(description="Permanently delete a whole feed entry and every image in it.")
        async def delete_group(
            group_id: Annotated[str, Field(description="The group id a tool reported.")],
        ) -> str:
            session_id = deps.store.session_of_group(group_id)
            if session_id is None:
                raise APIError(
                    f"No generation group {group_id!r}.",
                    status_code=404,
                    param="group_id",
                    code="not_found",
                )
            require_unlocked(deps.store, session_id)
            await deps.runner.cancel_running_in_group(group_id)
            deps.store.delete_group(group_id)
            return f"Deleted group {group_id} and everything in it."

    # ── Resources ──────────────────────────────────────────────────────────
    #
    # `list_models` above duplicates `qds://models` on purpose. Resources are
    # host-mediated: many clients never surface them to the model at all, while
    # a tool is always reachable by it. That is a mechanism reason, not a taste.

    @mcp.resource("qds://models", mime_type="text/plain", description="Image models and limits.")
    async def models_resource() -> str:
        return _models_text()

    @mcp.resource("qds://upscalers", mime_type="text/plain", description="Upscalers and their scales.")
    async def upscalers_resource() -> str:
        from qds.upscale import catalogue as upscale_catalogue

        return "\n".join(
            f"{spec.key}  {spec.display_name}  scales: "
            f"{', '.join(str(s) for s in upscale_catalogue.SCALES)}  ({spec.size_mb:.0f} MB)"
            for spec in upscale_catalogue.SPECS
        )

    @mcp.resource("qds://sessions", mime_type="text/plain", description="Playground sessions on this server.")
    async def sessions_resource() -> str:
        return await list_sessions()

    @mcp.resource(
        f"{deps.base_url}/playground/images/{{filename}}",
        mime_type="image/png",
        description="One generated image, at full resolution.",
    )
    async def image_resource(filename: str) -> bytes:
        """Full resolution, which a tool result deliberately never carries.

        Same authority as `GET /playground/images/{filename}`, and the same
        refusal on a locked session: this is another door onto one library, not
        a wider one.
        """
        row = deps.store.generated_image(filename)
        if row is None:
            raise APIError(
                f"No image {filename!r} on this server.",
                status_code=404,
                param="filename",
                code="not_found",
            )
        require_unlocked(deps.store, row["session_id"])
        return (deps.store.images_dir / filename).read_bytes()

    # ── Prompts ────────────────────────────────────────────────────────────
    #
    # User-initiated templates, which after `instructions` is the strongest
    # thing available for a model that struggles to pick a tool unaided.

    @mcp.prompt(description="Make a picture of something.")
    async def illustrate(subject: str, style: str | None = None) -> str:
        styled = f", in the style of {style}" if style else ""
        return (
            f"Call generate_image with a detailed description of: {subject}{styled}. "
            f"Write the description yourself -- say what is in the picture, how it is "
            f"lit and how it is framed -- then show me the result."
        )

    @mcp.prompt(description="Change something about a picture already made.")
    async def refine(image: str, change: str) -> str:
        return (
            f"Call refine_image on {image} with a prompt describing this change: {change}. "
            f"Then show me the result and say what changed."
        )

    return mcp
