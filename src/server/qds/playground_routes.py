"""The browser playground's control plane: `/playground/api` and its images.

The opposite of `/v1` in the one way that matters: a generation here outlives
the request that submitted it. It is a row in `PlaygroundStore`, run by the
single-worker `PlaygroundRunner`, and its images live outside `image_store`
where no TTL purge can reach them.

Everything under `/playground/api` carries the data-plane credential *and*
`deny_cross_site`, like the dashboard's own routes: a page on another origin
must not be able to spend this machine's GPU. `/playground/images/{filename}`
and its `/thumb` sibling are the documented exceptions, each for the reason its
own docstring gives.

The admission rules are `Admission`'s, never redecided here.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile, params
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from qds import admin, credential, playground_lock
from qds.admission import DEFAULT_IMAGE_STRENGTH, MAX_SEED, Admission, _save_upload
from qds.errors import APIError
from qds.logs import SERVER_LOGGER
from qds.playground import IMAGE_MEDIA_TYPES, THUMBNAIL_MEDIA_TYPE
from qds.registry import edit_enabled
from qds.upscale import catalogue as upscale_catalogue

logger = logging.getLogger(SERVER_LOGGER)

#: An unlock token, as the playground sends it. Module-level on purpose: with
#: postponed annotations FastAPI resolves names in the module namespace, so a
#: local alias inside the builder would be read as a required body field.
SessionToken = Annotated[str | None, Header(alias=playground_lock.UNLOCK_HEADER)]

#: How long a browser may keep a thumbnail. A day: long enough that scrolling a
#: project twice does not refetch it, short enough to bound the replay window the
#: route's `TEMPORARY:` note describes. The bytes themselves never go stale --
#: a filename is a `uuid4` and its pixels never change.
THUMBNAIL_MAX_AGE_S = 86400


class RenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    #: `None` or blank clears the title back to "first prompt".
    title: str | None = Field(default=None, max_length=1000)


class QueueStateRequest(BaseModel):
    """Hold or release the playground queue."""

    model_config = ConfigDict(extra="forbid")
    paused: bool


class SessionPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str


class UpscaleRequest(BaseModel):
    """Enlarge an image the session already owns.

    No image bytes: `image` names a file the server wrote and can attribute.
    `model` and `scale` are checked against the catalogue in the route rather
    than by an enum here, so the error names the valid values.
    """

    model_config = ConfigDict(extra="forbid")
    #: Filename of a *generated* image, as served by `/playground/images/`.
    image: str = Field(max_length=255)
    model: str = Field(max_length=64)
    scale: int
    #: Feed entry to join. Defaults to the source's, so an upscale grows the
    #: entry its image came from rather than starting a new one.
    group: str | None = Field(default=None, max_length=64)


def build_playground_router(admission: Admission, auth: params.Depends) -> APIRouter:
    """Both playground surfaces, as one router to include.

    Unprefixed, because the two are not siblings under one path: the API sits
    behind `/playground/api` with `deny_cross_site`, while the image routes sit
    at `/playground/images/{filename}` and `.../{filename}/thumb` without it.
    """
    router = APIRouter()
    settings = admission.settings
    playground = admission.playground
    runner = admission.runner
    engine = admission.engine

    playground_api = APIRouter(
        prefix="/playground/api",
        tags=["playground"],
        dependencies=[auth, Depends(admin.deny_cross_site)],
    )

    # Named locally so each route reads as one statement of its own rule. The
    # rules themselves live on `Admission`, shared with `/v1` and the MCP plane.
    unlocks = admission.unlocks
    unlock_throttles = admission.unlock_throttles
    not_found = admission.not_found
    assert_unlocked = admission.assert_unlocked
    submit_upscale = admission.submit_upscale
    no_generation = admission.no_generation
    no_image = admission.no_image

    def require_unlocked(session_id: str, x_qds_session_token: SessionToken = None) -> None:
        assert_unlocked(session_id, x_qds_session_token)

    unlocked = Depends(require_unlocked)

    @playground_api.post("/sessions", status_code=201)
    async def playground_create_session() -> dict:
        return playground.create_session()

    @playground_api.get("/sessions")
    async def playground_list_sessions() -> dict:
        # The pause rides on the list every tab already polls rather than on
        # `/v1/progress`: that stream is the *engine's* state and is shared with
        # `/v1` clients, and holding this queue is a playground control that has
        # no meaning there.
        return {"sessions": playground.list_sessions(), "paused": runner.paused}

    @playground_api.post("/queue")
    async def playground_set_queue_state(body: QueueStateRequest) -> dict:
        """Hold or release the playground queue, for every session at once.

        Not gated on a session, because it is not about one: there is a single
        FIFO worker behind every session's generations. It sits at the router's
        own auth level rather than admin's because it is reversible by anyone who
        can reach it, and that same credential already permits `/v1/cancel` and
        unbounded submission -- holding a queue is not more authority than
        emptying one. What it *is* that those are not is unbounded in time, which
        is why the state is published to every tab above.

        Idempotent, and deliberately not a claim about the engine: pausing takes
        effect at the runner's next boundary, so a 200 here does not mean nothing
        is being denoised.
        """
        await runner.set_paused(body.paused)
        return {"paused": runner.paused}

    @playground_api.get("/sessions/{session_id}", dependencies=[unlocked])
    async def playground_get_session(session_id: str) -> dict:
        detail = playground.get_session(session_id)
        if detail is None:
            raise not_found(session_id)
        return detail

    @playground_api.patch("/sessions/{session_id}", dependencies=[unlocked])
    async def playground_rename_session(session_id: str, body: RenameRequest) -> dict:
        session = playground.rename_session(session_id, body.title)
        if session is None:
            raise not_found(session_id)
        return session

    @playground_api.delete("/sessions/{session_id}", status_code=204, dependencies=[unlocked])
    async def playground_delete_session(session_id: str) -> None:
        # Stop the engine first: the worker is inside a generation whose record
        # is about to disappear, and it would otherwise keep the machine busy
        # producing images for a session nobody can see.
        await runner.cancel_running_in({session_id})
        playground.unlink(playground.delete_session(session_id))
        unlocks.revoke_session(session_id)
        unlock_throttles.forget(session_id)

    # ── Session passwords ──
    #
    # Setting, changing and removing all go through `unlocked`: on an open
    # session that passes trivially, on a protected one the token *is* the
    # proof of knowing the current password. The hash work runs in a thread —
    # ~100 ms of scrypt must not stall the event loop that serves previews.

    @playground_api.post("/sessions/{session_id}/password", dependencies=[unlocked])
    async def playground_set_password(session_id: str, body: SessionPasswordRequest) -> dict:
        try:
            record = await asyncio.to_thread(credential.hash_password, body.password)
        except credential.WeakPassword as exc:
            raise playground_lock.weak_password(str(exc)) from None
        if not playground.set_password(session_id, record):
            raise not_found(session_id)
        # Every earlier token was minted against the old password (or none);
        # the caller gets a fresh one so it stays where it is.
        unlocks.revoke_session(session_id)
        return {"token": unlocks.issue(session_id)}

    @playground_api.delete(
        "/sessions/{session_id}/password", status_code=204, dependencies=[unlocked]
    )
    async def playground_remove_password(session_id: str) -> None:
        if playground.password_record(session_id) is None:
            raise playground_lock.not_protected(session_id)
        playground.set_password(session_id, None)
        unlocks.revoke_session(session_id)
        unlock_throttles.forget(session_id)

    @playground_api.post("/sessions/{session_id}/unlock")
    async def playground_unlock(session_id: str, body: SessionPasswordRequest) -> dict:
        try:
            record = playground.password_record(session_id)
        except KeyError:
            raise not_found(session_id) from None
        if record is None:
            raise playground_lock.not_protected(session_id)
        throttle = unlock_throttles.for_session(session_id)
        wait = throttle.retry_after()
        if wait > 0:
            raise playground_lock.too_many_attempts(wait)
        if not await asyncio.to_thread(credential.verify_record, body.password, record):
            throttle.record_failure()
            logger.warning("playground: failed unlock attempt on session %s", session_id)
            raise playground_lock.invalid_password()
        throttle.record_success()
        return {"token": unlocks.issue(session_id), "session": playground.session_summary(session_id)}

    @playground_api.post("/sessions/{session_id}/lock", status_code=204)
    async def playground_lock_session(session_id: str, x_qds_session_token: SessionToken = None) -> None:
        """Give back the presented token. Only that one: another tab's unlock is
        its own; changing the password is what revokes them all."""
        try:
            playground.password_record(session_id)
        except KeyError:
            raise not_found(session_id) from None
        if unlocks.session_for(x_qds_session_token) == session_id:
            unlocks.revoke(x_qds_session_token)

    @playground_api.post("/sessions/{session_id}/generations", status_code=202, dependencies=[unlocked])
    async def playground_generate(
        session_id: str,
        prompt: Annotated[str, Form()],
        model: Annotated[str | None, Form()] = None,
        negative_prompt: Annotated[str | None, Form()] = None,
        n: Annotated[int, Form()] = 1,
        size: Annotated[str | None, Form()] = None,
        steps: Annotated[int | None, Form()] = None,
        seed: Annotated[int | None, Form()] = None,
        group: Annotated[str | None, Form()] = None,
        rewrite: Annotated[bool, Form()] = False,
        rewritten_prompt: Annotated[str | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
    ) -> dict:
        spec = admission.resolve_spec(model)
        if n < 1:
            raise APIError("n must be at least 1.", param="n", code="invalid_n")
        admission.check_n(n)
        admission.check_prompt(spec, prompt)
        # Blank is "none sent", not "an empty negative prompt": a browser form
        # posts the field whether or not it was typed in, and storing `""` would
        # both misreport the request and trip the capability check below on a
        # model that cannot take one.
        negative = (negative_prompt or "").strip() or None
        admission.check_capabilities(spec, negative_prompt=negative, guidance=None)
        carried = (rewritten_prompt or "").strip() or None
        rewrite = admission.check_rewrite(spec, prompt, requested=rewrite, carried=carried)
        width, height = admission.resolve_size(spec, size)
        if steps is not None and steps < 1:
            raise APIError("steps must be at least 1.", param="steps", code="invalid_steps")
        steps_val = steps or spec.default_steps
        if seed is not None and not (0 <= seed <= MAX_SEED):
            raise APIError(
                f"seed must be between 0 and {MAX_SEED}.", param="seed", code="invalid_seed"
            )
        seeds = admission.seeds_for(seed, n)

        kind, image_strength = "txt2img", None
        if image is not None:
            # Same decision as `/v1/images/edits` with no explicit strength.
            if edit_enabled(spec):
                kind = "edit"
            elif spec.supports_image_to_image:
                image_strength = DEFAULT_IMAGE_STRENGTH
            else:
                raise APIError(
                    f"Model '{spec.key}' supports neither editing nor image-to-image.",
                    param="model",
                    code="unsupported_parameter",
                )

        # The upload lands directly in the playground's never-purged directory, so
        # anything that goes wrong between writing it and owning it by a row must
        # remove it: nothing else ever will. `/v1/images/edits` gets this from its
        # scratch directory; here it is explicit.
        destination: Path | None = None
        try:
            if image is not None:
                destination = playground.context_path(Path(image.filename or "").suffix)
                await _save_upload(image, destination, settings.server.max_upload_mb)
            record = playground.add_generation(
                session_id,
                prompt=prompt,
                negative_prompt=negative,
                rewrite=rewrite,
                rewritten_prompt=carried,
                model=spec.public_name,
                kind=kind,
                n=n,
                width=width,
                height=height,
                steps=steps_val,
                steps_from_preset=steps is None and spec.preset is not None,
                seeds=seeds,
                image_strength=image_strength,
                context_image=destination.name if destination else None,
                group=group,
            )
        except KeyError as exc:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise APIError(
                f"No playground session {session_id!r}.",
                status_code=404,
                code="not_found",
            ) from exc
        except ValueError as exc:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise APIError(
                f"No generation group {group!r} in this session.",
                param="group",
                code="invalid_group",
            ) from exc
        except BaseException:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise
        runner.submit(record["id"])
        return record

    @playground_api.get("/upscalers")
    def playground_upscalers() -> dict:
        """What the image toolbar can offer, and whether it will cost a wait.

        `downloaded` is asked of the *file*, not the repository:
        `availability.scan_repos` answers "this repo is in the cache", which is
        right for a status report and wrong here -- it would say present for a
        repo from which some other file had been pulled. This decides whether a
        click starts a download, so it asks the exact question.
        """
        from qds.upscale.weights import is_downloaded

        return {
            "upscalers": [
                {
                    "id": spec.key,
                    "name": spec.display_name,
                    "scales": list(upscale_catalogue.SCALES),
                    "downloaded": is_downloaded(spec),
                    "sizeMb": spec.size_mb,
                    "license": spec.license,
                }
                for spec in upscale_catalogue.SPECS
            ]
        }

    @playground_api.post("/sessions/{session_id}/upscales", status_code=202, dependencies=[unlocked])
    async def playground_upscale(session_id: str, body: UpscaleRequest) -> dict:
        """JSON rather than multipart: the bytes are already here.

        See `submit_upscale` for why the mechanism is not in this route.
        """
        return submit_upscale(
            session_id, image=body.image, model=body.model, scale=body.scale, group=body.group
        )

    @playground_api.post("/generations/{generation_id}/cancel")
    async def playground_cancel(generation_id: str, x_qds_session_token: SessionToken = None) -> dict:
        session_id = playground.session_of_generation(generation_id)
        if session_id is None:
            raise no_generation(generation_id)
        assert_unlocked(session_id, x_qds_session_token)
        record = await runner.cancel(generation_id)
        if record is None:
            raise no_generation(generation_id)
        return record

    @playground_api.delete("/groups/{group_id}", status_code=204)
    async def playground_delete_group(group_id: str, x_qds_session_token: SessionToken = None) -> None:
        """Delete a whole feed entry: every generation of the lineage, and every
        file only it owned.

        The ordering is `playground_delete_session`'s, for the same reason: stop
        the engine first, because the worker may be inside a generation whose
        record is about to disappear and would otherwise keep the machine busy
        producing an image for an entry nobody can see.
        """
        session_id = playground.session_of_group(group_id)
        if session_id is None:
            raise APIError(
                f"No playground group {group_id!r}.", status_code=404, code="not_found"
            )
        assert_unlocked(session_id, x_qds_session_token)
        await runner.cancel_running_in_group(group_id)
        removed = playground.delete_group(group_id)
        if removed is None:  # pragma: no cover - deleted between the two calls
            raise APIError(
                f"No playground group {group_id!r}.", status_code=404, code="not_found"
            )
        playground.unlink(removed)

    @playground_api.get("/preview")
    async def playground_preview() -> Response:
        """The running generation's latest partially-denoised image, if there is one.

        A same-origin `<img>` sends the session cookie and no `Origin` header, so
        it satisfies both router dependencies — the same auth story as the feed's
        image fetches. 404 outside a run, or when the running job is a `/v1` one.

        The client's `?v=<preview_seq>` is a cache-buster, not a selector: the one
        slot always answers with its current frame, which can already be a newer
        one. Matching the counter exactly would fail-close a frame the client is
        entitled to whenever a fast model decodes the next one mid-fetch, and
        "latest" is what the caller wants either way.
        """
        payload = engine.preview()
        if payload is None:
            raise APIError("No preview is available.", status_code=404, code="not_found")
        return Response(payload, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @playground_api.delete("/images/{filename}", status_code=204)
    async def playground_delete_image(filename: str, x_qds_session_token: SessionToken = None) -> None:
        # `unlink` runs only when a DB row matched, and rows only ever hold names
        # minted by `save_image` (`uuid4().hex + ".png"`) or `context_path`
        # (`ctx-<uuid><suffix>`), so a crafted path never reaches the filesystem.
        session_id = playground.session_of_image(filename)
        if session_id is None:
            raise no_image(filename)
        assert_unlocked(session_id, x_qds_session_token)
        matched = playground.delete_image(filename)
        if matched is None:
            raise APIError(
                f"No playground image {filename!r}.", status_code=404, code="not_found"
            )
        _session_id, group_id = matched
        # Deleting the group's last image dissolves the group itself, or an entry
        # that is nothing but a prompt would be left behind; an active member
        # keeps the group, since its image is still coming.
        playground.unlink([filename, *playground.dissolve_empty_group(group_id)])

    router.include_router(playground_api)

    @router.get(
        "/playground/images/{filename}",
        dependencies=[auth],
        include_in_schema=False,
    )
    async def playground_image(
        filename: str,
        x_qds_session_token: SessionToken = None,
        t: Annotated[str | None, Query(alias=playground_lock.UNLOCK_QUERY)] = None,
    ) -> FileResponse:
        """A session's image, behind its lock.

        A row lookup rather than a static mount: the row says which session the
        file belongs to, and whether that session is locked. It also means a
        name no row holds — a traversal, a guess — is a 404 before any path is
        built. `?t=` is accepted here and only here, because an `<img>` sends no
        headers. `no-store`, so a relocked session is not replayed from cache.

        **`deny_cross_site` does not apply here, and only here.** Every other
        playground route keeps it. The reason is that on this one route the
        origin was never what authorized the request: the filename is a
        `uuid4().hex`, so naming a file *is* holding 122 bits of secret, and the
        session lock is checked underneath regardless. An origin check on top of
        that only refused a page that already knew the name -- which is to say,
        a page that already had the thing being protected.

        What it did break is real. `/mcp` hands a model this URL, and every MCP
        client renders in an origin of its own (`tauri://localhost` for one), so
        the guard turned the advertised link into a 403 whose JSON body then got
        saved as a `.png`. A URL published to clients that structurally cannot
        satisfy an origin check is a URL that does not work.

        The narrower guarantees are unchanged and are what this rests on: an
        unguessable name, a session lock enforced per request, `no-store`, and
        `/playground/api` still refusing cross-site outright -- so a hostile page
        can neither list what exists nor create anything.
        """
        session_id = playground.session_of_image(filename)
        if session_id is None:
            raise no_image(filename)
        assert_unlocked(session_id, x_qds_session_token or t)
        path = playground.images_dir / filename
        if not path.is_file():
            raise no_image(filename)
        # Declared, never guessed. Starlette's `FileResponse` falls back to
        # `mimetypes.guess_type(path)`, which turns a stored `.html` into an
        # inline document served from this server's own origin -- with the
        # dashboard's cookies. `context_path` no longer mints such a name; this
        # is what makes one that predates it inert anyway.
        return FileResponse(
            path,
            media_type=IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.get(
        "/playground/images/{filename}/thumb",
        dependencies=[auth],
        include_in_schema=False,
    )
    async def playground_image_thumbnail(
        filename: str,
        x_qds_session_token: SessionToken = None,
        t: Annotated[str | None, Query(alias=playground_lock.UNLOCK_QUERY)] = None,
    ) -> FileResponse:
        """The same image, small, behind the same lock.

        A gallery grid paints tens of tiles at once, and the stored files average
        1.9 MB, so serving the full image into a tile is ~190 MB for a hundred of
        them. This route serves a derived copy bounded to `THUMBNAIL_EDGE` on its
        longest edge -- ~20 KB measured.

        **Authority is this route's own, not inherited by proximity.** The row
        lookup comes first, so a name no row holds is a 404 before any path is
        built, and `assert_unlocked` runs on *this* request: a thumbnail is
        refused to exactly the caller the full image is refused to. `?t=` is
        accepted for the reason the image route accepts it -- an `<img>` sends no
        headers -- and nothing else about the lock is different here.

        `deny_cross_site` is absent, on the same argument the image route makes
        for itself and for the same narrow reason: the origin never authorized
        this request either. The path is `<uuid4 hex>/thumb`, so naming a tile is
        still holding 122 bits of secret, the session lock is still checked
        underneath, and the caller that needs it is the plugin pane rendering in
        an origin of its own (`tauri://localhost`) -- the very surface the guard
        broke on the image route. It also keeps the route under
        `pna.IMAGES_PREFIX`, so the private-network preflight an embedded client
        sends is granted here as well; a sibling path would have failed it.

        TEMPORARY: `private, max-age` rather than the image route's `no-store`.
        What this weakens, exactly: a session that has been re-locked can still
        have its *thumbnails* replayed from that one browser's cache until they
        expire, whereas its full images cannot. Authorised for this step so a
        scrolling grid does not refetch every tile, and scheduled to be revisited
        -- a validator with a short-lived token, or an unlock-scoped ETag, would
        give the grid its cache without the replay window. The full-image route
        keeps `no-store`, and so does the fallback below, so no full-resolution
        byte becomes cacheable by this decision.
        """
        session_id = playground.session_of_image(filename)
        if session_id is None:
            raise no_image(filename)
        assert_unlocked(session_id, x_qds_session_token or t)
        # In a thread: the derivation is ~40 ms of CPU on a 5120x2880 source, and
        # a cold grid asks for many at once. On the event loop that would stall
        # every other request, previews included, for the length of the grid.
        thumb = await asyncio.to_thread(playground.thumbnail, filename)
        if thumb is None:
            # A missing thumbnail degrades to the full image, never to a broken
            # tile: the derived artifact is an optimisation, and the caller has
            # already been authorized for the bytes it stands in for. `no-store`
            # here, because what is being served *is* the full-resolution file
            # and it keeps that file's régime wherever it is served from.
            path = playground.images_dir / filename
            if not path.is_file():
                raise no_image(filename)
            return FileResponse(
                path,
                media_type=IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                headers={
                    "Cache-Control": "private, no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        return FileResponse(
            thumb,
            media_type=THUMBNAIL_MEDIA_TYPE,
            headers={
                "Cache-Control": f"private, max-age={THUMBNAIL_MAX_AGE_S}",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return router
