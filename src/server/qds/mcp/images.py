"""The trust boundary for an image a model names.

`reference_image` names a file this server produced, and is resolved the way
`playground_upscale` resolves its source: a row must match before any path is
built, which *is* the traversal guard -- there is no string sanitising to get
wrong. `reference_path` names a file on the machine, is chosen by a model rather
than by a person, and is refused unless it resolves inside a configured root.
Empty roots means every such path is refused, which is the default.

Nothing goes the other way. A tool result carries the image's *name and URL*,
never its pixels: encoding a picture into a model's context was tried and
removed, because the person is the one who judges it and they have the
playground and the link.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from qds.errors import APIError
from qds.playground import IMAGE_SUFFIXES


def dimensions(source) -> tuple[int, int]:
    """The image's real size, without decoding its pixels.

    Reported in the text block so a model knows what it made -- and read from
    the header, so it costs a stat rather than a decode.
    """
    from PIL import Image

    with Image.open(source) as image:
        return image.size


def _resolved_within(path: Path, roots: list[str]) -> Path | None:
    """The file `path` resolves to, when that lands inside one of `roots`.

    Both sides resolved, so a symlink is followed *before* the comparison. A
    containment check against an unresolved path answers a question about the
    name rather than about the file, and a symlink is exactly how a model would
    make those two differ.

    It returns the resolved target rather than a bool because checking one path
    and then reading another is what makes a containment check advisory: every
    step after this one -- the suffix, the size, the decode, the copy -- must
    operate on the file that was actually judged, or a symlink swapped in after
    the check decides what gets published.
    """
    try:
        target = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    for root in roots:
        try:
            resolved_root = Path(root).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if target == resolved_root or resolved_root in target.parents:
            return target
    return None


def resolve_reference(
    deps,
    *,
    session_id: str,
    reference_image: str | None,
    reference_path: str | None,
) -> Path | None:
    """The file a generation should start from, copied into the store.

    Returns the *copy*'s path, or None when no reference was named. A copy and
    never a reference to the original, for the reason the upscale route gives:
    a row that points at a file it does not own is a row that breaks when
    someone deletes that file.

    Both arguments are refused together. They are two answers to one question,
    and quietly preferring one would make a model's mistake invisible.
    """
    if reference_image and reference_path:
        raise APIError(
            "Pass reference_image or reference_path, not both: they are two ways to name one picture.",
            param="reference_path",
            code="invalid_request",
        )

    if reference_image:
        source = _from_this_server(deps, session_id=session_id, filename=reference_image)
    elif reference_path:
        source = _from_the_filesystem(deps, raw=reference_path)
    else:
        return None

    destination = deps.store.context_path(source.suffix or ".png")
    shutil.copyfile(source, destination)
    return destination


def _from_this_server(deps, *, session_id: str, filename: str) -> Path:
    """A generated image of *this* session.

    `not_found` rather than `forbidden` for another session's, matching the
    upscale route: whether something exists should not depend on who is asking.
    """
    row = deps.store.generated_image(filename)
    if row is None or row["session_id"] != session_id:
        raise APIError(
            f"No image {filename!r} in this session. Only images this server "
            f"generated in this session can be used as a reference.",
            status_code=404,
            param="reference_image",
            code="not_found",
        )
    return deps.store.images_dir / filename


def _from_the_filesystem(deps, *, raw: str) -> Path:
    """A path the *model* chose, which is why every branch here refuses.

    The argument may have been written into a prompt by someone the operator
    never met. Publishing an arbitrary readable file into the playground store
    would put it behind an HTTP route that a default loopback install serves
    without a credential -- so this is a confidentiality boundary, not a
    convenience check, and it fails closed on an empty configuration.

    Every check below, and the path returned for copying, is the *resolved*
    file. Judging one path and reading another leaves a window in which a
    symlink swapped in after the check chooses what gets published.
    """
    roots = deps.settings.mcp.image_roots
    if not roots:
        raise APIError(
            "Reading an image from a filesystem path is switched off on this "
            "server. Set mcp.image_roots to the directories a model may read "
            "from, or pass reference_image to reuse an image generated here.",
            status_code=403,
            param="reference_path",
            code="image_root_not_configured",
        )

    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise APIError(
            f"reference_path must be an absolute path (got {raw!r}).",
            param="reference_path",
            code="invalid_reference_path",
        )
    resolved = _resolved_within(path, roots)
    if resolved is None:
        raise APIError(
            f"reference_path {raw!r} is outside every directory in "
            f"mcp.image_roots, so this server will not read it.",
            status_code=403,
            param="reference_path",
            code="reference_path_denied",
        )
    # The resolved suffix, deliberately: a symlink named `.png` pointing at a
    # private key is exactly the case this refuses.
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise APIError(
            f"reference_path must name an image ({', '.join(sorted(IMAGE_SUFFIXES))}), "
            f"got {resolved.suffix or 'no suffix'!r}.",
            param="reference_path",
            code="invalid_reference_path",
        )

    limit = deps.settings.server.max_upload_mb * 1024 * 1024
    if resolved.stat().st_size > limit:
        raise APIError(
            f"reference_path is larger than the {deps.settings.server.max_upload_mb:.0f} MB "
            f"this server accepts (max_upload_mb).",
            status_code=413,
            param="reference_path",
            code="file_too_large",
        )

    # Decoded, not trusted by suffix: the containment check says we may read the
    # file, and this says it is the kind of file we said it was.
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(resolved) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise APIError(
            f"reference_path {raw!r} is not an image this server can read.",
            param="reference_path",
            code="invalid_reference_path",
        ) from exc
    return resolved
