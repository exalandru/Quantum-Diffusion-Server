"""Durable state and execution for the browser playground.

The `/v1` data plane is synchronous: a generation lives exactly as long as its
HTTP request, and its images are TTL-purged. The playground needs the opposite —
a generation must survive the browser tab that started it, and its images must
survive until the user deletes the session. So this module owns two things the
`/v1` plane deliberately does not have:

* `PlaygroundStore` — a SQLite record per session/generation/image, plus an image
  directory outside `image_store` (no TTL purge can reach it).
* `PlaygroundRunner` — a single in-process FIFO worker that calls
  `engine.generate()` like any other caller. Engine serialization is untouched:
  the runner waits on the engine's own lock, and the images of one generation run
  one after the other.

**Interrupted records.** A generation that was `queued` or `running` when the
process died cannot be resumed — the weights, the latents and the request are all
gone. `mark_interrupted()`, called at startup, moves those rows to `failed`, so
every accepted generation reaches a terminal state even across a crash. That is
the invariant the interface relies on to stop showing a spinner forever.

Store access is synchronous under one lock. The statements are single-row and the
database is local, so the microseconds spent on the event loop are not worth an
executor hop or a new dependency.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qds.engine import GenerationJob
from qds.errors import APIError, translate_mflux_exception
from qds.logs import SERVER_LOGGER
from qds.registry import ModelSpec

logger = logging.getLogger(f"{SERVER_LOGGER}.playground")

#: Statuses a generation can still leave on its own.
ACTIVE_STATUSES = ("queued", "running")
#: How long a title may be before it is cut. A sidebar row, not a document.
TITLE_LIMIT = 80
#: Decode a preview of the partially-denoised image every N denoising steps; a
#: VAE decode is not free, so this is not every step. 0 would disable previews.
#: Playground generations only — `/v1` never asks for them.
PREVIEW_EVERY = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  title TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  -- A `credential.hash_password` record as JSON, or NULL for an open session.
  -- Never serialized: `_session_json` reports only whether it is set.
  password TEXT
);
CREATE TABLE IF NOT EXISTS generations (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  -- The lineage a generation belongs to: its own id when it started a group,
  -- the ancestor's group otherwise. Refining or varying an image appends to the
  -- group its image came from, and the interface renders one group as one entry
  -- — several images of the same idea, as if they had been asked for at once.
  group_id TEXT,
  prompt TEXT NOT NULL,
  -- What the image should avoid. NULL for a request that sent none, and for
  -- every model whose pipeline has no unconditional branch to apply it to --
  -- the engine drops it on those, and the route refuses it outright.
  negative_prompt TEXT,
  -- What the rewriter made of `prompt`, or NULL if nothing rewrote it. This is
  -- what `_run` sends to the engine; `prompt` above is never overwritten, so
  -- the feed can always show what the user actually typed and a variation can
  -- replay the exact text its ancestor was generated from.
  rewritten_prompt TEXT,
  -- Why a requested rewrite did not happen, or NULL. Not redundant with
  -- `rewritten_prompt IS NULL`: without it, "the rewriter failed and we
  -- generated from your prompt" and "no rewrite was asked for" are the same
  -- row, and the first one is something the user is owed an explanation for.
  --
  -- Only ever written by `record_rewrite`, i.e. only after a rewrite has been
  -- attempted. It is published to the client and rendered as a failure, so a
  -- value here means one happened.
  rewrite_error TEXT,
  -- 1 while a requested rewrite has not run yet, 0 otherwise.
  --
  -- A column of its own rather than a third status. `ACTIVE_STATUSES` is
  -- replicated as a literal `IN ('queued','running')` across seven queries --
  -- `mark_interrupted` among them -- and a `rewriting` status that one of them
  -- missed would strand a row outside every terminal path, breaking the
  -- "everything reaches a terminal state, even after a crash" invariant this
  -- module opens with. What is tracked here is not a lifecycle stage; it is one
  -- boolean about work still owed.
  --
  -- And a column of its own rather than a sentinel in `rewrite_error`, which is
  -- what this was first: that column is published as `rewriteError` and the feed
  -- renders any value in it as "Enhancing failed (…) — generated from your
  -- prompt". A queued generation would have said so about work that had not
  -- been attempted, and a run cancelled before its rewrite would have said so
  -- forever.
  rewrite_pending INTEGER NOT NULL DEFAULT 0,
  model TEXT NOT NULL,
  kind TEXT NOT NULL,
  n INTEGER NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  steps INTEGER NOT NULL,
  steps_from_preset INTEGER NOT NULL,
  seeds TEXT NOT NULL,
  image_strength REAL,
  context_image TEXT,
  status TEXT NOT NULL,
  error TEXT,
  created_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL
);
CREATE INDEX IF NOT EXISTS generations_by_session
  ON generations (session_id, created_at);
CREATE TABLE IF NOT EXISTS generation_images (
  generation_id TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  filename TEXT NOT NULL,
  seed INTEGER NOT NULL,
  PRIMARY KEY (generation_id, position)
);
"""


def _session_json(row: sqlite3.Row, *, generating: bool) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "generating": generating,
        "locked": row["password"] is not None,
    }


def _generation_json(row: sqlite3.Row, images: list[dict[str, Any]]) -> dict[str, Any]:
    # `or row["id"]`: a row written before `group_id` existed is its own group,
    # which is also what the startup backfill writes.
    context = row["context_image"]
    return {
        "id": row["id"],
        "sessionId": row["session_id"],
        "groupId": row["group_id"] or row["id"],
        "prompt": row["prompt"],
        "negativePrompt": row["negative_prompt"],
        # Two separate facts, deliberately both published. The feed shows the
        # typed prompt as the title and the rewrite behind a disclosure, so a
        # user is never shown words they did not write as if they had.
        "rewrittenPrompt": row["rewritten_prompt"],
        "rewriteError": row["rewrite_error"],
        "model": row["model"],
        "kind": row["kind"],
        "n": row["n"],
        "size": f"{row['width']}x{row['height']}",
        "steps": row["steps"],
        "seeds": json.loads(row["seeds"]),
        "contextImage": f"/playground/images/{context}" if context else None,
        "status": row["status"],
        "error": row["error"],
        "images": images,
        "createdAt": row["created_at"],
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
    }


class PlaygroundStore:
    """The playground's durable state: SQLite plus its own image directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.images_dir = self.directory / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # `check_same_thread=False`: the connection is reached both from the
        # event loop and from `TestClient`'s caller thread, and every statement
        # below goes through `self._lock`, which is the serialization sqlite3's
        # own check stands in for.
        self._db = sqlite3.connect(
            self.directory / "playground.db",
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Additive column migrations for a database an older build created.

        `CREATE TABLE IF NOT EXISTS` adds no column to a table that already
        exists, so a store written before `group_id` would keep a schema the
        reads below assume. Every existing row is its own group: that is what the
        feed showed before groups existed.
        """
        columns = {row["name"] for row in self._db.execute("PRAGMA table_info(generations)")}
        if "group_id" not in columns:
            self._db.execute("ALTER TABLE generations ADD COLUMN group_id TEXT")
            self._db.execute("UPDATE generations SET group_id = id WHERE group_id IS NULL")
            logger.info("playground store migrated: generations.group_id added")
        if "negative_prompt" not in columns:
            self._db.execute("ALTER TABLE generations ADD COLUMN negative_prompt TEXT")
            logger.info("playground store migrated: generations.negative_prompt added")
        # Both NULL on every existing row, which reads exactly right: nothing
        # rewrote them, and nothing failed to.
        if "rewritten_prompt" not in columns:
            self._db.execute("ALTER TABLE generations ADD COLUMN rewritten_prompt TEXT")
            logger.info("playground store migrated: generations.rewritten_prompt added")
        if "rewrite_error" not in columns:
            self._db.execute("ALTER TABLE generations ADD COLUMN rewrite_error TEXT")
            logger.info("playground store migrated: generations.rewrite_error added")
        if "rewrite_pending" not in columns:
            self._db.execute(
                "ALTER TABLE generations ADD COLUMN rewrite_pending INTEGER NOT NULL DEFAULT 0"
            )
            # An unreleased build tracked this as the literal 'pending' in
            # `rewrite_error`, which is published and rendered as "Enhancing
            # failed (pending)". Those rows would carry that claim forever, so
            # they are cleared here rather than left to be read as failures.
            # Narrow by construction: `record_rewrite` writes a sentence, never
            # this word, so nothing legitimate matches.
            self._db.execute(
                "UPDATE generations SET rewrite_error = NULL WHERE rewrite_error = 'pending'"
            )
            logger.info("playground store migrated: generations.rewrite_pending added")
        columns = {row["name"] for row in self._db.execute("PRAGMA table_info(sessions)")}
        if "password" not in columns:
            self._db.execute("ALTER TABLE sessions ADD COLUMN password TEXT")
            logger.info("playground store migrated: sessions.password added")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── Sessions ───────────────────────────────────────────────────────────

    def create_session(self) -> dict[str, Any]:
        now = time.time()
        session_id = uuid.uuid4().hex
        with self._lock:
            self._db.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, NULL, ?, ?)",
                (session_id, now, now),
            )
        return {
            "id": session_id,
            "title": None,
            "createdAt": now,
            "updatedAt": now,
            "generating": False,
            "locked": False,
        }

    def rename_session(self, session_id: str, title: str | None) -> dict[str, Any] | None:
        """Set a user-chosen title; `None` (or blank) returns the session to the
        auto-title, which `add_generation` fills from the next prompt."""
        now = time.time()
        cleaned = (title or "").strip()[:TITLE_LIMIT] or None
        with self._lock:
            cursor = self._db.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (cleaned, now, session_id),
            )
            if cursor.rowcount == 0:
                return None
            return self._session_row(session_id)

    def _session_row(self, session_id: str) -> dict[str, Any] | None:
        """Caller holds the lock."""
        row = self._db.execute(
            """
            SELECT s.*, EXISTS (
              SELECT 1 FROM generations g
              WHERE g.session_id = s.id AND g.status IN ('queued', 'running')
            ) AS generating
            FROM sessions s WHERE s.id = ?
            """,
            (session_id,),
        ).fetchone()
        return None if row is None else _session_json(row, generating=bool(row["generating"]))

    def session_summary(self, session_id: str) -> dict[str, Any] | None:
        """The session row alone, without its generations."""
        with self._lock:
            return self._session_row(session_id)

    # ── Passwords ──────────────────────────────────────────────────────────
    #
    # Persistence only: hashing and verification are `credential`'s, and they
    # run outside this lock — a scrypt takes ~100 ms and nothing else should
    # wait on it.

    def password_record(self, session_id: str) -> dict[str, Any] | None:
        """The stored hash record, `None` for an open session. `KeyError` when
        the session does not exist — the caller must tell those apart."""
        with self._lock:
            row = self._db.execute(
                "SELECT password FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        if row["password"] is None:
            return None
        try:
            record = json.loads(row["password"])
        except json.JSONDecodeError:
            # Damaged is still locked: `verify_record` fails closed on it.
            return {}
        return record if isinstance(record, dict) else {}

    def set_password(self, session_id: str, record: dict[str, Any] | None) -> bool:
        """Store a hash record, or clear it with `None`. False if no such session."""
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "UPDATE sessions SET password = ?, updated_at = ? WHERE id = ?",
                (None if record is None else json.dumps(record), now, session_id),
            )
        return cursor.rowcount > 0

    def session_of_image(self, filename: str) -> str | None:
        """Which session a file belongs to — generated or uploaded as context."""
        with self._lock:
            row = self._db.execute(
                """
                SELECT g.session_id AS session_id
                FROM generation_images gi JOIN generations g ON g.id = gi.generation_id
                WHERE gi.filename = ?
                UNION
                SELECT session_id FROM generations WHERE context_image = ?
                LIMIT 1
                """,
                (filename, filename),
            ).fetchone()
        return None if row is None else row["session_id"]

    def generated_image(self, filename: str) -> dict[str, Any] | None:
        """A *generated* image and its lineage, or None.

        Deliberately not `session_of_image`, which also matches
        `context_image`. Two reasons an upscale must not accept one of those:
        a context file keeps whatever suffix it was uploaded with, so it may be
        a JPEG or a WebP rather than the PNG this promises, and it has no seed
        -- while `generation_images.seed` is NOT NULL, so the row an upscale
        writes would have nothing to put there.

        The filename never reaches the filesystem before a row has matched,
        which is the traversal guard `GET /playground/images/{filename}`
        already relies on.
        """
        with self._lock:
            row = self._db.execute(
                """
                SELECT g.session_id AS session_id, g.group_id AS group_id,
                       g.prompt AS prompt, g.model AS model, gi.seed AS seed
                FROM generation_images gi JOIN generations g ON g.id = gi.generation_id
                WHERE gi.filename = ?
                LIMIT 1
                """,
                (filename,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "group_id": row["group_id"] or row["session_id"],
            "prompt": row["prompt"],
            "model": row["model"],
            "seed": row["seed"],
        }

    def generation_of_image(self, filename: str) -> dict[str, Any] | None:
        """Everything needed to *replay* a generated image, or None.

        `generated_image` above answers "who owns this file", which is what an
        upscale needs. A refinement or a variation needs more: it re-runs the
        settings that produced the image, so it needs the size, the step count,
        the kind, the negative and rewritten prompts, and the reference the
        original started from.

        Raw column names rather than `_generation_json`, deliberately. That
        serializer rewrites `context_image` into a URL for a browser, and a
        replay needs the filename to copy the file -- a URL would have to be
        parsed back into one, which is the sort of round trip that survives
        review and then breaks the first time the route changes.

        Restricted to `generation_images` for the same reason `generated_image`
        is: a context file has no seed and may not be a PNG.
        """
        with self._lock:
            row = self._db.execute(
                """
                SELECT g.id AS id, g.session_id AS session_id, g.group_id AS group_id,
                       g.prompt AS prompt, g.negative_prompt AS negative_prompt,
                       g.rewritten_prompt AS rewritten_prompt, g.model AS model,
                       g.kind AS kind, g.width AS width, g.height AS height,
                       g.steps AS steps, g.steps_from_preset AS steps_from_preset,
                       g.image_strength AS image_strength, g.context_image AS context_image,
                       gi.seed AS seed
                FROM generation_images gi JOIN generations g ON g.id = gi.generation_id
                WHERE gi.filename = ?
                LIMIT 1
                """,
                (filename,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            # A generation with no group is its own group, the same convention
            # `generated_image` applies.
            "group_id": row["group_id"] or row["session_id"],
            "prompt": row["prompt"],
            "negative_prompt": row["negative_prompt"],
            "rewritten_prompt": row["rewritten_prompt"],
            "model": row["model"],
            "kind": row["kind"],
            "width": row["width"],
            "height": row["height"],
            "steps": row["steps"],
            "steps_from_preset": bool(row["steps_from_preset"]),
            "image_strength": row["image_strength"],
            "context_image": row["context_image"],
            "seed": row["seed"],
        }

    def session_of_generation(self, generation_id: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT session_id FROM generations WHERE id = ?", (generation_id,)
            ).fetchone()
        return None if row is None else row["session_id"]

    def session_of_group(self, group_id: str) -> str | None:
        """Which session a lineage belongs to. `None` when no such group exists.

        A group is not a row of its own -- it is a column several generations
        share -- so this is the only way to name the session that must be
        unlocked before the group may be touched.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT session_id FROM generations WHERE group_id = ? LIMIT 1", (group_id,)
            ).fetchone()
        return None if row is None else row["session_id"]

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT s.*, EXISTS (
                  SELECT 1 FROM generations g
                  WHERE g.session_id = s.id AND g.status IN ('queued', 'running')
                ) AS generating
                FROM sessions s ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [_session_json(row, generating=bool(row["generating"])) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """The session and its generations, oldest first, images included.

        `rowid` breaks ties on `created_at`, which is a `time.time()` float and
        can repeat: the first generation of a group is the one whose request the
        feed shows and whose settings a new image of that group reuses, so insert
        order has to decide it rather than the clock's resolution.
        """
        with self._lock:
            row = self._db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            generations = self._db.execute(
                "SELECT * FROM generations WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
            generating = any(g["status"] in ACTIVE_STATUSES for g in generations)
            images = self._images_for(tuple(g["id"] for g in generations))
        return {
            "session": _session_json(row, generating=generating),
            "generations": [_generation_json(g, images.get(g["id"], [])) for g in generations],
        }

    def delete_session(self, session_id: str) -> list[str]:
        """Drop the session and report the image files the caller must unlink.

        The files are deleted by the caller, not here: an `unlink` failure must
        not leave the row half-deleted, and the rows are the record of truth
        about what exists.
        """
        now = time.time()
        with self._lock:
            filenames = [
                row["filename"]
                for row in self._db.execute(
                    """
                    SELECT filename FROM generation_images
                    WHERE generation_id IN (SELECT id FROM generations WHERE session_id = ?)
                    """,
                    (session_id,),
                )
            ]
            filenames += [
                row["context_image"]
                for row in self._db.execute(
                    "SELECT context_image FROM generations "
                    "WHERE session_id = ? AND context_image IS NOT NULL",
                    (session_id,),
                )
            ]
            # Terminal before the cascade removes them: a worker holding one of
            # these ids must see a status that tells it to stop.
            self._db.execute(
                "UPDATE generations SET status = 'cancelled', finished_at = ? "
                "WHERE session_id = ? AND status IN ('queued', 'running')",
                (now, session_id),
            )
            self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return filenames

    # ── Generations ────────────────────────────────────────────────────────

    def add_generation(
        self,
        session_id: str,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        rewrite: bool = False,
        rewritten_prompt: str | None = None,
        model: str,
        kind: str,
        n: int,
        width: int,
        height: int,
        steps: int,
        steps_from_preset: bool,
        seeds: list[int],
        image_strength: float | None = None,
        context_image: str | None = None,
        group: str | None = None,
    ) -> dict[str, Any]:
        """Record a queued generation.

        `group` appends this generation to an existing lineage instead of opening
        one. It is checked against the session, under the same lock as the
        insert: a group id is client-supplied, and a session must not be able to
        graft its generations onto another one's group. Raises `KeyError` for an
        unknown session and `ValueError` for a group that is not this session's.

        `rewrite` and `rewritten_prompt` are two different requests and only one
        may be made at a time. `rewrite=True` asks the runner to expand the
        prompt before generating; `rewritten_prompt` supplies text a previous
        generation already produced, which is how a variation reproduces its
        ancestor rather than re-sampling a *different* rewrite and calling the
        result a variation.
        """
        if rewrite and rewritten_prompt is not None:
            raise ValueError("a generation cannot both request a rewrite and supply one")
        now = time.time()
        generation_id = uuid.uuid4().hex
        with self._lock:
            session = self._db.execute(
                "SELECT id, title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise KeyError(session_id)
            if group is not None and not self._db.execute(
                "SELECT 1 FROM generations WHERE session_id = ? AND group_id = ? LIMIT 1",
                (session_id, group),
            ).fetchone():
                raise ValueError(group)
            self._db.execute(
                """
                INSERT INTO generations (
                  id, session_id, group_id, prompt, negative_prompt,
                  rewritten_prompt, rewrite_pending, model, kind, n,
                  width, height, steps, steps_from_preset, seeds, image_strength,
                  context_image, status, error, created_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', NULL, ?, NULL, NULL)
                """,
                (
                    generation_id,
                    session_id,
                    group or generation_id,
                    prompt,
                    negative_prompt,
                    rewritten_prompt,
                    # `_run` reads this to decide whether a rewrite is still owed.
                    int(rewrite),
                    model,
                    kind,
                    n,
                    width,
                    height,
                    steps,
                    int(steps_from_preset),
                    json.dumps(seeds),
                    image_strength,
                    context_image,
                    now,
                ),
            )
            if session["title"] is None:
                self._db.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (prompt.strip()[:TITLE_LIMIT] or None, now, session_id),
                )
            else:
                self._db.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
                )
            row = self._db.execute(
                "SELECT * FROM generations WHERE id = ?", (generation_id,)
            ).fetchone()
        return _generation_json(row, [])

    def get_generation(self, generation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM generations WHERE id = ?", (generation_id,)
            ).fetchone()
            if row is None:
                return None
            images = self._images_for((generation_id,))
        return _generation_json(row, images.get(generation_id, []))

    def claim(self, generation_id: str) -> sqlite3.Row | None:
        """Move a queued generation to `running`, once.

        Conditional on the current status, so a cancellation or a deletion that
        landed while the id sat on the queue wins the race by construction.
        """
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "UPDATE generations SET status = 'running', started_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (now, generation_id),
            )
            if cursor.rowcount == 0:
                return None
            return self._db.execute(
                "SELECT * FROM generations WHERE id = ?", (generation_id,)
            ).fetchone()

    def record_rewrite(
        self, generation_id: str, *, prompt: str | None, error: str | None
    ) -> None:
        """Record what the rewriter produced, or why it produced nothing.

        Writes all three columns every time, two of them to NULL as needed.

        Clearing `rewrite_pending` has no observable effect today -- `_run`
        tracks the flag in a local and `claim()` moves a row out of `queued`
        exactly once, so no row is ever re-read for it. It is written anyway so
        that the row's state matches what happened: a reader that trusts the
        column, including a future one that re-runs a generation, must not see
        work still owed that was already done.

        Deliberately unconditional on status. A generation cancelled between the
        rewrite and the first image still gets its rewrite recorded: the work
        happened, and hiding it would make the cancellation look like the reason
        no rewrite is shown.
        """
        with self._lock:
            self._db.execute(
                "UPDATE generations SET rewritten_prompt = ?, rewrite_error = ?, "
                "rewrite_pending = 0 WHERE id = ?",
                (prompt, error, generation_id),
            )

    def status_of(self, generation_id: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT status FROM generations WHERE id = ?", (generation_id,)
            ).fetchone()
        return None if row is None else row["status"]

    def cancel_queued(self, generation_id: str) -> bool:
        """Cancel a generation that has not started. Nothing else can be."""
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "UPDATE generations SET status = 'cancelled', finished_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (now, generation_id),
            )
            if cursor.rowcount:
                self._db.execute(
                    "UPDATE sessions SET updated_at = ? "
                    "WHERE id = (SELECT session_id FROM generations WHERE id = ?)",
                    (now, generation_id),
                )
            return bool(cursor.rowcount)

    def finish(self, generation_id: str, status: str, error: str | None = None) -> None:
        now = time.time()
        with self._lock:
            self._db.execute(
                "UPDATE generations SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, now, generation_id),
            )
            self._db.execute(
                "UPDATE sessions SET updated_at = ? "
                "WHERE id = (SELECT session_id FROM generations WHERE id = ?)",
                (now, generation_id),
            )

    def add_image(self, generation_id: str, position: int, filename: str, seed: int) -> bool:
        """Attach a finished image. False when the generation is gone.

        The foreign key is the test: a session deleted while its generation was
        running leaves no row to attach to, and the caller unlinks the file it
        just wrote rather than orphaning it in the directory.
        """
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO generation_images (generation_id, position, filename, seed) "
                    "VALUES (?, ?, ?, ?)",
                    (generation_id, position, filename, seed),
                )
            except sqlite3.IntegrityError:
                return False
            self._db.execute(
                "UPDATE sessions SET updated_at = ? "
                "WHERE id = (SELECT session_id FROM generations WHERE id = ?)",
                (time.time(), generation_id),
            )
            return True

    def delete_image(self, filename: str) -> tuple[str, str] | None:
        """Drop one image row. The `(session_id, group_id)` of the row that
        matched, `None` when no row has that filename.

        The caller unlinks the file, and only after a row matched: filenames come
        from the URL, and the row's existence is what proves the name is one of
        ours rather than a traversal attempt. The group is reported so the caller
        can dissolve it if that was its last image.
        """
        now = time.time()
        with self._lock:
            row = self._db.execute(
                "SELECT g.session_id, g.group_id FROM generation_images gi "
                "JOIN generations g ON g.id = gi.generation_id WHERE gi.filename = ?",
                (filename,),
            ).fetchone()
            if row is None:
                return None
            self._db.execute("DELETE FROM generation_images WHERE filename = ?", (filename,))
            self._db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, row["session_id"])
            )
            return (row["session_id"], row["group_id"])

    def dissolve_empty_group(self, group_id: str) -> list[str]:
        """Drop a group's generations when its last image has been deleted.

        The feed renders a group as one entry, and an entry that is nothing but a
        prompt is noise: deleting the last image deletes the group. The files the
        caller must unlink come back — the image rows cascade, but a member's
        `context_image` is a plain column and would otherwise stay on disk.

        The route calls this after *every* image deletion, so the "last" test
        lives here, under the same lock the deletion and the worker's `claim`
        take: a group that still has an image — or a member still `queued` or
        `running`, which will bring the group's next image — is left alone.
        Dissolving either would be data loss.
        """
        now = time.time()
        with self._lock:
            remaining = self._db.execute(
                "SELECT COUNT(*) AS count FROM generation_images gi "
                "JOIN generations g ON g.id = gi.generation_id WHERE g.group_id = ?",
                (group_id,),
            ).fetchone()["count"]
            active = self._db.execute(
                "SELECT 1 FROM generations "
                "WHERE group_id = ? AND status IN ('queued', 'running') LIMIT 1",
                (group_id,),
            ).fetchone()
            if remaining or active is not None:
                return []
            rows = self._db.execute(
                "SELECT session_id, context_image FROM generations WHERE group_id = ?",
                (group_id,),
            ).fetchall()
            if not rows:
                return []
            self._db.execute("DELETE FROM generations WHERE group_id = ?", (group_id,))
            self._db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, rows[0]["session_id"])
            )
        return [row["context_image"] for row in rows if row["context_image"]]

    def delete_group(self, group_id: str) -> list[str] | None:
        """Delete a whole lineage and report the files the caller must unlink.

        The deliberate difference from `dissolve_empty_group`: this one does not
        refuse a group with a `queued` or `running` member, it *cancels* it. The
        two are answers to different questions. Dissolving happens because the
        last image of a group was deleted, and a member still to come means the
        group is not empty after all; this happens because the user asked for the
        entry to go, and a member still to come is one more thing to stop.

        `delete_session` is the model, down to the ordering: the active rows are
        made terminal *before* the cascade removes them, so a worker holding one
        of these ids sees a status that tells it to stop rather than a missing
        row. Files are unlinked by the caller, never here -- an `unlink` failure
        must not leave the rows half-deleted, and the rows are the record of
        truth about what exists.

        `None` when no such group exists, which the route turns into a 404.
        """
        now = time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id, context_image FROM generations WHERE group_id = ?",
                (group_id,),
            ).fetchall()
            if not rows:
                return None
            session_id = rows[0]["session_id"]
            filenames = [
                row["filename"]
                for row in self._db.execute(
                    """
                    SELECT filename FROM generation_images
                    WHERE generation_id IN (SELECT id FROM generations WHERE group_id = ?)
                    """,
                    (group_id,),
                )
            ]
            filenames += [row["context_image"] for row in rows if row["context_image"]]
            self._db.execute(
                "UPDATE generations SET status = 'cancelled', finished_at = ? "
                "WHERE group_id = ? AND status IN ('queued', 'running')",
                (now, group_id),
            )
            self._db.execute("DELETE FROM generations WHERE group_id = ?", (group_id,))
            self._db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
        return filenames

    def mark_interrupted(self) -> int:
        """Fail every generation left mid-flight by a previous process."""
        with self._lock:
            cursor = self._db.execute(
                "UPDATE generations SET status = 'failed', "
                "error = 'Interrupted by server restart', finished_at = ? "
                "WHERE status IN ('queued', 'running')",
                (time.time(),),
            )
        if cursor.rowcount:
            logger.info(
                "%d playground generation(s) interrupted by a restart", cursor.rowcount
            )
        return cursor.rowcount

    # ── Files ──────────────────────────────────────────────────────────────

    def save_image(self, data: bytes) -> str:
        name = f"{uuid.uuid4().hex}.png"
        (self.images_dir / name).write_bytes(data)
        return name

    def context_path(self, suffix: str) -> Path:
        """Where an uploaded reference image goes. Named for what it is."""
        return self.images_dir / f"ctx-{uuid.uuid4().hex}{suffix or '.png'}"

    def unlink(self, filenames: list[str]) -> None:
        for name in filenames:
            (self.images_dir / name).unlink(missing_ok=True)

    # ── Internals ──────────────────────────────────────────────────────────

    def _images_for(self, ids: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
        """Images of several generations at once, called under the lock."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT * FROM generation_images WHERE generation_id IN ({placeholders}) "
            "ORDER BY generation_id, position",
            ids,
        ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["generation_id"], []).append(
                {"url": f"/playground/images/{row['filename']}", "seed": row["seed"]}
            )
        return grouped


class PlaygroundRunner:
    """One worker, one queue: playground generations run in submission order.

    It does not own the engine, and does not touch its lock: `engine.generate()`
    serializes against `/v1` requests exactly as it always did. What this adds is
    the *record* of the run, which is what makes a closed browser tab survivable.

    **Pausing** holds this queue and nothing else. `/v1` bypasses the runner
    entirely, so a paused playground does not stop the server generating -- it
    stops *this* queue starting anything more. It takes effect at the two
    boundaries the runner owns, before a claim and between the images of an
    `n>1` run; the image already being denoised always finishes, because the
    engine can only be interrupted by raising at a step and the alternative is
    throwing away work already paid for. Pause is runtime state: a restart
    clears it, and whatever it was holding is failed by `mark_interrupted()`.
    """

    def __init__(
        self,
        store: PlaygroundStore,
        engine: Any,
        idle_unloader: Any,
        resolve_spec: Callable[[str | None], ModelSpec],
        resolve_upscaler: Callable[[str], Any] | None = None,
        build_rewrite_job: Callable[[str], Any] | None = None,
    ):
        self._store = store
        self._engine = engine
        self._idle = idle_unloader
        self._resolve_spec = resolve_spec
        #: Injected the same way `resolve_spec` is, so the runner keeps its one
        #: dependency direction: it is handed how to look things up, it does not
        #: reach into a catalogue itself.
        self._resolve_upscaler = resolve_upscaler
        #: Turns a prompt into a `RewriteJob`, or returns None when rewriting is
        #: not configured. Injected for the same reason as the two above, and
        #: with one extra consequence: the runner never reads `Settings`, so it
        #: cannot drift from the answer the route gave at admission.
        self._build_rewrite_job = build_rewrite_job
        #: Created by `start()`, not here: an `asyncio.Queue` binds to the loop
        #: that first uses it, and one app may be run by several loops in turn
        #: (every `TestClient` context is a fresh one). Binding it at startup
        #: keeps the queue and the worker on the same loop by construction.
        self._queue: asyncio.Queue[str] | None = None
        self._task: asyncio.Task[None] | None = None
        #: Held work: True stops the runner starting anything more. Global, and
        #: runtime-only -- a restart clears it, and `mark_interrupted()` fails
        #: whatever was still held. Never persisted, because the queue it holds
        #: is not persisted either: `submit()` is a `put_nowait` on an in-memory
        #: queue, so a paused backlog does not survive a restart in any case.
        self._paused = False
        #: Wakes a parked worker. Created by `start()` for the same reason the
        #: queue is: it binds to the loop that first uses it.
        #:
        #: A `Condition` rather than an `Event` because the wake predicate has
        #: two arms -- "resumed" *or* "this generation was cancelled" -- and an
        #: `Event` can express only the first. Nothing takes this while holding
        #: `PlaygroundStore._lock`; the reverse order would deadlock.
        self._gate: asyncio.Condition | None = None
        #: The generation the worker is on, if any. Read by `cancel`, which must
        #: distinguish "still queued" (a row update suffices) from "running now".
        self.current_id: str | None = None
        #: Cancellation asked of the *running* generation, held here rather than
        #: left to the engine alone. `engine.request_cancel()` only bites while
        #: MLX is denoising: during the weight load, while queued behind another
        #: caller on the engine lock, and between the images of an `n>1` run it
        #: returns False and does nothing. Those are the longest moments of a
        #: run — precisely when someone presses Cancel — so the runner keeps the
        #: request and honours it at the next boundary it owns.
        self._cancel_requested: str | None = None

    def start(self) -> None:
        if self._task is None:
            self._queue = asyncio.Queue()
            self._gate = asyncio.Condition()
            # Reset with the queue, not left over from a previous run: `shutdown`
            # drops the task without clearing them, and a stale `current_id` or a
            # stale pause would apply to a queue that no longer exists.
            self._paused = False
            self.current_id = None
            self._cancel_requested = None
            self._task = asyncio.create_task(self._work(), name="playground-runner")

    async def shutdown(self) -> None:
        """Stop the worker. Any row it was on stays `running`.

        Deliberately: the next startup's `mark_interrupted()` is what turns it
        into a terminal `failed`, and it can do that correctly for a process that
        was killed outright — which this path cannot count on being the case.
        """
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @property
    def paused(self) -> bool:
        return self._paused

    async def set_paused(self, paused: bool) -> None:
        """Hold or release the queue.

        Holding takes effect at the next boundary the runner owns -- before a
        queued generation is claimed, and between the images of an `n>1` run.
        The image already being denoised always finishes: the engine can only be
        interrupted by raising at a denoising step, so there is no third option
        that does not throw away the work already paid for.

        A 200 from here therefore does **not** mean nothing is generating.
        """
        gate = self._gate
        if gate is None:  # pragma: no cover - the routes exist only once started
            raise RuntimeError("The playground runner is not running.")
        async with gate:
            self._paused = paused
            gate.notify_all()
        logger.info("playground queue %s", "paused" if paused else "resumed")

    async def _await_resume(self, generation_id: str) -> bool:
        """Park until the queue is released. False when the caller must stop.

        `wait_for` under the lock, never a bare `if paused: await wait()`: a
        release landing between the flag read and the lock acquisition would be
        lost, and the worker would park for good.

        The cancellation arm is what makes a Cancel pressed on a *parked* run
        land at once instead of on resume. It only ever fires at the per-image
        gate: at the pre-claim gate the row is still `queued`, so `cancel()`
        settles it through `cancel_queued()` and this worker simply finds
        nothing to claim.
        """
        gate = self._gate
        assert gate is not None  # set by `start()`, which creates this task
        if not self._paused:
            return self._cancel_requested != generation_id
        async with gate:
            await gate.wait_for(
                lambda: not self._paused or self._cancel_requested == generation_id
            )
        return self._cancel_requested != generation_id

    def submit(self, generation_id: str) -> None:
        if self._queue is None:  # pragma: no cover - the routes exist only once started
            raise RuntimeError("The playground runner is not running.")
        self._queue.put_nowait(generation_id)

    async def cancel(self, generation_id: str) -> dict[str, Any] | None:
        """Cancel by id, as far as this server can.

        Three cases, in order of how much has already happened:

        * **queued** — the row is cancelled and the worker will skip it.
        * **running, mid-denoise** — `engine.request_cancel()` stops it at the
          next step, which surfaces as `generation_stopped` below.
        * **running, not denoising** (loading weights, waiting on the engine
          lock, between images) — the engine refuses, so the request is recorded
          and applied at the next image boundary. The image being computed when
          the request lands is kept: it is paid for, and the record says
          `cancelled` with the images it did produce.

        A fourth case rides on the third: while the queue is **paused** the
        runner is parked at a boundary, so the engine refuses the request there
        too. The gate is woken so the record settles now rather than on resume.

        `engine.request_cancel()` is global. With one generation at a time that
        is nearly always this one; when an external `/v1` request holds the
        engine, it is that request which stops — the same semantics `/v1/cancel`
        already exposes at this auth level.
        """
        if not self._store.cancel_queued(generation_id) and generation_id == self.current_id:
            self._cancel_requested = generation_id
            self._engine.request_cancel()
            # A parked generation is not denoising, so `request_cancel` did
            # nothing: the worker is the only thing that can settle it, and it is
            # waiting on the gate. Waking it here is what keeps the record this
            # call returns from being read before the cancellation applies.
            await self._wake()
        return self._store.get_generation(generation_id)

    async def _wake(self) -> None:
        """Re-evaluate the gate's predicate on a parked worker, if there is one."""
        gate = self._gate
        if gate is None:  # pragma: no cover - the routes exist only once started
            return
        async with gate:
            gate.notify_all()
        # Let a woken worker settle its record before the caller reads it back.
        await asyncio.sleep(0)

    async def cancel_running_in(self, session_ids: set[str]) -> None:
        """Stop the current generation if it belongs to one of these sessions."""
        await self._stop_current(lambda row: row["sessionId"] in session_ids)

    async def cancel_running_in_group(self, group_id: str) -> None:
        """Stop the current generation if it belongs to this lineage."""
        await self._stop_current(lambda row: row["groupId"] == group_id)

    async def _stop_current(self, matches: Callable[[dict[str, Any]], bool]) -> None:
        """Stop the running generation when it matches, as far as this can.

        The recorded request is not a belt-and-braces copy of what the engine was
        told. `engine.request_cancel()` returns False and does nothing unless MLX
        is actually denoising (`engine.py`), so while the weights load, while the
        runner waits on the engine lock, between the images of an `n>1` run, and
        while the queue is paused, asking the engine alone loses the request
        entirely. Recording it here is what makes the runner honour it at the
        next boundary it owns -- the same mechanism `cancel` already relies on.
        """
        current = self.current_id
        if current is None:
            return
        row = self._store.get_generation(current)
        if row is None or not matches(row):
            return
        self._cancel_requested = current
        self._engine.request_cancel()
        await self._wake()

    async def _work(self) -> None:
        queue = self._queue
        assert queue is not None  # set by `start()`, which creates this task
        while True:
            generation_id = await queue.get()
            try:
                # Held before the claim, not after: the row stays `queued`, which
                # is what the feed should show, and nothing is half-started.
                if await self._await_resume(generation_id):
                    await self._run(generation_id)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                # One broken generation must not take the worker with it: every
                # later submission would then sit `queued` forever.
                logger.exception("Playground generation %s failed unexpectedly", generation_id)
                self._store.finish(generation_id, "failed", "Internal error")
            finally:
                self.current_id = None
                if self._cancel_requested == generation_id:
                    self._cancel_requested = None

    async def _run_upscale(self, generation_id: str, row: Any) -> None:
        """One enlargement: no seed loop, one image, one terminal status.

        Keeps the boundaries `_run` owns and drops the rest. `_await_resume`
        before starting, so a paused queue holds an upscale as it holds a
        generation; `with self._idle:` around the work, so the idle countdown
        is not armed while it runs; a cancellation check before committing, so
        a request that arrived during the work is not lost.
        """
        from qds.engine import UpscaleJob

        if self._resolve_upscaler is None:  # pragma: no cover - wired at start-up
            self._store.finish(generation_id, "failed", "Upscaling is not configured.")
            return

        spec = self._resolve_upscaler(row["model"])
        if spec is None:
            self._store.finish(
                generation_id, "failed", f"Unknown upscaler: {row['model']!r}."
            )
            return

        context = row["context_image"]
        if not context:  # pragma: no cover - the route always writes one
            self._store.finish(generation_id, "failed", "The source image is missing.")
            return
        source = self._store.images_dir / context
        if not source.is_file():
            # Said here rather than letting `Image.open` raise into
            # `translate_mflux_exception`, which would render a missing file as
            # "Weights or model not found" -- a real failure path with an
            # answer about something else entirely.
            self._store.finish(
                generation_id, "failed", "The source image is no longer on disk."
            )
            return

        try:
            if not await self._await_resume(generation_id):
                self._store.finish(generation_id, "cancelled")
                return
            if self._cancel_requested == generation_id:
                self._store.finish(generation_id, "cancelled")
                return

            with self._idle:
                png = await self._engine.upscale(
                    UpscaleJob(
                        spec=spec,
                        image_path=source,
                        target=(row["width"], row["height"]),
                    )
                )

            if self._cancel_requested == generation_id:
                # Requested while the tiles ran but after the last checkpoint.
                self._store.finish(generation_id, "cancelled")
                return

            filename = self._store.save_image(png)
            seeds: list[int] = json.loads(row["seeds"])
            if not self._store.add_image(generation_id, 0, filename, seeds[0] if seeds else 0):
                # The session or the row went away while this ran.
                self._store.unlink([filename])
                return
        except Exception as exc:
            error = translate_mflux_exception(exc)
            if error.code == "generation_stopped":
                self._store.finish(generation_id, "cancelled")
            else:
                self._store.finish(generation_id, "failed", error.message)
            return
        self._store.finish(generation_id, "completed")

    async def _rewrite_step(self, generation_id: str, typed: str) -> str:
        """Expand `typed`, or fall back to it and record why.

        Three outcomes, and they are deliberately not the same one.

        A **cancellation** propagates. The user asked for the run to stop, not
        for it to continue from the prompt they were trying to improve, and
        `_run`'s handler already turns `generation_stopped` into `cancelled`.

        Any **other failure** -- the weights are not downloaded, the decode
        timed out, `sanitise` refused the output -- keeps the generation and
        records the reason. Throwing away a generation the user asked for,
        because an optional step that improves it did not work, would be
        replacing detection with punishment. It is not silent: `rewrite_error`
        is a column the feed reads, so the row says "generated from your prompt,
        because ...".

        Success records the rewrite before generating, so a crash between the
        two leaves the work visible rather than lost.
        """
        typed = typed.strip()
        job = None if self._build_rewrite_job is None else self._build_rewrite_job(typed)
        if job is None:
            # Configuration changed between admission and execution. Nothing was
            # promised to the user here, so this is a fact, not a failure.
            self._store.record_rewrite(
                generation_id, prompt=None, error="Prompt rewriting is no longer configured."
            )
            return typed

        try:
            rewritten = await self._engine.rewrite(job)
        except APIError as exc:
            if exc.code == "generation_stopped":
                raise
            self._store.record_rewrite(generation_id, prompt=None, error=exc.message)
            logger.info(
                "Rewrite unavailable for %s: %s",
                generation_id,
                exc.message,
                extra={"event": "rewrite_failed", "fields": {"reason": exc.code}},
            )
            return typed
        except Exception as exc:
            self._store.record_rewrite(generation_id, prompt=None, error=str(exc))
            logger.info(
                "Rewrite failed for %s: %s",
                generation_id,
                exc,
                extra={"event": "rewrite_failed", "fields": {"reason": type(exc).__name__}},
            )
            return typed

        self._store.record_rewrite(generation_id, prompt=rewritten, error=None)
        return rewritten

    async def _run(self, generation_id: str) -> None:
        row = self._store.claim(generation_id)
        if row is None:
            # Cancelled or deleted while it waited its turn.
            return
        self.current_id = generation_id
        if row["kind"] == "upscale":
            # Diverted before `_resolve_spec`, which would raise `model_not_found`
            # on an upscaler key and fail the row with "Unknown model". An upscale
            # is a short, single job: no seed loop, no preset steps, no previews.
            await self._run_upscale(generation_id, row)
            return
        try:
            spec = self._resolve_spec(row["model"])
        except APIError as exc:
            # The model was disabled or removed between submission and execution.
            self._store.finish(generation_id, "failed", exc.message)
            return

        context = row["context_image"]
        image_path = self._store.images_dir / context if context else None
        seeds: list[int] = json.loads(row["seeds"])
        position = 0
        # What actually reaches the engine. A row that carries a rewrite from an
        # earlier generation -- a variation replaying its ancestor's -- uses it
        # as-is and asks for nothing.
        prompt = row["rewritten_prompt"] or row["prompt"]
        rewrite_pending = bool(row["rewrite_pending"])
        try:
            # One contiguous run of images per `with self._idle:` block, not one
            # per image. Unpaused there is exactly one block per generation,
            # which is the "armed per request, not per image" rule `idle.py`
            # spells out -- arming per image would release the weights *between*
            # the images of an `n=3` request at a delay of 0.
            #
            # The park is outside the block on purpose, and it is not a detail:
            # `IdleUnloader.__enter__` destroys the pending countdown and only
            # `__exit__` recreates it, on the way down to zero in-flight. The
            # unloader is one instance shared with the `/v1` plane, so parking
            # inside would suspend automatic release for the *whole server* --
            # tens of GB of unified memory held for an unbounded pause, which is
            # the exact failure the idle policy exists to prevent.
            while position < len(seeds):
                if not await self._await_resume(generation_id):
                    self._store.finish(generation_id, "cancelled")
                    return
                with self._idle:
                    if rewrite_pending:
                        # Inside this block, not before it, and that is not a
                        # matter of taste. `IdleUnloader.__enter__` destroys the
                        # pending countdown and only `__exit__` recreates it, on
                        # the way down to zero in-flight -- so a rewrite in a
                        # block of its own would arm a release *between* the
                        # rewrite and the first image, and at `idle_unload_s: 0`
                        # would unload the diffusion model there.
                        #
                        # Once, too: the outer loop re-enters this block after a
                        # pause, and the flag is what keeps a paused generation
                        # from being rewritten twice into two different prompts.
                        prompt = await self._rewrite_step(generation_id, row["prompt"])
                        rewrite_pending = False
                    while position < len(seeds) and not self._paused:
                        if self._cancel_requested == generation_id:
                            self._store.finish(generation_id, "cancelled")
                            return
                        if position and self._store.status_of(generation_id) != "running":
                            # Deleted between images.
                            return
                        seed = seeds[position]
                        png = await self._engine.generate(
                            GenerationJob(
                                spec=spec,
                                kind=row["kind"],
                                prompt=prompt,
                                negative_prompt=row["negative_prompt"],
                                width=row["width"],
                                height=row["height"],
                                steps=row["steps"],
                                seed=seed,
                                image_path=image_path,
                                image_strength=row["image_strength"],
                                steps_from_preset=bool(row["steps_from_preset"]),
                                preview_every=PREVIEW_EVERY,
                            )
                        )
                        filename = self._store.save_image(png)
                        if not self._store.add_image(generation_id, position, filename, seed):
                            self._store.unlink([filename])
                            return
                        position += 1
                        if self._cancel_requested == generation_id:
                            # Asked to stop while the engine was not in a position
                            # to be interrupted. The image above is recorded, and
                            # this is where the request finally takes effect.
                            self._store.finish(generation_id, "cancelled")
                            return
        except Exception as exc:
            # `CancelledError` is a `BaseException`: shutdown passes straight
            # through, leaving the row `running` for `mark_interrupted()`.
            error = translate_mflux_exception(exc)
            if error.code == "generation_stopped":
                self._store.finish(generation_id, "cancelled")
            else:
                self._store.finish(generation_id, "failed", error.message)
            return
        self._store.finish(generation_id, "completed")
