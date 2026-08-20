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
  updated_at REAL NOT NULL
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
        }

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
        """
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
                  id, session_id, group_id, prompt, model, kind, n, width, height,
                  steps, steps_from_preset, seeds, image_strength, context_image,
                  status, error, created_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', NULL, ?, NULL, NULL)
                """,
                (
                    generation_id,
                    session_id,
                    group or generation_id,
                    prompt,
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
    """

    def __init__(
        self,
        store: PlaygroundStore,
        engine: Any,
        idle_unloader: Any,
        resolve_spec: Callable[[str | None], ModelSpec],
    ):
        self._store = store
        self._engine = engine
        self._idle = idle_unloader
        self._resolve_spec = resolve_spec
        #: Created by `start()`, not here: an `asyncio.Queue` binds to the loop
        #: that first uses it, and one app may be run by several loops in turn
        #: (every `TestClient` context is a fresh one). Binding it at startup
        #: keeps the queue and the worker on the same loop by construction.
        self._queue: asyncio.Queue[str] | None = None
        self._task: asyncio.Task[None] | None = None
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

    def submit(self, generation_id: str) -> None:
        if self._queue is None:  # pragma: no cover - the routes exist only once started
            raise RuntimeError("The playground runner is not running.")
        self._queue.put_nowait(generation_id)

    def cancel(self, generation_id: str) -> dict[str, Any] | None:
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

        `engine.request_cancel()` is global. With one generation at a time that
        is nearly always this one; when an external `/v1` request holds the
        engine, it is that request which stops — the same semantics `/v1/cancel`
        already exposes at this auth level.
        """
        if not self._store.cancel_queued(generation_id) and generation_id == self.current_id:
            self._cancel_requested = generation_id
            self._engine.request_cancel()
        return self._store.get_generation(generation_id)

    def cancel_running_in(self, session_ids: set[str]) -> None:
        """Stop the current generation if it belongs to one of these sessions."""
        current = self.current_id
        if current is None:
            return
        row = self._store.get_generation(current)
        if row is not None and row["sessionId"] in session_ids:
            self._engine.request_cancel()

    async def _work(self) -> None:
        queue = self._queue
        assert queue is not None  # set by `start()`, which creates this task
        while True:
            generation_id = await queue.get()
            try:
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

    async def _run(self, generation_id: str) -> None:
        row = self._store.claim(generation_id)
        if row is None:
            # Cancelled or deleted while it waited its turn.
            return
        self.current_id = generation_id
        try:
            spec = self._resolve_spec(row["model"])
        except APIError as exc:
            # The model was disabled or removed between submission and execution.
            self._store.finish(generation_id, "failed", exc.message)
            return

        context = row["context_image"]
        image_path = self._store.images_dir / context if context else None
        seeds: list[int] = json.loads(row["seeds"])
        try:
            with self._idle:
                for position, seed in enumerate(seeds):
                    if self._cancel_requested == generation_id:
                        self._store.finish(generation_id, "cancelled")
                        return
                    if position and self._store.status_of(generation_id) != "running":
                        # Deleted between images.
                        return
                    png = await self._engine.generate(
                        GenerationJob(
                            spec=spec,
                            kind=row["kind"],
                            prompt=row["prompt"],
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
                    if self._cancel_requested == generation_id:
                        # Asked to stop while the engine was not in a position to
                        # be interrupted. The image above is recorded, and this
                        # is where the request finally takes effect.
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
