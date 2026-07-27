"""Moteur d'inférence in-process, avec un modèle gardé chaud en mémoire.

C'est le cœur du serveur. Le prototype lançait un binaire `mflux-*` par
image : les poids étaient rechargés à chaque requête. Ici le modèle reste
en mémoire entre les appels, ce qui fait passer une génération de plusieurs
minutes à quelques secondes une fois le modèle chargé.

Trois invariants :

* **un seul modèle vivant à la fois** — sur mémoire unifiée, en garder deux
  (un 9B + un autre) sature la machine ;
* **une seule génération à la fois** — un `asyncio.Lock` sérialise tout, et
  l'inférence tourne sur un unique thread worker, jamais dans la boucle
  d'événements ;
* **un seul callback enregistré par modèle** — `CallbackRegistry` de mflux
  n'a pas d'`unregister`, enregistrer par requête ferait grossir la liste
  indéfiniment.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mflux_server.errors import APIError, GenerationTimeout, translate_mflux_exception
from mflux_server.logs import SERVER_LOGGER, capture_stdout
from mflux_server.registry import ModelSpec, load_model

logger = logging.getLogger(SERVER_LOGGER)

#: Sous-modules à libérer au déchargement. Reprend ce que fait le
#: `MemorySaver` de mflux en interne (callbacks/instances/memory_saver.py:76-107),
#: faute de méthode de teardown publique dans la librairie.
_UNLOADABLE_ATTRS = (
    "transformer",
    "text_encoder",
    "clip_text_encoder",
    "t5_text_encoder",
    "vae",
    "image_encoder",
    "qwen_vl_encoder",
)

#: Au-delà, on purge le cache d'embeddings de Qwen : il est indexé par prompt
#: et n'a aucune borne (qwen_prompt_encoder.py:20-41).
_PROMPT_CACHE_MAX_ENTRIES = 16


@dataclass
class ProgressSnapshot:
    """État courant du moteur, lisible sans verrou.

    Écrit depuis le thread worker, lu depuis la boucle d'événements : ce sont de
    simples affectations d'attributs, donc atomiques sous le GIL. Un seul
    instantané suffit parce que `ModelEngine._lock` sérialise les générations —
    il n'y a jamais deux jobs en vol. C'est ce qui permet à `/v1/progress` de
    sonder plutôt que de bâtir une file inter-threads, et à plusieurs
    consommateurs SSE de coexister gratuitement.
    """

    state: str = "idle"  # idle | loading | generating
    model: str | None = None
    kind: str | None = None
    seed: int | None = None
    step: int = 0
    total: int = 0
    started_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        elapsed = None if self.started_at is None else round(time.monotonic() - self.started_at, 1)
        return {
            "state": self.state,
            "model": self.model,
            "kind": self.kind,
            "seed": self.seed,
            "step": self.step,
            "total": self.total,
            "elapsed_s": elapsed,
        }

    def reset(self) -> None:
        self.state = "idle"
        self.model = None
        self.kind = None
        self.seed = None
        self.step = 0
        self.total = 0
        self.started_at = None


@dataclass
class GenerationJob:
    spec: ModelSpec
    kind: str  # "txt2img" | "edit"
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    guidance: float | None = None
    negative_prompt: str | None = None
    image_path: Path | None = None
    image_strength: float | None = None


class _ProgressCallback:
    """Callback persistant : journalise la progression, arme le timeout et l'annulation.

    Interrompre une génération n'est possible que d'ici : une opération MLX
    en cours ne se laisse pas annuler depuis l'extérieur, et `asyncio` ne
    peut pas tuer le thread worker. Lever une exception depuis `call_in_loop`
    remonte bien — mflux n'intercepte que `KeyboardInterrupt` dans sa boucle.
    L'annulation demandée par `/v1/cancel` empruntre donc exactement le chemin
    déjà emprunté par le timeout.
    """

    def __init__(self, log_every: int = 1, progress: ProgressSnapshot | None = None):
        self._log_every = log_every
        self._progress = progress or ProgressSnapshot()
        self.deadline: float | None = None
        self.label: str = ""
        self.timed_out: bool = False
        self.cancel_requested: bool = False
        self.cancelled: bool = False

    def arm(self, label: str, deadline: float | None) -> None:
        self.label = label
        self.deadline = deadline
        self.timed_out = False
        self.cancel_requested = False
        self.cancelled = False

    def call_in_loop(self, *, t: int, config: Any, time_steps: Any, **_: Any) -> None:
        from mflux.utils.exceptions import StopImageGenerationException

        step = t + 1
        total = config.num_inference_steps

        if self.cancel_requested:
            self.cancelled = True
            raise StopImageGenerationException(f"Génération annulée à l'étape {step}/{total}")

        if self.deadline is not None and time.monotonic() > self.deadline:
            self.timed_out = True
            raise StopImageGenerationException(f"Timeout à l'étape {step}/{total}")

        self._progress.step = step
        self._progress.total = total
        if self._log_every and (step % self._log_every == 0 or step == total):
            logger.info(
                "%s — étape %d/%d",
                self.label,
                step,
                total,
                extra={"event": "generation_step", "fields": {"step": step, "total": total}},
            )


class ModelEngine:
    def __init__(self, *, request_timeout_s: float = 900.0, progress_log_every: int = 1):
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mflux-inference")
        self._request_timeout_s = request_timeout_s
        self._progress_log_every = progress_log_every
        self._snapshot = ProgressSnapshot()
        self._callback = _ProgressCallback(progress_log_every, self._snapshot)
        self._model: Any | None = None
        self._loaded: tuple[str, str] | None = None  # (clé du modèle, kind)

    # ── État ───────────────────────────────────────────────────────────────

    @property
    def loaded_model(self) -> str | None:
        return None if self._loaded is None else f"{self._loaded[0]}:{self._loaded[1]}"

    def progress(self) -> dict[str, Any]:
        """Instantané de progression. Sans verrou, appelable depuis la boucle.

        Prendre `self._lock` ici serait un blocage garanti : il est détenu
        pendant toute la génération, précisément quand on veut la suivre.
        """
        return {
            **self._snapshot.as_dict(),
            "loaded_model": self.loaded_model,
            "memory": self.memory_stats(),
        }

    def request_cancel(self) -> bool:
        """Demande l'arrêt de la génération en cours. `False` si rien ne tourne.

        Le drapeau est lu par `_ProgressCallback.call_in_loop`, donc l'arrêt
        prend effet à la prochaine étape de débruitage — pas instantanément.
        """
        if self._snapshot.state != "generating":
            return False
        self._callback.cancel_requested = True
        logger.info(
            "Annulation demandée pour %s",
            self._snapshot.model,
            extra={"event": "generation_cancel_requested", "fields": {"model": self._snapshot.model}},
        )
        return True

    def memory_stats(self) -> dict[str, float]:
        try:
            import mlx.core as mx
        except ImportError:  # pragma: no cover - mlx est une dépendance dure
            return {}
        return {
            "active_gb": round(mx.get_active_memory() / 1e9, 2),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "cache_gb": round(mx.get_cache_memory() / 1e9, 2),
        }

    # ── Cycle de vie ───────────────────────────────────────────────────────

    async def generate(self, job: GenerationJob) -> bytes:
        """Génère une image en PNG. Sérialisé : un appel à la fois."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(self._executor, self._generate_sync, job)
            except GenerationTimeout as exc:
                raise exc
            except Exception as exc:
                raise translate_mflux_exception(exc) from exc

    async def unload(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._unload_sync)

    def shutdown(self) -> None:
        """Arrête le moteur en bornant l'attente à une étape de débruitage.

        Deux pièges, dans cet ordre précis.

        D'abord, demander l'annulation **avant** de joindre le worker : sans ça
        `wait=True` attend la fin de la génération en cours, soit jusqu'à
        `request_timeout_s` (2400 s dans la config livrée). `timeout_graceful_shutdown`
        d'uvicorn ne couvre pas ce cas : il ne borne que l'attente des connexions
        HTTP, et le lifespan de fermeture — donc cet appel — tourne ensuite.

        Ensuite, ne libérer les sous-modules qu'**après** l'arrêt du worker :
        `_unload_sync` tourne sur le thread appelant, et poser `transformer = None`
        pendant qu'une génération l'utilise est une course.
        """
        self._callback.cancel_requested = True
        self._executor.shutdown(wait=True)
        self._unload_sync()
        self._snapshot.reset()

    # ── Implémentation (thread worker) ─────────────────────────────────────

    def _generate_sync(self, job: GenerationJob) -> bytes:
        try:
            model = self._ensure_model(job.spec, job.kind)
            label = f"{job.spec.key} seed={job.seed} {job.width}x{job.height}"
            deadline = time.monotonic() + self._request_timeout_s if self._request_timeout_s else None
            self._callback.arm(label, deadline)

            self._snapshot.state = "generating"
            self._snapshot.model = job.spec.key
            self._snapshot.kind = job.kind
            self._snapshot.seed = job.seed
            self._snapshot.step = 0
            self._snapshot.total = job.steps
            self._snapshot.started_at = time.monotonic()

            logger.info(
                "▶ %s — %d étapes",
                label,
                job.steps,
                extra={
                    "event": "generation_start",
                    "fields": {
                        "model": job.spec.key,
                        "kind": job.kind,
                        "seed": job.seed,
                        "steps": job.steps,
                        "width": job.width,
                        "height": job.height,
                    },
                },
            )
            started = time.monotonic()
            with capture_stdout():
                generated = model.generate_image(**self._generate_kwargs(job))
            elapsed = time.monotonic() - started
            logger.info(
                "✓ %s — %.1f s",
                label,
                elapsed,
                extra={
                    "event": "generation_done",
                    "fields": {"model": job.spec.key, "seed": job.seed, "elapsed_s": round(elapsed, 1)},
                },
            )

            self._trim_prompt_cache(model)
            return _to_png_bytes(generated)
        except Exception as exc:
            if self._callback.timed_out:
                raise GenerationTimeout(self._request_timeout_s) from exc
            raise translate_mflux_exception(exc) from exc
        finally:
            # Quel que soit le sort du job — succès, annulation, timeout, crash —
            # le moteur redevient disponible et les consommateurs SSE doivent le
            # voir. Le modèle chargé, lui, reste chaud.
            self._snapshot.reset()

    def _generate_kwargs(self, job: GenerationJob) -> dict[str, Any]:
        spec = job.spec
        kwargs: dict[str, Any] = {
            "seed": job.seed,
            "prompt": job.prompt,
            "num_inference_steps": job.steps,
            "width": job.width,
            "height": job.height,
            "scheduler": spec.scheduler,
        }

        guidance = job.guidance if job.guidance is not None else spec.default_guidance
        if guidance is not None:
            kwargs["guidance"] = guidance

        if spec.supports_negative_prompt and job.negative_prompt is not None:
            kwargs["negative_prompt"] = job.negative_prompt

        if job.kind == "edit":
            # Édition instructionnelle : les images de référence sont des
            # tokens de conditionnement, pas un latent de départ bruité.
            kwargs["image_paths"] = [str(job.image_path)] if job.image_path else []
        elif job.image_path is not None:
            kwargs["image_path"] = str(job.image_path)
            kwargs["image_strength"] = job.image_strength

        return kwargs

    def _ensure_model(self, spec: ModelSpec, kind: str) -> Any:
        target = (spec.key, kind)
        if self._loaded == target and self._model is not None:
            return self._model

        if self._model is not None:
            logger.info(
                "Déchargement de %s",
                self.loaded_model,
                extra={"event": "model_unload", "fields": {"model": self.loaded_model}},
            )
            self._unload_sync()

        # « loading » plutôt que « generating » : sur flux2-dev c'est une minute
        # ou plus, et l'UI doit pouvoir le dire au lieu d'afficher 0/50.
        self._snapshot.state = "loading"
        self._snapshot.model = spec.key
        self._snapshot.kind = kind
        self._snapshot.started_at = time.monotonic()

        logger.info(
            "Chargement de %s (%s) — %s",
            spec.key,
            kind,
            spec.repo,
            extra={"event": "model_loading", "fields": {"model": spec.key, "kind": kind, "repo": spec.repo}},
        )
        started = time.monotonic()
        with capture_stdout():
            model = load_model(spec, kind=kind)
        # Un unique callback, enregistré au chargement et jamais retiré :
        # CallbackRegistry ne sait pas désenregistrer.
        model.callbacks.register(self._callback)

        self._model = model
        self._loaded = target
        memory = self.memory_stats()
        logger.info(
            "Modèle %s prêt en %.1f s — mémoire %s",
            spec.key,
            time.monotonic() - started,
            memory,
            extra={
                "event": "model_ready",
                "fields": {
                    "model": spec.key,
                    "kind": kind,
                    "load_s": round(time.monotonic() - started, 1),
                    **memory,
                },
            },
        )
        return model

    def _unload_sync(self) -> None:
        model = self._model
        self._model = None
        self._loaded = None
        if model is None:
            return

        try:
            from mflux.callbacks.callback_registry import CallbackRegistry

            model.callbacks = CallbackRegistry()
        except Exception:  # pragma: no cover - défensif
            logger.debug("Impossible de réinitialiser les callbacks", exc_info=True)

        for attr in _UNLOADABLE_ATTRS:
            if hasattr(model, attr):
                try:
                    setattr(model, attr, None)
                except Exception:  # pragma: no cover - défensif
                    logger.debug("Impossible de libérer %s", attr, exc_info=True)

        prompt_cache = getattr(model, "prompt_cache", None)
        if isinstance(prompt_cache, dict):
            prompt_cache.clear()

        del model
        gc.collect()
        _clear_mlx_cache()

    def _trim_prompt_cache(self, model: Any) -> None:
        prompt_cache = getattr(model, "prompt_cache", None)
        if isinstance(prompt_cache, dict) and len(prompt_cache) > _PROMPT_CACHE_MAX_ENTRIES:
            prompt_cache.clear()
            gc.collect()
            _clear_mlx_cache()
            logger.debug("prompt_cache purgé")


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:  # pragma: no cover - mlx est une dépendance dure
        pass


def _to_png_bytes(generated: Any) -> bytes:
    """Extrait le PNG d'un `GeneratedImage` sans passer par le disque.

    `ZImage.generate_image` est annoté `-> Image.Image` mais renvoie bien un
    `GeneratedImage` (z_image.py:59 vs :137) — d'où le `getattr`.
    """
    image = getattr(generated, "image", generated)
    if image is None:
        raise APIError(
            "mflux n'a produit aucune image.", status_code=500, error_type="server_error", code="empty_result"
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
