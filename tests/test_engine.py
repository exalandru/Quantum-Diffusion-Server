"""Tests du moteur : cache, sérialisation, déchargement, callbacks.

Aucun poids n'est chargé — `load_model` est remplacé par un double. Les
objets mflux réellement utilisés (`CallbackRegistry`,
`StopImageGenerationException`) le sont pour de vrai : c'est justement leur
comportement qu'on veut verrouiller.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image

from mflux_server import engine as engine_module
from mflux_server.engine import GenerationJob, ModelEngine
from mflux_server.errors import APIError
from mflux_server.registry import BASE_SPECS_BY_KEY


class FakeGenerated:
    def __init__(self):
        self.image = Image.new("RGB", (2, 2), "green")


class FakeModel:
    """Imite juste ce que le moteur touche sur un modèle mflux."""

    def __init__(self, key: str, kind: str):
        from mflux.callbacks.callback_registry import CallbackRegistry

        self.key = key
        self.kind = kind
        self.callbacks = CallbackRegistry()
        self.transformer = object()
        self.text_encoder = object()
        self.vae = object()
        self.prompt_cache: dict[str, object] = {}
        self.calls: list[dict] = []
        self.delay = 0.0

    def generate_image(self, **kwargs):
        import time

        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        # Le vrai modèle notifie ses callbacks à chaque étape.
        for callback in self.callbacks.in_loop_callbacks():
            callback.call_in_loop(
                t=0,
                seed=kwargs["seed"],
                prompt=kwargs["prompt"],
                latents=None,
                config=_FakeConfig(kwargs["num_inference_steps"]),
                time_steps=None,
            )
        return FakeGenerated()


class _FakeConfig:
    def __init__(self, steps: int):
        self.num_inference_steps = steps


@pytest.fixture
def loaded(monkeypatch):
    created: list[FakeModel] = []

    def fake_load_model(spec, *, kind="txt2img"):
        model = FakeModel(spec.key, kind)
        created.append(model)
        return model

    monkeypatch.setattr(engine_module, "load_model", fake_load_model)
    return created


def job(key: str = "flux2-klein", kind: str = "txt2img", **kwargs) -> GenerationJob:
    spec = BASE_SPECS_BY_KEY[key]
    defaults = dict(
        prompt="un renard",
        width=1024,
        height=1024,
        steps=spec.default_steps,
        seed=42,
    )
    defaults.update(kwargs)
    return GenerationJob(spec=spec, kind=kind, **defaults)


def test_le_modele_reste_chaud_entre_deux_generations(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())
        await eng.generate(job(seed=43))
        return eng

    eng = asyncio.run(scenario())
    assert len(loaded) == 1, "les poids ont été rechargés"
    assert eng.loaded_model == "flux2-klein:txt2img"
    eng.shutdown()


def test_changer_de_modele_decharge_le_precedent(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("flux2-klein"))
        first = loaded[0]
        await eng.generate(job("z-image-turbo"))
        return eng, first

    eng, first = asyncio.run(scenario())
    assert len(loaded) == 2
    assert eng.loaded_model == "z-image-turbo:txt2img"
    # Les sous-modules du premier modèle ont bien été libérés.
    assert first.transformer is None
    assert first.text_encoder is None
    assert first.vae is None
    eng.shutdown()


def test_variante_edition_est_un_chargement_distinct(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("flux2-klein", kind="txt2img"))
        await eng.generate(job("flux2-klein", kind="edit"))
        return eng

    eng = asyncio.run(scenario())
    assert [model.kind for model in loaded] == ["txt2img", "edit"]
    eng.shutdown()


def test_le_callback_nest_enregistre_quune_fois(loaded):
    """`CallbackRegistry` n'a pas d'unregister : enregistrer par requête
    ferait grossir la liste indéfiniment."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        for seed in range(5):
            await eng.generate(job(seed=seed))
        return eng

    eng = asyncio.run(scenario())
    assert len(loaded[0].callbacks.in_loop_callbacks()) == 1
    eng.shutdown()


def test_les_generations_sont_serialisees(loaded):
    """Deux requêtes concurrentes ne doivent jamais se chevaucher : sur
    mémoire unifiée, deux modèles vivants saturent la machine."""
    overlaps = 0
    running = 0

    async def scenario():
        nonlocal overlaps, running
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job())  # charge le modèle
        original = loaded[0].generate_image

        def instrumented(**kwargs):
            nonlocal overlaps, running
            running += 1
            if running > 1:
                overlaps += 1
            try:
                return original(**kwargs)
            finally:
                running -= 1

        loaded[0].generate_image = instrumented
        loaded[0].delay = 0.02
        await asyncio.gather(*(eng.generate(job(seed=index)) for index in range(4)))
        return eng

    eng = asyncio.run(scenario())
    assert overlaps == 0
    eng.shutdown()


def test_le_prompt_cache_de_qwen_est_purge(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("qwen-image"))
        model = loaded[0]
        model.prompt_cache.update({f"prompt-{index}": object() for index in range(50)})
        await eng.generate(job("qwen-image", seed=7))
        return eng, model

    eng, model = asyncio.run(scenario())
    assert model.prompt_cache == {}
    eng.shutdown()


def test_le_timeout_interrompt_la_boucle_de_debruitage(loaded):
    """Seul le callback in-loop peut arrêter une génération : ni asyncio ni
    un thread ne savent annuler une opération MLX en cours."""

    async def scenario():
        eng = ModelEngine(request_timeout_s=0.05, progress_log_every=0)
        await eng.generate(job())  # charge le modèle, sous le délai
        loaded[0].delay = 0.2  # la prochaine étape arrivera après l'échéance
        with pytest.raises(APIError) as excinfo:
            await eng.generate(job(seed=1))
        return eng, excinfo.value

    eng, error = asyncio.run(scenario())
    assert error.status_code == 504
    assert error.code == "timeout"
    eng.shutdown()


def test_les_erreurs_mflux_sont_traduites(loaded, monkeypatch):
    from mflux.utils.exceptions import ModelConfigError

    def exploding_load(spec, *, kind="txt2img"):
        raise ModelConfigError("base model introuvable")

    monkeypatch.setattr(engine_module, "load_model", exploding_load)

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        with pytest.raises(APIError) as excinfo:
            await eng.generate(job())
        eng.shutdown()
        return excinfo.value

    error = asyncio.run(scenario())
    assert error.status_code == 400
    assert error.param == "model"


def test_le_png_est_produit_en_memoire(loaded):
    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        data = await eng.generate(job())
        eng.shutdown()
        return data

    data = asyncio.run(scenario())
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_arguments_passes_a_mflux(loaded):
    """Vérifie le câblage par famille : negative_prompt et guidance ne sont
    transmis que si le modèle les accepte."""

    async def scenario():
        eng = ModelEngine(progress_log_every=0)
        await eng.generate(job("flux2-klein", negative_prompt="flou", guidance=None))
        flux_kwargs = loaded[0].calls[0]
        await eng.generate(job("z-image", negative_prompt="flou", guidance=6.0))
        z_kwargs = loaded[1].calls[0]
        await eng.generate(job("flux2-klein", kind="edit", image_path="/tmp/in.png"))
        edit_kwargs = loaded[2].calls[0]
        await eng.generate(job("z-image", image_path="/tmp/in.png", image_strength=0.6))
        img2img_kwargs = loaded[3].calls[0]
        eng.shutdown()
        return flux_kwargs, z_kwargs, edit_kwargs, img2img_kwargs

    flux_kwargs, z_kwargs, edit_kwargs, img2img_kwargs = asyncio.run(scenario())

    # FLUX.2 Klein n'a pas de paramètre negative_prompt et sa guidance est figée.
    assert "negative_prompt" not in flux_kwargs
    assert flux_kwargs["guidance"] == 1.0
    assert flux_kwargs["scheduler"] == "flow_match_euler_discrete"

    assert z_kwargs["negative_prompt"] == "flou"
    assert z_kwargs["guidance"] == 6.0

    # Édition : liste d'images de conditionnement, pas d'image_strength.
    assert edit_kwargs["image_paths"] == ["/tmp/in.png"]
    assert "image_strength" not in edit_kwargs

    # img2img : latent de départ bruité.
    assert img2img_kwargs["image_path"] == "/tmp/in.png"
    assert img2img_kwargs["image_strength"] == 0.6
