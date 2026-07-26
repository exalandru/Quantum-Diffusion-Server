"""Conformité de la couche HTTP à ce qu'attend un client OpenAI."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from mflux_server.app import create_app
from mflux_server.settings import Settings
from tests.conftest import tiny_png


def generate(client: TestClient, **body):
    return client.post("/v1/images/generations", json={"prompt": "un renard", **body})


# ── /v1/models ─────────────────────────────────────────────────────────────


def test_liste_des_modeles_conforme(client):
    payload = client.get("/v1/models").json()
    assert payload["object"] == "list"
    assert {entry["id"] for entry in payload["data"]} == {
        "flux2-klein",
        "qwen-image",
        "z-image",
        "z-image-turbo",
    }
    for entry in payload["data"]:
        assert entry["object"] == "model"
        assert isinstance(entry["created"], int)
        assert entry["owned_by"] == "mflux"


def test_recuperation_dun_modele(client):
    payload = client.get("/v1/models/z-image-turbo").json()
    assert payload["id"] == "z-image-turbo"
    assert payload["mflux"]["default_size"] == "1280x720"
    assert payload["mflux"]["supports_guidance"] is False


def test_modele_inconnu_renvoie_une_erreur_openai(client):
    response = client.get("/v1/models/sdxl")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "model"
    assert error["code"] == "model_not_found"
    assert "sdxl" in error["message"]


# ── response_format ────────────────────────────────────────────────────────


def test_url_est_le_defaut_et_est_servie(client):
    """Le bug n°1 du prototype : `url` était silencieusement ignoré."""
    payload = generate(client).json()
    url = payload["data"][0]["url"]
    assert url.endswith(".png")
    assert "b64_json" not in payload["data"][0]

    served = client.get(url)
    assert served.status_code == 200
    assert served.content == tiny_png()


def test_b64_json(client):
    payload = generate(client, response_format="b64_json").json()
    assert base64.b64decode(payload["data"][0]["b64_json"]) == tiny_png()


def test_raw_renvoie_le_png(client):
    response = generate(client, response_format="raw")
    assert response.headers["content-type"] == "image/png"
    assert response.content == tiny_png()


def test_raw_avec_n_superieur_a_1_est_refuse(client):
    # Le prototype retombait silencieusement sur du b64.
    response = generate(client, response_format="raw", n=2)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "response_format"


def test_response_format_inconnu(client):
    response = generate(client, response_format="jpeg")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_response_format"


# ── Paramètres ─────────────────────────────────────────────────────────────


def test_taille_par_defaut_du_modele(client):
    payload = generate(client, size="auto").json()
    assert payload["mflux"]["size"] == "1920x1072"


def test_taille_arrondie_au_multiple_de_16_et_annoncee(client):
    payload = generate(client, size="1920x1080").json()
    assert payload["mflux"]["size"] == "1920x1072"


def test_taille_invalide(client):
    response = generate(client, size="grand")
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "size"


def test_parametres_openai_non_supportes_sont_ignores(client):
    response = generate(client, quality="hd", style="vivid", user="corin", background="opaque")
    assert response.status_code == 200


def test_n_borne_par_la_config(client, engine):
    assert generate(client, n=4).status_code == 200
    assert len(engine.jobs) == 4

    response = generate(client, n=5)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "n_too_large"


def test_seed_incremente_par_image(client, engine):
    generate(client, n=3, seed=100)
    assert [job.seed for job in engine.jobs] == [100, 101, 102]


def test_seed_absent_est_tire_au_hasard_mais_reste_consecutif(client, engine):
    generate(client, n=2)
    first, second = (job.seed for job in engine.jobs)
    assert second == first + 1


def test_steps_par_defaut_du_modele(client, engine):
    generate(client, model="flux2-klein")
    assert engine.jobs[0].steps == 4  # et non les 20 du prototype
    generate(client, model="z-image-turbo")
    assert engine.jobs[1].steps == 9


# ── Capacités des modèles ──────────────────────────────────────────────────


def test_negative_prompt_refuse_sur_flux2_klein(client):
    """Le bug n°2 : un 500 systématique sur le modèle par défaut."""
    response = generate(client, negative_prompt="flou")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "negative_prompt"
    assert error["code"] == "unsupported_parameter"


def test_negative_prompt_accepte_sur_qwen(client, engine):
    assert generate(client, model="qwen-image", negative_prompt="flou").status_code == 200
    assert engine.jobs[0].negative_prompt == "flou"


def test_guidance_refusee_sur_un_modele_distille(client):
    response = generate(client, guidance=3.5)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "guidance"


def test_guidance_acceptee_sur_z_image(client, engine):
    assert generate(client, model="z-image", guidance=5.0).status_code == 200
    assert engine.jobs[0].guidance == 5.0


# ── /v1/images/edits ───────────────────────────────────────────────────────


def edit(client: TestClient, **data):
    return client.post(
        "/v1/images/edits",
        data={"prompt": "ajoute un chapeau", **data},
        files={"image": ("source.png", tiny_png("blue"), "image/png")},
    )


def test_edits_utilise_la_variante_edition_quand_elle_existe(client, engine):
    assert edit(client).status_code == 200
    job = engine.jobs[0]
    assert job.kind == "edit"
    assert job.image_strength is None


def test_edits_bascule_en_img2img_si_strength_est_fourni(client, engine):
    assert edit(client, strength=0.7).status_code == 200
    job = engine.jobs[0]
    assert job.kind == "txt2img"
    assert job.image_strength == 0.7


def test_edits_sans_variante_edition_fait_de_limg2img(client, engine):
    assert edit(client, model="z-image").status_code == 200
    job = engine.jobs[0]
    assert job.kind == "txt2img"
    assert job.image_strength == 0.4


def test_mask_est_refuse_explicitement(client):
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "efface le fond"},
        files={
            "image": ("source.png", tiny_png(), "image/png"),
            "mask": ("mask.png", tiny_png("black"), "image/png"),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "mask"


def test_image_vide_refusee(client):
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "x"},
        files={"image": ("vide.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_upload_trop_gros_refuse(tmp_path, engine):
    settings = Settings.model_validate(
        {
            "server": {
                "image_store": str(tmp_path / "images"),
                "log_file": None,
                "max_upload_mb": 0.001,  # ~1 Ko
            }
        }
    )
    with TestClient(create_app(settings, engine)) as client:
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "x"},
            files={"image": ("gros.png", b"0" * 5000, "image/png")},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


# ── Transport : CORS, auth, health ─────────────────────────────────────────


def test_preflight_cors_autorise(client):
    response = client.options(
        "/v1/images/generations",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def secured_app(tmp_path, engine, api_key: str):
    settings = Settings.model_validate(
        {
            "server": {
                "image_store": str(tmp_path / "images"),
                "log_file": None,
                "api_key": api_key,
            }
        }
    )
    return create_app(settings, engine)


@pytest.fixture
def secured_client(tmp_path, engine):
    with TestClient(secured_app(tmp_path, engine, "cle-secrete")) as test_client:
        yield test_client


def test_auth_requise_quand_une_cle_est_configuree(secured_client):
    assert secured_client.get("/v1/models").status_code == 401
    assert secured_client.get("/v1/models", headers={"Authorization": "Bearer mauvaise"}).status_code == 401
    assert secured_client.get("/v1/models", headers={"Authorization": "Basic cle-secrete"}).status_code == 401
    assert (
        secured_client.get("/v1/models", headers={"Authorization": "Bearer cle-secrete"}).status_code == 200
    )


def test_cle_non_ascii_refuse_sans_planter(tmp_path, engine):
    """Une telle clé est inatteignable (les en-têtes HTTP sont latin-1),
    mais elle ne doit pas faire remonter un 500 depuis compare_digest."""
    with TestClient(secured_app(tmp_path, engine, "clé-secrète")) as test_client:
        assert test_client.get("/v1/models", headers={"Authorization": "Bearer x"}).status_code == 401


def test_health_reste_public(secured_client):
    payload = secured_client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["default_model"] == "flux2-klein"
    assert payload["loaded_model"] is None


def test_health_expose_le_modele_chaud(client):
    generate(client, model="z-image")
    assert client.get("/health").json()["loaded_model"] == "z-image:txt2img"


def test_capabilities(client):
    payload = client.get("/v1/capabilities").json()
    assert payload["max_n"] == 4
    assert payload["models"]["flux2-klein"]["supports_edit"] is True
    assert payload["models"]["z-image"]["supports_edit"] is False


def test_prompt_vide_renvoie_une_erreur_formatee(client):
    response = client.post("/v1/images/generations", json={"prompt": ""})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["param"] == "prompt"


def test_shutdown_libere_le_moteur(settings, engine):
    with TestClient(create_app(settings, engine)):
        pass
    assert engine.shutdown_called is True
