"""Logs JSON Lines et robustesse des chemins d'écriture.

Ces tests couvrent ce qui casse une fois le serveur lancé depuis un `.app`
plutôt que depuis un terminal : chemins relatifs au dossier courant, dossiers
parents inexistants, et format de sortie exploitable par un superviseur.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mflux_server.logs import JsonFormatter, setup_logging
from mflux_server.settings import ServerSettings


def _record(message: str = "coucou", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="mflux_server",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_une_ligne_est_un_json_valide():
    line = JsonFormatter().format(_record("étape 3/9"))
    payload = json.loads(line)
    assert "\n" not in line
    assert payload["message"] == "étape 3/9"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "mflux_server"
    assert "ts" in payload


def test_les_accents_ne_sont_pas_echappes():
    # `ensure_ascii=False` : les logs sont en français, autant qu'ils restent
    # lisibles dans un terminal comme dans l'app.
    assert "étape" in JsonFormatter().format(_record("étape 3/9"))


def test_les_champs_structures_sont_exposes():
    line = JsonFormatter().format(
        _record("étape 3/9", event="generation_step", fields={"step": 3, "total": 9})
    )
    payload = json.loads(line)
    assert payload["event"] == "generation_step"
    assert payload["fields"] == {"step": 3, "total": 9}


def test_sans_extra_pas_de_cles_parasites():
    payload = json.loads(JsonFormatter().format(_record()))
    assert "event" not in payload
    assert "fields" not in payload


def test_les_objets_non_serialisables_ne_font_pas_planter():
    # `memory_stats()` renvoie des floats, mais un `extra` mal fichu ne doit
    # jamais casser une génération.
    payload = json.loads(JsonFormatter().format(_record(event="x", fields={"path": Path("/tmp/a")})))
    assert payload["fields"]["path"] == "/tmp/a"


def test_le_mode_json_sort_sur_stdout_et_le_mode_texte_sur_stderr():
    """Séparation des canaux, et ce n'est pas cosmétique.

    mflux affiche sa barre de débruitage avec tqdm sur stderr, en fragments
    terminés par `\\r` sans retour à la ligne. Nos objets JSON s'y colleraient sur
    le même segment (`\\r 0%| | 0/40 [...]{"ts": ...}`) et un consommateur
    raisonnable les manquerait tous. Vérifié en vrai : 9 lignes sur stdout, 0
    non-JSON, tqdm resté sur stderr.
    """
    import sys

    setup_logging("INFO", None, json_lines=True)
    streams = [h.stream for h in logging.getLogger("mflux_server").handlers]
    assert streams == [sys.stdout]

    setup_logging("INFO", None, json_lines=False)
    streams = [h.stream for h in logging.getLogger("mflux_server").handlers]
    assert streams == [sys.stderr]


def test_le_dossier_du_fichier_de_log_est_cree(tmp_path):
    """`FileHandler` ne crée pas les dossiers parents et n'expanse pas `~`."""
    log_file = tmp_path / "logs" / "profond" / "mflux.log"
    setup_logging("INFO", log_file, json_lines=True)
    logging.getLogger("mflux_server").info("bonjour", extra={"event": "test"})

    for handler in logging.getLogger("mflux_server").handlers:
        handler.flush()
    assert log_file.exists()
    payload = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["event"] == "test"

    # Ne pas laisser un handler ouvert sur tmp_path pour les tests suivants.
    setup_logging("INFO", None)


# ── Chemins absolus ────────────────────────────────────────────────────────


def test_les_chemins_decriture_sont_rendus_absolus():
    """Sinon, lancé depuis un `.app`, le CWD est `/` et le démarrage échoue."""
    settings = ServerSettings.model_validate({"image_store": "images", "log_file": "mflux.log"})
    assert Path(settings.image_store).is_absolute()
    assert Path(settings.log_file).is_absolute()


def test_le_tilde_est_expanse():
    settings = ServerSettings.model_validate({"log_file": "~/mflux-probe.log"})
    assert "~" not in settings.log_file
    assert settings.log_file.startswith(str(Path.home()))


def test_desactiver_le_fichier_de_log_reste_possible():
    # La chaîne vide est le moyen documenté de couper le fichier via
    # MFLUX_SERVER_LOG_FILE ; elle ne doit pas devenir un chemin.
    assert ServerSettings.model_validate({"log_file": ""}).log_file == ""
    assert ServerSettings.model_validate({"log_file": None}).log_file is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("log_json", True), ("shutdown_grace_s", 5.0)],
)
def test_les_nouveaux_champs_existent(field, value):
    settings = ServerSettings.model_validate({field: value})
    assert getattr(settings, field) == value


def test_shutdown_grace_doit_etre_positif():
    with pytest.raises(ValueError):
        ServerSettings.model_validate({"shutdown_grace_s": 0})
