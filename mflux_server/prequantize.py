"""Pré-quantification de FLUX.2-dev vers un artefact MLX local.

Le repo amont est en bf16 : transformer 64,5 Go + encodeur texte 45,8 Go + VAE.
Soit ~111 Go de poids résidents, impossible sur 96 Go de mémoire unifiée. En
8 bits on tombe à ~58 Go, largement tenable — mais quantifier à la volée exige
justement de tenir le bf16 en mémoire d'abord. D'où cette conversion, à faire
une fois.

Deux précautions, chacune bornant une ressource :

* **un composant à la fois.** `WeightApplier.apply_and_quantize` charge tous les
  composants en bf16 avant de quantifier. On passe donc par les API « single »
  de mflux, en libérant entre chaque.
* **quantification bloc par bloc.** Un `nn.quantize` sur tout le transformer
  fait cohabiter 64,5 Go de bf16 et 34 Go de 8 bits, soit ~96 Go. En traitant
  chaque bloc puis en évaluant, le pic retombe autour de 66 Go.

L'ordre par défaut (transformer, puis encodeur, puis VAE) permet de purger le
cache HF entre deux composants : le pic disque passe de ~169 Go à ~97 Go.

Le rechargement ne demande aucun code : `WeightLoader._load_component` essaie
`_try_load_mflux_format` en premier, lit le `quantization_level` écrit ici dans
les métadonnées safetensors, et `WeightApplier` quantifie la structure avant de
poser les poids.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path

from mflux_server.logs import SERVER_LOGGER, setup_logging
from mflux_server.settings import ENV_PREFIX

logger = logging.getLogger(f"{SERVER_LOGGER}.prequantize")

#: Ordre imposé : le plus gros d'abord, pour pouvoir purger le cache HF entre
#: les étapes et borner le pic disque.
COMPONENT_ORDER = ("transformer", "text_encoder", "vae")


def _build_module(name: str, model_config):
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

    from mflux_server.flux2_dev.mistral3 import Mistral3TextEncoder

    if name == "transformer":
        return Flux2Transformer(**model_config.transformer_overrides)
    if name == "text_encoder":
        return Mistral3TextEncoder(**model_config.text_encoder_overrides)
    if name == "vae":
        return Flux2VAE()
    raise ValueError(f"Composant inconnu : {name!r}")


def _quantization_units(module) -> list:
    """Sous-modules à quantifier séparément pour borner le pic mémoire.

    `transformer_blocks` / `single_transformer_blocks` pour le transformer,
    `layers` pour l'encodeur texte. Le VAE n'en a pas : il est assez petit pour
    la passe globale.
    """
    units: list = []
    for attr in ("transformer_blocks", "single_transformer_blocks", "layers"):
        units.extend(getattr(module, attr, None) or [])
    return units


def _quantize_incrementally(module, *, bits: int, predicate) -> None:
    import mlx.core as mx
    from mlx import nn

    units = _quantization_units(module)
    for index, unit in enumerate(units, start=1):
        nn.quantize(unit, class_predicate=predicate, bits=bits)
        mx.eval(unit.parameters())
        mx.clear_cache()
        if index % 8 == 0 or index == len(units):
            logger.info(
                "  quantification %d/%d blocs — %s",
                index,
                len(units),
                _memory(),
                extra={
                    "event": "prequantize_progress",
                    "fields": {"block": index, "blocks": len(units)},
                },
            )

    # Passe finale pour les couches de tête (embeddings, projections, normes
    # modulées). Sans effet sur ce qui est déjà quantifié : les modules
    # `Quantized*` de MLX n'exposent pas `to_quantized`, donc le prédicat de
    # mflux les ignore.
    nn.quantize(module, class_predicate=predicate, bits=bits)
    mx.eval(module.parameters())
    mx.clear_cache()


def _memory() -> str:
    import mlx.core as mx

    return f"mlx actif {mx.get_active_memory() / 1e9:.1f} Go, pic {mx.get_peak_memory() / 1e9:.1f} Go"


def _directory_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def convert_component(name: str, *, repo: str, dest: Path, bits: int) -> None:
    import mlx.core as mx
    from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.saving.model_saver import ModelSaver

    from mflux_server.flux2_dev import flux2_dev_model_config, single_component_definition
    from mflux_server.flux2_dev.weights import Flux2DevWeightDefinition

    definition = single_component_definition(name)
    component = definition.get_components()[0]
    model_config = flux2_dev_model_config()

    logger.info("── %s ──────────────────────────────────────", name)
    logger.info(
        "Téléchargement / lecture de %s/%s",
        repo,
        component.hf_subdir,
        extra={
            "event": "prequantize_component_start",
            "fields": {"component": name, "repo": repo, "bits": bits},
        },
    )
    # `load` plutôt que `load_single` : il passe par `PathResolution`, donc
    # `--repo` accepte aussi un dossier local, et les patterns viennent de la
    # définition mono-composant — seul le sous-dossier voulu est téléchargé.
    weights = WeightLoader.load(weight_definition=definition, model_path=repo)

    resolved_bits, warning = QuantizationResolution.resolve(
        stored=weights.meta_data.quantization_level,
        requested=bits,
    )
    if warning:
        logger.warning(warning)
    if resolved_bits is None:
        raise ValueError(f"Aucune quantification résolue pour {name} (bits={bits!r})")

    module = _build_module(name, model_config)
    module.update(weights.components[component.name], strict=False)
    # On lâche la référence du loader avant de quantifier : sans ça les tableaux
    # bf16 restent vivants pendant toute la conversion.
    weights.components.clear()
    del weights
    gc.collect()
    mx.eval(module.parameters())
    logger.info("Poids bf16 posés — %s", _memory())

    _quantize_incrementally(
        module,
        bits=resolved_bits,
        predicate=Flux2DevWeightDefinition.quantization_predicate,
    )
    logger.info("Quantifié en %d bits — %s", resolved_bits, _memory())

    shim = _ComponentShim(name, module)
    if definition.get_tokenizers():
        shim.tokenizers = TokenizerLoader.load_all(
            definitions=definition.get_tokenizers(),
            model_path=repo,
        )

    ModelSaver.save_model(
        model=shim,
        bits=resolved_bits,
        base_path=str(dest),
        weight_definition=definition,
    )

    written = _directory_size_gb(dest / component.hf_subdir)
    logger.info(
        "Écrit dans %s (%.1f Go)",
        dest / component.hf_subdir,
        written,
        extra={
            "event": "prequantize_component_done",
            "fields": {"component": name, "bits": resolved_bits, "written_gb": round(written, 2)},
        },
    )

    del shim, module
    gc.collect()
    mx.clear_cache()
    logger.info("Libéré — %s", _memory())


class _ComponentShim:
    """Porteur d'attribut minimal pour `ModelSaver.save_model`.

    Le saver lit `getattr(model, component.name)` et, si présent,
    `model.tokenizers` — pas besoin du modèle complet.
    """

    def __init__(self, name: str, module) -> None:
        setattr(self, name, module)


def main() -> int:
    from mflux_server.flux2_dev import DEFAULT_MODEL_PATH, REPO
    from mflux_server.registry import QUANTIZE_CHOICES

    parser = argparse.ArgumentParser(
        prog="mflux-server-prequantize",
        description="Convertit black-forest-labs/FLUX.2-dev en artefact MLX quantifié, "
        "composant par composant pour tenir dans la mémoire unifiée.",
    )
    parser.add_argument("--dest", default=DEFAULT_MODEL_PATH, help="dossier de sortie")
    parser.add_argument("--repo", default=REPO, help="repo source")
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=QUANTIZE_CHOICES,
        help="bits de quantification",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(COMPONENT_ORDER),
        choices=list(COMPONENT_ORDER),
        help="composants à convertir, dans l'ordre donné (défaut : tous, du plus gros au plus petit)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        default=os.environ.get(f"{ENV_PREFIX}LOG_JSON", "").lower() in {"1", "true", "yes"},
        help="une ligne = un objet JSON, pour être suivi par un superviseur",
    )
    args = parser.parse_args()

    # Même configuration que le serveur, pour que l'app de bureau suive la
    # conversion exactement comme elle suit une génération.
    setup_logging(level="INFO", log_file=None, json_lines=args.json_logs)
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    for name in args.components:
        convert_component(name, repo=args.repo, dest=dest, bits=args.bits)
        logger.info(
            "Le bf16 de '%s' n'est plus utile : purgez-le du cache HF avant le composant suivant "
            "(`hf cache delete`, ou supprimez %s/ dans le snapshot) pour borner le pic disque.\n",
            name,
            name,
        )

    logger.info("Terminé — %s : %.1f Go", dest, _directory_size_gb(dest))
    logger.info(
        "Si ce chemin diffère du défaut, renseignez-le dans server-config.json (models.flux2-dev.model_path)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
