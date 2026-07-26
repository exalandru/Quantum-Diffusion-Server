# mflux-server

Serveur local qui expose [mflux](https://github.com/filipstrand/mflux) — l'implémentation MLX de FLUX, Qwen-Image et Z-Image pour Apple Silicon — derrière une **API compatible OpenAI Images**. De quoi brancher n'importe quel frontend qui parle OpenAI (Misty Studio, Open WebUI, le SDK `openai`…) sur des modèles de diffusion qui tournent en local.

Le modèle est **chargé une fois et gardé en mémoire** entre les requêtes, au lieu d'être rechargé par un nouveau process à chaque image.

## Installation

```sh
uv sync
```

mflux est une dépendance du projet — pas besoin de `uv tool install mflux` à côté. Les poids déjà présents dans le cache HuggingFace sont réutilisés tels quels.

## Lancement

```sh
uv run mflux-server
```

Le serveur écoute sur `http://127.0.0.1:8765`. Docs interactives sur `/docs`.

```sh
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/v1/models

curl http://127.0.0.1:8765/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "un renard roux dans la neige", "size": "1024x1024"}'
```

Avec le SDK officiel :

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="peu-importe")
result = client.images.generate(prompt="un renard roux dans la neige", size="1024x1024")
print(result.data[0].url)
```

## Performances

Mesures réelles sur M3 Ultra / 103 Go, poids déjà en cache HuggingFace :

| scénario | subprocess (ancien) | serveur, modèle chaud |
|---|---|---|
| `flux2-klein`, 1024×1024, 4 étapes | 34,3 s | **18,5 s** |
| `z-image-turbo`, 1280×720, 9 étapes | — | 45 s au 1ᵉʳ appel, puis **32 s** |

Soit environ **1,8×** sur le modèle par défaut, et ce gain se répète à chaque image.

Il faut avoir en tête d'où il vient, parce que ce n'est pas ce qu'on croit : mflux charge ses poids en **lazy/mmap**, donc un modèle 9B est « prêt » en une demi-seconde et le vrai coût est payé pendant la première génération. Ce que le serveur économise, c'est le démarrage d'un process Python complet (import de torch, transformers, mlx) et la rematérialisation des poids — pas un chargement de plusieurs minutes. Sur un modèle où l'inférence domine (`z-image-turbo` à ~3,5 s l'étape), le gain relatif est donc plus faible.

Corollaire : la mémoire est le vrai facteur limitant. Faire tourner un autre process mflux à côté du serveur fait évincer ses pages et triple le temps de la génération suivante.

## Modèles

| clé | repo | taille déf. | steps déf. | guidance | negative_prompt | img2img | édition |
|---|---|---|---|---|---|---|---|
| `flux2-klein` *(défaut)* | `black-forest-labs/FLUX.2-klein-9B` | 1920×1072 | 4 | figée à 1.0 | ❌ | ✅ | ✅ |
| `qwen-image` | `mlx-community/Qwen-Image-2512-8bit` | 1920×1072 | 20 | 3.5 | ✅ | ✅ | sur option |
| `z-image` | `mlx-community/Z-Image-bf16` | 1920×1072 | 50 | 4.0 | ✅ | ✅ | ❌ |
| `z-image-turbo` | `mlx-community/Z-Image-Turbo-bf16` | 1280×720 | 9 | forcée à 0 | ✅ | ✅ | ❌ |

Détails utiles :

- **`flux2-klein` est distillé.** 4 étapes suffisent, la guidance est figée à 1.0 et `negative_prompt` n'existe pas pour ce modèle — mflux refuse explicitement le paramètre. Le serveur renvoie un 400 clair plutôt que de laisser planter.
- **`z-image` et `z-image-turbo` sont quantifiés à 8 bits au chargement.** Les repos sont stockés en bf16 ; la quantization est donc réelle, mais payée une seule fois puisque le modèle reste chaud.
- **`qwen-image` est déjà quantifié 8 bits** dans ses métadonnées safetensors : y ajouter `quantize` serait sans effet.
- **L'édition de `qwen-image` est désactivée par défaut** : elle utilise un repo distinct (`Qwen/Qwen-Image-Edit-2509`), soit plusieurs Go à télécharger au premier appel. Active-la avec `"enable_edit": true`. L'édition de `flux2-klein` partage les mêmes poids que la génération, elle est donc active d'office.
- **Les dimensions sont tronquées au multiple de 16 inférieur** — c'est une contrainte de mflux. `1920x1080` devient `1920x1072`. Le serveur applique l'arrondi lui-même et renvoie la taille effective dans le champ `mflux.size` de la réponse.

`GET /v1/capabilities` renvoie ce tableau au format JSON.

## Endpoints

| Route | Méthode | Note |
|---|---|---|
| `/v1/images/generations` | POST | texte → image |
| `/v1/images/edits` | POST | multipart ; édition instructionnelle ou img2img |
| `/v1/models` | GET | format OpenAI standard |
| `/v1/models/{id}` | GET | + capacités du modèle sous la clé `mflux` |
| `/v1/capabilities` | GET | extension : tout le catalogue |
| `/health` | GET | public même avec une clé d'API ; indique le modèle chaud et la mémoire MLX |
| `/images/{nom}.png` | GET | images servies en `response_format="url"` |

### Paramètres

Standards OpenAI : `prompt`, `model`, `n`, `size`, `response_format`. Les paramètres sans équivalent (`quality`, `style`, `user`, `background`, `output_format`) sont acceptés et ignorés, pas rejetés.

Extensions — champs additionnels que les SDK OpenAI ignorent :

| champ | effet |
|---|---|
| `steps` | nombre d'étapes de débruitage |
| `seed` | graine ; avec `n > 1`, incrémentée à chaque image |
| `guidance` | échelle CFG, refusée sur les modèles distillés |
| `negative_prompt` | refusé sur `flux2-klein` |
| `strength` | *(edits uniquement)* force l'img2img au lieu de l'édition |
| `response_format: "raw"` | renvoie les octets PNG directement, `n=1` seulement |

`size` accepte `"auto"` (taille par défaut du modèle) et le format `"LxH"`.

### `/v1/images/edits` : édition ou img2img ?

Deux mécaniques réellement différentes, et le serveur choisit selon ce que tu envoies :

- **`strength` fourni** → img2img : l'image est encodée puis bruitée, la boucle démarre à une étape intermédiaire. Le résultat est une variation de l'image source.
- **`strength` absent, modèle avec variante d'édition** → édition instructionnelle : la boucle part du bruit pur et l'image sert de tokens de conditionnement. C'est ce qu'il faut pour « ajoute un chapeau à cette personne ».
- **`strength` absent, pas de variante d'édition** → img2img avec `strength = 0.4`.

Le paramètre `mask` d'OpenAI est refusé en 400 : aucun modèle du catalogue ne fait d'inpainting.

## Configuration

`server.config` (JSON). Toute clé de la section `server` est surchargeable par `MFLUX_SERVER_<CLÉ>` en majuscules — `MFLUX_SERVER_PORT=9000`, `MFLUX_SERVER_API_KEY=…`, `MFLUX_SERVER_CORS_ORIGINS=https://a.example,https://b.example`. `MFLUX_SERVER_CONFIG` pointe vers un autre fichier de config.

| clé | défaut | rôle |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8765` | binding |
| `api_key` | `null` | si renseignée, `Authorization: Bearer` exigé. **Obligatoire dès que `host` n'est pas local** |
| `cors_origins` | `["*"]` | origines autorisées |
| `max_n` | `4` | borne le `n` d'OpenAI (générations séquentielles) |
| `request_timeout_s` | `900` | interrompt la boucle de débruitage au-delà |
| `image_store` / `image_ttl_s` | `images` / `3600` | dossier et durée de vie des images servies en `url` |
| `max_upload_mb` | `25` | taille maximale d'une image envoyée à `/v1/images/edits` |
| `default_response_format` | `url` | valeur retenue si le client n'en envoie pas. `url` est le défaut d'OpenAI : le changer casse les SDK |
| `log_level` / `log_file` | `INFO` / `mflux.log` | `log_file: null` désactive le fichier |
| `progress_log_every` | `1` | log de progression toutes les N étapes ; `0` pour couper |

Par modèle, sous `models` : `enabled`, `quantize` (3/4/5/6/8, ou 0 pour aucune), `default_size`, `default_steps`, `default_guidance`, `enable_edit`.

### Accès depuis le réseau local

```json
{"server": {"host": "0.0.0.0", "api_key": "une-clé-longue-et-aléatoire"}}
```

Le serveur refuse de démarrer avec un host non local sans clé d'API.

## Limites connues

- **Une génération à la fois.** C'est volontaire : sur mémoire unifiée, deux modèles vivants saturent la machine. Les requêtes concurrentes sont mises en file, pas rejetées.
- **`n > 1` est séquentiel.** Le modèle reste chaud, mais les images sortent l'une après l'autre.
- **Le timeout ne couvre pas le chargement des poids.** Il n'est vérifié qu'entre deux étapes de débruitage — c'est le seul point d'interruption qu'offre mflux. Un premier appel qui télécharge 30 Go peut donc dépasser `request_timeout_s`.
- **Pas de progression côté client** (ni SSE ni `partial_images`). Elle est journalisée côté serveur ; la barre `tqdm` de mflux reste visible dans le terminal.
- **Pas de LoRA, ControlNet, inpainting, upscale.** mflux les propose, ils ne sont pas exposés ici.

## Développement

```sh
uv run pytest        # aucun poids n'est chargé
uv run ruff check .
uv run ruff format .
```

Les tests couvrent le registre, la conformité OpenAI et le moteur (cache, sérialisation, déchargement) avec un modèle factice. **L'inférence réelle se vérifie à la main** :

1. `uv run mflux-server`
2. une première génération sur `flux2-klein` — chronométrer, chargement inclus ;
3. **la relancer à l'identique : elle doit être nettement plus rapide.** C'est le test qui valide le cache ;
4. `negative_prompt` sur `flux2-klein` → 400 explicite ;
5. deux requêtes simultanées → sérialisées, mémoire stable dans `/health` ;
6. changer de modèle → déchargement visible dans les logs ;
7. une dizaine de prompts différents sur `qwen-image` → la mémoire ne doit pas dériver ;
8. brancher le frontend et vérifier qu'aucune erreur CORS n'apparaît dans la console du navigateur.

### Notes d'intégration mflux

Points non évidents, vérifiés dans le code de mflux 0.18.0, qui expliquent certains choix :

- **`ModelConfig.from_name()` est évité.** Sa résolution perd `sigma_*` et `text_encoder_overrides` (`config_resolution.py:112-128`), ce qui changerait le scheduler de Qwen. On passe la factory canonique + `model_path`.
- **`CallbackManager.register_callbacks` n'est jamais appelé.** Il installe un `MemorySaver` qui détruit `text_encoder` dès la première génération quand `num_seeds <= 1` (`memory_saver.py:45-47`) : la deuxième requête planterait. Il installe aussi un `BatterySaver` qui lance `pmset` avant chaque génération.
- **Un seul callback est enregistré, au chargement.** `CallbackRegistry` n'a pas d'`unregister` (`callback_registry.py:12-27`).
- **Le `prompt_cache` de Qwen est purgé** au-delà de 16 entrées : il est indexé par prompt et n'a aucune borne.
- **Le déchargement est manuel** — mflux n'expose aucune méthode de teardown. On remet les sous-modules à `None`, puis `gc.collect()` + `mx.clear_cache()`.
