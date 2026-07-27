# mflux-server

Serveur local qui expose [mflux](https://github.com/filipstrand/mflux) — l'implémentation MLX de FLUX, Qwen-Image et Z-Image pour Apple Silicon — derrière une **API compatible OpenAI Images**. De quoi brancher n'importe quel frontend qui parle OpenAI (Misty Studio, Open WebUI, le SDK `openai`…) sur des modèles de diffusion qui tournent en local.

Le modèle est **chargé une fois et gardé en mémoire** entre les requêtes, au lieu d'être rechargé par un nouveau process à chaque image.

## Installation

```sh
uv sync
```

mflux est une dépendance du projet — pas besoin de `uv tool install mflux` à côté. Les poids déjà présents dans le cache HuggingFace sont réutilisés tels quels.

Les modèles `black-forest-labs/*` sont *gated* : il faut un token HuggingFace avec l'accès accordé (`hf auth login`). `flux2-dev` demande en plus une conversion préalable, voir [FLUX.2-dev](#flux2-dev--32b-en-8-bits).

## Lancement

Il y a deux façons de s'en servir : l'app de bureau, ou le serveur en ligne de commande.

### App de bureau

[`desktop/`](desktop/README.md) contient **Quantum Diffusion Server**, un panneau de contrôle macOS (Tauri + React) qui installe son propre Python, démarre et surveille le serveur, expose la configuration dans un formulaire et pilote la préparation des modèles. Rien à installer sur la machine : le `.app` fait 57 Mo et se charge du reste.

```sh
cd desktop && npm install && npm run app:build
```

### En ligne de commande

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
| `flux2-dev` | `black-forest-labs/FLUX.2-dev` | 1024×1024 | 50 | 4.0 | ❌ | ✅ | ❌ |
| `qwen-image` | `mlx-community/Qwen-Image-2512-8bit` | 1920×1072 | 20 | 3.5 | ✅ | ✅ | sur option |
| `z-image` | `mlx-community/Z-Image-bf16` | 1920×1072 | 50 | 4.0 | ✅ | ✅ | ❌ |
| `z-image-turbo` | `mlx-community/Z-Image-Turbo-bf16` | 1280×720 | 9 | forcée à 0 | ✅ | ✅ | ❌ |

Détails utiles :

- **`flux2-klein` est distillé.** 4 étapes suffisent, la guidance est figée à 1.0 et `negative_prompt` n'existe pas pour ce modèle — mflux refuse explicitement le paramètre. Le serveur renvoie un 400 clair plutôt que de laisser planter.
- **`flux2-dev` exige une conversion préalable** et n'est pas utilisable tel quel : 32 milliards de paramètres, repo *gated* (token HF nécessaire), et surtout du code que mflux 0.18.0 ne fournit pas. Voir [FLUX.2-dev](#flux2-dev--32b-en-8-bits). Compter ~113 Go de téléchargement unique, un artefact local de ~58 Go, et ~58 Go résidents pendant la génération.
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
| `/v1/progress` | GET | extension : progression en Server-Sent Events |
| `/v1/cancel` | POST | extension : interrompt la génération en cours |
| `/v1/unload` | POST | extension : libère les poids résidents sans redémarrer |
| `/health` | GET | public même avec une clé d'API ; indique le modèle chaud et la mémoire MLX |
| `/images/{nom}.png` | GET | images servies en `response_format="url"` |

### Suivre, annuler, libérer

`GET /v1/progress` diffuse un instantané à chaque changement d'état :

```sh
curl -N http://127.0.0.1:8765/v1/progress
# data: {"state":"loading","model":"z-image-turbo",...}
# data: {"state":"generating","model":"z-image-turbo","step":3,"total":9,"elapsed_s":4.2,...}
# data: {"state":"idle",...}
```

`state` vaut `idle`, `loading` ou `generating`. La distinction compte : le chargement d'un modèle prend de quelques secondes à plusieurs minutes selon sa taille, et n'a pas de progression par étapes. Un commentaire `: ping` est émis toutes les 15 s quand rien ne bouge, pour que les déconnexions soient détectées.

`POST /v1/cancel` interrompt la génération en cours. MLX ne se laisse pas annuler de l'extérieur : l'arrêt passe par le callback de progression, donc il prend effet à l'étape de débruitage suivante — pas instantanément. La requête en cours se termine en **499 `generation_stopped`** et le serveur reste utilisable, modèle toujours chaud.

`POST /v1/unload` libère les poids sans redémarrer — utile pour rendre les dizaines de Go d'un gros modèle à la machine. La route prend le verrou du moteur : si une génération tourne, elle attend qu'elle finisse plutôt que de la casser.

```sh
curl -s -X POST http://127.0.0.1:8765/v1/unload
# {"loaded_model":null,"memory":{"active_gb":0.0,"peak_gb":35.16,"cache_gb":0.0}}
```

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

`server-config.json` (JSON). Toute clé de la section `server` est surchargeable par `MFLUX_SERVER_<CLÉ>` en majuscules — `MFLUX_SERVER_PORT=9000`, `MFLUX_SERVER_API_KEY=…`, `MFLUX_SERVER_CORS_ORIGINS=https://a.example,https://b.example`. `MFLUX_SERVER_CONFIG` pointe vers un autre fichier de config.

### Section `server`

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
| `log_level` / `log_file` | `INFO` / `mflux.log` | `log_file: null` désactive le fichier. Chemin rendu absolu, dossiers parents créés |
| `log_json` | `false` | une ligne = un objet JSON, **sur stdout**. Voir ci-dessous |
| `progress_log_every` | `1` | log de progression toutes les N étapes ; `0` pour couper |
| `shutdown_grace_s` | `10` | borne l'arrêt gracieux ; sans ça un SIGTERM en pleine génération attendrait `request_timeout_s` |

### Section `models` — par modèle

Chaque entrée peut contenir ces clés (toutes optionnelles, `null` = défaut du catalogue) :

| clé | valeurs | rôle |
|---|---|---|
| `enabled` | `true` / `false` | active ou retire le modèle du catalogue exposé |
| `default_size` | `"WxH"` ou `null` | résolution par défaut (ex: `"1024x1024"`). Troncée au multiple de 16. |
| `default_steps` | entier ≥ 1 ou `null` | nombre d'étapes de débruitage par défaut |
| `default_guidance` | float ≥ 0 ou `null` | échelle CFG par défaut. Refusée sur les modèles distillés (`flux2-klein`, `z-image-turbo`). |
| `quantize` | 3/4/5/6/8, 0 ou `null` | quantification au chargement. `0` = aucune (bf16). `null` = défaut du catalogue. |
| `model_path` | chemin, repo HF ou `null` | source des poids, à la place de celle du catalogue. Sert surtout à `flux2-dev`, dont l'artefact pré-quantifié est propre à la machine. |
| `enable_edit` | `true` / `false` ou `null` | active la variante d'édition instructionnelle. `null` = défaut du catalogue. |

Exemple complet :

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "api_key": null,
    "cors_origins": ["*"],
    "max_n": 4,
    "request_timeout_s": 2400,
    "image_store": "images",
    "image_ttl_s": 3600,
    "max_upload_mb": 25,
    "default_response_format": "url",
    "log_level": "INFO",
    "log_file": "mflux.log",
    "progress_log_every": 1
  },
  "default_model": "flux2-klein",
  "models": {
    "flux2-klein": {
      "enabled": true,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null,
      "enable_edit": true
    },
    "flux2-dev": {
      "enabled": true,
      "model_path": null,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": 8,
      "enable_edit": null
    },
    "qwen-image": {
      "enabled": true,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": null,
      "enable_edit": false
    },
    "z-image": {
      "enabled": true,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": 8,
      "enable_edit": null
    },
    "z-image-turbo": {
      "enabled": true,
      "default_size": null,
      "default_steps": null,
      "default_guidance": null,
      "quantize": 8,
      "enable_edit": null
    }
  }
}
```

#### Surcharge de `steps` et `size` — à la volée ou dans la config

Les deux paramètres sont surchargeables **à chaque requête** (priorité absolue), puis dans la config, puis par le catalogue :

```bash
# À la volée — ignore les défauts de config et du catalogue
curl http://127.0.0.1:8765/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "un renard roux dans la neige", "size": "1024x1024", "steps": 30}'
```

Dans la config, pour changer le comportement par défaut d'un modèle :

```json
{
  "models": {
    "z-image": {
      "default_size": "1024x1024",
      "default_steps": 30,
      "quantize": 8
    }
  }
}
```

#### Quantification — pourquoi `z-image` en 8-bit ?

Les repos `Z-Image-bf16` et `Z-Image-Turbo-bf16` sont stockés en bf16 (précision complète), mais le serveur les quantifie à 8 bits au chargement. Raisons :

- **Mémoire** : un modèle ~9B en bf16 prend ~18 Go. En int8, ~9 Go. Sur mémoire unifiée (Mac), ça fait la différence entre pouvoir ou non garder le modèle chaud + faire tourner autre chose.
- **Coût amorti** : la quantification est payée une seule fois au premier chargement, puis le modèle reste en mémoire.
- **Qualité** : la perte visuelle est négligeable pour de l'image générative.

Pour désactiver la quantification et lancer en bf16 :

```json
{
  "models": {
    "z-image": { "quantize": 0 }
  }
}
```

## FLUX.2-dev — 32B en 8 bits

`flux2-dev` est le seul modèle du catalogue qui demande une préparation, pour deux raisons cumulées.

**mflux 0.18.0 ne sait pas charger FLUX.2-dev.** Sa famille FLUX.2 est *klein-only* : `AVAILABLE_MODELS` ne contient que les variantes klein, et `Flux2Initializer` câble `Qwen3TextEncoder` et `Flux2KleinWeightDefinition` en dur. Mais l'écart réel est mince — vérifié tenseur par tenseur contre les index de poids du repo :

| composant | verdict |
|---|---|
| transformer | l'architecture est celle de klein en plus grand ; chaque clé de `transformer/config.json` est un kwarg de `Flux2Transformer`. Le mapping de poids de mflux couvre **329 des 331** tenseurs. |
| poids manquants | `time_guidance_embed.guidance_embedder.linear_{1,2}` — FLUX.2-dev est *guidance-distilled*, klein non. Deux `WeightTarget` ajoutés. |
| VAE | identique (`AutoencoderKLFlux2`, 32 canaux latents) : le mapping de mflux s'applique tel quel. |
| scheduler | les défauts `sigma_*` de `ModelConfig` correspondent déjà au `scheduler_config.json` du repo. |
| encodeur texte | **le seul vrai écart.** FLUX.2-dev empile trois états cachés d'un `Mistral3ForConditionalGeneration` (40 couches, hidden 5120 → 3 × 5120 = `joint_attention_dim` = 15360), là où klein utilise Qwen3. |

L'encodeur est donc porté en MLX dans [`mflux_server/flux2_dev/mistral3.py`](mflux_server/flux2_dev/mistral3.py). Le décodeur Mistral est du dense standard et *la seule* différence structurelle avec `Qwen3VLAttention` de mflux est l'absence de `q_norm`/`k_norm` par tête — le RMSNorm, le MLP SwiGLU, le RoPE et les helpers GQA sont réutilisés tels quels. Le résultat est validé contre `MistralModel` de transformers : mêmes poids, mêmes états cachés à 7e-7 près, sur les trois configurations de padding.

Conséquence agréable : **aucun Torch à l'inférence**. Tout tient en MLX, l'encodeur reste chaud avec le transformer, et il n'y a rien à recharger entre deux prompts.

**Le repo est en bf16, et ça ne rentre pas.** Transformer 64,5 Go + encodeur 45,8 Go + VAE, soit ~111 Go de poids résidents. En 8 bits on tombe à ~58 Go, confortable sur 96 Go de mémoire unifiée — mais quantifier au chargement suppose justement de tenir le bf16 en mémoire d'abord. D'où une conversion préalable, une fois :

```bash
uv run mflux-server-prequantize            # → ~/.cache/mflux-server/flux2-dev-mlx-8bit
```

Le repo est *gated* : il faut un token Hugging Face avec l'accès accordé (`hf auth login`). Compter ~113 Go de téléchargement et ~58 Go écrits. Le script travaille **composant par composant**, et quantifie le transformer **bloc par bloc** : sans ça le pic mémoire atteindrait ~96 Go, contre ~66 Go ainsi. L'ordre par défaut (transformer, encodeur, VAE) permet de purger le bf16 du cache HF entre deux étapes — le pic disque passe de ~169 Go à ~97 Go, et le script rappelle quoi supprimer.

Pour convertir un seul composant, par exemple pour valider l'encodeur avant d'engager les 64 Go du transformer :

```bash
uv run mflux-server-prequantize --components text_encoder
```

Le rechargement ne demande aucune configuration : mflux détecte le `quantization_level` inscrit dans les métadonnées safetensors et quantifie la structure avant de poser les poids. Si l'artefact est ailleurs, renseigne `model_path` :

```json
{
  "models": {
    "flux2-dev": { "model_path": "/Volumes/Assets/models/flux2-dev-mlx-8bit" }
  }
}
```

Sans artefact, le serveur refuse le chargement avec un message qui rappelle la commande — plutôt que de retomber silencieusement sur le repo bf16 et de tenter une quantification de 111 Go.

Deux limites à connaître :

- **Pas de `negative_prompt`.** FLUX.2-dev est guidance-distilled : la guidance est un scalaire embarqué dans le transformer, pas un CFG. Une seule passe par étape (deux fois plus rapide que du CFG), mais aucun prompt négatif possible. La guidance reste réglable, défaut 4.0.
- **Pas d'édition multi-images.** `/v1/images/edits` fonctionne en img2img, mais le conditionnement par tokens d'images de référence n'est pas implémenté.

Enfin, `request_timeout_s` passe à `2400` : 50 étapes sur un 32B dépassent largement les 900 s d'origine.

### Logs JSON pour un superviseur

`"log_json": true` (ou `MFLUX_SERVER_LOG_JSON=1`) fait passer les logs en JSON Lines, une ligne par objet :

```json
{"ts":"2026-07-27T14:19:02","level":"INFO","logger":"mflux_server","message":"z-image-turbo seed=42 1280x720 — étape 3/9","event":"generation_step","fields":{"step":3,"total":9}}
```

`event` vaut `model_loading`, `model_ready`, `model_unload`, `generation_start`, `generation_step`, `generation_done`, `generation_cancel_requested`, ou côté conversion `prequantize_component_start`, `prequantize_progress`, `prequantize_component_done`. Le `message` humain reste présent à côté des champs structurés.

**Dans ce mode les logs sortent sur stdout, pas sur stderr**, et l'access log d'uvicorn est coupé. La raison est concrète : mflux affiche sa barre de débruitage avec tqdm (`Config.time_steps`), qui écrit sur stderr des fragments terminés par `\r` **sans retour à la ligne**. Les objets JSON s'y collaient sur le même segment — `\r 0%| | 0/40 [00:00<?, ?it/s]{"ts": …}` — et un consommateur qui découpe sur `\n` les manquait tous. tqdm n'offre aucune variable d'environnement pour se taire, d'où la séparation des canaux :

- **stdout** : les événements structurés, une ligne = un JSON valide, rien d'autre ;
- **stderr** : le texte destiné aux humains, les barres de progression, et les logs de démarrage d'uvicorn.

`mflux-server-prequantize --json-logs` applique la même configuration à la conversion.

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

Spécifiques à `flux2-dev`, où l'on sort du chemin balisé :

- **`ModelConfig.from_name("black-forest-labs/FLUX.2-dev")` ne lève même pas.** `can_infer_substring` trouve l'alias `"dev"` dans le nom et fabrique silencieusement une config **FLUX.1**-dev (`config_resolution.py:57-64`). La config est donc construite à la main, et `registry._LOCAL_MODEL_CONFIGS` la résout à côté des factories de `ModelConfig`.
- **La guidance est pré-multipliée par 1000.** `Flux2Transformer.__call__` ne met la guidance à l'échelle que si elle vaut 1.0 ou moins (`flux2/…/transformer.py:91`), alors que le chemin FLUX.1 — le seul exercé en amont avec `guidance_embeds=True` — multiplie toujours par `num_train_steps` (`flux/…/transformer.py:155`). Aucun modèle livré par mflux n'active `guidance_embeds` sur le transformer FLUX.2, donc ce chemin n'y est pas testé. `test_la_guidance_doit_etre_premultipliee_par_mille` sert de vigie : si mflux corrige l'heuristique, il casse et il faudra retirer la compensation.
- **Le tokenizer complète à gauche, ce qui produisait des NaN.** Sous masque causal, une requête de padding en tête de séquence n'a qu'elle-même à regarder — et elle est masquée. Le softmax renvoie NaN, et la ligne suivante le propage (`0 × NaN = NaN`) : dès la deuxième couche, *toutes* les positions sont contaminées et le prompt entier part en NaN, sans aucune exception levée. Les lignes entièrement fermées sont donc rouvertes, comme le fait `AttentionMaskConverter._unmask_unattended` de transformers.
- **`LanguageTokenizer` de mflux ne convient pas.** Avec `use_chat_template=True` il envoie `[{"role": "user", …}]` et `add_generation_prompt=True` (`tokenizer.py:86-92`), là où FLUX.2-dev attend une conversation system + user, des contenus en listes de parts typées, et `add_generation_prompt=False`. Un tokenizer maison est branché via `TokenizerDefinition.encoder_class`.
- **Pas de `mx.compile` sur la boucle de débruitage**, contrairement à `Flux2Klein` qui l'active hors M1/M2 : sur 32B le graphe compilé peut dépasser le watchdog GPU de Metal.
- **Les composants sont quantifiés un par un**, parce que `WeightApplier.apply_and_quantize` charge tout le bf16 avant de quantifier. Le rechargement, en revanche, ne demande rien : `WeightLoader._load_component` essaie `_try_load_mflux_format` en premier (`weight_loader.py:89-92`) et lit le `quantization_level` écrit par `ModelSaver`.
