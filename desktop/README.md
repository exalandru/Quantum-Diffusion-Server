# Quantum Diffusion Server

Panneau de contrôle macOS pour [mflux-server](../README.md) : une app double-cliquable qui installe son propre Python, démarre et surveille le serveur, expose sa configuration dans un formulaire, et pilote la préparation des modèles.

Tauri 2 (Rust) + React 19 + Vite. **Apple Silicon uniquement** — mlx l'est par construction.

## Construire

```sh
cd desktop
npm install
npm run app:build     # → src-tauri/target/release/bundle/
```

Prérequis : Node, Rust, et `uv` dans le `PATH` (il est copié dans le bundle comme sidecar).

Résultat : `Quantum Diffusion Server.app` (**57 Mo**) et un `.dmg` (**25 Mo**), signés en ad-hoc (`signingIdentity: "-"`). Ni certificat Developer ID ni notarisation : c'est un usage personnel. Après un transfert d'une machine à l'autre, `xattr -d com.apple.quarantine "Quantum Diffusion Server.app"` peut être nécessaire (les guillemets comptent : le nom contient des espaces).

En développement : `npm run app:dev`. Attention, `tauri dev` hérite d'un dossier courant confortable et masque donc les problèmes de chemins relatifs — les vérifications sérieuses se font sur le `.app` construit.

## Pourquoi pas PyInstaller

Le venv du serveur pèse **1,1 Go** : torch 501 Mo (dépendance dure de mflux, importée au niveau module dans `weight_loader.py`), mlx 178 Mo dont un `mlx.metallib` de **150 Mo** de shaders Metal, opencv et matplotlib 142 Mo tirés par mflux sans qu'on s'en serve. Au total **186 binaires natifs** `.so`/`.dylib`. Ce metallib et ces 186 binaires sont exactement ce qui rend le gel fragile, et il faudrait tous les signer avec hardened runtime.

`uv`, lui, est un binaire unique de 50 Mo qui sait télécharger CPython et reconstituer l'environnement depuis `uv.lock` — 86 paquets, à l'identique. L'app reste donc à 57 Mo, aucun binaire natif tiers n'entre dans le bundle signé, et l'environnement s'installe dans l'espace de données au premier lancement.

## Ce que fait l'app

```
~/Library/Application Support/com.exalandru.qds/
├── server/               copie inscriptible du projet Python (source de uv sync)
├── env/                  environnement virtuel, ~1,0 Go
├── python/               CPython 3.12 géré par uv, ~67 Mo
├── uv-cache/             purgé après installation (1,5 Go récupérés)
├── images/               images servies en response_format="url"
└── server-config.json    piloté par l'onglet Configuration
```

**Premier lancement.** L'app copie les ressources du bundle vers `server/` — le bundle est en lecture seule — puis lance :

```sh
uv sync --frozen --no-dev --no-editable --managed-python --python 3.12 --project <appdata>/server
```

Environ 1,1 Go de téléchargement. `--no-editable` importe : sans lui le paquet installé pointerait vers la copie du projet. `--python 3.12` aussi, voir plus bas.

**Démarrage du serveur.** Le Rust choisit un port libre, crée les dossiers d'écriture, puis lance `<appdata>/env/bin/mflux-server` avec tout en absolu :

| variable | valeur |
|---|---|
| `MFLUX_SERVER_CONFIG` | `<appdata>/server-config.json` |
| `MFLUX_SERVER_HOST` / `PORT` | `127.0.0.1` / port choisi en Rust |
| `MFLUX_SERVER_IMAGE_STORE` | `<appdata>/images` |
| `MFLUX_SERVER_LOG_FILE` | `""` — c'est l'app qui capture |
| `MFLUX_SERVER_LOG_JSON` | `1` |
| `HF_HOME`, `HF_HUB_DISABLE_PROGRESS_BARS`, `PYTHONUNBUFFERED` | |

**Arrêt.** SIGTERM au groupe de processus, attente bornée à `shutdown_grace_s + 8 s`, puis SIGKILL. Le groupe et non le seul pid, sinon quitter l'app laisserait le serveur orphelin : macOS ne récolte pas les petits-enfants.

## Quatre pièges rencontrés, et leur correctif

Chacun a été trouvé en exécutant la chaîne pour de vrai, pas en la relisant.

**Le serveur ne démarrait pas depuis un `.app`.** `image_store` et `log_file` étaient relatifs au dossier courant et créés pendant `create_app`, avant même le bind. Lancé par le Finder, le dossier courant est `/`, en lecture seule → échec immédiat. Corrigé côté serveur : les deux chemins sont désormais rendus absolus à la validation, et `setup_logging` crée les dossiers parents. L'app pose en plus un `current_dir` explicite, en ceinture et bretelles.

**`server-config.json` était silencieusement ignoré.** Son chemin par défaut est relatif au *paquet* Python, donc `site-packages/` dans une installation par wheel, où le fichier n'existe pas — et une configuration absente n'est pas une erreur. Tous les réglages seraient partis à la poubelle sans un mot. D'où `MFLUX_SERVER_CONFIG` toujours explicite, et un `warning` côté serveur quand aucun fichier n'est trouvé.

**hatchling refusait de construire le wheel.** `pyproject.toml` déclare `readme = "README.md"` : sans ce fichier dans les ressources, `uv sync` échoue à l'étape de build. `README.md` fait donc partie de la charge embarquée — ce n'est pas de la documentation ici, c'est une dépendance de build.

**uv installait Python 3.13 au lieu de 3.12.** La découverte de `.python-version` dépend du dossier courant, imprévisible pour un sidecar. Comme `requires-python` autorise `>=3.12,<3.14`, uv prenait le plus récent — et `uv.lock`, dont les marqueurs distinguent 3.12 de 3.13 (`torch>=2.8`, `tokenizers`), résolvait un autre jeu de paquets que celui qui avait été testé. Le bootstrap lit maintenant `.python-version` dans la copie du projet et le passe en `--python`, ce qui supprime toute ambiguïté. Au passage, le glob `resources/server/**/*` de Tauri **ignore les fichiers cachés** : `.python-version` doit être listé explicitement dans `tauri.conf.json`.

## Architecture

Le Rust ne fait que ce que le navigateur ne peut pas : installer l'environnement, surveiller un processus, écrire un fichier.

| fichier | rôle |
|---|---|
| [src-tauri/src/paths.rs](src-tauri/src/paths.rs) | emplacements de travail, tous absolus |
| [src-tauri/src/bootstrap.rs](src-tauri/src/bootstrap.rs) | copie des ressources, `uv sync`, purge du cache |
| [src-tauri/src/supervisor.rs](src-tauri/src/supervisor.rs) | cycle de vie du serveur, échelle SIGTERM → SIGKILL, relais des sorties |
| [src-tauri/src/config.rs](src-tauri/src/config.rs) | `server-config.json`, écriture atomique |
| [src-tauri/src/lib.rs](src-tauri/src/lib.rs) | commandes exposées à React |

Tout le reste passe par l'API HTTP du serveur, que React interroge directement — y compris `/v1/progress` en Server-Sent Events, qu'il serait absurde de faire transiter par le pont IPC. Aucun schéma n'est dupliqué en Rust ni en TypeScript : le serveur valide déjà sa configuration et refuse ce qui n'a pas de sens par un 400.

La progression utilise `fetch` et non `EventSource`, parce que ce dernier ne permet pas d'en-tête `Authorization` et que `/v1/progress` est protégé comme le reste de `/v1`.

### Les deux canaux de sortie

Le serveur tourne avec `MFLUX_SERVER_LOG_JSON=1`, ce qui sépare :

- **stdout** : les événements structurés, une ligne = un JSON valide, rien d'autre ;
- **stderr** : le texte pour les humains, les barres tqdm, les logs de démarrage d'uvicorn.

Cette séparation n'est pas cosmétique. mflux affiche sa barre de débruitage avec tqdm, qui écrit sur stderr des fragments terminés par un retour chariot **sans saut de ligne** : les objets JSON s'y collaient sur le même segment (`\r 0%| | 0/40 [...]{"ts": …}`) et un consommateur qui découpe sur `\n` les manquait tous. tqdm n'offre aucune variable d'environnement pour se taire, d'où le déplacement du JSON sur stdout et la coupure de l'access log d'uvicorn qui l'aurait pollué.

L'onglet Logs affiche les deux : filtrage par niveau sur les événements structurés, texte brut affiché tel quel.

## Écrans

- **Installation** — s'affiche seul tant que l'environnement n'est pas prêt ou qu'il a été construit par une version antérieure de l'app. Sortie d'uv en direct : c'est un téléchargement d'un gigaoctet, un indicateur indéterminé ne suffirait pas.
- **Tableau de bord** — état, modèle chaud, mémoire MLX, démarrer/arrêter/redémarrer, barre de progression alimentée par SSE, annuler, libérer la mémoire, ouvrir `/docs`. « Serveur prêt » et « modèle chaud » sont deux états distincts : le serveur répond en une seconde mais ne charge aucun poids au démarrage.
- **Configuration** — formulaire sur `server-config.json`. Les contrôles sont grisés d'après `/v1/capabilities` : la guidance d'un modèle distillé est figée, le serveur refuse déjà toute valeur par un 400. Un rappel indique que la configuration n'est lue qu'au démarrage.
- **Modèles** — token HuggingFace (écrit là où `hf auth login` l'écrit, pour ne pas dupliquer un secret déjà en clair à côté), assistant de conversion de `flux2-dev` composant par composant, et le catalogue avec les capacités déclarées.
- **Logs** — les deux canaux, filtrables.

## Limites

- Pas de playground de génération : le serveur reste piloté par un frontend compatible OpenAI.
- Pas de mise à jour automatique (`tauri-plugin-updater`).
- Pas de rechargement à chaud de la configuration : l'app redémarre le processus.
- La sélection du port a une fenêtre de course inévitable — `mflux-server` n'accepte pas de socket préouverte et n'annonce pas le port obtenu.
- Les interactions de l'interface n'ont pas été cliquées automatiquement : capture d'écran et pilotage AppleScript demandent des autorisations qui ne s'accordent qu'à la main. Ce qui est vérifié, c'est la chaîne sous l'interface — bootstrap, environnement produit, serveur opérationnel, absence d'orphelins.
