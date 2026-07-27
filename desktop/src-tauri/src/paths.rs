//! Emplacements de travail de l'app.
//!
//! Tout vit sous `app_data_dir()`, c'est-à-dire
//! `~/Library/Application Support/com.exalandru.qds/`. Rien n'est écrit
//! dans le bundle : il est en lecture seule et remplacé à chaque mise à jour.
//!
//! Ces chemins ne sont pas un détail de confort : `mflux_server` crée son
//! dossier d'images et son fichier de log pendant l'initialisation de
//! l'application, avant même que le serveur n'écoute. Comme un processus lancé
//! par le Finder hérite de `/` comme dossier courant, tout chemin relatif
//! ferait échouer le démarrage. On passe donc au serveur des chemins absolus
//! dont les dossiers parents existent déjà.

use std::path::{Path, PathBuf};

use tauri::{AppHandle, Manager};

#[derive(Debug, Clone)]
pub struct Paths {
    /// Racine de travail, sous `~/Library/Application Support/`.
    pub data: PathBuf,
    /// Copie inscriptible du projet Python, source de `uv sync`.
    pub server: PathBuf,
    /// Environnement virtuel construit par uv.
    pub env: PathBuf,
    /// CPython géré par uv (python-build-standalone).
    pub python: PathBuf,
    /// Cache de téléchargement d'uv.
    pub uv_cache: PathBuf,
    /// Images servies en `response_format="url"`.
    pub images: PathBuf,
    /// `server-config.json` piloté par le formulaire de configuration.
    pub config: PathBuf,
    /// Version de l'app ayant produit `server/`, pour savoir quand resynchroniser.
    pub stamp: PathBuf,
}

impl Paths {
    pub fn new(app: &AppHandle) -> Result<Self, String> {
        let data = app
            .path()
            .app_data_dir()
            .map_err(|error| format!("dossier de données introuvable : {error}"))?;
        Ok(Self {
            server: data.join("server"),
            env: data.join("env"),
            python: data.join("python"),
            uv_cache: data.join("uv-cache"),
            images: data.join("images"),
            config: data.join("server-config.json"),
            stamp: data.join("server").join(".bootstrap-version"),
            data,
        })
    }

    /// Crée tout ce dans quoi on écrira. Appelé avant chaque démarrage : le
    /// serveur, lui, ne crée pas les dossiers parents de son fichier de log.
    pub fn ensure(&self) -> Result<(), String> {
        for directory in [&self.data, &self.images] {
            std::fs::create_dir_all(directory)
                .map_err(|error| format!("impossible de créer {} : {error}", directory.display()))?;
        }
        Ok(())
    }

    /// Le point d'entrée console installé par `uv sync`.
    pub fn server_bin(&self) -> PathBuf {
        self.env.join("bin").join("mflux-server")
    }

    /// L'outil de pré-quantification, installé par le même wheel.
    pub fn prequantize_bin(&self) -> PathBuf {
        self.env.join("bin").join("mflux-server-prequantize")
    }

    /// `true` si `uv sync` a déjà produit un environnement exploitable.
    pub fn env_ready(&self) -> bool {
        self.server_bin().is_file()
    }

    /// Version de l'app ayant produit la copie du projet, si elle existe.
    pub fn stamped_version(&self) -> Option<String> {
        std::fs::read_to_string(&self.stamp)
            .ok()
            .map(|value| value.trim().to_owned())
    }

    /// Emplacement du token HuggingFace, tel que `huggingface_hub` le résout.
    ///
    /// On réutilise le fichier écrit par `hf auth login` au lieu d'un second
    /// magasin de secrets : dupliquer le token dans le Keychain alors que le
    /// même se trouve en clair juste à côté n'apporterait rien.
    pub fn hf_token_file(hf_home: &Path) -> PathBuf {
        hf_home.join("token")
    }

    /// `HF_HOME` par défaut, celui de `huggingface_hub`.
    pub fn default_hf_home() -> PathBuf {
        std::env::var_os("HF_HOME")
            .map(PathBuf::from)
            .or_else(|| dirs_home().map(|home| home.join(".cache").join("huggingface")))
            .unwrap_or_else(|| PathBuf::from("/tmp/huggingface"))
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
