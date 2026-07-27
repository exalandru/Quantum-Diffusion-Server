//! Installation de l'environnement Python au premier lancement.
//!
//! On n'embarque pas les 1,1 Go de dépendances dans le bundle : torch pèse à lui
//! seul 501 Mo, mlx en ajoute 178 dont un `mlx.metallib` de 150 Mo de shaders
//! Metal, et l'ensemble compte 186 binaires natifs. `uv`, lui, est un binaire
//! unique de 50 Mo qui sait télécharger CPython et reconstituer l'environnement
//! depuis `uv.lock` — 86 paquets, à l'identique.
//!
//! Deux précautions :
//!
//! * on copie le projet du bundle vers l'espace de données avant de lancer
//!   `uv sync`, parce que le bundle est en lecture seule ;
//! * `--no-editable`, sans quoi le paquet installé pointerait vers cette copie
//!   plutôt que d'être vraiment installé.

use std::path::Path;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

use crate::paths::Paths;

/// Événement de progression du bootstrap, poussé vers l'interface.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum BootstrapEvent {
    Step { message: String },
    Output { line: String },
    Done,
    Failed { message: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BootstrapStatus {
    /// `true` si l'environnement est prêt et à jour.
    pub ready: bool,
    /// Version de l'app ayant produit l'environnement présent.
    pub installed_version: Option<String>,
    /// Version de l'app en cours d'exécution.
    pub app_version: String,
    pub env_path: String,
}

pub fn status(app: &AppHandle) -> Result<BootstrapStatus, String> {
    let paths = Paths::new(app)?;
    let app_version = app.package_info().version.to_string();
    let installed_version = paths.stamped_version();
    Ok(BootstrapStatus {
        ready: paths.env_ready() && installed_version.as_deref() == Some(app_version.as_str()),
        installed_version,
        app_version,
        env_path: paths.env.display().to_string(),
    })
}

fn emit(app: &AppHandle, event: BootstrapEvent) {
    // Une interface absente ne doit pas faire échouer l'installation.
    let _ = app.emit("bootstrap", event);
}

/// Recopie le projet Python et (re)construit l'environnement.
pub async fn run(app: AppHandle) -> Result<(), String> {
    let paths = Paths::new(&app)?;
    paths.ensure()?;
    let app_version = app.package_info().version.to_string();

    let result = install(&app, &paths, &app_version).await;
    match &result {
        Ok(()) => emit(&app, BootstrapEvent::Done),
        Err(message) => emit(
            &app,
            BootstrapEvent::Failed {
                message: message.clone(),
            },
        ),
    }
    result
}

async fn install(app: &AppHandle, paths: &Paths, app_version: &str) -> Result<(), String> {
    if paths.stamped_version().as_deref() != Some(app_version) {
        emit(
            app,
            BootstrapEvent::Step {
                message: "Copie du projet Python…".into(),
            },
        );
        copy_project(app, paths)?;
    }

    emit(
        app,
        BootstrapEvent::Step {
            message: "Installation de Python et des dépendances (environ 1,1 Go)…".into(),
        },
    );
    sync(app, paths).await?;

    if !paths.env_ready() {
        return Err(format!(
            "uv s'est terminé sans erreur mais {} est absent.",
            paths.server_bin().display()
        ));
    }

    std::fs::write(&paths.stamp, app_version)
        .map_err(|error| format!("impossible d'écrire {} : {error}", paths.stamp.display()))?;

    // Le cache de téléchargement pèse 1,6 Go après un premier sync, dont 1,5 de
    // roues déjà installées. uv relie l'environnement au cache par liens durs :
    // supprimer l'entrée de cache ne fait que retirer un lien, les fichiers
    // restent via site-packages — vérifié, l'environnement reste importable.
    emit(
        app,
        BootstrapEvent::Step {
            message: "Nettoyage du cache de téléchargement…".into(),
        },
    );
    prune_cache(app, paths).await;
    Ok(())
}

/// Récupère l'espace du cache uv. L'échec n'est pas fatal : ce n'est que du
/// disque, et l'environnement est déjà fonctionnel à ce stade.
async fn prune_cache(app: &AppHandle, paths: &Paths) {
    let Ok(command) = app.shell().sidecar("uv") else {
        return;
    };
    let command = command
        .args(["cache", "prune", "--ci"])
        .env("UV_CACHE_DIR", &paths.uv_cache)
        .env("UV_NO_CONFIG", "1");
    if let Ok((mut events, _child)) = command.spawn() {
        while let Some(event) = events.recv().await {
            if let CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) = event {
                let text = String::from_utf8_lossy(&bytes);
                for line in text.split(['\r', '\n']).filter(|line| !line.trim().is_empty()) {
                    emit(
                        app,
                        BootstrapEvent::Output {
                            line: line.trim_end().to_owned(),
                        },
                    );
                }
            }
        }
    }
}

/// Copie `resources/server/` du bundle vers `<appdata>/server/`.
fn copy_project(app: &AppHandle, paths: &Paths) -> Result<(), String> {
    let source = app
        .path()
        .resolve("resources/server", tauri::path::BaseDirectory::Resource)
        .map_err(|error| format!("ressources introuvables dans le bundle : {error}"))?;

    // On remplace intégralement : un reste d'une version précédente ferait
    // installer du code périmé.
    if paths.server.exists() {
        std::fs::remove_dir_all(&paths.server)
            .map_err(|error| format!("impossible de nettoyer {} : {error}", paths.server.display()))?;
    }
    copy_tree(&source, &paths.server)
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    std::fs::create_dir_all(destination)
        .map_err(|error| format!("impossible de créer {} : {error}", destination.display()))?;
    let entries = std::fs::read_dir(source)
        .map_err(|error| format!("impossible de lire {} : {error}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("lecture interrompue : {error}"))?;
        let target = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .map_err(|error| format!("type de fichier inconnu : {error}"))?;
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)
                .map_err(|error| format!("copie de {} échouée : {error}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// Version épinglée par le dépôt, lue dans la copie du projet.
///
/// On la passe explicitement à uv plutôt que de compter sur sa découverte
/// automatique : celle-ci dépend du dossier courant, qui est imprévisible pour
/// un sidecar lancé depuis un `.app`. Sans ça, uv prend le plus récent
/// interpréteur satisfaisant `requires-python` — mesuré : 3.13 au lieu de 3.12 —
/// et `uv.lock`, dont les marqueurs distinguent les deux, résout un autre jeu de
/// paquets que celui qui a été testé.
fn pinned_python(paths: &Paths) -> Result<String, String> {
    let file = paths.server.join(".python-version");
    let raw = std::fs::read_to_string(&file)
        .map_err(|error| format!("{} illisible : {error}", file.display()))?;
    let version = raw.trim().to_owned();
    if version.is_empty() {
        return Err(format!("{} est vide", file.display()));
    }
    Ok(version)
}

/// Lance `uv sync` en sidecar et relaie sa sortie vers l'interface.
async fn sync(app: &AppHandle, paths: &Paths) -> Result<(), String> {
    let python = pinned_python(paths)?;
    let command = app
        .shell()
        .sidecar("uv")
        .map_err(|error| format!("sidecar uv introuvable : {error}"))?
        .args([
            "sync",
            "--frozen",
            // Ni pytest ni ruff : ils ne servent qu'au développement.
            "--no-dev",
            // Sans ça, le paquet installé pointerait vers la copie du projet.
            "--no-editable",
            // Garantit qu'aucun Python système n'est requis.
            "--managed-python",
            "--python",
        ])
        .arg(&python)
        .args(["--project"])
        .arg(paths.server.as_os_str())
        // Chaque emplacement est explicite : rien ne doit atterrir dans le
        // dossier du projet ni dans les caches de l'utilisateur.
        .env("UV_PROJECT_ENVIRONMENT", &paths.env)
        .env("UV_PYTHON_INSTALL_DIR", &paths.python)
        .env("UV_CACHE_DIR", &paths.uv_cache)
        // Ignore un éventuel ~/.config/uv qui changerait la résolution.
        .env("UV_NO_CONFIG", "1");

    let (mut events, _child) = command
        .spawn()
        .map_err(|error| format!("impossible de lancer uv : {error}"))?;

    let mut failure: Option<String> = None;
    while let Some(event) = events.recv().await {
        match event {
            // uv écrit sa progression sur stderr.
            CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                for line in text.split(['\r', '\n']).filter(|line| !line.trim().is_empty()) {
                    emit(
                        app,
                        BootstrapEvent::Output {
                            line: line.trim_end().to_owned(),
                        },
                    );
                }
            }
            CommandEvent::Terminated(payload) => {
                if payload.code != Some(0) {
                    failure = Some(format!(
                        "uv sync a échoué (code {:?}, signal {:?})",
                        payload.code, payload.signal
                    ));
                }
            }
            CommandEvent::Error(message) => failure = Some(message),
            _ => {}
        }
    }

    match failure {
        Some(message) => Err(message),
        None => Ok(()),
    }
}
