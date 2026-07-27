//! Lecture et écriture de `server-config.json`.
//!
//! L'app est propriétaire de ce fichier : il vit dans son espace de données et
//! `MFLUX_SERVER_CONFIG` y pointe explicitement. Ce n'est pas facultatif —
//! `mflux_server` cherche sinon un chemin relatif à son propre paquet, donc
//! `site-packages/` dans une installation par wheel, où le fichier n'existe pas ;
//! et une configuration absente n'est pas une erreur de son point de vue. Sans
//! cette variable, tous les réglages seraient silencieusement ignorés.
//!
//! On ne valide rien ici : le serveur le fait déjà, et remonte une erreur de
//! démarrage explicite. Dupliquer le schéma en Rust ne ferait que le laisser
//! dériver.

use serde_json::{json, Value};

use crate::paths::Paths;

/// Configuration de départ, alignée sur celle du dépôt.
fn default_config() -> Value {
    json!({
        "server": {
            "host": "127.0.0.1",
            "port": 8765,
            "api_key": null,
            "cors_origins": ["*"],
            "max_n": 4,
            // 50 étapes sur un modèle 32B dépassent largement les 900 s d'origine.
            "request_timeout_s": 2400,
            "image_ttl_s": 3600,
            "max_upload_mb": 25,
            "default_response_format": "url",
            "log_level": "INFO",
            "progress_log_every": 1,
            "shutdown_grace_s": 10
        },
        "default_model": "flux2-klein",
        "models": {
            "flux2-klein": {"enabled": true, "quantize": null, "enable_edit": true},
            "flux2-dev": {"enabled": true, "quantize": 8, "model_path": null},
            "qwen-image": {"enabled": true, "quantize": null, "enable_edit": false},
            "z-image": {"enabled": true, "quantize": 8},
            "z-image-turbo": {"enabled": true, "quantize": 8}
        }
    })
}

/// Crée le fichier s'il manque. Appelé avant chaque démarrage.
pub fn ensure_exists(paths: &Paths) -> Result<(), String> {
    if paths.config.is_file() {
        return Ok(());
    }
    write(paths, &default_config())
}

pub fn read(paths: &Paths) -> Result<Value, String> {
    if !paths.config.is_file() {
        return Ok(default_config());
    }
    let text = std::fs::read_to_string(&paths.config)
        .map_err(|error| format!("lecture de {} impossible : {error}", paths.config.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("{} n'est pas un JSON valide : {error}", paths.config.display()))
}

pub fn write(paths: &Paths, value: &Value) -> Result<(), String> {
    paths.ensure()?;
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| format!("sérialisation impossible : {error}"))?;
    // Écriture puis remplacement atomique : une coupure au mauvais moment
    // laisserait sinon une configuration tronquée, que le serveur refuserait au
    // démarrage suivant.
    let temporary = paths.config.with_extension("json.tmp");
    std::fs::write(&temporary, format!("{text}\n"))
        .map_err(|error| format!("écriture de {} impossible : {error}", temporary.display()))?;
    std::fs::rename(&temporary, &paths.config)
        .map_err(|error| format!("remplacement de {} impossible : {error}", paths.config.display()))
}

/// Durée d'arrêt gracieux déclarée dans la configuration, pour dimensionner
/// l'attente avant SIGKILL.
pub fn shutdown_grace_s(paths: &Paths) -> f64 {
    read(paths)
        .ok()
        .and_then(|value| value.get("server")?.get("shutdown_grace_s")?.as_f64())
        .unwrap_or(10.0)
}
