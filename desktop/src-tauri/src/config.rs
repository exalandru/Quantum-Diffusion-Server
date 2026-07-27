//! Reading and writing `server-config.json`.
//!
//! The app owns this file: it lives in the app's data directory and
//! `MFLUX_SERVER_CONFIG` points at it explicitly. That is not optional —
//! otherwise `mflux_server` looks for a path relative to its own package, so
//! `site-packages/` in a wheel install, where the file does not exist; and from
//! its point of view a missing configuration is not an error. Without that
//! variable, every setting would be silently ignored.
//!
//! We validate nothing here: the server already does, and surfaces an explicit
//! startup error. Duplicating the schema in Rust would only let it drift.

use serde_json::{json, Value};

use crate::paths::Paths;

/// Starting configuration, aligned with the repo's.
fn default_config() -> Value {
    json!({
        "server": {
            "host": "127.0.0.1",
            "port": 8765,
            "api_key": null,
            "cors_origins": ["*"],
            "max_n": 4,
            // 50 steps on a 32B model far exceed the original 900s.
            "request_timeout_s": 2400,
            "image_ttl_s": 3600,
            "max_upload_mb": 25,
            "default_response_format": "url",
            "log_level": "INFO",
            "progress_log_every": 1,
            "shutdown_grace_s": 10
        },
        "default_model": "qwen-image",
        // Config-wide generation resolution; a per-model `default_size` still
        // wins over it, and `null` would leave every model on its catalogue size.
        "default_size": "1280x720",
        "models": {
            "flux2-klein": {"enabled": true, "quantize": null, "enable_edit": true},
            // Off by default: it answers 503 `model_not_prepared` until
            // `mflux-server-prequantize` has produced the local 8-bit artifact.
            // Shipping it enabled would advertise a model that cannot generate.
            "flux2-dev": {"enabled": false, "quantize": 8, "model_path": null},
            "qwen-image": {"enabled": true, "quantize": null, "enable_edit": false},
            "z-image": {"enabled": true, "quantize": 8},
            "z-image-turbo": {"enabled": true, "quantize": 8}
        }
    })
}

/// Create the file when missing. Called before every start.
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
        .map_err(|error| format!("could not read {}: {error}", paths.config.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("{} is not valid JSON: {error}", paths.config.display()))
}

pub fn write(paths: &Paths, value: &Value) -> Result<(), String> {
    paths.ensure()?;
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| format!("could not serialize: {error}"))?;
    // Write then rename atomically: an interruption at the wrong moment would
    // otherwise leave a truncated configuration, which the server would reject on
    // the next start.
    let temporary = paths.config.with_extension("json.tmp");
    std::fs::write(&temporary, format!("{text}\n"))
        .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
    std::fs::rename(&temporary, &paths.config)
        .map_err(|error| format!("could not replace {}: {error}", paths.config.display()))
}

/// Graceful-shutdown duration declared in the configuration, used to size the
/// wait before SIGKILL.
pub fn shutdown_grace_s(paths: &Paths) -> f64 {
    read(paths)
        .ok()
        .and_then(|value| value.get("server")?.get("shutdown_grace_s")?.as_f64())
        .unwrap_or(10.0)
}

/// Port declared in the configuration, `8765` by default.
///
/// The app used to pick a random free port on every start, which made the
/// address unstable and could not be configured. Reading it from the file makes
/// it both fixed across restarts and editable from the Configuration tab.
pub fn port(paths: &Paths) -> u16 {
    read(paths)
        .ok()
        .and_then(|value| value.get("server")?.get("port")?.as_u64())
        .and_then(|value| u16::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PORT)
}

pub const DEFAULT_PORT: u16 = 8765;
