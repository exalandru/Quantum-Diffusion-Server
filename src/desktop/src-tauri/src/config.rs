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
            "shutdown_grace_s": 10,
            // Release the model after this long without a generation. `null`
            // keeps it warm forever, `0` frees it as soon as a request ends.
            "idle_unload_s": null
        },
        // Both enabled models are Apache-2.0 and ungated: a fresh install
        // generates with no token, no access request and no licence to accept.
        // Everything gated or non-commercial ships off, and so do the slow ones.
        "default_model": "z-image-turbo",
        // Config-wide resolution; a per-model `default_size` still wins over it.
        "default_size": "1280x720",
        // 4 bits everywhere it means anything. Skipped on models whose weights
        // already carry their precision: mflux keeps the stored value and only
        // prints "Ignoring -q", so the setting would be a lie there.
        "default_quantize": 4,
        "models": {
            "z-image-turbo": {"enabled": true},
            "ernie-image-turbo": {"enabled": true},
            "z-image": {"enabled": false},
            "ernie-image": {"enabled": false},
            "qwen-image-2512": {"enabled": false, "enable_edit": false},
            // Gated and non-commercial, like the three below it.
            "flux2-klein": {"enabled": false, "enable_edit": true},
            // Also needs `mflux-server-prequantize` before it can answer at all.
            "flux2-dev": {"enabled": false, "quantize": 8, "model_path": null},
            "fibo-lite": {"enabled": false},
            "fibo": {"enabled": false},
            "ideogram-4": {"enabled": false, "preset": "V4_DEFAULT_20"}
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
    // the next start. The rename is what makes readers see all-or-nothing; the
    // fsync below is what makes the *contents* durable before that rename, which
    // rename alone does not promise.
    let temporary = paths.config.with_extension("json.tmp");
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary)
            .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
        file.write_all(format!("{text}\n").as_bytes())
            .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
        file.sync_all()
            .map_err(|error| format!("could not flush {}: {error}", temporary.display()))?;
    }
    // The file can hold an API key, so it is not world-readable.
    restrict(&temporary);
    std::fs::rename(&temporary, &paths.config)
        .map_err(|error| format!("could not replace {}: {error}", paths.config.display()))?;
    // And fsync the directory, so the rename itself survives a power cut.
    if let Ok(dir) = std::fs::File::open(&paths.data) {
        let _ = dir.sync_all();
    }
    Ok(())
}

fn restrict(file: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o600));
}

/// Graceful-shutdown duration declared in the configuration, used to size the
/// wait before SIGKILL.
pub fn shutdown_grace_s(paths: &Paths) -> f64 {
    read(paths)
        .ok()
        .and_then(|value| value.get("server")?.get("shutdown_grace_s")?.as_f64())
        .unwrap_or(10.0)
}

/// `storage.hf_home` from the configuration, when the user has chosen one.
///
/// A key lookup, not a schema: the type, the default and the validation all live
/// in `settings.py`, and this reads the value the server itself will read. `None`
/// means "unset", which leaves the fallback to `Paths::default_hf_home`.
pub fn hf_home(paths: &Paths) -> Option<std::path::PathBuf> {
    let value = read(paths)
        .ok()?
        .get("storage")?
        .get("hf_home")?
        .as_str()?
        .to_owned();
    if value.trim().is_empty() {
        return None;
    }
    Some(expand_tilde(&value))
}

/// `~` is expanded here because the value can be hand-edited in the JSON, and a
/// child process would otherwise be handed a literal `~` directory.
fn expand_tilde(value: &str) -> std::path::PathBuf {
    match value.strip_prefix("~/") {
        Some(rest) => match std::env::var_os("HOME") {
            Some(home) => std::path::PathBuf::from(home).join(rest),
            None => std::path::PathBuf::from(value),
        },
        None => std::path::PathBuf::from(value),
    }
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

/// Refuse a document that would switch off the model it also nominates by default.
///
/// The invariant is the generation server's — it will not start with a default
/// model that is disabled — and the Enabled switch could create exactly that
/// state with one click, from the one screen that could then no longer be
/// trusted to render. Checked here, at the write, because that is where every
/// path meets: the switch, the Configuration form, and anything added later.
///
/// A pure property of the document, deliberately: no catalogue is consulted and
/// no replacement default is chosen. Silently promoting another model to default
/// would be this process deciding something the user did not ask for, and the
/// refusal names the fix instead.
///
/// The same shape as `local_model_forget`'s refusal for the current default, and
/// for the same reason — the state is invalid whichever way it is reached.
pub fn refuse_disabled_default(value: &Value) -> Result<(), String> {
    let Some(default_model) = value.get("default_model").and_then(Value::as_str) else {
        // No default named: this document cannot break the invariant. A missing
        // key means the server's own default applies, and that model is not the
        // one being disabled here.
        return Ok(());
    };
    let disabled = value
        .get("models")
        .and_then(|models| models.get(default_model))
        .and_then(|entry| entry.get("enabled"))
        .and_then(Value::as_bool)
        .is_some_and(|enabled| !enabled);
    if disabled {
        return Err(format!(
            "\"{default_model}\" is currently the default model. Choose another default model \
             in Configuration before disabling it."
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn disabling_the_current_default_is_refused_and_names_the_fix() {
        let refused = refuse_disabled_default(&json!({
            "default_model": "z-image-turbo",
            "models": {"z-image-turbo": {"enabled": false}}
        }))
        .expect_err("must be refused");
        assert!(refused.contains("z-image-turbo"), "names the model: {refused}");
        assert!(refused.contains("Choose another default model"), "names the fix: {refused}");
    }

    #[test]
    fn disabling_any_other_model_is_allowed() {
        assert!(refuse_disabled_default(&json!({
            "default_model": "z-image-turbo",
            "models": {"z-image-turbo": {"enabled": true}, "flux2-dev": {"enabled": false}}
        }))
        .is_ok());
    }

    #[test]
    fn a_document_that_names_no_default_cannot_break_the_invariant() {
        assert!(refuse_disabled_default(&json!({"models": {"z-image": {"enabled": false}}})).is_ok());
        assert!(refuse_disabled_default(&json!({})).is_ok());
    }

    #[test]
    fn an_entry_with_no_enabled_key_is_enabled() {
        // `enabled` defaults to true on the Python side; absence is not disabled.
        assert!(refuse_disabled_default(&json!({
            "default_model": "z-image",
            "models": {"z-image": {"quantize": 4}}
        }))
        .is_ok());
    }
}
