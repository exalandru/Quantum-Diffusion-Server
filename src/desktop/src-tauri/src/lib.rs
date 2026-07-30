//! macOS control panel for mflux-server.
//!
//! The Rust side does only three things: install the Python environment with
//! `uv`, supervise the server process, and own its configuration file.
//! Everything else — status, progress, generation — goes through the server's
//! HTTP API, which React queries directly; including `/v1/progress` over
//! Server-Sent Events, which it would be absurd to funnel across the IPC bridge.

mod bootstrap;
mod config;
mod paths;
mod supervisor;

use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Manager, RunEvent, State};

use paths::Paths;
use supervisor::{SharedSupervisor, ServerStatus, Supervisor};

/// Overview rendered by the dashboard.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Overview {
    server: ServerStatus,
    bootstrap: bootstrap::BootstrapStatus,
    /// The `black-forest-labs/*` repos are *gated*: with no token, the first
    /// download fails with a 401.
    hf_token_present: bool,
    hf_home: String,
    data_dir: String,
    config_path: String,
    /// Whether `flux2-dev`'s pre-quantized artifact is present.
    flux2_dev_ready: bool,
}

#[tauri::command]
fn overview(app: AppHandle, state: State<'_, SharedSupervisor>) -> Result<Overview, String> {
    let paths = Paths::new(&app)?;
    let hf_home = Paths::default_hf_home();
    // `status()` reaps a self-exited child, hence the mutable guard.
    let server = state
        .try_lock()
        .map(|mut guard| guard.status())
        .unwrap_or_default();

    Ok(Overview {
        server,
        bootstrap: bootstrap::status(&app)?,
        hf_token_present: supervisor::hf_token_present(&hf_home),
        hf_home: hf_home.display().to_string(),
        data_dir: paths.data.display().to_string(),
        config_path: paths.config.display().to_string(),
        flux2_dev_ready: flux2_dev_artifact(&paths).is_some(),
    })
}

/// Path of the `flux2-dev` artifact when it exists, honouring a `model_path`
/// override from the configuration.
fn flux2_dev_artifact(paths: &Paths) -> Option<std::path::PathBuf> {
    let configured = config::read(paths)
        .ok()
        .and_then(|value| {
            value
                .get("models")?
                .get("flux2-dev")?
                .get("model_path")?
                .as_str()
                .map(str::to_owned)
        })
        .filter(|value| !value.is_empty());

    let raw = configured.unwrap_or_else(|| "~/.cache/mflux-server/flux2-dev-mlx-8bit".to_owned());
    let expanded = match raw.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(rest))?,
        None => std::path::PathBuf::from(raw),
    };
    expanded.is_dir().then_some(expanded)
}

#[tauri::command]
async fn bootstrap_run(app: AppHandle) -> Result<(), String> {
    bootstrap::run(app).await
}

#[tauri::command]
async fn server_start(app: AppHandle, state: State<'_, SharedSupervisor>) -> Result<u16, String> {
    let paths = Paths::new(&app)?;
    let port = {
        let mut guard = state.lock().await;
        guard.start(&app, &paths).await?
    };
    // The server loads no weights at startup: it listens within a second.
    supervisor::wait_healthy(port, Duration::from_secs(30)).await?;
    Ok(port)
}

#[tauri::command]
async fn server_stop(app: AppHandle, state: State<'_, SharedSupervisor>) -> Result<(), String> {
    let paths = Paths::new(&app)?;
    let grace = Duration::from_secs_f64(config::shutdown_grace_s(&paths).max(1.0));
    let mut guard = state.lock().await;
    guard.stop(grace).await
}

#[tauri::command]
async fn server_restart(app: AppHandle, state: State<'_, SharedSupervisor>) -> Result<u16, String> {
    server_stop(app.clone(), state.clone()).await?;
    server_start(app, state).await
}

#[tauri::command]
fn config_read(app: AppHandle) -> Result<Value, String> {
    config::read(&Paths::new(&app)?)
}

#[tauri::command]
fn config_write(app: AppHandle, value: Value) -> Result<(), String> {
    config::write(&Paths::new(&app)?, &value)
}

#[tauri::command]
fn hf_token_write(app: AppHandle, token: String) -> Result<(), String> {
    // We write where `hf auth login` writes, rather than opening a second
    // secret store: the token would sit in plaintext right next to it anyway.
    let _ = app;
    let hf_home = Paths::default_hf_home();
    std::fs::create_dir_all(&hf_home)
        .map_err(|error| format!("could not create {}: {error}", hf_home.display()))?;
    let file = Paths::hf_token_file(&hf_home);
    std::fs::write(&file, token.trim())
        .map_err(|error| format!("could not write {}: {error}", file.display()))?;
    restrict_permissions(&file);
    Ok(())
}

/// 0600 on the token file, the way the HuggingFace CLI does.
fn restrict_permissions(file: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o600));
}

#[tauri::command]
async fn models_status(app: AppHandle) -> Result<Value, String> {
    supervisor::models_status(&Paths::new(&app)?).await
}

#[tauri::command]
async fn model_fetch(app: AppHandle, key: String) -> Result<(), String> {
    let paths = Paths::new(&app)?;
    supervisor::run_fetch(app.clone(), &paths, key).await
}

#[tauri::command]
async fn prequantize_run(
    app: AppHandle,
    components: Vec<String>,
    dest: Option<String>,
) -> Result<(), String> {
    let paths = Paths::new(&app)?;
    supervisor::run_prequantize(app.clone(), &paths, components, dest).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let supervisor: SharedSupervisor = Default::default();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(supervisor)
        .invoke_handler(tauri::generate_handler![
            overview,
            bootstrap_run,
            server_start,
            server_stop,
            server_restart,
            config_read,
            config_write,
            hf_token_write,
            prequantize_run,
            models_status,
            model_fetch,
        ])
        .build(tauri::generate_context!())
        .expect("could not initialize Tauri")
        .run(|app, event| {
            // Quitting the app must never leave a server behind: it may be
            // holding tens of GB of weights in memory.
            if let RunEvent::Exit = event {
                if let Some(state) = app.try_state::<SharedSupervisor>() {
                    if let Ok(mut guard) = state.try_lock() {
                        guard.kill_now();
                    }
                }
            }
        });
}

// Makes `Supervisor` usable as shared Tauri state.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Supervisor>();
};
