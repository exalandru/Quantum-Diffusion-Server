//! macOS control panel for mflux-server.
//!
//! The Rust side does only three things: install the Python environment with
//! `uv`, supervise the server process, and own its configuration file.
//! Everything else — status, progress, generation — goes through the server's
//! HTTP API, which React queries directly; including `/v1/progress` over
//! Server-Sent Events, which it would be absurd to funnel across the IPC bridge.

mod bootstrap;
mod config;
mod install;
mod job;
mod paths;
mod supervisor;

use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Manager, RunEvent, State};

use bootstrap::SharedBootstrap;
use job::{JobStatus, SharedJobs};
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
    /// Where generated artifacts go when `storage.cache_dir` names nothing. Sent
    /// so the Configuration form can show the real default rather than describe
    /// it, and so the two sides derive it in one place each.
    default_cache_dir: String,
    data_dir: String,
    config_path: String,
}

#[tauri::command]
fn overview(
    app: AppHandle,
    state: State<'_, SharedSupervisor>,
    installer: State<'_, SharedBootstrap>,
) -> Result<Overview, String> {
    let paths = Paths::new(&app)?;
    let hf_home = paths.effective_hf_home();
    // `status()` reaps a self-exited child, hence the mutable guard.
    let server = state
        .try_lock()
        .map(|mut guard| guard.status())
        .unwrap_or_default();

    Ok(Overview {
        server,
        // The installer's own guard says whether a run is live; the record on disk
        // cannot, because it reads `installing` for an interrupted one too.
        bootstrap: bootstrap::status(&app, installer.is_running())?,
        hf_token_present: supervisor::hf_token_present(&hf_home),
        hf_home: hf_home.display().to_string(),
        default_cache_dir: paths.default_cache_dir().display().to_string(),
        data_dir: paths.data.display().to_string(),
        config_path: paths.config.display().to_string(),
    })
}

// The `flux2-dev` artifact used to be judged here, by `is_dir()` on a path
// literal duplicated from `flux2_dev/config.py`. Both halves were wrong: the
// converter creates that directory before downloading anything, so an empty one
// reported "artifact present"; and the default path lived in two places waiting
// to drift. Its state now arrives with the rest of the catalogue, from
// `mflux-server-fetch --status`, which is where the model rules live.

/// Install or rebuild the Python environment.
///
/// `false` means an install was already running and this call started nothing —
/// reported rather than raised, because a second click is not a failure.
#[tauri::command]
async fn bootstrap_run(
    app: AppHandle,
    installer: State<'_, SharedBootstrap>,
) -> Result<bool, String> {
    bootstrap::run(app.clone(), installer.inner()).await
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
    config::refuse_disabled_default(&value)?;
    config::write(&Paths::new(&app)?, &value)
}

#[tauri::command]
fn hf_token_write(app: AppHandle, token: String) -> Result<(), String> {
    // We write where `hf auth login` writes, rather than opening a second
    // secret store: the token would sit in plaintext right next to it anyway.
    // The token belongs beside the weights it authorises, so it follows the
    // configured storage root rather than a fixed home directory.
    let paths = Paths::new(&app)?;
    let hf_home = paths.effective_hf_home();
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

/// Ask macOS for a directory, and return the chosen path.
///
/// Rust owns the native dialog; whether the path is acceptable *configuration* is
/// `settings.py`'s call, and the server says so when the value is saved. Nothing
/// here inspects or validates the filesystem.
#[tauri::command]
async fn pick_directory(app: AppHandle, start: Option<String>) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let mut builder = app.dialog().file().set_title("Choose a storage folder");
    if let Some(start) = start.filter(|value| !value.is_empty()) {
        let path = std::path::PathBuf::from(start);
        // Opening on a directory that is not there (an unmounted volume, say)
        // would leave the panel somewhere arbitrary.
        if path.is_dir() {
            builder = builder.set_directory(path);
        }
    }
    let (sender, receiver) = tokio::sync::oneshot::channel();
    builder.pick_folder(move |chosen| {
        let _ = sender.send(chosen.map(|path| path.to_string()));
    });
    receiver
        .await
        .map_err(|error| format!("the folder chooser closed unexpectedly: {error}"))
}

#[tauri::command]
async fn models_status(app: AppHandle) -> Result<Value, String> {
    // Unchanged, and deliberately so: reading the catalogue goes through the
    // `env/` console script, not the generation server, so it keeps working with
    // the server stopped — which is when you most want to download weights.
    supervisor::models_status(&Paths::new(&app)?).await
}

/// Start a download. Returns as soon as the child is running; the outcome is
/// read back through `job_status`.
#[tauri::command]
async fn model_fetch(app: AppHandle, jobs: State<'_, SharedJobs>, key: String) -> Result<(), String> {
    let paths = Paths::new(&app)?;
    supervisor::start_fetch(app.clone(), &jobs, &paths, key).await
}

/// Start the FLUX.2-dev conversion. Same contract as `model_fetch`.
#[tauri::command]
async fn prequantize_run(
    app: AppHandle,
    jobs: State<'_, SharedJobs>,
    model: String,
    bits: u8,
    components: Vec<String>,
    dest: Option<String>,
) -> Result<(), String> {
    // Rust forwards the choice; whether it is a *valid* choice is the capability
    // contract's call, and the server enforces it.
    let paths = Paths::new(&app)?;
    supervisor::start_prequantize(app.clone(), &jobs, &paths, model, bits, components, dest).await
}

/// Identify a directory, without registering it. Advisory only.
#[tauri::command]
async fn local_model_inspect(app: AppHandle, path: String) -> Result<Value, String> {
    supervisor::run_import(&Paths::new(&app)?, &["inspect".into(), path]).await
}

/// Check a directory against one built-in catalogue entry. Binds nothing: the
/// override is written through the configuration, like every other per-model
/// setting.
#[tauri::command]
async fn local_model_locate(app: AppHandle, path: String, model: String) -> Result<Value, String> {
    supervisor::run_import(
        &Paths::new(&app)?,
        &["locate".to_owned(), path, "--model".to_owned(), model],
    )
    .await
}

/// Register a directory as a model. Python revalidates before persisting, so a
/// stale inspection result cannot be turned into a registration.
#[tauri::command]
async fn local_model_register(
    app: AppHandle,
    path: String,
    base_profile: String,
    name: Option<String>,
    api_name: Option<String>,
) -> Result<Value, String> {
    let mut args = vec![
        "register".to_owned(),
        path,
        "--base-profile".to_owned(),
        base_profile,
    ];
    if let Some(name) = name.filter(|value| !value.trim().is_empty()) {
        args.push("--name".to_owned());
        args.push(name);
    }
    if let Some(api_name) = api_name.filter(|value| !value.trim().is_empty()) {
        args.push("--api-name".to_owned());
        args.push(api_name);
    }
    supervisor::run_import(&Paths::new(&app)?, &args).await
}

/// Remove a registration. The model's files are never touched.
#[tauri::command]
async fn local_model_forget(app: AppHandle, id: String) -> Result<Value, String> {
    supervisor::run_import(&Paths::new(&app)?, &["forget".into(), id]).await
}

/// Current long-operation state. Rust is the authority here; React presents it.
#[tauri::command]
async fn job_status(jobs: State<'_, SharedJobs>) -> Result<JobStatus, String> {
    Ok(jobs.lock().await.status())
}

/// SIGTERM the active operation's process group, then SIGKILL after a grace
/// period if it is still there.
#[tauri::command]
async fn job_cancel(jobs: State<'_, SharedJobs>) -> Result<JobStatus, String> {
    // The pid comes back from the same critical section that sent the SIGTERM, so
    // a job started afterwards can never inherit the escalation.
    let (status, pid) = jobs.lock().await.request_cancel()?;
    job::arm_kill_after_grace(jobs.inner().clone(), pid);
    Ok(status)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let supervisor: SharedSupervisor = Default::default();
    let jobs: SharedJobs = Default::default();
    let installer: SharedBootstrap = Default::default();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(supervisor)
        .manage(jobs)
        .manage(installer)
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
            job_status,
            job_cancel,
            pick_directory,
            local_model_inspect,
            local_model_locate,
            local_model_register,
            local_model_forget,
        ])
        .build(tauri::generate_context!())
        .expect("could not initialize Tauri")
        .run(|app, event| {
            // Quitting the app must never leave a child behind: the server may be
            // holding tens of GB of weights, and a conversion runs for hours at a
            // ~66 GB peak. Both are their own process group, so one signal each
            // takes the whole subtree.
            if let RunEvent::Exit = event {
                if let Some(state) = app.try_state::<SharedSupervisor>() {
                    if let Ok(mut guard) = state.try_lock() {
                        guard.kill_now();
                    }
                }
                if let Some(state) = app.try_state::<SharedJobs>() {
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
