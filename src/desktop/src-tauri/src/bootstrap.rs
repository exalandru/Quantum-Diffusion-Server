//! Installing the Python environment on first launch.
//!
//! We do not ship the 1.1 GB of dependencies inside the bundle: torch alone is
//! 501 MB, mlx adds 178 more including a 150 MB `mlx.metallib` of Metal shaders,
//! and the whole tree holds 186 native binaries. `uv`, by contrast, is a single
//! 50 MB binary that can download CPython and rebuild the environment from
//! `uv.lock` — 86 packages, identically.
//!
//! Two precautions:
//!
//! * we copy the project out of the bundle into the data directory before running
//!   `uv sync`, because the bundle is read-only;
//! * `--no-editable`, without which the installed package would point back at
//!   that copy instead of being genuinely installed.

use std::path::Path;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

use crate::paths::Paths;

/// Bootstrap progress event, pushed to the interface.
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
    /// `true` when the environment is ready and up to date.
    pub ready: bool,
    /// App version that produced the environment currently present.
    pub installed_version: Option<String>,
    /// Version of the app currently running.
    pub app_version: String,
    pub env_path: String,
}

pub fn status(app: &AppHandle) -> Result<BootstrapStatus, String> {
    let paths = Paths::new(app)?;
    let app_version = app.package_info().version.to_string();
    let installed_version = paths.stamped_version();
    let up_to_date = installed_version.as_deref() == Some(app_version.as_str())
        && !bundle_is_newer(app, &paths);
    Ok(BootstrapStatus {
        ready: paths.env_ready() && up_to_date,
        installed_version,
        app_version,
        env_path: paths.env.display().to_string(),
    })
}

/// `true` when the bundled Python project is more recent than the installed one.
///
/// Comparing versions alone is not enough, and the gap is silent: a change to the
/// Python code with the app version left alone leaves `ready = true`, so the
/// bundle never gets copied, `uv sync` never re-runs, and the app keeps serving
/// whatever code the *first* install captured. Measured the hard way — new model
/// defaults shipped in the bundle while the server still answered with the old
/// ones, with nothing anywhere saying why.
///
/// The bundle's files carry their build time, so a mtime comparison against the
/// stamp catches every rebuild without hashing the tree. Unreadable timestamps
/// mean "no evidence of a change": we do not force a reinstall on a doubt.
fn bundle_is_newer(app: &AppHandle, paths: &Paths) -> bool {
    let Ok(stamp) = std::fs::metadata(&paths.stamp).and_then(|meta| meta.modified()) else {
        return false;
    };
    let Ok(source) = app
        .path()
        .resolve("resources/server", tauri::path::BaseDirectory::Resource)
    else {
        return false;
    };
    newest_mtime(&source).is_some_and(|bundled| bundled > stamp)
}

fn newest_mtime(path: &Path) -> Option<std::time::SystemTime> {
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_dir() {
        return metadata.modified().ok();
    }
    std::fs::read_dir(path)
        .ok()?
        .filter_map(|entry| newest_mtime(&entry.ok()?.path()))
        .max()
}

fn emit(app: &AppHandle, event: BootstrapEvent) {
    // A missing interface must not make the installation fail.
    let _ = app.emit("bootstrap", event);
}

/// Re-copy the Python project and (re)build the environment.
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
    // Unconditionally: the copy is a handful of files, and the previous version
    // check would skip it whenever only the *content* had changed — which is the
    // common case during development and exactly what `bundle_is_newer` detects.
    emit(
        app,
        BootstrapEvent::Step {
            message: "Copying the Python project…".into(),
        },
    );
    copy_project(app, paths)?;

    emit(
        app,
        BootstrapEvent::Step {
            message: "Installing Python and dependencies (about 1.1 GB)…".into(),
        },
    );
    sync(app, paths).await?;

    if !paths.env_ready() {
        return Err(format!(
            "uv exited without error but {} is missing.",
            paths.server_bin().display()
        ));
    }

    std::fs::write(&paths.stamp, app_version)
        .map_err(|error| format!("could not write {}: {error}", paths.stamp.display()))?;

    // The download cache weighs 1.6 GB after a first sync, 1.5 of which are
    // wheels that are already installed. uv links the environment to the cache
    // with hard links: removing a cache entry only drops one link, the files
    // survive through site-packages — verified, the environment stays importable.
    emit(
        app,
        BootstrapEvent::Step {
            message: "Cleaning up the download cache…".into(),
        },
    );
    prune_cache(app, paths).await;
    Ok(())
}

/// Reclaim the uv cache's space. Failure is not fatal: it is only disk, and the
/// environment already works at this point.
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

/// Copy `resources/server/` out of the bundle into `<appdata>/server/`.
fn copy_project(app: &AppHandle, paths: &Paths) -> Result<(), String> {
    let source = app
        .path()
        .resolve("resources/server", tauri::path::BaseDirectory::Resource)
        .map_err(|error| format!("resources not found in the bundle: {error}"))?;

    // We replace wholesale: a leftover from an earlier version would install
    // stale code.
    if paths.server.exists() {
        std::fs::remove_dir_all(&paths.server)
            .map_err(|error| format!("could not clean up {}: {error}", paths.server.display()))?;
    }
    copy_tree(&source, &paths.server)
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    std::fs::create_dir_all(destination)
        .map_err(|error| format!("could not create {}: {error}", destination.display()))?;
    let entries = std::fs::read_dir(source)
        .map_err(|error| format!("could not read {}: {error}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("read interrupted: {error}"))?;
        let target = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .map_err(|error| format!("unknown file type: {error}"))?;
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)
                .map_err(|error| format!("copying {} failed: {error}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// Version pinned by the repo, read from the project copy.
///
/// We hand it to uv explicitly rather than relying on its automatic discovery:
/// that depends on the current directory, which is unpredictable for a sidecar
/// launched from a `.app`. Without this, uv picks the newest interpreter
/// satisfying `requires-python` — measured: 3.13 instead of 3.12 — and
/// `uv.lock`, whose markers tell the two apart, resolves a different package set
/// from the one that was tested.
fn pinned_python(paths: &Paths) -> Result<String, String> {
    let file = paths.server.join(".python-version");
    let raw = std::fs::read_to_string(&file)
        .map_err(|error| format!("{} is unreadable: {error}", file.display()))?;
    let version = raw.trim().to_owned();
    if version.is_empty() {
        return Err(format!("{} is empty", file.display()));
    }
    Ok(version)
}

/// Run `uv sync` as a sidecar and relay its output to the interface.
async fn sync(app: &AppHandle, paths: &Paths) -> Result<(), String> {
    let python = pinned_python(paths)?;
    let command = app
        .shell()
        .sidecar("uv")
        .map_err(|error| format!("uv sidecar not found: {error}"))?
        .args([
            "sync",
            "--frozen",
            // Neither pytest nor ruff: they only serve development.
            "--no-dev",
            // Without this, the installed package would point at the project copy.
            "--no-editable",
            // Guarantees that no system Python is required.
            "--managed-python",
            "--python",
        ])
        .arg(&python)
        .args(["--project"])
        .arg(paths.server.as_os_str())
        // Every location is explicit: nothing should land in the project
        // directory nor in the user's caches.
        .env("UV_PROJECT_ENVIRONMENT", &paths.env)
        .env("UV_PYTHON_INSTALL_DIR", &paths.python)
        .env("UV_CACHE_DIR", &paths.uv_cache)
        // Ignore any ~/.config/uv that would change the resolution.
        .env("UV_NO_CONFIG", "1");

    let (mut events, _child) = command
        .spawn()
        .map_err(|error| format!("could not launch uv: {error}"))?;

    let mut failure: Option<String> = None;
    while let Some(event) = events.recv().await {
        match event {
            // uv writes its progress to stderr.
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
                        "uv sync failed (code {:?}, signal {:?})",
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
