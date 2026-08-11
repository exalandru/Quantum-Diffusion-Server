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

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

use crate::install::{self, BootstrapState, InstallRecord, InstallState};
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
    /// `true` only for [`BootstrapState::Ready`]. Kept because it is the gate the
    /// whole app hangs on, and one boolean is the right shape for one question.
    pub ready: bool,
    /// What the Setup screen must offer. `ready` is derived from this, never the
    /// other way round.
    pub state: BootstrapState,
    /// App version that produced the environment currently present. Diagnostic.
    pub installed_version: Option<String>,
    /// Version of the app currently running.
    pub app_version: String,
    pub env_path: String,
    /// Why the last install stopped, when it stopped. `None` otherwise.
    pub failure: Option<String>,
}

/// Single-flight for the installer.
///
/// React disabling its button is not a guard: it is one window's opinion, it is
/// lost on a reload, and it says nothing about a replayed IPC call. Two
/// overlapping runs would have one `remove_dir_all(server/)` racing the other's
/// `uv sync --project server`, on a directory neither owns exclusively.
///
/// Two mechanisms, because the harm does not care which process caused it. The
/// in-memory flag answers within this app; an advisory lock on a file in the data
/// directory answers across processes, which matters because a second copy of the
/// bundle can be launched (`open -n`, or a dev build beside the packaged one) and
/// would otherwise arrive with a guard of its own that has never heard of the
/// first. `flock` is released by the kernel when the holder dies, so it cannot go
/// stale the way a pid file would.
///
/// A lease rather than a flag the caller must clear: the release is a `Drop`, so
/// it happens on the error path and on an unwind as well as on success.
#[derive(Debug, Default)]
pub struct BootstrapGuard(Mutex<bool>);

pub type SharedBootstrap = Arc<BootstrapGuard>;

pub struct Lease {
    guard: SharedBootstrap,
    /// Held open for the lease's lifetime: closing the descriptor is what
    /// releases the lock.
    _lock: Option<std::fs::File>,
}

impl BootstrapGuard {
    /// A lease, or `None` when an install is already running anywhere.
    pub fn try_begin(self: &SharedBootstrap, lock_path: &Path) -> Option<Lease> {
        // `unwrap`: the only code under this mutex is a bool assignment, so it
        // cannot panic and the mutex cannot be poisoned.
        let mut running = self.0.lock().unwrap();
        if *running {
            return None;
        }
        let held = match take_lock(lock_path) {
            LockOutcome::HeldElsewhere => return None,
            LockOutcome::Acquired(file) => Some(file),
            // No lock file could be opened at all — a read-only or missing data
            // directory. Refusing to install for that reason would trade a rare
            // race for a certain dead end, and the in-process flag still holds.
            LockOutcome::Unavailable => None,
        };
        *running = true;
        Some(Lease {
            guard: self.clone(),
            _lock: held,
        })
    }

    pub fn is_running(&self) -> bool {
        *self.0.lock().unwrap()
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        *self.guard.0.lock().unwrap() = false;
    }
}

enum LockOutcome {
    Acquired(std::fs::File),
    HeldElsewhere,
    Unavailable,
}

/// Non-blocking exclusive `flock`, so "someone else is installing" is an answer
/// rather than a wait.
fn take_lock(path: &Path) -> LockOutcome {
    use std::os::unix::io::AsRawFd;

    let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)
    else {
        return LockOutcome::Unavailable;
    };
    // SAFETY: `file` owns the descriptor and outlives the call.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return LockOutcome::Acquired(file);
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::EWOULDBLOCK) => LockOutcome::HeldElsewhere,
        _ => LockOutcome::Unavailable,
    }
}

/// Where the app stands, decided from the install record rather than a version.
///
/// `running` comes from the guard because a live run is process state: the record
/// on disk says `installing` both while an install is in flight and after one was
/// interrupted, and only the guard tells those apart.
pub fn status(app: &AppHandle, running: bool) -> Result<BootstrapStatus, String> {
    let paths = Paths::new(app)?;
    let app_version = app.package_info().version.to_string();
    let env_ready = paths.env_ready();

    let record = install::read_record(&paths.install_record).or_else(|| {
        if running {
            // An install writes the record as its first act; adopting an old one
            // underneath it would race that write for no benefit.
            return None;
        }
        let adopted = install::migrate_legacy(&paths.server, &paths.stamp, env_ready)?;
        // Best effort: failing to persist the adoption costs a recomputation on
        // the next poll, never a rebuild.
        let _ = install::write_record(&paths.install_record, &adopted);
        Some(adopted)
    });

    let bundled = bundled_project(app)
        .ok()
        .and_then(|source| install::fingerprint(&source).ok());

    let state = install::decide(running, record.as_ref(), env_ready, bundled.as_deref());
    Ok(BootstrapStatus {
        ready: state == BootstrapState::Ready,
        state,
        installed_version: record.as_ref().map(|record| record.app_version.clone()),
        app_version,
        env_path: paths.env.display().to_string(),
        failure: record.as_ref().and_then(|record| record.error.clone()),
    })
}

/// Absolute path of the Python project inside the bundle.
///
/// `tauri.conf.json` maps `build/desktop/staging/resources/server/` **to** `server/`, so
/// `server` — the destination — is what `BaseDirectory::Resource` must be joined with. It
/// used to be an array of globs under `resources/`, where source and destination coincided;
/// when that became a map the Rust side kept resolving `resources/server`, i.e.
/// `Contents/Resources/resources/server`, which does not exist. Tauri's `resolve` only joins
/// and never checks (`tauri::path::resolve_path`), so nothing objected: `copy_project` failed
/// on every packaged install with "could not read …/resources/server", and `bundle_is_newer`
/// answered `false` forever, silently disabling change detection.
fn bundled_project(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .resolve("server", tauri::path::BaseDirectory::Resource)
        .map_err(|error| format!("resources not found in the bundle: {error}"))
}

fn emit(app: &AppHandle, event: BootstrapEvent) {
    // A missing interface must not make the installation fail.
    let _ = app.emit("bootstrap", event);
}

/// Re-copy the Python project and (re)build the environment.
///
/// Returns `false` when an install was already running and this call did nothing.
/// It is not an error: two clicks on one button are a user being impatient, and
/// the honest answer is the state, not a failure.
pub async fn run(app: AppHandle, guard: &SharedBootstrap) -> Result<bool, String> {
    let paths = Paths::new(&app)?;
    // Before the lease: the lock lives in the data directory, which may not exist
    // yet on the very first launch.
    paths.ensure()?;
    let Some(lease) = guard.try_begin(&paths.install_lock) else {
        return Ok(false);
    };
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
    drop(lease);
    result.map(|()| true)
}

async fn install(app: &AppHandle, paths: &Paths, app_version: &str) -> Result<(), String> {
    // Everything that can fail without consequence happens first, while the
    // installed copy is still untouched: resolving the payload, proving it is
    // readable, and reading its identity.
    let source = bundled_project(app)?;
    std::fs::read_dir(&source)
        .map_err(|error| format!("bundled project unreadable at {}: {error}", source.display()))?;
    let fingerprint = install::fingerprint(&source)?;
    let python_version = std::fs::read_to_string(source.join(".python-version"))
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());

    // From here on the install is destructive, so it says so on disk *first*. A
    // crash, a force-quit or a power cut between this line and the `ready` below
    // leaves `installing` behind — which the next launch reads as "repair", not
    // as "never installed" and never as "ready".
    let record = InstallRecord::new(
        InstallState::Installing,
        app_version,
        python_version.clone(),
        &fingerprint,
    );
    install::write_record(&paths.install_record, &record)?;

    let outcome = build(app, paths, &source).await;
    if let Err(message) = &outcome {
        let _ = install::write_record(
            &paths.install_record,
            &InstallRecord::new(
                InstallState::Failed,
                app_version,
                python_version,
                &fingerprint,
            )
            .with_error(message),
        );
        return outcome;
    }

    install::write_record(
        &paths.install_record,
        &InstallRecord::new(InstallState::Ready, app_version, python_version, &fingerprint),
    )?;

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

/// The destructive half: replace the project, then rebuild the environment.
async fn build(app: &AppHandle, paths: &Paths, source: &Path) -> Result<(), String> {
    // Unconditionally: the copy is a handful of files, and a version check would
    // skip it whenever only the *content* had changed — the common case during
    // development, and exactly what the fingerprint now detects.
    emit(
        app,
        BootstrapEvent::Step {
            message: "Copying the Python project…".into(),
        },
    );
    replace_project(source, &paths.server, &paths.staging)?;

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

/// Put the bundle's `Resources/server/` in place as `<appdata>/server/`.
///
/// Staged, and the staging is not decoration. Resolving a path is not evidence
/// that it exists, and with the readability check *after* the removal a wrong
/// resource path deleted a working install and only then failed — no project, no
/// marker, and every retry repeating the destruction. Copying into `staging`
/// first extends that guarantee from "the source was missing" to "the copy failed
/// halfway", which the previous shape could not survive either: the installed
/// project is removed only once a complete replacement exists beside it.
///
/// The swap is remove-then-rename rather than a single atomic operation. That
/// window is real and it is bounded by a `rename` within one directory; the
/// install record covers it, because a crash inside it leaves `installing` on
/// disk and the next launch offers a repair.
fn replace_project(source: &Path, server: &Path, staging: &Path) -> Result<(), String> {
    // `read_dir` is exactly what `copy_tree` will do first, so a source that
    // cannot be listed is refused before anything is touched.
    std::fs::read_dir(source)
        .map_err(|error| format!("bundled project unreadable at {}: {error}", source.display()))?;

    // A leftover from an interrupted run would otherwise merge into this one.
    if staging.exists() {
        std::fs::remove_dir_all(staging)
            .map_err(|error| format!("could not clean up {}: {error}", staging.display()))?;
    }
    copy_tree(source, staging)?;

    // We replace wholesale: a leftover from an earlier version would install
    // stale code.
    if server.exists() {
        std::fs::remove_dir_all(server)
            .map_err(|error| format!("could not clean up {}: {error}", server.display()))?;
    }
    std::fs::rename(staging, server).map_err(|error| {
        format!(
            "could not move {} into place at {}: {error}",
            staging.display(),
            server.display()
        )
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("qds-bootstrap-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn two_requests_produce_one_installation() {
        // The guard React cannot provide: a reloaded page or a replayed IPC call
        // reaches this without ever seeing the first one's disabled button.
        let root = temp("guard");
        let lock = root.join("bootstrap.lock");
        let guard: SharedBootstrap = Default::default();

        let first = guard.try_begin(&lock).expect("the first request installs");
        assert!(guard.is_running(), "an install in flight has to be observable");
        assert!(
            guard.try_begin(&lock).is_none(),
            "the second request must not start a second uv sync"
        );

        // And the lease is released by scope, so a failed install does not wedge
        // the installer permanently.
        drop(first);
        assert!(!guard.is_running());
        assert!(guard.try_begin(&lock).is_some(), "a later request installs normally");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_second_process_is_refused_too() {
        // `flock` binds to the open file description, not to the process, so a
        // second `open` here conflicts exactly as another app instance would.
        // Without this the guard would only be as authoritative as "one copy of
        // the bundle is running", which nothing enforces.
        let root = temp("guard-cross");
        let lock = root.join("bootstrap.lock");
        let guard: SharedBootstrap = Default::default();
        let lease = guard.try_begin(&lock).expect("the first request installs");

        assert!(
            matches!(take_lock(&lock), LockOutcome::HeldElsewhere),
            "another process must see the lock held"
        );
        drop(lease);
        assert!(
            matches!(take_lock(&lock), LockOutcome::Acquired(_)),
            "and get it once the holder is done"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn an_unreadable_source_leaves_the_installed_project_untouched() {
        // R0's destructive half, pinned: the resource path was wrong for two
        // releases and the removal happened first, so pressing Rebuild on a
        // working machine destroyed the install and only then reported why.
        let root = temp("r0");
        let server = root.join("server");
        std::fs::create_dir_all(server.join("mflux_server")).unwrap();
        std::fs::write(server.join("mflux_server/app.py"), "installed and working\n").unwrap();
        std::fs::write(server.join(".python-version"), "3.12\n").unwrap();

        let missing = root.join("Resources/resources/server");
        let error = replace_project(&missing, &server, &root.join("server.staging"))
            .expect_err("a missing payload cannot install");
        assert!(error.contains("resources/server"), "the message names the path: {error}");

        assert_eq!(
            std::fs::read_to_string(server.join("mflux_server/app.py")).unwrap(),
            "installed and working\n"
        );
        assert!(server.join(".python-version").is_file());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_successful_replacement_installs_the_payload_and_leaves_no_staging() {
        let root = temp("replace");
        let source = root.join("bundle");
        std::fs::create_dir_all(source.join("mflux_server")).unwrap();
        std::fs::write(source.join("mflux_server/app.py"), "new code\n").unwrap();
        std::fs::write(source.join("uv.lock"), "version = 1\n").unwrap();

        let server = root.join("server");
        std::fs::create_dir_all(&server).unwrap();
        std::fs::write(server.join("stale.py"), "old code\n").unwrap();

        let staging = root.join("server.staging");
        // A leftover from an interrupted run must not merge into this one.
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("leftover.py"), "half a copy\n").unwrap();

        replace_project(&source, &server, &staging).unwrap();

        assert_eq!(
            std::fs::read_to_string(server.join("mflux_server/app.py")).unwrap(),
            "new code\n"
        );
        assert!(!server.join("stale.py").exists(), "wholesale replacement");
        assert!(!server.join("leftover.py").exists(), "the stale staging was discarded");
        assert!(!staging.exists(), "staging is consumed by the swap");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_completion_record_lives_outside_the_directory_the_installer_replaces() {
        // The reason `.bootstrap-version` had to move: it sat inside `server/`,
        // which `replace_project` removes, so an interrupted install erased the
        // only evidence that a machine had ever been installed.
        let root = temp("outside");
        let server = root.join("server");
        let record = root.join("bootstrap.json");
        std::fs::create_dir_all(&server).unwrap();
        install::write_record(
            &record,
            &InstallRecord::new(InstallState::Ready, "0.2.0", None, "sha256:a"),
        )
        .unwrap();

        assert!(
            !record.starts_with(&server),
            "the record must not be inside the replaced directory"
        );
        let source = root.join("bundle");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("uv.lock"), "version = 1\n").unwrap();
        replace_project(&source, &server, &root.join("server.staging")).unwrap();

        assert!(install::read_record(&record).is_some(), "it survived the replacement");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
