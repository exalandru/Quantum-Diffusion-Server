//! Lifecycle of the Python server.
//!
//! Three constraints shape this file.
//!
//! **`mflux-server` takes no arguments**: everything goes through the environment
//! and a JSON file. It does not report the port it opened either, so this is
//! where we pick a free one and impose it.
//!
//! **The write paths must be absolute**, with their parent directories already
//! created: the server creates its image directory and its log file during
//! initialization, before it listens, and a `.app` launched by Finder inherits
//! `/` as its current directory.
//!
//! **Shutdown must be bounded.** `SIGTERM` triggers uvicorn's graceful shutdown,
//! bounded on the server side by `shutdown_grace_s`; measured at roughly 10s
//! mid-generation. But a second `SIGTERM` forces nothing — only `SIGINT` does, on
//! uvicorn's side — hence the SIGTERM-then-SIGKILL ladder. The signal goes to the
//! process *group*, otherwise quitting the app would orphan the server: macOS
//! does not reap grandchildren.

use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::job::{self, JobKind, SharedJobs};
use crate::paths::Paths;

/// Margin added to `shutdown_grace_s` before escalating to SIGKILL.
const KILL_AFTER: Duration = Duration::from_secs(8);

/// One line of server output, relayed to the interface.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerLine {
    /// `true` for stdout, which carries only JSON Lines in `log_json` mode.
    pub structured: bool,
    pub line: String,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerStatus {
    pub running: bool,
    pub port: Option<u16>,
    /// Set when the process stopped on its own.
    pub last_exit: Option<String>,
}

#[derive(Default)]
pub struct Supervisor {
    child: Option<Child>,
    port: Option<u16>,
    last_exit: Option<String>,
}

pub type SharedSupervisor = Arc<Mutex<Supervisor>>;

impl Supervisor {
    /// Current status, reaping the child if it exited on its own.
    ///
    /// `&mut self` because of that reaping, and it is not optional: without it
    /// `child` is never released when the server dies by itself, so the dashboard
    /// keeps reporting a running process, `last_exit` stays empty, and `start()`
    /// hands back the remembered port through its early return — leaving Start
    /// unable to relaunch anything until the user hits Stop. A fixed port makes
    /// self-exits likely (a port collision is one), so this has to be honest.
    ///
    /// `try_wait()` is non-blocking, so reaping lazily on each status poll costs
    /// nothing and needs neither a watcher task nor a channel.
    pub fn status(&mut self) -> ServerStatus {
        self.reap();
        ServerStatus {
            running: self.child.is_some(),
            port: self.port,
            last_exit: self.last_exit.clone(),
        }
    }

    /// Notice a process that exited without us asking, and record why.
    fn reap(&mut self) {
        let Some(child) = self.child.as_mut() else {
            return;
        };
        // A `None` means still running and an `Err` means the status is
        // unreadable; both leave the state alone.
        if let Ok(Some(status)) = child.try_wait() {
            self.last_exit = Some(match status.code() {
                Some(0) => "the server exited normally".to_owned(),
                Some(code) => format!("the server exited with code {code}"),
                // No code means a signal; the reason is usually in the logs.
                None => "the server was terminated by a signal".to_owned(),
            });
            self.child = None;
            self.port = None;
        }
    }

    /// Start the server, or return the current port if it is already running.
    pub async fn start(&mut self, app: &AppHandle, paths: &Paths) -> Result<u16, String> {
        // Reap first: without this the early return below would hand back the
        // port of a process that already died.
        self.reap();
        if let Some(port) = self.port {
            if self.child.is_some() {
                return Ok(port);
            }
        }
        if !paths.env_ready() {
            return Err(format!(
                "Python environment missing ({}). Run the installation first.",
                paths.server_bin().display()
            ));
        }
        paths.ensure()?;
        crate::config::ensure_exists(paths)?;

        // `ensure_exists` just ran, so the file is on disk and readable here.
        let port = crate::config::port(paths);
        ensure_port_free(port)?;
        let hf_home = paths.effective_hf_home();
        let mut command = Command::new(paths.server_bin());
        command
            // The current directory is set explicitly: we never rely on the
            // inherited one, which is `/` when launched from Finder.
            .current_dir(&paths.data)
            .env("MFLUX_SERVER_CONFIG", &paths.config)
            .env("MFLUX_SERVER_HOST", "127.0.0.1")
            .env("MFLUX_SERVER_PORT", port.to_string())
            .env("MFLUX_SERVER_IMAGE_STORE", &paths.images)
            // Empty string = no log file: we are the ones capturing.
            .env("MFLUX_SERVER_LOG_FILE", "")
            // stdout becomes a JSON Lines stream, stderr keeps tqdm.
            .env("MFLUX_SERVER_LOG_JSON", "1")
            .env("HF_HOME", &hf_home)
            // Cuts down the progress-bar noise on stderr.
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            // Without this, a redirected stdout would be block-buffered and
            // progress would arrive in bursts.
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            // Its own process group: lets us signal the whole subtree and keeps
            // the server from being orphaned when the app quits.
            .process_group(0);

        let mut child = command
            .spawn()
            .map_err(|error| format!("could not launch the server: {error}"))?;

        if let Some(stdout) = child.stdout.take() {
            pump(app.clone(), stdout, true, None);
        }
        if let Some(stderr) = child.stderr.take() {
            pump(app.clone(), stderr, false, None);
        }

        self.child = Some(child);
        self.port = Some(port);
        self.last_exit = None;
        Ok(port)
    }

    /// Stop the server: SIGTERM to the group, a bounded wait, then SIGKILL.
    pub async fn stop(&mut self, grace: Duration) -> Result<(), String> {
        let Some(mut child) = self.child.take() else {
            self.port = None;
            return Ok(());
        };
        let Some(pid) = child.id() else {
            // Already reaped.
            self.port = None;
            return Ok(());
        };

        signal_group(pid, libc::SIGTERM);
        let deadline = grace + KILL_AFTER;
        match tokio::time::timeout(deadline, child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                // uvicorn does not speed up on a second SIGTERM: we cut it off.
                signal_group(pid, libc::SIGKILL);
                let _ = child.wait().await;
            }
        }
        self.port = None;
        Ok(())
    }

    /// Terminate without waiting, for window close.
    pub fn kill_now(&mut self) {
        if let Some(child) = self.child.as_mut() {
            if let Some(pid) = child.id() {
                signal_group(pid, libc::SIGKILL);
            }
        }
        self.child = None;
        self.port = None;
    }
}

/// Send a signal to the whole process group (`-pgid`).
///
/// Shared with `job.rs`: the long model operations are signalled through exactly
/// this ladder rather than a second mechanism.
pub fn signal_group(pid: u32, signal: libc::c_int) {
    // `process_group(0)` makes the child its group leader, so its pid is also
    // its pgid.
    unsafe {
        libc::kill(-(pid as libc::pid_t), signal);
    }
}

/// Relay a stream to the interface, line by line.
///
/// `jobs` is `Some` for the long model operations: their structured lines are
/// also folded into the job status, so a terminal failure can report the reason
/// the child gave instead of only its exit code. The raw line still reaches the
/// Logs tab either way.
fn pump<R>(app: AppHandle, reader: R, structured: bool, jobs: Option<SharedJobs>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tauri::async_runtime::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }
            // On stderr, tqdm rewrites its bar with carriage returns and no
            // newline: we re-split so as not to accumulate one giant line.
            for fragment in line.split('\r').filter(|part| !part.trim().is_empty()) {
                if structured {
                    if let Some(jobs) = jobs.as_ref() {
                        job::note_line(jobs, fragment).await;
                    }
                }
                let _ = app.emit(
                    "server-line",
                    ServerLine {
                        structured,
                        line: fragment.to_owned(),
                    },
                );
            }
        }
    });
}

/// Check the configured port is bindable, and say what is wrong if it is not.
///
/// Worth doing rather than letting uvicorn discover it: on a bind failure the
/// server exits and the only symptom reaching the user would be `wait_healthy`
/// timing out 30 seconds later with "the server is not listening", while the real
/// `address already in use` sits buried in the log tab. Now that the port is
/// fixed, a collision is a routine situation — a server left over from a previous
/// session, or `uv run mflux-server` running alongside — so it deserves a precise
/// message straight away.
///
/// A race window remains between this check and the server binding, unavoidable
/// since `mflux-server` accepts no pre-opened socket.
fn ensure_port_free(port: u16) -> Result<(), String> {
    match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => Err(format!(
            "Port {port} is already in use. Something else is holding it — a server left over \
             from a previous session, or `uv run mflux-server` started alongside. Stop it, or \
             change the port in the Configuration tab."
        )),
        Err(error) => Err(format!("Cannot bind 127.0.0.1:{port}: {error}")),
    }
}

/// Wait for `/health` to answer. The server listens quickly — it loads no
/// weights at startup — but uvicorn still takes about a second to bind.
pub async fn wait_healthy(port: u16, timeout: Duration) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + timeout;
    let address = format!("127.0.0.1:{port}");
    while tokio::time::Instant::now() < deadline {
        if tokio::net::TcpStream::connect(&address).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    Err(format!("the server is not listening on {address}"))
}

/// Start the pre-quantization as an owned job, and return once it is running.
///
/// Deliberately not awaited to completion: this runs for hours, and the caller
/// used to be a `#[tauri::command]` whose promise was the only record that it was
/// happening. Progress and the terminal outcome are read back through
/// `job_status`, so a tab switch no longer loses the operation.
pub async fn start_prequantize(
    app: AppHandle,
    jobs: &SharedJobs,
    paths: &Paths,
    model: String,
    bits: u8,
    components: Vec<String>,
    dest: Option<String>,
) -> Result<(), String> {
    let binary = paths.prequantize_bin();
    if !binary.is_file() {
        return Err(format!("{} is missing.", binary.display()));
    }

    let mut command = Command::new(&binary);
    command
        .current_dir(&paths.data)
        .env("HF_HOME", paths.effective_hf_home())
        .env("PYTHONUNBUFFERED", "1")
        .arg("--json-logs")
        // Which model and which precision are the job's, not the script's
        // defaults: the capability contract decides what is offered, and the
        // request has to say what was chosen.
        .args(["--model", &model])
        .args(["--bits", &bits.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .process_group(0)
        // Defence in depth only. The explicit kill in `RunEvent::Exit` is what
        // actually guarantees the child dies; this covers the paths where the
        // manager is dropped without one.
        .kill_on_drop(true);
    if let Some(dest) = dest.as_deref().filter(|value| !value.is_empty()) {
        command.args(["--dest", dest]);
    }
    if !components.is_empty() {
        command.arg("--components").args(&components);
    }

    let target = format!("{model} @ {bits}-bit");
    spawn_job(app, jobs, command, JobKind::Prequantize, target, "conversion").await
}

/// Catalogue with cache state, straight from `mflux-server-fetch --status`.
///
/// Goes through the Python side rather than reading the HuggingFace cache in Rust:
/// the catalogue and the config live there, and `scan_cache_dir` already answers
/// the question. Cheap enough to call on every visit to the Models tab — the
/// script imports neither torch nor mflux.
pub async fn models_status(paths: &Paths) -> Result<serde_json::Value, String> {
    let binary = paths.fetch_bin();
    if !binary.is_file() {
        return Err(format!("{} is missing.", binary.display()));
    }
    let output = Command::new(&binary)
        .current_dir(&paths.data)
        .env("MFLUX_SERVER_CONFIG", &paths.config)
        .env("HF_HOME", paths.effective_hf_home())
        .arg("--status")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("could not read the catalogue: {error}"))?;

    if !output.status.success() {
        return Err(format!(
            "mflux-server-fetch --status failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("unreadable catalogue status: {error}"))
}

/// Run one `mflux-server-import` subcommand and hand back its JSON.
///
/// Plumbing only: this opens no directory, reads no `config.json`, and knows
/// nothing about families or profiles. Every verdict is Python's, and the answer
/// is parsed as JSON rather than scraped from human output.
pub async fn run_import(paths: &Paths, args: &[String]) -> Result<serde_json::Value, String> {
    let binary = paths.import_bin();
    if !binary.is_file() {
        return Err(format!("{} is missing.", binary.display()));
    }
    let output = Command::new(&binary)
        .current_dir(&paths.data)
        .env("MFLUX_SERVER_CONFIG", &paths.config)
        .env("HF_HOME", paths.effective_hf_home())
        .args(args)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("could not run the importer: {error}"))?;

    // A non-zero exit still carries a structured verdict, and that verdict is far
    // more useful than the exit code — so parse first, and fall back to stderr
    // only when there is nothing to parse.
    match serde_json::from_slice::<serde_json::Value>(&output.stdout) {
        Ok(value) => Ok(value),
        Err(_) => Err(format!(
            "the importer failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )),
    }
}

/// Start a weight download as an owned job, and return once it is running.
pub async fn start_fetch(
    app: AppHandle,
    jobs: &SharedJobs,
    paths: &Paths,
    key: String,
) -> Result<(), String> {
    let binary = paths.fetch_bin();
    if !binary.is_file() {
        return Err(format!("{} is missing.", binary.display()));
    }

    let mut command = Command::new(&binary);
    command
        .current_dir(&paths.data)
        // The same config the server reads, so an overridden `model_path` or
        // quantization is the one we fetch.
        .env("MFLUX_SERVER_CONFIG", &paths.config)
        .env("HF_HOME", paths.effective_hf_home())
        .env("PYTHONUNBUFFERED", "1")
        // Left on, unlike for the server: the download progress bar *is* the
        // feedback here.
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "0")
        .arg(&key)
        .arg("--json-logs")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .process_group(0)
        .kill_on_drop(true);

    spawn_job(app, jobs, command, JobKind::Fetch, key, "download").await
}

/// Spawn a child under the job manager: single-flight check, adoption, output
/// relay and the monitor task, all in the order that keeps them consistent.
async fn spawn_job(
    app: AppHandle,
    jobs: &SharedJobs,
    mut command: Command,
    kind: JobKind,
    target: String,
    what: &str,
) -> Result<(), String> {
    // The lock is taken *before* the spawn and held across it, so two commands
    // arriving together cannot both pass the check and leave one child
    // unreferenced — which is precisely what a React `disabled` attribute could
    // not prevent.
    let mut guard = jobs.lock().await;
    guard.ensure_free()?;

    let mut child = command
        .spawn()
        .map_err(|error| format!("could not launch the {what}: {error}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    guard.begin(kind, target, child);
    drop(guard);

    if let Some(stdout) = stdout {
        pump(app.clone(), stdout, true, Some(jobs.clone()));
    }
    if let Some(stderr) = stderr {
        pump(app.clone(), stderr, false, None);
    }
    job::monitor(jobs.clone());
    Ok(())
}

/// `true` when a HuggingFace token is available for the *gated* repos.
pub fn hf_token_present(hf_home: &Path) -> bool {
    if std::env::var("HF_TOKEN").is_ok_and(|value| !value.trim().is_empty()) {
        return true;
    }
    std::fs::read_to_string(Paths::hf_token_file(hf_home))
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}
