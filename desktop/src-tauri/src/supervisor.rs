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
    pub fn status(&self) -> ServerStatus {
        ServerStatus {
            running: self.child.is_some(),
            port: self.port,
            last_exit: self.last_exit.clone(),
        }
    }

    /// Start the server, or return the current port if it is already running.
    pub async fn start(&mut self, app: &AppHandle, paths: &Paths) -> Result<u16, String> {
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

        let port = free_port()?;
        let hf_home = Paths::default_hf_home();
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
            pump(app.clone(), stdout, true);
        }
        if let Some(stderr) = child.stderr.take() {
            pump(app.clone(), stderr, false);
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
fn signal_group(pid: u32, signal: libc::c_int) {
    // `process_group(0)` makes the child its group leader, so its pid is also
    // its pgid.
    unsafe {
        libc::kill(-(pid as libc::pid_t), signal);
    }
}

/// Relay a stream to the interface, line by line.
fn pump<R>(app: AppHandle, reader: R, structured: bool)
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

/// Reserve a free port by letting the OS choose it, then release it.
///
/// A race window remains between closing it and the server binding, unavoidable
/// since `mflux-server` accepts no pre-opened socket and does not announce the
/// port it obtained.
fn free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("no free port: {error}"))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| format!("port unreadable: {error}"))
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

/// Run the pre-quantization and relay its output like the server's.
pub async fn run_prequantize(
    app: AppHandle,
    paths: &Paths,
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
        .env("HF_HOME", Paths::default_hf_home())
        .env("PYTHONUNBUFFERED", "1")
        .arg("--json-logs")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .process_group(0);
    if let Some(dest) = dest.as_deref().filter(|value| !value.is_empty()) {
        command.args(["--dest", dest]);
    }
    if !components.is_empty() {
        command.arg("--components").args(&components);
    }

    let mut child = command
        .spawn()
        .map_err(|error| format!("could not launch the conversion: {error}"))?;
    if let Some(stdout) = child.stdout.take() {
        pump(app.clone(), stdout, true);
    }
    if let Some(stderr) = child.stderr.take() {
        pump(app.clone(), stderr, false);
    }

    let status = child
        .wait()
        .await
        .map_err(|error| format!("conversion interrupted: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("the conversion failed (code {:?})", status.code()))
    }
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
