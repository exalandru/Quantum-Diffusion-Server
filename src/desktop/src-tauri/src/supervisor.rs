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
        let mut command = qds_command(&paths.server_bin(), paths);
        command
            .env("MFLUX_SERVER_HOST", "127.0.0.1")
            .env("MFLUX_SERVER_PORT", port.to_string())
            .env("MFLUX_SERVER_IMAGE_STORE", &paths.images)
            // Empty string = no log file: we are the ones capturing.
            .env("MFLUX_SERVER_LOG_FILE", "")
            // stdout becomes a JSON Lines stream, stderr keeps tqdm.
            .env("MFLUX_SERVER_LOG_JSON", "1")
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

    let mut command = qds_command(&binary, paths);
    command
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
    let paths_for_completion = paths.clone();
    spawn_job_with(
        app,
        jobs,
        command,
        JobKind::Prequantize,
        target,
        "conversion",
        Some(Box::new(move |status: &job::JobStatus| {
            activate_if_variant_ready(&paths_for_completion, status);
        })),
    )
    .await
}

/// Select a variant the conversion has just declared complete.
///
/// **Who decides, and where.** Python is the only thing that can say an artifact
/// is whole: it validates every required component, checks the precision the
/// weights actually carry, and only then writes the completion marker and emits
/// `prequantize_done`. A run that converted a subset emits `prequantize_partial`
/// and never reaches that line — so "exited 0" is *not* the signal, and neither
/// the interface nor this function re-derives completeness. This reads the claim
/// and acts on it.
///
/// The configuration is written here rather than in the child because the
/// desktop process owns that file: two writers would race, and one of them would
/// be a short-lived subprocess with no knowledge of what the app had in flight.
///
/// A failure to write is logged and dropped on purpose. The conversion itself
/// succeeded and its artifact is on disk and valid; refusing to record the
/// selection is a smaller loss than failing a job that did what it was asked.
fn activate_if_variant_ready(paths: &Paths, status: &job::JobStatus) {
    if status.event.as_deref() != Some("prequantize_done") {
        return;
    }
    let Some(fields) = status.fields.as_ref() else {
        return;
    };
    let (Some(model), Some(bits)) = (
        fields.get("model").and_then(serde_json::Value::as_str),
        fields.get("bits").and_then(serde_json::Value::as_u64),
    ) else {
        eprintln!("a completed conversion named no model and bit depth; nothing selected");
        return;
    };

    if let Err(error) = select_variant(paths, model, bits) {
        eprintln!("{model}'s {bits}-bit copy is ready but could not be selected: {error}");
    } else {
        eprintln!("{model} now uses its {bits}-bit copy on the next server start");
    }
}

/// Write `models.<key>.prequantized_variant`, leaving every other key alone.
///
/// Read-modify-write of the whole document, which is what `config::write` takes.
/// Only this one key changes: other bit depths' artifacts are untouched on disk
/// and unmentioned here, and nothing else in the model's entry is rewritten.
fn select_variant(paths: &Paths, model: &str, bits: u64) -> Result<(), String> {
    let mut document = crate::config::read(paths)?;
    let root = document
        .as_object_mut()
        .ok_or_else(|| "the configuration is not an object".to_owned())?;
    let models = root
        .entry("models")
        .or_insert_with(|| serde_json::Value::Object(Default::default()))
        .as_object_mut()
        .ok_or_else(|| "models is not an object".to_owned())?;
    let entry = models
        .entry(model)
        .or_insert_with(|| serde_json::Value::Object(Default::default()))
        .as_object_mut()
        .ok_or_else(|| format!("models.{model} is not an object"))?;
    entry.insert("prequantized_variant".to_owned(), serde_json::json!(bits));
    crate::config::write(paths, &document)
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
    let output = qds_command(&binary, paths)
        .arg("--status")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("could not read the catalogue: {error}"))?;

    if !output.status.success() {
        // An expected configuration failure answers on stdout with a structured
        // reason. Reading it is what puts one actionable sentence in front of
        // the user instead of a Python traceback, which is what the whole
        // stderr used to become. The traceback is still on stderr for the Logs.
        if let Some(reason) = serde_json::from_slice::<serde_json::Value>(&output.stdout)
            .ok()
            .and_then(|value| {
                value
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
        {
            return Err(reason);
        }
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
    let output = qds_command(&binary, paths)
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

    let mut command = qds_command(&binary, paths);
    command
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
    command: Command,
    kind: JobKind,
    target: String,
    what: &str,
) -> Result<(), String> {
    spawn_job_with(app, jobs, command, kind, target, what, None).await
}

/// As `spawn_job`, with work to run the moment the child is judged successful.
#[allow(clippy::too_many_arguments)]
async fn spawn_job_with(
    app: AppHandle,
    jobs: &SharedJobs,
    mut command: Command,
    kind: JobKind,
    target: String,
    what: &str,
    on_complete: Option<job::OnComplete>,
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
    guard.begin_with(kind, target, child, on_complete);
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

/// A child process that reads this installation's state.
///
/// Every QDS child — the server, the catalogue reader, the importer, the
/// downloader, the converter — needs the same two facts: which configuration
/// file is this installation's, and where its weights are kept. Handing them
/// out in one place is not tidiness; it is the fix for a release blocker.
///
/// `mflux-server-prequantize` was spawned without `MFLUX_SERVER_CONFIG`. Its
/// `load_settings()` therefore fell back to the packaged default path, found
/// nothing, and ran on catalogue defaults — so a configured
/// `storage.cache_dir` was invisible to it and every conversion wrote to the
/// derived default instead. The same silence covered every model override the
/// converter should have honoured: `model_path`, `quantize`,
/// `prequantized_variant`.
///
/// Four of the five children set it and one did not, which is exactly the kind
/// of omission a shared constructor makes impossible rather than unlikely.
/// `no_bare_command_new` below keeps it that way.
fn qds_command(program: &Path, paths: &Paths) -> Command {
    let mut command = Command::new(program);
    command
        // Never the inherited working directory: it is `/` when the app is
        // launched from Finder.
        .current_dir(&paths.data)
        .env("MFLUX_SERVER_CONFIG", &paths.config)
        .env("HF_HOME", paths.effective_hf_home());
    command
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A `Paths` pointing at a temporary directory. Only `config` is read here.
    fn paths_in(dir: &std::path::Path) -> Paths {
        let empty = dir.join("unused");
        Paths {
            data: dir.to_path_buf(),
            server: empty.clone(),
            env: empty.clone(),
            python: empty.clone(),
            uv_cache: empty.clone(),
            images: dir.join("images"),
            config: dir.join("server-config.json"),
            staging: empty.clone(),
            install_lock: empty.clone(),
            install_record: empty.clone(),
            stamp: empty,
        }
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("qds-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn done_status(model: &str, bits: u64) -> job::JobStatus {
        job::JobStatus {
            event: Some("prequantize_done".to_owned()),
            fields: Some(json!({"model": model, "bits": bits, "variant_ready": true})),
            ..Default::default()
        }
    }

    #[test]
    fn a_ready_variant_is_selected_without_disturbing_anything_else() {
        // §10's requirement, as a write: other bit depths are artifacts on disk
        // and are not mentioned here at all, and the rest of the model's entry
        // survives untouched.
        let dir = temp_dir("select");
        let paths = paths_in(&dir);
        crate::config::write(
            &paths,
            &json!({
                "default_model": "z-image",
                "server": {"port": 8765},
                "models": {
                    "z-image": {"enabled": true, "quantize": 4, "prequantized_variant": 3},
                    "flux2-dev": {"enabled": false}
                }
            }),
        )
        .expect("seed");

        activate_if_variant_ready(&paths, &done_status("z-image", 8));

        let written = crate::config::read(&paths).expect("read back");
        let models = written.get("models").expect("models");
        let entry = models.get("z-image").expect("the model's entry");
        assert_eq!(entry.get("prequantized_variant"), Some(&json!(8)));
        // Everything else in the entry, and every other entry, is as it was.
        assert_eq!(entry.get("enabled"), Some(&json!(true)));
        assert_eq!(entry.get("quantize"), Some(&json!(4)));
        assert_eq!(models.get("flux2-dev"), Some(&json!({"enabled": false})));
        assert_eq!(written.get("default_model"), Some(&json!("z-image")));
        assert_eq!(written.get("server"), Some(&json!({"port": 8765})));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_partial_run_selects_nothing() {
        // The whole reason the decision is the child's: this run exited
        // successfully and the artifact is not usable.
        let dir = temp_dir("partial");
        let paths = paths_in(&dir);
        crate::config::write(&paths, &json!({"models": {"z-image": {"enabled": true}}}))
            .expect("seed");

        activate_if_variant_ready(
            &paths,
            &job::JobStatus {
                event: Some("prequantize_partial".to_owned()),
                fields: Some(json!({"model": "z-image", "bits": 8, "variant_ready": false})),
                ..Default::default()
            },
        );

        let written = crate::config::read(&paths).expect("read back");
        assert!(
            written["models"]["z-image"].get("prequantized_variant").is_none(),
            "a partial run must not select a variant"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_event_without_a_model_selects_nothing() {
        let dir = temp_dir("nameless");
        let paths = paths_in(&dir);
        crate::config::write(&paths, &json!({"models": {}})).expect("seed");

        activate_if_variant_ready(
            &paths,
            &job::JobStatus {
                event: Some("prequantize_done".to_owned()),
                fields: Some(json!({"bits": 8})),
                ..Default::default()
            },
        );

        let written = crate::config::read(&paths).expect("read back");
        assert_eq!(written["models"], json!({}));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_model_with_no_entry_yet_gains_one() {
        let dir = temp_dir("fresh");
        let paths = paths_in(&dir);
        crate::config::write(&paths, &json!({"default_model": "z-image"})).expect("seed");

        activate_if_variant_ready(&paths, &done_status("z-image", 4));

        let written = crate::config::read(&paths).expect("read back");
        assert_eq!(written["models"]["z-image"]["prequantized_variant"], json!(4));

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod child_env_tests {
    /// Every child must be built by `qds_command`, and this is what enforces it.
    ///
    /// The release blocker was one spawn site out of five that set `HF_HOME` and
    /// forgot `MFLUX_SERVER_CONFIG`, so the converter ran on catalogue defaults
    /// and wrote to the derived cache instead of the configured one. A reviewer
    /// reading that call site saw an environment being configured and had no
    /// reason to count the variables.
    ///
    /// A source check rather than a runtime one, because `tokio::process::Command`
    /// exposes no way to read back the environment it will apply — and because
    /// the property worth holding is about *every* spawn, including the next one
    /// somebody adds.
    #[test]
    fn no_bare_command_new_outside_the_shared_constructor() {
        // Only the production half: the tests below quote the name in their own
        // assertions, and a guard that counted its own text would be measuring
        // itself.
        let source = include_str!("supervisor.rs");
        let production = source.split("#[cfg(test)]").next().expect("a production half");
        let occurrences: Vec<&str> = production
            .lines()
            .filter(|line| line.contains("Command::new("))
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect();
        assert_eq!(
            occurrences.len(),
            1,
            "every child process must come from `qds_command`; found {occurrences:#?}"
        );
        assert!(
            occurrences[0].contains("let mut command = Command::new(program)"),
            "the one bare construction must be the shared constructor: {}",
            occurrences[0]
        );
    }

    #[test]
    fn the_shared_constructor_carries_the_configuration_and_the_weights() {
        let source = include_str!("supervisor.rs");
        let body = source
            .split("fn qds_command(")
            .nth(1)
            .expect("the shared constructor exists");
        let body = &body[..body.find("\n}").expect("its body ends")];
        assert!(body.contains("MFLUX_SERVER_CONFIG"), "the child reads this installation's config");
        assert!(body.contains("HF_HOME"), "and knows where its weights are");
        assert!(body.contains("current_dir"), "and never inherits `/` from Finder");
    }
}
