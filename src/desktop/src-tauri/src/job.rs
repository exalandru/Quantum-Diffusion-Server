//! Lifecycle of the long model operations: downloading weights, and the
//! FLUX.2-dev conversion.
//!
//! These used to be a `Command` awaited inside a `#[tauri::command]`, with no
//! handle kept anywhere. Three consequences, all observed:
//!
//! * the only guard against starting a second one was a React `disabled`
//!   attribute, and `App` unmounts `Models` on a tab switch — so leaving the tab
//!   and coming back re-armed the button while the first child was still running;
//! * there was no way to stop one, and a FLUX.2-dev conversion runs for hours at
//!   a ~66 GB peak;
//! * `RunEvent::Exit` only reached the *server* child, so quitting the app left
//!   the conversion running as an orphan under launchd, invisible.
//!
//! So the child lives here, in Tauri state, exactly like `Supervisor` holds the
//! server: one slot, signalled through the same process-group ladder. Runtime
//! state only — nothing is persisted, because nothing here survives the process.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use tokio::process::Child;
use tokio::sync::Mutex;

use crate::supervisor::signal_group;

/// How often the monitor task asks whether the child has exited.
///
/// Reaping under the same lock that cancellation takes is what makes the two
/// race-free, so it has to be a poll rather than a bare `wait()`: `wait()` would
/// own the child and reap it outside the lock, leaving a window where cancel
/// still believes it has a live pid.
const POLL: Duration = Duration::from_millis(250);

/// Grace given to a cancelled child before SIGKILL. Same shape as the server's
/// ladder; shorter, because these children have no in-flight HTTP to drain — the
/// cost of cutting one off is a partial download, which was already the outcome.
const CANCEL_GRACE: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Fetch,
    Prequantize,
}

impl JobKind {
    fn label(self) -> &'static str {
        match self {
            JobKind::Fetch => "a model download",
            JobKind::Prequantize => "the FLUX.2-dev conversion",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    #[default]
    Idle,
    Running,
    /// SIGTERM sent, child not yet reaped. Distinct from `Running` so the UI can
    /// stop offering Cancel twice, and from `Cancelled` so it does not claim the
    /// process is gone before it is.
    Cancelling,
    Completed,
    Failed,
    Cancelled,
}

impl JobState {
    fn is_active(self) -> bool {
        matches!(self, JobState::Running | JobState::Cancelling)
    }
}

/// What React needs to reconstruct the operation after `Models` remounts.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub state: JobState,
    pub kind: Option<JobKind>,
    /// Model key for a fetch; the component list for a conversion.
    pub target: Option<String>,
    /// Name of the last structured event on the child's stdout, verbatim.
    pub event: Option<String>,
    /// Its `fields` object, verbatim — we do not reshape the child's schema.
    pub fields: Option<Value>,
    /// Latest human-readable line from the structured stream while running; the
    /// terminal reason once finished.
    pub message: Option<String>,
    pub started_at_ms: Option<u64>,
    pub finished_at_ms: Option<u64>,
}

/// The reason a failed job reports: what the child said on its structured
/// stream, falling back to the exit code only when it said nothing.
///
/// The fallback used to be the whole story — every failure read "the download
/// failed (code Some(1))", whatever had actually gone wrong.
fn failure_message(error_message: Option<&str>, code: Option<i32>) -> String {
    match (error_message, code) {
        (Some(reason), _) => reason.to_owned(),
        (None, Some(code)) => format!("exited with code {code}"),
        (None, None) => "terminated by a signal".to_owned(),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Default)]
pub struct JobManager {
    /// Held here rather than by the awaiting task on purpose: reading the pid,
    /// signalling it and reaping it then all happen under one lock, so a cancel
    /// can never signal a pid that has already been reaped and reused.
    child: Option<Child>,
    /// Set by `request_cancel`, read when the child is reaped to tell a
    /// cancellation apart from a failure — the exit code cannot, since a
    /// SIGTERM'd Python exits non-zero either way.
    cancelling: bool,
    /// Last ERROR-level message from the structured stream, kept as the terminal
    /// reason candidate. Exit codes alone said "code Some(1)" and nothing else.
    error_message: Option<String>,
    status: JobStatus,
}

pub type SharedJobs = Arc<Mutex<JobManager>>;

impl JobManager {
    /// Adopt a freshly spawned child. The caller has already checked that no
    /// other job is active, under the same lock.
    pub fn begin(&mut self, kind: JobKind, target: String, child: Child) {
        self.child = Some(child);
        self.cancelling = false;
        self.error_message = None;
        self.status = JobStatus {
            state: JobState::Running,
            kind: Some(kind),
            target: Some(target),
            started_at_ms: Some(now_ms()),
            ..Default::default()
        };
    }

    /// `Err` when a job is already active, naming it so the message is actionable.
    pub fn ensure_free(&self) -> Result<(), String> {
        if self.status.state.is_active() {
            let what = self.status.kind.map(JobKind::label).unwrap_or("an operation");
            let target = self.status.target.clone().unwrap_or_default();
            return Err(format!(
                "{what}{} is already running. Only one heavy model operation runs at a time — \
                 they compete for the same unified memory and the same HuggingFace cache. \
                 Wait for it, or cancel it first.",
                if target.is_empty() {
                    String::new()
                } else {
                    format!(" ({target})")
                }
            ));
        }
        Ok(())
    }

    pub fn status(&mut self) -> JobStatus {
        self.reap();
        self.status.clone()
    }

    /// Notice a child that has exited and settle the terminal state.
    fn reap(&mut self) {
        let Some(child) = self.child.as_mut() else {
            return;
        };
        // `None` = still running, `Err` = status unreadable; both leave it alone.
        let Ok(Some(exit)) = child.try_wait() else {
            return;
        };
        self.child = None;
        self.status.finished_at_ms = Some(now_ms());
        if self.cancelling {
            self.status.state = JobState::Cancelled;
            self.status.message = Some("Cancelled.".to_owned());
        } else if exit.success() {
            self.status.state = JobState::Completed;
            self.status.message = None;
        } else {
            self.status.state = JobState::Failed;
            self.status.message = Some(failure_message(self.error_message.as_deref(), exit.code()));
        }
        self.cancelling = false;
    }

    /// SIGTERM the child's process group. `Err` when nothing is running.
    ///
    /// Returns the pid the SIGKILL escalation must target, read in this same
    /// critical section: fetching it in a second lock would let a job that
    /// finished meanwhile hand the escalation to whatever started next.
    pub fn request_cancel(&mut self) -> Result<(JobStatus, u32), String> {
        self.reap();
        if !self.status.state.is_active() {
            return Err("No model operation is running.".to_owned());
        }
        // Reading the pid under the same lock as `reap` is what keeps this from
        // ever signalling a reaped, recycled pid.
        let Some(pid) = self.child.as_ref().and_then(Child::id) else {
            return Err("The operation has already finished.".to_owned());
        };
        self.cancelling = true;
        self.status.state = JobState::Cancelling;
        self.status.message = Some("Stopping…".to_owned());
        signal_group(pid, libc::SIGTERM);
        Ok((self.status.clone(), pid))
    }

    /// Escalate a cancellation that SIGTERM did not finish. No-op if the child
    /// already exited or a different job has since started.
    fn escalate(&mut self, pid: u32) {
        self.reap();
        if !self.status.state.is_active() {
            return;
        }
        if self.child.as_ref().and_then(Child::id) == Some(pid) {
            signal_group(pid, libc::SIGKILL);
        }
    }

    /// Terminate without waiting, for application exit.
    pub fn kill_now(&mut self) {
        if let Some(pid) = self.child.as_ref().and_then(Child::id) {
            signal_group(pid, libc::SIGKILL);
        }
        self.child = None;
    }

    /// Fold one structured stdout line into the status.
    ///
    /// Only the fields the UI actually needs: the event name and its `fields` for
    /// progress, `message` for the current step, and an ERROR-level `message` as
    /// the terminal reason. Everything else on the line is ignored, and the raw
    /// line still goes to the Logs tab untouched — this is not a second protocol.
    fn note_line(&mut self, line: &str) {
        if !self.status.state.is_active() {
            return;
        }
        let Ok(Value::Object(record)) = serde_json::from_str::<Value>(line) else {
            return;
        };
        if let Some(event) = record.get("event").and_then(Value::as_str) {
            self.status.event = Some(event.to_owned());
            self.status.fields = record.get("fields").cloned();
        }
        if let Some(message) = record.get("message").and_then(Value::as_str) {
            self.status.message = Some(message.to_owned());
            if matches!(
                record.get("level").and_then(Value::as_str),
                Some("ERROR") | Some("CRITICAL")
            ) {
                self.error_message = Some(message.to_owned());
            }
        }
    }
}

/// Record a structured line against the active job, if any.
pub async fn note_line(jobs: &SharedJobs, line: &str) {
    jobs.lock().await.note_line(line);
}

/// Drive the job to a terminal state without waiting on React to poll.
///
/// A plain `child.wait()` would own the child outside the lock; polling
/// `try_wait` under it keeps pid handling in one place — see `POLL`.
pub fn monitor(jobs: SharedJobs) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(POLL).await;
            let mut guard = jobs.lock().await;
            guard.reap();
            if !guard.status.state.is_active() {
                return;
            }
        }
    });
}

/// SIGTERM → bounded wait → SIGKILL, the same ladder the server shutdown uses.
pub fn arm_kill_after_grace(jobs: SharedJobs, pid: u32) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(CANCEL_GRACE).await;
        jobs.lock().await.escalate(pid);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A manager that believes a job is running, without a real child. Enough to
    /// exercise the line folding and the single-flight check, which are the parts
    /// with logic in them.
    fn running() -> JobManager {
        JobManager {
            status: JobStatus {
                state: JobState::Running,
                kind: Some(JobKind::Fetch),
                target: Some("z-image-turbo".to_owned()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn structured_error_becomes_the_terminal_reason() {
        // The point of parsing the stream at all: "code Some(1)" told the user
        // nothing about *why*.
        let mut jobs = running();
        jobs.note_line(r#"{"ts":"t","level":"INFO","logger":"l","message":"Fetching z-image-turbo"}"#);
        jobs.note_line(r#"{"ts":"t","level":"ERROR","logger":"l","message":"Unknown model 'nope'"}"#);
        assert_eq!(jobs.error_message.as_deref(), Some("Unknown model 'nope'"));
        assert_eq!(failure_message(jobs.error_message.as_deref(), Some(1)), "Unknown model 'nope'");
    }

    #[test]
    fn without_a_structured_reason_the_exit_code_is_the_fallback() {
        assert_eq!(failure_message(None, Some(1)), "exited with code 1");
        assert_eq!(failure_message(None, None), "terminated by a signal");
    }

    #[test]
    fn an_info_line_updates_the_message_but_is_not_a_failure_reason() {
        let mut jobs = running();
        jobs.note_line(r#"{"level":"INFO","message":"Downloading / reading repo/x"}"#);
        assert_eq!(jobs.status.message.as_deref(), Some("Downloading / reading repo/x"));
        assert_eq!(jobs.error_message, None);
    }

    #[test]
    fn progress_events_are_kept_verbatim() {
        let mut jobs = running();
        jobs.note_line(
            r#"{"level":"INFO","message":"  quantized 8/56 blocks","event":"prequantize_progress","fields":{"block":8,"blocks":56}}"#,
        );
        assert_eq!(jobs.status.event.as_deref(), Some("prequantize_progress"));
        let fields = jobs.status.fields.as_ref().expect("fields kept");
        assert_eq!(fields.get("block").and_then(Value::as_u64), Some(8));
        assert_eq!(fields.get("blocks").and_then(Value::as_u64), Some(56));
    }

    #[test]
    fn non_json_output_is_ignored_rather_than_guessed_at() {
        // tqdm fragments and tracebacks share this stream's sibling; nothing here
        // may try to read meaning out of human text.
        let mut jobs = running();
        jobs.note_line("Traceback (most recent call last):");
        jobs.note_line("  4%|##        | 1.2G/28G [00:31<11:20, 39.4MB/s]");
        assert_eq!(jobs.status.message, None);
        assert_eq!(jobs.error_message, None);
    }

    #[test]
    fn a_finished_job_stops_absorbing_late_lines() {
        // Pumps outlive the child briefly; a straggler must not overwrite the
        // terminal reason that has already been settled.
        let mut jobs = running();
        jobs.status.state = JobState::Failed;
        jobs.status.message = Some("Unknown model 'nope'".to_owned());
        jobs.note_line(r#"{"level":"INFO","message":"late line"}"#);
        assert_eq!(jobs.status.message.as_deref(), Some("Unknown model 'nope'"));
    }

    #[test]
    fn single_flight_is_enforced_and_names_what_is_running() {
        let jobs = running();
        let refused = jobs.ensure_free().expect_err("a second job must be refused");
        assert!(refused.contains("a model download"), "names the kind: {refused}");
        assert!(refused.contains("z-image-turbo"), "names the target: {refused}");

        assert!(JobManager::default().ensure_free().is_ok());
    }

    #[test]
    fn a_terminal_job_no_longer_blocks_the_next_one() {
        let mut jobs = running();
        jobs.status.state = JobState::Cancelled;
        assert!(jobs.ensure_free().is_ok());
    }
}
