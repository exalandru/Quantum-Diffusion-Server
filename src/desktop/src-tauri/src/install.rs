//! What is installed, recorded where the installer cannot delete it.
//!
//! Two facts have to survive an interrupted install, and neither used to.
//!
//! *Completion.* The marker was `server/.bootstrap-version`, a bare version
//! string inside the very directory the installer removes and recopies. Quitting
//! during the ~1.1 GB `uv sync` therefore left new code with no marker at all —
//! indistinguishable from a machine that had never been installed, and offered
//! "Install" rather than "Repair". The record now lives beside the config, is
//! written `installing` *before* anything is touched and `ready` only after
//! `uv sync` has produced a usable environment, so an interruption is an
//! observable state rather than an absence.
//!
//! *Identity.* Readiness was `installed_version == app_version` plus an mtime
//! comparison against the bundle. Neither says what is actually installed: the
//! version is a label the developer controls, and the mtime answered `false`
//! whenever the bundle could not be read — which is exactly how a broken resource
//! path silently disabled change detection for two releases. The record stores a
//! content fingerprint of the inputs that can change the managed runtime, so
//! "up to date" is a statement about bytes.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bumped only for a change an older build could not read correctly.
pub const MARKER_VERSION: u32 = 1;

/// The payload entries that can change the managed Python runtime.
///
/// Deliberately not "the whole bundle": `README.md` is copied because the wheel
/// build reads it, but its contents cannot alter a resolved environment or the
/// server's behaviour, and hashing it would turn a typo into a forced 1.1 GB
/// rebuild. These four are the complete set of inputs `uv sync` consumes plus the
/// source it installs.
const PAYLOAD: [&str; 4] = [".python-version", "mflux_server", "pyproject.toml", "uv.lock"];

/// Entries that are build output or caches, never input.
///
/// `__pycache__` matters for a reason beyond tidiness: the *installed* copy is
/// fingerprinted when migrating an install that predates this record, and it may
/// have accumulated bytecode the bundle never had. Excluding it is what lets an
/// existing, perfectly good installation be recognised instead of rebuilt.
fn is_ignored(name: &str) -> bool {
    name == "__pycache__"
        || name == ".DS_Store"
        || name.ends_with(".pyc")
        || name.ends_with(".pyo")
        || name.ends_with(".tmp")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InstallState {
    /// An install began and has not reported success. On disk after a crash, this
    /// means *interrupted*, not *in progress*: a live run is known from the
    /// single-flight guard, which does not survive the process.
    Installing,
    /// The project was copied and `uv sync` produced a usable environment.
    Ready,
    /// An install ran and reported why it did not finish.
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallRecord {
    pub marker_version: u32,
    pub state: InstallState,
    /// Diagnostic only. It is not part of the readiness decision: a version
    /// string is a label, and two builds sharing one can hold different code.
    pub app_version: String,
    /// The interpreter `uv --python` was pinned to, as read from the payload.
    pub python_version: Option<String>,
    /// SHA-256 over [`PAYLOAD`]. The runtime's actual identity.
    pub payload_fingerprint: String,
    /// Why the last attempt stopped, when it stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl InstallRecord {
    pub fn new(
        state: InstallState,
        app_version: &str,
        python_version: Option<String>,
        payload_fingerprint: &str,
    ) -> Self {
        Self {
            marker_version: MARKER_VERSION,
            state,
            app_version: app_version.to_owned(),
            python_version,
            payload_fingerprint: payload_fingerprint.to_owned(),
            error: None,
        }
    }

    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_owned());
        self
    }
}

/// The record, or `None` when there is nothing trustworthy to read.
///
/// A record from a newer build is treated as absent rather than parsed
/// optimistically — but note what that costs and why it is still right: the app
/// will offer to install, which rewrites the record with this build's schema.
/// That is acceptable here and would not be for the model library, because an
/// install is reproducible from the bundle and a user's imported models are not.
pub fn read_record(path: &Path) -> Option<InstallRecord> {
    let text = std::fs::read_to_string(path).ok()?;
    let record: InstallRecord = serde_json::from_str(&text).ok()?;
    (record.marker_version <= MARKER_VERSION).then_some(record)
}

/// Write the record atomically and durably.
///
/// The same ladder as the config and the model library: a truncated record read
/// on the next launch would claim the wrong state about a 1.1 GB install.
pub fn write_record(path: &Path, record: &InstallRecord) -> Result<(), String> {
    use std::io::Write;

    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;

    let text = serde_json::to_string_pretty(record)
        .map_err(|error| format!("could not serialize the install record: {error}"))?;
    let temporary = path.with_extension("json.tmp");
    {
        let mut file = std::fs::File::create(&temporary)
            .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
        file.write_all(format!("{text}\n").as_bytes())
            .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
        file.sync_all()
            .map_err(|error| format!("could not flush {}: {error}", temporary.display()))?;
    }
    std::fs::rename(&temporary, path)
        .map_err(|error| format!("could not replace {}: {error}", path.display()))?;
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// SHA-256 over the payload entries, by content.
///
/// Deterministic by construction: entries are visited in sorted order, each
/// contributes its path relative to `root`, its length and its bytes, and nothing
/// contributes a timestamp, an inode or a traversal order. So the same tree
/// copied, rebuilt or restored from a backup fingerprints identically, while a
/// single changed byte in `uv.lock` does not.
pub fn fingerprint(root: &Path) -> Result<String, String> {
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for entry in PAYLOAD {
        collect(root, &root.join(entry), &mut files)?;
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));

    let mut hasher = Sha256::new();
    // Domain separator: a future change to what is hashed must not be able to
    // collide with a digest this version produced.
    hasher.update(b"qds-payload-v1");
    for (relative, path) in &files {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

/// Depth-first collection of regular files under `path`, missing entries skipped.
///
/// A missing payload entry is not an error here: its record is simply absent from
/// the digest, which makes the digest differ — that is the honest answer, and the
/// install itself reports a missing `.python-version` or `uv.lock` far more
/// usefully than a hash function could.
fn collect(root: &Path, path: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<(), String> {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return Ok(());
    };
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
    if is_ignored(name) {
        return Ok(());
    }
    if metadata.is_file() {
        let relative = path
            .strip_prefix(root)
            .map_err(|_| format!("{} is outside the payload", path.display()))?;
        out.push((relative.to_string_lossy().into_owned(), path.to_path_buf()));
        return Ok(());
    }
    if !metadata.is_dir() {
        // A symlink in the payload would make the digest depend on a target we do
        // not control. There are none, and if one appears it should be noticed.
        return Ok(());
    }
    let entries = std::fs::read_dir(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("read interrupted: {error}"))?;
        collect(root, &entry.path(), out)?;
    }
    Ok(())
}

/// What the Setup screen must offer. The backend is the authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum BootstrapState {
    /// Nothing has ever been installed here.
    Uninitialized,
    /// An install is running right now, in this process.
    Installing,
    /// Installed and usable, but built from a different payload.
    UpdateRequired,
    /// Installed and interrupted, or installed and since damaged.
    Broken,
    Ready,
}

/// The single readiness decision, kept pure so it can be tested exhaustively.
///
/// `bundled` is `None` when the bundle's fingerprint could not be computed. That
/// deliberately does not force a rebuild: an unreadable bundle says nothing about
/// whether the *installed* environment works, and the install path now validates
/// and reports the source itself instead of failing silently the way the mtime
/// comparison did.
pub fn decide(
    running: bool,
    record: Option<&InstallRecord>,
    env_ready: bool,
    bundled: Option<&str>,
) -> BootstrapState {
    if running {
        return BootstrapState::Installing;
    }
    let Some(record) = record else {
        return BootstrapState::Uninitialized;
    };
    if record.state != InstallState::Ready {
        return BootstrapState::Broken;
    }
    // The record is a claim about a past event; the environment is the thing that
    // has to exist now. `env/` alone was never proof of a finished install, and
    // the record alone is not proof that it is still there.
    if !env_ready {
        return BootstrapState::Broken;
    }
    match bundled {
        Some(bundled) if bundled != record.payload_fingerprint => BootstrapState::UpdateRequired,
        _ => BootstrapState::Ready,
    }
}

/// Recognise an install made before this record existed, without rebuilding it.
///
/// The old marker recorded only a version string, so it cannot say what was
/// installed — but the installed project *is* a byte copy of the payload it came
/// from, so fingerprinting it answers the question directly. An install whose
/// content still matches the bundle is therefore adopted as `ready` and left
/// alone, which is the whole point: nobody should re-download 1.1 GB because the
/// marker's format changed.
pub fn migrate_legacy(
    server: &Path,
    legacy_stamp: &Path,
    env_ready: bool,
) -> Option<InstallRecord> {
    if !env_ready {
        return None;
    }
    let version = std::fs::read_to_string(legacy_stamp).ok()?;
    let fingerprint = fingerprint(server).ok()?;
    let python = std::fs::read_to_string(server.join(".python-version"))
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    Some(InstallRecord::new(
        InstallState::Ready,
        version.trim(),
        python,
        &fingerprint,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A payload with every fingerprinted entry present, plus one that is not.
    fn payload(root: &Path) {
        std::fs::create_dir_all(root.join("mflux_server")).unwrap();
        std::fs::write(root.join(".python-version"), "3.12\n").unwrap();
        std::fs::write(root.join("pyproject.toml"), "[project]\nname='x'\n").unwrap();
        std::fs::write(root.join("uv.lock"), "version = 1\n").unwrap();
        std::fs::write(root.join("README.md"), "documentation\n").unwrap();
        std::fs::write(root.join("mflux_server/app.py"), "print('hi')\n").unwrap();
    }

    fn temp(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("qds-install-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn identical_content_with_different_timestamps_fingerprints_the_same() {
        // The property the mtime comparison could not offer: a rebuild that
        // changes nothing must not ask the user to reinstall 1.1 GB.
        let root = temp("mtime");
        let (left, right) = (root.join("a"), root.join("b"));
        payload(&left);
        payload(&right);
        let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
        for entry in [".python-version", "uv.lock", "pyproject.toml"] {
            let file = std::fs::File::options()
                .write(true)
                .open(right.join(entry))
                .unwrap();
            file.set_modified(old).unwrap();
        }
        assert_eq!(fingerprint(&left).unwrap(), fingerprint(&right).unwrap());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_changed_lock_python_or_source_changes_the_fingerprint() {
        let root = temp("changed");
        let base = root.join("base");
        payload(&base);
        let reference = fingerprint(&base).unwrap();

        for (file, content) in [
            ("uv.lock", "version = 2\n"),
            (".python-version", "3.13\n"),
            ("mflux_server/app.py", "print('bye')\n"),
            ("pyproject.toml", "[project]\nname='y'\n"),
        ] {
            let variant = root.join(file.replace('/', "-"));
            payload(&variant);
            std::fs::write(variant.join(file), content).unwrap();
            assert_ne!(
                reference,
                fingerprint(&variant).unwrap(),
                "{file} must change the fingerprint"
            );
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_new_source_file_changes_the_fingerprint() {
        // Content alone is not enough: adding an empty module changes what is
        // installed, and hashing bytes without paths would miss it.
        let root = temp("added");
        let base = root.join("base");
        payload(&base);
        let reference = fingerprint(&base).unwrap();
        std::fs::write(base.join("mflux_server/extra.py"), "").unwrap();
        assert_ne!(reference, fingerprint(&base).unwrap());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn ignored_and_unrelated_files_leave_the_fingerprint_alone() {
        let root = temp("ignored");
        let base = root.join("base");
        payload(&base);
        let reference = fingerprint(&base).unwrap();

        std::fs::write(base.join("README.md"), "rewritten entirely\n").unwrap();
        std::fs::write(base.join(".bootstrap-version"), "0.1.0").unwrap();
        std::fs::create_dir_all(base.join("mflux_server/__pycache__")).unwrap();
        std::fs::write(base.join("mflux_server/__pycache__/app.pyc"), "bytecode").unwrap();
        std::fs::write(base.join("mflux_server/app.pyc"), "bytecode").unwrap();

        assert_eq!(reference, fingerprint(&base).unwrap());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_record_round_trips_and_a_newer_schema_is_refused() {
        let root = temp("record");
        let file = root.join("bootstrap.json");
        let record = InstallRecord::new(InstallState::Ready, "0.2.0", Some("3.12".into()), "sha256:x");
        write_record(&file, &record).unwrap();
        let read = read_record(&file).unwrap();
        assert_eq!(read.state, InstallState::Ready);
        assert_eq!(read.payload_fingerprint, "sha256:x");
        assert_eq!(read.python_version.as_deref(), Some("3.12"));

        std::fs::write(&file, r#"{"marker_version":99,"state":"ready","app_version":"9",
            "python_version":null,"payload_fingerprint":"sha256:x"}"#).unwrap();
        assert!(read_record(&file).is_none());

        std::fs::write(&file, "not json").unwrap();
        assert!(read_record(&file).is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn an_interrupted_install_is_broken_not_uninitialized_and_not_ready() {
        // The distinction the old marker could not express, and the reason it had
        // to move out of the directory the installer deletes.
        let installing = InstallRecord::new(InstallState::Installing, "0.2.0", None, "sha256:a");
        assert_eq!(
            decide(false, Some(&installing), true, Some("sha256:a")),
            BootstrapState::Broken
        );
        let failed = InstallRecord::new(InstallState::Failed, "0.2.0", None, "sha256:a");
        assert_eq!(
            decide(false, Some(&failed), true, Some("sha256:a")),
            BootstrapState::Broken
        );
        assert_eq!(decide(false, None, true, Some("sha256:a")), BootstrapState::Uninitialized);
    }

    #[test]
    fn readiness_needs_the_record_the_environment_and_the_fingerprint() {
        let ready = InstallRecord::new(InstallState::Ready, "0.2.0", None, "sha256:a");
        assert_eq!(decide(false, Some(&ready), true, Some("sha256:a")), BootstrapState::Ready);
        // `env/` gone: the record's claim is about the past.
        assert_eq!(decide(false, Some(&ready), false, Some("sha256:a")), BootstrapState::Broken);
        // Different payload, same app version: exactly the case version equality
        // could not see.
        assert_eq!(
            decide(false, Some(&ready), true, Some("sha256:b")),
            BootstrapState::UpdateRequired
        );
        // Unreadable bundle: no evidence of a change, and the installed
        // environment still works.
        assert_eq!(decide(false, Some(&ready), true, None), BootstrapState::Ready);
        // A live run outranks everything.
        assert_eq!(decide(true, Some(&ready), true, Some("sha256:b")), BootstrapState::Installing);
        assert_eq!(decide(true, None, false, None), BootstrapState::Installing);
    }

    #[test]
    fn an_old_install_is_adopted_rather_than_rebuilt() {
        let root = temp("migrate");
        let server = root.join("server");
        payload(&server);
        let stamp = server.join(".bootstrap-version");
        std::fs::write(&stamp, "0.2.0").unwrap();
        // Bytecode the bundle never had: an install that has been *used* is still
        // the same install.
        std::fs::create_dir_all(server.join("mflux_server/__pycache__")).unwrap();
        std::fs::write(server.join("mflux_server/__pycache__/app.pyc"), "x").unwrap();

        let record = migrate_legacy(&server, &stamp, true).expect("a valid old install");
        assert_eq!(record.state, InstallState::Ready);
        assert_eq!(record.app_version, "0.2.0");
        assert_eq!(record.python_version.as_deref(), Some("3.12"));

        // The bundle it came from: adopted, and the app asks for nothing.
        let bundle = root.join("bundle");
        payload(&bundle);
        let bundled = fingerprint(&bundle).unwrap();
        assert_eq!(record.payload_fingerprint, bundled);
        assert_eq!(decide(false, Some(&record), true, Some(&bundled)), BootstrapState::Ready);

        // No environment: an orphaned project directory is not an install.
        assert!(migrate_legacy(&server, &stamp, false).is_none());
        // No old marker either: nothing to adopt.
        std::fs::remove_file(&stamp).unwrap();
        assert!(migrate_legacy(&server, &stamp, true).is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
