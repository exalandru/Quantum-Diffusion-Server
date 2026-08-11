//! Where the app does its work.
//!
//! Everything lives under `app_data_dir()`, that is
//! `~/Library/Application Support/com.exalandru.qds/`. Nothing is written inside
//! the bundle: it is read-only and gets replaced on every update.
//!
//! These paths are not a matter of convenience: `mflux_server` creates its image
//! directory and its log file while the application initializes, before the
//! server even listens. Since a process launched by Finder inherits `/` as its
//! current directory, any relative path would make startup fail. So we hand the
//! server absolute paths whose parent directories already exist.

use std::path::{Path, PathBuf};

use tauri::{AppHandle, Manager};

#[derive(Debug, Clone)]
pub struct Paths {
    /// Working root, under `~/Library/Application Support/`.
    pub data: PathBuf,
    /// Writable copy of the Python project, the source for `uv sync`.
    pub server: PathBuf,
    /// Virtual environment built by uv.
    pub env: PathBuf,
    /// uv-managed CPython (python-build-standalone).
    pub python: PathBuf,
    /// uv's download cache.
    pub uv_cache: PathBuf,
    /// Images served for `response_format="url"`.
    pub images: PathBuf,
    /// `server-config.json`, driven by the configuration form.
    pub config: PathBuf,
    /// Where a new project copy is assembled before it replaces `server/`.
    pub staging: PathBuf,
    /// `flock` target making the installer single-flight across processes, not
    /// only within one app. Never read; only its lock state matters.
    pub install_lock: PathBuf,
    /// What is installed and whether the install finished. Deliberately a sibling
    /// of `server/` rather than a file inside it: the installer replaces that
    /// directory wholesale, so a marker kept there could not survive the very
    /// interruption it exists to describe.
    pub install_record: PathBuf,
    /// Pre-record marker, `server/.bootstrap-version`. Read to recognise an
    /// install made by an earlier build; never written.
    pub stamp: PathBuf,
}

impl Paths {
    pub fn new(app: &AppHandle) -> Result<Self, String> {
        let data = app
            .path()
            .app_data_dir()
            .map_err(|error| format!("data directory not found: {error}"))?;
        Ok(Self {
            server: data.join("server"),
            env: data.join("env"),
            python: data.join("python"),
            uv_cache: data.join("uv-cache"),
            images: data.join("images"),
            config: data.join("server-config.json"),
            staging: data.join("server.staging"),
            install_lock: data.join("bootstrap.lock"),
            install_record: data.join("bootstrap.json"),
            stamp: data.join("server").join(".bootstrap-version"),
            data,
        })
    }

    /// Create everything we will write into. Called before every start: the
    /// server, for its part, does not create the parents of its log file.
    pub fn ensure(&self) -> Result<(), String> {
        for directory in [&self.data, &self.images] {
            std::fs::create_dir_all(directory)
                .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
        }
        Ok(())
    }

    /// The console entry point installed by `uv sync`.
    pub fn server_bin(&self) -> PathBuf {
        self.env.join("bin").join("mflux-server")
    }

    /// The pre-quantization tool, installed by the same wheel.
    pub fn prequantize_bin(&self) -> PathBuf {
        self.env.join("bin").join("mflux-server-prequantize")
    }

    /// The weight downloader, installed by the same wheel.
    pub fn fetch_bin(&self) -> PathBuf {
        self.env.join("bin").join("mflux-server-fetch")
    }

    /// The local-model importer, installed by the same wheel.
    pub fn import_bin(&self) -> PathBuf {
        self.env.join("bin").join("mflux-server-import")
    }

    /// `true` once `uv sync` has produced a usable environment.
    pub fn env_ready(&self) -> bool {
        self.server_bin().is_file()
    }

    /// Location of the HuggingFace token, the way `huggingface_hub` resolves it.
    ///
    /// We reuse the file `hf auth login` writes rather than opening a second
    /// secret store: duplicating the token into the Keychain while the very same
    /// one sits in plaintext right next to it would buy nothing.
    pub fn hf_token_file(hf_home: &Path) -> PathBuf {
        hf_home.join("token")
    }

    /// Fallback `HF_HOME`, matching `huggingface_hub`'s own default.
    ///
    /// Only reached when the configuration says nothing. `storage.hf_home` is the
    /// setting, it is owned and validated by `settings.py`, and this is the
    /// bootstrap value for the case where there is no configuration to read yet —
    /// not a second opinion about where weights belong.
    pub fn default_hf_home() -> PathBuf {
        std::env::var_os("HF_HOME")
            .map(PathBuf::from)
            .or_else(|| dirs_home().map(|home| home.join(".cache").join("huggingface")))
            .unwrap_or_else(|| PathBuf::from("/tmp/huggingface"))
    }

    /// The HuggingFace root every child must be given.
    ///
    /// One resolver, called from one place, so a status scan cannot read one cache
    /// while a download writes another. Python resolves the same value
    /// independently (`Settings.effective_hf_home`) and reaches the same answer
    /// because both prefer the configured key; a regression test pins that.
    ///
    /// Deliberately *not* created if it is missing: a path under an unmounted
    /// volume must stay absent so the availability rules can report
    /// `volume_unmounted`, rather than being quietly materialised as an empty
    /// directory on the boot disk.
    pub fn effective_hf_home(&self) -> PathBuf {
        crate::config::hf_home(self).unwrap_or_else(Self::default_hf_home)
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
