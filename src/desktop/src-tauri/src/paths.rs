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
    /// App version that produced `server/`, so we know when to re-sync.
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

    /// `true` once `uv sync` has produced a usable environment.
    pub fn env_ready(&self) -> bool {
        self.server_bin().is_file()
    }

    /// App version that produced the project copy, when there is one.
    pub fn stamped_version(&self) -> Option<String> {
        std::fs::read_to_string(&self.stamp)
            .ok()
            .map(|value| value.trim().to_owned())
    }

    /// Location of the HuggingFace token, the way `huggingface_hub` resolves it.
    ///
    /// We reuse the file `hf auth login` writes rather than opening a second
    /// secret store: duplicating the token into the Keychain while the very same
    /// one sits in plaintext right next to it would buy nothing.
    pub fn hf_token_file(hf_home: &Path) -> PathBuf {
        hf_home.join("token")
    }

    /// Default `HF_HOME`, matching `huggingface_hub`'s own.
    pub fn default_hf_home() -> PathBuf {
        std::env::var_os("HF_HOME")
            .map(PathBuf::from)
            .or_else(|| dirs_home().map(|home| home.join(".cache").join("huggingface")))
            .unwrap_or_else(|| PathBuf::from("/tmp/huggingface"))
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
