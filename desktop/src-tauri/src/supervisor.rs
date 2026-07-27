//! Cycle de vie du serveur Python.
//!
//! Trois contraintes dictent ce fichier.
//!
//! **`mflux-server` n'accepte aucun argument** : tout passe par l'environnement
//! et un fichier JSON. Il n'expose pas non plus le port qu'il a ouvert, donc
//! c'est ici qu'on en choisit un libre avant de le lui imposer.
//!
//! **Les chemins d'écriture doivent être absolus**, avec leurs dossiers parents
//! déjà créés : le serveur crée son dossier d'images et son fichier de log
//! pendant l'initialisation, avant d'écouter, et un `.app` lancé par le Finder
//! hérite de `/` comme dossier courant.
//!
//! **L'arrêt doit être borné.** `SIGTERM` déclenche l'arrêt gracieux d'uvicorn,
//! borné côté serveur par `shutdown_grace_s` ; mesuré à environ 10 s en pleine
//! génération. Mais un second `SIGTERM` ne force rien — seul `SIGINT` le fait
//! côté uvicorn — d'où l'échelle SIGTERM puis SIGKILL. Le signal est envoyé au
//! *groupe* de processus, sinon quitter l'app laisserait le serveur orphelin :
//! macOS ne récolte pas les petits-enfants.

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

/// Marge ajoutée à `shutdown_grace_s` avant de passer à SIGKILL.
const KILL_AFTER: Duration = Duration::from_secs(8);

/// Une ligne de sortie du serveur, relayée à l'interface.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerLine {
    /// `true` pour stdout, qui ne porte que du JSON Lines en mode `log_json`.
    pub structured: bool,
    pub line: String,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerStatus {
    pub running: bool,
    pub port: Option<u16>,
    /// Renseigné quand le processus s'est arrêté de lui-même.
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

    /// Démarre le serveur, ou renvoie le port courant s'il tourne déjà.
    pub async fn start(&mut self, app: &AppHandle, paths: &Paths) -> Result<u16, String> {
        if let Some(port) = self.port {
            if self.child.is_some() {
                return Ok(port);
            }
        }
        if !paths.env_ready() {
            return Err(format!(
                "Environnement Python absent ({}). Lance l'installation d'abord.",
                paths.server_bin().display()
            ));
        }
        paths.ensure()?;
        crate::config::ensure_exists(paths)?;

        let port = free_port()?;
        let hf_home = Paths::default_hf_home();
        let mut command = Command::new(paths.server_bin());
        command
            // Le dossier courant est posé explicitement : on ne compte jamais
            // sur celui hérité, qui vaut `/` depuis le Finder.
            .current_dir(&paths.data)
            .env("MFLUX_SERVER_CONFIG", &paths.config)
            .env("MFLUX_SERVER_HOST", "127.0.0.1")
            .env("MFLUX_SERVER_PORT", port.to_string())
            .env("MFLUX_SERVER_IMAGE_STORE", &paths.images)
            // Chaîne vide = pas de fichier de log : c'est nous qui capturons.
            .env("MFLUX_SERVER_LOG_FILE", "")
            // stdout devient un flux de JSON Lines, stderr garde tqdm.
            .env("MFLUX_SERVER_LOG_JSON", "1")
            .env("HF_HOME", &hf_home)
            // Réduit le bruit de barres de progression sur stderr.
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            // Sans ça, stdout redirigé serait bufferisé par blocs et la
            // progression arriverait par paquets.
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            // Propre groupe de processus : permet de signaler tout le sous-arbre
            // et évite de laisser le serveur orphelin quand l'app quitte.
            .process_group(0);

        let mut child = command
            .spawn()
            .map_err(|error| format!("impossible de lancer le serveur : {error}"))?;

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

    /// Arrête le serveur : SIGTERM au groupe, attente bornée, puis SIGKILL.
    pub async fn stop(&mut self, grace: Duration) -> Result<(), String> {
        let Some(mut child) = self.child.take() else {
            self.port = None;
            return Ok(());
        };
        let Some(pid) = child.id() else {
            // Déjà moissonné.
            self.port = None;
            return Ok(());
        };

        signal_group(pid, libc::SIGTERM);
        let deadline = grace + KILL_AFTER;
        match tokio::time::timeout(deadline, child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                // uvicorn n'accélère pas sur un second SIGTERM : on tranche.
                signal_group(pid, libc::SIGKILL);
                let _ = child.wait().await;
            }
        }
        self.port = None;
        Ok(())
    }

    /// Termine sans attendre, pour la fermeture de la fenêtre.
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

/// Envoie un signal à tout le groupe de processus (`-pgid`).
fn signal_group(pid: u32, signal: libc::c_int) {
    // `process_group(0)` fait du fils le leader de son groupe : son pid est donc
    // aussi son pgid.
    unsafe {
        libc::kill(-(pid as libc::pid_t), signal);
    }
}

/// Relaie un flux ligne par ligne vers l'interface.
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
            // Sur stderr, tqdm réécrit sa barre avec des retours chariot sans
            // saut de ligne : on redécoupe pour ne pas accumuler une ligne
            // géante.
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

/// Réserve un port libre en le laissant choisir par l'OS, puis le relâche.
///
/// Il reste une fenêtre de course entre la fermeture et le bind du serveur,
/// inévitable puisque `mflux-server` n'accepte pas de socket préouverte et
/// n'annonce pas le port qu'il a obtenu.
fn free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("aucun port libre : {error}"))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| format!("port illisible : {error}"))
}

/// Attend que `/health` réponde. Le serveur écoute vite — il ne charge aucun
/// poids au démarrage — mais uvicorn met tout de même une seconde à binder.
pub async fn wait_healthy(port: u16, timeout: Duration) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + timeout;
    let address = format!("127.0.0.1:{port}");
    while tokio::time::Instant::now() < deadline {
        if tokio::net::TcpStream::connect(&address).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    Err(format!("le serveur n'écoute pas sur {address}"))
}

/// Lance la pré-quantification et relaie sa sortie comme celle du serveur.
pub async fn run_prequantize(
    app: AppHandle,
    paths: &Paths,
    components: Vec<String>,
    dest: Option<String>,
) -> Result<(), String> {
    let binary = paths.prequantize_bin();
    if !binary.is_file() {
        return Err(format!("{} est absent.", binary.display()));
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
        .map_err(|error| format!("impossible de lancer la conversion : {error}"))?;
    if let Some(stdout) = child.stdout.take() {
        pump(app.clone(), stdout, true);
    }
    if let Some(stderr) = child.stderr.take() {
        pump(app.clone(), stderr, false);
    }

    let status = child
        .wait()
        .await
        .map_err(|error| format!("conversion interrompue : {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("la conversion a échoué (code {:?})", status.code()))
    }
}

/// `true` si un token HuggingFace est disponible pour les dépôts *gated*.
pub fn hf_token_present(hf_home: &Path) -> bool {
    if std::env::var("HF_TOKEN").is_ok_and(|value| !value.trim().is_empty()) {
        return true;
    }
    std::fs::read_to_string(Paths::hf_token_file(hf_home))
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}
