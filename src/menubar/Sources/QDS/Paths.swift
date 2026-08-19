import Foundation

/// Where everything the app owns lives, derived once.
///
/// All of it under `~/Library/Application Support/<bundle id>`, and the bundle
/// identifier is the only constant: two accounts on one Mac get their own, and
/// nothing here is written down twice.
///
/// The layout is the one the Tauri app used, deliberately. An existing install
/// keeps its configuration, its images and its downloaded weights across the
/// move to this app — those are gigabytes and hours, and relocating them to
/// express a rewrite would be the rewrite's problem becoming the user's.
struct Paths {
    static let bundleID = "com.exalandru.qds"

    let data: URL

    init(data: URL? = nil) {
        self.data =
            data
            ?? FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Paths.bundleID, isDirectory: true)
    }

    /// `server-config.json`. The server owns its contents; this app only reads
    /// them, except to seed a default file when none exists — which happens
    /// before any server is running, so there is no second writer.
    var config: URL { data.appendingPathComponent("server-config.json") }

    /// Generated images, served for `response_format="url"`.
    var images: URL { data.appendingPathComponent("images", isDirectory: true) }

    /// uv's tool environments, and the `qds` executable it links.
    var tools: URL { data.appendingPathComponent("tools", isDirectory: true) }
    var bin: URL { data.appendingPathComponent("bin", isDirectory: true) }
    var pythons: URL { data.appendingPathComponent("python", isDirectory: true) }
    var uvCache: URL { data.appendingPathComponent("uv-cache", isDirectory: true) }

    /// The installed server. `uv tool install` links it here.
    var qds: URL { bin.appendingPathComponent("qds") }

    /// What is installed and whether the install finished. A sibling of the
    /// tool directory rather than a file inside it: `uv tool install` replaces
    /// that directory wholesale, so a marker kept there could not survive the
    /// very interruption it exists to describe.
    var installRecord: URL { data.appendingPathComponent("bootstrap.json") }

    /// `flock` target making the installer single-flight across processes, not
    /// only within one app. Never read; only its lock state matters.
    var installLock: URL { data.appendingPathComponent("bootstrap.lock") }

    /// The server's stdout and stderr. The log *view* is `/admin/logs`, served
    /// by the server itself; this is for the case that view cannot exist —
    /// a server that died before it could answer anything.
    var serverLog: URL { data.appendingPathComponent("server.log") }

    /// The control-plane credential for local clients, written by the server at
    /// every start and readable only by this user.
    ///
    /// The tray cannot present the admin password — it is hashed, and there is
    /// nobody at the keyboard when it polls. It reads this instead. The boundary
    /// that claims is "something already able to read this directory", which
    /// could take the server over anyway by editing the configuration and
    /// waiting for a restart; the token grants the same authority without the
    /// wait, and never leaves the machine.
    var adminToken: URL { data.appendingPathComponent("admin-token") }

    /// The pid of the server this app started, so a later launch can recognise
    /// one it left behind.
    ///
    /// The app can die without running any cleanup — Force Quit sends SIGKILL,
    /// and a crash gives no notice at all — and the server, being in its own
    /// process group, survives and keeps the port. Without this file the next
    /// launch sees only "port 8765 is in use" and blames something else.
    var serverPid: URL { data.appendingPathComponent("server.pid") }

    func ensure() throws {
        for directory in [data, images, tools, bin, pythons, uvCache] {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
        }
    }
}
