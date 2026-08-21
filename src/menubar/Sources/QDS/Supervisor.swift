import Darwin
import Foundation

/// The server process: started, watched, and stopped.
///
/// A port of the Tauri app's `supervisor.rs`, and the shape is kept because the
/// reasons for it are unchanged:
///
/// * **The child gets its own process group.** A download or a conversion the
///   server spawns is a grandchild; signalling only the server would leave those
///   running as orphans under launchd, holding the HuggingFace cache and
///   invisible to whatever starts next. Every signal here goes to the group.
/// * **SIGTERM, then a bounded wait, then SIGKILL.** uvicorn does not speed up
///   on a second SIGTERM — only SIGINT does — so the escalation is the only
///   thing that bounds shutdown.
/// * **The port is checked before spawning.** Otherwise a leftover session, or
///   a `qds serve` started by hand, produces a child that exits immediately with
///   its reason buried in a log nobody is looking at.
///
/// `Foundation.Process` is not used: it cannot put a child in its own process
/// group, which is the first bullet above. `posix_spawn` can, so this drops to
/// the C API rather than giving up the property.
@MainActor
final class Supervisor {
    enum State: Equatable {
        case stopped
        case starting
        case running(port: Int)
        case stopping
    }

    private(set) var state: State = .stopped
    /// Why the last run ended, when it ended on its own. Otherwise an unexpected
    /// exit is invisible: the tray simply goes quiet.
    private(set) var lastExit: String?

    private let paths: Paths
    private var pid: pid_t?
    private var reaper: DispatchSourceProcess?
    private var onChange: () -> Void

    init(paths: Paths, onChange: @escaping () -> Void) {
        self.paths = paths
        self.onChange = onChange
    }

    // ── Starting ───────────────────────────────────────────────────────────

    func start(config: ServerConfig) async throws {
        guard state == .stopped else { return }
        try paths.ensure()
        try ServerConfig.seedIfMissing(at: paths.config)
        try Self.ensurePortFree(host: config.host, port: config.port)

        state = .starting
        lastExit = nil
        onChange()

        do {
            pid = try spawn(config: config)
            recordPid(pid!)
        } catch {
            state = .stopped
            onChange()
            throw error
        }
        watch(pid!)

        do {
            try await Self.waitHealthy(host: config.connectHost, port: config.port)
        } catch {
            // It never came up. Do not leave a half-started process behind for
            // the next attempt to trip over.
            await stop(grace: config.shutdownGrace)
            throw error
        }

        state = .running(port: config.port)
        onChange()
    }

    /// Refuse to start on a port something else holds.
    ///
    /// A pre-flight bind rather than a connect: a connect only finds a listener
    /// that answers, and the failure this prevents is any process holding the
    /// port at all.
    ///
    /// It binds **what the server will bind**, not always loopback, and the
    /// reason is measured rather than assumed: with `SO_REUSEADDR` set, binding
    /// `127.0.0.1:P` **succeeds** while another process holds `0.0.0.0:P` — the
    /// classic BSD rule that a more specific address may be bound under a
    /// wildcard. So the old loopback-only check passed cleanly and the server's
    /// own wildcard bind then failed, with its reason buried in a log nobody is
    /// reading, which is exactly the outcome this exists to prevent.
    ///
    /// `SO_REUSEADDR` stays because uvicorn sets it too: this pre-flight is
    /// useful only insofar as it predicts uvicorn's bind, so it has to ask the
    /// same question with the same options.
    nonisolated static func ensurePortFree(host: String, port: Int) throws {
        let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard socket >= 0 else { throw QDSError("cannot create a socket: \(errno)") }
        defer { Darwin.close(socket) }

        var yes: Int32 = 1
        setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        address.sin_addr.s_addr =
            ["0.0.0.0", "::", ""].contains(host) ? INADDR_ANY.bigEndian : inet_addr(host)

        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(socket, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if bound != 0 {
            if errno == EADDRINUSE {
                throw QDSError(
                    """
                    Port \(port) is already in use. Something else is holding it - a server left \
                    over from a previous session, or `qds serve` started alongside. Stop it, or \
                    change the port in the dashboard's Configuration tab.
                    """)
            }
            throw QDSError("Cannot bind \(host):\(port): \(String(cString: strerror(errno)))")
        }
    }

    /// Wait for the port to answer. The server listens quickly — it loads no
    /// weights at startup — but uvicorn still takes about a second to bind.
    nonisolated static func waitHealthy(
        host: String, port: Int, timeout: TimeInterval = 30
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if canConnect(host: host, port: port) { return }
            try? await Task.sleep(nanoseconds: 150_000_000)
        }
        throw QDSError("the server is not listening on \(host):\(port)")
    }

    nonisolated private static func canConnect(host: String, port: Int) -> Bool {
        let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard socket >= 0 else { return false }
        defer { Darwin.close(socket) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        address.sin_addr.s_addr = inet_addr(host)
        return withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(socket, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
    }

    private func spawn(config: ServerConfig) throws -> pid_t {
        // Everything the server needs comes through the environment, exactly as
        // it did before: the same variables, so a server started by this app and
        // one started by hand read their settings the same way.
        return try Self.spawnDetached(
            executable: paths.qds.path,
            argv: [paths.qds.path, "serve"],
            environment: Self.childEnvironment(paths: paths, config: config),
            logPath: paths.serverLog.path
        )
    }

    /// Start a process in its own process group, with output appended to a file.
    ///
    /// The whole reason this app does not use `Foundation.Process`, isolated so
    /// the property can actually be checked: a test spawns `sleep` through it
    /// and asserts `getpgid(pid) == pid`. Asserting that on the real server
    /// would mean installing one to find out.
    nonisolated static func spawnDetached(
        executable: String, argv: [String], environment: [String], logPath: String
    ) throws -> pid_t {
        var attributes: posix_spawnattr_t?
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        // Its own process group, so the ladder below reaches the server and
        // everything it spawns as one. `Foundation.Process` cannot do this.
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP))
        posix_spawnattr_setpgroup(&attributes, 0)

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        // stdout and stderr to a file rather than to pipes nobody drains: the
        // structured log is served by the server at `/admin/logs`, and this is
        // only for the case that endpoint cannot exist — a server that died
        // before it could answer anything.
        FileManager.default.createFile(atPath: logPath, contents: nil)
        posix_spawn_file_actions_addopen(
            &actions, 1, logPath, O_WRONLY | O_APPEND | O_CREAT, 0o644)
        posix_spawn_file_actions_adddup2(&actions, 1, 2)

        var pid: pid_t = 0
        let status = withCStrings(argv) { argvPointers in
            withCStrings(environment) { envPointers in
                posix_spawn(&pid, executable, &actions, &attributes, argvPointers, envPointers)
            }
        }
        guard status == 0 else {
            throw QDSError("could not start \(executable): \(String(cString: strerror(status)))")
        }
        return pid
    }

    /// The environment handed to the server, and the only channel it is
    /// configured through.
    nonisolated static func childEnvironment(paths: Paths, config: ServerConfig) -> [String] {
        childEnvironmentMap(paths: paths, config: config).map { "\($0.key)=\($0.value)" }
    }

    /// The same environment as a dictionary.
    ///
    /// `posix_spawn` wants `KEY=VALUE` strings and `Process` wants a
    /// dictionary; both callers need the *same* variables, and
    /// `QDS_SERVER_CONFIG` in particular. `Bootstrap` fetches the rewriter's
    /// weights through `Process`, and without that variable `qds fetch` would
    /// resolve its Hugging Face cache somewhere the server never looks.
    nonisolated static func childEnvironmentMap(paths: Paths, config: ServerConfig)
        -> [String: String]
    {
        var environment: [String: String] = [
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": NSHomeDirectory(),
            // The server reads its own settings from this file. Without it,
            // `qds` looks for one relative to its package — `site-packages/`,
            // where it does not exist — and every setting is silently ignored.
            "QDS_SERVER_CONFIG": paths.config.path,
            "QDS_SERVER_IMAGE_STORE": paths.images.path,
            "QDS_SERVER_LOG_JSON": "1",
            // Empty means "no file": this process captures the output.
            "QDS_SERVER_LOG_FILE": "",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        ]
        // Inherited rather than reasserted, so a token or a cache root the user
        // exported keeps working.
        for key in ["LANG", "LC_ALL", "TMPDIR", "HF_HOME", "HF_TOKEN"] {
            if let value = ProcessInfo.processInfo.environment[key] {
                environment[key] = value
            }
        }
        return environment
    }

    // ── Surviving our own death ────────────────────────────────────────────

    /// Note which process we started, so a later launch can find it.
    private func recordPid(_ pid: pid_t) {
        try? "\(pid)".write(to: paths.serverPid, atomically: true, encoding: .utf8)
    }

    private func forgetPid() {
        try? FileManager.default.removeItem(at: paths.serverPid)
    }

    /// Take charge of a server this app started and then failed to stop.
    ///
    /// Called once at launch. The app can die without running any cleanup —
    /// Force Quit is SIGKILL, a crash gives no notice — and the server is in its
    /// own process group precisely so that its children die with *it*, which
    /// also means it does not die with us. It keeps running, keeps the port, and
    /// the next Start fails with "port already in use" pointing at nothing.
    ///
    /// Adopted rather than killed: it is serving, and a user who force-quit the
    /// menu bar app did not ask for their generation to be interrupted. Watching
    /// it makes Stop and Restart work on it as if we had started it.
    func adoptOrphan(config: ServerConfig) {
        guard state == .stopped, let pid = recordedPid() else { return }
        guard Self.isAlive(pid) else {
            forgetPid()
            return
        }
        self.pid = pid
        watch(pid)
        state = .running(port: config.port)
        lastExit = nil
        onChange()
    }

    private func recordedPid() -> pid_t? {
        guard
            let text = try? String(contentsOf: paths.serverPid, encoding: .utf8),
            let value = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines))
        else { return nil }
        return value
    }

    /// Whether that pid is still a process **we may signal**.
    ///
    /// `kill(pid, 0)` asks exactly that. It cannot tell us the process is still
    /// *our server* rather than a recycled pid — so this is paired with the
    /// health check the caller runs before trusting the adoption, and with the
    /// group signal, which a non-leader recycled pid would not receive.
    nonisolated static func isAlive(_ pid: pid_t) -> Bool {
        kill(pid, 0) == 0 && getpgid(pid) == pid
    }

    // ── Watching ───────────────────────────────────────────────────────────

    private func watch(_ pid: pid_t) {
        let source = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit)
        source.setEventHandler { [weak self] in
            // `DispatchSource` fires on its own queue; everything below touches
            // main-actor state.
            Task { @MainActor in self?.reap(pid) }
        }
        source.resume()
        reaper = source
    }

    private func reap(_ pid: pid_t) {
        guard self.pid == pid else { return }
        var status: Int32 = 0
        waitpid(pid, &status, WNOHANG)
        self.pid = nil
        reaper?.cancel()
        reaper = nil
        forgetPid()

        if state != .stopping {
            // It went away on its own. Saying so is the difference between a
            // tray that reports a crash and one that just goes quiet.
            lastExit = Self.describe(status: status)
        }
        state = .stopped
        onChange()
    }

    nonisolated static func describe(status: Int32) -> String {
        if status & 0x7F == 0 {
            let code = (status >> 8) & 0xFF
            return code == 0 ? "the server exited" : "the server exited with code \(code)"
        }
        return "the server was terminated by signal \(status & 0x7F)"
    }

    // ── Stopping ───────────────────────────────────────────────────────────

    /// SIGTERM the group, wait, then SIGKILL it.
    ///
    /// The wait is `grace + 8s`: `grace` is what uvicorn was told to spend
    /// draining in-flight connections, and the rest is the margin for it to
    /// actually finish afterwards.
    func stop(grace: TimeInterval) async {
        guard let pid, state != .stopped else {
            state = .stopped
            onChange()
            return
        }
        state = .stopping
        onChange()

        Self.signalGroup(pid, SIGTERM)
        let deadline = Date().addingTimeInterval(grace + 8)
        while Date() < deadline {
            if self.pid == nil { return }  // `reap` settled it
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        if self.pid != nil {
            Self.signalGroup(pid, SIGKILL)
            // Give the reaper a moment; if it never fires the state is settled
            // below regardless, because a killed group is gone either way.
            try? await Task.sleep(nanoseconds: 300_000_000)
            reap(pid)
        }
    }

    /// Terminate without waiting, for application exit.
    ///
    /// Synchronous on purpose: `applicationWillTerminate` has no await to give,
    /// and a child left behind becomes an orphan under launchd.
    func killNow() {
        guard let pid else { return }
        Self.signalGroup(pid, SIGKILL)
        forgetPid()
        self.pid = nil
        state = .stopped
    }

    /// Signal the whole process group (`-pgid`).
    ///
    /// `posix_spawnattr_setpgroup(&attr, 0)` makes the child its own group
    /// leader, so its pid is also its process-group id.
    nonisolated static func signalGroup(_ pid: pid_t, _ signal: Int32) {
        kill(-pid, signal)
    }
}

struct QDSError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// Bridge `[String]` to the `char *const *` `posix_spawn` expects.
///
/// The pointers are valid only inside `body`, which is why this is a scoped
/// helper rather than a conversion returning an array: a `strdup`'d array that
/// outlived the call would either leak or dangle.
func withCStrings<T>(_ strings: [String], _ body: (UnsafePointer<UnsafeMutablePointer<CChar>?>) -> T)
    -> T
{
    var pointers: [UnsafeMutablePointer<CChar>?] = strings.map { strdup($0) }
    pointers.append(nil)
    defer { for pointer in pointers where pointer != nil { free(pointer) } }
    return pointers.withUnsafeBufferPointer { body($0.baseAddress!) }
}
