import AppKit
import Foundation
import Observation
import ServiceManagement

/// Everything the menu shows, and the only place the three sources are joined.
///
/// The sources stay separate on the way in — the supervisor knows whether a
/// process exists, `/health` knows whether it serves, `/v1/progress` knows what
/// it is doing, `/admin/jobs` knows about downloads — because they answer
/// different questions and can disagree. A supervised process that has stopped
/// serving is a real state, and collapsing them into one "status" earlier would
/// make it unreportable.
@MainActor
@Observable
final class MenuModel {
    private(set) var supervisor: Supervisor!
    private(set) var bootstrap: Bootstrap!

    var health: Health?
    var progress: Progress?
    var job: JobStatus?
    /// The last thing that went wrong, shown until it is superseded. Actions are
    /// rare and their failures are the only thing that reports them.
    var note: String?
    var busy: String?

    private let paths = Paths()
    private var config = ServerConfig()
    private var client: ServerClient!
    private var pollTask: Task<Void, Never>?
    private var progressTask: Task<Void, Never>?

    init() {
        supervisor = Supervisor(paths: paths, onChange: { [weak self] in self?.onSupervisorChange() })
        bootstrap = Bootstrap(paths: paths, onChange: { [weak self] in self?.touch() })
        reloadConfig()
        client = ServerClient(config: config, tokenFile: paths.adminToken)
        bootstrap.refresh()
    }

    /// The children's state, mirrored into properties `@Observable` can see.
    ///
    /// `Supervisor` and `Bootstrap` are plain classes, so reading
    /// `supervisor.state` from a view registers no dependency and the menu never
    /// re-renders when it changes. A `revision` counter was not enough either —
    /// nothing *read* it, so it tracked nothing. The only thing that works is
    /// for the view to read a property of this object, so the state is copied
    /// here whenever it moves.
    ///
    /// This was hiding two bugs, not one: an adopted server showed as stopped,
    /// and a server that died on its own was never reported at all. Both looked
    /// fine in testing because clicking a menu command also sets `busy`, which
    /// *is* tracked, so the redraw it caused made the rest appear to work.
    private(set) var supervisorState: Supervisor.State = .stopped
    private(set) var bootstrapState: Bootstrap.State = .absent
    private(set) var lastExit: String?

    private func touch() {
        supervisorState = supervisor.state
        bootstrapState = bootstrap.state
        lastExit = supervisor.lastExit
    }

    private func onSupervisorChange() {
        touch()
        if case .running = supervisor.state {
            startPolling()
        } else {
            stopPolling()
            health = nil
            progress = nil
            job = nil
        }
    }

    /// Re-read the file the *server* owns. Only connection facts are taken from
    /// it — the port to reach, the key to present, the grace to allow.
    func reloadConfig() {
        config = ServerConfig.read(at: paths.config)
        Task { await client?.update(config: config) }
    }

    // ── Lifetime ───────────────────────────────────────────────────────────

    var isReady: Bool {
        if case .ready = bootstrapState { return true }
        return false
    }

    /// The installed server is not the one this app carries.
    ///
    /// Distinct from "not installed": one means there is nothing to run, the
    /// other means what runs is the wrong build. Rebuilding without bumping the
    /// version makes the second the common case during development.
    var updateAvailable: Bool {
        if case .outdated = bootstrapState { return true }
        return false
    }

    var startLabel: String {
        switch bootstrapState {
        case .ready: return "Start Server"
        case .outdated: return "Update & Start Server"
        default: return "Install & Start Server"
        }
    }

    func startServer() {
        guard isReady else {
            Task { await bootstrap.install(); if isReady { startServer() } }
            return
        }
        act("Starting…") {
            self.reloadConfig()
            await self.client.update(config: self.config)
            try await self.supervisor.start(config: self.config)
        }
    }

    /// Install the bundled wheel over the running server, then bring it back.
    ///
    /// Stop first: `uv tool install --force` replaces the very files the running
    /// process is executing from, and a Python process whose `site-packages` is
    /// swapped underneath it fails in ways that are hard to read.
    func updateServer() {
        act("Updating…") {
            await self.supervisor.stop(grace: self.config.shutdownGrace)
            await self.bootstrap.install()
            guard self.isReady else {
                throw QDSError("The update did not complete; the server was left stopped.")
            }
            self.reloadConfig()
            await self.client.update(config: self.config)
            try await self.supervisor.start(config: self.config)
        }
    }

    func stopServer() {
        act("Stopping…") { await self.supervisor.stop(grace: self.config.shutdownGrace) }
    }

    func restartServer() {
        act("Restarting…") {
            await self.supervisor.stop(grace: self.config.shutdownGrace)
            self.reloadConfig()
            await self.client.update(config: self.config)
            try await self.supervisor.start(config: self.config)
        }
    }

    func cancelGeneration() {
        act("Cancelling…") { _ = try await self.client.cancelGeneration() }
    }

    func freeMemory() {
        act("Releasing…") { _ = try await self.client.unload() }
    }

    func openDashboard() {
        NSWorkspace.shared.open(config.baseURL.appendingPathComponent("dashboard/"))
    }

    func openPlayground() {
        NSWorkspace.shared.open(config.baseURL.appendingPathComponent("playground/"))
    }

    /// One slot, because the menu offers one action at a time: every command
    /// here is disabled while another runs, so a second result could only ever
    /// overwrite a message nobody had read.
    private func act(_ label: String, _ body: @escaping () async throws -> Void) {
        guard busy == nil else { return }
        busy = label
        note = nil
        Task {
            do { try await body() } catch { note = error.localizedDescription }
            busy = nil
        }
    }

    // ── Reading ────────────────────────────────────────────────────────────

    private func startPolling() {
        guard pollTask == nil else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.health = try? await self.client.health()
                // Only while something is running: an idle job polled every two
                // seconds for hours is a request storm reporting nothing.
                if self.job?.isActive == true || self.health != nil {
                    self.job = try? await self.client.jobs()
                }
                try? await Task.sleep(for: .seconds(5))
            }
        }
        startProgressStream()
    }

    private func startProgressStream() {
        guard progressTask == nil else { return }
        progressTask = Task { [weak self] in
            var attempt = 0
            while !Task.isCancelled {
                guard let self, case .running = self.supervisor.state else { return }
                do {
                    for try await frame in await self.client.progressFrames() {
                        attempt = 0
                        self.progress = frame
                    }
                } catch {
                    // A dropped stream is not worth a message: the tray still
                    // has `/health`, and the retry below is bounded.
                }
                if Task.isCancelled { return }
                self.progress = nil
                try? await Task.sleep(for: reconnectDelay(attempt: attempt))
                attempt += 1
            }
        }
    }

    private func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
        progressTask?.cancel()
        progressTask = nil
    }

    // ── What the menu says ─────────────────────────────────────────────────

    /// The one-line status, most specific fact first.
    ///
    /// Ordered by what a person opening the menu wants to know: whether
    /// something is happening, then whether it is up, then why it is not.
    var statusLine: String {
        if let busy { return busy }
        switch bootstrapState {
        case .installing(let step): return step
        case .failed(let reason): return reason
        case .absent: return "Server not installed"
        // Named rather than folded into "not installed": the server *is* there
        // and answering; what is stale is which build it is. Saying "not
        // installed" about a working server sends people looking for the wrong
        // problem.
        case .outdated(let installed): return "Update available - \(installed) installed"
        default: break
        }
        switch supervisorState {
        case .stopped:
            return lastExit.map { "Stopped - \($0)" } ?? "Stopped"
        case .starting: return "Starting…"
        case .stopping: return "Stopping…"
        case .running(let port):
            if let job, job.isActive {
                return jobLine(job)
            }
            if let progress, progress.isGenerating {
                return progress.total > 0
                    ? "Generating \(progress.step)/\(progress.total)"
                    : "Loading \(progress.model ?? "model")…"
            }
            if health?.status == "config_error" {
                return "Recovery mode - fix the configuration"
            }
            if let warm = health?.loaded_model { return "Running :\(port) - \(warm) warm" }
            return "Running :\(port)"
        }
    }

    private func jobLine(_ job: JobStatus) -> String {
        let what = job.kind == "prequantize" ? "Converting" : "Downloading"
        return "\(what) \(job.target ?? "")"
    }

    var isRunning: Bool {
        if case .running = supervisorState { return true }
        return false
    }

    var canCancelGeneration: Bool { isRunning && progress?.isGenerating == true }
    var hasWarmModel: Bool { isRunning && health?.loaded_model != nil }

    // ── Launch at login ────────────────────────────────────────────────────

    var launchesAtLogin: Bool {
        get { SMAppService.mainApp.status == .enabled }
        set {
            do {
                if newValue { try SMAppService.mainApp.register() }
                else { try SMAppService.mainApp.unregister() }
                touch()
            } catch {
                note = "Could not change the login item: \(error.localizedDescription)"
            }
        }
    }

    /// Take charge of a server left behind by a previous run of this app.
    ///
    /// Then confirm it: adoption is based on a recorded pid, which the operating
    /// system may have recycled. `/health` answering on the configured port is
    /// what turns "a process with that pid exists" into "our server is running".
    func adoptOrphan() {
        supervisor.adoptOrphan(config: config)
        guard isRunning else { return }
        Task {
            if (try? await client.health()) == nil {
                // Something holds the pid but does not serve. Reporting it as
                // running would make Stop and Restart act on a stranger.
                await supervisor.stop(grace: config.shutdownGrace)
            }
        }
    }

    /// Stop the server before the app goes.
    ///
    /// Quitting takes the server with it, and the menu says so. The alternative
    /// — leaving it running — means a server nothing on screen represents,
    /// holding a port and unified memory, which is precisely the orphan this
    /// app exists to prevent.
    func terminate() {
        supervisor.killNow()
    }
}
