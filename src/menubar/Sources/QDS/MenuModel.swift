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

    private let releaseCheck = ReleaseCheck()
    private let releaseCache = ReleaseCache()
    private let about = AboutWindowController()
    private let setup = SetupWindowController()
    private var releaseTask: Task<Void, Never>?

    init() {
        supervisor = Supervisor(paths: paths, onChange: { [weak self] in self?.onSupervisorChange() })
        bootstrap = Bootstrap(paths: paths, onChange: { [weak self] in self?.touch() })
        reloadConfig()
        client = ServerClient(config: config, tokenFile: paths.adminToken)
        bootstrap.refresh()
        restoreCachedRelease()
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

    /// The install's progress and output, mirrored for the same reason as the
    /// states above: `Bootstrap` is a plain class, so a view reading
    /// `bootstrap.progress` registers no dependency and never redraws.
    private(set) var installProgress = InstallProgress()
    private(set) var installTranscript: [String] = []
    /// Whether an install is running *or* has just settled with its window still
    /// up. Drives the setup window and the menu item that reopens it.
    private(set) var isInstalling = false

    /// Counts installs, so a delayed action can tell whether the run it was
    /// started for is still the current one.
    private var installGeneration = 0

    private func touch() {
        supervisorState = supervisor.state
        bootstrapState = bootstrap.state
        lastExit = supervisor.lastExit
        installProgress = bootstrap.progress
        installTranscript = bootstrap.transcript
        let wasInstalling = isInstalling
        isInstalling = bootstrap.isPresenting
        // A refused install says nothing about the run already in progress, so
        // it is reported the way any other rejected command is — as a note —
        // and never as a state that would disturb the running install.
        if let refused = bootstrap.refusal {
            bootstrap.refusal = nil
            note = refused
        }
        // A new run: count it, and forget that a *previous* window was
        // dismissed — the user closing the last install's window did not ask to
        // never see another one.
        if isInstalling && !wasInstalling {
            installGeneration += 1
            setup.reset()
        }
        syncSetupWindow()
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
            // Through `act`, like every other command, so a second click while
            // an install is running is refused rather than starting a *second*
            // `uv tool install` into the same directory. `flock` stops the two
            // from corrupting the install, but the loser still resets the shared
            // progress and transcript out from under the window — the second
            // click would appear to restart the first install from zero.
            //
            // This was reachable before the setup window existed; the window is
            // what makes an install long enough and visible enough to click at.
            act("Installing…") {
                await self.bootstrap.install()
                guard self.isReady else {
                    throw QDSError("The install did not complete; the server was not started.")
                }
                self.reloadConfig()
                await self.client.update(config: self.config)
                try await self.supervisor.start(config: self.config)
            }
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

    // ── The setup window ───────────────────────────────────────────────────

    /// Open the window when an install starts, close it once it has finished
    /// *and* succeeded.
    ///
    /// Only on success: a failure leaves it up, because the failure and its log
    /// are the whole reason the window exists. A window that vanished on the
    /// error would put the user back where they started, staring at a menu.
    private func syncSetupWindow() {
        if isInstalling {
            setup.present(model: self)
            return
        }
        guard setup.isOpen else { return }
        if case .ready = bootstrapState {
            // A beat, so "QDS is ready" and the four ticks are actually seen
            // rather than flashing past on the way out.
            //
            // Generation-stamped: an update started during the wait gets its own
            // window, and this timer must not close *that* one. Comparing a
            // counter rather than re-reading `isInstalling` because the second
            // install could also have finished by then, which would look
            // identical and close a window the user is still reading.
            let generation = installGeneration
            Task {
                try? await Task.sleep(for: .seconds(1.6))
                guard self.installGeneration == generation, !self.isInstalling else { return }
                self.setup.close()
            }
        }
    }

    /// Bring the window back after it was closed mid-install.
    func showSetupProgress() {
        setup.reveal(model: self)
    }

    func closeSetupWindow() {
        setup.close()
    }

    /// Stop the install, or skip the enhancer — `Bootstrap.cancel` decides which
    /// by looking at the phase, since only it knows whether the record is
    /// already written.
    func cancelInstall() {
        bootstrap.cancel()
    }

    /// Run the whole thing again from the top, after a failure.
    ///
    /// No guard of its own: `act` already refuses while another command runs,
    /// and a second one here would be a second authority on the same question.
    func retryInstall() {
        act("Installing…") {
            await self.bootstrap.install()
            guard self.isReady else {
                throw QDSError("The install did not complete.")
            }
            self.reloadConfig()
            await self.client.update(config: self.config)
            try await self.supervisor.start(config: self.config)
        }
    }

    /// The installer's output on the pasteboard, for a bug report.
    func copyInstallLog() {
        let text = installTranscript.joined(separator: "\n")
        guard !text.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
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

    // ── About, and the release check ───────────────────────────────────────

    /// What the GitHub release check found. Distinct from `updateAvailable`,
    /// which is about the *server* — see `newVersion` below.
    private(set) var releaseState: ReleaseState = .unknown

    /// The release to offer, or nothing.
    ///
    /// **Not the same fact as `updateAvailable`.** That one compares the wheel
    /// this app carries against the server installed on disk, and is satisfied
    /// by "Update Server & Restart" using bytes already on the machine. This one
    /// says a newer QDS was published, and the only thing to do about it is open
    /// a browser. Two things called "update" in one menu would make both
    /// unreadable, so this is worded as a version and that one is not.
    var newVersion: Release? {
        if case .available(let release) = releaseState { return release }
        return nil
    }

    func showAbout() {
        about.show(model: self)
        // Opening the window is the one moment somebody is definitely looking at
        // the answer, so a cached one that has gone stale is refreshed now
        // rather than at the next timer tick.
        checkForNewVersion()
    }

    func openNewVersion() {
        guard let release = newVersion else { return }
        NSWorkspace.shared.open(release.url)
    }

    /// Show what the last run found, before asking anything.
    ///
    /// The comparison is redone here rather than the verdict being cached: what
    /// is stored is the *release*, and whether it is newer depends on which
    /// version this build is — which changes when the app is updated. Caching
    /// "an update is available" instead would keep offering 2.3.0 to a 2.3.0
    /// that had just installed it.
    private func restoreCachedRelease() {
        guard let release = releaseCache.release else { return }
        releaseState = Self.verdict(for: release, current: Product.version)
    }

    /// Ask GitHub, at most once a day, and never twice at once.
    ///
    /// - Parameter force: skip the interval, for a check the user asked for.
    func checkForNewVersion(force: Bool = false) {
        // Nothing to compare against: a checkout build has no Info.plist and so
        // no version. Asking anyway would spend a request to learn nothing.
        guard Product.version != nil else {
            releaseState = .unknown
            return
        }
        guard releaseTask == nil else { return }
        guard force || ReleaseCheck.isDue(lastChecked: releaseCache.lastChecked) else { return }

        // Only *shown* as checking when there is nothing to show instead: a
        // cached answer replaced by "Checking…" flickers for no reason.
        if case .unknown = releaseState { releaseState = .checking }
        if case .failed = releaseState { releaseState = .checking }

        releaseTask = Task { [weak self] in
            guard let self else { return }
            defer { self.releaseTask = nil }
            do {
                let release = try await self.releaseCheck.latest()
                // The timestamp is written on success only. A failed check that
                // recorded one would make an offline day count as a day checked,
                // and the news would arrive up to 24 hours late.
                self.releaseCache.store(release.isOfferable ? release : nil)
                self.releaseState = Self.verdict(for: release, current: Product.version)
            } catch {
                // Silent by design: `note` is for actions the user took, and an
                // unreachable GitHub is not a failure of anything they did. The
                // window says so; the menu simply does not gain an item.
                self.releaseState = .failed
            }
        }
    }

    /// Newer, older, or not offerable — with the version this build claims.
    ///
    /// `nonisolated` because it is pure: it reads its two arguments and touches
    /// nothing else. That is what lets the ordering be tested directly, with no
    /// main actor, no network, no bundle and no clock — the comparison is the
    /// part that can be silently wrong, so it has to be the part a test can
    /// reach.
    nonisolated static func verdict(for release: Release, current: String?) -> ReleaseState {
        // A draft, a prerelease, or a tag that is not a version. Not "current":
        // nothing was established about which is newer.
        guard release.isOfferable, let published = release.version else { return .unknown }
        guard let current = current.flatMap(Version.init) else { return .unknown }
        return published > current ? .available(release) : .current
    }

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
