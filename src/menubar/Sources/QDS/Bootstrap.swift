import CryptoKit
import Foundation

/// Installing the server, from resources the app carries.
///
/// The app bundles two things: a `uv` binary and the `qds` wheel. First launch
/// runs `uv tool install <wheel>` into the app's own data directory, and that is
/// the whole installation — no PyPI, no network for the package itself, so a
/// first run works on a machine that has never seen this project.
///
/// (uv still downloads a managed CPython the first time, which does need the
/// network. That is one download, cached under the app's data directory, and the
/// tray says so when it fails rather than reporting a mysterious failure.)
///
/// This replaces the Tauri app's copy-the-project-then-`uv sync`: the wheel is
/// now the single distribution artifact, so there is no project tree to stage,
/// no lockfile to keep in step, and the same bytes install here as would install
/// from PyPI later.
@MainActor
final class Bootstrap {
    enum State: Equatable {
        /// Nothing installed, or an install that never finished.
        case absent
        /// Something is installed, but it is not what this bundle carries.
        case outdated(installed: String)
        case installing(String)
        case ready(version: String)
        case failed(String)
    }

    private(set) var state: State = .absent

    /// How far the install currently running has got, and what it is doing.
    ///
    /// Reset at the start of each `install()`, never carried between runs: a
    /// second attempt after a failure must not open showing the first attempt's
    /// bar half full.
    private(set) var progress = InstallProgress()

    /// The installer's output, as it arrives.
    ///
    /// Bounded, because it is a window's backing store rather than a log file:
    /// an install prints ~130 lines, and the failure this cap prevents is not
    /// this one but a future installer that decides to emit a line per file.
    /// The server's own log is `paths.serverLog`; this is only what the setup
    /// window shows.
    private(set) var transcript: [String] = []

    /// Lines kept in `transcript` before the oldest are dropped.
    static let transcriptLimit = 2000

    /// True from the moment `install()` starts until it settles, whatever the
    /// outcome. What decides whether the setup window is on screen.
    ///
    /// Distinct from `state == .installing`: that one also describes the step
    /// label, and a failure moves it to `.failed` while the window must stay up
    /// to *show* the failure.
    private(set) var isPresenting = false

    /// The child currently running, so it can be signalled.
    private var running: Process?

    /// Set when the user asked to stop, so the failure that follows is reported
    /// as their choice rather than as a broken install.
    private var cancelled = false

    /// Set when an install was refused because another one holds the lock.
    ///
    /// Separate from `state`, which belongs to the install that is *running*:
    /// the whole point of the refusal path is that it says nothing about, and
    /// changes nothing in, the run already in progress. Read and cleared by
    /// `MenuModel`, which shows it the way it shows any other command failure.
    var refusal: String?

    /// Set when the install finished but the rewriter's weights did not arrive.
    /// Informational: the server is ready, and Enhance still works — it just
    /// pays for the download the first time it is pressed.
    private(set) var rewriterNote: String?

    /// The same, for the upscalers. A separate line because they are separate
    /// facts: one can arrive and the other not, and a single note would have to
    /// pick which failure to describe.
    private(set) var upscalerNote: String?

    private let paths: Paths
    private let onChange: () -> Void

    init(paths: Paths, onChange: @escaping () -> Void) {
        self.paths = paths
        self.onChange = onChange
    }

    /// What the bundle carries. Absent only in a `swift run` from a checkout,
    /// where there is no `.app` around the binary.
    nonisolated static var bundledWheel: URL? {
        Bundle.main.url(forResource: nil, withExtension: "whl", subdirectory: "Resources")
            ?? Bundle.main.resourceURL.flatMap { resources in
                (try? FileManager.default.contentsOfDirectory(
                    at: resources, includingPropertiesForKeys: nil))?
                    .first { $0.pathExtension == "whl" }
            }
    }

    nonisolated static var bundledUV: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("uv")
    }

    /// The version the bundle would install, read from the wheel's filename.
    ///
    /// `qds-2.0.0-py3-none-any.whl` — the filename is the wheel's own identity,
    /// not a second place the version is written down.
    nonisolated static func version(ofWheel url: URL) -> String? {
        let parts = url.deletingPathExtension().lastPathComponent.split(separator: "-")
        return parts.count >= 2 ? String(parts[1]) : nil
    }

    /// A wheel's identity: the hash of its bytes.
    ///
    /// **Not its version.** The version is what the wheel is *called*, and
    /// during development it does not change between builds — so an install
    /// recorded as `2.0.0` and a freshly built `2.0.0` carrying a different
    /// server and a different dashboard compared equal, and the app reported
    /// "ready" while running last week's code. The bytes are what was actually
    /// installed, so the bytes are what gets recorded.
    ///
    /// (The Tauri app this replaced hashed its payload for exactly this reason.
    /// Moving to `uv tool install` dropped the property; this restores it.)
    nonisolated static func digest(ofWheel url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// What the record says is installed: `(digest, version)`, or `nil`.
    ///
    /// Read from the *record*, not inferred from the binary: `uv tool install`
    /// can be interrupted, and a `qds` on disk from a half-finished install
    /// would otherwise read as a complete one.
    func installed() -> (digest: String?, version: String)? {
        guard
            FileManager.default.isExecutableFile(atPath: paths.qds.path),
            let data = try? Data(contentsOf: paths.installRecord),
            let record = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            record["complete"] as? Bool == true,
            let version = record["version"] as? String
        else { return nil }
        // A record written before digests existed carries no `digest`. That is
        // not "nothing is installed" — something is, and it answers — it is
        // "this cannot be shown to be the build we carry". Returning `nil`
        // instead would make the menu say "Server not installed" about a working
        // server, which sends people looking for the wrong problem.
        return (record["digest"] as? String, version)
    }

    func refresh() {
        guard
            let wheel = Self.bundledWheel,
            Self.version(ofWheel: wheel) != nil,
            let wanted = Self.digest(ofWheel: wheel)
        else {
            state = .failed("This build carries no server wheel. Run `make build-app`.")
            onChange()
            return
        }
        switch installed() {
        case let .some(record) where record.digest == wanted:
            state = .ready(version: record.version)
        case let .some(record):
            // Same version number is the *common* case here, not the odd one:
            // rebuilding without bumping the version is what development is.
            // A record with no digest at all lands here too — installed, but
            // not shown to be this build.
            state = .outdated(installed: record.version)
        case .none:
            state = .absent
        }
        onChange()
    }

    /// Install the bundled wheel, unless another process is already doing it.
    ///
    /// Single-flight through an `flock` on a file, not through a flag in memory:
    /// two copies of the app can be launched, and the failure mode of two
    /// concurrent `uv tool install`s into one directory is a half-installed
    /// tool that neither of them notices.
    func install() async {
        // Take the lock before *anything* else, including the wheel check.
        //
        // Ordering discovered by the test below rather than by inspection: the
        // wheel guard also writes `state`, so with it above the lock a refused
        // second install could still overwrite a healthy run's state — the very
        // defect the lock-first ordering exists to prevent, just through a
        // different branch. Nothing may report anything about this instance
        // until it owns the run.
        let lock: FileLock
        do {
            try paths.ensure()
            lock = try FileLock(at: paths.installLock)
        } catch {
            state = .failed(error.localizedDescription)
            onChange()
            return
        }
        guard lock.acquired else {
            // Deliberately does not touch `progress`, `transcript`,
            // `isPresenting` or `state`: another install owns those right now,
            // and overwriting them is precisely the defect this ordering fixes.
            // The refusal is reported through the same channel the menu uses for
            // any other rejected command.
            refusal = "Another QDS is already installing the server."
            onChange()
            return
        }
        defer { lock.release() }

        guard let wheel = Self.bundledWheel, let uv = Self.bundledUV,
            let version = Self.version(ofWheel: wheel),
            let digest = Self.digest(ofWheel: wheel)
        else {
            state = .failed("This build carries no server wheel.")
            onChange()
            return
        }

        // Only now, with the run exclusively ours, does the state reset: a
        // retry after a failure must not open showing the previous attempt's
        // bar and log.
        progress = InstallProgress()
        transcript = []
        cancelled = false
        isPresenting = true
        state = .installing("Preparing…")
        onChange()
        defer {
            isPresenting = false
            onChange()
        }

        do {
            // The record is cleared first: if this run is interrupted, the next
            // launch has to see an incomplete install rather than a stale claim
            // that everything is fine.
            try? FileManager.default.removeItem(at: paths.installRecord)

            state = .installing("Installing the server (this takes a minute on first run)…")
            onChange()

            let result = try await run(
                uv, ["tool", "install", "--force", "--python", "3.12", wheel.path],
                environment: [
                    "UV_TOOL_DIR": paths.tools.path,
                    "UV_TOOL_BIN_DIR": paths.bin.path,
                    "UV_PYTHON_INSTALL_DIR": paths.pythons.path,
                    "UV_CACHE_DIR": paths.uvCache.path,
                    "HOME": NSHomeDirectory(),
                    "PATH": "/usr/bin:/bin",
                ],
                onLine: { [weak self] line in self?.absorb(line) })

            guard result.status == 0 else {
                // A cancel makes the child fail, and reporting *that* as a
                // broken install would blame the product for the user's choice.
                state = cancelled
                    ? .failed("Setup was stopped. Nothing was installed.")
                    : .failed(Self.explain(result))
                onChange()
                return
            }

            let record: [String: Any] = [
                "version": version, "digest": digest, "complete": true,
            ]
            try JSONSerialization.data(withJSONObject: record).write(to: paths.installRecord)

            // The install is complete at this point, and stays complete
            // whatever happens next. The optional weights are fetched *after*
            // the record is written, deliberately: the server is fully
            // functional without them, so an interrupted download must not
            // leave the install looking broken.
            //
            // Upscalers first: 42.5 MB against the rewriter's 2.2 GB, and the
            // rewriter is the phase carrying a Skip button.
            await fetchUpscalers()
            await fetchRewriter()

            state = .ready(version: version)
        } catch {
            state = .failed(error.localizedDescription)
        }
        onChange()
    }

    /// Stop whatever is running.
    ///
    /// Two different meanings, decided by which phase is running, because the
    /// install record has already been written by the time the rewriter starts:
    ///
    /// * during the install — a **cancel**. Nothing is installed, the record is
    ///   absent, and the next launch offers to install again.
    /// * during the rewriter fetch — a **skip**. The server is already installed
    ///   and fully functional; the enhancer downloads on first use instead, and
    ///   `rewriterNote` says so.
    ///
    /// `Process.terminate()`, never `kill(-pid, …)`, and the distinction is a
    /// safety property rather than a preference. A pid is only meaningful while
    /// its process is alive: between observing `isRunning` and calling `kill`,
    /// the child can exit, be reaped, and have its pid **recycled by the OS**.
    /// Signalling `-pid` at that point addresses whatever process group now owns
    /// that number — a stranger's, possibly the user's own work. `terminate()`
    /// signals through the `Process` object, which knows whether its child is
    /// still its child, so it cannot be aimed at a recycled pid.
    ///
    /// What that gives up is reaching grandchildren, and here it costs nothing:
    /// `Foundation.Process` does not make its child a group leader (which is
    /// exactly why `Supervisor` drops to `posix_spawn` for the *server*), so
    /// `kill(-pid, …)` would have signalled **this app's own process group** —
    /// the tray, and the running server with it. The old code guarded that with
    /// `getpgid(pid) == pid`, so the branch it protected was unreachable in
    /// practice and only the unsafe case could ever fire. `uv` and `qds fetch`
    /// both terminate their own children on SIGTERM.
    func cancel() {
        // The flag is set before the liveness check, deliberately: a stop that
        // races the child's own exit must still be reported as the user's
        // choice, not as "The server could not be installed: <last line>".
        cancelled = true
        guard let process = running, process.isRunning else { return }
        process.terminate()
    }

    /// Fold one line of installer output into the progress and the transcript.
    private func absorb(_ line: String) {
        transcript.append(line)
        if transcript.count > Self.transcriptLimit {
            transcript.removeFirst(transcript.count - Self.transcriptLimit)
        }
        progress.absorb(line)
        state = .installing(Self.label(for: progress))
        onChange()
    }

    /// Put this instance into the state a run in progress would be in.
    ///
    /// Exists so a test can assert what a *refused* second install may touch,
    /// which needs a first install's state to exist without a real four-minute
    /// `uv` run behind it. Named for what it is rather than hidden behind a
    /// plausible-looking API: a test seam is honest, a fake production method
    /// is not.
    func absorbForTesting(_ line: String) {
        absorb(line)
    }

    /// The one-line description of what is happening, for the window and the
    /// menu's status line.
    ///
    /// `nonisolated static` because it is pure — a function of the progress and
    /// nothing else — which is what lets a test assert the wording without a
    /// process, a pipe or a main actor.
    nonisolated static func label(for progress: InstallProgress) -> String {
        switch progress.phase {
        case .preparing:
            return "Preparing…"
        case .python:
            return "Downloading the Python runtime…"
        case .packages:
            // Three distinct situations, and `currentPackage == nil` covers two
            // of them. Both were found by replaying the real installer rather
            // than by inspection:
            //
            // * `Resolved` lands ~130ms before the first `Downloading`, so
            //   "nothing outstanding" at that point means downloads have not
            //   *started*.
            // * the last `Downloaded` also empties it, ~1.7s before `Prepared`
            //   arrives — so at that point it means they have *finished*.
            //
            // Treating the two alike made the label read
            // "Installing the server…" → "Downloading torch" → "Resolving
            // packages…" → "Installing the server…". A progress display that
            // walks backwards reads as a broken installer, which is the exact
            // impression this feature exists to remove.
            if let package = progress.currentPackage {
                return "Downloading \(package) — \(progress.packagesDone + 1) of \(progress.packagesTotal)"
            }
            // Anything announced at all means resolution is behind us.
            if progress.finishedDownloading || progress.packagesTotal > 0 {
                return "Installing the server…"
            }
            return "Resolving packages…"
        case .upscalers:
            return "Downloading the upscalers…"
        case .rewriter:
            return "Downloading the prompt enhancer…"
        }
    }

    /// Pull the upscalers' weights, best effort.
    ///
    /// Same shape and same reasoning as `fetchRewriter` below — an intent flag
    /// rather than catalogue keys in Swift, the server's own child environment
    /// so the files land in the cache the server reads, and a failure that is an
    /// informational line rather than `.failed`.
    ///
    /// Worth doing at install time for the same reason the rewriter is: leaving
    /// it to the first Upscale means a download behind a button press with no
    /// way to know it was coming. Unlike the rewriter it is nearly free — 42.5
    /// MB, about five seconds on a warm connection — which is why it is not
    /// skippable and does not need to be.
    private func fetchUpscalers() async {
        progress.beginUpscalers()
        state = .installing(Self.label(for: progress))
        onChange()
        do {
            try ServerConfig.seedIfMissing(at: paths.config)
            let config = ServerConfig.read(at: paths.config)
            let result = try await run(
                paths.qds, ["fetch", "--upscalers"],
                environment: Supervisor.childEnvironmentMap(paths: paths, config: config),
                onLine: { [weak self] line in self?.absorb(line) })
            if result.status != 0 {
                upscalerNote = "The upscalers will download the first time you use one."
            }
        } catch {
            upscalerNote = "The upscalers will download the first time you use one."
        }
        // Skipping *this* download must not cancel the next one.
        //
        // The flag is what makes a stop read as the user's choice, and it stays
        // set until cleared. Left set here, a Skip of the upscalers would arm
        // the rewriter fetch to be reported as cancelled too — and, worse, the
        // window's next Skip press would already find `cancelled` true. The two
        // fetches are independent choices, so each clears the flag it consumed.
        cancelled = false
    }

    /// Pull the prompt rewriter's weights, best effort.
    ///
    /// Python and mflux install themselves; this had to as well. Leaving it to
    /// the first Enhance meant a gigabyte downloading behind a button press,
    /// with no way to know it was coming.
    ///
    /// Three things here are load-bearing rather than incidental:
    ///
    /// * **`--rewriter`, never a catalogue key.** Which model this is belongs
    ///   to the server's catalogue. A key spelled out here would be catalogue
    ///   data duplicated into a second language, with no test able to keep the
    ///   two in step.
    /// * **`Supervisor.childEnvironment`, not this file's minimal one.**
    ///   `qds fetch` resolves its cache through `load_settings().apply_hf_home()`,
    ///   which reads `QDS_SERVER_CONFIG`. Without it the weights land in a
    ///   Hugging Face cache the server never looks at — a silent, expensive,
    ///   invisible failure. The config has to exist first, hence `seedIfMissing`.
    /// * **Best effort.** A failure here is an informational line, never
    ///   `.failed`. No network at install time is ordinary, and the feature
    ///   still works later: it downloads on first use and says so first.
    private func fetchRewriter() async {
        progress.beginRewriter()
        state = .installing(Self.label(for: progress))
        onChange()
        do {
            try ServerConfig.seedIfMissing(at: paths.config)
            let config = ServerConfig.read(at: paths.config)
            let result = try await run(
                paths.qds, ["fetch", "--rewriter"],
                environment: Supervisor.childEnvironmentMap(paths: paths, config: config),
                onLine: { [weak self] line in self?.absorb(line) })
            if result.status != 0 {
                rewriterNote = "The prompt enhancer will download the first time you use it."
            }
        } catch {
            rewriterNote = "The prompt enhancer will download the first time you use it."
        }
        // A skip here is not a cancelled install: the record is already written
        // and the server is ready. Clearing the flag keeps the *next* thing that
        // reads it — the failure branch of a subsequent run — honest.
        cancelled = false
    }

    /// Turn uv's output into something worth showing.
    ///
    /// The common first-run failure is no network while it fetches a managed
    /// CPython, and "exit status 2" is a poor way to say that.
    nonisolated static func explain(_ result: (status: Int32, output: String)) -> String {
        let output = result.output
        if output.contains("Failed to download") || output.contains("error sending request")
            || output.contains("dns error")
        {
            return
                "The server could not be installed: downloading Python needs a network connection. "
                + "Reconnect and try again."
        }
        let lastLine =
            output.split(separator: "\n").last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty })
            .map(String.init) ?? "exit status \(result.status)"
        return "The server could not be installed: \(lastLine)"
    }

    /// Run a child, reporting its output line by line as it arrives.
    ///
    /// - Parameter onLine: called on the main actor for each line, while the
    ///   process is still running. `nil` for callers that only want the result.
    ///
    /// Not `private`: the ordered main-actor hand-off below is a correctness
    /// property (`InstallProgress` discards a completion that arrives before its
    /// announcement), and an independent review pointed out it had no witness at
    /// all — `drain` was tested directly, so reverting the hand-off to a
    /// per-line `Task { @MainActor in … }` broke nothing. `internal` lets the
    /// test target drive a real child through the real path.
    @discardableResult
    func run(
        _ executable: URL, _ arguments: [String], environment: [String: String],
        onLine: (@MainActor @Sendable (String) -> Void)? = nil
    ) async throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()

        // Held so `cancel()` can reach it while it runs.
        running = process
        defer { running = nil }

        let handle = pipe.fileHandleForReading

        // Lines cross to the main actor through one stream, in order.
        //
        // The obvious spelling — `Task { @MainActor in onLine(line) }` per
        // line — is wrong: each `Task` is scheduled independently and Swift
        // guarantees no ordering between them, so a burst of twenty
        // announcements (which is exactly what uv emits, within ~10ms) could be
        // applied in any order. `InstallProgress` is order-sensitive — a
        // `Downloaded` applied before its `Downloading` is dropped as
        // unannounced — so that would silently corrupt both the count and the
        // log the user is reading.
        let (lines, feed) = AsyncStream.makeStream(of: String.self)

        let pump = Task { @MainActor in
            for await line in lines { onLine?(line) }
        }

        // The read happens off the main actor, and that is not a detail.
        //
        // `availableData` *blocks* until the child writes something — which,
        // during a 106 MB download, is seconds at a time. Draining on the main
        // actor would freeze the menu and the very window this feature exists to
        // draw: the app would look more hung than it does today, not less.
        let collected = await Task.detached(priority: .utility) {
            let data = Self.drain(handle) { feed.yield($0) }
            feed.finish()
            return data
        }.value

        // Every line has been applied before this returns, so a caller that
        // inspects the progress afterwards sees the whole run rather than
        // whatever had happened to arrive.
        await pump.value

        // Only after EOF, so this no longer blocks on anything meaningful.
        await Task.detached(priority: .utility) { process.waitUntilExit() }.value
        return (process.terminationStatus, String(data: collected, encoding: .utf8) ?? "")
    }

    /// Read a pipe to EOF, calling `onLine` for each line as it arrives.
    ///
    /// Never `readDataToEndOfFile()`. That call is what made this feature
    /// impossible before: it returns only at EOF — i.e. once the process is
    /// already over — so there was nothing to report *during* the four minutes
    /// an install takes. Draining as it arrives keeps the property the old code
    /// actually needed (a full pipe buffer with nobody reading it is how a child
    /// hangs forever) while making the output observable in time to matter.
    ///
    /// `nonisolated static` so the blocking loop provably cannot run on the main
    /// actor, and so a test can drive it with a pipe of its own.
    nonisolated static func drain(
        _ handle: FileHandle, onLine: @Sendable (String) -> Void
    ) -> Data {
        var collected = Data()
        var pending = Data()

        // Split on \n *and* \r. uv writes plain newlines when its output is a
        // pipe, but a renderer that redraws in place uses carriage returns, and
        // a reader that split only on newlines would hold an entire run in one
        // unterminated "line" and report nothing at all.
        func flush(upTo index: Data.Index) {
            let line = String(decoding: pending[pending.startIndex..<index], as: UTF8.self)
            pending = pending[pending.index(after: index)...]
            if !line.trimmingCharacters(in: .whitespaces).isEmpty { onLine(line) }
        }

        while true {
            let chunk = handle.availableData
            if chunk.isEmpty { break }  // EOF
            collected.append(chunk)
            pending.append(chunk)
            while let index = pending.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
                flush(upTo: index)
            }
        }
        // A final line with no terminator — how a child that is killed mid-write
        // leaves the pipe.
        let tail = String(decoding: pending, as: UTF8.self)
        if !tail.trimmingCharacters(in: .whitespaces).isEmpty { onLine(tail) }
        return collected
    }
}

/// An advisory lock held for the lifetime of the value.
struct FileLock {
    private let descriptor: Int32
    let acquired: Bool

    init(at url: URL) throws {
        descriptor = open(url.path, O_CREAT | O_RDWR, 0o644)
        guard descriptor >= 0 else { throw QDSError("cannot open \(url.path)") }
        acquired = flock(descriptor, LOCK_EX | LOCK_NB) == 0
    }

    func release() {
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}
