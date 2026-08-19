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
        guard let wheel = Self.bundledWheel, let uv = Self.bundledUV,
            let version = Self.version(ofWheel: wheel),
            let digest = Self.digest(ofWheel: wheel)
        else {
            state = .failed("This build carries no server wheel.")
            onChange()
            return
        }

        state = .installing("Preparing…")
        onChange()

        do {
            try paths.ensure()
            let lock = try FileLock(at: paths.installLock)
            guard lock.acquired else {
                state = .failed("Another QDS is already installing the server.")
                onChange()
                return
            }
            defer { lock.release() }

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
                ])

            guard result.status == 0 else {
                state = .failed(Self.explain(result))
                onChange()
                return
            }

            let record: [String: Any] = [
                "version": version, "digest": digest, "complete": true,
            ]
            try JSONSerialization.data(withJSONObject: record).write(to: paths.installRecord)
            state = .ready(version: version)
        } catch {
            state = .failed(error.localizedDescription)
        }
        onChange()
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

    private func run(
        _ executable: URL, _ arguments: [String], environment: [String: String]
    ) async throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()

        // Read before waiting: a full pipe buffer with nobody draining it is how
        // a subprocess hangs forever instead of finishing.
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
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
