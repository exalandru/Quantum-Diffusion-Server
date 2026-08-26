import XCTest

@testable import QDS

/// What can be tested without a menu bar.
///
/// The parts with decisions in them: where things live, what is read out of a
/// configuration this app does not own, how a raw exit status is described, and
/// how the bundled wheel's identity is derived. The rest of the app is a status
/// line over three pollers and a process — exercised by launching it, which is
/// what the manual smoke in the README covers.
final class PathsTests: XCTestCase {
    func testEverythingLivesUnderOneBundleIdentifiedDirectory() {
        let paths = Paths(data: URL(fileURLWithPath: "/tmp/qds-test"))
        XCTAssertEqual(paths.config.path, "/tmp/qds-test/server-config.json")
        XCTAssertEqual(paths.qds.path, "/tmp/qds-test/bin/qds")
        // The install record is a *sibling* of the tool directory on purpose:
        // `uv tool install` replaces that directory wholesale, so a marker kept
        // inside it could not survive the interruption it exists to describe.
        XCTAssertFalse(paths.installRecord.path.hasPrefix(paths.tools.path))
    }

    func testTheDefaultIsTheApplicationsOwnSupportDirectory() {
        let paths = Paths()
        XCTAssertTrue(paths.data.path.contains("Application Support"))
        XCTAssertTrue(paths.data.path.hasSuffix(Paths.bundleID))
    }
}

final class ServerConfigTests: XCTestCase {
    private func write(_ json: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-config-\(UUID().uuidString).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func testConnectionFactsAreReadFromTheFile() throws {
        let url = try write(
            #"{"server": {"host": "127.0.0.1", "port": 9123, "api_key": "k", "shutdown_grace_s": 25}}"#
        )
        let config = ServerConfig.read(at: url)
        XCTAssertEqual(config.port, 9123)
        XCTAssertEqual(config.apiKey, "k")
        XCTAssertEqual(config.shutdownGrace, 25)
    }

    func testAnUnreadableFileYieldsDefaultsRatherThanFailing() throws {
        let url = try write("{ this is not json")
        let config = ServerConfig.read(at: url)
        // Not this component's job to validate: the server refuses a broken
        // document far more precisely, and now stays up in recovery mode to say
        // so. Guessing here would at worst mean a tray that cannot connect.
        XCTAssertEqual(config.port, 8765)
        XCTAssertNil(config.apiKey)
    }

    func testAWildcardBindIsNotAnAddressToConnectTo() {
        var config = ServerConfig()
        config.host = "0.0.0.0"
        XCTAssertEqual(config.connectHost, "127.0.0.1")
        XCTAssertEqual(config.baseURL.absoluteString, "http://127.0.0.1:8765")
    }

    func testAnEmptyKeyIsNoKey() throws {
        let url = try write(#"{"server": {"api_key": ""}}"#)
        XCTAssertNil(ServerConfig.read(at: url).apiKey)
    }

    func testSeedingWritesOnlyWhenThereIsNoFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-seed-\(UUID().uuidString).json")
        try ServerConfig.seedIfMissing(at: url)
        let first = try Data(contentsOf: url)
        XCTAssertFalse(first.isEmpty)

        // The server owns this file. Seeding twice would make this app a second
        // writer, which is the property the whole design rests on.
        try ServerConfig.seedIfMissing(at: url)
        XCTAssertEqual(try Data(contentsOf: url), first)
    }

    func testTheSeededFileIsNotWorldReadable() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-seed-\(UUID().uuidString).json")
        try ServerConfig.seedIfMissing(at: url)
        let mode = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions]
        // It can hold an API key.
        XCTAssertEqual((mode as? NSNumber)?.int16Value ?? 0, 0o600)
    }

    func testTheSeededDefaultsStartWithTwoUngatedModels() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-seed-\(UUID().uuidString).json")
        try ServerConfig.seedIfMissing(at: url)
        let root =
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        let models = root["models"] as! [String: [String: Any]]
        let enabled = models.filter { $0.value["enabled"] as? Bool == true }.keys.sorted()
        // A fresh install has to generate with no token, no access request and
        // no licence to accept.
        XCTAssertEqual(enabled, ["ernie-image-turbo", "z-image-turbo"])
        XCTAssertEqual(root["default_model"] as? String, "z-image-turbo")
        XCTAssertTrue(enabled.contains(root["default_model"] as! String))
    }
}

final class SupervisorTests: XCTestCase {
    func testAnOccupiedPortIsRefusedWithAnActionableMessage() throws {
        // Hold a port, then ask the supervisor to check it.
        let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        defer { Darwin.close(socket) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0  // let the kernel choose
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        _ = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(socket, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        var bound = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &bound) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(socket, $0, &length)
            }
        }
        listen(socket, 1)
        let port = Int(bound.sin_port.bigEndian)

        XCTAssertThrowsError(try Supervisor.ensurePortFree(host: "127.0.0.1", port: port)) { error in
            let message = error.localizedDescription
            XCTAssertTrue(message.contains("\(port)"), message)
            // Actionable: it names both likely causes and where to change it.
            XCTAssertTrue(message.contains("qds serve"), message)
            XCTAssertTrue(message.contains("Configuration"), message)
        }
    }

    func testAFreePortIsAccepted() throws {
        XCTAssertNoThrow(try Supervisor.ensurePortFree(host: "127.0.0.1", port: 0))
    }

    func testAnExitStatusIsDescribedRatherThanPrinted() {
        // waitpid packs the code into the high byte; a signal into the low 7 bits.
        XCTAssertEqual(Supervisor.describe(status: 0), "the server exited")
        XCTAssertEqual(Supervisor.describe(status: 1 << 8), "the server exited with code 1")
        XCTAssertEqual(
            Supervisor.describe(status: SIGKILL), "the server was terminated by signal 9")
    }

    func testTheChildIsToldWhereItsConfigurationIs() {
        let paths = Paths(data: URL(fileURLWithPath: "/tmp/qds-env"))
        let environment = Supervisor.childEnvironment(paths: paths, config: ServerConfig())
        // Without this the server looks for a config relative to its own
        // package — `site-packages/`, where none exists — and every setting is
        // silently ignored.
        XCTAssertTrue(environment.contains("QDS_SERVER_CONFIG=/tmp/qds-env/server-config.json"))
        XCTAssertTrue(environment.contains("QDS_SERVER_IMAGE_STORE=/tmp/qds-env/images"))
        XCTAssertTrue(environment.contains("QDS_SERVER_LOG_JSON=1"))
    }
}

final class BootstrapTests: XCTestCase {
    func testTheVersionComesFromTheWheelsOwnName() {
        let wheel = URL(fileURLWithPath: "/x/qds-2.0.0-py3-none-any.whl")
        // The filename is the wheel's identity, not a second place to write the
        // version down.
        XCTAssertEqual(Bootstrap.version(ofWheel: wheel), "2.0.0")
    }

    func testAnUnrecognisableNameYieldsNoVersion() {
        XCTAssertNil(Bootstrap.version(ofWheel: URL(fileURLWithPath: "/x/qds.whl")))
    }

    /// The regression that made the app run last week's server.
    ///
    /// The install record held only the version, and the version does not change
    /// between development builds — so a freshly built `2.0.0` carrying a new
    /// server and a new dashboard compared equal to the installed `2.0.0`, and
    /// the app reported "ready" without reinstalling anything. Identity is the
    /// bytes, not the name.
    func testTwoWheelsWithTheSameNameButDifferentContentHaveDifferentIdentities() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-wheels-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let first = directory.appendingPathComponent("qds-2.0.0-py3-none-any.whl")
        try Data("build one".utf8).write(to: first)
        let digestOne = Bootstrap.digest(ofWheel: first)

        try Data("build two".utf8).write(to: first)
        let digestTwo = Bootstrap.digest(ofWheel: first)

        // The name — and therefore the version — is identical in both.
        XCTAssertEqual(Bootstrap.version(ofWheel: first), "2.0.0")
        XCTAssertNotNil(digestOne)
        XCTAssertNotEqual(digestOne, digestTwo, "identical bytes were read from different content")
    }

    func testTheSameBytesAlwaysHaveTheSameIdentity() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-\(UUID().uuidString).whl")
        try Data("same".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        // The negative of the test above: if the digest changed on every call,
        // that test would pass while making the app reinstall on every launch.
        XCTAssertEqual(Bootstrap.digest(ofWheel: url), Bootstrap.digest(ofWheel: url))
    }

    func testAMissingWheelHasNoIdentityRatherThanAnEmptyOne() {
        XCTAssertNil(Bootstrap.digest(ofWheel: URL(fileURLWithPath: "/nonexistent.whl")))
    }

    func testALostNetworkIsExplainedRatherThanNumbered() {
        let explained = Bootstrap.explain(
            (status: 2, output: "error: Failed to download python-3.12\ndns error"))
        XCTAssertTrue(explained.contains("network"), explained)
        XCTAssertFalse(explained.contains("status 2"), explained)
    }

    func testAnUnknownFailureStillCarriesTheLastThingSaid() {
        let explained = Bootstrap.explain((status: 1, output: "error: no such file\n\n"))
        XCTAssertTrue(explained.contains("no such file"), explained)
    }
}

final class BackoffTests: XCTestCase {
    func testTheProgressRetryIsBoundedAndStartsShort() {
        XCTAssertEqual(reconnectDelay(attempt: 0), .milliseconds(500))
        XCTAssertEqual(reconnectDelay(attempt: 1), .milliseconds(1000))
        // Bounded: a server that is genuinely down is retried every ten seconds
        // forever, not with an ever-growing delay that never recovers.
        XCTAssertEqual(reconnectDelay(attempt: 20), .milliseconds(10_000))
    }
}

final class ProcessGroupTests: XCTestCase {
    /// The property the whole `posix_spawn` detour exists for.
    ///
    /// A download or a conversion is a *grandchild* of this app. Signalling only
    /// the server would leave those running as orphans under launchd, holding
    /// the HuggingFace cache and invisible to whatever starts next. Signalling
    /// the group reaches them — which requires the child to be a group leader,
    /// which is what `Foundation.Process` cannot arrange.
    func testASpawnedChildLeadsItsOwnProcessGroupAndTakesItsChildrenDown() throws {
        let log = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-spawn-\(UUID().uuidString).log")
        let marker = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-grandchild-\(UUID().uuidString)")

        // A child that spawns its own child, exactly as the server spawns jobs.
        let script = "(while true; do touch '\(marker.path)'; sleep 0.05; done) & sleep 30"
        let pid = try Supervisor.spawnDetached(
            executable: "/bin/sh",
            argv: ["/bin/sh", "-c", script],
            environment: ["PATH=/usr/bin:/bin"],
            logPath: log.path
        )
        defer { Supervisor.signalGroup(pid, SIGKILL) }

        // Group leader: its pid *is* its process-group id, so `kill(-pid, …)`
        // addresses it and everything it started.
        XCTAssertEqual(getpgid(pid), pid, "the child is not its own process-group leader")

        // Wait for the grandchild to prove it is alive.
        var alive = false
        for _ in 0..<100 {
            if FileManager.default.fileExists(atPath: marker.path) {
                alive = true
                break
            }
            usleep(50_000)
        }
        XCTAssertTrue(alive, "the grandchild never started; this test would prove nothing")

        Supervisor.signalGroup(pid, SIGKILL)
        usleep(300_000)
        try? FileManager.default.removeItem(at: marker)
        usleep(400_000)
        // If the grandchild survived the group signal it would recreate this.
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: marker.path),
            "the grandchild outlived a signal sent to the group")
    }

    func testTheChildDoesNotInheritThisProcessesEnvironment() throws {
        setenv("QDS_TEST_LEAK", "1", 1)
        defer { unsetenv("QDS_TEST_LEAK") }
        let paths = Paths(data: URL(fileURLWithPath: "/tmp/qds-env"))
        let environment = Supervisor.childEnvironment(paths: paths, config: ServerConfig())
        // Only a named few are passed through; the rest of the app's
        // environment is not the server's business.
        XCTAssertFalse(environment.contains { $0.hasPrefix("QDS_TEST_LEAK=") })
    }
}

final class MenuBarIconTests: XCTestCase {
    /// The regression that made the app look like it had not launched.
    ///
    /// `Image(systemName:)` renders *nothing* for a name that does not resolve —
    /// no crash, no log — and the stopped-state name was `sparkles.slash`,
    /// which is not a real symbol. Since the app starts stopped, the status
    /// item was always invisible.
    func testEverySymbolTheStatusItemCanShowActuallyExists() {
        for name in MenuBarIcon.all {
            XCTAssertTrue(
                MenuBarIcon.resolves(name),
                "\(name) is not an SF Symbol; the menu bar item would be invisible")
        }
    }

    func testAMissingSymbolIsDetectedRatherThanRenderedAsNothing() {
        // The guard itself has to work, or the test above proves nothing.
        XCTAssertFalse(MenuBarIcon.resolves("sparkles.slash"))
        XCTAssertFalse(MenuBarIcon.resolves("definitely.not.a.symbol"))
    }

    func testTheTwoStatesAreDistinguishable() {
        XCTAssertNotEqual(MenuBarIcon.running, MenuBarIcon.stopped)
    }
}

final class NetworkBindingTests: XCTestCase {
    /// The case the loopback-only pre-flight missed — measured, not assumed.
    ///
    /// With `SO_REUSEADDR`, binding `127.0.0.1:P` succeeds while another process
    /// holds `0.0.0.0:P`: BSD lets a more specific address bind under a
    /// wildcard. The second assertion below pins that, because it is the reason
    /// the check had to change; the first pins the behaviour that replaces it.
    func testThePreflightBindsWhatTheServerWillBind() throws {
        // Hold a wildcard port, then ask for the same port on the wildcard host.
        let held = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        defer { Darwin.close(held) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = INADDR_ANY.bigEndian
        _ = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(held, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        var bound = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &bound) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(held, $0, &length) }
        }
        listen(held, 1)
        let port = Int(bound.sin_port.bigEndian)

        // What the server will actually attempt, and what now refuses.
        XCTAssertThrowsError(try Supervisor.ensurePortFree(host: "0.0.0.0", port: port))
        // And the old check, which did not — the bug, pinned so it stays fixed.
        XCTAssertNoThrow(try Supervisor.ensurePortFree(host: "127.0.0.1", port: port))
    }

    func testAFreePortIsAcceptedOnEitherHostForm() {
        XCTAssertNoThrow(try Supervisor.ensurePortFree(host: "0.0.0.0", port: 0))
        XCTAssertNoThrow(try Supervisor.ensurePortFree(host: "127.0.0.1", port: 0))
    }

    /// `canConnect` hands this to `inet_addr`, which answers `INADDR_NONE` for
    /// anything that is not a dotted quad — so a name would make the tray report
    /// "not listening" about a healthy server.
    func testAWildcardOrANameConnectsBackOverLoopback() {
        var config = ServerConfig()
        for host in ["0.0.0.0", "::", "", "nas.local", "macstudio.home"] {
            config.host = host
            XCTAssertEqual(config.connectHost, "127.0.0.1", host)
        }
    }

    func testAnAddressIsConnectedToDirectly() {
        var config = ServerConfig()
        config.host = "192.168.1.19"
        XCTAssertEqual(config.connectHost, "192.168.1.19")
        XCTAssertEqual(config.baseURL.absoluteString, "http://192.168.1.19:8765")
    }
}

final class VersionOrderingTests: XCTestCase {
    /// The reason this is not a string comparison.
    ///
    /// `"2.10.0" < "2.9.0"` is *true* alphabetically, so a string compare goes
    /// quiet exactly when the tenth minor release ships — and stays quiet for
    /// every release after it. Nothing about that failure is visible: the app
    /// simply never mentions an update again.
    func testTenSortsAfterNineRatherThanAlphabetically() {
        XCTAssertTrue(Version("2.9.0")! < Version("2.10.0")!)
        XCTAssertFalse(Version("2.10.0")! < Version("2.9.0")!)
        // The bug this pins, stated directly.
        XCTAssertTrue("2.10.0" < "2.9.0", "the string ordering changed; this test's premise is gone")
    }

    func testTheTagsVPrefixIsOptionalBecauseThisRepositoryUsesBoth() {
        // Real tags from this repository: `1.0.0` and `v2.1.0`.
        XCTAssertEqual(Version("v2.1.0"), Version("2.1.0"))
        XCTAssertTrue(Version("1.0.0")! < Version("v2.0.0")!)
    }

    /// The rendering defect the mockup comparison caught.
    ///
    /// The UI writes "Version \(version) is available", and this repository
    /// tags releases `v2.1.0`. Echoing the tag as written produced
    /// "Version v2.3.0" — the `v` saying again what the word already said. The
    /// spelling is normalised here rather than stripped at each call site,
    /// because there are two of them (the window and the menu item) and they
    /// must name the same release identically.
    func testTheDisplayedSpellingIsNormalisedRatherThanTheTagAsWritten() {
        XCTAssertEqual(Version("v2.3.0")!.description, "2.3.0")
        XCTAssertEqual(Version("2.3.0")!.description, "2.3.0")
        // Padding is not invented either: what was parsed is what is shown.
        XCTAssertEqual(Version("v2.1")!.description, "2.1")
    }

    func testAMissingComponentReadsAsZeroRatherThanAsNewer() {
        XCTAssertEqual(Version("2.1"), Version("2.1.0"))
        XCTAssertTrue(Version("2.1")! < Version("2.1.1")!)
    }

    /// Fail closed: a tag that is not a version must not be coerced into an
    /// ordering. `nil` reaches the UI as "could not check", which is honest;
    /// a guess would be an update offered on the strength of nothing.
    func testANonNumericOrOverflowingTagIsRefusedRatherThanGuessed() {
        XCTAssertNil(Version("nightly"))
        XCTAssertNil(Version(""))
        XCTAssertNil(Version("v"))
        XCTAssertNil(Version("2..0"))
        XCTAssertNil(Version("99999999999999999999.0.0"))
    }

    /// A prerelease suffix is not ordered, so `2.3.0-rc1` can never be reported
    /// as newer than `2.3.0`. GitHub's `prerelease` flag is what actually
    /// excludes them; this is the second, structural guard.
    func testAPrereleaseSuffixNeverSortsAboveTheMatchingFinal() {
        XCTAssertFalse(Version("2.3.0-rc1")! > Version("2.3.0")!)
    }
}

final class ReleaseDecodingTests: XCTestCase {
    /// A real response from
    /// `api.github.com/repos/exalandru/Quantum-Diffusion-Server/releases/latest`,
    /// trimmed to the fields this app reads.
    private let latest = Data(
        #"""
        {"tag_name":"v2.1.0","name":"v2.1.0","draft":false,"prerelease":false,
         "published_at":"2026-08-24T18:06:25Z",
         "html_url":"https://github.com/exalandru/Quantum-Diffusion-Server/releases/tag/v2.1.0"}
        """#.utf8)

    func testTheFieldsTheAppActuallyUsesAreRead() throws {
        let release = try ReleaseCheck.decode(latest)
        XCTAssertEqual(release.tagName, "v2.1.0")
        XCTAssertEqual(release.version, Version("2.1.0"))
        XCTAssertFalse(release.draft)
        XCTAssertFalse(release.prerelease)
        XCTAssertNotNil(release.publishedAt)
        XCTAssertTrue(release.isOfferable)
        XCTAssertEqual(release.url.absoluteString, release.htmlURL)
    }

    /// The flags are required rather than defaulted. A response missing them
    /// cannot be shown to be a stable release, and defaulting `prerelease` to
    /// `false` would advertise one.
    func testAResponseWithoutTheStabilityFlagsIsRefusedRatherThanAssumedStable() {
        let partial = Data(#"{"tag_name":"v9.9.9","html_url":"https://x/y"}"#.utf8)
        XCTAssertThrowsError(try ReleaseCheck.decode(partial))
    }

    func testADraftOrPrereleaseIsNotOfferable() throws {
        for flag in ["draft", "prerelease"] {
            let json = Data(
                """
                {"tag_name":"v9.9.9","html_url":"https://x/y",
                 "draft":\(flag == "draft"),"prerelease":\(flag == "prerelease")}
                """.utf8)
            XCTAssertFalse(try ReleaseCheck.decode(json).isOfferable, flag)
        }
    }

    func testAnUnparseableURLFallsBackToTheRepositoryRatherThanCrashing() throws {
        let json = Data(
            #"{"tag_name":"v9.9.9","html_url":"","draft":false,"prerelease":false}"#.utf8)
        XCTAssertEqual(try ReleaseCheck.decode(json).url, Product.repositoryURL)
    }
}

final class ReleaseVerdictTests: XCTestCase {
    private func release(_ tag: String, draft: Bool = false, prerelease: Bool = false) -> Release {
        Release(
            tagName: tag, htmlURL: "https://github.com/x/y/releases/tag/\(tag)",
            draft: draft, prerelease: prerelease, publishedAt: nil)
    }

    func testANewerPublishedReleaseIsOffered() {
        guard case .available(let found) = MenuModel.verdict(for: release("v2.3.0"), current: "2.2.0")
        else { return XCTFail("a newer release was not offered") }
        XCTAssertEqual(found.tagName, "v2.3.0")
    }

    /// The state this repository is actually in right now: `pyproject.toml` says
    /// 2.2.0 and the newest tag is v2.1.0. A check that only asked "are they
    /// different" would report an update *downwards* on every development build.
    func testABuildAheadOfTheNewestTagIsCurrentRatherThanOutdated() {
        XCTAssertEqual(MenuModel.verdict(for: release("v2.1.0"), current: "2.2.0"), .current)
    }

    func testTheSameVersionIsCurrent() {
        XCTAssertEqual(MenuModel.verdict(for: release("v2.2.0"), current: "2.2.0"), .current)
    }

    /// Fail closed at each of the three ways the comparison can be impossible.
    /// None of them may answer `.current`: that is a claim, and nothing here
    /// established it.
    func testAnImpossibleComparisonIsUnknownRatherThanUpToDate() {
        XCTAssertEqual(MenuModel.verdict(for: release("nightly"), current: "2.2.0"), .unknown)
        XCTAssertEqual(MenuModel.verdict(for: release("v2.3.0"), current: nil), .unknown)
        XCTAssertEqual(MenuModel.verdict(for: release("v2.3.0"), current: "not-a-version"), .unknown)
    }

    func testADraftOrPrereleaseIsNeverOffered() {
        XCTAssertEqual(
            MenuModel.verdict(for: release("v9.9.9", draft: true), current: "2.2.0"), .unknown)
        XCTAssertEqual(
            MenuModel.verdict(for: release("v9.9.9", prerelease: true), current: "2.2.0"), .unknown)
    }
}

final class ReleaseCheckScheduleTests: XCTestCase {
    func testTheFirstCheckIsDueAndAFreshOneIsNot() {
        XCTAssertTrue(ReleaseCheck.isDue(lastChecked: nil))
        let now = Date()
        XCTAssertFalse(ReleaseCheck.isDue(lastChecked: now, now: now))
        XCTAssertFalse(ReleaseCheck.isDue(lastChecked: now.addingTimeInterval(-3600), now: now))
    }

    func testACheckOlderThanTheIntervalIsDue() {
        let now = Date()
        XCTAssertTrue(
            ReleaseCheck.isDue(lastChecked: now.addingTimeInterval(-ReleaseCheck.interval), now: now))
    }

    /// The clock trap.
    ///
    /// A machine whose time moved backwards — a correction, a restore, a user
    /// setting it by hand — leaves a `lastChecked` in the future. `elapsed >=
    /// interval` on a negative elapsed is `false`, so without this the app would
    /// never check again for as long as that timestamp stood: a permanent,
    /// silent stall that no error would report.
    func testATimestampInTheFutureIsDueRatherThanStallingForever() {
        let now = Date()
        XCTAssertTrue(ReleaseCheck.isDue(lastChecked: now.addingTimeInterval(86_400 * 365), now: now))
    }
}

final class ProductIdentityTests: XCTestCase {
    /// The three URLs the About window offers all descend from one pair of
    /// coordinates, so "Show on GitHub" and the update check cannot point at
    /// different repositories.
    func testEveryLinkIsBuiltFromTheSameCoordinates() {
        XCTAssertEqual(Product.authorURL.absoluteString, "https://github.com/exalandru")
        XCTAssertEqual(
            Product.repositoryURL.absoluteString,
            "https://github.com/exalandru/Quantum-Diffusion-Server")
        XCTAssertEqual(
            Product.latestReleaseAPI.absoluteString,
            "https://api.github.com/repos/exalandru/Quantum-Diffusion-Server/releases/latest")
    }
}

@MainActor
final class CreditLinkTests: XCTestCase {
    /// The `-50` defect, pinned where it actually lived.
    ///
    /// The credit line was written as interpolated markdown:
    ///
    /// ```swift
    /// Text("Crafted by [\(owner)](\(url.absoluteString))")
    /// ```
    ///
    /// A literal string is a `LocalizedStringKey`, and SwiftUI parses the
    /// markdown on the *format* string — `"Crafted by [%@](%@)"` — before
    /// substituting. The link's destination became the literal `%@`, which
    /// reaches LaunchServices as the relative URL `%25@` and fails with `-50`
    /// (`paramErr`). Measured, not guessed: a probe clicked all four spellings
    /// and the two with an interpolated destination both delivered `%25@`.
    ///
    /// Nothing about it was visible. The text rendered correctly, the link was
    /// blue, the accessibility tree reported an `AXLink` — it only failed when
    /// pressed. So this asserts on the *destination*, which is the only thing
    /// that was ever wrong.
    func testTheCreditLinkPointsAtTheAuthorRatherThanAFormatPlaceholder() {
        let credit = AboutView.credit

        let links = credit.runs.compactMap(\.link)
        XCTAssertEqual(links.count, 1, "the credit line should carry exactly one link")
        XCTAssertEqual(links.first, Product.authorURL)

        // The regression, stated as what must never appear again.
        let destination = links.first?.absoluteString ?? ""
        XCTAssertFalse(destination.contains("%"), "the destination is a format placeholder: \(destination)")
        XCTAssertTrue(destination.hasPrefix("https://"), destination)
    }

    func testTheCreditReadsAsASentenceWithTheNameInIt() {
        let text = String(AboutView.credit.characters)
        XCTAssertEqual(text, "Crafted by exalandru")
    }

    /// The link is on the name only — not on "Crafted by".
    func testOnlyTheNameIsClickable() {
        let credit = AboutView.credit
        for run in credit.runs where run.link != nil {
            XCTAssertEqual(String(credit[run.range].characters), Product.owner)
        }
    }
}

/// The install parser, against output measured from the bundled wheel.
///
/// Every literal below was copied from a real `uv tool install` of
/// `qds-2.2.0-py3-none-any.whl` out of `QDS.app`, captured through a pipe —
/// which is the only way it is ever read in production. Inventing plausible
/// lines here would make this suite agree with a parser written from the same
/// imagination, which is the collusion the repository's own review rules name.
final class InstallProgressTests: XCTestCase {
    /// The measured happy path, abridged to its shape.
    ///
    /// Three properties at once, because they are what the window draws: the
    /// phase advances in order, the count reaches the announced total, and the
    /// bar ends full.
    func testARealRunAdvancesThroughEveryPhase() {
        var progress = InstallProgress()
        XCTAssertEqual(progress.phase, .preparing)
        XCTAssertNil(progress.fraction, "nothing has published a denominator yet")

        progress.absorb("Downloading cpython-3.12.13-macos-aarch64-none (download) (23.8MiB)")
        XCTAssertEqual(progress.phase, .python)
        XCTAssertNil(progress.fraction, "one file is not a fraction of anything")

        progress.absorb(" Downloaded cpython-3.12.13-macos-aarch64-none (download)")
        progress.absorb("Resolved 87 packages in 578ms")
        XCTAssertEqual(progress.phase, .packages)

        progress.absorb("Downloading torch (106.1MiB)")
        progress.absorb("Downloading pygments (1.2MiB)")
        XCTAssertEqual(progress.packagesTotal, 2)
        XCTAssertEqual(progress.packagesDone, 0)
        XCTAssertEqual(progress.fraction, 0)

        progress.absorb(" Downloaded pygments")
        XCTAssertEqual(progress.packagesDone, 1)

        progress.absorb(" Downloaded torch")
        progress.absorb("Prepared 87 packages in 3.91s")
        XCTAssertEqual(progress.fraction, 1)

        progress.beginRewriter()
        XCTAssertEqual(progress.phase, .rewriter)
        XCTAssertNil(progress.fraction, "the rewriter fetch publishes no progress at all")
    }

    /// The reason the bar is weighed in bytes rather than in packages.
    ///
    /// Measured: torch is 106.1 MiB and pygments is 1.2 MiB, so finishing
    /// pygments is 1% of the work and 50% of the count. A bar driven by the
    /// count jumps half way across for a file that took 0.6s, then sits still
    /// for the four seconds that actually matter — which is the specific lie
    /// this window exists to stop telling.
    func testTheBarIsWeighedBySizeRatherThanByPackageCount() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 578ms")
        progress.absorb("Downloading torch (106.1MiB)")
        progress.absorb("Downloading pygments (1.2MiB)")
        progress.absorb(" Downloaded pygments")

        XCTAssertEqual(progress.packagesDone, 1)
        XCTAssertEqual(progress.packagesTotal, 2)
        let fraction = try! XCTUnwrap(progress.fraction)
        // Half the packages, about a hundredth of the bytes.
        XCTAssertLessThan(fraction, 0.05, "the bar followed the count instead of the size")
        XCTAssertGreaterThan(fraction, 0)
    }

    /// The warm-cache path, which is what an *update* actually runs.
    ///
    /// Measured by installing twice into the same `UV_CACHE_DIR`: the second run
    /// prints `Resolved` then `Installed` with **no `Downloading` line at all**
    /// and finishes in under a second. A bar that treated "no downloads" as
    /// "zero of zero done" and rendered 100%, or as 0% and stuck, would both be
    /// wrong; there is simply no fraction to show.
    func testAWarmCacheDownloadsNothingAndOffersNoFraction() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Installed 87 packages in 581ms")
        XCTAssertEqual(progress.phase, .packages)
        XCTAssertEqual(progress.packagesTotal, 0)
        XCTAssertNil(progress.fraction, "0 of 0 packages is not a percentage")
    }

    /// `Prepared` on a warm cache must not read as a completed download.
    func testPreparedWithNothingDownloadedStillOffersNoFraction() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Prepared 87 packages in 12ms")
        XCTAssertNil(progress.fraction)
    }

    /// The two-parenthesis line, which a naive "first group" parser gets wrong.
    ///
    /// uv writes `Downloading cpython-… (download) (23.8MiB)`. Taking the first
    /// group yields the word "download", which parses as no size at all.
    func testTheSizeIsReadFromTheLastGroupNotTheFirst() {
        let line = "Downloading cpython-3.12.13-macos-aarch64-none (download) (23.8MiB)"
        let bytes = try! XCTUnwrap(InstallProgress.bytes(in: line))
        XCTAssertEqual(bytes, 23.8 * 1024 * 1024, accuracy: 1)
    }

    func testSizesAreReadInBinaryUnitsAsUvPrintsThem() {
        XCTAssertEqual(try! XCTUnwrap(InstallProgress.parseSize("1.5GiB")), 1.5 * 1024 * 1024 * 1024, accuracy: 1)
        XCTAssertEqual(try! XCTUnwrap(InstallProgress.parseSize("106.1MiB")), 106.1 * 1024 * 1024, accuracy: 1)
        XCTAssertEqual(try! XCTUnwrap(InstallProgress.parseSize("512KiB")), 512 * 1024, accuracy: 1)
        // Not a size: must not be read as zero, which would silently drop a
        // package's weight out of the denominator.
        XCTAssertNil(InstallProgress.parseSize("download"))
        XCTAssertNil(InstallProgress.parseSize(""))
    }

    func testAPackageNameStopsBeforeItsSize() {
        XCTAssertEqual(
            InstallProgress.packageName(afterPrefix: "Downloading ", in: "Downloading torch (106.1MiB)"),
            "torch")
        // `Downloaded` lines carry no size at all.
        XCTAssertEqual(
            InstallProgress.packageName(afterPrefix: "Downloaded ", in: "Downloaded torch"),
            "torch")
    }

    /// The managed CPython is not one of the 87 packages.
    ///
    /// It has the same line shape, so a parser that does not special-case it
    /// counts it — making the denominator wrong by one and adding 23.8 MiB to a
    /// total uv has not announced yet, so the bar starts somewhere other than
    /// zero and never reaches one.
    func testTheManagedPythonIsNotCountedAsAPackage() {
        var progress = InstallProgress()
        progress.absorb("Downloading cpython-3.12.13-macos-aarch64-none (download) (23.8MiB)")
        XCTAssertEqual(progress.packagesTotal, 0)
        XCTAssertEqual(progress.bytesTotal, 0)
    }

    /// A `Downloaded` for something never announced must not advance anything.
    ///
    /// Otherwise `packagesDone` can exceed `packagesTotal` and the fraction goes
    /// past 1 — a bar that overshoots its own end.
    func testAnUnannouncedCompletionIsIgnored() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb(" Downloaded something-we-never-saw-announced")
        XCTAssertEqual(progress.packagesDone, 0)
        XCTAssertNil(progress.fraction)
    }

    /// A repeated announcement must not be counted twice.
    func testARepeatedAnnouncementDoesNotInflateTheTotal() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Downloading torch (106.1MiB)")
        progress.absorb("Downloading torch (106.1MiB)")
        XCTAssertEqual(progress.packagesTotal, 1)
        XCTAssertEqual(progress.bytesTotal, 106.1 * 1024 * 1024, accuracy: 1)
    }

    /// The bar cannot overshoot, and the reason is structural rather than the
    /// clamp in `fraction`.
    ///
    /// Worth stating precisely, because the first version of this test was
    /// **vacuous** and an independent review caught it: it was named for the
    /// `min(1, …)` in `fraction` and passed with that clamp deleted. So did a
    /// second attempt. The clamp is in fact *unreachable*: `bytesDone` only ever
    /// grows by the size of a package that is currently in `outstanding` — which
    /// is removed in the same step — and each name contributes to `bytesTotal`
    /// at most once, so `bytesDone <= bytesTotal` holds by construction.
    ///
    /// The clamp stays as a cheap guard against a future edit breaking that
    /// invariant, but no test can witness it while the invariant holds, and
    /// pretending otherwise is what this comment exists to prevent. What is
    /// asserted here is the invariant itself, on the paths that move the two
    /// numbers.
    func testTheCreditedBytesCanNeverExceedTheAnnouncedOnes() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Downloading small (1MiB)")
        progress.absorb("Downloading big (100MiB)")

        // A duplicate completion: discarded, so it cannot double-credit.
        progress.absorb(" Downloaded big")
        progress.absorb(" Downloaded big")
        XCTAssertLessThanOrEqual(progress.bytesDone, progress.bytesTotal)
        XCTAssertEqual(progress.packagesDone, 1)

        // A completion for something never announced: also discarded.
        progress.absorb(" Downloaded never-announced")
        XCTAssertLessThanOrEqual(progress.bytesDone, progress.bytesTotal)

        progress.absorb(" Downloaded small")
        progress.absorb("Prepared 87 packages in 1s")
        XCTAssertEqual(progress.bytesDone, progress.bytesTotal)
        XCTAssertEqual(progress.packagesDone, progress.packagesTotal)
        XCTAssertEqual(progress.fraction, 1)
    }

    /// Unrecognised output changes nothing.
    ///
    /// The failure mode this prevents is not a crash: it is a window frozen at
    /// "Preparing…" for four minutes because uv reworded one line. Tolerance
    /// here is what keeps a wording change cosmetic instead of fatal.
    func testUnrecognisedLinesAreIgnoredRatherThanDerailingTheParse() {
        var progress = InstallProgress()
        progress.absorb("warning: `/tmp/x/bin` is not on your PATH.")
        progress.absorb(" + annotated-types==0.8.0")
        progress.absorb("Installed 1 executable: qds")
        progress.absorb("")
        XCTAssertEqual(progress.phase, .preparing)
        XCTAssertEqual(progress.packagesTotal, 0)
    }

    /// The phase never goes backwards.
    ///
    /// uv interleaves its output across concurrent downloads, and `Resolved`
    /// arriving after a `Downloading` must not drag the window back a step.
    func testThePhaseNeverRegresses() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Downloading torch (106.1MiB)")
        progress.beginRewriter()
        XCTAssertEqual(progress.phase, .rewriter)
        // Late output from the install, arriving after the rewriter started.
        progress.absorb("Prepared 87 packages in 3.91s")
        XCTAssertEqual(progress.phase, .rewriter, "a late line dragged the phase backwards")
    }

    /// The label names the biggest outstanding download, not the newest.
    ///
    /// uv announces twenty downloads within ~10ms of each other, so "most
    /// recent" changes faster than a person can read and settles on whichever
    /// tiny package happened to be announced last. The largest outstanding one
    /// is stable and is the honest answer to what the install is waiting for.
    func testTheLabelNamesTheLargestOutstandingDownload() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Downloading torch (106.1MiB)")
        progress.absorb("Downloading pygments (1.2MiB)")
        XCTAssertEqual(progress.currentPackage, "torch")

        progress.absorb(" Downloaded torch")
        XCTAssertEqual(progress.currentPackage, "pygments")

        progress.absorb(" Downloaded pygments")
        XCTAssertNil(progress.currentPackage, "nothing is outstanding")
    }

    func testHasReachedIsInclusiveAndOrdered() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        XCTAssertTrue(progress.hasReached(.preparing))
        XCTAssertTrue(progress.hasReached(.python))
        XCTAssertTrue(progress.hasReached(.packages))
        XCTAssertFalse(progress.hasReached(.rewriter))
    }
}

/// The incremental reader, and the property the whole feature rests on.
///
/// Before this, `Bootstrap.run` called `readDataToEndOfFile()` — which returns
/// only at EOF, i.e. once the child is already over. There was nothing to report
/// *during* the 3m21s an install was measured to take, so no window could have
/// shown anything. These tests are about output arriving while a process is
/// still running; a suite that only checked the final string would pass against
/// the old blocking implementation and prove nothing.
final class DrainTests: XCTestCase {
    /// Lines are delivered as they are written, not at EOF.
    ///
    /// The counterfactual is explicit: the writer holds the pipe open after the
    /// first burst, so a reader that waited for EOF would still be blocked when
    /// the expectation is checked, and this test would fail. That is what makes
    /// it a witness for streaming rather than for parsing.
    func testLinesArriveWhileTheWriterIsStillOpen() throws {
        let pipe = Pipe()
        let arrived = XCTestExpectation(description: "two lines arrived before EOF")
        arrived.expectedFulfillmentCount = 2

        let received = LineBox()
        let reader = Thread {
            _ = Bootstrap.drain(pipe.fileHandleForReading) { line in
                received.append(line)
                arrived.fulfill()
            }
        }
        reader.start()

        pipe.fileHandleForWriting.write(Data("Resolved 87 packages in 578ms\n".utf8))
        pipe.fileHandleForWriting.write(Data("Downloading torch (106.1MiB)\n".utf8))

        // The pipe is deliberately still open here. Under the old
        // `readDataToEndOfFile`, nothing would have been delivered yet.
        wait(for: [arrived], timeout: 5)
        XCTAssertEqual(
            received.lines,
            ["Resolved 87 packages in 578ms", "Downloading torch (106.1MiB)"])

        try pipe.fileHandleForWriting.close()
    }

    /// A line split across two writes is delivered once, whole.
    ///
    /// A pipe does not preserve write boundaries: a 4 KB chunk can end mid-line,
    /// and a reader that treated each chunk as a line would hand the parser
    /// `"Downloading tor"` and `"ch (106.1MiB)"` — neither of which parses, so
    /// the package would silently vanish from the denominator.
    func testALineSplitAcrossWritesIsReassembled() throws {
        let pipe = Pipe()
        let arrived = XCTestExpectation(description: "one whole line")
        let received = LineBox()

        let reader = Thread {
            _ = Bootstrap.drain(pipe.fileHandleForReading) { line in
                received.append(line)
                arrived.fulfill()
            }
        }
        reader.start()

        pipe.fileHandleForWriting.write(Data("Downloading tor".utf8))
        // Long enough for a naive reader to have delivered the fragment.
        Thread.sleep(forTimeInterval: 0.2)
        pipe.fileHandleForWriting.write(Data("ch (106.1MiB)\n".utf8))

        wait(for: [arrived], timeout: 5)
        XCTAssertEqual(received.lines, ["Downloading torch (106.1MiB)"])
        try pipe.fileHandleForWriting.close()
    }

    /// Carriage returns terminate a line too.
    ///
    /// uv writes plain newlines into a pipe today — measured — but a renderer
    /// that redraws in place uses `\r`, and a reader that split only on `\n`
    /// would accumulate an entire run into one unterminated line and report
    /// nothing at all until the process ended.
    func testCarriageReturnsTerminateLinesAsWellAsNewlines() throws {
        let pipe = Pipe()
        let arrived = XCTestExpectation(description: "three lines")
        arrived.expectedFulfillmentCount = 3
        let received = LineBox()

        let reader = Thread {
            _ = Bootstrap.drain(pipe.fileHandleForReading) { line in
                received.append(line)
                arrived.fulfill()
            }
        }
        reader.start()

        pipe.fileHandleForWriting.write(Data("one\rtwo\r\nthree\n".utf8))
        wait(for: [arrived], timeout: 5)
        XCTAssertEqual(received.lines, ["one", "two", "three"])
        try pipe.fileHandleForWriting.close()
    }

    /// A final line with no terminator is still delivered.
    ///
    /// How a child killed mid-write leaves the pipe — and, on a failure, that
    /// last fragment is often the error message.
    func testAnUnterminatedFinalLineIsStillDelivered() throws {
        let pipe = Pipe()
        let received = LineBox()
        let finished = XCTestExpectation(description: "drain returned")

        let reader = Thread {
            _ = Bootstrap.drain(pipe.fileHandleForReading) { received.append($0) }
            finished.fulfill()
        }
        reader.start()

        pipe.fileHandleForWriting.write(Data("error: no such file".utf8))
        try pipe.fileHandleForWriting.close()

        wait(for: [finished], timeout: 5)
        XCTAssertEqual(received.lines, ["error: no such file"])
    }

    /// The complete bytes are returned as well as streamed, because
    /// `Bootstrap.explain` reads the whole output to describe a failure.
    func testTheCompleteOutputIsReturnedForTheFailureExplanation() throws {
        let pipe = Pipe()
        let finished = XCTestExpectation(description: "drain returned")
        let box = DataBox()

        let reader = Thread {
            box.data = Bootstrap.drain(pipe.fileHandleForReading) { _ in }
            finished.fulfill()
        }
        reader.start()

        pipe.fileHandleForWriting.write(Data("error: Failed to download\ndns error\n".utf8))
        try pipe.fileHandleForWriting.close()
        wait(for: [finished], timeout: 5)

        let text = String(data: box.data, encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("dns error"), text)
        // And it still feeds the existing explanation path unchanged.
        XCTAssertTrue(Bootstrap.explain((status: 2, output: text)).contains("network"))
    }
}

/// A lock-guarded box, because the reader runs on its own thread.
private final class LineBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func append(_ line: String) {
        lock.lock()
        defer { lock.unlock() }
        storage.append(line)
    }

    var lines: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

private final class DataBox: @unchecked Sendable {
    var data = Data()
}

/// The status line the window and the menu both read.
final class InstallLabelTests: XCTestCase {
    func testEachPhaseIsNamedInPlainLanguage() {
        var progress = InstallProgress()
        XCTAssertEqual(Bootstrap.label(for: progress), "Preparing…")

        progress.absorb("Downloading cpython-3.12.13-macos-aarch64-none (download) (23.8MiB)")
        XCTAssertEqual(Bootstrap.label(for: progress), "Downloading the Python runtime…")

        progress.absorb("Resolved 87 packages in 578ms")
        progress.absorb("Downloading torch (106.1MiB)")
        XCTAssertEqual(Bootstrap.label(for: progress), "Downloading torch — 1 of 1")

        progress.beginRewriter()
        XCTAssertEqual(Bootstrap.label(for: progress), "Downloading the prompt enhancer…")
    }

    /// After `Prepared`, nothing is outstanding and the label must move on.
    ///
    /// The defect this pins: naming the last package forever would leave
    /// "Downloading torch — 87 of 87" on screen through the install step, which
    /// is a sentence about something that already finished.
    func testOnceEverythingIsDownloadedTheLabelStopsNamingAPackage() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 578ms")
        progress.absorb("Downloading torch (106.1MiB)")
        progress.absorb(" Downloaded torch")
        progress.absorb("Prepared 87 packages in 3.91s")
        XCTAssertEqual(Bootstrap.label(for: progress), "Installing the server…")
    }

    /// The warm-cache path never names a package, because none was downloaded.
    func testAWarmCacheNeverClaimsToBeDownloadingAnything() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Installed 87 packages in 581ms")
        XCTAssertEqual(Bootstrap.label(for: progress), "Installing the server…")
    }
}

/// The label must never walk backwards.
///
/// Found by running the real installer through the real reader, not by
/// inspection: uv emits `Resolved` about 130ms before its first `Downloading`,
/// and the first implementation treated "nothing outstanding" as "downloads are
/// finished". The window showed `Installing the server…` at 1.64s and then went
/// *back* to `Downloading torch` at 1.77s. A progress display that walks
/// backwards reads as a broken installer, which is the opposite of this
/// feature's purpose.
final class LabelMonotonicityTests: XCTestCase {
    /// The exact measured sequence, replayed.
    func testTheLabelDoesNotRegressBetweenResolutionAndTheFirstDownload() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 578ms")
        let afterResolve = Bootstrap.label(for: progress)
        XCTAssertEqual(afterResolve, "Resolving packages…")
        XCTAssertFalse(
            afterResolve.contains("Installing"),
            "resolution claimed the install had started; the next line walks it back")

        progress.absorb("Downloading torch (106.1MiB)")
        XCTAssertEqual(Bootstrap.label(for: progress), "Downloading torch — 1 of 1")
    }

    /// Once downloads are genuinely over, the label moves on and stays there.
    func testAfterPreparedTheLabelMovesOnAndStays() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 578ms")
        progress.absorb("Downloading torch (106.1MiB)")
        progress.absorb(" Downloaded torch")
        progress.absorb("Prepared 87 packages in 3.91s")
        XCTAssertEqual(Bootstrap.label(for: progress), "Installing the server…")
        XCTAssertTrue(progress.finishedDownloading)
    }

    /// A fully warm cache skips `Prepared` and goes straight to `Installed`.
    ///
    /// Measured by installing twice into one `UV_CACHE_DIR`. Without treating
    /// `Installed` as an end-of-download marker, an update would sit on
    /// "Resolving packages…" until the whole thing finished.
    func testAWarmCacheReachesTheInstallingLabelViaInstalledAlone() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        XCTAssertEqual(Bootstrap.label(for: progress), "Resolving packages…")
        progress.absorb("Installed 87 packages in 581ms")
        XCTAssertEqual(Bootstrap.label(for: progress), "Installing the server…")
        XCTAssertNil(progress.fraction, "nothing was downloaded, so there is no fraction")
    }

    /// Replay of the real transcript: the label may never return to a phase it
    /// has already left.
    ///
    /// The general property behind the specific defect above. Driven by the
    /// phase ordering rather than by string comparison, because the wording is
    /// allowed to change and the ordering is not.
    func testAcrossAWholeRunThePhaseOnlyEverMovesForward() {
        let transcript = [
            "Downloading cpython-3.12.13-macos-aarch64-none (download) (23.8MiB)",
            " Downloaded cpython-3.12.13-macos-aarch64-none (download)",
            "Resolved 87 packages in 578ms",
            "Downloading fonttools (2.7MiB)",
            "Downloading opencv-python (44.3MiB)",
            "Downloading torch (106.1MiB)",
            " Downloaded fonttools",
            " Downloaded opencv-python",
            " Downloaded torch",
            "Prepared 87 packages in 3.91s",
            "Installed 87 packages in 353ms",
            " + qds==2.2.0",
            "Installed 1 executable: qds",
        ]

        var progress = InstallProgress()
        var highest = InstallProgress.Phase.preparing
        var sawDownloading = false
        var sawInstallingLabel = false

        // The label's own ordering, independent of the phase enum: the phase
        // cannot distinguish "resolving", "downloading" and "installing", which
        // are all `.packages`, and every regression found so far lived inside
        // that one phase.
        func rank(_ label: String) -> Int {
            if label == "Preparing…" { return 0 }
            if label.contains("Python runtime") { return 1 }
            if label == "Resolving packages…" { return 2 }
            if label.hasPrefix("Downloading ") { return 3 }
            if label == "Installing the server…" { return 4 }
            if label.contains("prompt enhancer") { return 5 }
            return -1
        }

        var highestRank = 0
        for line in transcript {
            progress.absorb(line)
            XCTAssertTrue(
                progress.phase >= highest,
                "phase went backwards at \(line): \(progress.phase) < \(highest)")
            highest = progress.phase

            let label = Bootstrap.label(for: progress)
            let position = rank(label)
            XCTAssertNotEqual(position, -1, "unrecognised label: \(label)")
            // The defect this caught in review: after the last `Downloaded` and
            // before `Prepared`, nothing is outstanding — and the label fell
            // back to "Resolving packages…", i.e. two steps backwards, for one
            // frame. A one-frame flicker backwards is still a display that lies.
            XCTAssertGreaterThanOrEqual(
                position, highestRank,
                "the label went backwards at \(line): \(label)")
            highestRank = max(highestRank, position)

            if label.hasPrefix("Downloading ") && !label.contains("Python") {
                sawDownloading = true
                XCTAssertFalse(
                    sawInstallingLabel,
                    "the label returned to downloading after claiming to install: \(label)")
            }
            if label == "Installing the server…" { sawInstallingLabel = true }
        }

        // Both were actually exercised, or the assertions above proved nothing.
        XCTAssertTrue(sawDownloading, "the run never showed a download; this test is vacuous")
        XCTAssertTrue(sawInstallingLabel, "the run never reached the install step")
    }
}

/// Order is part of the contract, not an accident of scheduling.
///
/// `InstallProgress` is order-sensitive: a `Downloaded` applied before its
/// `Downloading` is discarded as unannounced, so the package silently vanishes
/// from both the count and the denominator. uv emits twenty announcements
/// within about ten milliseconds, so any per-line `Task { @MainActor in … }`
/// hand-off — which Swift explicitly does not order — could reorder them.
final class LineOrderingTests: XCTestCase {
    /// A burst delivered through the reader arrives in the order it was written.
    func testABurstOfLinesIsDeliveredInOrder() throws {
        let pipe = Pipe()
        let count = 200
        let arrived = XCTestExpectation(description: "all lines arrived")
        arrived.expectedFulfillmentCount = count
        let received = LineBox()

        let reader = Thread {
            _ = Bootstrap.drain(pipe.fileHandleForReading) {
                received.append($0)
                arrived.fulfill()
            }
        }
        reader.start()

        let written = (0..<count).map { "line-\($0)" }
        for line in written {
            pipe.fileHandleForWriting.write(Data("\(line)\n".utf8))
        }
        wait(for: [arrived], timeout: 10)
        XCTAssertEqual(received.lines, written, "lines were reordered in flight")
        try pipe.fileHandleForWriting.close()
    }

    /// The consequence, stated on the parser: applied in order, the totals are
    /// right; applied out of order, they are not.
    ///
    /// The second half is the counterfactual — without it, the first half would
    /// pass against an implementation that reorders and this test would prove
    /// nothing about why ordering matters.
    func testOutOfOrderCompletionsWouldCorruptTheCount() {
        let ordered = [
            "Resolved 87 packages in 578ms",
            "Downloading torch (106.1MiB)",
            "Downloading pygments (1.2MiB)",
            " Downloaded torch",
            " Downloaded pygments",
        ]

        var good = InstallProgress()
        for line in ordered { good.absorb(line) }
        XCTAssertEqual(good.packagesDone, 2)
        XCTAssertEqual(good.packagesTotal, 2)
        XCTAssertEqual(good.fraction, 1)

        // The same lines with each completion moved ahead of its announcement.
        let scrambled = [
            "Resolved 87 packages in 578ms",
            " Downloaded torch",
            " Downloaded pygments",
            "Downloading torch (106.1MiB)",
            "Downloading pygments (1.2MiB)",
        ]
        var bad = InstallProgress()
        for line in scrambled { bad.absorb(line) }
        // Both completions are discarded as unannounced: the bar would sit at
        // zero for the rest of the install.
        XCTAssertEqual(bad.packagesDone, 0)
        XCTAssertEqual(bad.fraction, 0)
        XCTAssertNotEqual(bad.packagesDone, good.packagesDone, "ordering makes no difference; this test is vacuous")
    }
}

/// `Bootstrap.run` end to end, through a real child process.
///
/// Added because an independent review found the ordering property had no
/// witness on the path that actually implements it: `DrainTests` drives `drain`
/// directly, so the `AsyncStream` + single-pump hand-off in `run` could be
/// reverted without failing anything.
///
/// **These tests still do not discriminate on ordering, and that is recorded
/// rather than papered over.** Reverting `run` to a per-line
/// `Task { @MainActor in … }` was measured against this suite: all four still
/// pass. The reason is that the main actor happens to dequeue those tasks in
/// submission order, and `run`'s later suspensions give it the chance to drain
/// them — so the *observable* behaviour is currently identical. What the
/// `AsyncStream` buys is that the ordering is **guaranteed** instead of being
/// an unspecified scheduling detail that a future runtime may change.
///
/// So: these witness that a real child's output arrives, complete, in order,
/// from both streams, before `run` returns. They do not witness the reason the
/// hand-off is shaped the way it is. No test can, short of a runtime that
/// reorders on purpose.
@MainActor
final class BootstrapRunTests: XCTestCase {
    private func bootstrap() -> Bootstrap {
        Bootstrap(
            paths: Paths(data: FileManager.default.temporaryDirectory
                .appendingPathComponent("qds-run-\(UUID().uuidString)")),
            onChange: {})
    }

    /// Lines reach the main actor in the order the child wrote them.
    ///
    /// 500 lines from a real `/bin/sh`, through the real pipe, the real drain
    /// and the real hand-off.
    func testOutputReachesTheMainActorInOrder() async throws {
        let box = LineBox()
        let result = try await bootstrap().run(
            URL(fileURLWithPath: "/bin/sh"),
            ["-c", "i=0; while [ $i -lt 500 ]; do echo line-$i; i=$((i+1)); done"],
            environment: ["PATH": "/usr/bin:/bin"],
            onLine: { line in box.append(line) })

        XCTAssertEqual(result.status, 0)
        XCTAssertEqual(box.lines, (0..<500).map { "line-\($0)" }, "lines were reordered on the way to the main actor")
    }

    /// Every line has been applied before `run` returns.
    ///
    /// The property `await pump.value` establishes: `install()` inspects
    /// `progress` immediately afterwards to decide what to report, so a hand-off
    /// still in flight would make that decision on partial output.
    func testEveryLineIsAppliedBeforeRunReturns() async throws {
        let box = LineBox()
        _ = try await bootstrap().run(
            URL(fileURLWithPath: "/bin/sh"),
            ["-c", "i=0; while [ $i -lt 200 ]; do echo n-$i; i=$((i+1)); done"],
            environment: ["PATH": "/usr/bin:/bin"],
            onLine: { line in box.append(line) })
        // Read with no waiting of any kind: if the pump were still draining,
        // this would be short.
        XCTAssertEqual(box.lines.count, 200)
    }

    /// A failing child yields its status and its output, for `explain`.
    func testAFailingChildReportsItsStatusAndOutput() async throws {
        let result = try await bootstrap().run(
            URL(fileURLWithPath: "/bin/sh"),
            ["-c", "echo 'error: Failed to download' >&2; exit 2"],
            environment: ["PATH": "/usr/bin:/bin"])
        XCTAssertEqual(result.status, 2)
        XCTAssertTrue(result.output.contains("Failed to download"), result.output)
        XCTAssertTrue(Bootstrap.explain(result).contains("network"))
    }

    /// stderr is captured, not only stdout.
    ///
    /// Load-bearing: `uv` writes **all** of its progress to stderr and leaves
    /// stdout empty, so a reader wired only to stdout would show nothing at all.
    func testStderrIsCapturedBecauseThatIsWhereUvWrites() async throws {
        let box = LineBox()
        _ = try await bootstrap().run(
            URL(fileURLWithPath: "/bin/sh"),
            ["-c", "echo 'Downloading torch (106.1MiB)' >&2"],
            environment: ["PATH": "/usr/bin:/bin"],
            onLine: { line in box.append(line) })
        XCTAssertEqual(box.lines, ["Downloading torch (106.1MiB)"])
    }
}

/// Single-flight, and what a refused install may touch.
///
/// From an independent review (MAJOR): the lock used to be taken *after*
/// `progress`, `transcript`, `cancelled`, `isPresenting` and `state` were reset,
/// so a second `install()` — reachable with two clicks, because the menu item
/// was never disabled — wiped the running install's bar and log, set
/// `.failed("Another QDS is already installing…")` over a healthy run, and
/// through its own `defer` closed the window on an install that was still going.
/// The guard fired, but only after the damage it exists to prevent.
final class InstallSingleFlightTests: XCTestCase {
    /// `flock` catches re-entrancy from *this* process, not only a second app.
    ///
    /// The premise the ordering fix rests on: if a same-process second call
    /// acquired the lock, ordering the reset later would fix nothing.
    func testTheLockIsSingleFlightWithinOneProcessToo() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-lock-\(UUID().uuidString)")
        let first = try FileLock(at: path)
        XCTAssertTrue(first.acquired)

        let second = try FileLock(at: path)
        XCTAssertFalse(second.acquired, "a second install in this process would have run concurrently")

        first.release()
        // And the lock is reusable once released, or a cancelled install could
        // never be retried.
        let third = try FileLock(at: path)
        XCTAssertTrue(third.acquired)
        third.release()
    }

    /// A refused install reports itself without disturbing the running one.
    ///
    /// Asserted on the real `Bootstrap`: the second call must leave the first
    /// call's observable state — its progress, its transcript, and the flag the
    /// window is driven by — exactly as it found it.
    @MainActor
    func testARefusedInstallLeavesTheRunningInstallsStateAlone() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qds-flight-\(UUID().uuidString)")
        let paths = Paths(data: directory)
        try paths.ensure()
        defer { try? FileManager.default.removeItem(at: directory) }

        let bootstrap = Bootstrap(paths: paths, onChange: {})

        // Hold the lock as a concurrent install would.
        let held = try FileLock(at: paths.installLock)
        XCTAssertTrue(held.acquired)
        defer { held.release() }

        // Stand in for a run in progress: state the second call must not touch.
        bootstrap.absorbForTesting("Resolved 87 packages in 578ms")
        bootstrap.absorbForTesting("Downloading torch (106.1MiB)")
        let progressBefore = bootstrap.progress
        let transcriptBefore = bootstrap.transcript
        XCTAssertEqual(transcriptBefore.count, 2)

        await bootstrap.install()

        XCTAssertEqual(bootstrap.progress, progressBefore, "the refused install reset the running one's progress")
        XCTAssertEqual(bootstrap.transcript, transcriptBefore, "the refused install wiped the running one's log")
        XCTAssertFalse(bootstrap.isPresenting, "the refused install drove the window")
        XCTAssertEqual(
            bootstrap.refusal, "Another QDS is already installing the server.",
            "the refusal was not reported")
        // And it did not overwrite the state with a failure about a healthy run.
        if case .failed = bootstrap.state {
            XCTFail("a refused install reported the running one as failed")
        }
    }
}

/// The upscaler phase, and where it sits.
final class UpscalerPhaseTests: XCTestCase {
    /// Upscalers are fetched *before* the rewriter.
    ///
    /// Not cosmetic. The rewriter phase is the one carrying a Skip button, and
    /// the two downloads are 42.5 MB against 2.2 GB (both measured). Ordered the
    /// other way, skipping the long download would also throw away the short one
    /// that had not started — so someone who skips a 2.2 GB wait would silently
    /// lose upscaling too.
    func testUpscalersComeBeforeTheRewriter() {
        XCTAssertLessThan(InstallProgress.Phase.upscalers, InstallProgress.Phase.rewriter)
        XCTAssertLessThan(InstallProgress.Phase.packages, InstallProgress.Phase.upscalers)
        // And the step list renders them in that order.
        XCTAssertEqual(
            InstallProgress.Phase.allCases.map(SetupView.name(of:)),
            ["Preparing", "Python runtime", "Server packages", "Upscalers", "Prompt enhancer"])
    }

    /// Both weight phases are optional; nothing before them is.
    ///
    /// The install record is written before either starts, so stopping in one
    /// leaves a complete, working server. The window words its button from this,
    /// and calling it "Cancel" would tell someone they are cancelling an install
    /// they have already finished.
    func testOnlyTheWeightPhasesAreSkippable() {
        XCTAssertFalse(InstallProgress.Phase.preparing.isOptional)
        XCTAssertFalse(InstallProgress.Phase.python.isOptional)
        XCTAssertFalse(InstallProgress.Phase.packages.isOptional)
        XCTAssertTrue(InstallProgress.Phase.upscalers.isOptional)
        XCTAssertTrue(InstallProgress.Phase.rewriter.isOptional)
    }

    func testEachWeightPhaseIsNamedInPlainLanguage() {
        var progress = InstallProgress()
        progress.beginUpscalers()
        XCTAssertEqual(Bootstrap.label(for: progress), "Downloading the upscalers…")
        progress.beginRewriter()
        XCTAssertEqual(Bootstrap.label(for: progress), "Downloading the prompt enhancer…")
    }

    /// Neither weight phase offers a fraction.
    ///
    /// `qds fetch` emits about two JSON lines for the whole download —
    /// `HF_HUB_DISABLE_PROGRESS_BARS=1` is set in `childEnvironmentMap` — so
    /// there is no denominator, and the window draws an indeterminate bar rather
    /// than inventing a percentage.
    func testTheWeightPhasesPublishNoFraction() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Downloading torch (106.1MiB)")
        progress.absorb(" Downloaded torch")
        progress.absorb("Prepared 87 packages in 1s")
        XCTAssertEqual(progress.fraction, 1)

        progress.beginUpscalers()
        XCTAssertNil(progress.fraction, "the upscaler fetch publishes no progress")
        progress.beginRewriter()
        XCTAssertNil(progress.fraction, "the rewriter fetch publishes no progress")
    }

    /// Fetch output must not be counted as a package.
    ///
    /// Found by replaying `qds fetch`'s real output through the parser rather
    /// than by reading it: the measured line
    /// "Downloading upscaler realesrgan-x4plus (33.5 MB) from …" has exactly
    /// uv's announcement shape, and was being counted — a 21st package named
    /// `upscaler realesrgan-x4plus`, appearing after the count had finished.
    /// Nothing on screen revealed it, because these phases publish no fraction,
    /// which is why the assertion is on the counters and not on the label.
    func testFetchOutputIsNotCountedAsAPackage() {
        var progress = InstallProgress()
        progress.absorb("Resolved 87 packages in 6ms")
        progress.absorb("Downloading torch (106.1MiB)")
        progress.absorb(" Downloaded torch")
        progress.absorb("Prepared 87 packages in 1s")
        let totalBefore = progress.packagesTotal
        let bytesBefore = progress.bytesTotal

        progress.beginUpscalers()
        progress.absorb("Fetching realesrgan-x4plus - 33.5 MB from mlx-community/Real-ESRGAN-x4plus")
        progress.absorb("Downloading upscaler realesrgan-x4plus (33.5 MB) from mlx-community/Real-ESRGAN-x4plus")
        progress.absorb(" Downloaded realesrgan-x4plus")
        progress.absorb("realesrgan-x4plus ready.")

        XCTAssertEqual(progress.packagesTotal, totalBefore, "a log line was counted as a package")
        XCTAssertEqual(progress.bytesTotal, bytesBefore, "a log line added bytes to the denominator")
        XCTAssertNil(progress.currentPackage, "a log line was reported as the current download")

        // The same for the rewriter phase, whose output has the same shape.
        progress.beginRewriter()
        progress.absorb("Fetching qwen3-4b-2507-4bit - 2263 MB from mlx-community/Qwen3-4B-Instruct-2507-4bit")
        XCTAssertEqual(progress.packagesTotal, totalBefore)
        XCTAssertNil(progress.currentPackage)
    }

    /// The phase never regresses across the whole run, upscalers included.
    ///
    /// `qds fetch` output is ordinary log text, and a line of it that happened to
    /// start with `Downloading ` must not drag the window back to the packages
    /// phase — which is exactly the shape `qds` logs
    /// ("Downloading upscaler realesrgan-x4plus (33.5 MB) from …").
    func testFetchOutputCannotDragThePhaseBackToPackages() {
        var progress = InstallProgress()
        progress.absorb("Prepared 87 packages in 1s")
        progress.beginUpscalers()

        // Real lines from `qds fetch`, measured against the installed server.
        progress.absorb("Fetching realesrgan-x4plus - 33.5 MB from mlx-community/Real-ESRGAN-x4plus")
        progress.absorb("Downloading upscaler realesrgan-x4plus (33.5 MB) from mlx-community/Real-ESRGAN-x4plus")
        progress.absorb("realesrgan-x4plus ready.")

        XCTAssertEqual(progress.phase, .upscalers, "fetch output dragged the phase backwards")
        XCTAssertEqual(Bootstrap.label(for: progress), "Downloading the upscalers…")
        XCTAssertNil(progress.fraction, "a log line was mistaken for a package download")
    }
}
