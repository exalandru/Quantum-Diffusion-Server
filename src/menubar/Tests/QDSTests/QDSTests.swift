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
