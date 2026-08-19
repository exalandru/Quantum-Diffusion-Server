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
