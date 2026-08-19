// swift-tools-version: 6.0
import PackageDescription

// A SwiftPM executable rather than an Xcode project: the whole thing is driven
// from the Makefile, and an `.xcodeproj` is a second place for build settings to
// live and drift. `scripts/bundle-menubar.sh` wraps the binary produced here
// into QDS.app.
//
// Swift 5 language mode on purpose. The app is a small amount of UI over three
// long-lived actors, and Swift 6's strict concurrency checking would mostly be
// satisfied by annotating code that is already single-threaded by construction —
// everything user-facing runs on the main actor, and the one place that does not
// (`Supervisor`'s reaper) is explicitly hopped back.
let package = Package(
    name: "QDS",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "QDS",
            path: "Sources/QDS",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "QDSTests",
            dependencies: ["QDS"],
            path: "Tests/QDSTests",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
