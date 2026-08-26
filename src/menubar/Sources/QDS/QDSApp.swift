import AppKit
import SwiftUI

/// The menubar app.
///
/// No window, no Dock icon (`LSUIElement` in the bundle's Info.plist). What
/// replaced the Tauri window is the dashboard the *server* serves: one interface,
/// written once, reachable from here and from any browser on the machine. This
/// app is what a web page cannot be — the thing that installs the server,
/// starts it, and is still there when it is not running.
@main
struct QDSApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        MenuBarExtra {
            MenuContent(model: delegate.model)
        } label: {
            MenuBarLabel(running: delegate.model.isRunning)
        }
        .menuBarExtraStyle(.menu)
    }
}

/// The two symbols the status item can show, and a guarantee it shows one.
///
/// This exists because it did not, and the app was invisible. The stopped state
/// used `sparkles.slash`, which is not a real SF Symbol; `Image(systemName:)`
/// renders nothing for a name that does not resolve, and the app starts
/// stopped — so the icon was *always* the missing one. Nothing crashed and
/// nothing was logged: the app simply appeared not to launch.
///
/// So: one filled shape and one outline of the *same* symbol, which is how a
/// menu bar reads state without changing what the icon is; both names checked
/// by a test; and a text fallback below, so a future rename can at worst make
/// the item ugly rather than absent.
enum MenuBarIcon {
    static let running = "sparkles.rectangle.stack.fill"
    static let stopped = "sparkles.rectangle.stack"

    /// Every name this app can pass to `Image(systemName:)`.
    static let all = [running, stopped]

    static func resolves(_ name: String) -> Bool {
        NSImage(systemSymbolName: name, accessibilityDescription: nil) != nil
    }
}

struct MenuBarLabel: View {
    let running: Bool

    var body: some View {
        let name = running ? MenuBarIcon.running : MenuBarIcon.stopped
        Group {
            if MenuBarIcon.resolves(name) {
                // A template image, so it inverts with the menu bar rather than
                // staying dark on a dark bar.
                Image(systemName: name)
            } else {
                // Never nothing. An unreadable menu bar item is still findable;
                // an absent one reads as an app that failed to start.
                Text("QDS")
            }
        }
        // Without this the item announces itself as its symbol name — VoiceOver
        // read "sparkles rectangle stack", which names the picture rather than
        // the thing, and says nothing about the state the icon is there to show.
        .accessibilityLabel(running ? "QDS - server running" : "QDS - server stopped")
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = MenuModel()

    private var signalSources: [DispatchSourceSignal] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        // No Dock icon and no menu bar of our own: this is a status item.
        NSApp.setActivationPolicy(.accessory)
        // A server this app started and never stopped is still out there.
        model.adoptOrphan()
        installSignalHandlers()
        // At most one request a day, and none at all if the last one was
        // recent — a launch is not by itself a reason to ask GitHub anything.
        model.checkForNewVersion()
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Synchronous, because this is the last moment there is. A child left
        // behind becomes an orphan under launchd, holding the port.
        model.terminate()
    }

    /// Stop the server when we are told to exit, not only when we are asked.
    ///
    /// `applicationWillTerminate` runs for Quit and for a logout, and **not**
    /// for a bare `SIGTERM` — which is what `kill`, `pkill` and a launchd
    /// shutdown send. Without this the server outlives us and keeps the port,
    /// and the next Start reports "port already in use" about a process nobody
    /// can see.
    ///
    /// This still cannot cover `SIGKILL` (Force Quit) or a crash, and nothing
    /// can: that is what `adoptOrphan` above is for. The two together are the
    /// whole answer — this one keeps it from happening, that one recovers when
    /// it does.
    private func installSignalHandlers() {
        for number in [SIGTERM, SIGINT, SIGHUP] {
            // The default disposition has to go, or the process dies before the
            // handler ever runs.
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
            source.setEventHandler { [weak self] in
                self?.model.terminate()
                exit(0)
            }
            source.resume()
            signalSources.append(source)
        }
    }
}

struct MenuContent: View {
    @Bindable var model: MenuModel

    var body: some View {
        Text(model.statusLine)

        if let note = model.note {
            Divider()
            Text(note)
        }

        Divider()

        Button("Open Playground") { model.openPlayground() }
            .disabled(!model.isRunning)
        Button("Open Dashboard") { model.openDashboard() }
            .disabled(!model.isRunning)

        Divider()

        // Only while an install is running: a way back to a window the user
        // closed. Absent the rest of the time, so the menu does not carry a
        // permanent item about something that is not happening.
        if model.isInstalling {
            Button("Show Setup Progress…") { model.showSetupProgress() }

            Divider()
        }

        if model.isRunning {
            Button("Stop Server") { model.stopServer() }
            Button("Restart Server") { model.restartServer() }
            // Offered while running, because that is when you notice: the app
            // was rebuilt, the server it started is the previous build, and
            // stopping first to be told so would be a step for nothing.
            if model.updateAvailable {
                Button("Update Server & Restart") { model.updateServer() }
            }
        } else {
            Button(model.startLabel) { model.startServer() }
        }

        Button("Cancel Generation") { model.cancelGeneration() }
            .disabled(!model.canCancelGeneration)
        Button("Free Memory") { model.freeMemory() }
            .disabled(!model.hasWarmModel)

        Divider()

        Toggle("Launch at Login", isOn: $model.launchesAtLogin)

        Divider()

        // Only when there is one, so the menu does not carry a permanent "you
        // are up to date" line nobody needs to read. This is the *app*, not the
        // server — see `MenuModel.newVersion`. The version is the normalised
        // one, so the menu and the window name it identically whichever way the
        // tag happened to be spelled.
        if let release = model.newVersion {
            Button("QDS \(release.version?.description ?? release.tagName) is available…") {
                model.openNewVersion()
            }
        }

        Button("About QDS") { model.showAbout() }

        // Named for what it does. "Quit" alone would be a lie: the server is a
        // child of this process, and it goes too.
        Button("Quit and Stop Server") { NSApplication.shared.terminate(nil) }
            .keyboardShortcut("q")
    }
}
