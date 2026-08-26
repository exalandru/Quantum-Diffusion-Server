import AppKit
import SwiftUI

/// The setup window: what the installer is doing, while it does it.
///
/// It exists because an install is 3–4 minutes of complete silence — measured at
/// 3m21s for the packages alone on an empty cache, plus 2.2 GB of prompt
/// enhancer after it — during which the only thing on screen was a menu the user
/// had to open to read. That is indistinguishable from a hung app, which is what
/// people reported it as.
///
/// Titled "Setting up QDS" rather than anything with *update* in it: this window
/// serves both the first install and an update over a running server, and the
/// menu already spends the word "update" twice — on the bundled-versus-installed
/// wheel (`MenuModel.updateAvailable`) and on a newer published release
/// (`MenuModel.newVersion`). A third meaning would make all three unreadable.
struct SetupView: View {
    @Bindable var model: MenuModel

    /// Collapsed by default, as agreed: the log is for when something looks
    /// wrong, and a wall of `uv` output is not what someone wants to watch for
    /// four minutes. A failure opens it — see `outcome`.
    @State private var showingLog = false
    /// Set once by a failure, so opening the log automatically does not fight
    /// the user closing it again.
    @State private var openedForFailure = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.bottom, 20)

            if case .failed(let reason) = model.bootstrapState {
                failureBanner(reason)
                    .padding(.bottom, 16)
            }

            phaseLine
                .padding(.bottom, 9)

            ProgressBar(fraction: model.installProgress.fraction, tone: barTone)

            steps
                .padding(.top, 14)

            logSection
                .padding(.top, 18)

            footer
                .padding(.top, 16)
        }
        .padding(24)
        .frame(width: 460)
        .background(Palette.bg)
        .onChange(of: model.bootstrapState) { _, new in
            // A failure opens the log itself: at that moment its contents are
            // the only thing that can explain what happened, and asking someone
            // to find a disclosure triangle to see the error is asking them to
            // guess that it is there.
            if case .failed = new, !openedForFailure {
                showingLog = true
                openedForFailure = true
            }
        }
    }

    // ── Head ───────────────────────────────────────────────────────────────

    private var header: some View {
        HStack(spacing: 13) {
            Group {
                if let image = NSApp.applicationIconImage {
                    Image(nsImage: image).resizable()
                } else {
                    Color.clear
                }
            }
            .frame(width: 44, height: 44)
            .shadow(color: .black.opacity(0.5), radius: 9, y: 4)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Palette.text)
                Text(subtitle)
                    .font(.system(size: 12.5))
                    .foregroundStyle(Palette.faint)
            }
            Spacer(minLength: 0)
        }
    }

    private var title: String {
        switch model.bootstrapState {
        case .failed: return "The server could not be installed"
        case .ready: return "QDS is ready"
        default: return "Installing the server"
        }
    }

    private var subtitle: String {
        let version = Bootstrap.bundledWheel.flatMap(Bootstrap.version(ofWheel:))
        let name = version.map { "Version \($0)" } ?? "Unversioned build"
        switch model.bootstrapState {
        case .failed: return "\(name) · nothing was changed"
        case .ready: return "\(name) · installed"
        default: return name
        }
    }

    private func failureBanner(_ reason: String) -> some View {
        Text(reason)
            .font(.system(size: 12.5))
            .foregroundStyle(Palette.text)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Palette.downTint)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Palette.down.opacity(0.35), lineWidth: 1))
            )
    }

    // ── Progress ───────────────────────────────────────────────────────────

    private var phaseLine: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(Bootstrap.label(for: model.installProgress))
                .font(.system(size: 13.5))
                .foregroundStyle(Palette.text)
            Spacer(minLength: 0)
            if let detail = sizeDetail {
                Text(detail)
                    .font(.system(size: 12))
                    .monospacedDigit()
                    .foregroundStyle(Palette.faint)
            }
        }
    }

    /// "42.1 MB of 280.4 MB", when a total is known.
    private var sizeDetail: String? {
        let progress = model.installProgress
        guard progress.phase == .packages, progress.bytesTotal > 0 else { return nil }
        return "\(Self.megabytes(progress.bytesDone)) of \(Self.megabytes(progress.bytesTotal))"
    }

    static func megabytes(_ bytes: Double) -> String {
        String(format: "%.1f MB", bytes / (1024 * 1024))
    }

    private var barTone: ProgressBar.Tone {
        switch model.bootstrapState {
        case .failed: return .failed
        case .ready: return .done
        default: return .running
        }
    }

    private var steps: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(InstallProgress.Phase.allCases, id: \.self) { phase in
                stepRow(phase)
            }
        }
    }

    private func stepRow(_ phase: InstallProgress.Phase) -> some View {
        let progress = model.installProgress
        let failed = { if case .failed = model.bootstrapState { return true } else { return false } }()
        let isCurrent = progress.phase == phase
        let isPast = progress.phase > phase || model.isReady

        let mark: String
        let colour: Color
        if failed && isCurrent {
            mark = "✕"
            colour = Palette.down
        } else if isPast {
            mark = "✓"
            colour = Palette.live
        } else if isCurrent {
            mark = "▸"
            colour = Palette.accent
        } else {
            mark = "·"
            colour = Palette.faint
        }

        return HStack(spacing: 8) {
            Text(mark)
                .font(.system(size: 11))
                .foregroundStyle(colour)
                .frame(width: 13)
            Text(Self.name(of: phase))
                .font(.system(size: 12.5))
                .foregroundStyle(
                    failed && isCurrent
                        ? Palette.down
                        : isCurrent ? Palette.text : isPast ? Palette.muted : Palette.faint)
            Spacer(minLength: 0)
        }
    }

    static func name(of phase: InstallProgress.Phase) -> String {
        switch phase {
        case .preparing: return "Preparing"
        case .python: return "Python runtime"
        case .packages: return "Server packages"
        case .upscalers: return "Upscalers"
        case .rewriter: return "Prompt enhancer"
        }
    }

    // ── The log ────────────────────────────────────────────────────────────

    private var logSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            Divider().overlay(Palette.lineSoft)
                .padding(.bottom, 14)

            Button {
                showingLog.toggle()
            } label: {
                HStack(spacing: 7) {
                    Text("▶")
                        .font(.system(size: 9))
                        .foregroundStyle(Palette.faint)
                        .rotationEffect(.degrees(showingLog ? 90 : 0))
                    Text(showingLog ? "Hide details" : "Show details")
                        .font(.system(size: 12.5))
                        .foregroundStyle(Palette.muted)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if showingLog {
                logBody.padding(.top, 11)
            }
        }
    }

    private var logBody: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(model.installTranscript.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(size: 11.5, design: .monospaced))
                            .foregroundStyle(Self.isError(line) ? Palette.down : Palette.muted)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .padding(.horizontal, 11)
                .padding(.vertical, 9)
            }
            .frame(height: 148)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Palette.sunken)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8).stroke(Palette.lineSoft, lineWidth: 1))
            )
            // Follow the tail, so the newest line is the one on screen. Without
            // this the log opens at the top and stays there while the install
            // runs, which reads as frozen — the exact impression this window
            // exists to dispel.
            .onChange(of: model.installTranscript.count) { _, count in
                guard count > 0 else { return }
                withAnimation(.easeOut(duration: 0.15)) {
                    proxy.scrollTo(count - 1, anchor: .bottom)
                }
            }
        }
    }

    static func isError(_ line: String) -> Bool {
        let lowered = line.lowercased()
        return lowered.hasPrefix("error") || lowered.contains("error:")
            || lowered.hasPrefix("warning") || lowered.contains("caused by")
    }

    // ── Foot ───────────────────────────────────────────────────────────────

    private var footer: some View {
        HStack(spacing: 10) {
            Text(footnote)
                .font(.system(size: 12))
                .foregroundStyle(Palette.faint)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)

            if case .failed = model.bootstrapState {
                Button("Copy Log") { model.copyInstallLog() }
                    .buttonStyle(QuietButton())
                Button("Try Again") { model.retryInstall() }
                    .buttonStyle(PrimaryButton())
            } else if model.isInstalling {
                // "Skip" once the record is written — the server is already
                // installed and what remains is an optional download. "Cancel"
                // before that, when stopping really does leave nothing behind.
                Button(model.installProgress.phase.isOptional ? "Skip" : "Cancel") {
                    model.cancelInstall()
                }
                .buttonStyle(QuietButton())
            } else {
                Button("Close") { model.closeSetupWindow() }
                    .buttonStyle(QuietButton())
            }
        }
    }

    private var footnote: String {
        switch model.bootstrapState {
        case .failed: return "Nothing was installed."
        case .ready: return "Closes on its own."
        default:
            if model.installProgress.phase.isOptional {
                return "The server already works — you can skip this."
            }
            return "This runs once."
        }
    }
}

/// The bar: a real fraction when one exists, a travelling sheen when none does.
///
/// The indeterminate case is the point. Three of the four phases publish no
/// denominator at all — `Preparing`, the CPython download, and the whole
/// rewriter fetch, which is the longest single wait in the install — and a bar
/// showing 0% or an invented percentage through them would be the same lie the
/// silence tells today, only with a widget around it.
struct ProgressBar: View {
    enum Tone { case running, done, failed }

    let fraction: Double?
    var tone: Tone = .running

    @State private var sliding = false

    private var fill: Color {
        switch tone {
        case .running: return Palette.accent
        case .done: return Palette.live
        case .failed: return Palette.down
        }
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Palette.sunken)
                    .overlay(Capsule().stroke(Palette.lineSoft, lineWidth: 1))

                if let fraction {
                    Capsule()
                        .fill(fill)
                        .frame(width: max(0, geometry.size.width * fraction))
                        .animation(.easeOut(duration: 0.25), value: fraction)
                } else if tone == .running {
                    Capsule()
                        .fill(fill)
                        .frame(width: geometry.size.width * 0.34)
                        .offset(x: sliding ? geometry.size.width * 0.66 : -geometry.size.width * 0.34)
                        .animation(
                            .easeInOut(duration: 1.6).repeatForever(autoreverses: true),
                            value: sliding
                        )
                        .onAppear { sliding = true }
                } else {
                    // Failed or done with no fraction: a full bar in the tone,
                    // rather than an empty track that says nothing.
                    Capsule().fill(fill)
                }
            }
        }
        .frame(height: 6)
        .clipShape(Capsule())
    }
}

/// One window, kept between openings.
///
/// `isReleasedWhenClosed` is the load-bearing line, for the reason spelled out
/// in `AboutWindowController`: a code-created `NSWindow` releases itself on
/// close, which under ARC with a controller also holding a strong reference is
/// an over-release and a crash the second time it opens.
@MainActor
final class SetupWindowController: NSObject, NSWindowDelegate {
    private var window: NSWindow?

    /// The user closed it during a run, so nothing may reopen it behind their
    /// back. Cleared by `reveal()` — an explicit request — and by `reset()` at
    /// the start of a new install.
    private(set) var dismissed = false

    /// Show the window for an install in progress.
    ///
    /// Idempotent and quiet: called on **every** line of installer output — 130
    /// of them in a measured run — so it must not activate the app or re-order
    /// the window each time. Doing that would drag focus off whatever the user
    /// was typing in, forty times in four seconds, which is a far worse defect
    /// than the silence this window replaces.
    ///
    /// It also honours a close: once the user dismisses it, only `reveal()`
    /// brings it back. Reopening it on the next line would make the close
    /// button appear broken and the `Show Setup Progress…` item pointless.
    func present(model: MenuModel) {
        guard !dismissed else { return }
        // Raise when it is not on screen — **not** merely when the object had
        // to be created. The window is deliberately kept between runs
        // (`isReleasedWhenClosed = false`), so after the first install's
        // auto-close it still exists while being invisible; keying off
        // existence meant the second install (an update, or a retry) updated a
        // window nobody could see and never brought it to the front.
        let visible = window?.isVisible == true
        make(model: model)
        guard !visible else { return }
        NSApp.activate()
        window?.makeKeyAndOrderFront(nil)
    }

    /// Bring it to the front because the user asked for it.
    func reveal(model: MenuModel) {
        dismissed = false
        make(model: model)
        // An `.accessory` application is not in the activation order, so a window
        // ordered front without this appears behind whatever the user was in.
        NSApp.activate()
        window?.makeKeyAndOrderFront(nil)
    }

    /// Forget a previous dismissal, so a new install presents itself.
    func reset() {
        dismissed = false
    }

    private func make(model: MenuModel) {
        guard window == nil else { return }
        let hosting = NSHostingController(rootView: SetupView(model: model))
        let window = NSWindow(contentViewController: hosting)
        window.styleMask = [.titled, .closable, .fullSizeContentView]
        window.title = "Setting up QDS"
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.isReleasedWhenClosed = false
        window.backgroundColor = Palette.windowBackground
        window.appearance = NSAppearance(named: .darkAqua)
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true
        window.delegate = self
        window.center()
        self.window = window
    }

    /// Closing the window never stops the install — the menu keeps reporting it
    /// and `Show Setup Progress…` brings this back.
    func windowWillClose(_ notification: Notification) {
        dismissed = true
    }

    /// Close it, if it is open.
    func close() {
        window?.close()
    }

    var isOpen: Bool { window?.isVisible == true }
}
