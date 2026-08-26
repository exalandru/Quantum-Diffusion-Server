import AppKit
import SwiftUI

/// The palette, from the dashboard's tokens.
///
/// Written out rather than derived from the system: this window is deliberately
/// dark whatever the system theme is, because it is a piece of QDS rather than a
/// standard About box, and a light one would be the only light surface the
/// product has. Values mirror `src/dashboard/src/styles.css`; there is no
/// mechanism keeping them in step, and there does not need to be — five colours
/// on one window is not a design system.
private enum Palette {
    static let bg = Color(red: 0.055, green: 0.078, blue: 0.110)
    static let raised = Color(red: 0.110, green: 0.141, blue: 0.184)
    static let line = Color(red: 0.192, green: 0.235, blue: 0.298)
    static let text = Color(red: 0.910, green: 0.929, blue: 0.961)
    static let muted = Color(red: 0.576, green: 0.631, blue: 0.710)
    static let faint = Color(red: 0.373, green: 0.427, blue: 0.502)
    static let accent = Color(red: 0.369, green: 0.722, blue: 1.0)
    static let accentLine = Color(red: 0.114, green: 0.435, blue: 0.722)
    static let accentTint = Color(red: 0.369, green: 0.722, blue: 1.0, opacity: 0.16)
    static let onAccent = Color(red: 0.024, green: 0.071, blue: 0.118)
}

/// The About window: what this is, which version, who made it, and whether a
/// newer one exists.
struct AboutView: View {
    let model: MenuModel

    var body: some View {
        VStack(spacing: 0) {
            icon
                .padding(.bottom, 20)

            Text(Product.displayName)
                .font(.system(size: 21, weight: .semibold))
                .foregroundStyle(Palette.text)
                .padding(.bottom, 5)

            Text(versionLine)
                .font(.system(size: 13))
                .monospacedDigit()
                .foregroundStyle(Palette.muted)

            updateStrip
                .padding(.top, 22)

            // Built explicitly, never as interpolated markdown.
            //
            // `Text("Crafted by [\(owner)](\(url))")` looks right and is not:
            // a literal string is a `LocalizedStringKey`, and SwiftUI parses
            // the markdown on the *format* string — `"Crafted by [%@](%@)"` —
            // before substituting the arguments. The link's destination is
            // therefore the literal `%@`, which reaches LaunchServices as the
            // relative URL `%25@` and fails with `-50` (`paramErr`). The
            // clickable text renders perfectly, so nothing looks wrong until
            // it is pressed.
            //
            // Interpolating only the *label* is safe; interpolating the
            // destination is not. Rather than rely on remembering which half of
            // the syntax is a trap, the attributed string is assembled here
            // with a real `URL` and no markdown parser involved at all.
            Text(Self.credit)
                .font(.system(size: 13))
                .foregroundStyle(Palette.muted)
                .tint(Palette.accent)
                .padding(.top, 24)

            Button("Show on GitHub") {
                NSWorkspace.shared.open(Product.repositoryURL)
            }
            .buttonStyle(QuietButton())
            .padding(.top, 22)
        }
        .padding(.horizontal, 30)
        .padding(.top, 30)
        .padding(.bottom, 24)
        .frame(width: 380)
        .background(Palette.bg)
        // Every link in this window opens the way the buttons do.
        //
        // `.handled` unconditionally, never `.systemAction` as a fallback: when
        // the destination was malformed the fallback ran SwiftUI's default
        // opener on the same bad URL, so one click produced *two* `-50`
        // dialogs. A failure here is already final — `NSWorkspace` answering
        // `false` means LaunchServices refused the URL, and asking a second
        // mechanism to refuse it again only reports the same thing twice.
        .environment(
            \.openURL,
            OpenURLAction { url in
                NSWorkspace.shared.open(url)
                return .handled
            })
    }

    /// "Crafted by exalandru", with the name carrying a real link.
    ///
    /// Assembled rather than parsed, and exposed so a test can read the link
    /// destination back out — the defect this replaces rendered correctly and
    /// only failed when clicked, so the witness has to inspect the attribute
    /// rather than the appearance.
    static var credit: AttributedString {
        var line = AttributedString("Crafted by ")
        var name = AttributedString(Product.owner)
        name.link = Product.authorURL
        line.append(name)
        return line
    }

    private var icon: some View {
        Group {
            if let image = NSApp.applicationIconImage {
                Image(nsImage: image).resizable()
            } else {
                // A checkout build has no bundle and therefore no icon. An empty
                // frame keeps the layout rather than collapsing it.
                Color.clear
            }
        }
        .frame(width: 108, height: 108)
        .shadow(color: .black.opacity(0.55), radius: 15, y: 7)
    }

    /// "Version 2.2.0", or an honest admission that this build cannot say.
    ///
    /// The second case is a `swift run` from a checkout, where there is no
    /// Info.plist to read. Printing nothing would leave a window whose whole
    /// purpose is to state a version stating none.
    private var versionLine: String {
        guard let version = Product.version else { return "Unversioned build" }
        return "Version \(version)"
    }

    @ViewBuilder
    private var updateStrip: some View {
        // A fixed height across every state, so the window does not resize under
        // the pointer when the check comes back.
        Group {
            switch model.releaseState {
            case .checking:
                Text("Checking for updates…")
                    .font(.system(size: 12))
                    .italic()
                    .foregroundStyle(Palette.faint)

            case .current:
                Text("You're running the latest version.")
                    .font(.system(size: 12))
                    .foregroundStyle(Palette.faint)

            case .available(let release):
                available(release)

            case .failed:
                Text("Could not check for updates.")
                    .font(.system(size: 12))
                    .foregroundStyle(Palette.faint)

            // Never checked, and cannot be: this build does not know its own
            // version, so there is nothing to compare. Saying "up to date" here
            // would be a claim nothing established.
            case .unknown:
                Color.clear
            }
        }
        .frame(maxWidth: .infinity, minHeight: 56)
    }

    private func available(_ release: Release) -> some View {
        HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                // The normalised version, never `release.tagName`: the tag may
                // carry a `v`, and "Version v2.3.0" says the word twice.
                Text("Version \(release.version?.description ?? release.tagName) is available")
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(Palette.text)
                if let published = release.publishedAt {
                    Text("Released \(Self.released(published))")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Palette.muted)
                }
            }
            .fixedSize(horizontal: false, vertical: true)

            Spacer(minLength: 0)

            Button("Get it") { NSWorkspace.shared.open(release.url) }
                .buttonStyle(PrimaryButton())
        }
        .padding(.leading, 13)
        .padding(.trailing, 10)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Palette.accentTint)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Palette.accentLine, lineWidth: 1))
        )
    }

    /// "3 days ago". Localised by the formatter, so this is not a phrase to
    /// assert on in a test — the parse it depends on is.
    static func released(_ date: Date, relativeTo now: Date = Date()) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter.localizedString(for: date, relativeTo: now)
    }
}

private struct QuietButton: ButtonStyle {
    @State private var hovering = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13))
            .foregroundStyle(Palette.text)
            .frame(height: 32)
            .padding(.horizontal, 16)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(hovering || configuration.isPressed ? Palette.line : Palette.raised)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Palette.line, lineWidth: 1))
            )
            .onHover { hovering = $0 }
    }
}

private struct PrimaryButton: ButtonStyle {
    @State private var hovering = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(Palette.onAccent)
            .frame(height: 27)
            .padding(.horizontal, 12)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Palette.accent.opacity(hovering || configuration.isPressed ? 0.85 : 1))
            )
            .onHover { hovering = $0 }
    }
}

/// One window, kept between openings.
///
/// `isReleasedWhenClosed` is the load-bearing line: a code-created `NSWindow`
/// defaults to releasing itself on close, which under ARC — with this
/// controller also holding a strong reference — is an over-release and a crash
/// the second time the user opens About. It has to be turned off explicitly.
@MainActor
final class AboutWindowController: NSObject, NSWindowDelegate {
    private var window: NSWindow?

    func show(model: MenuModel) {
        if window == nil {
            let hosting = NSHostingController(rootView: AboutView(model: model))
            let window = NSWindow(contentViewController: hosting)
            window.styleMask = [.titled, .closable, .fullSizeContentView]
            window.title = "About QDS"
            window.titlebarAppearsTransparent = true
            window.isMovableByWindowBackground = true
            window.isReleasedWhenClosed = false
            window.backgroundColor = NSColor(
                srgbRed: 0.055, green: 0.078, blue: 0.110, alpha: 1)
            // Dark whatever the system is set to: the window is drawn in the
            // product's own palette, and a light title bar over it would be the
            // one seam.
            window.appearance = NSAppearance(named: .darkAqua)
            // Nothing to resize or minimise; the close button is the whole set.
            window.standardWindowButton(.miniaturizeButton)?.isHidden = true
            window.standardWindowButton(.zoomButton)?.isHidden = true
            window.delegate = self
            window.center()
            self.window = window
        }

        // An `.accessory` application is not in the activation order, so a window
        // ordered front without this appears behind whatever the user was in.
        NSApp.activate()
        window?.makeKeyAndOrderFront(nil)
    }
}
