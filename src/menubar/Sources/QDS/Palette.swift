import SwiftUI

/// The palette, from the dashboard's tokens.
///
/// Written out rather than derived from the system: these windows are
/// deliberately dark whatever the system theme is, because they are pieces of
/// QDS rather than standard AppKit boxes, and a light one would be the only
/// light surface the product has. Values mirror `src/dashboard/src/styles.css`;
/// there is no mechanism keeping them in step, and there does not need to be —
/// a dozen colours on two windows is not a design system.
///
/// Shared rather than per-window: it was private to `AboutWindow` while there
/// was one window, and a second copy for the setup window would be the point at
/// which the two drift.
enum Palette {
    static let bg = Color(red: 0.055, green: 0.078, blue: 0.110)
    static let raised = Color(red: 0.110, green: 0.141, blue: 0.184)
    /// Below the panel rather than above it: what you read out of. The log body.
    static let sunken = Color(red: 0.063, green: 0.082, blue: 0.110)
    static let line = Color(red: 0.192, green: 0.235, blue: 0.298)
    static let lineSoft = Color(red: 0.102, green: 0.129, blue: 0.169)
    static let text = Color(red: 0.910, green: 0.929, blue: 0.961)
    static let muted = Color(red: 0.576, green: 0.631, blue: 0.710)
    static let faint = Color(red: 0.373, green: 0.427, blue: 0.502)
    static let accent = Color(red: 0.369, green: 0.722, blue: 1.0)
    static let accentLine = Color(red: 0.114, green: 0.435, blue: 0.722)
    static let accentTint = Color(red: 0.369, green: 0.722, blue: 1.0, opacity: 0.16)
    static let onAccent = Color(red: 0.024, green: 0.071, blue: 0.118)
    static let live = Color(red: 0.290, green: 0.871, blue: 0.502)
    static let liveTint = Color(red: 0.290, green: 0.871, blue: 0.502, opacity: 0.15)
    static let warn = Color(red: 0.984, green: 0.749, blue: 0.141)
    static let down = Color(red: 0.973, green: 0.443, blue: 0.443)
    static let downTint = Color(red: 0.973, green: 0.443, blue: 0.443, opacity: 0.15)

    /// The window background as an `NSColor`, for the title bar.
    static let windowBackground = NSColor(
        srgbRed: 0.055, green: 0.078, blue: 0.110, alpha: 1)
}

/// A quiet secondary button, as the About window draws it.
struct QuietButton: ButtonStyle {
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

/// The accent button, for the one action a window is offering.
struct PrimaryButton: ButtonStyle {
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
