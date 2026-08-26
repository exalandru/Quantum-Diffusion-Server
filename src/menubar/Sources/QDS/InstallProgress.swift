import Foundation

/// What an install is doing, derived from the installer's own output.
///
/// A separate value from `Bootstrap.State` because they answer different
/// questions: that one says whether a server is installed, this one says how far
/// along the installation currently running has got. Collapsing them would make
/// "installing, 40% of the packages" unrepresentable, which is the whole point
/// of the window this feeds.
///
/// **Pure.** No process, no file, no clock, no main actor: `absorb` is a
/// function from a line of text to a state change, so the parsing — the part
/// that can be silently wrong when uv changes its wording — is the part a test
/// can reach directly. Everything that has to touch a pipe lives in `Bootstrap`.
struct InstallProgress: Equatable {
    /// The four stretches an install passes through, in order.
    ///
    /// Ordered so `>=` means "has reached", which is what draws the step list.
    /// They are *phases of the run*, not steps in a script: a warm cache skips
    /// straight from `.preparing` to `.installing` with no download at all, and
    /// nothing here may assume otherwise.
    enum Phase: Int, Comparable, CaseIterable {
        /// Spawned, nothing measurable said yet.
        case preparing
        /// uv is fetching a managed CPython. One file, and the only phase that
        /// needs the network before anything else can happen.
        case python
        /// Resolving and downloading the wheels.
        case packages
        /// `qds fetch --upscalers`: the Real-ESRGAN checkpoints.
        ///
        /// Before the rewriter, deliberately. The whole upscaler catalogue is
        /// 42.5 MB and takes about five seconds — measured — against 2.2 GB for
        /// the rewriter, and the rewriter phase is the one carrying a *Skip*
        /// button. Ordered the other way, skipping the long download would also
        /// throw away the short one that had not started yet.
        case upscalers
        /// `qds fetch --rewriter`: the prompt enhancer's weights.
        case rewriter

        static func < (lhs: Phase, rhs: Phase) -> Bool { lhs.rawValue < rhs.rawValue }

        /// Whether stopping here leaves a working install behind.
        ///
        /// True from `.upscalers` on: the install record is written before
        /// either weight fetch starts, so the server is complete and usable and
        /// what is being abandoned is an optional download. The window words its
        /// button from this — *Skip* rather than *Cancel* — because telling
        /// someone they are cancelling an install they have already completed is
        /// how a safe action gets refused.
        var isOptional: Bool { self >= .upscalers }
    }

    private(set) var phase: Phase = .preparing

    /// Packages uv has announced, and how many have arrived.
    ///
    /// A count for the *label*, never for the bar — see `fraction`. uv announces
    /// every download within a few milliseconds of the first, so the denominator
    /// settles almost immediately, but it is still a number that can grow and
    /// the bar must not be driven by it.
    private(set) var packagesTotal = 0
    private(set) var packagesDone = 0

    /// The same downloads weighed by size, which is what the bar shows.
    ///
    /// uv states each package's size on its `Downloading` line, so this is
    /// measured rather than assumed. It matters more than it looks: on this
    /// machine torch is 106.1 MiB of a 280 MiB install and pygments is 1.2 MiB,
    /// so a bar driven by the package *count* jumps 5% for a file that took no
    /// time and then sits still through the one download that does.
    private(set) var bytesTotal: Double = 0
    private(set) var bytesDone: Double = 0

    /// The largest download still outstanding, for the label.
    ///
    /// The largest rather than the most recent: with twenty concurrent
    /// downloads "most recent" flickers between names every few milliseconds
    /// and reads as noise, while the biggest outstanding one is both stable and
    /// the honest answer to "what is it waiting for".
    private(set) var currentPackage: String?

    /// Whether uv has said every wheel is on disk (`Prepared` or `Installed`).
    ///
    /// The difference between "not downloading yet" and "no longer downloading",
    /// which `currentPackage == nil` alone cannot tell apart. Measured against
    /// the real installer: `Resolved` lands ~130ms before the first
    /// `Downloading`, so without this the window said "Installing the server…"
    /// and then went *back* to "Downloading torch" — a label that walks
    /// backwards, which reads as a bug in the installer.
    private(set) var finishedDownloading = false

    /// Sizes by package name, so a `Downloaded` line can be credited the bytes
    /// its `Downloading` line announced. uv does not repeat the size.
    private var announced: [String: Double] = [:]
    private var outstanding: Set<String> = []

    /// Where the bar should be, or `nil` when no honest fraction exists.
    ///
    /// `nil` is a real answer and the reason this is optional: during
    /// `.preparing`, during the CPython download, and throughout the rewriter
    /// fetch, nothing has published a denominator. A window that showed 0% —
    /// or worse, a made-up percentage — through the longest silence in the
    /// install would be exactly the lie this feature exists to remove. The view
    /// draws an indeterminate bar for `nil`.
    var fraction: Double? {
        guard phase == .packages, bytesTotal > 0 else { return nil }
        return min(1, bytesDone / bytesTotal)
    }

    /// Every phase up to and including this one has been entered.
    func hasReached(_ other: Phase) -> Bool { phase >= other }

    // ── Reading the installer's output ─────────────────────────────────────

    /// Advance the state by one line of installer output.
    ///
    /// Tolerant by construction: an unrecognised line changes nothing. uv's
    /// wording is not a contract, and the failure mode of a strict parser here
    /// would be a window stuck at "Preparing…" for four minutes — strictly
    /// worse than today, where at least the menu says something. Every claim
    /// below was read off a real run against the bundled wheel.
    mutating func absorb(_ raw: String) {
        let line = raw.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty else { return }

        // Past the packages phase, nothing in the output is a package.
        //
        // `qds fetch` writes ordinary log text, and some of it starts with the
        // same word uv uses — "Downloading upscaler realesrgan-x4plus (33.5 MB)
        // from …" is a real measured line. Parsed as an announcement it added a
        // 21st "package" called `upscaler realesrgan-x4plus` to a count that had
        // already finished. Nothing on screen showed it, because these phases
        // publish no fraction, which is precisely why it needed catching here
        // rather than being left to be noticed later.
        //
        // The transcript still records the line; only the *counting* stops.
        guard phase < .upscalers else { return }

        // The managed CPython. Matched before the general package rules because
        // it has the same shape and is not one of the 87 packages: counting it
        // would make the denominator wrong by one and put a 23.8 MiB download
        // into a total uv has not announced yet.
        if line.hasPrefix("Downloading cpython-") {
            phase = .python
            return
        }
        if line.hasPrefix("Downloaded cpython-") {
            // Downloaded, not yet resolved: `Resolved` below is what moves the
            // phase on, so an install whose resolution is slow does not sit on
            // a completed step.
            return
        }

        // Resolution is the boundary: before it uv is reading metadata, after it
        // it is fetching wheels. It appears whether or not a Python was needed.
        if line.hasPrefix("Resolved ") && line.contains(" package") {
            phase = max(phase, .packages)
            return
        }

        if line.hasPrefix("Downloading ") {
            phase = max(phase, .packages)
            let name = Self.packageName(afterPrefix: "Downloading ", in: line)
            guard !name.isEmpty else { return }
            let size = Self.bytes(in: line) ?? 0
            // Idempotent on the name: a repeated announcement must not count
            // twice. uv does not repeat them today; a parser that breaks when it
            // does is a parser that breaks silently.
            if announced[name] == nil {
                packagesTotal += 1
                bytesTotal += size
                announced[name] = size
                outstanding.insert(name)
            }
            currentPackage = Self.largest(of: outstanding, by: announced)
            return
        }

        if line.hasPrefix("Downloaded ") {
            let name = Self.packageName(afterPrefix: "Downloaded ", in: line)
            guard !name.isEmpty, outstanding.contains(name) else { return }
            outstanding.remove(name)
            packagesDone += 1
            bytesDone += announced[name] ?? 0
            currentPackage = Self.largest(of: outstanding, by: announced)
            return
        }

        // Every wheel is on disk. Said even when nothing was downloaded at all,
        // which is the warm-cache path an update takes: `Resolved` then
        // `Prepared` with no `Downloading` between them, so the bar was never
        // determinate and must not now be shown as complete.
        //
        // `Installed` counts too: on a fully warm cache uv skips `Prepared`
        // entirely and goes straight from `Resolved` to `Installed`.
        if (line.hasPrefix("Prepared ") || line.hasPrefix("Installed "))
            && line.contains(" package")
        {
            phase = max(phase, .packages)
            finishedDownloading = true
            outstanding.removeAll()
            currentPackage = nil
            // Only when there was something to finish. `bytesDone = bytesTotal`
            // with both at zero is not "100%", it is still "no fraction", and
            // `fraction` returns nil for it either way.
            bytesDone = bytesTotal
            packagesDone = packagesTotal
            return
        }
    }

    /// Enter the upscaler phase. Driven by the caller rather than by a line,
    /// because it is a different process: `qds fetch --upscalers`.
    mutating func beginUpscalers() {
        phase = max(phase, .upscalers)
        currentPackage = nil
    }

    /// Enter the rewriter phase. Driven by the caller rather than by a line,
    /// because it is a different process: `qds fetch --rewriter`, started only
    /// after the install record is written.
    mutating func beginRewriter() {
        phase = max(phase, .rewriter)
        currentPackage = nil
    }

    // ── Parsing helpers ────────────────────────────────────────────────────

    /// The package name in `Downloading torch (106.1MiB)`.
    static func packageName(afterPrefix prefix: String, in line: String) -> String {
        guard line.hasPrefix(prefix) else { return "" }
        let rest = line.dropFirst(prefix.count)
        // Up to the size in parentheses, when there is one. `Downloaded` lines
        // carry no size at all, so the whole remainder is the name.
        let name = rest.prefix { $0 != "(" }
        return name.trimmingCharacters(in: .whitespaces)
    }

    /// The size in the *last* parenthesised group, in bytes.
    ///
    /// The last, because uv writes `Downloading cpython-… (download) (23.8MiB)`
    /// — two groups, and the first is not a size. Returns `nil` rather than 0
    /// for a line with no size, so "no size stated" and "zero bytes" stay
    /// different facts.
    static func bytes(in line: String) -> Double? {
        var best: Double?
        var current = ""
        var inside = false
        for character in line {
            if character == "(" {
                inside = true
                current = ""
            } else if character == ")" {
                if inside, let value = parseSize(current) { best = value }
                inside = false
            } else if inside {
                current.append(character)
            }
        }
        return best
    }

    /// `106.1MiB` → bytes. Binary units, as uv prints them.
    static func parseSize(_ text: String) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        let units: [(String, Double)] = [
            ("GiB", 1024 * 1024 * 1024), ("MiB", 1024 * 1024), ("KiB", 1024), ("B", 1),
        ]
        for (suffix, scale) in units where trimmed.hasSuffix(suffix) {
            let number = trimmed.dropLast(suffix.count).trimmingCharacters(in: .whitespaces)
            guard let value = Double(number) else { return nil }
            return value * scale
        }
        return nil
    }

    private static func largest(of names: Set<String>, by sizes: [String: Double]) -> String? {
        names.max { (sizes[$0] ?? 0, $1) < (sizes[$1] ?? 0, $0) }
    }
}
