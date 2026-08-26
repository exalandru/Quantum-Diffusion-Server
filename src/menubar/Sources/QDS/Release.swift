import Foundation

/// Who this application is, written once.
///
/// The repository coordinates decide three things — where "Show on GitHub"
/// goes, whose profile the credit links to, and which releases the update check
/// reads — and those three must never be able to disagree.
enum Product {
    static let owner = "exalandru"
    static let repository = "Quantum-Diffusion-Server"

    static let displayName = "Quantum Diffusion Server"

    static var authorURL: URL { URL(string: "https://github.com/\(owner)")! }
    static var repositoryURL: URL { URL(string: "https://github.com/\(owner)/\(repository)")! }

    /// The newest release GitHub considers current. This endpoint already
    /// excludes drafts and prereleases, which is why the check does not have to
    /// page through `/releases` and pick one — though the two flags are still
    /// verified below, because trusting a remote server's filtering to be the
    /// only guard is how a prerelease ends up advertised as an update.
    static var latestReleaseAPI: URL {
        URL(string: "https://api.github.com/repos/\(owner)/\(repository)/releases/latest")!
    }

    /// This build's version, from the bundle rather than from a constant here.
    ///
    /// `scripts/bundle-menubar.sh` reads it out of `src/server/pyproject.toml`
    /// and writes it into `CFBundleShortVersionString`, so the wheel, the app
    /// and the tag it is compared against all descend from one number. A
    /// constant in this file would be a second place for it to be written down,
    /// and the two would drift the first time only one of them was bumped.
    ///
    /// `nil` in a `swift run` from a checkout, where there is no `.app` around
    /// the binary and therefore no Info.plist. The update check treats that as
    /// "cannot know", never as "up to date" — see `ReleaseState`.
    static var version: String? {
        guard let raw = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
            !raw.trimmingCharacters(in: .whitespaces).isEmpty
        else { return nil }
        return raw
    }
}

/// A dotted version, ordered by number rather than by spelling.
///
/// The whole reason this type exists rather than a `String` comparison: `"2.9.0"`
/// sorts *after* `"2.10.0"` alphabetically, so a string compare would go quiet
/// exactly when the tenth minor release shipped and stay quiet forever after.
struct Version: Comparable, CustomStringConvertible {
    /// The numeric components, most significant first.
    let components: [Int]

    /// The canonical spelling: the numeric components, dotted.
    ///
    /// Normalised rather than the tag as written, because the tag is written
    /// inconsistently — this repository has both `1.0.0` and `v2.1.0` — and the
    /// UI puts the word "Version" in front of it. Echoing the raw tag produced
    /// "Version v2.3.0", with the `v` doubling the word already there.
    var description: String { components.map(String.init).joined(separator: ".") }

    /// Parse a release tag, or refuse it.
    ///
    /// Tolerates the `v` prefix, because this repository's own tags are
    /// inconsistent about it — `1.0.0` and `v2.1.0` are both real tags here —
    /// and a check that only understood one of the two spellings would silently
    /// stop working the next time the other was used.
    ///
    /// Anything after the numeric core (`-rc1`, `+build`) is **not ordered**:
    /// `2.3.0-rc1` and `2.3.0` compare equal, so a prerelease can never be
    /// reported as newer than the matching final. That is the safe direction,
    /// and GitHub's `prerelease` flag is the mechanism that actually excludes
    /// them.
    ///
    /// Refuses rather than guesses. This string comes off the network, and a
    /// component that does not fit in an `Int` — or is not a number at all —
    /// must not be coerced into an ordering. `nil` here reaches the UI as
    /// "could not check", which is the fail-closed answer.
    init?(_ raw: String) {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.first == "v" || text.first == "V" { text.removeFirst() }

        let core = text.prefix { $0.isNumber || $0 == "." }
        let parts = core.split(separator: ".", omittingEmptySubsequences: false)
        guard !parts.isEmpty else { return nil }

        var numbers: [Int] = []
        for part in parts {
            // `Int(_:)` answers nil for an empty component ("2..0", "2.") and
            // for anything wider than 64 bits, so both land here.
            guard let number = Int(part), number >= 0 else { return nil }
            numbers.append(number)
        }

        components = numbers
    }

    /// Compared component by component, missing components reading as zero, so
    /// `2.1` and `2.1.0` are the same version rather than two.
    static func < (left: Version, right: Version) -> Bool {
        let width = max(left.components.count, right.components.count)
        for index in 0..<width {
            let a = index < left.components.count ? left.components[index] : 0
            let b = index < right.components.count ? right.components[index] : 0
            if a != b { return a < b }
        }
        return false
    }

    static func == (left: Version, right: Version) -> Bool {
        !(left < right) && !(right < left)
    }
}

/// A published release, as much of one as this app has any use for.
///
/// `draft` and `prerelease` are required rather than defaulted: a response that
/// does not carry them cannot be shown to be a stable release, and a decode
/// failure surfaces as "could not check" — which is the honest answer — instead
/// of a missing flag defaulting to `false` and advertising a draft.
struct Release: Codable, Equatable {
    var tagName: String
    var htmlURL: String
    var draft: Bool
    var prerelease: Bool
    var publishedAt: Date?

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case htmlURL = "html_url"
        case draft
        case prerelease
        case publishedAt = "published_at"
    }

    var version: Version? { Version(tagName) }

    var url: URL { URL(string: htmlURL) ?? Product.repositoryURL }

    /// Stable, and nameable as a version. Anything else is not something to
    /// offer somebody as an update.
    var isOfferable: Bool { !draft && !prerelease && version != nil }
}

/// What the update check found, including the two ways it can find nothing.
///
/// `unknown` and `failed` are kept apart from `current` deliberately. "We are up
/// to date" is a claim; "we could not ask" and "we do not know what version we
/// are" are not, and collapsing all three into one state would have the window
/// reassure the user about something it never established.
enum ReleaseState: Equatable {
    /// Not asked yet, or this build cannot name its own version.
    case unknown
    case checking
    /// This build is at or ahead of the newest published release.
    case current
    case available(Release)
    /// The network, GitHub, or the response was not usable.
    case failed
}

/// Asking GitHub what the newest release is.
///
/// Read-only, unauthenticated, and against one documented endpoint. The
/// unauthenticated limit is 60 requests an hour per address, which the interval
/// below stays three orders of magnitude under — but a 403 from it is still
/// treated as an ordinary failure, because a rate limit shared with everything
/// else on the machine is not this app's to reason about.
actor ReleaseCheck {
    /// How long a result stays good. A day: releases are not frequent enough to
    /// justify more, and an app left open for a week should still notice one.
    static let interval: TimeInterval = 24 * 60 * 60

    /// Whether a check is owed, given when the last one happened.
    ///
    /// A pure function so the policy has a witness. The clock is the trap here:
    /// a machine whose time moved backwards — a correction, a timezone-confused
    /// restore, a user setting it by hand — leaves a `lastChecked` in the
    /// future, and `elapsed >= interval` on a negative elapsed is `false`
    /// *forever*. So a future timestamp counts as due.
    static func isDue(
        lastChecked: Date?, now: Date = Date(), interval: TimeInterval = ReleaseCheck.interval
    ) -> Bool {
        guard let lastChecked else { return true }
        let elapsed = now.timeIntervalSince(lastChecked)
        if elapsed < 0 { return true }
        return elapsed >= interval
    }

    private let session: URLSession

    init() {
        let configuration = URLSessionConfiguration.ephemeral
        // Nothing waits on this, but a request left open forever is a task that
        // never ends and a connection nobody closes.
        configuration.timeoutIntervalForRequest = 15
        configuration.waitsForConnectivity = false
        session = URLSession(configuration: configuration)
    }

    func latest() async throws -> Release {
        var request = URLRequest(url: Product.latestReleaseAPI)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        // GitHub refuses requests without one.
        request.setValue(
            "QDS/\(Product.version ?? "dev") (+\(Product.repositoryURL.absoluteString))",
            forHTTPHeaderField: "User-Agent")

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw QDSError("HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0) from GitHub")
        }
        return try Self.decode(data)
    }

    /// Separated from the request so the parse has a witness that does not need
    /// the network.
    static func decode(_ data: Data) throws -> Release {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(Release.self, from: data)
    }
}

/// The last answer, so a relaunch does not have to be silent for a day.
///
/// GitHub is the source of truth; this is bounded derived state whose only
/// effect is a menu item and a line in a window. It is re-derived on the next
/// due check and never merged with anything — a stale entry can at worst
/// re-offer a release the user already has, which the comparison then rejects.
struct ReleaseCache {
    private static let releaseKey = "latestRelease"
    private static let checkedKey = "latestReleaseCheckedAt"

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    var lastChecked: Date? { defaults.object(forKey: Self.checkedKey) as? Date }

    var release: Release? {
        guard let data = defaults.data(forKey: Self.releaseKey) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(Release.self, from: data)
    }

    func store(_ release: Release?, checkedAt: Date = Date()) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if let release, let data = try? encoder.encode(release) {
            defaults.set(data, forKey: Self.releaseKey)
        } else {
            defaults.removeObject(forKey: Self.releaseKey)
        }
        defaults.set(checkedAt, forKey: Self.checkedKey)
    }
}
