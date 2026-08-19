import Foundation

/// What the tray reads from the server.
///
/// Three things, and each is read the way it is published rather than the way
/// that would be easiest here: `/health` on a timer because it is a snapshot,
/// `/v1/progress` as a stream because generation progress changes several times
/// a second, and `/admin/jobs` on a timer *only while a job is running* because
/// polling something idle every two seconds for hours is a log full of nothing.
///
/// Deliberately read-only except for the three commands the menu offers. The
/// dashboard is where the server is configured; this is a status line with a
/// few buttons.
struct Health: Decodable {
    var status: String
    var version: String
    var loaded_model: String?
    var error: String?
}

struct Progress: Decodable {
    var state: String
    var model: String?
    var step: Int
    var total: Int

    var isGenerating: Bool { state != "idle" }
}

struct JobStatus: Decodable {
    var state: String
    var kind: String?
    var target: String?
    var message: String?

    var isActive: Bool { state == "running" || state == "cancelling" }
}

actor ServerClient {
    private var config: ServerConfig
    private let session: URLSession
    private let tokenFile: URL
    private var cachedToken: String?
    private var tokenReadAt: Date = .distantPast

    init(config: ServerConfig, tokenFile: URL) {
        self.config = config
        self.tokenFile = tokenFile
        let configuration = URLSessionConfiguration.ephemeral
        // A stalled request must not outlive the poll interval that issued it.
        configuration.timeoutIntervalForRequest = 10
        self.session = URLSession(configuration: configuration)
    }

    func update(config: ServerConfig) {
        self.config = config
    }

    /// The local token, re-read when it looks stale.
    ///
    /// Cached rather than read per request — this polls every few seconds — but
    /// re-read on a schedule and on demand, because the server issues a new one
    /// every time it starts and the tray outlives several server lifetimes.
    private func adminToken(force: Bool = false) -> String? {
        if !force, let cached = cachedToken, Date().timeIntervalSince(tokenReadAt) < 5 {
            return cached
        }
        cachedToken = (try? String(contentsOf: tokenFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        tokenReadAt = Date()
        return cachedToken?.isEmpty == false ? cachedToken : nil
    }

    private func request(_ path: String, method: String = "GET") -> URLRequest {
        var request = URLRequest(url: config.baseURL.appendingPathComponent(path))
        request.httpMethod = method
        if let key = config.apiKey {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        // The control plane takes its own credential. `/v1` does not: a token
        // presented where an API key belongs authenticates nothing, by design.
        if path.hasPrefix("admin/"), let token = adminToken() {
            request.setValue(token, forHTTPHeaderField: "X-QDS-Admin-Token")
        }
        return request
    }

    private func decode<T: Decodable>(_ type: T.Type, _ path: String, method: String = "GET")
        async throws -> T
    {
        let (data, response) = try await session.data(for: request(path, method: method))
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw QDSError("HTTP \(status) from \(path)")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    /// Never gated, so this stays a reachability check rather than one that also
    /// depends on the key being right.
    func health() async throws -> Health { try await decode(Health.self, "health") }
    /// Job progress, retried once with a freshly read token.
    ///
    /// The server rewrites the token at every start, so the first call after a
    /// restart is expected to fail on a stale one. Swallowing that silently is
    /// what made the tray's job line vanish without explanation.
    func jobs() async throws -> JobStatus {
        do {
            return try await decode(JobStatus.self, "admin/jobs")
        } catch {
            _ = adminToken(force: true)
            return try await decode(JobStatus.self, "admin/jobs")
        }
    }

    @discardableResult
    func cancelGeneration() async throws -> Bool {
        _ = try await session.data(for: request("v1/cancel", method: "POST"))
        return true
    }

    @discardableResult
    func unload() async throws -> Bool {
        _ = try await session.data(for: request("v1/unload", method: "POST"))
        return true
    }

    /// Progress frames, until the caller stops iterating.
    ///
    /// `URLSession.bytes` line by line rather than a decoding wrapper: SSE is
    /// framed by blank lines and `data:` prefixes, and the keep-alive `: ping`
    /// comments carry no payload. Reconnection is the caller's — one loop owning
    /// one connection is what keeps "only one stream at a time" structural.
    func progressFrames() -> AsyncThrowingStream<Progress, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = self.request("v1/progress")
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    request.timeoutInterval = 3600
                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                        throw QDSError("progress stream refused")
                    }
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data:") else { continue }
                        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                        guard !payload.isEmpty, let data = payload.data(using: .utf8) else {
                            continue
                        }
                        continuation.yield(try JSONDecoder().decode(Progress.self, from: data))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// First retry, and the ceiling. Doubling in between.
///
/// The same bounds the dashboard uses, for the same reason: a server that is
/// genuinely down should be retried every ten seconds forever — often enough
/// that starting it recovers the tray without touching anything, rarely enough
/// that a stopped server does not fill the log for as long as the app is open.
func reconnectDelay(attempt: Int) -> Duration {
    .milliseconds(min(10_000, 500 * (1 << max(0, min(attempt, 5)))))
}
