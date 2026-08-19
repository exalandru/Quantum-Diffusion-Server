import Darwin
import Foundation

/// The three facts this app needs out of `server-config.json`, and nothing else.
///
/// **Read-only, by design.** The server owns this file: `PUT /admin/config`
/// validates a document against the very model the server boots with, and
/// writes it atomically. This app reads a port so it knows where to connect, a
/// key so it can authenticate, and a grace period so its kill ladder matches
/// the one uvicorn was configured for. Parsing more would be this app forming a
/// second opinion about a schema it does not own.
///
/// The one exception is `seedIfMissing`, below.
struct ServerConfig {
    var host: String = "127.0.0.1"
    var port: Int = 8765
    var apiKey: String?
    var shutdownGrace: TimeInterval = 10

    /// Read what is there. A missing or unreadable file yields the defaults —
    /// deliberately, because this is not the component that validates it: the
    /// server will refuse a broken document far more precisely than a key lookup
    /// here could, and it now stays up in recovery mode to say so. Guessing a
    /// port wrong at worst makes the tray report a server it cannot reach.
    static func read(at url: URL) -> ServerConfig {
        var config = ServerConfig()
        guard
            let data = try? Data(contentsOf: url),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let server = root["server"] as? [String: Any]
        else {
            return config
        }
        if let host = server["host"] as? String, !host.isEmpty { config.host = host }
        if let port = server["port"] as? Int, port > 0, port < 65536 { config.port = port }
        if let key = server["api_key"] as? String, !key.isEmpty { config.apiKey = key }
        if let grace = server["shutdown_grace_s"] as? Double, grace > 0 {
            config.shutdownGrace = grace
        }
        return config
    }

    /// A wildcard bind is an address to listen on, not one to connect to.
    ///
    /// Any *name* falls back to loopback too. `canConnect` passes this to
    /// `inet_addr`, which answers `INADDR_NONE` for anything that is not a
    /// dotted quad — so a hand-edited `host: "nas.local"` would have the tray
    /// report "not listening" about a perfectly healthy server.
    var connectHost: String {
        if ["0.0.0.0", "::", ""].contains(host) { return "127.0.0.1" }
        return inet_addr(host) == INADDR_NONE ? "127.0.0.1" : host
    }

    var baseURL: URL {
        URL(string: "http://\(connectHost):\(port)")!
    }

    /// Write a starting configuration when there is none.
    ///
    /// The only write this app ever makes to this file, and it happens before
    /// any server exists — so the "one writer" rule the server relies on is not
    /// broken by it. Without this the first launch starts a server whose
    /// configuration lives in `site-packages`, which is read-only and which no
    /// one can edit.
    ///
    /// The defaults mirror what the app shipped before: the two enabled models
    /// are Apache-2.0 and ungated, so a fresh install generates with no token,
    /// no access request and no licence to accept.
    static func seedIfMissing(at url: URL) throws {
        guard !FileManager.default.fileExists(atPath: url.path) else { return }

        let document: [String: Any] = [
            "server": [
                "host": "127.0.0.1",
                "port": 8765,
                "api_key": NSNull(),
                "cors_origins": ["*"],
                "max_n": 4,
                // 50 steps on a 32B model far exceed the original 900s.
                "request_timeout_s": 2400,
                "image_ttl_s": 3600,
                "max_upload_mb": 25,
                "default_response_format": "url",
                "log_level": "INFO",
                "progress_log_every": 1,
                "shutdown_grace_s": 10,
                // Release the model after this long without a generation. `null`
                // keeps it warm forever, `0` frees it as soon as a request ends.
                "idle_unload_s": NSNull(),
            ],
            "default_model": "z-image-turbo",
            "default_size": "1280x720",
            "default_quantize": 4,
            "models": [
                "z-image-turbo": ["enabled": true],
                "ernie-image-turbo": ["enabled": true],
                "z-image": ["enabled": false],
                "ernie-image": ["enabled": false],
                "qwen-image-2512": ["enabled": false, "enable_edit": false],
                "flux2-klein": ["enabled": false, "enable_edit": true],
                // Also needs `qds prequantize` before it can answer at all.
                "flux2-dev": ["enabled": false, "quantize": 8, "model_path": NSNull()],
                "fibo-lite": ["enabled": false],
                "fibo": ["enabled": false],
                "ideogram-4": ["enabled": false, "preset": "V4_DEFAULT_20"],
            ],
        ]

        let data = try JSONSerialization.data(
            withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
        // 0600 from creation: this file can hold an API key, and a window in
        // which it is world-readable is still a window.
        try data.write(to: url, options: [.atomic])
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}
