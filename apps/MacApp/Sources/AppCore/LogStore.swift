import Foundation

/// Bounded, thread-safe log buffer shared by the bootstrapper and the server supervisor.
///
/// Deliberately not an `ObservableObject`: AppCore stays free of SwiftUI so it can be unit
/// tested without a UI. AppHost subscribes and republishes.
public final class LogStore: @unchecked Sendable {
    private var lines: [OutputLine] = []
    private var subscribers: [UUID: @Sendable (OutputLine) -> Void] = [:]
    private let lock = NSLock()
    private let limit: Int
    private let fileHandle: FileHandle?

    /// - Parameter mirrorToFile: also append to `~/Library/Application Support/<id>/logs/app.log`.
    ///   On by default, and load-bearing: an app launched from Finder has nowhere to print, so
    ///   without this a failed bootstrap is completely undiagnosable — which is exactly how the
    ///   first launch of this app failed.
    public init(limit: Int = 2000, mirrorToFile: Bool = true) {
        self.limit = limit
        guard mirrorToFile else {
            self.fileHandle = nil
            return
        }
        try? FileManager.default.createDirectory(at: Paths.logsDir, withIntermediateDirectories: true)
        let url = Paths.logsDir.appendingPathComponent("app.log")
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        let handle = try? FileHandle(forWritingTo: url)
        try? handle?.seekToEnd()
        self.fileHandle = handle
    }

    deinit { try? fileHandle?.close() }

    public func append(_ line: OutputLine) {
        lock.lock()
        lines.append(line)
        if lines.count > limit { lines.removeFirst(lines.count - limit) }
        let handlers = Array(subscribers.values)
        let handle = fileHandle
        lock.unlock()

        if let handle {
            let stamp = ISO8601DateFormatter().string(from: line.date)
            let marker = line.stream == .stderr ? "!" : " "
            try? handle.write(contentsOf: Data("\(stamp)\(marker) \(line.text)\n".utf8))
        }
        for handler in handlers { handler(line) }
    }

    /// Record an app-level event (not subprocess output) into the same stream.
    public func note(_ message: String) {
        append(OutputLine(stream: .stdout, text: message))
    }

    public func snapshot() -> [OutputLine] {
        lock.lock(); defer { lock.unlock() }
        return lines
    }

    public func clear() {
        lock.lock(); defer { lock.unlock() }
        lines.removeAll()
    }

    @discardableResult
    public func subscribe(_ handler: @escaping @Sendable (OutputLine) -> Void) -> UUID {
        lock.lock(); defer { lock.unlock() }
        let token = UUID()
        subscribers[token] = handler
        return token
    }

    public func unsubscribe(_ token: UUID) {
        lock.lock(); defer { lock.unlock() }
        subscribers.removeValue(forKey: token)
    }
}
